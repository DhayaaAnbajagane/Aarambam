/* FFT module of the 2LPT executables: grid allocation, FFTW3-MPI plans and their wisdom file, and the FFTs of the
 * non-Gaussian stage with either backend (see lpt_fft.h for the layouts and the API, band_fft.h for the band one). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mpi.h>
#include "allvars.h"
#include "proto.h"
#include "lpt_fft.h"
#include "lpt_prof.h"
#include "band_fft.h"

int       Band_ny;
int      *Band_y;
int       Band_nz;
ptrdiff_t Band_sy, Band_sx;

static band_fft *Band;              /* FFTBackend band (NGPrecision double); NULL otherwise */
int              NGFloat;           /* NGPrecision float */
#ifndef AARAMBAM_NO_FLOAT
static band_fftf *BandF;            /* NGPrecision float: the single-precision band FFT */
#endif

static int lpt_plan_alignment;      /* fftw_alignment_of() of the grid the plans were made on */


/* ---- grids ---- */

/* One grid of TotalSizePlusAdditional reals. The plans are made once (initialize_ffts) and run on every grid
 * through FFTW3's new-array execute functions, which need the planning alignment: 64-byte aligned grids have it
 * for every SIMD instruction set. Freed with free(). */
void *lpt_alloc_grid(void)
{
  void *p = NULL;

  if(posix_memalign(&p, 64, sizeof(lpt_real) * (size_t) TotalSizePlusAdditional))
    return NULL;
  return p;
}

/* One grid of TotalSizePlusAdditional floats (NGPrecision float: the non-Gaussian stage's grids), FFT-aligned. */
void *lpt_alloc_grid_f(void)
{
  void *p = NULL;

  if(posix_memalign(&p, 64, sizeof(float) * (size_t) TotalSizePlusAdditional))
    return NULL;
  return p;
}

static void lpt_check_alignment(const void *field)
{
  if(fftw_alignment_of((double *) field) != lpt_plan_alignment)
    {
      printf("Task %d: grid passed to an FFT is not allocated with lpt_alloc_grid()\n", ThisTask);
      FatalError(1116);
    }
}

static void lpt_no_double_band(void)
{
  if(NGFloat)
    {
      printf("Task %d: double-precision band FFT called with NGPrecision float\n", ThisTask);
      FatalError(1123);
    }
}


/* ---- the FFTs ---- */

/* In-place FFT of one grid with Forward_plan (r2c) or Inverse_plan (c2r), normal x-slab layout. */
void lpt_fft(fftw_plan plan, lpt_real *field)
{
  PROF_BEGIN(PROF_FFT)
  lpt_check_alignment(field);
  if(plan == Inverse_plan)
    fftw_mpi_execute_dft_c2r(plan, (fftw_complex *) field, field);
  else
    fftw_mpi_execute_dft_r2c(plan, field, (fftw_complex *) field);
  PROF_END(PROF_FFT)
}

/* r2c of a real grid, result in the band layout. fftw: FFTW's transposed output (the y-slabs of
 * FFTW_MPI_TRANSPOSED_OUT), which skips the final all-to-all of the normal-order transform. band: band_fft. */
void lpt_fft_forward_band(lpt_real *field)
{
  lpt_no_double_band();
  PROF_BEGIN(PROF_FFT)
  lpt_check_alignment(field);
  if(Band)
    band_fft_forward(Band, field);
  else
    fftw_mpi_execute_dft_r2c(Forward_plan_t, field, (fftw_complex *) field);
  PROF_END(PROF_FFT)
}

/* c2r of a normal-layout k-space grid that is zero outside the band. fftw: the full transform. band: band_fft,
 * which only reads the band. */
void lpt_fft_inverse_band(lpt_real *field)
{
  lpt_no_double_band();
  if(!Band)
    {
      lpt_fft(Inverse_plan, field);
      return;
    }
  PROF_BEGIN(PROF_FFT)
  lpt_check_alignment(field);
  band_fft_inverse(Band, field);
  PROF_END(PROF_FFT)
}

/* Single precision (NGPrecision float): the same as lpt_fft_forward_band / lpt_fft_inverse_band on float grids from
 * lpt_alloc_grid_f, always with the band FFT. */
void lpt_fftf_forward_band(float *field)
{
#ifndef AARAMBAM_NO_FLOAT
  PROF_BEGIN(PROF_FFT)
  if(fftwf_alignment_of(field) != 0)
    FatalError(1116);
  band_fftf_forward(BandF, field);
  PROF_END(PROF_FFT)
#else
  (void) field;
  FatalError(1122);
#endif
}

void lpt_fftf_inverse_band(float *field)
{
#ifndef AARAMBAM_NO_FLOAT
  PROF_BEGIN(PROF_FFT)
  if(fftwf_alignment_of(field) != 0)
    FatalError(1116);
  band_fftf_inverse(BandF, field);
  PROF_END(PROF_FFT)
#else
  (void) field;
  FatalError(1122);
#endif
}

/* Band layout -> normal layout of a double grid, in place. fftw: a transpose. band: an all-to-all of the band, zero
 * elsewhere (NGPrecision float: the band layout of the float transform, moved in double precision). */
void lpt_band_to_normal(lpt_complex *data)
{
  PROF_BEGIN(PROF_FFT)
  lpt_check_alignment(data);
#ifndef AARAMBAM_NO_FLOAT
  if(BandF)
    band_fftf_to_slab_double(BandF, (fftw_complex *) data);
  else
#endif
  if(Band)
    band_fft_to_slab(Band, (fftw_complex *) data);
  else
    fftw_mpi_execute_r2r(Transpose_plan, (double *) data, (double *) data);
  PROF_END(PROF_FFT)
}


/* ---- wisdom file ---- */

/* Room for a wisdom directory (FFTWWisdomDir or AARAMBAM_FFTW_WISDOM_DIR, both at most sizeof(FFTWWisdomDir) - 1
 * characters, or the default cache directory) and for the file and temporary-file names in it. */
#define LPT_WISDOM_DIR_LEN  (sizeof(FFTWWisdomDir) + 32)
#define LPT_WISDOM_PATH_LEN (LPT_WISDOM_DIR_LEN + 64)

/* mkdir -p; returns 0 if dir exists (or was created) and is writable */
static int lpt_make_dir(const char *dir)
{
  char  buf[LPT_WISDOM_DIR_LEN];
  char *p;

  snprintf(buf, sizeof(buf), "%s", dir);
  for(p = buf + 1; *p; p++)
    if(*p == '/')
      {
        *p = 0;
        mkdir(buf, 0755);
        *p = '/';
      }
  mkdir(buf, 0755);
  return access(buf, W_OK);
}

/* Tag of this CPU model (hash of /proc/cpuinfo's "model name"): plans measured on one node type are kept apart
 * from those of another. */
static unsigned long lpt_cpu_tag(void)
{
  char          line[512];
  unsigned long h  = 5381;
  FILE         *fd = fopen("/proc/cpuinfo", "r");

  if(!fd)
    return 0;
  while(fgets(line, sizeof(line), fd))
    if(strncmp(line, "model name", 10) == 0)
      {
        for(char *c = line; *c && *c != '\n'; c++)
          h = h * 33 + (unsigned char) *c;
        break;
      }
  fclose(fd);
  return h & 0xffffffffUL;
}

/* Rank 0: the wisdom file to use, <dir>/fftw3_wisdom_<cpu>.txt. The directory is FFTWWisdomDir when it is a path
 * (created if needed; an unusable one is an error), else the AARAMBAM_FFTW_WISDOM_DIR environment variable (same
 * rules), else $XDG_CACHE_HOME/aarambam or ~/.cache/aarambam (if that cannot be created, plan with
 * FFTW_ESTIMATE). "none" in the tag or the variable: no file. Returns 0 for no file. */
static int lpt_wisdom_path(char *path, size_t n)
{
  const char *env = getenv("AARAMBAM_FFTW_WISDOM_DIR"), *given = NULL, *base;
  char        dir[LPT_WISDOM_DIR_LEN];
  int         n_dir;

  if(strcmp(FFTWWisdomDir, "none") == 0)
    return 0;
  if(strcmp(FFTWWisdomDir, "auto") != 0)
    given = FFTWWisdomDir;
  else if(env && env[0])
    given = env;

  if(given)
    {
      if(strcmp(given, "none") == 0)
        return 0;
      if(strlen(given) >= sizeof(FFTWWisdomDir))
        {
          printf("\nAARAMBAM_FFTW_WISDOM_DIR is longer than %d characters.\n", (int) sizeof(FFTWWisdomDir) - 1);
          fflush(stdout);
          FatalError(1119);
        }
      snprintf(dir, sizeof(dir), "%s", given);
      if(lpt_make_dir(dir) != 0)
        {
          printf("\nFFTWWisdomDir '%s' (%s) cannot be created or is not writable.\n", dir,
                 given == env ? "from AARAMBAM_FFTW_WISDOM_DIR" : "from the parameter file");
          fflush(stdout);
          FatalError(1119);
        }
    }
  else
    {
      if((base = getenv("XDG_CACHE_HOME")) && base[0])
        n_dir = snprintf(dir, sizeof(dir), "%s/aarambam", base);
      else if((base = getenv("HOME")) && base[0])
        n_dir = snprintf(dir, sizeof(dir), "%s/.cache/aarambam", base);
      else
        return 0;
      if(n_dir >= (int) sizeof(dir))
        {
          printf("\nThe default FFT plan directory (under %s) is too long: planning with FFTW_ESTIMATE. Set "
                 "FFTWWisdomDir to keep measured plans.\n", base);
          return 0;
        }

      if(lpt_make_dir(dir) != 0)
        {
          printf("\nCannot create the FFT plan directory %s: planning with FFTW_ESTIMATE. Set FFTWWisdomDir to a "
                 "writable directory to keep measured plans.\n", dir);
          return 0;
        }
    }

  snprintf(path, n, "%s/fftw3_wisdom_%08lx.txt", dir, lpt_cpu_tag());
  return 1;
}

/* Rank 0: write the accumulated wisdom (merged with the file, in case another run added plans meanwhile) to a
 * temporary file and rename it over the wisdom file, so concurrent runs never see a half-written file. */
static void lpt_save_wisdom(const char *path)
{
  char tmp[LPT_WISDOM_PATH_LEN + 32];

  fftw_import_wisdom_from_filename(path);
  snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long) getpid());
  if(!fftw_export_wisdom_to_filename(tmp) || rename(tmp, path) != 0)
    {
      remove(tmp);
      printf("\nCould not write the FFT plans to %s (they are measured again next time).\n", path);
    }
}


#ifndef AARAMBAM_NO_FLOAT
/* Single-precision plans (NGPrecision float) are kept in their own file next to the double one,
 * <dir>/fftw3f_wisdom_<cpu>.txt. FFTW's MPI wisdom functions exist only for double, so the broadcast and the gather
 * are done here with the wisdom as a string. */
static void lpt_float_wisdom_path(char *fpath, size_t n, const char *path)
{
  const char *tag = strstr(path, "/fftw3_wisdom_");

  snprintf(fpath, n, "%.*s/fftw3f_wisdom_%s", (int) (tag - path), path, tag + strlen("/fftw3_wisdom_"));
}

static void lpt_float_wisdom_load(const char *fpath, char **before, int *loaded)
{
  char *s = NULL;
  int   len = 0;

  if(ThisTask == 0)
    {
      *loaded = fftwf_import_wisdom_from_filename(fpath);
      *before = fftwf_export_wisdom_to_string();
      s       = *before;
      len     = s ? (int) strlen(s) + 1 : 0;
    }
  MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(len == 0)
    return;
  if(ThisTask != 0)
    s = malloc(len);
  MPI_Bcast(s, len, MPI_CHAR, 0, MPI_COMM_WORLD);
  if(ThisTask != 0)
    {
      fftwf_import_wisdom_from_string(s);
      free(s);
    }
}

static void lpt_float_wisdom_keep(const char *fpath, char *before, int loaded, double tplan)
{
  char *mine = fftwf_export_wisdom_to_string(), *all = NULL, tmp[LPT_WISDOM_PATH_LEN + 32];
  int   len  = mine ? (int) strlen(mine) + 1 : 1, *lens = NULL, *offs = NULL, total = 0;

  if(ThisTask == 0)
    {
      lens = malloc(sizeof(int) * NTask);
      offs = malloc(sizeof(int) * NTask);
    }
  MPI_Gather(&len, 1, MPI_INT, lens, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(ThisTask == 0)
    {
      for(int r = 0; r < NTask; r++)
        {
          offs[r] = total;
          total += lens[r];
        }
      all = malloc(total);
    }
  MPI_Gatherv(mine ? mine : "", len, MPI_CHAR, all, lens, offs, MPI_CHAR, 0, MPI_COMM_WORLD);
  free(mine);

  if(ThisTask == 0)
    {
      for(int r = 1; r < NTask; r++)
        if(lens[r] > 1)
          fftwf_import_wisdom_from_string(all + offs[r]);

      char *after = fftwf_export_wisdom_to_string();

      if(!before || !after || strcmp(before, after) != 0)
        {
          fftwf_import_wisdom_from_filename(fpath);
          snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", fpath, (long) getpid());
          if(!fftwf_export_wisdom_to_filename(tmp) || rename(tmp, fpath) != 0)
            {
              remove(tmp);
              printf("\nCould not write the single-precision FFT plans to %s.\n", fpath);
            }
          else
            printf("\nSingle-precision FFT plans measured in %.1f s and saved to %s\n", tplan, fpath);
        }
      else
        printf("\nSingle-precision FFT plans read from %s%s\n", fpath, loaded ? "" : " (no file)");
      free(after);
      free(before);
      free(lens);
      free(offs);
      free(all);
    }
}
#endif


/* ---- setup ---- */

/* FFTBackend and NGPrecision: checked, NGFloat set. Returns whether the NG stage uses a band FFT (FFTBackend band,
 * or NGPrecision float, whose FFTs are always band FFTs). */
static int lpt_fft_options(void)
{
  int band = strcmp(FFTBackend, "band") == 0;

  if(!band && strcmp(FFTBackend, "fftw") != 0)
    {
      if(ThisTask == 0)
        printf("\nFFTBackend '%s': must be fftw or band.\n", FFTBackend);
      fflush(stdout);
      FatalError(1120);
    }
  NGFloat = strcmp(NGPrecision, "float") == 0;
  if(!NGFloat && strcmp(NGPrecision, "double") != 0)
    {
      if(ThisTask == 0)
        printf("\nNGPrecision '%s': must be double or float.\n", NGPrecision);
      fflush(stdout);
      FatalError(1122);
    }
#ifdef AARAMBAM_NO_FLOAT
  if(NGFloat)
    {
      if(ThisTask == 0)
        printf("\nNGPrecision float: these executables were built without single-precision FFTW (libfftw3f). Build "
               "FFTW once more with --enable-float into the same prefix and reinstall, or use NGPrecision double.\n");
      fflush(stdout);
      FatalError(1122);
    }
#endif
  if(NGFloat && !band && ThisTask == 0)
    printf("\nNGPrecision float: the non-Gaussian FFTs use the band FFT (FFTBackend fftw applies to double only).\n");
  return band || NGFloat;
}

/* The x-slab layout of FFTW3-MPI's transforms: Local_nx, Local_x_start, Local_nx_table, Slab_to_task, and (n1, s1) the
 * y-slab of the transposed output. Returns the reals a grid needs for FFTW's transforms and transpose. */
static long long lpt_slab_layout(ptrdiff_t *n1, ptrdiff_t *s1)
{
  ptrdiff_t alloc_c, alloc_t, n0, s0, t0, ts0, t1, ts1;
  int      *slab_to_task_local;
  const int nzc = Nmesh / 2 + 1;     /* complex z length of the r2c layout */

  /* r2c layout: real [Local_nx][Nmesh][2 nzc] in x-slabs; the transposed output is [n1][Nmesh][nzc] */
  alloc_c = fftw_mpi_local_size_3d_transposed(Nmesh, Nmesh, nzc, MPI_COMM_WORLD, &n0, &s0, n1, s1);

  /* the transposed -> normal transpose of the k-space grid: an Nmesh (y) x Nmesh (x) matrix of 2 nzc doubles */
  alloc_t = fftw_mpi_local_size_many_transposed(2, (ptrdiff_t[]) {Nmesh, Nmesh}, 2 * nzc,
                                                FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD,
                                                &t0, &ts0, &t1, &ts1);
  if(t0 != *n1 || ts0 != *s1 || t1 != n0 || ts1 != s0)
    {
      printf("Task %d: FFTW3 transpose layout (%td+%td, %td+%td) does not match the FFT layout (%td+%td, %td+%td)\n",
             ThisTask, ts0, t0, ts1, t1, *s1, *n1, s0, n0);
      FatalError(1117);
    }

  Local_nx      = (int) n0;
  Local_x_start = (int) s0;

  Local_nx_table = malloc(sizeof(int) * NTask);
  MPI_Allgather(&Local_nx, 1, MPI_INT, Local_nx_table, 1, MPI_INT, MPI_COMM_WORLD);

  Slab_to_task       = malloc(sizeof(int) * Nmesh);
  slab_to_task_local = calloc(Nmesh, sizeof(int));
  for(int i = 0; i < Local_nx; i++)
    slab_to_task_local[Local_x_start + i] = ThisTask;
  MPI_Allreduce(slab_to_task_local, Slab_to_task, Nmesh, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  free(slab_to_task_local);

  return (2 * alloc_c > alloc_t) ? 2 * (long long) alloc_c : (long long) alloc_t;
}

/* Band_ny, Band_nz, Band_sx, Band_sy, Band_y from the layout of a band transform (band_fft.h: [x][row][kz]), and its
 * buffer reported. */
static void lpt_set_band_layout(int ny, int nz, const int *row_y, size_t buffer_bytes, int nrows, const char *what)
{
  double buf_mb = buffer_bytes / 1048576.0, buf_max;

  Band_ny = ny;
  Band_nz = nz;
  Band_sy = Band_nz;
  Band_sx = (ptrdiff_t) Band_ny * Band_nz;
  Band_y  = malloc(sizeof(int) * (Band_ny > 0 ? Band_ny : 1));
  memcpy(Band_y, row_y, sizeof(int) * Band_ny);

  MPI_Reduce(&buf_mb, &buf_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if(ThisTask == 0)
    printf("\n%s %d of %d rows of ky, %d of %d planes of kz; communication buffer %.1f MB per rank\n", what, nrows,
           Nmesh, Band_nz, Nmesh / 2 + 1, buf_max);
}

static void lpt_band_failed(const char *what)
{
  if(ThisTask == 0)
    printf("\n%s: the transfers per rank exceed 2^31 numbers or a plan failed; run with more MPI ranks%s.\n", what,
           NGFloat ? "" : " or FFTBackend fftw");
  fflush(stdout);
  FatalError(1121);
}

/* The FFTs of the NG stage and their k-space ("band") layout: the single-precision band FFT (NGPrecision float), the
 * band FFT (FFTBackend band), or FFTW's transposed y-slab (FFTBackend fftw, whose plans lpt_make_plans makes).
 * Returns the reals a grid needs for the band FFT (0 for fftw). */
static long long lpt_band_setup(int band, unsigned flags, ptrdiff_t n1, ptrdiff_t s1)
{
  int *row_in_band, nrows = 0;
  long long need = 0;

  if(!band)
    {
      /* FFTW's transposed y-slab, all of its rows */
      Band_ny = (int) n1;
      Band_nz = Nmesh / 2 + 1;
      Band_sy = (ptrdiff_t) Nmesh * Band_nz;
      Band_sx = Band_nz;
      Band_y  = malloc(sizeof(int) * (Band_ny > 0 ? Band_ny : 1));
      for(int i = 0; i < Band_ny; i++)
        Band_y[i] = (int) s1 + i;
      return 0;
    }

  row_in_band = malloc(sizeof(int) * Nmesh);
  for(int i = 0; i < Nmesh; i++)
    nrows += (row_in_band[i] = lpt_axis_in_band(i));

#ifndef AARAMBAM_NO_FLOAT
  if(NGFloat)
    {
      BandF = band_fftf_create(Nmesh, (int) lpt_kz_end(), row_in_band, Local_nx, Local_x_start, MPI_COMM_WORLD,
                               flags);
      if(!BandF)
        lpt_band_failed("NGPrecision float");
      lpt_set_band_layout(band_fftf_rows(BandF), band_fftf_nz(BandF), band_fftf_row_y(BandF),
                          band_fftf_buffer_bytes(BandF), nrows, "NGPrecision float: band FFT with");
      need = 2 * (long long) band_fftf_local_size(BandF);
    }
  else
#endif
    {
      Band = band_fft_create(Nmesh, (int) lpt_kz_end(), row_in_band, Local_nx, Local_x_start, MPI_COMM_WORLD, flags);
      if(!Band)
        lpt_band_failed("FFTBackend band");
      lpt_set_band_layout(band_fft_rows(Band), band_fft_nz(Band), band_fft_row_y(Band), band_fft_buffer_bytes(Band),
                          nrows, "FFTBackend band:");
      need = 2 * (long long) band_fft_local_size(Band);
    }
  free(row_in_band);
  return need;
}

/* FFTW3-MPI plans: the full r2c / c2r (lpt_fft), and for FFTBackend fftw the transposed r2c and the transpose. Made on
 * a grid from lpt_alloc_grid, so every grid they later run on has the planning alignment (FFTW_MEASURE overwrites it
 * while timing candidate plans). */
static void lpt_make_plans(int band, unsigned flags)
{
  const int nzc       = Nmesh / 2 + 1;
  lpt_real *plan_grid = lpt_alloc_grid();

  if(!plan_grid)
    {
      printf("Task %d: failed to allocate the FFT planning grid\n", ThisTask);
      FatalError(1);
    }

  Forward_plan = fftw_mpi_plan_dft_r2c_3d(Nmesh, Nmesh, Nmesh, plan_grid, (fftw_complex *) plan_grid,
                                          MPI_COMM_WORLD, flags);
  Inverse_plan = fftw_mpi_plan_dft_c2r_3d(Nmesh, Nmesh, Nmesh, (fftw_complex *) plan_grid, plan_grid,
                                          MPI_COMM_WORLD, flags);
  if(!band)
    {
      Forward_plan_t = fftw_mpi_plan_dft_r2c_3d(Nmesh, Nmesh, Nmesh, plan_grid, (fftw_complex *) plan_grid,
                                                MPI_COMM_WORLD, flags | FFTW_MPI_TRANSPOSED_OUT);
      Transpose_plan = fftw_mpi_plan_many_transpose(Nmesh, Nmesh, 2 * nzc, FFTW_MPI_DEFAULT_BLOCK,
                                                    FFTW_MPI_DEFAULT_BLOCK, plan_grid, plan_grid, MPI_COMM_WORLD,
                                                    flags);
    }
  if(!Forward_plan || !Inverse_plan || (!band && (!Forward_plan_t || !Transpose_plan)))
    {
      printf("Task %d: FFTW3 could not create the MPI plans\n", ThisTask);
      FatalError(1118);
    }

  lpt_plan_alignment = fftw_alignment_of(plan_grid);
  free(plan_grid);
}

/* The wisdom files of a run: the double-precision plans (FFTW3-MPI, and the band FFT's serial plans) and, for
 * NGPrecision float, the single-precision ones. */
typedef struct
{
  int   use, loaded, floaded;
  char  path[LPT_WISDOM_PATH_LEN], fpath[LPT_WISDOM_PATH_LEN + 8];
  char *before, *fbefore;      /* rank 0: the wisdom before planning, to tell whether new plans were measured */
}
lpt_wisdom;

/* Read the wisdom files (rank 0) and give every rank the same wisdom. */
static void lpt_wisdom_begin(lpt_wisdom *w)
{
  memset(w, 0, sizeof(*w));
  if(ThisTask == 0)
    w->use = lpt_wisdom_path(w->path, sizeof(w->path));
  MPI_Bcast(&w->use, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(!w->use)
    return;
  if(ThisTask == 0)
    {
      w->loaded = fftw_import_wisdom_from_filename(w->path);
      w->before = fftw_export_wisdom_to_string();
    }
  fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);
#ifndef AARAMBAM_NO_FLOAT
  if(NGFloat)
    {
      if(ThisTask == 0)
        lpt_float_wisdom_path(w->fpath, sizeof(w->fpath), w->path);
      lpt_float_wisdom_load(w->fpath, &w->fbefore, &w->floaded);
    }
#endif
}

/* Gather the plans of all ranks and save the files if new plans were measured (rank 0). */
static void lpt_wisdom_end(lpt_wisdom *w, double tplan)
{
  if(!w->use)
    return;
  fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
  if(ThisTask == 0)
    {
      char *after    = fftw_export_wisdom_to_string();
      int   measured = !w->before || !after || strcmp(w->before, after) != 0;

      if(measured)
        {
          lpt_save_wisdom(w->path);
          printf("\nFFT plans measured in %.1f s and saved to %s\n", tplan, w->path);
        }
      else
        printf("\nFFT plans read from %s%s\n", w->path, w->loaded ? "" : " (no file)");
      free(w->before);
      free(after);
    }
#ifndef AARAMBAM_NO_FLOAT
  if(NGFloat)
    lpt_float_wisdom_keep(w->fpath, w->fbefore, w->floaded, tplan);
#endif
}

/* FFTW3-MPI plans of the in-place Nmesh^3 transforms (slab-decomposed in x), the slab sizes, the FFTs of the
 * non-Gaussian stage (FFTBackend fftw: FFTW3-MPI's transposed transform and a transpose; band: band_fft.c;
 * NGPrecision float: band_fft.c in single precision), the band layout and the grid size. With a wisdom file
 * (FFTWWisdomDir, default "auto") the plans are measured (FFTW_MEASURE) once per grid size, rank count and CPU model
 * and kept in the file; later runs read them back and plan instantly. With FFTWWisdomDir none they are
 * FFTW_ESTIMATE plans (no measuring, no file). */
void initialize_ffts(void)
{
  const int  band = lpt_fft_options();
  ptrdiff_t  n1, s1;
  long long  total_size, band_size;
  int        additional;
  unsigned   flags;
  double     tplan;
  lpt_wisdom wisdom;

  MPI_Barrier(MPI_COMM_WORLD);
  total_size = lpt_slab_layout(&n1, &s1);

  lpt_wisdom_begin(&wisdom);
  flags = wisdom.use ? FFTW_MEASURE : FFTW_ESTIMATE;
  tplan = MPI_Wtime();

  band_size = lpt_band_setup(band, flags, n1, s1);
  if(band_size > total_size)
    total_size = band_size;

  additional = (Nmesh) * (2 * (Nmesh / 2 + 1));     /* additional plane on the right side */
  if(total_size + additional > AARAMBAM_MAX_SLAB_REALS)
    {
      printf("Task %d: its grid (%lld reals for a %d x %d x %d slab plus one plane) exceeds the 32-bit grid size "
             "of this code (%lld). Run with more MPI ranks.\n", ThisTask, total_size + additional, Local_nx, Nmesh,
             2 * (Nmesh / 2 + 1), (long long) AARAMBAM_MAX_SLAB_REALS);
      FatalError(1114);
    }
  TotalSizePlusAdditional = total_size + additional;

  lpt_make_plans(band, flags);
  tplan = MPI_Wtime() - tplan;

  lpt_wisdom_end(&wisdom, tplan);
}

void free_ffts(void)
{
  free(Slab_to_task);
  free(Band_y);
  fftw_destroy_plan(Inverse_plan);
  fftw_destroy_plan(Forward_plan);
  if(Forward_plan_t)
    fftw_destroy_plan(Forward_plan_t);
  if(Transpose_plan)
    fftw_destroy_plan(Transpose_plan);
  band_fft_destroy(Band);
  Band = NULL;
#ifndef AARAMBAM_NO_FLOAT
  band_fftf_destroy(BandF);
  BandF = NULL;
#endif
  fftw_mpi_cleanup();
}
