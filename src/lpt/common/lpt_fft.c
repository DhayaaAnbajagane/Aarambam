/* FFT module of the 2LPT executables: grid allocation, FFTW3-MPI plans and their wisdom file, and the FFTs of the
 * non-Gaussian stage (see lpt_fft.h for the layouts and the API). */
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

int  Band_ny;
int *Band_y;
int  Band_nz;

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

static void lpt_check_alignment(const void *field)
{
  if(fftw_alignment_of((double *) field) != lpt_plan_alignment)
    {
      printf("Task %d: grid passed to an FFT is not allocated with lpt_alloc_grid()\n", ThisTask);
      FatalError(1116);
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

/* r2c of a real grid, result in the band layout. FFTW backend: FFTW's transposed output ([y][x][z], the y-slabs
 * of FFTW_MPI_TRANSPOSED_OUT), which skips the final all-to-all of the normal-order transform. */
void lpt_fft_forward_band(lpt_real *field)
{
  PROF_BEGIN(PROF_FFT)
  lpt_check_alignment(field);
  fftw_mpi_execute_dft_r2c(Forward_plan_t, field, (fftw_complex *) field);
  PROF_END(PROF_FFT)
}

/* c2r of a normal-layout k-space grid that is zero outside the band. FFTW backend: the full transform. */
void lpt_fft_inverse_band(lpt_real *field)
{
  lpt_fft(Inverse_plan, field);
}

/* Band layout -> normal layout, in place. FFTW backend: a transpose, a pure data movement. */
void lpt_band_to_normal(lpt_complex *data)
{
  PROF_BEGIN(PROF_FFT)
  lpt_check_alignment(data);
  fftw_mpi_execute_r2r(Transpose_plan, (double *) data, (double *) data);
  PROF_END(PROF_FFT)
}


/* ---- wisdom file ---- */

/* mkdir -p; returns 0 if dir exists (or was created) and is writable */
static int lpt_make_dir(const char *dir)
{
  char  buf[600];
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
  char        dir[400];

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
        snprintf(dir, sizeof(dir), "%s/aarambam", base);
      else if((base = getenv("HOME")) && base[0])
        snprintf(dir, sizeof(dir), "%s/.cache/aarambam", base);
      else
        return 0;

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
  char tmp[700];

  fftw_import_wisdom_from_filename(path);
  snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long) getpid());
  if(!fftw_export_wisdom_to_filename(tmp) || rename(tmp, path) != 0)
    {
      remove(tmp);
      printf("\nCould not write the FFT plans to %s (they are measured again next time).\n", path);
    }
}


/* ---- setup ---- */

/* FFTW3-MPI plans of the in-place Nmesh^3 transforms (slab-decomposed in x), the slab sizes, the band layout and
 * the grid size. With a wisdom file (FFTWWisdomDir, default "auto") the plans are measured (FFTW_MEASURE) once per
 * grid size, rank count and CPU model and kept in the file; later runs read them back and plan instantly. With
 * FFTWWisdomDir none they are FFTW_ESTIMATE plans (no measuring, no file). */
void initialize_ffts(void)
{
  ptrdiff_t alloc_c, alloc_t, n0, s0, n1, s1, t0, ts0, t1, ts1;
  long long total_size;
  int       i, additional, use_wisdom = 0, loaded = 0;
  int      *slab_to_task_local;
  lpt_real *plan_grid;
  unsigned  flags;
  char      wisdom_path[600], *wisdom_before = NULL;
  double    tplan;
  const int nzc = Nmesh / 2 + 1;     /* complex z length of the r2c layout */

  MPI_Barrier(MPI_COMM_WORLD);

  /* r2c layout: real [Local_nx][Nmesh][2 nzc] in x-slabs; the transposed output is [n1][Nmesh][nzc] */
  alloc_c = fftw_mpi_local_size_3d_transposed(Nmesh, Nmesh, nzc, MPI_COMM_WORLD, &n0, &s0, &n1, &s1);

  /* the transposed -> normal transpose of the k-space grid: an Nmesh (y) x Nmesh (x) matrix of 2 nzc doubles */
  alloc_t = fftw_mpi_local_size_many_transposed(2, (ptrdiff_t[]) {Nmesh, Nmesh}, 2 * nzc,
                                                FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD,
                                                &t0, &ts0, &t1, &ts1);
  if(t0 != n1 || ts0 != s1 || t1 != n0 || ts1 != s0)
    {
      printf("Task %d: FFTW3 transpose layout (%td+%td, %td+%td) does not match the FFT layout (%td+%td, %td+%td)\n",
             ThisTask, ts0, t0, ts1, t1, s1, n1, s0, n0);
      FatalError(1117);
    }

  Local_nx      = (int) n0;
  Local_x_start = (int) s0;
  total_size    = (2 * alloc_c > alloc_t) ? 2 * (long long) alloc_c : (long long) alloc_t;   /* in reals */

  /* band layout of the FFTW backend: FFTW's transposed y-slab, all of its rows */
  Band_ny = (int) n1;
  Band_nz = nzc;
  Band_y  = malloc(sizeof(int) * (Band_ny > 0 ? Band_ny : 1));
  for(i = 0; i < Band_ny; i++)
    Band_y[i] = (int) s1 + i;

  Local_nx_table = malloc(sizeof(int) * NTask);
  MPI_Allgather(&Local_nx, 1, MPI_INT, Local_nx_table, 1, MPI_INT, MPI_COMM_WORLD);

  Slab_to_task       = malloc(sizeof(int) * Nmesh);
  slab_to_task_local = malloc(sizeof(int) * Nmesh);

  for(i = 0; i < Nmesh; i++)
    slab_to_task_local[i] = 0;

  for(i = 0; i < Local_nx; i++)
    slab_to_task_local[Local_x_start + i] = ThisTask;

  MPI_Allreduce(slab_to_task_local, Slab_to_task, Nmesh, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  free(slab_to_task_local);

  additional = (Nmesh) * (2 * (Nmesh / 2 + 1));     /* additional plane on the right side */

  if(total_size + additional > AARAMBAM_MAX_SLAB_REALS)
    {
      printf("Task %d: its grid (%lld reals for a %d x %d x %d slab plus one plane) exceeds the 32-bit grid size "
             "of this code (%lld). Run with more MPI ranks.\n", ThisTask, total_size + additional, Local_nx, Nmesh,
             2 * nzc, (long long) AARAMBAM_MAX_SLAB_REALS);
      FatalError(1114);
    }

  TotalSizePlusAdditional = total_size + additional;

  /* wisdom file: read on rank 0, the same plans on every rank */
  if(ThisTask == 0)
    use_wisdom = lpt_wisdom_path(wisdom_path, sizeof(wisdom_path));
  MPI_Bcast(&use_wisdom, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(use_wisdom)
    {
      if(ThisTask == 0)
        {
          loaded        = fftw_import_wisdom_from_filename(wisdom_path);
          wisdom_before = fftw_export_wisdom_to_string();
        }
      fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);
    }
  flags = use_wisdom ? FFTW_MEASURE : FFTW_ESTIMATE;

  /* plan on a grid from lpt_alloc_grid, so every grid the plans later run on has the planning alignment
   * (FFTW_MEASURE overwrites it while timing candidate plans) */
  plan_grid = lpt_alloc_grid();
  if(!plan_grid)
    {
      printf("Task %d: failed to allocate the FFT planning grid\n", ThisTask);
      FatalError(1);
    }

  tplan          = MPI_Wtime();
  Forward_plan   = fftw_mpi_plan_dft_r2c_3d(Nmesh, Nmesh, Nmesh, plan_grid, (fftw_complex *) plan_grid,
                                            MPI_COMM_WORLD, flags);
  Forward_plan_t = fftw_mpi_plan_dft_r2c_3d(Nmesh, Nmesh, Nmesh, plan_grid, (fftw_complex *) plan_grid,
                                            MPI_COMM_WORLD, flags | FFTW_MPI_TRANSPOSED_OUT);
  Inverse_plan   = fftw_mpi_plan_dft_c2r_3d(Nmesh, Nmesh, Nmesh, (fftw_complex *) plan_grid, plan_grid,
                                            MPI_COMM_WORLD, flags);
  Transpose_plan = fftw_mpi_plan_many_transpose(Nmesh, Nmesh, 2 * nzc, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                                plan_grid, plan_grid, MPI_COMM_WORLD, flags);
  tplan          = MPI_Wtime() - tplan;
  if(!Forward_plan || !Forward_plan_t || !Inverse_plan || !Transpose_plan)
    {
      printf("Task %d: FFTW3 could not create the MPI plans\n", ThisTask);
      FatalError(1118);
    }

  lpt_plan_alignment = fftw_alignment_of(plan_grid);
  free(plan_grid);

  /* keep newly measured plans */
  if(use_wisdom)
    {
      fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
      if(ThisTask == 0)
        {
          char *wisdom_after = fftw_export_wisdom_to_string();
          int   measured     = !wisdom_before || !wisdom_after || strcmp(wisdom_before, wisdom_after) != 0;

          if(measured)
            {
              lpt_save_wisdom(wisdom_path);
              printf("\nFFT plans measured in %.1f s and saved to %s\n", tplan, wisdom_path);
            }
          else
            printf("\nFFT plans read from %s%s\n", wisdom_path, loaded ? "" : " (no file)");
          free(wisdom_before);
          free(wisdom_after);
        }
    }
}

void free_ffts(void)
{
  free(Slab_to_task);
  free(Band_y);
  fftw_destroy_plan(Inverse_plan);
  fftw_destroy_plan(Forward_plan);
  fftw_destroy_plan(Forward_plan_t);
  fftw_destroy_plan(Transpose_plan);
  fftw_mpi_cleanup();
}
