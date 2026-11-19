#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <mpi.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_spline.h>
#include "allvars.h"
#include "proto.h"
#include <signal.h>
#include <assert.h>
#include <ctype.h>

#define ASSERT_ALLOC(cond) {                                                                                  \
  if(!(cond))                                                                                                 \
    {                                                                                                         \
      printf("failed to allocate %g Mbyte on Task %d\n", bytes / (1024.0 * 1024.0), ThisTask);                \
      printf("bailing out.\n");                                                                               \
      FatalError(1);                                                                                          \
    }                                                                                                         \
}

int frequency_of_primes (int n) {
  int i,j;
  int freq=n-1;
  for (i=2; i<=n; ++i) for (j=sqrt(i);j>1;--j) if (i%j==0) {--freq; break;}
  return freq;
}

#define TIMED_DONE_COLUMN 54

void print_timed_done(int n) {
  clock_t now = clock();
  double tot = ((double)(now - start_time)) / CLOCKS_PER_SEC;
  double diff = ((double)(now - previous_time)) / CLOCKS_PER_SEC;

  int tot_hours = (int) floor(tot / 3600.0);
  int tot_mins  = (int) floor(tot / 60.0) - 60 * tot_hours;
  double tot_secs = tot - 60.0 * (((double) tot_mins) + 60.0 * ((double) tot_hours));

  int diff_hours = (int) floor(diff / 3600.0);
  int diff_mins  = (int) floor(diff / 60.0) - 60 * diff_hours;
  double diff_secs = diff - 60.0 * (((double) diff_mins) + 60.0 * ((double) diff_hours));

  for (int i = 0; i < n; i++)
    printf(" ");
  printf("Done [%02d:%02d:%05.2f, %02d:%02d:%05.2f]\n",
         diff_hours, diff_mins, diff_secs, tot_hours, tot_mins, tot_secs);
  previous_time = now;
  return;
}

static void print_timed_done_aligned(int printed_chars) {
  int indent = TIMED_DONE_COLUMN - printed_chars;
  if(indent < 1)
    indent = 1;
  print_timed_done(indent);
}

void handle_signal(int signal) {
    // Handle the signal, clean up resources
    MPI_Finalize();
    exit(0);
}

int main(int argc, char **argv) {
  int timed_chars = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);
   
  // Setup signal handling
  signal(SIGINT, handle_signal);

  start_time = clock();
  previous_time = start_time;

  if(argc < 2) {
    if(ThisTask == 0) {
	  fprintf(stdout, "\nParameters are missing.\n");
	  fprintf(stdout, "Call with <ParameterFile>\n\n");
	}
    MPI_Finalize();
    exit(0);
  }

  if (ThisTask == 0)
    printf("RUNNING LPTPNG WITH %d TASKS", NTask);

  if (ThisTask == 0) {read_parameterfile(argv[1]);}
  broadcast_parameters();

  MPI_Barrier(MPI_COMM_WORLD);
  set_units();
  initialize_transferfunction(); 
  initialize_powerspectrum(); 
  initialize_ffts(); 
  read_glass(GlassFile);

  if (ThisTask == 0)
    print_setup();

  displacement_fields();

  if (ThisTask == 0) {timed_chars = printf("Writing initial conditions snapshot..."); fflush(stdout);}
  write_particle_data();
  if (ThisTask == 0) print_timed_done_aligned(timed_chars);
  if(NumPart)
    free(P);
  free_ffts();
  MPI_Barrier(MPI_COMM_WORLD);
  print_spec();
  MPI_Finalize();		/* clean up & finalize MPI */
  exit(0);
}

void print_setup(void) {
  char pstr[79];

  for (int i = 0; i < 79; i++)
    pstr[i] = '*';
  printf("%s\n", pstr);

  char exec[] = "2LPTNGBinned";


  pstr[0] = '*';
  for (int i = 1; i < 79/2 - 3; i++)
    pstr[i] = ' ';
  sprintf(&pstr[79/2 - 4], "%s", exec);
  for (int i = 79/2 + 4; i < 78; i++)
    pstr[i] = ' ';
  pstr[78] = '*';
  printf("%s\n", pstr);
  for (int i = 0; i < 79; i++)
    pstr[i] = '*';
  printf("%s\n", pstr);

  printf("\n");

  printf("      Box = %.2e   Nmesh = %d   Nsample = %d   Nmax = %d\n", Box, Nmesh, Nsample, Nmax);
  printf("                       Nglass = %d    GlassTileFac = %d\n\n", Nglass, GlassTileFac);
  printf("        Omega = %.4f         OmegaLambda = %.4f    OmegaBaryon = %.2e\n", Omega,  OmegaLambda, OmegaBaryon);
  printf("       sigma8 = %.4f     PrimoridalIndex = %.4f       Redshift = %.2e\n", Sigma8, PrimordialIndex, Redshift);
  printf("  HubbleParam = %.4f  OmegaDM_2ndSpecies = %.2e          fNL = %+.2e\n\n", HubbleParam, OmegaDM_2ndSpecies, Fnl); 
  printf("   FixedAmplitude = %d    PhaseFlip = % d   SphereMode = %d    Seed = %d\n", FixedAmplitude, PhaseFlip, SphereMode, Seed);
  printf("   BinnedPNGFilter(input k1/k2) = %d    BinnedPNGOutputFilter(k3) = %d    BinnedPNGCoordinate = %d\n", BinnedPNGFilter, BinnedPNGOutputFilter, BinnedPNGCoordinate);
  printf("   BinnedPNGUseInternalSymmetry = %d    BinnedPNGCacheBlockSize = %d\n", BinnedPNGUseInternalSymmetry, BinnedPNGCacheBlockSize);
  printf("   BinnedK1Edges = %s\n", FileWithBinnedK1Edges);
  printf("   BinnedK2Edges = %s\n", FileWithBinnedK2Edges);
  printf("   BinnedK3Edges(output) = %s\n", FileWithBinnedK3Edges);
  printf("   BinnedKernelValues = %s\n", FileWithBinnedKernelValues);

  for (int i = 0; i < 79; i++)
    pstr[i] = '*';
  printf("\n%s\n\n", pstr);
  return;
}

// Helper function for comparing doubles for qsort
int compare_doubles(const void *a, const void *b) {
    const double *da = (const double *) a;
    const double *db = (const double *) b;
    return (*da > *db) - (*da < *db);
}


/* -------------------------------------------------------------------------
 * Binned PNG hybrid-kernel support.
 *
 * PNGBinnedRunner's IC-friendly hybrid table represents the quadratic kernel
 * as
 *
 *   K(k1,k2,k3) = sum_{a,b,c} psi_a(k1) psi_b(k2) H_c(k3)
 *                  [K0_abc + K1_abc u3 + K2_abc u3^2 + K3_abc u3^3],
 *
 * where k1 and k2 are the two internal convolution legs, and k3=|k| is the
 * final/output mode.  This is different from the older cell-local polynomial
 * convention where k1 was treated as the output leg.  The expensive FFT work
 * is therefore one convolution for every (a,b) pair.  The k3 polynomial is
 * applied after the convolution, mode by mode.
 *
 * The parameter file should point FileWithBinnedKernelValues at the
 * PNGBinnedRunner hybrid text table, e.g. <prefix>_hybrid_values.txt.
 *
 * BinnedPNGFilter controls the global input-leg basis psi_a,psi_b:
 *   1 = top-hat / Heaviside on k1,k2
 *   2 = linear global interpolation on k1,k2 bin-center nodes
 *   3 = cubic  global interpolation on k1,k2 bin-center nodes
 *
 * BinnedPNGOutputFilter controls the cell-local output model along k3:
 *   1 = K0
 *   2 = K0 + K1 u3
 *   3 = K0 + K1 u3 + K2 u3^2 + K3 u3^3
 *
 * BinnedPNGCoordinate controls both the bin centers and interpolation variable:
 *   1 = k, 2 = log(k).  Use 2 for tables generated with
 *       --Binned_linear_coordinate log.
 * ------------------------------------------------------------------------- */

#define BPNG_FILTER_TOPHAT    1
#define BPNG_FILTER_LINEAR    2
#define BPNG_FILTER_CUBIC     3
#define BPNG_COORD_K          1
#define BPNG_COORD_LOG        2

typedef struct
{
  int nk[3];
  int input_filter;       /* k1,k2 basis: top-hat/linear/cubic */
  int output_filter;      /* k3 output polynomial: top-hat/linear/cubic */
  int coord_mode;
  int coeff_count;        /* 1, 2, or 4 output coefficients along k3 */
  double *edges[3];
  double *Kcoeff;         /* coeff-major: Kcoeff[q*Ncell + ((i*nk2+j)*nk3+l)] */
}
BinnedPNGTable;

static BinnedPNGTable BPNG = {{0,0,0}, BPNG_FILTER_TOPHAT, BPNG_FILTER_TOPHAT,
                              BPNG_COORD_K, 0, {NULL,NULL,NULL}, NULL};

static long long bpng_ncell(void)
{
  return ((long long) BPNG.nk[0]) * ((long long) BPNG.nk[1]) * ((long long) BPNG.nk[2]);
}

static long long bpng_cell_index(int i, int j, int l)
{
  return (((long long) i) * BPNG.nk[1] + j) * BPNG.nk[2] + l;
}

static double bpng_coeff(int q, int i, int j, int l)
{
  return BPNG.Kcoeff[((long long)q) * bpng_ncell() + bpng_cell_index(i,j,l)];
}

static int bpng_parse_doubles(char *line, double *vals, int maxvals)
{
  int n = 0;
  char *p = line;
  char *end = NULL;
  while(*p && n < maxvals)
    {
      while(*p && isspace((unsigned char)*p)) p++;
      if(*p == 0 || *p == '#') break;
      vals[n] = strtod(p, &end);
      if(end == p) break;
      n++;
      p = end;
    }
  return n;
}

static int bpng_count_edge_file(const char *path)
{
  FILE *fd = fopen(path, "r");
  char line[1024];
  double tmp[4];
  int n = 0;
  if(!fd)
    {
      printf("Could not open binned edge file '%s' on task %d\n", path, ThisTask);
      FatalError(9101);
    }
  while(fgets(line, sizeof(line), fd))
    if(bpng_parse_doubles(line, tmp, 4) >= 1)
      n++;
  fclose(fd);
  return n;
}

static void bpng_read_edge_file_root(const char *path, int axis)
{
  FILE *fd = fopen(path, "r");
  char line[1024];
  double tmp[4];
  int n = 0;
  if(!fd)
    {
      printf("Could not open binned edge file '%s' on task %d\n", path, ThisTask);
      FatalError(9102);
    }
  while(fgets(line, sizeof(line), fd))
    if(bpng_parse_doubles(line, tmp, 4) >= 1)
      BPNG.edges[axis][n++] = tmp[0];
  fclose(fd);
}

static void bpng_read_kernel_coefficients_root(void)
{
  char path[1000], line[8192];
  FILE *fd;
  double vals[80];
  long long nc = bpng_ncell();
  long long q;
  int nmin;

  for(q = 0; q < ((long long)BPNG.coeff_count) * nc; q++)
    BPNG.Kcoeff[q] = 0.0;

  snprintf(path, sizeof(path), "%s", FileWithBinnedKernelValues);

  fd = fopen(path, "r");
  if(!fd)
    {
      printf("Could not open binned kernel coefficient file '%s' on task %d\n", path, ThisTask);
      FatalError(9103);
    }

  /* hybrid_values.txt columns are
   *   i j l k1_center k2_center k3_center
   *   B[0:ncoef] K[0:ncoef] D[0:ncoef] S[0:ncoef]
   *   valid_center valid_box
   * so the K block starts at column 6+ncoef.
   * For ncoef=1 this is also compatible with the older values.txt layout,
   * where the K value is column 7.
   */
  nmin = 8 + 4 * BPNG.coeff_count;

  while(fgets(line, sizeof(line), fd))
    {
      int n = bpng_parse_doubles(line, vals, 80);
      int i, j, l, c, k_start;
      long long ind;
      if(n <= 0) continue;

      i = (int) vals[0];
      j = (int) vals[1];
      l = (int) vals[2];
      if(i < 0 || i >= BPNG.nk[0] || j < 0 || j >= BPNG.nk[1] || l < 0 || l >= BPNG.nk[2])
        continue;
      if(n < nmin)
        continue;

      ind = bpng_cell_index(i,j,l);
      k_start = 6 + BPNG.coeff_count;
      for(c = 0; c < BPNG.coeff_count; c++)
        BPNG.Kcoeff[((long long)c) * nc + ind] = vals[k_start + c];
    }
  fclose(fd);
}

static void bpng_read_and_broadcast_table(void)
{
  int axis, nedge[3], input_filter, output_filter, coord_mode;
  long long ncoef;

  if(ThisTask == 0)
    {
      input_filter = BinnedPNGFilter;
      output_filter = BinnedPNGOutputFilter;
      coord_mode = BinnedPNGCoordinate;

      if(input_filter != BPNG_FILTER_TOPHAT &&
         input_filter != BPNG_FILTER_LINEAR &&
         input_filter != BPNG_FILTER_CUBIC)
        {
          printf("Invalid BinnedPNGFilter=%d. Use 1=top-hat, 2=linear input interpolation, 3=cubic input interpolation.\n", input_filter);
          FatalError(9109);
        }

      if(output_filter != BPNG_FILTER_TOPHAT &&
         output_filter != BPNG_FILTER_LINEAR &&
         output_filter != BPNG_FILTER_CUBIC)
        {
          printf("Invalid BinnedPNGOutputFilter=%d. Use 1=heaviside, 2=linear k3 output, 3=cubic k3 output.\n", output_filter);
          FatalError(9111);
        }

      if(coord_mode != BPNG_COORD_K && coord_mode != BPNG_COORD_LOG)
        {
          printf("Invalid BinnedPNGCoordinate=%d. Use 1=k, 2=log(k).\n", coord_mode);
          FatalError(9110);
        }

      BPNG.input_filter = input_filter;
      BPNG.output_filter = output_filter;
      BPNG.coord_mode = coord_mode;
      BPNG.coeff_count = (output_filter == BPNG_FILTER_CUBIC)  ? 4 :
                         (output_filter == BPNG_FILTER_LINEAR) ? 2 : 1;

      nedge[0] = bpng_count_edge_file(FileWithBinnedK1Edges);
      nedge[1] = bpng_count_edge_file(FileWithBinnedK2Edges);
      nedge[2] = bpng_count_edge_file(FileWithBinnedK3Edges);

      for(axis = 0; axis < 3; axis++)
        {
          if(nedge[axis] < 2)
            FatalError(9104);
          BPNG.nk[axis] = nedge[axis] - 1;
          BPNG.edges[axis] = malloc(nedge[axis] * sizeof(double));
          if(!BPNG.edges[axis]) FatalError(9105);
        }

      bpng_read_edge_file_root(FileWithBinnedK1Edges, 0);
      bpng_read_edge_file_root(FileWithBinnedK2Edges, 1);
      bpng_read_edge_file_root(FileWithBinnedK3Edges, 2);

      ncoef = ((long long) BPNG.coeff_count) * bpng_ncell();
      BPNG.Kcoeff = malloc(ncoef * sizeof(double));
      if(!BPNG.Kcoeff) FatalError(9106);
      bpng_read_kernel_coefficients_root();

      printf("Loaded hybrid binned PNG kernel table: Nk=(%d,%d,%d), input_filter=%d, output_filter=%d, coord=%d, coeff_count=%d\n",
             BPNG.nk[0], BPNG.nk[1], BPNG.nk[2],
             BPNG.input_filter, BPNG.output_filter, BPNG.coord_mode, BPNG.coeff_count);
    }

  MPI_Bcast(BPNG.nk, 3, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&BPNG.input_filter, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&BPNG.output_filter, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&BPNG.coord_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&BPNG.coeff_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(ThisTask != 0)
    {
      for(axis = 0; axis < 3; axis++)
        {
          nedge[axis] = BPNG.nk[axis] + 1;
          BPNG.edges[axis] = malloc(nedge[axis] * sizeof(double));
          if(!BPNG.edges[axis]) FatalError(9107);
        }
      BPNG.Kcoeff = malloc(((long long)BPNG.coeff_count) * bpng_ncell() * sizeof(double));
      if(!BPNG.Kcoeff) FatalError(9108);
    }

  for(axis = 0; axis < 3; axis++)
    MPI_Bcast(BPNG.edges[axis], BPNG.nk[axis] + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(BPNG.Kcoeff, ((long long)BPNG.coeff_count) * bpng_ncell(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

static void bpng_free_table(void)
{
  int axis;
  for(axis = 0; axis < 3; axis++)
    {
      if(BPNG.edges[axis]) free(BPNG.edges[axis]);
      BPNG.edges[axis] = NULL;
    }
  if(BPNG.Kcoeff) free(BPNG.Kcoeff);
  BPNG.Kcoeff = NULL;
}

static int bpng_find_bin(const double *edges, int nbin, double k)
{
  int lo, hi, mid;
  if(k < edges[0] || k > edges[nbin]) return -1;
  if(k == edges[nbin]) return nbin - 1;
  lo = 0;
  hi = nbin;
  while(hi - lo > 1)
    {
      mid = (hi + lo) / 2;
      if(k < edges[mid]) hi = mid;
      else lo = mid;
    }
  return lo;
}

static double bpng_coord_value(double k)
{
  if(BPNG.coord_mode == BPNG_COORD_LOG)
    {
      if(k <= 0.0) return -1.0e300;
      return log(k);
    }
  return k;
}

static double bpng_axis_node_k(int axis, int node)
{
  double lo = BPNG.edges[axis][node];
  double hi = BPNG.edges[axis][node + 1];
  if(BPNG.coord_mode == BPNG_COORD_LOG)
    return sqrt(lo * hi);
  return 0.5 * (lo + hi);
}

static double bpng_axis_node_x(int axis, int node)
{
  return bpng_coord_value(bpng_axis_node_k(axis, node));
}

static double bpng_u_in_bin_axis(int axis, int bin, double k)
{
  double x, xc, dx;
  double xlo, xhi;
  if(bin < 0 || bin >= BPNG.nk[axis]) return 0.0;
  x = bpng_coord_value(k);
  xlo = bpng_coord_value(BPNG.edges[axis][bin]);
  xhi = bpng_coord_value(BPNG.edges[axis][bin + 1]);
  xc = 0.5 * (xlo + xhi);
  dx = xhi - xlo;
  if(dx == 0.0 || !isfinite(dx)) return 0.0;
  return (x - xc) / dx;
}

static int bpng_mode_allowed(double kx, double ky, double kz, double kmag)
{
  if(kmag == 0.0) return 0;
  if(SphereMode == 1)
    {
      if(kmag * Box / (2 * PI) > Nsample / 2) return 0;
    }
  else
    {
      if(fabs(kx) * Box / (2 * PI) > Nsample / 2) return 0;
      if(fabs(ky) * Box / (2 * PI) > Nsample / 2) return 0;
      if(fabs(kz) * Box / (2 * PI) > Nsample / 2) return 0;
    }
  return 1;
}

static int bpng_search_node_right(int axis, double x)
{
  int lo, hi, mid;
  int n = BPNG.nk[axis];
  lo = 0;
  hi = n;
  while(hi > lo)
    {
      mid = (hi + lo) / 2;
      if(x < bpng_axis_node_x(axis, mid))
        hi = mid;
      else
        lo = mid + 1;
    }
  return lo;
}

static double bpng_linear_input_basis_value(int axis, int node, double kmag)
{
  int n = BPNG.nk[axis];
  int right, i0, i1;
  double x, x0, x1, t;

  if(n <= 0) return 0.0;
  if(kmag < BPNG.edges[axis][0] || kmag > BPNG.edges[axis][n]) return 0.0;
  if(n == 1) return (node == 0) ? 1.0 : 0.0;
  if(node < 0 || node >= n) return 0.0;

  x = bpng_coord_value(kmag);
  if(x <= bpng_axis_node_x(axis, 0))
    return (node == 0) ? 1.0 : 0.0;
  if(x >= bpng_axis_node_x(axis, n - 1))
    return (node == n - 1) ? 1.0 : 0.0;

  right = bpng_search_node_right(axis, x);
  i0 = right - 1;
  if(i0 < 0) i0 = 0;
  if(i0 > n - 2) i0 = n - 2;
  i1 = i0 + 1;

  x0 = bpng_axis_node_x(axis, i0);
  x1 = bpng_axis_node_x(axis, i1);
  if(x1 == x0) return 0.0;
  t = (x - x0) / (x1 - x0);
  if(t < 0.0) t = 0.0;
  if(t > 1.0) t = 1.0;

  if(node == i0) return 1.0 - t;
  if(node == i1) return t;
  return 0.0;
}

static double bpng_cubic_input_basis_value(int axis, int node, double kmag)
{
  int n = BPNG.nk[axis];
  int right, base, m, q, idx[4];
  double x, xm, xq, den, w;

  if(n < 4)
    return bpng_linear_input_basis_value(axis, node, kmag);
  if(kmag < BPNG.edges[axis][0] || kmag > BPNG.edges[axis][n]) return 0.0;
  if(node < 0 || node >= n) return 0.0;

  x = bpng_coord_value(kmag);
  if(x <= bpng_axis_node_x(axis, 0))
    return (node == 0) ? 1.0 : 0.0;
  if(x >= bpng_axis_node_x(axis, n - 1))
    return (node == n - 1) ? 1.0 : 0.0;

  right = bpng_search_node_right(axis, x);
  base = right - 2;
  if(base < 0) base = 0;
  if(base > n - 4) base = n - 4;
  for(m = 0; m < 4; m++)
    idx[m] = base + m;

  for(m = 0; m < 4; m++)
    {
      if(node != idx[m]) continue;
      xm = bpng_axis_node_x(axis, idx[m]);
      w = 1.0;
      for(q = 0; q < 4; q++)
        {
          if(q == m) continue;
          xq = bpng_axis_node_x(axis, idx[q]);
          den = xm - xq;
          if(den == 0.0) return 0.0;
          w *= (x - xq) / den;
        }
      return w;
    }
  return 0.0;
}

static double bpng_input_basis_value(int axis, int node, double kmag)
{
  int ib;
  if(BPNG.input_filter == BPNG_FILTER_TOPHAT)
    {
      ib = bpng_find_bin(BPNG.edges[axis], BPNG.nk[axis], kmag);
      return (ib == node) ? 1.0 : 0.0;
    }
  if(BPNG.input_filter == BPNG_FILTER_LINEAR)
    return bpng_linear_input_basis_value(axis, node, kmag);
  if(BPNG.input_filter == BPNG_FILTER_CUBIC)
    return bpng_cubic_input_basis_value(axis, node, kmag);
  return 0.0;
}

static void bpng_build_input_filtered_field(fftw_complex *dst, fftw_complex *src, int axis, int node)
{
  ptrdiff_t ii, j, k, i;
  uint64_t coord;
  double kx, ky, kz, kmag, filt;

  memset(dst, 0, sizeof(fftw_real) * TotalSizePlusAdditional);

  for(ii = 0; ii < Local_nx; ii++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k <= Nmesh / 2; k++)
        {
          coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
          i = ii + Local_x_start;

          kx = (i < Nmesh / 2) ? i * 2 * PI / Box : -(Nmesh - i) * 2 * PI / Box;
          ky = (j < Nmesh / 2) ? j * 2 * PI / Box : -(Nmesh - j) * 2 * PI / Box;
          kz = (k < Nmesh / 2) ? k * 2 * PI / Box : -(Nmesh - k) * 2 * PI / Box;
          kmag = sqrt(kx*kx + ky*ky + kz*kz);

          if(!bpng_mode_allowed(kx, ky, kz, kmag)) continue;
          filt = bpng_input_basis_value(axis, node, kmag);
          if(filt == 0.0) continue;

          dst[coord].re = filt * src[coord].re;
          dst[coord].im = filt * src[coord].im;
        }
}

static void bpng_multiply_real_fields(fftw_real *out, fftw_real *a, fftw_real *b)
{
  ptrdiff_t i, j, k;
  uint64_t coord;
  memset(out, 0, sizeof(fftw_real) * TotalSizePlusAdditional);
  for(i = 0; i < Local_nx; i++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k < Nmesh; k++)
        {
          coord = (i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k;
          out[coord] = a[coord] * b[coord];
        }
}

static double bpng_output_k3_coefficient(int ia, int ib, int ic, double kmag)
{
  double u, u2;
  double coef;

  coef = bpng_coeff(0, ia, ib, ic);
  if(BPNG.coeff_count <= 1)
    return coef;

  u = bpng_u_in_bin_axis(2, ic, kmag);
  coef += bpng_coeff(1, ia, ib, ic) * u;

  if(BPNG.coeff_count <= 2)
    return coef;

  u2 = u * u;
  coef += bpng_coeff(2, ia, ib, ic) * u2
        + bpng_coeff(3, ia, ib, ic) * u2 * u;
  return coef;
}


static int bpng_internal_edges_are_identical(void)
{
  int a;
  if(BPNG.nk[0] != BPNG.nk[1]) return 0;
  for(a = 0; a <= BPNG.nk[0]; a++)
    if(BPNG.edges[0][a] != BPNG.edges[1][a]) return 0;
  return 1;
}

static double bpng_output_k3_pair_coefficient(int ia, int ib, int ic, double kmag, int use_symmetry)
{
  double coef;
  coef = bpng_output_k3_coefficient(ia, ib, ic, kmag);
  if(use_symmetry && ia != ib)
    coef += bpng_output_k3_coefficient(ib, ia, ic, kmag);
  return coef;
}

static fftw_real **bpng_alloc_real_field_cache(int nfield)
{
  int a;
  fftw_real **cache;
  if(nfield <= 0) return NULL;
  cache = (fftw_real **) malloc(nfield * sizeof(fftw_real *));
  if(!cache) FatalError(9301);
  for(a = 0; a < nfield; a++)
    {
      cache[a] = (fftw_real *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
      if(!cache[a]) FatalError(9302);
    }
  return cache;
}

static void bpng_free_real_field_cache(fftw_real **cache, int nfield)
{
  int a;
  if(!cache) return;
  for(a = 0; a < nfield; a++)
    if(cache[a]) free(cache[a]);
  free(cache);
}

static void bpng_fill_cached_real_field(fftw_real *field, fftw_complex *src, int axis, int node)
{
  bpng_build_input_filtered_field((fftw_complex *) field, src, axis, node);
  MPI_Barrier(MPI_COMM_WORLD);
  rfftwnd_mpi(Inverse_plan, 1, field, Workspace, FFTW_NORMAL_ORDER);
  MPI_Barrier(MPI_COMM_WORLD);
}

static void bpng_convolve_and_accumulate_hybrid(
    fftw_complex *field_a, fftw_real *field_a_real,
    fftw_complex *field_b, fftw_real *field_b_real,
    fftw_complex *c_tmp, fftw_real *p_tmp,
    long double *out_re, long double *out_im,
    int ia, int ib, int use_symmetry);

static void bpng_compute_non_gaussian_potential_symmetric_blocked(
    fftw_complex *cpot, long double *cpot_NG_re, long double *cpot_NG_im,
    fftw_complex *c_tmp, fftw_real *p_tmp, int block_size)
{
  int n, a0, b0, ia, ib, na, nb, la, lb, start_b;
  int timed_chars = 0;
  fftw_real **acache, **bcache;

  n = BPNG.nk[0];
  if(block_size <= 0) block_size = 1;
  if(block_size > n) block_size = n;

  for(a0 = 0; a0 < n; a0 += block_size)
    {
      na = block_size;
      if(a0 + na > n) na = n - a0;

      if(ThisTask == 0)
        {
          timed_chars = printf("  binned PNG internal block a=[%d,%d) of %d...", a0, a0 + na, n);
          fflush(stdout);
        }

      acache = bpng_alloc_real_field_cache(na);
      for(la = 0; la < na; la++)
        bpng_fill_cached_real_field(acache[la], cpot, 0, a0 + la);

      for(b0 = a0; b0 < n; b0 += block_size)
        {
          nb = block_size;
          if(b0 + nb > n) nb = n - b0;

          if(b0 == a0)
            bcache = acache;
          else
            {
              bcache = bpng_alloc_real_field_cache(nb);
              for(lb = 0; lb < nb; lb++)
                bpng_fill_cached_real_field(bcache[lb], cpot, 1, b0 + lb);
            }

          for(la = 0; la < na; la++)
            {
              ia = a0 + la;
              start_b = (b0 == a0) ? la : 0;
              for(lb = start_b; lb < nb; lb++)
                {
                  ib = b0 + lb;
                  bpng_convolve_and_accumulate_hybrid((fftw_complex *) acache[la], acache[la],
                                                      (fftw_complex *) bcache[lb], bcache[lb],
                                                      c_tmp, p_tmp,
                                                      cpot_NG_re, cpot_NG_im, ia, ib, 1);
                }
            }

          if(b0 != a0)
            bpng_free_real_field_cache(bcache, nb);
        }

      bpng_free_real_field_cache(acache, na);
      if(ThisTask == 0) print_timed_done_aligned(timed_chars);
    }
}

static void bpng_accumulate_hybrid_convolution(fftw_complex *conv, long double *out_re, long double *out_im, int ia, int ib, int use_symmetry)
{
  ptrdiff_t ii, j, k, i;
  uint64_t coord;
  double kx, ky, kz, kmag, coef;
  int ic;

  for(ii = 0; ii < Local_nx; ii++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k <= Nmesh / 2; k++)
        {
          coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
          i = ii + Local_x_start;

          kx = (i < Nmesh / 2) ? i * 2 * PI / Box : -(Nmesh - i) * 2 * PI / Box;
          ky = (j < Nmesh / 2) ? j * 2 * PI / Box : -(Nmesh - j) * 2 * PI / Box;
          kz = (k < Nmesh / 2) ? k * 2 * PI / Box : -(Nmesh - k) * 2 * PI / Box;
          kmag = sqrt(kx*kx + ky*ky + kz*kz);

          if(!bpng_mode_allowed(kx, ky, kz, kmag)) continue;
          ic = bpng_find_bin(BPNG.edges[2], BPNG.nk[2], kmag);
          if(ic < 0) continue;

          coef = bpng_output_k3_pair_coefficient(ia, ib, ic, kmag, use_symmetry);
          if(coef == 0.0 || !isfinite(coef)) continue;

          out_re[coord] += (long double) (Fnl * coef * conv[coord].re);
          out_im[coord] += (long double) (Fnl * coef * conv[coord].im);
        }
}

static void bpng_convolve_and_accumulate_hybrid(
    fftw_complex *field_a, fftw_real *field_a_real,
    fftw_complex *field_b, fftw_real *field_b_real,
    fftw_complex *c_tmp, fftw_real *p_tmp,
    long double *out_re, long double *out_im,
    int ia, int ib, int use_symmetry)
{
  bpng_multiply_real_fields(p_tmp, field_a_real, field_b_real);
  MPI_Barrier(MPI_COMM_WORLD);
  rfftwnd_mpi(Forward_plan, 1, p_tmp, Workspace, FFTW_NORMAL_ORDER);
  MPI_Barrier(MPI_COMM_WORLD);
  bpng_accumulate_hybrid_convolution(c_tmp, out_re, out_im, ia, ib, use_symmetry);
}

static void bpng_compute_non_gaussian_potential(fftw_complex *cpot, long double *cpot_NG_re, long double *cpot_NG_im)
{
  int ia, ib, ib_start, use_symmetry, cache_block;
  int timed_chars = 0;
  uint64_t bytes = 0;
  fftw_complex *cf1, *cf2, *c_tmp;
  fftw_real *f1, *f2, *p_tmp;

  if(ThisTask == 0)
    {
      printf("Computing hybrid binned-kernel non-Gaussian potential (k3 is output leg)...\n");
      fflush(stdout);
    }

  bpng_read_and_broadcast_table();

  use_symmetry = (BinnedPNGUseInternalSymmetry != 0);
  if(use_symmetry && !bpng_internal_edges_are_identical())
    {
      if(ThisTask == 0)
        printf("ERROR: BinnedPNGUseInternalSymmetry=1 but k1/k2 edges differ. Disable the flag or regenerate with matched k1/k2 bins.\n");
      FatalError(9310);
    }
  cache_block = BinnedPNGCacheBlockSize;
  if(cache_block < 0) cache_block = 0;

  c_tmp = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
  p_tmp = (fftw_real *) c_tmp;
  ASSERT_ALLOC(c_tmp);

  if(use_symmetry && cache_block > 0)
    {
      if(ThisTask == 0)
        {
          printf("  Using internal-leg symmetry and block cache size %d.\n", cache_block);
          fflush(stdout);
        }
      bpng_compute_non_gaussian_potential_symmetric_blocked(cpot, cpot_NG_re, cpot_NG_im,
                                                            c_tmp, p_tmp, cache_block);
      free(c_tmp);
      bpng_free_table();
      if(ThisTask == 0)
        {
          printf("Finished hybrid binned-kernel non-Gaussian potential.\n");
          fflush(stdout);
        }
      return;
    }

  cf1 = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
  cf2 = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
  f1 = (fftw_real *) cf1;
  f2 = (fftw_real *) cf2;
  ASSERT_ALLOC(cf1 && cf2 && c_tmp);

  if(ThisTask == 0 && use_symmetry)
    {
      printf("  Using internal-leg symmetry without block caching.\n");
      fflush(stdout);
    }

  for(ia = 0; ia < BPNG.nk[0]; ia++)
    {
      if(ThisTask == 0)
        {
          timed_chars = printf("  binned PNG internal k1 basis %d/%d...", ia + 1, BPNG.nk[0]);
          fflush(stdout);
        }

      bpng_build_input_filtered_field(cf1, cpot, 0, ia);
      MPI_Barrier(MPI_COMM_WORLD);
      rfftwnd_mpi(Inverse_plan, 1, f1, Workspace, FFTW_NORMAL_ORDER);
      MPI_Barrier(MPI_COMM_WORLD);

      ib_start = (use_symmetry ? ia : 0);
      for(ib = ib_start; ib < BPNG.nk[1]; ib++)
        {
          if(use_symmetry && ib == ia)
            {
              /* Reuse the already transformed field on the diagonal. */
              bpng_convolve_and_accumulate_hybrid(cf1, f1, cf1, f1, c_tmp, p_tmp,
                                                  cpot_NG_re, cpot_NG_im, ia, ib, use_symmetry);
            }
          else
            {
              bpng_build_input_filtered_field(cf2, cpot, 1, ib);
              MPI_Barrier(MPI_COMM_WORLD);
              rfftwnd_mpi(Inverse_plan, 1, f2, Workspace, FFTW_NORMAL_ORDER);
              MPI_Barrier(MPI_COMM_WORLD);

              bpng_convolve_and_accumulate_hybrid(cf1, f1, cf2, f2, c_tmp, p_tmp,
                                                  cpot_NG_re, cpot_NG_im, ia, ib, use_symmetry);
            }
        }
      if(ThisTask == 0) print_timed_done_aligned(timed_chars);
    }

  free(cf1);
  free(cf2);
  free(c_tmp);
  bpng_free_table();

  if(ThisTask == 0)
    {
      printf("Finished hybrid binned-kernel non-Gaussian potential.\n");
      fflush(stdout);
    }
}




void displacement_fields(void) {
  MPI_Request request;
  MPI_Status status;
  gsl_rng *random_generator;
  ptrdiff_t i, j, k, ii, jj, kk, axes;
  int n;
  int timed_chars = 0;
  int sendTask, recvTask;
  double t_of_k, phig, Beta, twb;
  fftw_complex *(cpot);  /* For computing nongaussian fnl ic */
  fftw_real *(pot);
    
  fftw_complex *(cpot_NG);  /* For computing nongaussian fnl ic */
    
  long double *(cpot_NG_re);
  long double *(cpot_NG_im);
    
  double fac, vel_prefac, vel_prefac2;
  double kvec[3], kmag, kmag2;
  double phase, ampl, hubble_a;
  double u, v, w;
  double f1, f2, f3, f4, f5, f6, f7, f8;
  double dis, dis2, maxdisp, max_disp_glob;
  unsigned int *seedtable;
// ******* FAVN *****
  double phase_shift; 
// ******* FAVN *****
    
  uint64_t bytes, nmesh3;
  uint64_t coord;
  fftw_complex *(cdisp[3]), *(cdisp2[3]) ; /* ZA and 2nd order displacements */
  fftw_real *(disp[3]), *(disp2[3]) ;

  fftw_complex *(cdigrad[6]);
  fftw_real *(digrad[6]);
    
  /* The binned-kernel implementation no longer reads modal AlphaTable or X0Table.
   * It only needs the explicit binned table files configured in the parameter file.
   * Keep Nmax>=Nsample as a sanity check because the input Gaussian modes are
   * generated only over the usual Nsample support.
   */
  assert(Nmax >= Nsample);

  MPI_Bcast(&SavePotentialField, 1, MPI_INT, 0, MPI_COMM_WORLD);

  #ifdef CORRECT_CIC
  double fx, fy, fz, ff, smth;
#endif

  hubble_a    = Hubble * sqrt(Omega / pow(InitTime, 3) + (1 - Omega - OmegaLambda) / pow(InitTime, 2) + OmegaLambda);
  vel_prefac  = InitTime * hubble_a * F_Omega(InitTime) / sqrt(InitTime);
  vel_prefac2 = InitTime * hubble_a * F2_Omega(InitTime) / sqrt(InitTime);

  phase_shift = 0.0;
  if (PhaseFlip==1)
      phase_shift = PI;

  fac = pow(2 * PI / Box, 1.5);

  maxdisp = 0;

  random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);

  gsl_rng_set(random_generator, Seed);

  if(!(seedtable = malloc(Nmesh * Nmesh * sizeof(unsigned int))))
    FatalError(4);

  for(i = 0; i < Nmesh / 2; i++)
    {
      for(j = 0; j < i; j++)
	seedtable[i * Nmesh + j] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[j * Nmesh + i] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i; j++)
	seedtable[(Nmesh - 1 - i) * Nmesh + j] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[(Nmesh - 1 - j) * Nmesh + i] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i; j++)
	seedtable[i * Nmesh + (Nmesh - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[j * Nmesh + (Nmesh - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i; j++)
	seedtable[(Nmesh - 1 - i) * Nmesh + (Nmesh - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[(Nmesh - 1 - j) * Nmesh + (Nmesh - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_generator);
    }


  if (ThisTask == 0) {timed_chars = printf("Setting Gaussian potential..."); fflush(stdout);}

  bytes=0; /*initialize*/
  cpot = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
  pot = (fftw_real *) cpot;
  ASSERT_ALLOC(cpot);

  cpot_NG_re = malloc(sizeof(long double) * TotalSizePlusAdditional);
  cpot_NG_im = malloc(sizeof(long double) * TotalSizePlusAdditional);

  ASSERT_ALLOC(cpot_NG_re);
  ASSERT_ALLOC(cpot_NG_im);

  memset(cpot_NG_re, 0, sizeof(long double) * TotalSizePlusAdditional);
  memset(cpot_NG_im, 0, sizeof(long double) * TotalSizePlusAdditional);
  memset(cpot, 0, sizeof(fftw_real) * TotalSizePlusAdditional);

  /* Ho in units of UnitLength_in_cm and c=1, i.e., internal units so far  */
  Beta = 1.5 * Omega / (2998. * 2998. / UnitLength_in_cm / UnitLength_in_cm * 3.085678e24 * 3.085678e24 ) / D0 ;  
  
  for(i = 0; i < Nmesh; i++)
    {
      ii = Nmesh - i;
      if(ii == Nmesh)
        ii = 0;
      if((i >= Local_x_start && i < (Local_x_start + Local_nx)) ||
         (ii >= Local_x_start && ii < (Local_x_start + Local_nx)))
        {
          for(j = 0; j < Nmesh; j++)
            {
              gsl_rng_set(random_generator, seedtable[i * Nmesh + j]);

              for(k = 0; k < Nmesh / 2; k++)
                {
                  phase = gsl_rng_uniform(random_generator) * 2 * PI;
// ***************** FAVN *****************
                  phase += phase_shift;
// ***************** FAVN *****************
                  do
                    ampl = gsl_rng_uniform(random_generator);

                  while(ampl == 0);

                  if(i == Nmesh / 2 || j == Nmesh / 2 || k == Nmesh / 2)
                    continue;
                  if(i == 0 && j == 0 && k == 0)
                    continue;

                  if(i < Nmesh / 2)
                    kvec[0] = i * 2 * PI / Box;
                  else
                    kvec[0] = -(Nmesh - i) * 2 * PI / Box;

                  if(j < Nmesh / 2)
                    kvec[1] = j * 2 * PI / Box;
                  else
                    kvec[1] = -(Nmesh - j) * 2 * PI / Box;

                  if(k < Nmesh / 2)
                    kvec[2] = k * 2 * PI / Box;
                  else
                    kvec[2] = -(Nmesh - k) * 2 * PI / Box;

                  kmag2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
                  kmag = sqrt(kmag2);

                  if(SphereMode == 1)
                    {
                      if(kmag * Box / (2 * PI) > Nsample / 2)       /* select a sphere in k-space */
                        continue;
                    }
                  else
                    {
                      if(fabs(kvec[0]) * Box / (2 * PI) > Nsample / 2)
                        continue;
                      if(fabs(kvec[1]) * Box / (2 * PI) > Nsample / 2)
                        continue;
                      if(fabs(kvec[2]) * Box / (2 * PI) > Nsample / 2)
                        continue;
                    }

                  phig = Anorm * exp( PrimordialIndex * log(kmag) );   /* initial normalized power */
// ************** FAVN/DSJ ***************
                  if (!FixedAmplitude)
                    phig *= -log(ampl);
// ***************** FAVN/DSJ *************

                  phig = sqrt(phig) * fac * Beta / kmag2;    /* amplitude of the initial gaussian potential */

                  if(k > 0)
                    {
                      if(i >= Local_x_start && i < (Local_x_start + Local_nx))
                           {

                            coord = ((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k;

                            cpot[coord].re = phig * cos(phase);
                            cpot[coord].im = phig * sin(phase);

                           }
                    }
                  else      /* k=0 plane needs special treatment */
                    {
                      if(i == 0)
                        {
                          if(j >= Nmesh / 2)
                            continue;
                          else
                            {
                              if(i >= Local_x_start && i < (Local_x_start + Local_nx))
                                {
                                  jj = Nmesh - j;   /* note: j!=0 surely holds at this point */

                                      coord = ((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k;

                                      cpot[coord].re = phig * cos(phase);
                                      cpot[coord].im = phig * sin(phase);


                                      coord = ((i - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) + k; 
                                      cpot[coord].re =  phig * cos(phase);
                                      cpot[coord].im = -phig * sin(phase);

                                }
                            }
                        }
                      else  /* here comes i!=0 : conjugate can be on other processor! */
                        {
                          if(i >= Nmesh / 2)
                            continue;
                          else
                            {
                              ii = Nmesh - i;
                              if(ii == Nmesh)
                                ii = 0;
                              jj = Nmesh - j;
                              if(jj == Nmesh)
                                jj = 0;

                              if(i >= Local_x_start && i < (Local_x_start + Local_nx))
                                  {

                                    coord = ((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k;

                                    cpot[coord].re = phig * cos(phase);
                                    cpot[coord].im = phig * sin(phase);

                                  }
                              if(ii >= Local_x_start && ii < (Local_x_start + Local_nx))
                                  {
                                    coord = ((ii - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) + k;

                                    cpot[coord].re = phig * cos(phase);
                                    cpot[coord].im = -phig * sin(phase);

                                  }
                            }
                        }
                    }
                }
            }
        }
    }

 if (ThisTask == 0) print_timed_done_aligned(timed_chars);
 /*** For non-local models it is important to keep all factors of SQRT(-1) as done below ***/
 /*** Notice also that there is a minus to convert from Bardeen to gravitational potential ***/

 
  /**** BINNED PRIMORDIAL POTENTIAL **************/

  cpot_NG = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
  ASSERT_ALLOC(cpot_NG);
  memset(cpot_NG, 0, sizeof(fftw_real) * TotalSizePlusAdditional);

  MPI_Barrier(MPI_COMM_WORLD);
  bpng_compute_non_gaussian_potential(cpot, cpot_NG_re, cpot_NG_im);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Move the long-double accumulator into the FFTW complex array used below. */
  for(ii = 0; ii < Local_nx; ii++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k <= Nmesh / 2 ; k++)
        {
          coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
          cpot_NG[coord].re = (double) cpot_NG_re[coord];
          cpot_NG[coord].im = (double) cpot_NG_im[coord];
        }

  free(cpot_NG_re);
  free(cpot_NG_im);

  MPI_Barrier(MPI_COMM_WORLD);

  //Trasnform to real space so we can write out the potential
  rfftwnd_mpi(Inverse_plan, 1, pot, Workspace, FFTW_NORMAL_ORDER);
  if(SavePotentialField==1) write_phi(pot, 0);
  // if(ThisTask == 0) printf("\n coord = 10 \n pot = %lg\n\n", pot[10]);
  MPI_Barrier(MPI_COMM_WORLD);   
  rfftwnd_mpi(Forward_plan, 1, pot, Workspace, FFTW_NORMAL_ORDER);

  nmesh3 = ((uint64_t) Nmesh) * ((uint64_t) Nmesh ) * ((uint64_t) Nmesh);    

  //Add NG potential to the full potential
  for(ii = 0; ii < Local_nx; ii++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k <= Nmesh / 2 ; k++){

          coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;

          //One factor of nmesh3 due to the FFT from the save potential step above
          cpot[coord].re /= (double) nmesh3; 
          cpot[coord].im /= (double) nmesh3; 
          
          //One factor due to converting c_tmp from real space to fourier space
          cpot[coord].re += cpot_NG[coord].re / (double) nmesh3; 
          cpot[coord].im += cpot_NG[coord].im / (double) nmesh3;
          
          
      }
    
  
  //Zero-out the mean mode alone
  if(ThisTask == 0) {
          cpot[0].re = 0.;
          cpot[0].im = 0.; 
       }

  
  free(cpot_NG);

MPI_Barrier(MPI_COMM_WORLD);
    
if (ThisTask == 0) {timed_chars = printf("Writing potential to disc..."); fflush(stdout);}
rfftwnd_mpi(Inverse_plan, 1, pot, Workspace, FFTW_NORMAL_ORDER);
if(SavePotentialField==1) write_phi(pot, 1);
MPI_Barrier(MPI_COMM_WORLD);   
rfftwnd_mpi(Forward_plan, 1, pot, Workspace, FFTW_NORMAL_ORDER);
    
for(ii = 0; ii < Local_nx; ii++){
    for(j = 0; j < Nmesh; j++){
      for(k = 0; k <= Nmesh / 2 ; k++){

          coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;

          cpot[coord].re = cpot[coord].re / (double) nmesh3;
          cpot[coord].im = cpot[coord].im / (double) nmesh3;

      }
    }
  }
if (ThisTask == 0) print_timed_done_aligned(timed_chars);

MPI_Barrier(MPI_COMM_WORLD);
if (ThisTask == 0) {timed_chars = printf("Computing gradient of non-Gaussian potential..."); fflush(stdout);}

/*****  Now 2LPT ****/

for(axes=0,bytes=0; axes < 3; axes++)
{
  cdisp[axes] = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
  disp[axes] = (fftw_real *) cdisp[axes];
}

ASSERT_ALLOC(cdisp[0] && cdisp[1] && cdisp[2]);


{

  /* first, clean the array */
  for(i = 0; i < Local_nx; i++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k <= Nmesh / 2; k++)
        for(axes = 0; axes < 3; axes++)
          {

        coord = (i * Nmesh + j) * (Nmesh / 2 + 1) + k;
        cdisp[axes][coord].re = 0;
        cdisp[axes][coord].im = 0;
          }


  for(ii = 0; ii < Local_nx; ii++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k <= Nmesh / 2 ; k++)
        {

          coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
          i = ii + Local_x_start;

               /*   if(i == 0 && j == 0 && k == 0); continue; */

                  if(i < Nmesh / 2)
                    kvec[0] = i * 2 * PI / Box;
                  else
                    kvec[0] = -(Nmesh - i) * 2 * PI / Box;

                  if(j < Nmesh / 2)
                    kvec[1] = j * 2 * PI / Box;
                  else
                    kvec[1] = -(Nmesh - j) * 2 * PI / Box;

                  if(k < Nmesh / 2)
                    kvec[2] = k * 2 * PI / Box;
                  else
                    kvec[2] = -(Nmesh - k) * 2 * PI / Box;

                  kmag2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
                  kmag = sqrt(kmag2);

                 t_of_k = TransferFunc(kmag);

                 twb = t_of_k / Dplus  / Beta;   

                 for(axes = 0; axes < 3; axes++)   
                                    {
                            cdisp[axes][coord].im = kvec[axes] * twb * cpot[coord].re;
                            cdisp[axes][coord].re = - kvec[axes] * twb * cpot[coord].im;
                                    }
        }


  free(cpot);

  if (ThisTask == 0) print_timed_done_aligned(timed_chars);


       MPI_Barrier(MPI_COMM_WORLD);
 
      /* Compute displacement gradient */

      if (ThisTask == 0) {timed_chars = printf("Computing 2LPT potential..."); fflush(stdout);}

      for(i = 0; i < 6; i++)
	{
	  cdigrad[i] = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
	  digrad[i] = (fftw_real *) cdigrad[i];
	  ASSERT_ALLOC(cdigrad[i]);
	}
      
      for(i = 0; i < Local_nx; i++)
	for(j = 0; j < Nmesh; j++)
	  for(k = 0; k <= Nmesh / 2; k++)
	    {
	      coord = (i * Nmesh + j) * (Nmesh / 2 + 1) + k;
	      if((i + Local_x_start) < Nmesh / 2)
		kvec[0] = (i + Local_x_start) * 2 * PI / Box;
	      else
		kvec[0] = -(Nmesh - (i + Local_x_start)) * 2 * PI / Box;
	      
	      if(j < Nmesh / 2)
		kvec[1] = j * 2 * PI / Box;
	      else
		kvec[1] = -(Nmesh - j) * 2 * PI / Box;
	      
	      if(k < Nmesh / 2)
		kvec[2] = k * 2 * PI / Box;
	      else
		kvec[2] = -(Nmesh - k) * 2 * PI / Box;
	      
	      /* Derivatives of ZA displacement  */
	      /* d(dis_i)/d(q_j)  -> sqrt(-1) k_j dis_i */
	      cdigrad[0][coord].re = -cdisp[0][coord].im * kvec[0]; /* disp0,0 */
	      cdigrad[0][coord].im = cdisp[0][coord].re * kvec[0];

	      cdigrad[1][coord].re = -cdisp[0][coord].im * kvec[1]; /* disp0,1 */
	      cdigrad[1][coord].im = cdisp[0][coord].re * kvec[1];

	      cdigrad[2][coord].re = -cdisp[0][coord].im * kvec[2]; /* disp0,2 */
	      cdigrad[2][coord].im = cdisp[0][coord].re * kvec[2];
	      
	      cdigrad[3][coord].re = -cdisp[1][coord].im * kvec[1]; /* disp1,1 */
	      cdigrad[3][coord].im = cdisp[1][coord].re * kvec[1];

	      cdigrad[4][coord].re = -cdisp[1][coord].im * kvec[2]; /* disp1,2 */
	      cdigrad[4][coord].im = cdisp[1][coord].re * kvec[2];

	      cdigrad[5][coord].re = -cdisp[2][coord].im * kvec[2]; /* disp2,2 */
	      cdigrad[5][coord].im = cdisp[2][coord].re * kvec[2];
	    }


      for(i = 0; i < 6; i++) rfftwnd_mpi(Inverse_plan, 1, digrad[i], Workspace, FFTW_NORMAL_ORDER);

      /* Compute second order source and store it in digrad[3]*/

      for(i = 0; i < Local_nx; i++)
	for(j = 0; j < Nmesh; j++)
	  for(k = 0; k < Nmesh; k++)
	    {
	      coord = (i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k;

	      digrad[3][coord] =

		digrad[0][coord]*(digrad[3][coord]+digrad[5][coord])+digrad[3][coord]*digrad[5][coord]
                -digrad[1][coord]*digrad[1][coord]-digrad[2][coord]*digrad[2][coord]-digrad[4][coord]*digrad[4][coord];
	    }

      rfftwnd_mpi(Forward_plan, 1, digrad[3], Workspace, FFTW_NORMAL_ORDER);
      
      /* The memory allocated for cdigrad[0], [1], and [2] will be used for 2nd order displacements */
      /* Freeing the rest. cdigrad[3] still has 2nd order displacement source, free later */

      for(axes = 0; axes < 3; axes++) 
	{
	  cdisp2[axes] = cdigrad[axes]; 
	  disp2[axes] = (fftw_real *) cdisp2[axes];
	}

      free(cdigrad[4]); free(cdigrad[5]); 

      /* Solve Poisson eq. and calculate 2nd order displacements */

      for(i = 0; i < Local_nx; i++)
	for(j = 0; j < Nmesh; j++)
	  for(k = 0; k <= Nmesh / 2; k++)
	    {
	      coord = (i * Nmesh + j) * (Nmesh / 2 + 1) + k;
	      if((i + Local_x_start) < Nmesh / 2)
		kvec[0] = (i + Local_x_start) * 2 * PI / Box;
	      else
		kvec[0] = -(Nmesh - (i + Local_x_start)) * 2 * PI / Box;
	      
	      if(j < Nmesh / 2)
		kvec[1] = j * 2 * PI / Box;
	      else
		kvec[1] = -(Nmesh - j) * 2 * PI / Box;
	      
	      if(k < Nmesh / 2)
		kvec[2] = k * 2 * PI / Box;
	      else
		kvec[2] = -(Nmesh - k) * 2 * PI / Box;

	      kmag2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
#ifdef CORRECT_CIC
	      /* calculate smooth factor for deconvolution of CIC interpolation */
	      fx = fy = fz = 1;
	      if(kvec[0] != 0)
		{
		  fx = (kvec[0] * Box / 2) / Nmesh;
		  fx = sin(fx) / fx;
		}
	      if(kvec[1] != 0)
		{
		  fy = (kvec[1] * Box / 2) / Nmesh;
		  fy = sin(fy) / fy;
		}
	      if(kvec[2] != 0)
		{
		  fz = (kvec[2] * Box / 2) / Nmesh;
		  fz = sin(fz) / fz;
		}
	      ff = 1 / (fx * fy * fz);
	      smth = ff * ff;
	      /*  */
#endif

	      /* cdisp2 = source * k / (sqrt(-1) k^2) */
	      for(axes = 0; axes < 3; axes++)
		{
		  if(kmag2 > 0.0) 
		    {
		      cdisp2[axes][coord].re = cdigrad[3][coord].im * kvec[axes] / kmag2;
		      cdisp2[axes][coord].im = -cdigrad[3][coord].re * kvec[axes] / kmag2;
		    }
		  else cdisp2[axes][coord].re = cdisp2[axes][coord].im = 0.0;
#ifdef CORRECT_CIC
		  cdisp[axes][coord].re *= smth;   cdisp[axes][coord].im *= smth;
		  cdisp2[axes][coord].re *= smth;  cdisp2[axes][coord].im *= smth;
#endif
		}
	    }
      
      /* Free cdigrad[3] */
      free(cdigrad[3]);

      MPI_Barrier(MPI_COMM_WORLD);

      /* Now, both cdisp, and cdisp2 have the ZA and 2nd order displacements */

      for(axes = 0; axes < 3; axes++)
	{
	  rfftwnd_mpi(Inverse_plan, 1, disp[axes], Workspace, FFTW_NORMAL_ORDER);
	  rfftwnd_mpi(Inverse_plan, 1, disp2[axes], Workspace, FFTW_NORMAL_ORDER);

	  /* now get the plane on the right side from neighbour on the right, 
	     and send the left plane */
      
	  recvTask = ThisTask;
	  do
	    {
	      recvTask--;
	      if(recvTask < 0)
		recvTask = NTask - 1;
	    }
	  while(Local_nx_table[recvTask] == 0);
      
	  sendTask = ThisTask;
	  do
	    {
	      sendTask++;
	      if(sendTask >= NTask)
		sendTask = 0;
	    }
	  while(Local_nx_table[sendTask] == 0);
      
	  /* use non-blocking send */
      
	  if(Local_nx > 0)
	    {
	      /* send ZA disp */
	      MPI_Isend(&(disp[axes][0]),
			sizeof(fftw_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
			MPI_BYTE, recvTask, 10, MPI_COMM_WORLD, &request);
	      
	      MPI_Recv(&(disp[axes][(Local_nx * Nmesh) * (2 * (Nmesh / 2 + 1))]),
		       sizeof(fftw_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
		       MPI_BYTE, sendTask, 10, MPI_COMM_WORLD, &status);
	      
	      MPI_Wait(&request, &status);

	      
	      /* send 2nd order disp */
	      MPI_Isend(&(disp2[axes][0]),
			sizeof(fftw_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
			MPI_BYTE, recvTask, 10, MPI_COMM_WORLD, &request);
	      
	      MPI_Recv(&(disp2[axes][(Local_nx * Nmesh) * (2 * (Nmesh / 2 + 1))]),
		       sizeof(fftw_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
		       MPI_BYTE, sendTask, 10, MPI_COMM_WORLD, &status);
	      
	      MPI_Wait(&request, &status);
	    }
	}

      if (ThisTask == 0) print_timed_done_aligned(timed_chars);
      if (ThisTask == 0) {timed_chars = printf("Computing displacements and velocitites..."); fflush(stdout);}

      /* read-out displacements */

      nmesh3 = (uint64_t)Nmesh * (uint64_t)Nmesh *(uint64_t) Nmesh;
      
      for(n = 0; n < NumPart; n++)
	{
          
	    {
	      u = P[n].Pos[0] / Box * Nmesh;
	      v = P[n].Pos[1] / Box * Nmesh;
	      w = P[n].Pos[2] / Box * Nmesh;
	      
	      i = (int) u;
	      j = (int) v;
	      k = (int) w;
	      
	      if(i == (Local_x_start + Local_nx))
		i = (Local_x_start + Local_nx) - 1;
	      if(i < Local_x_start)
		i = Local_x_start;
	      if(j == Nmesh)
		j = Nmesh - 1;
	      if(k == Nmesh)
		k = Nmesh - 1;
	      
	      u -= i;
	      v -= j;
	      w -= k;
	      
	      i -= Local_x_start;
	      ii = i + 1;
	      jj = j + 1;
	      kk = k + 1;
	      
	      if(jj >= Nmesh)
		jj -= Nmesh;
	      if(kk >= Nmesh)
		kk -= Nmesh;
	      
	      f1 = (1 - u) * (1 - v) * (1 - w);
	      f2 = (1 - u) * (1 - v) * (w);
	      f3 = (1 - u) * (v) * (1 - w);
	      f4 = (1 - u) * (v) * (w);
	      f5 = (u) * (1 - v) * (1 - w);
	      f6 = (u) * (1 - v) * (w); 
	      f7 = (u) * (v) * (1 - w);
	      f8 = (u) * (v) * (w);
	     
	      for(axes = 0; axes < 3; axes++)
		{
		  dis = disp[axes][(i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k] * f1 +
		    disp[axes][(i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + kk] * f2 +
		    disp[axes][(i * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + k] * f3 +
		    disp[axes][(i * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + kk] * f4 +
		    disp[axes][(ii * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k] * f5 +
		    disp[axes][(ii * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + kk] * f6 +
		    disp[axes][(ii * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + k] * f7 +
		    disp[axes][(ii * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + kk] * f8;

		  dis2 = disp2[axes][(i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k] * f1 +
		    disp2[axes][(i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + kk] * f2 +
		    disp2[axes][(i * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + k] * f3 +
		    disp2[axes][(i * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + kk] * f4 +
		    disp2[axes][(ii * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k] * f5 +
		    disp2[axes][(ii * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + kk] * f6 +
		    disp2[axes][(ii * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + k] * f7 +
		    disp2[axes][(ii * Nmesh + jj) * (2 * (Nmesh / 2 + 1)) + kk] * f8;
		  dis2 /= (float) nmesh3;
	      
		  
		  P[n].Pos[axes] += dis - 3./7. * dis2;
		  P[n].Vel[axes] = dis * vel_prefac - 3./7. * dis2 * vel_prefac2;

		  P[n].Pos[axes] = periodic_wrap(P[n].Pos[axes]);

		  if(fabs(dis - 3./7. * dis2 > maxdisp))
		    maxdisp = fabs(dis - 3./7. * dis2);
		}
	    }
	}
    }
 
      if (ThisTask == 0) print_timed_done_aligned(timed_chars);

  for(axes = 0; axes < 3; axes++) free(cdisp[axes]);
  for(axes = 0; axes < 3; axes++) free(cdisp2[axes]);

  gsl_rng_free(random_generator);

  MPI_Reduce(&maxdisp, &max_disp_glob, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

/*  if(ThisTask == 0)
    {
      printf("\nMaximum displacement (1D): %g kpc/h, in units of the part-spacing= %g\n",
	     max_disp_glob, max_disp_glob / (Box / Nmesh));
    }*/
}

double periodic_wrap(double x)
{
  return fmod(fmod(x, Box) + Box, Box);
  // while(x >= Box)
  //   x -= Box;

  // while(x < 0)
  //   x += Box;

  // return x;
}


void set_units(void)		/* ... set some units */
{
  UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s;
  G = GRAVITY / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2);
  Hubble = HUBBLE * UnitTime_in_s;
}



void initialize_ffts(void)
{
  
  int total_size, i, additional;
  int local_ny_after_transpose, local_y_start_after_transpose;
  int *slab_to_task_local;
  size_t bytes;

  MPI_Barrier(MPI_COMM_WORLD);
  
  Inverse_plan = rfftw3d_mpi_create_plan(MPI_COMM_WORLD,
					 Nmesh, Nmesh, Nmesh, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE);

  Forward_plan = rfftw3d_mpi_create_plan(MPI_COMM_WORLD,
					 Nmesh, Nmesh, Nmesh, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);

  rfftwnd_mpi_local_sizes(Forward_plan, &Local_nx, &Local_x_start,
			  &local_ny_after_transpose, &local_y_start_after_transpose, &total_size);

  Local_nx_table = malloc(sizeof(int) * NTask);
  MPI_Allgather(&Local_nx, 1, MPI_INT, Local_nx_table, 1, MPI_INT, MPI_COMM_WORLD);

  Slab_to_task = malloc(sizeof(int) * Nmesh);
  slab_to_task_local = malloc(sizeof(int) * Nmesh);

  for(i = 0; i < Nmesh; i++)
    slab_to_task_local[i] = 0;

  for(i = 0; i < Local_nx; i++)
    slab_to_task_local[Local_x_start + i] = ThisTask;

  MPI_Allreduce(slab_to_task_local, Slab_to_task, Nmesh, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  free(slab_to_task_local);

  additional = (Nmesh) * (2 * (Nmesh / 2 + 1));	/* additional plane on the right side */

  TotalSizePlusAdditional = total_size + additional;

  Workspace = (fftw_real *) malloc(sizeof(fftw_real) * total_size);

  ASSERT_ALLOC(Workspace)

}



void free_ffts(void)
{
  free(Workspace);
  free(Slab_to_task);
  rfftwnd_mpi_destroy_plan(Inverse_plan);
  rfftwnd_mpi_destroy_plan(Forward_plan);
}


int FatalError(int errnum)
{
  printf("FatalError called with number=%d\n", errnum);
  fflush(stdout);
  MPI_Abort(MPI_COMM_WORLD, errnum);
  exit(0);
}




static double A, B, alpha, beta, V, gf;

double fnl(double x)		/* Peacock & Dodds formula */
{
  return x * pow((1 + B * beta * x + pow(A * x, alpha * beta)) /
		 (1 + pow(pow(A * x, alpha) * gf * gf * gf / (V * sqrt(x)), beta)), 1 / beta);
}

void print_spec(void)
{
  double k, knl, po, dl, dnl, neff, kf, kstart, kend, po2, po1, DDD;
  char buf[1100];
  FILE *fd;

  if(ThisTask == 0)
    {
      sprintf(buf, "%s/inputspec_%s.txt", OutputDir, FileBase);

      fd = fopen(buf, "w");

      gf = GrowthFactor(0.001, 1.0) / (1.0 / 0.001);

      DDD = GrowthFactor(1.0 / (Redshift + 1), 1.0);

      fprintf(fd, "%12g %12g\n", Redshift, DDD);	/* print actual starting redshift and 
							   linear growth factor for this cosmology */

      kstart = 2 * PI / (1000.0 * (3.085678e24 / UnitLength_in_cm));	/* 1000 Mpc/h */
      kend = 2 * PI / (0.001 * (3.085678e24 / UnitLength_in_cm));	/* 0.001 Mpc/h */

      for(k = kstart; k < kend; k *= 1.025)
	{
	  po = PowerSpec(k);
	  dl = 4.0 * PI * k * k * k * po;

	  kf = 0.5;

	  po2 = PowerSpec(1.001 * k * kf);
	  po1 = PowerSpec(k * kf);

	  if(po != 0 && po1 != 0 && po2 != 0)
	    {
	      neff = (log(po2) - log(po1)) / (log(1.001 * k * kf) - log(k * kf));

	      if(1 + neff / 3 > 0)
		{
		  A = 0.482 * pow(1 + neff / 3, -0.947);
		  B = 0.226 * pow(1 + neff / 3, -1.778);
		  alpha = 3.310 * pow(1 + neff / 3, -0.244);
		  beta = 0.862 * pow(1 + neff / 3, -0.287);
		  V = 11.55 * pow(1 + neff / 3, -0.423) * 1.2;

		  dnl = fnl(dl);
		  knl = k * pow(1 + dnl, 1.0 / 3);
		}
	      else
		{
		  dnl = 0;
		  knl = 0;
		}
	    }
	  else
	    {
	      dnl = 0;
	      knl = 0;
	    }

	  fprintf(fd, "%12g %12g    %12g %12g\n", k, dl, knl, dnl);
	}
      fclose(fd);
    }
}