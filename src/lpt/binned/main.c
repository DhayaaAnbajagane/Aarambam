/* Binned method of 2LPTBinned: banner and the binned-kernel non-Gaussian potential; the shared
 * steps are in common/lpt_core.c. */
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
#include "binned_vars.h"
#include "proto.h"
#include <ctype.h>

#include "lpt_core.h"


void print_setup(void)
{
  lpt_print_banner("2LPTNGBinned");

  printf("      Box = %.2e   Nmesh = %d   Nsample = %d   Nmax = %d\n", Box, Nmesh, Nsample, Nmax);
  printf("                       Nglass = %d    GlassTileFac = %d\n\n", Nglass, GlassTileFac);
  printf("        Omega = %.4f         OmegaLambda = %.4f    OmegaBaryon = %.2e\n", Omega,  OmegaLambda, OmegaBaryon);
  printf("       sigma8 = %.4f     PrimoridalIndex = %.4f       Redshift = %.2e\n", Sigma8, PrimordialIndex, Redshift);
  printf("  HubbleParam = %.4f  OmegaDM_2ndSpecies = %.2e          fNL = %+.2e\n\n",
         HubbleParam, OmegaDM_2ndSpecies, Fnl);
  printf("   FixedAmplitude = %d    PhaseFlip = % d   SphereMode = %d    Seed = %d\n",
         FixedAmplitude, PhaseFlip, SphereMode, Seed);
  printf("   BinnedPNGFilter(input k1/k2) = %d    BinnedPNGOutputFilter(k3) = %d    BinnedPNGCoordinate = %d\n",
         BinnedPNGFilter, BinnedPNGOutputFilter, BinnedPNGCoordinate);
  printf("   BinnedPNGMultiGrid = %d    BinnedPNGCacheBlockSize = %d\n", BinnedPNGMultiGrid, BinnedPNGCacheBlockSize);
  printf("   BinnedK1Edges(k1 = k2) = %s\n", FileWithBinnedK1Edges);
  printf("   BinnedK3Edges(output) = %s\n", FileWithBinnedK3Edges);
  printf("   BinnedKernelValues = %s\n", FileWithBinnedKernelValues);

  printf("\n");
  lpt_print_rule();
  printf("\n");
  return;
}


/* The binned kernel (FileWithBinnedKernelValues, written by Aarambam.basis.binned) is
 *
 *   K(k1,k2,k3) = sum_{a,b,c} psi_a(k1) psi_b(k2) H_c(k3) [K0_abc + K1_abc u3 + K2_abc u3^2 + K3_abc u3^3],
 *
 * with k1, k2 the internal legs, k3 = |k| the output mode, H_c the indicator of k3 bin c and u3 the coordinate
 * within it. Each (a,b) pair costs one convolution; the k3 polynomial is applied mode by mode afterwards.
 *   BinnedPNGFilter (psi_a, psi_b):  1 top-hat, 2 linear, 3 cubic interpolation between the bin centres
 *   BinnedPNGOutputFilter (in u3):   1 K0, 2 up to K1 u3, 3 up to K3 u3^3
 *   BinnedPNGCoordinate:             1 k, 2 log k (the Python side writes 2, the only one it supports) */

#define BPNG_FILTER_TOPHAT    1
#define BPNG_FILTER_LINEAR    2
#define BPNG_FILTER_CUBIC     3
#define BPNG_COORD_K          1
#define BPNG_COORD_LOG        2

typedef struct
{
  int     nk[3];
  int     input_filter;      /* k1,k2 basis: top-hat/linear/cubic */
  int     output_filter;     /* k3 output polynomial: top-hat/linear/cubic */
  int     coord_mode;
  int     coeff_count;       /* 1, 2, or 4 output coefficients along k3 */
  double *edges[3];
  double *Kcoeff;            /* coeff-major: Kcoeff[q*Ncell + ((i*nk2+j)*nk3+l)] */
  double *node_x[3];         /* coordinate (k or log k) of each node, computed once in bpng_precompute_axes */
  double *bin_xc[3];         /* centre and width of each bin in that coordinate */
  double *bin_dx[3];
}
BinnedPNGTable;

static BinnedPNGTable BPNG = {{0, 0, 0}, BPNG_FILTER_TOPHAT, BPNG_FILTER_TOPHAT,
                              BPNG_COORD_K, 0, {NULL, NULL, NULL}, NULL,
                              {NULL, NULL, NULL}, {NULL, NULL, NULL}, {NULL, NULL, NULL}};

/* ModeTables 1: this rank's allowed modes, with |k|, output (k3) bin and position u in that bin, computed once
 * instead of in each of the ~Nk^2/2 accumulations and ~2 Nk filter builds. ~28 bytes per allowed mode. */
typedef struct
{
  long long  n;
  uint64_t  *coord;
  double    *kmag, *u;
  int       *ic;
}
BpngModeList;

static BpngModeList BML  = {0, NULL, NULL, NULL, NULL};   /* normal layout: input filter build */
static BpngModeList BMLT = {0, NULL, NULL, NULL, NULL};   /* band layout: NG accumulation */


/* ---- kernel table: cell indexing, reading and broadcasting ---- */

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
  return BPNG.Kcoeff[((long long) q) * bpng_ncell() + bpng_cell_index(i, j, l)];
}

static int bpng_parse_doubles(char *line, double *vals, int maxvals)
{
  int   n   = 0;
  char *p   = line;
  char *end = NULL;

  while(*p && n < maxvals)
    {
      while(*p && isspace((unsigned char) *p))
        p++;
      if(*p == 0 || *p == '#')
        break;

      vals[n] = strtod(p, &end);
      if(end == p)
        break;
      n++;
      p = end;
    }
  return n;
}

static int bpng_count_edge_file(const char *path)
{
  FILE  *fd = fopen(path, "r");
  char   line[1024];
  double tmp[4];
  int    n = 0;

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
  FILE  *fd = fopen(path, "r");
  char   line[1024];
  double tmp[4];
  int    n = 0;

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
  char      path[1000], line[8192];
  FILE     *fd;
  double    vals[80];
  long long nc    = bpng_ncell();
  long long q;
  int       nmin;
  long long nrows = 0;
  char     *seen;

  for(q = 0; q < ((long long) BPNG.coeff_count) * nc; q++)
    BPNG.Kcoeff[q] = 0.0;

  snprintf(path, sizeof(path), "%s", FileWithBinnedKernelValues);

  fd = fopen(path, "r");
  if(!fd)
    {
      printf("Could not open binned kernel coefficient file '%s' on task %d\n", path, ThisTask);
      FatalError(9103);
    }

  /* Columns: i j l k1_center k2_center k3_center, B[ncoef] K[ncoef] D[ncoef] S[ncoef], valid_center valid_box;
   * the K block starts at column 6 + ncoef. */
  nmin = 8 + 4 * BPNG.coeff_count;

  /* Every row must have the column count implied by BinnedPNGOutputFilter and every cell must appear once, so a
   * table written for another filter is an error, not a wrong kernel. */
  seen = (char *) calloc(nc, 1);
  if(!seen)
    FatalError(9112);

  while(fgets(line, sizeof(line), fd))
    {
      int       n = bpng_parse_doubles(line, vals, 80);
      int       i, j, l, c, k_start;
      long long ind;

      if(n <= 0)
        continue;

      if(n != nmin)
        {
          printf("Binned kernel table '%s': a row has %d columns, but BinnedPNGOutputFilter %d needs %d "
                 "(6 + 4 x %d coefficients + 2). Was the table written with a different Binned_filter?\n",
                 path, n, BinnedPNGOutputFilter, nmin, BPNG.coeff_count);
          FatalError(9113);
        }

      i = (int) vals[0];
      j = (int) vals[1];
      l = (int) vals[2];
      if(i < 0 || i >= BPNG.nk[0] || j < 0 || j >= BPNG.nk[1] || l < 0 || l >= BPNG.nk[2])
        {
          printf("Binned kernel table '%s': cell (%d, %d, %d) is outside the edge files' Nk = (%d, %d, %d)\n",
                 path, i, j, l, BPNG.nk[0], BPNG.nk[1], BPNG.nk[2]);
          FatalError(9114);
        }

      ind = bpng_cell_index(i, j, l);
      if(seen[ind])
        {
          printf("Binned kernel table '%s': cell (%d, %d, %d) appears more than once\n", path, i, j, l);
          FatalError(9114);
        }
      seen[ind] = 1;
      nrows++;

      k_start = 6 + BPNG.coeff_count;
      for(c = 0; c < BPNG.coeff_count; c++)
        BPNG.Kcoeff[((long long) c) * nc + ind] = vals[k_start + c];
    }

  fclose(fd);
  free(seen);

  if(nrows != nc)
    {
      printf("Binned kernel table '%s' has %lld cells, but the edge files define %lld\n", path, nrows, nc);
      FatalError(9115);
    }
}

static void bpng_read_and_broadcast_table(void)
{
  int       axis, nedge[3], input_filter, output_filter, coord_mode;
  long long ncoef;

  if(ThisTask == 0)
    {
      input_filter  = BinnedPNGFilter;
      output_filter = BinnedPNGOutputFilter;
      coord_mode    = BinnedPNGCoordinate;

      if(input_filter != BPNG_FILTER_TOPHAT &&
         input_filter != BPNG_FILTER_LINEAR &&
         input_filter != BPNG_FILTER_CUBIC)
        {
          printf("Invalid BinnedPNGFilter=%d. Use 1=top-hat, 2=linear input interpolation, "
                 "3=cubic input interpolation.\n", input_filter);
          FatalError(9109);
        }

      if(output_filter != BPNG_FILTER_TOPHAT &&
         output_filter != BPNG_FILTER_LINEAR &&
         output_filter != BPNG_FILTER_CUBIC)
        {
          printf("Invalid BinnedPNGOutputFilter=%d. Use 1=heaviside, 2=linear k3 output, 3=cubic k3 output.\n",
                 output_filter);
          FatalError(9111);
        }

      if(coord_mode != BPNG_COORD_K && coord_mode != BPNG_COORD_LOG)
        {
          printf("Invalid BinnedPNGCoordinate=%d. Use 1=k, 2=log(k).\n", coord_mode);
          FatalError(9110);
        }

      BPNG.input_filter  = input_filter;
      BPNG.output_filter = output_filter;
      BPNG.coord_mode    = coord_mode;
      BPNG.coeff_count   = (output_filter == BPNG_FILTER_CUBIC)  ? 4 :
                           (output_filter == BPNG_FILTER_LINEAR) ? 2 : 1;

      nedge[0] = bpng_count_edge_file(FileWithBinnedK1Edges);
      nedge[1] = nedge[0];                                         /* k2 shares the k1 edges */
      nedge[2] = bpng_count_edge_file(FileWithBinnedK3Edges);

      for(axis = 0; axis < 3; axis++)
        {
          if(nedge[axis] < 2)
            FatalError(9104);

          BPNG.nk[axis]    = nedge[axis] - 1;
          BPNG.edges[axis] = malloc(nedge[axis] * sizeof(double));
          if(!BPNG.edges[axis])
            FatalError(9105);
        }

      bpng_read_edge_file_root(FileWithBinnedK1Edges, 0);
      memcpy(BPNG.edges[1], BPNG.edges[0], nedge[0] * sizeof(double));
      bpng_read_edge_file_root(FileWithBinnedK3Edges, 2);
      for(axis = 0; axis < 3; axis += 2)
        for(int e = 1; e < nedge[axis]; e++)
          if(!(BPNG.edges[axis][e] > BPNG.edges[axis][e - 1]))
            {
              printf("Binned edge file '%s': the edges must increase strictly (edge %d is %g after %g).\n",
                     axis ? FileWithBinnedK3Edges : FileWithBinnedK1Edges, e, BPNG.edges[axis][e],
                     BPNG.edges[axis][e - 1]);
              FatalError(9119);
            }

      ncoef       = ((long long) BPNG.coeff_count) * bpng_ncell();
      BPNG.Kcoeff = malloc(ncoef * sizeof(double));
      if(!BPNG.Kcoeff)
        FatalError(9106);
      bpng_read_kernel_coefficients_root();

      printf("Loaded hybrid binned PNG kernel table: Nk=(%d,%d,%d), input_filter=%d, output_filter=%d, "
             "coord=%d, coeff_count=%d\n",
             BPNG.nk[0], BPNG.nk[1], BPNG.nk[2],
             BPNG.input_filter, BPNG.output_filter, BPNG.coord_mode, BPNG.coeff_count);
    }

  MPI_Bcast(BPNG.nk,             3, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&BPNG.input_filter,  1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&BPNG.output_filter, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&BPNG.coord_mode,    1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&BPNG.coeff_count,   1, MPI_INT, 0, MPI_COMM_WORLD);

  if(ThisTask != 0)
    {
      for(axis = 0; axis < 3; axis++)
        {
          nedge[axis]      = BPNG.nk[axis] + 1;
          BPNG.edges[axis] = malloc(nedge[axis] * sizeof(double));
          if(!BPNG.edges[axis])
            FatalError(9107);
        }

      BPNG.Kcoeff = malloc(((long long) BPNG.coeff_count) * bpng_ncell() * sizeof(double));
      if(!BPNG.Kcoeff)
        FatalError(9108);
    }

  for(axis = 0; axis < 3; axis++)
    MPI_Bcast(BPNG.edges[axis], BPNG.nk[axis] + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(BPNG.Kcoeff, ((long long) BPNG.coeff_count) * bpng_ncell(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

static void bpng_free_table(void)
{
  free(BPNG.Kcoeff);
  BPNG.Kcoeff = NULL;
  for(int axis = 0; axis < 3; axis++)
    {
      free(BPNG.edges[axis]);
      free(BPNG.node_x[axis]);
      free(BPNG.bin_xc[axis]);
      free(BPNG.bin_dx[axis]);
      BPNG.edges[axis] = BPNG.node_x[axis] = BPNG.bin_xc[axis] = BPNG.bin_dx[axis] = NULL;
    }
}


/* ---- bins, coordinates and the input-leg filters ---- */

static int bpng_find_bin(const double *edges, int nbin, double k)
{
  int lo, hi, mid;

  if(k < edges[0] || k > edges[nbin])
    return -1;
  if(k == edges[nbin])
    return nbin - 1;

  lo = 0;
  hi = nbin;
  while(hi - lo > 1)
    {
      mid = (hi + lo) / 2;
      if(k < edges[mid])
        hi = mid;
      else
        lo = mid;
    }
  return lo;
}

static double bpng_coord_value(double k)
{
  if(BPNG.coord_mode == BPNG_COORD_LOG)
    {
      if(k <= 0.0)
        return -1.0e300;
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
  return BPNG.node_x[axis][node];
}

/* Node coordinates and bin centres/widths depend only on the edges: computed once instead of in every per-mode
 * call. */
static void bpng_precompute_axes(void)
{
  int    axis, b;
  double xlo, xhi;

  for(axis = 0; axis < 3; axis++)
    {
      BPNG.node_x[axis] = (double *) malloc(sizeof(double) * BPNG.nk[axis]);
      BPNG.bin_xc[axis] = (double *) malloc(sizeof(double) * BPNG.nk[axis]);
      BPNG.bin_dx[axis] = (double *) malloc(sizeof(double) * BPNG.nk[axis]);
      if(!BPNG.node_x[axis] || !BPNG.bin_xc[axis] || !BPNG.bin_dx[axis])
        FatalError(9116);

      for(b = 0; b < BPNG.nk[axis]; b++)
        {
          BPNG.node_x[axis][b] = bpng_coord_value(bpng_axis_node_k(axis, b));

          xlo = bpng_coord_value(BPNG.edges[axis][b]);
          xhi = bpng_coord_value(BPNG.edges[axis][b + 1]);
          BPNG.bin_xc[axis][b] = 0.5 * (xlo + xhi);
          BPNG.bin_dx[axis][b] = xhi - xlo;
        }
    }
}

static double bpng_u_in_bin_axis(int axis, int bin, double k)
{
  double x, dx;

  if(bin < 0 || bin >= BPNG.nk[axis])
    return 0.0;

  x  = bpng_coord_value(k);
  dx = BPNG.bin_dx[axis][bin];
  if(dx == 0.0 || !isfinite(dx))
    return 0.0;

  return (x - BPNG.bin_xc[axis][bin]) / dx;
}

/* Modes used by the NG term: the Gaussian field's support (the Nsample test of lpt_setup_gaussian_potential),
 * never k = 0 and never the Nyquist planes, where the Gaussian field is zero and i*k is ill-defined. */
static int bpng_mode_allowed(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k, double kx, double ky, double kz, double kmag)
{
  if(kmag == 0.0)
    return 0;
  if(i == Nmesh / 2 || j == Nmesh / 2 || k == Nmesh / 2)     /* Gaussian field is zero there */
    return 0;

  if(SphereMode == 1)
    {
      if(kmag * Box / (2 * PI) > Nsample / 2)
        return 0;
    }
  else
    {
      if(fabs(kx) * Box / (2 * PI) > Nsample / 2)
        return 0;
      if(fabs(ky) * Box / (2 * PI) > Nsample / 2)
        return 0;
      if(fabs(kz) * Box / (2 * PI) > Nsample / 2)
        return 0;
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
  int    n = BPNG.nk[axis];
  int    right, i0, i1;
  double x, x0, x1, t;

  if(n <= 0)
    return 0.0;
  if(kmag < BPNG.edges[axis][0] || kmag > BPNG.edges[axis][n])
    return 0.0;
  if(n == 1)
    return (node == 0) ? 1.0 : 0.0;
  if(node < 0 || node >= n)
    return 0.0;

  x = bpng_coord_value(kmag);
  if(x <= bpng_axis_node_x(axis, 0))
    return (node == 0) ? 1.0 : 0.0;
  if(x >= bpng_axis_node_x(axis, n - 1))
    return (node == n - 1) ? 1.0 : 0.0;

  right = bpng_search_node_right(axis, x);
  i0    = right - 1;
  if(i0 < 0)
    i0 = 0;
  if(i0 > n - 2)
    i0 = n - 2;
  i1 = i0 + 1;

  x0 = bpng_axis_node_x(axis, i0);
  x1 = bpng_axis_node_x(axis, i1);
  if(x1 == x0)
    return 0.0;

  t = (x - x0) / (x1 - x0);
  if(t < 0.0)
    t = 0.0;
  if(t > 1.0)
    t = 1.0;

  if(node == i0)
    return 1.0 - t;
  if(node == i1)
    return t;
  return 0.0;
}

static double bpng_cubic_input_basis_value(int axis, int node, double kmag)
{
  int    n = BPNG.nk[axis];
  int    right, base, m, q, idx[4];
  double x, xm, xq, den, w;

  if(n < 4)
    return bpng_linear_input_basis_value(axis, node, kmag);
  if(kmag < BPNG.edges[axis][0] || kmag > BPNG.edges[axis][n])
    return 0.0;
  if(node < 0 || node >= n)
    return 0.0;

  x = bpng_coord_value(kmag);
  if(x <= bpng_axis_node_x(axis, 0))
    return (node == 0) ? 1.0 : 0.0;
  if(x >= bpng_axis_node_x(axis, n - 1))
    return (node == n - 1) ? 1.0 : 0.0;

  right = bpng_search_node_right(axis, x);
  base  = right - 2;
  if(base < 0)
    base = 0;
  if(base > n - 4)
    base = n - 4;
  for(m = 0; m < 4; m++)
    idx[m] = base + m;

  /* Lagrange weight of this node over the 4 nodes around x */
  for(m = 0; m < 4; m++)
    {
      if(node != idx[m])
        continue;

      xm = bpng_axis_node_x(axis, idx[m]);
      w  = 1.0;
      for(q = 0; q < 4; q++)
        {
          if(q == m)
            continue;

          xq  = bpng_axis_node_x(axis, idx[q]);
          den = xm - xq;
          if(den == 0.0)
            return 0.0;
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


/* ---- per-mode steps of the convolution: filter, product, output-leg coefficient ---- */

/* The output-leg polynomial of one cell, sum_q c[q] u^q for the BPNG.coeff_count coefficients c. */
static inline double bpng_cell_polynomial(const double *c, double u)
{
  double u2;
  double coef;

  coef = c[0];
  if(BPNG.coeff_count <= 1)
    return coef;

  coef += c[1] * u;
  if(BPNG.coeff_count <= 2)
    return coef;

  u2    = u * u;
  coef += c[2] * u2
        + c[3] * u2 * u;
  return coef;
}

/* The coefficients of cell (ia, ib, ic), in c[0 .. coeff_count-1]. */
static inline void bpng_cell_coefficients(int ia, int ib, int ic, double *c)
{
  for(int q = 0; q < BPNG.coeff_count; q++)
    c[q] = bpng_coeff(q, ia, ib, ic);
}

static double bpng_output_k3_coefficient_u(int ia, int ib, int ic, double u)
{
  double c[4] = {0, 0, 0, 0};

  bpng_cell_coefficients(ia, ib, ic, c);
  return bpng_cell_polynomial(c, u);
}

/* Output-leg coefficient of pair (ia, ib) in bin ic at bin coordinate u (bpng_u_in_bin_axis); the (ib, ia) cell is
 * added, since only ia <= ib pairs are convolved (k1 and k2 share their bins). */
static double bpng_output_k3_pair_coefficient_u(int ia, int ib, int ic, double u)
{
  double coef;

  coef = bpng_output_k3_coefficient_u(ia, ib, ic, u);
  if(ia != ib)
    coef += bpng_output_k3_coefficient_u(ib, ia, ic, u);
  return coef;
}


/* ---- ModeTables 1: per-rank lists of the allowed modes ---- */

static void bpng_build_mode_list_layout(BpngModeList *L, int transposed)
{
  ptrdiff_t ii, j, k, i, n0 = transposed ? Band_ny : Local_nx;
  double    kx, ky, kz, kmag;
  long long pass, n = 0, ntot = 0;

  for(pass = 0; pass < 2; pass++)          /* pass 0 counts, pass 1 fills */
    {
      if(pass == 1)
        {
          L->n     = n;
          L->coord = (uint64_t *) malloc(sizeof(uint64_t) * (n > 0 ? n : 1));
          L->kmag  = (double *)   malloc(sizeof(double)   * (n > 0 ? n : 1));
          L->u     = (double *)   malloc(sizeof(double)   * (n > 0 ? n : 1));
          L->ic    = (int *)      malloc(sizeof(int)      * (n > 0 ? n : 1));
          if(!L->coord || !L->kmag || !L->u || !L->ic)
            FatalError(9117);
          n = 0;
        }

      for(ii = 0; ii < n0; ii++)
        for(j = 0; j < Nmesh; j++)
          for(k = 0; k <= lpt_kz_end(); k++)
            {
              /* normal: ii = local x, j = y; band layout: ii = band row (y), j = x */
              ptrdiff_t ix = transposed ? j : ii + Local_x_start, iy = transposed ? Band_y[ii] : j;

              i    = ix;
              kx   = lpt_wavenumber(i);
              ky   = lpt_wavenumber(iy);
              kz   = lpt_wavenumber(k);
              kmag = sqrt(kx*kx + ky*ky + kz*kz);

              if(!bpng_mode_allowed(ix, iy, k, kx, ky, kz, kmag))
                continue;

              if(pass == 1)
                {
                  L->coord[n] = transposed ? lpt_band_coord(ii, j, k) : (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
                  L->kmag[n]  = kmag;
                  L->ic[n]    = bpng_find_bin(BPNG.edges[2], BPNG.nk[2], kmag);
                  L->u[n]     = (L->ic[n] < 0 || BPNG.coeff_count <= 1) ? 0.0 : bpng_u_in_bin_axis(2, L->ic[n], kmag);
                }
              n++;
            }
    }

  MPI_Reduce(&L->n, &ntot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if(ThisTask == 0 && transposed)
    printf("  ModeTables 1: %lld allowed modes tabulated twice (%.2f GB over all ranks).\n", ntot,
           2 * ntot * (double) (2 * sizeof(double) + sizeof(uint64_t) + sizeof(int)) / 1e9);
}

static void bpng_build_mode_list(void)
{
  bpng_build_mode_list_layout(&BML, 0);
  bpng_build_mode_list_layout(&BMLT, 1);
}

static void bpng_free_mode_list(void)
{
  BpngModeList *L[2] = {&BML, &BMLT};

  for(int q = 0; q < 2; q++)
    {
      free(L[q]->coord);
      free(L[q]->kmag);
      free(L[q]->u);
      free(L[q]->ic);

      L[q]->coord = NULL;
      L[q]->kmag  = L[q]->u = NULL;
      L[q]->ic    = NULL;
      L[q]->n     = 0;
    }
}


/* ---- multi-grid: bin pairs on smaller alias-free grids (BinnedPNGMultiGrid) ---- */

#include "binned_multigrid.h"


/* ---- the NG potential, for NGPrecision double (_d) and float (_f): binned_ng_impl.h ---- */

#define NGR      lpt_real
#define NGC      lpt_complex
#define NG(n)    n##_d
#define NG_ALLOC lpt_alloc_grid
#define NG_FWD   lpt_fft_forward_band
#define NG_INV   lpt_fft_inverse_band
#include "binned_ng_impl.h"
#undef NGR
#undef NGC
#undef NG
#undef NG_ALLOC
#undef NG_FWD
#undef NG_INV

#ifndef AARAMBAM_NO_FLOAT
#define NGR      float
#define NGC      lpt_complexf
#define NG(n)    n##_f
#define NG_ALLOC lpt_alloc_grid_f
#define NG_FWD   lpt_fftf_forward_band
#define NG_INV   lpt_fftf_inverse_band
#include "binned_ng_impl.h"
#undef NGR
#undef NGC
#undef NG
#undef NG_ALLOC
#undef NG_FWD
#undef NG_INV
#endif

void displacement_fields(void)
{
  double        Beta;
  lpt_complex  *cpot;           /* Gaussian potential */
  lpt_complex  *cpot_NG;        /* non-Gaussian term, accumulated in the band k-space layout */
  uint64_t      bytes = 0;

  if(Nmax < Nsample)
    {
      if(ThisTask == 0)
        printf("Nmax (%d) must be at least Nsample (%d).\n", Nmax, Nsample);
      FatalError(9118);
    }

  lpt_setup_gaussian_potential(&cpot, &Beta);

  /* the binned NG term, added to the Gaussian potential as is (no Bardeen-to-gravitational sign flip) */
  cpot_NG = (lpt_complex *) lpt_alloc_grid();
  ASSERT_ALLOC(cpot_NG);
  memset(cpot_NG, 0, sizeof(lpt_real) * TotalSizePlusAdditional);

  lpt_barrier();
#ifndef AARAMBAM_NO_FLOAT
  if(NGFloat)
    bpng_compute_non_gaussian_potential_f(cpot, cpot_NG);
  else
#endif
    bpng_compute_non_gaussian_potential_d(cpot, cpot_NG);
  lpt_barrier();

  /* cpot_NG was accumulated in the band k-space layout (see bpng_accumulate_hybrid_convolution): move it to the
   * normal x-slab layout used below. */
  lpt_band_to_normal(cpot_NG);

  lpt_barrier();

  lpt_displacements_from_potential(cpot, cpot_NG, 1.0, Beta);
}
