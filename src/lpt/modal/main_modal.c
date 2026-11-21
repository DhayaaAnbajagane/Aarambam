/* Modal (Legendre basis) method of 2LPTBasis: banner, AlphaTable/X0 tables and the separable
 * non-Gaussian potential; the shared steps are in common/lpt_core.c. */
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_spline.h>
#include "allvars.h"
#include "proto.h"

#include "lpt_core.h"


void print_setup(void)
{
  lpt_print_banner("Aarambam-2LPT-Basis");

  printf("      Box = %.2e   Nmesh = %d   Nsample = %d   Nmax = %d\n", Box, Nmesh, Nsample, Nmax);
  printf("                       Nglass = %d    GlassTileFac = %d\n\n", Nglass, GlassTileFac);
  printf("        Omega = %.4f         OmegaLambda = %.4f    OmegaBaryon = %.2e\n", Omega,  OmegaLambda, OmegaBaryon);
  printf("       sigma8 = %.4f     PrimoridalIndex = %.4f       Redshift = %.2e\n", Sigma8, PrimordialIndex, Redshift);
  printf("   HubbleParam = %.4f  OmegaDM_2ndSpecies = %.2e          fNL = %+.2e\n\n",
         HubbleParam, OmegaDM_2ndSpecies, Fnl);
  if(WhichSpectrum != 0)
    printf("   P(k) oscillation (%s k):  A_res = %.4f    w_res = %.4f      phi_res = %.4f\n\n",
           (WhichSpectrum == 1) ? "linear" : "log", A_res, w_res, phi_res);
  printf("   FixedAmplitude = %d    PhaseFlip = % d   SphereMode = %d    Seed = %d\n",
         FixedAmplitude, PhaseFlip, SphereMode, Seed);

  printf("\n");
  lpt_print_rule();
  printf("\n");
  return;
}


/* ---- per-mode helpers of the Legendre kernels ---- */

/* |k|^2 and |k| of mode (i, j, k); returns 0 for k = 0 and for modes outside the sampled cube (sphere with
 * SphereMode 1). */
static inline int modal_mode_in_band(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k, double *kmag2, double *kmag1)
{
  double kvec[3];

  if(i + j + k == 0)
    return 0;

  kvec[0] = lpt_wavenumber(i);
  kvec[1] = lpt_wavenumber(j);
  kvec[2] = lpt_wavenumber(k);

  *kmag2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
  *kmag1 = sqrt(*kmag2);

  if(SphereMode == 1)
    {
      if(*kmag1 * Box / (2 * PI) > Nsample / 2)       /* select a sphere in k-space */
        return 0;
    }
  else
    {
      if(fabs(kvec[0]) * Box / (2 * PI) > Nsample / 2)
        return 0;
      if(fabs(kvec[1]) * Box / (2 * PI) > Nsample / 2)
        return 0;
      if(fabs(kvec[2]) * Box / (2 * PI) > Nsample / 2)
        return 0;
    }
  return 1;
}

/* Argument of the Legendre polynomials: k^{(4 - n_s)/3} mapped linearly onto [-1, 1] over [kmin, kmax]. */
static inline double modal_legendre_x(double kmag1, double kmin_ns, double kmax_ns)
{
  double kmag_ns = pow(kmag1, (4 - PrimordialIndex)/3);

  return -1 + 2 * (kmag_ns - kmin_ns)/(kmax_ns - kmin_ns);
}

/* Advance the recursion buffers after an N >= 4 step: (P_{N-3}, P_{N-2}, P_{N-1}) -> (P_{N-2}, P_{N-1}, free). Only
 * in-band modes are ever read or written, and each step overwrites r[2] there, so no copy or clearing is needed. */
static inline void modal_rotate(lpt_complex *r[3])
{
  lpt_complex *t = r[0];

  r[0] = r[1];
  r[1] = r[2];
  r[2] = t;
}

/* Output-leg kernel of pair (N_i, N_j) at one mode: sum over N_k >= N_j of alpha[N_k] g_{N_k}(k3) / k3^2 (the
 * g_{N_k} for N_k >= 4 by the Legendre recursion started at max(4, N_j)). With pw/plk2 = NULL the powers and
 * Legendre values are computed here; otherwise they come from the ModeTables row (identical values). */
static inline long double modal_k3_factor(const double *alpha, unsigned int N_j,
                                          double kmag, double kmag1, double kmag2,
                                          const double *pw, const double *plk2, double (*X0Table)[2])
{
  long double  fnl_fac = 0;
  double       a = NAN, b = NAN, c, d;
  unsigned int N_k;

  for(N_k = N_j; N_k < N_modes; N_k++)
    {
      double np = ((double) N_k) - 1;

      if(N_k <= 3)
        fnl_fac += alpha[N_k] * (pw ? pw[N_k] : pow(kmag1, (4 - PrimordialIndex)/3 * (N_k - 3.)));
      else
        {
          /* start of the recursion: P_{N-3}, P_{N-2} */
          if((N_k == 4) || ((N_j >= 4) && (N_j == N_k)))
            {
              a = plk2 ? plk2[N_k - 1 - 2] : gsl_sf_legendre_Pl(N_k - 1 - 2, kmag)/kmag2;
              b = plk2 ? plk2[N_k - 1 - 1] : gsl_sf_legendre_Pl(N_k - 1 - 1, kmag)/kmag2;
            }

          c = ( (2 * np - 1) * kmag * b - (np - 1) * a ) / np;
          d = c - X0Table[N_k - 1][0]/kmag2;

          a = b;
          b = c;
          fnl_fac += alpha[N_k] * d;
        }
    }
  return fnl_fac;
}

/* ModeTables 1: the in-band modes of the band k-space layout (lpt_fft.h; where the NG potential is accumulated), with
 * a row of per-mode values  [x, k^2, k^{(4-n_s)/3 (N-3)} for N = 0..3, P_l(x)/k^2 for l = 0..N_modes-3]. */
typedef struct
{
  long long  n;
  int        nt;
  uint64_t  *coord;
  double    *v;
}
ModalModeList;

static void modal_build_mode_list(ModalModeList *L, double kmin_ns, double kmax_ns)
{
  ptrdiff_t    ii, i, j, k, kend = lpt_kz_end();
  double       kmag2, kmag1, kmag, *v;
  long long    ntot;
  int          l, nl = (N_modes >= 5) ? N_modes - 2 : 0;
  unsigned int N_k;

  L->nt = 6 + nl;
  for(int pass = 0; pass < 2; pass++)          /* pass 0 counts, pass 1 fills */
    {
      L->n = 0;
      for(ii = 0; ii < Band_ny; ii++)
        {
          j = Band_y[ii];
          if(!lpt_axis_in_band(j))
            continue;

          for(i = 0; i < Nmesh; i++)
            {
              if(!lpt_axis_in_band(i))
                continue;

              for(k = 0; k <= kend; k++)
                {
                  if(!modal_mode_in_band(i, j, k, &kmag2, &kmag1))
                    continue;

                  if(pass == 1)
                    {
                      kmag = modal_legendre_x(kmag1, kmin_ns, kmax_ns);
                      v    = L->v + L->n * L->nt;

                      L->coord[L->n] = lpt_band_coord(ii, i, k);
                      v[0] = kmag;
                      v[1] = kmag2;
                      for(N_k = 0; N_k <= 3; N_k++)
                        v[2 + N_k] = pow(kmag1, (4 - PrimordialIndex)/3 * (N_k - 3.));
                      for(l = 0; l < nl; l++)
                        v[6 + l] = gsl_sf_legendre_Pl(l, kmag)/kmag2;
                    }
                  L->n++;
                }
            }
        }

      if(pass == 0)
        {
          L->coord = malloc((L->n > 0 ? L->n : 1) * sizeof(uint64_t));
          L->v     = malloc((L->n > 0 ? L->n : 1) * L->nt * sizeof(double));
          if(!L->coord || !L->v)
            {
              printf("ModeTables 1: failed to allocate the mode table (%lld modes) on task %d\n", L->n, ThisTask);
              FatalError(102);
            }
        }
    }

  MPI_Reduce(&L->n, &ntot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if(ThisTask == 0)
    printf("  ModeTables 1: %lld modes tabulated (%.2f GB over all ranks).\n", ntot,
           ntot * (double) (sizeof(uint64_t) + L->nt * sizeof(double)) / 1e9);
}


/* ---- the NG stage, for NGPrecision double (_d) and float (_f): modal_ng_impl.h ---- */

#define NGR      lpt_real
#define NGC      lpt_complex
#define NG(n)    n##_d
#define NG_ALLOC lpt_alloc_grid
#define NG_FWD   lpt_fft_forward_band
#define NG_INV   lpt_fft_inverse_band
#include "modal_ng_impl.h"
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
#include "modal_ng_impl.h"
#undef NGR
#undef NGC
#undef NG
#undef NG_ALLOC
#undef NG_FWD
#undef NG_INV
#endif


/* ---- the NG potential ---- */

void displacement_fields(void)
{
  double        Beta;
  lpt_complex  *cpot;            /* Gaussian potential */
  lpt_complex  *cpot_NG;         /* non-Gaussian term, accumulated in the band k-space layout */
  unsigned int  bytes = 0;

  /* the basis is built up to Nmax, so it must cover the sampled modes (Nsample) */
  if(Nmax < Nsample)
    {
      if(ThisTask == 0)
        printf("Nmax (%d) must be at least Nsample (%d).\n", Nmax, Nsample);
      FatalError(9118);
    }

  /* the Legendre argument maps k^{(4-n_s)/3} over [kmin, kmax] onto [-1, 1] */
  double kmin = 2 * PI / Box;
  double kmax = 2 * PI / Box * Nmax;

  double kmin_ns = pow(kmin, (4 - PrimordialIndex)/3);
  double kmax_ns = pow(kmax, (4 - PrimordialIndex)/3);


  /**** alpha_{ijk} and X0 tables (read on task 0, broadcast) ****/

  double (*AlphaTable)[N_modes][N_modes] = malloc(sizeof(double) * N_modes * N_modes * N_modes);
  if(!AlphaTable)
    {
      printf("Failed to allocate AlphaTable on task %d\n", ThisTask);
      FatalError(101);
    }
  memset(AlphaTable, 0, N_modes * N_modes * N_modes * sizeof(double));

  if(ThisTask == 0)
    {
      FILE  *fd;
      char   buf[500];
      int    n1, n2, n3, nread = 0;
      double alpha;

      snprintf(buf, sizeof(buf), "%s", FileWithAlphaCoeff);
      if(!(fd = fopen(buf, "r")))
        {
          printf("can't read alpha coefficients in file '%s' on task %d\n", buf, ThisTask);
          FatalError(17);
        }

      while(fscanf(fd, " %d %d %d %lf ", &n1, &n2, &n3, &alpha) == 4)
        {
          if(n1 < 0 || n1 >= N_modes || n2 < 0 || n2 >= N_modes || n3 < 0 || n3 >= N_modes)
            {
              printf("Alpha table '%s' has mode (%d, %d, %d), outside N_modes = %d. "
                     "Was it decomposed with a different N_modes?\n", buf, n1, n2, n3, N_modes);
              FatalError(18);
            }

          AlphaTable[n3][n2][n1] = alpha;
          AlphaTable[n3][n1][n2] = alpha;
          AlphaTable[n2][n3][n1] = alpha;
          AlphaTable[n2][n1][n3] = alpha;
          AlphaTable[n1][n3][n2] = alpha;
          AlphaTable[n1][n2][n3] = alpha;
          nread++;
        }

      if(!feof(fd) || nread == 0)     /* stopped at an unreadable line, or nothing read: not a zero kernel */
        {
          printf("Alpha table '%s': unreadable line after %d rows\n", buf, nread);
          FatalError(103);
        }
      fclose(fd);
    }

  MPI_Bcast(AlphaTable, N_modes * N_modes * N_modes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double (*X0Table)[2] = malloc(sizeof(double) * N_modes * 2);
  if(X0Table == NULL)
    {
      printf("Failed to allocate X0Table on task %d\n", ThisTask);
      FatalError(101);
    }
  memset(X0Table, 0, N_modes * 2 * sizeof(double));

  if(ThisTask == 0)
    {
      FILE *fd;
      char  buf[500];
      int   n1, nrows = 0;
      char *seen = calloc(N_modes, 1);

      snprintf(buf, sizeof(buf), "%s", FileWithX0Coeff);
      if(!(fd = fopen(buf, "r")))
        {
          printf("can't read X0 coefficients in file '%s' on task %d\n", buf, ThisTask);
          FatalError(17);
        }

      do
        {
          double x0, x1;

          if(fscanf(fd, " %d %lf %lf ", &n1, &x0, &x1) == 3)
            {
              if(n1 < 0 || n1 >= N_modes)
                {
                  printf("X0 table '%s' has row %d, outside N_modes = %d\n", buf, n1, N_modes);
                  FatalError(18);
                }
              X0Table[n1][0] = x0;
              X0Table[n1][1] = x1;
              nrows += !seen[n1];
              seen[n1] = 1;
            }
          else
            break;
        }
      while(1);

      if(!feof(fd) || nrows != N_modes)     /* every row 0 .. N_modes-1, as written by the Legendre decomposition */
        {
          printf("X0 table '%s': %d of the rows 0 .. %d read%s. Was it written for N_modes = %d?\n", buf, nrows,
                 N_modes - 1, feof(fd) ? "" : " (stopped at an unreadable line)", N_modes);
          FatalError(104);
        }
      free(seen);
      fclose(fd);
    }

  MPI_Bcast(&(X0Table[0][0]), N_modes * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double t_012 = 0;
  if(N_modes >= 3 && AlphaTable[0][1][2] != 0)
    t_012 = -AlphaTable[1][1][1]/AlphaTable[0][1][2];
  if(ThisTask == 0)
    printf("Set coeff t_012 = %lg \n", t_012);


  /**** Gaussian potential ****/

  lpt_setup_gaussian_potential(&cpot, &Beta);

  /* the NG term is added to the Gaussian potential as is (no Bardeen-to-gravitational sign flip) */
  if(ThisTask == 0)
    {
      printf("Computing Legendre Basis non-Gaussian potential...\n");
      fflush(stdout);
    }


  cpot_NG = (lpt_complex *) lpt_alloc_grid();
  ASSERT_ALLOC(cpot_NG);
  memset(cpot_NG, 0, sizeof(lpt_real) * TotalSizePlusAdditional);

#ifndef AARAMBAM_NO_FLOAT
  if(NGFloat)
    modal_ng_stage_f(cpot, cpot_NG, AlphaTable, X0Table, t_012, kmin_ns, kmax_ns);
  else
#endif
    modal_ng_stage_d(cpot, cpot_NG, AlphaTable, X0Table, t_012, kmin_ns, kmax_ns);

  lpt_barrier();


  /**** Add the NG potential to the Gaussian one, then the displacements ****/

  /* cpot_NG was accumulated in the band k-space layout: back to the normal x-slab layout */
  lpt_band_to_normal(cpot_NG);

  free(AlphaTable);
  free(X0Table);

  lpt_barrier();

  lpt_displacements_from_potential(cpot, cpot_NG, Fnl, Beta);
}
