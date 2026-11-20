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
#include <signal.h>
#include <assert.h>

#include "lpt_core.h"

void print_setup(void) {
  char pstr[79];

  for (int i = 0; i < 79; i++)
    pstr[i] = '*';
  printf("%s\n", pstr);

  char exec[] = "Aarambam-2LPT-Basis";


  pstr[0] = '*';
  for (int i = 1; i < 79/2 - 9; i++)
    pstr[i] = ' ';
  sprintf(&pstr[79/2 - 10], "%s", exec);
  for (int i = 79/2 + 10; i < 78; i++)
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
  printf("   HubbleParam = %.4f  OmegaDM_2ndSpecies = %.2e          fNL = %+.2e\n\n", HubbleParam, OmegaDM_2ndSpecies, Fnl); 
  if(WhichSpectrum != 0)
    printf("   P(k) oscillation (%s k):  A_res = %.4f    w_res = %.4f      phi_res = %.4f\n\n",
           (WhichSpectrum == 1) ? "linear" : "log", A_res, w_res, phi_res);
  printf("   FixedAmplitude = %d    PhaseFlip = % d   SphereMode = %d    Seed = %d\n", FixedAmplitude, PhaseFlip, SphereMode, Seed);   

  for (int i = 0; i < 79; i++)
    pstr[i] = '*';
  printf("\n%s\n\n", pstr);
  return;
}


/* |k|^2 and |k| of mode (i, j, k); returns 0 for k = 0 and for modes outside the sampled band (SphereMode). */
static inline int modal_mode_in_band(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k, double *kmag2, double *kmag1)
{
  double kvec[3];

  if(i + j + k == 0)
    return 0;

  kvec[0] = (i < Nmesh / 2) ? i * 2 * PI / Box : -(Nmesh - i) * 2 * PI / Box;
  kvec[1] = (j < Nmesh / 2) ? j * 2 * PI / Box : -(Nmesh - j) * 2 * PI / Box;
  kvec[2] = (k < Nmesh / 2) ? k * 2 * PI / Box : -(Nmesh - k) * 2 * PI / Box;

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

/* Mode field N in Fourier space (normal x-slab layout), written to the in-band modes of out:
 *   N <= 3:  out = k^{(4 - n_s)/3 (N - 3)} k^{4 - n_s} phi;
 *   N >= 4:  r[2] = P_{N-1}(x) k^{2 - n_s} phi by the Legendre recursion from r[0] = P_{N-3}, r[1] = P_{N-2}
 *            (set from gsl when restart), and out = r[2] - X0_{N-1} k^{2 - n_s} phi.
 * After an N >= 4 step the caller rotates r (modal_rotate) so that r[0], r[1] hold P_{N-2}, P_{N-1}. */
static void modal_build_field(unsigned int N, int restart, fftw_complex *cpot, fftw_complex *r[3], fftw_complex *out,
                              double (*X0Table)[2], double kmin_ns, double kmax_ns)
{
  PROF_BEGIN(PROF_FIELD)
  ptrdiff_t ii, i, j, k, kend = lpt_kz_end();
  uint64_t coord;
  double kmag2, kmag1, kmag, Pl, pk;
  long double fnl_fac;
  double np = ((double) N) - 1; //For doing legendre calculations later

  for(ii = 0; ii < Local_nx; ii++)
    {
      i = ii + Local_x_start;
      if(!lpt_axis_in_band(i)) continue;
      for(j = 0; j < Nmesh; j++)
        {
          if(!lpt_axis_in_band(j)) continue;
          for(k = 0; k <= kend; k++)
            {
              coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
              if(!modal_mode_in_band(i, j, k, &kmag2, &kmag1)) continue;

              if(N <= 3)
                {
                  fnl_fac = pow(kmag1, (4 - PrimordialIndex)/3 * (N - 3.)) * pow(kmag1, 4 - PrimordialIndex); //including 1/P1(k) term here
                  out[coord].re = fnl_fac * cpot[coord].re;
                  out[coord].im = fnl_fac * cpot[coord].im;
                  continue;
                }

              kmag = modal_legendre_x(kmag1, kmin_ns, kmax_ns);
              //Use 2 - n_s instad of 4 - n_s since you implicitly account for the 1/k^2 factor
              //in switching from S(k1, k2, k3) --> B(k1, k2, k3).
              pk = pow(kmag1, (2 - PrimordialIndex));
              if(restart)
                {
                  Pl = gsl_sf_legendre_Pl(N - 1 - 2, kmag) * pk;
                  r[0][coord].re = cpot[coord].re * Pl;
                  r[0][coord].im = cpot[coord].im * Pl;

                  Pl = gsl_sf_legendre_Pl(N - 1 - 1, kmag) * pk;
                  r[1][coord].re = cpot[coord].re * Pl;
                  r[1][coord].im = cpot[coord].im * Pl;
                }

              r[2][coord].re = ( (2*np - 1) * kmag * r[1][coord].re - (np - 1) * r[0][coord].re ) / np ;
              r[2][coord].im = ( (2*np - 1) * kmag * r[1][coord].im - (np - 1) * r[0][coord].im ) / np ;

              Pl = X0Table[N - 1][0] * pk;
              out[coord].re = r[2][coord].re - cpot[coord].re * Pl;
              out[coord].im = r[2][coord].im - cpot[coord].im * Pl;
            }
        }
    }
  PROF_END(PROF_FIELD)
}

/* Advance the recursion buffers after an N >= 4 step: (P_{N-3}, P_{N-2}, P_{N-1}) -> (P_{N-2}, P_{N-1}, free). Only
 * in-band modes are ever read or written, and each step overwrites r[2] there, so no copy or clearing is needed. */
static inline void modal_rotate(fftw_complex *r[3])
{
  fftw_complex *t = r[0];
  r[0] = r[1]; r[1] = r[2]; r[2] = t;
}

/* Output-leg kernel of pair (N_i, N_j) at one mode: sum over N_k >= N_j of alpha[N_k] g_{N_k}(k3) / k3^2 (the
 * g_{N_k} for N_k >= 4 by the Legendre recursion started at max(4, N_j)). With pw/plk2 = NULL the powers and
 * Legendre values are computed here; otherwise they come from the ModeTables row (identical values). */
static inline long double modal_k3_factor(const double *alpha, unsigned int N_j, double kmag, double kmag1, double kmag2,
                                          const double *pw, const double *plk2, double (*X0Table)[2])
{
  long double fnl_fac = 0;
  double a = NAN, b = NAN, c, d;
  unsigned int N_k;

  for(N_k = N_j; N_k < N_modes; N_k++)
    {
      double np = ((double) N_k) - 1;

      if(N_k <= 3)
        fnl_fac += alpha[N_k] * (pw ? pw[N_k] : pow(kmag1, (4 - PrimordialIndex)/3 * (N_k - 3.)));
      else
        {
          //Fill in missing modes if start at N_k != 0
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

/* ModeTables 1: the in-band modes of the transposed k-space layout (where the NG potential is accumulated), with
 * a row of per-mode values  [x, k^2, k^{(4-n_s)/3 (N-3)} for N = 0..3, P_l(x)/k^2 for l = 0..N_modes-3]. */
typedef struct { long long n; int nt; uint64_t *coord; double *v; } ModalModeList;

static void modal_build_mode_list(ModalModeList *L, double kmin_ns, double kmax_ns)
{
  ptrdiff_t ii, i, j, k, kend = lpt_kz_end();
  double kmag2, kmag1, kmag, *v;
  long long ntot;
  int l, nl = (N_modes >= 5) ? N_modes - 2 : 0;
  unsigned int N_k;

  L->nt = 6 + nl;
  for(int pass = 0; pass < 2; pass++)          /* pass 0 counts, pass 1 fills */
    {
      L->n = 0;
      for(ii = 0; ii < Local_ny_t; ii++)
        {
          j = ii + Local_y_start_t;
          if(!lpt_axis_in_band(j)) continue;
          for(i = 0; i < Nmesh; i++)
            {
              if(!lpt_axis_in_band(i)) continue;
              for(k = 0; k <= kend; k++)
                {
                  if(!modal_mode_in_band(i, j, k, &kmag2, &kmag1)) continue;
                  if(pass == 1)
                    {
                      kmag = modal_legendre_x(kmag1, kmin_ns, kmax_ns);
                      v = L->v + L->n * L->nt;
                      L->coord[L->n] = (ii * Nmesh + i) * (Nmesh / 2 + 1) + k;
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


void displacement_fields(void) {



  ptrdiff_t i, j, k, ii;      /* 64-bit grid indices, as in the shared core */


  int ksum;
  double Beta;
  fftw_complex *(cpot);  /* For computing nongaussian fnl ic */
    
  fftw_complex *(cpot_NG);  /* For computing nongaussian fnl ic */
    
  fftw_complex *(c_tmp); /* For non-local fluctuations */    
  fftw_real *(p_tmp); /* For non-local fluctuations */

  long double *(cpot_NG_re);
  long double *(cpot_NG_im);
    
  unsigned int N, N_i, N_j, N_k;
  double kmag, kmag2, kmag1;


  long double fnl_fac;
// ******* FAVN *****
// ******* FAVN *****
    
  unsigned int bytes = 0;

  uint64_t coord;





    
  //DHAYAA: For computing temporary fields
  fftw_complex *(cfields[8]);
  fftw_real *(fields[8]);

  //Nmax sets how many modes we build our basis approximation fore. 
  //If it is lower than Nsample then we can't use the Legendre modes.
  //E.g., we can build up to k = 20 but only use up to k = 5 in the grid.
  assert(Nmax >= Nsample);

  //Some hyperparams
  double kmin = 2 * PI / Box;
  double kmax = 2 * PI / Box * Nmax;
    
  double kmin_ns = pow(kmin, (4 - PrimordialIndex)/3);
  double kmax_ns = pow(kmax, (4 - PrimordialIndex)/3);
    
  
  MPI_Bcast(&SavePotentialField, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //DHAYAA: Now need to read out the coefficients from a file
  //Load the Alpha Table
  double (*AlphaTable)[N_modes][N_modes] = malloc(sizeof(double) * N_modes * N_modes * N_modes);
  memset(AlphaTable, 0, N_modes * N_modes * N_modes * sizeof(double));

  if (ThisTask == 0)
  {
      FILE *fd;
      char buf[500];
      int n1, n2, n3;
      double alpha;

      snprintf(buf, sizeof(buf), "%s", FileWithAlphaCoeff);

      if (!(fd = fopen(buf, "r")))
      {
          printf("can't read alpha coefficients in file '%s' on task %d\n", buf, ThisTask);
          FatalError(17);
      }

      while (fscanf(fd, " %d %d %d %lf ", &n1, &n2, &n3, &alpha) == 4)
      {
          if (n1 < 0 || n1 >= N_modes || n2 < 0 || n2 >= N_modes || n3 < 0 || n3 >= N_modes)
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
      }

      fclose(fd);
  }

  // Broadcast the full AlphaTable to all tasks
  MPI_Bcast(AlphaTable, N_modes * N_modes * N_modes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  // Allocate heap memory for X0Table
  double (*X0Table)[2] = malloc(sizeof(double) * N_modes * 2);
  if (X0Table == NULL) {
      printf("Failed to allocate X0Table on task %d\n", ThisTask);
      FatalError(101);
  }

  memset(X0Table, 0, N_modes * 2 * sizeof(double));

  // Only Task 0 reads from file
  if (ThisTask == 0)
  {
      FILE *fd;
      char buf[500];

      int n1;

      snprintf(buf, sizeof(buf), "%s", FileWithX0Coeff);
      if (!(fd = fopen(buf, "r")))
      {
          printf("can't read X0 coefficients in file '%s' on task %d\n", buf, ThisTask);
          FatalError(17);
      }

      do
      {
          double x0, x1;
          if (fscanf(fd, " %d %lf %lf ", &n1, &x0, &x1) == 3)
          {
              if (n1 < 0 || n1 >= N_modes)
              {
                  printf("X0 table '%s' has row %d, outside N_modes = %d\n", buf, n1, N_modes);
                  FatalError(18);
              }
              X0Table[n1][0] = x0;
              X0Table[n1][1] = x1;
          }
          else
              break;
      } while (1);

      fclose(fd);
  }

  // Broadcast the full X0Table to all tasks
  MPI_Bcast(&(X0Table[0][0]), N_modes * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    

  double t_012 = 0;
  if (N_modes >= 3 && AlphaTable[0][1][2] != 0) t_012 = -AlphaTable[1][1][1]/AlphaTable[0][1][2];
  if(ThisTask == 0) {
      printf("Set coeff t_012 = %lg \n", t_012);
  }

  
#ifdef CORRECT_CIC

#endif





  lpt_setup_gaussian_potential(&cpot, &cpot_NG_re, &cpot_NG_im, &Beta);
 /*** For non-local models it is important to keep all factors of SQRT(-1) as done below ***/
 /*** No sign flip is applied to the NG term: it is added to the Gaussian potential as is ***/
 /*** (2LPTPNG noted a minus here to convert from Bardeen to gravitational potential)   ***/

  if (ThisTask == 0) {printf("Computing Legendre Basis non-Gaussian potential...\n"); fflush(stdout);};

  /**** NON-LOCAL PRIMORDIAL POTENTIAL **************/

  /* The NG potential is sum_{N_i <= N_j} FFT[g_i g_j](k3) K_{ij}(k3) over the real-space mode fields g_N.
   * ModeTables 0 rebuilds and inverse-transforms g_j for every pair (least memory). ModeTables 1 builds each g_N
   * once and keeps all N_modes of them in real space, and tabulates the per-mode values of K_{ij}: about N_modes
   * extra grids, half the FFTs. The two agree to rounding (the g_j of pairs with N_i >= 5 are recursed from 4
   * instead of from N_i). */
  fftw_real **gcache = NULL;
  ModalModeList ML = {0, 0, NULL, NULL};

  //We'll use these fields to compute temporary terms of interest to us
  for(N = 0; N < 8; N++){

      cfields[N] = NULL;
      fields[N]  = NULL;
      if(ModeTables && N >= 3) continue;      /* the g_i / g_j work grids are only used without the cache */
      cfields[N] = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
      fields[N]  = (fftw_real *) cfields[N];
      ASSERT_ALLOC(cfields[N]);

  }

  c_tmp = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
  p_tmp = (fftw_real *) c_tmp;
  ASSERT_ALLOC(c_tmp);

  cpot_NG = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
  ASSERT_ALLOC(cpot_NG);

  if(ModeTables)
    {
      gcache = malloc(N_modes * sizeof(fftw_real *));
      ASSERT_ALLOC(gcache);
      for(N = 0; N < N_modes; N++)
        {
          gcache[N] = (fftw_real *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
          ASSERT_ALLOC(gcache[N]);
        }
      if(ThisTask == 0)
        printf("  ModeTables 1: caching %d real-space mode fields (%.2f GB over all ranks).\n", N_modes,
               N_modes * (double) sizeof(fftw_real) * TotalSizePlusAdditional * NTask / 1e9);
    }

  lpt_barrier();



  /* first, clean the array */
  for(N = 0; N < 8; N++)
      if(cfields[N]) memset(cfields[N], 0, sizeof(fftw_real) * TotalSizePlusAdditional);
  memset(c_tmp, 0, sizeof(fftw_real) * TotalSizePlusAdditional);
  memset(cpot_NG, 0, sizeof(fftw_real) * TotalSizePlusAdditional);

  lpt_barrier();

  fftw_complex *ri[3] = {cfields[0], cfields[1], cfields[2]};   /* recursion buffers of g_i */

  if(ModeTables)
    {
      /* build every mode field once, in order, and keep it in real space */
      for(N = 0; N < N_modes; N++)
        {
          memset(gcache[N], 0, sizeof(fftw_real) * TotalSizePlusAdditional);
          modal_build_field(N, N == 4, cpot, ri, (fftw_complex *) gcache[N], X0Table, kmin_ns, kmax_ns);
          if(N >= 4) modal_rotate(ri);
          lpt_barrier();
          lpt_fft(Inverse_plan, gcache[N]);
          lpt_barrier();
        }
      modal_build_mode_list(&ML, kmin_ns, kmax_ns);
    }

  /* Loop to start generating all fields */
  for(N_i = 0; N_i < N_modes; N_i++){

      fftw_real *gi, *gj;

      if(ModeTables)
          gi = gcache[N_i];
      else {
          //Generate the first field in fourier space
          memset(cfields[3], 0, sizeof(fftw_real) * TotalSizePlusAdditional); //Needed to prevent memory issue in loop
          modal_build_field(N_i, N_i == 4, cpot, ri, cfields[3], X0Table, kmin_ns, kmax_ns);
          if(N_i >= 4) modal_rotate(ri);   //so the next iteration continues the recursion
          lpt_barrier();

          lpt_fft(Inverse_plan, fields[3]);
          lpt_barrier();
          gi = fields[3];
      }

      fftw_complex *rj[3] = {cfields[4], cfields[5], cfields[6]};   /* recursion buffers of g_j (restarted at N_i) */

      for(N_j = N_i; N_j < N_modes; N_j++){

          //Routine that determines if we can skip the FFTs assuming
          //the coefficients are zero for all these objects.
          ksum = 0;
          for (N_k = N_j; N_k < N_modes; N_k++){
            if (fabs(AlphaTable[N_i][N_j][N_k]) > 0) ksum = 1;
          }

          if(ModeTables)
              gj = gcache[N_j];
          else {
              /* g_j is needed for this pair, and for N_j >= 4 also to carry the recursion to a later pair */
              int needed = ksum;
              for(unsigned int M = N_j + 1; M < N_modes && N_j >= 4 && !needed; M++)
                for(N_k = M; N_k < N_modes; N_k++)
                  if(fabs(AlphaTable[N_i][M][N_k]) > 0) needed = 1;

              if(needed) {
                  //Generate the 2nd field, also in fourier space
                  memset(cfields[7], 0, sizeof(fftw_real) * TotalSizePlusAdditional); //Needed to prevent memory issue in loop
                  modal_build_field(N_j, (N_j == 4) || ((N_i >= 4) && (N_i == N_j)), cpot, rj, cfields[7], X0Table,
                                    kmin_ns, kmax_ns);
                  if(N_j >= 4) modal_rotate(rj);
                  lpt_barrier();
              }
              gj = fields[7];
          }

          if (ThisTask == 0) {printf("Computing combination %d, %d ....", N_i, N_j); fflush(stdout);};

          if (ksum == 0.0) {
            if (ThisTask == 0 ) {
              int print_size = 0;
              if (N_i >= 10) print_size += 1;
              if (N_j >= 10) print_size += 1;
              print_timed_done(23 - print_size);
            }
            continue;
          }

          if(!ModeTables) {
              lpt_fft(Inverse_plan, fields[7]);
              lpt_barrier();
          }

          memset(p_tmp, 0, sizeof(fftw_real) * TotalSizePlusAdditional);
          lpt_barrier();

      PROF_BEGIN(PROF_MULT)
          //Multiply terms in realspace [\phi gi] x [\phi gj]
          for(i = 0; i < Local_nx; i++)
            for(j = 0; j < Nmesh; j++)
              for(k = 0; k < Nmesh; k++)
                    {
                      coord = (i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k;

                      p_tmp[coord] = gi[coord] * gj[coord];

                    }


      PROF_END(PROF_MULT)
          lpt_barrier();
          lpt_fft_forward_t(p_tmp);
          lpt_barrier();



          PROF_BEGIN(PROF_ACCUM)
          //Add the g3(k) kernel. This can be summed linearly for fixed gi and gj.
          const double *alpha = AlphaTable[N_i][N_j];
          if(ModeTables)
            for(long long m = 0; m < ML.n; m++)
              {
                const double *v = ML.v + m * ML.nt;
                fnl_fac = modal_k3_factor(alpha, N_j, v[0], 0, v[1], v + 2, v + 6, X0Table);
                coord = ML.coord[m];
                cpot_NG_re[coord] += (long double) (fnl_fac * c_tmp[coord].re);
                cpot_NG_im[coord] += (long double) (fnl_fac * c_tmp[coord].im);
              }
          else
          for(ii = 0; ii < Local_ny_t; ii++)          /* transposed layout: [y (local)][x][z] */
            {
              j = ii + Local_y_start_t;
              if(!lpt_axis_in_band(j)) continue;
              for(i = 0; i < Nmesh; i++)
                {
                  if(!lpt_axis_in_band(i)) continue;
                  for(k = 0; k <= lpt_kz_end(); k++)
                  {

                      coord = (ii * Nmesh + i) * (Nmesh / 2 + 1) + k;
                      if(!modal_mode_in_band(i, j, k, &kmag2, &kmag1)) continue;
                      kmag = modal_legendre_x(kmag1, kmin_ns, kmax_ns);

                      fnl_fac = modal_k3_factor(alpha, N_j, kmag, kmag1, kmag2, NULL, NULL, X0Table);

                      cpot_NG_re[coord] += (long double) (fnl_fac * c_tmp[coord].re);
                      cpot_NG_im[coord] += (long double) (fnl_fac * c_tmp[coord].im);
                  }
                }
            }

          PROF_END(PROF_ACCUM)
          if (ThisTask == 0 ) {
            int print_size = 0;
            if (N_i >= 10) print_size += 1;
            if (N_j >= 10) print_size += 1;
            print_timed_done(23 - print_size);
          }
          lpt_barrier();
          }
      } //Close the Ni,Nj,Nk loop


  //Now need to include the extra templates for cancelling divergent terms :P
  //Only the pairs (0, 1) and (0, 2) carry a t_012 term, so the other pairs N_i, N_j <= 3 are not computed.

  //Free the g_i / g_j work fields (2..7), since we won't use them anymore
  for(N = 2; N < 8; N++) free(cfields[N]);
  free(ML.coord);
  free(ML.v);

  if(N_modes >= 3 && t_012 != 0){

      fftw_real *g0, *gj;

      if(ModeTables)
          g0 = gcache[0];
      else {
          //Generate the first field in fourier space
          memset(cfields[0], 0, sizeof(fftw_real) * TotalSizePlusAdditional); //Needed to prevent memory issue in loop
          modal_build_field(0, 0, cpot, ri, cfields[0], X0Table, kmin_ns, kmax_ns);
          lpt_barrier();
          lpt_fft(Inverse_plan, fields[0]);
          lpt_barrier();
          g0 = fields[0];
      }

      for(N_j = 1; N_j <= 2; N_j++){

          if(ModeTables)
              gj = gcache[N_j];
          else {
              //Generate the 2nd field, also in fourier space
              memset(cfields[1], 0, sizeof(fftw_real) * TotalSizePlusAdditional); //Needed to prevent memory issue in loop
              modal_build_field(N_j, 0, cpot, ri, cfields[1], X0Table, kmin_ns, kmax_ns);
              lpt_barrier();
              lpt_fft(Inverse_plan, fields[1]);
              lpt_barrier();
              gj = fields[1];
          }

          memset(p_tmp, 0, sizeof(fftw_real) * TotalSizePlusAdditional);
          lpt_barrier();

          //Multiply terms in realspace [\phi gi] x [\phi gj]
          for(i = 0; i < Local_nx; i++)
            for(j = 0; j < Nmesh; j++)
              for(k = 0; k < Nmesh; k++)
                    {
                      coord = (i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k;
                      p_tmp[coord] = g0[coord] * gj[coord];

                    }


          lpt_barrier();
          lpt_fft_forward_t(p_tmp);
          lpt_barrier();

          if (ThisTask == 0) {printf("Computing Beta combination %d, %d ....", 0, N_j); fflush(stdout);};

          for(ii = 0; ii < Local_ny_t; ii++)          /* transposed layout: [y (local)][x][z] */
            {
              j = ii + Local_y_start_t;
              if(!lpt_axis_in_band(j)) continue;
              for(i = 0; i < Nmesh; i++)
                {
                  if(!lpt_axis_in_band(i)) continue;
                  for(k = 0; k <= lpt_kz_end(); k++)
                  {

                      coord = (ii * Nmesh + i) * (Nmesh / 2 + 1) + k;
                      if(!modal_mode_in_band(i, j, k, &kmag2, &kmag1)) continue;

                      fnl_fac = 0;

                      //The t_012 piece (cancels the divergent (0,1,2) term).
                      if (N_j == 1) //Subtract from the original kernel
                          fnl_fac += -AlphaTable[0][1][2] * t_012 * pow(kmag1, (4 - PrimordialIndex)/3 * (2 - 3.0));
                      else          //Now add the t_012 piece
                          fnl_fac +=  AlphaTable[0][1][2] * t_012 * pow(kmag1, (4 - PrimordialIndex)/3 * (1 - 3.0));

                      cpot_NG_re[coord] += (long double) (fnl_fac * c_tmp[coord].re);
                      cpot_NG_im[coord] += (long double) (fnl_fac * c_tmp[coord].im);
                  }
                }
            }

          if (ThisTask == 0 ) print_timed_done(18);
          lpt_barrier();
          }
      } //Close the Beta loop

  free(c_tmp);

  for(N = 0; N < 2; N++){
      free(cfields[N]);
  }
  if(gcache)
    {
      for(N = 0; N < N_modes; N++) free(gcache[N]);
      free(gcache);
    }

  lpt_barrier();

  //Add NG potential to the full potential
  //Copy the long-double accumulator (transposed k-space layout) into the FFTW complex array, then transpose it
  //back to the normal x-slab layout used below
  for(coord = 0; coord < (uint64_t) Local_ny_t * Nmesh * (Nmesh / 2 + 1); coord++){
      cpot_NG[coord].re = cpot_NG_re[coord];
      cpot_NG[coord].im = cpot_NG_im[coord];
  }
  lpt_transposed_to_normal(cpot_NG);

  free(cpot_NG_re);
  free(cpot_NG_im);
  free(AlphaTable);
  free(X0Table);

  lpt_barrier();
  
  

  lpt_displacements_from_potential(cpot, cpot_NG, Fnl, Beta);
}
