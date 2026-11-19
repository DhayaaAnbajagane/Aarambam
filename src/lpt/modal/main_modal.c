#include <math.h>
#include <stdint.h>
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

// Helper function for comparing doubles for qsort


void displacement_fields(void) {



  int i, j, k, ii;


  int ksum;
  double Beta;
  fftw_complex *(cpot);  /* For computing nongaussian fnl ic */
    
  fftw_complex *(cpot_NG);  /* For computing nongaussian fnl ic */
  fftw_real *(pot_NG);
    
  fftw_complex *(c_tmp); /* For non-local fluctuations */    
  fftw_real *(p_tmp); /* For non-local fluctuations */

  long double *(cpot_NG_re);
  long double *(cpot_NG_im);
    
  unsigned int N, N_i, N_j, N_k;
  double kvec[3], kmag, kmag2, kmag1, kmag_ns;


  long double fnl_fac;
// ******* FAVN *****
// ******* FAVN *****
    
  double a, b, c, d; //For holding coeffs

  unsigned int bytes = 0;

  int coord;





    
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
    
  double Pl; //for storing legendre coeffs
  
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
    

  double t_012 = -AlphaTable[1][1][1]/AlphaTable[0][1][2];
  double u_002 = -AlphaTable[0][1][1]/AlphaTable[0][0][2];
  if (AlphaTable[0][1][2] == 0) t_012 = 0;
  if (AlphaTable[0][0][2] == 0) u_002 = 0;
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

  //We'll use these fields to compute temporary terms of interest to us
  for(N = 0; N < 8; N++){

      cfields[N] = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
      fields[N]  = (fftw_real *) cfields[N];
      ASSERT_ALLOC(cfields[N]);

  }

  c_tmp = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
  p_tmp = (fftw_real *) c_tmp;
  ASSERT_ALLOC(c_tmp);

  cpot_NG = (fftw_complex *) malloc(sizeof(fftw_real) * TotalSizePlusAdditional);
  pot_NG  = (fftw_real *) cpot_NG;
  ASSERT_ALLOC(cpot_NG);

  MPI_Barrier(MPI_COMM_WORLD);

  
    
  /* first, clean the array */
  for(N = 0; N < 8; N++)
      memset(cfields[N], 0, sizeof(fftw_real) * TotalSizePlusAdditional);
  memset(c_tmp, 0, sizeof(fftw_real) * TotalSizePlusAdditional);
  memset(cpot_NG, 0, sizeof(fftw_real) * TotalSizePlusAdditional);

  MPI_Barrier(MPI_COMM_WORLD);

  /* Loop to start generating all fields */      
  for(N_i = 0; N_i < N_modes; N_i++){
      
      double np = ((double) N_i) - 1; //For doing legendre calculations later
          
      memset(cfields[2], 0, sizeof(fftw_real) * TotalSizePlusAdditional); //Needed to prevent memory issue in loop
      memset(cfields[3], 0, sizeof(fftw_real) * TotalSizePlusAdditional); //Needed to prevent memory issue in loop
      
      //Generate the first field in fourier space
      for(ii = 0; ii < Local_nx; ii++)
          for(j = 0; j < Nmesh; j++)
              for(k = 0; k <= Nmesh / 2 ; k++)
              {

                  coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
                  i = ii + Local_x_start;

                  kvec[0] = (i < Nmesh / 2) ? i * 2 * PI / Box : -(Nmesh - i) * 2 * PI / Box;
                  kvec[1] = (j < Nmesh / 2) ? j * 2 * PI / Box : -(Nmesh - j) * 2 * PI / Box;
                  kvec[2] = (k < Nmesh / 2) ? k * 2 * PI / Box : -(Nmesh - k) * 2 * PI / Box;

                  kmag2   = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
                  kmag1   = sqrt(kmag2);
                  kmag_ns = pow(kmag1, (4 - PrimordialIndex)/3);
                  kmag    = -1 + 2 * (kmag_ns - kmin_ns)/(kmax_ns - kmin_ns);

                  if(i + j + k == 0) continue;

                  if(SphereMode == 1)
                    {
                      if(kmag1 * Box / (2 * PI) > Nsample / 2)       /* select a sphere in k-space */
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

                  if (N_i <= 3){
                      fnl_fac = pow(kmag1, (4 - PrimordialIndex)/3 * (N_i - 3.)) * pow(kmag1, 4 - PrimordialIndex); //including 1/P1(k) term here
                      
                      cfields[3][coord].re = fnl_fac * cpot[coord].re;
                      cfields[3][coord].im = fnl_fac * cpot[coord].im;
                      
                  }
                  else {
                      
                      //Setup the initial kernels only at the start
                      if (N_i == 4){
                          
                          //Use 2 - n_s instad of 4 - n_s since you implicitly account for the 1/k^2 factor
                          //in switching from S(k1, k2, k3) --> B(k1, k2, k3).
                          Pl = gsl_sf_legendre_Pl(N_i - 1 - 2, kmag) * pow(kmag1, (2 - PrimordialIndex));
                          cfields[0][coord].re = cpot[coord].re * Pl;
                          cfields[0][coord].im = cpot[coord].im * Pl;

                          Pl = gsl_sf_legendre_Pl(N_i - 1 - 1, kmag) * pow(kmag1, (2 - PrimordialIndex));
                          cfields[1][coord].re = cpot[coord].re * Pl;
                          cfields[1][coord].im = cpot[coord].im * Pl;
                          
                      }
                      
                      //Doesn't need 1/kmag2 since its already in cfields[4] and cfields[3]
                      cfields[2][coord].re = ( (2*np - 1) * kmag * cfields[1][coord].re - (np - 1) * cfields[0][coord].re ) / np ;
                      cfields[2][coord].im = ( (2*np - 1) * kmag * cfields[1][coord].im - (np - 1) * cfields[0][coord].im ) / np ;
                      
                      Pl = X0Table[N_i - 1][0] * pow(kmag1, (2 - PrimordialIndex));
                      cfields[3][coord].re = cfields[2][coord].re - cpot[coord].re * Pl;
                      cfields[3][coord].im = cfields[2][coord].im - cpot[coord].im * Pl;
                      
                  }
              }
      
      
      MPI_Barrier(MPI_COMM_WORLD);
      
      
      //Copy kernels over so next iteration works ok
      if (N_i <= 3) {}
      else {
          memcpy(cfields[0], cfields[1], sizeof(fftw_real) * TotalSizePlusAdditional);
          memcpy(cfields[1], cfields[2], sizeof(fftw_real) * TotalSizePlusAdditional);
      }
      
      rfftwnd_mpi(Inverse_plan, 1, fields[3], Workspace, FFTW_NORMAL_ORDER);
      MPI_Barrier(MPI_COMM_WORLD);
      
      memset(cfields[4], 0, sizeof(fftw_real) * TotalSizePlusAdditional);
      memset(cfields[5], 0, sizeof(fftw_real) * TotalSizePlusAdditional);

      for(N_j = N_i; N_j < N_modes; N_j++){
          
          double np = ((double) N_j) - 1; //For doing legendre calculations later
              
          memset(cfields[6], 0, sizeof(fftw_real) * TotalSizePlusAdditional); //Needed to prevent memory issue in loop
          memset(cfields[7], 0, sizeof(fftw_real) * TotalSizePlusAdditional); //Needed to prevent memory issue in loop
          
          //Generate the 2nd field, also in fourier space
          for(ii = 0; ii < Local_nx; ii++)
              for(j = 0; j < Nmesh; j++)
                  for(k = 0; k <= Nmesh / 2 ; k++)
                  {

                      coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
                      i = ii + Local_x_start;

                      kvec[0] = (i < Nmesh / 2) ? i * 2 * PI / Box : -(Nmesh - i) * 2 * PI / Box;
                      kvec[1] = (j < Nmesh / 2) ? j * 2 * PI / Box : -(Nmesh - j) * 2 * PI / Box;
                      kvec[2] = (k < Nmesh / 2) ? k * 2 * PI / Box : -(Nmesh - k) * 2 * PI / Box;

                      kmag2   = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
                      kmag1   = sqrt(kmag2);
                      kmag_ns = pow(kmag1, (4 - PrimordialIndex)/3);
                      kmag    = -1 + 2 * (kmag_ns - kmin_ns)/(kmax_ns - kmin_ns);

                      if(SphereMode == 1)
                        {
                          if(kmag1 * Box / (2 * PI) > Nsample / 2)       /* select a sphere in k-space */
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

                      if(i + j + k == 0) continue;
                      
                      if (N_j <= 3){
                          fnl_fac = pow(kmag1, (4 - PrimordialIndex)/3 * (N_j - 3.)) * pow(kmag1, 4 - PrimordialIndex); //including 1/P1(k) term here
                          cfields[7][coord].re = fnl_fac * cpot[coord].re;
                          cfields[7][coord].im = fnl_fac * cpot[coord].im;
                          }
                      else {
                          
                          //To fill-in modes that we haven't evaluated yet if we start at N_j != 0
                          if ( (N_j == 4) || ( (N_i >= 4) && (N_i == N_j) )){ 
                              Pl = gsl_sf_legendre_Pl(N_j - 1 - 2, kmag) * pow(kmag1, (2 - PrimordialIndex));
                              cfields[4][coord].re = cpot[coord].re * Pl;
                              cfields[4][coord].im = cpot[coord].im * Pl;

                              Pl = gsl_sf_legendre_Pl(N_j - 1 - 1, kmag) * pow(kmag1, (2 - PrimordialIndex));
                              cfields[5][coord].re = cpot[coord].re * Pl;
                              cfields[5][coord].im = cpot[coord].im * Pl;
                          }
                          
                          //Doesn't need 1/kmag2 since its already in cfields[4] and cfields[3]
                          cfields[6][coord].re = ( (2*np - 1) * kmag * cfields[5][coord].re - (np - 1) * cfields[4][coord].re ) / np ;
                          cfields[6][coord].im = ( (2*np - 1) * kmag * cfields[5][coord].im - (np - 1) * cfields[4][coord].im ) / np ;
                          
                          Pl = X0Table[N_j - 1][0] * pow(kmag1, (2 - PrimordialIndex));
                          cfields[7][coord].re = cfields[6][coord].re - cpot[coord].re * Pl;
                          cfields[7][coord].im = cfields[6][coord].im - cpot[coord].im * Pl;
                          
                          
                          }
                      
                      

                  }
          
          
          
          MPI_Barrier(MPI_COMM_WORLD);
          
          if (N_j <= 3) {}
          else {
              memcpy(cfields[4], cfields[5], sizeof(fftw_real) * TotalSizePlusAdditional);
              memcpy(cfields[5], cfields[6], sizeof(fftw_real) * TotalSizePlusAdditional);
          }

          if (ThisTask == 0) {printf("Computing combination %d, %d ....", N_i, N_j); fflush(stdout);};
          
          //Routine that determines if we can skip the FFTs assuming
          //the coefficients are zero for all these objects.
          ksum = 0;
          for (N_k = N_j; N_k < N_modes; N_k++){
            if (fabs(AlphaTable[N_i][N_j][N_k]) > 0) ksum = 1;
          }
          if (ksum == 0.0) {
            if (ThisTask == 0 ) {
              int print_size = 0;
              if (N_i >= 10) print_size += 1;
              if (N_j >= 10) print_size += 1;
              print_timed_done(23 - print_size);
            }
            continue;
          }
          
          rfftwnd_mpi(Inverse_plan, 1, fields[7], Workspace, FFTW_NORMAL_ORDER);
          MPI_Barrier(MPI_COMM_WORLD);

          memset(p_tmp, 0, sizeof(fftw_real) * TotalSizePlusAdditional);
          MPI_Barrier(MPI_COMM_WORLD);
          
          //Multiply terms in realspace [\phi gi] x [\phi gj]
          for(i = 0; i < Local_nx; i++)
            for(j = 0; j < Nmesh; j++)
              for(k = 0; k < Nmesh; k++)
                    {
                      coord = (i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k;

                      p_tmp[coord] = fields[3][coord] * fields[7][coord];
                  
                    }
          

          MPI_Barrier(MPI_COMM_WORLD);
          rfftwnd_mpi(Forward_plan, 1, p_tmp, Workspace, FFTW_NORMAL_ORDER);
          MPI_Barrier(MPI_COMM_WORLD);

          
          
          for(ii = 0; ii < Local_nx; ii++)
              for(j = 0; j < Nmesh; j++)
                  for(k = 0; k <= Nmesh / 2 ; k++)
                  {

                      coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
                      i = ii + Local_x_start;

                      if(i == 0 && j == 0 && k == 0)
                        {
                          cpot_NG[0].re=0.;
                          cpot_NG[0].im=0.;

                          continue;
                        }

                      kvec[0] = (i < Nmesh / 2) ? i * 2 * PI / Box : -(Nmesh - i) * 2 * PI / Box;
                      kvec[1] = (j < Nmesh / 2) ? j * 2 * PI / Box : -(Nmesh - j) * 2 * PI / Box;
                      kvec[2] = (k < Nmesh / 2) ? k * 2 * PI / Box : -(Nmesh - k) * 2 * PI / Box;

                      kmag2   = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
                      kmag1   = sqrt(kmag2);
                      kmag_ns = pow(kmag1, (4 - PrimordialIndex)/3);
                      kmag    = -1 + 2 * (kmag_ns - kmin_ns)/(kmax_ns - kmin_ns);

                      if(SphereMode == 1)
                        {
                          if(kmag1 * Box / (2 * PI) > Nsample / 2)       /* select a sphere in k-space */
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

                      fnl_fac = 0;
                      
                      a = NAN;
                      b = NAN;
                      c = NAN;
                      d = NAN;
                      
                      //Add the g3(k) kernel. This can be summed linearly for fixed gi and gj.
                      for(N_k = N_j; N_k < N_modes; N_k++){
                          
                          double np = ((double) N_k) - 1;
                          
                          if (N_k <= 3) {
                              fnl_fac += AlphaTable[N_i][N_j][N_k] * pow(kmag1, (4 - PrimordialIndex)/3 * (N_k - 3.));
                              
                          }
                          else { 
                              
                              //Fill in missing modes if start at N_k != 0
                              if ( (N_k == 4) || ( (N_j >= 4) && (N_j == N_k) )) {
                                  a = gsl_sf_legendre_Pl(N_k - 1 - 2, kmag)/kmag2; 
                                  b = gsl_sf_legendre_Pl(N_k - 1 - 1, kmag)/kmag2;
                              }
                              
                              c = ( (2 * np - 1) * kmag * b - (np - 1) * a ) / np;
                              d = c - X0Table[N_k - 1][0]/kmag2;
                                  
                              a = b * 1;
                              b = c * 1;
                              fnl_fac += AlphaTable[N_i][N_j][N_k] * d;
                              
                          }
                          
                          
                          
                      }//End of N_k loop
                      
                      
                      // cpot_NG[coord].re += fnl_fac * c_tmp[coord].re;
                      // cpot_NG[coord].im += fnl_fac * c_tmp[coord].im;

                      cpot_NG_re[coord] += (long double) (fnl_fac * c_tmp[coord].re);
                      cpot_NG_im[coord] += (long double) (fnl_fac * c_tmp[coord].im);
                      

                  }

          if (ThisTask == 0 ) {
            int print_size = 0;
            if (N_i >= 10) print_size += 1;
            if (N_j >= 10) print_size += 1;
            print_timed_done(23 - print_size);
          }
          MPI_Barrier(MPI_COMM_WORLD);
          }
      } //Close the Ni,Nj,Nk loop

  
  //Now need to include the extra templates for cancelling divergent terms :P
   
  //Free six of the fields (2..7), since we won't use them anymore
  for(N = 2; N < 8; N++) free(cfields[N]);
  
  for(N_i = 0; N_i <= 3; N_i++){
      
      memset(cfields[0], 0, sizeof(fftw_real) * TotalSizePlusAdditional); //Needed to prevent memory issue in loop
      
      //Generate the first field in fourier space
      for(ii = 0; ii < Local_nx; ii++)
          for(j = 0; j < Nmesh; j++)
              for(k = 0; k <= Nmesh / 2 ; k++)
              {

                  coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
                  i = ii + Local_x_start;

                  kvec[0] = (i < Nmesh / 2) ? i * 2 * PI / Box : -(Nmesh - i) * 2 * PI / Box;
                  kvec[1] = (j < Nmesh / 2) ? j * 2 * PI / Box : -(Nmesh - j) * 2 * PI / Box;
                  kvec[2] = (k < Nmesh / 2) ? k * 2 * PI / Box : -(Nmesh - k) * 2 * PI / Box;

                  kmag2   = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
                  kmag1   = sqrt(kmag2);

                  if(SphereMode == 1)
                    {
                      if(kmag1 * Box / (2 * PI) > Nsample / 2)       /* select a sphere in k-space */
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

                  if(i + j + k == 0) continue;

                  fnl_fac = pow(kmag1, (4 - PrimordialIndex)/3 * (N_i - 3.)) * pow(kmag1, 4 - PrimordialIndex);
                  
                  cfields[0][coord].re = fnl_fac * cpot[coord].re;
                  cfields[0][coord].im = fnl_fac * cpot[coord].im;
              
              }
      
      
      MPI_Barrier(MPI_COMM_WORLD);
      rfftwnd_mpi(Inverse_plan, 1, fields[0], Workspace, FFTW_NORMAL_ORDER);
      MPI_Barrier(MPI_COMM_WORLD);

      for(N_j = N_i; N_j <= 3; N_j++){
          
          memset(cfields[1], 0, sizeof(fftw_real) * TotalSizePlusAdditional); //Needed to prevent memory issue in loop
          
          //Generate the 2nd field, also in fourier space
          for(ii = 0; ii < Local_nx; ii++)
              for(j = 0; j < Nmesh; j++)
                  for(k = 0; k <= Nmesh / 2 ; k++)
                  {

                      coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
                      i = ii + Local_x_start;

                      kvec[0] = (i < Nmesh / 2) ? i * 2 * PI / Box : -(Nmesh - i) * 2 * PI / Box;
                      kvec[1] = (j < Nmesh / 2) ? j * 2 * PI / Box : -(Nmesh - j) * 2 * PI / Box;
                      kvec[2] = (k < Nmesh / 2) ? k * 2 * PI / Box : -(Nmesh - k) * 2 * PI / Box;

                      kmag2   = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
                      kmag1   = sqrt(kmag2);

                      if(SphereMode == 1)
                        {
                          if(kmag1 * Box / (2 * PI) > Nsample / 2)       /* select a sphere in k-space */
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

                      if(i + j + k == 0) continue;
                      
                      fnl_fac = pow(kmag1, (4 - PrimordialIndex)/3. * (N_j - 3.)) * pow(kmag1, 4 - PrimordialIndex);

                      cfields[1][coord].re = fnl_fac * cpot[coord].re;
                      cfields[1][coord].im = fnl_fac * cpot[coord].im;
                      
                      

                  }
          
          
          MPI_Barrier(MPI_COMM_WORLD);
          rfftwnd_mpi(Inverse_plan, 1, fields[1], Workspace, FFTW_NORMAL_ORDER);
          MPI_Barrier(MPI_COMM_WORLD);

          memset(p_tmp, 0, sizeof(fftw_real) * TotalSizePlusAdditional);
          MPI_Barrier(MPI_COMM_WORLD);
          
          //Multiply terms in realspace [\phi gi] x [\phi gj]
          for(i = 0; i < Local_nx; i++)
            for(j = 0; j < Nmesh; j++)
              for(k = 0; k < Nmesh; k++)
                    {
                      coord = (i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k; 
                      p_tmp[coord] = fields[0][coord] * fields[1][coord];

                    }

          
          MPI_Barrier(MPI_COMM_WORLD);
          rfftwnd_mpi(Forward_plan, 1, p_tmp, Workspace, FFTW_NORMAL_ORDER);
          MPI_Barrier(MPI_COMM_WORLD);
          
          if (ThisTask == 0) {printf("Computing Beta combination %d, %d ....", N_i, N_j); fflush(stdout);};

          for(ii = 0; ii < Local_nx; ii++)
              for(j = 0; j < Nmesh; j++)
                  for(k = 0; k <= Nmesh / 2 ; k++)
                  {

                      coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
                      i = ii + Local_x_start;

                      if(i == 0 && j == 0 && k == 0) continue;

                      kvec[0] = (i < Nmesh / 2) ? i * 2 * PI / Box : -(Nmesh - i) * 2 * PI / Box;
                      kvec[1] = (j < Nmesh / 2) ? j * 2 * PI / Box : -(Nmesh - j) * 2 * PI / Box;
                      kvec[2] = (k < Nmesh / 2) ? k * 2 * PI / Box : -(Nmesh - k) * 2 * PI / Box;

                      kmag2   = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
                      kmag1   = sqrt(kmag2);

                      if(SphereMode == 1)
                        {
                          if(kmag1 * Box / (2 * PI) > Nsample / 2)       /* select a sphere in k-space */
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

                      fnl_fac = 0;
                      
                      
                      //First handle the t_012 piece (cancels the divergent (0,1,2) term). 
                      {
                      if ( (N_i == 0) && (N_j == 1) ) { //Subtract from the original kernel
                          fnl_fac += -AlphaTable[0][1][2] * t_012 * pow(kmag1, (4 - PrimordialIndex)/3 * (2 - 3.0));
                          }
                          
                      if ( (N_i == 0) && (N_j == 2) ) //Now add the t_012 piece
                          fnl_fac +=  AlphaTable[0][1][2] * t_012 * pow(kmag1, (4 - PrimordialIndex)/3 * (1 - 3.0));

                      }

                      cpot_NG_re[coord] += (long double) (fnl_fac * c_tmp[coord].re);
                      cpot_NG_im[coord] += (long double) (fnl_fac * c_tmp[coord].im);  
                                          
                      
                  }

          if (ThisTask == 0 ) print_timed_done(18);
          MPI_Barrier(MPI_COMM_WORLD);
          }
      } //Close the Ni,Nj,Nk loop
  
  free(c_tmp);

  for(N = 0; N < 2; N++){
      free(cfields[N]);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  //Add NG potential to the full potential
  for(ii = 0; ii < Local_nx; ii++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k <= Nmesh / 2 ; k++){

          coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;

          //Copy the long-double accumulator into the FFTW complex array
          cpot_NG[coord].re = cpot_NG_re[coord]; 
          cpot_NG[coord].im = cpot_NG_im[coord];           
      }

  free(cpot_NG_re);
  free(cpot_NG_im);
  free(AlphaTable);
  free(X0Table);

  MPI_Barrier(MPI_COMM_WORLD);
  
  

  lpt_displacements_from_potential(cpot, cpot_NG, Fnl, Beta);
}
