#include <math.h>
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

#define ASSERT_ALLOC(cond) {                                                                                  \
  if(!cond)                                                                                                   \
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

void print_timed_done(int n) {
  clock_t tot_time = clock() - start_time;
  int tot_hours = (int) floor(((double) tot_time) / 60. / 60. / CLOCKS_PER_SEC);
  int tot_mins = (int) floor(((double) tot_time) / 60. / CLOCKS_PER_SEC) - 60 * tot_hours;
  double tot_secs = (((double) tot_time) / CLOCKS_PER_SEC) - 60. * (((double) tot_mins) * 60. * ((double) tot_hours));
  clock_t diff_time = clock() - previous_time;
  int diff_hours = (int) floor(((double) diff_time) / 60. / 60. / CLOCKS_PER_SEC);
  int diff_mins = (int) floor(((double) diff_time) / 60. / CLOCKS_PER_SEC) - 60 *diff_hours;
  double diff_secs = (((double) diff_time) / CLOCKS_PER_SEC) - 60. * (((double) diff_mins) * 60. * ((double) diff_hours));
  for (int i = 0; i < n; i++)
   printf(" ");
  printf("Done [%02d:%02d:%05.2f, %02d:%02d:%05.2f]\n", diff_hours, diff_mins, diff_secs, tot_hours, tot_mins, tot_secs);
  previous_time = clock();
  return;
}

void handle_signal(int signal) {
    // Handle the signal, clean up resources
    MPI_Finalize();
    exit(0);
}

int main(int argc, char **argv) {

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

  if (ThisTask == 0) {printf("Writing initial conditions snapshot..."); fflush(stdout);};
  write_particle_data();
  if (ThisTask == 0 ) print_timed_done(16);
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

  char exec[] = "2LPTAarambam";


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
  printf("   HubbleParam = %.4f  OmegaDM_2ndSpecies = %.2e          fNL = %+.2e\n\n", HubbleParam, OmegaDM_2ndSpecies, Fnl); 
  printf("   FixedAmplitude = %d    PhaseFlip = % d   SphereMode = %d    Seed = %d\n", FixedAmplitude, PhaseFlip, SphereMode, Seed);   

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


void displacement_fields(void) {
  MPI_Request request;
  MPI_Status status;
  gsl_rng *random_generator;
  int i, j, k, ii, jj, kk, axes;
  int n;
  int sendTask, recvTask;
  int ksum;
  double t_of_k, phig, Beta, twb;
  fftw_complex *(cpot);  /* For computing nongaussian fnl ic */
  fftw_real *(pot);
    
  fftw_complex *(cpot_NG);  /* For computing nongaussian fnl ic */
  fftw_real *(pot_NG);
    
  fftw_complex *(c_tmp); /* For non-local fluctuations */    
  fftw_real *(p_tmp); /* For non-local fluctuations */

  long double *(cpot_NG_re);
  long double *(cpot_NG_im);
    
  unsigned int N, N_i, N_j, N_k;
  double fac, vel_prefac, vel_prefac2;
  double kvec[3], kmag, kmag2, kmag1, kmag_ns;
  double phase, ampl, hubble_a;
  double u, v, w;
  double f1, f2, f3, f4, f5, f6, f7, f8;
  double dis, dis2, maxdisp, max_disp_glob;
  long double fnl_fac;
  unsigned int *seedtable;
// ******* FAVN *****
  double phase_shift; 
// ******* FAVN *****
    
  double a, b, c, d; //For holding coeffs

  unsigned int bytes, nmesh3;
  int coord;
  fftw_complex *(cdisp[3]), *(cdisp2[3]) ; /* ZA and 2nd order displacements */
  fftw_real *(disp[3]), *(disp2[3]) ;

  fftw_complex *(cdigrad[6]);
  fftw_real *(digrad[6]);
    
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

      sprintf(buf, FileWithAlphaCoeff);

      if (!(fd = fopen(buf, "r")))
      {
          printf("can't read alpha coefficients in file '%s' on task %d\n", buf, ThisTask);
          FatalError(17);
      }

      while (fscanf(fd, " %d %d %d %lf ", &n1, &n2, &n3, &alpha) == 4)
      {
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
      double alpha;
      int n1;

      sprintf(buf, FileWithX0Coeff);
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


  if (ThisTask == 0) {printf("Setting Gaussian potential..."); fflush(stdout);};

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
  /* Beta = 3/2 H(z)^2 a^3 Om(a) / D0 = 3/2 Ho^2 Om0 / D0 at redshift z = 0.0 */ 
  Beta = 1.5 * Omega / (2998. * 2998. / UnitLength_in_cm / UnitLength_in_cm * 3.085678e24 * 3.085678e24 ) / D0 ;     

  // if(ThisTask == 0){
  //     printf("\n BETA: %e \n", Beta);
  //     printf("UnitLength_in_cm: %e \n", UnitLength_in_cm);   
  //     printf("D0: %e \n", D0);
  //     printf("Anorm: %e \n", Anorm);   
  // }
  
  
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

 if (ThisTask == 0 ) print_timed_done(25);
 /*** For non-local models it is important to keep all factors of SQRT(-1) as done below ***/
 /*** Notice also that there is a minus to convert from Bardeen to gravitational potential ***/

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
   
  //Free four of the fields, since we won't use them anymore
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
                      
                      
                      //First handle the s_012 and t_012 piece. 
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

          //One factor of nmesh3 due to the FFT from the save potential step above
          cpot_NG[coord].re = cpot_NG_re[coord]; 
          cpot_NG[coord].im = cpot_NG_im[coord];           
      }

  free(cpot_NG_re);
  free(cpot_NG_im);
  free(AlphaTable);
  free(X0Table);

  MPI_Barrier(MPI_COMM_WORLD);
  
  

  //Trasnform to real space so we can write out the potential
  rfftwnd_mpi(Inverse_plan, 1, pot, Workspace, FFTW_NORMAL_ORDER);
  if(SavePotentialField==1) write_phi(pot, 0);
  // if(ThisTask == 0) printf("\n coord = 10 \n pot = %lg\n\n", pot[10]);
  MPI_Barrier(MPI_COMM_WORLD);   
  rfftwnd_mpi(Forward_plan, 1, pot, Workspace, FFTW_NORMAL_ORDER);

  nmesh3 = ((unsigned int) Nmesh) * ((unsigned int) Nmesh ) * ((unsigned int) Nmesh);    

  //Add NG potential to the full potential
  for(ii = 0; ii < Local_nx; ii++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k <= Nmesh / 2 ; k++){

          coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;

          //One factor of nmesh3 due to the FFT from the save potential step above
          cpot[coord].re /= (double) nmesh3; 
          cpot[coord].im /= (double) nmesh3; 
          
          //One factor due to converting c_tmp from real space to fourier space
          cpot[coord].re += Fnl * cpot_NG[coord].re / (double) nmesh3; 
          cpot[coord].im += Fnl * cpot_NG[coord].im / (double) nmesh3;
          
          
      }
    
  
  //Zero-out the mean mode alone
  if(ThisTask == 0) {
          cpot[0].re = 0.;
          cpot[0].im = 0.; 
       }

  
  free(cpot_NG);

MPI_Barrier(MPI_COMM_WORLD);
    
if (ThisTask == 0) {printf("Writing potential to disc..."); fflush(stdout);};
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
if (ThisTask == 0 ) print_timed_done(26);

MPI_Barrier(MPI_COMM_WORLD);
if (ThisTask == 0) {printf("Computing gradient of non-Gaussian potential..."); fflush(stdout);};

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
        cdisp[axes][(i * Nmesh + j) * (Nmesh / 2 + 1) + k].re = 0;
        cdisp[axes][(i * Nmesh + j) * (Nmesh / 2 + 1) + k].im = 0;
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

  if (ThisTask == 0 ) print_timed_done(7);


       MPI_Barrier(MPI_COMM_WORLD);
 
      /* Compute displacement gradient */

      if (ThisTask == 0) {printf("Computing 2LPT potential..."); fflush(stdout);};

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

      if (ThisTask == 0 ) print_timed_done(27);
      if (ThisTask == 0) {printf("Computing displacements and velocitites..."); fflush(stdout);};

      /* read-out displacements */

      nmesh3 = Nmesh * Nmesh * Nmesh;
      
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
 
      if (ThisTask == 0 ) print_timed_done(12);

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
  char buf[1000];
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
