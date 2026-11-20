#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <mpi.h>
#include <gsl/gsl_rng.h>
#include "allvars.h"
#include "proto.h"
#include "lpt_core.h"
#include <signal.h>
#include <assert.h>

/* Code shared by the modal (2LPTBasis) and binned (2LPTBinned) executables: program driver, timing,
 * the Gaussian potential, and everything after the method-specific non-Gaussian term (adding it,
 * writing the potentials, Zel'dovich + 2LPT displacements, particle read-out). Each method supplies
 * print_setup() and displacement_fields(). */

#define TIMED_DONE_COLUMN 54


/* ---- barrier and the optional profiling timers ---- */

void lpt_barrier(void)
{
  PROF_BEGIN(PROF_WAIT)
  MPI_Barrier(MPI_COMM_WORLD);
  PROF_END(PROF_WAIT)
}

void lpt_prof_report(void)
{
#ifdef AARAMBAM_PROFILE
  const char *name[PROF_NCAT] = {"FFTs", "field build (filters/kernels)", "real-space products",
                                 "NG accumulation", "memset/memcpy", "barrier wait"};
  double      tmax[PROF_NCAT], tmin[PROF_NCAT];

  MPI_Reduce(lpt_prof_time, tmax, PROF_NCAT, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(lpt_prof_time, tmin, PROF_NCAT, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      printf("\nAARAMBAM_PROFILE (min / max over ranks):\n");
      for(int c = 0; c < PROF_NCAT; c++)
        printf("  %-32s %10.3f %10.3f s\n", name[c], tmin[c], tmax[c]);
    }
#endif
}


/* ---- progress output ---- */

void print_timed_done(int n)
{
  clock_t now  = clock();
  double  tot  = ((double)(now - start_time)) / CLOCKS_PER_SEC;
  double  diff = ((double)(now - previous_time)) / CLOCKS_PER_SEC;

  int    tot_hours = (int) floor(tot / 3600.0);
  int    tot_mins  = (int) floor(tot / 60.0) - 60 * tot_hours;
  double tot_secs  = tot - 60.0 * (((double) tot_mins) + 60.0 * ((double) tot_hours));

  int    diff_hours = (int) floor(diff / 3600.0);
  int    diff_mins  = (int) floor(diff / 60.0) - 60 * diff_hours;
  double diff_secs  = diff - 60.0 * (((double) diff_mins) + 60.0 * ((double) diff_hours));

  for(int i = 0; i < n; i++)
    printf(" ");
  printf("Done [%02d:%02d:%05.2f, %02d:%02d:%05.2f]\n",
         diff_hours, diff_mins, diff_secs, tot_hours, tot_mins, tot_secs);

  previous_time = now;
  return;
}

void print_timed_done_aligned(int printed_chars)
{
  int indent = TIMED_DONE_COLUMN - printed_chars;

  if(indent < 1)
    indent = 1;
  print_timed_done(indent);
}

void handle_signal(int signal)
{
  // Handle the signal, clean up resources
  MPI_Finalize();
  exit(0);
}


/* ---- program driver ---- */

int main(int argc, char **argv)
{
  int timed_chars = 0;

  MPI_Init(&argc, &argv);
  fftw_mpi_init();
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  // Setup signal handling
  signal(SIGINT, handle_signal);

  start_time    = clock();
  previous_time = start_time;

  if(argc < 2)
    {
      if(ThisTask == 0)
        {
          fprintf(stdout, "\nParameters are missing.\n");
          fprintf(stdout, "Call with <ParameterFile>\n\n");
        }
      MPI_Finalize();
      exit(0);
    }

  if(ThisTask == 0)
    printf("RUNNING LPTPNG WITH %d TASKS", NTask);

  if(ThisTask == 0)
    read_parameterfile(argv[1]);
  broadcast_parameters();

  lpt_barrier();
  set_units();
  initialize_transferfunction();
  initialize_powerspectrum();
  initialize_ffts();
  read_glass(GlassFile);

  if(ThisTask == 0)
    print_setup();

  displacement_fields();

  if(ThisTask == 0)
    {
      timed_chars = printf("Writing initial conditions snapshot...");
      fflush(stdout);
    }
  write_particle_data();
  if(ThisTask == 0)
    print_timed_done_aligned(timed_chars);

  if(NumPart)
    free(P);
  free_ffts();
  lpt_barrier();

  print_spec();
  lpt_prof_report();

  MPI_Finalize();     /* clean up & finalize MPI */
  exit(0);
}


/* ---- Gaussian potential ---- */

/* Gaussian primordial potential on this rank's slab (the random phases do not depend on the number of
 * ranks). Allocates and returns cpot, and the factor Beta. */
void lpt_setup_gaussian_potential(lpt_complex **cpot_out, double *Beta_out)
{
  gsl_rng      *random_generator;
  ptrdiff_t     i, j, k, ii, jj;
  int           timed_chars = 0;
  double        phig, Beta;
  lpt_complex  *cpot;              /* Gaussian potential */
  double        fac;
  double        kvec[3], kmag, kmag2;
  double        phase, ampl;
  unsigned int *seedtable;
  double        phase_shift;       /* pi for PhaseFlip = 1 */
  uint64_t      bytes = 0;
  uint64_t      coord;

  phase_shift = 0.0;
  if(PhaseFlip == 1)
    phase_shift = PI;

  fac = pow(2 * PI / Box, 1.5);

  /* one seed per (i, j) column, drawn in a fixed order, so the phases do not depend on the number of ranks */
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

  if(ThisTask == 0)
    {
      timed_chars = printf("Setting Gaussian potential...");
      fflush(stdout);
    }

  bytes = 0; /*initialize*/
  cpot  = (lpt_complex *) lpt_alloc_grid();
  ASSERT_ALLOC(cpot);

  memset(cpot, 0, sizeof(lpt_real) * TotalSizePlusAdditional);

  /* Ho in units of UnitLength_in_cm and c=1, i.e., internal units so far  */
  Beta = 1.5 * Omega / (2998. * 2998. / UnitLength_in_cm / UnitLength_in_cm * 3.085678e24 * 3.085678e24 ) / D0 ;

  for(i = 0; i < Nmesh; i++)
    {
      ii = Nmesh - i;
      if(ii == Nmesh)
        ii = 0;

      /* this rank holds column i or its conjugate */
      if((i >= Local_x_start && i < (Local_x_start + Local_nx)) ||
         (ii >= Local_x_start && ii < (Local_x_start + Local_nx)))
        {
          for(j = 0; j < Nmesh; j++)
            {
              if(!lpt_axis_in_band(i) || !lpt_axis_in_band(j))      /* no sampled mode in this column */
                continue;
              gsl_rng_set(random_generator, seedtable[i * Nmesh + j]);

              for(k = 0; k < Nmesh / 2 && k <= Nsample / 2 + 1; k++)   /* later draws are not used */
                {
                  phase  = gsl_rng_uniform(random_generator) * 2 * PI;
                  phase += phase_shift;

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
                  kmag  = sqrt(kmag2);

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
                  switch(WhichSpectrum)    /* oscillating primordial P(k), see PowerSpec() in power.c */
                    {
                    case 0:
                      phig *= 1.0;
                      break;

                    case 1:
                      phig *= (1.0 + A_res * cos(w_res * kmag + phi_res));
                      break;

                    case 2:
                      phig *= (1.0 + A_res * cos(w_res * log(kmag) + phi_res));
                      break;
                    }

                  if(!FixedAmplitude)       /* Rayleigh-distributed amplitude unless FixedAmplitude */
                    phig *= -log(ampl);

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

  if(ThisTask == 0)
    print_timed_done_aligned(timed_chars);

  gsl_rng_free(random_generator);
  free(seedtable);

  *cpot_out = cpot;
  *Beta_out = Beta;
}


/* ---- NG term added, potentials written, ZA + 2LPT displacements ---- */

/* Everything after the non-Gaussian term: cpot (Gaussian, Fourier space) += ng_scale * cpot_NG,
 * potentials written if SavePotentialField, then the ZA and 2LPT displacements and velocities of P[].
 * Frees cpot and cpot_NG. ng_scale is Fnl for the modal method and 1 for the binned one (whose
 * kernel already includes Fnl). */
void lpt_displacements_from_potential(lpt_complex *cpot, lpt_complex *cpot_NG, double ng_scale, double Beta)
{
  MPI_Request   request;
  MPI_Status    status;
  ptrdiff_t     i, j, k, ii, jj, kk, axes;
  int           n;
  int           timed_chars = 0;
  int           sendTask, recvTask;
  double        t_of_k, twb;
  lpt_real     *pot;
  double        vel_prefac, vel_prefac2;
  double        kvec[3], kmag, kmag2;
  double        hubble_a;
  double        u, v, w;
  double        f1, f2, f3, f4, f5, f6, f7, f8;
  double        dis, dis2, maxdisp, max_disp_glob;
  uint64_t      bytes = 0, nmesh3;
  uint64_t      coord;
  lpt_complex  *(cdisp[3]), *(cdisp2[3]);  /* ZA and 2nd order displacements */
  lpt_real     *(disp[3]), *(disp2[3]);
  lpt_complex  *(cdigrad[6]);              /* displacement gradient */
  lpt_real     *(digrad[6]);

#ifdef CORRECT_CIC
  double fx, fy, fz, ff, smth;
#endif

  pot = (lpt_real *) cpot;

  hubble_a    = Hubble * sqrt(Omega / pow(InitTime, 3) + (1 - Omega - OmegaLambda) / pow(InitTime, 2) + OmegaLambda);
  vel_prefac  = InitTime * hubble_a * F_Omega(InitTime) / sqrt(InitTime);
  vel_prefac2 = InitTime * hubble_a * F2_Omega(InitTime) / sqrt(InitTime);

  maxdisp = 0;

  //Transform to real space and back only to write out the potential (the round trip multiplies it by nmesh3)
  if(SavePotentialField == 1)
    {
      lpt_fft(Inverse_plan, pot);
      write_phi(pot, 0);
      lpt_barrier();
      lpt_fft(Forward_plan, pot);
    }

  nmesh3 = ((uint64_t) Nmesh) * ((uint64_t) Nmesh ) * ((uint64_t) Nmesh);
  const double rt = (SavePotentialField == 1) ? (double) nmesh3 : 1.0;   /* FFT round trip of the potential */

  //Add NG potential to the full potential
  for(ii = 0; ii < Local_nx; ii++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k <= Nmesh / 2 ; k++)
        {
          coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;

          //Undo the factor nmesh3 of the save-potential round trip above, if there was one
          cpot[coord].re /= rt;
          cpot[coord].im /= rt;

          //One factor due to converting c_tmp from real space to fourier space
          cpot[coord].re += ng_scale * cpot_NG[coord].re / (double) nmesh3;
          cpot[coord].im += ng_scale * cpot_NG[coord].im / (double) nmesh3;
        }

  //Zero-out the mean mode alone
  if(ThisTask == 0)
    {
      cpot[0].re = 0.;
      cpot[0].im = 0.;
    }

  free(cpot_NG);

  lpt_barrier();

  if(ThisTask == 0)
    {
      timed_chars = printf("Writing potential to disc...");
      fflush(stdout);
    }
  if(SavePotentialField == 1)
    {
      lpt_fft(Inverse_plan, pot);
      write_phi(pot, 1);
      lpt_barrier();
      lpt_fft(Forward_plan, pot);
    }

  for(ii = 0; ii < Local_nx; ii++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k <= Nmesh / 2 ; k++)
        {
          coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;

          cpot[coord].re = cpot[coord].re / rt;
          cpot[coord].im = cpot[coord].im / rt;
        }

  if(ThisTask == 0)
    print_timed_done_aligned(timed_chars);

  lpt_barrier();
  if(ThisTask == 0)
    {
      timed_chars = printf("Computing gradient of non-Gaussian potential...");
      fflush(stdout);
    }


  /*****  Now 2LPT ****/

  for(axes = 0, bytes = 0; axes < 3; axes++)
    {
      cdisp[axes] = (lpt_complex *) lpt_alloc_grid();
      disp[axes]  = (lpt_real *) cdisp[axes];
    }
  ASSERT_ALLOC(cdisp[0] && cdisp[1] && cdisp[2]);

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

  /* ZA displacement: dis = i k T(k) phi / (D+ Beta) */
  for(ii = 0; ii < Local_nx; ii++)
    for(j = 0; j < Nmesh; j++)
      for(k = 0; k <= Nmesh / 2 ; k++)
        {
          coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
          i     = ii + Local_x_start;

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
          kmag  = sqrt(kmag2);

          t_of_k = TransferFunc(kmag);
          twb    = t_of_k / Dplus  / Beta;

          for(axes = 0; axes < 3; axes++)
            {
              cdisp[axes][coord].im = kvec[axes] * twb * cpot[coord].re;
              cdisp[axes][coord].re = - kvec[axes] * twb * cpot[coord].im;
            }
        }

  free(cpot);

  if(ThisTask == 0)
    print_timed_done_aligned(timed_chars);

  lpt_barrier();

  /* Compute displacement gradient */

  if(ThisTask == 0)
    {
      timed_chars = printf("Computing 2LPT potential...");
      fflush(stdout);
    }

  for(i = 0; i < 6; i++)
    {
      cdigrad[i] = (lpt_complex *) lpt_alloc_grid();
      digrad[i]  = (lpt_real *) cdigrad[i];
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

  for(i = 0; i < 6; i++)
    lpt_fft(Inverse_plan, digrad[i]);

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

  lpt_fft(Forward_plan, digrad[3]);

  /* The memory allocated for cdigrad[0], [1], and [2] will be used for 2nd order displacements */
  /* Freeing the rest. cdigrad[3] still has 2nd order displacement source, free later */
  for(axes = 0; axes < 3; axes++)
    {
      cdisp2[axes] = cdigrad[axes];
      disp2[axes]  = (lpt_real *) cdisp2[axes];
    }

  free(cdigrad[4]);
  free(cdigrad[5]);

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
          ff   = 1 / (fx * fy * fz);
          smth = ff * ff;
#endif

          /* cdisp2 = source * k / (sqrt(-1) k^2) */
          for(axes = 0; axes < 3; axes++)
            {
              if(kmag2 > 0.0)
                {
                  cdisp2[axes][coord].re = cdigrad[3][coord].im * kvec[axes] / kmag2;
                  cdisp2[axes][coord].im = -cdigrad[3][coord].re * kvec[axes] / kmag2;
                }
              else
                cdisp2[axes][coord].re = cdisp2[axes][coord].im = 0.0;

#ifdef CORRECT_CIC
              cdisp[axes][coord].re  *= smth;
              cdisp[axes][coord].im  *= smth;
              cdisp2[axes][coord].re *= smth;
              cdisp2[axes][coord].im *= smth;
#endif
            }
        }

  /* Free cdigrad[3] */
  free(cdigrad[3]);

  lpt_barrier();

  /* Now, both cdisp, and cdisp2 have the ZA and 2nd order displacements */
  for(axes = 0; axes < 3; axes++)
    {
      lpt_fft(Inverse_plan, disp[axes]);
      lpt_fft(Inverse_plan, disp2[axes]);

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
                    sizeof(lpt_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
                    MPI_BYTE, recvTask, 10, MPI_COMM_WORLD, &request);

          MPI_Recv(&(disp[axes][(Local_nx * Nmesh) * (2 * (Nmesh / 2 + 1))]),
                   sizeof(lpt_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
                   MPI_BYTE, sendTask, 10, MPI_COMM_WORLD, &status);

          MPI_Wait(&request, &status);

          /* send 2nd order disp */
          MPI_Isend(&(disp2[axes][0]),
                    sizeof(lpt_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
                    MPI_BYTE, recvTask, 10, MPI_COMM_WORLD, &request);

          MPI_Recv(&(disp2[axes][(Local_nx * Nmesh) * (2 * (Nmesh / 2 + 1))]),
                   sizeof(lpt_real) * Nmesh * (2 * (Nmesh / 2 + 1)),
                   MPI_BYTE, sendTask, 10, MPI_COMM_WORLD, &status);

          MPI_Wait(&request, &status);
        }
    }

  if(ThisTask == 0)
    print_timed_done_aligned(timed_chars);
  if(ThisTask == 0)
    {
      timed_chars = printf("Computing displacements and velocitites...");
      fflush(stdout);
    }

  /* read-out displacements: CIC interpolation of the ZA and 2LPT fields at each particle */
  nmesh3 = (uint64_t)Nmesh * (uint64_t)Nmesh *(uint64_t) Nmesh;

  for(n = 0; n < NumPart; n++)
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
          P[n].Vel[axes]  = dis * vel_prefac - 3./7. * dis2 * vel_prefac2;
          P[n].Pos[axes]  = periodic_wrap(P[n].Pos[axes]);

          if(fabs(dis - 3./7. * dis2) > maxdisp)
            maxdisp = fabs(dis - 3./7. * dis2);
        }
    }

  if(ThisTask == 0)
    print_timed_done_aligned(timed_chars);

  for(axes = 0; axes < 3; axes++)
    free(cdisp[axes]);
  for(axes = 0; axes < 3; axes++)
    free(cdisp2[axes]);

  /* random_generator and seedtable are freed in lpt_setup_gaussian_potential */

  MPI_Reduce(&maxdisp, &max_disp_glob, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}


/* ---- utilities ---- */

double periodic_wrap(double x)
{
  return fmod(fmod(x, Box) + Box, Box);
}

void set_units(void)     /* ... set some units */
{
  UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s;
  G             = GRAVITY / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2);
  Hubble        = HUBBLE * UnitTime_in_s;
}

int FatalError(int errnum)
{
  printf("FatalError called with number=%d\n", errnum);
  fflush(stdout);
  MPI_Abort(MPI_COMM_WORLD, errnum);
  exit(0);
}


/* ---- input spectrum written next to the ICs (with the Peacock & Dodds non-linear estimate) ---- */

static double A, B, alpha, beta, V, gf;

double fnl(double x)     /* Peacock & Dodds formula */
{
  return x * pow((1 + B * beta * x + pow(A * x, alpha * beta)) /
                 (1 + pow(pow(A * x, alpha) * gf * gf * gf / (V * sqrt(x)), beta)), 1 / beta);
}

void print_spec(void)
{
  double k, knl, po, dl, dnl, neff, kf, kstart, kend, po2, po1, DDD;
  char   buf[1100];
  FILE  *fd;

  if(ThisTask == 0)
    {
      sprintf(buf, "%s/inputspec_%s.txt", OutputDir, FileBase);

      fd = fopen(buf, "w");

      gf  = GrowthFactor(0.001, 1.0) / (1.0 / 0.001);
      DDD = GrowthFactor(1.0 / (Redshift + 1), 1.0);

      /* print actual starting redshift and linear growth factor for this cosmology */
      fprintf(fd, "%12g %12g\n", Redshift, DDD);

      kstart = 2 * PI / (1000.0 * (3.085678e24 / UnitLength_in_cm));    /* 1000 Mpc/h */
      kend   = 2 * PI / (0.001 * (3.085678e24 / UnitLength_in_cm));     /* 0.001 Mpc/h */

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
                  A     = 0.482 * pow(1 + neff / 3, -0.947);
                  B     = 0.226 * pow(1 + neff / 3, -1.778);
                  alpha = 3.310 * pow(1 + neff / 3, -0.244);
                  beta  = 0.862 * pow(1 + neff / 3, -0.287);
                  V     = 11.55 * pow(1 + neff / 3, -0.423) * 1.2;

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
