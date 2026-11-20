/* Shared core of the 2LPT executables (common/lpt_core.c) and the hooks each method supplies. */
#ifndef AARAMBAM_LPT_CORE_H
#define AARAMBAM_LPT_CORE_H

#include <stddef.h>
#include "allvars.h"

#define ASSERT_ALLOC(cond) {                                                                                  \
  if(!(cond))                                                                                                 \
    {                                                                                                         \
      printf("failed to allocate %g Mbyte on Task %d\n", bytes / (1024.0 * 1024.0), ThisTask);                \
      printf("bailing out.\n");                                                                               \
      FatalError(1);                                                                                          \
    }                                                                                                         \
}

/* Optional per-category timers. Build with -DAARAMBAM_PROFILE to accumulate the wall time of each category
 * on every rank; lpt_prof_report() then prints the maximum over ranks. Without the flag they compile away. */
enum lpt_prof_cat { PROF_FFT, PROF_FIELD, PROF_MULT, PROF_ACCUM, PROF_MEM, PROF_WAIT, PROF_NCAT };
#ifdef AARAMBAM_PROFILE
extern double lpt_prof_time[PROF_NCAT];
#define PROF_BEGIN(cat) { double _prof_t0 = MPI_Wtime();
#define PROF_END(cat)   lpt_prof_time[cat] += MPI_Wtime() - _prof_t0; }
#else
#define PROF_BEGIN(cat) {
#define PROF_END(cat)   }
#endif
void lpt_fft(rfftwnd_mpi_plan plan, fftw_real *field);   /* in-place rfftwnd_mpi with Workspace, timed */
void lpt_fft_forward_t(fftw_real *field);               /* forward FFT, output left in transposed order */
void lpt_transposed_to_normal(fftw_complex *data);      /* transposed -> normal order k-space, in place */
void lpt_barrier(void);                                   /* MPI_Barrier on MPI_COMM_WORLD, timed as PROF_WAIT */
void lpt_prof_report(void);

/* Conservative band test on one grid axis: an index magnitude above Nsample / 2 + 1 is outside the sampled band
 * for either SphereMode, so k-space loops can skip whole planes and rows before their exact per-mode test. */
static inline int lpt_axis_in_band(ptrdiff_t n)
{
  ptrdiff_t m = (n < Nmesh / 2) ? n : Nmesh - n;
  return m <= Nsample / 2 + 1;
}

/* Last z index (of 0 .. Nmesh / 2 in the r2c layout) that can be in band. */
static inline ptrdiff_t lpt_kz_end(void)
{
  return (Nsample / 2 + 1 < Nmesh / 2) ? Nsample / 2 + 1 : Nmesh / 2;
}

void print_timed_done(int n);
void print_timed_done_aligned(int printed_chars);
void lpt_setup_gaussian_potential(fftw_complex **cpot_out, long double **re_out, long double **im_out,
                                  double *Beta_out);
void lpt_displacements_from_potential(fftw_complex *cpot, fftw_complex *cpot_NG, double ng_scale, double Beta);

/* supplied by each method */
void print_setup(void);
void displacement_fields(void);

#endif
