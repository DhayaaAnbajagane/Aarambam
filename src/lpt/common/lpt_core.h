/* Shared core of the 2LPT executables (common/lpt_core.c) and the hooks each method supplies. */
#ifndef AARAMBAM_LPT_CORE_H
#define AARAMBAM_LPT_CORE_H

#include "allvars.h"

#define ASSERT_ALLOC(cond) {                                                                                  \
  if(!(cond))                                                                                                 \
    {                                                                                                         \
      printf("failed to allocate %g Mbyte on Task %d\n", bytes / (1024.0 * 1024.0), ThisTask);                \
      printf("bailing out.\n");                                                                               \
      FatalError(1);                                                                                          \
    }                                                                                                         \
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
