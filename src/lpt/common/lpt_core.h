/* Shared core of the 2LPT executables (common/lpt_core.c) and the hooks each method supplies. */
#ifndef AARAMBAM_LPT_CORE_H
#define AARAMBAM_LPT_CORE_H

#include <stddef.h>
#include "allvars.h"
#include "lpt_fft.h"
#include "lpt_prof.h"

#define ASSERT_ALLOC(cond) {                                                                                  \
  if(!(cond))                                                                                                 \
    {                                                                                                         \
      printf("failed to allocate %g Mbyte on Task %d\n", bytes / (1024.0 * 1024.0), ThisTask);                \
      printf("bailing out.\n");                                                                               \
      FatalError(1);                                                                                          \
    }                                                                                                         \
}

/* wavenumber of grid index n along one axis (FFT order: n >= Nmesh/2 are the negative frequencies) */
static inline double lpt_wavenumber(ptrdiff_t n)
{
  return (n < Nmesh / 2) ? n * 2 * PI / Box : -(Nmesh - n) * 2 * PI / Box;
}

void lpt_barrier(void);                                  /* MPI_Barrier on MPI_COMM_WORLD, timed as PROF_WAIT */
void lpt_prof_report(void);

void print_timed_done(int n);
void print_timed_done_aligned(int printed_chars);
void lpt_print_banner(const char *name);                  /* boxed name at the top of the setup print-out */
void lpt_print_rule(void);                               /* a row of '*' */
void lpt_setup_gaussian_potential(lpt_complex **cpot_out, double *Beta_out);
void lpt_displacements_from_potential(lpt_complex *cpot, lpt_complex *cpot_NG, double ng_scale, double Beta);

/* supplied by each method */
void print_setup(void);
void displacement_fields(void);

#endif
