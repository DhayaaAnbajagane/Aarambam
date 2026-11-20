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

void lpt_barrier(void);                                   /* MPI_Barrier on MPI_COMM_WORLD, timed as PROF_WAIT */
void lpt_prof_report(void);

void print_timed_done(int n);
void print_timed_done_aligned(int printed_chars);
void lpt_setup_gaussian_potential(lpt_complex **cpot_out, double *Beta_out);
void lpt_displacements_from_potential(lpt_complex *cpot, lpt_complex *cpot_NG, double ng_scale, double Beta);

/* supplied by each method */
void print_setup(void);
void displacement_fields(void);

#endif
