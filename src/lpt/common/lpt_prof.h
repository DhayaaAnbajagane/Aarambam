/* Optional per-category timers of the 2LPT executables. Build with -DAARAMBAM_PROFILE to accumulate the wall time of
 * each category on every rank; lpt_prof_report() (lpt_core.c) then prints the minimum and maximum over ranks.
 * Without the flag they compile away. */
#ifndef AARAMBAM_LPT_PROF_H
#define AARAMBAM_LPT_PROF_H

#include <mpi.h>

enum lpt_prof_cat { PROF_FFT, PROF_FIELD, PROF_MULT, PROF_ACCUM, PROF_MEM, PROF_WAIT, PROF_NCAT };

#ifdef AARAMBAM_PROFILE
extern double lpt_prof_time[PROF_NCAT];     /* defined in allvars.c */
#define PROF_BEGIN(cat) { double _prof_t0 = MPI_Wtime();
#define PROF_END(cat)   lpt_prof_time[cat] += MPI_Wtime() - _prof_t0; }
#else
#define PROF_BEGIN(cat) {
#define PROF_END(cat)   }
#endif

#endif
