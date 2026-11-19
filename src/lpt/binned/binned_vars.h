#ifndef AARAMBAM_BINNED_VARS_H
#define AARAMBAM_BINNED_VARS_H

/* Binned PNG kernel table (parameters defined in params_binned.c).
 * The parameter file gives explicit file paths rather than a prefix.
 * BinnedPNGFilter: 1=heaviside/top-hat, 2=linear, 3=cubic.
 * BinnedPNGCoordinate: 1=k, 2=log(k).
 * The loaded kernel table is always multiplied by the config Fnl.
 */
extern char     FileWithBinnedK1Edges[500];
extern char     FileWithBinnedK2Edges[500];
extern char     FileWithBinnedK3Edges[500];
extern char     FileWithBinnedKernelValues[500];
extern int      BinnedPNGFilter;
extern int      BinnedPNGOutputFilter;
extern int      BinnedPNGCoordinate;
extern int      BinnedPNGUseInternalSymmetry;
extern int      BinnedPNGCacheBlockSize;

#endif
