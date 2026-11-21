#ifndef AARAMBAM_BINNED_VARS_H
#define AARAMBAM_BINNED_VARS_H

/* Binned PNG kernel table (parameters defined in params_binned.c).
 * BinnedPNGFilter: 1=heaviside/top-hat, 2=linear, 3=cubic.
 * BinnedPNGCoordinate: 1=k, 2=log(k).
 * The internal legs k1 and k2 share the k1 edges; k3 (the output leg) has its own.
 * BinnedPNGMultiGrid: 1 = each bin pair on the smallest alias-free grid (binned_multigrid.h), 0 = all on Nmesh^3.
 * The loaded kernel table is always multiplied by the config Fnl.
 */
extern char     FileWithBinnedK1Edges[500];
extern char     FileWithBinnedK3Edges[500];
extern char     FileWithBinnedKernelValues[500];
extern int      BinnedPNGFilter;
extern int      BinnedPNGOutputFilter;
extern int      BinnedPNGCoordinate;
extern int      BinnedPNGCacheBlockSize;
extern int      BinnedPNGMultiGrid;

#endif
