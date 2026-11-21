/* FFT module of the 2LPT executables (lpt_fft.c): grid allocation, the FFTW3-MPI plans and their wisdom file, and
 * the FFTs of the non-Gaussian stage behind a small API whose backend is chosen at run time (FFTBackend).
 *
 * Layouts. Real grids and "normal" k-space grids are x-slabs: real [Local_nx][Nmesh][2 (Nmesh/2+1)], complex
 * [Local_nx][Nmesh][Nmesh/2+1], x = Local_x_start + ii. The non-Gaussian stage works on the sampled band of k-space
 * (|kx|, |ky|, kz <= Nsample/2 + 1, see lpt_axis_in_band) and uses a "band" k-space layout: Band_ny rows of ky per
 * rank, row jj holding ky = Band_y[jj] for all kx and kz < Band_nz, at
 *
 *     coord = lpt_band_coord(jj, i, k) = jj * Band_sy + i * Band_sx + k,   x = i,  y = Band_y[jj],  z = k
 *
 * FFTBackend fftw: FFTW3-MPI's transposed output, [y (Band_ny = FFTW's y-slab)][x][z (Band_nz = Nmesh/2+1)].
 * FFTBackend band: band_fft.c's layout, [x][y (Band_ny = an even share of the in-band rows)][z (Band_nz = kz_end+1)].
 *
 * The API (all in place, unnormalised, collective):
 *     lpt_fft(Forward_plan | Inverse_plan, f)   full r2c / c2r, normal layout (FFTW3-MPI, any backend)
 *     lpt_fft_forward_band(f)                   r2c of a real grid when only the sampled band of the result is
 *                                               needed: k-space result in the band layout
 *     lpt_fft_inverse_band(f)                   c2r of a normal-layout k-space grid that is zero outside the band
 *     lpt_band_to_normal(g)                     band layout -> normal layout (zero outside the band)
 * Every grid an FFT runs on must come from lpt_alloc_grid(). */
#ifndef AARAMBAM_LPT_FFT_H
#define AARAMBAM_LPT_FFT_H

#include <stddef.h>
#include "allvars.h"

extern int  Band_ny;     /* rows of the band layout on this rank */
extern int *Band_y;      /* their ky indices */
extern int  Band_nz;     /* kz stored per (x, y) */
extern ptrdiff_t Band_sy, Band_sx;   /* strides of a row and of x */

static inline ptrdiff_t lpt_band_coord(ptrdiff_t jj, ptrdiff_t i, ptrdiff_t k)
{
  return jj * Band_sy + i * Band_sx + k;
}

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

/* NGPrecision float: the non-Gaussian stage's real-space fields, products and their FFTs in single precision, on float
 * grids from lpt_alloc_grid_f (same element count, half the bytes), always with the band FFT. The layouts are those
 * above; the accumulated NG term stays double and lpt_band_to_normal moves it in double. */
extern int NGFloat;

typedef struct
{
  float re, im;
}
lpt_complexf;

extern unsigned Lpt_plan_flags;                 /* FFTW planner flags of the run: FFTW_MEASURE with a wisdom file */
void  lpt_wisdom_snapshot(void);                /* before making plans after initialize_ffts ... */
void  lpt_wisdom_save_new(const char *what);    /* ... and after: new plans of all ranks go to the wisdom files */

void *lpt_alloc_grid(void);                     /* one grid of TotalSizePlusAdditional reals, FFT-aligned */
void *lpt_alloc_grid_f(void);                   /* the same number of floats */
void  lpt_fftf_forward_band(float *field);
void  lpt_fftf_inverse_band(float *field);
void  lpt_fft(fftw_plan plan, lpt_real *field);
void  lpt_fft_forward_band(lpt_real *field);
void  lpt_fft_inverse_band(lpt_real *field);
void  lpt_band_to_normal(lpt_complex *data);

#endif
