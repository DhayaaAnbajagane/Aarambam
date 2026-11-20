/* Band-limited slab FFT (band_fft.c): the 3D real FFTs of a grid of which only a band of k-space matters, built from
 * serial FFTW3 transforms and MPI_Alltoallv. Self-contained: it needs MPI and FFTW3, not the rest of the 2LPT code.
 *
 * The grid is N^3, distributed over the ranks of comm in x-slabs (rank r holds x0_r .. x0_r + nx_r - 1, the ranks in
 * order of x): real [nx][N][2 nzc], complex [nx][N][nzc] with nzc = N/2 + 1 (the "slab" layout, as FFTW3-MPI's).
 * The band is the rows ky with row_in_band[ky] != 0 and the planes kz < nkz = kz_end + 1, all kx. In the "band"
 * layout each rank holds an even share of the band rows, ny rows with ky = band_fft_row_y(b)[q], as [N (x)][ny][nkz]:
 *
 *     coord = (x * ny + q) * nkz + kz
 *
 * Compared with the full transform the y and x transforms and the all-to-all only touch the band, and the band rows
 * are spread evenly over the ranks (so is the k-space work that follows).
 *
 *   band_fft_forward(b, f)    real slab grid -> its FFT on the band, in the band layout (r2c, FFTW sign -1)
 *   band_fft_inverse(b, f)    complex slab grid -> real slab grid (c2r); entries outside the band are taken as zero
 *   band_fft_to_slab(b, g)    band layout -> complex slab layout, zero outside the band
 *
 * All in place, unnormalised (as FFTW) and collective. A grid needs room for band_fft_local_size(b) complex numbers
 * (both layouts) and SIMD alignment (fftw_malloc, or posix_memalign to 64 bytes). */
#ifndef AARAMBAM_BAND_FFT_H
#define AARAMBAM_BAND_FFT_H

#include <stddef.h>
#include <mpi.h>
#include <fftw3.h>

typedef struct band_fft band_fft;

/* nx, x0: this rank's slab. flags: FFTW planner flags of the serial transforms (FFTW_ESTIMATE, FFTW_MEASURE ...).
 * Returns NULL (on every rank) if the all-to-all counts would overflow an int or a plan cannot be made. */
band_fft  *band_fft_create(int N, int kz_end, const int *row_in_band, int nx, int x0, MPI_Comm comm, unsigned flags);
void       band_fft_destroy(band_fft *b);

void       band_fft_forward(band_fft *b, double *f);
void       band_fft_inverse(band_fft *b, double *f);
void       band_fft_to_slab(band_fft *b, fftw_complex *g);

ptrdiff_t  band_fft_local_size(const band_fft *b);   /* complex numbers a grid needs on this rank */
int        band_fft_rows(const band_fft *b);         /* band rows on this rank (ny) */
const int *band_fft_row_y(const band_fft *b);        /* their ky, increasing */
int        band_fft_nz(const band_fft *b);           /* nkz */
size_t     band_fft_buffer_bytes(const band_fft *b); /* the communication buffer on this rank */

#endif
