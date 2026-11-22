/* Band-limited slab FFT: see band_fft.h for the layouts.
 *
 * forward:  z r2c of every row (serial, all of the slab), y c2c of the planes kz < nkz, all-to-all of the band
 *           (ky in band, kz < nkz) from x-slabs to band rows, x c2c of the band rows.
 * inverse:  the same backwards: all-to-all of the band of the slab grid to band rows, x c2c, all-to-all back (the
 *           slab is zeroed outside the band), y c2c of the planes kz < nkz, z c2r of every row.
 *
 * The all-to-all sends only the band: (2 (Nsample/2+1) + 1) / N of the rows and nkz / nzc of the planes, about 4/9
 * of the grid for N = 3/2 Nsample. It receives straight into the band layout ([x][q][kz]: the x range of each sender
 * is contiguous) and sends straight from it on the way back, so one buffer (on the slab side) is enough. */
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "band_fft.h"

int Bf_new_plans = 0;   /* see BF_PLAN_WISDOM_FIRST */

/* double: band_fft_*, FFTW's fftw_* */
#define BF_T      band_fft
#define BF(n)     band_fft_##n
#define BF_R      double
#define BF_C      fftw_complex
#define BF_PLAN   fftw_plan
#define BF_X(n)   fftw_##n
#define BF_MPI    MPI_DOUBLE
#include "band_fft_impl.h"
#undef BF_T
#undef BF
#undef BF_R
#undef BF_C
#undef BF_PLAN
#undef BF_X
#undef BF_MPI

#ifndef AARAMBAM_NO_FLOAT
/* float: band_fftf_*, FFTW's single-precision fftwf_* */
#define BF_T      band_fftf
#define BF(n)     band_fftf_##n
#define BF_R      float
#define BF_C      fftwf_complex
#define BF_PLAN   fftwf_plan
#define BF_X(n)   fftwf_##n
#define BF_MPI    MPI_FLOAT
#include "band_fft_impl.h"

/* band layout -> slab layout of a double grid with the layout of a float transform: the same data movement as
 * band_fftf_to_slab, in double precision (a temporary buffer of the double size) */
void band_fftf_to_slab_double(band_fftf *b, fftw_complex *c)
{
  const int     N = b->N, nzc = b->nzc, nkz = b->nkz;
  size_t        nbuf = (size_t) b->nx * b->nband * nkz;
  fftw_complex *buf = fftw_alloc_complex(nbuf > 0 ? nbuf : 1), *p = buf;
  MPI_Datatype  dcplx;

  MPI_Type_contiguous(2, MPI_DOUBLE, &dcplx);
  MPI_Type_commit(&dcplx);
  MPI_Alltoallv(c, b->r_cnt, b->r_dsp, dcplx, buf, b->s_cnt, b->s_dsp, dcplx, b->comm);
  MPI_Type_free(&dcplx);

  for(int x = 0; x < b->nx; x++)
    for(int y = 0; y < N; y++)
      {
        int z0 = b->in_band[y] ? nkz : 0;
        memset(c + ((size_t) x * N + y) * nzc + z0, 0, sizeof(fftw_complex) * (nzc - z0));
      }
  for(int r = 0; r < b->np; r++)
    for(int x = 0; x < b->nx; x++)
      for(int q = b->row0[r]; q < b->row0[r + 1]; q++, p += nkz)
        memcpy(c + ((size_t) x * N + b->band_y[q]) * nzc, p, sizeof(fftw_complex) * nkz);
  fftw_free(buf);
}
#endif
