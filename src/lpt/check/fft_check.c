/* 2LPTFFTCheck: checks the band FFT backend (band_fft.c, FFTBackend band) against FFTW3-MPI (FFTBackend fftw) on
 * the ranks it is started on, for an N^3 grid whose band is that of Nsample (as the 2LPT executables set it up):
 *
 *   1. forward, plane waves:  band_fft_forward of a sum of plane waves (in and out of the band) against their
 *                             exact transform, every element of the band layout on every rank
 *   2. forward, random field: band_fft_forward + band_fft_to_slab against FFTW3-MPI's r2c, in the slab layout
 *                             (zero outside the band, exactly)
 *   3. inverse:               band_fft_inverse of a k-space grid with modes in and out of the band against
 *                             FFTW3-MPI's c2r of the same grid with the out-of-band modes removed
 *
 * Usage: mpirun -np P 2LPTFFTCheck N Nsample [measure]       (N even; measure: FFTW_MEASURE plans, else ESTIMATE)
 * Prints the errors relative to the largest value and exits with 1 if one exceeds 1e-12. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <fftw3-mpi.h>
#include "band_fft.h"

#define TOL 1e-12

static int N, Nsample, kz_end;

static int in_band_axis(ptrdiff_t n)               /* as lpt_axis_in_band */
{
  ptrdiff_t m = (n < N / 2) ? n : N - n;
  return m <= Nsample / 2 + 1;
}

static double hash_value(ptrdiff_t x, ptrdiff_t y, ptrdiff_t z)
{
  unsigned long long h = (unsigned long long) ((x * N + y) * N + z) * 0x9E3779B97F4A7C15ULL;

  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ULL;
  h ^= h >> 32;
  return (double) (h >> 11) / 9007199254740992.0 - 0.5;
}

/* plane waves A cos(2 pi k.x / N + phi), kz > 0: the r2c transform is N^3 A e^{i phi} / 2 at k, zero elsewhere */
#define NWAVE 6
static int    wave_k[NWAVE][3];
static double wave_a[NWAVE], wave_phi[NWAVE];

static void set_waves(void)
{
  int kc = Nsample / 2 + 1, m = 0;
  int ks[NWAVE][3] = {{1, 2, 3}, {-kc, kc, kc}, {0, -1, 1}, {kc + 1, 0, 1}, {2, -(kc + 2), 2}, {3, 1, kc + 1}};

  for(m = 0; m < NWAVE; m++)
    {
      for(int d = 0; d < 3; d++)
        {
          int k = ks[m][d];

          k = (d == 2) ? (k < 1 ? 1 : (k > N / 2 - 1 ? N / 2 - 1 : k)) : ((k % N) + N) % N;
          wave_k[m][d] = k;
        }
      wave_a[m]   = 1.0 + 0.3 * m;
      wave_phi[m] = 0.7 * m + 0.2;
    }
}

static void exact_transform(ptrdiff_t x, ptrdiff_t y, ptrdiff_t z, double *re, double *im)
{
  *re = *im = 0;
  for(int m = 0; m < NWAVE; m++)
    if(wave_k[m][0] == x && wave_k[m][1] == y && wave_k[m][2] == z)
      {
        *re += 0.5 * (double) N * N * N * wave_a[m] * cos(wave_phi[m]);
        *im += 0.5 * (double) N * N * N * wave_a[m] * sin(wave_phi[m]);
      }
}

static int report(const char *what, double err, double scale, int rank)
{
  double e = err / scale, emax;

  MPI_Allreduce(&e, &emax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if(rank == 0)
    printf("  %-28s max|err| / max = %.2e  %s\n", what, emax, emax <= TOL ? "ok" : "FAIL");
  return emax <= TOL;
}

int main(int argc, char **argv)
{
  int       rank, np, ok = 1;
  ptrdiff_t alloc, nx, x0, local;
  unsigned  flags;

  MPI_Init(&argc, &argv);
  fftw_mpi_init();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  if(argc < 3 || atoi(argv[1]) < 4 || atoi(argv[1]) % 2)
    {
      if(rank == 0)
        printf("usage: mpirun -np P 2LPTFFTCheck N Nsample [measure]   (N even)\n");
      MPI_Finalize();
      return 2;
    }
  N       = atoi(argv[1]);
  Nsample = atoi(argv[2]);
  kz_end  = (Nsample / 2 + 1 < N / 2) ? Nsample / 2 + 1 : N / 2;      /* as lpt_kz_end */
  flags   = (argc > 3 && strcmp(argv[3], "measure") == 0) ? FFTW_MEASURE : FFTW_ESTIMATE;

  const int nzc = N / 2 + 1;
  int      *rows = malloc(sizeof(int) * N);

  for(int y = 0; y < N; y++)
    rows[y] = in_band_axis(y);

  alloc = fftw_mpi_local_size_3d(N, N, nzc, MPI_COMM_WORLD, &nx, &x0);
  band_fft *b = band_fft_create(N, kz_end, rows, (int) nx, (int) x0, MPI_COMM_WORLD, flags);

  if(!b)
    {
      if(rank == 0)
        printf("band_fft_create failed\n");
      MPI_Finalize();
      return 1;
    }
  local = band_fft_local_size(b) > alloc ? band_fft_local_size(b) : alloc;

  double       *f = fftw_alloc_real(2 * local), *g = fftw_alloc_real(2 * local);
  fftw_complex *cf = (fftw_complex *) f, *cg = (fftw_complex *) g;
  fftw_plan     pf = fftw_mpi_plan_dft_r2c_3d(N, N, N, g, cg, MPI_COMM_WORLD, flags);
  fftw_plan     pb = fftw_mpi_plan_dft_c2r_3d(N, N, N, cg, g, MPI_COMM_WORLD, flags);
  const int     ny = band_fft_rows(b), nkz = band_fft_nz(b);
  const int    *ky = band_fft_row_y(b);
  double        err, scale, v;

  if(rank == 0)
    printf("2LPTFFTCheck: N = %d, Nsample = %d, %d ranks, %s plans; band: kz < %d, %d rows of ky on rank 0\n", N,
           Nsample, np, flags == FFTW_MEASURE ? "measured" : "estimated", nkz, ny);

  /* 1. forward, plane waves, against the exact transform */
  set_waves();
  for(ptrdiff_t x = 0; x < nx; x++)
    for(ptrdiff_t y = 0; y < N; y++)
      for(ptrdiff_t z = 0; z < N; z++)
        {
          v = 0;
          for(int m = 0; m < NWAVE; m++)
            {
              long long phase = ((long long) wave_k[m][0] * (x0 + x) + (long long) wave_k[m][1] * y +
                                 (long long) wave_k[m][2] * z) % N;

              v += wave_a[m] * cos(2 * M_PI * (double) phase / N + wave_phi[m]);
            }
          f[((x * N) + y) * 2 * nzc + z] = v;
        }
  band_fft_forward(b, f);
  err   = 0;
  scale = 0.5 * (double) N * N * N * wave_a[NWAVE - 1];
  for(ptrdiff_t x = 0; x < N; x++)
    for(int q = 0; q < ny; q++)
      for(int z = 0; z < nkz; z++)
        {
          double re, im;
          fftw_complex *a = cf + (x * ny + q) * nkz + z;

          exact_transform(x, ky[q], z, &re, &im);
          err = fmax(err, hypot((*a)[0] - re, (*a)[1] - im));
        }
  ok &= report("forward, plane waves", err, scale, rank);

  /* 2. forward, random field, against FFTW3-MPI (slab layout) */
  for(ptrdiff_t x = 0; x < nx; x++)
    for(ptrdiff_t y = 0; y < N; y++)
      for(ptrdiff_t z = 0; z < N; z++)
        f[((x * N) + y) * 2 * nzc + z] = g[((x * N) + y) * 2 * nzc + z] = hash_value(x0 + x, y, z);
  band_fft_forward(b, f);
  band_fft_to_slab(b, cf);
  fftw_execute(pf);
  err = scale = 0;
  for(ptrdiff_t i = 0; i < nx * N * nzc; i++)
    {
      int y = (int) ((i / nzc) % N), z = (int) (i % nzc);

      scale = fmax(scale, hypot(cg[i][0], cg[i][1]));
      if(rows[y] && z < nkz)
        err = fmax(err, hypot(cf[i][0] - cg[i][0], cf[i][1] - cg[i][1]));
      else if(cf[i][0] != 0 || cf[i][1] != 0)
        err = INFINITY;                                  /* outside the band: exactly zero */
    }
  MPI_Allreduce(MPI_IN_PLACE, &scale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  ok &= report("forward, random field", err, scale, rank);

  /* 3. inverse: band_fft_inverse of the full spectrum (cf) against FFTW's c2r of its band part (cg) */
  memcpy(cf, cg, sizeof(fftw_complex) * nx * N * nzc);
  for(ptrdiff_t i = 0; i < nx * N * nzc; i++)
    if(!rows[(i / nzc) % N] || i % nzc >= nkz)
      cg[i][0] = cg[i][1] = 0;
  band_fft_inverse(b, f);
  fftw_execute(pb);
  err = scale = 0;
  for(ptrdiff_t x = 0; x < nx; x++)
    for(ptrdiff_t y = 0; y < N; y++)
      for(ptrdiff_t z = 0; z < N; z++)
        {
          ptrdiff_t i = ((x * N) + y) * 2 * nzc + z;

          scale = fmax(scale, fabs(g[i]));
          err   = fmax(err, fabs(f[i] - g[i]));
        }
  MPI_Allreduce(MPI_IN_PLACE, &scale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  ok &= report("inverse", err, scale, rank);

  if(rank == 0)
    printf("%s\n", ok ? "PASS" : "FAIL");

  fftw_destroy_plan(pf);
  fftw_destroy_plan(pb);
  band_fft_destroy(b);
  fftw_free(f);
  fftw_free(g);
  free(rows);
  MPI_Finalize();
  return ok ? 0 : 1;
}
