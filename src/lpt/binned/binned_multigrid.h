/* Multi-grid binned NG stage (BinnedPNGMultiGrid 1), included by binned/main.c before binned_ng_impl.h.
 *
 * The filtered field f_a has modes only up to K_a per axis (the largest |n_x|, |n_y|, n_z of an allowed mode where its
 * input filter is non-zero; measured, so any filter type works), so f_a f_b has modes only up to K_a + K_b. On an L^3
 * grid the output modes |n| <= K_out = min(K_out_max, K_a + K_b) of FFT[f_a f_b] are then free of aliases if
 * L > K_a + K_b + K_out: a smaller grid gives the same products at the modes that are kept, and the pair adds exactly
 * zero above K_out. Each pair gets the smallest such L on a ladder of FFT-friendly sizes (ratio MG_LADDER_RATIO) and
 * one of three paths:
 *
 *   small  (L <= MG_SMALL_MAX): whole pairs per rank, serial pruned FFTW transforms on the L^3 grid;
 *   medium (MG_SMALL_MAX < L < Nmesh): all ranks together, a band_fft on the L^3 grid per ladder size;
 *   full   (L >= Nmesh, or a medium L with fewer x planes than ranks or whose fields need more memory than the
 *          full path's cache blocks): the distributed Nmesh^3 path of
 *          binned_ng_impl.h, which skips the pairs done here (bpng_mg_pair_done) and the fields only they use
 *          (bpng_mg_field_needed).
 *
 * The small and medium results are accumulated on the band |n_x|, |n_y|, n_z <= Nsample/2 (on every rank), summed
 * over the ranks and added into the band layout of the NG term. The FFTs use the run's planner flags (FFTW_MEASURE
 * with a wisdom file); plans measured here are added to the wisdom files. */

#include <limits.h>
#include <unistd.h>
#include "band_fft.h"

#define MG_SMALL_MAX     128      /* largest grid of the small (per-rank) path: at most 17 MB per real double field */
#define MG_LADDER_RATIO  1.2      /* successive grid sizes of the ladder */
#ifndef MG_MEM_FRACTION            /* the tables of every rank on a node may take at most this fraction of its memory */
#define MG_MEM_FRACTION  0.25
#endif

static int            MgActive    = 0;
static int            MgN         = 0;          /* bins of the input legs */
static unsigned char *MgPairDone  = NULL;       /* [a * MgN + b], a <= b: pair done on a smaller grid */
static unsigned char *MgFieldFull = NULL;       /* [a]: field needed by some full-grid pair */

static int bpng_mg_pair_done(int a, int b)
{
  if(!MgActive)
    return 0;
  return a <= b ? MgPairDone[a * MgN + b] : MgPairDone[b * MgN + a];
}

static int bpng_mg_field_needed(int a)
{
  return MgActive ? MgFieldFull[a] : 1;
}

static void bpng_mg_free(void)
{
  free(MgPairDone);
  free(MgFieldFull);
  MgPairDone = MgFieldFull = NULL;
  MgActive   = 0;
}

static int mg_good_size(int n)       /* 2^a 3^b 5^c 7^d */
{
  int p[4] = {2, 3, 5, 7};

  for(int q = 0; q < 4; q++)
    while(n % p[q] == 0)
      n /= p[q];
  return n == 1;
}

static int mg_next_good(int n)       /* smallest even FFT-friendly size >= n */
{
  if(n % 2)
    n++;
  while(!mg_good_size(n))
    n += 2;
  return n;
}

static int mg_small_max(void)        /* MG_SMALL_MAX; the environment variable AARAMBAM_MG_SMALL_MAX overrides it (tests) */
{
  const char *s = getenv("AARAMBAM_MG_SMALL_MAX");
  return (s && *s) ? atoi(s) : MG_SMALL_MAX;
}

static double mg_cost(int L)
{
  return (double) L * L * L * log2((double) L);
}

typedef struct
{
  int a, b, level, rank;     /* level: grid size L; rank: owner of a small pair, -1 for a medium pair */
}
MgPair;

static int mg_pair_order(const void *x, const void *y)    /* largest grid first, then (a, b) */
{
  const MgPair *p = x, *q = y;

  if(p->level != q->level)
    return q->level - p->level;
  if(p->a != q->a)
    return p->a - q->a;
  return p->b - q->b;
}

/* Tables of the small and medium paths. The band |n_x|, |n_y|, n_z <= KB = Nsample/2 has index
 * ((n_x + KB) D + n_y + KB) NZ + n_z, D = 2 KB + 1, NZ = KB + 1. */
typedef struct
{
  int          n, KB, D, NZ, S, nm, kout_max;
  long long    nB;
  long long   *shell;        /* allowed modes sorted by max(|n_x|, |n_y|, n_z): shell r is [shell[r], shell[r+1]) */
  int         *m_idx, *m_s;  /* band index and s = |n|^2 of each */
  short       *m_nx, *m_ny, *m_nz;
  int         *band_s;       /* per band index: |n|^2 if the mode is allowed, else -1 */
  double      *filt;         /* [s * n + a]: input filter of node a at |n|^2 = s */
  int         *s_ic;         /* per s: output bin (-1 outside) */
  double      *s_u;          /* and the position in it */
  int         *Ka;           /* per node: largest per-axis mode index where its filter is non-zero */
  int         *s_amax;       /* per s: largest max(|n_x|, |n_y|, n_z) of an allowed mode with |n|^2 = s (-1: none) */
  char        *s_live;       /* per s: the Gaussian potential is non-zero at some mode with |n|^2 = s */
  lpt_complex *G;            /* Gaussian potential on the band */
  lpt_complex *A;            /* NG term of the small and medium pairs on the band */
  double      *coef;         /* per s: Fnl * scale * output coefficient of the current pair (0: skipped) */
}
MgCtx;

static int mg_kout(const MgCtx *X, int a, int b)
{
  return X->Ka[a] + X->Ka[b] < X->kout_max ? X->Ka[a] + X->Ka[b] : X->kout_max;
}

/* X->coef[s] for s up to 3 kout^2: the arithmetic of bpng_accumulate_hybrid_convolution per mode (the cells (a, b) and
 * (b, a)), times Fnl and scale. */
static void mg_pair_coefficients(MgCtx *X, int a, int b, int kout, double scale)
{
  const int nb3 = BPNG.nk[2], pair = (a != b), smax = 3 * kout * kout;
  double    ca[nb3][4], cb[nb3][4];

  for(int ic = 0; ic < nb3; ic++)
    {
      for(int q = 0; q < 4; q++)
        ca[ic][q] = cb[ic][q] = 0;
      bpng_cell_coefficients(a, b, ic, ca[ic]);
      if(pair)
        bpng_cell_coefficients(b, a, ic, cb[ic]);
    }
  for(int s = 0; s <= smax && s < X->S; s++)
    {
      const int ic = X->s_ic[s];
      double    c;

      X->coef[s] = 0;
      if(ic < 0)
        continue;
      c = bpng_cell_polynomial(ca[ic], X->s_u[s]);
      if(pair)
        c += bpng_cell_polynomial(cb[ic], X->s_u[s]);
      if(c == 0.0 || !isfinite(c))
        continue;
      X->coef[s] = Fnl * c * scale;
    }
}


/* ---- small and medium paths, for double and float: binned_multigrid_impl.h ---- */

#define MGR          double
#define MGC          fftw_complex
#define MG(n)        n##_d
#define MGPLAN       fftw_plan
#define MG_IODIM     fftw_iodim64
#define MG_GURU_DFT  fftw_plan_guru64_dft
#define MG_GURU_C2R  fftw_plan_guru64_dft_c2r
#define MG_GURU_R2C  fftw_plan_guru64_dft_r2c
#define MG_EXEC      fftw_execute
#define MG_EXEC_C2R  fftw_execute_dft_c2r
#define MG_DESTROY   fftw_destroy_plan
#define MG_MALLOC    fftw_malloc
#define MG_FREE      fftw_free
#define BF           band_fft
#define BF_CREATE    band_fft_create
#define BF_DESTROY   band_fft_destroy
#define BF_FWD       band_fft_forward
#define BF_INV       band_fft_inverse
#define BF_LSZ       band_fft_local_size
#define BF_ROWS      band_fft_rows
#define BF_NZ        band_fft_nz
#define BF_ROWY      band_fft_row_y
#include "binned_multigrid_impl.h"
#undef MGR
#undef MGC
#undef MG
#undef MGPLAN
#undef MG_IODIM
#undef MG_GURU_DFT
#undef MG_GURU_C2R
#undef MG_GURU_R2C
#undef MG_EXEC
#undef MG_EXEC_C2R
#undef MG_DESTROY
#undef MG_MALLOC
#undef MG_FREE
#undef BF
#undef BF_CREATE
#undef BF_DESTROY
#undef BF_FWD
#undef BF_INV
#undef BF_LSZ
#undef BF_ROWS
#undef BF_NZ
#undef BF_ROWY

#ifndef AARAMBAM_NO_FLOAT
#define MGR          float
#define MGC          fftwf_complex
#define MG(n)        n##_f
#define MGPLAN       fftwf_plan
#define MG_IODIM     fftwf_iodim64
#define MG_GURU_DFT  fftwf_plan_guru64_dft
#define MG_GURU_C2R  fftwf_plan_guru64_dft_c2r
#define MG_GURU_R2C  fftwf_plan_guru64_dft_r2c
#define MG_EXEC      fftwf_execute
#define MG_EXEC_C2R  fftwf_execute_dft_c2r
#define MG_DESTROY   fftwf_destroy_plan
#define MG_MALLOC    fftwf_malloc
#define MG_FREE      fftwf_free
#define BF           band_fftf
#define BF_CREATE    band_fftf_create
#define BF_DESTROY   band_fftf_destroy
#define BF_FWD       band_fftf_forward
#define BF_INV       band_fftf_inverse
#define BF_LSZ       band_fftf_local_size
#define BF_ROWS      band_fftf_rows
#define BF_NZ        band_fftf_nz
#define BF_ROWY      band_fftf_row_y
#include "binned_multigrid_impl.h"
#undef MGR
#undef MGC
#undef MG
#undef MGPLAN
#undef MG_IODIM
#undef MG_GURU_DFT
#undef MG_GURU_C2R
#undef MG_GURU_R2C
#undef MG_EXEC
#undef MG_EXEC_C2R
#undef MG_DESTROY
#undef MG_MALLOC
#undef MG_FREE
#undef BF
#undef BF_CREATE
#undef BF_DESTROY
#undef BF_FWD
#undef BF_INV
#undef BF_LSZ
#undef BF_ROWS
#undef BF_NZ
#undef BF_ROWY
#endif


/* ---- tables, pair plan and the two passes ---- */

static void mg_build_tables(MgCtx *X, lpt_complex *cpot)
{
  const int n = BPNG.nk[0], KB = Nsample / 2, D = 2 * KB + 1, NZ = KB + 1, S = 3 * KB * KB + 1;

  X->n  = n;
  X->KB = KB;
  X->D  = D;
  X->NZ = NZ;
  X->S  = S;
  X->nB = (long long) D * D * NZ;

  /* allowed modes of the band (the test of the full-grid path), counted per shell max(|n_x|, |n_y|, n_z) */
  X->shell  = calloc(KB + 2, sizeof(long long));
  X->band_s = malloc(sizeof(int) * X->nB);
  if(!X->shell || !X->band_s)
    FatalError(9420);
  for(int nx = -KB; nx <= KB; nx++)
    for(int ny = -KB; ny <= KB; ny++)
      for(int nz = 0; nz <= KB; nz++)
        {
          const ptrdiff_t i = nx < 0 ? nx + Nmesh : nx, j = ny < 0 ? ny + Nmesh : ny;
          const double    kx = lpt_wavenumber(i), ky = lpt_wavenumber(j), kz = lpt_wavenumber(nz);
          const long long idx = ((long long) (nx + KB) * D + (ny + KB)) * NZ + nz;
          int             am = abs(nx) > abs(ny) ? abs(nx) : abs(ny);

          am = nz > am ? nz : am;
          if(bpng_mode_allowed(i, j, nz, kx, ky, kz, sqrt(kx * kx + ky * ky + kz * kz)))
            {
              X->band_s[idx] = nx * nx + ny * ny + nz * nz;
              X->shell[am + 1]++;
            }
          else
            X->band_s[idx] = -1;
        }
  for(int r = 1; r <= KB + 1; r++)
    X->shell[r] += X->shell[r - 1];

  X->nm    = (int) X->shell[KB + 1];
  X->m_idx = malloc(sizeof(int) * (X->nm > 0 ? X->nm : 1));
  X->m_s   = malloc(sizeof(int) * (X->nm > 0 ? X->nm : 1));
  X->m_nx  = malloc(sizeof(short) * (X->nm > 0 ? X->nm : 1));
  X->m_ny  = malloc(sizeof(short) * (X->nm > 0 ? X->nm : 1));
  X->m_nz  = malloc(sizeof(short) * (X->nm > 0 ? X->nm : 1));
  long long *fill = calloc(KB + 1, sizeof(long long));
  if(!X->m_idx || !X->m_s || !X->m_nx || !X->m_ny || !X->m_nz || !fill)
    FatalError(9421);
  for(int nx = -KB; nx <= KB; nx++)
    for(int ny = -KB; ny <= KB; ny++)
      for(int nz = 0; nz <= KB; nz++)
        {
          const long long idx = ((long long) (nx + KB) * D + (ny + KB)) * NZ + nz;
          if(X->band_s[idx] < 0)
            continue;
          int am = abs(nx) > abs(ny) ? abs(nx) : abs(ny);
          am = nz > am ? nz : am;
          const long long m = X->shell[am] + fill[am]++;
          X->m_idx[m] = (int) idx;
          X->m_s[m]   = X->band_s[idx];
          X->m_nx[m]  = nx;
          X->m_ny[m]  = ny;
          X->m_nz[m]  = nz;
        }
  free(fill);

  /* what depends on |k| only, per s = |n|^2 */
  X->filt = calloc((size_t) n * S, sizeof(double));
  X->s_ic = malloc(sizeof(int) * S);
  X->s_u  = malloc(sizeof(double) * S);
  X->coef = calloc(S, sizeof(double));
  if(!X->filt || !X->s_ic || !X->s_u || !X->coef)
    FatalError(9422);
  X->s_amax = malloc(sizeof(int) * S);
  X->s_live = calloc(S, 1);
  if(!X->s_amax || !X->s_live)
    FatalError(9426);
  for(int s = 0; s < S; s++)
    X->s_amax[s] = -1;
  for(int r = 0; r <= KB; r++)
    for(long long m = X->shell[r]; m < X->shell[r + 1]; m++)
      X->s_amax[X->m_s[m]] = r;                   /* shells in increasing r: the last one is the largest */
  X->s_ic[0] = -1;
  X->s_u[0]  = 0;
  for(int s = 1; s < S; s++)
    {
      const double kmag = sqrt((double) s) * 2 * PI / Box;
      for(int a = 0; a < n; a++)
        X->filt[(size_t) s * n + a] = bpng_input_basis_value(0, a, kmag);
      X->s_ic[s] = bpng_find_bin(BPNG.edges[2], BPNG.nk[2], kmag);
      X->s_u[s]  = (X->s_ic[s] < 0 || BPNG.coeff_count <= 1) ? 0.0 : bpng_u_in_bin_axis(2, X->s_ic[s], kmag);
    }

  /* the Gaussian potential on the band, on every rank */
  X->G = calloc(X->nB, sizeof(lpt_complex));
  if(!X->G)
    FatalError(9423);
  for(ptrdiff_t ii = 0; ii < Local_nx; ii++)
    {
      const ptrdiff_t i = ii + Local_x_start, nx = i < Nmesh / 2 ? i : i - Nmesh;
      if(nx < -KB || nx > KB)
        continue;
      for(ptrdiff_t j = 0; j < Nmesh; j++)
        {
          const ptrdiff_t ny = j < Nmesh / 2 ? j : j - Nmesh;
          if(ny < -KB || ny > KB)
            continue;
          for(int nz = 0; nz <= KB; nz++)
            X->G[((nx + KB) * D + (ny + KB)) * NZ + nz] = cpot[(ii * Nmesh + j) * (Nmesh / 2 + 1) + nz];
        }
    }
  MPI_Allreduce(MPI_IN_PLACE, X->G, 2 * X->nB, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for(long long m = 0; m < X->nm; m++)
    if(X->G[X->m_idx[m]].re != 0 || X->G[X->m_idx[m]].im != 0)
      X->s_live[X->m_s[m]] = 1;
}

static void mg_free_tables(MgCtx *X)
{
  free(X->shell);
  free(X->band_s);
  free(X->m_idx);
  free(X->m_s);
  free(X->m_nx);
  free(X->m_ny);
  free(X->m_nz);
  free(X->filt);
  free(X->s_ic);
  free(X->s_u);
  free(X->s_amax);
  free(X->s_live);
  free(X->coef);
  free(X->Ka);
  free(X->G);
  free(X->A);
}

/* K_a per node, which nodes have a non-zero field, K_out_max; then the grid size and path of every pair (identical on
 * every rank) and the owners of the small pairs (greedy: a pair costs one r2c on its grid plus a c2r for each of its
 * fields the rank does not have on that grid yet). Returns the number of small and medium pairs, in P. */
static int mg_plan_pairs(MgCtx *X, MgPair *P, int *nsmall, int *nmed, int *nfull)
{
  const int n = X->n, small_max = mg_small_max();
  int      *live = calloc(n, sizeof(int)), levels[64], nlev = 0, np = 0;

  X->Ka       = calloc(n, sizeof(int));
  X->kout_max = 0;
  for(int s = 0; s < X->S; s++)                 /* everything here depends on |n|^2 only */
    {
      const double *w = X->filt + (size_t) s * n;
      if(X->s_amax[s] < 0)
        continue;
      if(X->s_ic[s] >= 0 && X->s_amax[s] > X->kout_max)
        X->kout_max = X->s_amax[s];
      for(int a = 0; a < n; a++)
        if(w[a] != 0)
          {
            X->Ka[a] = X->s_amax[s] > X->Ka[a] ? X->s_amax[s] : X->Ka[a];
            live[a] |= X->s_live[s];
          }
    }

  levels[nlev++] = mg_next_good(8);
  while(levels[nlev - 1] < Nmesh && nlev < 63)
    {
      const int prev = levels[nlev - 1], next = mg_next_good((int) ceil(prev * MG_LADDER_RATIO));
      levels[nlev++] = next > prev ? next : mg_next_good(prev + 2);
    }

  *nsmall = *nmed = *nfull = 0;
  for(int a = 0; a < n; a++)
    for(int b = a; b < n; b++)
      {
        if(!live[a] || !live[b])             /* a zero field: the pair adds exactly zero */
          continue;
        const int big = X->Ka[a] > X->Ka[b] ? X->Ka[a] : X->Ka[b];
        int       need = X->Ka[a] + X->Ka[b] + mg_kout(X, a, b) + 1, L = 0;
        need = need < 2 * big + 2 ? 2 * big + 2 : need;       /* the fields themselves fit */
        for(int l = 0; l < nlev; l++)
          if(levels[l] >= need)
            {
              L = levels[l];
              break;
            }
        /* a medium grid needs at least one x plane per rank */
        if(L == 0 || L >= Nmesh || (L > small_max && L < NTask))
          {
            MgFieldFull[a] = MgFieldFull[b] = 1;
            (*nfull)++;
            continue;
          }
        MgPairDone[a * n + b] = 1;
        P[np++] = (MgPair) {a, b, L, L <= small_max ? 0 : -1};
        if(L <= small_max)
          (*nsmall)++;
        else
          (*nmed)++;
      }
  qsort(P, np, sizeof(MgPair), mg_pair_order);

  /* A medium level holds all its fields at once: if they take more than the full path's two cache blocks
   * (BinnedPNGCacheBlockSize fields of Nmesh^3), its pairs go to the full grid, so the multi-grid stage never needs
   * more memory than the block size the run chose. */
  {
    const int    block  = BinnedPNGCacheBlockSize < 1 ? 1 : (BinnedPNGCacheBlockSize > n ? n : BinnedPNGCacheBlockSize);
    const double budget = 2.0 * block * Nmesh * (double) Nmesh * (Nmesh / 2 + 1);
    unsigned char *uses = malloc(n);
    int            q    = 0;

    if(!uses)
      FatalError(9427);
    for(int p0 = 0, p1; p0 < np; p0 = p1)
      {
        const int L = P[p0].level;
        int       nf = 0;

        for(p1 = p0; p1 < np && P[p1].level == L; p1++)
          ;
        if(P[p0].rank >= 0)             /* a small level: kept */
          {
            for(int p = p0; p < p1; p++)
              P[q++] = P[p];
            continue;
          }
        memset(uses, 0, n);
        for(int p = p0; p < p1; p++)
          uses[P[p].a] = uses[P[p].b] = 1;
        for(int a = 0; a < n; a++)
          nf += uses[a];
        if(nf * (double) L * L * (L / 2 + 1) <= budget)
          {
            for(int p = p0; p < p1; p++)
              P[q++] = P[p];
            continue;
          }
        for(int p = p0; p < p1; p++)    /* to the full grid */
          {
            MgPairDone[P[p].a * n + P[p].b] = 0;
            MgFieldFull[P[p].a] = MgFieldFull[P[p].b] = 1;
            (*nmed)--;
            (*nfull)++;
          }
      }
    np = q;
    free(uses);
  }

  double        *load = calloc(NTask, sizeof(double));
  unsigned char *have = calloc((size_t) NTask * nlev * n, 1);
  if(!load || !have)
    FatalError(9424);
  for(int p = 0; p < np; p++)
    {
      if(P[p].rank < 0)
        continue;
      int    l = 0, best = 0;
      double bc = 1e300;
      while(levels[l] != P[p].level)
        l++;
      for(int r = 0; r < NTask; r++)
        {
          const unsigned char *h = have + ((size_t) r * nlev + l) * n;
          const double c = load[r] + mg_cost(P[p].level) * (1 + !h[P[p].a] + (P[p].b != P[p].a && !h[P[p].b]));
          if(c < bc)
            {
              bc   = c;
              best = r;
            }
        }
      have[((size_t) best * nlev + l) * n + P[p].a] = have[((size_t) best * nlev + l) * n + P[p].b] = 1;
      load[best] = bc;
      P[p].rank  = best;
    }
  free(load);
  free(have);
  free(live);
  return np;
}

/* Add X->A (summed over the ranks) into this rank's part of the band layout of the NG term. */
static void mg_add_to_band(MgCtx *X, lpt_complex *out)
{
  const int KB = X->KB, D = X->D, NZ = X->NZ;

  MPI_Allreduce(MPI_IN_PLACE, X->A, 2 * X->nB, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for(int jj = 0; jj < Band_ny; jj++)
    {
      const int ny = Band_y[jj] < Nmesh / 2 ? Band_y[jj] : Band_y[jj] - Nmesh;
      if(ny < -KB || ny > KB)
        continue;
      for(ptrdiff_t i = 0; i < Nmesh; i++)
        {
          const ptrdiff_t nx = i < Nmesh / 2 ? i : i - Nmesh;
          if(nx < -KB || nx > KB)
            continue;
          for(int nz = 0; nz <= KB && nz < Band_nz; nz++)
            {
              const lpt_complex v = X->A[((nx + KB) * D + (ny + KB)) * NZ + nz];
              const ptrdiff_t   c = lpt_band_coord(jj, i, nz);
              out[c].re += v.re;
              out[c].im += v.im;
            }
        }
    }
}

/* Whether the tables of mg_build_tables fit. Every rank holds the whole band, about 50 bytes per band mode
 * (band_s, G, A and the mode lists): ~0.4 GB at Nsample 256, ~3.4 GB at 512 per rank. They must fit the int indexing
 * and counts, and the ranks of a node together may use at most MG_MEM_FRACTION of its memory. Collective. */
static int mg_tables_fit(double *gb_node)
{
  const long long KB = Nsample / 2, nB = (2 * KB + 1) * (2 * KB + 1) * (KB + 1);
  const double    bytes = 50.0 * (double) nB;
  long long       pages = sysconf(_SC_PHYS_PAGES), psize = sysconf(_SC_PAGESIZE);
  int             nlocal, fit, fit_all;
  MPI_Comm        node;

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  MPI_Comm_size(node, &nlocal);
  MPI_Comm_free(&node);
  *gb_node = nlocal * bytes / 1e9;
  fit = 2 * nB <= INT_MAX && (pages <= 0 || psize <= 0 || nlocal * bytes <= MG_MEM_FRACTION * (double) pages * psize);
  MPI_Allreduce(&fit, &fit_all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  return fit_all;
}

/* The small- and medium-grid pairs, added into out (the band layout of the NG term). Collective. Sets what the
 * full-grid path skips (bpng_mg_pair_done, bpng_mg_field_needed). Does nothing with BinnedPNGMultiGrid 0, or when its
 * tables would not fit (mg_tables_fit): then every pair runs on the full grid. */
static void bpng_multigrid_pairs(lpt_complex *cpot, lpt_complex *out)
{
  MgCtx X;
  int   timed_chars = 0, nsmall, nmed, nfull;

  MgActive = 0;
  if(!BinnedPNGMultiGrid)
    return;
  double gb_node;
  if(!mg_tables_fit(&gb_node))
    {
      if(ThisTask == 0)
        printf("  BinnedPNGMultiGrid: off for this run; its tables would take %.1f GB per node (every rank holds the "
               "sampled band). All bin pairs run on the full grid.\n", gb_node);
      return;
    }

  if(ThisTask == 0)
    {
      timed_chars = printf("  multi-grid pairs...");
      fflush(stdout);
    }

  memset(&X, 0, sizeof(X));
  mg_build_tables(&X, cpot);
  MgN         = X.n;
  MgPairDone  = calloc((size_t) X.n * X.n, 1);
  MgFieldFull = calloc(X.n, 1);
  MgPair *P   = malloc(sizeof(MgPair) * (X.n * (X.n + 1) / 2 + 1));
  X.A         = calloc(X.nB, sizeof(lpt_complex));
  if(!MgPairDone || !MgFieldFull || !P || !X.A)
    FatalError(9425);
  const int np = mg_plan_pairs(&X, P, &nsmall, &nmed, &nfull);

  lpt_wisdom_snapshot();

  for(int p0 = 0; p0 < np;)                 /* small grids: this rank's pairs, one grid size at a time */
    {
      int p1 = p0, mine = 0;
      while(p1 < np && P[p1].level == P[p0].level)
        mine += (P[p1++].rank == ThisTask);
      if(P[p0].rank >= 0 && mine)
        {
#ifndef AARAMBAM_NO_FLOAT
          if(NGFloat)
            mg_small_level_f(&X, P, p0, p1, Lpt_plan_flags);
          else
#endif
            mg_small_level_d(&X, P, p0, p1, Lpt_plan_flags);
        }
      p0 = p1;
    }
  lpt_barrier();
  for(int p0 = 0; p0 < np;)                 /* medium grids: all ranks, one grid size at a time */
    {
      int p1 = p0;
      while(p1 < np && P[p1].level == P[p0].level)
        p1++;
      if(P[p0].rank < 0)
        {
#ifndef AARAMBAM_NO_FLOAT
          if(NGFloat)
            mg_medium_level_f(&X, P, p0, p1, Lpt_plan_flags);
          else
#endif
            mg_medium_level_d(&X, P, p0, p1, Lpt_plan_flags);
        }
      p0 = p1;
    }

  lpt_wisdom_save_new("multi-grid");
  mg_add_to_band(&X, out);

  free(P);
  mg_free_tables(&X);
  MgActive = 1;
  if(ThisTask == 0)
    {
      print_timed_done_aligned(timed_chars);
      printf("  %d bin pairs on small grids (<= %d), %d on medium grids, %d on the full grid.\n", nsmall,
             mg_small_max(), nmed, nfull);
    }
}
