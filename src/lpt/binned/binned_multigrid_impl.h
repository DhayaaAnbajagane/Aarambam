/* The small-grid and medium-grid paths of the multi-grid NG stage (binned_multigrid.h), included twice: for
 * NGPrecision double (MG(n) = n_d, MGR = double, FFTW's fftw_* and band_fft) and for float (n_f, float, fftwf_*,
 * band_fftf). The fields, products and FFTs are in that precision; the accumulated NG term (MgCtx.A) is double. */

/* ---- small grids: pruned serial transforms, whole pairs per rank ---- */

/* Pruned 3D transforms on an L^3 grid ([x][y][z], z fastest; complex rows LZ = L/2 + 1, real rows L): along x and y
 * only the lines that carry non-zero input (c2r of a field with modes up to K) or needed output (r2c of a product
 * read up to K) are transformed; z (the r2c axis) is transformed in full. */
typedef struct
{
  MGPLAN x0, x1, y;          /* x lines for y = 0..K and y = L-K..L-1 (z <= K); y lines for all x (z <= K) */
  int    made;
}
MG(MgPass);

static void MG(mg_plan_pass)(MG(MgPass) *p, int L, int K, MGC *C, int sign, unsigned flags)
{
  const ptrdiff_t LZ = L / 2 + 1;
  MG_IODIM dx = {L, L * LZ, L * LZ}, dy = {L, LZ, LZ};
  MG_IODIM h0[2] = {{K + 1, LZ, LZ}, {K + 1, 1, 1}}, h1[2] = {{K, LZ, LZ}, {K + 1, 1, 1}};
  MG_IODIM hy[2] = {{L, L * LZ, L * LZ}, {K + 1, 1, 1}};

  p->x0   = MG_GURU_DFT(1, &dx, 2, h0, C, C, sign, flags);
  p->x1   = K > 0 ? MG_GURU_DFT(1, &dx, 2, h1, C + (L - K) * LZ, C + (L - K) * LZ, sign, flags) : NULL;
  p->y    = MG_GURU_DFT(1, &dy, 2, hy, C, C, sign, flags);
  p->made = 1;
  if(!p->x0 || !p->y || (K > 0 && !p->x1))
    FatalError(9401);
}

static void MG(mg_exec_xy)(MG(MgPass) *p, int x_first)
{
  if(!x_first)
    MG_EXEC(p->y);
  MG_EXEC(p->x0);
  if(p->x1)
    MG_EXEC(p->x1);
  if(x_first)
    MG_EXEC(p->y);
}

static void MG(mg_destroy_pass)(MG(MgPass) *p)
{
  if(!p->made)
    return;
  MG_DESTROY(p->x0);
  if(p->x1)
    MG_DESTROY(p->x1);
  MG_DESTROY(p->y);
  p->made = 0;
}

/* This rank's pairs among P[p0 .. p1) (all of one small grid size L): the fields they need are built and transformed
 * once on this rank, then each pair is multiplied, transformed and accumulated into X->A. */
static void MG(mg_small_level)(MgCtx *X, const MgPair *P, int p0, int p1, unsigned flags)
{
  const int       L = P[p0].level, n = X->n, KB = X->KB;
  const ptrdiff_t L3 = (ptrdiff_t) L * L * L, LZ = L / 2 + 1;
  const double    scale = pow((double) Nmesh / L, 3);      /* r2c on L^3 vs on Nmesh^3 */

  MGR        *R   = MG_MALLOC(sizeof(MGR) * L3);
  MGC        *C   = MG_MALLOC(sizeof(MGC) * L * L * LZ);
  MG_IODIM    dz  = {L, 1, 1}, hc = {(ptrdiff_t) L * L, LZ, L}, hr = {(ptrdiff_t) L * L, L, LZ};
  MGPLAN      pzi = MG_GURU_C2R(1, &dz, 1, &hc, C, R, flags);     /* z lines, all (x, y) */
  MGPLAN      pzf = MG_GURU_R2C(1, &dz, 1, &hr, R, C, flags);
  MG(MgPass) *bw  = calloc(KB + 1, sizeof(MG(MgPass))), *fw = calloc(KB + 1, sizeof(MG(MgPass)));
  MGR       **F   = calloc(n, sizeof(MGR *));
  int        *pos = malloc(sizeof(int) * (X->nm > 0 ? X->nm : 1));          /* band mode m on the L grid */

  if(!R || !C || !pzi || !pzf || !bw || !fw || !F || !pos)
    FatalError(9400);
  for(long long m = 0; m < X->nm; m++)
    pos[m] = ((X->m_nx[m] + L) % L * L + (X->m_ny[m] + L) % L) * LZ + X->m_nz[m];

  for(int p = p0; p < p1; p++)
    {
      if(P[p].rank != ThisTask)
        continue;
      const int a = P[p].a, b = P[p].b, ab[2] = {a, b}, kout = mg_kout(X, a, b);

      for(int q = 0; q < 2; q++)                    /* fields on this grid: embed the modes, c2r */
        {
          const int f = ab[q], K = X->Ka[f];
          if(F[f])
            continue;
          F[f] = MG_MALLOC(sizeof(MGR) * L3);
          if(!F[f])
            FatalError(9402);
          if(!bw[K].made)
            MG(mg_plan_pass)(&bw[K], L, K, C, FFTW_BACKWARD, flags);
          PROF_BEGIN(PROF_FIELD)
          memset(C, 0, sizeof(MGC) * L * L * LZ);
          for(long long m = 0; m < X->shell[K + 1]; m++)
            {
              double w = X->filt[(size_t) X->m_s[m] * n + f];
              if(w == 0.0)
                continue;
              C[pos[m]][0] = w * X->G[X->m_idx[m]].re;
              C[pos[m]][1] = w * X->G[X->m_idx[m]].im;
            }
          PROF_END(PROF_FIELD)
          PROF_BEGIN(PROF_FFT)
          MG(mg_exec_xy)(&bw[K], 1);                /* x over the y band, then y over the z band */
          MG_EXEC_C2R(pzi, C, F[f]);
          PROF_END(PROF_FFT)
        }

      if(!fw[kout].made)
        MG(mg_plan_pass)(&fw[kout], L, kout, C, FFTW_FORWARD, flags);
      PROF_BEGIN(PROF_MULT)
      const MGR *fa = F[a], *fb = F[b];
      for(ptrdiff_t x = 0; x < L3; x++)
        R[x] = fa[x] * fb[x];
      PROF_END(PROF_MULT)
      PROF_BEGIN(PROF_FFT)
      MG_EXEC(pzf);
      MG(mg_exec_xy)(&fw[kout], 0);                 /* y over the z band, then x over the y band */
      PROF_END(PROF_FFT)

      PROF_BEGIN(PROF_ACCUM)
      mg_pair_coefficients(X, a, b, kout, scale);
      for(long long m = 0; m < X->shell[kout + 1]; m++)
        {
          double c = X->coef[X->m_s[m]];
          if(c == 0.0)
            continue;
          X->A[X->m_idx[m]].re += c * C[pos[m]][0];
          X->A[X->m_idx[m]].im += c * C[pos[m]][1];
        }
      PROF_END(PROF_ACCUM)
    }

  for(int f = 0; f < n; f++)
    if(F[f])
      MG_FREE(F[f]);
  for(int k = 0; k <= KB; k++)
    {
      MG(mg_destroy_pass)(&bw[k]);
      MG(mg_destroy_pass)(&fw[k]);
    }
  free(F);
  free(pos);
  free(bw);
  free(fw);
  MG_DESTROY(pzf);
  MG_DESTROY(pzi);
  MG_FREE(R);
  MG_FREE(C);
}


/* ---- medium grids: a band FFT on the L^3 grid, all ranks ---- */

/* All pairs P[p0 .. p1) of one medium grid size L, collectively: x slabs of the L^3 grid on every rank, each field of
 * the level transformed once, then the pairs in turn. */
static void MG(mg_medium_level)(MgCtx *X, const MgPair *P, int p0, int p1, unsigned flags)
{
  const int    L = P[p0].level, n = X->n, KB = X->KB, D = X->D, NZ = X->NZ, nzc = L / 2 + 1;
  const double scale = pow((double) Nmesh / L, 3);
  int          Kl = 0, *need = calloc(n, sizeof(int)), *row_in = calloc(L, sizeof(int));

  for(int p = p0; p < p1; p++)              /* band of this level: the largest field support and output read */
    {
      const int ko = mg_kout(X, P[p].a, P[p].b);
      need[P[p].a] = need[P[p].b] = 1;
      Kl = X->Ka[P[p].a] > Kl ? X->Ka[P[p].a] : Kl;
      Kl = X->Ka[P[p].b] > Kl ? X->Ka[P[p].b] : Kl;
      Kl = ko > Kl ? ko : Kl;
    }
  for(int y = 0; y < L; y++)
    row_in[y] = abs(y < L / 2 ? y : y - L) <= Kl;

  const int nx = L / NTask + (ThisTask < L % NTask);
  const int x0 = ThisTask * (L / NTask) + (ThisTask < L % NTask ? ThisTask : L % NTask);
  BF       *bf = BF_CREATE(L, Kl, row_in, nx, x0, MPI_COMM_WORLD, flags);
  if(!bf)
    FatalError(9410);
  const ptrdiff_t lsz = BF_LSZ(bf);
  const int       ny = BF_ROWS(bf), nkz = BF_NZ(bf), *rowy = BF_ROWY(bf);
  MGR           **F  = calloc(n, sizeof(MGR *));
  MGR            *Pr = MG_MALLOC(sizeof(MGC) * (lsz > 0 ? lsz : 1));
  if(!F || !Pr)
    FatalError(9411);

  for(int f = 0; f < n; f++)                 /* the level's fields: this rank's x planes from the band, c2r */
    {
      if(!need[f])
        continue;
      const int K = X->Ka[f];
      F[f]        = MG_MALLOC(sizeof(MGC) * (lsz > 0 ? lsz : 1));
      if(!F[f])
        FatalError(9412);
      PROF_BEGIN(PROF_FIELD)
      MGC *g = (MGC *) F[f];
      memset(g, 0, sizeof(MGC) * lsz);
      for(int ii = 0; ii < nx; ii++)
        {
          const int sx = x0 + ii < L / 2 ? x0 + ii : x0 + ii - L;
          if(abs(sx) > K)
            continue;
          for(int y = 0; y < L; y++)
            {
              const int sy = y < L / 2 ? y : y - L;
              if(abs(sy) > K)
                continue;
              for(int z = 0; z <= K; z++)
                {
                  const long long idx = ((long long) (sx + KB) * D + (sy + KB)) * NZ + z;
                  const int       s   = X->band_s[idx];
                  if(s < 0)
                    continue;
                  const double    w = X->filt[(size_t) s * n + f];
                  const ptrdiff_t c = ((ptrdiff_t) ii * L + y) * nzc + z;
                  g[c][0] = w * X->G[idx].re;
                  g[c][1] = w * X->G[idx].im;
                }
            }
        }
      PROF_END(PROF_FIELD)
      PROF_BEGIN(PROF_FFT)
      BF_INV(bf, F[f]);
      PROF_END(PROF_FFT)
    }

  for(int p = p0; p < p1; p++)
    {
      const int a = P[p].a, b = P[p].b, kout = mg_kout(X, a, b);

      PROF_BEGIN(PROF_MULT)
      for(int ii = 0; ii < nx; ii++)
        for(int y = 0; y < L; y++)
          {
            const ptrdiff_t r = ((ptrdiff_t) ii * L + y) * 2 * nzc;
            for(int z = 0; z < L; z++)
              Pr[r + z] = F[a][r + z] * F[b][r + z];
            for(int z = L; z < 2 * nzc; z++)        /* row padding: the only other input of the r2c FFT */
              Pr[r + z] = 0;
          }
      PROF_END(PROF_MULT)
      PROF_BEGIN(PROF_FFT)
      BF_FWD(bf, Pr);
      PROF_END(PROF_FFT)

      PROF_BEGIN(PROF_ACCUM)
      mg_pair_coefficients(X, a, b, kout, scale);
      const MGC *c = (const MGC *) Pr;
      for(int x = 0; x < L; x++)              /* band layout of band_fft: (x * ny + q) * nkz + kz */
        {
          const int sx = x < L / 2 ? x : x - L;
          if(abs(sx) > kout)
            continue;
          for(int q = 0; q < ny; q++)
            {
              const int sy = rowy[q] < L / 2 ? rowy[q] : rowy[q] - L;
              if(abs(sy) > kout)
                continue;
              for(int z = 0; z <= kout && z < nkz; z++)
                {
                  const long long idx = ((long long) (sx + KB) * D + (sy + KB)) * NZ + z;
                  const int       s   = X->band_s[idx];
                  if(s < 0 || X->coef[s] == 0.0)
                    continue;
                  const ptrdiff_t k = ((ptrdiff_t) x * ny + q) * nkz + z;
                  X->A[idx].re += X->coef[s] * c[k][0];
                  X->A[idx].im += X->coef[s] * c[k][1];
                }
            }
        }
      PROF_END(PROF_ACCUM)
    }

  for(int f = 0; f < n; f++)
    if(F[f])
      MG_FREE(F[f]);
  free(F);
  free(need);
  free(row_in);
  MG_FREE(Pr);
  BF_DESTROY(bf);
}
