/* Body of band_fft.c, written once for both precisions (included by band_fft.c with BF_T, BF(name), BF_R, BF_C,
 * BF_PLAN, BF_X(fftw name) and BF_MPI defined for double or for float). */

struct BF_T
{
  int          N, nzc, nkz, nx, x0, np, rank;
  int          nband;          /* band rows in total */
  int         *band_y;         /* their ky */
  int         *row0;           /* rank r holds band rows row0[r] .. row0[r+1]-1 */
  int         *in_band;        /* [N]: row ky in band */
  int          ny;             /* band rows here */
  int         *nx_of, *x0_of;  /* slabs of every rank */
  ptrdiff_t    local_size;     /* complex numbers per grid */
  BF_C *buf;           /* slab-side buffer: nx * nband * nkz */
  int         *s_cnt, *s_dsp, *r_cnt, *r_dsp;     /* slab -> band counts (complex numbers); band -> slab swaps them */
  MPI_Comm     comm;
  MPI_Datatype cplx;
  BF_PLAN    pz_f, py_f, px_f, px_b, py_b, pz_b;
};

int        BF(rows)(const BF_T *b) { return b->ny; }
const int *BF(row_y)(const BF_T *b) { return b->band_y + b->row0[b->rank]; }
int        BF(nz)(const BF_T *b) { return b->nkz; }
ptrdiff_t  BF(local_size)(const BF_T *b) { return b->local_size; }
size_t     BF(buffer_bytes)(const BF_T *b)
{
  return sizeof(BF_C) * (size_t) b->nx * b->nband * b->nkz;
}


/* ---- the two all-to-alls ---- */

/* slab layout (band part) -> band layout */
static void BF(slab_to_band)(BF_T *b, BF_C *c)
{
  const int    N = b->N, nzc = b->nzc, nkz = b->nkz;
  BF_C *p = b->buf;

  for(int r = 0; r < b->np; r++)
    for(int x = 0; x < b->nx; x++)
      for(int q = b->row0[r]; q < b->row0[r + 1]; q++, p += nkz)
        memcpy(p, c + ((size_t) x * N + b->band_y[q]) * nzc, sizeof(BF_C) * nkz);

  MPI_Alltoallv(b->buf, b->s_cnt, b->s_dsp, b->cplx, c, b->r_cnt, b->r_dsp, b->cplx, b->comm);
}

/* band layout -> slab layout, zero outside the band */
void BF(to_slab)(BF_T *b, BF_C *c)
{
  const int     N = b->N, nzc = b->nzc, nkz = b->nkz;
  BF_C *p = b->buf;

  MPI_Alltoallv(c, b->r_cnt, b->r_dsp, b->cplx, b->buf, b->s_cnt, b->s_dsp, b->cplx, b->comm);

  for(int x = 0; x < b->nx; x++)
    for(int y = 0; y < N; y++)
      {
        int z0 = b->in_band[y] ? nkz : 0;
        memset(c + ((size_t) x * N + y) * nzc + z0, 0, sizeof(BF_C) * (nzc - z0));
      }
  for(int r = 0; r < b->np; r++)
    for(int x = 0; x < b->nx; x++)
      for(int q = b->row0[r]; q < b->row0[r + 1]; q++, p += nkz)
        memcpy(c + ((size_t) x * N + b->band_y[q]) * nzc, p, sizeof(BF_C) * nkz);
}


/* ---- the transforms ---- */

void BF(forward)(BF_T *b, BF_R *f)
{
  BF_C *c = (BF_C *) f;

  if(b->nx > 0)
    {
      BF_X(execute_dft_r2c)(b->pz_f, f, c);
      BF_X(execute_dft)(b->py_f, c, c);
    }
  BF(slab_to_band)(b, c);
  if(b->ny > 0)
    BF_X(execute_dft)(b->px_f, c, c);
}

void BF(inverse)(BF_T *b, BF_R *f)
{
  BF_C *c = (BF_C *) f;

  BF(slab_to_band)(b, c);
  if(b->ny > 0)
    BF_X(execute_dft)(b->px_b, c, c);
  BF(to_slab)(b, c);
  if(b->nx > 0)
    {
      BF_X(execute_dft)(b->py_b, c, c);
      BF_X(execute_dft_c2r)(b->pz_b, c, f);
    }
}


/* ---- setup ---- */

BF_T *BF(create)(int N, int kz_end, const int *row_in_band, int nx, int x0, MPI_Comm comm, unsigned flags)
{
  BF_T *b = calloc(1, sizeof(*b));
  long long s_tot = 0, r_tot = 0;
  int       ok, all_ok;

  b->N    = N;
  b->nzc  = N / 2 + 1;
  b->nkz  = kz_end + 1;
  b->nx   = nx;
  b->x0   = x0;
  b->comm = comm;
  MPI_Comm_size(comm, &b->np);
  MPI_Comm_rank(comm, &b->rank);

  b->in_band = malloc(sizeof(int) * N);
  b->band_y  = malloc(sizeof(int) * N);
  for(int y = 0; y < N; y++)
    if((b->in_band[y] = row_in_band[y] != 0))
      b->band_y[b->nband++] = y;

  b->row0 = malloc(sizeof(int) * (b->np + 1));
  for(int r = 0; r <= b->np; r++)
    b->row0[r] = (int) ((long long) r * b->nband / b->np);
  b->ny = b->row0[b->rank + 1] - b->row0[b->rank];

  b->nx_of = malloc(sizeof(int) * b->np);
  b->x0_of = malloc(sizeof(int) * b->np);
  MPI_Allgather(&nx, 1, MPI_INT, b->nx_of, 1, MPI_INT, comm);
  MPI_Allgather(&x0, 1, MPI_INT, b->x0_of, 1, MPI_INT, comm);

  /* slab -> band: to rank r my x-slab of its rows; from rank r its x-slab of my rows, at its x offset */
  b->s_cnt = malloc(sizeof(int) * b->np);
  b->s_dsp = malloc(sizeof(int) * b->np);
  b->r_cnt = malloc(sizeof(int) * b->np);
  b->r_dsp = malloc(sizeof(int) * b->np);
  for(int r = 0; r < b->np; r++)
    {
      long long sc = (long long) nx * (b->row0[r + 1] - b->row0[r]) * b->nkz;
      long long rc = (long long) b->nx_of[r] * b->ny * b->nkz, rd = (long long) b->x0_of[r] * b->ny * b->nkz;

      b->s_cnt[r] = (int) sc;
      b->s_dsp[r] = (int) s_tot;
      b->r_cnt[r] = (int) rc;
      b->r_dsp[r] = (int) rd;
      s_tot += sc;
      r_tot  = (rd + rc > r_tot) ? rd + rc : r_tot;
    }
  ok = s_tot <= INT_MAX && r_tot <= INT_MAX;

  b->local_size = (ptrdiff_t) nx * N * b->nzc;
  if((ptrdiff_t) N * b->ny * b->nkz > b->local_size)
    b->local_size = (ptrdiff_t) N * b->ny * b->nkz;

  b->buf = BF_X(alloc_complex)(s_tot > 0 ? (size_t) s_tot : 1);
  MPI_Type_contiguous(2, BF_MPI, &b->cplx);
  MPI_Type_commit(&b->cplx);

  /* serial plans, made on a scratch grid (FFTW_MEASURE overwrites it) */
  if(ok)
    {
      BF_C *c = BF_X(alloc_complex)(b->local_size > 0 ? (size_t) b->local_size : 1);
      BF_R         *f = (BF_R *) c;
      int           n[1] = {N};
      const int     N_nzc = N * b->nzc, row = b->ny * b->nkz;

      if(nx > 0)
        {
          BF_X(iodim) ty    = {N, b->nzc, b->nzc};
          BF_X(iodim) hy[2] = {{nx, N_nzc, N_nzc}, {b->nkz, 1, 1}};

          b->pz_f = BF_X(plan_many_dft_r2c)(1, n, nx * N, f, NULL, 1, 2 * b->nzc, c, NULL, 1, b->nzc, flags);
          b->pz_b = BF_X(plan_many_dft_c2r)(1, n, nx * N, c, NULL, 1, b->nzc, f, NULL, 1, 2 * b->nzc, flags);
          b->py_f = BF_X(plan_guru_dft)(1, &ty, 2, hy, c, c, FFTW_FORWARD, flags);
          b->py_b = BF_X(plan_guru_dft)(1, &ty, 2, hy, c, c, FFTW_BACKWARD, flags);
          ok      = ok && b->pz_f && b->pz_b && b->py_f && b->py_b;
        }
      if(b->ny > 0)
        {
          BF_X(iodim) tx = {N, row, row};
          BF_X(iodim) hx = {row, 1, 1};

          b->px_f = BF_X(plan_guru_dft)(1, &tx, 1, &hx, c, c, FFTW_FORWARD, flags);
          b->px_b = BF_X(plan_guru_dft)(1, &tx, 1, &hx, c, c, FFTW_BACKWARD, flags);
          ok      = ok && b->px_f && b->px_b;
        }
      BF_X(free)(c);
    }

  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
  if(!all_ok)
    {
      BF(destroy)(b);
      return NULL;
    }
  return b;
}

void BF(destroy)(BF_T *b)
{
  if(!b)
    return;

  BF_PLAN plans[6] = {b->pz_f, b->py_f, b->px_f, b->px_b, b->py_b, b->pz_b};

  for(int i = 0; i < 6; i++)
    if(plans[i])
      BF_X(destroy_plan)(plans[i]);
  MPI_Type_free(&b->cplx);
  BF_X(free)(b->buf);
  free(b->in_band);
  free(b->band_y);
  free(b->row0);
  free(b->nx_of);
  free(b->x0_of);
  free(b->s_cnt);
  free(b->s_dsp);
  free(b->r_cnt);
  free(b->r_dsp);
  free(b);
}
