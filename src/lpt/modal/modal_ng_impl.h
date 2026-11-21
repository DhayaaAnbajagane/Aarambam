/* The grid-touching part of the modal non-Gaussian stage (the mode fields g_N in real space, their products and FFTs,
 * and the accumulation of the NG term), included twice by main_modal.c: for NGPrecision double (NG(name) = name_d,
 * NGR = lpt_real, NGC = lpt_complex) and for float (name_f, float, lpt_complexf). The Gaussian potential, the Legendre
 * recursion buffers and the accumulated NG term are double in both; NG_ALLOC, NG_FWD and NG_INV are the grid
 * allocation and the band FFTs of the precision. */

/* Mode field N in Fourier space (normal x-slab layout), written to the in-band modes of out:
 *   N <= 3:  out = k^{(4 - n_s)/3 (N - 3)} k^{4 - n_s} phi;
 *   N >= 4:  r[2] = P_{N-1}(x) k^{2 - n_s} phi by the Legendre recursion from r[0] = P_{N-3}, r[1] = P_{N-2}
 *            (set from gsl when restart), and out = r[2] - X0_{N-1} k^{2 - n_s} phi.
 * After an N >= 4 step the caller rotates r (modal_rotate) so that r[0], r[1] hold P_{N-2}, P_{N-1}. The recursion
 * buffers r are double; out has the precision of the NG stage (NGC). */
static void NG(modal_build_field)(unsigned int N, int restart, lpt_complex *cpot, lpt_complex *r[3], NGC *out,
                                  double (*X0Table)[2], double kmin_ns, double kmax_ns)
{
  PROF_BEGIN(PROF_FIELD)
  ptrdiff_t   ii, i, j, k, kend = lpt_kz_end();
  uint64_t    coord;
  double      kmag2, kmag1, kmag, Pl, pk;
  long double fnl_fac;
  double      np = ((double) N) - 1;     /* degree of the Legendre recursion step */

  for(ii = 0; ii < Local_nx; ii++)
    {
      i = ii + Local_x_start;
      if(!lpt_axis_in_band(i))
        continue;

      for(j = 0; j < Nmesh; j++)
        {
          if(!lpt_axis_in_band(j))
            continue;

          for(k = 0; k <= kend; k++)
            {
              coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;
              if(!modal_mode_in_band(i, j, k, &kmag2, &kmag1))
                continue;

              if(N <= 3)
                {
                  /* includes the 1/P(k) factor */
                  fnl_fac = pow(kmag1, (4 - PrimordialIndex)/3 * (N - 3.)) * pow(kmag1, 4 - PrimordialIndex);

                  out[coord].re = fnl_fac * cpot[coord].re;
                  out[coord].im = fnl_fac * cpot[coord].im;
                  continue;
                }

              kmag = modal_legendre_x(kmag1, kmin_ns, kmax_ns);

              /* 2 - n_s rather than 4 - n_s: the 1/k^2 of S(k1, k2, k3) -> B(k1, k2, k3) is folded in */
              pk = pow(kmag1, (2 - PrimordialIndex));

              if(restart)
                {
                  Pl = gsl_sf_legendre_Pl(N - 1 - 2, kmag) * pk;
                  r[0][coord].re = cpot[coord].re * Pl;
                  r[0][coord].im = cpot[coord].im * Pl;

                  Pl = gsl_sf_legendre_Pl(N - 1 - 1, kmag) * pk;
                  r[1][coord].re = cpot[coord].re * Pl;
                  r[1][coord].im = cpot[coord].im * Pl;
                }

              r[2][coord].re = ( (2*np - 1) * kmag * r[1][coord].re - (np - 1) * r[0][coord].re ) / np ;
              r[2][coord].im = ( (2*np - 1) * kmag * r[1][coord].im - (np - 1) * r[0][coord].im ) / np ;

              Pl = X0Table[N - 1][0] * pk;
              out[coord].re = r[2][coord].re - cpot[coord].re * Pl;
              out[coord].im = r[2][coord].im - cpot[coord].im * Pl;
            }
        }
    }
  PROF_END(PROF_FIELD)
}


/* p = a b on the real-space grid, row padding zeroed (the only other input of the r2c FFT: no full-grid clear needed) */
static void NG(modal_multiply_fields)(NGR *p, const NGR *a, const NGR *b)
{
  PROF_BEGIN(PROF_MULT)
  for(ptrdiff_t i = 0; i < Local_nx; i++)
    for(ptrdiff_t j = 0; j < Nmesh; j++)
      {
        const uint64_t row = (i * Nmesh + j) * (2 * (Nmesh / 2 + 1));

        for(ptrdiff_t k = 0; k < Nmesh; k++)
          p[row + k] = a[row + k] * b[row + k];
        for(ptrdiff_t k = Nmesh; k < 2 * (Nmesh / 2 + 1); k++)
          p[row + k] = 0;
      }
  PROF_END(PROF_MULT)
}

/* rank 0: "Done" for the pair line "Computing combination i, j ....", aligned whatever the width of i and j */
static void NG(modal_pair_done)(unsigned int N_i, unsigned int N_j)
{
  if(ThisTask == 0)
    print_timed_done(23 - (N_i >= 10) - (N_j >= 10));
}

/* The NG stage: cpot_NG (double, band layout, zeroed) += sum over the mode pairs of K_ij FFT[g_i g_j], and the Beta
 * terms. */
static void NG(modal_ng_stage)(lpt_complex *cpot, lpt_complex *cpot_NG, double (*AlphaTable)[N_modes][N_modes],
                               double (*X0Table)[2], double t_012, double kmin_ns, double kmax_ns)
{
  ptrdiff_t     i, j, k, ii;     /* 64-bit grid indices, as in the shared core */
  int           ksum;
  NGC          *c_tmp;           /* product of two mode fields (Fourier space after the forward FFT) */
  NGR          *p_tmp;           /* the same array in real space */
  NGC          *cg_i, *cg_j;     /* work grids of g_i, g_j (ModeTables 0) */
  NGR          *g_ir, *g_jr;
  unsigned int  N, N_i, N_j, N_k;
  double        kmag, kmag2, kmag1;
  long double   fnl_fac;
  uint64_t      coord;
  unsigned int  bytes = 0;
  lpt_complex  *(cfields[8]);    /* recursion buffers (k-space, double) */

  /* The NG potential is sum_{N_i <= N_j} FFT[g_i g_j](k3) K_{ij}(k3) over the real-space mode fields g_N.
   * ModeTables 0 rebuilds and inverse-transforms g_j for every pair (least memory). ModeTables 1 builds each g_N
   * once and keeps all N_modes of them in real space, and tabulates the per-mode values of K_{ij}: about N_modes
   * extra grids, half the FFTs. The two agree to rounding (the g_j of pairs with N_i >= 5 are recursed from 4
   * instead of from N_i). */
  NGR         **gcache = NULL;
  ModalModeList ML     = {0, 0, NULL, NULL};

  /* recursion buffers of the mode fields (k-space, double): 0-2 for g_i, 4-6 for g_j */
  for(N = 0; N < 8; N++)
    {
      cfields[N] = NULL;
      if(N == 3 || N == 7 || (ModeTables && N >= 3))
        continue;

      cfields[N] = (lpt_complex *) lpt_alloc_grid();
      ASSERT_ALLOC(cfields[N]);
    }

  /* work grids of g_i and g_j (without the cache), in the precision of the NG stage */
  cg_i = cg_j = NULL;
  if(!ModeTables)
    {
      cg_i = (NGC *) NG_ALLOC();
      cg_j = (NGC *) NG_ALLOC();
      ASSERT_ALLOC(cg_i && cg_j);
    }
  g_ir = (NGR *) cg_i;
  g_jr = (NGR *) cg_j;

  c_tmp = (NGC *) NG_ALLOC();
  p_tmp = (NGR *) c_tmp;
  ASSERT_ALLOC(c_tmp);

  if(ModeTables)
    {
      gcache = malloc(N_modes * sizeof(NGR *));
      ASSERT_ALLOC(gcache);
      for(N = 0; N < N_modes; N++)
        {
          gcache[N] = (NGR *) NG_ALLOC();
          ASSERT_ALLOC(gcache[N]);
        }

      if(ThisTask == 0)
        printf("  ModeTables 1: caching %d real-space mode fields (%.2f GB over all ranks).\n", N_modes,
               N_modes * (double) sizeof(NGR) * TotalSizePlusAdditional * NTask / 1e9);
    }

  lpt_barrier();

  /* first, clean the array */
  for(N = 0; N < 8; N++)
    if(cfields[N])
      memset(cfields[N], 0, sizeof(lpt_real) * TotalSizePlusAdditional);
  if(cg_i)
    {
      memset(cg_i, 0, sizeof(NGR) * TotalSizePlusAdditional);
      memset(cg_j, 0, sizeof(NGR) * TotalSizePlusAdditional);
    }
  memset(c_tmp, 0, sizeof(NGR) * TotalSizePlusAdditional);

  lpt_barrier();

  lpt_complex *ri[3] = {cfields[0], cfields[1], cfields[2]};   /* recursion buffers of g_i */

  if(ModeTables)
    {
      /* build every mode field once, in order, and keep it in real space */
      for(N = 0; N < N_modes; N++)
        {
          memset(gcache[N], 0, sizeof(NGR) * TotalSizePlusAdditional);
          NG(modal_build_field)(N, N == 4, cpot, ri, (NGC *) gcache[N], X0Table, kmin_ns, kmax_ns);
          if(N >= 4)
            modal_rotate(ri);
          lpt_barrier();

          NG_INV(gcache[N]);
          lpt_barrier();
        }

      modal_build_mode_list(&ML, kmin_ns, kmax_ns);
    }

  /* Loop over the mode pairs (N_i <= N_j) */
  for(N_i = 0; N_i < N_modes; N_i++)
    {
      NGR *gi, *gj;

      if(ModeTables)
        gi = gcache[N_i];
      else
        {
          /* g_i in Fourier space (zeroed first: the previous FFT left this grid in real space) */
          memset(cg_i, 0, sizeof(NGR) * TotalSizePlusAdditional);
          NG(modal_build_field)(N_i, N_i == 4, cpot, ri, cg_i, X0Table, kmin_ns, kmax_ns);
          if(N_i >= 4)
            modal_rotate(ri);     /* the next N_i continues the recursion */
          lpt_barrier();

          NG_INV(g_ir);
          lpt_barrier();
          gi = g_ir;
        }

      lpt_complex *rj[3] = {cfields[4], cfields[5], cfields[6]};   /* recursion buffers of g_j (restarted at N_i) */

      for(N_j = N_i; N_j < N_modes; N_j++)
        {
          /* the pair can be skipped if all its alpha coefficients are zero */
          ksum = 0;
          for(N_k = N_j; N_k < N_modes; N_k++)
            if(fabs(AlphaTable[N_i][N_j][N_k]) > 0)
              ksum = 1;

          if(ModeTables)
            gj = gcache[N_j];
          else
            {
              /* g_j is needed for this pair, and for N_j >= 4 also to carry the recursion to a later pair */
              int needed = ksum;

              for(unsigned int M = N_j + 1; M < N_modes && N_j >= 4 && !needed; M++)
                for(N_k = M; N_k < N_modes; N_k++)
                  if(fabs(AlphaTable[N_i][M][N_k]) > 0)
                    needed = 1;

              if(needed)
                {
                  /* g_j in Fourier space (zeroed first, as g_i) */
                  memset(cg_j, 0, sizeof(NGR) * TotalSizePlusAdditional);
                  NG(modal_build_field)(N_j, (N_j == 4) || ((N_i >= 4) && (N_i == N_j)), cpot, rj, cg_j, X0Table,
                                        kmin_ns, kmax_ns);
                  if(N_j >= 4)
                    modal_rotate(rj);
                  lpt_barrier();
                }
              gj = g_jr;
            }

          if(ThisTask == 0)
            {
              printf("Computing combination %d, %d ....", N_i, N_j);
              fflush(stdout);
            }

          if(ksum == 0)
            {
              NG(modal_pair_done)(N_i, N_j);
              continue;
            }

          if(!ModeTables)
            {
              NG_INV(g_jr);
              lpt_barrier();
            }

          /* product of the two mode fields in real space */
          NG(modal_multiply_fields)(p_tmp, gi, gj);
          lpt_barrier();

          NG_FWD(p_tmp);
          lpt_barrier();

          /* output-leg kernel K_ij(k3), summed over N_k for this pair */
          PROF_BEGIN(PROF_ACCUM)
          const double *alpha = AlphaTable[N_i][N_j];

          if(ModeTables)
            {
              for(long long m = 0; m < ML.n; m++)
                {
                  const double *v = ML.v + m * ML.nt;

                  fnl_fac = modal_k3_factor(alpha, N_j, v[0], 0, v[1], v + 2, v + 6, X0Table);
                  coord   = ML.coord[m];
                  cpot_NG[coord].re += (double) (fnl_fac * c_tmp[coord].re);
                  cpot_NG[coord].im += (double) (fnl_fac * c_tmp[coord].im);
                }
            }
          else
            {
              for(ii = 0; ii < Band_ny; ii++)             /* band layout: rows of y */
                {
                  j = Band_y[ii];
                  if(!lpt_axis_in_band(j))
                    continue;

                  for(i = 0; i < Nmesh; i++)
                    {
                      if(!lpt_axis_in_band(i))
                        continue;

                      for(k = 0; k <= lpt_kz_end(); k++)
                        {
                          coord = lpt_band_coord(ii, i, k);
                          if(!modal_mode_in_band(i, j, k, &kmag2, &kmag1))
                            continue;

                          kmag    = modal_legendre_x(kmag1, kmin_ns, kmax_ns);
                          fnl_fac = modal_k3_factor(alpha, N_j, kmag, kmag1, kmag2, NULL, NULL, X0Table);

                          cpot_NG[coord].re += (double) (fnl_fac * c_tmp[coord].re);
                          cpot_NG[coord].im += (double) (fnl_fac * c_tmp[coord].im);
                        }
                    }
                }
            }
          PROF_END(PROF_ACCUM)

          NG(modal_pair_done)(N_i, N_j);
          lpt_barrier();
        }
    }


  /**** Extra templates cancelling the divergent (0,1,2) term ****/

  /* only the pairs (0, 1) and (0, 2) carry a t_012 term; the recursion buffers 2..7 are no longer needed */
  for(N = 2; N < 8; N++)
    free(cfields[N]);
  free(ML.coord);
  free(ML.v);

  if(N_modes >= 3 && t_012 != 0)
    {
      NGR *g0, *gj;

      if(ModeTables)
        g0 = gcache[0];
      else
        {
          memset(cg_i, 0, sizeof(NGR) * TotalSizePlusAdditional);
          NG(modal_build_field)(0, 0, cpot, ri, cg_i, X0Table, kmin_ns, kmax_ns);
          lpt_barrier();

          NG_INV(g_ir);
          lpt_barrier();
          g0 = g_ir;
        }

      for(N_j = 1; N_j <= 2; N_j++)
        {
          if(ModeTables)
            gj = gcache[N_j];
          else
            {
              memset(cg_j, 0, sizeof(NGR) * TotalSizePlusAdditional);
              NG(modal_build_field)(N_j, 0, cpot, ri, cg_j, X0Table, kmin_ns, kmax_ns);
              lpt_barrier();

              NG_INV(g_jr);
              lpt_barrier();
              gj = g_jr;
            }

          NG(modal_multiply_fields)(p_tmp, g0, gj);
          lpt_barrier();

          NG_FWD(p_tmp);
          lpt_barrier();

          if(ThisTask == 0)
            {
              printf("Computing Beta combination %d, %d ....", 0, N_j);
              fflush(stdout);
            }

          for(ii = 0; ii < Band_ny; ii++)             /* band layout: rows of y */
            {
              j = Band_y[ii];
              if(!lpt_axis_in_band(j))
                continue;

              for(i = 0; i < Nmesh; i++)
                {
                  if(!lpt_axis_in_band(i))
                    continue;

                  for(k = 0; k <= lpt_kz_end(); k++)
                    {
                      coord = lpt_band_coord(ii, i, k);
                      if(!modal_mode_in_band(i, j, k, &kmag2, &kmag1))
                        continue;

                      fnl_fac = 0;

                      /* the t_012 template cancelling the divergent (0, 1, 2) term */
                      if(N_j == 1)
                        fnl_fac += -AlphaTable[0][1][2] * t_012 * pow(kmag1, (4 - PrimordialIndex)/3 * (2 - 3.0));
                      else
                        fnl_fac +=  AlphaTable[0][1][2] * t_012 * pow(kmag1, (4 - PrimordialIndex)/3 * (1 - 3.0));

                      cpot_NG[coord].re += (double) (fnl_fac * c_tmp[coord].re);
                      cpot_NG[coord].im += (double) (fnl_fac * c_tmp[coord].im);
                    }
                }
            }

          if(ThisTask == 0)
            print_timed_done(18);
          lpt_barrier();
        }
    }

  free(c_tmp);
  for(N = 0; N < 2; N++)
    free(cfields[N]);
  free(cg_i);
  free(cg_j);

  if(gcache)
    {
      for(N = 0; N < N_modes; N++)
        free(gcache[N]);
      free(gcache);
    }
}
