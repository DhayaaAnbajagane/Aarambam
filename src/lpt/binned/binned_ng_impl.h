/* The grid-touching part of the binned non-Gaussian stage (the filtered input fields, their products and FFTs, and
 * the accumulation of the NG term), included twice by main.c: for NGPrecision double (NG(name) = name_d, NGR = lpt_real,
 * NGC = lpt_complex) and for float (name_f, float, lpt_complexf). The Gaussian potential (cpot) and the accumulated NG
 * term (cpot_NG / out) are double in both; NG_ALLOC, NG_FWD and NG_INV are the grid allocation and the band FFTs of
 * the precision. */

/* Returns whether the field is nonzero on some rank (collective): a field that is zero everywhere (a bin outside the
 * k range of the grid) gives products and FFTs that are exactly zero, so its pairs can be skipped. */
static int NG(bpng_build_input_filtered_field)(NGC *dst, lpt_complex *src, int axis, int node)
{
  int nonzero = 0, nonzero_all;

  PROF_BEGIN(PROF_FIELD)
  ptrdiff_t ii, j, k, i;
  uint64_t  coord;
  double    kx, ky, kz, kmag, filt;

  memset(dst, 0, sizeof(NGR) * TotalSizePlusAdditional);

  if(ModeTables)
    {
      for(long long m = 0; m < BML.n; m++)
        {
          filt = bpng_input_basis_value(axis, node, BML.kmag[m]);
          if(filt == 0.0)
            continue;

          coord = BML.coord[m];
          dst[coord].re = filt * src[coord].re;
          dst[coord].im = filt * src[coord].im;
          nonzero |= (dst[coord].re != 0 || dst[coord].im != 0);
        }
    }
  else
    {
      for(ii = 0; ii < Local_nx; ii++)
        {
          i = ii + Local_x_start;
          if(!lpt_axis_in_band(i))
            continue;

          for(j = 0; j < Nmesh; j++)
            {
              if(!lpt_axis_in_band(j))
                continue;

              for(k = 0; k <= lpt_kz_end(); k++)
                {
                  coord = (ii * Nmesh + j) * (Nmesh / 2 + 1) + k;

                  kx   = lpt_wavenumber(i);
                  ky   = lpt_wavenumber(j);
                  kz   = lpt_wavenumber(k);
                  kmag = sqrt(kx*kx + ky*ky + kz*kz);

                  if(!bpng_mode_allowed(i, j, k, kx, ky, kz, kmag))
                    continue;
                  filt = bpng_input_basis_value(axis, node, kmag);
                  if(filt == 0.0)
                    continue;

                  dst[coord].re = filt * src[coord].re;
                  dst[coord].im = filt * src[coord].im;
                  nonzero |= (dst[coord].re != 0 || dst[coord].im != 0);
                }
            }
        }
    }
  PROF_END(PROF_FIELD)
  MPI_Allreduce(&nonzero, &nonzero_all, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  return nonzero_all;
}

/* out = a * b on the real-space grid, row padding zeroed (no full-grid clear needed). */
static void NG(bpng_multiply_real_fields)(NGR *out, NGR *a, NGR *b)
{
  PROF_BEGIN(PROF_MULT)
  ptrdiff_t i, j, k;
  uint64_t  coord;

  for(i = 0; i < Local_nx; i++)
    for(j = 0; j < Nmesh; j++)
      {
        for(k = 0; k < Nmesh; k++)
          {
            coord      = (i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k;
            out[coord] = a[coord] * b[coord];
          }

        for(k = Nmesh; k < 2 * (Nmesh / 2 + 1); k++)      /* row padding: the only other input of the r2c FFT */
          out[(i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k] = 0;
      }
  PROF_END(PROF_MULT)
}


/* ---- the NG potential: loops over bin pairs ---- */

static NGR **NG(bpng_alloc_real_field_cache)(int nfield)
{
  int        a;
  NGR      **cache;

  if(nfield <= 0)
    return NULL;

  cache = (NGR **) malloc(nfield * sizeof(NGR *));
  if(!cache)
    FatalError(9301);

  for(a = 0; a < nfield; a++)
    {
      cache[a] = (NGR *) NG_ALLOC();
      if(!cache[a])
        FatalError(9302);
    }
  return cache;
}

static void NG(bpng_free_real_field_cache)(NGR **cache, int nfield)
{
  int a;

  if(!cache)
    return;

  for(a = 0; a < nfield; a++)
    if(cache[a])
      free(cache[a]);
  free(cache);
}

/* Filtered input field psi_node(k) phi(k), transformed to real space. Returns 0 (and skips the FFT) if it is zero
 * everywhere. */
static int NG(bpng_fill_cached_real_field)(NGR *field, lpt_complex *src, int axis, int node)
{
  if(!NG(bpng_build_input_filtered_field)((NGC *) field, src, axis, node))
    return 0;
  lpt_barrier();
  NG_INV(field);
  lpt_barrier();
  return 1;
}

/* out += Fnl K_ab(k3) conv(k3). conv is a forward FFT in the band layout (lpt_fft.h); out is accumulated in the
 * same layout. */
static void NG(bpng_accumulate_hybrid_convolution)(NGC *conv, lpt_complex *out, int ia, int ib)
{
  PROF_BEGIN(PROF_ACCUM)
  ptrdiff_t jj, j, k, i;
  uint64_t  coord;
  double    kx, ky, kz, kmag, coef;
  int       ic;

  if(ModeTables)
    {
      /* the coefficients of this pair (and of its symmetric partner (ib, ia)) for every output bin, read once per
       * pair instead of once per mode; the modes stay in memory order. Same arithmetic as
       * bpng_output_k3_pair_coefficient_u. */
      const int pair = ia != ib, nb3 = BPNG.nk[2];
      double    ca[nb3][4], cb[nb3][4];

      for(ic = 0; ic < nb3; ic++)
        {
          for(int q = 0; q < 4; q++)
            ca[ic][q] = cb[ic][q] = 0;
          bpng_cell_coefficients(ia, ib, ic, ca[ic]);
          if(pair)
            bpng_cell_coefficients(ib, ia, ic, cb[ic]);
        }

      for(long long m = 0; m < BMLT.n; m++)
        {
          ic = BMLT.ic[m];
          if(ic < 0)
            continue;

          coef = bpng_cell_polynomial(ca[ic], BMLT.u[m]);
          if(pair)
            coef += bpng_cell_polynomial(cb[ic], BMLT.u[m]);
          if(coef == 0.0 || !isfinite(coef))
            continue;

          coord = BMLT.coord[m];
          out[coord].re += Fnl * coef * conv[coord].re;
          out[coord].im += Fnl * coef * conv[coord].im;
        }
    }
  else
    {
      for(jj = 0; jj < Band_ny; jj++)             /* band layout: rows of y */
        {
          j = Band_y[jj];
          if(!lpt_axis_in_band(j))
            continue;

          for(i = 0; i < Nmesh; i++)
            {
              if(!lpt_axis_in_band(i))
                continue;

              for(k = 0; k <= lpt_kz_end(); k++)
                {
                  coord = lpt_band_coord(jj, i, k);

                  kx   = lpt_wavenumber(i);
                  ky   = lpt_wavenumber(j);
                  kz   = lpt_wavenumber(k);
                  kmag = sqrt(kx*kx + ky*ky + kz*kz);

                  if(!bpng_mode_allowed(i, j, k, kx, ky, kz, kmag))
                    continue;
                  ic = bpng_find_bin(BPNG.edges[2], BPNG.nk[2], kmag);
                  if(ic < 0)
                    continue;

                  /* u once per mode: the symmetric pair uses it twice */
                  coef = bpng_output_k3_pair_coefficient_u(ia, ib, ic,
                                                           (BPNG.coeff_count <= 1) ? 0.0
                                                                                   : bpng_u_in_bin_axis(2, ic, kmag));
                  if(coef == 0.0 || !isfinite(coef))
                    continue;

                  out[coord].re += Fnl * coef * conv[coord].re;
                  out[coord].im += Fnl * coef * conv[coord].im;
                }
            }
        }
    }
  PROF_END(PROF_ACCUM)
}

/* One pair: product of the two real-space fields, forward FFT, output-leg accumulation. */
static void NG(bpng_convolve_and_accumulate_hybrid)(NGR *field_a, NGR *field_b, NGC *c_tmp, NGR *p_tmp,
                                                    lpt_complex *out, int ia, int ib)
{
  NG(bpng_multiply_real_fields)(p_tmp, field_a, field_b);
  lpt_barrier();

  NG_FWD(p_tmp);
  lpt_barrier();

  NG(bpng_accumulate_hybrid_convolution)(c_tmp, out, ia, ib);
}

/* ia <= ib pairs, with the filtered real-space fields of block_size k1 bins (and of the k2 bins they pair with)
 * kept in memory, so each field is transformed once per block instead of once per pair. Pairs done by the multi-grid
 * stage are skipped, and fields only they use are not built. */
static void NG(bpng_compute_non_gaussian_potential_symmetric_blocked)(
    lpt_complex *cpot, lpt_complex *cpot_NG,
    NGC *c_tmp, NGR *p_tmp, int block_size)
{
  int        n, a0, b0, ia, ib, na, nb, la, lb, start_b;
  int        timed_chars = 0, npair = 0, nskip = 0;
  NGR      **acache, **bcache;
  int       *anz, *bnz;           /* filtered field nonzero somewhere */

  n = BPNG.nk[0];
  if(block_size <= 0)
    block_size = 1;
  if(block_size > n)
    block_size = n;

  for(a0 = 0; a0 < n; a0 += block_size)
    {
      na = block_size;
      if(a0 + na > n)
        na = n - a0;

      if(ThisTask == 0)
        {
          timed_chars = printf("  binned PNG internal block a=[%d,%d) of %d...", a0, a0 + na, n);
          fflush(stdout);
        }

      acache = NG(bpng_alloc_real_field_cache)(na);
      anz    = malloc(sizeof(int) * na);
      for(la = 0; la < na; la++)
        anz[la] = bpng_mg_field_needed(a0 + la) && NG(bpng_fill_cached_real_field)(acache[la], cpot, 0, a0 + la);

      for(b0 = a0; b0 < n; b0 += block_size)
        {
          nb = block_size;
          if(b0 + nb > n)
            nb = n - b0;

          if(b0 == a0)
            {
              bcache = acache;
              bnz    = anz;
            }
          else
            {
              bcache = NG(bpng_alloc_real_field_cache)(nb);
              bnz    = malloc(sizeof(int) * nb);
              for(lb = 0; lb < nb; lb++)
                bnz[lb] = bpng_mg_field_needed(b0 + lb) && NG(bpng_fill_cached_real_field)(bcache[lb], cpot, 1, b0 + lb);
            }

          for(la = 0; la < na; la++)
            {
              ia      = a0 + la;
              start_b = (b0 == a0) ? la : 0;

              for(lb = start_b; lb < nb; lb++)
                {
                  ib = b0 + lb;
                  if(bpng_mg_pair_done(ia, ib))
                    continue;
                  npair++;
                  if(!anz[la] || !bnz[lb])        /* a zero field: the pair adds exactly zero */
                    {
                      nskip++;
                      continue;
                    }
                  NG(bpng_convolve_and_accumulate_hybrid)(acache[la], bcache[lb], c_tmp, p_tmp, cpot_NG, ia, ib);
                }
            }

          if(b0 != a0)
            {
              NG(bpng_free_real_field_cache)(bcache, nb);
              free(bnz);
            }
        }

      NG(bpng_free_real_field_cache)(acache, na);
      free(anz);
      if(ThisTask == 0)
        print_timed_done_aligned(timed_chars);
    }
  if(ThisTask == 0 && nskip > 0)
    printf("  %d of %d bin pairs skipped: a filtered field is zero on this grid (bin outside its k range).\n",
           nskip, npair);
}

/* All ia <= ib pairs with two work fields: the k2 field is rebuilt and transformed for every pair (least memory).
 * Pairs done by the multi-grid stage are skipped. */
static void NG(bpng_compute_non_gaussian_potential_unblocked)(lpt_complex *cpot, lpt_complex *cpot_NG,
                                                              NGC *c_tmp, NGR *p_tmp)
{
  int      ia, ib, timed_chars = 0;
  uint64_t bytes = 0;
  NGC     *cf1 = (NGC *) NG_ALLOC(), *cf2 = (NGC *) NG_ALLOC();
  NGR     *f1  = (NGR *) cf1, *f2 = (NGR *) cf2;

  ASSERT_ALLOC(cf1 && cf2);

  if(ThisTask == 0)
    {
      printf("  Without block caching (BinnedPNGCacheBlockSize 0).\n");
      fflush(stdout);
    }

  for(ia = 0; ia < BPNG.nk[0]; ia++)
    {
      if(ThisTask == 0)
        {
          timed_chars = printf("  binned PNG internal k1 basis %d/%d...", ia + 1, BPNG.nk[0]);
          fflush(stdout);
        }

      if(!bpng_mg_field_needed(ia) || !NG(bpng_build_input_filtered_field)(cf1, cpot, 0, ia))   /* none to do here */
        {
          if(ThisTask == 0)
            print_timed_done_aligned(timed_chars);
          continue;
        }
      lpt_barrier();
      NG_INV(f1);
      lpt_barrier();

      for(ib = ia; ib < BPNG.nk[1]; ib++)
        {
          if(bpng_mg_pair_done(ia, ib))
            continue;
          if(ib == ia)
            {
              /* Reuse the already transformed field on the diagonal. */
              NG(bpng_convolve_and_accumulate_hybrid)(f1, f1, c_tmp, p_tmp, cpot_NG, ia, ib);
            }
          else
            {
              if(!NG(bpng_build_input_filtered_field)(cf2, cpot, 1, ib))
                continue;
              lpt_barrier();
              NG_INV(f2);
              lpt_barrier();

              NG(bpng_convolve_and_accumulate_hybrid)(f1, f2, c_tmp, p_tmp, cpot_NG, ia, ib);
            }
        }

      if(ThisTask == 0)
        print_timed_done_aligned(timed_chars);
    }

  free(cf1);
  free(cf2);
}

static void NG(bpng_compute_non_gaussian_potential)(lpt_complex *cpot, lpt_complex *cpot_NG)
{
  int           cache_block;
  uint64_t      bytes = 0;
  NGC          *c_tmp;
  NGR          *p_tmp;

  if(ThisTask == 0)
    {
      printf("Computing hybrid binned-kernel non-Gaussian potential (k3 is output leg)...\n");
      fflush(stdout);
    }

  bpng_read_and_broadcast_table();
  bpng_precompute_axes();
  if(ModeTables)
    bpng_build_mode_list();

  bpng_multigrid_pairs(cpot, cpot_NG);        /* pairs that fit a smaller grid (BinnedPNGMultiGrid) */

  cache_block = BinnedPNGCacheBlockSize;
  if(cache_block < 0)
    cache_block = 0;

  c_tmp = (NGC *) NG_ALLOC();
  p_tmp = (NGR *) c_tmp;
  ASSERT_ALLOC(c_tmp);

  if(cache_block > 0)       /* blocked path: filtered fields cached in real space */
    {
      if(ThisTask == 0)
        {
          printf("  Using block cache size %d.\n", cache_block);
          fflush(stdout);
        }
      NG(bpng_compute_non_gaussian_potential_symmetric_blocked)(cpot, cpot_NG, c_tmp, p_tmp, cache_block);
    }
  else
    NG(bpng_compute_non_gaussian_potential_unblocked)(cpot, cpot_NG, c_tmp, p_tmp);

  free(c_tmp);
  bpng_mg_free();
  if(ModeTables)
    bpng_free_mode_list();
  bpng_free_table();

  if(ThisTask == 0)
    {
      printf("Finished hybrid binned-kernel non-Gaussian potential.\n");
      fflush(stdout);
    }
}
