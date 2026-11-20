/* The grid-touching part of the binned non-Gaussian stage (the filtered input fields, their products and FFTs, and
 * the accumulation of the NG term), included twice by main.c: for NGPrecision double (NG(name) = name_d, NGR = lpt_real,
 * NGC = lpt_complex) and for float (name_f, float, lpt_complexf). The Gaussian potential (cpot) and the accumulated NG
 * term (cpot_NG / out) are double in both; NG_ALLOC, NG_FWD and NG_INV are the grid allocation and the band FFTs of
 * the precision. */

static void NG(bpng_build_input_filtered_field)(NGC *dst, lpt_complex *src, int axis, int node)
{
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

                  kx   = (i < Nmesh / 2) ? i * 2 * PI / Box : -(Nmesh - i) * 2 * PI / Box;
                  ky   = (j < Nmesh / 2) ? j * 2 * PI / Box : -(Nmesh - j) * 2 * PI / Box;
                  kz   = (k < Nmesh / 2) ? k * 2 * PI / Box : -(Nmesh - k) * 2 * PI / Box;
                  kmag = sqrt(kx*kx + ky*ky + kz*kz);

                  if(!bpng_mode_allowed(i, j, k, kx, ky, kz, kmag))
                    continue;
                  filt = bpng_input_basis_value(axis, node, kmag);
                  if(filt == 0.0)
                    continue;

                  dst[coord].re = filt * src[coord].re;
                  dst[coord].im = filt * src[coord].im;
                }
            }
        }
    }
  PROF_END(PROF_FIELD)
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

/* Filtered input field psi_node(k) phi(k), transformed to real space. */
static void NG(bpng_fill_cached_real_field)(NGR *field, lpt_complex *src, int axis, int node)
{
  NG(bpng_build_input_filtered_field)((NGC *) field, src, axis, node);
  lpt_barrier();
  NG_INV(field);
  lpt_barrier();
}

static void NG(bpng_convolve_and_accumulate_hybrid)(
    NGC *field_a, NGR *field_a_real,
    NGC *field_b, NGR *field_b_real,
    NGC *c_tmp, NGR *p_tmp,
    lpt_complex *out,
    int ia, int ib, int use_symmetry);

/* ia <= ib pairs, with the filtered real-space fields of block_size k1 bins (and of the k2 bins they pair with)
 * kept in memory, so each field is transformed once per block instead of once per pair. */
static void NG(bpng_compute_non_gaussian_potential_symmetric_blocked)(
    lpt_complex *cpot, lpt_complex *cpot_NG,
    NGC *c_tmp, NGR *p_tmp, int block_size)
{
  int        n, a0, b0, ia, ib, na, nb, la, lb, start_b;
  int        timed_chars = 0;
  NGR      **acache, **bcache;

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
      for(la = 0; la < na; la++)
        NG(bpng_fill_cached_real_field)(acache[la], cpot, 0, a0 + la);

      for(b0 = a0; b0 < n; b0 += block_size)
        {
          nb = block_size;
          if(b0 + nb > n)
            nb = n - b0;

          if(b0 == a0)
            bcache = acache;
          else
            {
              bcache = NG(bpng_alloc_real_field_cache)(nb);
              for(lb = 0; lb < nb; lb++)
                NG(bpng_fill_cached_real_field)(bcache[lb], cpot, 1, b0 + lb);
            }

          for(la = 0; la < na; la++)
            {
              ia      = a0 + la;
              start_b = (b0 == a0) ? la : 0;

              for(lb = start_b; lb < nb; lb++)
                {
                  ib = b0 + lb;
                  NG(bpng_convolve_and_accumulate_hybrid)((NGC *) acache[la], acache[la],
                                                          (NGC *) bcache[lb], bcache[lb],
                                                      c_tmp, p_tmp,
                                                      cpot_NG, ia, ib, 1);
                }
            }

          if(b0 != a0)
            NG(bpng_free_real_field_cache)(bcache, nb);
        }

      NG(bpng_free_real_field_cache)(acache, na);
      if(ThisTask == 0)
        print_timed_done_aligned(timed_chars);
    }
}

/* out += Fnl K_ab(k3) conv(k3). conv is a forward FFT in the band layout (lpt_fft.h); out is accumulated in the
 * same layout. */
static void NG(bpng_accumulate_hybrid_convolution)(NGC *conv, lpt_complex *out,
                                               int ia, int ib, int use_symmetry)
{
  PROF_BEGIN(PROF_ACCUM)
  ptrdiff_t jj, j, k, i;
  uint64_t  coord;
  double    kx, ky, kz, kmag, coef;
  int       ic;

  if(ModeTables)
    {
      for(long long m = 0; m < BMLT.n; m++)
        {
          ic = BMLT.ic[m];
          if(ic < 0)
            continue;

          coef = bpng_output_k3_pair_coefficient_u(ia, ib, ic, BMLT.u[m], use_symmetry);
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

                  kx   = (i < Nmesh / 2) ? i * 2 * PI / Box : -(Nmesh - i) * 2 * PI / Box;
                  ky   = (j < Nmesh / 2) ? j * 2 * PI / Box : -(Nmesh - j) * 2 * PI / Box;
                  kz   = (k < Nmesh / 2) ? k * 2 * PI / Box : -(Nmesh - k) * 2 * PI / Box;
                  kmag = sqrt(kx*kx + ky*ky + kz*kz);

                  if(!bpng_mode_allowed(i, j, k, kx, ky, kz, kmag))
                    continue;
                  ic = bpng_find_bin(BPNG.edges[2], BPNG.nk[2], kmag);
                  if(ic < 0)
                    continue;

                  /* u once per mode: the symmetric pair uses it twice */
                  coef = bpng_output_k3_pair_coefficient_u(ia, ib, ic,
                                                           (BPNG.coeff_count <= 1) ? 0.0
                                                                                   : bpng_u_in_bin_axis(2, ic, kmag),
                                                           use_symmetry);
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
static void NG(bpng_convolve_and_accumulate_hybrid)(
    NGC *field_a, NGR *field_a_real,
    NGC *field_b, NGR *field_b_real,
    NGC *c_tmp, NGR *p_tmp,
    lpt_complex *out,
    int ia, int ib, int use_symmetry)
{
  NG(bpng_multiply_real_fields)(p_tmp, field_a_real, field_b_real);
  lpt_barrier();

  NG_FWD(p_tmp);
  lpt_barrier();

  NG(bpng_accumulate_hybrid_convolution)(c_tmp, out, ia, ib, use_symmetry);
}

static void NG(bpng_compute_non_gaussian_potential)(lpt_complex *cpot, lpt_complex *cpot_NG)
{
  int           ia, ib, ib_start, use_symmetry, cache_block;
  int           timed_chars = 0;
  uint64_t      bytes = 0;
  NGC          *cf1, *cf2, *c_tmp;
  NGR          *f1, *f2, *p_tmp;

  if(ThisTask == 0)
    {
      printf("Computing hybrid binned-kernel non-Gaussian potential (k3 is output leg)...\n");
      fflush(stdout);
    }

  bpng_read_and_broadcast_table();
  bpng_precompute_axes();
  if(ModeTables)
    bpng_build_mode_list();

  use_symmetry = (BinnedPNGUseInternalSymmetry != 0);
  if(use_symmetry && !bpng_internal_edges_are_identical())
    {
      if(ThisTask == 0)
        printf("ERROR: BinnedPNGUseInternalSymmetry=1 but k1/k2 edges differ. Disable the flag or regenerate "
               "with matched k1/k2 bins.\n");
      FatalError(9310);
    }

  cache_block = BinnedPNGCacheBlockSize;
  if(cache_block < 0)
    cache_block = 0;

  c_tmp = (NGC *) NG_ALLOC();
  p_tmp = (NGR *) c_tmp;
  ASSERT_ALLOC(c_tmp);

  /* blocked path: filtered fields cached in real space */
  if(use_symmetry && cache_block > 0)
    {
      if(ThisTask == 0)
        {
          printf("  Using internal-leg symmetry and block cache size %d.\n", cache_block);
          fflush(stdout);
        }

      NG(bpng_compute_non_gaussian_potential_symmetric_blocked)(cpot, cpot_NG,
                                                            c_tmp, p_tmp, cache_block);

      free(c_tmp);
      if(ModeTables)
        bpng_free_mode_list();
      bpng_free_table();

      if(ThisTask == 0)
        {
          printf("Finished hybrid binned-kernel non-Gaussian potential.\n");
          fflush(stdout);
        }
      return;
    }

  /* unblocked path: two work fields, the k2 field rebuilt for every pair */
  cf1 = (NGC *) NG_ALLOC();
  cf2 = (NGC *) NG_ALLOC();
  f1  = (NGR *) cf1;
  f2  = (NGR *) cf2;
  ASSERT_ALLOC(cf1 && cf2 && c_tmp);

  if(ThisTask == 0 && use_symmetry)
    {
      printf("  Using internal-leg symmetry without block caching.\n");
      fflush(stdout);
    }

  for(ia = 0; ia < BPNG.nk[0]; ia++)
    {
      if(ThisTask == 0)
        {
          timed_chars = printf("  binned PNG internal k1 basis %d/%d...", ia + 1, BPNG.nk[0]);
          fflush(stdout);
        }

      NG(bpng_build_input_filtered_field)(cf1, cpot, 0, ia);
      lpt_barrier();
      NG_INV(f1);
      lpt_barrier();

      ib_start = (use_symmetry ? ia : 0);
      for(ib = ib_start; ib < BPNG.nk[1]; ib++)
        {
          if(use_symmetry && ib == ia)
            {
              /* Reuse the already transformed field on the diagonal. */
              NG(bpng_convolve_and_accumulate_hybrid)(cf1, f1, cf1, f1, c_tmp, p_tmp,
                                                  cpot_NG, ia, ib, use_symmetry);
            }
          else
            {
              NG(bpng_build_input_filtered_field)(cf2, cpot, 1, ib);
              lpt_barrier();
              NG_INV(f2);
              lpt_barrier();

              NG(bpng_convolve_and_accumulate_hybrid)(cf1, f1, cf2, f2, c_tmp, p_tmp,
                                                  cpot_NG, ia, ib, use_symmetry);
            }
        }

      if(ThisTask == 0)
        print_timed_done_aligned(timed_chars);
    }

  free(cf1);
  free(cf2);
  free(c_tmp);
  if(ModeTables)
    bpng_free_mode_list();
  bpng_free_table();

  if(ThisTask == 0)
    {
      printf("Finished hybrid binned-kernel non-Gaussian potential.\n");
      fflush(stdout);
    }
}
