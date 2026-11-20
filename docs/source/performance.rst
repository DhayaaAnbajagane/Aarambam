Running faster
==============

The 2LPT executables have a few options that make them faster at the cost of memory or of an extra
build step. None of them changes the ICs beyond rounding. This page lists them in the order in which
they usually matter; :doc:`conventions` has the details and the timings behind them.

Quick recipe
------------

With enough memory, the fastest settings are

.. code-block:: python

   cfg = Aarambam.utils.make_config_binned(..., ModeTables = 1, BinnedPNGCacheBlockSize = 64,
                                           FFTBackend = 'band')
   cfg = Aarambam.utils.make_config_basis(..., ModeTables = 1, FFTBackend = 'band')

or, in a config file written by hand,

.. code-block:: text

   ModeTables                1
   BinnedPNGCacheBlockSize   64      # binned only
   FFTBackend                band

and leaving ``FFTWWisdomDir`` at its default. On one 48-core node, the binned method at
:math:`384^3` / ``Nsample 256`` on 32 ranks then takes 10.9 s, against 21.4 s with
``ModeTables 0`` and the ``fftw`` backend, and 43.4 s for version 1.0.0.

If a relative precision of about :math:`10^{-6}` on the non-Gaussian term is enough, add
``NGPrecision float`` (``NGPrecision = 'float'``): binned 24% and basis 9% faster at
:math:`192^3` on 8 ranks, with half the memory for that stage (see `Single precision`_).

The options
-----------

==============================  =======================================  ========================================
option                          gain                                     cost
==============================  =======================================  ========================================
``ModeTables 1``                basis 2.3-2.6x; binned 1.5-1.7x (with    basis: about ``N_modes`` extra grids;
                                a full cache block, next line)           binned: about one extra grid
``BinnedPNGCacheBlockSize 64``  binned: all filtered fields kept, fewer  about one extra grid per cached field
                                passes over the grid
``FFTBackend band``             14-16% (basis), 22-25% (binned) at       one buffer of about 0.44 of a grid per
                                ``Nmesh = 3/2 Nsample``                  rank; not timed across nodes
``FFTWWisdomDir`` (default)     FFTs 7-27% faster with measured plans    the first run with a new grid size, rank
                                                                         count or CPU type measures them (seconds
                                                                         to ~30 s at :math:`1024^3`)
``Aarambam_NATIVE=ON``          a few per cent                           build option; the executables then run
                                                                         only on CPUs like the build machine's
``NGPrecision float``           binned 15-24%, basis 0-9% at 192^3       NG term to ~1e-6 relative; a second,
                                (more at larger grids for that stage)    single-precision FFTW build
==============================  =======================================  ========================================

One grid is ``8 Nmesh^3`` bytes summed over all ranks: 57 MB at :math:`192^3`, 0.46 GB at
:math:`384^3`, 8.6 GB at :math:`1024^3`. The Python config writers set ``ModeTables`` and
``BinnedPNGCacheBlockSize`` to ``'auto'`` by default: the fast values up to ``Nmesh = 512`` and the
lean ones above. Set them explicitly when you know your memory allows more (see
`Speed and memory <conventions.html#speed-and-memory>`_).

* **ModeTables, BinnedPNGCacheBlockSize**: see
  `Speed and memory <conventions.html#speed-and-memory>`_.
* **FFTBackend**: ``band`` does the FFTs of the non-Gaussian term with the code's own slab FFT,
  which only transforms and communicates the sampled band of k-space and shares it evenly between
  the ranks. The default stays ``fftw`` (FFTW3's MPI transforms). See
  `FFT backends <conventions.html#fft-backends>`_.
* **FFTWWisdomDir**: keep the default ``auto`` (or a directory of your own), not ``none``: measured
  FFT plans are faster, and they are kept on disk so that only the first run with a given grid
  size, rank count and CPU type pays for measuring them. If many jobs run the same setup, point
  them all at one directory. See `FFT plans <conventions.html#fft-plans>`_.
* **FFTW build**: build FFTW3 with its SIMD kernels (``--enable-sse2 --enable-avx --enable-avx2``,
  and ``--enable-avx512`` where available), as in :doc:`installation`.
* **Aarambam_NATIVE**: see :doc:`installation`. Only worth it when all the nodes you run on have
  the same CPU type.
* **NGPrecision**: see `Single precision`_ below.

``2LPTBinned`` also skips, automatically, every pair of bins in which one filtered field is zero on the grid:
kernel bins below the box's fundamental mode or above the sampled k range have no modes. The result is
unchanged (those pairs add exactly zero); the log says how many pairs were skipped. With the default kernel
table, ``Lbox 512`` and ``Nsample 128`` that is 57 of the 210 pairs, and the binned run at :math:`192^3` on 8
ranks takes 2.68 s instead of 3.25 s.

Single precision
----------------

``NGPrecision float`` runs the non-Gaussian stage's real-space fields, their products and their
FFTs in single precision (always with the band FFT), and keeps everything else in double: the
Gaussian potential, the k-space mode recursion of the basis method, the accumulated non-Gaussian
term and the 2LPT stage. The default is ``double``. Measured against ``double`` (tests at
:math:`32^3`-:math:`48^3`, and a standalone copy of the binned stage at :math:`192^3` and
:math:`384^3`):

* the non-Gaussian term agrees to 3-4e-7 (binned) and up to 1.3e-6 (basis, whose sum over mode
  pairs cancels more) relative rms; the Gaussian part is unchanged. For :math:`f_{\rm NL} = 100`
  the term is about 1% of the potential, so the potential changes by ~1e-8 relative, below the
  float32 rounding of the particle positions in the snapshot.
* wall times at :math:`192^3` / ``Nsample 128`` (fastest settings otherwise):

  =====================  ==============  ==============
  run                    double          float
  =====================  ==============  ==============
  Binned, 8 ranks        3.27 s          2.49 s (-24%)
  Basis, 8 ranks         2.26 s          2.05 s (-9%)
  Binned, 48 ranks       1.49 s          1.27 s (-15%)
  Basis, 48 ranks        1.06 s          no gain
  =====================  ==============  ==============

  The stage itself is about 40% faster (45% at 48 ranks, 41% at :math:`384^3`); the rest of a run
  (Gaussian setup, 2LPT, start-up) does not change, so the gain is largest where the non-Gaussian
  stage dominates: the binned method, few ranks, large grids.

It needs FFTW built a second time with ``--enable-float`` (see :doc:`installation`). The
single-precision FFT plans are kept next to the double ones (``fftw3f_wisdom_*.txt`` in
``FFTWWisdomDir``).

Number of ranks
---------------

The grid is split between the ranks in slabs of ``Nmesh`` planes, so at most ``Nmesh`` ranks have
work. Use a rank count that divides ``Nmesh``, so that all slabs are the same size. Time is spent
mostly in FFTs, products of real-space fields and the all-to-all communication of the FFTs. On one
node, run one rank per core (``OMP_NUM_THREADS=1``; the 2LPT executables do not use threads).
