Running faster
==============

The 2LPT executables have a few options that make them faster at the cost of memory or of an extra
build step. None changes the ICs beyond rounding. :doc:`conventions` says what each does.

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

with ``FFTWWisdomDir`` left at its default. On one 48-core node, the binned method at
:math:`384^3` / ``Nsample 256`` on 32 ranks then takes 10.9 s, against 21.4 s with ``ModeTables 0``
and the ``fftw`` backend, and 43.4 s for version 1.0.0.

If a relative precision of about :math:`10^{-6}` on the non-Gaussian term is enough, add
``NGPrecision float``: binned 24% and basis 9% faster at :math:`192^3` on 8 ranks, with half the
memory for that stage (see `Single precision`_).

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
``BinnedPNGMultiGrid 1``        binned 14-21% (8 ranks), 32-40%          on by default; the first run measures a
                                (1 rank) at 192^3                        few more FFT plans; NG term to rounding,
                                                                         not bit for bit
==============================  =======================================  ========================================

One grid is ``8 Nmesh^3`` bytes summed over all ranks (57 MB at :math:`192^3`, 0.46 GB at
:math:`384^3`). The Python config writers default ``ModeTables`` and ``BinnedPNGCacheBlockSize`` to
``'auto'``: the fast values up to ``Nmesh = 512``, the lean ones above. Set them explicitly when your
memory allows more. Build FFTW with its SIMD kernels (see :doc:`installation`), and keep the FFT plan
files: if many jobs run the same setup, point them all at one ``FFTWWisdomDir``.

Some speed-ups need no option: ``2LPTBinned`` skips the bin pairs that cannot contribute (57 of 210
with the default table, ``Lbox 512`` and ``Nsample 128``: 2.68 s instead of 3.25 s at :math:`192^3`
on 8 ranks), the Gaussian draws are shared evenly between the ranks, the wisdom files are rewritten
only when new plans were measured, and particle positions are wrapped into the box without glibc's
slow ``fmod`` (the last three together: 1.59 s to 1.49 s at :math:`192^3` on 8 ranks, binned, float,
same ICs bit for bit).

Timings
-------

On one 48-core node (Xeon Gold 6248R), with the default kernels (``N_modes = 15``, 20 binned bins),
``Nmesh = 3/2 Nsample``, FFTW 3.3.10 built with SIMD and the FFT plans read from a wisdom file:

=========================  ===============  ==================  ==================
run                        version 1.0.0    ``ModeTables 0``    ``ModeTables 1``
=========================  ===============  ==================  ==================
Basis  192/128, 8 ranks    14.7 s           7.3 s               2.8 s
Basis  384/256, 32 ranks   35.1 s           17.8 s              7.8 s
Binned 192/128, 8 ranks    16.6 s           7.5 s               4.4 s
Binned 384/256, 32 ranks   43.4 s           21.4 s              14.5 s
=========================  ===============  ==================  ==================

(Version 1.0.0 used FFTW2. The binned ``ModeTables 1`` column uses the full cache block, the
``ModeTables 0`` column a block of 4.) The rest of the time is mostly FFTs, part of it waiting: with
``Nmesh = 3/2 Nsample`` a third of the slabs hold no sampled modes. The ``band`` backend removes
most of that wait:

==========================================  ========  ========  =====
run                                         ``fftw``  ``band``  saved
==========================================  ========  ========  =====
Basis  192/128, 8 ranks, ``ModeTables 0``   7.4 s     6.3 s     15%
Basis  192/128, 8 ranks, ``ModeTables 1``   2.8 s     2.4 s     14%
Basis  384/256, 32 ranks, ``ModeTables 0``  17.8 s    15.3 s    14%
Basis  384/256, 32 ranks, ``ModeTables 1``  7.8 s     6.5 s     16%
Binned 192/128, 8 ranks, ``ModeTables 0``   7.4 s     5.7 s     23%
Binned 192/128, 8 ranks, ``ModeTables 1``   4.3 s     3.4 s     22%
Binned 384/256, 32 ranks, ``ModeTables 0``  21.4 s    16.2 s    24%
Binned 384/256, 32 ranks, ``ModeTables 1``  14.5 s    10.9 s    25%
==========================================  ========  ========  =====

These are single-node timings up to :math:`384^3`; across nodes the band backend sends less data
than FFTW's transforms, but it has not been timed there.

Building with ``-DAARAMBAM_PROFILE`` in the C flags makes both executables print, at the end of a
run, the time spent in FFTs, field building, products, accumulation and barriers (minimum and
maximum over ranks).

Multi-grid binned stage
-----------------------

``BinnedPNGMultiGrid 1`` (the default) computes each pair of bins on the smallest grid on which it is
exact (see `Speed and memory <conventions.html#speed-and-memory>`_). Low-k bins hold few modes, so
most pairs need far less than the full grid: with the default table at :math:`192^3` /
``Nsample 128``, 64 of the 153 pairs run on grids of at most 108 points per side, 68 on 140 or 168,
and 21 on the full 192. Wall times (fastest settings otherwise):

=====================  ==============  ==============
setup                  multi-grid off  multi-grid on
=====================  ==============  ==============
1 rank, double         15.2 s          9.1 s
1 rank, float          11.0 s          7.5 s
8 ranks, double        2.60 s          2.05 s
8 ranks, float         2.04 s          1.76 s
=====================  ==============  ==============

The non-Gaussian term agrees with ``BinnedPNGMultiGrid 0`` to :math:`\sim 10^{-14}` relative in
double. In single precision it is closer to the double result than the full-grid float path (2.5e-7
against 1.3e-6 relative rms), because the smaller grids accumulate less rounding. The gain grows with
``Nmesh / Nsample`` and with the grid.

Single precision
----------------

``NGPrecision float`` runs the non-Gaussian stage's real-space fields, their products and their
FFTs in single precision (always with the band FFT). The Gaussian potential, the modal method's
k-space mode recursion, the accumulated non-Gaussian term and the 2LPT stage stay double.

* The non-Gaussian term agrees with ``double`` to 3-4e-7 (binned) and up to 1.3e-6 (basis, whose sum
  over mode pairs cancels more) relative rms. For :math:`f_{\rm NL} = 100` the term is about 1% of
  the potential, so the potential changes by ~1e-8 relative, below the float32 rounding of the
  particle positions in the snapshot.
* Wall times at :math:`192^3` / ``Nsample 128`` (fastest settings otherwise):

  =====================  ==============  ==============
  run                    double          float
  =====================  ==============  ==============
  Binned, 8 ranks        3.27 s          2.49 s (-24%)
  Basis, 8 ranks         2.26 s          2.05 s (-9%)
  Binned, 48 ranks       1.49 s          1.27 s (-15%)
  Basis, 48 ranks        1.06 s          no gain
  =====================  ==============  ==============

  The stage itself is about 40% faster; the rest of a run does not change, so the gain is largest
  where the non-Gaussian stage dominates: the binned method, few ranks, large grids.

It needs FFTW built a second time with ``--enable-float`` (see :doc:`installation`). The
single-precision plans are kept next to the double ones (``fftw3f_wisdom_*.txt``).

Number of ranks
---------------

The grid is split into slabs of ``Nmesh`` planes, so at most ``Nmesh`` ranks have work. Use a rank
count that divides ``Nmesh``, so that all slabs are the same size. On one node, run one rank per core
with ``OMP_NUM_THREADS=1`` (the 2LPT executables do not use threads).
