Conventions
===========

A few conventions that are easy to get wrong when adding models or setting up runs.

Template normalisation
----------------------

Models implement either ``raw_bispectrum(k1, k2, k3)`` or ``bispectrum(k1, k2, k3)``.
A ``raw_bispectrum`` is summed over all 6 permutations of ``(k1, k2, k3)`` by the
decomposition classes, so each term carries a weight of ``1/6`` (fully symmetric
term), ``3/6`` (term with 3 distinct permutations) or ``6/6`` (term with 6 distinct
permutations). The shipped templates are normalised so that the symmetrised shape
satisfies :math:`S(k, k, k) = 1` in the equilateral limit, where
:math:`B = S / (k_1 k_2 k_3)^2` (with :math:`k \to k^{(4-n_s)/3}` to include the tilt).

Amplitude in the simulations
----------------------------

The 2LPT executables multiply the decomposed template by ``Fnl`` such that the
primordial potential has

.. math::

   B_\Phi(k_1, k_2, k_3) = 6 \, f_{\rm NL} \, B_{\rm template}(k_1, k_2, k_3).

For the ``Local`` template, whose normalisation carries a factor :math:`1/3`, this
reduces to the standard :math:`2 f_{\rm NL} [P_\Phi(k_1) P_\Phi(k_2) + 2\ {\rm perms}]`.
Keep the factor of 6 in mind when comparing measured bispectra (or halo bias) with
theory predictions computed directly from a template.

Grid sizes
----------

* ``Nmax`` (in units of the fundamental mode :math:`2\pi/L_{\rm box}`) sets the largest
  wavenumber of the decomposition. The executables assert ``Nmax >= Nsample``.
* The non-Gaussian term is built from products of fields that are band-limited to
  ``Nsample/2``. Using ``Nmesh >= 3/2 Nsample`` keeps the aliased power of those
  products (and the Nyquist planes) out of the sampled modes; ``Nmesh = Nsample`` runs
  but is not alias-free.

Speed and memory
----------------

Two config tags trade memory for speed. Neither changes the ICs beyond rounding, and the
Python config writers set both to ``'auto'`` by default.

* ``ModeTables`` (both executables, default 0 when absent from a config file).

  * ``0`` computes every per-mode quantity on the fly and uses the least memory.
  * ``1`` precomputes them once.

    * ``2LPTBinned`` tabulates |k|, the output bin and the in-bin coordinate of every sampled
      mode (about one extra grid at ``Nmesh = 3/2 Nsample``). The result is bit-identical.
    * ``2LPTBasis`` additionally builds each of the ``N_modes`` mode fields once and keeps
      them in real space, instead of rebuilding and transforming one for every mode pair.
      This halves the FFTs, costs about ``N_modes`` extra grids, and agrees with
      ``ModeTables 0`` to rounding (:math:`\sim 10^{-16}` relative).

  * ``'auto'`` is 1 for the binned method, and 1 for the modal one when
    ``Nmesh <= AUTO_FAST_MAX_NMESH`` (512).

* ``BinnedPNGCacheBlockSize`` (binned only): how many filtered real-space fields are kept per
  block, about that many extra grids. The C code caps it at the number of :math:`k_1` bins. The
  result does not depend on it. ``'auto'`` is 64 (all bins) up to ``Nmesh = 512`` and 4 above.

One grid is ``8 Nmesh^3`` bytes summed over ranks: 57 MB at :math:`192^3`, 0.46 GB at
:math:`384^3`, and 8.6 GB at :math:`1024^3`. For a large run where memory is tight, set
``ModeTables 0`` and a small block size. Where memory allows, ``ModeTables 1`` is faster at
every size.

Wall times on one 48-core node (Xeon Gold 6248R), with the default kernels (``N_modes = 15``,
20 binned bins), ``Nmesh = 3/2 Nsample``, FFTW 3.3.10 built with SIMD (see :doc:`installation`) and
the FFT plans read from a wisdom file (see `FFT plans`_):

=========================  ===============  ==================  ==================
run                        before (FFTW2)   ``ModeTables 0``    ``ModeTables 1``
=========================  ===============  ==================  ==================
Basis  192/128, 8 ranks    14.7 s           7.3 s               2.8 s
Basis  384/256, 32 ranks   35.1 s           17.8 s              7.8 s
Binned 192/128, 8 ranks    16.6 s           7.5 s               4.4 s
Binned 384/256, 32 ranks   43.4 s           21.4 s              14.5 s
=========================  ===============  ==================  ==================

("Before" is the code of version 1.0.0, which used FFTW2. The binned ``ModeTables 1`` column uses the
full cache block, the binned ``ModeTables 0`` column a block of 4.) In every row the rest of the time
is mostly FFTs. Part of it is waiting: with ``Nmesh = 3/2 Nsample``, a third of the slabs hold no
sampled modes. The ``band`` FFT backend (see `FFT backends`_) removes most of that wait.

Building with ``-DAARAMBAM_PROFILE`` in the C flags makes both executables print the time spent in
FFTs, field building, products, accumulation and barriers (minimum and maximum over ranks) at the
end of a run.

FFT plans
~~~~~~~~~

FFTW can measure which FFT algorithm is fastest for a given grid size and number of ranks
(``FFTW_MEASURE``) and store the result as "wisdom". The first run with a given ``Nmesh``, number
of ranks and CPU model measures its plans, which takes about 2 s at :math:`192^3` and 30 s at
:math:`1024^3`, and saves them in a wisdom file, one per CPU model. Every later run with the same grid
and rank count reads them back and starts at once. The ``FFTWWisdomDir`` tag (both executables)
says where the files are kept:

* a directory path: that directory. It is created if needed; if it cannot be created or written,
  the run stops with an error.
* ``auto`` (the default, also when the tag is absent): ``$XDG_CACHE_HOME/aarambam/`` or
  ``~/.cache/aarambam/``, or the ``AARAMBAM_FFTW_WISDOM_DIR`` environment variable when it is set
  (a directory, or ``none``). If the default directory cannot be created, the run plans with
  ``FFTW_ESTIMATE`` instead.
* ``none``: plan with ``FFTW_ESTIMATE``, no measuring and no file.

Measured plans make each FFT 7% (at :math:`1024^3`) to 27% (at :math:`192^3`) faster. They can
differ from run to run and between rank counts, so ICs made with them agree with ``none`` (and
with each other) to rounding (:math:`\sim 10^{-16}`) rather than bit for bit; runs that reuse
the same wisdom file use the same plans. The log says whether the plans were measured or read.

Building with ``Aarambam_NATIVE=ON`` (see :doc:`installation`) compiles for the build machine's
CPU (AVX2 / AVX-512 and fused multiply-adds). That gains a few per cent at most, and the
executables then only run on CPUs with the same instruction sets.

FFT backends
~~~~~~~~~~~~

The non-Gaussian term needs many FFTs of grids of which only the sampled band of k-space matters
(:math:`|k_x|, |k_y|, k_z \le` ``Nsample/2 + 1`` in grid units). The ``FFTBackend`` tag (both
executables; ``make_config_*(..., FFTBackend = 'fftw' | 'band')``) chooses how they are done:

* ``fftw`` (the default, also when the tag is absent): FFTW3's MPI transforms of the whole grid.
* ``band``: the 2LPT code's own slab FFT (``src/lpt/common/band_fft.c``), built on FFTW3's serial
  transforms and one ``MPI_Alltoallv``. It transforms in y and x and communicates only the band,
  4/9 of the grid for ``Nmesh = 3/2 Nsample``, and gives every rank an equal share of the band, so
  no rank waits on the others. It keeps one communication buffer of about
  :math:`(N_{\rm sample}/N_{\rm mesh})^2` of a grid per rank (0.44 for ``Nmesh = 3/2 Nsample``).

Both give the same ICs to rounding. With ``FFTWWisdomDir none`` (``FFTW_ESTIMATE`` plans) they are
bit-for-bit identical, since both run the same serial FFTW kernels on every line of the grid; with
measured plans they agree to :math:`\sim 10^{-16}`. Run times on one 48-core node, with plans read
from a wisdom file:

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

(The binned rows use the cache blocks of the speed table above.) These are single-node timings at
up to :math:`384^3`; across nodes the band backend sends less data than FFTW's transforms, but it
has not been timed there, nor on larger grids. ``2LPTFFTCheck`` (installed next to the
executables) compares the two backends on the ranks it is started on, e.g.
``mpirun -np 2 2LPTFFTCheck 48 32``; the tests run it on 1 to 7 ranks.

Size limits
-----------

Some sizes in the 2LPT executables and in the Gadget-1 snapshot format are 32-bit integers. The
executables check these limits at startup and stop with a message instead of silently overflowing:

* Each MPI rank's grid, about ``Local_nx * Nmesh * 2 (Nmesh/2 + 1)`` reals, must stay below
  :math:`2^{31}` (the code's grid size and the potential files' header are ``int``), i.e. 16 GB
  per grid per rank. Use more ranks for large ``Nmesh``.
* Each rank writes one snapshot file, whose position block (12 bytes per particle) must fit a
  4-byte block marker, i.e. fewer than about 179 million particles per rank.
* Particle IDs are 32-bit by default, so at most :math:`2^{31}-1` particles in total
  (about :math:`1290^3`). Build with ``Aarambam_64BIT_IDS=ON`` (see :doc:`installation`) to write
  8-byte IDs; the code reading the snapshots must then expect 64-bit IDs.

Grid indices and the particle gather are 64-bit, so these are the only limits.
