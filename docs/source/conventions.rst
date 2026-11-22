Conventions
===========

Things that are easy to get wrong when adding templates or setting up runs. Timings are on
:doc:`performance`.

Template normalisation
----------------------

A template implements ``raw_shape(k1, k2, k3)`` (see ``Aarambam.models.ShapeTemplate``), or
``raw_bispectrum`` or ``bispectrum`` directly. The decompositions sum ``raw_bispectrum`` over the 6
permutations of ``(k1, k2, k3)``, so each term carries a weight of ``1/6`` (fully symmetric), ``3/6``
(3 distinct permutations) or ``6/6`` (6 distinct permutations). The shipped templates satisfy
:math:`S(k, k, k) = 1` in the equilateral limit, where :math:`B = S / (k_1 k_2 k_3)^2` and
:math:`k \to k^{(4-n_s)/3}` includes the tilt. The resonance templates keep the amplitude convention
of their papers instead.

Amplitude in the simulations
----------------------------

The 2LPT executables multiply the decomposed template by ``Fnl`` such that the primordial potential
has

.. math::

   B_\Phi(k_1, k_2, k_3) = 6 \, f_{\rm NL} \, B_{\rm template}(k_1, k_2, k_3).

For the ``Local`` template, whose normalisation carries a factor :math:`1/3`, this is the standard
:math:`2 f_{\rm NL} [P_\Phi(k_1) P_\Phi(k_2) + 2\ {\rm perms}]`. Keep the factor of 6 in mind when
comparing measured bispectra (or halo bias) with predictions computed directly from a template.

Grid sizes
----------

* ``Nmesh`` must be even and ``Nsample`` at most ``Nmesh``; the executables check both.
* ``Nmax`` (in units of the fundamental mode :math:`2\pi/L_{\rm box}`) is the largest wavenumber of
  the decomposition. The executables stop with an error unless ``Nmax >= Nsample``.
* The non-Gaussian term is built from products of fields band-limited to ``Nsample/2``.
  ``Nmesh >= 3/2 Nsample`` keeps the aliased power of those products out of the sampled modes;
  ``Nmesh = Nsample`` runs but is not alias-free.

Speed and memory
----------------

These tags trade memory for speed. None changes the ICs beyond rounding. The Python config writers
set ``ModeTables`` and ``BinnedPNGCacheBlockSize`` to ``'auto'`` by default.

* ``ModeTables`` (both executables; 0 when absent from a config file)

  * ``0`` computes every per-mode quantity on the fly: least memory.
  * ``1`` precomputes them. ``2LPTBinned`` tabulates :math:`|k|`, the output bin and the in-bin
    coordinate of every sampled mode (about one extra grid; same result to rounding). ``2LPTBasis`` also
    keeps the ``N_modes`` mode fields in real space instead of rebuilding them for every mode pair
    (half the FFTs, about ``N_modes`` extra grids, same result to :math:`\sim 10^{-16}`).
  * ``'auto'``: 1 for the binned method; for the modal one, 1 up to ``Nmesh = 512``
    (``AUTO_FAST_MAX_NMESH``) and 0 above.

* ``BinnedPNGCacheBlockSize`` (binned): how many filtered real-space fields are kept at once, about
  that many extra grids. It is capped at the number of :math:`k_1` bins and does not change the
  result. ``'auto'``: 64 (all bins) up to ``Nmesh = 512``, 4 above.

* ``BinnedPNGMultiGrid`` (binned, default 1): computes each pair of :math:`k_1, k_2` bins on the
  smallest grid on which it is exact. A filtered field has modes up to its bin's upper edge
  :math:`K_a`, so a product of two has modes up to :math:`K_a + K_b`, and its modes up to
  :math:`K_{\rm out}` are exact on any grid of :math:`L > K_a + K_b + K_{\rm out}` points per side
  (the pair adds nothing above :math:`K_{\rm out}`). Pairs with :math:`L \le 128` run whole on one
  rank with serial FFTs, larger ones on all ranks with a band FFT of size :math:`L` (unless the fields
  of that size would need more memory than two cache blocks of ``BinnedPNGCacheBlockSize`` full grids);
  the rest stay on the full grid. The non-Gaussian term agrees with ``0`` (every pair on the full grid) to
  :math:`\sim 10^{-14}` relative. With it on, the result depends on the number of ranks at that
  level rather than not at all, because the small-grid pairs are summed in a rank-dependent order
  (the Gaussian potential is the same bit for bit on any number of ranks). Every rank keeps tables of the
  whole sampled band, about :math:`25\,N_{\rm sample}^3` bytes (0.4 GB at ``Nsample 256``, 3.4 GB at 512);
  when those of all the ranks on a node would take more than a quarter of its memory, the run says so and
  puts every pair on the full grid.

``2LPTBinned`` also skips every pair of bins in which one filtered field has no modes on the grid
(bins below the fundamental mode or above the sampled range); the log says how many.

The internal legs :math:`k_1` and :math:`k_2` share their bins (``FileWithBinnedK1Edges``); the
output leg :math:`k_3` has its own (``FileWithBinnedK3Edges``). The edges in both files must increase
strictly. Older parameter files may contain ``FileWithBinnedK2Edges`` and
``BinnedPNGUseInternalSymmetry``; they are accepted and ignored.

One grid is ``8 Nmesh^3`` bytes summed over all ranks: 57 MB at :math:`192^3`, 0.46 GB at
:math:`384^3`, 8.6 GB at :math:`1024^3`. Where memory is tight, use ``ModeTables 0`` and a small
block size; otherwise ``ModeTables 1`` is faster at every size.

FFT plans
~~~~~~~~~

FFTW can measure which FFT algorithm is fastest for a grid size and rank count (``FFTW_MEASURE``)
and store the result as "wisdom". The first run with a given ``Nmesh``, rank count and CPU model
measures its plans (about 2 s at :math:`192^3`, 30 s at :math:`1024^3`) and saves them, one file per
CPU model; later runs read them back. ``FFTWWisdomDir`` (both executables) says where:

* a directory: created if needed; the run stops with an error if it cannot be created or written.
* ``auto`` (the default, also when the tag is absent): ``$XDG_CACHE_HOME/aarambam/`` or
  ``~/.cache/aarambam/``, or ``$AARAMBAM_FFTW_WISDOM_DIR`` when set (a directory, or ``none``). If
  the default directory cannot be created, the run uses ``FFTW_ESTIMATE``.
* ``none``: ``FFTW_ESTIMATE`` plans, no measuring and no file.

Measured plans make each FFT 7-27% faster. They can differ between runs and rank counts, so ICs
made with them agree with ``none`` (and with each other) to :math:`\sim 10^{-16}` rather than bit for
bit; runs that read the same file use the same plans. A run whose plans are all in the file neither
measures nor rewrites it, so many runs can share one directory; new plans are merged in and the file
is written atomically. The log says whether the plans were measured or read.

FFT backends
~~~~~~~~~~~~

The non-Gaussian term needs many FFTs of which only the sampled band of k-space matters
(:math:`|k_x|, |k_y|, k_z \le` ``Nsample/2 + 1``). ``FFTBackend`` (both executables) chooses how
they are done:

* ``fftw`` (the default): FFTW3's MPI transforms of the whole grid.
* ``band``: the code's own slab FFT (``src/lpt/common/band_fft.c``), built on FFTW3's serial
  transforms and one ``MPI_Alltoallv``. It transforms and communicates only the band (4/9 of the
  grid for ``Nmesh = 3/2 Nsample``) and gives every rank an equal share of it. It keeps a buffer of
  about :math:`(N_{\rm sample}/N_{\rm mesh})^2` of a grid per rank.

Both give the same ICs to rounding. ``2LPTFFTCheck``
(installed next to the executables) compares them on the ranks it is started on, e.g.
``mpirun -np 2 2LPTFFTCheck 48 32`` (grid size, ``Nsample``).

Single precision
~~~~~~~~~~~~~~~~

``NGPrecision float`` (both executables; default ``double``) computes the non-Gaussian stage's
real-space fields, products and FFTs in single precision. See :doc:`performance` for what stays
double, the precision (~4e-7 binned, ~1e-6 basis, relative) and the timings.

Size limits
-----------

Some sizes in the executables and in the Gadget-1 snapshot format are 32-bit integers. The
executables check these limits at startup and stop with a message:

* Each rank's grid, about ``Local_nx * Nmesh * 2 (Nmesh/2 + 1)`` reals, must stay below
  :math:`2^{31}` (16 GB per grid per rank). Use more ranks for large ``Nmesh``.
* There is one snapshot file per rank that holds particles (``FileBase.0``, ``FileBase.1``, ..., in
  rank order; just ``FileBase`` on one rank). Each file's position block (12 bytes per particle) must
  fit a 4-byte block marker: fewer than about 179 million particles per rank.
* Particle IDs are 32-bit by default: at most :math:`2^{31}-1` particles (about :math:`1290^3`).
  Build with ``Aarambam_64BIT_IDS=ON`` (see :doc:`installation`) for 8-byte IDs; the code reading
  the snapshots must then expect them.
