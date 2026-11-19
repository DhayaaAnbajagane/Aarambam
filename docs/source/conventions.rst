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

Size limits
-----------

The 2LPT executables use FFTW2 and the Gadget-1 snapshot format, which limit some sizes to
32-bit integers. They check these limits at startup and stop with a message instead of
silently overflowing:

* Each MPI rank's FFT slab, ``Local_nx * Nmesh * 2 (Nmesh/2 + 1)`` reals, must stay below
  :math:`2^{31}` (FFTW2's own ``int`` sizes). Use more ranks for large ``Nmesh``.
* Each rank writes one snapshot file, whose position block (12 bytes per particle) must fit a
  4-byte block marker, i.e. fewer than about 179 million particles per rank.
* Particle IDs are 32-bit by default, so at most :math:`2^{31}-1` particles in total
  (about :math:`1290^3`). Build with ``Aarambam_64BIT_IDS=ON`` (see :doc:`installation`) to write
  8-byte IDs; the code reading the snapshots must then expect 64-bit IDs.

Grid indices and the particle gather are 64-bit, so these are the only limits.
