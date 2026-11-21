"""Base class for the bispectrum templates in ``Aarambam.models``."""
import numpy as np

__all__ = ['ShapeTemplate']


class ShapeTemplate:
    """
    Base class for bispectrum templates written as a shape function in the
    rescaled wavenumbers :math:`\\tilde k = k^{(4-n_s)/3}`.

    A template implements ``raw_shape(k1, k2, k3)``: one term of the shape
    (with its permutation weight 1/6, 3/6 or 6/6), evaluated on the rescaled
    wavenumbers. The decomposition classes sum ``raw_bispectrum`` over the six
    permutations of :math:`(k_1, k_2, k_3)`, and this class supplies

    .. math::

        B_{\\rm raw}(k_1, k_2, k_3) = \\frac{S_{\\rm raw}(\\tilde k_1, \\tilde k_2, \\tilde k_3)}
                                       {N \\, (\\tilde k_1 \\tilde k_2 \\tilde k_3)^2},

    so that :math:`B \\propto P(k)^2` scaling holds for any :math:`n_s`. With
    ``normalise = 'equilateral'`` (the default), :math:`N` is the symmetrised
    shape at :math:`k_1 = k_2 = k_3 = 1`, i.e. the template satisfies
    :math:`S(1, 1, 1) = 1`. Templates with their own normalisation convention
    set ``normalise = None`` (then :math:`N = 1`).

    Templates are not instantiated directly: pass the class to ``Decomposer.go``,
    together with its parameters as keywords (e.g. ``mass = 1``). They are read
    from ``self.args``, which also holds ``n_s``.
    """

    normalise = 'equilateral'

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape, with its permutation weight; subclasses implement it.

        Parameters
        ----------
        k1, k2, k3 : ndarray
            Rescaled wavenumbers :math:`\\tilde k`.
        """
        raise NotImplementedError("A ShapeTemplate must implement raw_shape(k1, k2, k3)")

    def shape_normalisation(self):
        """Symmetrised shape at k1 = k2 = k3 = 1 (all six permutations are equal there)."""
        if self.normalise is None:
            return 1.0
        if self.normalise != 'equilateral':
            raise ValueError(f"normalise must be 'equilateral' or None, not {self.normalise!r}")
        one = np.ones(1)
        return 6 * float(self.raw_shape(one, one, one)[0])

    def raw_bispectrum(self, k1, k2, k3):
        """One permutation term of the bispectrum at wavenumbers ``k1, k2, k3`` (h/Mpc):
        ``raw_shape`` of the rescaled k over :math:`N (\\tilde k_1 \\tilde k_2 \\tilde k_3)^2`."""

        p  = (4 - self.args['n_s'])/3
        k1 = np.power(k1, p)
        k2 = np.power(k2, p)
        k3 = np.power(k3, p)

        return self.raw_shape(k1, k2, k3) / self.shape_normalisation() / (k1 * k2 * k3)**2
