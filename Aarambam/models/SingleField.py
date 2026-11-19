"""Single-field templates: local, equilateral, orthogonal, quasi-single-field, DBI and the EFT shapes."""
import numpy as np
from scipy import special
from .base import ShapeTemplate

__all__ = ['Local', 'Equilateral', 'Orthogonal', 'QuasiSingleField', 'DBI', 'SenatoreEFT1', 'SenatoreEFT2']
class Local(ShapeTemplate):
    """
    The standard local type non-Gaussianity.

    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        return k1**2/(k2*k3) * 3/6


class Equilateral(ShapeTemplate):

    """
    The standard equilateral type non-Gaussianity.

    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        S_1 = k1**2/(k2*k3) * 3/6
        S_2 = 1 * 1/6
        S_3 = k3/k1 * 6/6
        return (-1) * S_1 + (-2) * S_2 + 1 * S_3


class Orthogonal(ShapeTemplate):
    
    """
    The standard orthogonal type non-Gaussianity. This is the simpler template in Senatore++
    which is actively used in LSS analyses, and not the updated on in their appendix that is
    better for LSS work.

    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        S_1 = k1**2/(k2*k3) * 3/6
        S_2 = 1 * 1/6
        S_3 = k3/k1 * 6/6
        return (-3) * S_1 + (-8) * S_2 + 3 * S_3


class QuasiSingleField(ShapeTemplate):
    
    """
    The Quasi Single Field model. See, for example, Equation 2.6 in
    Sohn+ 2024 (https://arxiv.org/abs/2404.07203).

    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    - mass : float
        The order :math:`\\nu` of the Bessel function :math:`Y_\\nu` in the template, with
        :math:`\\nu = \\sqrt{9/4 - (m/H)^2}` in :math:`[0, 3/2)` (Sohn+ 2024, below Eq. 2.6).
        Larger :math:`\\nu` (lighter field) gives a more squeezed shape.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        nu    = self.args['mass']
        kappa = k1*k2*k3/(k1 + k2 + k3)**3
        return np.sqrt(kappa) * special.yv(nu, 8 * kappa) * 1/6


class DBI(ShapeTemplate):

    """
    The Dirac-Born-Infeld (DBI) model. See, for example, Eq(7) in https://arxiv.org/pdf/1303.5084.
    There are no additional parameter in this model.

    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        S1 = -3/7 / np.power(k1 + k2 + k3, 2)
        S2 = k1**5 * 3/6
        S3 = (2*k1**4*k2 - 3*k1**3*k2**2) * 6/6
        S4 = (k1**3*k2*k3 - 4*k1**2*k2**2*k3) * 6/6
        return S1 * (S2 + S3 + S4) / (k1*k2*k3)


class SenatoreEFT1(ShapeTemplate):
    """
    The EFT-based template for the first cubic (self)-interaction of the inflaton.
    See Eq(5) in https://arxiv.org/pdf/1303.5084. The equilateral and orthogonal
    templates are linear combinations of this template with ``SenatoreEFT2``.

    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        S1 = -9/17 / np.power(k1 + k2 + k3, 3)
        S2 = k1**6 * 3/6
        S3 = (3*k1**5*k2 - k1**4*k2**2 - 3*k1**3*k2**3) * 6/6
        S4 = (3*k1**4*k2*k3 - 9*k1**3*k2**2*k3 - 2*np.power(k1*k2*k3, 2)) *  6/6
        return S1 * (S2 + S3 + S4) / (k1*k2*k3)


class SenatoreEFT2(ShapeTemplate):
    """
    The EFT-based template for the second cubic (self)-interaction of the inflaton.
    See Eq(6) in https://arxiv.org/pdf/1303.5084. The equilateral and orthogonal
    templates are linear combinations of this template with ``SenatoreEFT1``.

    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        return 27 * (k1*k2*k3) / np.power(k1 + k2 + k3, 3) * 1/6 #Fully symmetric, so 1/6 for the permutation sum
