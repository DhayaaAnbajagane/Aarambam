"""Single-field templates: local, equilateral, orthogonal, quasi-single-field, DBI and the EFT shapes."""
import numpy as np
from scipy import special
from .base import ShapeTemplate

__all__ = ['Local', 'Equilateral', 'Orthogonal', 'QuasiSingleField', 'DBI', 'SenatoreEFT1', 'SenatoreEFT2']


class Local(ShapeTemplate):
    """Local non-Gaussianity. No parameters."""

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        return k1**2/(k2*k3) * 3/6


class Equilateral(ShapeTemplate):
    """Equilateral template. No parameters."""

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        S_1 = k1**2/(k2*k3) * 3/6
        S_2 = 1 * 1/6
        S_3 = k3/k1 * 6/6
        return (-1) * S_1 + (-2) * S_2 + 1 * S_3


class Orthogonal(ShapeTemplate):
    """Orthogonal template (the standard one of Senatore et al. 2010, not the LSS-optimised version of their
    appendix). No parameters."""

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        S_1 = k1**2/(k2*k3) * 3/6
        S_2 = 1 * 1/6
        S_3 = k3/k1 * 6/6
        return (-3) * S_1 + (-8) * S_2 + 3 * S_3


class QuasiSingleField(ShapeTemplate):
    """
    Quasi-single-field template (Sohn+ 2024, arXiv:2404.07203, Eq. 2.6).

    Parameters
    ----------
    mass : float
        Order :math:`\\nu = \\sqrt{9/4 - (m/H)^2}` of the Bessel function, in [0, 3/2). Larger is more squeezed.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        nu    = self.args['mass']
        kappa = k1*k2*k3/(k1 + k2 + k3)**3
        return np.sqrt(kappa) * special.yv(nu, 8 * kappa) * 1/6


class DBI(ShapeTemplate):
    """DBI template (arXiv:1303.5084, Eq. 7). No parameters."""

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        S1 = -3/7 / np.power(k1 + k2 + k3, 2)
        S2 = k1**5 * 3/6
        S3 = (2*k1**4*k2 - 3*k1**3*k2**2) * 6/6
        S4 = (k1**3*k2*k3 - 4*k1**2*k2**2*k3) * 6/6
        return S1 * (S2 + S3 + S4) / (k1*k2*k3)


class SenatoreEFT1(ShapeTemplate):
    """EFT template of the first cubic inflaton interaction (arXiv:1303.5084, Eq. 5). No parameters."""

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        S1 = -9/17 / np.power(k1 + k2 + k3, 3)
        S2 = k1**6 * 3/6
        S3 = (3*k1**5*k2 - k1**4*k2**2 - 3*k1**3*k2**3) * 6/6
        S4 = (3*k1**4*k2*k3 - 9*k1**3*k2**2*k3 - 2*np.power(k1*k2*k3, 2)) *  6/6
        return S1 * (S2 + S3 + S4) / (k1*k2*k3)


class SenatoreEFT2(ShapeTemplate):
    """EFT template of the second cubic inflaton interaction (arXiv:1303.5084, Eq. 6). No parameters."""

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        return 27 * (k1*k2*k3) / np.power(k1 + k2 + k3, 3) * 1/6 #Fully symmetric, so 1/6 for the permutation sum
