"""Oscillatory (resonance and feature) templates of Planck 2018 IX (arXiv:1905.05697)."""
import numpy as np
from .base import ShapeTemplate

__all__ = ['LogResonance', 'LinearResonance', 'K2cosResonance', 'KsinResonance', 'NBDsin']
class LogResonance(ShapeTemplate):
    """
    Logarithmic oscillations in the bispectrum, i.e. oscillations in :math:`\\log k`.
    See https://arxiv.org/pdf/1905.05697, Equation (12)
    
    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    - w_res : float
        Frequency of the oscillations in the bispectrum.
    - phi : float
        The phase of the oscillations.
    """

    normalise = None #Amplitude convention of the cited template, not S(1,1,1) = 1

    def raw_shape(self, k1, k2, k3): #https://arxiv.org/pdf/1905.05697, Equation (12)
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        w, phi = self.args['w_res'], self.args['phi']
        return np.sin(w * np.log(k1 + k2 + k3) + phi) * 1/6


class LinearResonance(ShapeTemplate):
    """
    Linear oscillations in the bispectrum, i.e. oscillations in k.
    See https://arxiv.org/pdf/1905.05697, Equation (15)
    
    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    - w_res : float
        Frequency of the oscillations in the bispectrum.
    - phi : float
        The phase of the oscillations.
    """

    normalise = None #Amplitude convention of the cited template, not S(1,1,1) = 1

    def raw_shape(self, k1, k2, k3): #https://arxiv.org/pdf/1905.05697, Equation (15)
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        w, phi = self.args['w_res'], self.args['phi']
        return np.sin(w * (k1 + k2 + k3) + phi) * 1/6


class K2cosResonance(ShapeTemplate):

    """
    The oscillatory bispectra from Adshead+ 2012, Eq(16) in https://arxiv.org/pdf/1905.05697.
    Fairly similar to the linear resonance model but with a exponential dampening.
    
    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    - w_res : float
        Frequency of the oscillations in the bispectrum.
    - alpha : float
        Damping parameter: the template is multiplied by
        :math:`\\alpha\\omega / (K \\sinh(\\alpha\\omega K))` with :math:`K = k_1 + k_2 + k_3`.
    """

    normalise = None #Amplitude convention of the cited template, not S(1,1,1) = 1

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        w, alpha = self.args['w_res'], self.args['alpha']
        K = k1 + k2 + k3
        return K**2 * ( (alpha*w) / (K * np.sinh(alpha*w*K))) * np.cos(w*K) * 1/6


class KsinResonance(ShapeTemplate):

    """
    A different oscillatory bispectra from Adshead+ 2012, Eq(17) in https://arxiv.org/pdf/1905.05697.
    A complement to the :math:`K^2\\cos` model provided in `Aarambam`.
    
    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    - w_res : float
        Frequency of the oscillations in the bispectrum.
    - alpha : float
        Damping parameter: the template is multiplied by
        :math:`\\alpha\\omega / (K \\sinh(\\alpha\\omega K))` with :math:`K = k_1 + k_2 + k_3`.
    """

    normalise = None #Amplitude convention of the cited template, not S(1,1,1) = 1

    def raw_shape(self, k1, k2, k3): #Eq(17) in https://arxiv.org/pdf/1905.05697
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        w, alpha = self.args['w_res'], self.args['alpha']
        K = k1 + k2 + k3
        return K * ( (alpha*w) / (K * np.sinh(alpha*w*K))) * np.sin(w*K) * 1/6


class NBDsin(ShapeTemplate):
    """
    Oscillatory bispectra caused by excited initial states, Non-Bunch Davies (NBD) vacua
    
    This object must be used with the `utils.Decomposer` class. That class instance
    should be provided with the parameters necessary to compute this model (mass, spin, etc.)

    Parameters
    ----------
    - n_s : float
        Scalar spectral index used to rescale each wavenumber by
        :math:`k^{(4-n_s)/3}` for near scale invariance.
    - w_res : float
        Frequency of the oscillations in the bispectrum.
    - phi : float
        The phase of the oscillations.
    """

    normalise = None #Amplitude convention of the cited template, not S(1,1,1) = 1

    def raw_shape(self, k1, k2, k3): #Eq(21) in https://arxiv.org/pdf/1502.01592
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        K  = k1 + k2 + k3
        w, phi = self.args['w_res'], self.args['phi']
        S1 = np.exp(-w*(K - 2*k1)) + np.exp(-w*(K - 2*k2)) + np.exp(-w*(K - 2*k3))
        S2 = np.sin(w*K + phi)
        return S1 * S2 * 1/6
