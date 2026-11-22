"""Oscillatory (resonance and feature) templates of Planck 2018 IX (arXiv:1905.05697)."""
import numpy as np
from .base import ShapeTemplate

__all__ = ['LogResonance', 'LinearResonance', 'K2cosResonance', 'KsinResonance', 'NBDsin']


class LogResonance(ShapeTemplate):
    """
    Oscillations in log k (Planck 2018 IX, arXiv:1905.05697, Eq. 12).

    Parameters
    ----------
    w_res : float
        Frequency.
    phi : float
        Phase.
    """

    normalise = None #Amplitude convention of the cited template, not S(1,1,1) = 1

    def raw_shape(self, k1, k2, k3): #https://arxiv.org/pdf/1905.05697, Equation (12)
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        w, phi = self.args['w_res'], self.args['phi']
        return np.sin(w * np.log(k1 + k2 + k3) + phi) * 1/6


class LinearResonance(ShapeTemplate):
    """
    Oscillations in k (Planck 2018 IX, arXiv:1905.05697, Eq. 15).

    Parameters
    ----------
    w_res : float
        Frequency.
    phi : float
        Phase.
    """

    normalise = None #Amplitude convention of the cited template, not S(1,1,1) = 1

    def raw_shape(self, k1, k2, k3): #https://arxiv.org/pdf/1905.05697, Equation (15)
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        w, phi = self.args['w_res'], self.args['phi']
        return np.sin(w * (k1 + k2 + k3) + phi) * 1/6


class K2cosResonance(ShapeTemplate):
    """
    Damped :math:`K^2 \\cos` oscillations of Adshead+ 2012 (arXiv:1905.05697, Eq. 16).

    Parameters
    ----------
    w_res : float
        Frequency.
    alpha : float
        Damping: the template is multiplied by :math:`\\alpha\\omega / (K \\sinh(\\alpha\\omega K))`,
        :math:`K = k_1 + k_2 + k_3`.
    """

    normalise = None #Amplitude convention of the cited template, not S(1,1,1) = 1

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        w, alpha = self.args['w_res'], self.args['alpha']
        K = k1 + k2 + k3
        return K**2 * ( (alpha*w) / (K * np.sinh(alpha*w*K))) * np.cos(w*K) * 1/6


class KsinResonance(ShapeTemplate):
    """
    Damped :math:`K \\sin` oscillations of Adshead+ 2012 (arXiv:1905.05697, Eq. 17).

    Parameters
    ----------
    w_res : float
        Frequency.
    alpha : float
        Damping, as in ``K2cosResonance``.
    """

    normalise = None #Amplitude convention of the cited template, not S(1,1,1) = 1

    def raw_shape(self, k1, k2, k3): #Eq(17) in https://arxiv.org/pdf/1905.05697
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        w, alpha = self.args['w_res'], self.args['alpha']
        K = k1 + k2 + k3
        return K * ( (alpha*w) / (K * np.sinh(alpha*w*K))) * np.sin(w*K) * 1/6


class NBDsin(ShapeTemplate):
    """
    Oscillations from a non-Bunch-Davies initial state (arXiv:1502.01592, Eq. 21).

    Parameters
    ----------
    w_res : float
        Frequency.
    phi : float
        Phase.
    """

    normalise = None #Amplitude convention of the cited template, not S(1,1,1) = 1

    def raw_shape(self, k1, k2, k3): #Eq(21) in https://arxiv.org/pdf/1502.01592
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        K  = k1 + k2 + k3
        w, phi = self.args['w_res'], self.args['phi']
        S1 = np.exp(-w*(K - 2*k1)) + np.exp(-w*(K - 2*k2)) + np.exp(-w*(K - 2*k3))
        S2 = np.sin(w*K + phi)
        return S1 * S2 * 1/6
