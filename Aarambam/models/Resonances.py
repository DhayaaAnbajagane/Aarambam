import numpy as np

class LogResonance:
    """
    Logarithmic oscillations in the bispectrum, i.e. oscillations in :math:\\log k
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

    def raw_bispectrum(self, k1, k2, k3): #https://arxiv.org/pdf/1905.05697, Equation (12)
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)

        w, phi = self.args['w_res'], self.args['phi']
        S = np.sin(w * np.log(k1 + k2 + k3) + phi) * 1/6
                
        return S / (k1 * k2 * k3)**2
    

class LinearResonance:
    """
    Linear oscillations in the bispectrum, i.e. oscillations in k.
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

    def raw_bispectrum(self, k1, k2, k3): #https://arxiv.org/pdf/1905.05697, Equation (15)
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        
        w, phi = self.args['w_res'], self.args['phi']
        S = np.sin(w * (k1 + k2 + k3) + phi) * 1/6
                
        return S / (k1 * k2 * k3)**2
    

class K2cosResonance:

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
    - phi : float
        The phase of the oscillations.
    """

    def raw_bispectrum(self, k1, k2, k3):
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)

        w, alpha = self.args['w_res'], self.args['alpha']
        K = k1 + k2 + k3
        S = K**2 * ( (alpha*w) / (K * np.sinh(alpha*w*K))) * np.cos(w*K) * 1/6
                
        return S / (k1 * k2 * k3)**2


class KsinResonance:

    """
    A different oscillatory bispectra from Adshead+ 2012, Eq(16) in https://arxiv.org/pdf/1905.05697.
    A complement to the :math:``K^2\\cos`` model provided in `Aarambam`.
    
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

    def raw_bispectrum(self, k1, k2, k3): #Eq(17) in https://arxiv.org/pdf/1905.05697
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)

        w, alpha = self.args['w_res'], self.args['alpha']
        K = k1 + k2 + k3
        S = K * ( (alpha*w) / (K * np.sinh(alpha*w*K))) * np.sin(w*K) * 1/6
                
        return S / (k1 * k2 * k3)**2

class NBDsin:
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
    def raw_bispectrum(self, k1, k2, k3): #Eq(21) in https://arxiv.org/pdf/1502.01592
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        K  = k1 + k2 + k3

        w, phi = self.args['w_res'], self.args['phi']

        S1 = np.exp(-w*(K - 2*k1)) + np.exp(-w*(K - 2*k2)) + np.exp(-w*(K - 2*k3))
        S2 = np.sin(w*K + phi)
        B  = S1 * S2 / np.power(k1*k2*k3, 2) * 1/6

        return B