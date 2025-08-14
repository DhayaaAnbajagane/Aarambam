import numpy as np

class LogResonance:
    
    def raw_bispectrum(self, k1, k2, k3): #https://arxiv.org/pdf/1905.05697, Equation (12)
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)

        w, phi = self.args['w_res'], self.args['phi']
        S = np.sin(w * np.log(k1 + k2 + k3) + phi) * 1/6
                
        return S / (k1 * k2 * k3)**2
    

class LinearResonance:

    def raw_bispectrum(self, k1, k2, k3): #https://arxiv.org/pdf/1905.05697, Equation (15)
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        
        w, phi = self.args['w_res'], self.args['phi']
        S = np.sin(w * (k1 + k2 + k3) + phi) * 1/6
                
        return S / (k1 * k2 * k3)**2
    

class K2cosResonance:

    def raw_bispectrum(self, k1, k2, k3): #Eq(16) in https://arxiv.org/pdf/1905.05697
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)

        w, alpha = self.args['w_res'], self.args['alpha']
        K = k1 + k2 + k3
        S = K**2 * ( (alpha*w) / (K * np.sinh(alpha*w*K))) * np.cos(w*K) * 1/6
                
        return S / (k1 * k2 * k3)**2


class KsinResonance:

    def raw_bispectrum(self, k1, k2, k3): #Eq(17) in https://arxiv.org/pdf/1905.05697
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)

        w, alpha = self.args['w_res'], self.args['alpha']
        K = k1 + k2 + k3
        S = K * ( (alpha*w) / (K * np.sinh(alpha*w*K))) * np.sin(w*K) * 1/6
                
        return S / (k1 * k2 * k3)**2

class NBDsin:

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