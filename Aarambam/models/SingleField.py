import numpy as np
from scipy import special


class Local:
        
    def bispectrum(self, k1, k2, k3):
            
        S_1 = k1**2 * (k2 * k3) ** (self.args['n_s'] - 2)
        S_2 = k2**2 * (k3 * k1) ** (self.args['n_s'] - 2)
        S_3 = k3**2 * (k1 * k2) ** (self.args['n_s'] - 2)

        S = (S_1 + S_2 + S_3)/3 #1/3 is normalizing from Equil limit
        
        return S/(k1 * k2 * k3)**2
    
    
class Equilateral:
        
    def bispectrum(self, k1, k2, k3):
        
        k1_sq = k1 * k1   
        k1_li = np.power(k1, (self.args['n_s'] + 2) / 3)
        k1_co = np.power(k1, 2 * (self.args['n_s'] - 1) / 3)
        k1_in = np.power(k1, self.args['n_s'] - 2)

        k2_sq = k2 * k2
        k2_li = np.power(k2, (self.args['n_s'] + 2) / 3)
        k2_co = np.power(k2, 2 * (self.args['n_s'] - 1) / 3)
        k2_in = np.power(k2, self.args['n_s'] - 2)

        k3_sq = k3 * k3
        k3_li = np.power(k3, (self.args['n_s'] + 2) / 3)
        k3_co = np.power(k3, 2 * (self.args['n_s'] - 1) / 3)
        k3_in = np.power(k3, self.args['n_s'] - 2)

        S_1 = (k1_sq * k2_in * k3_in + 
               k2_sq * k3_in * k1_in + 
               k3_sq * k1_in * k2_in)

        S_2 = k1_co * k2_co * k3_co

        S_3 = (k1_in * k2_co * k3_li + 
               k1_in * k3_co * k2_li + 
               k2_in * k1_co * k3_li + 
               k2_in * k3_co * k1_li + 
               k3_in * k1_co * k2_li + 
               k3_in * k2_co * k1_li)

        S = ((-1) * S_1 + (-2) * S_2 + 1 * S_3)
        
        
        return S/(k1 * k2 * k3)**2
    

class Orthogonal:
    
    def bispectrum(self, k1, k2, k3):
        
        k1_sq = k1 * k1   
        k1_li = np.power(k1, (self.args['n_s'] + 2) / 3)
        k1_co = np.power(k1, 2 * (self.args['n_s'] - 1) / 3)
        k1_in = np.power(k1, self.args['n_s'] - 2)

        k2_sq = k2 * k2
        k2_li = np.power(k2, (self.args['n_s'] + 2) / 3)
        k2_co = np.power(k2, 2 * (self.args['n_s'] - 1) / 3)
        k2_in = np.power(k2, self.args['n_s'] - 2)

        k3_sq = k3 * k3
        k3_li = np.power(k3, (self.args['n_s'] + 2) / 3)
        k3_co = np.power(k3, 2 * (self.args['n_s'] - 1) / 3)
        k3_in = np.power(k3, self.args['n_s'] - 2)

        S_1 = (k1_sq * k2_in * k3_in + 
               k2_sq * k3_in * k1_in + 
               k3_sq * k1_in * k2_in)

        S_2 = k1_co * k2_co * k3_co

        S_3 = (k1_in * k2_co * k3_li + 
               k1_in * k3_co * k2_li + 
               k2_in * k1_co * k3_li + 
               k2_in * k3_co * k1_li + 
               k3_in * k1_co * k2_li + 
               k3_in * k2_co * k1_li)

        S = ((-3) * S_1 + (-8) * S_2 + 3 * S_3)
        
        return S/(k1 * k2 * k3)**2
    

class QuasiSingleField:
    
    def raw_bispectrum(self, k1, k2, k3):
        
        mu = self.args['mass']
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        
        kT    = k1 + k2 + k3
        kappa = k1*k2*k3/kT**3
        
        S = 3 * np.sqrt(3 * kappa) * special.yv(mu, 8 * kappa) / special.yv(mu, 8/27)
        
        return S/(k1 * k2 * k3)**2 * 1/6


class DBI:

    def raw_bispectrum(self, k1, k2, k3): #Eq(7) in https://arxiv.org/pdf/1303.5084
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        
        S1 = -3/7 / np.power(k1 + k2 + k3, 2)
        S2 = k1**5 * 3/6
        S3 = (2*k1**4*k2 - 3*k1**3*k2**2) * 6/6
        S4 = (k1**3*k2*k3 - 4*k1**2*k2**2*k3) * 6/6
        B  = S1 * (S2 + S3 + S4) / np.power(k1*k2*k3, 3) 

        return B
    

class SenatoreEFT1:
    def raw_bispectrum(self, k1, k2, k3): #Eq(5) in https://arxiv.org/pdf/1303.5084
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        
        S1 = -9/17 / np.power(k1 + k2 + k3, 3)
        S2 = k1**6 * 3/6
        S3 = (3*k1**5*k2 - k1**4*k2**2 - 3*k1**3*k2**3) * 6/6
        S4 = (3*k1**4*k2*k3 - 9*k1**3*k2**2*k3 - 2*np.power(k1*k2*k3, 2)) *  6/6
        B  = S1 * (S2 + S3 + S4) / np.power(k1*k2*k3, 3)

        return B


class SenatoreEFT2:
    def raw_bispectrum(self, k1, k2, k3): #Eq(6) in https://arxiv.org/pdf/1303.5084
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        
        S1 = 27 / np.power(k1 + k2 + k3, 3)
        B  = S1 / (k1*k2*k3) 

        return B