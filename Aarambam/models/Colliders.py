import numpy as np
from scipy import special

class ScalarI:

    def raw_bispectrum(self, k1, k2, k3):
    
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        
        mu    = self.args['mass']
        beta  = mu**2 + 1/4
        kT    = k1 + k2 + k3
        delta = np.angle( special.gamma(5/2 + 1j*mu) * special.gamma(-1j*mu)*(1 + 1j*np.sinh(np.pi*mu)) )
        
        S_11  = 2*k1*k2*k3 / (beta*kT**2) / (k1 + k2)
        S_12  = (1 + 4*k3/(beta + 2)/(k1 + k2) + (beta + 4)*k3**2/(beta + 2)**2/(k1 + k2)**2)
        S_13  = np.power(kT/(k1 + k2), -beta/(beta + 2))
        
        S_21  = k1*k2*k3/(6*np.cosh(np.pi*mu)*(k1 + k2)**3)
        S_22  = -2*(2*mu**4 - 1) * (k3**2/kT**2 + k3*(4*kT - k3)/kT**2*np.log(kT/(k1 + k2)) + np.log(kT/(k1 + k2))**2)
        S_23  = mu**2 * np.power(k3/(k1 + k2), (1 + 16*mu**2)/(1 + 8*mu**2))
        S_24  = (k3*(6*kT - k3 + 8*mu**2*(8*kT - k3))/(1 + 8*mu**2)/kT**2 
                 +2*(3 + 68*mu**2 + 384*mu**4)/(1 + 8*mu**2)**2*np.log(kT/(k1 + k2))
                )
        
        S_31  = k1*k2/(k1 + k2)**2
        S_32  = np.sqrt(np.pi**3 * beta * (beta + 2) / mu / np.sinh(2*np.pi*mu))
        S_33  = np.sqrt(k3/(k1 + k2))
        S_34  = np.cos(mu * np.log(k3/2/(k1 + k2)) + delta)
        
        S     = S_11*S_12*S_13 + S_21 * (S_22 + S_23*S_24) + S_31*S_32*S_33*S_34

        #Now the normalization
        N_11  = 1/(beta * 3**2)
        N_12  = 1 + 2 / (beta + 2) + (beta + 4) / (4 * (beta + 2)**2)
        N_13  = np.power(3/2, -beta/(beta + 2))

        N_21  = 1 / (48 * np.cosh(np.pi * mu))
        N_22  = -2 * (2 * mu**4 - 1) * (1/9 + 11/9 * np.log(3/2) + np.log(3/2)**2)
        N_23  = mu**2 * np.power(1/2, (1 + 16*mu**2)/(1 + 8*mu**2))
        N_24  = (17 + 184 * mu**2) / (9 * (1 + 8 * mu**2)) + 2 * (3 + 68 * mu**2 + 384 * mu**4) / (1 + 8 * mu**2)**2 * np.log(3/2)

        N_31  = 1/4
        N_32  = np.sqrt(np.pi**3 * beta * (beta + 2) / mu / np.sinh(2*np.pi*mu))
        N_33  = np.sqrt(1/2)
        N_34  = np.cos(mu * np.log(1/2/2) + delta)

        N     = N_11*N_12*N_13 + N_21 * (N_22 + N_23*N_24) + N_31*N_32*N_33*N_34
        N     = N * 3

        S     = S/N * 3/6

        
        return S/(k1 * k2 * k3)**2 
    

class ScalarII:
        
    def raw_bispectrum(self, k1, k2, k3):
        
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        
        #Have to manually copy the ScalarI template, since we have to handle normalization separately.
        mu    = self.args['mass']
        beta  = mu**2 + 1/4
        kT    = k1 + k2 + k3
        delta = np.angle( special.gamma(5/2 + 1j*mu) * special.gamma(-1j*mu)*(1 + 1j*np.sinh(np.pi*mu)) )
        
        S_11  = 2*k1*k2*k3 / (beta*kT**2) / (k1 + k2)
        S_12  = (1 + 4*k3/(beta + 2)/(k1 + k2) + (beta + 4)*k3**2/(beta + 2)**2/(k1 + k2)**2)
        S_13  = np.power(kT/(k1 + k2), -beta/(beta + 2))
        
        
        S_21  = k1*k2*k3/(6*np.cosh(np.pi*mu)*(k1 + k2)**3)
        S_22  = -2*(2*mu**4 - 1) * (k3**2/kT**2 + k3*(4*kT - k3)/kT**2*np.log(kT/(k1 + k2)) + np.log(kT/(k1 + k2))**2)
        S_23  = mu**2 * np.power(k3/(k1 + k2), (1 + 16*mu**2)/(1 + 8*mu**2))
        S_24  = (k3*(6*kT - k3 + 8*mu**2*(8*kT - k3))/(1 + 8*mu**2)/kT**2 
                 +2*(3 + 68*mu**2 + 384*mu**4)/(1 + 8*mu**2)**2*np.log(kT/(k1 + k2))
                )
        
        S_31  = k1*k2/(k1 + k2)**2
        S_32  = np.sqrt(np.pi**3 * beta * (beta + 2) / mu / np.sinh(2*np.pi*mu))
        S_33  = np.sqrt(k3/(k1 + k2))
        S_34  = np.cos(mu * np.log(k3/2/(k1 + k2)) + delta)
        
        SI    = S_11*S_12*S_13 + S_21 * (S_22 + S_23*S_24) + S_31*S_32*S_33*S_34

        #Now the normalization
        N_11  = 1/(beta * 3**2)
        N_12  = 1 + 2 / (beta + 2) + (beta + 4) / (4 * (beta + 2)**2)
        N_13  = np.power(3/2, -beta/(beta + 2))

        N_21  = 1 / (48 * np.cosh(np.pi * mu))
        N_22  = -2 * (2 * mu**4 - 1) * (1/9 + 11/9 * np.log(3/2) + np.log(3/2)**2)
        N_23  = mu**2 * np.power(1/2, (1 + 16*mu**2)/(1 + 8*mu**2))
        N_24  = (17 + 184 * mu**2) / (9 * (1 + 8 * mu**2)) + 2 * (3 + 68 * mu**2 + 384 * mu**4) / (1 + 8 * mu**2)**2 * np.log(3/2)

        N_31  = 1/4
        N_32  = np.sqrt(np.pi**3 * beta * (beta + 2) / mu / np.sinh(2*np.pi*mu))
        N_33  = np.sqrt(1/2)
        N_34  = np.cos(mu * np.log(1/2/2) + delta)

        NI    = N_11*N_12*N_13 + N_21 * (N_22 + N_23*N_24) + N_31*N_32*N_33*N_34


        #Now the scalarII template
        d_1   = np.angle((special.gamma(1/2 + 1j*mu) + special.gamma(3/2 + 1j*mu)) 
                         * special.gamma(-1j*mu)*(1j + 1/np.sinh(np.pi*mu)) 
                        )
        d_2   = np.angle(special.gamma(5/2 + 1j*mu) * special.gamma(-1j*mu)*(1j + 1/np.sinh(np.pi*mu)) 
                        )
        
        S_41  = k3*(k3**2 - k1**2 - k2**2)/beta/(k1 + k2)**3
        S_42  = (6 - 6*beta*k3/(beta + 2)/kT + 
                 2*beta*(beta + 1)*k3**2/(beta + 2)**2/kT**2  +
                 (k1**2 + k2**2)/(k1*k2) * (2 - beta*k3/(beta + 2)/kT)
                )
        S_43  = np.power(kT/(k1 + k2), -beta/(beta + 2))
        
        S_51  = (k3**2 - k1**2 - k2**2)/(k1*k2) * np.sqrt(k3/(k1 + k2))
        S_52  = (np.sqrt(np.pi**3*(beta + 2)/(mu * np.sinh(2*np.pi*mu))) * 
                 np.cos(mu * np.log(k3/2/(k1 + k2)) + d_1)
                )
        S_53  = ((k1*k2)/(k1 + k2)**2 * 
                 np.sqrt(np.pi**3*(beta + 2)/(mu * np.sinh(2*np.pi*mu))) * 
                 np.cos(mu * np.log(k3/2/(k1 + k2)) + d_2)
                )
        
        S_61  = k3 * (k1**2 + k2**2 - k3**2)/(12*np.cosh(np.pi*mu) * k1 * k2 * (k1 + k2)**4)
        S_62  = 2*(2*mu**4 - 1)*(k1 + k2)
        S_63  = (k1**2 + k2**2 + 3*k1*k2)*np.log(kT/(k1 + k2))**2
        S_64  = k3/kT**2 * (k1**3 + k2**3 + (k1**2 + k2**2)*k3 + 7*k1*k2*(k1 + k2) + 5*k1*k2*k3) * np.log(kT/(k1 + k2))
        S_65  = k1*k2*k3**2/kT**2
        S_66  = mu**2*k3 * np.power(k3/(k1 + k2), 8*mu**2/(8*mu**2 + 1))
        S_67  = (32*mu**2 + 3)/(8*mu**2 + 1) * (k1*k2/(8*mu**2 + 1) - k1**2 - k2**2 - 5*k1*k2) * np.log(kT/(k1 + k2))
        S_68  = 2/(8*mu**2 + 1)*(k1*k2*k3)/kT - k3/kT*(k1**3 + k2**3 + (k1**2 + k2**2)*k3 + 11*k1*k2*(k1 + k2) + 9*k1*k2*k3)
        
        SII   = S_41*S_42*S_43 + S_51*(S_52 + S_53) + S_61 * (S_62*(S_63 + S_64 + S_65) + S_66*(S_67 + S_68))
        
        #Once again normalization for SII

        N_41  = -1/(8*beta)
        N_42  = 6 - 2 * beta / (beta + 2) + 2 * beta * (beta + 1) / (9 * (beta + 2)**2) + 2 * (2 - beta / (3 * (beta + 2)))
        N_43  = np.power(3/2, -beta/(beta + 2))

        N_51  = -1 / np.sqrt(2)
        N_52  = np.sqrt(np.pi**3 * (beta + 2) / (mu * np.sinh(2 * np.pi * mu))) * np.cos(mu * np.log(1/4) + d_1)
        N_53  = (1/4) * np.sqrt(np.pi**3 * (beta + 2) / (mu * np.sinh(2 * np.pi * mu))) * np.cos(mu * np.log(1/4) + d_2)

        N_61  = 1 / (192 * np.cosh(np.pi * mu))
        N_62  = 4 * (2 * mu**4 - 1) 
        N_63  = 5 * np.log(3/2)**2
        N_64  = (23 / 9) * np.log(3/2)
        N_65  = 1/9
        N_66  = mu**2 * np.power(1/2, 8 * mu**2 / (8 * mu**2 + 1))
        N_67  = (32 * mu**2 + 3) / (8 * mu**2 + 1) * (1 / (8 * mu**2 + 1) - 7) * np.log(3/2)
        N_68  = 2 / (3 * (8 * mu**2 + 1)) - 35 / 3

        NII   = N_41*N_42*N_43 + N_51*(N_52 + N_53) + N_61 * (N_62*(N_63 + N_64 + N_65) + N_66*(N_67 + N_68))

        S     = -SI - 0.14*SII
        N     = -NI - 0.14*NII
        N     = N * 3

        S     = S / N * 3/6
        
        return S / (k1 * k2 * k3)**2
    

class HeavySpinCollider:
        
    def raw_bispectrum(self, k1, k2, k3):
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        kT = k1 + k2 + k3

        s  = self.args['spin']
        
        P_s = special.eval_legendre(s, (k1**2 + k3**2 - k2**2)/(2*k1*k3))
        S_1 = k2 / np.power(k1*k3, 1 - s) / np.power(kT, 2*s + 1)
        S_2 = (2*s - 1) * ((k1 + k3) * kT + 2*s*k1*k3) + kT**2
        S   = P_s * S_1 * S_2
        
        N = special.eval_legendre(s, 1/2) / np.power(3, 2*s + 1) * ((2*s - 1)*(2*s + 6) + 9) * 6
        S = S/N * 6/6
        
        return S/(k1 * k2 * k3)**2


class LowSpeedCollider:
    
    def raw_bispectrum(self, k1, k2, k3):
        
        k1_sq = k1 * k1   
        k1_co = np.power(k1, 2 * (self.args['n_s'] - 1) / 3)
        k1_in = np.power(k1, self.args['n_s'] - 2)

        k2_co = np.power(k2, 2 * (self.args['n_s'] - 1) / 3)
        k2_in = np.power(k2, self.args['n_s'] - 2)

        k3_li = np.power(k3, (self.args['n_s'] + 2) / 3)
        k3_co = np.power(k3, 2 * (self.args['n_s'] - 1) / 3)
        k3_in = np.power(k3, self.args['n_s'] - 2)

        S_1 = k1_sq * k2_in * k3_in * 3/6
        S_2 = k1_co * k2_co * k3_co * 1/6
        S_3 = k1_in * k2_co * k3_li * 6/6
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        
        
        alpha = self.args['alpha']
        S_4   = k1**2/(3*k2*k3) * 1/(1 + alpha*(k1**2/(3*k2*k3))**2) * 3/6
        S     = (-1) * S_1 + (-2) * S_2 + 1 * S_3 + S_4
        
        N = 1 + (1/3 * 1/(1 + alpha/9) * 3/6)
        S = S/N
        
        return S/(k1 * k2 * k3)**2
    
    
class MultiSpeedCollider:
    
    def raw_bispectrum(self, k1, k2, k3):
        
        k1 = np.power(k1, (4 - self.args['n_s'])/3)
        k2 = np.power(k2, (4 - self.args['n_s'])/3)
        k3 = np.power(k3, (4 - self.args['n_s'])/3)
        
        c1, c2, c3 = self.args['c1'], self.args['c2'], self.args['c3']
        
        S = (k1*k2*k3) / (c1*k1 + c2*k2 + c3*k3)**3
        
        N = 1/(c1 + c2 + c3)**3
        N = N * 6
        S = S/N * 6/6 #Normalize to Equil limit and add symmetry factor
        
        return S/(k1 * k2 * k3)**2