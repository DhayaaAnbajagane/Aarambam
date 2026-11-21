"""Cosmological-collider templates of Sohn+ 2024 (arXiv:2404.07203)."""
import numpy as np, numbers
from scipy import special
from .base import ShapeTemplate

__all__ = ['ScalarI', 'ScalarII', 'HeavySpinCollider', 'LowSpeedCollider', 'MultiSpeedCollider']


def _scalar_I_shape(k1, k2, k3, mu):
    """Unnormalised Scalar-I shape of Sohn+ 2024 (Eq. 2.15, with the extra small-mass term of their appendix)."""
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

    return S_11*S_12*S_13 + S_21 * (S_22 + S_23*S_24) + S_31*S_32*S_33*S_34

class ScalarI(ShapeTemplate):
    """
    Scalar-I collider template, interaction :math:`\\dot\\phi^2 \\sigma` (Sohn+ 2024, arXiv:2404.07203, Eq. 2.15).

    Parameters
    ----------
    mass : float
        :math:`\\mu = \\sqrt{(m/H)^2 - 9/4} > 0`; sets the frequency of the oscillation.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        return _scalar_I_shape(k1, k2, k3, self.args['mass']) * 3/6 #Symmetry factor


class ScalarII(ShapeTemplate):
    """
    Scalar-II collider template, a combination of :math:`\\dot\\phi^2 \\sigma` and
    :math:`(\\partial\\phi)^2 \\sigma` (Sohn+ 2024, Eq. 2.20).

    Parameters
    ----------
    mass : float
        :math:`\\mu = \\sqrt{(m/H)^2 - 9/4} > 0`.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""

        mu    = self.args['mass']
        beta  = mu**2 + 1/4
        kT    = k1 + k2 + k3
        SI    = _scalar_I_shape(k1, k2, k3, mu)

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
        S_68  = 2/(8*mu**2 + 1)*(k1*k2*k3)/kT - k3/kT**2*(k1**3 + k2**3 + (k1**2 + k2**2)*k3 + 11*k1*k2*(k1 + k2) + 9*k1*k2*k3)

        SII   = S_41*S_42*S_43 + S_51*(S_52 + S_53) + S_61 * (S_62*(S_63 + S_64 + S_65) + S_66*(S_67 + S_68))

        return (-SI - 0.14*SII) * 3/6 #Symmetry factor


class HeavySpinCollider(ShapeTemplate):
    """
    Heavy spinning-particle collider template, :math:`m \\gg H` (Sohn+ 2024, Eq. 2.24).

    Parameters
    ----------
    spin : int
        Spin of the exchanged particle.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""

        s  = self.args['spin']
        assert isinstance(s, numbers.Integral), f"The provided spin must be an integer. You provided {type(s)}"

        kT  = k1 + k2 + k3
        P_s = special.eval_legendre(s, (k1**2 + k3**2 - k2**2)/(2*k1*k3))
        S_1 = k2 / np.power(k1*k3, 1 - s) / np.power(kT, 2*s + 1)
        S_2 = (2*s - 1) * ((k1 + k3) * kT + 2*s*k1*k3) + kT**2

        return P_s * S_1 * S_2 * 6/6


class LowSpeedCollider(ShapeTemplate):
    """
    Low-speed collider template (Sohn+ 2024, Eq. 2.33).

    Parameters
    ----------
    alpha : float
        :math:`\\alpha = c_s m / H`. Large alpha gives the equilateral template.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""

        #Equilateral template, written in the rescaled k. This matches the tilt of
        #the ``Equilateral`` model exactly.
        S_1 = k1**2/(k2*k3) * 3/6
        S_2 = 1 * 1/6
        S_3 = k3/k1 * 6/6

        alpha = self.args['alpha']
        S_4   = k1**2/(3*k2*k3) * 1/(1 + alpha*(k1**2/(3*k2*k3))**2) * 3/6

        return (-1) * S_1 + (-2) * S_2 + 1 * S_3 + S_4


class MultiSpeedCollider(ShapeTemplate):
    """
    Multi-speed collider template, massless exchange with different sound speeds (Sohn+ 2024, Eq. 2.34).

    Parameters
    ----------
    c1, c2, c3 : float
        Sound speeds of the three legs, in units of c.
    """

    def raw_shape(self, k1, k2, k3):
        """One permutation term of the shape in the rescaled k (see ``ShapeTemplate``)."""
        c1, c2, c3 = self.args['c1'], self.args['c2'], self.args['c3']
        return (k1*k2*k3) / (c1*k1 + c2*k2 + c3*k3)**3 * 6/6
