"""Template normalisation, tilt and symmetry checks (pure Python, seconds)."""
import numpy as np
import pytest

SCALE_INVARIANT = ['Local', 'Equilateral', 'Orthogonal', 'QuasiSingleField', 'DBI', 'SenatoreEFT1', 'SenatoreEFT2',
                   'ScalarI', 'ScalarII', 'HeavySpinCollider', 'LowSpeedCollider', 'MultiSpeedCollider']
RESONANCES      = ['LogResonance', 'LinearResonance', 'K2cosResonance', 'KsinResonance', 'NBDsin']
PARAMS          = dict(mass=1.0, spin=2, alpha=1.0, c1=1.0, c2=0.7, c3=0.4, w_res=10.0, phi=0.3)


def make(am, name, n_s, **extra):
    args = dict(PARAMS); args.update(extra)
    if name == 'QuasiSingleField':
        args['mass'] = 0.5
    C = type('C', (getattr(am.models, name), am.binned.BasicBinnedDecompose), {})
    return C(n_s=n_s, Lbox=1000., Nmax=256, Binned_progress=False, **args)


@pytest.mark.parametrize('name', SCALE_INVARIANT)
@pytest.mark.parametrize('n_s', [1.0, 0.96])
def test_normalisation_and_tilt(am, name, n_s):
    """S(1, 1, 1) = 1, and S scales as lambda^(2 (n_s - 1)) on a generic triangle (a shape term of the wrong degree
    would show up here even at n_s = 1)."""
    m = make(am, name, n_s)
    one = np.ones(1)
    assert m.shape_function(one, one, one)[0] == pytest.approx(1.0, rel=1e-10)
    k1, k2, k3 = np.array([0.2]), np.array([0.5]), np.array([0.6])
    for lam in (0.01, 3.0):
        ratio = m.shape_function(lam*k1, lam*k2, lam*k3)[0] / m.shape_function(k1, k2, k3)[0]
        assert ratio == pytest.approx(lam**(2*(n_s - 1)), rel=1e-8)


@pytest.mark.parametrize('name', SCALE_INVARIANT + RESONANCES)
def test_symmetric(am, name):
    m = make(am, name, 0.96)
    rng = np.random.default_rng(0)
    k1, k2 = 10**rng.uniform(-2, 0, 50), 10**rng.uniform(-2, 0, 50)
    k3 = np.abs(k1 - k2) + rng.uniform(0.1, 0.9, 50) * (k1 + k2 - np.abs(k1 - k2))
    B = m.bispectrum(k1, k2, k3)
    np.testing.assert_allclose(m.bispectrum(k2, k3, k1), B, rtol=1e-12)
    np.testing.assert_allclose(m.bispectrum(k3, k2, k1), B, rtol=1e-12)


@pytest.mark.parametrize('n_s', [1.0, 0.96])
def test_low_speed_collider_equilateral_limit(am, n_s):
    lsc = make(am, 'LowSpeedCollider', n_s, alpha=1e12)
    eq  = make(am, 'Equilateral', n_s)
    for kk in (0.006, 0.05, 1.0):
        k = np.array([kk])
        assert lsc.bispectrum(k, k, k)[0] / eq.bispectrum(k, k, k)[0] == pytest.approx(1.0, rel=1e-8)


def test_heavy_spin_integer_types(am):
    k = np.ones(1)
    assert np.isfinite(make(am, 'HeavySpinCollider', 1.0, spin=np.int64(2)).bispectrum(k, k, k)[0])
    with pytest.raises(AssertionError):
        make(am, 'HeavySpinCollider', 1.0, spin=2.0).bispectrum(k, k, k)


def test_user_template(am):
    class MyLocal(am.models.ShapeTemplate):
        def raw_shape(self, k1, k2, k3):
            return k1**2/(k2*k3) * 3/6 * 7.0     #arbitrary amplitude, removed by the normalisation

    C = type('C', (MyLocal, am.binned.BasicBinnedDecompose), {})
    mine, ref = C(n_s=0.96, Lbox=1000., Nmax=256, Binned_progress=False), make(am, 'Local', 0.96)
    k1, k2, k3 = np.array([0.01, 0.3]), np.array([0.5, 0.3]), np.array([0.5, 0.4])
    np.testing.assert_allclose(mine.bispectrum(k1, k2, k3), ref.bispectrum(k1, k2, k3), rtol=1e-12)

    class ZeroAtEquilateral(am.models.ShapeTemplate):     #cannot be normalised at k1 = k2 = k3: an error, not NaNs
        def raw_shape(self, k1, k2, k3):
            return (k1 - k2) / k3

    C = type('C', (ZeroAtEquilateral, am.binned.BasicBinnedDecompose), {})
    with pytest.raises(ValueError):
        C(n_s=0.96, Lbox=1000., Nmax=256, Binned_progress=False).bispectrum(k1, k2, k3)
