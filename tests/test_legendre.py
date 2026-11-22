"""Legendre decomposition with the compiled CMB-BEST extension, and its tables run through 2LPTBasis (skipped without
the extension, e.g. on the source tree; the GitHub workflow runs it on the installed package)."""
import os

import numpy as np
import pytest

try:
    from Aarambam.basis import legendre
except ImportError:         #the package is there, its compiled part is not: importorskip would raise
    pytest.skip('compiled cmbbest extension not available', allow_module_level=True)


@pytest.fixture(scope='module')
def local_tables(am, tmp_path_factory):
    """The local template decomposed on a small basis (N_modes 6, Nmax 32), tables written to a directory."""
    d = tmp_path_factory.mktemp('legendre')
    D = am.utils.Decomposer(n_s=0.96, Lbox=500., Nmax=32, N_modes=6, ModeTol=1e-8, MaxModeCount=25)
    return D.go(am.models.Local, legendre.BasicLegendreDecompose, outdir=str(d)), str(d)


def test_local_template_reproduced(local_tables):
    """The local shape is a sum of power-law modes: the expansion reproduces it in the three limits."""
    out, d = local_tables
    np.testing.assert_allclose(out['S_apprx'], out['S_original'], rtol=1e-4)
    alpha = np.loadtxt(os.path.join(d, 'AlphaTable.dat'), ndmin=2)
    ai    = np.loadtxt(os.path.join(d, 'AiTable.dat'), ndmin=2)
    assert alpha.shape[1] == 4 and np.count_nonzero(alpha[:, 3]) > 0
    assert ai.shape == (6, 3)
    np.testing.assert_array_equal(ai[:, 0], np.arange(6))


@pytest.mark.mpi
def test_tables_run_in_2lptbasis(am, lpt_binaries, local_tables, tmp_path):
    """The written tables drive 2LPTBasis (32^3, 2 ranks): a ~1% non-Gaussian term at Fnl = 100."""
    from test_c_smoke import write_config, run, potential

    _, d = local_tables
    cfg = write_config(am, 'basis', tmp_path, Lbox=500., N_modes=6, AlphaPath=os.path.join(d, 'AlphaTable.dat'),
                       AiPath=os.path.join(d, 'AiTable.dat'))
    r = run(lpt_binaries, 'basis', cfg, 2)
    assert r.returncode == 0, r.stdout[-2000:]
    g, ng = potential(str(tmp_path), 'Gauss'), potential(str(tmp_path), 'Nongauss')
    assert np.all(np.isfinite(ng)) and 0 < np.std(ng - g) < 0.1 * np.std(g)
