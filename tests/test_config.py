"""Config writers (pure Python)."""
import os, subprocess, sys

import pytest

from conftest import ROOT


def values(cfg):
    out = {}
    for line in cfg.splitlines():
        f = line.split()
        if len(f) >= 2 and not f[0].startswith('%'):
            out[f[0]] = f[1]
    return out


def test_import_without_tmpdir():
    env = {k: v for k, v in os.environ.items() if k != 'TMPDIR'}
    code = ("import sys, types, os; r = sys.argv[1]; sys.path.insert(0, r)\n"
            "for n, p in [('Aarambam', 'Aarambam'), ('Aarambam.basis', 'Aarambam/basis')]:\n"
            "    m = types.ModuleType(n); m.__path__ = [os.path.join(r, p)]; sys.modules[n] = m\n"
            "import Aarambam.utils as U; print(U.DEFAULT_BINNED_INPUTS['OutputDir'])\n")
    r = subprocess.run([sys.executable, '-c', code, ROOT], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize('method', ['basis', 'resbasis', 'binned'])
def test_example_configs(am, method):
    cfg = values(getattr(am.utils, f'make_example_config_{method}')())
    assert cfg['NumFilesWrittenInParallel'] == '1'
    assert int(cfg['Nmax']) >= int(cfg['Nsample'])
    if method == 'resbasis':
        assert {'A_res', 'w_res', 'phi_res'} <= set(cfg) and cfg['WhichSpectrum'] in ('1', '2')
    else:
        assert cfg['WhichSpectrum'] == '0'


def test_floats_round_trip(am):
    d = dict(am.utils.DEFAULT_BINNED_INPUTS); d.update(Fnl=1e-7, Om=0.30891234567891234, sigma8=0.8)
    cfg = values(am.utils.make_config_binned(**d))
    assert float(cfg['Fnl']) == 1e-7
    assert float(cfg['Omega']) == 0.30891234567891234


def test_utils_reexports(am):
    for name in am.utils.__all__:
        assert hasattr(am.utils, name)
    assert am.utils.Decomposer is am.decomposer.Decomposer
    assert am.utils.collate_potential is am.postproc.collate_potential


def test_decomposer_requires_matching_basis(am):
    D = am.utils.Decomposer(n_s=0.96, Lbox=1000., Nmax=32, N_modes=5, ModeTol=1e-3, MaxModeCount=10)
    with pytest.raises(ValueError):
        D.go(am.models.Local, am.binned.BasicBinnedDecompose)
