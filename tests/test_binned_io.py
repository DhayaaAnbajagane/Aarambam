"""Binned decomposition tables and their link to the C config (pure Python, tiny Nk)."""
import os

import numpy as np
import pytest

#Binned_filter -> (BinnedPNGFilter, BinnedPNGOutputFilter) codes of the C code
FILTERS = {'heaviside': (1, 1), 'hybrid_cubic_k3cubic': (3, 3), 'hybrid_linear_k3linear': (2, 2), 'hybrid_cubic': (3, 1)}


def decompose(am, outdir, filt):
    D = am.utils.Decomposer(n_s=0.96, Lbox=1000., Nmax=32, Binned_Nk=4, Binned_filter=filt)
    return D.go(am.models.Local, am.binned.BasicBinnedDecompose, outdir=str(outdir), Binned_progress=False)


@pytest.mark.parametrize('filt', sorted(FILTERS))
def test_kernel_inputs_from_metadata(am, tmp_path, filt):
    decompose(am, tmp_path, filt)
    inp = am.utils.binned_kernel_inputs(str(tmp_path))
    assert (inp['BinnedPNGFilterCode'], inp['BinnedPNGOutputFilterCode']) == FILTERS[filt]

    #the table has exactly the column count the C reader requires for that output filter
    ncoef = {1: 1, 2: 2, 3: 4}[inp['BinnedPNGOutputFilterCode']]
    assert np.loadtxt(inp['BinnedKernelValuesPath'], ndmin=2).shape[1] == 6 + 4*ncoef + 2

    d = dict(am.utils.DEFAULT_BINNED_INPUTS); d.update(inp)
    am.utils.make_config_binned(**d)
    d.update(BinnedPNGOutputFilterCode=inp['BinnedPNGOutputFilterCode'] % 3 + 1)
    with pytest.raises(ValueError):
        am.utils.make_config_binned(**d)


def test_heaviside_removes_stale_hybrid_table(am, tmp_path):
    decompose(am, tmp_path, 'hybrid_cubic_k3cubic')
    assert os.path.exists(tmp_path / 'BinnedKernel_hybrid_values.txt')
    decompose(am, tmp_path, 'heaviside')
    assert not os.path.exists(tmp_path / 'BinnedKernel_hybrid_values.txt')


def test_linear_k_coordinate_rejected(am, tmp_path):
    D = am.utils.Decomposer(n_s=0.96, Lbox=1000., Nmax=32, Binned_Nk=4, Binned_linear_coordinate='k')
    with pytest.raises(NotImplementedError):
        D.go(am.models.Local, am.binned.BasicBinnedDecompose, Binned_progress=False)


def test_k1_k2_share_edges(am, tmp_path):
    """No k2 edges file or config line: the internal legs k1 and k2 share the k1 bins."""
    decompose(am, tmp_path, 'hybrid_cubic_k3cubic')
    assert not os.path.exists(tmp_path / 'BinnedKernel_k2_edges.txt')
    inp = am.utils.binned_kernel_inputs(str(tmp_path))
    assert 'BinnedK2EdgesPath' not in inp
    d = dict(am.utils.DEFAULT_BINNED_INPUTS); d.update(inp)
    cfg = am.utils.make_config_binned(**d)
    assert 'K2Edges' not in cfg and 'InternalSymmetry' not in cfg
    assert any(l.split()[:2] == ['BinnedPNGMultiGrid', '1'] for l in cfg.splitlines())

    from Aarambam.basis.binned.io import load_outputs
    edges = load_outputs(str(tmp_path))['edges']
    np.testing.assert_array_equal(edges[0], edges[1])


def test_different_k1_k2_edges_rejected(am):
    class Skewed(am.binned.BasicBinnedDecompose):
        def generate_edges(self):
            e = super().generate_edges()
            return (e[0], e[1] * 1.01, e[2])

    D = am.utils.Decomposer(n_s=0.96, Lbox=1000., Nmax=32, Binned_Nk=4)
    with pytest.raises(ValueError):
        D.go(am.models.Local, Skewed, Binned_progress=False)
