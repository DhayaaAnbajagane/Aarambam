"""aarambam command line (pure Python parts)."""
import os, shutil

import numpy as np
import pytest


@pytest.mark.parametrize('method', ['basis', 'resbasis', 'binned'])
def test_example_config(am, tmp_path, method):
    path = tmp_path / f'{method}.cfg'
    am.cli.main(['example-config', method, '--config_path', str(path)])
    assert path.read_text() == getattr(am.utils, f'make_example_config_{method}')()


def test_camb2input(am, tmp_path, defaults_dir):
    shutil.copy(os.path.join(defaults_dir, 'TransferFunction.dat'), tmp_path / 'TransferFunction.dat')
    am.cli.main(['camb2input', '--transfer_path', str(tmp_path / 'TransferFunction.dat')])
    np.testing.assert_allclose(np.loadtxt(tmp_path / 'TransferFunctionConverted.dat'),
                               np.loadtxt(os.path.join(defaults_dir, 'TransferFunctionConverted.dat')), rtol=0)


def test_collate(am, tmp_path):
    from test_postprocess import write_slab
    a = np.arange(27.).reshape(3, 3, 3)
    for t in ('Gauss', 'Nongauss'):
        write_slab(tmp_path / f'{t}_potential_5', a, 3, 0)
    am.cli.main(['collate', '--file_dir', str(tmp_path)])
    np.testing.assert_array_equal(np.load(tmp_path / 'Nongauss_potential.npy'), a)
    assert not list(tmp_path.glob('*_potential_5'))


def test_resbasis_runs_the_modal_binary(am):
    assert am.cli.BINARIES['resbasis'] == am.cli.BINARIES['basis'] == '2LPTBasis'


def test_missing_binary_message(am, monkeypatch, tmp_path):
    import Aarambam.cli as cli
    monkeypatch.setattr(cli, 'files', lambda pkg: tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.run_2lpt('binned', ['x.cfg'])
    assert 'not found' in str(e.value)
