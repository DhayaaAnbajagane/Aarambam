"""Test configuration.

The Python tests run from a source checkout without the compiled Cython extension
(``Aarambam.basis.legendre.cmbbest``): if ``import Aarambam`` fails because it is missing, the
package and ``Aarambam.basis`` are registered as plain namespaces so the pure-Python modules
(models, config, postprocess, decomposer, basis.binned, cli) import on their own.

Tests that need the compiled 2LPT executables and ``mpirun`` are marked ``mpi`` and are skipped
unless both are found (see ``lpt_binaries`` below and tests/test_c_smoke.py).

``AARAMBAM_TEST_INSTALLED=1`` tests the installed package instead of the source tree (its compiled
extension and executables; the GitHub workflow does this after ``pip install .``).
"""
import importlib, os, shutil, sys, types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_aarambam():
    if os.environ.get('AARAMBAM_TEST_INSTALLED') == '1':    #the installed package (CI): compiled extension, executables
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != ROOT]
        import Aarambam  # noqa: F401
        return
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        import Aarambam  # noqa: F401  (works when cmbbest is compiled or installed)
        return
    except ImportError:
        for k in [k for k in sys.modules if k == 'Aarambam' or k.startswith('Aarambam.')]:
            del sys.modules[k]
    for name, path in [('Aarambam', os.path.join(ROOT, 'Aarambam')),
                       ('Aarambam.basis', os.path.join(ROOT, 'Aarambam', 'basis'))]:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod


_import_aarambam()


def pytest_configure(config):
    config.addinivalue_line('markers', 'mpi: needs the compiled 2LPT executables and mpirun')


@pytest.fixture(scope='session')
def am():
    """Namespace with the pure-Python Aarambam modules."""
    return types.SimpleNamespace(
        utils      = importlib.import_module('Aarambam.utils'),
        config     = importlib.import_module('Aarambam.config'),
        postproc   = importlib.import_module('Aarambam.postprocess'),
        decomposer = importlib.import_module('Aarambam.decomposer'),
        models     = importlib.import_module('Aarambam.models'),
        binned     = importlib.import_module('Aarambam.basis.binned'),
        cli        = importlib.import_module('Aarambam.cli'))


@pytest.fixture(scope='session')
def defaults_dir():
    return os.path.join(ROOT, 'Aarambam', 'defaults')


def _find_binaries():
    """Directory holding 2LPTBasis and 2LPTBinned: $AARAMBAM_LPT_BIN, else the installed package."""
    cands = [os.environ.get('AARAMBAM_LPT_BIN', '')]
    try:
        from importlib.resources import files
        cands.append(str(files('Aarambam') / 'bin'))
    except Exception:
        pass
    for d in cands:
        if d and all(os.path.isfile(os.path.join(d, b)) for b in ('2LPTBasis', '2LPTBinned')):
            return d
    return None


@pytest.fixture(scope='session')
def lpt_binaries():
    d = _find_binaries()
    mpirun = os.environ.get('AARAMBAM_MPIRUN') or shutil.which('mpirun')
    if d is None or mpirun is None:
        pytest.skip('2LPT executables (set AARAMBAM_LPT_BIN) or mpirun not found')
    return types.SimpleNamespace(dir = d, mpirun = mpirun)
