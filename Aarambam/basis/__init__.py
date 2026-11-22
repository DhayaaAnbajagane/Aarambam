# Aarambam/basis/__init__.py
import types as _types

from . import binned
from .binned import BasicBinnedDecompose

try:
    from . import legendre
    from .legendre import BasicLegendreDecompose
except ImportError as _err:     #without the compiled extension (Aarambam_PYTHON_EXTENSION=OFF, or a source checkout)
    if 'cmbbest' not in f"{getattr(_err, 'name', '')} {_err}":
        raise
    _msg = ('the Legendre decomposition needs the compiled extension Aarambam.basis.legendre.cmbbest, which is not '
            f'installed ({_err}); reinstall Aarambam with Aarambam_PYTHON_EXTENSION=ON (the default)')

    class _NoLegendre(_types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('__'):
                raise AttributeError(name)
            raise ImportError(_msg)

    legendre = _NoLegendre('Aarambam.basis.legendre', _msg)

    class BasicLegendreDecompose:
        def __init__(self, *args, **kwargs):
            raise ImportError(_msg)

__all__ = ['legendre', 'binned', 'BasicLegendreDecompose', 'BasicBinnedDecompose']
