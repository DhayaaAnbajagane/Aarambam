# Aarambam/basis/__init__.py
from . import legendre, binned

from .legendre import BasicLegendreDecompose
from .binned   import BasicBinnedDecompose

__all__ = ['legendre', 'binned', 'BasicLegendreDecompose', 'BasicBinnedDecompose']
