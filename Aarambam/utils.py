"""Backwards-compatible namespace: everything here lives in config, postprocess and decomposer."""
from .config import *
from .config import __all__ as _config_all
from .postprocess import *
from .postprocess import __all__ as _post_all
from .decomposer import Decomposer

__all__ = _config_all + _post_all + ['Decomposer']
