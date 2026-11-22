from . import utils, models
from .basis import legendre, binned

try:
    from importlib.metadata import version as _version, PackageNotFoundError
    __version__ = _version("Aarambam")
except Exception: #Not installed (e.g. imported from a source checkout)
    __version__ = "1.1.0"
