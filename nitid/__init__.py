"""Public Nitid package namespace.

``NITID`` is the supported entry point for new applications. Re-exports from
the internal ``dfine`` package preserve the existing Python API.
"""

from dfine import *  # noqa: F403
from dfine import __version__
from dfine.nitid import NITID, NitidModelSpec, parse_nitid_model_name

__all__ = [
    *[name for name in globals() if not name.startswith("_")],
    "__version__",
    "NITID",
    "NitidModelSpec",
    "parse_nitid_model_name",
]
