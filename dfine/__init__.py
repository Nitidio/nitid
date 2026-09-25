"""Compatibility namespace for existing ``dfine`` imports.

New applications should import :class:`nitid.NITID` from :mod:`nitid`.
"""

from dfine.media import Frame, FrameMetadata, FrameSink, FrameSource
from dfine.model import DFINE
from dfine.nitid import NITID
from dfine.results import SemanticMask
from dfine.utils.reporting import BugReport, bugreport

__version__ = "0.1.0"
__all__ = [
    "BugReport",
    "DFINE",
    "NITID",
    "Frame",
    "FrameMetadata",
    "FrameSink",
    "FrameSource",
    "SemanticMask",
    "bugreport",
]
