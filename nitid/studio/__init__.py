"""Nitid Studio: a local web workbench served by the ``nitid`` package.

This package is the backend skeleton: settings, security guards, static
serving of the compiled single-page app, and a versioned ``/api/v1`` router.

Importing ``nitid.studio`` must stay cheap and must not import FastAPI or
uvicorn. They live in the optional ``studio`` extra
(``pip install "nitid[studio]"``) and are imported lazily by
:mod:`nitid.studio.launcher` and :mod:`nitid.studio.app`.
"""

from nitid.studio.settings import StudioSettings

__all__ = ["StudioSettings"]
