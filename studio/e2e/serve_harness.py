"""Serve the parity harness with the real Studio app (same middleware, headers and API).

Identical to ``nitid studio`` except that the static files come from a directory given on the
command line, so the browser engine harness (``harness/``) can be loaded under the exact
CSP/COOP/COEP headers of the product and call the real ``/api/v1/onnx`` endpoint.

    PYTHONPATH=<repo> python studio/e2e/serve_harness.py MODELS_DIR STATIC_DIR

Prints ``Open: http://127.0.0.1:PORT/#t=TOKEN`` once listening, like ``nitid studio``.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from nitid.studio.app import create_app
from nitid.studio.launcher import access_url, bind_socket, make_server
from nitid.studio.settings import StudioSettings


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    models_dir, static_dir = Path(argv[1]), Path(argv[2])
    settings = StudioSettings(port=0, models_dir=models_dir, open_browser=False)
    sock = bind_socket(settings.host, settings.port)
    settings = dataclasses.replace(settings, port=sock.getsockname()[1])
    server = make_server(create_app(settings, static_dir=static_dir), log_level="warning")
    print(f"Open: {access_url(settings)}", flush=True)
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
