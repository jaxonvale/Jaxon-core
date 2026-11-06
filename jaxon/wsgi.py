"""Production entry point: `gunicorn jaxon.wsgi:app`.

Refuses to start without JAXON_TOKEN, so a public deployment is never left open.
"""

import os

from .server import create_app

if not os.environ.get("JAXON_TOKEN"):
    raise RuntimeError("Set JAXON_TOKEN (a long random password) before running Jaxon on a server.")

app = create_app()
