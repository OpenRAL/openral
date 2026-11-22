"""A slow HTTP origin on 127.0.0.1 (stdlib ``http.server``) for proxy tests.

Answers every ``GET`` with ``200`` and the request's ``Host`` after ``delay_s``. It reads the
whole request before answering, so a client that half-closes its side after sending still gets
the reply: what ``remote_router`` must pass on (uvicorn drops such a connection while a reply
is pending, so it can't stand in here).
"""

from __future__ import annotations

import http.server
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def slow_http_origin(delay_s: float = 0.2) -> Iterator[int]:
    """Serve on a free port for the block; yields the port."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            time.sleep(delay_s)
            body = (self.headers.get("Host") or "").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
