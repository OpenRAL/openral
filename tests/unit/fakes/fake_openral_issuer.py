"""Network-boundary doubles for the dashboard's remote listener (ADR-0129).

* ``FakeIssuer`` — openral.com's token side: a real ES256 key pair (``cryptography``),
  its JWKS served over real HTTP on 127.0.0.1 (counting fetches), and ``mint`` for
  robot tokens shaped as Interface B specifies. Replaces ``GET <iss>/jwks`` and the
  website's ``POST /api/robot-token``; the dashboard's verifier runs unmodified.
* ``FakeBridge`` — ``foxglove_bridge``'s WebSocket on 127.0.0.1: accepts the
  ``foxglove.websocket.v1`` subprotocol and echoes every frame, so a test can prove
  ``/ws`` really relays both ways.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from websockets.asyncio.server import ServerConnection, serve
from websockets.typing import Subprotocol

ISSUER = "https://www.openral.com/api/auth"
ROBOT = "k3x9"
AUD = f"https://r-{ROBOT}.openral.com"
ORG = "acme"


class FakeIssuer:
    """openral.com's signing key and JWKS endpoint."""

    def __init__(self) -> None:
        """Generate a key; serve nothing until ``serving``."""
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.kid = "kid-1"
        self.fetches = 0
        self.url = ""

    def jwks(self) -> dict[str, Any]:
        """The public JWKS, as Better Auth's ``/api/auth/jwks`` serves it."""
        jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(self.key.public_key()))
        return {"keys": [{**jwk, "kid": self.kid, "alg": "ES256", "use": "sig"}]}

    @contextmanager
    def serving(self) -> Iterator[FakeIssuer]:
        """Serve the JWKS at ``self.url`` for the duration of the block."""
        issuer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                issuer.fetches += 1
                body = json.dumps(issuer.jwks()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.url = f"http://127.0.0.1:{server.server_address[1]}/api/auth/jwks"
        try:
            yield self
        finally:
            server.shutdown()
            server.server_close()

    def mint(
        self, *, header: dict[str, Any] | None = None, lifetime_s: int = 300, **claims: Any
    ) -> str:
        """A robot token for ``k3x9`` of org ``acme``; keyword args override claims."""
        now = int(time.time())
        payload: dict[str, Any] = {
            "iss": ISSUER,
            "aud": AUD,
            "sub": "user-7f3a",
            "client_id": "openral-web",
            "org": ORG,
            "role": "operator",
            "robot": ROBOT,
            "jti": "tok-91",
            "iat": now,
            "exp": now + lifetime_s,
        }
        payload.update(claims)
        payload = {k: v for k, v in payload.items() if v is not None}
        return jwt.encode(
            payload,
            self.key,
            algorithm="ES256",
            headers={"kid": self.kid, "typ": None, **(header or {})},
        )


class FakeBridge:
    """A ``foxglove_bridge`` stand-in that echoes every frame."""

    def __init__(self) -> None:
        """Not listening until ``running``."""
        self.port = 0

    @contextmanager
    def running(self) -> Iterator[FakeBridge]:
        """Listen on 127.0.0.1 in a background event loop for the duration of the block."""
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        stop: asyncio.Future[None] | None = None

        async def echo(conn: ServerConnection) -> None:
            async for message in conn:
                await conn.send(message)

        async def main() -> None:
            nonlocal stop
            stop = loop.create_future()
            # A short close timeout: shutdown must not outwait the join below on a
            # connection the client side dropped without a close frame.
            async with serve(
                echo,
                "127.0.0.1",
                0,
                subprotocols=[Subprotocol("foxglove.websocket.v1")],
                close_timeout=0.5,
            ) as server:
                self.port = next(iter(server.sockets)).getsockname()[1]
                ready.set()
                await stop

        thread = threading.Thread(target=loop.run_until_complete, args=(main(),), daemon=True)
        thread.start()
        ready.wait(5.0)
        try:
            yield self
        finally:
            assert stop is not None
            loop.call_soon_threadsafe(stop.set_result, None)
            thread.join(5.0)
            loop.close()
