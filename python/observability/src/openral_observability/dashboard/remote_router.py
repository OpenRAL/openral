"""One tunnel port for several linked robots on one host (a ``deploy sim`` scene).

Every robot's Cloudflare Tunnel ingress is set by openral.com to ``localhost:4319``
(:data:`~openral_observability.dashboard.tunnel.TUNNEL_PORT`), so on a host that runs
several linked robots (``openral deploy sim`` with a scene of several robots, each its own
openral.com robot) every tunnel reaches the same port. Each robot's dashboard opens its
remote listener on a port of its own instead (``$OPENRAL_DASHBOARD_REMOTE_PORT``, see
:data:`~openral_observability.dashboard.tunnel.REMOTE_PORT_ENV`), and this router holds
``127.0.0.1:4319``: it reads the ``Host`` of each connection's first request (the robot's own
``r-<id>.openral.com``, which cloudflared passes through) and splices the connection, unchanged,
to that robot's listener. Nothing is checked here: every robot verifies its own tokens.

Routed once per connection: each robot's tunnel is its own cloudflared, whose ingress admits
only that robot's host, so a connection only ever carries that robot's requests. A connection
whose later requests named another host would still reach the first robot, which refuses
their tokens (``aud``). A host no robot here answers to gets ``421 Misdirected Request``, a
malformed head ``400``, a robot whose dashboard is down ``502``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import re
from collections.abc import Mapping
from urllib.parse import urlsplit

import structlog

from openral_observability.dashboard.tunnel import TUNNEL_PORT

__all__ = ["HEAD_TIMEOUT_S", "request_host", "route_host", "serve_router"]

_logger = structlog.get_logger(__name__)

_HEAD_MAX = 64 * 1024  # a request head larger than this is refused
HEAD_TIMEOUT_S = 10.0
"""How long a new connection gets to send its first request head before it is closed."""
#: RFC 9112 §5: a field name is a token, followed directly by ":" (no whitespace before it).
_FIELD = re.compile(rb"^[!#$%&'*+.^_`|~0-9A-Za-z-]+:")
_REQUEST_LINE = re.compile(rb"^[A-Z]+ \S+ HTTP/1\.[01]$")


def route_host(aud: str) -> str:
    """The ``Host`` a robot's tunnel requests carry: its enrolment ``aud``'s hostname.

    Example:
        >>> route_host("https://r-a1a1a.openral.com")
        'r-a1a1a.openral.com'
    """
    return (urlsplit(aud).hostname or "").lower()


def request_host(head: bytes) -> str:
    r"""The host a request head names (lower case, no port), read as strictly as its robot will.

    Raises:
        ValueError: Not an HTTP/1 request head: a bad request line, a field line that is not
            ``name: value`` (whitespace before the colon, an obsolete folded line), or not
            exactly one ``Host``.

    Example:
        >>> request_host(b"GET /api/state HTTP/1.1\r\nHost: R-A1.openral.com:443\r\n\r\n")
        'r-a1.openral.com'
    """
    lines = head.removesuffix(b"\r\n\r\n").split(b"\r\n")
    if not _REQUEST_LINE.match(lines[0]):
        raise ValueError("not an HTTP/1 request line")
    hosts = []
    for line in lines[1:]:
        if not _FIELD.match(line):
            raise ValueError("a malformed field line")
        name, _, value = line.partition(b":")
        if name.lower() == b"host":
            hosts.append(value.strip().decode("latin-1"))
    if len(hosts) != 1 or not hosts[0]:
        raise ValueError("not exactly one Host")
    return (urlsplit(f"//{hosts[0]}").hostname or "").lower()


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy one direction until its EOF, then pass the half-close on."""
    with contextlib.suppress(ConnectionError, OSError):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()


async def _refuse(writer: asyncio.StreamWriter, status: str) -> None:
    body = status.encode()
    writer.write(
        f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
        + body
    )
    with contextlib.suppress(ConnectionError, OSError):
        await writer.drain()


async def _handle(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, routes: Mapping[str, int]
) -> None:
    up_writer: asyncio.StreamWriter | None = None
    try:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT_S)
        except TimeoutError:
            return  # an idle connection: closed
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            await _refuse(writer, "400 Bad Request")
            return
        try:
            host = request_host(head)
        except ValueError:
            await _refuse(writer, "400 Bad Request")
            return
        port = routes.get(host)
        if port is None:
            _logger.warning("dashboard.router_unknown_host", host=host)
            await _refuse(writer, "421 Misdirected Request")
            return
        try:
            up_reader, up_writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await _refuse(writer, "502 Bad Gateway")  # that robot's dashboard is down
            return
        up_writer.write(head)  # the head as sent: the robot reads exactly what the client wrote
        await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))
    finally:
        for w in (writer, up_writer):
            if w is not None:
                with contextlib.suppress(ConnectionError, OSError):
                    w.close()


async def serve_router(routes: Mapping[str, int], port: int = TUNNEL_PORT) -> asyncio.Server:
    """Listen on ``127.0.0.1:port``, splicing each connection to ``routes[host]``.

    ``routes``: each robot's tunnel host (:func:`route_host`) to its remote listener port.

    Raises:
        OSError: ``port`` is taken (another dashboard holds the tunnel port).
    """
    table = {h.lower(): p for h, p in routes.items()}
    return await asyncio.start_server(
        lambda r, w: _handle(r, w, table), "127.0.0.1", port, limit=_HEAD_MAX
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m openral_observability.dashboard.remote_router --route HOST=PORT ...``.

    Exits 1 (logged) when the port is taken. On SIGINT it stops at once: open connections
    (a 3D view's WebSocket) are closed, not waited for.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--route", action="append", default=[], metavar="HOST=PORT")
    parser.add_argument("--port", type=int, default=TUNNEL_PORT)
    args = parser.parse_args(argv)
    routes = {h: int(p) for h, _, p in (r.partition("=") for r in args.route)}

    async def run() -> int:
        try:
            server = await serve_router(routes, args.port)
        except OSError as exc:
            _logger.error("dashboard.router_port_taken", port=args.port, error=str(exc))
            return 1
        _logger.info("dashboard.router_up", port=args.port, routes=routes)
        try:
            await server.serve_forever()
        finally:
            # Not `async with server`: its wait_closed would wait for every open connection.
            # asyncio.run then cancels their handlers, whose `finally` closes both sockets.
            server.close()
        return 0

    with contextlib.suppress(KeyboardInterrupt):
        return asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
