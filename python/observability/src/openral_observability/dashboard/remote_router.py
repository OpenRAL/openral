"""One tunnel port for several linked robots on one host (a ``deploy sim`` scene).

Every robot's Cloudflare Tunnel ingress is set by openral.com to ``localhost:4319``
(:data:`~openral_observability.dashboard.tunnel.TUNNEL_PORT`), so on a host that runs
several linked robots (``openral deploy sim`` with a scene of several robots, each its own
openral.com robot) every tunnel reaches the same port. Each robot's dashboard opens its
remote listener on a port of its own instead (``$OPENRAL_DASHBOARD_REMOTE_PORT``, see
:data:`~openral_observability.dashboard.tunnel.REMOTE_PORT_ENV`), and this router holds
``127.0.0.1:4319``: it reads each request's ``Host`` (the robot's own ``r-<id>.openral.com``,
which cloudflared passes through) and splices the connection to that robot's listener.
Nothing is checked here: every robot verifies its own tokens.

One request per connection: the request is sent on with ``Connection: close``, so the robot
answers it and closes, and the next request (on a new connection) is routed afresh. A
WebSocket upgrade keeps its connection, spliced both ways until either side closes. A host
no robot here answers to gets ``421 Misdirected Request``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from collections.abc import Mapping
from urllib.parse import urlsplit

import structlog

from openral_observability.dashboard.tunnel import TUNNEL_PORT

__all__ = ["route_host", "routed_head", "serve_router"]

_logger = structlog.get_logger(__name__)

_HEAD_MAX = 64 * 1024  # a request head larger than this is refused
_HOP = ("connection", "keep-alive", "proxy-connection")


def route_host(aud: str) -> str:
    """The ``Host`` a robot's tunnel requests carry: its enrolment ``aud``'s hostname.

    Example:
        >>> route_host("https://r-a1a1a.openral.com")
        'r-a1a1a.openral.com'
    """
    return (urlsplit(aud).hostname or "").lower()


def routed_head(head: bytes) -> tuple[str, bytes]:
    r"""The request's host (lower case, no port) and its head as sent on to the robot.

    A plain request goes on with ``Connection: close`` (one request per connection); a
    WebSocket upgrade keeps its ``Connection: Upgrade``.

    Raises:
        ValueError: Not an HTTP/1 request head, or no ``Host``.

    Example:
        >>> host, out = routed_head(
        ...     b"GET /api/state HTTP/1.1\r\nHost: R-A1.openral.com:443\r\n"
        ...     b"Connection: keep-alive\r\n\r\n"
        ... )
        >>> host, out.decode().split("\r\n")[1:3]
        ('r-a1.openral.com', ['Host: R-A1.openral.com:443', 'Connection: close'])
    """
    text = head.decode("latin-1")
    lines = text.split("\r\n")
    if not lines[1:] or not lines[0].endswith(("HTTP/1.1", "HTTP/1.0")):
        raise ValueError("not an HTTP/1 request")
    fields = [
        (k.strip(), v.strip()) for k, _, v in (line.partition(":") for line in lines[1:] if line)
    ]
    host = next((v for k, v in fields if k.lower() == "host"), "")
    if not host:
        raise ValueError("no Host")
    upgrade = any(k.lower() == "upgrade" for k, _ in fields) and any(
        k.lower() == "connection" and "upgrade" in v.lower() for k, v in fields
    )
    if not upgrade:
        fields = [(k, v) for k, v in fields if k.lower() not in _HOP] + [("Connection", "close")]
    out = "\r\n".join([lines[0], *(f"{k}: {v}" for k, v in fields)]) + "\r\n\r\n"
    return urlsplit(f"//{host}").hostname or "", out.encode("latin-1")


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    with contextlib.suppress(ConnectionError, asyncio.IncompleteReadError):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    with contextlib.suppress(ConnectionError, OSError):
        writer.close()


async def _refuse(writer: asyncio.StreamWriter, status: str) -> None:
    body = status.encode()
    writer.write(
        f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
        + body
    )
    with contextlib.suppress(ConnectionError, OSError):
        await writer.drain()
        writer.close()


async def _handle(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, routes: Mapping[str, int]
) -> None:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
        host, out = routed_head(head)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
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
    up_writer.write(out)
    await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))


async def serve_router(routes: Mapping[str, int], port: int = TUNNEL_PORT) -> asyncio.Server:
    """Listen on ``127.0.0.1:port``, splicing each request to ``routes[host]``.

    ``routes``: each robot's tunnel host (:func:`route_host`) to its remote listener port.

    Raises:
        OSError: ``port`` is taken (another dashboard holds the tunnel port).
    """
    table = {h.lower(): p for h, p in routes.items()}
    return await asyncio.start_server(
        lambda r, w: _handle(r, w, table), "127.0.0.1", port, limit=_HEAD_MAX
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m openral_observability.dashboard.remote_router --route HOST=PORT ...``."""
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
        async with server:
            await server.serve_forever()
        return 0

    with contextlib.suppress(KeyboardInterrupt):
        return asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
