"""The tunnel router: several linked robots on one host share the tunnel port (4319).

Two real remote listeners (``remote_listener(create_app(...))`` under uvicorn, each with its
own enrolment and token check) sit behind a real router on a free port; requests arrive as
the robots' tunnels send them, with each robot's own ``Host``. The issuer is the network-
boundary fake (``tests/unit/fakes/fake_openral_issuer.py``); the upgrade test's backend is a
raw socket peer standing in for ``foxglove_bridge``'s WebSocket.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn
from openral_core import RobotEnrolment
from openral_observability.dashboard import TelemetryStore, create_app
from openral_observability.dashboard.remote_auth import JwksCache, TokenVerifier, remote_listener
from openral_observability.dashboard.remote_router import route_host, routed_head, serve_router

from tests.unit.fakes.fake_openral_issuer import AUD, ISSUER, ORG, ROBOT, FakeIssuer

_OTHER = "b2b2b"
_OTHER_AUD = f"https://r-{_OTHER}.openral.com"
_SITE = "https://www.openral.com"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def issuer() -> Iterator[FakeIssuer]:
    with FakeIssuer().serving() as served:
        yield served


@contextlib.asynccontextmanager
async def _listener(issuer: FakeIssuer, tmp_path: Path, robot: str, aud: str) -> AsyncIterator[int]:
    """One robot's remote listener on a free port, as its dashboard opens it."""
    enrolment = RobotEnrolment(v=1, robot=robot, aud=aud, org=ORG, iss=ISSUER, jwks=issuer.url)
    verifier = TokenVerifier(enrolment, JwksCache(issuer.url, tmp_path / f"{robot}-jwks.json"))
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            remote_listener(create_app(TelemetryStore(), verifier=verifier)),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
        )
    )
    task = asyncio.create_task(server.serve())
    while not server.started:  # noqa: ASYNC110  # reason: uvicorn exposes only a started flag
        await asyncio.sleep(0.02)
    try:
        yield port
    finally:
        server.should_exit = True
        await task


@contextlib.asynccontextmanager
async def _scene(issuer: FakeIssuer, tmp_path: Path) -> AsyncIterator[str]:
    """Two linked robots behind the router; yields the router's base URL."""
    async with (
        _listener(issuer, tmp_path, ROBOT, AUD) as port_a,
        _listener(issuer, tmp_path, _OTHER, _OTHER_AUD) as port_b,
    ):
        router_port = _free_port()
        routes = {route_host(AUD): port_a, route_host(_OTHER_AUD): port_b}
        server = await serve_router(routes, router_port)
        async with server:
            yield f"http://127.0.0.1:{router_port}"


def _headers(host: str, token: str) -> dict[str, str]:
    return {"Host": host, "Authorization": f"Bearer {token}", "Origin": _SITE}


@pytest.mark.asyncio
async def test_each_robots_requests_reach_that_robot(issuer: FakeIssuer, tmp_path: Path) -> None:
    async with _scene(issuer, tmp_path) as base, httpx.AsyncClient(base_url=base) as client:
        token_a = issuer.mint(role="viewer")
        token_b = issuer.mint(role="viewer", aud=_OTHER_AUD, robot=_OTHER)
        a = await client.get("/api/config", headers=_headers(route_host(AUD), token_a))
        b = await client.get("/api/config", headers=_headers(route_host(_OTHER_AUD), token_b))
        # One pooled connection could carry both: each request is answered and closed, so the
        # second is routed afresh (``Connection: close``).
        again = await client.get("/api/config", headers=_headers(route_host(AUD), issuer.mint()))
    assert (a.status_code, a.json()["robot_id"]) == (200, ROBOT), a.text
    assert (b.status_code, b.json()["robot_id"]) == (200, _OTHER), b.text
    assert (again.status_code, again.json()["robot_id"]) == (200, ROBOT), again.text


@pytest.mark.asyncio
async def test_a_robot_still_checks_every_token_itself(issuer: FakeIssuer, tmp_path: Path) -> None:
    async with _scene(issuer, tmp_path) as base, httpx.AsyncClient(base_url=base) as client:
        # Robot b's token sent to robot a's host: routed to a, which refuses its audience.
        crossed = await client.get(
            "/api/config", headers=_headers(route_host(AUD), issuer.mint(aud=_OTHER_AUD))
        )
        none = await client.get("/api/config", headers={"Host": route_host(_OTHER_AUD)})
    assert crossed.status_code == 401
    assert none.status_code == 401


@pytest.mark.asyncio
async def test_unknown_hosts_and_bad_heads_are_refused(issuer: FakeIssuer, tmp_path: Path) -> None:
    async with _scene(issuer, tmp_path) as base, httpx.AsyncClient(base_url=base) as client:
        unknown = await client.get("/api/config", headers={"Host": "r-nobody.openral.com"})
        reader, writer = await asyncio.open_connection("127.0.0.1", int(base.rsplit(":", 1)[1]))
        writer.write(b"not http\r\n\r\n")
        await writer.drain()
        bad = await reader.read()
        writer.close()
    assert unknown.status_code == 421
    assert bad.startswith(b"HTTP/1.1 400")


@pytest.mark.asyncio
async def test_a_robot_whose_dashboard_is_down_is_a_502() -> None:
    server = await serve_router({"r-gone.openral.com": _free_port()}, _free_port())
    port = server.sockets[0].getsockname()[1]
    async with server, httpx.AsyncClient() as client:
        resp = await client.get(
            f"http://127.0.0.1:{port}/api/config", headers={"Host": "r-gone.openral.com"}
        )
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_an_upgrade_keeps_its_connection_both_ways() -> None:
    """The 3D view's WebSocket: after the 101 both directions flow until a side closes."""
    received: list[bytes] = []

    async def bridge(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        received.append(head)
        writer.write(
            b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n"
        )
        while data := await reader.read(1024):
            writer.write(data.upper())  # an echo, so both directions are seen
            await writer.drain()
        writer.close()

    backend = await asyncio.start_server(bridge, "127.0.0.1", 0)
    router = await serve_router({"r-a.openral.com": backend.sockets[0].getsockname()[1]}, 0)
    async with backend, router:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", router.sockets[0].getsockname()[1]
        )
        writer.write(
            b"GET /ws HTTP/1.1\r\nHost: r-a.openral.com\r\nConnection: Upgrade\r\n"
            b"Upgrade: websocket\r\n\r\n"
        )
        status = await reader.readuntil(b"\r\n\r\n")
        for frame in (b"tf", b"urdf"):
            writer.write(frame)
            await writer.drain()
            assert await reader.readexactly(len(frame)) == frame.upper()
        writer.close()
    assert status.startswith(b"HTTP/1.1 101")
    assert b"Connection: Upgrade" in received[0]  # the upgrade is kept, not closed


def test_a_plain_request_goes_on_with_connection_close() -> None:
    host, out = routed_head(
        b"POST /api/actions/estop HTTP/1.1\r\nHost: r-k3x9.openral.com\r\n"
        b"Connection: keep-alive\r\nKeep-Alive: timeout=5\r\nContent-Length: 2\r\n\r\n"
    )
    lines = out.decode().split("\r\n")
    assert host == "r-k3x9.openral.com"
    assert "Connection: close" in lines and "Keep-Alive: timeout=5" not in lines
    assert "Content-Length: 2" in lines  # the body that follows is spliced as is
    with pytest.raises(ValueError, match="no Host"):
        routed_head(b"GET / HTTP/1.1\r\nAccept: */*\r\n\r\n")
