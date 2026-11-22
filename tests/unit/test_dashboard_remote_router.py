"""The tunnel router: several linked robots on one host share the tunnel port (4319).

Two real remote listeners (``remote_listener(create_app(...))`` under uvicorn, each with its
own enrolment and token check) sit behind a real router on a free port; requests arrive as the
robots' tunnels send them, one robot's ``Host`` per connection. The issuer and the WebSocket
peer are the network-boundary fakes of ``tests/unit/fakes/fake_openral_issuer.py``; the
half-close test's origin is ``tests/unit/fakes/fake_http_origin.py``.
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
import websockets
from openral_core import RobotEnrolment
from openral_observability.dashboard import TelemetryStore, create_app, remote_router
from openral_observability.dashboard.remote_auth import JwksCache, TokenVerifier, remote_listener
from openral_observability.dashboard.remote_router import request_host, route_host, serve_router

from tests.unit.fakes.fake_http_origin import slow_http_origin
from tests.unit.fakes.fake_openral_issuer import AUD, ISSUER, ORG, ROBOT, FakeBridge, FakeIssuer

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
        server = await serve_router({route_host(AUD): port_a, route_host(_OTHER_AUD): port_b}, 0)
        try:
            yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        finally:
            server.close()


def _headers(host: str, token: str | None = None) -> dict[str, str]:
    auth = {"Authorization": f"Bearer {token}"} if token else {}
    return {"Host": host, "Origin": _SITE, **auth}


@pytest.mark.asyncio
async def test_each_robots_tunnel_reaches_that_robot(issuer: FakeIssuer, tmp_path: Path) -> None:
    """Each robot's cloudflared is its own client: keep-alive requests stay on its robot."""
    async with (
        _scene(issuer, tmp_path) as base,
        httpx.AsyncClient(base_url=base) as tunnel_a,
        httpx.AsyncClient(base_url=base) as tunnel_b,
    ):
        for _ in range(3):  # one pooled connection each, several requests on it
            a = await tunnel_a.get("/api/config", headers=_headers(route_host(AUD), issuer.mint()))
            token_b = issuer.mint(aud=_OTHER_AUD, robot=_OTHER)
            b = await tunnel_b.get("/api/config", headers=_headers(route_host(_OTHER_AUD), token_b))
            assert (a.status_code, a.json()["robot_id"]) == (200, ROBOT), a.text
            assert (b.status_code, b.json()["robot_id"]) == (200, _OTHER), b.text


@pytest.mark.asyncio
async def test_a_robot_still_checks_every_token_itself(issuer: FakeIssuer, tmp_path: Path) -> None:
    async with _scene(issuer, tmp_path) as base:
        async with httpx.AsyncClient(base_url=base) as client:
            # Robot b's token on robot a's host: routed to a, which refuses its audience.
            crossed = await client.get(
                "/api/config", headers=_headers(route_host(AUD), issuer.mint(aud=_OTHER_AUD))
            )
        async with httpx.AsyncClient(base_url=base) as client:
            none = await client.get("/api/config", headers=_headers(route_host(_OTHER_AUD)))
        # A connection is routed by its first request: a later one naming another robot's
        # host still reaches the first robot, which refuses that robot's token.
        async with httpx.AsyncClient(base_url=base) as client:
            await client.get("/api/config", headers=_headers(route_host(AUD), issuer.mint()))
            switched = await client.get(
                "/api/config",
                headers=_headers(route_host(_OTHER_AUD), issuer.mint(aud=_OTHER_AUD, robot=_OTHER)),
            )
    assert crossed.status_code == 401
    assert none.status_code == 401
    assert switched.status_code == 401


@pytest.mark.asyncio
async def test_unknown_hosts_and_malformed_heads_are_refused(
    issuer: FakeIssuer, tmp_path: Path
) -> None:
    async with _scene(issuer, tmp_path) as base:
        port = int(base.rsplit(":", 1)[1])
        async with httpx.AsyncClient(base_url=base) as client:
            unknown = await client.get("/api/config", headers={"Host": "r-nobody.openral.com"})

        async def raw(head: bytes) -> bytes:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(head)
            await writer.drain()
            reply = await reader.read()
            writer.close()
            return reply

        bad = [
            await raw(b"not http\r\n\r\n"),
            await raw(b"GET / HTTP/1.1\r\nHost : r-k3x9.openral.com\r\n\r\n"),  # space before :
            await raw(b"GET / HTTP/1.1\r\nHost: r-k3x9.openral.com\r\n folded\r\n\r\n"),
            await raw(b"GET / HTTP/1.1\r\nHost: a\r\nHost: r-k3x9.openral.com\r\n\r\n"),
        ]
    assert unknown.status_code == 421
    assert all(reply.startswith(b"HTTP/1.1 400") for reply in bad), bad


@pytest.mark.asyncio
async def test_a_half_closed_request_still_gets_its_answer() -> None:
    """A client that ends its sending side after the request still reads the reply: the
    half-close is passed on, not turned into a close of the robot's side."""
    with slow_http_origin(delay_s=0.2) as origin:
        server = await serve_router({"r-a.openral.com": origin}, 0)
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server.sockets[0].getsockname()[1]
        )
        writer.write(b"GET / HTTP/1.1\r\nHost: r-a.openral.com\r\nConnection: close\r\n\r\n")
        writer.write_eof()
        reply = await asyncio.wait_for(reader.read(), 10)
        writer.close()
        server.close()
    assert reply.startswith(b"HTTP/1.") and b" 200 " in reply.split(b"\r\n")[0], reply[:80]
    assert reply.endswith(b"r-a.openral.com")


@pytest.mark.asyncio
async def test_an_idle_connection_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection that never sends a head doesn't hold the router (or its shutdown)."""
    monkeypatch.setattr(remote_router, "HEAD_TIMEOUT_S", 0.2)
    server = await serve_router({}, 0)
    reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
    assert await asyncio.wait_for(reader.read(), 5) == b""  # closed by the router
    writer.close()
    server.close()


@pytest.mark.asyncio
async def test_a_robot_whose_dashboard_is_down_is_a_502() -> None:
    server = await serve_router({"r-gone.openral.com": _free_port()}, 0)
    port = server.sockets[0].getsockname()[1]
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"http://127.0.0.1:{port}/api/config", headers={"Host": "r-gone.openral.com"}
        )
    server.close()
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_the_3d_views_websocket_flows_both_ways() -> None:
    """A WebSocket upgrade, spliced as is: frames both ways until a side closes."""
    with FakeBridge().running() as bridge:
        server = await serve_router({"r-a.openral.com": bridge.port}, 0)
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(
            f"ws://r-a.openral.com:{port}/",
            host="127.0.0.1",
            port=port,
            subprotocols=[websockets.Subprotocol("foxglove.websocket.v1")],
        ) as ws:
            for frame in ("tf", "urdf"):
                await ws.send(frame)
                assert await asyncio.wait_for(ws.recv(), 5) == frame
        server.close()


def test_the_host_is_read_strictly() -> None:
    assert request_host(b"GET / HTTP/1.1\r\nHost: R-K3X9.openral.com:443\r\n\r\n") == (
        "r-k3x9.openral.com"
    )
    for head in (
        b"GET / HTTP/1.1\r\nAccept: */*\r\n\r\n",  # no Host
        b"GET / HTTP/1.1\r\nHost:\r\n\r\n",  # an empty one
        b"GET /\r\nHost: a\r\n\r\n",  # not HTTP/1
        b"GET / HTTP/1.1\r\nHost\t: a\r\n\r\n",
    ):
        with pytest.raises(ValueError):
            request_host(head)
