"""The robot console's ⚙ relays openral.com's Robot settings (``/api/robot-settings``).

Robot settings have one source of truth, openral.com: the dashboard forwards the body
unchanged to ``<site>/api/robot/console`` with the robot's own credential and returns
the answer as it is. openral.com is an ``httpx.MockTransport`` (a network-boundary
double); the robot's enrolment and ``robot.secret`` are real files, as
``openral robot link`` writes them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from openral_core import REMOTE_CONTROL_ENV, RobotEnrolment
from openral_observability.dashboard import TelemetryStore, create_app
from openral_observability.dashboard.remote_auth import JwksCache, TokenVerifier, remote_listener

from tests.unit.fakes.fake_openral_issuer import (
    AUD,
    ISSUER,
    ISSUER_ORIGIN,
    ORG,
    ROBOT,
    FakeIssuer,
)

_SECRET = "ors_test-robot-credential-0000"
_CONSOLE = {
    "scope": "robot",
    "canWrite": True,
    "settings": {},
    "values": {"OPENRAL_REASONER_MODEL": {"set": True, "value": "claude-sonnet-5-5"}},
    "keys": [],
    "inherited": {},
    "robot": {"id": ROBOT, "name": "so101", "org": ORG, "fleet": None},
}


class _Site:
    """openral.com's ``/api/robot/console``: records each request, answers ``status``/``body``.

    Installed as the transport of every ``httpx.AsyncClient`` that names none (the relay's);
    the test's own clients name theirs. ``down`` makes it unreachable.
    """

    def __init__(self) -> None:
        self.seen: list[httpx.Request] = []
        self.status = 200
        self.body: bytes = json.dumps(_CONSOLE).encode()
        self.down = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/api/robot/console":  # the fake issuer's JWKS, on loopback
            with httpx.Client(transport=httpx.HTTPTransport()) as real:
                answer = real.get(str(request.url))
            return httpx.Response(answer.status_code, content=answer.content)
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        self.seen.append(request)
        return httpx.Response(self.status, content=self.body)


@pytest.fixture
def site(monkeypatch: pytest.MonkeyPatch) -> _Site:
    fake = _Site()
    real = httpx.AsyncClient

    class Client(real):  # type: ignore[misc,valid-type] # reason: subclassing a patched name
        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("transport", httpx.MockTransport(fake))
            super().__init__(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    return fake


def _link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The files ``openral robot link`` leaves: the enrolment and ``robot.secret`` beside it."""
    path = tmp_path / "config" / "robot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    enrolment = RobotEnrolment(
        v=1, robot=ROBOT, aud=AUD, org=ORG, iss=ISSUER, jwks=f"{ISSUER}/jwks"
    )
    path.write_text(enrolment.model_dump_json())
    path.with_name("robot.secret").write_text(_SECRET + "\n")
    monkeypatch.setenv("OPENRAL_ROBOT_ENROLMENT", str(path))


def _local(client: str = "127.0.0.1") -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(TelemetryStore()), client=(client, 50000))
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:4318", timeout=15.0)


@pytest.mark.asyncio
async def test_each_method_is_forwarded_unchanged_with_the_robots_credential(
    site: _Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _link(tmp_path, monkeypatch)
    put = {"name": "OPENRAL_REASONER_MODEL", "value": "claude-opus-5-5"}
    async with _local() as c:
        got = await c.get("/api/robot-settings")
        assert (got.status_code, got.json()) == (200, _CONSOLE)
        assert (await c.put("/api/robot-settings", json=put)).status_code == 200
        assert (await c.patch("/api/robot-settings", json={"name": "so101-b"})).status_code == 200
        # The path the pinned console still calls relays the same way.
        assert (await c.get("/api/settings")).json() == _CONSOLE
    assert [
        (r.method, str(r.url), r.headers["authorization"], r.headers["user-agent"])
        for r in site.seen
    ] == [
        (m, f"{ISSUER_ORIGIN}/api/robot/console", f"Bearer {_SECRET}", "openral-robot")
        for m in ("GET", "PUT", "PATCH", "GET")
    ]
    assert json.loads(site.seen[1].content) == put
    assert json.loads(site.seen[2].content) == {"name": "so101-b"}


@pytest.mark.asyncio
async def test_openral_coms_status_and_json_come_back_as_they_are(
    site: _Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _link(tmp_path, monkeypatch)
    site.status, site.body = 403, b'{"error":"only an org admin can change this"}'
    async with _local() as c:
        r = await c.put("/api/robot-settings", json={"name": "OPENRAL_RSKILLS", "value": "x"})
    assert (r.status_code, r.json()) == (403, {"error": "only an org admin can change this"})

    site.status, site.body = 204, b""
    async with _local() as c:
        r = await c.patch("/api/robot-settings", json={"name": "so101-b"})
    assert (r.status_code, r.content) == (204, b"")


@pytest.mark.asyncio
async def test_a_refused_robot_credential_is_409_never_a_token_401(
    site: _Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The console takes a 401 for its own expired token and mints another."""
    _link(tmp_path, monkeypatch)
    site.status, site.body = 401, b'{"error":"unknown robot credential"}'
    async with _local() as c:
        r = await c.get("/api/robot-settings")
    assert r.status_code == 409
    assert "link it again" in r.json()["error"]


@pytest.mark.asyncio
async def test_an_unlinked_robot_answers_409_and_an_unreachable_site_502(
    site: _Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_ROBOT_ENROLMENT", str(tmp_path / "none" / "robot.json"))
    async with _local() as c:
        r = await c.get("/api/robot-settings")
    assert (r.status_code, r.json()) == (409, {"error": "this robot isn't linked to openral.com"})

    _link(tmp_path, monkeypatch)
    site.down = True
    async with _local() as c:
        assert (await c.get("/api/robot-settings")).status_code == 502


@pytest.mark.asyncio
async def test_the_lan_may_read_but_not_change_and_another_site_may_not_change(
    site: _Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _link(tmp_path, monkeypatch)
    async with _local("192.168.1.20") as lan:
        got = await lan.get("/api/robot-settings")
        assert (got.status_code, got.json()["canWrite"]) == (200, False)
        put = await lan.put("/api/robot-settings", json={"name": "OPENRAL_RSKILLS", "value": ""})
        assert put.status_code == 403
        assert (await lan.patch("/api/robot-settings", json={"name": "x"})).status_code == 403
    async with _local() as c:
        csrf = await c.patch(
            "/api/robot-settings", json={"name": "x"}, headers={"Origin": "https://evil.example"}
        )
        assert csrf.status_code == 403
    assert [r.method for r in site.seen] == ["GET"]


@pytest.fixture
def issuer() -> Iterator[FakeIssuer]:
    with FakeIssuer().serving() as served:
        yield served


@pytest.mark.asyncio
async def test_the_tunnel_reads_only_and_only_with_remote_control_on(
    issuer: FakeIssuer,
    site: _Site,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _link(tmp_path, monkeypatch)
    verifier = TokenVerifier(
        RobotEnrolment(v=1, robot=ROBOT, aud=AUD, org=ORG, iss=ISSUER, jwks=issuer.url),
        JwksCache(issuer.url, tmp_path / "jwks.json"),
    )
    app = remote_listener(create_app(TelemetryStore(), verifier=verifier))
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 40000))

    def auth() -> dict[str, str]:
        token = issuer.mint(role="operator")
        return {"Authorization": f"Bearer {token}", "Origin": "https://www.openral.com"}

    async with httpx.AsyncClient(transport=transport, base_url=AUD, timeout=15.0) as c:
        monkeypatch.setenv(REMOTE_CONTROL_ENV, "0")
        assert (await c.get("/api/robot-settings", headers=auth())).status_code == 403
        off = await c.patch("/api/robot-settings", json={"name": "x"}, headers=auth())
        assert off.status_code == 403
        assert site.seen == []
        monkeypatch.setenv(REMOTE_CONTROL_ENV, "1")
        got = await c.get("/api/robot-settings", headers=auth())
        assert (got.status_code, got.json()["canWrite"]) == (200, False)
        # A write would use the robot's credential past openral.com's check of the person.
        on = await c.patch("/api/robot-settings", json={"name": "so101-b"}, headers=auth())
        assert on.status_code == 403
    assert [r.method for r in site.seen] == ["GET"]


@pytest.mark.asyncio
async def test_a_robot_linked_again_since_the_dashboard_started_is_not_relayed(
    issuer: FakeIssuer, site: _Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate admits callers for the link it started with; the new link must not answer."""
    _link(tmp_path, monkeypatch)
    started_with = RobotEnrolment(
        v=1, robot="old1", aud="https://r-old1.openral.com", org=ORG, iss=ISSUER, jwks=issuer.url
    )
    verifier = TokenVerifier(started_with, JwksCache(issuer.url, tmp_path / "jwks.json"))
    transport = httpx.ASGITransport(
        app=create_app(TelemetryStore(), verifier=verifier), client=("127.0.0.1", 50000)
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1:4318", timeout=15.0
    ) as c:
        r = await c.get("/api/robot-settings")
    assert r.status_code == 409
    assert "linked again" in r.json()["error"]
    assert site.seen == []
