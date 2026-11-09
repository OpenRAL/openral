"""The dashboard's remote listener: openral.com tokens, CORS, roles, the 3D proxy (ADR-0129).

Requests go through ``remote_listener(app)`` exactly as the tunnel's do. The issuer
(ES256 key + JWKS over real HTTP) and ``foxglove_bridge`` are network-boundary fakes
(``tests/unit/fakes/fake_openral_issuer.py``); ``ros2`` is a real executable on PATH
that logs its argv. Everything between them is the production code.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import stat
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import FastAPI
from openral_core import (
    DeployRuntime,
    RobotDescription,
    RobotEnrolment,
    RSkillManifest,
    build_robot_manifest,
    pseudonymous_actor_id,
)
from openral_observability import semconv
from openral_observability.dashboard import TelemetryStore, create_app
from openral_observability.dashboard.remote_auth import JwksCache, TokenVerifier, remote_listener
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.proto.logs.v1.logs_pb2 import LogRecord, ResourceLogs, ScopeLogs
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.unit.fakes.fake_openral_issuer import AUD, ISSUER, ORG, ROBOT, FakeBridge, FakeIssuer

_REPO = Path(__file__).resolve().parents[2]
_SKILL = "OpenRAL/rskill-smolvla-so101-eraser_place-bf16"
_SITE = "https://www.openral.com"


@pytest.fixture
def issuer() -> Iterator[FakeIssuer]:
    with FakeIssuer().serving() as served:
        yield served


@pytest.fixture
def ros2_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `ros2` on PATH that logs its argv and accepts goals / resets."""
    log = tmp_path / "ros2.jsonl"
    shim = tmp_path / "ros2"
    shim.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:3] == ['action', 'send_goal']:\n"
        "    print('Goal accepted with ID: g-1', flush=True)\n"
        "elif sys.argv[1:3] == ['service', 'call']:\n"
        "    print(\"response: success=True, message=''\", flush=True)\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("OPENRAL_ACTOR_KEY", "deployer-key")
    return log


def _announced(log: Path) -> list[dict[str, str]]:
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    return [json.loads(c[-1]) for c in calls if "/openral/operator_action" in c]


def _verifier(issuer: FakeIssuer, tmp_path: Path, *, cooldown_s: float = 60.0) -> TokenVerifier:
    enrolment = RobotEnrolment(v=1, robot=ROBOT, aud=AUD, org=ORG, iss=ISSUER, jwks=issuer.url)
    return TokenVerifier(
        enrolment, JwksCache(issuer.url, tmp_path / "jwks.json", cooldown_s=cooldown_s)
    )


def _store_with_manifest() -> TelemetryStore:
    """A store holding a real so101_follower manifest, ingested as the runtime sends it."""
    robot = RobotDescription.from_yaml(str(_REPO / "robots/so101_follower/robot.yaml"))
    skill = RSkillManifest.from_yaml(
        str(_REPO / "rskills/rskill-smolvla-so101-eraser_place-bf16/rskill.yaml")
    )
    manifest = build_robot_manifest(
        robot,
        DeployRuntime(enable_reasoner=True),
        [skill],
        bridge_on=True,
        write_controls=True,
        remote_control=True,
    )
    record = LogRecord(
        event_name=semconv.EVENT_ROBOT_MANIFEST,
        body=AnyValue(string_value=manifest.model_dump_json(by_alias=True)),
    )
    store = TelemetryStore()
    store.ingest_logs([ResourceLogs(scope_logs=[ScopeLogs(log_records=[record])])])
    return store


def _remote(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=remote_listener(app), client=("127.0.0.1", 40000))
    return httpx.AsyncClient(transport=transport, base_url=AUD, timeout=10.0)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Origin": _SITE}


def _seg(part: dict[str, object] | bytes) -> str:
    raw = part if isinstance(part, bytes) else json.dumps(part).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unsigned(claims: dict[str, object], alg: str) -> str:
    return f"{_seg({'alg': alg, 'kid': 'kid-1'})}.{_seg(claims)}."


def _hs256(claims: dict[str, object], secret: bytes, kid: str) -> str:
    signing_input = f"{_seg({'alg': 'HS256', 'kid': kid})}.{_seg(claims)}"
    mac = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_seg(mac)}"


# ── token validation, Interface B order ───────────────────────────────────────


@pytest.mark.asyncio
async def test_a_valid_token_is_a_remote_principal(issuer: FakeIssuer, tmp_path: Path) -> None:
    app = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path))
    async with _remote(app) as client:
        resp = await client.get("/api/config", headers=_auth(issuer.mint(role="viewer")))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["mode"], body["role"], body["robot_id"]) == ("remote", "viewer", ROBOT)
    assert resp.headers["access-control-allow-origin"] == _SITE
    assert "access-control-allow-credentials" not in resp.headers


def _bad_tokens(issuer: FakeIssuer) -> dict[str, str]:
    now = int(time.time())
    public_pem = issuer.key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )
    claims = jwt.decode(issuer.mint(), options={"verify_signature": False})
    return {
        "wrong aud": issuer.mint(aud="https://r-other.openral.com"),
        "aud list": issuer.mint(aud=[AUD, "https://r-other.openral.com"]),
        "wrong iss": issuer.mint(iss="https://evil.example/api/auth"),
        "wrong robot": issuer.mint(robot="other"),
        "wrong org": issuer.mint(org="other-org"),
        "free-robot org on an org robot": issuer.mint(org=""),
        "expired": issuer.mint(iat=now - 400, exp=now - 100),
        "lifetime over 300 s": issuer.mint(lifetime_s=301),
        "issued in the future": issuer.mint(iat=now + 3600, exp=now + 3700),
        "no jti": issuer.mint(jti=None),
        "no sub": issuer.mint(sub=None),
        "unknown role": issuer.mint(role="owner"),
        "typ JWT": issuer.mint(header={"typ": "JWT"}),
        "alg none": _unsigned(claims, "none"),
        # The classic confusion: HMAC "signed" with the issuer's public key (forged by hand,
        # since PyJWT itself refuses to use an asymmetric key as an HMAC secret).
        "alg HS256 keyed with the public key": _hs256(claims, public_pem, issuer.kid),
        "another key, same kid": jwt.encode(
            claims, FakeIssuer().key, algorithm="ES256", headers={"kid": issuer.kid}
        ),
        "garbage": "not-a-jwt",
    }


@pytest.mark.asyncio
async def test_bad_tokens_are_401(issuer: FakeIssuer, tmp_path: Path) -> None:
    app = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path))
    async with _remote(app) as client:
        no_header = await client.get("/api/config", headers={"Origin": _SITE})
        statuses = {
            name: (await client.get("/api/config", headers=_auth(token))).status_code
            for name, token in _bad_tokens(issuer).items()
        }
    assert no_header.status_code == 401
    assert no_header.headers["www-authenticate"].startswith("Bearer")
    assert no_header.headers["access-control-allow-origin"] == _SITE  # readable by the UI
    assert statuses == dict.fromkeys(statuses, 401)


@pytest.mark.asyncio
async def test_unenrolled_robot_admits_nobody(issuer: FakeIssuer) -> None:
    async with _remote(create_app(TelemetryStore())) as client:
        resp = await client.get("/api/config", headers=_auth(issuer.mint()))
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_jwks_is_cached_on_disk_and_refetched_at_most_once_a_minute(
    issuer: FakeIssuer, tmp_path: Path
) -> None:
    app = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path))
    async with _remote(app) as client:
        ok = [await client.get("/api/config", headers=_auth(issuer.mint())) for _ in range(3)]
        unknown_kid = issuer.mint(header={"kid": "kid-2"})
        rotated = [await client.get("/api/config", headers=_auth(unknown_kid)) for _ in range(3)]
    assert [r.status_code for r in ok] == [200, 200, 200]
    assert [r.status_code for r in rotated] == [401, 401, 401]
    assert issuer.fetches == 1  # unknown kids inside the cooldown cause no refetch
    # A restarted dashboard starts from the disk cache, not the network.
    restarted = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path))
    async with _remote(restarted) as client:
        assert (await client.get("/api/config", headers=_auth(issuer.mint()))).status_code == 200
    assert issuer.fetches == 1


@pytest.mark.asyncio
async def test_a_rotated_key_is_picked_up_after_the_cooldown(
    issuer: FakeIssuer, tmp_path: Path
) -> None:
    app = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path, cooldown_s=0.0))
    async with _remote(app) as client:
        assert (await client.get("/api/config", headers=_auth(issuer.mint()))).status_code == 200
        issuer.kid = "kid-2"
        resp = await client.get("/api/config", headers=_auth(issuer.mint()))
    assert resp.status_code == 200
    assert issuer.fetches == 2


@pytest.mark.asyncio
async def test_a_retired_key_stops_verifying_after_the_hourly_refetch(
    issuer: FakeIssuer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key the issuer drops (rotation after a leak) is not trusted forever."""
    from openral_observability.dashboard import remote_auth

    app = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path))
    old = issuer.mint()
    async with _remote(app) as client:
        assert (await client.get("/api/config", headers=_auth(old))).status_code == 200
        issuer.kid = "kid-2"  # the issuer now publishes only its new key
        monkeypatch.setattr(remote_auth, "_JWKS_MAX_AGE_S", 0.0)  # an hour later
        resp = await client.get("/api/config", headers=_auth(old))
    assert resp.status_code == 401
    assert issuer.fetches == 2


# ── CORS and the refused surface ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cors_preflight_is_answered_before_auth(issuer: FakeIssuer, tmp_path: Path) -> None:
    app = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path))
    preflight = {
        "Origin": _SITE,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization, content-type",
    }
    async with _remote(app) as client:
        ok = await client.options("/api/actions/estop", headers=preflight)
        evil = await client.options(
            "/api/actions/estop", headers={**preflight, "Origin": "https://evil.example"}
        )
        evil_get = await client.get(
            "/api/config", headers={**_auth(issuer.mint()), "Origin": "https://evil.example"}
        )
    assert ok.status_code == 204
    assert ok.headers["access-control-allow-origin"] == _SITE
    assert "Authorization" in ok.headers["access-control-allow-headers"]
    assert ok.headers["access-control-max-age"] == "600"
    assert "access-control-allow-credentials" not in ok.headers
    assert evil.status_code == 403 and "access-control-allow-origin" not in evil.headers
    assert evil_get.status_code == 403


@pytest.mark.asyncio
async def test_allowed_origins_come_from_the_env(
    issuer: FakeIssuer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_DASHBOARD_ALLOWED_ORIGINS", "https://staging.example.org")
    app = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path))
    async with _remote(app) as client:
        site = await client.get("/api/config", headers=_auth(issuer.mint()))
        staging = await client.get(
            "/api/config", headers={**_auth(issuer.mint()), "Origin": "https://staging.example.org"}
        )
    assert site.status_code == 403
    assert staging.status_code == 200


@pytest.mark.asyncio
async def test_otlp_receiver_is_not_reachable_remotely(issuer: FakeIssuer, tmp_path: Path) -> None:
    app = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path))
    async with _remote(app) as client:
        authed = await client.post("/v1/logs", content=b"", headers=_auth(issuer.mint()))
        anon = await client.post("/v1/traces", content=b"")
    assert (authed.status_code, anon.status_code) == (404, 404)


# ── roles and the remote-control flag (ADR-0129 §6) ──────────────────────────


@pytest.mark.asyncio
async def test_viewer_may_only_estop(
    issuer: FakeIssuer, tmp_path: Path, ros2_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_DASHBOARD_REMOTE_CONTROL", "1")
    monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", "1")
    app = create_app(_store_with_manifest(), verifier=_verifier(issuer, tmp_path))
    viewer = _auth(issuer.mint(role="viewer"))
    async with _remote(app) as client:
        estop = await client.post("/api/actions/estop", headers=viewer)
        reset = await client.post("/api/actions/estop_reset", json={"args": {}}, headers=viewer)
        prompt = await client.post(
            "/api/actions/prompt", json={"args": {"text": "go"}}, headers=viewer
        )
        skill = await client.post(f"/api/actions/skill:{_SKILL}", json={"args": {}}, headers=viewer)
        from openral_observability.dashboard.app import _BACKGROUND_TASKS

        await asyncio.gather(*_BACKGROUND_TASKS)
    assert estop.status_code == 200, estop.text
    assert (reset.status_code, prompt.status_code, skill.status_code) == (403, 403, 403)
    announced = _announced(ros2_log)
    assert [(a["action"], a["channel_label"]) for a in announced] == [("estop", "remote")]
    assert announced[0]["actor_id"] == pseudonymous_actor_id("user-7f3a", "deployer-key")
    assert announced[0]["detail"] == "jti=tok-91"


@pytest.mark.asyncio
async def test_transcribe_follows_the_prompt_rules_remotely(
    issuer: FakeIssuer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whisper on the robot's computer: an operator's, over the tunnel with remote control."""
    monkeypatch.delenv("OPENRAL_DASHBOARD_REMOTE_CONTROL", raising=False)
    app = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path))
    async with _remote(app) as client:
        viewer = await client.post(
            "/api/transcribe", content=b"x", headers=_auth(issuer.mint(role="viewer"))
        )
        operator = await client.post(
            "/api/transcribe", content=b"x", headers=_auth(issuer.mint(role="operator"))
        )
    assert viewer.status_code == 403 and "may not run" in viewer.json()["error"]
    assert operator.status_code == 403 and "remote control is off" in operator.json()["error"]


@pytest.mark.asyncio
async def test_operator_needs_the_remote_control_flag(
    issuer: FakeIssuer, tmp_path: Path, ros2_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", "1")
    monkeypatch.delenv("OPENRAL_DASHBOARD_REMOTE_CONTROL", raising=False)
    store = _store_with_manifest()
    app = create_app(store, verifier=_verifier(issuer, tmp_path))
    operator = _auth(issuer.mint(role="operator"))
    async with _remote(app) as client:
        refused = await client.post("/api/actions/estop_reset", json={"args": {}}, headers=operator)
        manifest = (await client.get("/api/manifest", headers=operator)).json()
        monkeypatch.setenv("OPENRAL_DASHBOARD_REMOTE_CONTROL", "1")
        reset = await client.post("/api/actions/estop_reset", json={"args": {}}, headers=operator)
        skill = await client.post(
            f"/api/actions/skill:{_SKILL}",
            json={"args": {}},
            headers=_auth(issuer.mint(role="admin")),
        )
        from openral_observability.dashboard.app import _BACKGROUND_TASKS

        await asyncio.gather(*_BACKGROUND_TASKS)
    assert refused.status_code == 403 and "remote control is off" in refused.json()["error"]
    assert {a["id"]: a["remote"] for a in manifest["actions"]}["estop_reset"] is False
    assert reset.status_code == 200, reset.text
    assert skill.status_code == 202, skill.text


# ── streams close at the token's exp ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_ends_at_token_expiry(issuer: FakeIssuer, tmp_path: Path) -> None:
    app = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path))
    now = int(time.time())
    token = issuer.mint(iat=now, exp=now + 2)
    started = time.monotonic()
    async with _remote(app) as client:
        resp = await client.get("/api/stream?images=0", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.text.startswith("data: ")
    assert time.monotonic() - started < 10


# ── the 3D view: tickets, layout, /ws ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_viz_ticket_is_256_bits_and_needs_a_token(issuer: FakeIssuer, tmp_path: Path) -> None:
    app = create_app(TelemetryStore(), verifier=_verifier(issuer, tmp_path))
    async with _remote(app) as client:
        issued = await client.post("/api/viz/ticket", headers=_auth(issuer.mint(role="viewer")))
        anonymous = await client.post("/api/viz/ticket", headers={"Origin": _SITE})
    assert issued.json()["expires_in"] == 900
    assert len(base64.urlsafe_b64decode(issued.json()["ticket"] + "=")) == 32  # 256 bits
    assert anonymous.status_code == 401


@pytest.mark.asyncio
async def test_viz_index_carries_the_robot_layout_under_its_csp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Lichtblick's default-layout slot, as its index.html ships it (no layoutUrl).
    monkeypatch.syspath_prepend(str(_REPO / "packages" / "openral_foxglove_bringup"))
    slot = "globalThis.L = [/*LICHTBLICK_SUITE_DEFAULT_LAYOUT_PLACEHOLDER*/][0];"
    (tmp_path / "index.html").write_text(f"<html><script>{slot}</script></html>")
    app = create_app(_store_with_manifest())
    app.state.viz_dir = tmp_path
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 5))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:4318") as c:
        resp = await c.get("/viz/")
        gone = await c.get("/api/viz/layout.json")
    script = resp.text.split("<script>")[1].split("</script>")[0]
    layout = json.loads(script.removeprefix("globalThis.L = [").removesuffix("][0];"))
    image_topics = {
        cfg["imageMode"]["imageTopic"]
        for name, cfg in layout["configById"].items()
        if name.startswith("Image!")
    }
    assert image_topics == {"/openral/cameras/top/image", "/openral/cameras/wrist/image"}
    assert layout["configById"]["3D!scene"]["followTf"] == "base"
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    csp = resp.headers["content-security-policy"]
    assert f"'unsafe-eval' 'sha256-{digest}';" in csp  # the hash is of the page as served
    assert resp.headers["cache-control"] == "no-store"
    assert gone.status_code == 404


def _ticket(client: TestClient, headers: dict[str, str] | None = None) -> str:
    resp = client.post("/api/viz/ticket", headers=headers or {})
    assert resp.status_code == 200, resp.text
    return str(resp.json()["ticket"])


def test_ws_proxy_needs_a_ticket_remotely_and_relays_both_ways(
    issuer: FakeIssuer, tmp_path: Path
) -> None:
    with FakeBridge().running() as bridge:
        app = create_app(
            TelemetryStore(),
            verifier=_verifier(issuer, tmp_path),
            bridge_url=f"ws://127.0.0.1:{bridge.port}",
        )
        remote = remote_listener(app)
        with TestClient(remote, base_url=AUD, client=("127.0.0.1", 40000)) as client:
            for query, origin in (
                ("", "https://viz.openral.com"),
                ("?ticket=forged", "https://viz.openral.com"),
            ):
                with (
                    pytest.raises(WebSocketDisconnect) as refused,
                    client.websocket_connect(f"/ws{query}", headers={"Origin": origin}),
                ):
                    pass
                assert refused.value.code == 1008
            ticket = _ticket(client, _auth(issuer.mint(role="viewer")))
            with (
                pytest.raises(WebSocketDisconnect),
                client.websocket_connect(
                    f"/ws?ticket={ticket}", headers={"Origin": "https://evil.example"}
                ),
            ):
                pass
            with client.websocket_connect(
                f"/ws?ticket={ticket}",
                headers={"Origin": "https://viz.openral.com"},
                subprotocols=["foxglove.websocket.v1"],
            ) as ws:
                assert ws.accepted_subprotocol == "foxglove.websocket.v1"
                ws.send_text('{"op": "subscribe"}')
                assert ws.receive_text() == '{"op": "subscribe"}'
                ws.send_bytes(b"\x01\x02")
                assert ws.receive_bytes() == b"\x01\x02"


def test_ws_proxy_ticket_is_optional_only_on_loopback(tmp_path: Path) -> None:
    with FakeBridge().running() as bridge:
        app = create_app(TelemetryStore(), bridge_url=f"ws://127.0.0.1:{bridge.port}")
        foxglove = {"subprotocols": ["foxglove.websocket.v1"]}
        with TestClient(app, base_url="http://127.0.0.1:4318", client=("127.0.0.1", 5)) as local:
            with local.websocket_connect("/ws", **foxglove) as ws:
                ws.send_text("hi")
                assert ws.receive_text() == "hi"
            with (  # a page on another site
                pytest.raises(WebSocketDisconnect),
                local.websocket_connect("/ws", headers={"Origin": "http://evil.example"}),
            ):
                pass
        with TestClient(app, base_url="http://10.0.0.2:4318", client=("10.0.0.7", 5)) as lan:
            with pytest.raises(WebSocketDisconnect), lan.websocket_connect("/ws"):
                pass
            ticket = _ticket(lan)  # a LAN viewer may ask for one
            with lan.websocket_connect(f"/ws?ticket={ticket}", **foxglove) as ws:
                ws.send_text("hi")
                assert ws.receive_text() == "hi"


def test_a_remote_port_needs_a_linked_robot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openral_observability.dashboard.server import run_dashboard

    monkeypatch.setenv("OPENRAL_ROBOT_ENROLMENT", str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="openral robot link"):
        run_dashboard(remote_port=4319)
