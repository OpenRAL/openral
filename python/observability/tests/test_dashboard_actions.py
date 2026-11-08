"""``POST /api/actions/{id}`` and ``GET /api/manifest`` on the local listener (ADR-0129).

The manifest is a real one — ``robots/so101_follower`` with the in-tree rSkills —
delivered the way the runtime delivers it: an ``openral.robot.manifest`` OTLP log
record posted to ``/v1/logs``. ``ros2`` and ``openral`` are real executables on PATH
that log their argv (CLAUDE.md §1.11 process-boundary doubles), so each test proves
the dashboard spawned the right child. Remote-listener rules (roles, the
remote-control flag, tokens) are in ``tests/unit/test_dashboard_remote.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from openral_core import (
    DeployRuntime,
    RobotDescription,
    RobotManifest,
    RSkillManifest,
    build_robot_manifest,
)
from openral_observability import semconv
from openral_observability.dashboard import TelemetryStore, create_app
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.proto.logs.v1.logs_pb2 import LogRecord, ResourceLogs, ScopeLogs

_REPO = Path(__file__).resolve().parents[3]
_SKILL = "OpenRAL/rskill-smolvla-so101-eraser_place-bf16"


def _manifest() -> RobotManifest:
    robot = RobotDescription.from_yaml(str(_REPO / "robots/so101_follower/robot.yaml"))
    installed = [
        RSkillManifest.from_yaml(str(p)) for p in sorted(_REPO.glob("rskills/*so101*/rskill.yaml"))
    ]
    return build_robot_manifest(
        robot,
        DeployRuntime(enable_reasoner=True),
        installed,
        bridge_on=True,
        write_controls=True,
        remote_control=False,
    )


def _manifest_logs(manifest: RobotManifest) -> bytes:
    record = LogRecord(
        event_name=semconv.EVENT_ROBOT_MANIFEST,
        body=AnyValue(string_value=manifest.model_dump_json(by_alias=True)),
    )
    req = ExportLogsServiceRequest(
        resource_logs=[ResourceLogs(scope_logs=[ScopeLogs(log_records=[record])])]
    )
    return req.SerializeToString()


def _client(
    store: TelemetryStore | None = None, client: tuple[str, int] = ("127.0.0.1", 50000)
) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(store or TelemetryStore()), client=client)
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:4318", timeout=10.0)


async def _drain_background() -> None:
    """Let the post-E-STOP announcement / skill-result drain finish inside this test's loop."""
    from openral_observability.dashboard.app import _BACKGROUND_TASKS

    await asyncio.gather(*_BACKGROUND_TASKS)


async def _with_manifest(client: httpx.AsyncClient) -> None:
    resp = await client.post("/v1/logs", content=_manifest_logs(_manifest()))
    assert resp.status_code == 200, resp.text


@pytest.fixture
def ros2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Path]:
    """Put a `ros2` on PATH that appends its argv to a log and answers like ROS would.

    ``mode`` picks the answer to ``action send_goal`` / ``service call``: ``accepted``,
    ``rejected``, ``slow`` (no answer for 5 s), ``reset_ok`` or ``reset_cooldown``.
    """

    def install(mode: str) -> Path:
        log = tmp_path / "ros2_calls.jsonl"
        shim = tmp_path / "ros2"
        shim.write_text(
            f"#!{sys.executable}\n"
            "import json, sys, time\n"
            f"open({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
            f"mode = {mode!r}\n"
            "if sys.argv[1:3] == ['action', 'send_goal']:\n"
            "    if mode == 'slow':\n"
            "        time.sleep(5)\n"
            "    elif mode == 'rejected':\n"
            "        print('Goal was rejected :(', flush=True); sys.exit(1)\n"
            "    print('Goal accepted with ID: test-123', flush=True)\n"
            "    print('  success: True', flush=True)\n"
            "elif sys.argv[1:3] == ['service', 'call']:\n"
            "    ok = mode == 'reset_ok'\n"
            "    print(f\"response: success={ok}, message=''\", flush=True)\n"
        )
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
        return log

    return install


def _calls(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _announced(log: Path) -> list[dict[str, str]]:
    return [json.loads(c[-1]) for c in _calls(log) if "/openral/operator_action" in c]


# ── manifest + config ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_manifest_is_404_until_the_runtime_sends_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", "1")
    store = TelemetryStore()
    async with _client(store) as client:
        assert (await client.get("/api/manifest")).status_code == 404
        before = store.snapshot()["manifest_rev"]
        await _with_manifest(client)
        body = (await client.get("/api/manifest")).json()
    assert store.snapshot()["manifest_rev"] == before + 1
    assert body["schema"] == 1
    assert body["robot_id"] == "so101_follower"
    assert {"cam:top", "cam:wrist", "joints", "viz3d"} <= {p["id"] for p in body["panels"]}
    assert f"skill:{_SKILL}" in {a["id"] for a in body["actions"]}
    # The manifest record is cached, not shown as an event-log row.
    assert all(e["kind"] != "openral.robot" for e in store.snapshot()["events"])


@pytest.mark.asyncio
async def test_served_manifest_follows_this_dashboards_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The emitter had write controls on; this dashboard has them off: skills drop out,
    # and `remote` follows this dashboard's remote-control flag (E-STOP always remote).
    monkeypatch.delenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", raising=False)
    monkeypatch.setenv("OPENRAL_DASHBOARD_REMOTE_CONTROL", "1")
    async with _client() as client:
        await _with_manifest(client)
        actions = {a["id"]: a for a in (await client.get("/api/manifest")).json()["actions"]}
    assert set(actions) == {"estop", "estop_reset", "prompt"}
    assert all(a["remote"] for a in actions.values())


@pytest.mark.asyncio
async def test_lan_client_sees_read_only_plus_estop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", "1")
    store = TelemetryStore()
    async with _client(store) as local:
        await _with_manifest(local)
        local_config = (await local.get("/api/config")).json()
    async with _client(store, client=("10.0.0.7", 50000)) as lan:
        config = (await lan.get("/api/config")).json()
        actions = (await lan.get("/api/manifest")).json()["actions"]
        prompt = await lan.post("/api/actions/prompt", json={"args": {"text": "go"}})
        # A robot computer exporting to a dashboard on the LAN (ADR-0017) still can.
        otlp = await lan.post("/v1/logs", content=_manifest_logs(_manifest()))
    assert (local_config["mode"], local_config["role"]) == ("local", "operator")
    assert local_config["robot_id"] == "so101_follower"
    assert (config["mode"], config["role"]) == ("lan", "viewer")
    assert config["write_controls"] is True
    assert [a["id"] for a in actions] == ["estop"]
    assert prompt.status_code == 403
    assert otlp.status_code == 200


# ── E-STOP ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("client_addr", [("127.0.0.1", 1), ("10.0.0.7", 1)])
async def test_estop_fires_without_a_manifest_from_loopback_and_lan(
    ros2: Callable[[str], Path], client_addr: tuple[str, int]
) -> None:
    log = ros2("accepted")
    store = TelemetryStore()
    async with _client(store, client=client_addr) as client:
        resp = await client.post("/api/actions/estop")  # no body: a stop never waits on one
        await _drain_background()
    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True
    assert store.snapshot()["topics"]["safety"]["estopped"] is True
    assert any("/openral/estop" in c for c in _calls(log))


# ── reset, prompt, skills ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_action_is_404() -> None:
    async with _client() as client:
        no_manifest = await client.post("/api/actions/estop_reset", json={"args": {}})
        await _with_manifest(client)
        unknown = await client.post("/api/actions/skill:OpenRAL/rskill-nope", json={"args": {}})
    assert no_manifest.status_code == 404
    assert unknown.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "status", "latched"), [("reset_ok", 200, False), ("reset_cooldown", 409, True)]
)
async def test_estop_reset(
    ros2: Callable[[str], Path], mode: str, status: int, latched: bool
) -> None:
    log = ros2(mode)
    store = TelemetryStore()
    store.set_estopped(True)
    async with _client(store) as client:
        await _with_manifest(client)
        resp = await client.post("/api/actions/estop_reset", json={"args": {}})
    assert resp.status_code == status, resp.text
    assert store.snapshot()["topics"]["safety"]["estopped"] is latched
    assert [a["action"] for a in _announced(log)] == ["estop_reset"]


@pytest.mark.asyncio
async def test_prompt_validates_args_then_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv_log = tmp_path / "openral_argv.json"
    shim = tmp_path / "openral"
    shim.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(argv_log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    async with _client() as client:
        await _with_manifest(client)
        bodies: list[dict[str, object]] = [
            {"args": {}},
            {"args": {"text": 42}},
            {"args": {"text": ""}},
            {"args": []},
        ]
        bad = [await client.post("/api/actions/prompt", json=body) for body in bodies]
        resp = await client.post("/api/actions/prompt", json={"args": {"text": "pick the eraser"}})
    assert [r.status_code for r in bad] == [400, 400, 400, 400]
    assert resp.status_code == 200, resp.text
    assert json.loads(argv_log.read_text()) == [
        "prompt",
        "pick the eraser",
        "--topic",
        "/openral/prompt_in/dashboard",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "status"), [("accepted", 202), ("rejected", 409)])
async def test_skill_dispatches_execute_rskill(
    ros2: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch, mode: str, status: int
) -> None:
    monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", "1")
    log = ros2(mode)
    async with _client() as client:
        await _with_manifest(client)
        resp = await client.post(f"/api/actions/skill:{_SKILL}", json={"args": {}})
        await _drain_background()
    assert resp.status_code == status, resp.text
    if status == 202:
        assert resp.json() == {
            "ok": True,
            "goal_id": "test-123",
            "detail": "skill dispatched; track progress via dashboard telemetry",
        }
    (goal_call,) = [c for c in _calls(log) if c[:2] == ["action", "send_goal"]]
    assert json.loads(goal_call[-1])["rskill_id"] == _SKILL
    assert [(a["action"], a["detail"]) for a in _announced(log)] == [("skill_execute", _SKILL)]


@pytest.mark.asyncio
async def test_skill_accept_timeout_is_504(
    ros2: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", "1")
    monkeypatch.setenv("OPENRAL_DASHBOARD_SKILL_ACCEPT_TIMEOUT_S", "1")
    ros2("slow")
    async with _client() as client:
        await _with_manifest(client)
        resp = await client.post(f"/api/actions/skill:{_SKILL}", json={"args": {}})
    assert resp.status_code == 504, resp.text
    assert "accept" in resp.json()["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "true", "True", "yes", "on", "1 ", " 1", "2"])
async def test_skills_need_write_controls_exactly_1(
    ros2: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # Only "1" enables write controls: a looser truthiness test must never unlock them.
    log = ros2("accepted")
    monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", value)
    async with _client() as client:
        await _with_manifest(client)
        resp = await client.post(f"/api/actions/skill:{_SKILL}", json={"args": {}})
    assert resp.status_code == 403, resp.text
    assert _calls(log) == []


# ── CSRF on the local listener ────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/actions/estop", "/api/actions/prompt", "/v1/logs"])
async def test_cross_origin_post_is_refused(ros2: Callable[[str], Path], path: str) -> None:
    # A page on another site POSTing through the operator's browser is refused
    # before anything shells out.
    log = ros2("accepted")
    async with _client() as client:
        resp = await client.post(
            path,
            content=b'{"args": {"text": "go"}}',
            headers={"origin": "http://evil.example", "content-type": "text/plain"},
        )
        assert resp.status_code == 403, resp.text
        assert _calls(log) == []
        same = await client.post("/api/actions/estop", headers={"origin": "http://127.0.0.1:4318"})
        await _drain_background()
    assert same.status_code == 200


def test_an_emitted_manifest_reaches_the_dashboard() -> None:
    # producer.emit_robot_manifest → real OTel SDK → the OTLP/HTTP wire encoding → store.
    from openral_observability.producer import emit_robot_manifest
    from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor

    exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]  # reason: the SDK's __init__ is unannotated
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    manifest = _manifest()
    emit_robot_manifest(manifest, logger_provider=provider)
    (record,) = exporter.get_finished_logs()
    assert record.log_record.event_name == semconv.EVENT_ROBOT_MANIFEST
    store = TelemetryStore()
    store.ingest_logs(list(encode_logs([record]).resource_logs))
    assert store.manifest() == manifest
