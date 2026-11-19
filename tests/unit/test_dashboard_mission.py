"""``GET /api/mission`` and its event stream: the few facts a program follows a robot by (#659)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from openral_observability.dashboard import TelemetryStore, create_app
from openral_observability.dashboard.app import mission_view

from tests.unit.fakes.fake_openral_issuer import FakeIssuer
from tests.unit.test_dashboard_remote import _auth, _remote, _store_with_manifest, _verifier


@pytest.fixture
def issuer() -> Iterator[FakeIssuer]:
    with FakeIssuer().serving() as served:
        yield served


def _snapshot(**reasoner: object) -> dict[str, object]:
    return {
        "topics": {
            "reasoner": reasoner,
            "safety": {"estopped": False, "checks": {"big": "x" * 10_000}},
            "safety_status": {"latched": False, "drop_reason_label": "none"},
            "hal_estop": {"hal": {"estopped": False}, "odd": None},
            "perception": {"cameras": {"top": {"jpeg_b64": "x" * 100_000}}},
        },
        "events": ["e"] * 200,
    }


def test_mission_view_is_small_and_carries_the_mission() -> None:
    mission = {
        "goal": "pick the orange",
        "correlation_id": "fleet-7:step-1",
        "max_attempts": 3,
        "splits": {},
        "tasks": [
            {"id": "t1", "text": "pick", "status": "done", "attempts": 1, "failures": 0,
             "verdict": "ok", "steps": [{"big": "x" * 5000}]},
            {"id": "t2", "text": "place", "status": "active", "attempts": 0, "failures": 0},
        ],
    }  # fmt: skip
    view = mission_view(_snapshot(mission=mission, tick_idx=4, tool="emit_prompt", message="?"))
    assert view["mission"] == {
        "goal": "pick the orange",
        "correlation_id": "fleet-7:step-1",
        "finished": False,
        "tasks": [
            {"id": "t1", "text": "pick", "status": "done", "attempts": 1, "failures": 0,
             "verdict": "ok"},
            {"id": "t2", "text": "place", "status": "active", "attempts": 0, "failures": 0,
             "verdict": None},
        ],
    }  # fmt: skip
    assert view["reasoner"]["tool"] == "emit_prompt" and view["reasoner"]["tick_idx"] == 4
    assert (view["awaiting_operator"], view["question"]) == (None, None)  # an older openral
    assert (view["stopped"], view["estopped"], view["latched"]) == (False, False, False)
    assert view["latch_reason"] is None  # a reason only while latched
    assert len(json.dumps(view)) < 1000


@pytest.mark.parametrize(
    ("statuses", "finished"),
    [(["done", "abandoned"], True), (["done", "verifying"], False), ([], False)],
)
def test_mission_view_says_when_the_mission_is_finished(
    statuses: list[str], finished: bool
) -> None:
    tasks = [{"id": f"t{i}", "status": s} for i, s in enumerate(statuses)]
    view = mission_view(_snapshot(mission={"goal": "g", "tasks": tasks}))
    assert view["mission"]["finished"] is finished


@pytest.mark.parametrize(
    ("topics", "expected"),
    [
        ({}, (False, False, False, None)),
        ({"safety": {"estopped": True}}, (True, True, False, None)),
        ({"hal_estop": {"left": {"estopped": True}}}, (True, True, False, None)),
        (
            {"safety_status": {"latched": True, "drop_reason_label": "collision"}},
            (True, False, True, "collision"),
        ),
        ({"safety": None, "safety_status": None, "hal_estop": None, "reasoner": None},
         (False, False, False, None)),
    ],
)  # fmt: skip
def test_mission_view_stop_state(topics: dict[str, object], expected: tuple[object, ...]) -> None:
    view = mission_view({"topics": topics})
    assert (view["stopped"], view["estopped"], view["latched"], view["latch_reason"]) == expected
    assert view["mission"] is None


@pytest.mark.asyncio
async def test_a_viewer_reads_the_mission_remotely(issuer: FakeIssuer, tmp_path: Path) -> None:
    store = _store_with_manifest()
    store.set_estopped(True)
    app = create_app(store, verifier=_verifier(issuer, tmp_path))
    before = time.time()
    async with _remote(app) as client:
        resp = await client.get("/api/mission", headers=_auth(issuer.mint(role="viewer")))
        anonymous = await client.get("/api/mission")
    body = resp.json()
    assert resp.status_code == 200, resp.text
    assert (body["mission"], body["stopped"], body["estopped"]) == (None, True, True)
    assert body["ts_unix"] >= before
    assert len(resp.content) < 1000
    assert anonymous.status_code == 401


async def _subscribed(store: TelemetryStore) -> None:
    """Until the stream under test has subscribed: a publish before that is in its snapshot."""
    for _ in range(500):
        if store._subscribers:  # reason: test sync on the store's own state
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the stream never subscribed")


@pytest.mark.asyncio
async def test_the_mission_stream_sends_only_changes(issuer: FakeIssuer, tmp_path: Path) -> None:
    store = TelemetryStore()
    app = create_app(store, verifier=_verifier(issuer, tmp_path))
    now = int(time.time())
    token = issuer.mint(iat=now, exp=now + 3)

    async def poke() -> None:
        await _subscribed(store)
        store.set_estopped(False)  # a publish that changes nothing in the view
        await asyncio.sleep(0.3)
        store.set_estopped(True)
        await asyncio.sleep(0.3)
        store.set_estopped(True)  # again: no change

    async with _remote(app) as client:
        task = asyncio.create_task(poke())
        resp = await client.get("/api/mission/stream", headers=_auth(token))
        await task
    frames = [json.loads(f[6:]) for f in resp.text.split("\n\n") if f.startswith("data: ")]
    assert [f["stopped"] for f in frames] == [False, True]
    assert all("ts_unix" not in f and "events" not in f for f in frames)


@pytest.mark.asyncio
async def test_a_busy_unchanged_stream_still_sends_keepalives(
    issuer: FakeIssuer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Telemetry that doesn't change the view must not starve the keepalive."""
    from openral_observability.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "_SSE_KEEPALIVE_S", 0.3)
    store = TelemetryStore()
    app = create_app(store, verifier=_verifier(issuer, tmp_path))
    now = int(time.time())
    token = issuer.mint(iat=now, exp=now + 3)

    async def busy() -> None:
        await _subscribed(store)
        for _ in range(25):  # unchanged view, a publish every 0.1 s: never a 0.3 s gap
            await asyncio.sleep(0.1)
            store.set_estopped(False)

    async with _remote(app) as client:
        task = asyncio.create_task(busy())
        resp = await client.get("/api/mission/stream", headers=_auth(token))
        await task
    assert resp.text.count(": keepalive") >= 3
    assert resp.text.count("data: ") == 1
