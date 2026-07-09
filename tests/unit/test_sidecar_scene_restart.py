"""``SidecarClient`` scene-aware restart on identity mismatch.

A sidecar is spawned detached (own session) and reused across runs by
pinging ``host:port`` first. When a *different* scene reuses the port, the
ping identity contradicts the request. Previously ``connect`` raised
``ROSConfigError`` ("already serving a different scene") and the run died —
the Isaac ``:5757`` collision seen in the scene×rSkill sweep, where the first
scene's sidecar lingered and every later scene failed. Now, when
``auto_spawn`` is set, the stale sidecar is reaped (via its recorded PID) and
a fresh one spawned. ``auto_spawn=False`` keeps the explicit error (we must
not restart a hand-launched sidecar).

CLAUDE.md §1.11 — the only doubles are at the process/network boundary
(``zmq`` import, ``os.killpg``, the port probe), exactly the seam this module
is meant to isolate.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest
from openral_core.exceptions import ROSConfigError
from openral_sim.sidecar import SidecarClient


@pytest.fixture
def _fake_zmq() -> Iterator[None]:
    """Stub ``zmq`` so ``connect`` runs without the opt-in sidecar group."""
    fake = types.ModuleType("zmq")
    fake.Context = types.SimpleNamespace(instance=object)  # type: ignore[attr-defined]
    saved = sys.modules.get("zmq")
    sys.modules["zmq"] = fake
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["zmq"] = saved
        else:
            sys.modules.pop("zmq", None)


def _client(auto_spawn: bool) -> SidecarClient:
    return SidecarClient(
        name="isaac",
        host="127.0.0.1",
        port=59999,
        timeout_ms=1000,
        boot_timeout_s=1.0,
        launch_argv=["python", "sidecar.py"],
        auto_spawn=auto_spawn,
        expected_identity={"task": "isaac_sim/bowl_plate", "layout": "bowl_plate"},
    )


def test_identity_mismatch_pure() -> None:
    c = _client(auto_spawn=True)
    assert c._identity_mismatch({"task": "isaac_sim/bowl_plate", "layout": "bowl_plate"}) == {}
    mis = c._identity_mismatch({"task": "isaac_sim/lift_cube", "layout": "lift_cube"})
    assert set(mis) == {"task", "layout"}


def test_connect_restarts_stale_sidecar_on_mismatch(
    _fake_zmq: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """auto_spawn + a wrong-scene sidecar → reap + respawn, not raise."""
    c = _client(auto_spawn=True)
    monkeypatch.setattr(c, "_init_socket", lambda: None)
    # An existing sidecar is up, serving a DIFFERENT scene.
    monkeypatch.setattr(
        c, "_ping_reply", lambda: {"task": "isaac_sim/lift_cube", "layout": "lift_cube"}
    )
    reaped: list[str] = []
    spawned: list[bool] = []
    monkeypatch.setattr(c, "_reap_stale_sidecar", reaped.append)
    monkeypatch.setattr(c, "_spawn", lambda: spawned.append(True))
    monkeypatch.setattr(c, "_wait_for_boot", lambda: True)

    c.connect()

    assert reaped == ["tcp://127.0.0.1:59999"], "stale sidecar not reaped on scene mismatch"
    assert spawned == [True], "fresh sidecar not spawned after reap"


def test_connect_raises_on_mismatch_when_not_auto_spawn(
    _fake_zmq: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """auto_spawn=False keeps the explicit 'stop it / distinct port' error."""
    c = _client(auto_spawn=False)
    monkeypatch.setattr(c, "_init_socket", lambda: None)
    monkeypatch.setattr(c, "_ping_reply", lambda: {"task": "isaac_sim/lift_cube"})
    with pytest.raises(ROSConfigError, match="already serving a different"):
        c.connect()


def test_reap_stale_kills_recorded_pid_and_returns_when_port_frees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openral_sim.sidecar as sc

    monkeypatch.setattr(sc, "read_sidecar_identity", lambda port: {"pid": 4242})
    monkeypatch.setattr(sc.os, "getpgid", lambda pid: pid)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(sc.os, "killpg", lambda pgid, signum: killed.append((pgid, signum)))
    c = _client(auto_spawn=True)
    monkeypatch.setattr(c, "_is_port_busy", lambda: False)  # frees immediately

    c._reap_stale_sidecar("tcp://127.0.0.1:59999")

    assert killed and killed[0][0] == 4242


def test_reap_stale_raises_when_port_stays_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    import openral_sim.sidecar as sc

    class _Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            self.t += 100.0
            return self.t

    monkeypatch.setattr(sc, "read_sidecar_identity", lambda port: {"pid": 4242})
    monkeypatch.setattr(sc.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(sc.os, "killpg", lambda *a: None)
    monkeypatch.setattr(sc.time, "sleep", lambda *_a: None)
    monkeypatch.setattr(sc.time, "monotonic", _Clock())
    c = _client(auto_spawn=True)
    monkeypatch.setattr(c, "_is_port_busy", lambda: True)  # never frees

    with pytest.raises(ROSConfigError, match="could not be reaped"):
        c._reap_stale_sidecar("tcp://127.0.0.1:59999")
