"""``SidecarClient`` reaps the sidecars it spawned when ``openral`` exits.

Sidecars spawn detached (own session) so a terminal Ctrl-C during boot can't
kill them — but a hard-killed ``openral`` (SIGTERM from ``timeout`` /
orchestrator, or a crash before ``close()``) then leaks the sidecar, which
keeps holding GPU memory and its port. That leak broke later in-process runs
in the scene×rSkill sweep (a stale RLDX sidecar held ~6 GB). The client now
registers each spawned child for reaping at process exit (atexit + a chained,
main-thread-only SIGTERM handler).

CLAUDE.md §1.11 — doubles only at the process boundary (``os.killpg``,
``atexit.register``, fake ``Popen``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import openral_sim.sidecar as sc
import pytest
from openral_sim.sidecar import SidecarClient


class _FakeChild:
    """Minimal ``subprocess.Popen`` stand-in: just ``pid`` + ``poll``."""

    def __init__(self, pid: int, *, alive: bool = True) -> None:
        self.pid = pid
        self._alive = alive

    def poll(self) -> int | None:
        return None if self._alive else 0


@pytest.fixture(autouse=True)
def _isolate_reaper_state() -> Iterator[None]:
    """Save/restore the module-global reaper registry + install flag."""
    saved_children = list(sc._spawned_children)
    saved_installed = set(sc._exit_reaper_installed)
    sc._spawned_children.clear()
    sc._exit_reaper_installed.clear()
    try:
        yield
    finally:
        sc._spawned_children[:] = saved_children
        sc._exit_reaper_installed.clear()
        sc._exit_reaper_installed |= saved_installed


def test_reap_spawned_children_sigterms_live_and_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(sc.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(sc.os, "killpg", lambda pgid, _signum: killed.append(pgid))
    live = _FakeChild(111, alive=True)
    dead = _FakeChild(222, alive=False)
    sc._spawned_children.extend([live, dead])

    sc._reap_spawned_children()

    assert killed == [111]  # live one signalled, dead one skipped
    assert sc._spawned_children == []  # registry cleared


def test_register_spawned_child_tracks_and_installs_atexit(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list[Any] = []
    monkeypatch.setattr(sc.atexit, "register", registered.append)
    # Force the worker-thread branch (current != main) so no real SIGTERM
    # handler is installed: each ``object`` call returns a distinct instance.
    monkeypatch.setattr(sc.threading, "current_thread", object)
    monkeypatch.setattr(sc.threading, "main_thread", object)

    child = _FakeChild(333)
    sc._register_spawned_child(child)  # type: ignore[arg-type]

    assert child in sc._spawned_children
    assert sc._reap_spawned_children in registered
    # Idempotent: a second registration does not re-install atexit.
    sc._register_spawned_child(_FakeChild(334))  # type: ignore[arg-type]
    assert registered.count(sc._reap_spawned_children) == 1


def test_terminate_child_deregisters_from_reaper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc.atexit, "register", lambda _fn: None)
    monkeypatch.setattr(sc.threading, "current_thread", object)
    monkeypatch.setattr(sc.threading, "main_thread", object)
    client = SidecarClient(
        name="rldx",
        host="127.0.0.1",
        port=40001,
        timeout_ms=1,
        boot_timeout_s=1.0,
        launch_argv=[],
        auto_spawn=True,
    )
    dead = _FakeChild(444, alive=False)  # already exited
    client._child = dead  # type: ignore[assignment]
    sc._spawned_children.append(dead)  # type: ignore[arg-type]

    client._terminate_child()

    assert dead not in sc._spawned_children
    assert client._child is None
