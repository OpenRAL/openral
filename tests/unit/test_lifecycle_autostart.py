"""Unit tests for ``tools/lifecycle_autostart.py:_drive_transition``.

Regression coverage for the robocasa-kitchen ``openral deploy sim`` boot
crash: ``on_configure`` runs synchronously on its executor and on first boot
blocks over a minute (MuJoCo + robosuite import + ``env.reset``, plus a
``uv`` build of the robocasa editable package). Autostart previously
hardcoded a 30 s ``spin_until_future_complete`` per transition, so it timed
out mid-configure — ``future.result()`` returned ``None``, the immediate
``get_state`` read returned ``''`` (executor still busy), and a
near-successful transition was reported as ``did not advance the FSM``: the
process exited 1 and the launch died. Fix: the per-transition spin budget is
now a ``--transition-timeout-s`` parameter (default 300 s), with a short
grace poll of post-call state before declaring failure.

The lifecycle ``change_state``/``get_state`` services are a ROS
process/network boundary, so faking the service clients is allowed under
CLAUDE.md §1.11. ``rclpy`` + ``lifecycle_msgs`` are imported at module scope;
the whole module skips when ROS 2 isn't installed.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("rclpy", reason="lifecycle_autostart needs a ROS 2 (rclpy) install")
pytest.importorskip("lifecycle_msgs", reason="lifecycle_autostart needs lifecycle_msgs")

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "lifecycle_autostart.py"


def _load_tool() -> Any:
    """Import ``tools/lifecycle_autostart.py`` as a standalone module."""
    spec = importlib.util.spec_from_file_location("lifecycle_autostart_under_test", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeCurrentState:
    def __init__(self, label: str) -> None:
        self.label = label


class _FakeGetStateResult:
    def __init__(self, label: str) -> None:
        self.current_state = _FakeCurrentState(label)


class _FakeChangeStateResult:
    def __init__(self, success: bool) -> None:
        self.success = success


class _FakeFuture:
    def __init__(self, result: Any) -> None:
        self._result = result

    def result(self) -> Any:
        return self._result


class _FakeChangeStateClient:
    """Returns a fixed ``ChangeState`` response (``None`` = spin timed out)."""

    def __init__(self, resp: Any) -> None:
        self._resp = resp

    def call_async(self, _req: Any) -> _FakeFuture:
        return _FakeFuture(self._resp)


class _FakeGetStateClient:
    """Returns successive state labels, holding the last one forever."""

    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    def call_async(self, _req: Any) -> _FakeFuture:
        label = self._labels.pop(0) if len(self._labels) > 1 else self._labels[0]
        return _FakeFuture(_FakeGetStateResult(label))


class _FakeRclpy:
    """Records the per-transition spin timeout; never actually spins."""

    def __init__(self) -> None:
        self.spin_timeouts: list[float | None] = []

    def spin_until_future_complete(
        self, _node: Any, _future: Any, timeout_sec: float | None = None
    ) -> None:
        self.spin_timeouts.append(timeout_sec)

    def spin_once(self, _node: Any, timeout_sec: float | None = None) -> None:
        return None


class _FakeTime:
    """``monotonic`` ticks by 1.0 each call so the grace loop terminates."""

    def __init__(self) -> None:
        self._t = 0.0

    def monotonic(self) -> float:
        now = self._t
        self._t += 1.0
        return now


def _patch_runtime(mod: Any) -> _FakeRclpy:
    fake_rclpy = _FakeRclpy()
    mod.rclpy = fake_rclpy
    mod.time = _FakeTime()
    return fake_rclpy


def test_drive_transition_uses_caller_timeout() -> None:
    """The change_state spin is bounded by the passed ``transition_timeout_s``."""
    mod = _load_tool()
    fake_rclpy = _patch_runtime(mod)
    mod._drive_transition(
        node=object(),
        target_node="/openral_hal_panda_mobile",
        change_state_client=_FakeChangeStateClient(_FakeChangeStateResult(success=True)),
        get_state_client=_FakeGetStateClient(["inactive"]),
        transition_id=1,
        transition_label="configure",
        transition_timeout_s=123.0,
    )
    # First spin is the change_state wait — it must use the full budget,
    # not the legacy hardcoded 30 s.
    assert fake_rclpy.spin_timeouts[0] == 123.0


def test_drive_transition_succeeds_when_state_advances_despite_no_response() -> None:
    """A timed-out spin (resp=None) is fine when the post-call state advanced.

    This is the robocasa first-boot case: on_configure overran the spin
    so ``future.result()`` is ``None``, but the FSM did reach INACTIVE.
    """
    mod = _load_tool()
    _patch_runtime(mod)
    mod._drive_transition(
        node=object(),
        target_node="/openral_hal_panda_mobile",
        change_state_client=_FakeChangeStateClient(None),
        get_state_client=_FakeGetStateClient(["inactive"]),
        transition_id=1,
        transition_label="configure",
        transition_timeout_s=300.0,
    )  # must not raise


def test_drive_transition_polls_grace_window_before_success() -> None:
    """A spurious ``success=false`` settles via a later post-call state poll.

    Mirrors the Jazzy ``change_state`` race: the response says false but
    the FSM transitions a moment later, so the second state poll sees it.
    """
    mod = _load_tool()
    _patch_runtime(mod)
    mod._drive_transition(
        node=object(),
        target_node="/openral_slam_toolbox",
        change_state_client=_FakeChangeStateClient(_FakeChangeStateResult(success=False)),
        get_state_client=_FakeGetStateClient(["unconfigured", "inactive"]),
        transition_id=1,
        transition_label="configure",
        transition_timeout_s=300.0,
    )  # must not raise — second poll returns "inactive"


def test_drive_transition_raises_on_genuine_failure() -> None:
    """A response failure with a never-advancing state raises after the grace poll."""
    mod = _load_tool()
    _patch_runtime(mod)
    with pytest.raises(RuntimeError, match=r"did not advance the FSM within 300\.0s"):
        mod._drive_transition(
            node=object(),
            target_node="/openral_hal_panda_mobile",
            change_state_client=_FakeChangeStateClient(_FakeChangeStateResult(success=False)),
            get_state_client=_FakeGetStateClient(["unconfigured"]),
            transition_id=1,
            transition_label="configure",
            transition_timeout_s=300.0,
        )


_TARGET_NODE = textwrap.dedent(
    """
    import rclpy
    from rclpy.lifecycle import LifecycleNode

    rclpy.init()
    rclpy.spin(LifecycleNode("autostart_redundant_target"))
    """
)


def test_redundant_activate_does_not_crash_a_real_lifecycle_node() -> None:
    """A transition the target can no longer take is refused, not fatal.

    rclpy's ``change_state`` server turns an invalid *labelled* request into
    ``success=False``, but an id-only request reaches
    ``trigger_transition_by_id`` unguarded and its ``RCLError`` kills the
    target's executor. The tool sent ids only, so an ACTIVATE that another
    activator had already made redundant crashed the node it was starting
    (reproduced against this real node: exit 1, "Transition is not
    registered"). It now sends the label too, and the post-call state check
    accepts the already-active node.
    """
    import rclpy
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState, GetState

    from tests.sim.safety._kernel_subprocess import isolated_domain_id

    mod = _load_tool()
    domain = isolated_domain_id()
    target = subprocess.Popen(
        [sys.executable, "-c", _TARGET_NODE], env={**os.environ, "ROS_DOMAIN_ID": str(domain)}
    )
    ctx_started = False
    try:
        rclpy.init(domain_id=domain)
        ctx_started = True
        node = rclpy.create_node("autostart_redundant_driver")
        name = "/autostart_redundant_target"
        cs = mod._wait_for_service(node, f"{name}/change_state", 20.0, ChangeState)
        gs = mod._wait_for_service(node, f"{name}/get_state", 20.0, GetState)
        for tid, label in (
            (Transition.TRANSITION_CONFIGURE, "configure"),
            (Transition.TRANSITION_ACTIVATE, "activate"),
        ):
            mod._drive_transition(node, name, cs, gs, tid, label, 10.0)
        # A second activator: ACTIVATE again on a node that is already active.
        mod._drive_transition(node, name, cs, gs, Transition.TRANSITION_ACTIVATE, "activate", 10.0)
        assert mod._read_state(node, name, gs) == "active"
        with pytest.raises(subprocess.TimeoutExpired):
            target.wait(timeout=0.5)
        node.destroy_node()
    finally:
        if ctx_started:
            rclpy.shutdown()
        target.kill()
        target.wait()
