"""Unit tests for ``tools/lifecycle_autostart.py:_drive_transition``.

Regression coverage for the robocasa-kitchen ``openral deploy sim`` boot
crash: the HAL's ``on_configure`` runs synchronously on its executor
and on a first-boot blocks for over a minute (MuJoCo + robosuite import
+ ``env.reset``, plus a ``uv`` build of the robocasa editable package).
The autostart previously hardcoded a 30 s ``spin_until_future_complete``
on each transition, so it timed out mid-configure: ``future.result()``
returned ``None`` and the immediate ``get_state`` read returned ``''``
(executor still busy), and a transition that was about to succeed was
reported as ``did not advance the FSM`` — the process exited 1 and the
whole launch died. The fix makes the per-transition spin budget a
``--transition-timeout-s`` parameter (default 300 s; the HAL autostart
passes it explicitly) and polls the post-call state for a short grace
window before declaring failure.

The lifecycle ``change_state`` / ``get_state`` services are a ROS
process/network boundary, so faking the service clients here is allowed
under CLAUDE.md §1.11 (doubles permitted at process/network boundaries).
``rclpy`` + ``lifecycle_msgs`` are imported by the tool at module scope;
when ROS 2 isn't installed (pure-Python CI) the whole module is skipped.
"""

from __future__ import annotations

import importlib.util
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
