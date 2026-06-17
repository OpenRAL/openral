"""Bare ``MujocoArmHAL`` staleness recovery + idle-step (deploy-sim regression).

Two coupled regressions surfaced by ``openral deploy sim`` against a bare
:class:`~openral_hal._mujoco_arm.MujocoArmHAL` (e.g. ``OpenArmMujocoHAL``):

1. **Latched ``ROSPerceptionStale``.** ``read_state`` reads live in-process
   ``MjData`` (always the current simulator state), yet it refreshed the
   staleness clock only on the *success* path. If the single-threaded executor
   stalled >``staleness_limit_s`` between two publish ticks — e.g. a slow camera
   render hogging the thread — the next ``read_state`` raised *before* the
   refresh, so the clock never advanced and every subsequent read raised too.
   The HAL bricked itself permanently ("Joint state is X s old" logged
   constantly) despite the data being perfectly current.

2. **No dedicated publisher thread.** A bare ``MujocoArmHAL`` lacked
   ``idle_step``, so the lifecycle node published ``/joint_states`` from a timer
   on the executor (starved by rendering) instead of ADR-0049's dedicated
   thread + ``ProprioSnapshot``, and its cameras froze when idle (ADR-0034).

These tests build a *real* native-MuJoCo arm (no mocks, CLAUDE.md §1.11). A
bare arm's ``connect()`` does not render, so no GL/display is required; the only
external need is the menagerie MJCF, fetched on first use — unavailable →
``pytest.skip``.
"""

from __future__ import annotations

import time

import pytest
from openral_core.exceptions import ROSConfigError, ROSEStopRequested


def _build_bare_arm() -> object:
    """Return a connected bare ``MujocoArmHAL`` (Franka panda, native MuJoCo).

    Skips cleanly when ``mujoco`` is missing or the menagerie MJCF cannot be
    fetched (offline CI), per CLAUDE.md §1.11 (skip, never fake).
    """
    pytest.importorskip("mujoco")
    from openral_hal.franka_panda import FrankaPandaHAL

    hal = FrankaPandaHAL(gravity_enabled=False)
    try:
        hal.connect()
    except ROSConfigError as exc:  # missing mujoco / un-fetchable MJCF on this host
        pytest.skip(f"cannot build native MuJoCo arm: {exc}")
    return hal


def test_read_state_recovers_after_executor_stall_no_latch() -> None:
    """A >limit servicing gap must NOT latch ``read_state`` into permanent staleness.

    Reproduces the deploy-sim symptom deterministically: rewind the staleness
    clock past the limit (as a stalled executor would) and confirm the *next*
    read self-heals — the data is the live ``MjData``, so a transient stall is a
    diagnostic, not a fatal latched ``ROSPerceptionStale``.
    """
    hal = _build_bare_arm()

    # A fresh read works.
    first = hal.read_state()  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
    assert first.position, "expected a non-empty joint vector"

    # Simulate a 1.0 s executor stall (> the 0.5 s default limit) before the
    # next serviced read — exactly what camera-render starvation produces.
    hal._last_state_time -= 1.0  # type: ignore[attr-defined]  # reason: white-box stall injection

    # Must return live state, not raise, and must leave the clock fresh so the
    # call after it is trivially within the limit (the anti-latch guarantee).
    recovered = hal.read_state()  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
    assert recovered.position == pytest.approx(first.position)
    assert (time.monotonic() - hal._last_state_time) < 0.5  # type: ignore[attr-defined]

    # The latch regression would raise on this third call; it must not.
    again = hal.read_state()  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
    assert again.position == pytest.approx(first.position)


def test_idle_step_advances_state_and_is_recoverable() -> None:
    """``idle_step`` HOLD-steps the sim, returns True, and refreshes the clock.

    Gives the bare arm the ADR-0049 dedicated-thread + ADR-0034 idle-camera
    treatment: the lifecycle gates both on a callable ``idle_step``.
    """
    hal = _build_bare_arm()
    assert callable(getattr(hal, "idle_step", None)), "MujocoArmHAL must expose idle_step"

    hal._last_state_time -= 1.0  # type: ignore[attr-defined]  # reason: prove idle_step refreshes it
    assert hal.idle_step() is True  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
    # idle_step advanced the sim → clock fresh, so the publisher thread's
    # snapshot read is never stale while idle.
    assert (time.monotonic() - hal._last_state_time) < 0.5  # type: ignore[attr-defined]


def test_idle_step_returns_false_after_estop() -> None:
    """SAFETY: a latched estop (disconnected HAL) must make ``idle_step`` a no-op False.

    The idle stepper must never autonomously step a HAL that has e-stopped.
    """
    hal = _build_bare_arm()
    with pytest.raises(ROSEStopRequested):
        hal.estop()  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface; estop always raises
    assert hal.idle_step() is False  # type: ignore[attr-defined]  # reason: disconnected → no step


def test_last_action_ns_updates_on_send_action() -> None:
    """``last_action_ns`` (the idle-step hand-off timestamp) advances on each action."""
    pytest.importorskip("numpy")
    hal = _build_bare_arm()
    from openral_core import Action, ControlMode

    assert int(hal.last_action_ns) == 0  # type: ignore[attr-defined]  # never actuated

    state = hal.read_state()  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface
    hold = list(state.position)
    action = Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[hold])
    hal.send_action(action)  # type: ignore[attr-defined]  # reason: MujocoArmHAL surface

    assert int(hal.last_action_ns) > 0  # type: ignore[attr-defined]  # stamped by send_action
