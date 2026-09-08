"""Integration coverage for the sim-only idle stepper's WIRED bridge+HAL path.

2026-06-04 idle-stepper amendment added this coverage. The unit suite
(``python/hal/tests/test_sim_attached_idle_step.py``) already covers
``SimAttachedHAL.idle_step`` in isolation (frame-advance, estop-suppress, terminal-reset,
crash-containment) and the pure ``should_idle_step`` predicate. Deferred and added here:
integration-level proof that the WIRED ``SimSensorBridge`` (real lifecycle node) +
``SimAttachedHAL`` (real sim env) together keep an idle scene live and yield to an active
action stream. Regression guarded: an idle ``openral deploy sim`` scene freezing the
perception bus, since the env only steps on ``/openral/safe_action`` receipt, so cameras go
stale with no skill running.

Test form — in-process real-component integration, not ``launch_testing``: a launch_testing
variant asserting on the idle timer firing on its own wall-clock cadence is exactly the
flakiness source this test was deferred to avoid — the assertion hinges on the precise
interleaving of the rclpy timer, the camera-publish timer, and ``last_action_ns``, none of
which a black-box launch test can pin deterministically (a slow CI host, GC pause, or
executor jitter flips the result).

So this test wires the real components — a real ``rclpy`` ``LifecycleNode`` (genuine
``create_timer``/``create_publisher``/``get_logger``), a real ``SimSensorBridge``, a real
``SimAttachedHAL`` over a real native-MuJoCo sim env (no mocks, CLAUDE.md §1.11) — and drives
the bridge's own idle-tick callback (``_idle_step_tick``) directly a controlled number of
times rather than waiting on the wall-clock timer, exercising the full wiring (bridge
predicate → ``hal.idle_step`` → ``env.step`` → ``read_images`` re-cache) deterministically.
Timing tolerances are generous (multi-second idle-hold) so the yield assertion never races
the clock.

Backend: native-MuJoCo ``so101_box`` scene (``scenes/sim/so101_tube_insertion.yaml``) —
exercises ``SimSensorBridge`` with live ``mujoco_handles`` + rendered camera frames, builds in
~5 s, needs neither ``libero`` nor ``robocasa`` (neither installed here), avoiding the ~60 s
robocasa kitchen build entirely.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

# A real rclpy LifecycleNode is constructed below, so a sourced ROS 2 is
# required (the SimSensorBridge timer/publisher/logger surface is genuine).
_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — this test constructs a real rclpy LifecycleNode.",
)

# A multi-second idle-hold so the yield-under-load assertion can never race the
# wall clock: a real send_action stamp followed (microseconds later) by an idle
# tick stays well inside the hold even on a slow host. Far larger than the
# production 200 ms default; we drive the tick deterministically, so the only
# requirement is that the stamp-to-tick gap < hold, which this guarantees.
_IDLE_HOLD_MS = 5_000.0


def _first_hwc_frame(images: dict[str, Any]) -> NDArray[Any]:
    """Return the first HWC (RGB) frame in a ``read_images()`` dict."""
    for arr in images.values():
        a = np.asarray(arr)
        if a.ndim == 3:  # reason: HWC image
            return a
    raise AssertionError(f"no HWC frame in images keys={sorted(images)}")


@pytest.fixture
def wired_bridge_and_hal() -> Iterator[tuple[Any, Any]]:
    """Yield a real (SimSensorBridge, SimAttachedHAL) wired to a real sim env.

    Skips cleanly when sim deps (mujoco/openral_sim) are absent. Uses the native-MuJoCo so101
    box scene — its action width isn't introspectable, so ``env_action_dim=6`` is passed
    explicitly (mirrors the unit suite's ``_build_so101_hal``).
    """
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")

    import rclpy
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml
    from openral_hal.sim_sensor_bridge import SimSensorBridge
    from rclpy.lifecycle import LifecycleNode

    rclpy.init()
    node: Any = None
    bridge: Any = None
    try:
        env, seed = build_sim_env_from_yaml(
            "scenes/sim/so101_tube_insertion.yaml",
            robot_id_fallback="so101_follower",
        )
        desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
        hal = SimAttachedHAL(env, desc, env_reset_seed=seed, env_action_dim=6)
        hal.connect()

        node = LifecycleNode("test_sim_idle_stepper_wiring")
        bridge = SimSensorBridge(node, hal, desc, viewer_enabled=False, idle_hold_ms=_IDLE_HOLD_MS)
        # Wire the real idle timer through the production setup path. Both gates must hold:
        # callable idle_step AND live MuJoCo handles (so101_box does) — assert the timer
        # exists so a broken wiring fails loudly rather than vacuously.
        bridge._setup_idle_stepper()
        assert bridge._idle_timer is not None, (
            "idle timer not created — both gates (callable idle_step + live "
            "mujoco_handles) should hold for the so101_box native backend"
        )
        yield bridge, hal
    finally:
        if bridge is not None:
            bridge.teardown()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_idle_liveness_wired_bridge_keeps_scene_advancing(
    wired_bridge_and_hal: tuple[Any, Any],
) -> None:
    """IDLE LIVENESS: with no action stream, driving the bridge's idle tick keeps frames advancing.

    Drives the real ``SimSensorBridge._idle_step_tick`` callback (same code the production
    timer fires) several times against a real wired HAL+env with no actions sent —
    ``last_action_ns == 0`` makes ``should_idle_step`` engage every tick, so ``read_images()``
    advances.

    Asserts on frame distinctness, never an exact step count or sleep: the frozen-scene
    regression leaves frames byte-identical, a live scene doesn't. No wall-clock timing
    relied on.
    """
    bridge, hal = wired_bridge_and_hal

    before = _first_hwc_frame(hal.read_images()).copy()
    # Several idle ticks so settling dynamics produce a visible delta; driven
    # directly (deterministic) rather than awaiting the wall-clock timer.
    for _ in range(8):
        bridge._idle_step_tick()
    after = _first_hwc_frame(hal.read_images())

    assert before.shape == after.shape
    # Wired bridge → hal.idle_step → env.step → read_images re-cache advanced
    # the scene. The frozen-perception-bus regression would leave these equal.
    assert not np.array_equal(before, after), (
        "idle scene did not advance — the bridge's idle tick failed to step the "
        "wired env (frozen-perception-bus regression)"
    )


def test_idle_stepper_yields_under_active_action_stream(
    wired_bridge_and_hal: tuple[Any, Any],
) -> None:
    """YIELD UNDER LOAD: a real send_action suppresses the idle tick (no double-step).

    Sends a real ``Action`` through ``hal.send_action`` (stamps ``last_action_ns``), then
    drives the idle tick — while the action is recent (within the idle-hold), the tick must
    not also step the env. Counts actual idle steps via a spy wrapping ``hal.idle_step``
    (``True`` return only). Generous ``_IDLE_HOLD_MS`` (5 s) keeps the send-to-tick gap always
    inside the hold. Also confirms the env resumes idle-stepping once the action is far enough
    in the past, proving suppression is the hold predicate, not a permanently-wedged stepper.
    """
    from openral_core.schemas import Action, ControlMode

    bridge, hal = wired_bridge_and_hal

    # Count idle steps that actually fired (idle_step() -> True). The spy must
    # mirror the real signature — the bridge calls
    # ``idle_step(wall_dt_s=1/camera_rate_hz)``, and a spy that rejected the
    # keyword made the bridge log "idle stepper disabled after error" and
    # latch the stepper OFF, so the resume assertion below failed for a reason
    # that had nothing to do with the hold predicate under test.
    fired: dict[str, int] = {"n": 0}
    real_idle_step = hal.idle_step

    def _counting_idle_step(wall_dt_s: float | None = None) -> bool:
        stepped = bool(real_idle_step(wall_dt_s))
        if stepped:
            fired["n"] += 1
        return stepped

    hal.idle_step = _counting_idle_step  # test spy over the real method (hal is Any here)

    # so101_box exposes 6 arm joints (all role "unknown") — a 6-wide
    # JOINT_POSITION row is a valid real action for this manifest.
    n_arm = sum(
        1
        for j in hal.description.joints
        if j.role != "gripper"
        and (hal.description.base_joints is None or j.name not in hal.description.base_joints)
    )
    action = Action(
        control_mode=ControlMode.JOINT_POSITION,
        joint_targets=[[0.0] * n_arm],
    )

    # Active stream: send a real action, then immediately tick the idle path.
    hal.send_action(action)
    assert hal.last_action_ns > 0, "send_action did not stamp last_action_ns"
    bridge._idle_step_tick()
    assert fired["n"] == 0, (
        "idle stepper stepped while a real action was within the idle-hold "
        "window — it must yield to the active skill (no double-stepping)"
    )

    # Now make the last action look old (predicate engages) and tick again: the
    # idle stepper must resume, proving the suppression above was the hold
    # predicate yielding — not a permanently-disabled stepper. Rewinding
    # ``_last_action_ns`` past the hold is the deterministic equivalent of
    # waiting out the quiet window.
    hal._last_action_ns -= int(_IDLE_HOLD_MS * 1_000_000) * 2  # white-box rewind past the idle-hold
    bridge._idle_step_tick()
    assert fired["n"] == 1, (
        "idle stepper did not resume once the last action aged past the "
        "idle-hold window — suppression should be the hold predicate, not a "
        "wedged stepper"
    )
