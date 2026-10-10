"""Deploy-sim continuous mode for the LIBERO backend.

lerobot's ``LiberoEnv.step`` resets the episode *inline* the instant the task
succeeds or the horizon is hit (``if terminated: self.reset()``), re-randomising
the whole scene mid-mission and re-creating the MjData (which orphans the passive
viewer). For a continuous deploy twin the reasoner/mission own episode
boundaries, so ``_LiberoSim.enable_continuous`` (called by ``SimAttachedHAL``)
suppresses that. ``openral sim run`` keeps the per-episode reset.

Real LIBERO env, no mocks (CLAUDE.md §1.11); skips cleanly when the suite can't
be provisioned (missing deps / robosuite>=1.5 conflict / no GL backend).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_REQUIRED = ("robosuite", "libero", "mujoco")
_MISSING = tuple(m for m in _REQUIRED if importlib.util.find_spec(m) is None)


def _robosuite_conflict() -> bool:
    import importlib.metadata as md

    if importlib.util.find_spec("robosuite") is None:
        return False
    try:
        return not md.version("robosuite").startswith("1.4")
    except md.PackageNotFoundError:
        return False


pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(bool(_MISSING), reason=f"LIBERO deps missing: {_MISSING}"),
    pytest.mark.skipif(_robosuite_conflict(), reason="robosuite>=1.5 blocks the LIBERO runtime"),
]


def _build(max_steps: int):
    from openral_core import SceneSpec, SimEnvironment, TaskSpec, VLASpec
    from openral_sim.backends.libero import _build_libero_scene

    os.environ.setdefault("MUJOCO_GL", "egl")
    scene = SceneSpec(
        id="libero_object", backend="mujoco", observation_height=128, observation_width=128
    )
    task = TaskSpec(
        id="libero_object/0",
        scene_id="libero_object",
        instruction="",
        success_key="is_success",
        max_steps=max_steps,
    )
    env_cfg = SimEnvironment(
        scene=scene,
        task=task,
        vla=VLASpec(id="zero", weights_uri="local://none"),
        robot_id="franka_panda",
    )
    return _build_libero_scene(env_cfg)


def test_enable_continuous_never_terminates_or_resets_past_horizon() -> None:
    # Tiny horizon so the default would terminate (and lerobot reset inline) by
    # step 3; continuous must keep going and keep the arm in place.
    sim = _build(max_steps=3)
    try:
        # Production order (SimAttachedHAL): enable_continuous at __init__,
        # BEFORE connect()'s first reset. LiberoEnv builds its robosuite env
        # lazily inside reset, so an eager-only ignore_done lands on nothing and
        # the horizon `done` still hard-raises mid-goal — the live deploy-sim
        # mid-episode auto-reset. reset() must re-apply it.
        sim.enable_continuous()
        sim.reset(seed=0)
        # Regression: ignore_done must land on the env that actually latches `done`
        # (Libero_*_Manipulation), NOT the OffScreenRenderEnv wrapper — otherwise
        # the horizon/success `done` still hard-raises and the HAL re-randomises.
        rs = sim._robosuite_env()
        assert rs is not None and hasattr(rs, "horizon")
        assert type(rs).__name__ != "OffScreenRenderEnv"
        assert getattr(rs, "ignore_done", False) is True
        handles = sim.mujoco_handles()
        assert handles is not None, "need MjData to detect a reset's teleport"
        data = handles[1]
        prev = np.array(data.qpos[:7])
        terminated, max_jump = [], 0.0
        for _ in range(8):  # well past horizon=3
            result = sim.step(np.zeros(7, dtype=np.float32))
            terminated.append(result.terminated)
            cur = np.array(data.qpos[:7])
            max_jump = max(max_jump, float(np.abs(cur - prev).max()))
            prev = cur
        # Never terminates: lerobot's inline reset is suppressed + robosuite
        # ignore_done keeps the post-horizon step from raising.
        assert not any(terminated)
        # No reset: a re-randomising reset teleports the arm back to home (a large
        # qpos discontinuity); continuous only ever sees small physics deltas.
        assert max_jump < 0.5, f"arm teleported ({max_jump:.3f}) — the scene reset"
    finally:
        sim.close() if hasattr(sim, "close") else None


def test_a_pinned_control_rate_makes_one_step_one_control_period() -> None:
    """Regression for issue #358: ``backend_options.control_freq_hz`` holds LIBERO to the robot.

    Without the pin LIBERO steps at its native 20 Hz (the published benchmark
    protocol, unchanged); a deploy scene pins the 30 Hz ``franka_panda`` rate so
    30 steps advance 1.0 s of sim time and ``SimAttachedHAL.connect`` accepts it.
    """
    from openral_core import RobotDescription, SceneSpec, SimEnvironment, TaskSpec, VLASpec
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_sim.backends.libero import _build_libero_scene

    os.environ.setdefault("MUJOCO_GL", "egl")
    env_cfg = SimEnvironment(
        scene=SceneSpec(
            id="libero_object",
            backend="mujoco",
            observation_height=128,
            observation_width=128,
            backend_options={"control_freq_hz": 30.0},
        ),
        task=TaskSpec(
            id="libero_object/0",
            scene_id="libero_object",
            instruction="",
            success_key="is_success",
            max_steps=50,
        ),
        vla=VLASpec(id="zero", weights_uri="local://none"),
        robot_id="franka_panda",
    )
    sim = _build_libero_scene(env_cfg)
    try:
        sim.reset(seed=0)
        assert sim.sim_dt_per_tick_s == pytest.approx(1 / 30, rel=0.01)
        zero = np.zeros(sim.action_dim, dtype=np.float32)
        sim.step(zero)
        t0 = sim.sim_time_ns()
        for _ in range(30):
            sim.step(zero)
        t1 = sim.sim_time_ns()
        assert t0 is not None and t1 is not None
        assert (t1 - t0) / 1e9 == pytest.approx(1.0, rel=0.01)
        hal = SimAttachedHAL(sim, RobotDescription.from_yaml("robots/franka_panda/robot.yaml"))
        hal.connect()
    finally:
        sim.close() if hasattr(sim, "close") else None


def test_the_benchmark_rate_stays_at_libero_native_20_hz() -> None:
    """No pin: the eval tier keeps the 20 Hz protocol its published numbers were measured at."""
    sim = _build(max_steps=50)
    try:
        sim.reset(seed=0)
        assert sim.sim_dt_per_tick_s == pytest.approx(1 / 20, rel=0.01)
    finally:
        sim.close() if hasattr(sim, "close") else None
