"""Deploy-sim continuous mode for the LIBERO backend (ADR-0036).

lerobot's ``LiberoEnv.step`` resets the episode *inline* the instant the task
succeeds or the horizon is hit (``if terminated: self.reset()``), re-randomising
the whole scene mid-mission and re-creating the MjData (which orphans the passive
viewer). For a continuous deploy twin the reasoner/mission own episode
boundaries, so :meth:`_LiberoSim.enable_continuous` (called by ``SimAttachedHAL``)
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
        scene=scene, task=task, vla=VLASpec(id="zero", weights_uri="local://none"),
        robot_id="franka_panda",
    )
    return _build_libero_scene(env_cfg)


def test_enable_continuous_never_terminates_or_resets_past_horizon() -> None:
    # Tiny horizon so the default would terminate (and lerobot reset inline) by
    # step 3; continuous must keep going and keep the arm in place.
    sim = _build(max_steps=3)
    try:
        sim.reset(seed=0)
        sim.enable_continuous()
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
