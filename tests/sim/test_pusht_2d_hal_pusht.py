"""Sim test: ``SimAttachedHAL.read_state`` publishes the PushT tip as the measured state.

The PushT rollout used to return only ``state``, so a scene-attached ``openral deploy
sim`` on ``pusht_2d`` published ``/joint_states`` of ``(0, 0)``. It now emits the tip
position (``agent_pos``, canvas pixels — the manifest's ``tip_x`` / ``tip_y`` units)
as ``joint_positions``.

Asserted on the real gym-pusht env and the real ``pusht_2d`` manifest: ``read_state``
equals the env's own ``agent_pos`` at reset and after a step toward a target.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

_REPO = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        importlib.util.find_spec("gym_pusht") is None,
        reason="PushT sim test requires gym_pusht (just sync --group pusht)",
    ),
]


@pytest.fixture(scope="module")
def hal() -> Iterator[Any]:
    import openral_sim.backends  # noqa: F401 — registers the scene factory
    from openral_core import (
        BenchmarkScene,
        RobotDescription,
        SimEnvironment,
        VLASpec,
        load_scene_strict,
    )
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_sim.registry import SCENES

    sc = load_scene_strict(str(_REPO / "scenes" / "benchmark" / "pusht.yaml"), BenchmarkScene)
    env_cfg = SimEnvironment(
        robot_id="pusht_2d",
        scene=sc.scene,
        task=sc.task,
        vla=VLASpec(id="zero", weights_uri="stub"),
        seed=0,
        n_episodes=1,
    )
    rollout = SCENES.get("pusht")(env_cfg)
    desc = RobotDescription.from_yaml(str(_REPO / "robots" / "pusht_2d" / "robot.yaml"))
    sim_hal = SimAttachedHAL(rollout, desc)
    sim_hal.connect()
    yield sim_hal
    sim_hal.disconnect()
    rollout.close()


def _agent_pos(hal: Any) -> NDArray[np.float64]:
    return np.asarray(hal.env._env.unwrapped.agent.position, dtype=np.float64)


def test_read_state_is_the_tip_at_reset(hal: Any) -> None:
    state = hal.read_state()
    assert state.name == ["tip_x", "tip_y"]
    np.testing.assert_allclose(state.position, _agent_pos(hal), atol=1e-3)
    assert state.velocity == []  # gym-pusht reports no tip velocity: none, not zeros
    print(f"/joint_states at reset: {dict(zip(state.name, state.position, strict=True))}")


def test_read_state_follows_the_tip(hal: Any) -> None:
    from openral_core import Action, ControlMode

    start = np.asarray(hal.read_state().position)
    target = np.clip(start + 60.0, 20.0, 490.0)
    for _ in range(10):
        hal.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[target.tolist()])
        )
    state = hal.read_state()
    np.testing.assert_allclose(state.position, _agent_pos(hal), atol=1e-3)
    assert np.linalg.norm(np.asarray(state.position) - start) > 5.0
    print(f"/joint_states after 10 steps toward {target.tolist()}: {state.position}")
