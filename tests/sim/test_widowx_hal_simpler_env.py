"""Sim test: ``SimAttachedHAL.read_state`` publishes SimplerEnv's measured WidowX joints.

The SimplerEnv rollout used to return only ``state`` (qpos + qvel, unnamed), so a
scene-attached ``openral deploy sim`` on ``widowx`` published an all-zero
``/joint_states``. It now emits ``joint_names`` / ``joint_positions`` /
``joint_velocities`` off the live ManiSkill articulation, the ``left_finger`` slide as
the manifest's normalised ``gripper``.

What is asserted, on the real ManiSkill3 Bridge env and the real ``widowx`` manifest:

* every manifest joint is reported, by name, equal to the articulation's own qpos;
* the gripper is ``left_finger`` normalised over the articulation's limits, in
  ``[0, 1]``;
* after a step the state still matches the articulation (it is re-read per obs).

Skips when ``mani_skill`` / ``simpler_env`` are missing or SAPIEN has no Vulkan device.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_MISSING = tuple(
    m for m in ("mani_skill", "simpler_env", "gymnasium") if importlib.util.find_spec(m) is None
)

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_MISSING),
        reason="SimplerEnv sim test requires "
        + ", ".join(_MISSING)
        + " (just sync --group simpler-env, then `openral install simpler-env`)",
    ),
]


@pytest.fixture(scope="module")
def hal() -> Iterator[Any]:
    import openral_sim.backends  # noqa: F401 — registers the scene factory
    from openral_core import RobotDescription, SimEnvironment, SimScene, VLASpec
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_sim.registry import SCENES

    scene_env = SimScene.from_yaml(
        str(_REPO / "scenes" / "benchmark" / "widowx_carrot_on_plate.yaml")
    )
    env_cfg = SimEnvironment(
        robot_id="widowx",
        scene=scene_env.scene,
        task=scene_env.task,
        vla=VLASpec(id="zero", weights_uri="stub"),
        seed=0,
        n_episodes=1,
    )
    try:
        rollout = SCENES.get("simpler_env")(env_cfg)
    except RuntimeError as exc:
        if "vk::" in str(exc) or "vulkan" in str(exc).lower():
            pytest.skip(f"SAPIEN rendering needs a working Vulkan driver: {exc}")
        raise
    desc = RobotDescription.from_yaml(str(_REPO / "robots" / "widowx" / "robot.yaml"))
    sim_hal = SimAttachedHAL(rollout, desc)
    sim_hal.connect()
    yield sim_hal
    sim_hal.disconnect()
    rollout.close()


def _articulation(hal: Any) -> dict[str, tuple[float, float, float]]:
    """``{joint: (qpos, lower, upper)}`` read straight off the ManiSkill robot."""
    robot = hal.env._env.unwrapped.agent.robot
    names = [j.name for j in robot.get_active_joints()]
    qpos = robot.get_qpos().cpu().numpy().reshape(-1)
    lim = robot.get_qlimits().cpu().numpy().reshape(-1, 2)
    return {n: (float(qpos[i]), float(lim[i, 0]), float(lim[i, 1])) for i, n in enumerate(names)}


def _assert_matches_articulation(hal: Any) -> None:
    state = hal.read_state()
    art = _articulation(hal)
    assert state.name == [j.name for j in hal.description.joints]
    for name, value in zip(state.name, state.position, strict=True):
        if name == "gripper":
            q, lo, hi = art["left_finger"]
            expected = (q - lo) / (hi - lo)
            assert 0.0 <= value <= 1.0
        else:
            expected = art[name][0]
        assert value == pytest.approx(expected, abs=1e-5), name
    assert len(state.velocity) == len(state.name)
    print(
        f"/joint_states: {dict(zip(state.name, np.round(state.position, 4).tolist(), strict=True))}"
    )


def test_read_state_is_the_articulation_at_reset(hal: Any) -> None:
    _assert_matches_articulation(hal)


def test_read_state_tracks_the_articulation_after_steps(hal: Any) -> None:
    for _ in range(5):
        assert hal.idle_step() is True
    _assert_matches_articulation(hal)
