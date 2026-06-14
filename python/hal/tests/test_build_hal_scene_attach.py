# python/hal/tests/test_build_hal_scene_attach.py
"""build_hal scene-attach path (ADR-0034): sim_env_yaml -> SimAttachedHAL."""

from __future__ import annotations

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_hal import build_hal

pytest.importorskip("openral_sim")  # scene backends are an optional group
pytest.importorskip("mujoco")

_FRANKA = "robots/franka_panda/robot.yaml"
_SCENE = "scenes/sim/tabletop_cube_push.yaml"  # native MjSpec scene: mujoco-only, no robosuite/GPU


def test_sim_env_yaml_returns_sim_attached_hal() -> None:
    from openral_hal.sim_attached import SimAttachedHAL

    desc = RobotDescription.from_yaml(_FRANKA)
    hal = build_hal(desc, mode="sim", sim_env_yaml=_SCENE)
    assert isinstance(hal, SimAttachedHAL)
    assert hal.description.name == "franka_panda"


def test_sim_without_scene_still_builds_bare_twin() -> None:
    desc = RobotDescription.from_yaml(_FRANKA)
    hal = build_hal(desc, mode="sim")  # no sim_env_yaml
    assert type(hal).__name__ != "SimAttachedHAL"


def test_real_mode_with_sim_env_yaml_raises() -> None:
    desc = RobotDescription.from_yaml(_FRANKA)
    with pytest.raises(ROSConfigError, match="sim_env_yaml"):
        build_hal(desc, mode="real", sim_env_yaml=_SCENE)


def test_franka_sim_joint_names_match_native_mjcf() -> None:
    desc = RobotDescription.from_yaml(_FRANKA)
    sim_names = {j.sim_joint_name or j.name for j in desc.joints if j.role != "gripper"}
    assert {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"} <= sim_names
