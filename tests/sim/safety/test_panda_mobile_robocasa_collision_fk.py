"""RoboCasa's ground-level mobile base and the safety FK share one frame."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.sim.conftest import mujoco_renderer_probe_error

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROBOT = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
_SCENE = _REPO_ROOT / "scenes" / "deploy" / "robocasa_baguette.yaml"


def _robocasa_unavailable() -> str:
    if importlib.util.find_spec("robocasa") is None:
        return "robocasa not installed"
    from openral_sim._deps import _has_robocasa_kitchen

    return "" if _has_robocasa_kitchen() else "RoboCasa kitchen fork is not active"


_ROBOCASA_ERROR = _robocasa_unavailable()
_RENDERER_ERROR = mujoco_renderer_probe_error() if not _ROBOCASA_ERROR else ""

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(bool(_ROBOCASA_ERROR), reason=_ROBOCASA_ERROR or "RoboCasa unavailable"),
    pytest.mark.skipif(
        bool(_RENDERER_ERROR), reason=_RENDERER_ERROR or "no MuJoCo offscreen renderer"
    ),
]


@pytest.fixture(scope="module")
def panda_mobile_hal() -> Iterator[Any]:
    from openral_core import RobotDescription
    from openral_hal import build_hal

    hal = build_hal(
        RobotDescription.from_yaml(_ROBOT),
        mode="sim",
        sim_env_yaml=str(_SCENE),
    )
    hal.connect()
    try:
        yield hal
    finally:
        hal.disconnect()


def test_collision_root_matches_robocasa_mobile_base(panda_mobile_hal: Any) -> None:
    """The manifest joint-1 origin matches RoboCasa link1 relative to base_link."""
    import mujoco

    model, data = panda_mobile_hal.mujoco_handles()
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_base")
    link1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_link1")
    base_rotation = np.asarray(data.xmat[base_id], dtype=np.float64).reshape(3, 3)
    link1_in_base = base_rotation.T @ (
        np.asarray(data.xpos[link1_id], dtype=np.float64)
        - np.asarray(data.xpos[base_id], dtype=np.float64)
    )
    joint1 = next(
        joint for joint in panda_mobile_hal.description.joints if joint.name == "panda_joint1"
    )

    assert link1_in_base == pytest.approx(joint1.origin_xyz, abs=1e-6)
