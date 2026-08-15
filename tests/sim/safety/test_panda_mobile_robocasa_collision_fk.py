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
# scipy arrives transitively with robosuite; it is not a declared OpenRAL dependency,
# so the lean CI env has to skip rather than fail collection.
_SCIPY_MISSING = importlib.util.find_spec("scipy") is None

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(bool(_ROBOCASA_ERROR), reason=_ROBOCASA_ERROR or "RoboCasa unavailable"),
    pytest.mark.skipif(
        bool(_RENDERER_ERROR), reason=_RENDERER_ERROR or "no MuJoCo offscreen renderer"
    ),
    pytest.mark.skipif(_SCIPY_MISSING, reason="scipy not installed"),
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


def test_arm_boxes_enclose_the_real_robocasa_collision_meshes(
    panda_mobile_hal: Any,
) -> None:
    """Every world-collision OBB contains its matching RoboCasa arm mesh."""
    import mujoco
    from openral_core import BoxShape
    from scipy.spatial.transform import Rotation

    model, _data = panda_mobile_hal.mujoco_handles()
    for link_index in range(1, 8):
        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"robot0_link{link_index}_collision",
        )
        geometry = next(
            item
            for item in panda_mobile_hal.description.collision_geometry
            if item.link_name == f"panda_link{link_index}"
        )
        assert isinstance(geometry.shape, BoxShape)

        aabb_center = np.asarray(model.geom_aabb[geom_id, :3], dtype=np.float64)
        aabb_half = np.asarray(model.geom_aabb[geom_id, 3:], dtype=np.float64)
        geom_rotation = Rotation.from_quat(
            [
                model.geom_quat[geom_id, 1],
                model.geom_quat[geom_id, 2],
                model.geom_quat[geom_id, 3],
                model.geom_quat[geom_id, 0],
            ]
        ).as_matrix()
        corners = np.asarray(
            [
                aabb_center + aabb_half * np.asarray([x, y, z], dtype=np.float64)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ]
        )
        corners_in_link = corners @ geom_rotation.T + np.asarray(model.geom_pos[geom_id])

        box_origin = np.asarray(geometry.origin_xyz_rpy[:3])
        box_rotation = Rotation.from_euler("xyz", geometry.origin_xyz_rpy[3:]).as_matrix()
        corners_in_box = (corners_in_link - box_origin) @ box_rotation

        assert np.all(
            np.abs(corners_in_box)
            <= np.asarray(geometry.shape.half_extents_m, dtype=np.float64) + 1e-6
        ), geometry.link_name
