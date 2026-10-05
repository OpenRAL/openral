"""The world map covers each robot's own reach, measured from the kernel's collision model.

``deploy_e2e.launch.py`` mapped one fixed ball, 1.05 m about ``(0, 0, 0.5)`` in
``base_frame``, measured on panda_mobile. The OpenArm's ``base_frame`` is its torso
top and its arms hang down: their reach went 108 mm below that ball, so a bin the
arm could reach was never in the kernel's world map (an empty cell reads as free).
``_coverage_ball`` now measures centre and radius per robot over its joint limits.
These check it on the real manifests against an independent, denser sample of the
same kinematics — the reach the kernel's own forward kinematics would pose.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH = REPO_ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"
# What must stay inside the ball beyond a link's surface: the kernel's real world
# margin (20 mm) plus half a 20 mm cell, the window it scans around each link.
KERNEL_WINDOW_M = 0.03


@pytest.fixture(scope="module")
def launch_module() -> object:
    """The real launch file, loaded by path — it is not an importable package."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_coverage", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_coverage"] = module
    spec.loader.exec_module(module)
    return module


def _robot(robot_id: str) -> tuple[dict[str, object], list[tuple[float, float]], list[int]]:
    """The kernel's collision params, joint limits and base dofs for a real manifest."""
    pytest.importorskip("openral_safety", reason="the collision lowering lives in openral_safety")
    from openral_core import RobotDescription
    from openral_safety.envelope_loader import collision_params_from_description

    desc = RobotDescription.from_yaml(str(REPO_ROOT / "robots" / robot_id / "robot.yaml"))
    base = set(desc.base_joints or [])
    return (
        collision_params_from_description(desc),
        [j.position_limits for j in desc.joints],
        [i for i, j in enumerate(desc.joints) if j.name in base],
    )


def _dense_reach(
    launch_module: object, robot_id: str, seed: int = 11
) -> tuple[np.ndarray, np.ndarray]:
    """``(points, radii)`` of every moving primitive over 131 072 random configurations."""
    params, limits, base = _robot(robot_id)
    model = launch_module._ReachModel(params, limits, base)  # type: ignore[attr-defined]
    rng = np.random.default_rng(seed)
    lo, hi = model.limits[:, 0], model.limits[:, 1]
    pts, rads = [], []
    for _ in range(2):
        pick = rng.integers(0, 3, size=(65536, len(lo)))
        q = np.where(pick == 0, lo, np.where(pick == 1, hi, rng.uniform(lo, hi, pick.shape)))
        p, r = model.extremes(q)
        pts.append(p.reshape(-1, 3))
        rads.append(np.tile(r, len(q)))
    return np.concatenate(pts), np.concatenate(rads)


@pytest.mark.parametrize("robot_id", ["openarm", "panda_mobile", "franka_panda"])
def test_every_checked_link_stays_inside_the_ball_and_the_clip(
    launch_module: object, robot_id: str
) -> None:
    """No sampled pose puts a moving primitive's check window outside the map."""
    params, limits, base = _robot(robot_id)
    ball = launch_module._coverage_ball(params, limits, base, 0.015)  # type: ignore[attr-defined]
    points, radii = _dense_reach(launch_module, robot_id)
    reach = np.linalg.norm(points - np.asarray(ball.centre), axis=1) + radii
    assert reach.max() <= ball.radius - KERNEL_WINDOW_M, (robot_id, ball, reach.max())
    assert np.all(points - radii[:, None] >= np.asarray(ball.clip_lo) + KERNEL_WINDOW_M)
    assert np.all(points + radii[:, None] <= np.asarray(ball.clip_hi) - KERNEL_WINDOW_M)
    # The clip lies inside the ball's bounding box, so octomap never integrates a return
    # the bridge would not publish anyway.
    for axis in range(3):
        assert ball.clip_lo[axis] >= ball.centre[axis] - ball.radius
        assert ball.clip_hi[axis] <= ball.centre[axis] + ball.radius


def test_openarm_ball_reaches_below_its_hanging_arms(launch_module: object) -> None:
    """The OpenArm's lowest reach is mapped; under the old fixed ball it was not."""
    params, limits, base = _robot("openarm")
    ball = launch_module._coverage_ball(params, limits, base, 0.015)  # type: ignore[attr-defined]
    points, radii = _dense_reach(launch_module, "openarm")
    bottom = points[:, 2] - radii
    lowest = int(np.argmin(bottom))
    assert ball.centre[2] - ball.radius <= bottom[lowest] - KERNEL_WINDOW_M
    assert ball.clip_lo[2] <= bottom[lowest] - KERNEL_WINDOW_M
    # The regression this replaces: the old ball's surface directly above that point.
    x, y = points[lowest, :2]
    old_bottom = 0.5 - math.sqrt(1.05**2 - x**2 - y**2)
    assert bottom[lowest] < old_bottom


def test_a_ball_the_bridge_cannot_publish_is_refused_at_launch(launch_module: object) -> None:
    """Past the bridge's cell cap the grid would be published empty; refuse instead."""
    from openral_core.exceptions import ROSConfigError

    params, limits, base = _robot("franka_panda")
    with pytest.raises(ROSConfigError, match="more than the voxel bridge publishes"):
        launch_module._coverage_ball(params, limits, base, 0.005)  # type: ignore[attr-defined]


def test_a_robot_without_a_collision_model_keeps_the_default_ball(launch_module: object) -> None:
    """No kernel voxel check reads its map; the pre-per-robot ball stays, said out loud."""
    ball = launch_module._coverage_ball(  # type: ignore[attr-defined]
        {"self_collision_enabled": False}, [], [], 0.015
    )
    assert ball.centre == (0.0, 0.0, 0.5)
    assert ball.radius == 1.05
