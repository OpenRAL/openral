"""The perception legs' default camera (``_primary_rgb_camera``) for every in-tree robot.

The continuous detector, the on-demand locators' default ``locate_in_view`` camera and the
reasoner's completion camera all follow it, so a change here moves all three: this pins the
answer per manifest so any change is a visible diff. The OpenAMRobot 2.0 must get its
context camera ``top`` (``head_zed``'s colour stream), not a wrist camera.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LAUNCH = _ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"

EXPECTED = {
    "aloha_agilex": "top",
    "aloha_bimanual": "top",
    "anvil_openarm_v2": "wrist_left",
    "droid_franka_robotiq": "exterior",
    "franka_panda": "top",
    "g1": "head",
    "galaxea_a1": "front",
    "google_robot": "head",
    "gr1": "head",
    "h1": "",
    "h2": "",
    "makermods_metal": "top",
    "makermods_metal_station": "top",
    "openamrobot_v2": "top",
    "openarm": "top",
    "panda_mobile": "shoulder_left",
    "panda_mobile_vslam": "shoulder_left",
    "pusht_2d": "top",
    "r1pro": "head",
    "rizon4": "",
    "sawyer": "front",
    "so100_follower": "top",
    "so101_follower": "top",
    "ur10e": "",
    "ur5e": "",
    "widowx": "top",
}


@pytest.fixture(scope="module")
def launch() -> object:
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_primary_rgb", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_primary_rgb"] = module
    spec.loader.exec_module(module)
    return module


def test_every_robot_manifest_is_pinned() -> None:
    found = {p.parent.name for p in (_ROOT / "robots").glob("*/robot.yaml")}
    assert found == set(EXPECTED), "add/remove the robot's primary camera in EXPECTED"


@pytest.mark.parametrize(("robot", "camera"), sorted(EXPECTED.items()))
@pytest.mark.parametrize("hal_mode", ["sim", "real"])
def test_primary_rgb_camera(launch: object, robot: str, camera: str, hal_mode: str) -> None:
    from openral_core import RobotDescription, publishing_sensors

    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / robot / "robot.yaml"))
    publishing = publishing_sensors(desc.sensors, [], hal_mode)
    got = launch._primary_rgb_camera(publishing)  # type: ignore[attr-defined]
    if hal_mode == "sim":
        assert got == camera
    else:  # real: only bound cameras publish; the pick must be one of them
        assert got == "" or any(s.name == got for s in publishing)
    if robot == "openamrobot_v2":
        assert got == "top"
