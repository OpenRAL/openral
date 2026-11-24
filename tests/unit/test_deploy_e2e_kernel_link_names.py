"""Link-name parameters reach an MJCF-lowered kernel in that model's naming (#754 review).

``deploy_e2e.launch.py`` loads the kernel's collision model from the robot's MJCF when it
yields one, and that model names each link after its MJCF body. The grasp exemption's
``grasp_contact_links`` and ``tracked_grasp_extra_contact_links`` were passed in the
manifest's naming, which such a kernel refuses at configure. ``_mjcf_kernel_link_names``
translates them as the collision lowering maps links to bodies. Real manifests, real MJCFs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH = REPO_ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"


@pytest.fixture(scope="module")
def launch_module() -> Any:
    """The real launch file, loaded by path — it is not an importable package."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    pytest.importorskip("openral_safety", reason="the collision lowering lives in openral_safety")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_link_names", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_link_names"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("robot_id", "expected"),
    [
        ("so100_follower", {"gripper_finger": "Moving_Jaw", "gripper_base": "Fixed_Jaw"}),
        ("franka_panda", {"panda_finger_pair": "left_finger", "panda_hand": "hand"}),
    ],
)
def test_grasp_links_are_named_as_the_mjcf_kernel_names_them(
    launch_module: Any, robot_id: str, expected: dict[str, str]
) -> None:
    mujoco = pytest.importorskip("mujoco")
    from openral_core import RobotDescription
    from openral_core.assets import AssetFetchError, resolve_asset
    from openral_safety.mjcf_lowering import lower_collision_params

    desc = RobotDescription.from_yaml(str(REPO_ROOT / "robots" / robot_id / "robot.yaml"))
    assert desc.assets.mjcf is not None
    try:
        path = resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetFetchError as exc:
        pytest.skip(f"{robot_id} twin not fetchable: {exc}")
    model = mujoco.MjModel.from_xml_path(str(path))
    names = launch_module._mjcf_kernel_link_names(desc, model)
    assert {k: names[k] for k in expected} == expected
    lowered = lower_collision_params(model, [j.name for j in desc.joints])
    kernel = set(lowered["collision_link_names"])
    wanted = [*desc.grasp_contact_links(), *desc.fixed_jaw_links()]
    assert {names.get(link, link) for link in wanted} <= kernel
