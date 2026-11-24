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


def test_hand_links_reach_the_kernel_in_its_naming(launch_module: Any) -> None:
    """Workspace-contact and exemption-provider hand links are renamed like the grasp ones."""
    params = {
        "workspace_contact_links": ["gripper_finger", "wrist"],
        "exemption_provider_links": ["gripper_base"],
        "grasp_contact_links": ["Moving_Jaw"],
    }
    names = {"gripper_finger": "Moving_Jaw", "gripper_base": "Fixed_Jaw", "wrist": "Wrist"}
    out = launch_module._hand_links_in_kernel_naming(params, names)
    assert out["workspace_contact_links"] == ["Moving_Jaw", "Wrist"]
    assert out["exemption_provider_links"] == ["Fixed_Jaw"]
    assert out["grasp_contact_links"] == ["Moving_Jaw"]
    assert launch_module._hand_links_in_kernel_naming(params, {}) == params


def test_a_real_deploy_refuses_geometry_fitted_to_a_twin_alone(launch_module: Any) -> None:
    """ALOHA's kernel model is gym-aloha's (no URDF of a real rig): sim only (#754 review)."""
    from openral_core import RobotDescription
    from openral_core.exceptions import ROSConfigError

    aloha = RobotDescription.from_yaml(str(REPO_ROOT / "robots/aloha_bimanual/robot.yaml"))
    launch_module._refuse_twin_only_geometry_on_real(aloha, "sim")
    with pytest.raises(ROSConfigError, match="fitted to the sim twin only"):
        launch_module._refuse_twin_only_geometry_on_real(aloha, "real")
    openarm = RobotDescription.from_yaml(str(REPO_ROOT / "robots/openarm/robot.yaml"))
    launch_module._refuse_twin_only_geometry_on_real(openarm, "real")  # has its URDF


def test_self_filter_hand_links_are_named_as_the_collision_model(launch_module: Any) -> None:
    """The robot self-filter poses the kernel's model, so its hand links use its naming."""
    from openral_core import RobotDescription

    so100 = RobotDescription.from_yaml(str(REPO_ROOT / "robots/so100_follower/robot.yaml"))
    rename = {"gripper_finger": "Moving_Jaw", "gripper_base": "Fixed_Jaw"}
    plain = launch_module._self_filter_params({}, so100, "/joint_states", 0.02)
    named = launch_module._self_filter_params(
        {}, so100, "/joint_states", 0.02, kernel_link_name=rename
    )
    assert named["robot_mask_hand_links"] == [
        rename.get(h, h) for h in plain["robot_mask_hand_links"]
    ]
    assert "Moving_Jaw" in named["robot_mask_hand_links"]
