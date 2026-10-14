# SPDX-License-Identifier: Apache-2.0
"""RoboCasa provides the generic scene inputs: RGB-D frames, static fixtures, declarations.

On the live ``scenes/deploy/robocasa_pnp.yaml`` kitchen (PickPlaceCounterToCabinet):

* the sensor bridge publishes the panda_mobile ``front_depth`` camera through the one
  ``openral_sim.depth_frames`` contract (depth image + CameraInfo + cloud, colour when the
  MJCF camera renders the raster's K);
* ``mujoco_geometry.static_collider_boxes`` exports the kitchen's fixtures and never the
  robot;
* ``_RoboCasaSim.scene_ground_truth`` names the task object and its destination, and
  ``tools/derive_scene_inputs.py`` turns them into a derived scene that validates.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from openral_core import DeployScene, RobotDescription
from openral_sim._deps import _has_robocasa_kitchen

from tests.sim.conftest import mujoco_renderer_probe_error

_ROBOCASA_INACTIVE = not _has_robocasa_kitchen()
_RENDERER_ERROR = mujoco_renderer_probe_error() if not _ROBOCASA_INACTIVE else ""

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(_ROBOCASA_INACTIVE, reason="RoboCasa kitchen fork is not active"),
    pytest.mark.skipif(
        bool(_RENDERER_ERROR), reason=_RENDERER_ERROR or "no MuJoCo offscreen renderer"
    ),
]

_REPO = Path(__file__).resolve().parents[2]
SCENE = "scenes/deploy/robocasa_pnp.yaml"
ROBOT = _REPO / "robots" / "panda_mobile" / "robot.yaml"


def _tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "derive_scene_inputs", _REPO / "tools" / "derive_scene_inputs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def kitchen() -> Any:
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(SCENE)
    env.reset(seed=seed)
    yield env
    env.close()


def test_static_fixtures_never_include_the_robot(kitchen: Any) -> None:
    import mujoco
    from openral_hal.depth_cloud import resolve_base_frame_body_name, robot_self_body_ids
    from openral_sim.backends.mujoco_geometry import static_collider_boxes

    model, data = kitchen.mujoco_handles()
    desc = RobotDescription.from_yaml(str(ROBOT))
    base = resolve_base_frame_body_name(model, description=desc)
    assert base is not None
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base)
    robot = robot_self_body_ids(model, [j.sim_joint_name for j in desc.joints])
    boxes = static_collider_boxes(model, data, base_body_id=base_id, exclude_body_ids=robot)
    assert len(boxes["boxes"]) > 50  # a kitchen's counters, cabinets, walls
    robot_bodies = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in robot}
    assert not {b["path"].split("/")[0] for b in boxes["boxes"]} & robot_bodies
    assert all(np.isfinite(b["half_extents_m"]).all() for b in boxes["boxes"])


def test_ground_truth_names_the_object_and_the_cabinet(kitchen: Any) -> None:
    import mujoco
    from openral_hal.depth_cloud import resolve_base_frame_body_name

    model, _ = kitchen.mujoco_handles()
    base = resolve_base_frame_body_name(model, description=RobotDescription.from_yaml(str(ROBOT)))
    truth = kitchen.scene_ground_truth(
        base_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base)
    )
    assert "obj" in truth["objects"]
    (placement,) = truth["placements"]
    assert placement["object"] == "obj"
    assert "cab" in placement["target_name"] or "cabinet" in placement["target_name"].lower()


def test_derived_scene_validates_without_hand_written_inputs(kitchen: Any, tmp_path: Path) -> None:
    out = tmp_path / "robocasa_pnp_derived.yaml"
    assert _tool().main(["--config", SCENE, "--out", str(out)]) == 0
    derived = yaml.safe_load(out.read_text())
    scene = DeployScene.model_validate(derived)
    assert scene.grasp_declaration is not None
    search = scene.grasp_declaration.search_box
    assert search is not None
    # panda_mobile drives, so its declarations are in odom (the sim's world body), which
    # the producer re-expresses in the grid frame every tick: they stay true off the
    # reset pose. The box holds the object where the world puts it.
    odom = RobotDescription.from_yaml(str(ROBOT)).odom_frame
    assert search.frame_id == odom
    workspace = scene.grasp_declaration.workspace_box
    assert workspace is None or workspace.frame_id == odom
    obj = np.asarray(kitchen.scene_ground_truth(base_body_id=0)["objects"]["obj"]["center_m"])
    assert np.all(np.abs(obj - np.asarray(search.pose.xyz)) <= np.asarray(search.half_extents))
    # panda_mobile drives: base-frame fixtures would be true only at the reset pose, so the
    # tool declares none (the export itself is test_static_fixtures_never_include_the_robot).
    assert not (derived.get("runtime") or {}).get("static_geometry_file")


def test_front_depth_publishes_through_the_one_frame_contract(kitchen: Any) -> None:
    rclpy = pytest.importorskip("rclpy", reason="ROS 2 (rclpy) not sourced")
    from openral_core import CameraTopicKind, camera_topic
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_sensor_bridge import SimSensorBridge
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2

    desc = RobotDescription.from_yaml(str(ROBOT))
    hal = SimAttachedHAL(kitchen, desc)
    hal.connect()
    rclpy.init()
    node = rclpy.create_node("test_robocasa_depth_frames")
    bridge = SimSensorBridge(node=node, hal=hal, description=desc, viewer_enabled=False)
    got: dict[str, Any] = {}
    for key, kind, topic in (
        ("depth", Image, camera_topic("front_depth", CameraTopicKind.DEPTH_IMAGE)),
        ("info", CameraInfo, camera_topic("front_depth", CameraTopicKind.DEPTH_CAMERA_INFO)),
        ("cloud", PointCloud2, camera_topic("front_depth", CameraTopicKind.POINTS)),
        ("colour", Image, camera_topic("front_depth")),
    ):
        node.create_subscription(
            kind, topic, lambda m, k=key: got.__setitem__(k, m), qos_profile_sensor_data
        )
    try:
        bridge._setup_depth()
        bridge._publish_depth_clouds()
        deadline = time.monotonic() + 5.0
        while len(got) < 4 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert {"depth", "info", "cloud"} <= set(got), got.keys()
        assert (got["depth"].width, got["depth"].height) == (got["info"].width, got["info"].height)
        if "front_depth" in bridge._depth_colour_disabled:
            assert "colour" not in got  # depth only, warned once
        else:
            assert (got["colour"].width, got["colour"].height) == (
                got["depth"].width,
                got["depth"].height,
            )
            assert "rgb" in {f.name for f in got["cloud"].fields}
    finally:
        bridge.teardown()
        node.destroy_node()
        rclpy.shutdown()
