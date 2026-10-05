# SPDX-License-Identifier: Apache-2.0
"""Live ROS: a sim depth camera publishes colour registered to its depth, optically framed.

The vision attachment leg (``DeployRuntime.vision_attachment``) needs four streams of ONE
camera: RGB + its ``CameraInfo`` (the segmenter masks the colour) and metric depth + its
``CameraInfo`` (the HAL bridge back-projects the mask through the depth), all in one
REP-103 optical frame on TF. A real ZED driver publishes exactly that. Before this, the
MuJoCo OpenArm twin published none of it for ``head_zed``: the MJCF had no head camera,
the depth path stamped optical-convention data with the ZED *body* frame
(``zed_camera_link``), and nothing rendered the head camera's colour.

What this pins, on the real OpenArm manifest, the real tabletop composition
(``scenes/deploy/openarm_tabletop.yaml``'s composer — three free cubes on a table), the
real ``MujocoArmHAL`` (which splices ``head_zed`` from its ``sim_placement``) and the real
``SimSensorBridge``:

* colour, depth and both ``CameraInfo`` share one optical frame
  (``zed_camera_link_optical_frame``), one raster size and one K;
* the optical frame hangs off the mount by the REP-103 rotation on ``/tf_static``;
* registration: a scene cube's centre, projected through that K from the camera's
  MuJoCo pose (ground truth is fine in a test), lands on a pixel that is that cube's
  colour in the RGB and whose depth is that cube's surface.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like its neighbours. Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build && source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 pytest tests/integration/test_sim_depth_registered_colour_live.py
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy node — set OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT_YAML = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
_SCENE_YAML = _REPO_ROOT / "scenes" / "deploy" / "openarm_tabletop.yaml"
_CAMERA = "head_zed"
_OPTICAL = "zed_camera_link_optical_frame"
_TIMEOUT_S = 30.0


def _compose_scene(tmp_path: Path) -> str:
    """Write the deploy scene's composed MJCF beside its meshes, like the HAL node does."""
    import importlib

    from openral_core import DeployScene

    scene = DeployScene.from_yaml(str(_SCENE_YAML))
    assert scene.composition is not None
    module, _, fn = scene.composition.composer.partition(":")
    xml, meshdir = getattr(importlib.import_module(module), fn)(**scene.composition.params)
    path = meshdir.parent / f"test_registered_colour_{os.getpid()}.xml"
    path.write_text(xml)
    return str(path)


def test_head_camera_colour_is_registered_to_its_depth(tmp_path: Path) -> None:
    rclpy = pytest.importorskip("rclpy")
    mujoco = pytest.importorskip("mujoco")
    from openral_core import CameraTopicKind, RobotDescription, camera_topic
    from openral_hal.resolver import build_hal
    from openral_hal.sim_sensor_bridge import SimSensorBridge
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_msgs.msg import TFMessage

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))
    mjcf_path = _compose_scene(tmp_path)
    hal = build_hal(description, mode="sim", transport={"mjcf_path": mjcf_path})
    hal.connect()
    rclpy.init()
    got: dict[str, Any] = {}
    try:
        node = Node("test_sim_depth_registered_colour")
        bridge = SimSensorBridge(node, hal, description, viewer_enabled=False)
        bridge.setup()
        best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        for key, msg_type, topic, qos in (
            ("rgb", Image, camera_topic(_CAMERA), best_effort),
            (
                "rgb_info",
                CameraInfo,
                camera_topic(_CAMERA, CameraTopicKind.CAMERA_INFO),
                best_effort,
            ),
            ("depth", Image, camera_topic(_CAMERA, CameraTopicKind.DEPTH_IMAGE), best_effort),
            (
                "depth_info",
                CameraInfo,
                camera_topic(_CAMERA, CameraTopicKind.DEPTH_CAMERA_INFO),
                latched,
            ),
            ("tf_static", TFMessage, "/tf_static", latched),
        ):
            node.create_subscription(msg_type, topic, lambda m, k=key: got.__setitem__(k, m), qos)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin = threading.Thread(target=executor.spin, daemon=True)
        spin.start()
        deadline = time.monotonic() + _TIMEOUT_S
        keys = {"rgb", "rgb_info", "depth", "depth_info", "tf_static"}
        while time.monotonic() < deadline and not keys <= got.keys():
            time.sleep(0.05)
        executor.shutdown()
        spin.join(timeout=5.0)
        assert keys <= got.keys(), f"missing streams: {sorted(keys - got.keys())}"

        rgb, depth, rgb_info, depth_info = (
            got["rgb"],
            got["depth"],
            got["rgb_info"],
            got["depth_info"],
        )
        for msg in (rgb, depth, rgb_info, depth_info):
            assert msg.header.frame_id == _OPTICAL
        assert (rgb.width, rgb.height) == (depth.width, depth.height)
        assert (rgb_info.width, rgb_info.height) == (depth.width, depth.height)
        assert list(rgb_info.k) == list(depth_info.k)
        mount = [
            t
            for t in got["tf_static"].transforms
            if t.child_frame_id == _OPTICAL and t.header.frame_id == "zed_camera_link"
        ]
        assert mount, "no zed_camera_link -> optical on /tf_static"
        r = mount[0].transform.rotation
        assert np.allclose([r.x, r.y, r.z, r.w], [-0.5, 0.5, -0.5, 0.5])

        # Registration against ground truth: project each cube's top-face centre.
        model, data = hal.mujoco_handles()
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, _CAMERA)
        assert cam >= 0, "the camera rig did not splice head_zed"
        cam_pos = np.asarray(data.cam_xpos[cam])
        opt = np.asarray(data.cam_xmat[cam]).reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        fx, fy, cx, cy = depth_info.k[0], depth_info.k[4], depth_info.k[2], depth_info.k[5]
        colour = np.frombuffer(bytes(rgb.data), np.uint8).reshape(rgb.height, rgb.width, 3)
        metric = np.frombuffer(bytes(depth.data), "<f4").reshape(depth.height, depth.width)
        checked = 0
        for body, channel in (("cube_red", 0), ("cube_green", 1), ("cube_blue", 2)):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            gid = int(model.body_geomadr[bid])
            top = np.asarray(data.xpos[bid]) + np.array([0.0, 0.0, float(model.geom_size[gid][2])])
            p = opt.T @ (top - cam_pos)
            if p[2] <= 0.0:
                continue
            u, v = round(fx * p[0] / p[2] + cx), round(fy * p[1] / p[2] + cy)
            if not (0 <= u < rgb.width and 0 <= v < rgb.height):
                continue
            pixel = colour[v, u].astype(int)
            assert int(np.argmax(pixel)) == channel, f"{body}: pixel {pixel} at ({u},{v})"
            assert abs(float(metric[v, u]) - p[2]) < 0.01, f"{body}: depth {metric[v, u]} vs {p[2]}"
            checked += 1
        assert checked >= 2, "fewer than two cubes in view of the head camera"
        bridge.teardown()
        node.destroy_node()
    finally:
        rclpy.shutdown()
        hal.disconnect()
        Path(mjcf_path).unlink(missing_ok=True)
