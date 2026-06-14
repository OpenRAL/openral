"""Integration test: the query_scene ROS service path, end-to-end (ADR-0047).

Brings up the REAL ``scene_vlm_node`` as a subprocess (real rclpy node, real
``openral_msgs/srv/QueryScene`` IDL, real ``QwenSceneVlm`` backend talking to a
live NF4 Qwen3.5-4B sidecar), publishes a real camera ``sensor_msgs/Image``,
calls ``/openral/perception/query_scene``, and asserts a grounded text answer.
No mocks (CLAUDE.md §1.11) — this exercises the integration-tier glue the unit
tests can't: frame caching + the service handler calling the VLM backend.

Gated (CLAUDE.md §12) on a provisioned sidecar venv (``OPENRAL_QWEN_VLM_SIDECAR_VENV``)
+ a local GPU, and on the ROS stack being importable (``openral_msgs`` built,
``rclpy`` sourced). Run with ROS 2 + the worktree ``install/`` sourced and the
openral src dirs on ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import time

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_FIXTURES = _REPO / "tests" / "unit" / "fixtures"
_MANIFEST = _REPO / "rskills" / "qwen35-4b-nf4" / "rskill.yaml"
_NODE = (
    _REPO / "packages" / "openral_perception_ros" / "openral_perception_ros" / "scene_vlm_node.py"
)


def _gpu_present() -> bool:
    return shutil.which("nvidia-smi") is not None


def _ros_available() -> bool:
    try:
        import rclpy  # noqa: F401
        from openral_msgs.srv import QueryScene  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENRAL_QWEN_VLM_SIDECAR_VENV")
    or not _gpu_present()
    or not _ros_available(),
    reason="needs a provisioned Qwen sidecar venv + GPU + a built/sourced ROS stack "
    "(OPENRAL_QWEN_VLM_SIDECAR_VENV; openral_msgs built; rclpy sourced).",
)


def test_query_scene_service_e2e() -> None:
    import numpy as np
    import rclpy
    from openral_msgs.srv import QueryScene
    from PIL import Image as PILImage
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Image

    img = PILImage.open(_FIXTURES / "coco_sample.jpg").convert("RGB")
    width, height = img.size
    rgb = np.asarray(img, dtype=np.uint8)

    # Launch the REAL node as a subprocess (the node class is defined inside
    # main() behind the ROS imports, so it can't be imported directly).
    proc = subprocess.Popen(
        [
            sys.executable,
            str(_NODE),
            "--ros-args",
            "-p",
            f"manifest_path:={_MANIFEST}",
            "-p",
            "image_topic:=/test/cam",
            "-p",
            "primary_camera:=cam0",
            "-p",
            "sidecar_port:=5759",
        ],
    )
    rclpy.init()
    node = rclpy.create_node("scene_vlm_itest")
    try:
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        pub = node.create_publisher(Image, "/test/cam", qos)
        msg = Image()
        msg.height = height
        msg.width = width
        msg.encoding = "rgb8"
        msg.step = width * 3
        msg.data = rgb.tobytes()
        client = node.create_client(QueryScene, "/openral/perception/query_scene")

        # Wait for the node to advertise the service (it builds the backend at
        # startup), publishing frames so a frame is cached by the time we query.
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline and not client.service_is_ready():
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.2)
        assert client.service_is_ready(), "query_scene service never came up"

        # Ensure at least one frame is cached at the node before the query.
        for _ in range(15):
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.1)

        req = QueryScene.Request()
        req.question = "How many cats are in this image? Answer with a number."
        req.camera = ""
        future = client.call_async(req)
        end = time.monotonic() + 180.0
        while time.monotonic() < end and not future.done():
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.2)
        assert future.done(), "query_scene service call timed out"

        resp = future.result()
        print(f"query_scene -> ok={resp.ok} camera={resp.camera!r} answer={resp.answer!r}")
        assert resp.ok, "service reported failure (no frame or sidecar error)"
        assert resp.answer.strip(), "empty answer"
        assert "2" in resp.answer or "two" in resp.answer.lower(), (
            f"expected the two-cats image to be counted, got: {resp.answer!r}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
