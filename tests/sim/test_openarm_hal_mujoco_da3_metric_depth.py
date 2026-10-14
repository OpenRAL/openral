# SPDX-License-Identifier: Apache-2.0
"""DA3METRIC's metres against the simulator's ground truth, on the same camera and frame.

``DeployRuntime.estimated_depth_camera`` converts DA3METRIC's canonical output to metres
with the camera's focal *scaled to DA3's processing raster*
(``openral_perception_ros.metric_depth``). The upstream README gives
``metric_depth = focal * net_output / 300`` but does not say at which resolution
``focal`` is taken, so this measures it: the MuJoCo OpenArm twin's ``head_zed`` publishes
colour registered pixel-for-pixel to its ray-cast depth (``openral_deploy`` tabletop
scene, real ``MujocoArmHAL`` + ``SimSensorBridge``); the colour goes through a real
``DA3METRIC-LARGE`` sidecar and the converted depth is compared with the ray-cast.

Needs a GPU and the DA3 sidecar venv (first boot provisions it and downloads the
weights), so it is opt-in: ``OPENRAL_TEST_ROS_LIVE=1 OPENRAL_TEST_DA3=1``.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_GPU = shutil.which("nvidia-smi") is not None and (
    subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=False).returncode == 0
)
pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (os.getenv("OPENRAL_TEST_ROS_LIVE") and os.getenv("OPENRAL_TEST_DA3")),
        reason="opt-in: OPENRAL_TEST_ROS_LIVE=1 OPENRAL_TEST_DA3=1 (DA3 sidecar venv + GPU)",
    ),
    pytest.mark.skipif(not _GPU, reason="no working NVIDIA GPU (nvidia-smi -L failed)"),
]

_REPO = Path(__file__).resolve().parents[2]
_CAMERA = "head_zed"
_MODEL = "depth-anything/DA3METRIC-LARGE"
_PORT = 5779
_BOOT_S = 1200.0
#: Median DA3/ray-cast depth ratio this conversion must land within. Monocular metric
#: depth on a synthetic render is not exact; a wrong focal resolution is off by the
#: raster scale (about 2-4x here), far outside this band.
_RATIO_BAND = (0.75, 1.33)


def _frames() -> tuple[np.ndarray, np.ndarray, Any]:
    """One registered (rgb, ray-cast depth, CameraInfo) of ``head_zed``."""
    import rclpy
    from openral_core import CameraTopicKind, DeployScene, RobotDescription, camera_topic
    from openral_hal.resolver import build_hal
    from openral_hal.sim_sensor_bridge import SimSensorBridge
    from openral_perception_ros.depth_convert import image_msg_to_depth_array
    from openral_perception_ros.image_convert import image_to_bgr_bytes
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    scene = DeployScene.from_yaml(str(_REPO / "scenes/deploy/openarm_tabletop.yaml"))
    assert scene.composition is not None
    module, _, fn = scene.composition.composer.partition(":")
    import importlib

    xml, meshdir = getattr(importlib.import_module(module), fn)(**scene.composition.params)
    mjcf = meshdir.parent / f"test_da3_metric_{os.getpid()}.xml"
    mjcf.write_text(xml)
    description = RobotDescription.from_yaml(str(_REPO / "robots/openarm/robot.yaml"))
    hal = build_hal(description, mode="sim", transport={"mjcf_path": str(mjcf)})
    hal.connect()
    rclpy.init()
    got: dict[str, Any] = {}
    try:
        node = Node("test_da3_metric_depth")
        bridge = SimSensorBridge(node, hal, description, viewer_enabled=False)
        bridge.setup()
        for key, kind, topic in (
            ("rgb", Image, camera_topic(_CAMERA)),
            ("info", CameraInfo, camera_topic(_CAMERA, CameraTopicKind.DEPTH_CAMERA_INFO)),
            ("depth", Image, camera_topic(_CAMERA, CameraTopicKind.DEPTH_IMAGE)),
        ):
            node.create_subscription(
                kind, topic, lambda m, k=key: got.__setitem__(k, m), qos_profile_sensor_data
            )
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin = threading.Thread(target=executor.spin, daemon=True)
        spin.start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and len(got) < 3:
            time.sleep(0.05)
        executor.shutdown()
        # rclpy's shutdown() does not wait for a running callback: a depth timer still in
        # mujoco.mj_ray at interpreter exit segfaults (pthread_exit through pybind11).
        spin.join(timeout=5.0)
        assert not spin.is_alive(), "executor thread still running a callback"
        bridge.teardown()
        node.destroy_node()
    finally:
        rclpy.shutdown()
        hal.disconnect()
        mjcf.unlink(missing_ok=True)
    assert len(got) == 3, f"head_zed streams missing: {sorted(got)}"
    bgr, w, h = image_to_bgr_bytes(got["rgb"])
    rgb = np.frombuffer(bgr, np.uint8).reshape(h, w, 3)[..., ::-1]
    return np.ascontiguousarray(rgb), image_msg_to_depth_array(got["depth"]), got["info"]


@contextlib.contextmanager
def _sidecar() -> Any:
    import msgpack
    import zmq

    proc = subprocess.Popen(
        [
            sys.executable,
            str(_REPO / "tools/da3_depth_sidecar.py"),
            "--model",
            _MODEL,
            "--port",
            str(_PORT),
        ],
    )
    ctx = zmq.Context()
    try:
        deadline = time.monotonic() + _BOOT_S
        while True:
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 2000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://127.0.0.1:{_PORT}")
            sock.send(msgpack.packb({"op": "ping"}))
            try:
                assert msgpack.unpackb(sock.recv(), raw=False)["ok"]
                sock.setsockopt(zmq.RCVTIMEO, 60000)
                yield sock
                return
            except zmq.Again:
                sock.close(linger=0)
                assert proc.poll() is None, f"DA3 sidecar exited ({proc.returncode})"
                assert time.monotonic() < deadline, "DA3 sidecar did not answer in time"
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=20)
        ctx.term()


def test_da3metric_metres_match_the_ray_cast(tmp_path: Path) -> None:
    import io

    import msgpack
    from openral_perception_ros.metric_depth import (
        da3_metric_depth,
        is_metric_model,
        scale_intrinsics,
    )
    from PIL import Image as PILImage

    rgb, truth, info = _frames()
    h, w = truth.shape
    buf = io.BytesIO()
    PILImage.fromarray(rgb, "RGB").save(buf, format="PNG")
    with _sidecar() as sock:
        sock.send(msgpack.packb({"op": "depth", "image": buf.getvalue(), "process_res": 504}))
        rep = msgpack.unpackb(sock.recv(), raw=False)
    assert rep["ok"] and is_metric_model(rep["model"]), rep.get("error", rep.get("model"))
    dh, dw = int(rep["h"]), int(rep["w"])
    canonical = np.frombuffer(rep["depth"], np.float32).reshape(dh, dw)
    k = (float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5]))
    fx, fy, _, _ = scale_intrinsics(k, (w, h), (dw, dh))
    est = da3_metric_depth(canonical, fx=fx, fy=fy)
    # Ground truth sampled at DA3's raster (nearest pixel centre).
    rows = np.clip(((np.arange(dh) + 0.5) * h / dh).astype(int), 0, h - 1)
    cols = np.clip(((np.arange(dw) + 0.5) * w / dw).astype(int), 0, w - 1)
    gt = truth[np.ix_(rows, cols)]
    valid = np.isfinite(gt) & (gt > 0.05) & (gt < 3.0) & (est > 0.0)
    assert valid.sum() > 0.2 * valid.size, "too few pixels with ground truth"
    ratio = float(np.median(est[valid] / gt[valid]))
    at_camera_res = ratio * (0.5 * (k[0] + k[1])) / (0.5 * (fx + fy))
    lo, hi = _RATIO_BAND
    assert lo < ratio < hi, (
        f"DA3METRIC/ray-cast median ratio {ratio:.3f} with focal at DA3's raster "
        f"({dw}x{dh}); with focal at the camera's {w}x{h} it would be {at_camera_res:.3f}"
    )
