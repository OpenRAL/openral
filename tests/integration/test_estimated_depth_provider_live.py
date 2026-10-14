# SPDX-License-Identifier: Apache-2.0
"""Live ROS: the depth provider's metric mode turns an RGB camera into a world-map cloud.

``DeployRuntime.estimated_depth_camera`` runs the real ``depth_provider_node`` on an RGB
camera against a DA3METRIC sidecar. Here the sidecar is a fake at its ZMQ boundary
(``fakes/fake_da3_sidecar.py``) answering with the canonical depth of a plane 1.5 m in
front of the camera; everything else is real: the node process, its parameters, ROS
transport, ``CameraInfo`` in, depth + ``CameraInfo`` + ``PointCloud2`` out. Pinned:

* the cloud is the plane at 1.5 m, in the RGB ``CameraInfo``'s optical frame, through the
  camera's own K scaled to DA3's raster (never DA3's estimated K);
* a sidecar serving a non-metric checkpoint (``DA3-SMALL``) yields no cloud at all.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like its neighbours.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS, reason="live rclpy node — set OPENRAL_TEST_ROS_LIVE=1 with ROS sourced."
)

_REPO = Path(__file__).resolve().parents[2]
_NODE = _REPO / "packages/openral_perception_ros/openral_perception_ros/depth_provider_node.py"
_W, _H, _F = 640, 480, 500.0
_RASTER = (120, 160)  # DA3's (h, w): a quarter of the camera image
_PLANE_Z_M = 1.5
_FRAME = "test_camera_optical_frame"


@contextlib.contextmanager
def _provider(prefix: str, port: int, log: Path) -> Iterator[subprocess.Popen[bytes]]:
    params = {
        "image_topic": f"{prefix}/image",
        "rgb_camera_info_topic": f"{prefix}/camera_info",
        "depth_topic": f"{prefix}/depth",
        "camera_info_topic": f"{prefix}/depth/camera_info",
        "points_topic": f"{prefix}/points",
        "sidecar_port": str(port),
        "max_range_m": "3.0",
    }
    argv = [sys.executable, str(_NODE), "--ros-args"]
    for key, value in params.items():
        argv += ["-p", f"{key}:={value}"]
    argv += ["-r", f"__node:=depth_provider_{uuid.uuid4().hex[:6]}"]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_REPO / 'packages/openral_perception_ros'}:{env.get('PYTHONPATH', '')}"
    with log.open("wb") as out:
        proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env)
        try:
            yield proc
        finally:
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - shutdown hardening
                proc.kill()
                proc.wait(timeout=5)


def _run(model: str, tmp_path: Path, wait_s: float) -> tuple[dict[str, Any], int, str]:
    """Feed the provider RGB + CameraInfo for ``wait_s``; return what it published."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2

    sys.path.insert(0, str(Path(__file__).parent))
    from fakes.fake_da3_sidecar import FakeDa3Sidecar

    focal_raster = _F * _RASTER[1] / _W
    sidecar = FakeDa3Sidecar(
        model=model, plane_z_m=_PLANE_Z_M, raster_hw=_RASTER, focal_px=focal_raster
    )
    prefix = f"/test_{uuid.uuid4().hex[:8]}/cam"
    log = tmp_path / "provider.log"
    got: dict[str, Any] = {}
    rclpy.init()
    node = rclpy.create_node(f"test_estimated_depth_{uuid.uuid4().hex[:6]}")
    try:
        img_pub = node.create_publisher(Image, f"{prefix}/image", qos_profile_sensor_data)
        info_pub = node.create_publisher(CameraInfo, f"{prefix}/camera_info", 1)
        node.create_subscription(
            PointCloud2,
            f"{prefix}/points",
            lambda m: got.__setitem__("cloud", m),
            qos_profile_sensor_data,
        )
        node.create_subscription(
            CameraInfo, f"{prefix}/depth/camera_info", lambda m: got.__setitem__("info", m), 1
        )
        info = CameraInfo()
        info.header.frame_id = _FRAME
        info.width, info.height = _W, _H
        info.k = [_F, 0.0, (_W - 1) / 2.0, 0.0, _F, (_H - 1) / 2.0, 0.0, 0.0, 1.0]
        image = Image()
        image.header.frame_id = "test_camera"  # sim stamps the camera name; K's frame wins
        image.height, image.width, image.encoding = _H, _W, "rgb8"
        image.step = _W * 3
        image.data = bytes(np.full((_H, _W, 3), 120, np.uint8).tobytes())
        with _provider(prefix, sidecar.port, log):
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline and "cloud" not in got:
                now = node.get_clock().now().to_msg()
                info.header.stamp = now
                image.header.stamp = now
                info_pub.publish(info)
                img_pub.publish(image)
                rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sidecar.close()
    return got, sidecar.requests, log.read_text(errors="replace")


def test_a_metric_sidecar_gives_the_plane_as_a_cloud_in_the_camera_frame(
    tmp_path: Path,
) -> None:
    from sensor_msgs_py.point_cloud2 import read_points_numpy

    got, requests, log = _run("depth-anything/DA3METRIC-LARGE", tmp_path, wait_s=30.0)
    assert "cloud" in got, f"no cloud after {requests} sidecar requests; provider log:\n{log}"
    cloud = got["cloud"]
    assert cloud.header.frame_id == _FRAME
    pts = read_points_numpy(cloud, field_names=("x", "y", "z"))
    assert len(pts) == _RASTER[0] * _RASTER[1]
    assert np.allclose(pts[:, 2], _PLANE_Z_M, atol=1e-4)
    # The camera's K at DA3's raster (a quarter of 640x480), not DA3's own K (fx=1).
    info = got["info"]
    assert info.header.frame_id == _FRAME
    assert (info.width, info.height) == (_RASTER[1], _RASTER[0])
    assert info.k[0] == pytest.approx(_F / 4.0)
    # The plane spans the camera's field of view at 1.5 m: x reaches about +-w/2 * z / f.
    assert float(pts[:, 0].max()) == pytest.approx(_W / 2.0 * _PLANE_Z_M / _F, rel=0.02)


def test_a_non_metric_sidecar_feeds_the_map_nothing(tmp_path: Path) -> None:
    got, requests, log = _run("depth-anything/DA3-SMALL", tmp_path, wait_s=8.0)
    assert requests > 0, f"the provider never asked the sidecar; log:\n{log}"
    assert "cloud" not in got
    assert "not DA3METRIC" in log
