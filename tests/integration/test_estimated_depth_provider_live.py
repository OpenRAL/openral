# SPDX-License-Identifier: Apache-2.0
"""Live ROS: the depth provider's metric mode turns an RGB camera into a world-map cloud.

``DeployRuntime.estimated_depth_camera`` runs the real ``depth_provider_node`` on an RGB
camera against a DA3METRIC sidecar. Here the sidecar is a fake at its ZMQ boundary
(``fakes/fake_da3_sidecar.py``) answering with the canonical depth of a plane 1.5 m in
front of the camera; everything else is real: the node process, its parameters, ROS
transport, ``CameraInfo`` in, depth + ``CameraInfo`` + ``PointCloud2`` out. Pinned:

* the cloud is the plane at 1.5 m, in the RGB ``CameraInfo``'s optical frame, through the
  camera's own K scaled to DA3's raster (never DA3's estimated K);
* a sidecar serving a non-metric checkpoint (``DA3-SMALL``) yields no cloud at all;
* a full-resolution ``CameraInfo`` for a downscaled image still gives the plane at 1.5 m,
  while one of another aspect ratio (another view: K would be wrong) yields no cloud;
* a distorted lens (the real Arducam ``plumb_bob`` calibration) is rectified first, and
  the cloud goes through the rectified pinhole K; an unsupported lens model yields none.

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
# The fake sidecar and the provider node both speak ZMQ: the opt-in `sidecar-wire` group.
pytest.importorskip("zmq")

_REPO = Path(__file__).resolve().parents[2]
_NODE = _REPO / "packages/openral_perception_ros/openral_perception_ros/depth_provider_node.py"
_W, _H, _F = 640, 480, 500.0
_RASTER = (120, 160)  # DA3's (h, w): a quarter of the camera image
_PLANE_Z_M = 1.5
_FRAME = "test_camera_optical_frame"


@contextlib.contextmanager
def _provider(prefix: str, port: int, log: Path, frame: str) -> Iterator[subprocess.Popen[bytes]]:
    params = {
        "image_topic": f"{prefix}/image",
        "rgb_camera_info_topic": f"{prefix}/camera_info",
        "depth_topic": f"{prefix}/depth",
        "camera_info_topic": f"{prefix}/depth/camera_info",
        "points_topic": f"{prefix}/points",
        "sidecar_port": str(port),
        "max_range_m": "3.0",
    }
    if frame:  # rcl refuses an empty `-p name:=`
        params["depth_frame_id"] = frame
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


def _run(
    model: str,
    tmp_path: Path,
    wait_s: float,
    frame: str = "",
    info_scale: tuple[float, float] = (1.0, 1.0),
    lens: tuple[str, list[float]] = ("plumb_bob", [0.0] * 5),
    focal_raster: float | None = None,
) -> tuple[dict[str, Any], int, str]:
    """Feed the provider RGB + CameraInfo for ``wait_s``; return what it published."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2

    from tests.integration.fakes.fake_da3_sidecar import FakeDa3Sidecar

    # The focal DA3 "sees" at its raster: the camera's, or a rectified lens's pinhole.
    focal_raster = focal_raster if focal_raster is not None else _F * _RASTER[1] / _W
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
        # The camera's K at ``info_scale`` x the image: a driver's full-resolution
        # CameraInfo beside a downscaled republish, or (unequal scales) another view.
        sx, sy = info_scale
        info.width, info.height = round(_W * sx), round(_H * sy)
        fx, fy = _F * sx, _F * sy
        info.k = [fx, 0.0, info.width / 2.0 - 0.5, 0.0, fy, info.height / 2.0 - 0.5, 0.0, 0.0, 1.0]
        info.distortion_model, info.d = lens[0], list(lens[1])
        image = Image()
        image.header.frame_id = "test_camera"  # sim stamps the camera name; K's frame wins
        image.height, image.width, image.encoding = _H, _W, "rgb8"
        image.step = _W * 3
        image.data = bytes(np.full((_H, _W, 3), 120, np.uint8).tobytes())
        with _provider(prefix, sidecar.port, log, frame):
            deadline = time.monotonic() + wait_s
            # Cloud and CameraInfo arrive on separate topics, in either order.
            while time.monotonic() < deadline and not ("cloud" in got and "info" in got):
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


def test_the_launch_resolved_frame_wins_over_the_camera_info_frame(tmp_path: Path) -> None:
    """A top camera riding another sensor's mount stamps its CameraInfo elsewhere."""
    got, _, log = _run(
        "depth-anything/DA3METRIC-LARGE",
        tmp_path,
        wait_s=30.0,
        frame="zed_camera_link_optical_frame",
    )
    assert "cloud" in got, log
    assert got["cloud"].header.frame_id == "zed_camera_link_optical_frame"
    assert got["info"].header.frame_id == "zed_camera_link_optical_frame"


def test_a_full_resolution_camera_info_scales_to_a_downscaled_image(tmp_path: Path) -> None:
    """A driver's own CameraInfo at 2x the republished image: K scales, the plane holds."""
    from sensor_msgs_py.point_cloud2 import read_points_numpy

    got, requests, log = _run(
        "depth-anything/DA3METRIC-LARGE", tmp_path, wait_s=30.0, info_scale=(2.0, 2.0)
    )
    assert "cloud" in got, f"no cloud after {requests} sidecar requests; provider log:\n{log}"
    pts = read_points_numpy(got["cloud"], field_names=("x", "y", "z"))
    assert np.allclose(pts[:, 2], _PLANE_Z_M, atol=1e-4)
    assert got["info"].k[0] == pytest.approx(_F / 4.0)


def test_a_camera_info_of_another_aspect_feeds_the_map_nothing(tmp_path: Path) -> None:
    """A 640x360 CameraInfo does not describe a 640x480 image: its K would misplace depth."""
    got, requests, log = _run(
        "depth-anything/DA3METRIC-LARGE", tmp_path, wait_s=8.0, info_scale=(1.0, 0.75)
    )
    assert requests == 0, "a frame whose CameraInfo cannot describe it never reaches DA3"
    assert "cloud" not in got
    assert "is not a resize of CameraInfo" in log


# The calibrated left wrist Arducam (robots/openarm/robot.yaml), plumb_bob.
_ARDUCAM_D = [-0.072761672, -0.056848830, 0.002163274, 0.001326299, 0.035500496]


def test_a_distorted_lens_goes_through_its_rectified_pinhole(tmp_path: Path) -> None:
    from openral_perception_ros.lens_rectify import build_rectifier
    from sensor_msgs_py.point_cloud2 import read_points_numpy

    rect = build_rectifier(
        (_F, _F, _W / 2.0 - 0.5, _H / 2.0 - 0.5), (_W, _H), "plumb_bob", _ARDUCAM_D
    )
    assert rect.maps is not None and rect.k[0] == rect.k[1] != _F
    got, requests, log = _run(
        "depth-anything/DA3METRIC-LARGE",
        tmp_path,
        wait_s=30.0,
        lens=("plumb_bob", _ARDUCAM_D),
        focal_raster=rect.k[0] * _RASTER[1] / _W,
    )
    assert "cloud" in got, f"no cloud after {requests} sidecar requests; provider log:\n{log}"
    pts = read_points_numpy(got["cloud"], field_names=("x", "y", "z"))
    assert np.allclose(pts[:, 2], _PLANE_Z_M, atol=1e-4)
    # The rectified pinhole, not the raw K, and no distortion left on the depth CameraInfo.
    assert got["info"].k[0] == pytest.approx(rect.k[0] / 4.0)
    assert not any(got["info"].d)
    assert "lens plumb_bob" in log


def test_an_unsupported_lens_feeds_the_map_nothing(tmp_path: Path) -> None:
    got, requests, log = _run(
        "depth-anything/DA3METRIC-LARGE",
        tmp_path,
        wait_s=8.0,
        lens=("kannala_brandt", [0.1, 0.0, 0.0, 0.0]),
    )
    assert requests == 0 and "cloud" not in got
    assert "unsupported distortion_model" in log
