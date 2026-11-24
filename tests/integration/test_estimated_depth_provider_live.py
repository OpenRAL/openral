# SPDX-License-Identifier: Apache-2.0
"""Live ROS: the depth provider's metric mode turns an RGB camera into a world-map cloud.

``DeployRuntime.estimated_depth_camera`` runs the real ``depth_provider_node`` on an RGB
camera with a ``kind: depth_estimator`` rSkill's manifest (DA3METRIC by default, Depth
Pro 2 where named). Here the sidecar is a fake at its ZMQ boundary
(``fakes/fake_depth_sidecar.py``) answering with a plane 1.5 m in front of the camera;
everything else is real: the node process, its parameters, ROS transport, ``CameraInfo``
in, depth + ``CameraInfo`` + ``PointCloud2`` out. Pinned:

* the cloud is the plane at 1.5 m, in the RGB ``CameraInfo``'s optical frame, through the
  camera's own K scaled to DA3's raster (never DA3's estimated K);
* a sidecar serving another model than the skill's (``DA3-SMALL``) yields no cloud at all,
  and a ``relative`` skill is refused at startup;
* a ``metric`` skill (Depth Pro 2) is sent the rectified ``fx`` as ``f_px`` and its metres
  pass through to the same plane;
* a full-resolution ``CameraInfo`` for a downscaled image still gives the plane at 1.5 m,
  while one of another aspect ratio (another view: K would be wrong) yields no cloud;
* a distorted lens (the real Arducam ``plumb_bob`` calibration) is rectified first, and
  the cloud goes through the rectified pinhole K; an unsupported lens model yields none;
* the node is a managed lifecycle node, driven through its ``change_state`` service:
  deactivated it calls no sidecar and publishes no cloud, reactivated the cloud resumes
  (the reasoner's ``lifecycle_transition`` on ``openral_estimated_depth_provider``).

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
_DA3M = "rskill-da3metric_large-any-metric_depth-bf16"
_DP2 = "rskill-depth_pro2_s-any-metric_depth-bf16"
_RASTER = (120, 160)  # DA3's (h, w): a quarter of the camera image
_PLANE_Z_M = 1.5
_FRAME = "test_camera_optical_frame"


@contextlib.contextmanager
def _provider(prefix: str, port: int, log: Path, frame: str, skill: str) -> Iterator[str]:
    """Run the provider lifecycle node (UNCONFIGURED); yield its node name."""
    params = {
        "manifest_path": str(_REPO / "rskills" / skill / "rskill.yaml"),
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
    name = f"depth_provider_{uuid.uuid4().hex[:6]}"
    argv += ["-r", f"__node:={name}"]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_REPO / 'packages/openral_perception_ros'}:{env.get('PYTHONPATH', '')}"
    with log.open("wb") as out:
        proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env)
        try:
            yield name
        finally:
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - shutdown hardening
                proc.kill()
                proc.wait(timeout=5)


def _transition(node: Any, provider: str, *transitions: str) -> bool:
    """Drive ``provider`` through ``transitions`` via its ``change_state`` service."""
    import rclpy
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState

    client = node.create_client(ChangeState, f"/{provider}/change_state")
    try:
        if not client.wait_for_service(timeout_sec=15.0):
            return False
        for name in transitions:
            req = ChangeState.Request()
            req.transition.id = getattr(Transition, f"TRANSITION_{name.upper()}")
            fut = client.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=15.0)
            if not fut.done() or not fut.result().success:
                return False
        return True
    finally:
        node.destroy_client(client)


def _run(
    model: str,
    tmp_path: Path,
    wait_s: float,
    frame: str = "",
    info_scale: tuple[float, float] = (1.0, 1.0),
    lens: tuple[str, list[float]] = ("plumb_bob", [0.0] * 5),
    focal_raster: float | None = None,
    skill: str = _DA3M,
    toggle: bool = False,
) -> tuple[dict[str, Any], Any, str]:
    """Feed the provider RGB + CameraInfo for ``wait_s``; return what it published.

    ``toggle``: once the first cloud is in, deactivate the provider and feed 3 s
    (``clouds_inactive`` / ``requests_inactive`` count what still came out / went to the
    sidecar), then activate it again and feed until a cloud is back (``resumed``).
    """
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2

    from tests.integration.fakes.fake_depth_sidecar import FakeDepthSidecar

    # The focal DA3 "sees" at its raster: the camera's, or a rectified lens's pinhole.
    focal_raster = focal_raster if focal_raster is not None else _F * _RASTER[1] / _W
    sidecar = FakeDepthSidecar(
        model=model,
        plane_z_m=_PLANE_Z_M,
        raster_hw=_RASTER,
        focal_px=focal_raster,
        metric=skill == _DP2,
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
        with _provider(prefix, sidecar.port, log, frame, skill) as provider:
            # A refused manifest fails configure; the tests read that from the log.
            got["active"] = _transition(node, provider, "configure", "activate")

            def feed(seconds: float, done: Any) -> None:
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline and not done():
                    now = node.get_clock().now().to_msg()
                    info.header.stamp = now
                    image.header.stamp = now
                    info_pub.publish(info)
                    img_pub.publish(image)
                    rclpy.spin_once(node, timeout_sec=0.2)

            # Cloud and CameraInfo arrive on separate topics, in either order.
            feed(wait_s, lambda: "cloud" in got and "info" in got)
            if toggle and "cloud" in got:
                assert _transition(node, provider, "deactivate"), log.read_text()
                # The LLM may repeat it: refused, never fatal (the node keeps serving).
                got["second_deactivate"] = _transition(node, provider, "deactivate")
                rclpy.spin_once(node, timeout_sec=0.5)  # drain a cloud already in flight
                got.pop("cloud", None)
                before = sidecar.requests
                feed(3.0, lambda: False)
                got["clouds_inactive"] = int("cloud" in got)
                got["requests_inactive"] = sidecar.requests - before
                assert _transition(node, provider, "activate"), log.read_text()
                got.pop("cloud", None)
                feed(wait_s, lambda: "cloud" in got)
                got["resumed"] = "cloud" in got
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sidecar.close()
    return got, sidecar, log.read_text(errors="replace")


def test_a_metric_sidecar_gives_the_plane_as_a_cloud_in_the_camera_frame(
    tmp_path: Path,
) -> None:
    from sensor_msgs_py.point_cloud2 import read_points_numpy

    got, sidecar, log = _run("depth-anything/DA3METRIC-LARGE", tmp_path, wait_s=30.0)
    assert "cloud" in got, f"no cloud after {sidecar.requests} requests; provider log:\n{log}"
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


def test_deactivating_stops_the_cloud_and_activating_resumes_it(tmp_path: Path) -> None:
    """The reasoner's lifecycle peer: INACTIVE calls no sidecar and publishes no cloud."""
    got, sidecar, log = _run("depth-anything/DA3METRIC-LARGE", tmp_path, wait_s=30.0, toggle=True)
    assert got["active"], log
    assert "resumed" in got, f"no first cloud after {sidecar.requests} requests; log:\n{log}"
    assert got["clouds_inactive"] == 0, "a deactivated provider still published a cloud"
    assert got["requests_inactive"] == 0, "a deactivated provider still called the sidecar"
    assert got["second_deactivate"] is False and "change_state refused" in log
    assert got["resumed"], f"no cloud after reactivating; log:\n{log}"
    assert "deactivated; no depth or cloud" in log


def test_a_sidecar_serving_another_model_feeds_the_map_nothing(tmp_path: Path) -> None:
    """The DA3METRIC skill's provider against a sidecar that serves DA3-SMALL."""
    got, sidecar, log = _run("depth-anything/DA3-SMALL", tmp_path, wait_s=8.0)
    assert sidecar.requests > 0, f"the provider never asked the sidecar; log:\n{log}"
    assert "cloud" not in got
    assert "not the rSkill's 'depth-anything/DA3METRIC-LARGE'" in log


def test_a_metric_skill_is_sent_the_focal_and_its_metres_pass_through(tmp_path: Path) -> None:
    """Depth Pro 2: the rectified fx goes out as ``f_px``; the plane comes back in metres."""
    from sensor_msgs_py.point_cloud2 import read_points_numpy

    got, sidecar, log = _run("apple/DepthPro2-S", tmp_path, wait_s=30.0, skill=_DP2)
    assert "cloud" in got, f"no cloud after {sidecar.requests} requests; provider log:\n{log}"
    assert sidecar.f_px and all(f == pytest.approx(_F, rel=1e-3) for f in sidecar.f_px)
    pts = read_points_numpy(got["cloud"], field_names=("x", "y", "z"))
    assert np.allclose(pts[:, 2], _PLANE_Z_M, atol=1e-4)


def test_a_relative_skill_is_refused_in_metric_mode(tmp_path: Path) -> None:
    """DA3-SMALL is not metric: the provider refuses to configure rather than map its depth."""
    got, sidecar, log = _run(
        "depth-anything/DA3-SMALL", tmp_path, wait_s=8.0, skill="rskill-da3_small-any-depth-bf16"
    )
    assert got["active"] is False
    assert sidecar.requests == 0 and "cloud" not in got
    assert "predicts relative depth" in log


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

    got, sidecar, log = _run(
        "depth-anything/DA3METRIC-LARGE", tmp_path, wait_s=30.0, info_scale=(2.0, 2.0)
    )
    assert "cloud" in got, f"no cloud after {sidecar.requests} requests; provider log:\n{log}"
    pts = read_points_numpy(got["cloud"], field_names=("x", "y", "z"))
    assert np.allclose(pts[:, 2], _PLANE_Z_M, atol=1e-4)
    assert got["info"].k[0] == pytest.approx(_F / 4.0)


def test_a_camera_info_of_another_aspect_feeds_the_map_nothing(tmp_path: Path) -> None:
    """A 640x360 CameraInfo does not describe a 640x480 image: its K would misplace depth."""
    got, sidecar, log = _run(
        "depth-anything/DA3METRIC-LARGE", tmp_path, wait_s=8.0, info_scale=(1.0, 0.75)
    )
    assert sidecar.requests == 0, "a frame whose CameraInfo cannot describe it never reaches DA3"
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
    got, sidecar, log = _run(
        "depth-anything/DA3METRIC-LARGE",
        tmp_path,
        wait_s=30.0,
        lens=("plumb_bob", _ARDUCAM_D),
        focal_raster=rect.k[0] * _RASTER[1] / _W,
    )
    assert "cloud" in got, f"no cloud after {sidecar.requests} requests; provider log:\n{log}"
    pts = read_points_numpy(got["cloud"], field_names=("x", "y", "z"))
    assert np.allclose(pts[:, 2], _PLANE_Z_M, atol=1e-4)
    # The rectified pinhole, not the raw K, and no distortion left on the depth CameraInfo.
    assert got["info"].k[0] == pytest.approx(rect.k[0] / 4.0)
    assert not any(got["info"].d)
    assert "lens plumb_bob" in log


def test_an_unsupported_lens_feeds_the_map_nothing(tmp_path: Path) -> None:
    got, sidecar, log = _run(
        "depth-anything/DA3METRIC-LARGE",
        tmp_path,
        wait_s=8.0,
        lens=("kannala_brandt", [0.1, 0.0, 0.0, 0.0]),
    )
    assert sidecar.requests == 0 and "cloud" not in got
    assert "unsupported distortion_model" in log
