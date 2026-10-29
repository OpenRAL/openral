# SPDX-License-Identifier: Apache-2.0
"""Any calibrated lens becomes an exact pinhole before DA3 sees it (metric depth).

``openral_perception_ros.lens_rectify`` undoes the lens a ROS ``CameraInfo`` describes
(``plumb_bob``, ``rational_polynomial``, ``equidistant``), so DA3METRIC's metric scale
and the world-collision back-projection both hold to the image edges. Pinned here
against OpenCV's own forward models (``cv2.projectPoints`` / ``cv2.fisheye.projectPoints``):

* a 3-D point lands, in the rectified image, exactly where the rectified pinhole K puts
  it (the remap reads the raw pixel the lens put it at), across the whole image;
* the remap never reads outside the raw image (no invented black border for DA3);
* the camera's pixel aspect is kept (DA3 assumes square pixels);
* a pinhole ``CameraInfo`` passes through untouched; anything it cannot rectify is named.

The Arducam wrist calibration is the real one from ``robots/openarm/robot.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import RobotDescription
from openral_perception_ros.lens_rectify import (
    build_rectifier,
    distortion_problem,
    rectified_stream_k,
)
from openral_perception_ros.metric_depth import (
    DA3_CANONICAL_FOCAL_PX,
    da3_metric_depth,
    points_from_depth,
    scale_intrinsics,
)

cv2 = pytest.importorskip("cv2")

_REPO = Path(__file__).resolve().parents[2]


def _arducam() -> tuple[tuple[float, float, float, float], tuple[int, int], list[float]]:
    """The calibrated left wrist Arducam (plumb_bob, 960x600)."""
    arm = RobotDescription.from_yaml(str(_REPO / "robots/openarm/robot.yaml"))
    k = next(s.intrinsics for s in arm.sensors if s.name == "wrist_left")
    assert k is not None and k.distortion_model == "plumb_bob"
    return (k.fx, k.fy, k.cx, k.cy), (k.width, k.height), list(k.distortion_coeffs)


def _lenses() -> list[tuple[str, tuple[float, float, float, float], tuple[int, int], list[float]]]:
    k, size, d = _arducam()
    return [
        ("plumb_bob", k, size, d),  # the real Arducam
        ("plumb_bob", (500.0, 500.0, 319.5, 239.5), (640, 480), [-0.3, 0.1, 0.0, 0.0, 0.0]),
        (
            "rational_polynomial",
            (520.0, 515.0, 322.0, 241.0),
            (640, 480),
            [0.5, -0.1, 0.001, -0.002, 0.0, 0.6, -0.05, 0.0],
        ),
        ("equidistant", (300.0, 300.0, 319.5, 239.5), (640, 480), [0.05, -0.01, 0.002, 0.0]),
    ]


def _raw_pixels(
    model: str, k: tuple[float, float, float, float], d: list[float], pts: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Where OpenCV's forward lens model puts camera-frame points in the raw image."""
    fx, fy, cx, cy = k
    cam = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    zero = np.zeros(3)
    if model == "equidistant":
        uv, _ = cv2.fisheye.projectPoints(
            pts.reshape(-1, 1, 3), zero, zero, cam, np.asarray(d, dtype=np.float64)
        )
    else:
        uv, _ = cv2.projectPoints(pts, zero, zero, cam, np.asarray(d, dtype=np.float64))
    return np.asarray(uv, dtype=np.float64).reshape(-1, 2)


@pytest.mark.parametrize(("model", "k", "size", "d"), _lenses())
def test_a_point_lands_where_the_rectified_pinhole_puts_it(
    model: str, k: tuple[float, float, float, float], size: tuple[int, int], d: list[float]
) -> None:
    r = build_rectifier(k, size, model, d)
    assert r.maps is not None
    fx, fy, cx, cy = r.k
    w, h = size
    # A grid of rectified pixels over the whole image, edges included.
    us, vs = np.meshgrid(np.linspace(2, w - 3, 9), np.linspace(2, h - 3, 7))
    u, v = np.round(us.ravel()).astype(int), np.round(vs.ravel()).astype(int)
    # The ray through each rectified pixel, per the rectified pinhole K, at any depth.
    z = 1.5
    rays = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, np.full(u.shape, z)], axis=1)
    # The remap samples the raw image at map(u, v): exactly where the lens put that ray.
    sampled = np.stack([r.maps[0][v, u], r.maps[1][v, u]], axis=1)
    error = np.abs(sampled - _raw_pixels(model, k, d, rays)).max()
    assert error < 0.05, error


@pytest.mark.parametrize(("model", "k", "size", "d"), _lenses())
def test_the_remap_never_reads_outside_the_raw_image(
    model: str, k: tuple[float, float, float, float], size: tuple[int, int], d: list[float]
) -> None:
    r = build_rectifier(k, size, model, d)
    assert r.maps is not None
    w, h = size
    mx, my = r.maps
    assert mx.min() >= -0.5 and mx.max() <= w - 0.5
    assert my.min() >= -0.5 and my.max() <= h - 0.5
    white = np.full((h, w, 3), 255, np.uint8)
    assert int(r.apply(white).min()) > 0  # no black border reaches DA3


@pytest.mark.parametrize(("model", "k", "size", "d"), _lenses())
def test_the_pixel_aspect_is_kept(
    model: str, k: tuple[float, float, float, float], size: tuple[int, int], d: list[float]
) -> None:
    r = build_rectifier(k, size, model, d)
    assert r.k[0] / r.k[1] == pytest.approx(k[0] / k[1], rel=1e-9)


def test_a_rectified_point_comes_back_at_its_metric_depth() -> None:
    """The real Arducam: rectify, DA3 at its raster, back-project — the point is in place."""
    k, size, d = _arducam()
    r = build_rectifier(k, size, "plumb_bob", d)
    point = np.array([0.31, -0.17, 0.9])  # off-axis, where the lens bends most
    dw, dh = 504, 315  # DA3's raster for a 960x600 frame at process_res 504
    fx, fy, cx, cy = scale_intrinsics(r.k, r.size, (dw, dh))
    col, row = round(fx * point[0] / point[2] + cx), round(fy * point[1] / point[2] + cy)
    canonical = np.zeros((dh, dw), dtype=np.float32)
    canonical[row, col] = point[2] * DA3_CANONICAL_FOCAL_PX / (0.5 * (fx + fy))
    depth = da3_metric_depth(canonical, fx=fx, fy=fy)
    (got,) = points_from_depth(depth, fx=fx, fy=fy, cx=cx, cy=cy, max_range_m=3.0)
    assert np.allclose(got, point, atol=point[2] / min(fx, fy))


@pytest.mark.parametrize(
    ("model", "d"),
    [("none", []), ("", []), ("plumb_bob", [0.0] * 5), ("equidistant", [0.0] * 4)],
)
def test_a_pinhole_camera_info_passes_through(model: str, d: list[float]) -> None:
    k = (500.0, 500.0, 319.5, 239.5)
    r = build_rectifier(k, (640, 480), model, d)
    img = np.arange(480 * 640 * 3, dtype=np.uint32).astype(np.uint8).reshape(480, 640, 3)
    assert r.maps is None and r.k == k and r.apply(img) is img


@pytest.mark.parametrize(
    ("model", "d", "match"),
    [
        ("kannala_brandt", [0.1, 0.0, 0.0, 0.0], "unsupported distortion_model"),
        ("", [0.1, 0.0, 0.0, 0.0, 0.0], "unsupported distortion_model ''"),
        ("plumb_bob", [0.1, 0.0], "needs 5 coefficients"),
        ("rational_polynomial", [0.1] * 5, "needs 8 coefficients"),
        ("equidistant", [0.1] * 5, "needs 4 coefficients"),
        ("plumb_bob", [0.1, float("nan"), 0.0, 0.0, 0.0], "not all finite"),
    ],
)
def test_what_it_cannot_rectify_is_named(model: str, d: list[float], match: str) -> None:
    problem = distortion_problem(model, d)
    assert problem is not None and match in problem
    with pytest.raises(ValueError, match=match.split(" '", maxsplit=1)[0]):
        build_rectifier((500.0, 500.0, 319.5, 239.5), (640, 480), model, d)


def test_a_rectified_stream_reads_its_projection_matrix() -> None:
    p = [498.2, 0.0, 321.4, 0.0, 0.0, 498.2, 238.9, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert rectified_stream_k(p) == (498.2, 498.2, 321.4, 238.9)
