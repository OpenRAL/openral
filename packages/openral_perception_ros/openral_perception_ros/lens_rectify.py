"""Lens rectification for DA3 metric depth: any calibrated lens to an exact pinhole image.

DA3METRIC's canonical depth is metric only for a pinhole image (``focal * out / 300``),
and the world-collision cloud is back-projected through a pinhole K. A raw image from
a lens with distortion breaks both, increasingly towards the image edges. So the metric
provider rectifies every frame *before* inference, with the camera's own calibration in
ROS ``CameraInfo`` terms (``distortion_model`` + ``d``, REP-104):

* ``plumb_bob`` (``[k1, k2, p1, p2, k3]``) and ``rational_polynomial``
  (``[k1, k2, p1, p2, k3, k4, k5, k6]``): ``cv2.initUndistortRectifyMap``;
* ``equidistant`` (fisheye, ``[k1, k2, k3, k4]``): ``cv2.fisheye.initUndistortRectifyMap``;
* no distortion (``none``, an empty model, or all-zero coefficients): the image is
  already pinhole and passes through untouched.

Only the lens is undone: the rectification rotation ``R`` (stereo) is not applied, so the
rectified image stays in the camera's own optical frame and TF still places it. The
rectified K keeps only valid pixels (``alpha`` / ``balance`` 0) and the camera's
pixel aspect (one zoom for both axes): the network never sees
the black border a remap would otherwise invent. An image that is already rectified
(``image_pipeline``'s ``image_rect``, which shares the raw stream's ``CameraInfo``) is
declared so by the deploy, and is read through ``CameraInfo.p`` instead
(:func:`rectified_stream_k`); nothing is undistorted twice. Any other model, a wrong
coefficient count or a non-finite value refuses the frame (:func:`distortion_problem`).

OpenCV and numpy only; the provider node (``depth_provider_node``) owns the ROS I/O.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DISTORTION_COEFF_COUNTS",
    "PinholeRectifier",
    "build_rectifier",
    "distortion_problem",
    "rectified_stream_k",
]

#: Coefficients each supported ``CameraInfo.distortion_model`` carries (REP-104 / OpenCV).
DISTORTION_COEFF_COUNTS: dict[str, int] = {
    "plumb_bob": 5,
    "rational_polynomial": 8,
    "equidistant": 4,
}


def _is_pinhole(model: str, coeffs: Sequence[float]) -> bool:
    return (model in ("", "none") or model in DISTORTION_COEFF_COUNTS) and not any(coeffs)


def distortion_problem(model: str, coeffs: Sequence[float]) -> str | None:
    """Why ``CameraInfo``'s ``distortion_model`` / ``d`` cannot be rectified, or ``None``.

    Example:
        >>> distortion_problem("plumb_bob", [0.1, -0.2, 0.0, 0.0, 0.05])
        >>> distortion_problem("", [])
        >>> distortion_problem("plumb_bob", [0.1, -0.2])
        'distortion_model plumb_bob needs 5 coefficients, got 2'
        >>> distortion_problem("kannala_brandt", [0.1, 0.0, 0.0, 0.0]).split(" (")[0]
        "unsupported distortion_model 'kannala_brandt'"
    """
    if not all(math.isfinite(c) for c in coeffs):
        return f"distortion coefficients {list(coeffs)} are not all finite"
    if _is_pinhole(model, coeffs):
        return None
    need = DISTORTION_COEFF_COUNTS.get(model)
    if need is None:
        return (
            f"unsupported distortion_model {model!r} "
            f"(supported: {', '.join(DISTORTION_COEFF_COUNTS)})"
        )
    if len(coeffs) != need:
        return f"distortion_model {model} needs {need} coefficients, got {len(coeffs)}"
    return None


def rectified_stream_k(p: Sequence[float]) -> tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` of an already-rectified stream, from ``CameraInfo.p`` (3x4).

    ROS ``CameraInfo`` describes the raw image (``k``, ``d``); a rectified image is
    described by the projection matrix ``p`` (``sensor_msgs/CameraInfo``).

    Example:
        >>> rectified_stream_k([500.0, 0, 320.0, 0, 0, 501.0, 240.0, 0, 0, 0, 1, 0])
        (500.0, 501.0, 320.0, 240.0)
    """
    return float(p[0]), float(p[5]), float(p[2]), float(p[6])


@dataclass(frozen=True)
class PinholeRectifier:
    """One lens's raw -> pinhole remap at one image size, built once per calibration.

    Attributes:
        k: ``(fx, fy, cx, cy)`` of the rectified pinhole image, at ``size``.
        size: ``(width, height)`` of the images it takes and returns.
        maps: OpenCV ``remap`` tables, or ``None`` when the lens is already pinhole.
    """

    k: tuple[float, float, float, float]
    size: tuple[int, int]
    maps: tuple[NDArray[Any], NDArray[Any]] | None

    def apply(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """The rectified ``image`` (``H x W x C``); the input itself when already pinhole."""
        if self.maps is None:
            return image
        import cv2

        out = cv2.remap(
            image, self.maps[0], self.maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        return np.asarray(out, dtype=np.uint8)


def build_rectifier(
    k: tuple[float, float, float, float],
    size: tuple[int, int],
    model: str,
    coeffs: Sequence[float],
) -> PinholeRectifier:
    """The remap from a raw ``size`` image with intrinsics ``k`` and this lens to a pinhole.

    ``k`` must already be at ``size`` (a uniform resize leaves the coefficients alone:
    they act on normalised coordinates). Call :func:`distortion_problem` first; this
    raises ``ValueError`` for anything it refuses, or when OpenCV cannot find a usable
    pinhole for the lens.

    Example:
        >>> r = build_rectifier((500.0, 500.0, 319.5, 239.5), (640, 480), "plumb_bob", [0.0] * 5)
        >>> r.maps is None, r.k
        (True, (500.0, 500.0, 319.5, 239.5))
        >>> r = build_rectifier(
        ...     (500.0, 500.0, 319.5, 239.5), (640, 480), "plumb_bob", [-0.3, 0.1, 0.0, 0.0, 0.0]
        ... )
        >>> r.maps is not None, r.k[0] == r.k[1]  # remapped, square pixels kept
        (True, True)
    """
    problem = distortion_problem(model, coeffs)
    if problem is not None:
        raise ValueError(problem)
    if _is_pinhole(model, coeffs):
        return PinholeRectifier(k=k, size=size, maps=None)
    import cv2

    fx, fy, cx, cy = k
    cam = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    d = np.asarray(coeffs, dtype=np.float64)
    if model == "equidistant":
        new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            cam, d, size, np.eye(3), balance=0.0
        )
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(cam, d, np.eye(3), new, size, cv2.CV_32FC1)
    else:
        new, _roi = cv2.getOptimalNewCameraMatrix(cam, d, size, 0.0, size)
        # OpenCV fits each axis on its own, which would stretch the pixels; one zoom
        # (the tighter axis's) keeps the camera's pixel aspect for the network.
        zoom = max(new[0, 0] / fx, new[1, 1] / fy)
        new[0, 0], new[1, 1] = fx * zoom, fy * zoom
        m1, m2 = cv2.initUndistortRectifyMap(cam, d, np.eye(3), new, size, cv2.CV_32FC1)
    rect = (float(new[0, 0]), float(new[1, 1]), float(new[0, 2]), float(new[1, 2]))
    if not all(math.isfinite(v) for v in rect) or rect[0] <= 0.0 or rect[1] <= 0.0:
        raise ValueError(f"no usable pinhole for {model} {list(coeffs)}: K {rect}")
    return PinholeRectifier(k=rect, size=size, maps=(m1, m2))
