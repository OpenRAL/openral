"""Shared gaze geometry — look-at rotations and camera gaze poses.

Promotes the look-at math that was previously triplicated across the MuJoCo
scene composers (``openral_sim.backends.{so101_box,openarm_robosuite,
tabletop_push}._assets``) into one public helper, and adds the full-pose
variant the ``rskill-moveit-multi-look_at`` rSkill consumes.

Lives in ``openral_core`` so every layer can compute camera
orientations from one source — in particular the layer-0 HAL camera rig
(``openral_hal._camera_rig``) places manifest cameras without a backward
dependency on world-state (layer 2). ``openral_world_state.geometry``
re-exports this module verbatim for back-compat. Kept out of
``openral_core.__init__`` so importing the core schemas stays numpy-free on the
fast CLI path (CLAUDE.md §1.5).

Camera conventions differ per consumer, so the **view axis is explicit**:

- ``"-z"`` — MuJoCo cameras: look along local -Z with +Y up in the image.
- ``"+z"`` — ROS optical frames (REP-103 ``*_optical``): +Z forward, +X right,
  +Y down.
- ``"+x"`` — body-frame forward (the approach-viewpoint convention:
  +X forward, +Z up).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from openral_core.schemas import Pose6D

__all__ = [
    "ViewAxis",
    "compute_gaze_pose",
    "homogeneous_from_quat_xyz",
    "look_at_quat_wxyz",
    "quat_xyzw_to_yaw",
    "rotation_to_quat_wxyz",
    "yaw_to_quat_wxyz",
    "yaw_to_quat_xyzw",
]

ViewAxis = Literal["-z", "+z", "+x"]

_ZERO_NORM = 1e-9
_PARALLEL = 0.999
_QUAT_NORM_EPS = 1e-12  # below this squared-norm a quaternion is degenerate


def homogeneous_from_quat_xyz(
    translation: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    """Build a ``(4, 4)`` homogeneous transform from a translation + xyzw quaternion.

    The canonical "TF2 ``TransformStamped`` → 4x4 matrix" step. It lives here,
    beside the rest of the shared rotation math, because every layer that reads
    TF needs it: the layer-0 HAL's vision attachment bridge composes
    ``attach_link <- camera_optical`` this way, and layer 2's object lift has
    done the same since it was written.
    :func:`openral_world_state.object_lift.homogeneous_from_quat_xyz` now
    delegates here, so there is one implementation rather than one per layer
    (CLAUDE.md §1.13).

    Args:
        translation: ``(x, y, z)`` translation in metres.
        quat_xyzw: Rotation quaternion as ``(x, y, z, w)``; need not be unit
            length (it is normalised internally).

    Returns:
        A ``(4, 4)`` float64 homogeneous transform matrix.

    Raises:
        ROSConfigError: If the quaternion norm is effectively zero — a
            degenerate rotation is a malformed input, not something to guess a
            fallback for.

    Example:
        >>> homogeneous_from_quat_xyz((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)).tolist()
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]
    """
    from openral_core.exceptions import ROSConfigError  # noqa: PLC0415 — avoids an import cycle

    x, y, z, w = quat_xyzw
    n = x * x + y * y + z * z + w * w
    if n < _QUAT_NORM_EPS:
        raise ROSConfigError(f"degenerate quaternion {quat_xyzw!r}")
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1.0 - (yy + zz)
    m[0, 1] = xy - wz
    m[0, 2] = xz + wy
    m[1, 0] = xy + wz
    m[1, 1] = 1.0 - (xx + zz)
    m[1, 2] = yz - wx
    m[2, 0] = xz - wy
    m[2, 1] = yz + wx
    m[2, 2] = 1.0 - (xx + yy)
    m[0, 3], m[1, 3], m[2, 3] = translation
    return m


def yaw_to_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    """Unit quaternion ``(x, y, z, w)`` (ROS order) for ``Rz(yaw)``.

    Example:
        >>> yaw_to_quat_xyzw(0.0)
        (0.0, 0.0, 0.0, 1.0)
    """
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    """Unit quaternion ``(w, x, y, z)`` (MuJoCo order) for ``Rz(yaw)``.

    Example:
        >>> yaw_to_quat_wxyz(0.0)
        (1.0, 0.0, 0.0, 0.0)
    """
    half = yaw * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def quat_xyzw_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Planar yaw (rad, CCW about +Z, in ``[-pi, pi]``) from an ``(x, y, z, w)`` quaternion.

    Standard ZYX extraction reduced to the yaw term; roll/pitch are ignored
    (planar-base consumers: odometry, occupancy grids, SLAM poses).

    Example:
        >>> quat_xyzw_to_yaw(0.0, 0.0, 0.0, 1.0)
        0.0
    """
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _basis_to_quat_wxyz(rot: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Convert a right-handed 3x3 rotation matrix to a ``(w, x, y, z)`` quaternion.

    Shepperd's trace method with the largest-diagonal fallback — the same
    algorithm the sim composers used, so promoting the helper is
    behaviour-preserving for them.
    """
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = float(rot[2, 1] - rot[1, 2]) * s
        y = float(rot[0, 2] - rot[2, 0]) * s
        z = float(rot[1, 0] - rot[0, 1]) * s
        return (w, x, y, z)
    i = int(np.argmax(np.diag(rot)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(1.0 + float(rot[i, i] - rot[j, j] - rot[k, k]))
    qi = 0.25 * s
    qj = float(rot[j, i] + rot[i, j]) / s
    qk = float(rot[k, i] + rot[i, k]) / s
    w = float(rot[k, j] - rot[j, k]) / s
    q = [w, 0.0, 0.0, 0.0]
    q[1 + i] = qi
    q[1 + j] = qj
    q[1 + k] = qk
    return (q[0], q[1], q[2], q[3])


def rotation_to_quat_wxyz(rot: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Public matrix→quaternion conversion (``w, x, y, z``), Shepperd's method.

    Used by the ``rskill-moveit-multi-look_at`` skill to re-express a camera gaze pose for
    the camera's mount link after a homogeneous-matrix composition.

    Example:
        >>> import numpy as np
        >>> rotation_to_quat_wxyz(np.eye(3))
        (1.0, 0.0, 0.0, 0.0)
    """
    return _basis_to_quat_wxyz(rot)


def look_at_quat_wxyz(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    view_axis: ViewAxis = "-z",
) -> tuple[float, float, float, float]:
    """Quaternion (``w, x, y, z``) orienting a camera at ``eye`` to face ``target``.

    The rotated frame's ``view_axis`` points from ``eye`` toward ``target``.
    Degenerate inputs fall back instead of raising: ``target == eye`` returns
    the MuJoCo straight-down quat for ``"-z"`` (preserving the sim composers'
    documented behaviour) and identity otherwise; a gaze nearly parallel to
    ``up`` swaps in a +Y up vector.

    Args:
        eye: Camera position.
        target: Point to look at, same frame as ``eye``.
        up: World up used to level the camera roll.
        view_axis: Which camera axis must point at the target (see module
            docstring for the conventions).

    Returns:
        Unit quaternion in ``(w, x, y, z)`` order (MuJoCo convention).

    Example:
        >>> look_at_quat_wxyz((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), view_axis="+x")
        (1.0, 0.0, 0.0, 0.0)
    """
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    forward = target_v - eye_v
    norm = float(np.linalg.norm(forward))
    if norm < _ZERO_NORM:
        # Sim composers' documented fallback: 180° flip about X looks straight
        # down for a -Z camera. Other conventions get identity.
        return (0.0, 1.0, 0.0, 0.0) if view_axis == "-z" else (1.0, 0.0, 0.0, 0.0)
    forward /= norm
    up_v = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(up_v, forward))) > _PARALLEL:
        # Looking nearly straight up/down the up vector: pick an alternate up.
        up_v = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)

    if view_axis == "-z":
        z_axis = -forward
        x_axis = np.cross(up_v, z_axis)
        x_axis /= float(np.linalg.norm(x_axis))
        y_axis = np.cross(z_axis, x_axis)
    elif view_axis == "+z":
        z_axis = forward
        x_axis = np.cross(forward, up_v)
        x_axis /= float(np.linalg.norm(x_axis))
        y_axis = np.cross(z_axis, x_axis)
    else:  # "+x"
        x_axis = forward
        y_axis = np.cross(up_v, forward)
        y_axis /= float(np.linalg.norm(y_axis))
        z_axis = np.cross(x_axis, y_axis)

    rot = np.column_stack([x_axis, y_axis, z_axis])
    return _basis_to_quat_wxyz(rot)


def compute_gaze_pose(
    camera_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
    *,
    frame_id: str = "map",
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    view_axis: ViewAxis = "+z",
) -> Pose6D:
    """Full 6-DOF camera pose at ``camera_xyz`` whose view axis hits ``target_xyz``.

    The pose the ``rskill-moveit-multi-look_at`` rSkill plans the
    camera frame to: position fixed at ``camera_xyz``, orientation from
    :func:`look_at_quat_wxyz`. Defaults to the ROS optical-frame convention
    (``"+z"`` forward) since real camera ``frame_id``s are optical frames.

    Args:
        camera_xyz: Where the camera sits, in ``frame_id``.
        target_xyz: The point to aim at, in ``frame_id``.
        frame_id: tf2 frame both points are expressed in.
        up: World up used to level the camera roll.
        view_axis: Camera forward-axis convention.

    Returns:
        A :class:`~openral_core.Pose6D` (``quat_xyzw`` order).

    Example:
        >>> pose = compute_gaze_pose((0.0, 0.0, 1.0), (1.0, 0.0, 1.0))
        >>> pose.frame_id
        'map'
        >>> pose.xyz
        (0.0, 0.0, 1.0)
    """
    w, x, y, z = look_at_quat_wxyz(camera_xyz, target_xyz, up=up, view_axis=view_axis)
    return Pose6D(xyz=camera_xyz, quat_xyzw=(x, y, z, w), frame_id=frame_id)
