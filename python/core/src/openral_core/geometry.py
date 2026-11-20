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
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    AttachedCollisionObject,
    DeployScene,
    JointType,
    PlaceRegion,
    Pose6D,
    RobotDescription,
)

__all__ = [
    "ViewAxis",
    "base_displaced",
    "base_prismatic",
    "compute_gaze_pose",
    "freeze_released_attachment",
    "homogeneous_from_quat_xyz",
    "is_gravity_aligned",
    "look_at_quat_wxyz",
    "quat_xyzw_to_yaw",
    "rebase_box",
    "rebase_pose",
    "rotation_to_quat_wxyz",
    "scene_for_robot",
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
    ``openral_world_state.object_lift.homogeneous_from_quat_xyz`` now
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
    ``look_at_quat_wxyz``. Defaults to the ROS optical-frame convention
    (``"+z"`` forward) since real camera ``frame_id``s are optical frames.

    Args:
        camera_xyz: Where the camera sits, in ``frame_id``.
        target_xyz: The point to aim at, in ``frame_id``.
        frame_id: tf2 frame both points are expressed in.
        up: World up used to level the camera roll.
        view_axis: Camera forward-axis convention.

    Returns:
        A ``Pose6D`` (``quat_xyzw`` order).

    Example:
        >>> pose = compute_gaze_pose((0.0, 0.0, 1.0), (1.0, 0.0, 1.0))
        >>> pose.frame_id
        'map'
        >>> pose.xyz
        (0.0, 0.0, 1.0)
    """
    w, x, y, z = look_at_quat_wxyz(camera_xyz, target_xyz, up=up, view_axis=view_axis)
    return Pose6D(xyz=camera_xyz, quat_xyzw=(x, y, z, w), frame_id=frame_id)


def freeze_released_attachment(
    held: AttachedCollisionObject,
    *,
    base_link: str,
    t_base_from_link: NDArray[np.float64],
    stamp_ns: int,
) -> AttachedCollisionObject:
    """Re-express a released payload as a record fixed in the base frame.

    The kernel places an attached object at ``link_world[attach_link] ·
    pose_in_link`` for every configuration a candidate predicts, so a record
    left on the hand would ride the retreating hand through the whole chunk while
    the real object stays put. Attached to the collision model's root link (the
    manifest's ``base_frame``, which arm FK never moves; on a mobile base the base
    joints carry it, and the root link with it) it stays where it was released while
    the base stands still — a base move invalidates it, so the release window closes
    then — and ``released_by_link`` names the hand that let it go: the kernel
    then treats it as a frozen payload (2026-11-02, Spark ``v_off_dr``: read as a
    carried payload, the frozen banana latched a world stop on the bowl 170 ms after
    release). It is never checked against the world, stays an obstacle to every robot
    link, and the hand plus ``touch_links`` are exempt against it only until the
    kernel measures them clear or its release timeout passes.

    ``touch_links`` become the held record's attach link plus its touch links —
    exactly the links the held record already exempted (the hand and the jaws
    are in contact with the payload at release).

    Args:
        held: The attachment the gripper held until it let go.
        base_link: The collision model's root link (``RobotDescription.base_frame``); fixed
            relative to the world only while the base is still.
        t_base_from_link: ``(4, 4)`` pose of ``held.attach_link`` in ``base_link``
            when the gripper let go.
        stamp_ns: The release instant: the frozen record's ``stamp_ns``. The kernel keys
            the release window on it and opens it closed when it predates the last base
            motion, so the attach-time stamp would close the window of every payload
            carried across a drive.

    ``support_contact`` is dropped: a held payload's own witness is the grasp-time
    ``map_support_under:*`` one (``region_attachment``), a claim about where the payload
    rested at ATTACH, and a frozen payload is not checked against the world at all.

    Returns:
        The frozen record: same object, geometry, evidence and force attestation, no
        support witness; base-frame pose; ``released_by_link`` the held attach link;
        ``evidence_ref`` suffixed ``|frozen_release``; ``stamp_ns`` the release instant.
    """
    pose = t_base_from_link @ homogeneous_from_quat_xyz(
        held.pose_in_link.xyz, held.pose_in_link.quat_xyzw
    )
    w, x, y, z = rotation_to_quat_wxyz(pose[:3, :3])
    touch = [held.attach_link, *(link for link in held.touch_links if link != held.attach_link)]
    return AttachedCollisionObject.model_validate(
        {
            **held.model_dump(),
            "attach_link": base_link,
            "touch_links": touch,
            "pose_in_link": {
                "xyz": tuple(float(v) for v in pose[:3, 3]),
                "quat_xyzw": (x, y, z, w),
                "frame_id": base_link,
            },
            "evidence_ref": f"{held.evidence_ref or held.object_id}|frozen_release",
            "support_contact": None,
            "released_by_link": held.attach_link,
            "stamp_ns": stamp_ns,
        }
    )


#: The most a fixed frame's ``frame <- box frame`` rotation may tilt off vertical for a
#: declared box to be re-expressed yaw-only (``rebase_box``): a mobile base on a level
#: floor. Beyond it the box's footprint is not a column in the grid frame.
_REBASE_MAX_TILT_RAD = math.radians(0.5)
_GRAVITY_TOL = 1e-6


def is_gravity_aligned(quat_xyzw: tuple[float, float, float, float]) -> bool:
    """Whether a box orientation is yaw-only: its z axis is the frame's (``rebase_box``).

    The one predicate the deploy's scene check and ``rebase_box`` share, so a box the deploy
    accepts is one the consumer can re-express.

    Example:
        >>> is_gravity_aligned((0.0, 0.0, 0.3826834, 0.9238795))
        True
        >>> is_gravity_aligned((0.0998334, 0.0, 0.0, 0.9950042))
        False
    """
    rot = homogeneous_from_quat_xyz((0.0, 0.0, 0.0), quat_xyzw)[:3, :3]
    return abs(float(rot[2, 2]) - 1.0) <= _GRAVITY_TOL


def rebase_box(box: PlaceRegion, t_frame_from_box: NDArray[np.float64], frame: str) -> PlaceRegion:
    """A gravity-aligned declared ``box`` re-expressed in ``frame``, still gravity-aligned.

    A scene may declare a box in a fixed frame (``odom``, ``map``) so it stays put while a
    mobile base moves: a ``GraspDeclaration.search_box`` / ``workspace_box`` (the HAL's
    grasp-target leg) or a ``PlaceZone.region`` (the Layer-2 object tracker). Its consumer
    re-expresses it in the voxel grid's frame every tick through ``t_frame_from_box``
    (tf2 ``frame <- box``). The centre is transformed and the box's yaw composed with the
    transform's; the transform's own tilt (at most ``_REBASE_MAX_TILT_RAD``) is dropped, so
    the result is yaw-only like the box. Lives here (moved down from
    ``openral_hal._grasp_target_leg``) so Layer 0 and Layer 2 share one copy.

    Raises:
        ROSConfigError: The box or the transform is tilted past the bound.

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> box = PlaceRegion(
        ...     frame_id="odom",
        ...     half_extents=(0.1, 0.2, 0.3),
        ...     pose=Pose6D(xyz=(1.0, 0.0, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="odom"),
        ... )
        >>> yaw90 = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 0.5], [0, 0, 0, 1.0]])
        >>> r = rebase_box(box, yaw90, "base_link")
        >>> r.frame_id, [round(v, 3) for v in r.pose.xyz], [round(v, 3) for v in r.pose.quat_xyzw]
        ('base_link', [1.0, 3.0, 0.5], [0.0, 0.0, 0.707, 0.707])
    """
    rot = np.asarray(t_frame_from_box, dtype=np.float64)[:3, :3]
    box_rot = homogeneous_from_quat_xyz((0.0, 0.0, 0.0), box.pose.quat_xyzw)[:3, :3]
    if not is_gravity_aligned(box.pose.quat_xyzw):
        raise ROSConfigError(f"declared box in {box.frame_id!r} must be gravity-aligned.")
    tilt = math.acos(min(1.0, max(-1.0, float(rot[2, 2]))))
    if tilt > _REBASE_MAX_TILT_RAD:
        raise ROSConfigError(
            f"{frame} <- {box.frame_id} is tilted {math.degrees(tilt):.2f} deg; a declared "
            f"box re-expresses only under {math.degrees(_REBASE_MAX_TILT_RAD):.2f} deg."
        )
    xyz = rot @ np.asarray(box.pose.xyz, dtype=np.float64) + np.asarray(t_frame_from_box)[:3, 3]
    composed = rot @ box_rot
    yaw = math.atan2(float(composed[1, 0]), float(composed[0, 0]))
    pose = box.pose.model_copy(
        update={
            "xyz": tuple(float(v) for v in xyz),
            "quat_xyzw": yaw_to_quat_xyzw(yaw),
            "frame_id": frame,
        }
    )
    return box.model_copy(update={"frame_id": frame, "pose": pose})


def rebase_pose(pose: Pose6D, t_frame_from_pose: NDArray[np.float64], frame: str) -> Pose6D:
    """``pose`` re-expressed in ``frame`` through ``t_frame_from_pose`` (tf2 ``frame <- pose``).

    The full rigid motion, tilt included (``rebase_box`` keeps a declared box yaw-only; a
    payload primitive or a released record carries whatever orientation it was measured
    with). The Layer-2 object tracker carries its base-frame poses across a base move with it.

    Example:
        >>> yaw90 = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 0.5], [0, 0, 0, 1.0]])
        >>> p = Pose6D(xyz=(1.0, 0.0, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="old")
        >>> r = rebase_pose(p, yaw90, "new")
        >>> r.frame_id, [round(v, 3) for v in r.xyz], [round(v, 3) for v in r.quat_xyzw]
        ('new', [1.0, 3.0, 0.5], [0.0, 0.0, 0.707, 0.707])
    """
    m = np.asarray(t_frame_from_pose, dtype=np.float64) @ homogeneous_from_quat_xyz(
        pose.xyz, pose.quat_xyzw
    )
    w, x, y, z = rotation_to_quat_wxyz(m[:3, :3])
    return pose.model_copy(
        update={
            "xyz": (float(m[0, 3]), float(m[1, 3]), float(m[2, 3])),
            "quat_xyzw": (x, y, z, w),
            "frame_id": frame,
        }
    )


def base_displaced(
    reference: Sequence[float] | None,
    now: Sequence[float] | None,
    prismatic: Sequence[bool],
    *,
    planar_m: float,
    turn_rad: float,
) -> bool:
    """Whether a mobile base's planar dofs left ``reference`` by more than the bounds.

    The prismatic dofs' planar distance (against ``planar_m``) and each revolute dof's wrapped
    angle (against ``turn_rad``) are tested separately. Unknown (``None``) or non-finite on
    either side counts as moved: fail-closed. Shared by the HAL's release window / grasp-target
    leg and the Layer-2 object tracker's station reset.

    Args:
        reference: The ``base_joints`` positions to compare against, in manifest order.
        now: The same dofs now.
        prismatic: Per dof, whether it is prismatic (else revolute / continuous).
        planar_m: Planar distance bound (m).
        turn_rad: Per-dof rotation bound (rad).

    Returns:
        ``True`` past either bound, or when either side is unknown.

    Example:
        >>> mask = (True, True, False)
        >>> base_displaced(
        ...     (0.0, 0.0, 3.14), (0.005, 0.0, -3.14), mask, planar_m=0.01, turn_rad=0.01
        ... )
        False
        >>> base_displaced((0.0, 0.0, 0.0), (0.02, 0.0, 0.0), mask, planar_m=0.01, turn_rad=0.01)
        True
        >>> base_displaced(
        ...     (0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0), mask, planar_m=1.0, turn_rad=1.0
        ... )
        True
    """
    if reference is None or now is None:
        return True
    deltas = [b - a for a, b in zip(reference, now, strict=True)]
    if not all(math.isfinite(d) for d in deltas):
        return True
    planar = math.hypot(*(d for d, p in zip(deltas, prismatic, strict=True) if p))
    turn = max(
        (abs(math.remainder(d, math.tau)) for d, p in zip(deltas, prismatic, strict=True) if not p),
        default=0.0,
    )
    return planar > planar_m or turn > turn_rad


def base_prismatic(description: RobotDescription) -> tuple[bool, ...]:
    """Per ``base_joints`` entry, whether it is prismatic (``base_displaced``'s mask).

    Example:
        >>> base_prismatic(RobotDescription.from_yaml("robots/openamrobot_v2/robot.yaml"))
        (True, True, False)
    """
    kinds = {j.name: j.joint_type for j in description.joints}
    return tuple(kinds[name] is JointType.PRISMATIC for name in description.base_joints or ())


def scene_for_robot(scene: DeployScene, name: str, *, map_frame: str = "map") -> DeployScene:
    """Robot ``name``'s own one-robot scene, from a scene with several (``DeployScene.robots``).

    Its ``robot_id``, ``base_pose`` and ``robot_unit`` are the entry's. Its map starts at its
    ``base_pose``, so every ``nav_goals`` pose and ``place_zones`` region, declared in the shared
    ``world`` frame, is re-expressed in ``map_frame`` (``RobotDescription.map_frame``): the
    frames the one-robot deploy already reads. Everything else is the scene's, shared.

    Raises:
        ROSConfigError: The scene has no robot ``name``, or a zone does not re-express
            (``rebase_box``: tilted).

    Example:
        >>> from openral_core import NavGoal, SceneRobot, SceneSpec
        >>> at = lambda x, y, yaw: Pose6D(
        ...     xyz=(x, y, 0.0), quat_xyzw=yaw_to_quat_xyzw(yaw), frame_id="world"
        ... )
        >>> scene = DeployScene(
        ...     scene=SceneSpec(id="aisle", backend="isaacsim"),
        ...     robots=[
        ...         SceneRobot(
        ...             name="south", robot_id="openamrobot_v2", base_pose=at(0.0, -1.0, 0.0)
        ...         ),
        ...         SceneRobot(  # at y = 2, facing south (-y)
        ...             name="north",
        ...             robot_id="openamrobot_v2",
        ...             base_pose=at(0.0, 2.0, -math.pi / 2),
        ...         ),
        ...     ],
        ...     nav_goals=[NavGoal(name="s5", pose=at(0.0, 1.0, 0.0))],
        ... )
        >>> north = scene_for_robot(scene, "north")
        >>> north.robot_id, north.robots
        ('openamrobot_v2', [])
        >>> goal = north.nav_goals[0].pose  # 1 m straight ahead of it
        >>> goal.frame_id, [round(v, 3) + 0.0 for v in goal.xyz]
        ('map', [1.0, 0.0, 0.0])
    """
    robot = next((r for r in scene.robots if r.name == name), None)
    if robot is None:
        raise ROSConfigError(f"scene {scene.scene.id!r} has no robot {name!r}.")
    start = robot.base_pose
    world_from_map = homogeneous_from_quat_xyz(start.xyz, start.quat_xyzw)
    map_from_world = np.linalg.inv(world_from_map).astype(np.float64)
    goals = [
        g.model_copy(update={"pose": rebase_pose(g.pose, map_from_world, map_frame)})
        for g in scene.nav_goals
    ]
    zones = [
        z.model_copy(update={"region": rebase_box(z.region, map_from_world, map_frame)})
        for z in scene.place_zones
    ]
    return scene.model_copy(
        update={
            "robots": [],
            "robot_id": robot.robot_id,
            "base_pose": start,
            "robot_unit": robot.robot_unit,
            "nav_goals": goals,
            "place_zones": zones,
        }
    )
