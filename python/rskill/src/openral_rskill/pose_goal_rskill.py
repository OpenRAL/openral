"""``PoseGoalRskill`` — move the end-effector to a Cartesian pose via MoveGroup (ADR-0054).

A :class:`~openral_rskill.ros_action_rskill.ROSActionRskill` whose goal is a
small ``pose`` block (target position + orientation for a constrained link),
lowered at dispatch time into MoveGroup ``position_constraints`` +
``orientation_constraints``. The generic Cartesian sibling of
:class:`~openral_rskill.look_at_rskill.LookAtRskill`: look-at computes its pose
(camera gaze) and leaves optical roll free, whereas a generic pose is supplied
directly and constrains all three axes.

The pose→constraints lowering (:func:`build_pose_constraints`) is the **shared**
implementation `LookAtRskill` also uses, so there is one place the link-offset
math lives (ADR-0054 D2/D3). Orientation is a quaternion array whose component
order the manifest declares via ``quaternion_order`` (default ``"xyzw"``, Q2).

The constrained link's tool offset (``link_t_target``) is identity in v1; the
ADR-0054 phase-6 follow-up sources it from a ``RobotDescription`` tool frame
(mirroring how look-at sources the camera mount from ``SensorSpec``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from openral_core import Pose6D
from openral_core.exceptions import ROSConfigError
from openral_world_state.geometry import rotation_to_quat_wxyz
from openral_world_state.object_lift import homogeneous_from_quat_xyz

from openral_rskill.ros_action_rskill import ROSActionRskill

__all__ = ["PoseGoalRskill", "build_pose_constraints", "pose_from_block"]

log = structlog.get_logger(__name__)

NDArrayOrNone = Any  # reason: numpy NDArray | None alias keeps signatures readable

_SOLID_PRIMITIVE_SPHERE = 2  # shape_msgs/SolidPrimitive.SPHERE
_DEFAULT_POSITION_TOLERANCE_M = 0.01
_DEFAULT_ORIENTATION_TOLERANCE_RAD = 0.05
_XYZ_LEN = 3
_QUAT_LEN = 4
_QUATERNION_ORDERS = ("xyzw", "wxyz")
_TF_LOOKUP_TIMEOUT_S = 5.0


def build_pose_constraints(
    *,
    pose: Pose6D,
    link_name: str,
    link_t_target: NDArrayOrNone = None,
    position_tolerance_m: float = _DEFAULT_POSITION_TOLERANCE_M,
    orientation_axis_tolerances_rad: tuple[float, float, float] = (
        _DEFAULT_ORIENTATION_TOLERANCE_RAD,
        _DEFAULT_ORIENTATION_TOLERANCE_RAD,
        _DEFAULT_ORIENTATION_TOLERANCE_RAD,
    ),
) -> dict[str, Any]:
    """Lower a target pose into one MoveGroup ``goal_constraints`` entry.

    When ``link_t_target`` is given the goal is re-expressed for the constrained
    link (``goal_link = goal_target @ inv(link_t_target)``); otherwise ``pose``
    *is* the constrained link's goal. ``orientation_axis_tolerances_rad`` is the
    per-axis (x, y, z) absolute tolerance — a generic Cartesian pose passes the
    same value on all three; ``build_look_at_constraints`` passes ``π`` on the
    optical (z) axis to leave roll free.

    Args:
        pose: Target pose for the constrained link (or for the tool frame when
            ``link_t_target`` is given). ``pose.frame_id`` is the constraint
            ``header.frame_id``.
        link_name: The MoveGroup link the constraints apply to.
        link_t_target: Optional homogeneous link→target offset.
        position_tolerance_m: Sphere radius for the position constraint.
        orientation_axis_tolerances_rad: ``(x, y, z)`` absolute tolerances.

    Returns:
        One ``goal_constraints`` entry (``position_constraints`` +
        ``orientation_constraints``).
    """
    if link_t_target is not None:
        x, y, z, w = pose.quat_xyzw
        goal_target = homogeneous_from_quat_xyz(pose.xyz, (x, y, z, w))
        goal_link = goal_target @ np.linalg.inv(link_t_target)
        gw, gx, gy, gz = rotation_to_quat_wxyz(goal_link[:3, :3])
        position = (float(goal_link[0, 3]), float(goal_link[1, 3]), float(goal_link[2, 3]))
        quat_xyzw = (gx, gy, gz, gw)
    else:
        position = pose.xyz
        quat_xyzw = pose.quat_xyzw

    tol_x, tol_y, tol_z = orientation_axis_tolerances_rad
    frame_id = pose.frame_id
    return {
        "position_constraints": [
            {
                "header": {"frame_id": frame_id},
                "link_name": link_name,
                "constraint_region": {
                    "primitives": [
                        {
                            "type": _SOLID_PRIMITIVE_SPHERE,
                            "dimensions": [position_tolerance_m],
                        }
                    ],
                    "primitive_poses": [
                        {
                            "position": {
                                "x": position[0],
                                "y": position[1],
                                "z": position[2],
                            },
                            "orientation": {"w": 1.0},
                        }
                    ],
                },
                "weight": 1.0,
            }
        ],
        "orientation_constraints": [
            {
                "header": {"frame_id": frame_id},
                "link_name": link_name,
                "orientation": {
                    "x": quat_xyzw[0],
                    "y": quat_xyzw[1],
                    "z": quat_xyzw[2],
                    "w": quat_xyzw[3],
                },
                "absolute_x_axis_tolerance": tol_x,
                "absolute_y_axis_tolerance": tol_y,
                "absolute_z_axis_tolerance": tol_z,
                "weight": 1.0,
            }
        ],
    }


def pose_from_block(block: dict[str, Any]) -> tuple[Pose6D, str, str | None, float, float]:
    """Parse a ``pose`` goal block.

    Returns ``(pose, link_name, tool_frame, pos_tol, orient_tol)``. Orientation
    is a 4-float quaternion whose component order is given by
    ``block["quaternion_order"]`` (``"xyzw"`` default, or ``"wxyz"`` — the
    manifest fixes the convention, ADR-0054 Q2). Position is ``[x, y, z]``.
    ``tool_frame`` (optional) is the TCP/tool frame the target pose is expressed
    *for*; when set, the executor TF-looks-up ``link_name ← tool_frame`` to offset
    the constraint (ADR-0054 Q3) — otherwise the pose is for ``link_name`` itself.

    Raises:
        ROSConfigError: On a missing/ill-typed field or an unknown
            ``quaternion_order``.
    """
    try:
        frame_id = str(block["frame_id"])
        link_name = str(block["link_name"])
        position = block["position"]
        orientation = block["orientation"]
    except (KeyError, TypeError) as exc:
        raise ROSConfigError(
            f"pose block needs frame_id, link_name, position, orientation: {exc}"
        ) from exc
    if not _is_floats(position, _XYZ_LEN):
        raise ROSConfigError(f"pose.position must be [x, y, z] numbers; got {position!r}.")
    if not _is_floats(orientation, _QUAT_LEN):
        raise ROSConfigError(
            f"pose.orientation must be a 4-number quaternion; got {orientation!r}."
        )
    order = str(block.get("quaternion_order", "xyzw"))
    if order not in _QUATERNION_ORDERS:
        raise ROSConfigError(
            f"pose.quaternion_order must be one of {_QUATERNION_ORDERS}; got {order!r}."
        )
    a, b, c, d = (float(v) for v in orientation)
    quat_xyzw = (a, b, c, d) if order == "xyzw" else (b, c, d, a)
    pose = Pose6D(
        xyz=(float(position[0]), float(position[1]), float(position[2])),
        quat_xyzw=quat_xyzw,
        frame_id=frame_id,
    )
    tool = block.get("tool_frame")
    tool_frame = str(tool) if tool is not None else None
    pos_tol = float(block.get("position_tolerance_m", _DEFAULT_POSITION_TOLERANCE_M))
    orient_tol = float(block.get("orientation_tolerance_rad", _DEFAULT_ORIENTATION_TOLERANCE_RAD))
    return pose, link_name, tool_frame, pos_tol, orient_tol


def _is_floats(value: Any, length: int) -> bool:  # noqa: ANN401  # reason: validates arbitrary JSON values
    return (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(isinstance(v, (int, float)) for v in value)
    )


class PoseGoalRskill(ROSActionRskill):
    """Cartesian-pose MoveGroup skill (``ros_integration.goal_builder: "pose"``).

    Consumes the merged goal's ``pose`` block (instead of raw constraints) and
    lowers it into a full-orientation MoveGroup pose goal on the first
    :meth:`step`, then dispatches + replays exactly like the parent.
    """

    def _configure_impl(self) -> None:
        super()._configure_impl()
        block = self._goal_dict.pop("pose", None)
        if not isinstance(block, dict):
            raise ROSConfigError(
                f"PoseGoalRskill({self.name!r}): the merged goal JSON has no 'pose' object "
                "— the manifest's default_goal_json must carry one (the LLM's "
                "goal_params_json may override its fields)."
            )
        (
            self._pose,
            self._link_name,
            self._tool_frame,
            self._pos_tol,
            self._orient_tol,
        ) = pose_from_block(block)
        self._constraints_lowered = False
        self._tf_buffer: Any = None
        if self._tool_frame is not None:
            # A tool/TCP offset is requested → TF is the only source of frames
            # (CLAUDE.md). Build the listener now; the lookup runs lazily on the
            # first step() once the host node is spinning (ADR-0054 Q3).
            try:
                import tf2_ros  # noqa: PLC0415  # reason: ROS runtime dep, absent in pure-unit environments
            except ImportError as exc:
                raise ROSConfigError(
                    f"PoseGoalRskill({self.name!r}): pose.tool_frame is set but tf2_ros is "
                    "unavailable — TF is the only source of the link←tool offset. Source a "
                    "ROS 2 workspace, or drop tool_frame to constrain link_name directly."
                ) from exc
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
        log.info(
            "pose_goal_rskill.configured",
            name=self.name,
            link_name=self._link_name,
            tool_frame=self._tool_frame,
            frame_id=self._pose.frame_id,
        )

    def _resolve_tool_offset(self) -> NDArrayOrNone:
        """TF-look-up ``link_name ← tool_frame`` → homogeneous offset, or ``None``."""
        if self._tool_frame is None:
            return None
        import rclpy.duration  # noqa: PLC0415  # reason: ROS runtime dep
        import rclpy.time  # noqa: PLC0415  # reason: ROS runtime dep
        from openral_core.exceptions import ROSRuntimeError  # noqa: PLC0415

        try:
            tf = self._tf_buffer.lookup_transform(
                self._link_name,
                self._tool_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=_TF_LOOKUP_TIMEOUT_S),
            )
        except Exception as exc:  # reason: tf2 raises several lookup/extrapolation types
            raise ROSRuntimeError(
                f"PoseGoalRskill({self.name!r}): no TF {self._link_name!r} ← "
                f"{self._tool_frame!r} within {_TF_LOOKUP_TIMEOUT_S}s — is "
                "robot_state_publisher up and the tool frame in the URDF/static "
                f"broadcasters? ({exc})"
            ) from exc
        t = tf.transform.translation
        q = tf.transform.rotation
        return homogeneous_from_quat_xyz((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

    def _lower_constraints(self) -> None:
        entry = build_pose_constraints(
            pose=self._pose,
            link_name=self._link_name,
            link_t_target=self._resolve_tool_offset(),
            position_tolerance_m=self._pos_tol,
            orientation_axis_tolerances_rad=(self._orient_tol, self._orient_tol, self._orient_tol),
        )
        request = self._goal_dict.setdefault("request", {})
        request["goal_constraints"] = [entry]
        self._constraints_lowered = True

    def _step_impl(self, world_state: Any) -> Any:  # noqa: ANN401  # reason: matches parent WorldState/Action signature
        if not self._constraints_lowered:
            self._lower_constraints()
        return super()._step_impl(world_state)
