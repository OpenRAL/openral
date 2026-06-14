"""``LookAtRskill`` — aim a robot camera at a 3-D point via MoveGroup (ADR-0044 Phase 3).

A :class:`~openral_rskill.ros_action_rskill.ROSActionRskill` whose goal is not
authored as raw MoveGroup constraints but as a small ``look_at`` block
(``target_xyz`` + ``camera``), lowered at dispatch time into MoveGroup
**pose-goal** constraints:

1. Resolve the named camera from ``RobotDescription.sensors`` (default
   ``"wrist"``; :class:`~openral_core.exceptions.ROSConfigError` listing the
   available sensors when absent — explicit beats implicit).
2. Look up the camera's *current* pose over TF2 (the only source of frames,
   CLAUDE.md) in the goal frame.
3. Place the camera goal: at its current position (pure re-aim) or at
   ``standoff_m`` from the target along the current line of approach.
4. Orient it with :func:`~openral_world_state.geometry.compute_gaze_pose`
   (ROS optical convention: camera ``+Z`` hits the target) and lower the
   result into ``position_constraints`` + ``orientation_constraints`` for the
   camera's link. Roll about the optical axis is left free (tolerance π) —
   image roll doesn't change what the camera sees, and the slack buys the
   planner reachability.

The planned trajectory then replays waypoint-per-chunk through
``/openral/candidate_action`` exactly like the parent class — the safety
kernel sees every aiming step.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import structlog
from openral_core import Pose6D, RobotDescription, SensorSpec
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_world_state.geometry import compute_gaze_pose
from openral_world_state.object_lift import homogeneous_from_quat_xyz

from openral_rskill.pose_goal_rskill import build_pose_constraints
from openral_rskill.ros_action_rskill import ROSActionRskill

__all__ = ["LookAtRskill", "build_look_at_constraints", "resolve_camera_sensor"]

log = structlog.get_logger(__name__)

NDArrayOrNone = Any  # reason: numpy NDArray | None alias keeps signatures readable

_DEFAULT_CAMERA = "wrist"
_DEFAULT_POSITION_TOLERANCE_M = 0.02
_DEFAULT_ORIENTATION_TOLERANCE_RAD = 0.15
_TF_LOOKUP_TIMEOUT_S = 5.0
_XYZ_LEN = 3
_MIN_DIRECTION_NORM = 1e-9


def resolve_camera_sensor(description: RobotDescription | None, camera: str) -> SensorSpec:
    """Find the named camera in the robot manifest, or fail with the menu.

    Raises:
        ROSConfigError: When no description is available or no sensor matches
            ``camera`` — the message lists the robot's sensor names rather
            than silently guessing which camera to aim (ADR-0044).
    """
    if description is None:
        raise ROSConfigError(
            "LookAtRskill needs the host RobotDescription to resolve the camera; none was provided."
        )
    for sensor in description.sensors:
        if sensor.name == camera:
            return sensor
    available = ", ".join(repr(s.name) for s in description.sensors) or "<none>"
    raise ROSConfigError(
        f"LookAtRskill: robot {description.name!r} declares no sensor named "
        f"{camera!r}; available sensors: {available}. Pass look_at.camera "
        "explicitly or add the camera to the robot manifest."
    )


def _camera_mount(sensor: SensorSpec) -> tuple[str, NDArrayOrNone]:
    """The robot link MoveGroup constrains, and the link→camera offset.

    A sensor whose ``frame_id`` is itself a robot link (no declared static
    mount — franka's LIBERO eye-in-hand on ``panda_hand``) is constrained
    directly with an identity offset. A sensor with ``parent_frame`` +
    ``static_transform_xyz_rpy`` (so101-style wrist cam) constrains the
    parent link through the declared mount transform.
    """
    if sensor.parent_frame is not None and sensor.static_transform_xyz_rpy is not None:
        x, y, z, roll, pitch, yaw = sensor.static_transform_xyz_rpy
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        link_t_cam = homogeneous_from_quat_xyz((x, y, z), (qx, qy, qz, qw))
        return sensor.parent_frame, link_t_cam
    return sensor.frame_id, None


def build_look_at_constraints(
    *,
    camera_goal: Pose6D,
    link_name: str,
    link_t_cam: NDArrayOrNone = None,
    position_tolerance_m: float = _DEFAULT_POSITION_TOLERANCE_M,
    orientation_tolerance_rad: float = _DEFAULT_ORIENTATION_TOLERANCE_RAD,
) -> dict[str, Any]:
    """Lower a camera gaze pose into one MoveGroup ``goal_constraints`` entry.

    When ``link_t_cam`` is given the camera goal is re-expressed for the
    mount link (``goal_link = goal_cam @ inv(link_t_cam)``); otherwise the
    camera frame *is* the constrained link. Roll about the camera's optical
    axis (link ``z`` when the offset is identity) gets tolerance π — what the
    camera sees is roll-invariant.
    """
    # Delegate to the shared ADR-0054 lowering. Look-at is the gaze specialisation:
    # roll about the optical (z) axis is free (tolerance π) — seeing the target is
    # roll-invariant, and the slack buys planner reachability.
    return build_pose_constraints(
        pose=camera_goal,
        link_name=link_name,
        link_t_target=link_t_cam,
        position_tolerance_m=position_tolerance_m,
        orientation_axis_tolerances_rad=(
            orientation_tolerance_rad,
            orientation_tolerance_rad,
            math.pi,
        ),
    )


class LookAtRskill(ROSActionRskill):
    """Camera-aiming MoveGroup skill (``ros_integration.goal_builder: look_at``).

    Consumes the merged goal's ``look_at`` block instead of raw constraints:

    - ``target_xyz`` (required): the 3-D point to aim at, in ``frame_id``.
    - ``frame_id``: planning frame both poses are expressed in.
    - ``camera``: sensor name from the robot manifest (default ``"wrist"``).
    - ``standoff_m``: optional — re-position the camera this far from the
      target along its current line of approach; omitted = re-aim in place.
    - ``position_tolerance_m`` / ``orientation_tolerance_rad``: constraint
      tolerances.

    The lowering needs the camera's *current* pose, which only TF can give —
    so it happens lazily on the first :meth:`step` (the host node is spinning
    by then), right before the parent dispatches the wrapped MoveGroup goal.
    """

    def _configure_impl(self) -> None:
        super()._configure_impl()
        params = self._goal_dict.pop("look_at", None)
        if not isinstance(params, dict):
            raise ROSConfigError(
                f"LookAtRskill({self.name!r}): the merged goal JSON has no 'look_at' "
                "object — the manifest's default_goal_json must carry one (and the "
                "LLM's goal_params_json may override its fields)."
            )
        target = params.get("target_xyz")
        if (
            not isinstance(target, (list, tuple))
            or len(target) != _XYZ_LEN
            or not all(isinstance(v, (int, float)) for v in target)
        ):
            raise ROSConfigError(
                f"LookAtRskill({self.name!r}): look_at.target_xyz must be [x, y, z] "
                f"numbers; got {target!r}."
            )
        self._target_xyz = (float(target[0]), float(target[1]), float(target[2]))
        self._goal_frame = str(params.get("frame_id", "map"))
        camera = str(params.get("camera", _DEFAULT_CAMERA))
        standoff = params.get("standoff_m")
        self._standoff_m = float(standoff) if standoff is not None else None
        self._position_tol = float(
            params.get("position_tolerance_m", _DEFAULT_POSITION_TOLERANCE_M)
        )
        self._orientation_tol = float(
            params.get("orientation_tolerance_rad", _DEFAULT_ORIENTATION_TOLERANCE_RAD)
        )
        self._sensor = resolve_camera_sensor(self._description, camera)
        self._link_name, self._link_t_cam = _camera_mount(self._sensor)
        self._constraints_lowered = False

        try:
            import tf2_ros  # noqa: PLC0415  # reason: ROS runtime dep, absent in pure-unit environments
        except ImportError as exc:
            raise ROSConfigError(
                f"LookAtRskill({self.name!r}): tf2_ros is unavailable — TF2 is the "
                "only source of the camera's current pose. Source a ROS 2 workspace."
            ) from exc
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
        log.info(
            "look_at_rskill.configured",
            name=self.name,
            camera=self._sensor.name,
            camera_frame=self._sensor.frame_id,
            constrained_link=self._link_name,
            target_xyz=self._target_xyz,
            goal_frame=self._goal_frame,
            standoff_m=self._standoff_m,
        )

    def _current_camera_xyz(self) -> tuple[float, float, float]:
        """The camera frame's current position in the goal frame, via TF2."""
        import rclpy.duration  # noqa: PLC0415  # reason: ROS runtime dep
        import rclpy.time  # noqa: PLC0415  # reason: ROS runtime dep

        try:
            tf = self._tf_buffer.lookup_transform(
                self._goal_frame,
                self._sensor.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=_TF_LOOKUP_TIMEOUT_S),
            )
        except Exception as exc:  # reason: tf2 raises several lookup/extrapolation types
            raise ROSRuntimeError(
                f"LookAtRskill({self.name!r}): no TF {self._goal_frame!r} ← "
                f"{self._sensor.frame_id!r} within {_TF_LOOKUP_TIMEOUT_S}s — is "
                "robot_state_publisher up and the camera frame in the URDF/static "
                f"broadcasters? ({exc})"
            ) from exc
        t = tf.transform.translation
        return (float(t.x), float(t.y), float(t.z))

    def _lower_constraints(self) -> None:
        """Compute the gaze pose from live TF and write the MoveGroup constraints."""
        current = self._current_camera_xyz()
        if self._standoff_m is not None:
            direction = np.asarray(current, dtype=np.float64) - np.asarray(
                self._target_xyz, dtype=np.float64
            )
            norm = float(np.linalg.norm(direction))
            unit = direction / norm if norm > _MIN_DIRECTION_NORM else np.asarray((-1.0, 0.0, 0.0))
            goal_xyz_arr = np.asarray(self._target_xyz) + unit * self._standoff_m
            goal_xyz = (
                float(goal_xyz_arr[0]),
                float(goal_xyz_arr[1]),
                float(goal_xyz_arr[2]),
            )
        else:
            goal_xyz = current
        camera_goal = compute_gaze_pose(
            goal_xyz, self._target_xyz, frame_id=self._goal_frame, view_axis="+z"
        )
        entry = build_look_at_constraints(
            camera_goal=camera_goal,
            link_name=self._link_name,
            link_t_cam=self._link_t_cam,
            position_tolerance_m=self._position_tol,
            orientation_tolerance_rad=self._orientation_tol,
        )
        request = self._goal_dict.setdefault("request", {})
        request["goal_constraints"] = [entry]
        self._constraints_lowered = True
        log.info(
            "look_at_rskill.constraints_lowered",
            name=self.name,
            camera_current_xyz=current,
            camera_goal_xyz=camera_goal.xyz,
            target_xyz=self._target_xyz,
        )

    def _step_impl(self, world_state: Any) -> Any:  # noqa: ANN401  # reason: matches parent's WorldState/Action signature without re-importing
        if not self._constraints_lowered:
            self._lower_constraints()
        return super()._step_impl(world_state)
