"""The depth-camera extrinsic gate: pass limits and report verification.

The kernel's world-voxel check places every obstacle through the depth camera's
``parent_frame -> frame_id`` mount, and nothing else measures that pose
(``openral calibrate camera`` fits intrinsics only). The camera sees the robot itself, so
the robot is the calibration target, for any robot with an MJCF model and a depth camera
in its manifest: ``tools/depth_extrinsic_capture.py`` drives the robot through the poses
committed in ``robots/<id>/calibration/extrinsic_fit_poses.yaml`` and records depth points
with the REAL joint readings; ``tools/depth_extrinsic_check.py check`` fits the mount to
the robot's own meshes posed at those readings and writes a report to
``robots/<id>/calibration/<unit>/<sensor>_extrinsic.json`` (per robot unit;
``calibration/<sensor>_extrinsic.json`` for a robot without ``units/``). This module is
the part both that tool and ``openral deploy run``'s preflight need: the pass limits,
derived from the real world-voxel margin, and :func:`verify_extrinsic_report`.

Kept out of ``openral_core.__init__`` and free of numpy, so the launch file can read
:data:`REAL_WORLD_VOXEL_MARGIN_M` without loading anything heavy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from openral_core.schemas import RobotDescription, SensorSpec

#: Clearance the kernel's world-voxel check keeps around the robot on a real deploy
#: (``world_voxel_margin_m``; sim runs at 0). The launch passes this value, and every
#: extrinsic limit below is derived from it, so tightening the margin tightens the gate.
REAL_WORLD_VOXEL_MARGIN_M: Final[float] = 0.02
#: Range at which an angular error is converted to a position error. A head camera sees
#: its workspace within 1 m; a closer (wrist) camera only makes the same angle smaller.
CHECKED_RANGE_M: Final[float] = 1.0
#: Height error and the tilt error at ``CHECKED_RANGE_M`` each get half the margin, so
#: together they never exceed it.
MAX_HEIGHT_ERR_M: Final[float] = REAL_WORLD_VOXEL_MARGIN_M / 2.0
MAX_TILT_DEG: Final[float] = math.degrees(math.atan(MAX_HEIGHT_ERR_M / CHECKED_RANGE_M))
#: Planar (x, y) error of the camera, and the yaw error that moves a point as far at
#: ``CHECKED_RANGE_M``.
MAX_PLANAR_ERR_M: Final[float] = 0.75 * REAL_WORLD_VOXEL_MARGIN_M
MAX_YAW_DEG: Final[float] = math.degrees(math.atan(MAX_PLANAR_ERR_M / CHECKED_RANGE_M))
#: Median distance of the robot's own returns from its meshes after the fit. Larger means
#: the joint readings, the meshes or the recording disagree with the robot, and a mount
#: fitted to them cannot be trusted to the limits above.
MAX_FIT_RESIDUAL_M: Final[float] = REAL_WORLD_VOXEL_MARGIN_M / 2.0
#: Fewer poses cannot leave one out and still pin all six degrees of freedom.
MIN_FIT_POSES: Final[int] = 4
#: Restarted from a mount displaced by one limit along an axis, the fit must come back to
#: within this fraction of that limit. On an axis the recording cannot see (an arm hanging
#: straight down shows neither the camera's height nor its yaw) the fit stays wherever it
#: starts, so whatever value it returned for that axis is not a measurement.
MAX_AXIS_UNRECOVERED: Final[float] = 0.5

#: The four error metrics and their limits; applied to the mount error and held-out spread.
ERROR_LIMITS: Final[dict[str, float]] = {
    "height_m": MAX_HEIGHT_ERR_M,
    "planar_m": MAX_PLANAR_ERR_M,
    "tilt_deg": MAX_TILT_DEG,
    "yaw_deg": MAX_YAW_DEG,
}
#: The six mount axes the recovery test displaces, each with the metric that measures it.
FIT_AXES: Final[dict[str, str]] = {
    "x": "planar_m",
    "y": "planar_m",
    "z": "height_m",
    "roll": "tilt_deg",
    "pitch": "tilt_deg",
    "yaw": "yaw_deg",
}


def extrinsic_criteria() -> dict[str, float | int]:
    """The shipped pass criteria, exactly as a report records them.

    Example:
        >>> extrinsic_criteria()["min_fit_poses"]
        4
    """
    return {
        "max_height_err_m": MAX_HEIGHT_ERR_M,
        "max_planar_err_m": MAX_PLANAR_ERR_M,
        "max_tilt_deg": MAX_TILT_DEG,
        "max_yaw_deg": MAX_YAW_DEG,
        "max_fit_residual_m": MAX_FIT_RESIDUAL_M,
        "max_axis_unrecovered": MAX_AXIS_UNRECOVERED,
        "min_fit_poses": MIN_FIT_POSES,
    }


def extrinsic_report_path(robot_yaml: Path, sensor: str, unit: str | None = None) -> Path:
    """Where ``sensor``'s extrinsic report lives, next to the manifest.

    ``<manifest dir>/calibration/<unit>/<sensor>_extrinsic.json`` for a robot unit
    (``robots/<id>/units/<unit>.yaml``: each unit mounts its camera, so each measures its
    own pose); ``<manifest dir>/calibration/<sensor>_extrinsic.json`` when ``unit`` is
    ``None`` — only for a robot that ships no ``units/`` (a real deploy of one that does
    refuses without a unit: ``openral_core.resolve_sensor_overlays``).

    Example:
        >>> extrinsic_report_path(Path("robots/openarm/robot.yaml"), "head_zed", "thor").as_posix()
        'robots/openarm/calibration/thor/head_zed_extrinsic.json'
        >>> extrinsic_report_path(Path("robots/g1/robot.yaml"), "head").as_posix()
        'robots/g1/calibration/head_extrinsic.json'
    """
    calibration = robot_yaml.parent / "calibration"
    return (calibration / unit if unit else calibration) / f"{sensor}_extrinsic.json"


def fit_poses_path(robot_yaml: Path) -> Path:
    """The robot's committed extrinsic-fit poses: ``<manifest dir>/calibration/...yaml``.

    One file per robot type, shared by its units: the poses depend on the robot's
    kinematics and where its cameras look, not on which cell it stands in.

    Example:
        >>> fit_poses_path(Path("robots/openarm/robot.yaml")).as_posix()
        'robots/openarm/calibration/extrinsic_fit_poses.yaml'
    """
    return robot_yaml.parent / "calibration" / "extrinsic_fit_poses.yaml"


def checkable_depth_sensor(description: RobotDescription, sensor: str) -> SensorSpec:
    """The manifest sensor ``sensor``, if its extrinsic can be measured and gated.

    Raises:
        ROSConfigError: no such sensor; not a depth camera (an RGB-only camera yields
            no points to fit); or no ``parent_frame`` + ``static_transform_xyz_rpy`` in
            the manifest (the pose under test).

    Example:
        >>> from openral_core import RobotDescription
        >>> desc = RobotDescription.from_yaml("robots/openarm/robot.yaml")
        >>> checkable_depth_sensor(desc, "head_zed").parent_frame
        'openarm_base'
    """
    spec = next((s for s in description.sensors if s.name == sensor), None)
    if spec is None:
        names = [s.name for s in description.sensors]
        raise ROSConfigError(f"robot {description.name!r} has no sensor {sensor!r}; has {names}")
    if not spec.is_depth_camera:
        raise ROSConfigError(
            f"sensor {sensor!r} is {spec.modality!r}"
            + (" without intrinsics" if spec.intrinsics is None else "")
            + ": the extrinsic check fits the robot's own depth returns, so it covers depth "
            "cameras only (modality depth/point_cloud with intrinsics)."
        )
    if spec.parent_frame is None or spec.static_transform_xyz_rpy is None:
        raise ROSConfigError(
            f"sensor {sensor!r} has no parent_frame + static_transform_xyz_rpy in the robot "
            "manifest; that mount is the pose the extrinsic check measures and gates."
        )
    return spec


def _within(value: object, limit: float) -> bool:
    """``value <= limit`` for a finite number; NaN, inf and non-numbers never pass."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value <= limit
    )


def _number(value: object) -> float | None:
    """``value`` as a finite float, else ``None``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _section(res: dict[str, Any], key: str) -> dict[str, Any]:
    value = res.get(key)
    return value if isinstance(value, dict) else {}


def residual_failures(res: dict[str, Any]) -> list[str]:
    """Every way the fit residuals ``res`` miss the shipped limits (empty = pass).

    For each metric (height, planar, tilt, yaw) the manifest pose's error against the fit
    PLUS the held-out spread (the worst refit with one pose left out) must stay within its
    limit; the fit must use enough poses, sit close to the robot's meshes, and see every
    axis. Written as "not within", so a NaN, inf or missing value fails instead of slipping
    through a ``>`` comparison.

    Example:
        >>> residual_failures({"poses": 2})[0]
        '2 pose(s); 4 needed'
    """
    failures = []
    poses = res.get("poses")
    if not isinstance(poses, int) or isinstance(poses, bool) or poses < MIN_FIT_POSES:
        failures.append(f"{poses} pose(s); {MIN_FIT_POSES} needed")
    if not _within(res.get("median_residual_m"), MAX_FIT_RESIDUAL_M):
        failures.append(
            f"median robot residual {res.get('median_residual_m')} m not <= {MAX_FIT_RESIDUAL_M}"
        )
    error, spread = _section(res, "mount_error"), _section(res, "heldout_spread")
    for key, limit in ERROR_LIMITS.items():
        e, h = _number(error.get(key)), _number(spread.get(key))
        if e is None or h is None or not _within(e + h, limit):
            failures.append(
                f"{key}: mount error {error.get(key)} + held-out spread {spread.get(key)} "
                f"not <= {limit}"
            )
    unrecovered = _section(res, "axis_unrecovered")
    for axis in FIT_AXES:
        if not _within(unrecovered.get(axis), MAX_AXIS_UNRECOVERED):
            failures.append(
                f"axis {axis} unobservable: restarted one limit off, the fit stayed "
                f"{unrecovered.get(axis)} of a limit away (need <= {MAX_AXIS_UNRECOVERED}); "
                "record more varied poses"
            )
    return failures


def verify_extrinsic_report(
    spec: SensorSpec, report_path: Path, *, base_frame: str, unit: str | None = None
) -> list[str]:
    """Every reason ``report_path`` does not clear ``spec``'s CURRENT manifest pose.

    Empty means verified. Refuses a missing or unreadable report, one that did not pass,
    one whose stored residuals fail this module's limits (the stored verdict is never
    trusted alone), one measured for another unit, sensor, frame pair, base frame or pose
    (stale after a manifest or unit-overlay edit), and one checked against looser criteria.

    Args:
        spec: The sensor as ``unit`` publishes it (manifest entry with the unit's
            ``SensorOverlay`` applied; see :func:`checkable_depth_sensor`).
        report_path: The ``check`` report (see :func:`extrinsic_report_path`).
        base_frame: The robot base frame the mount was fitted in.
        unit: The robot unit the report must be for (``None`` = a robot without units).
    """
    if not report_path.is_file():
        return [f"no extrinsic report at {report_path}"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"unreadable extrinsic report {report_path}: {exc}"]
    if not isinstance(report, dict):
        return [f"extrinsic report {report_path} is not a JSON object"]
    crit = _section(report, "criteria")
    problems = []
    if report.get("passed") is not True:
        problems.append(f"report did not pass: {report.get('failures')}")
    problems += [
        f"residuals fail the shipped limits: {f}"
        for f in residual_failures(_section(report, "residuals"))
    ]
    if report.get("unit") != unit:
        problems.append(f"report is for unit {report.get('unit')!r}, not {unit!r}")
    measured = (
        report.get("sensor"),
        report.get("parent_frame"),
        report.get("frame_id"),
        report.get("base_frame"),
    )
    if measured != (spec.name, spec.parent_frame, spec.frame_id, base_frame):
        problems.append(
            f"report is for (sensor, parent, frame, base) {measured}, not "
            f"{(spec.name, spec.parent_frame, spec.frame_id, base_frame)}"
        )
    pose = list(spec.static_transform_xyz_rpy or ())
    reported = report.get("static_transform_xyz_rpy")
    if not (
        isinstance(reported, list)
        and len(reported) == len(pose) > 0
        and all(
            _within(abs(float(a) - b), 1e-9) if isinstance(a, (int, float)) else False
            for a, b in zip(reported, pose, strict=True)
        )
    ):
        problems.append(f"manifest pose {pose} != checked pose {reported}")
    for key, limit in extrinsic_criteria().items():
        value = _number(crit.get(key))
        # A max_* may only be as tight or tighter; a min_* only as high or higher.
        tighter = key.startswith("max_") and value is not None and value <= limit + 1e-12
        higher = key.startswith("min_") and value is not None and value >= limit - 1e-12
        if not (tighter or higher):
            problems.append(f"criterion {key}={crit.get(key)} is looser than {limit}")
    return problems
