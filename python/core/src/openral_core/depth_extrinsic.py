"""The depth-camera extrinsic gate: pass limits and report verification.

The kernel's world-voxel check places every obstacle through the depth camera's
``parent_frame -> frame_id`` mount, and nothing else measures that pose
(``openral calibrate camera`` fits intrinsics only). ``tools/depth_extrinsic_check.py
check`` measures it from a recorded bag and writes a report to
``robots/<id>/calibration/<sensor>_extrinsic.json``; this module is the part both that
tool and ``openral deploy run``'s preflight need: the pass limits, derived from the
real world-voxel margin, and :func:`verify_extrinsic_report`.

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
#: Range at which a tilt error is converted to a position error. A head camera sees its
#: table within 1 m; a closer (wrist) camera only makes the same tilt smaller in metres.
CHECKED_RANGE_M: Final[float] = 1.0
#: Height error and the tilt error at ``CHECKED_RANGE_M`` each get half the margin, so
#: together they never exceed it.
MAX_HEIGHT_ERR_M: Final[float] = REAL_WORLD_VOXEL_MARGIN_M / 2.0
MAX_TILT_DEG: Final[float] = math.degrees(math.atan(MAX_HEIGHT_ERR_M / CHECKED_RANGE_M))
#: Planar (x, y, yaw) error measured at the markers.
MAX_MARKER_ERR_M: Final[float] = 0.75 * REAL_WORLD_VOXEL_MARGIN_M
#: One marker cannot tell a yaw error from a translation.
MIN_MARKERS: Final[int] = 2


def extrinsic_report_path(robot_yaml: Path, sensor: str) -> Path:
    """``<manifest dir>/calibration/<sensor>_extrinsic.json`` — next to the manifest.

    Example:
        >>> extrinsic_report_path(Path("robots/openarm/robot.yaml"), "head_zed").as_posix()
        'robots/openarm/calibration/head_zed_extrinsic.json'
    """
    return robot_yaml.parent / "calibration" / f"{sensor}_extrinsic.json"


def checkable_depth_sensor(description: RobotDescription, sensor: str) -> SensorSpec:
    """The manifest sensor ``sensor``, if its extrinsic can be measured and gated.

    Raises:
        ROSConfigError: no such sensor; not a depth camera (an RGB-only camera yields
            no cloud to fit a table plane to); or no ``parent_frame`` +
            ``static_transform_xyz_rpy`` in the manifest (the pose under test).

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
            + ": the extrinsic check fits a table plane to a depth cloud, so it covers depth "
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


def residual_failures(
    res: dict[str, Any],
    *,
    max_tilt_deg: float = MAX_TILT_DEG,
    max_height_err_m: float = MAX_HEIGHT_ERR_M,
    max_marker_err_m: float = MAX_MARKER_ERR_M,
) -> list[str]:
    """Every way the residuals ``res`` miss the limits (empty = pass).

    Written as "not within", so a NaN or inf residual — or limit — fails instead of
    slipping through a ``>`` comparison.

    Example:
        >>> residual_failures({"tilt_deg": 0.1, "height_err_m": 0.0, "markers": []})
        ['0 marker(s); 2 needed to fix yaw']
    """
    failures = []
    if not _within(res.get("tilt_deg"), max_tilt_deg):
        failures.append(f"table tilt {res.get('tilt_deg')} deg not <= {max_tilt_deg}")
    height = res.get("height_err_m")
    if not _within(abs(height) if isinstance(height, (int, float)) else height, max_height_err_m):
        failures.append(f"table height error {height} m not within {max_height_err_m}")
    markers = res.get("markers") or []
    if len(markers) < MIN_MARKERS:
        failures.append(f"{len(markers)} marker(s); {MIN_MARKERS} needed to fix yaw")
    failures += [
        f"marker {m.get('expected_xy')} off by {m.get('error_m')} m (limit {max_marker_err_m})"
        for m in markers
        if not _within(m.get("error_m"), max_marker_err_m)
    ]
    return failures


def verify_extrinsic_report(spec: SensorSpec, report_path: Path, *, base_frame: str) -> list[str]:
    """Every reason ``report_path`` does not clear ``spec``'s CURRENT manifest pose.

    Empty means verified. Refuses a missing or unreadable report, one that did not pass,
    one whose stored residuals fail this module's limits (the stored verdict is never
    trusted alone), one measured for another sensor, frame pair, base frame or pose
    (stale after a manifest edit), and one checked against looser criteria.

    Args:
        spec: The manifest sensor (see :func:`checkable_depth_sensor`).
        report_path: The ``check`` report (see :func:`extrinsic_report_path`).
        base_frame: The robot base frame the table and markers were measured in.
    """
    if not report_path.is_file():
        return [f"no extrinsic report at {report_path}"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"unreadable extrinsic report {report_path}: {exc}"]
    if not isinstance(report, dict):
        return [f"extrinsic report {report_path} is not a JSON object"]
    crit = report.get("criteria")
    crit = crit if isinstance(crit, dict) else {}
    residuals = report.get("residuals")
    problems = []
    if report.get("passed") is not True:
        problems.append(f"report did not pass: {report.get('failures')}")
    problems += [
        f"residuals fail the shipped limits: {f}"
        for f in residual_failures(residuals if isinstance(residuals, dict) else {})
    ]
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
    for key, limit in (
        ("max_tilt_deg", MAX_TILT_DEG),
        ("max_height_err_m", MAX_HEIGHT_ERR_M),
        ("max_marker_err_m", MAX_MARKER_ERR_M),
    ):
        if not _within(crit.get(key), limit + 1e-12):
            problems.append(f"criterion {key}={crit.get(key)} is looser than {limit}")
    min_markers = crit.get("min_markers")
    if not isinstance(min_markers, int) or min_markers < MIN_MARKERS:
        problems.append(f"criterion min_markers={min_markers} < {MIN_MARKERS}")
    return problems
