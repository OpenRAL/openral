"""Measure a depth camera's extrinsic against the table it looks at, from a recorded bag.

The safety kernel's world-voxel check trusts ``parent_frame -> frame_id`` for the depth
camera absolutely: every obstacle it will ever stop on is placed through that transform.
Intrinsics do not enter (the depth driver builds the cloud from its own calibration); the
EXTRINSIC does, and ``openral calibrate camera`` does not measure it. This tool does,
offline, from a bag the operator records with the robot held still:

* **Table plane** (roll, pitch, height). Points in ``--table-roi`` are fitted to a plane
  in the robot BASE frame (the manifest's ``base_frame``); its tilt from horizontal and its
  height against the measured ``--table-z`` are the residuals.
* **Markers** (x, y, yaw). Flat objects a few cm thick placed at tape-measured base-frame
  ``--marker X Y`` positions; the centroid of the cloud standing on the table within
  ``--marker-radius`` of each is compared with where it was placed. Two or more are
  required to pass: one marker cannot tell a yaw error from a translation.

The pose under test is the ``--sensor`` entry exactly as a deploy of ``--unit`` publishes
it: the robot manifest's entry with that unit's overlay (``robots/<id>/units/<unit>.yaml``)
applied (a depth camera with ``parent_frame`` + ``static_transform_xyz_rpy``; RGB-only
cameras are refused, there is no cloud to fit). A robot that ships ``units/`` requires
``--unit``: its mounts are per cell. The bag supplies the clouds, the camera-internal TF below the mount frame
(``frame_id -> <cloud frame>``, from the driver's own URDF), and — when ``parent_frame`` is
not the base frame (a head on ``torso_link``, a wrist camera on ``gripper``) — the TF chain
``base_frame -> parent_frame`` at the recorded joint pose (record ``/tf`` + ``/tf_static``
from ``robot_state_publisher``). That chain must hold still across the clouds used. A camera
whose parent IS the base frame needs only the driver in the bag.

The report records the pose, and ``verify`` (``openral_core.depth_extrinsic``) refuses a
report for another unit, whose pose no longer matches the manifest + unit overlay, or whose
criteria are looser than the limits derived from the real world-voxel margin. ``openral
deploy run`` applies the same gate, for the selected unit, before any real launch with the
world-voxel check on. The committed report lives next to the manifest, in
``robots/<id>/calibration/<unit>/<sensor>_extrinsic.json`` (``calibration/<sensor>_extrinsic.json``
for a robot without ``units/``).

The fitted corrections are also composed into ``suggested_static_transform_xyz_rpy`` (in
``parent_frame``), to be copied into the unit overlay (or the manifest, without units). Passing on the bag the suggestion was
fitted to proves nothing (it is zero by construction); verify on a SECOND bag with the
markers moved.

Run (ROS 2 sourced, for rosbag2_py / tf2)::

    uv run python tools/depth_extrinsic_check.py check \\
        --robot robots/openarm/robot.yaml --sensor head_zed --unit thor --bag <bag_dir> \\
        --cloud-topic /zed/zed_node/point_cloud/cloud_registered \\
        --table-z -0.20 --table-roi 0.30 0.70 -0.30 0.30 \\
        --marker 0.45 0.15 --marker 0.55 -0.15 \\
        --out robots/openarm/calibration/thor/head_zed_extrinsic.json
    uv run python tools/depth_extrinsic_check.py verify \\
        --robot robots/openarm/robot.yaml --sensor head_zed --unit thor
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from openral_core import RobotDescription, apply_sensor_overlays, load_robot_unit
from openral_core.depth_extrinsic import (
    MAX_HEIGHT_ERR_M,
    MAX_MARKER_ERR_M,
    MAX_TILT_DEG,
    MIN_MARKERS,
    checkable_depth_sensor,
    extrinsic_report_path,
    residual_failures,
    verify_extrinsic_report,
)
from openral_core.exceptions import ROSConfigError

_TABLE_BAND_M = 0.15  # first-pass search band around --table-z; a bad pose is cm off
_INLIER_M = 0.01  # plane refit keeps points this close to the first fit
_MIN_PLANE_POINTS = 500
_MIN_MARKER_POINTS = 30
#: How far ``base_frame -> parent_frame`` may wander across the clouds used (m / rad):
#: a wrist or head that moves mid-recording has no single extrinsic to fit.
_PARENT_STILL_M = 1e-3
_PARENT_STILL_RAD = 1e-3

Points = NDArray[np.float64]
Rot = NDArray[np.float64]


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> Rot:
    """``static_transform_publisher``'s convention: fixed-axis XYZ, R = Rz Ry Rx."""
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _matrix_to_rpy(r: Rot) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    return math.atan2(float(r[2, 1]), float(r[2, 2])), pitch, math.atan2(r[1, 0], r[0, 0])


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> Rot:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _align(n: NDArray[np.float64]) -> Rot:
    """Smallest rotation taking unit vector ``n`` onto +z (Rodrigues)."""
    axis = np.cross(n, [0.0, 0.0, 1.0])
    s, c = float(np.linalg.norm(axis)), float(n[2])
    if s < 1e-12:
        return np.eye(3)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    out: Rot = np.eye(3) + s * kx + (1 - c) * (kx @ kx)
    return out


def _fit_plane(pts: Points) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    centroid = pts.mean(axis=0)
    normal = np.linalg.svd(pts - centroid, full_matrices=False)[2][2]
    return centroid, (normal if normal[2] > 0 else -normal)


def _evaluate(
    mount_pts: Points,
    rot: Rot,
    trans: NDArray[np.float64],
    *,
    table_z: float,
    roi: tuple[float, float, float, float],
    markers: list[tuple[float, float]],
    marker_radius: float,
    min_marker_height: float,
) -> dict[str, Any]:
    """Residuals of one candidate mount pose, all in the robot base frame."""
    pts = mount_pts @ rot.T + trans
    xmin, xmax, ymin, ymax = roi
    in_roi = (
        (pts[:, 0] >= xmin)
        & (pts[:, 0] <= xmax)
        & (pts[:, 1] >= ymin)
        & (pts[:, 1] <= ymax)
        & (np.abs(pts[:, 2] - table_z) <= _TABLE_BAND_M)
    )
    plane = pts[in_roi]
    if len(plane) < _MIN_PLANE_POINTS:
        raise ValueError(
            f"only {len(plane)} cloud points in the table ROI within {_TABLE_BAND_M} m of "
            f"--table-z {table_z}; need {_MIN_PLANE_POINTS}. Check the ROI, the table height, "
            "and that the pose is within ~15 cm of true."
        )
    centroid, normal = _fit_plane(plane)
    plane = plane[np.abs((plane - centroid) @ normal) <= _INLIER_M]
    centroid, normal = _fit_plane(plane)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    z_at_centre = (
        centroid[2]
        - (normal[0] * (cx - centroid[0]) + normal[1] * (cy - centroid[1])) / (normal[2])
    )
    found: list[dict[str, Any]] = []
    for mx, my in markers:
        # Height above the FITTED table plane, not the nominal table_z: with a height or
        # tilt error in the pose under test, a nominal cut would drop real marker points
        # (or admit table points) and bias the marker residual it is meant to measure.
        near = (np.hypot(pts[:, 0] - mx, pts[:, 1] - my) <= marker_radius) & (
            (pts - centroid) @ normal >= min_marker_height
        )
        if int(near.sum()) < _MIN_MARKER_POINTS:
            raise ValueError(
                f"marker at ({mx}, {my}): {int(near.sum())} points standing "
                f">= {min_marker_height} m above the table within {marker_radius} m; need "
                f"{_MIN_MARKER_POINTS}. Is it in view, and thick enough?"
            )
        c = pts[near, :2].mean(axis=0)
        found.append(
            {
                "expected_xy": [mx, my],
                "measured_xy": [float(c[0]), float(c[1])],
                "error_m": float(math.hypot(c[0] - mx, c[1] - my)),
            }
        )
    return {
        "plane_points": len(plane),
        "plane_normal": [float(v) for v in normal],
        "tilt_deg": math.degrees(math.acos(min(1.0, float(normal[2])))),
        "height_err_m": float(z_at_centre - table_z),
        "markers": found,
    }


def _suggest(
    mount_pts: Points, rot: Rot, trans: NDArray[np.float64], kw: dict[str, Any]
) -> tuple[Rot, NDArray[np.float64]]:
    """Compose tilt -> height -> planar (x, y, yaw) corrections onto the mount pose."""
    first = _evaluate(mount_pts, rot, trans, **{**kw, "markers": []})
    # Tilt: rotate about the camera origin so the table comes out horizontal.
    rot = _align(np.array(first["plane_normal"])) @ rot
    first = _evaluate(mount_pts, rot, trans, **{**kw, "markers": []})
    trans = trans - np.array([0.0, 0.0, first["height_err_m"]])
    found = _evaluate(mount_pts, rot, trans, **kw)["markers"]
    if not found:
        return rot, trans
    meas = np.array([m["measured_xy"] for m in found])
    want = np.array([m["expected_xy"] for m in found])
    yaw = 0.0
    if len(found) >= 2:  # 2-D Kabsch: best rotation about base z, then translation
        a, b = meas - meas.mean(axis=0), want - want.mean(axis=0)
        yaw = math.atan2(float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()), float((a * b).sum()))
    rz = _rpy_to_matrix(0.0, 0.0, yaw)
    shift = want.mean(axis=0) - (rz[:2, :2] @ meas.mean(axis=0))
    return rz @ rot, rz @ trans + np.array([shift[0], shift[1], 0.0])


def _as_rt(tf: Any) -> tuple[Rot, NDArray[np.float64]]:
    r, t = tf.rotation, tf.translation
    return _quat_to_matrix(r.x, r.y, r.z, r.w), np.array([t.x, t.y, t.z])


def _read_bag(
    bag: Path,
    cloud_topic: str,
    *,
    mount_frame: str,
    parent_frame: str,
    base_frame: str,
    max_clouds: int,
    stride: int,
) -> tuple[Points, str, int, Rot, NDArray[np.float64]]:
    """Clouds from ``bag`` in ``mount_frame``, and the recorded ``base_frame <- parent_frame``.

    The cloud is re-expressed in the mount frame through the camera-internal TF; the mount
    itself (``parent_frame -> mount_frame``) is the manifest pose under test and is never
    read from the bag. ``base_frame <- parent_frame`` is identity when they are the same
    frame, else looked up at each cloud's stamp and required to hold still.
    """
    import rosbag2_py
    from rclpy.duration import Duration
    from rclpy.serialization import deserialize_message
    from rclpy.time import Time
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs_py.point_cloud2 import read_points_numpy
    from tf2_ros import Buffer, TransformException

    # Any: rosbag2_py has pybind stubs only when a ROS overlay is sourced, so a typed
    # reader would make `mypy --strict tools/` disagree between CI (no ROS) and a dev host.
    reader: Any = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    topics = reader.get_all_topics_and_types()
    types = {t.name: t.type for t in topics}
    if cloud_topic not in types:
        raise ValueError(f"{bag} has no {cloud_topic}; topics: {sorted(types)}")
    # Hold the whole recording: the chain is looked up at each cloud's own stamp.
    buf = Buffer(cache_time=Duration(seconds=24 * 3600))
    clouds = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in ("/tf", "/tf_static"):
            for tf in deserialize_message(data, get_message(types[topic])).transforms:
                if topic == "/tf_static":
                    buf.set_transform_static(tf, "bag")
                else:
                    buf.set_transform(tf, "bag")
        elif topic == cloud_topic:
            clouds.append(deserialize_message(data, get_message(types[topic])))
    if not clouds:
        raise ValueError(f"{bag}: no messages on {cloud_topic}")
    clouds = clouds[-max_clouds:]
    frame = clouds[0].header.frame_id
    try:
        rot, trans = _as_rt(buf.lookup_transform(mount_frame, frame, Time()).transform)
        chain = [
            _as_rt(
                buf.lookup_transform(
                    base_frame, parent_frame, Time.from_msg(c.header.stamp)
                ).transform
            )
            for c in clouds
            if parent_frame != base_frame
        ] or [(np.eye(3), np.zeros(3))]
    except TransformException as exc:
        raise ValueError(
            f"{bag}: TF lookup failed ({exc}). The bag needs {mount_frame} -> {frame} (the "
            "driver's own TF)"
            + (
                f" and {base_frame} -> {parent_frame} at each cloud's stamp: record /tf and "
                "/tf_static from robot_state_publisher with the robot held still."
                if parent_frame != base_frame
                else "."
            )
        ) from exc
    base_rot, base_trans = chain[0]
    for r, t in chain[1:]:
        moved = float(np.linalg.norm(t - base_trans))
        turned = math.acos(max(-1.0, min(1.0, (float(np.trace(base_rot.T @ r)) - 1.0) / 2.0)))
        if moved > _PARENT_STILL_M or turned > _PARENT_STILL_RAD:
            raise ValueError(
                f"{parent_frame} moved relative to {base_frame} during the recording "
                f"({moved * 1000:.1f} mm, {math.degrees(turned):.2f} deg): hold the robot "
                "still while recording."
            )
    pts = np.concatenate(
        [
            read_points_numpy(c, field_names=("x", "y", "z"), skip_nans=True)[::stride]
            for c in clouds
        ]
    ).astype(np.float64)
    return pts @ rot.T + trans, frame, len(clouds), base_rot, base_trans


def _unit_description(robot_yaml: Path, sensor: str, unit: str | None) -> RobotDescription:
    """The manifest with ``unit``'s sensor overlay applied: what a deploy of it publishes.

    Raises:
        ROSConfigError: ``unit`` is invalid, or ``None`` for a robot that ships ``units/``
            (checked after ``sensor`` itself, so an RGB-only camera is refused as such).
    """
    description = RobotDescription.from_yaml(str(robot_yaml))
    if unit is not None:
        overlays = load_robot_unit(robot_yaml, unit).sensors
        description = description.model_copy(
            update={"sensors": apply_sensor_overlays(description.sensors, overlays)}
        )
    checkable_depth_sensor(description, sensor)
    units_dir = robot_yaml.parent / "units"
    if unit is None and units_dir.is_dir():
        have = ", ".join(sorted(p.stem for p in units_dir.glob("*.yaml")))
        raise ROSConfigError(
            f"{robot_yaml.parent.name} ships per-unit overlays ({have}); its camera mounts "
            "are per unit, so pass --unit"
        )
    return description


def check(args: argparse.Namespace) -> int:
    """Measure the unit's pose against the bag and write the JSON report; 0 iff it passes."""
    description = _unit_description(args.robot, args.sensor, args.unit)
    spec = checkable_depth_sensor(description, args.sensor)
    assert spec.parent_frame is not None and spec.static_transform_xyz_rpy is not None
    base_frame = description.base_frame
    mount_pts, cloud_frame, n_clouds, base_rot, base_trans = _read_bag(
        args.bag,
        args.cloud_topic,
        mount_frame=spec.frame_id,
        parent_frame=spec.parent_frame,
        base_frame=base_frame,
        max_clouds=args.max_clouds,
        stride=args.stride,
    )
    x, y, z, roll, pitch, yaw = spec.static_transform_xyz_rpy
    # The camera's pose in the BASE frame: the recorded parent pose, then the manifest mount.
    rot = base_rot @ _rpy_to_matrix(roll, pitch, yaw)
    trans = base_rot @ np.array([x, y, z]) + base_trans
    kw: dict[str, Any] = {
        "table_z": args.table_z,
        "roi": tuple(args.table_roi),
        "markers": [tuple(m) for m in args.marker],
        "marker_radius": args.marker_radius,
        "min_marker_height": args.min_marker_height,
    }
    res = _evaluate(mount_pts, rot, trans, **kw)
    s_rot, s_trans = _suggest(mount_pts, rot, trans, kw)
    criteria = {
        "max_tilt_deg": args.max_tilt_deg,
        "max_height_err_m": args.max_height_err_m,
        "max_marker_err_m": args.max_marker_err_m,
        "min_markers": MIN_MARKERS,
    }
    failures = residual_failures(
        res,
        max_tilt_deg=args.max_tilt_deg,
        max_height_err_m=args.max_height_err_m,
        max_marker_err_m=args.max_marker_err_m,
    )
    report = {
        "unit": args.unit,
        "sensor": spec.name,
        "parent_frame": spec.parent_frame,
        "frame_id": spec.frame_id,
        "base_frame": base_frame,
        "parent_in_base_xyz_rpy": [*(float(v) for v in base_trans), *_matrix_to_rpy(base_rot)],
        "static_transform_xyz_rpy": list(spec.static_transform_xyz_rpy),
        "bag": str(args.bag),
        "cloud_topic": args.cloud_topic,
        "cloud_frame": cloud_frame,
        "clouds": n_clouds,
        "table_z": args.table_z,
        "table_roi": list(args.table_roi),
        "criteria": criteria,
        "residuals": res,
        "passed": not failures,
        "failures": failures,
        # Back from the base frame to the manifest's parent_frame.
        "suggested_static_transform_xyz_rpy": [
            *(float(v) for v in base_rot.T @ (s_trans - base_trans)),
            *_matrix_to_rpy(base_rot.T @ s_rot),
        ],
        "suggested_residuals": _evaluate(mount_pts, s_rot, s_trans, **kw),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print("PASS" if not failures else "FAIL: " + "; ".join(failures), file=sys.stderr)
    return 0 if not failures else 1


def verify(args: argparse.Namespace) -> int:
    """0 iff the report passed, at no looser criteria, for the unit's CURRENT pose."""
    description = _unit_description(args.robot, args.sensor, args.unit)
    spec = checkable_depth_sensor(description, args.sensor)
    report = args.report or extrinsic_report_path(args.robot, spec.name, args.unit)
    problems = verify_extrinsic_report(
        spec, report, base_frame=description.base_frame, unit=args.unit
    )
    for p in problems:
        print(f"REFUSE: {p}", file=sys.stderr)
    if not problems:
        print(f"extrinsic verified: {spec.name} {list(spec.static_transform_xyz_rpy or ())}")
    return 1 if problems else 0


def _finite_float(text: str) -> float:
    """argparse type: a finite float (``nan``/``inf`` would defeat every limit check)."""
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"{text!r} is not a finite number")
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``check`` (bag -> report) or ``verify`` (report vs robot manifest)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--robot", type=Path, required=True, help="robots/<id>/robot.yaml")
        p.add_argument("--sensor", required=True, help="the manifest's depth camera name")
        p.add_argument(
            "--unit", default=None, help="robots/<id>/units/<unit>.yaml overlay (this cell)"
        )
    c = sub.choices["check"]
    finite = _finite_float
    c.add_argument("--bag", type=Path, required=True)
    c.add_argument("--cloud-topic", required=True, help="the depth driver's PointCloud2")
    c.add_argument("--table-z", type=finite, required=True, help="table top, base frame (m)")
    c.add_argument(
        "--table-roi", type=finite, nargs=4, required=True, metavar=("XMIN", "XMAX", "YMIN", "YMAX")
    )
    c.add_argument(
        "--marker", type=finite, nargs=2, action="append", default=[], metavar=("X", "Y")
    )
    c.add_argument("--marker-radius", type=finite, default=0.08)
    c.add_argument("--min-marker-height", type=finite, default=0.01)
    c.add_argument("--max-clouds", type=int, default=10)
    c.add_argument("--stride", type=int, default=4, help="keep every Nth point per cloud")
    c.add_argument("--max-tilt-deg", type=finite, default=MAX_TILT_DEG)
    c.add_argument("--max-height-err-m", type=finite, default=MAX_HEIGHT_ERR_M)
    c.add_argument("--max-marker-err-m", type=finite, default=MAX_MARKER_ERR_M)
    c.add_argument("--out", type=Path, default=None)
    sub.choices["verify"].add_argument(
        "--report",
        type=Path,
        default=None,
        help="default: <robot dir>/calibration/<unit>/<sensor>_extrinsic.json",
    )
    args = parser.parse_args(argv)
    try:
        return check(args) if args.cmd == "check" else verify(args)
    except (ValueError, ROSConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
