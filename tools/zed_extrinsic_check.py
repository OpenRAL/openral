"""Measure a head camera's extrinsic against the table it looks at, from a recorded bag.

The safety kernel's world-voxel check trusts ``parent_frame -> frame_id`` for the depth
camera absolutely: every obstacle it will ever stop on is placed through that transform.
Intrinsics do not enter (the ZED SDK builds the cloud from its own factory calibration);
the EXTRINSIC does, and ``openral calibrate camera`` does not measure it. This tool does,
offline, from a bag the operator records with the arms unpowered:

* **Table plane** (roll, pitch, height). Points in ``--table-roi`` are fitted to a plane
  in the robot base frame; its tilt from horizontal and its height against the measured
  ``--table-z`` are the residuals.
* **Markers** (x, y, yaw). Flat objects a few cm thick placed at tape-measured base-frame
  ``--marker X Y`` positions; the centroid of the cloud standing on the table within
  ``--marker-radius`` of each is compared with where it was placed. Two or more are
  required to pass: one marker cannot tell a yaw error from a translation.

The pose under test is the ROBOT MANIFEST's ``--sensor`` entry: the camera is bolted to
the robot, so its mount is robot geometry, published by every scene on that robot, and no
scene may restate it (``openral_core.check_scene_sensor_overrides``). The bag supplies
only the clouds and the camera-internal TF below the mount frame (``zed_camera_link ->
<cloud frame>``, from the ZED wrapper's own URDF), so the bag can be recorded with the
ZED driver alone. The report records that pose, and ``verify`` refuses a report whose pose
no longer matches the manifest or whose criteria are looser than this tool's — the gate
``tools/openarm_world_voxel_run.sh`` applies before a real-arm launch. The committed
report lives next to the manifest, in ``robots/<id>/calibration/<sensor>_extrinsic.json``.

The fitted corrections are also composed into ``suggested_static_transform_xyz_rpy``, to
be copied into the manifest. Passing on the bag the suggestion was fitted to proves
nothing (it is zero by construction); verify on a SECOND bag with the markers moved.

Run (ROS 2 sourced, for rosbag2_py / tf2)::

    uv run python tools/zed_extrinsic_check.py check \\
        --robot robots/openarm/robot.yaml --bag <bag_dir> \\
        --cloud-topic /zed/zed_node/point_cloud/cloud_registered \\
        --table-z -0.20 --table-roi 0.30 0.70 -0.30 0.30 \\
        --marker 0.45 0.15 --marker 0.55 -0.15 --out report.json
    uv run python tools/zed_extrinsic_check.py verify \\
        --robot robots/openarm/robot.yaml \\
        --report robots/openarm/calibration/head_zed_extrinsic.json
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

#: Pass criteria. Proposed, not measured on a rig: each is half or less of the real
#: world-voxel margin (20 mm) at the ranges the head camera sees the table (<= 1 m,
#: where 0.75 deg is 13 mm). ``verify`` refuses a report checked against looser ones.
MAX_TILT_DEG = 0.75
MAX_HEIGHT_ERR_M = 0.010
MAX_MARKER_ERR_M = 0.015
MIN_MARKERS = 2

_TABLE_BAND_M = 0.15  # first-pass search band around --table-z; a bad pose is cm off
_INLIER_M = 0.01  # plane refit keeps points this close to the first fit
_MIN_PLANE_POINTS = 500
_MIN_MARKER_POINTS = 30

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


def _manifest_sensor(robot_yaml: Path, sensor: str) -> Any:
    """The sensor's mount exactly as every deploy publishes it: the robot manifest's entry."""
    from openral_core import RobotDescription

    for spec in RobotDescription.from_yaml(str(robot_yaml)).sensors:
        if spec.name == sensor:
            if spec.parent_frame is None or spec.static_transform_xyz_rpy is None:
                raise ValueError(f"sensor {sensor!r} has no parent_frame + static transform")
            return spec
    raise ValueError(f"no sensor named {sensor!r} in {robot_yaml}")


def _read_bag(
    bag: Path, cloud_topic: str, mount_frame: str, max_clouds: int, stride: int
) -> tuple[Points, str, int]:
    """Clouds from ``bag``, re-expressed in ``mount_frame`` through the camera-internal TF."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rclpy.time import Time
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs_py.point_cloud2 import read_points_numpy
    from tf2_ros import Buffer

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    topics = reader.get_all_topics_and_types()  # type: ignore[no-untyped-call]  # reason: pybind stub
    types = {t.name: t.type for t in topics}
    if cloud_topic not in types:
        raise ValueError(f"{bag} has no {cloud_topic}; topics: {sorted(types)}")
    buf = Buffer()
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
    tf = buf.lookup_transform(mount_frame, frame, Time()).transform
    rot = _quat_to_matrix(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
    trans = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
    pts = np.concatenate(
        [
            read_points_numpy(c, field_names=("x", "y", "z"), skip_nans=True)[::stride]
            for c in clouds
        ]
    ).astype(np.float64)
    return pts @ rot.T + trans, frame, len(clouds)


def check(args: argparse.Namespace) -> int:
    """Measure the manifest's pose against the bag and write the JSON report; 0 iff it passes."""
    spec = _manifest_sensor(args.robot, args.sensor)
    mount_pts, cloud_frame, n_clouds = _read_bag(
        args.bag, args.cloud_topic, spec.frame_id, args.max_clouds, args.stride
    )
    x, y, z, roll, pitch, yaw = spec.static_transform_xyz_rpy
    rot, trans = _rpy_to_matrix(roll, pitch, yaw), np.array([x, y, z])
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
    failures = []
    if res["tilt_deg"] > args.max_tilt_deg:
        failures.append(f"table tilt {res['tilt_deg']:.2f} deg > {args.max_tilt_deg}")
    if abs(res["height_err_m"]) > args.max_height_err_m:
        failures.append(f"table height error {res['height_err_m'] * 1000:+.1f} mm")
    if len(res["markers"]) < MIN_MARKERS:
        failures.append(f"{len(res['markers'])} marker(s); {MIN_MARKERS} needed to fix yaw")
    failures += [
        f"marker {m['expected_xy']} off by {m['error_m'] * 1000:.1f} mm"
        for m in res["markers"]
        if m["error_m"] > args.max_marker_err_m
    ]
    report = {
        "sensor": spec.name,
        "parent_frame": spec.parent_frame,
        "frame_id": spec.frame_id,
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
        "suggested_static_transform_xyz_rpy": [
            *(float(v) for v in s_trans),
            *_matrix_to_rpy(s_rot),
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
    """0 iff ``--report`` passed, at no looser criteria, for the manifest's CURRENT pose."""
    spec = _manifest_sensor(args.robot, args.sensor)
    if not args.report.is_file():
        print(f"REFUSE: no extrinsic report at {args.report}", file=sys.stderr)
        return 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    crit = report.get("criteria", {})
    problems = []
    if report.get("passed") is not True:
        problems.append(f"report did not pass: {report.get('failures')}")
    if (report.get("sensor"), report.get("parent_frame"), report.get("frame_id")) != (
        spec.name,
        spec.parent_frame,
        spec.frame_id,
    ):
        problems.append("report is for a different sensor or frame pair")
    reported = report.get("static_transform_xyz_rpy") or []
    if len(reported) != 6 or not np.allclose(reported, spec.static_transform_xyz_rpy, atol=1e-9):
        problems.append(
            f"manifest pose {list(spec.static_transform_xyz_rpy)} != checked pose {reported}"
        )
    for key, limit in (
        ("max_tilt_deg", MAX_TILT_DEG),
        ("max_height_err_m", MAX_HEIGHT_ERR_M),
        ("max_marker_err_m", MAX_MARKER_ERR_M),
    ):
        if not isinstance(crit.get(key), (int, float)) or crit[key] > limit:
            problems.append(f"criterion {key}={crit.get(key)} is looser than {limit}")
    if not isinstance(crit.get("min_markers"), int) or crit["min_markers"] < MIN_MARKERS:
        problems.append(f"criterion min_markers={crit.get('min_markers')} < {MIN_MARKERS}")
    for p in problems:
        print(f"REFUSE: {p}", file=sys.stderr)
    if not problems:
        print(f"extrinsic verified: {spec.name} {list(spec.static_transform_xyz_rpy)}")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``check`` (bag -> report) or ``verify`` (report vs robot manifest)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--robot", type=Path, required=True, help="robots/<id>/robot.yaml")
        p.add_argument("--sensor", default="head_zed")
    c = sub.choices["check"]
    c.add_argument("--bag", type=Path, required=True)
    c.add_argument("--cloud-topic", required=True, help="the depth driver's PointCloud2")
    c.add_argument("--table-z", type=float, required=True, help="table top, base frame (m)")
    c.add_argument(
        "--table-roi", type=float, nargs=4, required=True, metavar=("XMIN", "XMAX", "YMIN", "YMAX")
    )
    c.add_argument("--marker", type=float, nargs=2, action="append", default=[], metavar=("X", "Y"))
    c.add_argument("--marker-radius", type=float, default=0.08)
    c.add_argument("--min-marker-height", type=float, default=0.01)
    c.add_argument("--max-clouds", type=int, default=10)
    c.add_argument("--stride", type=int, default=4, help="keep every Nth point per cloud")
    c.add_argument("--max-tilt-deg", type=float, default=MAX_TILT_DEG)
    c.add_argument("--max-height-err-m", type=float, default=MAX_HEIGHT_ERR_M)
    c.add_argument("--max-marker-err-m", type=float, default=MAX_MARKER_ERR_M)
    c.add_argument("--out", type=Path, default=None)
    sub.choices["verify"].add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return check(args) if args.cmd == "check" else verify(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
