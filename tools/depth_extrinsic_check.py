"""Measure a depth camera's extrinsic by fitting it to the robot it looks at.

The safety kernel's world-voxel check trusts ``parent_frame -> frame_id`` for the depth
camera absolutely: every obstacle it will ever stop on is placed through that transform.
Intrinsics do not enter (the depth driver builds the cloud from its own calibration); the
EXTRINSIC does, and ``openral calibrate camera`` does not measure it. This tool does, from
the robot itself (``tools/_extrinsic_fit.py``): no markers, no tape measure. Any robot with
a depth camera in its manifest and an MJCF or URDF model.

* ``plan``   Offline, no robot: validate ``robots/<id>/calibration/extrinsic_fit_poses.yaml``
  (limits, self-contact along the ramps between poses, how low the robot reaches) and
  simulate a capture through the camera's field of view to show every axis is observable.
* ``check``  Fit the unit's mount to a capture from ``tools/depth_extrinsic_capture.py``
  (depth points at each pose + the REAL joint readings) and write the report.
* ``verify`` The gate ``openral deploy run`` applies: the committed report passed, at no
  looser criteria, for the unit's CURRENT mount.

The report's residuals (``openral_core.depth_extrinsic.residual_failures``): the manifest
mount's error against the fit plus the held-out spread (refit with each pose left out)
must stay within limits derived from the world-voxel margin; the median distance of the
robot's returns from its meshes must be small; and every axis must be observable
(restarted a full limit off, the fit must come back). ``suggested_static_transform_xyz_rpy``
is the fitted mount: copy it into the unit overlay (or the manifest, for a robot without
units) and run ``check`` again on the same capture; it now passes only if the fit is
consistent across poses. The committed report lives in
``robots/<id>/calibration/<unit>/<sensor>_extrinsic.json``.

Run::

    uv run python tools/depth_extrinsic_check.py plan \\
        --robot robots/openarm/robot.yaml --sensor head_zed --unit thor
    uv run python tools/depth_extrinsic_check.py check \\
        --robot robots/openarm/robot.yaml --sensor head_zed --unit thor \\
        --capture ~/extrinsic_capture_thor \\
        --out robots/openarm/calibration/thor/head_zed_extrinsic.json
    uv run python tools/depth_extrinsic_check.py verify \\
        --robot robots/openarm/robot.yaml --sensor head_zed --unit thor
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from openral_core import RobotDescription, apply_sensor_overlays, load_robot_unit
from openral_core.depth_extrinsic import (
    checkable_depth_sensor,
    extrinsic_criteria,
    extrinsic_report_path,
    fit_poses_path,
    residual_failures,
    verify_extrinsic_report,
)
from openral_core.exceptions import ROSConfigError

# Run as `python tools/depth_extrinsic_check.py`, sys.path[0] is `tools/`; the repo root
# goes on the path for the `tools.` package form, which is also what mypy resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools._extrinsic_fit import (  # noqa: E402
    FitPoseFile,
    PoseCapture,
    RobotSurface,
    fit_mount,
    matrix_to_xyzrpy,
    read_capture,
    simulate_capture,
    xyzrpy_to_matrix,
)

#: ``plan`` samples each ramp between consecutive poses at this many points.
_RAMP_SAMPLES = 21
#: A mesh interpenetration deeper than this along a ramp is reported as self-contact.
_CONTACT_DEPTH_M = 0.002
#: ``plan`` simulates a capture with the mount displaced by this much from the manifest,
#: so the fit has something to find: 10 mm in each axis, 0.5 deg about each.
_PLAN_OFFSET_XYZRPY = (0.01, 0.01, 0.01, math.radians(0.5), math.radians(0.5), math.radians(0.5))


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


def _poses(
    args: argparse.Namespace, description: RobotDescription
) -> list[tuple[str, dict[str, float]]]:
    path = args.poses or fit_poses_path(args.robot)
    if not path.is_file():
        raise ROSConfigError(f"no extrinsic-fit poses at {path}; write them and run `plan`")
    return FitPoseFile.load(path).targets(args.robot.parent.name, description)


def plan(args: argparse.Namespace) -> int:
    """Validate the pose file offline; 0 iff every pose and ramp is clear and the fit sees all axes."""
    description = _unit_description(args.robot, args.sensor, args.unit)
    spec = checkable_depth_sensor(description, args.sensor)
    assert spec.static_transform_xyz_rpy is not None
    targets = _poses(args, description)
    robot = RobotSurface(description, args.robot.parent)
    problems = []
    names = [j.name for j in description.joints]
    rest = {n: 0.0 for n in names}
    path = [("rest", rest), *targets, ("rest", rest)]
    for (a, qa), (b, qb) in itertools.pairwise(path):
        depth = robot.self_contact_depth(qa, qb, _RAMP_SAMPLES)
        if depth is not None and depth > _CONTACT_DEPTH_M:
            problems.append(f"ramp {a} -> {b}: robot meshes interpenetrate {depth * 1000:.1f} mm")
    for name, q in targets:
        robot.set(q)
        surf, _ = robot.surface()
        print(
            f"  {name:24s} lowest robot point z={float(surf[:, 2].min()):+.3f} m in {description.base_frame}"
        )
    true_mount = xyzrpy_to_matrix(spec.static_transform_xyz_rpy) @ xyzrpy_to_matrix(
        _PLAN_OFFSET_XYZRPY
    )
    rng = np.random.default_rng(0)
    captures = [
        PoseCapture(name, simulate_capture(robot, spec, true_mount, q, rng=rng, noise_m=0.004), q)
        for name, q in targets
    ]
    for c in captures:
        print(f"  {c.name:24s} {len(c.points)} simulated points in view")
    _, residuals = fit_mount(robot, spec, captures)
    unseen = [f for f in residual_failures(residuals) if "mount error" not in f]
    problems += [f"simulated fit: {f}" for f in unseen]
    print(
        json.dumps(
            {
                "axis_unrecovered": residuals["axis_unrecovered"],
                "heldout_spread": residuals["heldout_spread"],
            },
            indent=2,
        )
    )
    for p in problems:
        print(f"PLAN: {p}", file=sys.stderr)
    print(
        "PLAN OK: the poses are clear of self-contact and the fit sees every axis. Check the "
        "lowest points against this cell's table and fixtures before recording."
        if not problems
        else "PLAN FAILED",
        file=sys.stderr,
    )
    return 1 if problems else 0


def check(args: argparse.Namespace) -> int:
    """Fit the unit's mount to the capture and write the JSON report; 0 iff it passes."""
    description = _unit_description(args.robot, args.sensor, args.unit)
    spec = checkable_depth_sensor(description, args.sensor)
    assert spec.parent_frame is not None and spec.static_transform_xyz_rpy is not None
    meta, captures = read_capture(args.capture)
    if (meta.get("sensor"), meta.get("frame_id")) != (spec.name, spec.frame_id):
        raise ROSConfigError(
            f"capture is of ({meta.get('sensor')}, {meta.get('frame_id')}), not "
            f"({spec.name}, {spec.frame_id})"
        )
    robot = RobotSurface(description, args.robot.parent)
    mount, residuals = fit_mount(
        robot, spec, captures, progress=lambda s: print(f"  {s}", file=sys.stderr)
    )
    failures = residual_failures(residuals)
    report = {
        "method": "robot_fit",
        "unit": args.unit,
        "sensor": spec.name,
        "parent_frame": spec.parent_frame,
        "frame_id": spec.frame_id,
        "base_frame": description.base_frame,
        "static_transform_xyz_rpy": list(spec.static_transform_xyz_rpy),
        "capture": str(args.capture),
        "capture_meta": meta,
        "criteria": extrinsic_criteria(),
        "residuals": residuals,
        "passed": not failures,
        "failures": failures,
        "suggested_static_transform_xyz_rpy": [round(v, 6) for v in matrix_to_xyzrpy(mount)],
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``plan`` (poses, offline), ``check`` (capture -> report) or ``verify``."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "check", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--robot", type=Path, required=True, help="robots/<id>/robot.yaml")
        p.add_argument("--sensor", required=True, help="the manifest's depth camera name")
        p.add_argument(
            "--unit", default=None, help="robots/<id>/units/<unit>.yaml overlay (this cell)"
        )
    sub.choices["plan"].add_argument(
        "--poses",
        type=Path,
        default=None,
        help="default: <robot dir>/calibration/extrinsic_fit_poses.yaml",
    )
    c = sub.choices["check"]
    c.add_argument(
        "--capture", type=Path, required=True, help="tools/depth_extrinsic_capture.py output"
    )
    c.add_argument("--out", type=Path, default=None)
    sub.choices["verify"].add_argument(
        "--report",
        type=Path,
        default=None,
        help="default: <robot dir>/calibration/<unit>/<sensor>_extrinsic.json",
    )
    args = parser.parse_args(argv)
    try:
        return {"plan": plan, "check": check, "verify": verify}[args.cmd](args)
    except (ValueError, ROSConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
