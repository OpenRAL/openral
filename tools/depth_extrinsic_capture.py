"""Record the robot for the extrinsic fit: move it through the committed poses, capture depth.

MOVES THE ROBOT. Drives it through ``robots/<id>/calibration/extrinsic_fit_poses.yaml``
and, at each pose, records depth points from ``--cloud-topic`` (re-expressed in the
camera's mount frame through the driver's own TF) together with the REAL joint readings.
``tools/depth_extrinsic_check.py check --capture <out>`` then fits the camera mount to the
robot's own meshes. Robot-agnostic: joints, limits, speeds and the camera all come from
the manifest (with ``--unit``'s overlay applied).

Every waypoint goes through the safety kernel, never to a controller directly: the tool
publishes on ``/openral/candidate_action`` with the same ``ROSPublishingHAL`` and the same
bounded ramp (``starting_pose_ramp_steps`` at the manifest's
``starting_pose_max_joint_speed_*``) the rskill runner uses to reach a skill's starting
pose, and waits for the HAL's ``/openral/action_applied`` before the next. The kernel
checks limits, velocity and self-collision on every waypoint and drops anything unsafe.
Its world-voxel check is off (that is what this calibrates), so the operator is the
guard against the environment: clear the space in front of the robot first.

Run it with the robot's deploy graph up and NO goal active (the runner idle), attended,
hand on the hardware E-stop. Refuses unless both gates are exported and stdin is a
terminal, and asks before EVERY move:

    OPENRAL_EXTRINSIC_CAPTURE_ALLOW_MOTION=1 OPENRAL_EXTRINSIC_CAPTURE_ATTENDED=1 \\
    uv run python tools/depth_extrinsic_capture.py \\
        --robot robots/openarm/robot.yaml --sensor head_zed --unit thor \\
        --cloud-topic /zed/zed_node/point_cloud/cloud_registered \\
        --joint-states-topic /openral_hal_openarm/joint_states \\
        --out ~/extrinsic_capture_thor

A deadman note: the watchdog arms on a goal, and this tool sends none, so an abort here is
the operator's E-stop, the kernel's checks, or the tool's own stop on a latched
``/openral/safety_status`` or ``/openral/estop``. A crash mid-ramp leaves the robot holding
the last applied waypoint (position control), never moving on its own.
"""

from __future__ import annotations

import argparse
import math
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from openral_core.depth_extrinsic import checkable_depth_sensor, fit_poses_path
from openral_core.exceptions import ROSConfigError, ROSEStopRequested

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools._extrinsic_fit import FitPoseFile, PoseCapture, write_capture  # noqa: E402
from tools.depth_extrinsic_check import _unit_description  # noqa: E402

GATES = ("OPENRAL_EXTRINSIC_CAPTURE_ALLOW_MOTION", "OPENRAL_EXTRINSIC_CAPTURE_ATTENDED")
#: A pose counts as held when every joint stayed within this over the recording window.
_STILL_RAD = 2e-3


def refuse_reason(env: dict[str, str], interactive: bool) -> str | None:
    """Why this tool must not move the robot, or ``None`` when every gate is open.

    Example:
        >>> refuse_reason({}, True)
        'export OPENRAL_EXTRINSIC_CAPTURE_ALLOW_MOTION=1 and OPENRAL_EXTRINSIC_CAPTURE_ATTENDED=1'
        >>> refuse_reason(dict.fromkeys(GATES, "1"), False)
        'stdin is not a terminal: every move needs a typed confirmation'
    """
    if not all(env.get(g) == "1" for g in GATES):
        return f"export {GATES[0]}=1 and {GATES[1]}=1"
    if not interactive:
        return "stdin is not a terminal: every move needs a typed confirmation"
    return None


def ramp(current: list[float], target: list[float], steps: int) -> list[list[float]]:
    """The runner's linear joint-space ramp: ``steps`` waypoints ending exactly at ``target``.

    Example:
        >>> ramp([0.0, 1.0], [1.0, 1.0], 2)
        [[0.5, 1.0], [1.0, 1.0]]
    """
    return [
        [a + (b - a) * (i / steps) for a, b in zip(current, target, strict=True)]
        for i in range(1, steps + 1)
    ]


def main(argv: list[str] | None = None) -> int:  # reason: one linear, attended procedure
    """Parse, gate, then move and record; 0 when every pose was captured."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--unit", default=None)
    parser.add_argument("--poses", type=Path, default=None)
    parser.add_argument("--cloud-topic", required=True, help="the depth driver's PointCloud2")
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--clouds", type=int, default=15, help="clouds recorded per pose")
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--stride", type=int, default=4, help="keep every Nth point per cloud")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    reason = refuse_reason(dict(os.environ), sys.stdin.isatty())
    if reason:
        print(f"REFUSED: {reason}", file=sys.stderr)
        return 2
    description = _unit_description(args.robot, args.sensor, args.unit)
    spec = checkable_depth_sensor(description, args.sensor)
    targets = FitPoseFile.load(args.poses or fit_poses_path(args.robot)).targets(
        args.robot.parent.name, description
    )

    import rclpy
    from openral_core.schemas import Action, ControlMode
    from openral_msgs.msg import SafetyStatus
    from openral_rskill_ros.rskill_runner_node import (
        resolve_control_rate_source,
        starting_pose_joint_bounds,
        starting_pose_ramp_steps,
    )
    from openral_runner.ros_publishing_hal import ROSPublishingHAL
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py.point_cloud2 import read_points_numpy
    from std_msgs.msg import Empty
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = rclpy.create_node("depth_extrinsic_capture")
    abort: list[str] = []
    node.create_subscription(Empty, "/openral/estop", lambda _m: abort.append("/openral/estop"), 10)
    node.create_subscription(
        SafetyStatus,
        "/openral/safety_status",
        lambda m: abort.append(f"safety_status latched: {m.detail}") if m.latched else None,
        10,
    )
    clouds: list[Any] = []
    node.create_subscription(PointCloud2, args.cloud_topic, clouds.append, qos_profile_sensor_data)
    tf = Buffer()
    TransformListener(tf, node)
    tick = [0]
    session = secrets.randbits(64) or 1
    hal = ROSPublishingHAL(
        node=node,  # a plain Node has every API the adapter uses
        description=description,
        skill_id_getter=lambda: "tools/depth_extrinsic_capture",
        tick_index_getter=lambda: tick[0],
        runner_session_id=session,
        joint_state_topic=args.joint_states_topic,
        safety_abort_getter=lambda: abort[0] if abort else None,
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    hal.connect()
    names = [j.name for j in description.joints]
    bounds = starting_pose_joint_bounds(0.0, 0.0, description)
    rate_hz, _ = resolve_control_rate_source(0.0, description)

    def joints_now() -> list[float]:
        state = hal.read_state()
        by_name = dict(zip(state.name, state.position, strict=True))
        return [float(by_name[n]) for n in names]

    def move_to(target: list[float]) -> None:
        start = joints_now()
        steps = starting_pose_ramp_steps(start, target, bounds, rate_hz)
        for waypoint in ramp(start, target, steps):
            if abort:
                raise ROSEStopRequested(f"capture aborted: {abort[0]}")
            tick[0] += 1
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[waypoint],
                    joint_names=names,
                    stamp_ns=time.time_ns(),
                )
            )
            time.sleep(1.0 / rate_hz)

    print(
        f"session {session:#018x}; {len(targets)} poses; kernel-checked ramps at the manifest's "
        "starting-pose speed. Keep a hand on the hardware E-stop."
    )
    home = joints_now()
    captures = []
    try:
        for name, q in targets:
            target = [q[n] for n in names]
            if (
                input(f"Move to pose {name!r}? Space clear, hand on E-stop - type 'go': ").strip()
                != "go"
            ):
                print("stopped by the operator")
                break
            move_to(target)
            time.sleep(args.settle_s)
            reached = joints_now()
            off = max(abs(a - b) for a, b in zip(reached, target, strict=True))
            if off > max(tol for _, tol in bounds):
                raise ROSConfigError(f"pose {name!r} not reached: worst joint {off:.3f} off")
            clouds.clear()
            readings = []
            deadline = time.monotonic() + 30.0
            while len(clouds) < args.clouds and time.monotonic() < deadline:
                readings.append(joints_now())
                time.sleep(0.05)
            if len(clouds) < args.clouds:
                raise ROSConfigError(f"only {len(clouds)} clouds on {args.cloud_topic} in 30 s")
            spread = float(np.ptp(np.asarray(readings), axis=0).max())
            if spread > _STILL_RAD:
                raise ROSConfigError(
                    f"pose {name!r}: joints moved {spread:.4f} rad while recording"
                )
            batch = clouds[: args.clouds]
            t = tf.lookup_transform(spec.frame_id, batch[0].header.frame_id, Time()).transform
            r, p = t.rotation, t.translation
            rot = np.array(
                [
                    [
                        1 - 2 * (r.y**2 + r.z**2),
                        2 * (r.x * r.y - r.z * r.w),
                        2 * (r.x * r.z + r.y * r.w),
                    ],
                    [
                        2 * (r.x * r.y + r.z * r.w),
                        1 - 2 * (r.x**2 + r.z**2),
                        2 * (r.y * r.z - r.x * r.w),
                    ],
                    [
                        2 * (r.x * r.z - r.y * r.w),
                        2 * (r.y * r.z + r.x * r.w),
                        1 - 2 * (r.x**2 + r.y**2),
                    ],
                ]
            )
            pts = np.concatenate(
                [
                    read_points_numpy(c, field_names=("x", "y", "z"), skip_nans=True)[
                        :: args.stride
                    ]
                    for c in batch
                ]
            ).astype(np.float64)
            mean = np.asarray(readings).mean(axis=0)
            captures.append(
                PoseCapture(
                    name,
                    pts @ rot.T + np.array([p.x, p.y, p.z]),
                    dict(zip(names, (float(v) for v in mean), strict=True)),
                )
            )
            print(f"  {name}: {len(pts)} points, worst joint {off:.4f} rad off target")
    finally:
        if (
            captures
            and not abort
            and input("Return to where the robot started? type 'go': ").strip() == "go"
        ):
            move_to(home)
        write_capture(
            args.out,
            {
                "sensor": spec.name,
                "frame_id": spec.frame_id,
                "unit": args.unit,
                "robot": args.robot.parent.name,
                "cloud_topic": args.cloud_topic,
                "joint_states_topic": args.joint_states_topic,
                "session": f"{session:#018x}",
                "recorded_unix_s": math.floor(time.time()),
            },
            captures,
        )
        hal.disconnect()
        executor.shutdown()
        rclpy.shutdown()
    print(f"wrote {len(captures)} pose(s) to {args.out}")
    return 0 if len(captures) == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
