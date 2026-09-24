# SPDX-License-Identifier: Apache-2.0
"""Measure joint-state freshness as the real ros2_control HAL sees it, with no robot.

Backs the `safety.joint_state_staleness_limit_s` a real manifest declares.
That limit is a safety control on the actuation path: `RosControlHAL.read_state`
raises `ROSPerceptionStale` when the newest `/joint_states` is older than it, so
a policy cannot act on a dead robot's last known pose. Too loose and a dead bus
goes unnoticed for many ticks; too tight and ordinary DDS + executor jitter
aborts a healthy run. The number therefore has to come from a measurement on
the deploy host, not from a schema default.

What it does: builds the robot's real ros2_control HAL from its manifest
(`--robot`, via `build_hal(mode="real")`, so it reads at the manifest's window),
publishes `sensor_msgs/JointState` at `--rate` (the controller manager's joint
state rate, e.g. OpenArm 750 Hz) with that HAL's joint names on its joint-state
topic, over real DDS on this host, into `RosControlTransport` — its production
QoS and callback — and records

* callback inter-arrival gaps (what a hiccuping or dead bus stretches),
* header-stamp → callback latency (DDS + executor delay),
* the read-side age `now - last_arrival()` sampled at the runner's tick rate
  (default: the manifest's `action_spec.control_freq_hz`),
  plus whether `hal.read_state()` ever raised at the declared limit.

`--load` adds a pure-Python busy thread in the same process to contend for the
GIL, the way the HAL node's other callbacks do. It is harsher than production
(it also starves this probe's own publisher thread), which is the point.

Thor, 2026-09-23 (Fast-DDS, jazzy, 14 cores, another user's GPU policy server
resident), 60 s each:

    idle  gap p50 1.33 / p99 1.60 / p99.9 2.64 / max 4.56 ms; latency max 2.6 ms;
          read-side age max 32 ms; 0 gaps > 10 ms
    load  gap p50 5.9 / p99 10.3 / max 24.1 ms; latency max 42.8 ms;
          read-side age max 24 ms; 110 gaps > 10 ms, 0 gaps > 33 ms

From which `robots/openarm/robot.yaml` declares `joint_state_staleness_limit_s: 0.1`: three
30 Hz control periods, more than twice the worst observed latency, and a dead
bus is caught within three ticks instead of fifteen.

Run on the deploy host with a sourced ROS 2 overlay:

    uv run python tools/joint_state_staleness_probe.py --robot robots/openarm/robot.yaml \
        --rate 750 --duration 60
    uv run python tools/joint_state_staleness_probe.py --robot robots/ur5e/robot.yaml \
        --rate 500 --duration 60 --load

Only ros2_control robots (`RosControlHAL` subclasses) are supported; a serial or
vendor-SDK HAL (SO-100, ALOHA, Galaxea) is refused with a clear error.

Publisher and subscriber share one process and one executor on purpose: the
measured term is the HAL node's own callback path, not inter-process transport.
Uses `ROS_DOMAIN_ID` 77 unless one is already set, so it never touches a live
graph.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
from typing import Any


def pct(xs: list[float], p: float) -> float:
    """The p-th percentile of *xs* (nearest rank); NaN when empty."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, round(p / 100.0 * (len(s) - 1)))]


def build_probe_hal(robot_yaml: str) -> Any:  # reason: returns a RosControlHAL; import deferred
    """The robot's real ros2_control HAL, built from its manifest like a deploy does.

    Raises:
        SystemExit: The manifest's real HAL is not a ``RosControlHAL``.

    Example:
        >>> hal = build_probe_hal("robots/ur5e/robot.yaml")
        >>> hal.joint_state_topic, hal.description.safety.joint_state_staleness_limit_s
        ('/joint_states', 0.5)
    """
    from openral_core.schemas import RobotDescription
    from openral_hal import build_hal
    from openral_hal.ros_control import RosControlHAL

    desc = RobotDescription.from_yaml(robot_yaml)
    # require_can_links only exists on the OpenArm HAL (build_hal drops it
    # elsewhere): the probe needs no hardware links.
    hal = build_hal(desc, mode="real", transport={"require_can_links": False})
    if not isinstance(hal, RosControlHAL):
        raise SystemExit(
            f"{desc.name}: real HAL {type(hal).__name__} is not a ros2_control HAL; "
            "this probe measures RosControlTransport only."
        )
    return hal


def run(
    robot_yaml: str, duration_s: float, rate_hz: float, read_hz: float | None, load: bool
) -> int:
    """Drive the probe and print the freshness statistics; returns a process exit code."""
    import rclpy
    from openral_core.exceptions import ROSPerceptionStale
    from openral_hal.ros_control_transport import RosControlTransport
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.subscription import Subscription
    from rclpy.utilities import get_rmw_implementation_identifier
    from sensor_msgs.msg import JointState

    ctx = rclpy.Context()
    ctx.init()
    pub_node = Node("js_source", context=ctx)
    sub_node = Node("hal_side", context=ctx)
    hal = build_probe_hal(robot_yaml)
    # RosControlHAL refuses a manifest without a positive control rate.
    tick_hz = read_hz if read_hz is not None else float(hal.description.control_rate_hz or 0.0)
    topic = hal.joint_state_topic
    names = hal.ros2_control_joint_names()
    tr = RosControlTransport(
        sub_node,
        command_topics=hal.command_topics(),
        joint_names=names,
        joint_state_topic=topic,
        command_kinds=hal.command_bindings(),
    )
    hal.attach_transport(tr.publish, tr.state, tr.last_arrival)
    hal.connect()

    sub = next(v for v in vars(tr).values() if isinstance(v, Subscription))
    original = sub.callback
    gaps: list[float] = []
    lat: list[float] = []
    last = [0.0]

    def instrumented(msg: Any) -> None:  # reason: rosidl message classes are untyped
        now = time.monotonic()
        if last[0]:
            gaps.append(now - last[0])
        last[0] = now
        lat.append(time.time() - (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9))
        original(msg)

    sub.callback = instrumented

    pub = pub_node.create_publisher(
        JointState,
        topic,
        QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10
        ),
    )
    stop = threading.Event()
    published = [0]

    def source() -> None:
        period = 1.0 / rate_hz
        nxt = time.perf_counter()
        while not stop.is_set():
            m = JointState()
            m.header.stamp = pub_node.get_clock().now().to_msg()
            m.name = names
            m.position = [0.0] * len(names)
            m.velocity = [0.0] * len(names)
            m.effort = [0.0] * len(names)
            pub.publish(m)
            published[0] += 1
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()

    ages: list[float] = []
    stale_raises = [0]

    def reader() -> None:
        period = 1.0 / tick_hz
        nxt = time.perf_counter() + 1.0
        while not stop.is_set():
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            nxt += period
            arrival = tr.last_arrival()
            if arrival:
                ages.append(time.monotonic() - arrival)
            try:
                hal.read_state()
            except ROSPerceptionStale:
                stale_raises[0] += 1

    def gil_hog() -> None:
        while not stop.is_set():
            sum(range(20000))

    threads = [
        threading.Thread(target=source, daemon=True),
        threading.Thread(target=reader, daemon=True),
    ]
    if load:
        threads.append(threading.Thread(target=gil_hog, daemon=True))
    for t in threads:
        t.start()

    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(sub_node)
    executor.add_node(pub_node)
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        executor.spin_once(timeout_sec=0.01)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)

    received = len(gaps) + 1
    print(
        f"robot={hal.description.name} topic={topic} "
        f"limit_s={hal.description.safety.joint_state_staleness_limit_s} "
        f"rmw={get_rmw_implementation_identifier()} load={'gil-hog' if load else 'idle'} "
        f"duration_s={duration_s:.0f} rate_hz={rate_hz:.0f}"
    )
    print(
        f"published={published[0]} received={received} "
        f"achieved_rx_hz={received / duration_s:.0f} dropped={published[0] - received}"
    )
    if not gaps or not ages:
        print("no samples — is a ROS 2 overlay sourced and the domain free?")
        return 1
    print(
        f"callback gap ms: p50={pct(gaps, 50) * 1e3:.3f} p99={pct(gaps, 99) * 1e3:.3f} p99.9={pct(gaps, 99.9) * 1e3:.3f} max={max(gaps) * 1e3:.3f} mean={statistics.mean(gaps) * 1e3:.3f}"
    )
    print(
        f"stamp->callback latency ms: p50={pct(lat, 50) * 1e3:.3f} p99={pct(lat, 99) * 1e3:.3f} max={max(lat) * 1e3:.3f}"
    )
    print(
        f"read-side age at {tick_hz:.0f} Hz ms: p50={pct(ages, 50) * 1e3:.3f} "
        f"p99={pct(ages, 99) * 1e3:.3f} max={max(ages) * 1e3:.3f} "
        f"samples={len(ages)} stale_raises={stale_raises[0]}"
    )
    print(f"gaps>10ms={sum(g > 0.010 for g in gaps)} gaps>33ms={sum(g > 0.033 for g in gaps)}")
    hal.disconnect()
    tr.close()
    pub_node.destroy_node()
    sub_node.destroy_node()
    ctx.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--robot", required=True, help="robots/<id>/robot.yaml (ros2_control)")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds to run")
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="JointState Hz the robot's controller manager publishes (OpenArm 750)",
    )
    parser.add_argument(
        "--read-hz", type=float, default=None, help="reader tick Hz (default: manifest rate)"
    )
    parser.add_argument("--load", action="store_true", help="add a GIL-contending thread")
    args = parser.parse_args(argv)
    os.environ.setdefault("ROS_DOMAIN_ID", "77")
    return run(args.robot, args.duration, args.rate, args.read_hz, args.load)


if __name__ == "__main__":
    sys.exit(main())
