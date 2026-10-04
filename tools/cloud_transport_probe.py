"""Does a large best-effort ``PointCloud2`` arrive? Delivery ratio and latency, measured.

The deploy graph's depth clouds are megabytes (an Isaac 512x384 depth camera
back-projects to ~2.4-2.8 MB of xyz float32 per frame) on ``BEST_EFFORT`` topics.
Under Jazzy's default Fast DDS a best-effort sample larger than the 512 KiB
shared-memory segment of its publisher's participant is lost almost every time
(``docs/reference/dds-large-messages.md``). This probe measures exactly that, over
real DDS and real ``sensor_msgs/PointCloud2``, so a transport change is judged on
numbers rather than on reasoning.

Two roles, run as separate processes (the transport under test is the one
between processes; Fast DDS reads its XML profile once per process)::

    # terminal 1
    python tools/cloud_transport_probe.py sub --topic /probe/cloud --count 60
    # terminal 2
    python tools/cloud_transport_probe.py pub --topic /probe/cloud --count 60 --hz 4

Each prints one JSON line. ``pub`` reports the ``publish()`` call cost (a
producer must never block on a cloud); ``sub`` reports how many distinct clouds
arrived and their publish-to-receive latency. Put a node between them (the
robot self-filter: ``--joint`` names make ``pub`` also stream the joint states it
needs) to measure a chain. ``tests/integration/test_large_cloud_transport_live.py``
runs this through the real ``robot_self_filter``.

Needs a sourced ROS 2 overlay. The result is RMW- and profile-specific; the
``pub`` line names both.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState, PointCloud2, PointField


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def _cloud(n_points: int, frame: str) -> PointCloud2:
    """``n_points`` xyz float32 in a 1 m box 1.5 m ahead of ``frame`` (clear of any arm)."""
    rng = np.random.default_rng(0)
    pts = rng.uniform((1.5, -0.5, 0.0), (2.5, 0.5, 1.0), size=(n_points, 3)).astype("<f4")
    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.height = 1
    msg.width = n_points
    msg.fields = [
        PointField(name=axis, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, axis in enumerate("xyz")
    ]
    msg.point_step = 12
    msg.row_step = 12 * n_points
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


def _wait_matched(node: Node, pub: Any, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


def run_pub(args: argparse.Namespace) -> dict[str, Any]:
    """Publish ``count`` clouds at ``hz`` (and joint states between them, if named)."""
    node = Node("cloud_transport_probe_pub")
    reliability = (
        QoSReliabilityPolicy.RELIABLE if args.reliable else QoSReliabilityPolicy.BEST_EFFORT
    )
    # The sim sensor bridge's depth-cloud profile (sim_sensor_bridge.py, depth_qos).
    qos = QoSProfile(reliability=reliability, durability=QoSDurabilityPolicy.VOLATILE, depth=5)
    pub = node.create_publisher(PointCloud2, args.topic, qos)
    joint_pub = node.create_publisher(JointState, args.joint_topic, 50) if args.joint else None
    msg = _cloud(args.points, args.frame)
    joints = JointState(name=list(args.joint), position=[0.0] * len(args.joint))

    def publish_joints() -> None:
        if joint_pub is not None:
            joints.header.stamp = node.get_clock().now().to_msg()
            joint_pub.publish(joints)

    _wait_matched(node, pub, args.discovery_s)
    # A chain node needs joint states that bracket the first cloud.
    warm_until = time.monotonic() + (1.0 if joint_pub is not None else 0.2)
    while time.monotonic() < warm_until:
        publish_joints()
        time.sleep(0.01)
    period = 1.0 / args.hz
    calls_ms: list[float] = []
    next_t = time.monotonic()
    for _ in range(args.count):
        stamp = node.get_clock().now()
        msg.header.stamp = stamp.to_msg()
        t0 = time.monotonic()
        pub.publish(msg)
        calls_ms.append((time.monotonic() - t0) * 1e3)
        next_t += period
        while time.monotonic() < next_t:  # joint states at ~100 Hz between clouds
            publish_joints()
            time.sleep(min(0.01, max(0.0, next_t - time.monotonic())))
    for _ in range(100):  # let a chain node finish its last cloud
        publish_joints()
        time.sleep(0.01)
    node.destroy_node()
    return {
        "role": "pub",
        "rmw": os.environ.get("RMW_IMPLEMENTATION", "(jazzy default: Fast-DDS)"),
        "fastdds_profile": os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", ""),
        "sent": args.count,
        "bytes": len(msg.data),
        "hz": args.hz,
        "publish_ms_p50": _percentile(calls_ms, 50),
        "publish_ms_p99": _percentile(calls_ms, 99),
        "publish_ms_max": max(calls_ms),
    }


def run_sub(args: argparse.Namespace) -> dict[str, Any]:
    """Count distinct clouds (by capture stamp) until ``count`` arrive or ``timeout_s``."""
    node = Node("cloud_transport_probe_sub")
    reliability = (
        QoSReliabilityPolicy.RELIABLE if args.reliable else QoSReliabilityPolicy.BEST_EFFORT
    )
    qos = QoSProfile(
        reliability=reliability, durability=QoSDurabilityPolicy.VOLATILE, depth=args.depth
    )
    stamps: set[int] = set()
    latency_ms: list[float] = []
    sizes: list[int] = []

    def on_cloud(msg: PointCloud2) -> None:
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        stamps.add(stamp_ns)
        latency_ms.append((node.get_clock().now().nanoseconds - stamp_ns) / 1e6)
        sizes.append(len(msg.data))

    node.create_subscription(PointCloud2, args.topic, on_cloud, qos)
    print(json.dumps({"role": "sub", "ready": True}), flush=True)
    deadline = time.monotonic() + args.timeout_s
    while len(stamps) < args.count and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    return {
        "role": "sub",
        "received": len(stamps),
        "expected": args.count,
        "ratio": len(stamps) / args.count,
        "bytes_max": max(sizes, default=0),
        "latency_ms_p50": _percentile(latency_ms, 50),
        "latency_ms_p95": _percentile(latency_ms, 95),
        "latency_ms_max": max(latency_ms, default=float("nan")),
    }


def main() -> None:
    """CLI entry: ``pub`` or ``sub``, one JSON result line on stdout."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("role", choices=("pub", "sub"))
    parser.add_argument("--topic", default="/probe/cloud")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--reliable", action="store_true", help="RELIABLE instead of BEST_EFFORT")
    parser.add_argument("--points", type=int, default=233_000, help="pub: ~2.8 MB of xyz")
    parser.add_argument("--hz", type=float, default=4.0, help="pub: cloud rate")
    parser.add_argument("--frame", default="base_link", help="pub: the cloud's frame_id")
    parser.add_argument("--joint", nargs="*", default=[], help="pub: also stream these joints")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--discovery-s", type=float, default=10.0)
    parser.add_argument("--depth", type=int, default=1, help="sub: KEEP_LAST depth")
    parser.add_argument("--timeout-s", type=float, default=60.0, help="sub: give up after")
    args = parser.parse_args()
    rclpy.init()
    try:
        result = run_pub(args) if args.role == "pub" else run_sub(args)
    finally:
        rclpy.try_shutdown()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
