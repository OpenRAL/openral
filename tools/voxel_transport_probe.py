"""What a finer world-voxel grid costs on the wire, at each candidate resolution.

The third and last cost term on `PLAN.md`'s 25 → 15 mm lever. The other two are
measured and both cheap — the kernel *consuming* a 15 mm grid is p99 0.825 ms,
and `openral_octomap_bridge` *producing* one is 1.60 ms against a 100 ms publish
period. Neither touches the term that actually grows: `OccupancyVoxels.occupancy`
is a **dense** `uint8[]`, so 25 → 15 mm takes one message from 0.61 MB to
2.80 MB, ten times a second.

This measures that directly rather than reasoning about it, because on this lever
every number reasoned about has been wrong — the original strike by 32×, and the
un-strike's cell count by 7×.

**Method.** Two processes over real DDS, real `openral_msgs`, publishing at the
deployed 10 Hz onto `/openral/world_voxels` under the kernel's own QoS for that
topic — `RELIABLE`, `KEEP_LAST(1)`, `VOLATILE`, read from
`lifecycle_kernel.cpp`. Grid sizes come from the shipped coverage radius
(`deploy_e2e.launch.py::_octomap_coverage_radius`, 1.05 m), so the cell counts are
the deployed ones. Reports publish-call cost, publish→receive latency, delivery
count, and the rate actually achieved.

**What the latency means.** It is map **staleness**: how old the world is when
the kernel reads it. That is the number the lever trades against, because a finer
grid buys static precision and pays in age, and at any nonzero arm speed age is
also millimetres.

Run::

    uv run python tools/voxel_transport_probe.py sweep
    uv run python tools/voxel_transport_probe.py sweep --resolutions 0.025 0.015

Needs a sourced ROS 2 overlay. Reports the RMW it measured — the result is
transport-specific and does not carry across implementations.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

import rclpy
from openral_msgs.msg import OccupancyVoxels
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

RADIUS_M = 1.05  # deploy_e2e.launch.py::_octomap_coverage_radius
TARGET_HZ = 10.0
WARMUP = 5


def qos() -> QoSProfile:
    q = QoSProfile(depth=1)
    q.reliability = QoSReliabilityPolicy.RELIABLE
    q.durability = QoSDurabilityPolicy.VOLATILE
    q.history = QoSHistoryPolicy.KEEP_LAST
    return q


def per_axis(res: float) -> int:
    return int(2.0 * RADIUS_M / res) + 1


def make_msg(res: float) -> tuple[Any, int]:
    n = per_axis(res)
    m = OccupancyVoxels()
    m.resolution = res
    m.size_x = m.size_y = m.size_z = n
    m.occupancy = bytes(n * n * n)
    return m, n


def run_pub(res: float, count: int) -> None:
    rclpy.init(args=[])
    node = Node("voxel_probe_pub")
    pub = node.create_publisher(OccupancyVoxels, "/openral/world_voxels", qos())
    msg, n = make_msg(res)
    while pub.get_subscription_count() < 1:
        rclpy.spin_once(node, timeout_sec=0.05)
    time.sleep(0.5)
    period = 1.0 / TARGET_HZ
    sent: list[float] = []
    for i in range(count + WARMUP):
        t0 = time.perf_counter()
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        publish_ms = (time.perf_counter() - t0) * 1e3
        if i >= WARMUP:
            sent.append(publish_ms)
        slack = period - (time.perf_counter() - t0)
        if slack > 0:
            time.sleep(slack)
    print(json.dumps({"role": "pub", "res": res, "per_axis": n, "bytes": n**3, "publish_ms": sent}))
    node.destroy_node()
    rclpy.shutdown()


def run_sub(res: float, count: int) -> None:
    rclpy.init(args=[])
    node = Node("voxel_probe_sub")
    got: list[float] = []

    def cb(m: Any) -> None:
        now = node.get_clock().now().nanoseconds
        st = m.header.stamp
        got.append((now - (st.sec * 10**9 + st.nanosec)) / 1e6)

    node.create_subscription(OccupancyVoxels, "/openral/world_voxels", cb, qos())
    deadline = time.time() + 60.0
    started = None
    while time.time() < deadline and len(got) < count + WARMUP:
        rclpy.spin_once(node, timeout_sec=0.1)
        if got and started is None:
            started = time.time()
    elapsed = (time.time() - started) if started else 0.0
    lat = got[WARMUP:]
    print(
        json.dumps(
            {
                "role": "sub",
                "res": res,
                "received": len(got),
                "latency_ms": lat,
                "elapsed_s": elapsed,
            }
        )
    )
    node.destroy_node()
    rclpy.shutdown()


def run_sweep(resolutions: list[float], count: int) -> int:
    """Drive both roles per resolution and print the table.

    Separate PROCESSES, not threads: intra-process publish/subscribe short-
    circuits the transport this is measuring, which would report a cost that
    does not exist in the deployed graph (bridge and kernel are separate nodes).
    """
    import subprocess

    here = os.path.abspath(__file__)
    env = dict(os.environ)
    env.setdefault("ROS_DOMAIN_ID", "91")
    env.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    rows: list[tuple[float, Any, Any]] = []
    for res in resolutions:
        sub = subprocess.Popen(
            [sys.executable, here, "sub", "--res", str(res), "--count", str(count)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        time.sleep(3.0)  # let the subscription match before the first publish
        pub = subprocess.run(
            [sys.executable, here, "pub", "--res", str(res), "--count", str(count)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        sub_out, _ = sub.communicate(timeout=90)
        try:
            p, s = json.loads(pub.stdout.strip()), json.loads(sub_out.strip())
        except (json.JSONDecodeError, ValueError):
            print(f"  {res * 1e3:.1f} mm: no result (probe produced no JSON)")
            continue
        rows.append((res, p, s))

    print(f"  RMW: {os.environ.get('RMW_IMPLEMENTATION', '(jazzy default: Fast-DDS)')}")
    print(
        f"  {'res':>8} {'axis':>6} {'MB':>7} {'recv':>6} {'p50':>9} {'p99':>9} {'pub p99':>9} {'Hz':>6}"
    )
    for res, p, s in rows:
        lat = sorted(s["latency_ms"])
        pub_ms = sorted(p["publish_ms"])
        if not lat:
            print(f"  {res * 1e3:>7.1f}m received {s['received']} — no latencies")
            continue

        def pct(a: list[float]) -> float:
            return a[min(len(a) - 1, int(len(a) * 0.99))]

        hz = len(lat) / s["elapsed_s"] if s["elapsed_s"] > 0 else 0.0
        print(
            f"  {res * 1e3:>7.1f}m {p['per_axis']:>6} {p['bytes'] / 1e6:>7.2f} "
            f"{s['received']:>6} {statistics.median(lat):>8.2f}m {pct(lat):>8.2f}m "
            f"{pct(pub_ms):>8.2f}m {hz:>6.2f}"
        )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("role", choices=["pub", "sub", "sweep"])
    ap.add_argument("--res", type=float)
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.025, 0.020, 0.015])
    ap.add_argument("--count", type=int, default=30)
    a = ap.parse_args()
    if a.role == "sweep":
        sys.exit(run_sweep(a.resolutions, a.count))
    if a.res is None:
        ap.error("--res is required for the pub and sub roles")
    (run_pub if a.role == "pub" else run_sub)(a.res, a.count)
