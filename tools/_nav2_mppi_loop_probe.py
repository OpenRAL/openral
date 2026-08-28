"""Time the WHOLE MPPI control loop against its 50 ms budget, on a live graph.

`openral_nav2_bringup` deferred `CostCritic.consider_footprint` because the
flag's own cost was measured in isolation
(`benchmark/cost_critic_footprint_bench.cpp`: +8.1 / +9.7 ms) while the loop
AROUND CostCritic was not. This measures the loop: it attaches to a running
`openral deploy sim` graph, drives a real `NavigateToPose`, and reads
`controller_server`'s own CPU time from `/proc` divided by the control cycles
it actually produced.

**CPU per cycle, not wall-clock per cycle, deliberately.** The stack runs
`use_sim_time:=True`, so the controller's 20 Hz is 20 Hz of SIMULATION time.
Wall-clock spacing between commands would measure how fast MuJoCo steps, not
whether the controller fits its budget. Process CPU time is independent of the
clock source and is directly comparable to the 50 ms period.

Four arms, so the flag's effect is separable from the footprint's::

    bare  x consider_footprint=false   # what ships today
    bare  x consider_footprint=true
    grown x consider_footprint=false
    grown x consider_footprint=true    # the arm that decides the flip

**Every run validates which polygon the costmap actually adopted** by reading
`/local_costmap/published_footprint` and measuring its longest edge (frame
invariant: 0.72 m bare, 1.23 m grown). A run whose costmap never took the arm's
polygon is reported `arm_valid: false` rather than counted -- the shipped
`payload_footprint_node` publishes the bare polygon at 2 Hz whenever nothing is
attached, and without this check the two publishers silently alternate.

Private (`_` prefix): it attaches to a graph someone else launched and is
evidence tooling, not a shipped entry point. The round it produced is recorded
in `docs/reference/robocasa-carry-survey.md`; its raw output is
`docs/reference/data/nav2-mppi-loop-2026-08-28.jsonl`.

Usage (with a deploy-sim graph already up and Nav2 ACTIVE)::

    ros2 param set /controller_server FollowPath.CostCritic.consider_footprint true
    python tools/_nav2_mppi_loop_probe.py --grown --seconds=25
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import rclpy
from geometry_msgs.msg import Point32, Polygon
from nav2_msgs.action import (  # type: ignore[import-untyped]  # reason: generated ROS action package, no py.typed
    NavigateToPose,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

BARE = [(0.35, 0.25), (-0.35, 0.25), (-0.35, -0.25), (0.35, -0.25)]
GROWN = [(0.86, 0.25), (-0.35, 0.25), (-0.35, -0.25), (0.86, -0.25)]
CLK = os.sysconf("SC_CLK_TCK")


def controller_pid() -> int:

    out = subprocess.run(
        ["pgrep", "-f", "nav2_controller/controller_server"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    if not out:
        raise SystemExit("controller_server not running")
    return int(out[0])


def cpu_seconds(pid: int) -> float:
    with open(f"/proc/{pid}/stat") as fh:
        parts = fh.read().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / CLK  # utime + stime


class Probe(Node):  # type: ignore[misc]  # reason: rclpy ships no py.typed, so Node is Any
    def __init__(self, grown: bool) -> None:
        # The stack runs use_sim_time:=True; a wall-clock TF listener asks for
        # stamps the sim-time buffer has never held.
        from rclpy.parameter import Parameter

        super().__init__(
            "mppi_loop_probe",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.fp = [
            self.create_publisher(Polygon, t, 10)
            for t in ("/local_costmap/footprint", "/global_costmap/footprint")
        ]
        self.poly = GROWN if grown else BARE
        self.cycles = 0
        from geometry_msgs.msg import PolygonStamped, Twist

        self.seen: dict[str, int] = {}
        # All cmd_vel* on this graph are plain Twist (checked live).
        # /cmd_vel_nav is controller_server's own output -- one message per
        # control cycle, which is what the 50 ms budget bounds.
        for topic in ("/cmd_vel_nav", "/cmd_vel"):
            self.create_subscription(Twist, topic, self._make_cb(topic), 100)
        # Read back what the costmap actually adopted, so a measurement can
        # never claim a grown footprint the costmap never accepted.
        self.published_fp: list[tuple[float, float]] = []
        self.create_subscription(
            PolygonStamped, "/local_costmap/published_footprint", self._on_fp, 10
        )
        self.create_timer(0.05, self._pub_fp)  # 20 Hz: out-publish the 2 Hz node
        self.ac = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def _on_fp(self, m: Any) -> None:
        self.published_fp = [(round(p.x, 3), round(p.y, 3)) for p in m.polygon.points]

    def _make_cb(self, topic: str) -> Callable[[Any], None]:
        def _cb(_m: Any) -> None:
            self.cycles += 1
            self.seen[topic] = self.seen.get(topic, 0) + 1

        return _cb

    def _pub_fp(self) -> None:
        p = Polygon()
        p.points = [Point32(x=float(a), y=float(b), z=0.0) for a, b in self.poly]
        for pub in self.fp:
            pub.publish(p)


def main() -> int:
    grown = "--grown" in sys.argv
    dur = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--seconds=")), "40"))
    rclpy.init()
    n = Probe(grown)

    def spin(t: float) -> None:
        end = time.monotonic() + t
        while time.monotonic() < end:
            rclpy.spin_once(n, timeout_sec=0.02)

    spin(8.0)  # settle: /clock + TF + footprint

    if not n.ac.wait_for_server(timeout_sec=15.0):
        print(json.dumps({"error": "navigate_to_pose action server absent"}))
        return 2

    # The sim-time TF buffer holds only a sliver of history, so a single
    # "latest" lookup races the writer. Retry against the buffer's own newest
    # stamp instead of asking for a time it may already have dropped.
    tf = None
    last = ""
    for _ in range(60):
        spin(0.5)
        for stamp in (rclpy.time.Time(), None):
            try:
                if stamp is None:
                    t_avail = n.buf.get_latest_common_time("map", "base_link")
                    tf = n.buf.lookup_transform("map", "base_link", t_avail)
                else:
                    tf = n.buf.lookup_transform("map", "base_link", stamp)
                break
            except Exception as e:
                last = str(e)
        if tf is not None:
            break
    if tf is None:
        print(json.dumps({"error": f"no map->base_link TF: {last}"}))
        return 2
    x0, y0 = tf.transform.translation.x, tf.transform.translation.y

    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = "map"
    goal.pose.pose.position.x = x0 + 1.2
    goal.pose.pose.position.y = y0
    goal.pose.pose.orientation.w = 1.0
    fut = n.ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=15.0)
    if fut.result() is None or not fut.result().accepted:
        print(json.dumps({"error": "goal rejected", "start": [x0, y0]}))
        return 3

    gh = fut.result()
    res_fut = gh.get_result_async()
    pid = controller_pid()
    n.seen.clear()
    c0, w0 = cpu_seconds(pid), time.monotonic()
    spin(dur)
    c1, w1 = cpu_seconds(pid), time.monotonic()
    n.status = "done" if res_fut.done() else "still-running"

    # controller_server's OWN output is one message per control cycle;
    # /cmd_vel is the smoothed republish and would double-count.
    cyc = n.seen.get("/cmd_vel_nav", 0)
    # The costmap republishes the adopted polygon in the map frame, rotated and
    # translated. Its longest edge is frame-invariant, so it identifies WHICH
    # polygon was in force: bare is 0.72 m long (0.70 + 2x0.01 padding), grown
    # is 1.23 m. A run whose costmap never took the arm's polygon measures the
    # wrong thing and is marked invalid rather than reported.
    pts = n.published_fp
    fp_len = 0.0
    if len(pts) >= 2:
        fp_len = max(math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))
    want = 1.23 if grown else 0.72
    valid = abs(fp_len - want) < 0.06
    out = {
        "arm": "grown" if grown else "bare",
        "start_xy": [round(x0, 3), round(y0, 3)],
        "wall_s": round(w1 - w0, 2),
        "controller_cpu_s": round(c1 - c0, 3),
        "cycles": cyc,
        "cpu_ms_per_cycle": round((c1 - c0) / cyc * 1000, 2) if cyc else None,
        "budget_ms": 50.0,
        "by_topic": n.seen,
        "published_footprint": n.published_fp,
        "published_max_x": round(max((x for x, _ in n.published_fp), default=0.0), 3),
        "footprint_len_m": round(fp_len, 3),
        "arm_valid": bool(valid),
        "goal_status": getattr(n, "status", None),
    }
    print(json.dumps(out))
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
