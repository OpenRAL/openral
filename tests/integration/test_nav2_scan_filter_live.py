"""Live proof that the payload scan filter keeps Nav2's costmap honest.

Nav2 here is **base-only**: the costmaps' footprint is the manifest's bare
chassis and nothing grows it, because the arm and anything carried belong to the
3-D safety kernel (see ``packages/openral_nav2_bringup/README.md``, "Nav2 is
base-only"). That makes this node load-bearing rather than cosmetic — with no
footprint growing over the payload, an unfiltered payload return is simply an
obstacle that moves with the robot, one it can never drive away from.

The unit suite
(``packages/openral_nav2_bringup/test/test_payload_scan_filter.py``) pins the
geometry the filter computes. It cannot pin the part that actually matters: that
a **real** ``nav2_costmap_2d``, reading the filtered topic the shipped
``config/nav2_panda_mobile.yaml`` points every observation source at, ends up
with no cell marked for the carried object — and still marks a real obstacle at
the same bearing. That claim is about Nav2's parameter and topic contract, so
only Nav2's own binary can settle it.

Three tests, one per direction the node can be wrong:

* the carried object never marks the cost grid;
* a self-return is removed while a real obstacle on the same bearing survives;
* a filter that cannot place the chassis removes **nothing** — the control that
  proves the first two measured the filter and not ``footprint_clearing_enabled``.

Real components throughout (CLAUDE.md §1.11): the upstream ``nav2_costmap_2d``
node from ``ros-${ROS_DISTRO}-nav2-bringup``, the production
``payload_scan_filter_node`` run as its own process through its real ``main()``,
and the real ``robots/panda_mobile/robot.yaml`` for the chassis outline. The one
constructed input is the ``WorldStateStamped`` carrying the attachment.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy + a real nav2_costmap_2d — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "and source install/setup.bash first."
)

pytestmark = pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT_YAML = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
_NODE_DIR = _REPO_ROOT / "packages" / "openral_nav2_bringup" / "openral_nav2_bringup"
_SCAN_FILTER_NODE = _NODE_DIR / "payload_scan_filter_node.py"

# The standalone `nav2_costmap_2d` executable runs one `Costmap2DROS` named
# `costmap`, whose footprint topics are RELATIVE — so launching it under
# `__ns:=/local_costmap` reproduces the exact topic names the production
# `nav2_bringup` graph uses, which is the half of the contract the shipped
# `config/nav2_panda_mobile.yaml` and `DEFAULT_FOOTPRINT_TOPICS` depend on.
# Verified against the Jazzy binary, not assumed: `ros2 node info` on the
# namespaced node lists `/local_costmap/footprint` inbound
# (`geometry_msgs/Polygon`) and `/local_costmap/published_footprint` outbound
# (`geometry_msgs/PolygonStamped`).
_COSTMAP_NAMESPACE = "/local_costmap"
_COSTMAP_NODE = "/local_costmap/costmap"

#: ``openral_nav2_bringup.payload_scan_filter_node.DEFAULT_OUTPUT_TOPIC``,
#: repeated rather than imported so collecting this module never needs the
#: colcon overlay. The unit suite pins that this string, the node's default and
#: every source in ``config/nav2_panda_mobile.yaml`` still agree.
_FILTERED_SCAN_TOPIC = "/openral/nav2/scan"

_ATTACH_LINK = "panda_link7"
_SCAN_FRAME = "base_scan"
_SCAN_Z_IN_BASE = 0.35
_LINK_X_IN_BASE = 0.55
_PAYLOAD_X_IN_LINK = 0.20
_PAYLOAD_HALF_X = 0.10
#: Where the payload's centre lands in ``base_link`` / ``odom`` — 0.75 m ahead,
#: well clear of the 0.35 m chassis, so the costmap's own
#: ``footprint_clearing_enabled`` cannot be what removes it.
_PAYLOAD_X_IN_BASE = _LINK_X_IN_BASE + _PAYLOAD_X_IN_LINK
# nav2_costmap_2d's `footprint_padding` default, verified live: the published
# polygon is the received one grown by this on every axis.
_FOOTPRINT_PADDING_M = 0.01


# Same costmap, plus the obstacle layer reading the FILTERED scan topic the
# shipped `config/nav2_panda_mobile.yaml` points every source at.
_COSTMAP_PARAMS_WITH_OBSTACLES = f"""\
/**:
  ros__parameters:
    update_frequency: 5.0
    publish_frequency: 5.0
    global_frame: odom
    robot_base_frame: base_link
    rolling_window: true
    width: 4
    height: 4
    resolution: 0.05
    footprint: "[[0.35, 0.25], [-0.35, 0.25], [-0.35, -0.25], [0.35, -0.25]]"
    plugins: ["obstacle_layer"]
    obstacle_layer:
      plugin: "nav2_costmap_2d::ObstacleLayer"
      enabled: True
      observation_sources: scan
      scan:
        topic: {_FILTERED_SCAN_TOPIC}
        data_type: "LaserScan"
        clearing: True
        marking: True
        max_obstacle_height: 2.0
        raytrace_max_range: 3.0
        obstacle_max_range: 2.5
    always_send_full_costmap: True
"""

# The self-filter's costmap. Identical to the payload one except that
# `footprint_clearing_enabled` is turned OFF.
#
# That is not a convenience — it is what makes the test mean anything. A
# self-return lands *inside* the chassis polygon by definition, and with the
# upstream default (`True`, verified live) the obstacle layer frees every cell
# under the footprint on each update, so the cell would read clear whether or
# not the filter did its job and the degraded control below could never fail.
# Turning it off isolates the filter — and reproduces the one consumer that has
# no costmap-side clearing at all: `collision_monitor`, which reads the scan raw
# and is exactly what brakes for the robot's own chassis today.
_COSTMAP_PARAMS_SELF_RETURNS = _COSTMAP_PARAMS_WITH_OBSTACLES.replace(
    "      enabled: True\n",
    "      enabled: True\n      footprint_clearing_enabled: False\n",
)


def _spin_until(
    executor: Any, predicate: Any, *, timeout_s: float = 20.0, each: Any = None
) -> bool:
    """Spin until ``predicate()`` holds, running ``each()`` on every pass.

    ``each`` re-publishes the world state on every iteration rather than in one
    burst: ``/openral/world_state_fast`` is VOLATILE and 30 Hz on a real robot,
    so a single publish that lands before DDS discovery matches is simply lost
    and the test would fail on the transport, not on the behaviour.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if each is not None:
            each()
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return predicate()


@contextmanager
def _process(argv: list[str], log_path: Path) -> Iterator[subprocess.Popen[bytes]]:
    """Run a node in its own process GROUP, teeing its log where a failure can quote it.

    The group is the load-bearing part. ``ros2 run <pkg> <exe>`` **forks** the
    node rather than exec-ing it, so signalling the ``Popen`` handle reaps the
    ``ros2`` wrapper and orphans the node — which keeps publishing, on the same
    topics, under the same node name. Every test in this file uses
    ``/local_costmap/costmap_raw``, so one orphan makes later tests sample a
    costmap they never configured; and the test immediately before the
    silhouette sweep is the degraded control, whose whole job is to leave the
    chassis marked. That is the grid the sweep then refused, twice, in CI.

    Signalling the whole group is what actually stops the node. Verified by
    ``pgrep -f nav2_costmap_2d`` returning nothing after a run, where it used to
    return one process per test.
    """
    with log_path.open("wb") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield proc
        finally:
            group = os.getpgid(proc.pid)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - shutdown hardening
                pass
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)
            proc.wait(timeout=10)


def _lifecycle(transition: str, *, timeout_s: float = 45.0) -> None:
    """Drive the costmap through ``transition``, waiting for it to be ready.

    Retried rather than one-shot: the node needs a moment to advertise
    ``change_state`` after ``Popen`` returns, and ``activate`` fails until the
    ``base_link -> odom`` TF it checks has been discovered. Both are startup
    races, not behaviour, and a one-shot call makes the test flaky on a loaded
    host. Stops on the first success, so it can never re-drive a transition
    that already happened.
    """
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["ros2", "lifecycle", "set", _COSTMAP_NODE, transition],
            check=False,
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0:
            return
        last = (result.stdout + result.stderr).decode(errors="replace").strip()
        time.sleep(0.5)
    raise AssertionError(f"costmap never accepted `{transition}` within {timeout_s}s: {last}")


def _static_transforms(node: Any) -> Any:
    """``odom -> base_link -> {panda_link7, base_scan}``, the frames the nodes need.

    The costmap refuses to activate without ``base_link -> odom``; the footprint
    publisher needs the attach link to place the payload; the scan filter needs
    ``base_scan -> panda_link7`` to place it in the sensor's frame. The lidar
    sits at the attach link's height here so the scan plane cuts through the
    carried object — the case where a payload becomes a costmap obstacle at all.
    A static broadcaster is TRANSIENT_LOCAL, so the subprocesses get these
    however late they join.
    """
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import StaticTransformBroadcaster

    broadcaster = StaticTransformBroadcaster(node)
    transforms = []
    for parent, child, x, z in (
        ("odom", "base_link", 0.0, 0.0),
        ("base_link", _ATTACH_LINK, _LINK_X_IN_BASE, _SCAN_Z_IN_BASE),
        ("base_link", _SCAN_FRAME, 0.0, _SCAN_Z_IN_BASE),
    ):
        t = TransformStamped()
        t.header.stamp = node.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = x
        t.transform.translation.z = z
        t.transform.rotation.w = 1.0
        transforms.append(t)
    broadcaster.sendTransform(transforms)
    return broadcaster


def _world_state(*, carrying: bool, revision: int) -> Any:
    """A ``WorldStateStamped`` with (or without) one boxed payload attached."""
    from openral_msgs.msg import (
        AttachedCollisionObject,
        AttachedCollisionPrimitive,
        WorldStateStamped,
    )

    msg = WorldStateStamped()
    msg.attachment_revision = revision
    if not carrying:
        return msg

    primitive = AttachedCollisionPrimitive()
    primitive.shape_type = AttachedCollisionPrimitive.SHAPE_BOX
    primitive.shape_dimensions = [_PAYLOAD_HALF_X, 0.06, 0.06]
    primitive.pose_in_object.orientation.w = 1.0

    obj = AttachedCollisionObject()
    obj.object_id = "baguette"
    obj.attach_link = _ATTACH_LINK
    obj.pose_in_link.position.x = _PAYLOAD_X_IN_LINK
    obj.pose_in_link.orientation.w = 1.0
    obj.primitives = [primitive]
    msg.attached_objects = [obj]
    return msg


def _payload_scan() -> Any:
    """A 360-beam fan that sees a wall at 3 m, except where the payload is.

    The forward beams stop on the carried object at its true distance. This is
    the sensor picture a lidar mounted at the payload's height actually
    produces, and the one that made the payload a costmap obstacle.
    """
    import math

    from sensor_msgs.msg import LaserScan

    n_beams = 360
    scan = LaserScan()
    scan.header.frame_id = _SCAN_FRAME
    scan.angle_min = -math.pi
    scan.angle_max = math.pi
    scan.angle_increment = 2.0 * math.pi / n_beams
    scan.range_min = 0.05
    scan.range_max = 12.0
    ranges = [3.0] * n_beams
    for angle in (-0.06, -0.03, 0.0, 0.03, 0.06):
        index = round((angle - scan.angle_min) / scan.angle_increment) % n_beams
        ranges[index] = _PAYLOAD_X_IN_BASE
    scan.ranges = ranges
    return scan


def _cost_at(costmap: Any, x_m: float, y_m: float) -> int:
    """The costmap cell covering ``(x, y)`` in its own global frame."""
    meta = costmap.metadata
    mx = int((x_m - meta.origin.position.x) / meta.resolution)
    my = int((y_m - meta.origin.position.y) / meta.resolution)
    return int(costmap.data[my * meta.size_x + mx])


def test_the_carried_object_never_becomes_a_costmap_obstacle(tmp_path: Path) -> None:
    """The scan filter's real claim, measured in a real costmap's cost grid.

    One unchanging sensor picture, two attachment states. While the object is
    attached its returns must not mark the costmap at all; the moment World
    State says it is released, the very same returns must mark it — otherwise
    the filter is not removing a payload, it is blinding Nav2.

    The payload sits 0.75 m ahead, well outside the 0.35 m chassis, so Nav2's
    own ``footprint_clearing_enabled`` (default ``True``, verified live) cannot
    be what keeps the cell free.
    """
    import rclpy
    from nav2_msgs.msg import Costmap
    from openral_msgs.msg import WorldStateStamped
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import LaserScan

    lethal_threshold = 253

    params_file = tmp_path / "costmap_obstacles.yaml"
    params_file.write_text(_COSTMAP_PARAMS_WITH_OBSTACLES)

    rclpy.init()
    node = Node("test_nav2_payload_scan_filter")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    costs: list[int] = []

    def _on_costmap(msg: Costmap) -> None:
        costs.append(_cost_at(msg, _PAYLOAD_X_IN_BASE, 0.0))

    try:
        broadcaster = _static_transforms(node)
        assert broadcaster is not None
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        state_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        node.create_subscription(Costmap, f"{_COSTMAP_NAMESPACE}/costmap_raw", _on_costmap, 1)
        scan_pub = node.create_publisher(LaserScan, "/scan", sensor_qos)
        state_pub = node.create_publisher(WorldStateStamped, "/openral/world_state_fast", state_qos)
        scan = _payload_scan()

        filter_log = tmp_path / "payload_scan_filter.log"
        filter_argv = [sys.executable, str(_SCAN_FILTER_NODE)]
        costmap_argv = [
            "ros2",
            "run",
            "nav2_costmap_2d",
            "nav2_costmap_2d",
            "--ros-args",
            "-r",
            f"__ns:={_COSTMAP_NAMESPACE}",
            "--params-file",
            str(params_file),
        ]

        with (
            _process(filter_argv, filter_log),
            _process(costmap_argv, tmp_path / "costmap_obstacles.log"),
        ):
            _lifecycle("configure")
            _lifecycle("activate")

            def _tick(carrying: bool, revision: int) -> Any:
                state = _world_state(carrying=carrying, revision=revision)

                def _publish() -> None:
                    state_pub.publish(state)
                    scan.header.stamp = node.get_clock().now().to_msg()
                    scan_pub.publish(scan)

                return _publish

            # 1. Carrying: the payload's own returns must never mark the map.
            carrying = _tick(carrying=True, revision=1)
            assert _spin_until(executor, lambda: len(costs) > 40, timeout_s=25.0, each=carrying), (
                f"the costmap never published; filter said:\n"
                f"{filter_log.read_text(errors='replace')[-2000:]}"
            )
            assert max(costs) < lethal_threshold, (
                f"the carried object marked the costmap (peak cost {max(costs)}); filter said:\n"
                f"{filter_log.read_text(errors='replace')[-2000:]}"
            )

            # 2. Released: the identical scan must mark it again, or the filter
            #    is not removing a payload — it is blinding Nav2.
            costs.clear()
            released = _tick(carrying=False, revision=2)
            assert _spin_until(
                executor,
                lambda: bool(costs) and costs[-1] >= lethal_threshold,
                timeout_s=25.0,
                each=released,
            ), f"the released object never marked the costmap (last cost {costs[-1:]})"
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


#: Forward beams at this range land 0.20 m ahead of ``base_link`` — inside the
#: manifest's 0.35 m chassis, so the robot is the only thing that can be there.
_SELF_RETURN_M = 0.20
#: The same forward beams, moved 0.25 m past the chassis edge. Nothing about
#: the robot explains this one, so it must survive to the cost grid.
_OBSTACLE_AT_SAME_BEARING_M = 0.60
#: A ``base_frame`` no broadcaster ever publishes. The self-filter's
#: ``base_frame <- base_scan`` lookup then fails on every scan while the scan's
#: own frame stays valid, so the costmap still consumes it — which is what makes
#: this a *control* rather than a blackout.
_UNRESOLVABLE_BASE_FRAME = "no_such_base_frame"

#: How many of the 360 ring beams must come back non-finite before the filter
#: counts as live. **All** of them, deliberately: with a payload attached the
#: ring is 355 chassis beams plus 5 on the carried box, so any threshold below
#: 360 is satisfied by the chassis half alone and would let the payload half
#: still be warming up — and a payload return that reaches the grid is as
#: permanent as a chassis one.
_MIN_DROPPED_FOR_A_LIVE_FILTER = 360

_FORWARD_BEAM_ANGLES = (-0.06, -0.03, 0.0, 0.03, 0.06)


def _forward_scan(range_m: float) -> Any:
    """A 360-beam fan seeing a wall at 3 m, with the forward beams at ``range_m``."""
    import math

    from sensor_msgs.msg import LaserScan

    n_beams = 360
    scan = LaserScan()
    scan.header.frame_id = _SCAN_FRAME
    scan.angle_min = -math.pi
    scan.angle_max = math.pi
    scan.angle_increment = 2.0 * math.pi / n_beams
    # Matches the manifest's `range_min_m` since #194 lowered it to the sensor
    # minimum. It used to be 0.55 m — a radial cutoff sized to hide the chassis,
    # the blunt instrument this filter replaces — and this fixture already
    # ignored it, because gating here would hide the very returns under test.
    scan.range_min = 0.05
    scan.range_max = 12.0
    ranges = [3.0] * n_beams
    for angle in _FORWARD_BEAM_ANGLES:
        index = round((angle - scan.angle_min) / scan.angle_increment) % n_beams
        ranges[index] = range_m
    scan.ranges = ranges
    return scan


def _wait_for_a_filtered_scan(
    node: Any, executor: Any, *, publish: Any, filter_log: Path, timeout_s: float = 40.0
) -> None:
    """Block until the filter's own output proves it is filtering.

    The node fails open until its TF buffer holds ``base_frame <- scan_frame``,
    so "the process is up" is not "the filter is working". The proof is the
    output topic: a scan whose beams have been replaced by ``inf``.
    """
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import LaserScan

    dropped: list[int] = []

    def _on_filtered(msg: LaserScan) -> None:
        dropped.append(sum(1 for r in msg.ranges if not math.isfinite(r)))

    sub = node.create_subscription(
        LaserScan,
        _FILTERED_SCAN_TOPIC,
        _on_filtered,
        QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        ),
    )
    try:
        assert _spin_until(
            executor,
            lambda: bool(dropped) and dropped[-1] >= _MIN_DROPPED_FOR_A_LIVE_FILTER,
            timeout_s=timeout_s,
            each=publish,
        ), (
            f"the filter never dropped a beam in {timeout_s}s (best {max(dropped, default=0)} of "
            f"{_MIN_DROPPED_FOR_A_LIVE_FILTER} needed), so the costmap would have been fed an "
            f"unfiltered scan; filter said:\n{filter_log.read_text(errors='replace')[-2000:]}"
        )
    finally:
        node.destroy_subscription(sub)


@contextmanager
def _self_filter_rig(
    tmp_path: Path, *, filter_argv: list[str], state: Any = None, warmup_scan: Any = None
) -> Iterator[Any]:
    """A live costmap on the filtered topic, plus the filter process under test.

    Yields ``(executor, publish_of, samples, filter_log, latest)``:
    ``samples[probe]`` is the cost sampled at that probe point on every costmap
    update, ``latest`` holds the most recent whole ``Costmap`` (for callers that
    sweep the grid rather than probe it), and ``publish_of(scan)`` returns a
    callable that re-publishes that scan alongside ``state``.

    ``state`` defaults to an empty ``WorldStateStamped``, so the payload half of
    the filter is provably not what is doing the removing. A caller that wants
    the payload half acting passes the attachment in.

    ``warmup_scan`` closes a startup race that is **not** cosmetic. The filter
    fails open by design — a scan that arrives before its TF buffer has the
    ``base_frame <- scan_frame`` transform is republished unfiltered — and a
    self-return that reaches the cost grid even once is **permanent**: the
    filter then removes exactly the beam whose ray would have cleared it, and
    a costmap with ``footprint_clearing_enabled: False`` has nothing else that
    would. (Measured: 32 cells marked by one unfiltered ring survived 20 s of
    all-``inf`` filtered scans, and cleared the moment a real return was put on
    the same bearings.) So a caller that asserts on the *steady state* passes
    its scan in here, and the costmap is not configured until the filter's own
    output proves the filter is live.
    """
    import rclpy
    from nav2_msgs.msg import Costmap
    from openral_msgs.msg import WorldStateStamped
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import LaserScan

    params_file = tmp_path / "costmap_self.yaml"
    params_file.write_text(_COSTMAP_PARAMS_SELF_RETURNS)

    rclpy.init()
    node = Node("test_nav2_self_return_filter")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        broadcaster = _static_transforms(node)
        assert broadcaster is not None
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        state_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        samples: dict[float, list[int]] = {}
        probes: list[float] = [_SELF_RETURN_M, _OBSTACLE_AT_SAME_BEARING_M]
        for probe in probes:
            samples[probe] = []

        latest: list[Any] = []

        def _on_costmap(msg: Costmap) -> None:
            latest[:] = [msg]
            for probe in probes:
                samples[probe].append(_cost_at(msg, probe, 0.0))

        node.create_subscription(Costmap, f"{_COSTMAP_NAMESPACE}/costmap_raw", _on_costmap, 1)
        scan_pub = node.create_publisher(LaserScan, "/scan", sensor_qos)
        state_pub = node.create_publisher(WorldStateStamped, "/openral/world_state_fast", state_qos)

        published_state = WorldStateStamped() if state is None else state

        def _publish_of(scan: Any) -> Any:
            def _publish() -> None:
                state_pub.publish(published_state)
                scan.header.stamp = node.get_clock().now().to_msg()
                scan_pub.publish(scan)

            return _publish

        costmap_argv = [
            "ros2",
            "run",
            "nav2_costmap_2d",
            "nav2_costmap_2d",
            "--ros-args",
            "-r",
            f"__ns:={_COSTMAP_NAMESPACE}",
            "--params-file",
            str(params_file),
        ]
        filter_log = tmp_path / "self_scan_filter.log"
        with _process(filter_argv, filter_log):
            if warmup_scan is not None:
                # The costmap process is not started until this returns. A gate
                # that merely delays `configure` would still leave a window in
                # which an unfiltered scan is published while a costmap process
                # exists, and one mark that lands inside the chassis is
                # permanent (see this function's docstring). Not existing yet is
                # the only airtight version.
                _wait_for_a_filtered_scan(
                    node, executor, publish=_publish_of(warmup_scan), filter_log=filter_log
                )
            with _process(costmap_argv, tmp_path / "costmap_self.log"):
                _lifecycle("configure")
                _lifecycle("activate")
                yield executor, _publish_of, samples, filter_log, latest
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_a_self_return_is_removed_and_a_real_obstacle_at_the_same_bearing_is_not(
    tmp_path: Path,
) -> None:
    """The robot half of the filter, measured in a real costmap's cost grid.

    A 2-D lidar on a real base sees the base. Unfiltered, those returns become
    permanent costmap obstacles and the robot concludes it is surrounded by
    itself — the failure sim hides completely, because
    ``synthesize_laser_scan_2d`` re-casts through the robot's own MuJoCo
    kinematic tree and never emits a self-return in the first place.

    One beam carries one range, so a self-return and a real obstacle cannot
    share a beam of a single scan. The discriminating construction is therefore
    the same *bearing*, twice: forward beams at 0.20 m are inside the chassis
    and must vanish; the identical beams at 0.60 m are 0.25 m past it and must
    reach the cost grid untouched. Both phases run with an empty attachment set,
    so the payload half is provably not what is acting.
    """
    lethal_threshold = 253
    filter_argv = [
        sys.executable,
        str(_SCAN_FILTER_NODE),
        "--ros-args",
        "-p",
        f"robot_yaml:={_ROBOT_YAML}",
    ]

    with _self_filter_rig(tmp_path, filter_argv=filter_argv) as (
        executor,
        publish_of,
        samples,
        filter_log,
        _latest,
    ):
        # 1. The chassis return must never mark the grid.
        self_hit = publish_of(_forward_scan(_SELF_RETURN_M))
        seen = samples[_SELF_RETURN_M]
        assert _spin_until(executor, lambda: len(seen) > 40, timeout_s=25.0, each=self_hit), (
            f"the costmap never published; filter said:\n"
            f"{filter_log.read_text(errors='replace')[-2000:]}"
        )
        assert max(seen) < lethal_threshold, (
            f"the robot's own chassis marked the costmap (peak cost {max(seen)}); "
            f"filter said:\n{filter_log.read_text(errors='replace')[-2000:]}"
        )

        # 2. The same bearing, 0.25 m further out, is not the robot — and the
        #    filter must not have learned to eat that bearing.
        beyond = samples[_OBSTACLE_AT_SAME_BEARING_M]
        beyond.clear()
        obstacle = publish_of(_forward_scan(_OBSTACLE_AT_SAME_BEARING_M))
        assert _spin_until(
            executor,
            lambda: bool(beyond) and beyond[-1] >= lethal_threshold,
            timeout_s=25.0,
            each=obstacle,
        ), (
            f"a real obstacle 0.25 m past the chassis was deleted as self "
            f"(last cost {beyond[-1:]}); filter said:\n"
            f"{filter_log.read_text(errors='replace')[-2000:]}"
        )


def test_a_self_filter_that_cannot_place_the_chassis_removes_nothing(tmp_path: Path) -> None:
    """The fail-closed direction, and the control for the test above.

    Same node, same scan, same costmap — only the ``base_frame`` is one nobody
    broadcasts, so every ``base_frame <- base_scan`` lookup fails. Dropping a
    real obstacle because we mistook it for the robot is the dangerous error
    here, so a self-filter that cannot place the chassis must remove *nothing*,
    and the 0.20 m return must arrive in the cost grid exactly as the raw
    sensor reported it.

    This doubles as the proof that the sibling test measured the filter: if
    something else (footprint clearing, the range gate, the rolling window)
    were keeping that cell free, this assertion would fail too.
    """
    lethal_threshold = 253
    filter_argv = [
        sys.executable,
        str(_SCAN_FILTER_NODE),
        "--ros-args",
        "-p",
        f"robot_yaml:={_ROBOT_YAML}",
        "-p",
        f"base_frame:={_UNRESOLVABLE_BASE_FRAME}",
    ]

    with _self_filter_rig(tmp_path, filter_argv=filter_argv) as (
        executor,
        publish_of,
        samples,
        filter_log,
        _latest,
    ):
        seen = samples[_SELF_RETURN_M]
        self_hit = publish_of(_forward_scan(_SELF_RETURN_M))
        assert _spin_until(
            executor,
            lambda: bool(seen) and seen[-1] >= lethal_threshold,
            timeout_s=25.0,
            each=self_hit,
        ), (
            f"a degraded self-filter still removed the return (last cost {seen[-1:]}); "
            f"filter said:\n{filter_log.read_text(errors='replace')[-2000:]}"
        )


#: Every beam of the ring lands this far out. 0.20 m is inside the manifest
#: chassis at *every* bearing (its narrowest half-extent is 0.25 m), so the ring
#: is the real-hardware picture in full: a 2-D lidar that sees nothing but the
#: robot it is bolted to. The probe-point tests above put five beams inside the
#: chassis; this puts 355.
_RING_SELF_RETURN_M = 0.20

#: ``nav2_costmap_2d``'s ``LETHAL_OBSTACLE``. The probe tests compare against
#: 253 (``INSCRIBED_INFLATED_OBSTACLE``) because either value fails a robot; a
#: silhouette *sweep* has to be stricter about what it means, and the claim
#: being proved is that no cell inside the robot was **marked** — 253 is what an
#: inflation layer writes near a legitimate obstacle elsewhere.
_LETHAL_OBSTACLE = 254


def _ring_scan(*, payload_x_m: float | None = None) -> Any:
    """A 360-beam fan that returns off the robot at every bearing.

    Optionally the five forward beams are pushed out to ``payload_x_m`` instead,
    which is where a carried object sits — so one scan carries both halves of
    the silhouette the sweep below asserts on.
    """
    import math

    from sensor_msgs.msg import LaserScan

    n_beams = 360
    scan = LaserScan()
    scan.header.frame_id = _SCAN_FRAME
    scan.angle_min = -math.pi
    scan.angle_max = math.pi
    scan.angle_increment = 2.0 * math.pi / n_beams
    scan.range_min = 0.05
    scan.range_max = 12.0
    ranges = [_RING_SELF_RETURN_M] * n_beams
    if payload_x_m is not None:
        for angle in _FORWARD_BEAM_ANGLES:
            index = round((angle - scan.angle_min) / scan.angle_increment) % n_beams
            ranges[index] = payload_x_m
    scan.ranges = ranges
    return scan


def _silhouette_mask(costmap: Any, *, with_payload: bool) -> Any:
    """Which of ``costmap``'s cell centres lie inside the robot (∪ the payload).

    The predicates are the shipped node's own — ``base_footprint_polygon`` off
    the real ``robots/panda_mobile/robot.yaml`` for the chassis, and
    ``points_in_primitive`` for the carried box — so this measures the same
    geometry the filter measures rather than a second opinion about it.

    Evaluated **in the scan plane** (``_SCAN_Z_IN_BASE``), which is the only
    height a 2-D costmap can be marked from and, in this rig, the payload box's
    own centre height with the box axis-aligned — so the cross-section taken
    here is its full ground projection, not a slice of it.

    The rig's ``odom -> base_link`` is identity, so the costmap's own frame and
    ``base_link`` share an origin and no transform is needed.
    """
    import numpy as np
    from openral_core import RobotDescription
    from openral_nav2_bringup._footprint_geometry import SHAPE_BOX, base_footprint_polygon
    from openral_nav2_bringup.payload_scan_filter_node import (
        points_in_convex_polygon,
        points_in_primitive,
    )

    meta = costmap.metadata
    xs = meta.origin.position.x + (np.arange(meta.size_x) + 0.5) * meta.resolution
    ys = meta.origin.position.y + (np.arange(meta.size_y) + 0.5) * meta.resolution
    grid_x, grid_y = np.meshgrid(xs, ys)
    points_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    chassis = base_footprint_polygon(RobotDescription.from_yaml(str(_ROBOT_YAML)))
    inside = points_in_convex_polygon(points_xy, chassis)
    if with_payload:
        transform = np.eye(4)
        transform[:3, 3] = (_PAYLOAD_X_IN_BASE, 0.0, _SCAN_Z_IN_BASE)
        points_xyz = np.concatenate(
            [points_xy, np.full((points_xy.shape[0], 1), _SCAN_Z_IN_BASE)], axis=1
        )
        inside = inside | points_in_primitive(
            points_xyz, SHAPE_BOX, [_PAYLOAD_HALF_X, 0.06, 0.06], transform
        )
    return inside


def _marked_cells_inside_silhouette(
    costmap: Any, *, with_payload: bool
) -> list[tuple[float, float]]:
    """The ``(x, y)`` of every ``LETHAL_OBSTACLE`` cell inside the silhouette."""
    import numpy as np

    meta = costmap.metadata
    data = np.asarray(costmap.data, dtype=np.int32)
    inside = _silhouette_mask(costmap, with_payload=with_payload)
    hits = np.flatnonzero(inside & (data == _LETHAL_OBSTACLE))
    return [
        (
            float(meta.origin.position.x + (int(i) % meta.size_x + 0.5) * meta.resolution),
            float(meta.origin.position.y + (int(i) // meta.size_x + 0.5) * meta.resolution),
        )
        for i in hits
    ]


def test_no_costmap_cell_inside_the_robot_or_payload_silhouette_is_marked(tmp_path: Path) -> None:
    """Issue #108's costmap-clean claim, swept rather than probed.

    "The costmaps contain no floating or self obstacles" is a statement about
    the **whole** silhouette, and the probe-point tests above cannot make it: a
    return that marks some *other* cell inside the robot — a different bearing,
    a raytrace artefact, a stale mark the rolling window carried along — passes
    them and still leaves the base surrounded by itself.

    So this drives the full real-hardware picture (355 beams returning off the
    chassis at every bearing, five off a carried box) through the shipped
    filter into a real ``nav2_costmap_2d``, then sweeps every cell of the
    published grid and asserts **zero** are marked inside the chassis ∪ payload
    silhouette.

    It asserts on the **steady state**: the rig withholds the costmap until the
    filter's own output proves it is filtering, because a self-return that
    reaches the grid once is permanent (the filter then removes the very beam
    whose ray would clear it). That transient is a real property of the node's
    fail-open design, recorded in the package README rather than hidden here.

    ``footprint_clearing_enabled`` is off in this costmap (see
    ``_COSTMAP_PARAMS_SELF_RETURNS``), so Nav2's own footprint clearing cannot
    be what keeps the silhouette clean — and it reproduces the one consumer
    that has no costmap-side clearing at all, ``collision_monitor``, which reads
    the scan raw.

    This is the assertion ADR-0099 obliges: with Nav2 base-only, nothing grows
    the footprint over a payload any more, so the scan filter is the only thing
    keeping the robot and what it carries out of Nav2's world.
    """
    filter_argv = [
        sys.executable,
        str(_SCAN_FILTER_NODE),
        "--ros-args",
        "-p",
        f"robot_yaml:={_ROBOT_YAML}",
    ]

    ring = _ring_scan(payload_x_m=_PAYLOAD_X_IN_BASE)
    # The rig cannot publish while `ros2 lifecycle set` blocks, and the node's
    # default 0.5 s `attached_state_timeout_s` expires inside that window — so
    # the first scan after activation could find no attachment, pass the five
    # payload beams through, and mark the payload silhouette permanently. The
    # sweep measures containment, not attachment freshness, so the timeout is
    # widened past the window rather than raced against.
    filter_argv += ["-p", "attached_state_timeout_s:=30.0"]
    with _self_filter_rig(
        tmp_path,
        filter_argv=filter_argv,
        state=_world_state(carrying=True, revision=1),
        # The costmap is not configured until this scan comes back filtered.
        # Without that gate the rig can feed it one unfiltered ring during the
        # filter's TF warm-up, and those marks never clear — see the rig's
        # docstring, and the "permanent self-marks" note in the package README.
        warmup_scan=ring,
    ) as (executor, publish_of, samples, filter_log, latest):
        publish = publish_of(ring)
        # One entry per costmap update; 40 of them is the same settling the
        # probe tests wait for, so the grid has been marked and re-marked many
        # times over before the sweep reads it.
        updates = samples[_SELF_RETURN_M]

        assert _spin_until(executor, lambda: len(updates) > 40, timeout_s=25.0, each=publish), (
            f"the costmap never published; filter said:\n"
            f"{filter_log.read_text(errors='replace')[-2000:]}"
        )
        # The payload half needs its own control, for the same reason the
        # chassis half has one: a mis-placed box or a changed dimension
        # convention would make "no marked cell inside the payload" true by
        # covering no cells at all.
        chassis_only = int(_silhouette_mask(latest[0], with_payload=False).sum())
        with_payload = int(_silhouette_mask(latest[0], with_payload=True).sum())
        assert with_payload > chassis_only, (
            f"the payload contributed no cells to the silhouette ({with_payload} vs "
            f"{chassis_only} for the chassis alone), so the payload half of this assertion "
            "is vacuous"
        )

        marked = _marked_cells_inside_silhouette(latest[0], with_payload=True)
        assert not marked, (
            f"{len(marked)} costmap cells are marked inside the robot/payload silhouette, "
            f"first at {marked[:5]} (chassis-only silhouette is {chassis_only} cells, with the "
            f"payload {with_payload}; the ring returns at {_RING_SELF_RETURN_M} m and the payload "
            f"at {_PAYLOAD_X_IN_BASE} m, so the x of a marked cell says which half let it "
            f"through); filter said:\n{filter_log.read_text(errors='replace')[-2000:]}"
        )


def test_the_silhouette_sweep_fails_when_the_self_filter_is_not_running(tmp_path: Path) -> None:
    """The control: the sweep above measured the filter, not the rig.

    Same ring, same costmap, same assertion — with the filter given no
    ``robot_yaml``, which is the documented degradation to "the robot's own
    returns are not filtered". The chassis must then mark its own silhouette.

    Without this, "zero marked cells inside the robot" is a claim the rolling
    window, the range gate or an unwired topic could satisfy on their own. That
    is not hypothetical here: #183 found this file's payload test passing
    vacuously because the deploy image had never built the package the filter
    lives in.
    """
    filter_argv = [sys.executable, str(_SCAN_FILTER_NODE)]

    with _self_filter_rig(tmp_path, filter_argv=filter_argv) as (
        executor,
        publish_of,
        _samples,
        filter_log,
        latest,
    ):
        publish = publish_of(_ring_scan())
        assert _spin_until(
            executor,
            lambda: (
                bool(latest)
                and bool(_marked_cells_inside_silhouette(latest[0], with_payload=False))
            ),
            timeout_s=25.0,
            each=publish,
        ), (
            "an unconfigured self-filter still left the silhouette clean, so the sweep "
            f"above proves nothing; filter said:\n{filter_log.read_text(errors='replace')[-2000:]}"
        )
