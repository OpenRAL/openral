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

import os
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
    """Run a node as its own process, teeing its log where a failure can quote it."""
    with log_path.open("wb") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield proc
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - shutdown hardening
                proc.kill()
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


@contextmanager
def _self_filter_rig(tmp_path: Path, *, filter_argv: list[str]) -> Iterator[Any]:
    """A live costmap on the filtered topic, plus the filter process under test.

    Yields ``(executor, publish, costs)``: ``costs[i]`` is sampled at the probe
    point on every costmap update, and ``publish(scan)`` returns a callable that
    re-publishes that scan (and an empty world state, so the payload half is
    provably not what is doing the removing).
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

        def _on_costmap(msg: Costmap) -> None:
            for probe in probes:
                samples[probe].append(_cost_at(msg, probe, 0.0))

        node.create_subscription(Costmap, f"{_COSTMAP_NAMESPACE}/costmap_raw", _on_costmap, 1)
        scan_pub = node.create_publisher(LaserScan, "/scan", sensor_qos)
        state_pub = node.create_publisher(WorldStateStamped, "/openral/world_state_fast", state_qos)

        def _publish_of(scan: Any) -> Any:
            def _publish() -> None:
                state_pub.publish(WorldStateStamped())
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
        with (
            _process(filter_argv, filter_log),
            _process(costmap_argv, tmp_path / "costmap_self.log"),
        ):
            _lifecycle("configure")
            _lifecycle("activate")
            yield executor, _publish_of, samples, filter_log
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
