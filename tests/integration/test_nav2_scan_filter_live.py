"""Live proof that the payload scan filter keeps Nav2's costmap honest.

Nav2 is base-only (``packages/openral_nav2_bringup/README.md``): footprint is the bare
chassis; arm/payload go through the 3-D safety kernel instead of footprint growth, so an
unfiltered payload return is an obstacle the robot can never drive away from.

The unit suite (``packages/openral_nav2_bringup/test/test_payload_scan_filter.py``) pins the
filter's geometry only; only a real ``nav2_costmap_2d`` reading the filtered topic every
source in ``config/nav2_panda_mobile.yaml`` points at can prove the carried object marks no
cell while a real obstacle at the same bearing still does.

Three tests: (1) carried object never marks the cost grid, (2) a self-return is removed while
a real obstacle on the same bearing survives, (3) control — a filter that cannot place the
chassis removes nothing, proving (1)/(2) measured the filter, not
``footprint_clearing_enabled``.

Real components (CLAUDE.md §1.11): upstream ``nav2_costmap_2d``
(``ros-${ROS_DISTRO}-nav2-bringup``), production ``payload_scan_filter_node`` via its real
``main()``, real ``robots/panda_mobile/robot.yaml`` chassis outline. Only
``WorldStateStamped`` is constructed.
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

# The standalone `nav2_costmap_2d` executable runs one `Costmap2DROS` named `costmap`, whose
# footprint topics are RELATIVE — `__ns:=/local_costmap` reproduces the production nav2_bringup
# topic names that config/nav2_panda_mobile.yaml and DEFAULT_FOOTPRINT_TOPICS depend on.
# Confirmed on the Jazzy binary via `ros2 node info`: `/local_costmap/footprint` inbound
# (geometry_msgs/Polygon), `/local_costmap/published_footprint` outbound
# (geometry_msgs/PolygonStamped).
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
# nav2_costmap_2d's `footprint_padding` default: published polygon = received polygon
# grown by this on every axis.
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

# Self-filter costmap = payload one with `footprint_clearing_enabled` turned OFF.
# Required, not cosmetic: a self-return lands inside the chassis polygon, and with the
# upstream default (True) the obstacle layer frees every footprint cell each update, so the
# cell would read clear regardless of the filter — the degraded control below could never
# fail. OFF isolates the filter and reproduces `collision_monitor`, which reads the scan raw
# with no costmap-side clearing and is what brakes for the robot's own chassis today.
_COSTMAP_PARAMS_SELF_RETURNS = _COSTMAP_PARAMS_WITH_OBSTACLES.replace(
    "      enabled: True\n",
    "      enabled: True\n      footprint_clearing_enabled: False\n",
)


def _spin_until(
    executor: Any, predicate: Any, *, timeout_s: float = 20.0, each: Any = None
) -> bool:
    """Spin until ``predicate()`` holds, running ``each()`` on every pass.

    ``each`` re-publishes each iteration, not once: ``/openral/world_state_fast`` is VOLATILE
    and 30 Hz on a real robot, so a publish before DDS discovery matches is simply lost.
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

    Group is load-bearing: ``ros2 run <pkg> <exe>`` forks rather than execs, so signalling
    the ``Popen`` handle reaps only the wrapper and orphans the node, which keeps publishing
    under the same name/topics (every test here uses ``/local_costmap/costmap_raw``) and
    corrupts later tests. Signalling the whole group is what actually stops it — confirmed via
    ``pgrep -f nav2_costmap_2d`` returning nothing after a run.
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


def _lifecycle(
    transition: str, *, costmap_node: str = _COSTMAP_NODE, timeout_s: float = 45.0
) -> None:
    """Drive the costmap through ``transition``, waiting for it to be ready.

    Retried, not one-shot: ``change_state`` takes a moment to advertise after ``Popen``
    returns, and ``activate`` fails until ``base_link -> odom`` TF is discovered — both
    startup races, not behaviour. Stops on first success. ``costmap_node`` defaults to this
    file's node; parameterized so ``test_nav2_global_costmap_height_live.py`` can reuse the
    same retry loop.
    """
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["ros2", "lifecycle", "set", costmap_node, transition],
            check=False,
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0:
            return
        last = (result.stdout + result.stderr).decode(errors="replace").strip()
        time.sleep(0.5)
    raise AssertionError(
        f"{costmap_node} never accepted `{transition}` within {timeout_s}s: {last}"
    )


def _static_transforms(node: Any) -> Any:
    """``odom -> base_link -> {panda_link7, base_scan}``, the frames the nodes need.

    Costmap needs ``base_link -> odom`` to activate; footprint publisher needs the attach
    link; scan filter needs ``base_scan -> panda_link7``. Lidar is placed at the attach link's
    height so the scan plane cuts through the carried object. TRANSIENT_LOCAL, so late-joining
    subprocesses still get these.
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

    Same sensor picture, two attachment states: attached, returns must not mark the costmap;
    released, the same returns must mark it — else the filter is blinding Nav2, not removing
    a payload. Payload sits 0.75 m ahead, outside the 0.35 m chassis, so
    ``footprint_clearing_enabled`` (default True) cannot be what keeps the cell free.
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

#: ``self_tf_grace_s`` for the tests that want the #212 startup gate to open
#: again within a test's patience. The shipped default is 5.0 s, sized for a
#: real bringup where TF and the first scans arrive together.
_SHORT_TF_GRACE_S = 6.0
#: What option 1 of #212 would have written for a beam the chassis half
#: dropped: ``range_max`` less nav2's own ``inf``-substitution epsilon, so
#: ``laser_geometry`` keeps the point (its cutoff is strict ``<``) and
#: ``ObstacleLayer`` raytraces the bearing clear without marking the endpoint.
_RANGE_MAX_BEAM_M = 12.0 - 0.0001

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
    # Matches manifest range_min_m since #194 lowered it to the sensor minimum (was 0.55 m,
    # a radial cutoff sized to hide the chassis — the blunt instrument this filter replaces).
    # This fixture already ignored that cutoff: gating here would hide the returns under test.
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

    Yields ``(executor, publish_of, samples, filter_log, latest)``: ``samples[probe]`` is the
    cost at that probe point on each costmap update, ``latest`` is the most recent whole
    ``Costmap``, ``publish_of(scan)`` returns a callable republishing that scan alongside
    ``state``.

    ``state`` defaults to an empty ``WorldStateStamped`` so the payload half of the filter is
    provably not doing the removing; pass an attachment to exercise that half.

    ``warmup_scan``: the filter fails open until its TF buffer has ``base_frame <- scan_frame``,
    and a self-return that reaches the grid even once is permanent (measured: 32 cells from
    one unfiltered ring survived 20 s of all-``inf`` filtered scans, clearing only once a real
    return hit the same bearings). Callers asserting steady state pass their scan here; the
    costmap isn't configured until the filter's own output proves it is live.
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
                # Costmap process isn't started until this returns — a gate that only delays
                # `configure` would leave a window where an unfiltered scan hits an existing
                # costmap process, and one mark inside the chassis is permanent (see this
                # function's docstring).
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

    Unfiltered, a 2-D lidar's self-returns become permanent costmap obstacles — a failure
    sim hides this, since ``synthesize_laser_scan_2d`` re-casts through the robot's own
    MuJoCo tree and never emits a self-return. Same bearing, twice: forward beams at 0.20 m
    (inside the chassis) must vanish; at 0.60 m (0.25 m past it) must reach the grid untouched.
    Both phases run with an empty attachment set, so the payload half is provably not acting.
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
    """The fail-closed direction, and the control for the sibling test above.

    Same node/scan/costmap but ``base_frame`` is one nobody broadcasts, so every
    ``base_frame <- base_scan`` lookup fails; a self-filter that cannot place the chassis must
    remove nothing — the 0.20 m return must reach the grid unchanged (else something else,
    e.g. footprint clearing, was keeping that cell free, invalidating the sibling test too).

    Since #212 the pass-through isn't immediate — the node withholds every scan until chassis
    TF resolves once — so this also measures the bounded half of that gate: with a TF that
    never resolves, ``self_tf_grace_s`` must expire and hand scans back, or a mistyped
    ``base_frame`` would blind the costmap forever.
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
        "-p",
        f"self_tf_grace_s:={_SHORT_TF_GRACE_S}",
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
        # Falling back silently would be the worse bug of the two: Nav2 is
        # being handed scans whose chassis returns are permanent.
        assert "UNFILTERED" in filter_log.read_text(errors="replace"), (
            "the grace window expired without saying so; filter said:\n"
            f"{filter_log.read_text(errors='replace')[-2000:]}"
        )


def test_the_filter_withholds_every_scan_until_it_can_place_the_chassis(
    tmp_path: Path,
) -> None:
    """The #212 startup gate: no output at all rather than one unfiltered scan.

    The self half can't fail open like the payload half: a payload return is cleared by the
    next scan's ray on the same bearing, but a chassis return isn't (the working filter
    removes exactly that ray), so one unfiltered scan during TF warm-up is permanent — which
    is what failed #207's silhouette sweep in CI. So the node publishes nothing while a
    self-polygon is configured and ``base_frame <- scan_frame`` has never resolved (measured
    with a ``base_frame`` nobody broadcasts, holding the gate shut for the whole grace window).

    Second half proves non-vacuity and boundedness: the window is bounded, so a mistyped
    ``base_frame`` costs only ``self_tf_grace_s`` of blindness, not all of it — if the
    subscription had never matched, no scan would arrive after the grace either.
    """
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import LaserScan

    # Comfortably inside the grace window even after the node's own startup,
    # which is what pushes the window's start later, never earlier.
    withheld_window_s = 4.0
    filter_argv = [
        sys.executable,
        str(_SCAN_FILTER_NODE),
        "--ros-args",
        "-p",
        f"robot_yaml:={_ROBOT_YAML}",
        "-p",
        f"base_frame:={_UNRESOLVABLE_BASE_FRAME}",
        "-p",
        f"self_tf_grace_s:={_SHORT_TF_GRACE_S}",
    ]

    rclpy.init()
    node = Node("test_nav2_self_filter_startup_gate")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        received: list[LaserScan] = []
        node.create_subscription(LaserScan, _FILTERED_SCAN_TOPIC, received.append, sensor_qos)
        scan_pub = node.create_publisher(LaserScan, "/scan", sensor_qos)
        scan = _forward_scan(_SELF_RETURN_M)

        def _publish() -> None:
            scan.header.stamp = node.get_clock().now().to_msg()
            scan_pub.publish(scan)

        filter_log = tmp_path / "startup_gate_filter.log"
        with _process(filter_argv, filter_log):
            deadline = time.monotonic() + withheld_window_s
            while time.monotonic() < deadline:
                _publish()
                executor.spin_once(timeout_sec=0.05)
            assert not received, (
                f"the filter published {len(received)} scan(s) it could not place the chassis "
                f"in — every chassis return in them is a permanent costmap mark (#212); "
                f"filter said:\n{filter_log.read_text(errors='replace')[-2000:]}"
            )

            assert _spin_until(executor, lambda: bool(received), timeout_s=25.0, each=_publish), (
                f"the filter never published after its {_SHORT_TF_GRACE_S}s grace expired, so "
                f"Nav2 would be blind for good; filter said:\n"
                f"{filter_log.read_text(errors='replace')[-2000:]}"
            )
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_raytrace_clearing_a_dropped_beam_would_erase_a_real_obstacle(tmp_path: Path) -> None:
    """Why a dropped beam stays ``inf`` — the measurement that rejects #212 option 1.

    Option 1 (publish ``range_max`` instead of ``inf`` for a dropped beam, so Nav2 raytraces
    it clear) is wrong because Nav2 clears the ray out to ``raytrace_max_range`` (3.0 m in the
    shipped config, well past the chassis) — not just the endpoint — and a chassis-occluded
    bearing never gets re-marked, so that clears cells marked from other robot poses too.

    Measured with the exact bytes option 1 would emit: a real obstacle 0.25 m past the
    chassis edge, already lethal, is deleted by one ``range_max`` beam on its bearing (the
    filter passes 12 m through untouched, since it's outside any chassis) — pinning Nav2's
    own behaviour rather than guessing at it.
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
        # 1. A real obstacle on the forward bearings, marked lethal.
        beyond = samples[_OBSTACLE_AT_SAME_BEARING_M]
        obstacle = publish_of(_forward_scan(_OBSTACLE_AT_SAME_BEARING_M))
        assert _spin_until(
            executor,
            lambda: bool(beyond) and beyond[-1] >= lethal_threshold,
            timeout_s=25.0,
            each=obstacle,
        ), (
            f"the obstacle never marked, so the erasure below would prove nothing "
            f"(last cost {beyond[-1:]}); filter said:\n"
            f"{filter_log.read_text(errors='replace')[-2000:]}"
        )

        # 2. The same bearings, carrying what option 1 emits for a dropped beam.
        beyond.clear()
        healing = publish_of(_forward_scan(_RANGE_MAX_BEAM_M))
        assert _spin_until(
            executor,
            lambda: bool(beyond) and beyond[-1] == 0,
            timeout_s=25.0,
            each=healing,
        ), (
            f"expected the range_max beam to raytrace the obstacle away "
            f"(last cost {beyond[-1:]}) — if it no longer does, #212 option 1 is worth "
            f"re-opening; filter said:\n{filter_log.read_text(errors='replace')[-2000:]}"
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

    Uses the shipped node's own predicates — ``base_footprint_polygon`` off the real
    ``robots/panda_mobile/robot.yaml``, ``points_in_primitive`` for the carried box — so this
    measures the same geometry the filter measures. Evaluated in the scan plane
    (``_SCAN_Z_IN_BASE``), the payload box's own centre height, so this is its full ground
    projection. ``odom -> base_link`` is identity in this rig, so no transform is needed.
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

    "No floating or self obstacles" is a whole-silhouette claim the probe-point tests above
    can't make (a mark on some other cell inside the robot still passes them). So this drives
    355 beams off the chassis at every bearing (five off a carried box) through the shipped
    filter into a real ``nav2_costmap_2d``, sweeps every published cell, and asserts zero are
    marked inside the chassis ∪ payload silhouette.

    Asserts on steady state: the rig withholds the costmap until the filter's own output
    proves it is filtering, since a self-return reaching the grid once is permanent (a real
    fail-open transient, documented in the package README, not hidden here).
    ``footprint_clearing_enabled`` is off (see ``_COSTMAP_PARAMS_SELF_RETURNS``), so Nav2's own
    clearing can't be what keeps this clean, reproducing ``collision_monitor`` — the one
    consumer with no costmap-side clearing at all.

    ADR-0099: with Nav2 base-only, nothing grows the footprint over a payload, so the scan
    filter is the only thing keeping the robot and its payload out of Nav2's world.
    """
    filter_argv = [
        sys.executable,
        str(_SCAN_FILTER_NODE),
        "--ros-args",
        "-p",
        f"robot_yaml:={_ROBOT_YAML}",
    ]

    ring = _ring_scan(payload_x_m=_PAYLOAD_X_IN_BASE)
    # The rig cannot publish while `ros2 lifecycle set` blocks, and the node's default 0.5 s
    # `attached_state_timeout_s` expires inside that window — the first scan after activation
    # could find no attachment and mark the payload silhouette permanently. The sweep measures
    # containment, not attachment freshness, so the timeout is widened past the window.
    filter_argv += ["-p", "attached_state_timeout_s:=30.0"]
    with _self_filter_rig(
        tmp_path,
        filter_argv=filter_argv,
        state=_world_state(carrying=True, revision=1),
        # The costmap is not configured until this scan comes back filtered — without that
        # gate the rig could feed one unfiltered ring during TF warm-up, and those marks never
        # clear (see the rig's docstring and the package README's "permanent self-marks" note).
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

    Same ring, same costmap, same assertion — with the filter given no ``robot_yaml``, the
    documented degradation to "the robot's own returns are not filtered". The chassis must
    then mark its own silhouette, or "zero marked cells" is a claim the rolling window, the
    range gate, or an unwired topic could satisfy on their own — not hypothetical: #183 found
    this file's payload test passing vacuously because the deploy image never built the
    package the filter lives in.
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
