"""Live proof that Nav2's **global** costmap marks anything at all (issue #211).

The global costmap was empty. Not sparse — empty: `0` non-zero cells and a
maximum cost of `0` over 101 published samples across two deploy scenes, while
the local costmap on the same graph, the same second and the same scan topic
peaked at `254` with ~2000 non-zero cells. `NavfnPlanner` planned every accepted
`NavigateToPose` goal against a blank 20 x 20 m grid, and nothing warned: a cloud
emptied by the height filter is indistinguishable from a scan with no returns.

The cause is that nav2 filters observation z in the costmap's **own**
`global_frame`, and the two costmaps do not share one. `map`'s z origin is not
the floor — slam_toolbox scan-matches in 2-D and publishes the `map -> odom` z
that flattens `base_link` to 0, and `base_link` is the arm mount 0.700 m up the
pedestal (ADR-0095). So the lidar plane sits at **+0.300 m** in `odom` and
**-0.400 m** in `map`, and a floor of `0.0` keeps every return in the local
costmap and discards every one in the global.

Two tests, and neither means anything without the other:

* the shipped `config/nav2_panda_mobile.yaml` global block marks a real
  obstacle in the `map` frame;
* the same rig with the **pre-#211** height gate restored marks nothing
  anywhere — the regression this PR closes, and the proof that the height gate
  is what the first test measured rather than some incidental difference.

Real components (CLAUDE.md §1.11): the upstream `nav2_costmap_2d` node from
`ros-${ROS_DISTRO}-nav2-bringup`, the **shipped** config file read off disk
rather than a copy of it, and the real `robots/panda_mobile/robot.yaml` for the
`base_link -> base_scan` mount. Nothing here constructs a parameter the
production graph does not use.

The scan filter is deliberately absent. #211 is about a gate applied after the
observation reaches the buffer, so the filter that decides which beams get there
is not part of the claim — `tests/integration/test_nav2_scan_filter_live.py`
owns that half.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

# Reused rather than copied: the process-group teardown, the retried lifecycle
# driver, the spin loop and the cell lookup are the same problem in both files,
# and a second copy of the teardown in particular is how orphaned costmap nodes
# get back in (see that module's `_process` docstring).
from tests.integration.test_nav2_scan_filter_live import (
    _FILTERED_SCAN_TOPIC,
    _LIVE_ROS,
    _LIVE_ROS_REASON,
    _REPO_ROOT,
    _SCAN_FRAME,
    _cost_at,
    _forward_scan,
    _lifecycle,
    _process,
    _spin_until,
)

pytestmark = pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)

_ROBOT_YAML = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
_NAV2_CONFIG = (
    _REPO_ROOT / "packages" / "openral_nav2_bringup" / "config" / "nav2_panda_mobile.yaml"
)

#: Namespaced so the standalone `nav2_costmap_2d` executable's single
#: `Costmap2DROS` lands on the same relative topics the production
#: `nav2_bringup` graph gives the global costmap.
_NAMESPACE = "/global_costmap"
_COSTMAP_NODE = "/global_costmap/costmap"

#: `odom -> base_link`, measured on the live graph and stated in the manifest:
#: the HAL publishes the full RoboCasa `robot0_base_pos`, so `base_link` IS the
#: arm-mount frame 0.700 m up the pedestal, not ground level.
_BASE_Z_IN_ODOM = 0.700
#: `map -> odom`. slam_toolbox scan-matches in 2-D and publishes the correction
#: that puts `base_link` at z = 0 in `map`, which is exactly the negation of the
#: line above. This is the number that makes `map`'s floor -0.700 m.
_ODOM_Z_IN_MAP = -_BASE_Z_IN_ODOM

#: Far enough ahead to be outside the 0.35 m chassis (so
#: `footprint_clearing_enabled` cannot free it) and inside `obstacle_max_range`
#: 2.5 m (so it is allowed to mark).
_OBSTACLE_M = 1.0
_LETHAL_THRESHOLD = 253


def _scan_mount_z() -> float:
    """`base_link -> base_scan` z, from the manifest that owns that geometry.

    Read rather than hardcoded because `sim_e2e.launch.py` publishes this exact
    field as the static TF; a test that duplicated the number could keep passing
    after the mount moved.
    """
    from openral_core import RobotDescription

    description = RobotDescription.from_yaml(_ROBOT_YAML)
    for sensor in description.sensors:
        if sensor.frame_id == _SCAN_FRAME:
            assert sensor.static_transform_xyz_rpy is not None
            return float(sensor.static_transform_xyz_rpy[2])
    raise AssertionError(f"{_ROBOT_YAML} declares no {_SCAN_FRAME!r} sensor")


def _global_costmap_params(*, pre_211_height_gate: bool) -> str:
    """The shipped global costmap block, as a params file for the standalone node.

    Read off `config/nav2_panda_mobile.yaml` so the fix under test is the one
    that ships. Four values are overridden and none of them touches height: the
    grid is shrunk from 20 m to 6 m and run at 5 Hz instead of 1 Hz so the test
    fits the integration budget.

    With ``pre_211_height_gate`` the four height keys are set back to what the
    graph ran before this PR — layer-level `min`/`max` unset (nav2 defaults
    `0.0` / `2.0`) and source-level `0.0` / `2.0`. That is a restoration, not a
    mutilation: it is the exact configuration the empty costmaps were measured
    on.
    """
    raw = yaml.safe_load(_NAV2_CONFIG.read_text())
    params = copy.deepcopy(raw["global_costmap"]["global_costmap"]["ros__parameters"])
    params["width"] = 6
    params["height"] = 6
    params["update_frequency"] = 5.0
    params["publish_frequency"] = 5.0

    if pre_211_height_gate:
        layer = params["obstacle_layer"]
        layer["min_obstacle_height"] = 0.0
        layer["max_obstacle_height"] = 2.0
        layer["scan"]["min_obstacle_height"] = 0.0
        layer["scan"]["max_obstacle_height"] = 2.0

    return yaml.safe_dump({"/**": {"ros__parameters": params}}, sort_keys=False)


def _map_odom_base_scan(node: Any) -> Any:
    """`map -> odom -> base_link -> base_scan`, with the live graph's z values.

    The chain is the whole point of the test, so it is built from the measured
    numbers rather than flattened: the scan lands at +0.300 m in `odom` and
    -0.400 m in `map`, and only a costmap that reads height in `map` can be
    wrong about it.
    """
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import StaticTransformBroadcaster

    broadcaster = StaticTransformBroadcaster(node)
    transforms = []
    for parent, child, z in (
        ("map", "odom", _ODOM_Z_IN_MAP),
        ("odom", "base_link", _BASE_Z_IN_ODOM),
        ("base_link", _SCAN_FRAME, _scan_mount_z()),
    ):
        t = TransformStamped()
        t.header.stamp = node.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.z = z
        t.transform.rotation.w = 1.0
        transforms.append(t)
    broadcaster.sendTransform(transforms)
    return broadcaster


def _run_global_costmap(
    tmp_path: Path, *, pre_211_height_gate: bool, label: str
) -> tuple[list[int], list[int]]:
    """Publish a 1 m obstacle at the global costmap and report what it marked.

    Returns ``(peak cost at the obstacle per sample, non-zero cells per sample)``
    — the two quantities `tools/_nav2_costmap_silhouette_probe.py` reports as
    `max_cost_seen` and `nonzero_cells_max` on the real scenes.
    """
    import rclpy
    from nav2_msgs.msg import Costmap
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import LaserScan

    params_file = tmp_path / f"global_costmap_{label}.yaml"
    params_file.write_text(_global_costmap_params(pre_211_height_gate=pre_211_height_gate))

    costs: list[int] = []
    nonzero: list[int] = []

    rclpy.init()
    node = Node("test_nav2_global_costmap_height")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def _on_costmap(msg: Costmap) -> None:
        costs.append(_cost_at(msg, _OBSTACLE_M, 0.0))
        nonzero.append(sum(1 for cell in msg.data if cell))

    try:
        broadcaster = _map_odom_base_scan(node)
        assert broadcaster is not None
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        node.create_subscription(Costmap, f"{_NAMESPACE}/costmap_raw", _on_costmap, 1)
        scan_pub = node.create_publisher(LaserScan, _FILTERED_SCAN_TOPIC, sensor_qos)
        scan = _forward_scan(_OBSTACLE_M)

        costmap_argv = [
            "ros2",
            "run",
            "nav2_costmap_2d",
            "nav2_costmap_2d",
            "--ros-args",
            "-r",
            f"__ns:={_NAMESPACE}",
            "--params-file",
            str(params_file),
        ]
        with _process(costmap_argv, tmp_path / f"global_costmap_{label}.log"):
            _lifecycle("configure", costmap_node=_COSTMAP_NODE)
            _lifecycle("activate", costmap_node=_COSTMAP_NODE)

            def _publish() -> None:
                scan.header.stamp = node.get_clock().now().to_msg()
                scan_pub.publish(scan)

            assert _spin_until(
                executor,
                lambda: len(costs) > 40 or (bool(costs) and costs[-1] >= _LETHAL_THRESHOLD),
                timeout_s=40.0,
                each=_publish,
            ), "the global costmap never published a single sample"
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return costs, nonzero


def test_the_global_costmap_marks_an_obstacle_in_the_map_frame(tmp_path: Path) -> None:
    """The shipped config marks, with the lidar plane 0.400 m *below* `map`'s origin.

    This is issue #211's acceptance evidence, deterministically: the probe's
    `max_cost_seen` and `nonzero_cells_max` both above zero, on the same
    parameters the deploy graph loads.
    """
    costs, nonzero = _run_global_costmap(tmp_path, pre_211_height_gate=False, label="shipped")

    assert max(nonzero) > 0, (
        "the global costmap marked nothing anywhere — the #211 regression is back "
        f"(peak cost at the obstacle {max(costs)})"
    )
    assert max(costs) >= _LETHAL_THRESHOLD, (
        f"the 1 m obstacle never reached LETHAL in the map frame (peak {max(costs)}, "
        f"{max(nonzero)} non-zero cells elsewhere)"
    )


def test_the_pre_fix_height_gate_empties_the_global_costmap(tmp_path: Path) -> None:
    """The control: restore the old height gate and the whole grid goes blank.

    Without this the test above proves only that a costmap can mark, not that
    the height gate was what stopped it. With it the pair is a bisect: same
    node, same scan, same TF chain, four parameters different, and the costmap
    goes from `254` to nothing at all.

    It also pins the half that is easy to get wrong — nav2 applies the cut
    **twice**, once in `ObservationBuffer` from `<source>.min_obstacle_height`
    and again in `ObstacleLayer::updateBounds` from the layer-level parameter of
    the same name. Widening only the source pair leaves this test failing.
    """
    costs, nonzero = _run_global_costmap(tmp_path, pre_211_height_gate=True, label="pre211")

    assert max(nonzero) == 0, (
        f"expected the pre-#211 height gate to empty the costmap, but {max(nonzero)} cells "
        "are non-zero — the reproduction no longer reproduces"
    )
    assert max(costs) == 0, f"peak cost at the obstacle was {max(costs)}, expected 0"
