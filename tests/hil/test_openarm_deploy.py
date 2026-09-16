# SPDX-License-Identifier: Apache-2.0
"""Full-graph HIL gate for the real OpenArm v2 bimanual deployment.

Runs `[self-hosted, lab-openarm]`. Like `test_galaxea_a1_deploy.py`, this test
attaches to an **already-running** `openral deploy run` graph rather than
starting one; unlike it, this test **never commands a joint**. Every assertion
is an observation.

That is not conservatism, it is forced. Starting the graph is itself the
motion event on this robot: `deploy_e2e.launch.py` includes
`openral_hal_openarm`'s `real_bringup.launch.py` when `hal_mode:=real`, which
starts upstream `openarm_bringup`, whose `OpenArmHW::on_activate` calls
`enable_all()` and then `return_to_zero()` — an unramped MIT position command
to 0.0 on all seven joints per side, issued *before* the current pose is
sampled, then a 200 x 10 ms ramp to zero. A pytest process must never be what
triggers that, so bringing the cell up stays an operator act with a hand on the
hardware E-stop, and this test proves what the operator brought up:

* both motor buses up and ERROR-ACTIVE,
* `controller_manager` plus all four `JointTrajectoryController`s active,
* real `/joint_states` flowing from the vendor's `joint_state_broadcaster`
  carrying all sixteen ros2_control joints, at rate,
* the TF tree complete from `world` down to both end effectors,
* the C++ safety kernel ACTIVE, its envelope loaded at 16 DoF, and
  deny-by-default — no `safe_action` appears while nothing proposes one,
* the world map populated from the real ZED cloud.

The last one is why `scenes/deploy/openarm_bench.yaml` exists. The manifest's
`head_zed` depth spec auto-enables the octomap leg, but the cloud topic keeps
its sim-only launch default unless the scene pins it, and the result is an
empty octree behind an entirely healthy-looking graph.

Operator procedure (three terminals, one `ROS_DOMAIN_ID`, unique per run --
a finished graph leaves Fast-DDS shm segments that poison the next run on the
same domain into silent zero traffic)::

    ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedm \
        publish_tf:=false publish_map_tf:=false
    openral deploy run --config scenes/deploy/openarm_bench.yaml   # MOVES THE ARM
    OPENARM_DEPLOY_HIL=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        pytest -q tests/hil/test_openarm_deploy.py

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required: sourcing a ROS overlay pulls the
`launch_testing` pytest plugin off system Python, which is incompatible with the
workspace's pytest.

Environment:
    OPENARM_DEPLOY_HIL: Must be ``"1"``. Without it the module skips even on a
        wired rig, so a stray `pytest tests/hil/` never attaches to a live cell.

Gating is skip, never fail: no CAN links means no arm on this host, which is a
fact about the bench and not about the code.
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from openral_hal.openarm_real import OPENARM_REAL_DESCRIPTION

from tests.hil._can_gate import _can_links_up

if TYPE_CHECKING:
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
    from openral_msgs.msg import ActionChunk, OccupancyVoxels
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

_LEFT_CAN = "openarm_left"
_RIGHT_CAN = "openarm_right"
_CAN_LINKS = (_LEFT_CAN, _RIGHT_CAN)

_ENABLED = os.environ.get("OPENARM_DEPLOY_HIL", "0") == "1"
_ROS2_AVAILABLE = importlib.util.find_spec("rclpy") is not None

# ros2_control joint names, not the Skill-facing logical ones. `sim_joint_name`
# carries the URDF name the controllers and `joint_state_broadcaster` use
# (`openarm_left_joint1`, `openarm_left_finger_joint1`), and reading it here
# keeps exactly one copy of that mapping in the repo -- the same source
# `OpenArmRealHAL` fans its 16-DoF action out with.
_EXPECTED_JOINTS = tuple(
    joint.sim_joint_name for joint in OPENARM_REAL_DESCRIPTION.joints if joint.sim_joint_name
)
_N_DOF = len(OPENARM_REAL_DESCRIPTION.joints)

# The four controllers `openarm_bringup`'s bimanual config spawns, each side's
# seven joints and its gripper commanded separately. Pinned against the HAL's
# own constants by `test_openarm_bringup_agreement.py`.
_EXPECTED_CONTROLLERS = (
    "left_joint_trajectory_controller",
    "left_gripper_controller",
    "right_joint_trajectory_controller",
    "right_gripper_controller",
)
_JOINT_STATE_BROADCASTER = "joint_state_broadcaster"

# `joint_state_broadcaster` is spawned at the controller_manager's update rate
# but publishes on its own `publish_rate` (100 Hz in the bimanual config). Floor
# the assertion well under that: this measures "state is genuinely streaming",
# not the vendor's scheduling jitter on a loaded Jetson.
_MIN_JOINT_STATE_HZ = 20.0
_RATE_WINDOW_S = 3.0

_GRAPH_TIMEOUT_S = 25.0
_MAP_TIMEOUT_S = 30.0
_SERVICE_TIMEOUT_S = 10.0
_ESTOP_TIMEOUT_S = 5.0
# A fresh rclpy participant needs a couple of seconds of spinning before DDS
# discovery delivers `/joint_states`. A shorter settle reads as "zero joints
# seen" and looks exactly like a transport bug.
_DISCOVERY_SETTLE_S = 3.0

pytestmark = [
    pytest.mark.skipif(
        not _ENABLED,
        reason="OPENARM_DEPLOY_HIL=1 is required to attach to a live OpenArm deploy graph.",
    ),
    pytest.mark.skipif(
        not _can_links_up(_CAN_LINKS),
        reason=f"{_LEFT_CAN} / {_RIGHT_CAN} are not both up — no OpenArm wired to this host",
    ),
    pytest.mark.skipif(
        not _ROS2_AVAILABLE,
        reason="rclpy is unavailable; source the OpenRAL ROS overlay to run this gate.",
    ),
]


@dataclass
class _Observed:
    """Everything one single-threaded ROS node collected from the live graph."""

    joint_states: list[JointState] = field(default_factory=list)
    joint_state_stamps: list[float] = field(default_factory=list)
    diagnostics: dict[str, DiagnosticStatus] = field(default_factory=dict)
    safe_actions: list[ActionChunk] = field(default_factory=list)
    voxels: list[OccupancyVoxels] = field(default_factory=list)

    def on_joint_state(self, message: JointState) -> None:
        self.joint_states.append(message)
        self.joint_state_stamps.append(time.monotonic())

    def on_diagnostics(self, message: DiagnosticArray) -> None:
        for status in message.status:
            self.diagnostics[status.name] = status

    def on_safe_action(self, message: ActionChunk) -> None:
        self.safe_actions.append(message)

    def on_voxels(self, message: OccupancyVoxels) -> None:
        self.voxels.append(message)


def _diagnostic_values(status: DiagnosticStatus) -> dict[str, str]:
    return {item.key: item.value for item in status.values}


def _spin_until(node: Node, predicate: Callable[[], bool], timeout_s: float) -> bool:
    import rclpy

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
    return False


def _spin_for(node: Node, seconds: float) -> None:
    import rclpy

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


@pytest.fixture(scope="module")
def live_graph() -> Any:  # type: ignore[misc]  # reason: generator fixture, rclpy types are lazy
    """One node attached to the running deploy graph, shared by every assertion.

    Module-scoped on purpose: each `rclpy` participant pays the DDS discovery
    settle once, and one observer is a truer picture of a single graph than six
    independently-racing ones.
    """
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from openral_msgs.msg import ActionChunk, OccupancyVoxels
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = rclpy.create_node("openral_hil_openarm_deploy")
    observed = _Observed()
    control_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    safety_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node, spin_thread=False)
    estop_pub = node.create_publisher(Empty, "/openral/estop", safety_qos)
    node.create_subscription(
        JointState, "/joint_states", observed.on_joint_state, qos_profile_sensor_data
    )
    node.create_subscription(DiagnosticArray, "/diagnostics", observed.on_diagnostics, 10)
    # Subscribed but never published to. An empty list here is the
    # deny-by-default evidence, so the subscription must exist from the start.
    node.create_subscription(
        ActionChunk, "/openral/safe_action", observed.on_safe_action, control_qos
    )
    node.create_subscription(
        OccupancyVoxels, "/openral/world_voxels", observed.on_voxels, qos_profile_sensor_data
    )

    _spin_for(node, _DISCOVERY_SETTLE_S)
    try:
        yield node, observed, tf_buffer
    finally:
        # CLAUDE.md §2: every HIL tier test leaves an e-stop on teardown. This
        # one never moved the arm, so the stop is belt-and-braces rather than a
        # recovery -- but a gate that can attach to a powered bimanual cell and
        # walk away without asking it to stop is not a gate worth shipping.
        try:
            deadline = time.monotonic() + _ESTOP_TIMEOUT_S
            while time.monotonic() < deadline:
                estop_pub.publish(Empty())
                rclpy.spin_once(node, timeout_sec=0.05)
                status = observed.diagnostics.get("openral_hal_openarm")
                if status is not None and _diagnostic_values(status).get("estopped") == "true":
                    break
        finally:
            del tf_listener
            node.destroy_node()
            rclpy.try_shutdown()


def test_graph_is_observable(live_graph: Any) -> None:  # pragma: no cover
    """The deploy graph's own nodes are up before anything else is asserted."""
    node, observed, _ = live_graph
    assert _spin_until(
        node,
        lambda: (
            bool(observed.joint_states)
            and "openral_safety_kernel" in observed.diagnostics
            and "openral_hal_openarm" in observed.diagnostics
        ),
        _GRAPH_TIMEOUT_S,
    ), (
        "OpenArm deploy graph did not become observable within "
        f"{_GRAPH_TIMEOUT_S:.0f} s — seen diagnostics: {sorted(observed.diagnostics)}"
    )


def test_both_motor_buses_are_up_and_the_hal_is_connected(
    live_graph: Any,
) -> None:  # pragma: no cover
    """The HAL's cached preflight agrees with the buses the gate checked."""
    node, observed, _ = live_graph
    assert _spin_until(
        node, lambda: "openral_hal_openarm" in observed.diagnostics, _GRAPH_TIMEOUT_S
    )
    values = _diagnostic_values(observed.diagnostics["openral_hal_openarm"])
    assert values["connected"] == "True"
    assert values["estopped"] == "false"
    # `preflight_can_links` reports a down bus as "<name> (DOWN)", so the bare
    # name is the assertion. A HAL reporting "connected" over a dead motor bus
    # is the worst failure mode there is: every send_action succeeds and the arm
    # never moves.
    assert values["left_can"] == _LEFT_CAN
    assert values["right_can"] == _RIGHT_CAN


def test_controller_manager_has_all_four_trajectory_controllers_active(
    live_graph: Any,
) -> None:  # pragma: no cover
    """Every topic the HAL publishes to has a live controller behind it.

    A `JointTrajectoryController` that is loaded but inactive accepts the
    message and drops it. Nothing on the publisher side raises, nothing on
    `/joint_states` looks unusual, and the arm simply does not move —
    indistinguishable from a policy that chose to hold still.
    """
    node, _, _ = live_graph
    controller_manager_msgs = pytest.importorskip(
        "controller_manager_msgs.srv",
        reason="controller_manager_msgs is unavailable; source the ros2_control overlay",
    )
    client = node.create_client(
        controller_manager_msgs.ListControllers, "/controller_manager/list_controllers"
    )
    try:
        assert client.wait_for_service(timeout_sec=_SERVICE_TIMEOUT_S), (
            "/controller_manager/list_controllers never appeared — openarm_bringup is not running"
        )
        import rclpy

        future = client.call_async(controller_manager_msgs.ListControllers.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=_SERVICE_TIMEOUT_S)
        response = future.result()
        assert response is not None, "list_controllers did not answer"
        by_name = {controller.name: controller.state for controller in response.controller}
        for name in (*_EXPECTED_CONTROLLERS, _JOINT_STATE_BROADCASTER):
            assert by_name.get(name) == "active", (
                f"{name} is {by_name.get(name, 'absent')!r}, expected 'active'; "
                f"controller_manager reports {by_name}"
            )
    finally:
        node.destroy_client(client)


def test_joint_states_carry_every_joint_at_rate(live_graph: Any) -> None:  # pragma: no cover
    """Real feedback from the vendor's broadcaster, not a stale latched frame."""
    node, observed, _ = live_graph
    assert _spin_until(node, lambda: bool(observed.joint_states), _GRAPH_TIMEOUT_S)

    latest = observed.joint_states[-1]
    assert len(latest.name) == len(set(latest.name)), "duplicate joint name in /joint_states"
    by_name = dict(zip(latest.name, latest.position, strict=True))
    missing = sorted(set(_EXPECTED_JOINTS) - set(by_name))
    assert not missing, f"/joint_states is missing OpenArm joints: {missing}"
    assert all(math.isfinite(by_name[name]) for name in _EXPECTED_JOINTS), (
        "a joint reported a non-finite position — the bus is answering with garbage"
    )

    before = len(observed.joint_states)
    started = time.monotonic()
    _spin_for(node, _RATE_WINDOW_S)
    elapsed = time.monotonic() - started
    rate_hz = (len(observed.joint_states) - before) / elapsed
    assert rate_hz >= _MIN_JOINT_STATE_HZ, (
        f"/joint_states arrived at {rate_hz:.1f} Hz over {elapsed:.1f} s, "
        f"below the {_MIN_JOINT_STATE_HZ:.0f} Hz floor — state is not streaming"
    )


def test_tf_tree_reaches_both_end_effectors(live_graph: Any) -> None:  # pragma: no cover
    """TF2 is the only source of frames, so a hole in it is a silent data loss.

    `world -> openarm_base` is the bridge the manifest declares
    (`assets.urdf.base_to_root_xyz_rpy`); without it nothing publishes
    `openarm_base` and every consumer that asks for it — the sensor mounts, the
    octomap bridge, the kernel's world-collision check — fails quietly.
    """
    node, _, tf_buffer = live_graph
    from rclpy.duration import Duration
    from rclpy.time import Time

    base = OPENARM_REAL_DESCRIPTION.base_frame
    targets = [base, "openarm_left_ee_base_link", "openarm_right_ee_base_link"]
    deadline = time.monotonic() + _GRAPH_TIMEOUT_S
    unresolved = list(targets)
    while time.monotonic() < deadline and unresolved:
        _spin_for(node, 0.2)
        unresolved = [
            frame
            for frame in targets
            if not tf_buffer.can_transform("world", frame, Time(), Duration(seconds=0.0))
        ]
    assert not unresolved, f"TF tree incomplete — world -> {unresolved} did not resolve"


def test_safety_kernel_is_active_and_denies_by_default(
    live_graph: Any,
) -> None:  # pragma: no cover
    """The kernel is up at the right DoF and emits nothing unbidden.

    Deny-by-default is the property under test: this node publishes no
    candidate action at any point, so any `safe_action` on the wire would mean
    the kernel is a source of commands rather than a filter on them.
    """
    node, observed, _ = live_graph
    assert _spin_until(
        node, lambda: "openral_safety_kernel" in observed.diagnostics, _GRAPH_TIMEOUT_S
    )
    kernel = observed.diagnostics["openral_safety_kernel"]
    values = _diagnostic_values(kernel)
    assert values["envelope_loaded"] == "true", (
        f"kernel envelope not loaded: {kernel.message!r} {values}"
    )
    assert values["n_dof"] == str(_N_DOF), (
        f"kernel is configured for {values['n_dof']} DoF, expected {_N_DOF}"
    )

    lifecycle_msgs = pytest.importorskip(
        "lifecycle_msgs.srv", reason="lifecycle_msgs is unavailable; source the ROS overlay"
    )
    client = node.create_client(lifecycle_msgs.GetState, "/openral_safety_kernel/get_state")
    try:
        assert client.wait_for_service(timeout_sec=_SERVICE_TIMEOUT_S)
        import rclpy

        future = client.call_async(lifecycle_msgs.GetState.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=_SERVICE_TIMEOUT_S)
        response = future.result()
        assert response is not None
        assert response.current_state.label == "active", (
            f"safety kernel is {response.current_state.label!r}; an INACTIVE kernel "
            "passes nothing and the arm is dead, but a MISSING one is worse"
        )
    finally:
        node.destroy_client(client)

    _spin_for(node, 1.0)
    assert not observed.safe_actions, (
        f"{len(observed.safe_actions)} safe_action(s) appeared with no candidate published — "
        "the kernel is not deny-by-default"
    )


def test_world_map_populates_from_the_real_depth_cloud(
    live_graph: Any,
) -> None:  # pragma: no cover
    """The scene's `octomap_cloud_topic` actually reaches the kernel's grid.

    The failure this exists to catch is silent by construction: with the launch
    default topic (`/openral/cameras/front_depth/points`, published only by the
    sim sensor bridge) `octomap_server`, the voxel bridge and the dashboard all
    report healthy and the octree is simply empty forever.
    """
    node, observed, _ = live_graph
    assert _spin_until(node, lambda: bool(observed.voxels), _MAP_TIMEOUT_S), (
        f"no OccupancyVoxels on /openral/world_voxels within {_MAP_TIMEOUT_S:.0f} s — "
        "check that the ZED driver is up on this ROS_DOMAIN_ID and that the scene's "
        "runtime.octomap_cloud_topic matches what it publishes"
    )
    grid = observed.voxels[-1]
    assert grid.resolution > 0.0
    expected_cells = grid.size_x * grid.size_y * grid.size_z
    assert len(grid.occupancy) == expected_cells, (
        f"grid declares {grid.size_x}x{grid.size_y}x{grid.size_z} = {expected_cells} cells "
        f"but carries {len(grid.occupancy)}"
    )
    # An all-zero quaternion is the unset default and is not a unit quaternion.
    # Every consumer refuses it rather than assuming identity, because a
    # silently wrong orientation is a fail-OPEN misread of the world.
    norm = math.sqrt(
        grid.orientation.x**2
        + grid.orientation.y**2
        + grid.orientation.z**2
        + grid.orientation.w**2
    )
    assert abs(norm - 1.0) < 1e-6, f"voxel grid orientation is not a unit quaternion (|q|={norm})"
    occupied = sum(1 for cell in grid.occupancy if cell)
    assert occupied > 0, (
        "world map is structurally valid but completely empty — this is exactly the "
        "silent octomap-topic failure the bench scene pins `octomap_cloud_topic` to avoid"
    )
