# SPDX-License-Identifier: Apache-2.0
"""`/openral/estop` must stop a real `ros2_control` controller — proven for every such HAL.

Issue #295: the lifecycle e-stop used to end at `RosControlHAL.estop()` flipping a flag. The
controller that already held a trajectory was never told anything, so for every ros2_control
robot the *physical* stop was unproven. This file runs each adapter's **production lifecycle
node** (`ManifestHALLifecycleNode`, `hal_mode:=real`, the robot's own `robot.yaml`) against
the real stack with only the motors left out — real `controller_manager`, a real
`JointTrajectoryController` per controller the HAL commands, a real `joint_state_broadcaster`,
`mock_components/GenericSystem` as the hardware — and asserts the issue's five points:

1. the downstream stop is emitted **and acknowledged**: `controller_manager` reports the
   HAL's controllers `inactive` after `/openral/estop`, and the HAL's `DownstreamStopReport`
   says `stopped`;
2. the active trajectory cannot continue receiving commands: a `/openral/safe_action` chunk
   published after the stop does not move the (echoing) hardware;
3. `/openral/estop_cleared` follows the declared recovery policy;
4. `RESTART_REQUIRED` adapters (UR5e, UR10e, Franka, Sawyer) stay latched, controllers stay
   inactive, and a later chunk still moves nothing;
5. the `RESETTABLE` adapter (OpenArm) resumes only after `reset_estop` re-activated all four
   controllers — confirmed by the manager — and then a chunk moves the arm again.

The UR vendor stop (`/dashboard_client/stop`) is served by `tests/integration/fakes/
fake_ur_dashboard.py`, a double at the vendor-process boundary CLAUDE.md §1.11 allows; the
Sawyer super-stop is observed on `/robot/set_super_stop` by a real subscriber. What this
cannot prove is the pendant program / FCI reflex / intera state actually stopping — that is
the attended HIL tier (`tests/hil/test_{ur5e,ur10e,franka_panda,sawyer}.py`).

Gates: `ros2` CLI + `rclpy` + ros2_control packages + `openral_msgs` (the node's
`/openral/safe_action` subscriber is the real IDL). Missing any → skips cleanly.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.integration._ros2_control_stack import (
    bring_up,
    unavailable,
    write_controller_config,
    write_generic_system_urdf,
)
from tests.sim.safety._kernel_subprocess import isolated_domain_id

# Confine this graph to its own DDS domain before any rclpy import — the stack's children
# inherit it, and the point is to stay off any real robot's domain. `setdefault` keeps an
# operator's own export authoritative.
os.environ.setdefault("ROS_DOMAIN_ID", str(isolated_domain_id()))

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SKIP_REASON = unavailable() or (
    "openral_msgs not importable — colcon-build the workspace"
    if importlib.util.find_spec("openral_msgs") is None
    else None
)
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))

#: Every ros2_control `hal.real` adapter, by manifest directory, with its declared policy.
#: The fleet conformance test (`tests/unit/test_real_hal_estop_fleet_conformance.py`) is what
#: keeps this list honest: a new ros2_control real HAL fails there until it is added here.
_ADAPTERS = ["ur5e", "ur10e", "franka_panda", "sawyer", "openarm"]

_STOP_TIMEOUT_S = 10.0


def _real_hal(robot: str) -> Any:
    """Build the manifest's real HAL off-robot, just to read its wiring."""
    from openral_core import RobotDescription
    from openral_hal.resolver import build_hal

    desc = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / robot / "robot.yaml"))
    return build_hal(desc, mode="real", transport={"require_can_links": False})


def _controller_joints(hal: Any) -> dict[str, list[str]]:
    """Which ros2_control joints each of the HAL's controllers commands."""
    names = hal.controller_names()
    joints = hal.ros2_control_joint_names()
    if len(names) == 1:
        return {names[0]: list(joints)}
    # The bimanual OpenArm splits its 16 joints 7/1/7/1 across four controllers; the HAL
    # publishes each slice under its own topic, so the split is read off its groups.
    groups = hal._command_groups
    return {topic.strip("/").split("/")[0]: list(jn) for topic, _, jn in groups}


def _manifest_for(robot: str, tmp: Path) -> Path:
    """The robot's own manifest, with off-robot construction defaults where needed."""
    src = _REPO_ROOT / "robots" / robot / "robot.yaml"
    if robot != "openarm":
        return src
    # OpenArm's real HAL refuses to connect unless both CAN links are up (a preflight, not a
    # transport). There are no CAN buses here, so the manifest copy declares the preflight
    # off; everything else — controllers, joint names, recovery policy — is untouched.
    data = yaml.safe_load(src.read_text())
    data.setdefault("hal", {}).setdefault("parameters", {}).setdefault("defaults", {})[
        "require_can_links"
    ] = False
    dst = tmp / "robot.yaml"
    dst.write_text(yaml.safe_dump(data, sort_keys=False))
    return dst


@pytest.fixture(params=_ADAPTERS)
def adapter(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[dict[str, Any]]:
    """Bring up the real stack shaped like `request.param`'s controllers, then the node."""
    robot = str(request.param)
    hal = _real_hal(robot)
    controllers = _controller_joints(hal)
    urdf = write_generic_system_urdf(
        tmp_path / f"{robot}.urdf", name=robot, joints=hal.ros2_control_joint_names()
    )
    config = write_controller_config(tmp_path / "controllers.yaml", controllers)
    with bring_up(tmp_path, urdf=urdf, config=config, controllers=list(controllers)):
        yield {
            "robot": robot,
            "controllers": list(controllers),
            "joints": hal.ros2_control_joint_names(),
            "recovery": hal.estop_recovery,
            "vendor_services": hal.vendor_stop_services(),
            "vendor_topics": hal.vendor_stop_topics(),
            "manifest": _manifest_for(robot, tmp_path),
        }


class _Harness:
    """The production node plus the observer / publisher side, on one spinning executor."""

    def __init__(self, robot: str, manifest: Path) -> None:
        """Build the production node, the observer node and their shared executor."""
        import rclpy
        from controller_manager_msgs.srv import ListControllers
        from openral_hal.lifecycle import ManifestHALLifecycleNode
        from openral_msgs.msg import ActionChunk
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import JointState as RosJointState
        from std_msgs.msg import Empty

        rclpy.init()
        self.node: Any = ManifestHALLifecycleNode(f"openral_hal_{robot}_estop_live")
        self.node.set_parameters(
            [
                Parameter("robot_yaml", value=str(manifest)),
                Parameter("hal_mode", value="real"),
                Parameter("publish_rate_hz", value=30.0),
                Parameter("viewer_enabled", value=False),
            ]
        )
        self.helper: Any = rclpy.create_node(f"openral_hal_{robot}_estop_observer")
        self.joint_states: list[Any] = []
        self.super_stops = 0
        state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        safety_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        chunk_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=50,
        )
        self.helper.create_subscription(
            RosJointState, "/joint_states", self.joint_states.append, state_qos
        )
        self.helper.create_subscription(
            Empty, "/robot/set_super_stop", lambda _m: self._count_super_stop(), safety_qos
        )
        self._estop_pub = self.helper.create_publisher(Empty, "/openral/estop", safety_qos)
        self._cleared_pub = self.helper.create_publisher(
            Empty, "/openral/estop_cleared", safety_qos
        )
        self._chunk_pub = self.helper.create_publisher(
            ActionChunk, "/openral/safe_action", chunk_qos
        )
        self._list = self.helper.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        self._empty = Empty
        self._chunk = ActionChunk
        self._list_type = ListControllers
        self._rclpy = rclpy

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.helper)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _count_super_stop(self) -> None:
        """Count one intera super-stop publish."""
        self.super_stops += 1

    def _spin(self) -> None:
        """Executor loop for both nodes until ``close``."""
        while not self._stop.is_set():
            self.executor.spin_once(timeout_sec=0.05)

    def start(self) -> None:
        """Start spinning, then configure + activate the production node."""
        self._thread.start()
        assert str(self.node.trigger_configure()).endswith("SUCCESS"), "configure failed"
        assert str(self.node.trigger_activate()).endswith("SUCCESS"), "activate failed"

    def close(self) -> None:
        """Deactivate + cleanup the node, stop spinning, tear rclpy down."""
        with suppress(Exception):
            self.node.trigger_deactivate()
            self.node.trigger_cleanup()
        self._stop.set()
        self._thread.join(timeout=5.0)
        with suppress(Exception):
            self.executor.shutdown()
        self.helper.destroy_node()
        self.node.destroy_node()
        self._rclpy.shutdown()

    # ── stimuli ────────────────────────────────────────────────────────────

    def publish_chunk(self, targets: list[float]) -> None:
        """Publish one ``ActionChunk`` with ``targets`` on ``/openral/safe_action``."""
        chunk = self._chunk()
        chunk.n_dof = len(targets)
        chunk.horizon = 1
        chunk.flat = [float(v) for v in targets]
        chunk.rskill_id = "openral/test-estop-live"
        self._chunk_pub.publish(chunk)

    def estop(self) -> None:
        """Publish ``/openral/estop``."""
        self._estop_pub.publish(self._empty())

    def estop_cleared(self) -> None:
        """Publish ``/openral/estop_cleared``."""
        self._cleared_pub.publish(self._empty())

    # ── observations ───────────────────────────────────────────────────────

    def positions(self, joints: list[str]) -> dict[str, float] | None:
        """Latest ``/joint_states`` positions for ``joints``; ``None`` before the first message."""
        if not self.joint_states:
            return None
        msg = self.joint_states[-1]
        by_name = dict(zip(msg.name, msg.position, strict=False))
        return {j: float(by_name[j]) for j in joints if j in by_name}

    def controller_states(self) -> dict[str, str]:
        """Ask the manager directly, from the observer side, not through the HAL."""
        assert self._list.wait_for_service(timeout_sec=5.0), "list_controllers unavailable"
        future = self._list.call_async(self._list_type.Request())
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert future.done(), "list_controllers did not answer"
        return {str(c.name): str(c.state) for c in future.result().controller}

    def wait(self, predicate: Any, timeout_s: float) -> bool:
        """Poll ``predicate`` until true or ``timeout_s`` elapses."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return predicate()


def _reached(h: _Harness, joints: list[str], target: list[float]) -> bool:
    """True once every joint reads within 1e-3 rad of ``target``."""
    got = h.positions(joints)
    if got is None or len(got) != len(joints):
        return False
    return max(abs(got[j] - t) for j, t in zip(joints, target, strict=True)) < 1e-3


def _hal_report(h: _Harness) -> Any:
    """The node's HAL's ``last_stop_report`` (the evidence under test)."""
    hal = h.node._hal
    return hal.last_stop_report


def test_estop_reaches_and_stops_the_downstream_controller(
    adapter: dict[str, Any], tmp_path: Path
) -> None:
    """Points 1-5 of issue #295's reproduction, on the adapter's production node."""
    from openral_hal.protocol import EStopRecovery

    from tests.integration.fakes.fake_ur_dashboard import FakeURDashboard

    joints: list[str] = adapter["joints"]
    controllers: list[str] = adapter["controllers"]
    h = _Harness(adapter["robot"], adapter["manifest"])
    # Own node + executor thread, like the separate driver process it stands in for: the
    # e-stop callback blocks the harness executor while waiting for this Trigger.
    dashboard = (
        FakeURDashboard() if "/dashboard_client/stop" in adapter["vendor_services"] else None
    )
    try:
        h.start()
        # The real transport reads the broadcaster; the node must see fresh state before
        # any command is meaningful.
        assert h.wait(
            lambda: h.positions(joints) is not None and len(h.positions(joints)) == len(joints),
            15.0,
        ), (  # type: ignore[arg-type]  # reason: guarded by the None check
            f"joint_state_broadcaster never reported all of {joints}"
        )
        for name in controllers:
            assert h.controller_states().get(name) == "active", h.controller_states()

        # A nontrivial trajectory, and proof it is being executed: the mock echoes commands
        # into state, so the controller reaching the target means it accepted the chunk.
        moving = [0.25 + 0.03 * i for i in range(len(joints))]
        h.publish_chunk(moving)
        assert h.wait(lambda: _reached(h, joints, moving), 10.0), (
            f"the chunk never reached the controller: {h.positions(joints)}"
        )

        # ── 1. stop emitted and acknowledged ─────────────────────────────────
        h.estop()
        assert h.wait(lambda: h.node._estopped, 5.0), "node never latched /openral/estop"
        assert h.wait(lambda: _hal_report(h) is not None, _STOP_TIMEOUT_S), (
            "the HAL produced no DownstreamStopReport"
        )
        report = _hal_report(h)
        assert report.stopped, f"downstream stop NOT acknowledged: {report}"
        assert set(report.controllers) == set(controllers)
        states = h.controller_states()
        for name in controllers:
            assert states.get(name) == "inactive", f"{name}: {states}"
        if dashboard is not None:
            assert dashboard.stop_calls == 1, "UR dashboard stop was not called"
            assert not dashboard.program_running
            assert report.vendor_stop == "/dashboard_client/stop"
        if "/robot/set_super_stop" in adapter["vendor_topics"]:
            assert h.wait(lambda: h.super_stops >= 1, 5.0), "super stop never published"
            assert report.vendor_stop == "/robot/set_super_stop"
        # The diagnostics heartbeat carries the acknowledgement too.
        _level, message, fields = h.node._heartbeat_status(adapter["robot"])
        assert message == "estop latched" and fields["downstream_stop"] == "acknowledged"

        # ── 2. the trajectory cannot continue receiving commands ─────────────
        held = h.positions(joints)
        assert held is not None
        after_stop = [v - 0.2 for v in moving]
        h.publish_chunk(after_stop)
        time.sleep(1.5)
        now = h.positions(joints)
        assert now == held, f"joints moved after the stop: {held} -> {now}"
        assert not _reached(h, joints, after_stop)

        # ── 3-5. recovery follows the declared policy ────────────────────────
        h.estop_cleared()
        if adapter["recovery"] is EStopRecovery.RESTART_REQUIRED:
            time.sleep(1.0)
            assert h.node._estopped, "RESTART_REQUIRED node cleared its latch in process"
            states = h.controller_states()
            for name in controllers:
                assert states.get(name) == "inactive", f"{name} re-activated: {states}"
            h.publish_chunk(after_stop)
            time.sleep(1.0)
            assert h.positions(joints) == held, "a latched node let a chunk through"
        else:
            assert adapter["recovery"] is EStopRecovery.RESETTABLE
            assert h.wait(lambda: not h.node._estopped, _STOP_TIMEOUT_S), (
                "RESETTABLE node did not clear after reset_estop"
            )
            assert _hal_report(h) is None, "the stop report outlived the reset"
            states = h.controller_states()
            for name in controllers:
                assert states.get(name) == "active", f"{name} not re-activated: {states}"
            resumed = [v + 0.1 for v in moving]
            h.publish_chunk(resumed)
            assert h.wait(lambda: _reached(h, joints, resumed), 10.0), (
                f"after reset the chunk never reached the controller: {h.positions(joints)}"
            )
    finally:
        if dashboard is not None:
            dashboard.close()
        h.close()


def test_a_refused_vendor_stop_is_reported_unacknowledged_and_still_latches(
    tmp_path: Path,
) -> None:
    """UR only: the controller is still deactivated, but the report says the stop failed."""
    from tests.integration.fakes.fake_ur_dashboard import FakeURDashboard

    hal = _real_hal("ur5e")
    controllers = _controller_joints(hal)
    urdf = write_generic_system_urdf(
        tmp_path / "ur5e.urdf", name="ur5e", joints=hal.ros2_control_joint_names()
    )
    config = write_controller_config(tmp_path / "controllers.yaml", controllers)
    with bring_up(tmp_path, urdf=urdf, config=config, controllers=list(controllers)):
        h = _Harness("ur5e_refused", _manifest_for("ur5e", tmp_path))
        dashboard = FakeURDashboard(refuse=True, name="fake_ur_dashboard_refusing")
        try:
            h.start()
            joints = hal.ros2_control_joint_names()
            assert h.wait(lambda: h.positions(joints) is not None, 15.0)
            h.estop()
            assert h.wait(lambda: _hal_report(h) is not None, _STOP_TIMEOUT_S)
            report = _hal_report(h)
            assert not report.stopped
            assert "not connected" in report.detail
            assert dashboard.stop_calls == 1
            assert h.controller_states().get("scaled_joint_trajectory_controller") == "inactive"
            assert h.node._estopped
            _level, message, fields = h.node._heartbeat_status("ur5e")
            assert message == "estop latched"
            assert fields["downstream_stop"] == "unacknowledged"
        finally:
            dashboard.close()
            h.close()
