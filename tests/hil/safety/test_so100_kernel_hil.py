"""ADR-0020 PR-I — HIL safety tests for the SO-100 follower arm.

Requires a physically connected SO-100 arm AND a sourced ROS 2
environment with ``openral_safety_kernel`` built. Gated by:

* The presence of ``SO100_PORT`` (default ``/dev/ttyUSB0``).
* The ``ros2`` binary being on PATH (for ``ros2 run``).
* ``openral_msgs`` + ``openral_safety`` importable in the workspace
  venv.

CI label: ``[self-hosted, lab-so100]`` — see
``.github/workflows/hil-so100.yml``.

Safety rules (CLAUDE.md §7.3):

* Each test must be idempotent and ≤120 s per test (most are <20 s).
* The fixture always disconnects the HAL on teardown, including on
  exceptions.
* No test commands the arm faster than 30% of velocity limit.
* The C++ safety kernel runs in a separate subprocess; the parent
  always terminates it on teardown (best-effort SIGTERM then SIGKILL).
"""

from __future__ import annotations

import contextlib
import os
import time
import uuid
from typing import Any

import pytest

from tests.sim.safety._kernel_subprocess import start_kernel, terminate_kernel

SO100_PORT = os.environ.get("SO100_PORT", "/dev/ttyUSB0")

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
pytest.importorskip("openral_hal")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SO100_PORT),
    reason=f"SO-100 not connected on {SO100_PORT}",
)


# Joint limits sourced from robots/so100_follower/robot.yaml; the
# kernel reads each field as a ROS parameter (ADR-0020 PR-K). See
# `tests/sim/safety/_kernel_subprocess.kernel_params_from_envelope`
# for the canonical conversion path used by `openral deploy sim`.
_ENVELOPE_PARAMS_FROM_SO100_MANIFEST: dict[str, object] = {
    "n_dof": 6,
    "robot_name": "so100_follower",
    "joint_position_min": [-2.0944, -1.7453, -1.7453, -1.7453, -2.7925, 0.0],
    "joint_position_max": [2.0944, 1.7453, 1.7453, 1.7453, 2.7925, 1.0],
    "joint_velocity_max": [3.15, 3.15, 3.15, 3.15, 3.15, 3.15],
    "joint_torque_max": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    "workspace_box_min_xyz": [-0.4, -0.4, 0.0],
    "workspace_box_max_xyz": [0.4, 0.4, 0.6],
    "max_ee_speed_m_s": 0.5,
    "max_ee_accel_m_s2": 2.0,
    "max_force_n": 10.0,
    "max_torque_nm": 3.0,
    "contact_force_threshold_n": 5.0,
    "deadman_required": True,
}


def _activate(node_name: str, helper: Any) -> bool:
    import rclpy
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState

    client = helper.create_client(ChangeState, f"/{node_name}/change_state")
    if not client.wait_for_service(timeout_sec=10.0):
        return False
    for t in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
        req = ChangeState.Request()
        req.transition.id = t
        fut = client.call_async(req)
        deadline = time.time() + 5.0
        while time.time() < deadline and not fut.done():
            rclpy.spin_once(helper, timeout_sec=0.05)
        if not fut.done() or not fut.result().success:  # type: ignore[union-attr]
            return False
    return True


@pytest.fixture
def kernel_with_so100_envelope() -> Any:
    """Bring up the C++ safety kernel with the SO-100 envelope file."""
    import rclpy

    node_name = f"so100_safety_kernel_hil_{uuid.uuid4().hex[:8]}"
    domain_id = 50 + (os.getpid() % 50)
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    proc = start_kernel(
        _ENVELOPE_PARAMS_FROM_SO100_MANIFEST,
        node_name,
        domain_id,
        estop_reset_cooldown_s=0.5,
    )
    if True:  # preserve indent of original tempfile context
        time.sleep(1.5)
        rclpy.init()
        helper = rclpy.create_node("so100_hil_helper")
        try:
            assert _activate(node_name, helper), "kernel activation failed"
            yield helper, node_name
        finally:
            with contextlib.suppress(Exception):
                rclpy.shutdown()
            terminate_kernel(proc)


@pytest.fixture
def so100_hal() -> Any:
    """Connect the real SO-100 HAL; disconnect on teardown."""
    from openral_hal.so100_follower import SO100FollowerHAL

    hal = SO100FollowerHAL(port=SO100_PORT, calibrate_on_connect=False)
    hal.connect()
    try:
        yield hal
    finally:
        with contextlib.suppress(Exception):
            hal.disconnect()


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSO100SafetyKernelHIL:
    """Real SO-100 + real C++ kernel + real ROS topic round-trip."""

    def test_kernel_passes_in_range_chunks_with_real_arm_connected(
        self,
        kernel_with_so100_envelope: Any,
        so100_hal: Any,
    ) -> None:
        """Real arm connected + kernel running: in-range chunks republish OK.

        Does NOT actuate the arm. The HAL is connected only to prove the
        port is reachable; the chunks travel from the test publisher
        through the kernel and the test observes /openral/safe_action.
        """
        from openral_msgs.msg import ActionChunk
        from rclpy.executors import SingleThreadedExecutor

        helper, _node_name = kernel_with_so100_envelope
        received: list[ActionChunk] = []
        safe_sub = helper.create_subscription(
            ActionChunk, "/openral/safe_action", received.append, 10
        )
        pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)

        executor = SingleThreadedExecutor()
        executor.add_node(helper)
        # Wait for discovery.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1:
                break
            executor.spin_once(timeout_sec=0.05)
        assert pub.get_subscription_count() >= 1
        assert safe_sub.get_publisher_count() >= 1

        # Sample the arm's current pose; build an ActionChunk holding it
        # verbatim so we don't actuate but exercise the round-trip.
        state = so100_hal.read_state()
        assert len(state.position) == 6
        for i in range(5):
            chunk = ActionChunk()
            chunk.control_mode = 0
            chunk.horizon = 1
            chunk.n_dof = 6
            chunk.flat = list(state.position)
            chunk.trace_id = f"hil-trace-{i:03d}"
            pub.publish(chunk)
            spin_deadline = time.time() + 0.2
            while time.time() < spin_deadline:
                executor.spin_once(timeout_sec=0.02)
        assert len(received) >= 4, f"expected ~5 safe_action republishes, got {len(received)}"

    def test_external_estop_latches_kernel_with_real_arm(
        self,
        kernel_with_so100_envelope: Any,
        so100_hal: Any,
    ) -> None:
        """External /openral/estop → kernel latch → subsequent chunks drop."""
        from openral_msgs.msg import ActionChunk
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import Empty

        helper, _node_name = kernel_with_so100_envelope
        received_safe: list[ActionChunk] = []
        received_estop: list[Empty] = []
        helper.create_subscription(
            ActionChunk,
            "/openral/safe_action",
            received_safe.append,
            10,
        )
        helper.create_subscription(
            Empty,
            "/openral/estop",
            received_estop.append,
            10,
        )
        pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
        estop_pub = helper.create_publisher(Empty, "/openral/estop", 10)

        executor = SingleThreadedExecutor()
        executor.add_node(helper)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if pub.get_subscription_count() >= 1 and estop_pub.get_subscription_count() >= 1:
                break
            executor.spin_once(timeout_sec=0.05)

        # Fire the external estop.
        estop_pub.publish(Empty())
        deadline = time.time() + 1.5
        while time.time() < deadline:
            executor.spin_once(timeout_sec=0.05)

        # Now publish a chunk holding the arm's current pose — kernel
        # should drop it because the latch is set.
        state = so100_hal.read_state()
        chunk = ActionChunk()
        chunk.control_mode = 0
        chunk.horizon = 1
        chunk.n_dof = 6
        chunk.flat = list(state.position)
        chunk.trace_id = "hil-latched"
        pub.publish(chunk)
        deadline = time.time() + 0.5
        while time.time() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert len(received_safe) == 0, (
            "latched kernel must drop subsequent chunks even with real arm connected"
        )

    def test_force_violation_drops_chunk_with_real_arm(
        self,
        kernel_with_so100_envelope: Any,
        so100_hal: Any,
    ) -> None:
        """Inject a torque chunk > envelope: kernel drops + estops.

        The chunk targets joint_torque mode with a torque (10 Nm) that
        exceeds the SO-100 ceiling (5 Nm). The kernel must refuse
        before the HAL ever sees the command — proving the kernel is
        the gate.
        """
        from openral_msgs.msg import ActionChunk, FailureTrigger
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import Empty

        helper, _node_name = kernel_with_so100_envelope
        # Tap state so the assertions below have something to assert on.
        _ = so100_hal.read_state()
        received_safe: list[ActionChunk] = []
        received_fail: list[FailureTrigger] = []
        received_estop: list[Empty] = []
        helper.create_subscription(
            ActionChunk,
            "/openral/safe_action",
            received_safe.append,
            10,
        )
        helper.create_subscription(
            FailureTrigger,
            "/openral/failure/safety",
            received_fail.append,
            50,
        )
        helper.create_subscription(
            Empty,
            "/openral/estop",
            received_estop.append,
            10,
        )
        pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)

        executor = SingleThreadedExecutor()
        executor.add_node(helper)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if pub.get_subscription_count() >= 1:
                break
            executor.spin_once(timeout_sec=0.05)

        bad = ActionChunk()
        bad.control_mode = 2  # JOINT_TORQUE
        bad.horizon = 1
        bad.n_dof = 6
        bad.flat = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # joint 0 torque = 10 Nm > 5 Nm
        bad.trace_id = "hil-force-violation"
        pub.publish(bad)

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if len(received_fail) >= 1 and len(received_estop) >= 1:
                break
            executor.spin_once(timeout_sec=0.05)

        assert len(received_fail) >= 1, "expected FailureTrigger on force violation"
        assert len(received_estop) >= 1, "expected /openral/estop publish on violation"
        assert len(received_safe) == 0, "violating chunk must not be republished"
        ft = received_fail[0]
        assert ft.kind == FailureTrigger.KIND_FORCE
        assert ft.severity == FailureTrigger.SEVERITY_ABORT


# ── Geometric collision (ADR-0030) on the lab SO-100 ───────────────────────────

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SO100_MANIFEST = os.path.join(_REPO_ROOT, "robots", "so100_follower", "robot.yaml")

# Synthetic 3-link arm (capsules along +X, hinge about Z) whose MJCF joint names
# do NOT match the manifest order passed to the lowering — at q=[0,π,0] link1 and
# link3 (non-adjacent, not in the allowed-collision matrix) overlap. Used for the
# self-collision case because the real SO-100's own collision geometry does not
# self-intersect anywhere in its joint limits (a property of the small open arm,
# verified through the kernel) — so a *self*-collision must be exercised with a
# folding model, while the *world* case below uses the real SO-100 geometry.
_TRI_LINK_MJCF = """
<mujoco>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="Alpha" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.36 0 0" size="0.05"/>
      <body name="link2" pos="0.4 0 0">
        <joint name="Beta" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.36 0 0" size="0.05"/>
        <body name="link3" pos="0.4 0 0">
          <joint name="Gamma" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.36 0 0" size="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _so100_world_collision_params() -> dict[str, object]:
    """SO-100 envelope + real (manifest) collision model with the world-obstacle
    check armed. Self-collision is left OFF: the manifest capsules are slightly
    fat and overlap at the home pose, which would mask the world signal — the
    deploy path uses the MJCF model (collision-free), so this isolates the
    world-collision path on the real SO-100 geometry without that artefact."""
    import yaml
    from openral_core import RobotDescription
    from openral_safety.envelope_loader import (
        collision_params_from_description,
        compute_intersection,
        kernel_params_from_envelope,
    )

    with open(_SO100_MANIFEST) as fh:
        robot = RobotDescription.model_validate(yaml.safe_load(fh))
    params: dict[str, object] = dict(kernel_params_from_envelope(compute_intersection(robot, None)))
    params.update(collision_params_from_description(robot))
    params["self_collision_enabled"] = False
    params["world_collision_enabled"] = True
    params["world_collision_margin_m"] = 0.0
    params["world_collision_deadline_ms"] = 5000.0
    params["world_collision_max_primitives"] = 16
    return params


def _synthetic_self_collision_params() -> dict[str, object]:
    """3-link folding model (mismatched MJCF joint names) with self-collision on."""
    mujoco = pytest.importorskip("mujoco")
    from openral_safety.mjcf_lowering import lower_collision_params

    model = mujoco.MjModel.from_xml_string(_TRI_LINK_MJCF)
    collision = lower_collision_params(model, ["j_a", "j_b", "j_c"])
    assert collision["collision_dof_index"] == [0, 1, 2], "MJCF FK froze (dof_index bug)"
    params: dict[str, object] = {
        "n_dof": 3,
        "robot_name": "tri_link",
        "joint_position_min": [-3.2, -3.2, -3.2],
        "joint_position_max": [3.2, 3.2, 3.2],
        "joint_velocity_max": [100.0, 100.0, 100.0],
        "joint_torque_max": [100.0, 100.0, 100.0],
    }
    params.update(collision)
    return params


@pytest.fixture
def kernel_with_params() -> Any:
    """Factory fixture: start the kernel with an arbitrary param dict, activate
    it, yield a spinning helper node, and always terminate on teardown."""
    import rclpy

    procs: list[Any] = []
    helpers: list[Any] = []

    def _start(params: dict[str, object]) -> tuple[Any, str]:
        node_name = f"so100_collision_hil_{uuid.uuid4().hex[:8]}"
        domain_id = 50 + (os.getpid() % 50)
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
        proc = start_kernel(params, node_name, domain_id, estop_reset_cooldown_s=0.2)
        procs.append(proc)
        time.sleep(1.5)
        if not rclpy.ok():
            rclpy.init()
        helper = rclpy.create_node(f"helper_{uuid.uuid4().hex[:8]}")
        helpers.append(helper)
        assert _activate(node_name, helper), "kernel activation failed"
        return helper, node_name

    try:
        yield _start
    finally:
        with contextlib.suppress(Exception):
            rclpy.shutdown()
        for p in procs:
            terminate_kernel(p)


class TestSO100GeometricCollisionHIL:
    """ADR-0030 geometric collision on the real SO-100 lab rig + real C++ kernel.

    Hardware-safe by construction: every case asserts the kernel DROPS the
    candidate chunk (no `/openral/safe_action` republish), so the colliding
    command never reaches the HAL and the arm is never actuated into a
    collision. These do not connect the HAL — the module-level skip already
    guarantees the lab arm is present; the kernel is the unit under test.
    """

    def test_world_obstacle_overlapping_arm_is_rejected(
        self,
        kernel_with_params: Any,
    ) -> None:
        """A world capsule co-located with the SO-100 → KIND_COLLISION (world)."""
        from openral_msgs.msg import ActionChunk, FailureTrigger, WorldCollision
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Empty

        helper, _node = kernel_with_params(_so100_world_collision_params())
        safe: dict[str, ActionChunk] = {}
        failures: list[FailureTrigger] = []
        estops: list[Empty] = []
        safe_sub = helper.create_subscription(
            ActionChunk, "/openral/safe_action", lambda m: safe.__setitem__(m.trace_id, m), 10
        )
        helper.create_subscription(FailureTrigger, "/openral/failure/safety", failures.append, 50)
        helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
        pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
        world_pub = helper.create_publisher(
            WorldCollision,
            "/openral/world_collision",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        executor = SingleThreadedExecutor()
        executor.add_node(helper)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1:
                break
            executor.spin_once(timeout_sec=0.05)

        rest = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]

        def send(trace: str) -> None:
            chunk = ActionChunk()
            chunk.control_mode = 0  # JOINT_POSITION
            chunk.horizon = 1
            chunk.n_dof = 6
            chunk.flat = rest
            chunk.rskill_id = "openral/so100-world-hil"
            chunk.trace_id = trace
            pub.publish(chunk)
            end = time.time() + 1.0
            while time.time() < end:
                executor.spin_once(timeout_sec=0.02)

        def publish_world(x: float, y: float, z: float, r: float, hl: float) -> None:
            wc = WorldCollision()
            wc.header.frame_id = "base"
            wc.header.stamp = helper.get_clock().now().to_msg()
            wc.radius = [r]
            wc.half_length = [hl]
            wc.origin_xyzrpy = [x, y, z, 0.0, 0.0, 0.0]
            wc.object_id = ["lab_obstacle"]
            world_pub.publish(wc)
            end = time.time() + 0.5
            while time.time() < end:
                executor.spin_once(timeout_sec=0.02)

        # A fresh, far obstacle → the home pose passes.
        publish_world(2.0, 0.0, 0.2, 0.05, 0.05)
        send("clear")
        assert "clear" in safe, "home pose must pass with a distant obstacle"
        assert not estops, "no estop on a clear world"

        # Obstacle co-located with the arm base → world collision, dropped + estop.
        publish_world(0.0, 0.0, 0.10, 0.12, 0.10)
        send("collide")
        assert "collide" not in safe, "world-colliding chunk must NOT reach safe_action"
        assert estops, "world collision must fire /openral/estop"
        assert failures, "world collision must publish a FailureTrigger"
        trigger = failures[-1]
        assert trigger.kind == FailureTrigger.KIND_COLLISION
        import json

        evidence = json.loads(trigger.evidence_json)
        assert evidence["collision_kind"] == "world"
        assert evidence["link_b_or_object"] == "lab_obstacle"

    def test_self_colliding_config_is_rejected(
        self,
        kernel_with_params: Any,
    ) -> None:
        """Folded config on a self-folding model → KIND_COLLISION (self).

        The real SO-100's MJCF collision geometry does not self-intersect within
        its joint limits, so this exercises the kernel's self-collision path with
        a 3-link folding model whose MJCF joint names differ from the manifest —
        also a regression guard that the MJCF dof_index mapping keeps the FK live
        (a frozen FK would pass the folded config exactly like the straight one).
        """
        import json

        from openral_msgs.msg import ActionChunk, FailureTrigger
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import Empty

        helper, _node = kernel_with_params(_synthetic_self_collision_params())
        safe: dict[str, ActionChunk] = {}
        failures: list[FailureTrigger] = []
        estops: list[Empty] = []
        safe_sub = helper.create_subscription(
            ActionChunk, "/openral/safe_action", lambda m: safe.__setitem__(m.trace_id, m), 10
        )
        helper.create_subscription(FailureTrigger, "/openral/failure/safety", failures.append, 50)
        helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
        pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
        executor = SingleThreadedExecutor()
        executor.add_node(helper)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1:
                break
            executor.spin_once(timeout_sec=0.05)

        def send(q: list[float], trace: str) -> None:
            chunk = ActionChunk()
            chunk.control_mode = 0
            chunk.horizon = 1
            chunk.n_dof = 3
            chunk.flat = q
            chunk.rskill_id = "openral/so100-self-hil"
            chunk.trace_id = trace
            pub.publish(chunk)
            end = time.time() + 1.0
            while time.time() < end:
                executor.spin_once(timeout_sec=0.02)

        send([0.0, 0.0, 0.0], "straight")
        assert "straight" in safe, "straight arm must pass"
        assert not estops, "straight arm must not estop"

        send([0.0, 3.14159, 0.0], "folded")
        assert "folded" not in safe, "folded self-colliding config must NOT pass"
        assert estops, "self collision must fire /openral/estop"
        assert failures, "self collision must publish a FailureTrigger"
        trigger = failures[-1]
        assert trigger.kind == FailureTrigger.KIND_COLLISION
        evidence = json.loads(trigger.evidence_json)
        assert evidence["collision_kind"] == "self"
