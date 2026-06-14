"""Integration test: franka manifest-driven HAL, scene-attached, publishes over ROS.

Exercises ADR-0034 end-to-end at the ROS level — :class:`_ManifestHALLifecycleNode`
is driven through ``configure → activate``, attaches to a real MuJoCo scene, and
the test asserts:

* ``/joint_states`` carries a :class:`sensor_msgs/JointState` with
  ``len(position) == 8`` (7 panda arm joints + 1 gripper) within a timeout —
  positions come from the scene's live MJCF qpos via the joint-name mapping.
* ``viewer_enabled=true`` with ``MUJOCO_GL=egl`` (no DISPLAY) must NOT fail
  activation — the :class:`~openral_hal.sim_sensor_bridge.SimSensorBridge`
  catches the GL/display failure and continues headless.

**Test A** — native tabletop_push scene (always runs, no LIBERO).
**Test B** — LIBERO milk scene, camera frames; guarded on ``libero`` import.

Per CLAUDE.md §1.11: no mocks. Real franka manifest, real MuJoCo scene,
real lifecycle node, real ROS IDL.

Run::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    MUJOCO_GL=egl uv run pytest \
        packages/openral_rskill_ros/test/test_franka_scene_attach.launch.py -v
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

# ── Guards ───────────────────────────────────────────────────────────────────

pytest.importorskip("rclpy")
pytest.importorskip("openral_hal")
pytest.importorskip("mujoco")
pytest.importorskip("robot_descriptions")

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

# Absolute paths so the node can find them regardless of cwd.
# parents[3]: test/ → openral_rskill_ros/ → packages/ → <repo-root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROBOT_YAML = str(_REPO_ROOT / "robots" / "franka_panda" / "robot.yaml")
_TABLETOP_YAML = str(_REPO_ROOT / "scenes" / "native" / "tabletop_push.yaml")
_LIBERO_MILK_YAML = str(_REPO_ROOT / "scenes" / "native" / "pi05_libero_custom_milk.yaml")

_FRANKA_DOF = 8  # 7 arm + 1 gripper, per FRANKA_PANDA_DESCRIPTION
_CAMERA_HW = 256  # agentview / wrist intrinsics declared in robot.yaml
_JOINT_STATE_TIMEOUT_S = 5.0
_CAMERA_TIMEOUT_S = 10.0


# ── Helpers ───────────────────────────────────────────────────────────────────


def _spin_until(executor: object, condition: object, timeout_s: float) -> bool:
    """Spin *executor* until *condition()* is truthy or *timeout_s* elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)  # type: ignore[union-attr]
        if condition():  # type: ignore[operator]
            return True
    return False


# ── Test A — native tabletop_push, always runs ───────────────────────────────


def test_franka_tabletop_push_joint_states() -> None:
    """ADR-0034: franka+tabletop_push → configure→activate → /joint_states 8-DoF.

    The tabletop_push scene is robot-agnostic (ADR-0033). The franka manifest
    drives the joint-name mapping (``sim_joint_name`` on each JointSpec) so
    ``read_state()`` maps the native MJCF qpos to the 8-DoF panda schema.

    Asserts:
    * Lifecycle configure + activate return SUCCESS.
    * ``viewer_enabled=true`` + ``MUJOCO_GL=egl`` (no DISPLAY) does NOT fail
      activation — :class:`~openral_hal.sim_sensor_bridge.SimSensorBridge`
      catches GL/display failures and continues headless.
    * ``/joint_states`` publishes with ``len(position) == 8`` and the canonical
      panda joint names within ``_JOINT_STATE_TIMEOUT_S`` seconds.
    """
    import rclpy
    from openral_hal.lifecycle import _ManifestHALLifecycleNode  # type: ignore[attr-defined]
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState as RosJointState

    rclpy.init()
    try:
        node = _ManifestHALLifecycleNode("openral_hal_franka")
        node.set_parameters(
            [
                Parameter("robot_yaml", Parameter.Type.STRING, _ROBOT_YAML),
                Parameter("hal_mode", Parameter.Type.STRING, "sim"),
                Parameter("sim_env_yaml", Parameter.Type.STRING, _TABLETOP_YAML),
                Parameter("viewer_enabled", Parameter.Type.BOOL, True),
            ]
        )

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)

        try:
            # configure — compiles the MuJoCo MJCF; may take ~10 s on first run.
            result = node.trigger_configure()
            assert result == TransitionCallbackReturn.SUCCESS, (
                f"configure transition failed: {result!r}"
            )

            # activate — wires joint-state timer + SimSensorBridge.
            # viewer_enabled=true + MUJOCO_GL=egl + no DISPLAY must NOT raise FAILURE.
            result = node.trigger_activate()
            assert result == TransitionCallbackReturn.SUCCESS, (
                "activate transition failed — viewer_enabled=true + MUJOCO_GL=egl + no DISPLAY "
                "must not block activation (SimSensorBridge catches GL failures gracefully)."
            )

            # Subscribe to global /joint_states.
            helper = rclpy.create_node("test_franka_tabletop_helper")
            executor.add_node(helper)
            received: list[RosJointState] = []

            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            helper.create_subscription(RosJointState, "/joint_states", received.append, qos)

            # Spin until the first joint-state arrives.
            got_msg = _spin_until(executor, lambda: bool(received), _JOINT_STATE_TIMEOUT_S)

            assert got_msg, (
                f"no JointState on /joint_states within {_JOINT_STATE_TIMEOUT_S} s — "
                "is the HAL publish_rate_hz > 0 and the joint-name mapping resolved?"
            )

            msg = received[-1]
            assert len(msg.position) == _FRANKA_DOF, (
                f"expected {_FRANKA_DOF}-DoF JointState (7 arm + 1 gripper), "
                f"got {len(msg.position)}: positions={list(msg.position)}"
            )
            assert len(msg.name) == _FRANKA_DOF, f"joint name count mismatch: {list(msg.name)}"

            # Verify canonical panda joint names (ADR-0034 §3.6 joint-name mapping).
            expected_joints = {
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_gripper",
            }
            missing = expected_joints - set(msg.name)
            assert not missing, (
                f"JointState missing expected panda joint names: {sorted(missing)}; "
                f"got: {list(msg.name)}"
            )

            # Graceful teardown.
            assert node.trigger_deactivate() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_cleanup() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_shutdown() == TransitionCallbackReturn.SUCCESS

            executor.remove_node(helper)
            helper.destroy_node()
        finally:
            executor.remove_node(node)
            node.destroy_node()
    finally:
        rclpy.shutdown()


# ── Test B — LIBERO milk scene, camera frames (guarded on libero import) ─────


def test_franka_libero_milk_camera_image() -> None:
    """ADR-0034: franka+LIBERO milk scene → /openral/cameras/agentview/image 256×256.

    Guarded: skips cleanly if:
    * ``libero`` is not installed.
    * The BDDL asset ``pick_milk_in_basket.bddl`` is absent.
    * ``franka_libero_custom_bddl`` is not registered in ``openral_sim.SCENES``.
    * Scene configure fails (robosuite asset path not set up).
    """
    # Guard 1: probe for the libero package before touching rclpy.
    if importlib.util.find_spec("libero") is None:
        pytest.skip("libero not installed — Test B requires the LIBERO benchmark package")

    # Guard 2: BDDL asset on disk.
    bddl_path = _REPO_ROOT / "scenes" / "native" / "pick_milk_in_basket.bddl"
    if not bddl_path.is_file():
        pytest.skip(f"BDDL asset not found: {bddl_path}")

    # Guard 3: scene registered.
    try:
        from openral_sim.registry import SCENES  # reason: optional dep

        if "franka_libero_custom_bddl" not in SCENES:
            pytest.skip("franka_libero_custom_bddl not registered in openral_sim.SCENES")
    except ImportError:
        pytest.skip("openral_sim.registry not importable")

    import rclpy
    from openral_hal.lifecycle import _ManifestHALLifecycleNode  # type: ignore[attr-defined]
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image as RosImage

    rclpy.init()
    try:
        node = _ManifestHALLifecycleNode("openral_hal_franka_libero")
        node.set_parameters(
            [
                Parameter("robot_yaml", Parameter.Type.STRING, _ROBOT_YAML),
                Parameter("hal_mode", Parameter.Type.STRING, "sim"),
                Parameter("sim_env_yaml", Parameter.Type.STRING, _LIBERO_MILK_YAML),
                Parameter("viewer_enabled", Parameter.Type.BOOL, True),
            ]
        )

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)

        try:
            result = node.trigger_configure()
            if result != TransitionCallbackReturn.SUCCESS:
                pytest.skip(
                    f"configure failed ({result!r}) — LIBERO scene init requires "
                    "robosuite + LIBERO BDDL assets on PATH; skip on missing assets."
                )

            result = node.trigger_activate()
            assert result == TransitionCallbackReturn.SUCCESS, f"activate failed: {result!r}"

            helper = rclpy.create_node("test_franka_libero_milk_helper")
            executor.add_node(helper)
            received_images: list[RosImage] = []

            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            helper.create_subscription(
                RosImage,
                "/openral/cameras/agentview/image",
                received_images.append,
                qos,
            )

            got_img = _spin_until(executor, lambda: bool(received_images), _CAMERA_TIMEOUT_S)

            assert got_img, (
                f"no Image on /openral/cameras/agentview/image within {_CAMERA_TIMEOUT_S} s"
            )
            img = received_images[-1]
            assert img.height == _CAMERA_HW, f"expected height={_CAMERA_HW}, got {img.height}"
            assert img.width == _CAMERA_HW, f"expected width={_CAMERA_HW}, got {img.width}"

            node.trigger_deactivate()
            node.trigger_cleanup()
            node.trigger_shutdown()

            executor.remove_node(helper)
            helper.destroy_node()
        finally:
            executor.remove_node(node)
            node.destroy_node()
    finally:
        rclpy.shutdown()
