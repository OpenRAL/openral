# Tests · HIL bridges

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

Lab-runner-only `rclpy` bridges that wire the real-HW HAL adapters
(`UR5eRealHAL` / `UR10eRealHAL` / `FrankaPandaRealHAL` / `SawyerRealHAL` /
`AlohaHAL`) onto a live `ros2_control` controller stack via the HALs'
injected `publish_fn` / `state_fn` callables. Off-lab they are guarded
behind `importlib.util.find_spec("rclpy") is None` plus a per-robot env
probe; the unit-lane conformance tests use `SimTransport` instead.

### `tests/hil/_ros_control_transport.py`
_Single-controller bridge. Used by UR5e, UR10e, Franka Panda, Sawyer._

- `_make_trajectory_publisher(node, command_topic) -> Publisher` — Module-private helper that creates a `trajectory_msgs/JointTrajectory` publisher with the shared HIL QoS profile. Reused by `_aloha_ros_transport.py`. (L58)
- `RosControlHILTransport(node, joint_names, *, command_topic, joint_state_topic)` — Subscribes to `joint_state_topic`, exposes `publish(_topic, msg)` for the HAL's outgoing trajectories and `state()` for the cached joint state. Helpers: `spin_once`, `wait_for_first_state`, `last_stamp`. (L68)
- `make_hil_transport(node_name, joint_names, *, command_topic, joint_state_topic) -> tuple[Node, RosControlHILTransport, Callable[[], None]]` — Factory that initialises rclpy, creates the node, and returns a teardown callable. (L179)

### `tests/hil/_aloha_ros_transport.py`
_4-way fan-out bridge for the bimanual ALOHA HAL._

- `AlohaHILTransport(node, joint_names, *, left_arm_command_topic, right_arm_command_topic, left_gripper_command_topic, right_gripper_command_topic, joint_state_topic)` — Owns four `JointTrajectory` publishers (two arms 6-DOF + two grippers 1-DOF — Trossen gripper modeled as a 1-DOF `JointTrajectoryController`) and one aggregated `JointState` subscriber. `publish(topic, msg)` dispatches by topic match; unknown topics raise `ValueError`. (L67)
- `make_aloha_hil_transport(node_name, *, left/right arm + gripper command topics, joint_state_topic) -> tuple[Node, AlohaHILTransport, Callable[[], None]]` — Factory; derives the canonical 14 joint names from the public `ALOHA_REAL_DESCRIPTION.joints` so the bridge stays in lock-step with the HAL manifest. (L228)

