# Tests · sim helpers

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

Shared subprocess + envelope wiring for the C++ safety-kernel digital-twin sim tests, pulled out of the four per-robot kernel-twin tests so each stays focused on its embodiment-specific assertions.

### `tests/sim/safety/_kernel_subprocess.py`
_Shared lifecycle helpers for kernel-twin tests._

- `isolated_domain_id() -> int` — Pick an unused `ROS_DOMAIN_ID` so concurrent kernel subprocesses don't cross-talk. (L43)
- `start_kernel(*, domain_id, robot_yaml, ...) -> subprocess.Popen` — Launch the `openral_safety_kernel` binary with the test's robot manifest under an isolated DDS domain. (L133)
- `terminate_kernel(proc, *, sigint_grace_s=2.0) -> None` — SIGINT → SIGKILL teardown. (L206)
- `activate_kernel_node(domain_id, *, node_name="openral_safety_kernel") -> None` — Run the configure → activate lifecycle transitions against the spawned kernel node (uses `ros2 lifecycle set …`); extracted from the four kernel-twin tests so the lifecycle ceremony lives once. (L230)
- `kernel_param_args_from_dict(params) -> list[str]` — Format a dict of kernel parameters as `-p key:=value` argv pairs, for tests that hand-roll specific envelope values rather than drive the kernel from a real `RobotDescription`. (L86)
- `kernel_param_args(robot_description) -> list[str]` — Synthesise the safety envelope from a robot manifest and emit each canonical field as a `--ros-args -p key:=value` argv list (mirrors `deploy_e2e.launch.py` in-process). (L101)

### `tests/integration/_ros2_control_stack.py`
_A real `ros2_control` graph with only the motors left out — `controller_manager` (`ros2_control_node`), real `JointTrajectoryController`s, a real `joint_state_broadcaster`, `mock_components/GenericSystem` as the hardware. Shared by `tests/sim/test_openarm_hal_ros2_control.py` (the actuation path) and `tests/integration/test_real_hal_estop_ros2_control_live.py` (the e-stop path, issue #295)._

- `unavailable() -> str | None` — Why this host cannot run the stack (`ros2` CLI, `rclpy`, message packages, the five ros2_control packages), or None. Callers `pytest.skip` on it; the dependency is never faked. (L47)
- `write_generic_system_urdf(path, *, name, joints) -> Path` — Minimal serial-chain URDF whose `<ros2_control>` block is `GenericSystem`, one revolute joint per name with a position command interface and position + velocity state interfaces. The mock echoes commands into state, so a joint reads back what it was last told. (L78)
- `write_controller_config(path, controllers, *, update_rate=100) -> Path` — `controller_manager` params: a broadcaster plus one JTC per `{name: joints}` entry. (L117)
- `spawn(argv, log, env)` / `terminate(process)` — One ROS node per process group; SIGINT → SIGKILL teardown that takes `ros2 run`'s child with it. (L140)
- `bring_up(tmp, *, urdf, config, controllers)` [contextmanager] — `robot_state_publisher` + `ros2_control_node`, then the spawner per controller (its exit code is the readiness signal). Children inherit `ROS_DOMAIN_ID`; never sets `ROS_AUTOMATIC_DISCOVERY_RANGE` (see `tests/sim/conftest.py` for the hang). (L163)
- `driver_node(name)` [contextmanager] — `(node, spin)` on a private rclpy context. (L235)
- `wait_until(spin, predicate, timeout_s) -> bool` — Poll, never a bare sleep; DDS discovery takes a beat. (L266)

