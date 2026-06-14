# Tests · sim helpers

> Part of the OpenRAL [public-symbol inventory](../METHODS.md). Hand-curated; `(LNN)` markers are refreshed by `tools/refresh_methods_linenos.py`.

Shared subprocess + envelope wiring for the C++ safety-kernel digital-twin sim tests. Pulled out of the four per-robot kernel-twin tests in the 2026-05 cleanup (`test_kernel_with_so100_digital_twin.py`, `…_openarm_twin.py`, `…_rizon4_twin.py`, `…_h1_humanoid_twin.py`) so each test stays focused on its embodiment-specific assertions.

### `tests/sim/safety/_kernel_subprocess.py`
_Shared lifecycle helpers for kernel-twin tests._

- `isolated_domain_id() -> int` — Pick an unused `ROS_DOMAIN_ID` so concurrent kernel subprocesses don't cross-talk. (L43)
- `start_kernel(*, domain_id, robot_yaml, ...) -> subprocess.Popen` — Launch the `openral_safety_kernel` binary with the test's robot manifest under an isolated DDS domain. (L133)
- `terminate_kernel(proc, *, sigint_grace_s=2.0) -> None` — SIGINT → SIGKILL teardown. (L206)
- `activate_kernel_node(domain_id, *, node_name="openral_safety_kernel") -> None` — Run the configure → activate lifecycle transitions against the spawned kernel node (uses `ros2 lifecycle set …`). New in this branch; extracted from the four kernel-twin tests so the lifecycle ceremony lives once. (L230)
- `kernel_param_args_from_dict(params) -> list[str]` — Format a dict of kernel parameters as `-p key:=value` argv pairs, for tests that hand-roll specific envelope values rather than drive the kernel from a real `RobotDescription`. (L86)
- `kernel_param_args(robot_description) -> list[str]` — Synthesise the safety envelope from a robot manifest and emit each canonical field as a `--ros-args -p key:=value` argv list (mirrors `sim_e2e.launch.py` in-process, ADR-0020 PR-K). (L101)

