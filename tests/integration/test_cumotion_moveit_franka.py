"""ADR-0065 live e2e — MoveIt planning through the cuMotion pipeline.

Launches a real Panda MoveIt graph with ``isaac_ros_cumotion`` registered as a
planning pipeline, starts NVIDIA's real cuMotion action server on the local GPU,
then drives the in-tree ``rskill-moveit-joints`` manifest through
``JointGoalRskill``. The rSkill's capability gate injects
``MotionPlanRequest.pipeline_id = "isaac_ros_cumotion"``; a passing plan proves
the MoveIt request went through the cuMotion-backed pipeline rather than the
default OMPL demo.

Skipped only when the required live components are absent: ROS 2/openral_msgs,
NVIDIA GPU, Isaac ROS cuMotion packages, or the upstream Panda MoveIt demo
packages. No mocks or fake planners.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set or openral_msgs unavailable; source the ROS 2 overlay first.",
)

pytest.importorskip("rclpy")


def _ros_pkg_available(name: str) -> bool:
    if shutil.which("ros2") is None:
        return False
    result = subprocess.run(
        ["ros2", "pkg", "prefix", name],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0 and result.stdout.strip() != ""


def _require_live_cumotion_stack() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not on PATH; cuMotion e2e requires a local NVIDIA GPU")
    gpu = subprocess.run(
        ["nvidia-smi", "-L"], check=False, capture_output=True, text=True, timeout=10
    )
    if gpu.returncode != 0 or not gpu.stdout.strip():
        pytest.skip("no NVIDIA GPU visible to nvidia-smi; cuMotion e2e requires CUDA hardware")
    missing = [
        pkg
        for pkg in (
            "isaac_ros_cumotion",
            "isaac_ros_cumotion_moveit",
            "isaac_ros_cumotion_robot_description",
            "moveit_resources_panda_description",
            "moveit_resources_panda_moveit_config",
        )
        if not _ros_pkg_available(pkg)
    ]
    if missing:
        pytest.skip(f"missing ROS package(s) required for cuMotion MoveIt e2e: {missing}")


def _wait_for_action(action_name: str, timeout_s: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["ros2", "action", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            continue
        if action_name in result.stdout.splitlines():
            return True
        time.sleep(1.0)
    return False


@contextlib.contextmanager
def _cumotion_moveit_graph() -> Any:
    _require_live_cumotion_stack()
    repo_root = Path(__file__).resolve().parents[2]
    launch_path = repo_root / "tests" / "integration" / "cumotion_panda_moveit_launch.py"
    env = {**os.environ, "ROS_DOMAIN_ID": str(120 + (os.getpid() % 100))}
    os.environ["ROS_DOMAIN_ID"] = env["ROS_DOMAIN_ID"]
    with tempfile.NamedTemporaryFile(prefix="openral-cumotion-moveit-", suffix=".log") as log:
        proc = subprocess.Popen(
            ["ros2", "launch", str(launch_path)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if not _wait_for_action("/cumotion/move_group", timeout_s=90.0):
                log.seek(0)
                pytest.fail(
                    "cuMotion MoveGroup action did not come up. Launch log:\n"
                    + log.read().decode(errors="replace")[-8000:]
                )
            if not _wait_for_action("/move_action", timeout_s=90.0):
                log.seek(0)
                pytest.fail(
                    "MoveIt /move_action did not come up. Launch log:\n"
                    + log.read().decode(errors="replace")[-8000:]
                )
            time.sleep(8.0)
            yield
        finally:
            import signal

            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)


def test_moveit_joints_rskill_plans_with_cumotion_pipeline() -> None:
    """MoveGroup request with cuMotion pipeline id returns replayable waypoints."""
    import rclpy
    from openral_core import RobotDescription, ROSRskillGoalSatisfied, RSkillManifest
    from openral_core.schemas import ControlMode
    from openral_rskill.joint_goal_rskill import JointGoalRskill
    from openral_rskill.ros_action_rskill import CUMOTION_PIPELINE_ID

    repo_root = Path(__file__).resolve().parents[2]
    manifest = RSkillManifest.from_yaml(
        str(repo_root / "rskills" / "rskill-moveit-joints" / "rskill.yaml")
    )
    description = RobotDescription.from_yaml(
        str(repo_root / "robots" / "franka_panda" / "robot.yaml")
    )
    description.capabilities = description.capabilities.model_copy(
        update={
            "gpu_vram_gb": 24.0,
            "cuda_compute_capability": (8, 9),
            "cuda_toolkit_version": "13.2",
        }
    )

    with _cumotion_moveit_graph():
        rclpy.init()
        node: Any = None
        skill: Any = None
        try:
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node

            node = Node("test_cumotion_moveit_rskill_harness")
            skill = JointGoalRskill(
                manifest=manifest,
                ros_node=node,
                robot_description=description,
                prompt="move to home with cuMotion",
                prompt_metadata_json="",
            )
            skill.configure()
            assert skill._goal_dict["request"]["pipeline_id"] == CUMOTION_PIPELINE_ID
            skill.activate()

            import threading

            executor = SingleThreadedExecutor()
            executor.add_node(node)
            spin_thread = threading.Thread(target=executor.spin, daemon=True)
            spin_thread.start()
            try:
                first = skill.step(world_state=None)  # type: ignore[arg-type]
                assert first.control_mode is ControlMode.JOINT_POSITION
                assert first.horizon == 1
                assert len(first.joint_targets[0]) == len(description.joints)

                waypoint_count = 1
                for _ in range(10_000):
                    try:
                        nxt = skill.step(world_state=None)  # type: ignore[arg-type]
                    except ROSRskillGoalSatisfied:
                        break
                    waypoint_count += 1
                    assert nxt.control_mode is ControlMode.JOINT_POSITION
                    assert len(nxt.joint_targets[0]) == len(description.joints)
                else:
                    pytest.fail(f"{skill.name} did not finish after {waypoint_count} waypoints")
                assert waypoint_count >= 1
            finally:
                executor.shutdown()
                spin_thread.join(timeout=2.0)
        finally:
            if skill is not None and skill.state.value not in {"finalized", "error"}:
                skill.shutdown()
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()
