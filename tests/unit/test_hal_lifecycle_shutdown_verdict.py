# SPDX-License-Identifier: Apache-2.0
"""Regression: the terminal task-success verdict must survive a signal teardown.

``SimAttachedHAL.disconnect`` emits ``sim.task_success_final`` — the one
greppable statement of whether a ``deploy sim`` session ended with the scene's
task completed. It was reached only from the lifecycle ``cleanup`` / ``shutdown``
transition, and ``rclpy`` runs neither on SIGINT: its ``rclpy.init``-installed
handler shuts the context down and raises out of ``spin``. The deploy harness
tears the graph down exactly that way (``openral_cli.deploy_sim``
``_terminate_launch_group`` SIGINTs the launch's process group), so the HAL
process died with the verdict unemitted in every field run — while the unit
tests that call ``disconnect`` directly stayed green.

``HALLifecycleNodeBase.shutdown_hal`` is the teardown both ``main()`` factories
now run in their ``finally``. These tests invoke it directly on a real node.

Real components (CLAUDE.md §1.11): a real ``HALLifecycleNodeBase`` subclass on a
real ``rclpy`` context, a real ``SimAttachedHAL`` over a real ``RobotDescription``
manifest; the ``SimRollout`` boundary is the shared ``FakeSimEnv`` recorder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from openral_core import RobotDescription, SceneSpec, TaskSpec

from tests.unit.fakes.fake_sim_env import FakeSimEnv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT_YAML = _REPO_ROOT / "robots" / "so101_follower" / "robot.yaml"


def _rclpy_available() -> bool:
    try:
        import rclpy
        import rclpy.lifecycle  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _rclpy_available(),
    reason="rclpy / lifecycle not on PYTHONPATH; source a ROS 2 install to run",
)


def _sim_env(*, task_success_value: bool) -> FakeSimEnv:
    """A rollout carrying a real SceneSpec/TaskSpec and a success predicate."""
    return FakeSimEnv(
        action_dim=6,
        has_sim_clock=True,
        sim_dt_ns=20_000_000,
        task_success_value=task_success_value,
        scene=SceneSpec(id="tabletop_push", backend="mujoco"),
        task=TaskSpec(
            id="tabletop_push/push_to_goal",
            scene_id="tabletop_push",
            instruction="push the red cube onto the green goal marker",
        ),
    )


def _final_lines(captured: str) -> list[dict[str, Any]]:
    """The ``sim.task_success_final <json>`` lines mirrored onto stdout."""
    return [
        json.loads(line.split(" ", 1)[1])
        for line in captured.splitlines()
        if line.startswith("sim.task_success_final ")
    ]


def _node(env: FakeSimEnv, name: str) -> Any:
    """A real lifecycle node whose HAL is a real SimAttachedHAL over ``env``."""
    from openral_hal.lifecycle import HALLifecycleNodeBase
    from openral_hal.sim_attached import SimAttachedHAL

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))

    class _SimHALNode(HALLifecycleNodeBase):  # type: ignore[misc, valid-type]
        def _create_hal(self) -> object:
            return SimAttachedHAL(env, description)

    return _SimHALNode(name)


def test_signal_teardown_emits_the_terminal_verdict(capfd: pytest.CaptureFixture[str]) -> None:
    """`shutdown_hal` disconnects the HAL, so SIGINT still states the verdict."""
    import rclpy

    rclpy.init()
    env = _sim_env(task_success_value=True)
    node = _node(env, "openral_hal_shutdown_verdict_test")
    try:
        assert str(node.trigger_configure()).endswith("SUCCESS")
        capfd.readouterr()  # drop the bring-up chatter

        node.shutdown_hal()

        finals = _final_lines(capfd.readouterr().out)
        assert len(finals) == 1, (
            "a signal-driven teardown must still close the session with exactly "
            f"one sim.task_success_final line; got {finals}"
        )
        assert finals[0]["success"] is True
        assert finals[0]["scene_id"] == "tabletop_push"
        assert node._hal is None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_signal_teardown_is_idempotent_with_the_cleanup_path(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Cleanup then shutdown (and shutdown twice) still emit exactly one verdict."""
    import rclpy

    rclpy.init()
    env = _sim_env(task_success_value=False)
    node = _node(env, "openral_hal_shutdown_verdict_idempotent_test")
    try:
        assert str(node.trigger_configure()).endswith("SUCCESS")
        capfd.readouterr()

        # The orderly path first — a lifecycle cleanup already disconnected.
        assert str(node.trigger_cleanup()).endswith("SUCCESS")
        node.shutdown_hal()
        node.shutdown_hal()

        finals = _final_lines(capfd.readouterr().out)
        assert len(finals) == 1, (
            f"the two teardown paths must not double-report the verdict; got {finals}"
        )
        assert finals[0]["success"] is False
    finally:
        node.destroy_node()
        rclpy.shutdown()
