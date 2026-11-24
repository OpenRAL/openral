# SPDX-License-Identifier: Apache-2.0
"""Regression: the terminal task-success verdict must survive a signal teardown.

``SimAttachedHAL.disconnect`` emits ``sim.task_success_final`` — the one
greppable record of whether a ``deploy sim`` session finished its task. It
fired only from the lifecycle ``cleanup``/``shutdown`` transition, but
``rclpy`` runs neither on SIGINT (its handler shuts the context and raises
out of ``spin``); the deploy harness tears down exactly that way
(``openral_cli.deploy_sim._terminate_launch_group`` SIGINTs the launch's
process group), so the verdict went unemitted in every field run while
disconnect-calling unit tests stayed green.

``HALLifecycleNodeBase.shutdown_hal`` is the teardown both ``main()``
factories now run in ``finally``; these tests invoke it directly.

Real components (CLAUDE.md §1.11): a real node/context, a real
``SimAttachedHAL`` over a real manifest; ``SimRollout`` is the ``FakeSimEnv`` recorder.
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
    from openral_hal.protocol import HAL
    from openral_hal.sim_attached import SimAttachedHAL

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))

    class _SimHALNode(HALLifecycleNodeBase):
        def _create_hal(self) -> HAL:
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


def test_signal_teardown_releases_viewers_before_the_hal_disconnects() -> None:
    """`shutdown_hal` runs `on_signal_shutdown_pre_disconnect` first, then disconnects.

    The signal path runs no lifecycle transition, so the sensor bridge's teardown
    (which closes the MuJoCo passive viewer) never ran: the viewer thread outlived
    `main()`, and GLFW's atexit `terminate` aborted (-6) or segfaulted (-11) the HAL
    process on every twin (#742). The manifest node closes its viewer in this hook.
    """
    import rclpy

    rclpy.init()
    calls: list[str] = []
    env = _sim_env(task_success_value=True)
    node = _node(env, "openral_hal_shutdown_viewer_order_test")
    node.on_signal_shutdown_pre_disconnect = lambda: calls.append("pre_disconnect")
    try:
        assert str(node.trigger_configure()).endswith("SUCCESS")
        hal = node._hal
        disconnect = hal.disconnect

        def recorded_disconnect() -> None:
            calls.append("disconnect")
            disconnect()

        hal.disconnect = recorded_disconnect
        node.shutdown_hal()
        node.shutdown_hal()  # idempotent: the hook does not run again without a HAL
        assert calls == ["pre_disconnect", "disconnect"]
    finally:
        node.destroy_node()
        rclpy.shutdown()
