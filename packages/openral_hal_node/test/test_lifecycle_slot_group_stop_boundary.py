"""A slot group cut short by a stop must not outlive it into the next goal.

Thor twin pass (2026-10-03): a kernel stop landed mid-tick, leaving tick 195
staged with 2/4 slots on the MuJoCo twin's HAL. That HAL is latch-only (it does
not opt into ``estop()``), so nothing dropped the partial: it survived
``openral estop reset`` and goal 2's first slot reported it as
``incomplete slot group``. Real lifecycle node, real ``OpenArmMujocoHAL`` (the
twin's adapter, MuJoCo loaded), real ADR-0102 slot actions.
"""

from __future__ import annotations

from typing import cast

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("mujoco")

from openral_core.schemas import Action, ControlMode
from openral_hal.lifecycle import ManifestHALLifecycleNode
from openral_hal.openarm import OpenArmMujocoHAL
from openral_hal.protocol import HAL, LifecycleEStopHAL

_SESSION = 0x5E55


def _tick(tick: int) -> list[Action]:
    """The four slots `_dispatch_slots` emits for the OpenArm bimanual contract."""
    arms = [
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[[0.0] * 16],
            joint_names=[f"{side}_joint{i}" for i in range(1, 8)],
            tick_index=tick,
            tick_group_size=4,
            runner_session_id=_SESSION,
        )
        for side in ("left", "right")
    ]
    grips = [
        Action(
            control_mode=ControlMode.GRIPPER_POSITION,
            gripper=[0.0],
            ee_name=f"{side}_gripper",
            tick_index=tick,
            tick_group_size=4,
            runner_session_id=_SESSION,
        )
        for side in ("left", "right")
    ]
    return [arms[0], grips[0], arms[1], grips[1]]


@pytest.fixture
def node_and_hal() -> object:
    rclpy.init()
    node = ManifestHALLifecycleNode("openral_hal_openarm_twin")
    hal = OpenArmMujocoHAL()
    hal.connect()
    # The precondition the bug needed: the twin's HAL never receives estop().
    assert not isinstance(hal, LifecycleEStopHAL)
    node._hal = cast(HAL, hal)
    try:
        yield node, hal
    finally:
        hal.disconnect()
        node.destroy_node()
        rclpy.try_shutdown()


def test_a_partial_tick_staged_before_a_stop_is_dropped_by_estop_and_reset(
    node_and_hal: tuple[ManifestHALLifecycleNode, OpenArmMujocoHAL],
) -> None:
    node, hal = node_and_hal
    for slot in _tick(195)[:2]:
        hal.send_action(slot)
    node._on_estop(object())
    node._on_estop_cleared(object())
    assert not node._estopped
    # Goal 2's first tick commits with no incomplete-group error about 195.
    for slot in _tick(196):
        hal.send_action(slot)
    assert hal.last_committed_tick == 196


def test_estop_cleared_alone_drops_a_partial_tick(
    node_and_hal: tuple[ManifestHALLifecycleNode, OpenArmMujocoHAL],
) -> None:
    # A stop whose /openral/estop this node never latched must not leave one either.
    node, hal = node_and_hal
    for slot in _tick(195)[:2]:
        hal.send_action(slot)
    node._on_estop_cleared(object())
    for slot in _tick(196):
        hal.send_action(slot)
    assert hal.last_committed_tick == 196


def test_the_watermark_survives_the_discard(
    node_and_hal: tuple[ManifestHALLifecycleNode, OpenArmMujocoHAL],
) -> None:
    # Discard, not reset: a pre-stop tick replayed after the stop is still stale.
    from openral_core.exceptions import ROSRuntimeError

    node, hal = node_and_hal
    for slot in _tick(194):
        hal.send_action(slot)
    node._on_estop(object())
    node._on_estop_cleared(object())
    with pytest.raises(ROSRuntimeError, match="stale slot group"):
        hal.send_action(_tick(194)[0])
