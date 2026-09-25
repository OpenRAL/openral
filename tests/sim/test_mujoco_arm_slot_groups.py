"""ADR-0102 slot groups on the shared ``MujocoArmHAL`` twin, on robots other than OpenArm.

The runner splits a slot-dispatched policy's action into one typed ``Action``
per slot (arm ``JOINT_POSITION`` slots padded to full dof and carrying
``joint_names``, ``GRIPPER_POSITION`` slots addressed by ``ee_name``), all
sharing one ``tick_index`` / ``tick_group_size``. Every ``MujocoArmHAL`` twin
must stage them, commit the tick as ONE step once the last slot lands, and
refuse a group whose tick is not after the last committed one. Before this
lived in the base class only the OpenArm twin did it: ALOHA, Franka and the
rest refused every gripper slot and applied arm slots one padded vector at a
time (audit A.md F3, B.md 3).

Real MuJoCo physics, real manifests (``ALOHA_DESCRIPTION``,
``FRANKA_PANDA_DESCRIPTION``); no mocks.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest

try:
    import mujoco  # noqa: F401
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

try:
    import gym_aloha

    _aloha_mjcf = os.path.join(
        os.path.dirname(gym_aloha.__file__), "assets", "bimanual_viperx_transfer_cube.xml"
    )
    if not os.path.isfile(_aloha_mjcf):
        raise FileNotFoundError(f"missing MJCF at {_aloha_mjcf}")
    _ALOHA_ERROR: str | None = None
except Exception as exc:
    _ALOHA_ERROR = str(exc)

try:
    from robot_descriptions import panda_mj_description as _panda_desc

    _ = _panda_desc.MJCF_PATH  # triggers lazy clone / cache lookup
    _PANDA_ERROR: str | None = None
except Exception as exc:
    _PANDA_ERROR = str(exc)

from openral_core import Action, ControlMode, ROSRuntimeError
from openral_hal import AlohaMujocoHAL, FrankaPandaHAL
from openral_hal._mujoco_arm import MujocoArmHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
]


@dataclass(frozen=True)
class _Case:
    """One robot's slot contract: arm slots by joint name, gripper slots by ee."""

    make: Callable[[], MujocoArmHAL]
    target: tuple[float, ...]  # full-dof, description.joints order
    arm_slots: tuple[tuple[str, ...], ...]
    gripper_slots: tuple[str, ...]


# ALOHA: gym-aloha keyframe "home" plus small arm deltas (all-zeros self-collides).
_ALOHA = _Case(
    make=lambda: AlohaMujocoHAL(gravity_enabled=False, settle_steps=3000),
    target=(0.2, -0.8, 1.0, 0.2, -0.2, 0.1, 0.024, -0.2, -0.8, 1.0, -0.2, -0.2, -0.1, 0.024),
    arm_slots=(
        tuple(
            f"left_{j}"
            for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
        ),
        tuple(
            f"right_{j}"
            for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
        ),
    ),
    gripper_slots=("left_gripper", "right_gripper"),
)

# Franka: a joint+gripper contract (the shape of lingbot-va-galaxea-a1 / LIBERO slots).
_FRANKA = _Case(
    make=lambda: FrankaPandaHAL(gravity_enabled=False, settle_steps=1500),
    target=(0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7, 1.0),
    arm_slots=(tuple(f"panda_joint{i}" for i in range(1, 8)),),
    gripper_slots=("panda_gripper",),
)

_CASES = [
    pytest.param(
        _ALOHA,
        id="aloha",
        marks=pytest.mark.skipif(_ALOHA_ERROR is not None, reason=f"gym-aloha: {_ALOHA_ERROR}"),
    ),
    pytest.param(
        _FRANKA,
        id="franka",
        marks=pytest.mark.skipif(_PANDA_ERROR is not None, reason=f"Panda MJCF: {_PANDA_ERROR}"),
    ),
]


def _group(hal: MujocoArmHAL, case: _Case, tick: int) -> list[Action]:
    """Every slot action of one tick, as ``rskill_runner_node._dispatch_slots`` emits them."""
    names = [j.name for j in hal.description.joints]
    size = len(case.arm_slots) + len(case.gripper_slots)
    out: list[Action] = []
    for slot in case.arm_slots:
        padded = [0.0] * len(names)
        for name in slot:
            padded[names.index(name)] = case.target[names.index(name)]
        out.append(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[padded],
                joint_names=list(slot),
                tick_index=tick,
                tick_group_size=size,
                stamp_ns=time.time_ns(),
            )
        )
    for ee in case.gripper_slots:
        out.append(
            Action(
                control_mode=ControlMode.GRIPPER_POSITION,
                horizon=1,
                gripper=[case.target[names.index(ee)]],
                ee_name=ee,
                tick_index=tick,
                tick_group_size=size,
                stamp_ns=time.time_ns(),
            )
        )
    return out


@pytest.mark.parametrize("case", _CASES)
def test_slot_group_commits_as_one_step_once_the_last_slot_lands(case: _Case) -> None:
    hal = case.make()
    hal.connect()
    try:
        names = [j.name for j in hal.description.joints]
        before = list(hal.read_state().position)
        group = _group(hal, case, tick=1)

        for slot in group[:-1]:
            hal.send_action(slot)
        # Nothing moves before the tick is complete — no half-committed group.
        assert hal.last_committed_tick == 0
        assert list(hal.read_state().position) == pytest.approx(before, abs=1e-6)

        hal.send_action(group[-1])
        assert hal.last_committed_tick == 1
        after = hal.read_state().position
        for slot in case.arm_slots:
            for name in slot:
                i = names.index(name)
                assert after[i] == pytest.approx(case.target[i], abs=5e-3), name
    finally:
        hal.disconnect()


@pytest.mark.parametrize("case", _CASES)
def test_group_of_an_already_committed_tick_is_refused(case: _Case) -> None:
    hal = case.make()
    hal.connect()
    try:
        for slot in _group(hal, case, tick=3):
            hal.send_action(slot)
        assert hal.last_committed_tick == 3

        for stale_tick in (3, 2):
            with pytest.raises(ROSRuntimeError, match="stale slot group"):
                hal.send_action(_group(hal, case, tick=stale_tick)[0])
        assert hal.last_committed_tick == 3

        # The refusal staged nothing: the next tick completes normally.
        for slot in _group(hal, case, tick=4):
            hal.send_action(slot)
        assert hal.last_committed_tick == 4

        # A reconnect restarts the numbering.
        hal.disconnect()
        assert hal.last_committed_tick == 0
    finally:
        hal.disconnect()


@pytest.mark.parametrize("case", _CASES)
def test_estop_drops_a_half_staged_group(case: _Case) -> None:
    from openral_core import ROSEStopRequested

    hal = case.make()
    hal.connect()
    for slot in _group(hal, case, tick=1)[:-1]:
        hal.send_action(slot)
    with pytest.raises(ROSEStopRequested):
        hal.estop()
    hal.connect()
    # The survivors of tick 1 must never be committed after the stop: tick 2
    # starts clean and completes on its own slots.
    for slot in _group(hal, case, tick=2):
        hal.send_action(slot)
    assert hal.last_committed_tick == 2
    hal.disconnect()


@pytest.mark.parametrize("case", _CASES)
def test_slot_is_refused_while_disconnected(case: _Case) -> None:
    hal = case.make()
    with pytest.raises(ROSRuntimeError, match="not connected"):
        hal.send_action(_group(hal, case, tick=1)[0])


@pytest.mark.parametrize("case", _CASES)
def test_tick_one_after_a_higher_watermark_is_a_restarted_runner_everything_else_is_a_replay(
    case: _Case,
) -> None:
    """The slot-group tick rule (hazard log Entry 035), on non-OpenArm twins.

    A tick at or below the committed watermark is a replay and is refused,
    EXCEPT tick 1 arriving while the watermark is above 1: runner ticks are
    process-monotonic from 1, so that is a restarted runner's fresh numbering,
    and refusing it would wedge the HAL for the rest of its node's life.
    """
    hal = case.make()
    hal.connect()
    try:

        def commit(tick: int) -> None:
            for slot in _group(hal, case, tick=tick):
                hal.send_action(slot)
            assert hal.last_committed_tick == tick

        commit(7)
        for stale_tick in (7, 5):  # the committed tick replayed, and a lower one
            with pytest.raises(ROSRuntimeError, match="stale slot group"):
                hal.send_action(_group(hal, case, tick=stale_tick)[0])
            assert hal.last_committed_tick == 7

        commit(1)  # a restarted runner: adopted, not refused

        # Tick 1 replayed once tick 1 is the committed one: a replay again.
        with pytest.raises(ROSRuntimeError, match="stale slot group"):
            hal.send_action(_group(hal, case, tick=1)[0])
        commit(2)
    finally:
        hal.disconnect()


@pytest.mark.parametrize("case", _CASES)
def test_estop_keeps_the_watermark_so_a_pre_stop_tick_is_still_refused(case: _Case) -> None:
    from openral_core import ROSEStopRequested

    hal = case.make()
    hal.connect()
    try:
        for tick in (1, 2, 3):
            for slot in _group(hal, case, tick=tick):
                hal.send_action(slot)
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        hal.connect()
        assert hal.last_committed_tick == 3
        with pytest.raises(ROSRuntimeError, match="stale slot group"):
            hal.send_action(_group(hal, case, tick=3)[0])
        # Disconnect clears the watermark: numbering restarts.
        hal.disconnect()
        assert hal.last_committed_tick == 0
    finally:
        hal.disconnect()
