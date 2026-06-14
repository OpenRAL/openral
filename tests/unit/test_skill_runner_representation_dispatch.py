"""Deploy-sim representation→slot expansion in rskill_runner_node (ADR-0036).

A VLA skill may declare only ``action_contract.representation`` (e.g.
``delta_ee_6d_plus_gripper``) and NO explicit ``action_contract.slots``.
Before this change the runner fell through to the legacy single-surface
path and emitted one ``Action(control_mode=JOINT_POSITION, ...)`` for the
whole vector — which the joint-space safety envelope rejects (``n_dof 7
!= 8`` on franka). The runner now expands the representation to the
canonical typed slots and dispatches them, so it emits
``Action(CARTESIAN_DELTA)`` + ``Action(GRIPPER_POSITION)`` instead.

These tests pin the routing the ``_step_impl`` hook relies on, using the
real ``robots/franka_panda/robot.yaml`` ``RobotDescription`` and the real
``canonical_slots_for_representation`` helper plus the real
``_dispatch_slots`` dispatcher (no mocks, CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# The module under test bundles the full lifecycle node (rclpy / IDL).
# Skip cleanly when those aren't sourced — the helpers are still defined
# at module top-level, just unreachable.
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_core import ControlMode, RobotDescription
from openral_core.schemas import (
    ActionRepresentation,
    canonical_slots_for_representation,
)
from openral_rskill_ros.rskill_runner_node import _dispatch_slots

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _franka() -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"))


class TestCanonicalSlotsForRepresentation:
    def test_joint_representation_returns_none(self) -> None:
        # Joint reps keep the legacy whole-vector JOINT_POSITION path.
        assert (
            canonical_slots_for_representation(
                ActionRepresentation.JOINT_POSITIONS, dim=8, description=_franka()
            )
            is None
        )

    def test_delta_ee_6d_plus_gripper_layout(self) -> None:
        slots = canonical_slots_for_representation(
            ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER, dim=7, description=_franka()
        )
        assert slots is not None
        assert [s.control_mode for s in slots] == [
            ControlMode.CARTESIAN_DELTA,
            ControlMode.GRIPPER_POSITION,
        ]
        # franka's primary end-effector names the cartesian + gripper slots.
        assert slots[0].range == (0, 5)
        assert slots[0].ee == "panda_hand"
        assert slots[1].range == (6, 6)


class TestDispatchExpandedSlots:
    def test_delta_ee_6d_plus_gripper_dispatches_cartesian_and_gripper(self) -> None:
        """The canonical slots route the 7-vec to CARTESIAN_DELTA + GRIPPER.

        This is exactly the routing the ``_step_impl`` representation hook
        relies on: expand → ``_dispatch_slots``. A realistic small OSC
        delta vector + a gripper command; the cartesian Action must carry
        the first 6 values and the gripper Action the 7th.
        """
        franka = _franka()
        slots = canonical_slots_for_representation(
            ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER, dim=7, description=franka
        )
        assert slots is not None
        policy_action = np.array([0.1, -0.2, 0.05, 0.0, 0.03, -0.01, -0.9], dtype=np.float32)

        actions = _dispatch_slots(slots, policy_action, description=franka)

        assert [a.control_mode for a in actions] == [
            ControlMode.CARTESIAN_DELTA,
            ControlMode.GRIPPER_POSITION,
        ]

        cart, grip = actions
        # Cartesian delta carries the first 6 values as a single 6-tuple.
        assert cart.cartesian_delta is not None
        assert len(cart.cartesian_delta) == 1
        assert cart.cartesian_delta[0] == pytest.approx(
            (0.1, -0.2, 0.05, 0.0, 0.03, -0.01), abs=1e-6
        )
        assert cart.ee_name == "panda_hand"
        assert cart.frame_id == "panda_hand"
        # No joint targets — the joint-space envelope is never engaged.
        assert cart.joint_targets is None

        # Gripper Action carries only the 7th value.
        assert grip.gripper is not None
        assert grip.gripper == pytest.approx([-0.9], abs=1e-6)
        assert grip.ee_name == "panda_hand"
        assert grip.joint_targets is None
