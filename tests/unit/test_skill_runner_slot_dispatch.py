"""Unit tests for ``_dispatch_slots`` in ``rskill_runner_node`` (ADR-0028b step 3).

Exercises the pure byte-routing function that splits a flat policy
action vector into typed :class:`openral_core.Action` objects per the
manifest's :class:`openral_core.ActionContract.slots` declaration.

The full rskill_runner_node module is gated behind ``_ROS2_AVAILABLE``
(``rclpy`` import), so we load the module directly via
``importlib.util.spec_from_file_location`` instead of through the
package ``__init__`` — same trick used by other host-side tests that
need to exercise helpers without booting ROS 2.

The slot loop assumes the :class:`ActionContract` validator already
enforced coverage + per-mode field requirements; the cases here cover
the runtime byte-routing only.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from openral_core import Action, ActionSlot, ControlMode


def _load_skill_runner_module() -> ModuleType:
    """Load ``rskill_runner_node`` bypassing the ROS2-gated package init."""
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py"
    spec = importlib.util.spec_from_file_location("_test_rskill_runner_node", src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner_mod() -> ModuleType:
    return _load_skill_runner_module()


def _robocasa_12d_slots() -> list[ActionSlot]:
    """Canonical RoboCasa365 layout — five slots, three non-discard."""
    return [
        ActionSlot(
            range=(0, 5),
            control_mode=ControlMode.CARTESIAN_DELTA,
            ee="panda_hand",
            frame="panda_link0",
        ),
        ActionSlot(
            range=(6, 6),
            control_mode=ControlMode.GRIPPER_POSITION,
            ee="panda_gripper",
        ),
        ActionSlot(range=(7, 7), discard=True),
        ActionSlot(
            range=(8, 10),
            control_mode=ControlMode.BODY_TWIST,
            frame="base_link",
        ),
        ActionSlot(range=(11, 11), discard=True),
    ]


# ─── RoboCasa 12-D layout ────────────────────────────────────────────────────


def test_robocasa_12d_emits_three_typed_actions(runner_mod: ModuleType) -> None:
    """The 5-slot RoboCasa layout produces 3 non-discard Actions."""
    vec = np.array(
        [0.014, 0.0, -0.003, 0.001, 0.0, 0.0, -0.989, 0.001, 0.0, 0.0, 0.0, -0.991],
        dtype=np.float32,
    )
    actions = runner_mod._dispatch_slots(_robocasa_12d_slots(), vec)
    assert len(actions) == 3
    cart, grip, twist = actions

    assert cart.control_mode is ControlMode.CARTESIAN_DELTA
    assert cart.ee_name == "panda_hand"
    assert cart.frame_id == "panda_link0"
    assert cart.cartesian_delta is not None
    assert tuple(round(v, 4) for v in cart.cartesian_delta[0]) == (
        0.014,
        0.0,
        -0.003,
        0.001,
        0.0,
        0.0,
    )

    assert grip.control_mode is ControlMode.GRIPPER_POSITION
    assert grip.ee_name == "panda_gripper"
    assert grip.frame_id is None
    assert grip.gripper is not None
    assert math.isclose(grip.gripper[0], -0.989, abs_tol=1e-5)

    assert twist.control_mode is ControlMode.BODY_TWIST
    assert twist.frame_id == "base_link"
    assert twist.ee_name is None
    # 3-D planar slice (forward, side, yaw) padded to 6-D twist.
    assert twist.body_twist == [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]


def test_discard_slots_produce_no_action(runner_mod: ModuleType) -> None:
    """Slots marked discard=True emit no Action — bytes are dropped."""
    slots = [
        ActionSlot(range=(0, 0), discard=True),
        ActionSlot(range=(1, 1), control_mode=ControlMode.GRIPPER_POSITION, ee="g"),
        ActionSlot(range=(2, 2), discard=True),
    ]
    actions = runner_mod._dispatch_slots(slots, np.array([99.0, 0.5, 99.0], dtype=np.float32))
    assert len(actions) == 1
    assert actions[0].control_mode is ControlMode.GRIPPER_POSITION
    assert actions[0].gripper == [0.5]


# ─── Per-mode dispatch coverage ──────────────────────────────────────────────


def test_joint_position_routes_to_joint_targets(runner_mod: ModuleType) -> None:
    slots = [
        ActionSlot(
            range=(0, 6),
            control_mode=ControlMode.JOINT_POSITION,
        )
    ]
    vec = np.arange(7, dtype=np.float32)
    actions = runner_mod._dispatch_slots(slots, vec)
    assert len(actions) == 1
    a = actions[0]
    assert a.control_mode is ControlMode.JOINT_POSITION
    assert a.joint_targets == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    assert a.cartesian_delta is None
    assert a.body_twist is None
    assert a.gripper is None


def test_joint_velocity_routes_to_joint_velocities(runner_mod: ModuleType) -> None:
    slots = [ActionSlot(range=(0, 1), control_mode=ControlMode.JOINT_VELOCITY)]
    actions = runner_mod._dispatch_slots(slots, np.array([1.0, 2.0], dtype=np.float32))
    assert actions[0].joint_velocities == [[1.0, 2.0]]
    assert actions[0].joint_targets is None


def test_cartesian_twist_routes_correctly(runner_mod: ModuleType) -> None:
    slots = [
        ActionSlot(
            range=(0, 5),
            control_mode=ControlMode.CARTESIAN_TWIST,
            ee="panda_hand",
            frame="panda_link0",
        )
    ]
    vec = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03], dtype=np.float32)
    actions = runner_mod._dispatch_slots(slots, vec)
    a = actions[0]
    assert a.control_mode is ControlMode.CARTESIAN_TWIST
    assert tuple(round(v, 4) for v in a.cartesian_twist[0]) == (
        0.1,
        0.2,
        0.3,
        0.01,
        0.02,
        0.03,
    )


def test_body_twist_6d_passthrough(runner_mod: ModuleType) -> None:
    """6-D body twist passes through verbatim, no zero-padding."""
    slots = [ActionSlot(range=(0, 5), control_mode=ControlMode.BODY_TWIST, frame="base_link")]
    vec = np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.3], dtype=np.float32)
    actions = runner_mod._dispatch_slots(slots, vec)
    assert tuple(round(v, 4) for v in actions[0].body_twist[0]) == (
        0.5,
        0.1,
        0.0,
        0.0,
        0.0,
        0.3,
    )


def test_body_twist_invalid_width_rejected(runner_mod: ModuleType) -> None:
    """4-D / 5-D BODY_TWIST slots are rejected (only 3-D planar or 6-D full)."""
    slots = [ActionSlot(range=(0, 3), control_mode=ControlMode.BODY_TWIST, frame="base_link")]
    with pytest.raises(ValueError, match="must be 3-D"):
        runner_mod._dispatch_slots(slots, np.zeros(4, dtype=np.float32))


def test_gripper_binary_routes_to_gripper(runner_mod: ModuleType) -> None:
    slots = [ActionSlot(range=(0, 0), control_mode=ControlMode.GRIPPER_BINARY, ee="g")]
    actions = runner_mod._dispatch_slots(slots, np.array([1.0], dtype=np.float32))
    assert actions[0].control_mode is ControlMode.GRIPPER_BINARY
    assert actions[0].gripper == [1.0]


# ─── Action objects are well-formed ──────────────────────────────────────────


def test_returned_actions_are_action_instances(runner_mod: ModuleType) -> None:
    actions = runner_mod._dispatch_slots(
        _robocasa_12d_slots(),
        np.zeros(12, dtype=np.float32),
    )
    assert all(isinstance(a, Action) for a in actions)
    # Each Action carries horizon=1 — single-step chunks.
    assert all(a.horizon == 1 for a in actions)
