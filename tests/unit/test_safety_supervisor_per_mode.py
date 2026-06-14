"""Per-control-mode envelope dispatch tests for ``SafetyPassthroughNode``
(ADR-0028b step 5).

The supervisor used to enforce ``n_dof`` + per-joint bounds and reject
non-joint chunks implicitly (every non-joint chunk passed through with
zero validation). ADR-0028b adds per-mode dispatch:

* JOINT_*       → existing path (unchanged — first test below pins this).
* CARTESIAN_DELTA / CARTESIAN_TWIST / BODY_TWIST → per-axis bounds.
* GRIPPER_*     → width range bound.

All new bounds default to ``-1.0`` ("no enforcement declared, skip")
so a legacy launch that doesn't override them keeps passing every
non-joint chunk through verbatim. The "joint path unchanged" property
is the most important guard: the existing ADR-0018 supervisor test
(``packages/openral_safety/test/test_supervisor_node.py``) is gated on
``openral_msgs`` and skipped in this worktree, so we re-pin the joint
behaviour here in pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

rclpy = pytest.importorskip("rclpy")

from openral_core import (  # noqa: E402  # reason: imported after rclpy importorskip guard
    CONTROL_MODE_TO_UINT8,
    ControlMode,
)
from openral_safety.supervisor_node import (  # noqa: E402  # reason: imported after rclpy importorskip guard
    SafetyPassthroughNode,
)


@dataclass
class _StubChunk:
    """Minimal ActionChunk shape — the supervisor only reads three fields.

    The real ``openral_msgs/ActionChunk`` IDL is built by colcon; this
    stub lets the per-mode dispatch be exercised in a unit context
    without a sourced ROS 2 install.
    """

    control_mode: int
    n_dof: int
    flat: list[float]


@pytest.fixture(scope="module")
def ros_init() -> Any:
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros_init: None) -> Any:
    n = SafetyPassthroughNode(node_name=f"sup_per_mode_{id(object())}")
    yield n
    n.destroy_node()


def _set(node: Any, name: str, value: float) -> None:
    """Override a declared double parameter."""
    from rclpy.parameter import Parameter

    node.set_parameters([Parameter(name, Parameter.Type.DOUBLE, value)])


def _chunk(mode: ControlMode, *, flat: list[float], n_dof: int) -> _StubChunk:
    return _StubChunk(
        control_mode=CONTROL_MODE_TO_UINT8[mode],
        n_dof=n_dof,
        flat=flat,
    )


# ─── JOINT path — must remain byte-identical to the pre-ADR-0028b behaviour ──


def test_joint_path_unchanged_pass_when_no_envelope_declared(node: Any) -> None:
    """No n_dof / min_joint / max_joint set → joint chunk passes through."""
    msg = _chunk(ControlMode.JOINT_POSITION, n_dof=6, flat=[0.0] * 6)
    kind, _ = node._envelope_violation(msg)
    assert kind is None


def test_joint_path_n_dof_mismatch_still_rejects(node: Any) -> None:
    """n_dof check still fires when configured (legacy contract)."""
    from rclpy.parameter import Parameter

    node.set_parameters([Parameter("n_dof", Parameter.Type.INTEGER, 6)])
    msg = _chunk(ControlMode.JOINT_POSITION, n_dof=7, flat=[0.0] * 7)
    kind, reason = node._envelope_violation(msg)
    assert kind == "n_dof"
    assert "expected n_dof=6" in reason


def test_joint_path_per_joint_bounds_still_reject(node: Any) -> None:
    """Per-joint min/max array still fires when configured."""
    from rclpy.parameter import Parameter

    node.set_parameters(
        [
            Parameter("n_dof", Parameter.Type.INTEGER, 3),
            Parameter("min_joint", Parameter.Type.DOUBLE_ARRAY, [-1.0, -1.0, -1.0]),
            Parameter("max_joint", Parameter.Type.DOUBLE_ARRAY, [1.0, 1.0, 1.0]),
        ]
    )
    msg = _chunk(ControlMode.JOINT_POSITION, n_dof=3, flat=[0.5, 2.0, 0.0])
    kind, reason = node._envelope_violation(msg)
    assert kind == "workspace"
    assert "joint[1]=2.0000" in reason


# ─── CARTESIAN_DELTA ─────────────────────────────────────────────────────────


def test_cartesian_delta_passes_when_bounds_unset(node: Any) -> None:
    msg = _chunk(ControlMode.CARTESIAN_DELTA, n_dof=6, flat=[10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    kind, _ = node._envelope_violation(msg)
    assert kind is None


def test_cartesian_delta_rejects_xyz_step_too_large(node: Any) -> None:
    _set(node, "max_cartesian_step_m", 0.05)
    msg = _chunk(ControlMode.CARTESIAN_DELTA, n_dof=6, flat=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    kind, reason = node._envelope_violation(msg)
    assert kind == "cartesian_step"
    assert "|dxyz|=0.1000" in reason


def test_cartesian_delta_passes_when_within_xyz_bound(node: Any) -> None:
    _set(node, "max_cartesian_step_m", 0.05)
    msg = _chunk(ControlMode.CARTESIAN_DELTA, n_dof=6, flat=[0.01, 0.02, 0.03, 0.0, 0.0, 0.0])
    kind, _ = node._envelope_violation(msg)
    # |dxyz| = sqrt(0.0001 + 0.0004 + 0.0009) = ~0.0374, under 0.05.
    assert kind is None


def test_cartesian_delta_rejects_rotation_step_too_large(node: Any) -> None:
    _set(node, "max_cartesian_step_rad", 0.2)
    msg = _chunk(ControlMode.CARTESIAN_DELTA, n_dof=6, flat=[0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    kind, reason = node._envelope_violation(msg)
    assert kind == "cartesian_step_rot"
    assert "|drotvec|=0.5000" in reason


def test_cartesian_delta_shape_rejection(node: Any) -> None:
    """4-D cartesian_delta chunk (bug shape) rejected when bounds active."""
    _set(node, "max_cartesian_step_m", 0.05)
    msg = _chunk(ControlMode.CARTESIAN_DELTA, n_dof=4, flat=[0.01, 0.01, 0.01, 0.0])
    kind, reason = node._envelope_violation(msg)
    assert kind == "cartesian_shape"
    assert "n_dof=4" in reason


# ─── CARTESIAN_TWIST ─────────────────────────────────────────────────────────


def test_cartesian_twist_linear_bound_rejects(node: Any) -> None:
    _set(node, "max_ee_speed_m_s", 1.0)
    msg = _chunk(ControlMode.CARTESIAN_TWIST, n_dof=6, flat=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    kind, reason = node._envelope_violation(msg)
    assert kind == "ee_linear_speed"
    assert "|v_ee|=2.0000" in reason


def test_cartesian_twist_angular_bound_rejects(node: Any) -> None:
    _set(node, "max_ee_angular_speed_rad_s", 1.0)
    msg = _chunk(ControlMode.CARTESIAN_TWIST, n_dof=6, flat=[0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    kind, reason = node._envelope_violation(msg)
    assert kind == "ee_angular_speed"
    assert "|w_ee|=2.0000" in reason


# ─── BODY_TWIST ──────────────────────────────────────────────────────────────


def test_body_twist_passes_when_bounds_unset(node: Any) -> None:
    """Existing nav2 cmd_vel path: BODY_TWIST without per-mode bounds passes."""
    msg = _chunk(ControlMode.BODY_TWIST, n_dof=6, flat=[100.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    kind, _ = node._envelope_violation(msg)
    assert kind is None


def test_body_twist_linear_bound_rejects(node: Any) -> None:
    _set(node, "max_base_linear_speed_m_s", 1.0)
    msg = _chunk(ControlMode.BODY_TWIST, n_dof=6, flat=[1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    kind, reason = node._envelope_violation(msg)
    assert kind == "base_linear_speed"
    assert "|v_base|=1.5000" in reason


def test_body_twist_angular_bound_rejects(node: Any) -> None:
    _set(node, "max_base_angular_speed_rad_s", 1.5)
    msg = _chunk(ControlMode.BODY_TWIST, n_dof=6, flat=[0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    kind, reason = node._envelope_violation(msg)
    assert kind == "base_angular_speed"
    assert "|w_base|=2.0000" in reason


def test_body_twist_planar_zero_padding_passes(node: Any) -> None:
    """The skill_runner's BODY_TWIST emission zero-pads (vy=0, vz=0, wx=0, wy=0)
    for planar bases. The supervisor's Euclidean bound over each triplet
    handles this naturally.
    """
    _set(node, "max_base_linear_speed_m_s", 1.0)
    _set(node, "max_base_angular_speed_rad_s", 1.0)
    msg = _chunk(ControlMode.BODY_TWIST, n_dof=6, flat=[0.5, 0.0, 0.0, 0.0, 0.0, 0.3])
    kind, _ = node._envelope_violation(msg)
    assert kind is None


# ─── GRIPPER ─────────────────────────────────────────────────────────────────


def test_gripper_passes_when_bounds_unset(node: Any) -> None:
    msg = _chunk(ControlMode.GRIPPER_POSITION, n_dof=1, flat=[99.0])
    kind, _ = node._envelope_violation(msg)
    assert kind is None


def test_gripper_min_bound_rejects(node: Any) -> None:
    _set(node, "gripper_min", 0.0)
    msg = _chunk(ControlMode.GRIPPER_POSITION, n_dof=1, flat=[-0.5])
    kind, reason = node._envelope_violation(msg)
    assert kind == "gripper_range"
    assert "width=-0.5000 < gripper_min=0.0000" in reason


def test_gripper_max_bound_rejects(node: Any) -> None:
    _set(node, "gripper_max", 1.0)
    msg = _chunk(ControlMode.GRIPPER_POSITION, n_dof=1, flat=[2.0])
    kind, reason = node._envelope_violation(msg)
    assert kind == "gripper_range"
    assert "width=2.0000 > gripper_max=1.0000" in reason


def test_gripper_in_range_passes(node: Any) -> None:
    _set(node, "gripper_min", 0.0)
    _set(node, "gripper_max", 1.0)
    msg = _chunk(ControlMode.GRIPPER_POSITION, n_dof=1, flat=[0.5])
    kind, _ = node._envelope_violation(msg)
    assert kind is None


# ─── Unknown control_mode ────────────────────────────────────────────────────


def test_unknown_control_mode_rejected(node: Any) -> None:
    msg = _StubChunk(control_mode=42, n_dof=1, flat=[0.0])
    kind, reason = node._envelope_violation(msg)
    assert kind == "control_mode"
    assert "uint8 42" in reason
