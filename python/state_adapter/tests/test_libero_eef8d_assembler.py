"""Unit tests for the ``libero_eef8d`` task-space layout assembler (ADR-0027).

Proves the deploy-path state vector byte-matches the LIBERO benchmark's
task-space construction (``eef_pos(3) ‖ eef_axisangle(3) ‖ gripper_qpos(2)``),
so a LIBERO VLA receives the same proprioception in ``openral deploy sim`` as
in ``openral sim run`` (where it scores 0.9–1.0 on libero_spatial).

The ``tf_lookup`` Protocol is the documented rclpy-free injection point
(``_protocol.TfLookup``), so these run without a ROS graph — no mock of any
internal logic, just a real callable returning a real ``TransformView``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from openral_core import ROSConfigError, StateContractBindings

from openral_state_adapter._protocol import TransformView
from openral_state_adapter.layouts.libero_eef8d import (
    _quat_xyzw_to_axisangle,
    assemble_libero_eef8d,
)

# Real franka_panda bindings (matches robots/franka_panda/robot.yaml frames),
# not "foo"/"test" placeholders (CLAUDE.md §1.11).
_EEF_FRAME = "panda_hand_tcp"
_WORLD_FRAME = "world"
_GRIPPER_2 = ["panda_finger_joint1", "panda_finger_joint2"]


def _tf(pos: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float]):
    def _lookup(target_frame: str, source_frame: str) -> TransformView:
        assert target_frame == _WORLD_FRAME
        assert source_frame == _EEF_FRAME
        return TransformView(position=pos, quaternion_xyzw=quat_xyzw)

    return _lookup


def test_identity_quat_gives_zero_axisangle() -> None:
    """A near-identity rotation maps to the zero axis-angle vector."""
    aa = _quat_xyzw_to_axisangle((0.0, 0.0, 0.0, 1.0))
    assert np.allclose(aa, np.zeros(3), atol=1e-6)


def test_quat_90deg_about_z_matches_pi_over_2() -> None:
    """A 90° rotation about +z → axis-angle ``[0, 0, π/2]`` (benchmark formula)."""
    s = math.sin(math.pi / 4.0)
    c = math.cos(math.pi / 4.0)
    aa = _quat_xyzw_to_axisangle((0.0, 0.0, s, c))
    assert np.allclose(aa, np.array([0.0, 0.0, math.pi / 2.0]), atol=1e-5)


def test_full_8d_ordering_and_values() -> None:
    """Output is exactly ``[eef_pos(3), eef_axisangle(3), gripper(2)]``."""
    pos = (0.35, -0.12, 1.05)
    s, c = math.sin(math.pi / 6.0), math.cos(math.pi / 6.0)  # 60° about z
    quat = (0.0, 0.0, s, c)
    bindings = StateContractBindings(
        eef_frame=_EEF_FRAME, world_frame=_WORLD_FRAME, gripper_qpos_joints=_GRIPPER_2
    )
    joints = {"panda_finger_joint1": 0.03, "panda_finger_joint2": 0.028, "panda_joint1": 0.0}

    out = assemble_libero_eef8d(bindings, joints, _tf(pos, quat))

    assert out.shape == (8,)
    assert out.dtype == np.float32
    expected = np.concatenate(
        [np.asarray(pos), _quat_xyzw_to_axisangle(quat), np.array([0.03, 0.028])]
    ).astype(np.float32)
    assert np.allclose(out, expected, atol=1e-6)


def test_single_gripper_joint_mirrors_to_v_minus_v() -> None:
    """A 1-joint parallel-gripper abstraction reconstructs ``[v, -v]``."""
    bindings = StateContractBindings(
        eef_frame=_EEF_FRAME, world_frame=_WORLD_FRAME, gripper_qpos_joints=["panda_finger_joint1"]
    )
    out = assemble_libero_eef8d(bindings, {"panda_finger_joint1": 0.041}, _tf((0, 0, 0), (0, 0, 0, 1)))
    assert np.allclose(out[6:8], np.array([0.041, -0.041]), atol=1e-6)


def test_world_frame_binding_is_forwarded_to_tf_lookup() -> None:
    """The manifest-set ``world_frame`` is the TF target (not the ``"map"`` default)."""
    bindings = StateContractBindings(
        eef_frame=_EEF_FRAME, world_frame=_WORLD_FRAME, gripper_qpos_joints=_GRIPPER_2
    )
    seen: dict[str, str] = {}

    def _lookup(target_frame: str, source_frame: str) -> TransformView:
        seen["target"] = target_frame
        return TransformView(position=(0.1, 0.2, 0.3), quaternion_xyzw=(0, 0, 0, 1))

    out = assemble_libero_eef8d(
        bindings, {"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0}, _lookup
    )
    assert seen["target"] == _WORLD_FRAME
    assert np.allclose(out[0:3], np.array([0.1, 0.2, 0.3]), atol=1e-6)


def test_missing_eef_frame_raises() -> None:
    bindings = StateContractBindings(world_frame=_WORLD_FRAME, gripper_qpos_joints=_GRIPPER_2)
    with pytest.raises(ROSConfigError, match=r"eef_frame is required"):
        assemble_libero_eef8d(bindings, {}, _tf((0, 0, 0), (0, 0, 0, 1)))


def test_bad_gripper_joint_count_raises() -> None:
    bindings = StateContractBindings(
        eef_frame=_EEF_FRAME,
        world_frame=_WORLD_FRAME,
        gripper_qpos_joints=["a", "b", "c"],
    )
    with pytest.raises(ROSConfigError, match=r"expects 1 .* or 2 .* gripper joints"):
        assemble_libero_eef8d(
            bindings, {"a": 0.0, "b": 0.0, "c": 0.0}, _tf((0, 0, 0), (0, 0, 0, 1))
        )
