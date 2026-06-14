"""Regression test for the ADR-0028b ``/openral/safe_action`` decoder.

The HAL lifecycle node's ``_on_safe_action`` callback used to hardcode
every incoming chunk as :class:`ControlMode.JOINT_POSITION`, throwing
away the wire ``control_mode`` field. Consequence: per-mode chunks
(CARTESIAN_DELTA, GRIPPER_POSITION, BODY_TWIST) arriving from the C++
safety kernel were silently misrouted into ``Action.joint_targets``,
where the HAL packer rejected them with a single WARN — visible only
as "arm never moves in ``openral deploy sim``".

This module pins the decoder (:func:`openral_hal.lifecycle.
decode_action_chunk`) against every control_mode the F1/F5 publisher
emits, so the same lie can't return. Real ``ActionChunk``-shaped
inputs, real Pydantic :class:`Action` outputs, no mocks per CLAUDE.md
§1.11 — the decoder is duck-typed so any object exposing the wire
field names works as the test double.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from openral_core.schemas import CONTROL_MODE_TO_UINT8, Action, ControlMode
from openral_hal.lifecycle import decode_action_chunk


@dataclass
class FakeChunk:
    """Duck-typed stand-in for the rosidl-generated ``ActionChunk``.

    ``decode_action_chunk`` reads its fields via ``getattr`` only, so a
    plain dataclass is a real (not mocked) substitute — the rosidl
    class itself is just a struct with no behaviour.
    """

    flat: list[float] = field(default_factory=list)
    n_dof: int = 0
    horizon: int = 1
    control_mode: int = 0


class TestDecodeActionChunk:
    def test_joint_position_round_trip(self) -> None:
        chunk = FakeChunk(
            flat=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            n_dof=7,
            horizon=1,
            control_mode=CONTROL_MODE_TO_UINT8[ControlMode.JOINT_POSITION],
        )
        action = decode_action_chunk(chunk)
        assert isinstance(action, Action)
        assert action.control_mode is ControlMode.JOINT_POSITION
        assert action.joint_targets == [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]

    def test_cartesian_delta_populates_typed_field_not_joint_targets(self) -> None:
        """The regression. CARTESIAN_DELTA must land in
        :attr:`Action.cartesian_delta`, not :attr:`Action.joint_targets`.
        """
        chunk = FakeChunk(
            flat=[-0.97, -0.28, -0.27, 0.0, -0.34, -0.10],
            n_dof=6,
            horizon=1,
            control_mode=CONTROL_MODE_TO_UINT8[ControlMode.CARTESIAN_DELTA],
        )
        action = decode_action_chunk(chunk)
        assert isinstance(action, Action)
        assert action.control_mode is ControlMode.CARTESIAN_DELTA
        assert action.cartesian_delta == [(-0.97, -0.28, -0.27, 0.0, -0.34, -0.10)]
        assert not action.joint_targets

    def test_gripper_position_uses_flat_list_not_nested_rows(self) -> None:
        chunk = FakeChunk(
            flat=[0.04, 0.04, 0.0],
            n_dof=1,
            horizon=3,
            control_mode=CONTROL_MODE_TO_UINT8[ControlMode.GRIPPER_POSITION],
        )
        action = decode_action_chunk(chunk)
        assert isinstance(action, Action)
        assert action.control_mode is ControlMode.GRIPPER_POSITION
        assert action.gripper == [0.04, 0.04, 0.0]

    def test_body_twist_populates_typed_field(self) -> None:
        chunk = FakeChunk(
            flat=[0.2, 0.0, 0.0, 0.0, 0.0, 0.3],
            n_dof=6,
            horizon=1,
            control_mode=CONTROL_MODE_TO_UINT8[ControlMode.BODY_TWIST],
        )
        action = decode_action_chunk(chunk)
        assert isinstance(action, Action)
        assert action.control_mode is ControlMode.BODY_TWIST
        assert action.body_twist == [(0.2, 0.0, 0.0, 0.0, 0.0, 0.3)]

    def test_cartesian_twist_populates_typed_field(self) -> None:
        chunk = FakeChunk(
            flat=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            n_dof=6,
            horizon=1,
            control_mode=CONTROL_MODE_TO_UINT8[ControlMode.CARTESIAN_TWIST],
        )
        action = decode_action_chunk(chunk)
        assert isinstance(action, Action)
        assert action.control_mode is ControlMode.CARTESIAN_TWIST
        assert action.cartesian_twist == [(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)]

    def test_joint_velocity_and_joint_torque_route_to_typed_fields(self) -> None:
        for mode, field_name in (
            (ControlMode.JOINT_VELOCITY, "joint_velocities"),
            (ControlMode.JOINT_TORQUE, "joint_torques"),
        ):
            chunk = FakeChunk(
                flat=[0.1, -0.2, 0.3],
                n_dof=3,
                horizon=1,
                control_mode=CONTROL_MODE_TO_UINT8[mode],
            )
            action = decode_action_chunk(chunk)
            assert isinstance(action, Action)
            assert action.control_mode is mode
            assert getattr(action, field_name) == [[0.1, -0.2, 0.3]]
            assert not action.joint_targets, (
                f"{mode!r} must not bleed into joint_targets — that was the bug."
            )

    def test_multi_step_horizon_splits_rows(self) -> None:
        chunk = FakeChunk(
            flat=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            n_dof=6,
            horizon=2,
            control_mode=CONTROL_MODE_TO_UINT8[ControlMode.CARTESIAN_DELTA],
        )
        action = decode_action_chunk(chunk)
        assert isinstance(action, Action)
        assert action.horizon == 2
        assert action.cartesian_delta == [
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            (1.0, 1.1, 1.2, 1.3, 1.4, 1.5),
        ]

    def test_empty_flat_returns_none(self) -> None:
        assert decode_action_chunk(FakeChunk(flat=[], n_dof=0)) is None
        assert decode_action_chunk(FakeChunk(flat=[1.0], n_dof=0)) is None

    @pytest.mark.parametrize(
        "unwired_mode",
        [ControlMode.CARTESIAN_POSE, ControlMode.FOOT_PLACEMENT],
    )
    def test_modes_not_on_wire_drop_to_none(self, unwired_mode: ControlMode) -> None:
        """Modes the publisher refuses to encode must drop on the
        receive side too — otherwise the prior code's "fabricate a
        JOINT_POSITION lie" lives on by another name.
        """
        chunk = FakeChunk(
            flat=[0.0, 0.0, 0.0],
            n_dof=3,
            horizon=1,
            control_mode=CONTROL_MODE_TO_UINT8[unwired_mode],
        )
        assert decode_action_chunk(chunk) is None
