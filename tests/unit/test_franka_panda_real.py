"""Unit tests for :class:`FrankaPandaRealHAL` — the real-hardware Franka adapter.

The adapter wraps :class:`RosControlHAL` so the heavy hot-path logic is
already covered by ``tests/unit/test_hal.py``; this file pins the
Franka-specific surface (FCI metadata, manifest pointer, e-stop topic) and
the closed-loop behaviour against :class:`SimTransport` (real, not mocked).
"""

from __future__ import annotations

import time

import pytest
from openral_core import (
    Action,
    ControlMode,
    ROSConfigError,
    ROSEStopRequested,
    ROSPerceptionStale,
    ROSRuntimeError,
    ROSSafetyViolation,
)
from openral_core.schemas import JointState
from openral_hal.franka_panda import FRANKA_PANDA_DESCRIPTION
from openral_hal.franka_panda_real import (
    FRANKA_PANDA_REAL_DESCRIPTION,
    FrankaPandaRealHAL,
)
from openral_hal.protocol import HAL
from openral_hal.sim_transport import SimTransport

# ── Fixtures ──────────────────────────────────────────────────────────────────

_N_JOINTS = len(FRANKA_PANDA_REAL_DESCRIPTION.joints)


@pytest.fixture()
def transport() -> SimTransport:
    return SimTransport(n_joints=_N_JOINTS)


@pytest.fixture()
def hal(transport: SimTransport) -> FrankaPandaRealHAL:
    return FrankaPandaRealHAL(
        fci_ip="172.16.0.2",
        publish_fn=transport.publish,
        state_fn=transport.state,
    )


def _hold_action() -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0] * _N_JOINTS],
        stamp_ns=time.time_ns(),
    )


# ── Construction ──────────────────────────────────────────────────────────────


class TestConstruction:
    def test_empty_fci_ip_rejected(self) -> None:
        with pytest.raises(ROSConfigError):
            FrankaPandaRealHAL(fci_ip="")

    def test_whitespace_fci_ip_rejected(self) -> None:
        with pytest.raises(ROSConfigError):
            FrankaPandaRealHAL(fci_ip="   ")

    def test_default_controller_name(self, hal: FrankaPandaRealHAL) -> None:
        assert hal.controller_name == "franka_arm_controller"

    def test_fci_ip_stored(self, hal: FrankaPandaRealHAL) -> None:
        assert hal.fci_ip == "172.16.0.2"

    def test_description_is_franka_panda(self, hal: FrankaPandaRealHAL) -> None:
        assert hal.description.name == "franka_panda"
        assert len(hal.description.joints) == 8


# ── Manifest pointer (issue #56 acceptance criterion) ─────────────────────────


class TestManifestPointer:
    def test_sdk_kind_is_closed_with_api(self) -> None:
        assert FRANKA_PANDA_REAL_DESCRIPTION.sdk_kind == "closed_with_api"

    def test_hal_real_resolves_to_real_hal(self) -> None:
        assert (
            FRANKA_PANDA_REAL_DESCRIPTION.hal.real
            == "openral_hal.franka_panda_real:FrankaPandaRealHAL"
        )

    def test_real_description_inherits_kinematics_from_sim(self) -> None:
        """Sim baseline ↔ real-HW description share kinematics + safety + caps."""
        sim = FRANKA_PANDA_DESCRIPTION.model_dump()
        real = FRANKA_PANDA_REAL_DESCRIPTION.model_dump()
        for shared_field in ("name", "joints", "end_effectors", "capabilities", "safety"):
            assert sim[shared_field] == real[shared_field]
        # ADR-0031: only sdk_kind differs; the hal entrypoints are shared.
        assert sim["sdk_kind"] != real["sdk_kind"]
        assert sim["hal"] == real["hal"]


# ── HAL Protocol conformance (closed-loop SimTransport) ───────────────────────


class TestProtocolConformance:
    def test_satisfies_hal_protocol(self, hal: FrankaPandaRealHAL) -> None:
        assert isinstance(hal, HAL)

    def test_read_state_before_connect_raises(self, hal: FrankaPandaRealHAL) -> None:
        with pytest.raises(ROSRuntimeError):
            hal.read_state()

    def test_send_action_before_connect_raises(self, hal: FrankaPandaRealHAL) -> None:
        with pytest.raises(ROSRuntimeError):
            hal.send_action(_hold_action())

    def test_connect_then_read_state(self, hal: FrankaPandaRealHAL) -> None:
        hal.connect()
        try:
            state = hal.read_state()
            assert isinstance(state, JointState)
            assert state.name == [j.name for j in FRANKA_PANDA_REAL_DESCRIPTION.joints]
            assert len(state.position) == _N_JOINTS
        finally:
            hal.disconnect()

    def test_disconnect_idempotent(self, hal: FrankaPandaRealHAL) -> None:
        hal.connect()
        hal.disconnect()
        # Second call must not raise.
        hal.disconnect()

    def test_send_action_publishes_to_franka_controller(
        self, hal: FrankaPandaRealHAL, transport: SimTransport
    ) -> None:
        hal.connect()
        try:
            hal.send_action(_hold_action())
        finally:
            hal.disconnect()
        assert transport.call_count == 1
        topic, _msg = transport.calls[0]
        assert topic == "/franka_arm_controller/joint_trajectory"


# ── Safety ────────────────────────────────────────────────────────────────────


class TestSafety:
    def test_estop_always_raises(self, hal: FrankaPandaRealHAL) -> None:
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()

    def test_estop_is_safety_violation(self, hal: FrankaPandaRealHAL) -> None:
        hal.connect()
        with pytest.raises(ROSSafetyViolation):
            hal.estop()

    def test_estop_publishes_recovery_message(
        self, hal: FrankaPandaRealHAL, transport: SimTransport
    ) -> None:
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        assert any(topic == "/error_recovery/goal" for topic, _msg in transport.calls)

    def test_after_estop_send_action_fails(
        self, hal: FrankaPandaRealHAL, transport: SimTransport
    ) -> None:
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        # Drop into a clean state — sending an action without reconnecting fails.
        with pytest.raises(ROSRuntimeError):
            hal.send_action(_hold_action())


# ── Staleness guard (delegated to RosControlHAL) ──────────────────────────────


class TestStaleness:
    def test_read_state_raises_when_stale(self, transport: SimTransport) -> None:
        hal = FrankaPandaRealHAL(
            fci_ip="172.16.0.2",
            publish_fn=transport.publish,
            state_fn=transport.state,
            staleness_limit_s=0.0,  # any age fails
        )
        hal.connect()
        time.sleep(0.001)
        try:
            with pytest.raises(ROSPerceptionStale):
                hal.read_state()
        finally:
            hal.disconnect()
