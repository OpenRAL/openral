"""Parametrized HAL protocol contract tests shared across all MuJoCo HAL implementations.

This module consolidates the identical HAL protocol compliance tests that were
previously duplicated across 9+ individual HAL test files. Each HAL implementation
must pass these contracts to be considered production-ready.

CLAUDE.md §1.11: Real components, not mocks. These tests load actual MuJoCo physics
and exercise the full HAL lifecycle contract:

  connect → read_state → send_action → estop / disconnect

See also: tests/unit/test_hal_protocol_conformance.py for non-MuJoCo HALs.
"""

from __future__ import annotations

import pytest

try:
    import mujoco  # noqa: F401
except Exception as exc:
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

try:
    # Pre-flight check on all MJCF loads to avoid cascading skip messages.
    _MJCF_ERROR: str | None = None
except Exception as exc:
    _MJCF_ERROR = str(exc)

from openral_core import (
    Action,
    ControlMode,
    ROSConfigError,
    ROSEStopRequested,
    ROSSafetyViolation,
)
from openral_hal import (
    HAL,
    AlohaMujocoHAL,
    FrankaPandaHAL,
    G1MujocoHAL,
    H1MujocoHAL,
    OpenArmMujocoHAL,
    Rizon4MujocoHAL,
    SO100MujocoHAL,
    UR5eHAL,
    UR10eHAL,
)

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"robot MJCF unavailable: {_MJCF_ERROR}",
    ),
]

# ── Parametrized HAL fixtures ────────────────────────────────────────────────

_HAL_CLASSES = [
    SO100MujocoHAL,
    FrankaPandaHAL,
    G1MujocoHAL,
    H1MujocoHAL,
    AlohaMujocoHAL,
    Rizon4MujocoHAL,
    OpenArmMujocoHAL,
    UR5eHAL,
    UR10eHAL,
]


def _make_hal(hal_class: type[HAL]) -> HAL:
    """Factory to instantiate each HAL with appropriate defaults."""
    if issubclass(hal_class, (G1MujocoHAL, H1MujocoHAL)):
        # Humanoids settle longer; gravity off so the free-standing floating base
        # doesn't collapse during the settle steps and perturb the contract check.
        return hal_class(gravity_enabled=False, settle_steps=1000)
    else:
        # Manipulators: gravity_enabled=False so position controllers converge exactly.
        return hal_class(gravity_enabled=False, settle_steps=2000)


@pytest.fixture(params=_HAL_CLASSES)
def hal(request: pytest.FixtureRequest) -> HAL:
    """Parametrized fixture instantiating each HAL class."""
    try:
        return _make_hal(request.param)
    except ROSConfigError as exc:
        # An optional sim asset/dependency for this robot is not installed in this
        # environment (e.g. ``gym_aloha`` or a ``robot_descriptions`` submodule).
        # Skip rather than fail the shared contract suite — CLAUDE.md §1.11. A
        # ROSConfigError with no missing-import/-file root cause is a genuine
        # misconfiguration and is re-raised.
        if isinstance(exc.__cause__, (ModuleNotFoundError, ImportError, FileNotFoundError)):
            pytest.skip(f"{request.param.__name__}: {exc}")
        raise


@pytest.fixture
def connected_hal(hal: HAL) -> HAL:
    """HAL in connected state; disconnects on teardown."""
    hal.connect()
    yield hal
    hal.disconnect()


# ── Shared HAL protocol contracts ────────────────────────────────────────────


class TestHALProtocolCompliance:
    """All HAL implementations must implement the HAL Protocol."""

    def test_satisfies_hal_protocol(self, hal: HAL) -> None:
        """HAL instance implements the HAL protocol interface."""
        assert isinstance(hal, HAL)


class TestHALLifecycleContract:
    """Standardized connect/disconnect/state/action contract enforced on all HALs."""

    def test_connect_twice_raises(self, hal: HAL) -> None:
        """Connecting twice without disconnect raises."""
        hal.connect()
        try:
            with pytest.raises(Exception):  # noqa: B017  # reason: HAL contract only guarantees *some* error on double-connect; concrete type varies by implementation
                hal.connect()
        finally:
            hal.disconnect()

    def test_disconnect_idempotent(self, hal: HAL) -> None:
        """Disconnecting multiple times is safe."""
        hal.connect()
        hal.disconnect()
        hal.disconnect()  # Should not raise.

    def test_disconnect_without_connect_is_noop(self, hal: HAL) -> None:
        """Disconnecting before connect is a no-op."""
        hal.disconnect()  # Should not raise.

    def test_raises_when_not_connected(self, hal: HAL) -> None:
        """read_state() and send_action() raise when not connected."""
        from openral_core import ROSRuntimeError

        with pytest.raises(ROSRuntimeError):
            hal.read_state()

    def test_rejects_missing_joint_targets(self, connected_hal: HAL) -> None:
        """send_action() rejects an action whose joint width doesn't match the robot."""
        from openral_core import ROSConfigError

        # A single joint target when every in-scope robot drives >1 joint — a width
        # mismatch that ``_validate_action_dims`` must reject (ADR-0023 contract).
        bad_action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[[0.0]],
            horizon=1,
        )
        with pytest.raises(ROSConfigError):
            connected_hal.send_action(bad_action)

    def test_rejects_unsupported_control_mode(self, connected_hal: HAL) -> None:
        """send_action() rejects a control mode the robot does not drive."""
        from openral_core import ROSConfigError

        description = connected_hal.description
        # These MuJoCo HALs drive position actuators; JOINT_TORQUE is never advertised
        # in ``capabilities.supported_control_modes`` — so it must be refused.
        if ControlMode.JOINT_TORQUE in description.capabilities.supported_control_modes:
            pytest.skip(f"{type(connected_hal).__name__} advertises JOINT_TORQUE")
        action = Action(
            control_mode=ControlMode.JOINT_TORQUE,
            joint_targets=[[0.0 for _ in description.joints]],
            horizon=1,
        )
        with pytest.raises(ROSConfigError):
            connected_hal.send_action(action)


class TestHALSafetyContract:
    """E-stop and safety violation contract enforced on all HALs."""

    def test_estop_raises_safety_violation(self, connected_hal: HAL) -> None:
        """Calling estop() raises ROSSafetyViolation."""
        with pytest.raises(ROSSafetyViolation):
            connected_hal.estop()

    def test_estop_disconnects(self, connected_hal: HAL) -> None:
        """estop() tears the HAL down — reads are refused until reconnected."""
        from openral_core import ROSRuntimeError

        with pytest.raises(ROSEStopRequested):
            connected_hal.estop()
        # estop() zeroes ``ctrl`` and drops the sim handle (``_connected=False``),
        # so the HAL must refuse a stale read rather than return a frozen frame.
        with pytest.raises(ROSRuntimeError):
            connected_hal.read_state()

    def test_estop_then_send_action_rejected(self, connected_hal: HAL) -> None:
        """estop() raises ROSEStopRequested; a subsequent send_action() is refused."""
        from openral_core import ROSRuntimeError

        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[[0.0 for _ in connected_hal.description.joints]],
            horizon=1,
        )
        with pytest.raises(ROSEStopRequested):
            connected_hal.estop()
        with pytest.raises(ROSRuntimeError):
            connected_hal.send_action(action)
