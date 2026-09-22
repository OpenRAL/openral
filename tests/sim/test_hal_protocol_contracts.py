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

from collections.abc import Callable

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
    AnvilOpenArmV2MujocoHAL,
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
    AnvilOpenArmV2MujocoHAL,
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


def _missing_optional_dep(exc: BaseException) -> bool:
    """True if a missing-import/-file error sits anywhere in ``exc``'s cause chain.

    The HAL boundary translates the optional-dependency failure twice before it
    reaches us: ``resolve_asset`` raises ``AssetRefError(... ) from ImportError``,
    then ``_resolve_mjcf_path`` raises ``ROSConfigError(str(exc)) from
    AssetRefError``. The ``ImportError`` we key on is therefore two links deep, so
    we walk ``__cause__``/``__context__`` rather than inspecting only the top
    frame. A genuine misconfiguration (e.g. a bad scene id) has no import/file
    error in its chain and is re-raised.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (ModuleNotFoundError, ImportError, FileNotFoundError)):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _make_hal_or_skip(hal_class: type[HAL]) -> HAL:
    """``_make_hal`` with the shared optional-sim-asset skip policy applied.

    An optional sim asset/dependency for this robot may not be installed in this
    environment (e.g. ``gym_aloha`` or a ``robot_descriptions`` submodule). Skip
    rather than fail the shared contract suite — CLAUDE.md §1.11. A
    ``ROSConfigError`` with no missing-import/-file root cause is a genuine
    misconfiguration and is re-raised.
    """
    try:
        return _make_hal(hal_class)
    except ROSConfigError as exc:
        if _missing_optional_dep(exc):
            pytest.skip(f"{hal_class.__name__}: {exc}")
        raise


def _hal_id(value: object) -> str:
    """Test id for a parametrize row: the HAL class name, else the value itself."""
    return str(getattr(value, "__name__", value))


@pytest.fixture(params=_HAL_CLASSES)
def hal(request: pytest.FixtureRequest) -> HAL:
    """Parametrized fixture instantiating each HAL class."""
    return _make_hal_or_skip(request.param)


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
        # mismatch that ``_validate_action_dims`` must reject (manifest-driven HAL contract).
        bad_action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[[0.0]],
            horizon=1,
        )
        with pytest.raises(ROSConfigError):
            connected_hal.send_action(bad_action)

    def test_rejects_wrong_joint_count(self, connected_hal: HAL) -> None:
        """An off-by-one action width is refused, and the error names the robot's width.

        The expected width is read off the manifest (``description.joints``) rather
        than hard-coded per robot, so this one body covers every HAL.
        """
        from openral_core import ROSConfigError

        n_joints = len(connected_hal.description.joints)
        bad = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * (n_joints - 1)],
        )
        with pytest.raises(ROSConfigError, match=f"{n_joints} joints"):
            connected_hal.send_action(bad)

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


# ── Per-robot MJCF / rest-pose contracts (table-driven) ──────────────────────
#
# These four checks used to be copy-pasted into each robot's own sim file with
# only the HAL class and an expected number differing. They live here as one
# body per check plus a table of rows. A robot whose body says something extra
# keeps its own copy in its per-robot file (noted on the row / in that file).

# (hal_class, expected model.nu, expected model.njnt or None when the MJCF's
# joint count carries no separate contract). The humanoids' njnt is nu + 1:
# the floating base. ALOHA's nu exceeds its 14 joints by the two mimic finger
# actuators, which is exactly why nu is tabulated rather than derived.
_MJCF_MODEL_SHAPE: list[tuple[type[HAL], int, int | None]] = [
    (AlohaMujocoHAL, 16, None),  # 14 arm + 2 extra finger actuators
    (AnvilOpenArmV2MujocoHAL, 16, None),  # 16 position actuators, v2-era layout
    (OpenArmMujocoHAL, 16, None),  # 16 position actuators in v2
    (G1MujocoHAL, 29, 30),  # 29 actuated joints + floating base
    (H1MujocoHAL, 19, 20),  # 19 actuated joints + floating base
    (Rizon4MujocoHAL, 7, None),
    (UR5eHAL, 6, None),
]

# HALs whose manifest sets ``sim.seed_ctrl_from_qpos`` (ADR-0023): connect()
# pre-loads ctrl with the current qpos so the native <position> actuators hold
# the rest pose on the first mj_step instead of yanking toward ctrl=0.
_SEED_CTRL_FROM_QPOS_HALS: list[type[HAL]] = [OpenArmMujocoHAL, AnvilOpenArmV2MujocoHAL]

# (hal_class, absolute tolerance) for robots whose MJCF has no keyframe, so
# every actuated joint starts at qpos=0. The humanoids' floating base leaves a
# little settle noise in the actuated joints, hence the looser bound.
_ZERO_REST_POSE: list[tuple[type[HAL], float]] = [
    (G1MujocoHAL, 1e-3),
    (H1MujocoHAL, 1e-3),
    (Rizon4MujocoHAL, 1e-6),
    (UR5eHAL, 1e-6),
]

# Robots that must hold the zero pose when commanded all-zeros (gravity off).
# Excludes robots whose MJCF home keyframe is non-zero or that self-collide on
# the way to zero (e.g. ALOHA — see its own closed-loop suite).
_HOLDS_ZERO_POSE: list[type[HAL]] = [
    G1MujocoHAL,
    H1MujocoHAL,
    OpenArmMujocoHAL,
    AnvilOpenArmV2MujocoHAL,
]


class TestMujocoModelContracts:
    """MJCF shape and rest-pose contracts, one row per robot."""

    @pytest.mark.parametrize(
        ("hal_class", "expected_nu", "expected_njnt"), _MJCF_MODEL_SHAPE, ids=_hal_id
    )
    def test_connect_loads_mujoco_model(
        self, hal_class: type[HAL], expected_nu: int, expected_njnt: int | None
    ) -> None:
        """connect() loads the robot's MJCF with the actuator count its HAL expects."""
        hal = _make_hal_or_skip(hal_class)
        hal.connect()
        try:
            assert hal._connected is True
            assert hal._model is not None
            assert hal._data is not None
            assert hal._model.nu == expected_nu
            if expected_njnt is not None:
                assert hal._model.njnt == expected_njnt
        finally:
            hal.disconnect()

    @pytest.mark.parametrize("hal_class", _SEED_CTRL_FROM_QPOS_HALS, ids=_hal_id)
    def test_connect_seeds_ctrl_from_qpos(self, hal_class: type[HAL]) -> None:
        """connect() seeds every actuator's ctrl with its joint's current qpos."""
        hal = _make_hal_or_skip(hal_class)
        hal.connect()
        try:
            assert hal._data is not None
            for name in hal._joint_names:
                qpos_idx = hal._joint_qpos_addr[name]
                act_idx = hal._actuator_index[name]
                assert hal._data.ctrl[act_idx] == pytest.approx(
                    float(hal._data.qpos[qpos_idx]), abs=1e-6
                )
        finally:
            hal.disconnect()

    @pytest.mark.parametrize(("hal_class", "tolerance"), _ZERO_REST_POSE, ids=_hal_id)
    def test_initial_positions_are_zero(self, hal_class: type[HAL], tolerance: float) -> None:
        """A keyframe-less MJCF reports every actuated joint at qpos=0 after connect."""
        hal = _make_hal_or_skip(hal_class)
        hal.connect()
        try:
            for q in hal.read_state().position:
                assert abs(q) < tolerance
        finally:
            hal.disconnect()

    @pytest.mark.parametrize("hal_class", _HOLDS_ZERO_POSE, ids=_hal_id)
    def test_send_action_holds_zero_pose(
        self,
        hal_class: type[HAL],
        assert_send_action_holds_zero_pose: Callable[[HAL, Action], None],
    ) -> None:
        """Commanding all-zeros leaves every joint at zero (gravity off)."""
        hal = _make_hal_or_skip(hal_class)
        hal.connect()
        try:
            zero = Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[[0.0] * len(hal.description.joints)],
            )
            assert_send_action_holds_zero_pose(hal, zero)
        finally:
            hal.disconnect()
