"""HAL Protocol conformance — parametrized contract test for every concrete HAL.

CLAUDE.md §5.1 says *"types are the contract"*; this file pins the runtime
contract :class:`openral_hal.protocol.HAL` declares so a typo or
signature drift in any HAL implementation fails the unit lane immediately
instead of waiting for a sim or HIL run.

Why a parametrized test instead of per-HAL contract files?  Ten HALs ship
today (`RosControlHAL`, `SO100FollowerHAL`, `UR5eHAL`, `UR10eHAL`,
`FrankaPandaHAL`, `FrankaPandaRealHAL`, `SawyerRealHAL`, `AlohaHAL`,
`UR5eRealHAL`, `UR10eRealHAL`); duplicating the same assertions across
ten files would mean every Protocol change touches ten files.  Here a
new HAL becomes a one-line entry in :data:`HAL_BUILDERS`.

The five real-HW adapters (Franka FCI, Sawyer, ALOHA, UR5e, UR10e) are
exercised against an injected
:class:`~openral_hal.sim_transport.SimTransport` (a recorded RTDE-shaped
fixture per CLAUDE.md §5.4); their live HIL runs live in
``tests/hil/test_franka_panda.py``, ``test_sawyer.py``, ``test_aloha.py``,
``test_ur5e.py`` and ``test_ur10e.py`` and are gated by the lab runner
labels.

Coverage (asserted against every HAL listed in :data:`HAL_BUILDERS`)
-------------------------------------------------------------------
- ``description`` attribute is a populated :class:`RobotDescription`.
- The instance is structurally a :class:`HAL` (``runtime_checkable``).
- ``read_state`` / ``send_action`` before ``connect`` raise
  :class:`ROSRuntimeError` (per :class:`HAL` docstring).
- ``connect`` then ``read_state`` returns a :class:`JointState` whose
  ``name`` matches ``description.joints``.
- ``disconnect`` is idempotent (calling twice doesn't raise).
- ``estop`` always raises :class:`ROSEStopRequested` (a
  :class:`ROSSafetyViolation` subclass — never silently caught).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from openral_core import (
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_core.exceptions import ROSEStopRequested, ROSRuntimeError
from openral_core.schemas import JointState
from openral_hal.protocol import HAL
from openral_hal.ros_control import RosControlHAL
from openral_hal.sim_transport import SimTransport

if TYPE_CHECKING:
    pass


# ── HAL builders ─────────────────────────────────────────────────────────────

# Each builder returns ``(hal, cleanup)`` where ``cleanup`` is called in
# ``finally`` regardless of test outcome.  The builder is responsible for
# any optional-dep ``importorskip`` calls so tests skip cleanly on machines
# without MuJoCo / robot_descriptions / lerobot.

HALBuilder = Callable[[], tuple[HAL, Callable[[], None]]]


def _build_minimal_description(n_joints: int = 2) -> RobotDescription:
    return RobotDescription(
        name="conformance_robot",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name=f"j{i}",
                joint_type=JointType.REVOLUTE,
                parent_link="base" if i == 0 else f"link_{i - 1}",
                child_link=f"link_{i}",
            )
            for i in range(n_joints)
        ],
        capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        safety=SafetyEnvelope(),
    )


def _ros_control_builder() -> tuple[HAL, Callable[[], None]]:
    desc = _build_minimal_description(n_joints=2)
    transport = SimTransport(n_joints=2)
    hal = RosControlHAL(
        desc,
        controller_name="conformance_ctrl",
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    return hal, lambda: None


def _so100_follower_builder() -> tuple[HAL, Callable[[], None]]:
    pytest.importorskip("lerobot")
    from openral_hal.so100_follower import SO100FollowerHAL  # reason: optional dep
    from openral_hal.so100_sim import (  # reason: optional dep
        SO100DigitalTwin,
        SO100DigitalTwinConfig,
    )

    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    hal = SO100FollowerHAL(robot=twin)
    return hal, lambda: None


def _ur5e_builder() -> tuple[HAL, Callable[[], None]]:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco not installed")
    if importlib.util.find_spec("robot_descriptions") is None:
        pytest.skip("robot_descriptions not installed")
    from openral_hal.ur import UR5eHAL  # reason: optional dep

    hal = UR5eHAL(gravity_enabled=False)
    return hal, lambda: None


def _ur10e_builder() -> tuple[HAL, Callable[[], None]]:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco not installed")
    if importlib.util.find_spec("robot_descriptions") is None:
        pytest.skip("robot_descriptions not installed")
    from openral_hal.ur import UR10eHAL  # reason: optional dep

    hal = UR10eHAL(gravity_enabled=False)
    return hal, lambda: None


def _franka_builder() -> tuple[HAL, Callable[[], None]]:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco not installed")
    if importlib.util.find_spec("robot_descriptions") is None:
        pytest.skip("robot_descriptions not installed")
    from openral_hal.franka_panda import FrankaPandaHAL  # reason: optional dep

    hal = FrankaPandaHAL(gravity_enabled=False)
    return hal, lambda: None


def _franka_real_builder() -> tuple[HAL, Callable[[], None]]:
    """Build a :class:`FrankaPandaRealHAL` against an in-memory transport.

    Real ``franka_ros2`` / ``libfranka`` are not part of the unit lane; the
    transport is a real :class:`SimTransport` that records publishes and
    feeds back zeroed joint state.  This exercises the full HAL Protocol
    contract without any ROS 2 installation.
    """
    from openral_hal.franka_panda_real import FrankaPandaRealHAL

    n = len(_franka_n_joints())
    transport = SimTransport(n_joints=n)
    hal = FrankaPandaRealHAL(
        fci_ip="172.16.0.2",
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    return hal, lambda: None


def _franka_n_joints() -> list[str]:
    from openral_hal.franka_panda import FRANKA_PANDA_DESCRIPTION

    return [j.name for j in FRANKA_PANDA_DESCRIPTION.joints]


def _sawyer_real_builder() -> tuple[HAL, Callable[[], None]]:
    """Build a :class:`SawyerRealHAL` against an in-memory transport."""
    from openral_hal.sawyer_real import SAWYER_DESCRIPTION, SawyerRealHAL

    transport = SimTransport(n_joints=len(SAWYER_DESCRIPTION.joints))
    hal = SawyerRealHAL(
        hostname="sawyer.local",
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    return hal, lambda: None


def _aloha_builder() -> tuple[HAL, Callable[[], None]]:
    """Build an :class:`AlohaHAL` against an in-memory transport."""
    from openral_hal.aloha import ALOHA_DESCRIPTION, AlohaHAL

    transport = SimTransport(n_joints=len(ALOHA_DESCRIPTION.joints))
    hal = AlohaHAL(
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    return hal, lambda: None


def _ur5e_real_builder() -> tuple[HAL, Callable[[], None]]:
    """Build a UR5eRealHAL backed by a SimTransport (recorded RTDE fixture).

    Runs without a live UR5e or live ros2_control by injecting the same
    typed in-memory transport the RosControlHAL unit tests use; the HIL
    counterpart is ``tests/hil/test_ur5e.py``.
    """
    from openral_hal.ur_real import UR5eRealHAL

    transport = SimTransport(n_joints=6)
    hal = UR5eRealHAL(
        robot_ip="192.0.2.10",
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    return hal, lambda: None


def _ur10e_real_builder() -> tuple[HAL, Callable[[], None]]:
    """Build a UR10eRealHAL backed by a SimTransport.

    Same shape as :func:`_ur5e_real_builder`; HIL counterpart is
    ``tests/hil/test_ur10e.py``.
    """
    from openral_hal.ur_real import UR10eRealHAL

    transport = SimTransport(n_joints=6)
    hal = UR10eRealHAL(
        robot_ip="192.0.2.11",
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    return hal, lambda: None


HAL_BUILDERS: dict[str, HALBuilder] = {
    "RosControlHAL": _ros_control_builder,
    "SO100FollowerHAL+SO100DigitalTwin": _so100_follower_builder,
    "UR5eHAL": _ur5e_builder,
    "UR10eHAL": _ur10e_builder,
    "FrankaPandaHAL": _franka_builder,
    "FrankaPandaRealHAL": _franka_real_builder,
    "SawyerRealHAL": _sawyer_real_builder,
    "AlohaHAL": _aloha_builder,
    "UR5eRealHAL+SimTransport": _ur5e_real_builder,
    "UR10eRealHAL+SimTransport": _ur10e_real_builder,
}


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("hal_name", list(HAL_BUILDERS.keys()))
def test_hal_has_robot_description(hal_name: str) -> None:
    hal, cleanup = HAL_BUILDERS[hal_name]()
    try:
        assert isinstance(hal.description, RobotDescription)
        assert len(hal.description.joints) >= 1
    finally:
        cleanup()


@pytest.mark.parametrize("hal_name", list(HAL_BUILDERS.keys()))
def test_hal_satisfies_runtime_checkable_protocol(hal_name: str) -> None:
    """Every HAL must structurally satisfy the ``HAL`` Protocol at runtime."""
    hal, cleanup = HAL_BUILDERS[hal_name]()
    try:
        assert isinstance(hal, HAL)
    finally:
        cleanup()


@pytest.mark.parametrize("hal_name", list(HAL_BUILDERS.keys()))
def test_hal_read_state_before_connect_raises(hal_name: str) -> None:
    hal, cleanup = HAL_BUILDERS[hal_name]()
    try:
        with pytest.raises(ROSRuntimeError):
            hal.read_state()
    finally:
        cleanup()


@pytest.mark.parametrize("hal_name", list(HAL_BUILDERS.keys()))
def test_hal_send_action_before_connect_raises(hal_name: str) -> None:
    from openral_core.schemas import Action  # reason: keep imports lazy

    hal, cleanup = HAL_BUILDERS[hal_name]()
    try:
        n = len(hal.description.joints)
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * n],
        )
        with pytest.raises(ROSRuntimeError):
            hal.send_action(action)
    finally:
        cleanup()


@pytest.mark.parametrize("hal_name", list(HAL_BUILDERS.keys()))
def test_hal_connect_then_read_state_returns_joint_state(hal_name: str) -> None:
    hal, cleanup = HAL_BUILDERS[hal_name]()
    try:
        hal.connect()
        state = hal.read_state()
        assert isinstance(state, JointState)
        # Names must align with the description's joint inventory.
        assert state.name == [j.name for j in hal.description.joints]
        assert len(state.position) == len(state.name)
        hal.disconnect()
    finally:
        cleanup()


@pytest.mark.parametrize("hal_name", list(HAL_BUILDERS.keys()))
def test_hal_disconnect_is_idempotent(hal_name: str) -> None:
    hal, cleanup = HAL_BUILDERS[hal_name]()
    try:
        hal.connect()
        hal.disconnect()
        # Second disconnect must not raise.
        hal.disconnect()
    finally:
        cleanup()


@pytest.mark.parametrize("hal_name", list(HAL_BUILDERS.keys()))
def test_hal_estop_always_raises_estoprequested(hal_name: str) -> None:
    """The :class:`HAL` Protocol mandates that ``estop`` always raises.

    Per CLAUDE.md §10 / Protocol docstring, ``ROSEStopRequested`` is the
    exact exception type.  It MUST be a ``ROSSafetyViolation`` subclass
    so the safety supervisor catches it at the boundary; that is asserted
    structurally at the import site (one test).
    """
    hal, cleanup = HAL_BUILDERS[hal_name]()
    try:
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()
    finally:
        cleanup()


def test_estoprequested_is_safety_violation_subclass() -> None:
    """One-time structural check — ``ROSEStopRequested`` must inherit from
    ``ROSSafetyViolation`` so safety-supervisor handlers catching the latter
    also catch the former.  Catches accidental hierarchy refactors.
    """
    from openral_core.exceptions import ROSSafetyViolation

    assert issubclass(ROSEStopRequested, ROSSafetyViolation)
