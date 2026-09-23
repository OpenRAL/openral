"""Fleet conformance: every committed `hal.real` adapter can be stopped from `/openral/estop`.

Derived from the manifests, not from a hand-kept list: every `robots/*/robot.yaml` whose
`hal.real` is set is built through the production `build_hal(mode="real")` seam and held to
the lifecycle e-stop contract (issue #295):

1. it implements `LifecycleEStopHAL` — so `HALLifecycleNodeBase._on_estop` forwards the stop
   to it rather than latching locally and hoping;
2. its `estop_recovery` is an explicit `EStopRecovery` policy, and a `RESETTABLE` adapter also
   implements `reset_estop`;
3. a ros2_control adapter is `ControllerStoppable` (so the lifecycle node wires the
   `controller_manager` stop seam to it) and every command topic it publishes to belongs to a
   controller its stop deactivates — a topic outside the stop is a controller the e-stop
   would leave running;
4. an Interbotix adapter is `InterbotixStoppable`;
5. it reports its downstream stop (`DownstreamStopReporting`) — the lifecycle node logs FATAL
   on an unacknowledged one, which only works if the HAL can say.

Adapters that own their own bus or sidecar (SO-100/SO-101 over serial, Galaxea A1 over its
ROS 1 sidecar) prove their stop in their own gates (`tests/unit/test_so100_follower_hal.py`
against the digital twin, `tests/hil/test_so101_serial_live.py`, `tests/hil/test_galaxea_a1*.py`);
here they are held to (1) and (2) only, which is the part a new adapter can forget.

A new real HAL that omits any of this fails here before it ever reaches a robot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_hal.protocol import (
    HAL,
    EStopRecovery,
    LifecycleEStopHAL,
    ResettableLifecycleEStopHAL,
)
from openral_hal.resolver import build_hal
from openral_hal.ros_control import (
    ControllerStoppable,
    DownstreamStopReporting,
)
from openral_hal.ros_control_transport import RosControlDrivable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFESTS = sorted(_REPO_ROOT.glob("robots/*/robot.yaml"))

#: Construction overrides that keep a real HAL constructible off-robot. Keys a
#: constructor does not accept are dropped by `build_hal`; nothing here connects.
_OFF_ROBOT_KWARGS: dict[str, object] = {"require_can_links": False}


def _real_manifests() -> list[Path]:
    """Every ``robots/*/robot.yaml`` whose ``hal.real`` is set."""
    out = []
    for path in _MANIFESTS:
        desc = RobotDescription.from_yaml(str(path))
        if desc.hal.real:
            out.append(path)
    return out


_REAL = _real_manifests()


def _build(path: Path) -> HAL:
    """Build the manifest's real HAL off-robot through ``build_hal``."""
    desc = RobotDescription.from_yaml(str(path))
    return build_hal(desc, mode="real", transport=_OFF_ROBOT_KWARGS)


def test_the_fleet_is_not_empty() -> None:
    """Guards the glob itself: an empty parametrization would pass vacuously."""
    assert {p.parent.name for p in _REAL} >= {
        "aloha_bimanual",
        "franka_panda",
        "galaxea_a1",
        "openarm",
        "sawyer",
        "so100_follower",
        "so101_follower",
        "ur5e",
        "ur10e",
    }


@pytest.mark.parametrize("manifest", _REAL, ids=[p.parent.name for p in _REAL])
def test_every_real_hal_opts_into_the_lifecycle_estop_with_an_explicit_policy(
    manifest: Path,
) -> None:
    """Every real hal opts into the lifecycle estop with an explicit policy."""
    hal = _build(manifest)
    assert isinstance(hal, LifecycleEStopHAL), (
        f"{type(hal).__name__} ({manifest.parent.name}) is not a LifecycleEStopHAL: "
        "/openral/estop would latch locally and never reach its hardware."
    )
    policy = hal.estop_recovery
    assert isinstance(policy, EStopRecovery), f"{type(hal).__name__}: {policy!r}"
    if policy is EStopRecovery.RESETTABLE:
        assert isinstance(hal, ResettableLifecycleEStopHAL), (
            f"{type(hal).__name__} declares RESETTABLE without reset_estop()."
        )


@pytest.mark.parametrize("manifest", _REAL, ids=[p.parent.name for p in _REAL])
def test_every_ros2_control_real_hal_can_stop_every_controller_it_commands(
    manifest: Path,
) -> None:
    """Every ros2 control real hal can stop every controller it commands."""
    hal = _build(manifest)
    if not isinstance(hal, RosControlDrivable):
        pytest.skip(f"{type(hal).__name__} is not a ros2_control adapter (own bus / sidecar).")
    assert isinstance(hal, ControllerStoppable), (
        f"{type(hal).__name__} is RosControlDrivable but not ControllerStoppable; the "
        "lifecycle node would refuse to configure it in real mode."
    )
    assert isinstance(hal, DownstreamStopReporting)
    controllers = hal.controller_names()
    assert controllers, f"{type(hal).__name__} names no controller to deactivate."
    for topic in hal.command_bindings():
        owner = topic.strip("/").split("/")[0]
        assert owner in controllers, (
            f"{type(hal).__name__} publishes on {topic!r} but its e-stop only deactivates "
            f"{controllers}: that controller would keep running."
        )


@pytest.mark.parametrize("manifest", _REAL, ids=[p.parent.name for p in _REAL])
def test_every_interbotix_real_hal_can_cut_torque_on_every_arm(manifest: Path) -> None:
    """Every interbotix real hal can cut torque on every arm."""
    from openral_hal.aloha import InterbotixStoppable

    hal = _build(manifest)
    if not isinstance(hal, InterbotixStoppable):
        pytest.skip(f"{type(hal).__name__} is not an Interbotix adapter.")
    assert isinstance(hal, DownstreamStopReporting)
    assert hal.arm_namespaces(), f"{type(hal).__name__} names no arm to torque off."


def test_a_real_hal_without_the_contract_is_rejected() -> None:
    """The checks above must fail on a HAL that only has the five Protocol methods.

    A stand-in that satisfies `HAL` and nothing else — the shape of the adapters this issue
    found (`estop()` flips a flag and raises) — is exactly what the fleet test exists to keep
    out. It is the input under test, not a double for anything the assertions rely on.
    """
    from types import SimpleNamespace

    from openral_core.exceptions import ROSEStopRequested

    class LatchOnly:
        description = SimpleNamespace(name="latch_only")

        def connect(self) -> None:
            """Protocol stub for the latch-only shape under test."""
            ...

        def disconnect(self) -> None:
            """Protocol stub for the latch-only shape under test."""
            ...

        def read_state(self) -> None:
            """Protocol stub for the latch-only shape under test."""
            ...

        def send_action(self, action: object) -> None:
            """Protocol stub for the latch-only shape under test."""
            ...

        def estop(self) -> None:
            """Protocol stub for the latch-only shape under test."""
            raise ROSEStopRequested("latched")

    assert isinstance(LatchOnly(), HAL)
    assert not isinstance(LatchOnly(), LifecycleEStopHAL)
    assert not isinstance(LatchOnly(), ControllerStoppable)
    assert not isinstance(LatchOnly(), DownstreamStopReporting)
