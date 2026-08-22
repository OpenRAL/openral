"""HIL test: a physically wired Enactic OpenArm v2 on this host.

Skipped automatically unless both CAN interfaces the OpenArm provisioning
pins by udev are present and up.  On a lab host with the arm wired in, this
is the only test that exercises the real transport end-to-end.

Two tiers, deliberately separated:

- **Transport** — the bus exists, is CAN FD at the OpenArm's bitrates, and
  ``openral detect`` identifies the arm from it.  Runs whenever the links
  are up, whether or not the motors are powered.  Read-only: nothing here
  opens a CAN socket.
- **Motor round-trip** — the motors answer.  Additionally requires the
  ``openarm_can`` bindings and a bus in ``ERROR-ACTIVE``; an unpowered arm
  leaves the links up but in ``ERROR-PASSIVE``, which is a skip and not a
  failure, because it is a fact about the bench, not about the code.

The motor round-trip is a **read**: it queries motor state and never calls
``enable_all()``, so it cannot energise or move the arm.  Commanding motion
belongs to the ros2_control stack (CLAUDE.md §1.5), not to a pytest process.
"""

from __future__ import annotations

import pytest

_LEFT = "openarm_left"
_RIGHT = "openarm_right"

# Motor CAN ids and types for the v2 7-DoF arm + 1 gripper, from
# `openarm_hardware`'s V10 default configuration (the same table the
# ros2_control plugin uses).
_ARM_SEND_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
_ARM_RECV_IDS = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
_GRIPPER_SEND_ID = 0x08
_GRIPPER_RECV_ID = 0x18


def _openarm_links() -> dict[str, object]:
    """Return the OpenArm's CAN interfaces by name, if the host has them."""
    from openral_cli.autodetect import enumerate_can_interfaces

    return {i.name: i for i in enumerate_can_interfaces() if i.name in (_LEFT, _RIGHT)}


def _links_are_up() -> bool:
    links = _openarm_links()
    return len(links) == 2 and all(getattr(i, "is_up", False) for i in links.values())


def _motors_answer() -> bool:
    """True when both buses are ERROR-ACTIVE — i.e. something is ACKing."""
    if not _links_are_up():
        return False
    return all(getattr(i, "state", "") == "ERROR-ACTIVE" for i in _openarm_links().values())


requires_links = pytest.mark.skipif(
    not _links_are_up(),
    reason=f"{_LEFT} / {_RIGHT} are not both up — no OpenArm wired to this host",
)
requires_powered_motors = pytest.mark.skipif(
    not _motors_answer(),
    reason="OpenArm CAN links are up but not ERROR-ACTIVE — motors unpowered",
)


# ── Transport ─────────────────────────────────────────────────────────────────


@requires_links
def test_both_buses_are_can_fd_at_the_openarm_bitrates() -> None:  # pragma: no cover
    # The Damiao motors run CAN FD: 1 Mbit/s arbitration, 5 Mbit/s data.
    # A classic-CAN link at 500k would enumerate fine and never talk.
    for name, iface in _openarm_links().items():
        assert iface.fd_enabled, f"{name} is not in CAN FD mode"
        assert iface.bitrate == 1_000_000, f"{name} nominal bitrate is {iface.bitrate}"
        assert iface.data_bitrate == 5_000_000, f"{name} data bitrate is {iface.data_bitrate}"
        assert iface.mtu == 72, f"{name} MTU {iface.mtu} is not a CAN FD MTU"


@requires_links
def test_detect_identifies_the_arm_from_the_bus_alone() -> None:  # pragma: no cover
    # No ROS running, no USB serial device — CAN is the only evidence.
    from openral_detect import detect_hardware
    from openral_detect.assemble import _infer_bh_robot_type

    report = detect_hardware(dds_timeout_s=0.0, include={"can"})
    assert [m.bh_robot_type for m in report.can.matches] == ["openarm"]
    assert _infer_bh_robot_type(report) == "openarm"


@requires_links
def test_the_real_hal_passes_its_bus_preflight() -> None:  # pragma: no cover
    from pathlib import Path

    from openral_core.schemas import RobotDescription
    from openral_hal.openarm_real import OpenArmRealHAL

    manifest = Path(__file__).resolve().parents[2] / "robots" / "openarm" / "robot.yaml"
    hal = OpenArmRealHAL(RobotDescription.from_yaml(str(manifest)))
    hal.connect()
    try:
        assert hal.health().fields["left_can"] == _LEFT
        assert hal.health().fields["right_can"] == _RIGHT
    finally:
        hal.disconnect()


@requires_links
def test_an_unpowered_bus_is_diagnosed_not_silently_passed() -> None:  # pragma: no cover
    # Whichever state the bench is in, the report must say so in words.
    from openral_detect.probes import probe_can

    warnings: list[str] = []
    result = probe_can(warnings=warnings)
    passive = [i.name for m in result.matches for i in m.interfaces if i.state == "ERROR-PASSIVE"]
    if passive:
        assert any("unpowered" in w for w in warnings), (
            f"{passive} are ERROR-PASSIVE but no warning explained it"
        )
    else:
        assert not any("unpowered" in w for w in warnings)


# ── Motor round-trip ──────────────────────────────────────────────────────────


@requires_links
@requires_powered_motors
def test_every_motor_answers_a_state_query() -> None:  # pragma: no cover
    """Each arm's 7 joints + gripper reply to a refresh over CAN FD.

    Read-only: ``enable_all()`` is never called, so the motors stay
    de-energised and the arm cannot move.
    """
    oa = pytest.importorskip("openarm_can", reason="openarm_can bindings not installed")

    motor_types = [
        oa.MotorType.DM8009,
        oa.MotorType.DM8009,
        oa.MotorType.DM4340,
        oa.MotorType.DM4340,
        oa.MotorType.DM4310,
        oa.MotorType.DM4310,
        oa.MotorType.DM4310,
    ]
    for interface in (_LEFT, _RIGHT):
        arm = oa.OpenArm(interface, True)
        arm.init_arm_motors(motor_types, _ARM_SEND_IDS, _ARM_RECV_IDS, [oa.ControlMode.MIT] * 7)
        arm.init_gripper_motor(
            oa.MotorType.DM4310, _GRIPPER_SEND_ID, _GRIPPER_RECV_ID, oa.ControlMode.MIT
        )
        arm.set_callback_mode_all(oa.CallbackMode.STATE)
        arm.refresh_all()
        arm.recv_all(5000)

        motors = list(arm.get_arm().get_motors()) + list(arm.get_gripper().get_motors())
        assert len(motors) == 8, f"{interface}: expected 8 motors, got {len(motors)}"
        # A motor that never replied leaves position/velocity/torque all at
        # exactly 0.0.  A real arm at rest has non-zero torque on at least
        # the gravity-loaded joints, and encoder noise on the rest.
        readings = [(m.get_position(), m.get_velocity(), m.get_torque()) for m in motors]
        assert any(any(v != 0.0 for v in r) for r in readings), (
            f"{interface}: every motor read exactly zero — none of them answered"
        )
