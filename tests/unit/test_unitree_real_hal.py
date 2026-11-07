# SPDX-License-Identifier: Apache-2.0
"""`UnitreeHgRealHAL` — the real G1 HAL proposes; NVIDIA's C++ S0 graph disposes.

Driven through the production HAL code path over `SimTransport` (the typed in-memory
ros2_control transport + controller table; no mocks, CLAUDE.md §1.11) and built from the
real `robots/g1/robot.yaml` through `build_hal`, exactly as `deploy run` builds it.

What is pinned:

* legs belong to the S0 policy: any JOINT_POSITION that names or implies a leg joint is
  refused with `ROSCapabilityMismatch` and nothing reaches the wire;
* every upper-body message carries all upper-body joints (NVIDIA's forward controller
  snaps an omitted joint back to its activation pose);
* the velocity command is a heartbeat (zero unless a same-tick BODY_TWIST set it), so
  the policy's input-staleness timeout only fires when OpenRAL is really gone;
* the e-stop is ONE swap to `freeze_controller`, never a bare deactivation — the hg
  hardware interface puts unclaimed joints at zero torque, which would drop the robot.
"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import (
    ROSCapabilityMismatch,
    ROSConfigError,
    ROSEStopRequested,
)
from openral_core.schemas import Action, ControlMode
from openral_hal import build_hal
from openral_hal.protocol import EStopRecovery
from openral_hal.ros_control import (
    ControllerSwapSeam,
    ControllerSwitchReport,
    TriggerReport,
)
from openral_hal.sim_transport import SimTransport
from openral_hal.unitree_real import UnitreeHgRealHAL

_REPO = Path(__file__).resolve().parents[2]
_G1_YAML = _REPO / "robots" / "g1" / "robot.yaml"
_UPPER_TOPIC = "/upper_body_forward_joint_command_controller/joint_commands_trajectory"
_TWIST_TOPIC = "/inference_controller/cmd_vel"
_CHAIN = (
    "safety_controller",
    "inference_controller",
    "upper_body_forward_joint_command_controller",
)


def _desc() -> RobotDescription:
    return RobotDescription.from_yaml(str(_G1_YAML))


@functools.cache
def _names() -> tuple[str, ...]:
    return tuple(j.name for j in _desc().joints)


def _wired() -> tuple[UnitreeHgRealHAL, SimTransport]:
    """Real-mode HAL from the manifest, wired to a STRICT simulated controller table."""
    desc = _desc()
    hal = build_hal(desc, mode="real")
    assert isinstance(hal, UnitreeHgRealHAL)
    names = [j.name for j in desc.joints]
    transport = SimTransport(
        n_joints=len(names),
        joint_names=names,
        controllers=[*_CHAIN, "freeze_controller"],
    )
    # freeze_controller is loaded inactive by the bringup.
    transport.deactivate_controllers(["freeze_controller"], timeout_s=1.0)
    # SimTransport's state is current at every read, so "now" is its true arrival stamp.
    hal.attach_transport(transport.publish, transport.state, time.monotonic)
    hal.attach_controller_stop(transport)
    hal.connect()
    return hal, transport


def _row(values: dict[str, float]) -> list[float]:
    return [values.get(n, 0.0) for n in _names()]


def _joint_action(
    values: dict[str, float],
    *,
    joint_names: Sequence[str] | None,
    tick: int = 0,
    group: int = 1,
) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[_row(values)],
        joint_names=None if joint_names is None else list(joint_names),
        tick_index=tick,
        tick_group_size=group,
    )


def _twist_action(twist: tuple[float, ...], *, tick: int = 0, group: int = 1) -> Action:
    return Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[twist],
        tick_index=tick,
        tick_group_size=group,
    )


#: The default (GR00T Decoupled WBC) split's upper body: both 7-DoF arms, from the policy file.
_UPPER = tuple(UnitreeHgRealHAL().upper_body_joints())
_LEFT_ARM = [n for n in _UPPER if n.startswith("left_")]
_RIGHT_ARM = [n for n in _UPPER if n.startswith("right_")]


# ── Construction / wiring ─────────────────────────────────────────────────────


def test_build_hal_binds_the_g1_manifest_and_splits_the_29_joints() -> None:
    desc = _desc()
    hal = build_hal(desc, mode="real")
    assert isinstance(hal, UnitreeHgRealHAL)
    assert hal.description is desc
    assert hal.estop_recovery is EStopRecovery.RESTART_REQUIRED
    upper, s0 = set(hal.upper_body_joints()), set(hal.s0_joints())
    assert not upper & s0
    assert upper | s0 == {j.name for j in desc.joints}
    assert len(desc.joints) == 29
    assert all("hip" in n or "knee" in n or "ankle" in n or "waist" in n for n in s0)


def test_every_command_topic_belongs_to_a_controller_the_estop_swaps_out() -> None:
    hal = UnitreeHgRealHAL()
    assert list(hal.command_bindings()) == [_UPPER_TOPIC, _TWIST_TOPIC]
    owners = {t.strip("/").split("/")[0] for t in hal.command_bindings()}
    assert owners <= set(hal.controller_names())
    assert hal.controller_names() == list(_CHAIN)


def test_unknown_model_and_unknown_policies_are_refused() -> None:
    with pytest.raises(ROSConfigError, match="no hg-family model"):
        UnitreeHgRealHAL(robot_model="h7")
    with pytest.raises(ROSConfigError, match="does not exist"):
        UnitreeHgRealHAL(s0_policy="no_such_policy")


def test_the_joint_split_follows_the_selected_policy_file() -> None:
    """GR00T WBC (default) owns the whole waist; AGILE leaves waist yaw to OpenRAL."""
    gr00t = UnitreeHgRealHAL()
    agile = UnitreeHgRealHAL(s0_policy="agile_velocity")
    assert gr00t.s0_policy == "gr00t_decoupled_wbc"
    assert {"waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"} <= set(gr00t.s0_joints())
    assert len(gr00t.s0_joints()) == 15 and len(gr00t.upper_body_joints()) == 14
    assert "waist_yaw_joint" in agile.upper_body_joints()
    assert "waist_yaw_joint" not in agile.s0_joints()


def test_a_policy_that_does_not_cover_the_body_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A policy file whose split leaves a joint unowned is refused at construction."""
    import yaml

    src = _REPO / "packages" / "openral_hal_unitree" / "config"
    doc = yaml.safe_load((src / "g1" / "policies" / "gr00t_decoupled_wbc.yaml").read_text())
    doc["openral_policy"]["s0_joints"].remove("waist_yaw_joint")  # now owned by nobody
    (tmp_path / "g1" / "policies").mkdir(parents=True)
    (tmp_path / "g1" / "policies" / "gr00t_decoupled_wbc.yaml").write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("OPENRAL_UNITREE_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ROSConfigError, match="owned by nobody"):
        UnitreeHgRealHAL()


# ── Leg refusal ───────────────────────────────────────────────────────────────


def test_a_whole_vector_joint_action_claims_the_legs_and_is_refused() -> None:
    hal, transport = _wired()
    with pytest.raises(ROSCapabilityMismatch, match="S0 whole-body"):
        hal.send_action(_joint_action({}, joint_names=None))
    assert transport.calls == []


def test_naming_a_single_leg_joint_is_refused_before_anything_is_published() -> None:
    hal, transport = _wired()
    with pytest.raises(ROSCapabilityMismatch, match="left_knee_joint"):
        hal.send_action(
            _joint_action({"left_knee_joint": 0.3}, joint_names=[*_LEFT_ARM, "left_knee_joint"])
        )
    assert transport.calls == []


@pytest.mark.parametrize("joint", ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])
def test_the_policy_owned_waist_axes_are_refused_too(joint: str) -> None:
    hal, transport = _wired()
    with pytest.raises(ROSCapabilityMismatch):
        hal.send_action(_joint_action({joint: 0.1}, joint_names=[joint]))
    assert transport.calls == []


# ── Upper body ────────────────────────────────────────────────────────────────


def test_an_arm_command_carries_every_upper_body_joint_then_a_zero_twist() -> None:
    hal, transport = _wired()
    commanded = {n: 0.1 * (i + 1) for i, n in enumerate(_LEFT_ARM)}
    hal.send_action(_joint_action(commanded, joint_names=_LEFT_ARM))
    (upper_topic, upper), (twist_topic, twist) = transport.calls
    assert upper_topic == _UPPER_TOPIC
    assert upper["joint_names"] == list(_UPPER)
    targets = upper["joint_targets"]
    assert isinstance(targets, list)
    (row,) = targets
    by_name = dict(zip(_UPPER, row, strict=True))
    for n, v in commanded.items():
        assert by_name[n] == pytest.approx(v)
    # Unowned upper-body joints hold their measured position (SimTransport starts at 0).
    assert all(by_name[n] == 0.0 for n in _RIGHT_ARM)
    # One control period at the manifest's 20 Hz.
    assert upper["time_from_start_s"] == pytest.approx(1 / 20)
    assert twist_topic == _TWIST_TOPIC
    assert twist == {"twist": [0.0] * 6}


def test_a_second_arm_command_keeps_the_first_arms_target_not_the_activation_pose() -> None:
    hal, transport = _wired()
    left = {n: 0.2 for n in _LEFT_ARM}
    right = {n: -0.3 for n in _RIGHT_ARM}
    hal.send_action(_joint_action(left, joint_names=_LEFT_ARM))
    hal.send_action(_joint_action(right, joint_names=_RIGHT_ARM))
    upper_msgs = [m for t, m in transport.calls if t == _UPPER_TOPIC]
    targets = upper_msgs[-1]["joint_targets"]
    assert isinstance(targets, list)
    (row,) = targets
    by_name = dict(zip(_UPPER, row, strict=True))
    assert all(by_name[n] == pytest.approx(0.2) for n in _LEFT_ARM)
    assert all(by_name[n] == pytest.approx(-0.3) for n in _RIGHT_ARM)
    # And the simulated robot reached both (applied by name).
    state = hal.read_state()
    pos = dict(zip(state.name, state.position, strict=True))
    assert pos["left_elbow_joint"] == pytest.approx(0.2)
    assert pos["right_elbow_joint"] == pytest.approx(-0.3)
    assert pos["left_knee_joint"] == 0.0


def test_a_non_finite_upper_body_target_is_refused() -> None:
    hal, transport = _wired()
    with pytest.raises(ROSConfigError, match="non-finite"):
        hal.send_action(_joint_action({"left_elbow_joint": math.nan}, joint_names=_LEFT_ARM))
    assert transport.calls == []


# ── Body twist ────────────────────────────────────────────────────────────────


def test_body_twist_goes_to_the_policy_and_leaves_the_arms_alone() -> None:
    hal, transport = _wired()
    hal.send_action(_twist_action((0.3, -0.1, 0.0, 0.0, 0.0, 0.4)))
    assert transport.calls == [(_TWIST_TOPIC, {"twist": [0.3, -0.1, 0.0, 0.0, 0.0, 0.4]})]


@pytest.mark.parametrize(
    "twist",
    [(0.1, 0.0, 0.2, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.3, 0.0, 0.0), (math.inf,) + (0.0,) * 5],
)
def test_a_non_planar_or_non_finite_twist_is_refused(twist: tuple[float, ...]) -> None:
    hal, transport = _wired()
    with pytest.raises(ROSConfigError):
        hal.send_action(_twist_action(twist))
    assert transport.calls == []


def test_a_slot_group_sibling_republishes_the_ticks_twist_not_zero() -> None:
    hal, transport = _wired()
    twist = (0.2, 0.0, 0.0, 0.0, 0.0, 0.1)
    hal.send_action(_twist_action(twist, tick=7, group=2))
    hal.send_action(_joint_action({}, joint_names=_LEFT_ARM, tick=7, group=2))
    assert transport.calls[-1] == (_TWIST_TOPIC, {"twist": list(twist)})
    # The next tick without a twist slot falls back to zero.
    hal.send_action(_joint_action({}, joint_names=_LEFT_ARM, tick=8, group=2))
    assert transport.calls[-1] == (_TWIST_TOPIC, {"twist": [0.0] * 6})


def test_base_height_has_no_action_field_to_ride_on() -> None:
    """Documented limit: `Action` carries no base height, so there is nothing to forward."""
    assert "base_height" not in Action.model_fields
    assert "body_height" not in Action.model_fields


# ── E-stop ────────────────────────────────────────────────────────────────────


def test_estop_swaps_the_s0_chain_for_freeze_in_one_request() -> None:
    hal, transport = _wired()
    switches_before = list(transport.switch_calls)
    with pytest.raises(ROSEStopRequested, match="acknowledged"):
        hal.estop()
    new = transport.switch_calls[len(switches_before) :]
    assert new == [("swap", (*_CHAIN, "freeze_controller"))]
    report = hal.last_stop_report
    assert report is not None and report.stopped
    assert transport.controller_state("freeze_controller") == "active"
    assert all(transport.controller_state(c) == "inactive" for c in _CHAIN)
    # A late command after the stop never reaches the wire (and the HAL refuses it).
    transport.publish(_TWIST_TOPIC, {"twist": [1.0, 0, 0, 0, 0, 0]})
    assert transport.dropped_calls[-1][0] == _TWIST_TOPIC


def test_a_failing_swap_is_reported_unacknowledged_and_still_raises() -> None:
    desc = _desc()
    hal = build_hal(desc, mode="real")
    assert isinstance(hal, UnitreeHgRealHAL)
    transport = SimTransport(
        n_joints=len(desc.joints), switch_faults={"swap": RuntimeError("executor down")}
    )
    hal.attach_transport(transport.publish, transport.state)
    hal.attach_controller_stop(transport)
    hal.connect()
    with pytest.raises(ROSEStopRequested, match="NOT acknowledged"):
        hal.estop()
    report = hal.last_stop_report
    assert report is not None and not report.stopped
    assert "executor down" in report.detail


class _DeactivateOnlySeam:
    """The input under test: a stop seam that can deactivate but cannot swap."""

    def deactivate_controllers(
        self, names: Sequence[str], *, timeout_s: float
    ) -> ControllerSwitchReport:
        """Seam surface under test (never called: the HAL refuses the seam first)."""
        return ControllerSwitchReport(ok=True, controllers=tuple(names))

    def activate_controllers(
        self, names: Sequence[str], *, timeout_s: float
    ) -> ControllerSwitchReport:
        """Seam surface under test (never called)."""
        return ControllerSwitchReport(ok=True, controllers=tuple(names))

    def call_trigger(self, service: str, *, timeout_s: float) -> TriggerReport:
        """Seam surface under test (never called)."""
        return TriggerReport(success=True)

    def publish_empty(self, topic: str) -> None:
        """Seam surface under test (never called)."""


def test_a_seam_that_can_only_deactivate_is_refused_at_wire_up() -> None:
    seam = _DeactivateOnlySeam()
    assert not isinstance(seam, ControllerSwapSeam)
    hal = UnitreeHgRealHAL()
    with pytest.raises(ROSConfigError, match="zero torque"):
        hal.attach_controller_stop(seam)
    # Not left half-attached: an e-stop now reports "no seam", it never deactivates.
    hal.connect()
    with pytest.raises(ROSEStopRequested):
        hal.estop()
    assert hal.last_stop_report is not None
    assert not hal.last_stop_report.stopped


def test_the_production_transports_implement_the_swap_seam() -> None:
    from openral_hal.ros_control_transport import RosControlTransport

    assert isinstance(SimTransport(n_joints=1), ControllerSwapSeam)
    assert callable(RosControlTransport.swap_controllers)
