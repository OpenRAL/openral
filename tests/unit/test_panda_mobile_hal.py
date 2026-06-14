"""ADR-0025 — :class:`PandaMobileHAL` body_twist + joint_position contract.

Exercises the HAL Protocol surface against in-memory state — no
robosuite, no MuJoCo. Pinning the two-mode routing (BODY_TWIST for
the holonomic base, JOINT_POSITION for the arm) lets the higher
layers (safety supervisor, RskillRunnerNode, Nav2 rSkill replay) be
contract-tested without the full sim stack.
"""

from __future__ import annotations

import math

import pytest
from openral_core import ROSConfigError
from openral_core.schemas import Action, ControlMode
from openral_hal.panda_mobile import (
    PANDA_MOBILE_BASE_JOINT_NAMES,
    PANDA_MOBILE_JOINT_NAMES,
    PandaMobileHAL,
)


def test_connect_disconnect_state_round_trip() -> None:
    hal = PandaMobileHAL()
    hal.connect()
    state = hal.read_state()
    assert state.name == PANDA_MOBILE_JOINT_NAMES
    # ADR-0028a — 11 joints: 3 base + 7 arm + 1 panda_gripper.
    assert len(state.position) == 11
    assert all(v == 0.0 for v in state.position)
    hal.disconnect()
    with pytest.raises(ROSConfigError, match="connect"):
        hal.read_state()


def test_send_action_before_connect_rejected() -> None:
    hal = PandaMobileHAL()
    twist = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    with pytest.raises(ROSConfigError, match="connect"):
        hal.send_action(twist)


def test_body_twist_forward_integrates_base_x() -> None:
    """Forward twist (vx=1, others=0) for one dt advances base_x by dt."""
    hal = PandaMobileHAL(dt_s=0.1)
    hal.connect()
    twist = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    hal.send_action(twist)
    x, y, yaw = hal.base_pose
    assert x == pytest.approx(0.1)
    assert y == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)


def test_body_twist_respects_current_yaw() -> None:
    """Body-frame vx becomes world vy after a 90° yaw rotation."""
    hal = PandaMobileHAL(dt_s=0.1)
    hal.connect()
    # Rotate +π/2 first (wz·dt = π/2 ⇒ wz = 5π).
    spin = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2 / 0.1]],
    )
    hal.send_action(spin)
    assert hal.base_pose[2] == pytest.approx(math.pi / 2, abs=1e-9)
    # Now body-vx=1 should move world-y (because forward is now +y).
    forward = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    hal.send_action(forward)
    x, y, _yaw = hal.base_pose
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.1, abs=1e-9)


def test_base_twist_latches_command_then_clears_on_non_twist() -> None:
    """``base_twist`` (the /odom twist source) tracks the last BODY_TWIST."""
    hal = PandaMobileHAL(dt_s=0.1)
    hal.connect()
    assert hal.base_twist == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    hal.send_action(
        Action(
            control_mode=ControlMode.BODY_TWIST,
            horizon=1,
            body_twist=[[-0.3, 0.1, 0.0, 0.0, 0.0, 0.2]],
        )
    )
    assert hal.base_twist == pytest.approx((-0.3, 0.1, 0.0, 0.0, 0.0, 0.2))

    # A non-BODY_TWIST action clears the latch (base no longer commanded).
    hal.send_action(
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
        )
    )
    assert hal.base_twist == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_body_twist_rejects_non_planar_components() -> None:
    """Linear-z / angular-x / angular-y must be zero for a planar base."""
    hal = PandaMobileHAL()
    hal.connect()
    bad = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[0.0, 0.0, 0.5, 0.0, 0.0, 0.0]],  # vz != 0
    )
    with pytest.raises(ROSConfigError, match="holonomic planar"):
        hal.send_action(bad)


def test_joint_position_arm_only_sets_seven_arm_joints() -> None:
    hal = PandaMobileHAL()
    hal.connect()
    target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    hal.send_action(
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[target],
        )
    )
    state = hal.read_state()
    # First three slots (base) stay zero; arm slots receive the target;
    # gripper (last slot) stays zero (arm-only target).
    n_base = len(PANDA_MOBILE_BASE_JOINT_NAMES)
    assert state.position[:n_base] == [0.0, 0.0, 0.0]
    assert state.position[n_base : n_base + len(target)] == target
    assert state.position[-1] == 0.0  # gripper untouched by arm-only target


def test_joint_position_full_state_replay_accepted() -> None:
    """10-DoF replay sets every base+arm slot (used by MoveIt trajectory replay).

    The 10-wide form is preserved alongside the new 11-wide
    (base+arm+gripper) form per ADR-0028c — gripper stays where it
    was when the chunk's row is 10 floats.
    """
    hal = PandaMobileHAL()
    hal.connect()
    target = [1.0, 2.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    hal.send_action(
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[target],
        )
    )
    state = hal.read_state()
    assert state.position[: len(target)] == target
    assert state.position[-1] == 0.0  # gripper untouched by 10-wide replay


def test_joint_position_rejects_wrong_width() -> None:
    hal = PandaMobileHAL()
    hal.connect()
    # ADR-0028a — error message now lists three accepted widths
    # (arm-only / base+arm / base+arm+gripper).
    with pytest.raises(ROSConfigError, match=r"arm-only.*base\+arm.*gripper"):
        hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[[0.0, 0.0, 0.0]],
            )
        )


def test_cartesian_delta_tracks_latest_command() -> None:
    """ADR-0028c — CARTESIAN_DELTA is recorded for dashboard observability.

    The digital-twin HAL has no Jacobian; cartesian deltas don't move
    the qpos. They DO get stamped onto ``_last_cartesian_delta`` so
    the lifecycle node can surface "what did the policy command this
    tick" without involving the sim.
    """
    hal = PandaMobileHAL()
    hal.connect()
    delta = (0.01, 0.0, -0.003, 0.001, 0.0, 0.0)
    hal.send_action(
        Action(
            control_mode=ControlMode.CARTESIAN_DELTA,
            horizon=1,
            cartesian_delta=[delta],
            ee_name="panda_hand",
            frame_id="panda_link0",
        )
    )
    assert hal._last_cartesian_delta == delta
    # qpos unchanged — Jacobian-free digital twin.
    state = hal.read_state()
    assert all(v == 0.0 for v in state.position)


def test_cartesian_delta_rejects_wrong_width() -> None:
    """CARTESIAN_DELTA must carry a 6-vec (xyz + axis-angle).

    Pydantic enforces 6-tuple shape at construction, so the HAL's
    runtime guard is defence in depth — exercise it by hand-crafting
    a 5-wide payload that bypasses construction validation.
    """
    hal = PandaMobileHAL()
    hal.connect()
    # Build a malformed Action by setting cartesian_delta directly
    # after construction (Pydantic Action enforces the 6-tuple on
    # construction; this exercises the HAL's defence-in-depth check).
    a = Action(
        control_mode=ControlMode.CARTESIAN_DELTA,
        horizon=1,
        cartesian_delta=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
    )
    a.cartesian_delta = [(0.0, 0.0, 0.0, 0.0, 0.0)]  # type: ignore[list-item]  # reason: deliberate width breach
    with pytest.raises(ROSConfigError, match="6 floats per row"):
        hal.send_action(a)


def test_gripper_position_writes_qpos() -> None:
    """ADR-0028c — GRIPPER_POSITION sets the trailing qpos slot directly."""
    hal = PandaMobileHAL()
    hal.connect()
    hal.send_action(
        Action(
            control_mode=ControlMode.GRIPPER_POSITION,
            horizon=1,
            gripper=[0.6],
            ee_name="panda_gripper",
        )
    )
    state = hal.read_state()
    # Gripper is the last slot (index 10).
    assert state.position[-1] == 0.6
    # Other slots untouched.
    assert all(v == 0.0 for v in state.position[:-1])


def test_joint_position_11_wide_sets_all_including_gripper() -> None:
    """ADR-0028a + 0028c — full chain replay (base + arm + gripper) works."""
    hal = PandaMobileHAL()
    hal.connect()
    target = [1.0, 2.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 11 wide
    hal.send_action(
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[target],
        )
    )
    state = hal.read_state()
    assert state.position == target


def test_unsupported_control_mode_rejected() -> None:
    hal = PandaMobileHAL()
    hal.connect()
    with pytest.raises(ROSConfigError, match="unsupported control_mode"):
        hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_VELOCITY,
                horizon=1,
                joint_targets=[[0.0] * 7],
            )
        )


def test_estop_latch_blocks_actions_until_reset() -> None:
    hal = PandaMobileHAL()
    hal.connect()
    hal.estop()
    hal.send_action(
        Action(
            control_mode=ControlMode.BODY_TWIST,
            horizon=1,
            body_twist=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        )
    )
    # Latched: base position unchanged.
    assert hal.base_pose == (0.0, 0.0, 0.0)
    assert hal.estop_latched is True
    hal.reset_estop()
    hal.send_action(
        Action(
            control_mode=ControlMode.BODY_TWIST,
            horizon=1,
            body_twist=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        )
    )
    assert hal.base_pose[0] > 0.0


def test_panda_mobile_robot_yaml_declares_body_twist() -> None:
    """ADR-0025 — robot.yaml must advertise body_twist + joint_position
    + has_lidar so the rSkill compat checks for Nav2 / SLAM accept it.
    """
    import pathlib

    from openral_core import RobotDescription

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    description = RobotDescription.from_yaml(
        str(repo_root / "robots" / "panda_mobile" / "robot.yaml")
    )
    modes = {m.value for m in description.capabilities.supported_control_modes}
    assert "body_twist" in modes, f"got modes={modes}"
    assert "joint_position" in modes
    assert description.capabilities.has_lidar is True


def test_base_sim_joint_names_returns_canonical_mjcf_triple() -> None:
    """ADR-0025 — `extract_base_sim_joint_names` returns the OmronMobileBase triple.

    Robocasa's `PandaMobile` composition exposes the base joints as
    `mobilebase0_joint_mobile_{forward,side,yaw}`; the description's
    base `JointSpec.sim_joint_name` values are pinned to those via
    `robots/panda_mobile/robot.yaml`. The generic helper reads them
    so the sim-adapter side never hardcodes the robosuite namespace.
    """
    from openral_core import extract_base_sim_joint_names
    from openral_hal.panda_mobile import PANDA_MOBILE_DESCRIPTION

    triple = extract_base_sim_joint_names(PANDA_MOBILE_DESCRIPTION)
    assert triple == (
        "mobilebase0_joint_mobile_forward",
        "mobilebase0_joint_mobile_side",
        "mobilebase0_joint_mobile_yaw",
    )


def test_base_sim_joint_names_returns_none_when_missing() -> None:
    """If any of the first 3 joints lacks `sim_joint_name`, the helper returns None.

    Callers (lifecycle node, sim adapter) treat `None` as "fall back
    to the module defaults" rather than crashing.
    """
    from openral_core import extract_base_sim_joint_names
    from openral_core.schemas import (
        JointSpec,
        JointType,
        RobotCapabilities,
        RobotDescription,
        SafetyEnvelope,
    )

    desc = RobotDescription(
        name="panda_mobile_stub",
        embodiment_kind="mobile_manipulator",
        joints=[
            JointSpec(
                name="base_x",
                joint_type=JointType.PRISMATIC,
                parent_link="world",
                child_link="base_x_link",
            ),
            JointSpec(
                name="base_y",
                joint_type=JointType.PRISMATIC,
                parent_link="base_x_link",
                child_link="base_y_link",
            ),
            JointSpec(
                name="base_yaw",
                joint_type=JointType.REVOLUTE,
                parent_link="base_y_link",
                child_link="base_link",
            ),
        ],
        capabilities=RobotCapabilities(embodiment_tags=["panda_mobile"]),
        safety=SafetyEnvelope(),
    )
    assert extract_base_sim_joint_names(desc) is None


def test_panda_mobile_robot_yaml_carries_sim_joint_names() -> None:
    """Every joint in the canonical description has a populated `sim_joint_name`.

    ADR-0025 contract — the on-disk `robots/panda_mobile/robot.yaml`
    declares the MJCF override for every joint, so consumers can
    rely on it being present for the full chain.

    ADR-0028a — the chain is now 11 joints (3 base + 7 arm + 1
    parallel-gripper width DoF). The trailing gripper carries the
    canonical robosuite ``gripper0_right_finger_joint1`` override so a sim
    backend can resolve it via the same ``mj_name2id`` path the base
    + arm use.
    """
    from openral_hal.panda_mobile import PANDA_MOBILE_DESCRIPTION

    for spec in PANDA_MOBILE_DESCRIPTION.joints:
        assert spec.sim_joint_name is not None, (
            f"joint {spec.name!r} has no sim_joint_name in robots/panda_mobile/robot.yaml; "
            "ADR-0025 requires every panda_mobile joint to encode its MJCF override."
        )
    base_names = [j.sim_joint_name for j in PANDA_MOBILE_DESCRIPTION.joints[:3]]
    assert base_names == [
        "mobilebase0_joint_mobile_forward",
        "mobilebase0_joint_mobile_side",
        "mobilebase0_joint_mobile_yaw",
    ]
    arm_names = [j.sim_joint_name for j in PANDA_MOBILE_DESCRIPTION.joints[3:10]]
    assert arm_names == [f"robot0_joint{i}" for i in range(1, 8)]
    gripper = PANDA_MOBILE_DESCRIPTION.joints[10]
    assert gripper.name == "panda_gripper"
    assert gripper.role == "gripper"
    assert gripper.sim_joint_name == "gripper0_right_finger_joint1"
