"""Unit tests for ``_dispatch_slots`` in ``rskill_runner_node``.

Exercises the pure byte-routing function that splits a flat policy action vector into
typed ``openral_core.Action`` objects per the manifest's
``openral_core.ActionContract.slots`` declaration.

The module is gated behind ``_ROS2_AVAILABLE`` (``rclpy`` import), so it's loaded
directly via ``importlib.util.spec_from_file_location`` rather than the package
``__init__``, to exercise helpers without booting ROS 2. Assumes
``ActionContract`` validation already enforced slot coverage + per-mode field
requirements; only byte-routing is covered here.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from openral_core import Action, ActionSlot, ControlMode
from openral_rskill._policy_io import PolicyIOCodec


def _load_skill_runner_module() -> ModuleType:
    """Load ``rskill_runner_node`` bypassing the ROS2-gated package init."""
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py"
    spec = importlib.util.spec_from_file_location("_test_rskill_runner_node", src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner_mod() -> ModuleType:
    return _load_skill_runner_module()


def _robocasa_12d_slots() -> list[ActionSlot]:
    """Canonical RoboCasa365 layout — five slots, three non-discard."""
    return [
        ActionSlot(
            range=(0, 5),
            control_mode=ControlMode.CARTESIAN_DELTA,
            ee="panda_hand",
            frame="panda_link0",
        ),
        ActionSlot(
            range=(6, 6),
            control_mode=ControlMode.GRIPPER_POSITION,
            ee="panda_gripper",
        ),
        ActionSlot(range=(7, 7), discard=True),
        ActionSlot(
            range=(8, 10),
            control_mode=ControlMode.BODY_TWIST,
            frame="base_link",
        ),
        ActionSlot(range=(11, 11), discard=True),
    ]


# ─── RoboCasa 12-D layout ────────────────────────────────────────────────────


def test_robocasa_12d_emits_three_typed_actions(runner_mod: ModuleType) -> None:
    """The 5-slot RoboCasa layout produces 3 non-discard Actions."""
    vec = np.array(
        [0.014, 0.0, -0.003, 0.001, 0.0, 0.0, -0.989, 0.001, 0.0, 0.0, 0.0, -0.991],
        dtype=np.float32,
    )
    actions = runner_mod._dispatch_slots(_robocasa_12d_slots(), vec)
    assert len(actions) == 3
    cart, grip, twist = actions

    assert cart.control_mode is ControlMode.CARTESIAN_DELTA
    assert cart.ee_name == "panda_hand"
    assert cart.frame_id == "panda_link0"
    assert cart.cartesian_delta is not None
    assert tuple(round(v, 4) for v in cart.cartesian_delta[0]) == (
        0.014,
        0.0,
        -0.003,
        0.001,
        0.0,
        0.0,
    )

    assert grip.control_mode is ControlMode.GRIPPER_POSITION
    assert grip.ee_name == "panda_gripper"
    assert grip.frame_id is None
    assert grip.gripper is not None
    assert math.isclose(grip.gripper[0], -0.989, abs_tol=1e-5)

    assert twist.control_mode is ControlMode.BODY_TWIST
    assert twist.frame_id == "base_link"
    assert twist.ee_name is None
    # 3-D planar slice (forward, side, yaw) padded to 6-D twist.
    assert twist.body_twist == [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]


def test_cartesian_scale_is_copied_without_changing_policy_bytes(
    runner_mod: ModuleType,
) -> None:
    raw = np.array(
        [-0.176, 0.095, -0.309, 0.0, 0.013, 0.104, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        dtype=np.float32,
    )
    scale = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
    cart = runner_mod._dispatch_slots(
        _robocasa_12d_slots(),
        raw,
        cartesian_delta_scale=scale,
    )[0]
    assert cart.cartesian_delta is not None
    assert cart.cartesian_delta[0] == pytest.approx(tuple(raw[:6]))
    assert cart.cartesian_delta_scale == scale
    grouped = runner_mod._dispatch_slots(_robocasa_12d_slots(), raw, cartesian_delta_scale=scale)
    assert {action.tick_group_size for action in grouped} == {3}


def test_discard_slots_produce_no_action(runner_mod: ModuleType) -> None:
    """Slots marked discard=True emit no Action — bytes are dropped."""
    slots = [
        ActionSlot(range=(0, 0), discard=True),
        ActionSlot(range=(1, 1), control_mode=ControlMode.GRIPPER_POSITION, ee="g"),
        ActionSlot(range=(2, 2), discard=True),
    ]
    actions = runner_mod._dispatch_slots(slots, np.array([99.0, 0.5, 99.0], dtype=np.float32))
    assert len(actions) == 1
    assert actions[0].control_mode is ControlMode.GRIPPER_POSITION
    assert actions[0].gripper == [0.5]


# ─── Per-mode dispatch coverage ──────────────────────────────────────────────


def test_joint_position_routes_to_joint_targets(runner_mod: ModuleType) -> None:
    slots = [
        ActionSlot(
            range=(0, 6),
            control_mode=ControlMode.JOINT_POSITION,
        )
    ]
    vec = np.arange(7, dtype=np.float32)
    actions = runner_mod._dispatch_slots(slots, vec)
    assert len(actions) == 1
    a = actions[0]
    assert a.control_mode is ControlMode.JOINT_POSITION
    assert a.joint_targets == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    assert a.cartesian_delta is None
    assert a.body_twist is None
    assert a.gripper is None


def test_joint_velocity_routes_to_joint_velocities(runner_mod: ModuleType) -> None:
    slots = [ActionSlot(range=(0, 1), control_mode=ControlMode.JOINT_VELOCITY)]
    actions = runner_mod._dispatch_slots(slots, np.array([1.0, 2.0], dtype=np.float32))
    assert actions[0].joint_velocities == [[1.0, 2.0]]
    assert actions[0].joint_targets is None


def test_declared_input_bounds_clip_before_joint_velocity_dispatch(
    runner_mod: ModuleType,
) -> None:
    slots = [
        ActionSlot(
            range=(0, 2),
            control_mode=ControlMode.JOINT_VELOCITY,
            input_bounds=(-1.0, 1.0),
        )
    ]
    clipped = runner_mod._clip_input_bounds(slots, np.array([1.015625, -1.25, 0.5], np.float32))
    actions = runner_mod._dispatch_slots(slots, clipped)
    assert actions[0].joint_velocities == [[1.0, -1.0, 0.5]]


def test_input_bounds_clip_in_policy_units_before_conversion(runner_mod: ModuleType) -> None:
    """Bounds are policy units: clip 150 to 100, then scale to 1.0 (not 1.5 unclipped)."""
    from openral_rskill._policy_io import PolicyIOCodec

    slots = [
        ActionSlot(
            range=(0, 0),
            control_mode=ControlMode.GRIPPER_POSITION,
            ee="g",
            input_bounds=(0.0, 100.0),
        )
    ]
    actions = runner_mod._policy_action_to_actions(
        np.array([150.0], dtype=np.float32),
        codec=PolicyIOCodec(gripper_scale=100.0),
        slots=slots,
        description=None,
        cartesian_delta_scale=None,
    )
    assert actions[0].gripper == [1.0]


def test_cartesian_twist_routes_correctly(runner_mod: ModuleType) -> None:
    slots = [
        ActionSlot(
            range=(0, 5),
            control_mode=ControlMode.CARTESIAN_TWIST,
            ee="panda_hand",
            frame="panda_link0",
        )
    ]
    vec = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03], dtype=np.float32)
    actions = runner_mod._dispatch_slots(slots, vec)
    a = actions[0]
    assert a.control_mode is ControlMode.CARTESIAN_TWIST
    assert tuple(round(v, 4) for v in a.cartesian_twist[0]) == (
        0.1,
        0.2,
        0.3,
        0.01,
        0.02,
        0.03,
    )


def test_body_twist_6d_passthrough(runner_mod: ModuleType) -> None:
    """6-D body twist passes through verbatim, no zero-padding."""
    slots = [ActionSlot(range=(0, 5), control_mode=ControlMode.BODY_TWIST, frame="base_link")]
    vec = np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.3], dtype=np.float32)
    actions = runner_mod._dispatch_slots(slots, vec)
    assert tuple(round(v, 4) for v in actions[0].body_twist[0]) == (
        0.5,
        0.1,
        0.0,
        0.0,
        0.0,
        0.3,
    )


def test_body_twist_invalid_width_rejected(runner_mod: ModuleType) -> None:
    """4-D / 5-D BODY_TWIST slots are rejected (only 3-D planar or 6-D full)."""
    slots = [ActionSlot(range=(0, 3), control_mode=ControlMode.BODY_TWIST, frame="base_link")]
    with pytest.raises(ValueError, match="must be 3-D"):
        runner_mod._dispatch_slots(slots, np.zeros(4, dtype=np.float32))


def test_gripper_binary_routes_to_gripper(runner_mod: ModuleType) -> None:
    slots = [ActionSlot(range=(0, 0), control_mode=ControlMode.GRIPPER_BINARY, ee="g")]
    actions = runner_mod._dispatch_slots(slots, np.array([1.0], dtype=np.float32))
    assert actions[0].control_mode is ControlMode.GRIPPER_BINARY
    assert actions[0].gripper == [1.0]


# ─── Action objects are well-formed ──────────────────────────────────────────


def test_returned_actions_are_action_instances(runner_mod: ModuleType) -> None:
    actions = runner_mod._dispatch_slots(
        _robocasa_12d_slots(),
        np.zeros(12, dtype=np.float32),
    )
    assert all(isinstance(a, Action) for a in actions)
    # Each Action carries horizon=1 — single-step chunks.
    assert all(a.horizon == 1 for a in actions)


# ─── The slot path applies the SAME policy<->robot codec as the joint path ───


def _so101_degrees_slot_manifest() -> tuple[object, object]:
    """Real SO-101 degrees checkpoint, redeclared with a two-slot contract.

    Validating it through ``RSkillManifest`` pins that ``joint_units: degrees``
    together with ``slots`` is a legal manifest — legal only because the runner
    now converts the slot path too.
    """
    from openral_core import RobotDescription, RSkillManifest

    description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    raw = RSkillManifest.from_yaml(
        "rskills/rskill-smolvla-so101-eraser_place-bf16/rskill.yaml"
    ).model_dump(mode="json")
    raw["action_contract"]["slots"] = [
        {
            "range": [0, 4],
            "control_mode": "joint_position",
            "joint_names": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
        },
        {"range": [5, 5], "control_mode": "gripper_position", "ee": "gripper"},
    ]
    manifest = RSkillManifest.model_validate(raw)
    assert manifest.action_contract is not None
    assert manifest.action_contract.joint_units is not None
    assert manifest.action_contract.joint_units.value == "degrees"
    return manifest, description


def test_degrees_slot_manifest_reaches_the_wire_in_radians(runner_mod: ModuleType) -> None:
    """A degrees checkpoint with slots: radians on the wire, gripper descaled.

    Before the codec the slot path handed the RAW policy vector to
    ``_dispatch_slots`` — 90 (degrees) went out as 90 rad and cleared nothing.
    """
    from openral_rskill._policy_io import PolicyIOCodec

    manifest, description = _so101_degrees_slot_manifest()
    codec = PolicyIOCodec.from_manifest(manifest, description)
    # In-range values: the slot path now also pre-clamps to the joint envelope
    # (wrist_roll tops out at 2.7925 rad), and this test pins the unit codec.
    policy_action = np.array([90.0, -45.0, 30.0, 0.0, 90.0, 50.0], dtype=np.float32)
    actions = runner_mod._policy_action_to_actions(
        policy_action,
        codec=codec,
        slots=manifest.action_contract.slots,
        description=description,
        cartesian_delta_scale=None,
    )
    joint, grip = actions
    assert joint.control_mode is ControlMode.JOINT_POSITION
    np.testing.assert_allclose(
        joint.joint_targets[0],
        [math.pi / 2, -math.pi / 4, math.pi / 6, 0.0, math.pi / 2, 0.0],
        rtol=1e-5,
        atol=1e-6,
    )
    assert grip.control_mode is ControlMode.GRIPPER_POSITION
    assert grip.gripper == pytest.approx([0.5])


def test_joint_path_converts_and_clamps(runner_mod: ModuleType) -> None:
    """Without slots the whole vector is converted, then pre-clamped to the envelope."""
    from openral_core import RobotDescription, RSkillManifest
    from openral_rskill._policy_io import PolicyIOCodec

    description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    manifest = RSkillManifest.from_yaml(
        "rskills/rskill-smolvla-so101-eraser_place-bf16/rskill.yaml"
    )
    codec = PolicyIOCodec.from_manifest(manifest, description)
    action = runner_mod._policy_action_to_actions(
        np.array([10.0, 500.0, 0.0, 0.0, 0.0, 50.0], dtype=np.float32),
        codec=codec,
        slots=None,
        description=description,
        cartesian_delta_scale=None,
    )
    assert action.control_mode is ControlMode.JOINT_POSITION
    targets = action.joint_targets[0]
    assert targets[0] == pytest.approx(math.radians(10.0), rel=1e-5)
    assert targets[1] < 1.7453  # 500 deg clamped inside the envelope
    assert targets[5] == pytest.approx(0.5)


def test_joint_position_slots_are_clamped_inside_the_robots_joint_limits(
    runner_mod: ModuleType,
) -> None:
    """A JOINT_POSITION slot target past a joint limit is pulled strictly inside it.

    Real fixture: the OpenArm v2 manifest and its 16-D bimanual slot contract.
    On qorin1 (2026-09-22) the restock π0.5's first tick proposed left_joint5 at
    -1.58973 rad against a -1.5708 limit; the kernel E-stopped. The
    single-surface path already clamps with a 1e-3 epsilon (the kernel checks
    open intervals); slot dispatch now does the same for named joints.
    """
    import yaml
    from openral_core import RobotDescription

    repo_root = Path(__file__).resolve().parents[2]
    desc = RobotDescription.model_validate(
        yaml.safe_load((repo_root / "robots" / "openarm" / "robot.yaml").read_text())
    )
    left = [f"left_joint{i}" for i in range(1, 8)]
    right = [f"right_joint{i}" for i in range(1, 8)]
    slots = [
        ActionSlot(range=(0, 6), control_mode=ControlMode.JOINT_POSITION, joint_names=left),
        ActionSlot(range=(7, 7), control_mode=ControlMode.GRIPPER_POSITION, ee="left_gripper"),
        ActionSlot(range=(8, 14), control_mode=ControlMode.JOINT_POSITION, joint_names=right),
        ActionSlot(range=(15, 15), control_mode=ControlMode.GRIPPER_POSITION, ee="right_gripper"),
    ]
    lo, hi = desc.joints[[j.name for j in desc.joints].index("left_joint5")].position_limits
    assert lo == pytest.approx(-1.5708, abs=1e-4)

    vec = np.zeros(16, dtype=np.float32)
    vec[4] = -1.58973  # left_joint5, 0.019 rad past its limit
    vec[12] = float(hi) + 0.5  # right_joint5, well past the other end
    actions = runner_mod._policy_action_to_actions(
        vec,
        codec=PolicyIOCodec.from_manifest(None, desc),
        slots=slots,
        description=desc,
        cartesian_delta_scale=None,
    )

    left_action = next(a for a in actions if a.joint_names == left)
    right_action = next(a for a in actions if a.joint_names == right)
    assert left_action.joint_targets[0][4] == pytest.approx(float(lo) + 1e-3)
    assert right_action.joint_targets[0][12] == pytest.approx(float(hi) - 1e-3)
    # An in-range target is untouched, and nothing is clamped onto the limit itself.
    assert left_action.joint_targets[0][0] == 0.0
    assert left_action.joint_targets[0][4] > float(lo)


def test_an_unnamed_joint_position_slot_is_clamped_in_description_order(
    runner_mod: ModuleType,
) -> None:
    """A JOINT_POSITION slot without ``joint_names`` is clamped too (audit A.md F4).

    Such a slot is a whole-vector action in ``RobotDescription.joints`` order
    (``_slot_joint_names``). The runner's old slot clamp keyed on names only,
    so it proposed these targets raw; the clamp now lives in the one codec
    (``PolicyIOCodec``) with the same epsilon as the whole-vector path.
    Real fixture: the Franka Panda manifest, an arm + gripper contract.
    """
    import yaml
    from openral_core import RobotDescription

    repo_root = Path(__file__).resolve().parents[2]
    desc = RobotDescription.model_validate(
        yaml.safe_load((repo_root / "robots" / "franka_panda" / "robot.yaml").read_text())
    )
    n = len(desc.joints)
    slots = [ActionSlot(range=(0, n - 1), control_mode=ControlMode.JOINT_POSITION)]
    limits = [j.position_limits for j in desc.joints]
    lo4, hi4 = limits[3]
    vec = np.zeros(n, dtype=np.float32)
    vec[3] = float(hi4) + 0.3  # panda_joint4 past its upper limit
    vec[0] = float(limits[0][0]) - 0.3  # panda_joint1 past its lower limit
    codec = PolicyIOCodec.from_manifest(None, desc)

    (action,) = runner_mod._policy_action_to_actions(
        vec, codec=codec, slots=slots, description=desc, cartesian_delta_scale=None
    )
    row = action.joint_targets[0]
    assert row[3] == pytest.approx(float(hi4) - 1e-3)
    assert row[0] == pytest.approx(float(limits[0][0]) + 1e-3)
    # Same result as the whole-vector path: one clamp, one epsilon.
    assert row == pytest.approx(list(codec.clamp(codec.to_robot_action(vec))))
    assert float(lo4) < row[3] < float(hi4)


def test_a_joint_position_slot_whose_width_differs_from_its_joint_names_is_refused() -> None:
    """A width/``joint_names`` mismatch never reaches the clamp (#289's slice fix).

    ``PolicyIOCodec._slots_to_robot_units`` indexes ``slot.joint_names[k]`` over
    the slot width, so a mismatch there would clamp against the wrong joint or
    truncate. ``ActionSlot``'s own validator refuses it at manifest load, naming
    ``joint_names``, so no codec-side guard is needed. Real joint names: the
    Franka Panda manifest.
    """
    import yaml
    from openral_core import RobotDescription
    from pydantic import ValidationError

    repo_root = Path(__file__).resolve().parents[2]
    desc = RobotDescription.model_validate(
        yaml.safe_load((repo_root / "robots" / "franka_panda" / "robot.yaml").read_text())
    )
    names = [j.name for j in desc.joints][:7]
    for width in (6, 8):  # narrower and wider than the seven names
        with pytest.raises(ValidationError, match=r"joint_names length \(7\) must equal"):
            ActionSlot(
                range=(0, width - 1), control_mode=ControlMode.JOINT_POSITION, joint_names=names
            )
    ActionSlot(range=(0, 6), control_mode=ControlMode.JOINT_POSITION, joint_names=names)
