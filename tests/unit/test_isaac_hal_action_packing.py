"""HAL arm commands reach the Isaac manifest scene by name, as absolute targets.

The regression: ``SimAttachedHAL``'s default packer assumes the robosuite slot
order (base first, one gripper in the LAST slot) and per-step deltas, while the
Isaac scene's layout is ``[non-base joints in manifest order, base twist]``. On
a live Isaac 6.1 run an arm JOINT_POSITION drove panda_mobile's base 1 m, a
gripper "open" spun it 1 rad, and every OpenArm slot tick was refused (a
zero-padded 16-wide row vs the packer's 14). The env now packs its own actions
(``pack_action`` / ``idle_action`` hooks, ``pack_isaac_action``).

Two halves, both GPU-free and on the real manifests:

* ``pack_isaac_action`` against ``IsaacActionLayout.from_description``;
* the full deploy path — real warehouse YAML → real loader + factory →
  ``SimAttachedHAL`` → a real sidecar subprocess (the real wire + serve loop,
  Kit stubbed at the process boundary) that records the vector Isaac would get.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from openral_core import Action, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import ControlMode
from openral_sim.backends.isaac_sim import IsaacActionLayout, pack_isaac_action

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAKE_SIDECAR = _REPO_ROOT / "tests" / "unit" / "fakes" / "isaac_sidecar_no_kit.py"


def _desc(robot: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / robot / "robot.yaml"))


def _nan_eq(got: np.ndarray, want: list[float]) -> None:
    np.testing.assert_array_equal(np.isnan(got), np.isnan(np.asarray(want)))
    np.testing.assert_allclose(np.nan_to_num(got), np.nan_to_num(np.asarray(want)), atol=1e-6)


NAN = math.nan

# ── pack_isaac_action on the real manifests ───────────────────────────────────


def test_panda_mobile_layout_is_manifest_order_then_twist() -> None:
    layout = IsaacActionLayout.from_description(_desc("panda_mobile"))
    assert layout.slots == (*(f"panda_joint{i}" for i in range(1, 8)), "panda_gripper")
    assert layout.dim == 11


def test_arm_command_fills_arm_slots_and_never_the_base() -> None:
    layout = IsaacActionLayout.from_description(_desc("panda_mobile"))
    arm = [0.0, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0]
    out = pack_isaac_action(
        Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[arm]), layout, None
    )
    # Absolute targets in the arm slots, gripper untouched (HOLD), base still.
    _nan_eq(out, [*arm, NAN, 0.0, 0.0, 0.0])


def test_gripper_command_fills_the_gripper_slot_and_holds_the_arm() -> None:
    layout = IsaacActionLayout.from_description(_desc("franka_panda"))
    prev = np.array([0.1, 0.2, 0.3, -1.0, 0.0, 1.0, 0.0, NAN], dtype=np.float32)
    # franka's end-effector is panda_hand; its gripper joint is panda_gripper.
    out = pack_isaac_action(
        Action(control_mode=ControlMode.GRIPPER_POSITION, gripper=[1.0], ee_name="panda_hand"),
        layout,
        prev,
    )
    _nan_eq(out, [0.1, 0.2, 0.3, -1.0, 0.0, 1.0, 0.0, 1.0])


def test_twist_holds_every_joint_and_a_joint_command_stops_the_base() -> None:
    layout = IsaacActionLayout.from_description(_desc("panda_mobile"))
    twist = pack_isaac_action(
        Action(control_mode=ControlMode.BODY_TWIST, body_twist=[(0.5, 0.0, 0.0, 0.0, 0.0, 0.2)]),
        layout,
        None,
    )
    _nan_eq(twist, [NAN] * 8 + [0.5, 0.0, 0.2])
    arm = pack_isaac_action(
        Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[[0.0] * 7]), layout, twist
    )
    assert arm[-3:].tolist() == [0.0, 0.0, 0.0]


def test_openarm_slot_rows_are_placed_by_joint_names() -> None:
    desc = _desc("openarm")
    layout = IsaacActionLayout.from_description(desc)
    names = [j.name for j in desc.joints]
    right = [f"right_joint{i}" for i in range(1, 8)]
    row = [0.0] * len(names)
    for name in right:
        row[names.index(name)] = -0.3
    out = pack_isaac_action(
        Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[row], joint_names=right),
        layout,
        None,
    )
    want = [NAN] * 16
    for name in right:
        want[layout.slots.index(name)] = -0.3
    _nan_eq(out, want)  # the zero-padded left arm is NOT commanded to 0 rad
    grip = pack_isaac_action(
        Action(control_mode=ControlMode.GRIPPER_POSITION, gripper=[-0.7], ee_name="right_gripper"),
        layout,
        out,
    )
    assert grip[layout.slots.index("right_gripper")] == pytest.approx(-0.7)
    assert np.isnan(grip[layout.slots.index("left_gripper")])


def test_ambiguous_gripper_and_unsupported_modes_are_refused() -> None:
    layout = IsaacActionLayout.from_description(_desc("openarm"))
    with pytest.raises(ROSConfigError, match="grippers are"):
        pack_isaac_action(
            Action(control_mode=ControlMode.GRIPPER_POSITION, gripper=[0.5]), layout, None
        )
    with pytest.raises(ROSConfigError, match="joint_velocity"):
        pack_isaac_action(
            Action(control_mode=ControlMode.JOINT_VELOCITY, joint_velocities=[[0.0] * 16]),
            layout,
            None,
        )


# ── the deploy path: SimAttachedHAL → real factory → recording sidecar ───────


@pytest.fixture
def recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pytest.importorskip("zmq", reason="isaacsim group (pyzmq) not installed")
    pytest.importorskip("msgpack", reason="isaacsim group (msgpack) not installed")
    out = tmp_path / "actions.jsonl"
    monkeypatch.setenv("OPENRAL_ISAAC_SIDECAR_PYTHON", sys.executable)
    monkeypatch.setenv("OPENRAL_ISAAC_SIDECAR_SCRIPT", str(_FAKE_SIDECAR))
    monkeypatch.setenv("OPENRAL_TEST_ISAAC_ARGV_OUT", str(tmp_path / "argv.json"))
    monkeypatch.setenv("OPENRAL_TEST_ISAAC_ACTIONS_OUT", str(out))
    # openarm.urdf's package://openarm_description meshes: a sourced workspace
    # wins over the public fetch, so an empty package dir keeps this offline
    # (the stubbed sidecar never reads a mesh).
    (tmp_path / "share" / "openarm_description").mkdir(parents=True)
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(tmp_path))
    return out


def _hal(robot: str, scene: str):
    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(str(_REPO_ROOT / "scenes" / "deploy" / scene))
    hal = SimAttachedHAL(env, _desc(robot), env_reset_seed=seed)
    hal.connect()
    return hal, env


def _rows(path: Path) -> list[list[float]]:
    if not path.exists():  # nothing stepped yet
        return []
    return [
        [NAN if v is None else v for v in json.loads(line)]
        for line in path.read_text().splitlines()
    ]


def test_hal_drives_panda_mobile_arm_gripper_and_base_into_their_own_slots(
    recorded: Path,
) -> None:
    hal, env = _hal("panda_mobile", "isaac_panda_mobile_warehouse.yaml")
    try:
        arm = [0.0, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0]
        hal.send_action(Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[arm]))
        hal.send_action(Action(control_mode=ControlMode.GRIPPER_POSITION, gripper=[1.0]))
        hal.send_action(
            Action(
                control_mode=ControlMode.BODY_TWIST,
                body_twist=[(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)],
            )
        )
        hal.idle_step()
    finally:
        hal.disconnect()
        env.close()  # stops the sidecar — a leaked one is adopted by the next test
    joint, grip, twist, idle = _rows(recorded)[-4:]
    _nan_eq(np.asarray(joint), [*arm, NAN, 0.0, 0.0, 0.0])  # base untouched
    _nan_eq(np.asarray(grip), [*arm, 1.0, 0.0, 0.0, 0.0])  # gripper slot, not wz
    _nan_eq(np.asarray(twist), [*arm, 1.0, 0.5, 0.0, 0.0])  # the arm holds
    _nan_eq(np.asarray(idle), [NAN] * 8 + [0.0, 0.0, 0.0])  # idle HOLDs, never 0 rad


def test_hal_commits_an_openarm_slot_tick_as_one_step(recorded: Path) -> None:
    desc = _desc("openarm")
    names = [j.name for j in desc.joints]
    left = [f"left_joint{i}" for i in range(1, 8)]
    right = [f"right_joint{i}" for i in range(1, 8)]

    def padded(value: float, sel: list[str]) -> list[list[float]]:
        row = [0.0] * len(names)
        for name in sel:
            row[names.index(name)] = value
        return [row]

    def slot(**kw: object) -> Action:
        return Action(tick_index=1, tick_group_size=4, **kw)  # type: ignore[arg-type]

    hal, env = _hal("openarm", "isaac_openarm_warehouse.yaml")
    try:
        before = len(_rows(recorded))
        for action in (
            slot(
                control_mode=ControlMode.JOINT_POSITION,
                joint_targets=padded(0.3, left),
                joint_names=left,
            ),
            slot(control_mode=ControlMode.GRIPPER_POSITION, gripper=[0.7], ee_name="left_gripper"),
            slot(
                control_mode=ControlMode.JOINT_POSITION,
                joint_targets=padded(-0.3, right),
                joint_names=right,
            ),
            slot(
                control_mode=ControlMode.GRIPPER_POSITION, gripper=[-0.7], ee_name="right_gripper"
            ),
        ):
            hal.send_action(action)
    finally:
        hal.disconnect()
        env.close()
    rows = _rows(recorded)
    assert len(rows) == before + 1, "a slot tick must commit as exactly one env step"
    # Manifest order: left arm, left gripper, right arm, right gripper.
    _nan_eq(np.asarray(rows[-1]), [0.3] * 7 + [0.7] + [-0.3] * 7 + [-0.7])


def test_a_mobile_manipulation_tick_keeps_its_twist(recorded: Path) -> None:
    """Twist + arm + gripper slots of ONE tick commit as one step with the twist
    intact (a later slot must not zero what an earlier one packed)."""
    hal, env = _hal("panda_mobile", "isaac_panda_mobile_warehouse.yaml")
    arm = [0.0, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0]
    try:
        for action in (
            Action(
                control_mode=ControlMode.BODY_TWIST,
                body_twist=[(0.3, 0.0, 0.0, 0.0, 0.0, 0.1)],
                tick_index=1,
                tick_group_size=3,
            ),
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                joint_targets=[arm],
                tick_index=1,
                tick_group_size=3,
            ),
            Action(
                control_mode=ControlMode.GRIPPER_POSITION,
                gripper=[1.0],
                tick_index=1,
                tick_group_size=3,
            ),
        ):
            hal.send_action(action)
        assert hal.base_twist[0] == pytest.approx(0.3)  # latched for /odom
    finally:
        hal.disconnect()
        env.close()
    _nan_eq(np.asarray(_rows(recorded)[-1]), [*arm, 1.0, 0.3, 0.0, 0.1])


def test_reconnect_drops_pre_reset_targets(recorded: Path) -> None:
    """After connect() re-resets the env, a base twist must not re-send the arm
    targets commanded before the reset."""
    hal, env = _hal("panda_mobile", "isaac_panda_mobile_warehouse.yaml")
    try:
        hal.send_action(Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[[0.5] * 7]))
        hal.connect()
        hal.send_action(
            Action(
                control_mode=ControlMode.BODY_TWIST,
                body_twist=[(0.2, 0.0, 0.0, 0.0, 0.0, 0.0)],
            )
        )
    finally:
        hal.disconnect()
        env.close()
    _nan_eq(np.asarray(_rows(recorded)[-1]), [NAN] * 8 + [0.2, 0.0, 0.0])


def test_fixed_joints_take_no_slot_but_pad_named_rows() -> None:
    layout = IsaacActionLayout(joints=("mount", "j1", "grip"), roles=("fixed", "arm", "gripper"))
    assert layout.slots == ("j1", "grip")
    out = pack_isaac_action(
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[[0.0, 0.7, 0.0]],  # padded to ALL joints, fixed included
            joint_names=["j1"],
        ),
        layout,
        None,
    )
    _nan_eq(out, [0.7, NAN])
