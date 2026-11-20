"""The sim HAL shell's optional env hooks: ``read_joint_state``, ``read_images``, ``idle_step``.

Issue #683 made the bare MuJoCo twin a ``SimAttachedHAL`` (the one sim
HAL shell) whose env (``MujocoTwinRollout``) answers getattr-gated hooks. These tests
wrap a real ``MujocoTwinRollout`` in a PLAIN ``SimAttachedHAL`` (not ``MujocoArmHAL``)
to show the shell itself uses each hook, then show the hooks are opt-in: the
native-MuJoCo ``so101_box`` rollout (``openral_sim``, a real backend that defines
none of them) still takes the shell's other paths. The so101_box deploy-scene test
builds its composed bare twin the way the HAL node does and checks the five gaps
#682 measured plus the scene-attached surface it now has.

Real manifests, real MuJoCo, no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _renderer_probe import requires_renderer
from openral_core import (
    Action,
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    ControlMode,
    DeployScene,
    Pose6D,
    RobotDescription,
)
from openral_core.code_refs import import_code_ref
from openral_core.exceptions import ROSConfigError
from openral_hal import build_hal
from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal.mujoco_twin import MujocoTwinRollout
from openral_hal.sim_attached import SimAttachedHAL

pytest.importorskip("mujoco")

_HOOKS = ("read_joint_state", "read_images", "idle_step")
PlainShell = Callable[[str], tuple[SimAttachedHAL, MujocoTwinRollout]]


@pytest.fixture
def plain_shell(
    robot_description: Callable[[str], RobotDescription], connect_twin: Callable[..., Any]
) -> Callable[[str], tuple[SimAttachedHAL, MujocoTwinRollout]]:
    """Connect a plain ``SimAttachedHAL`` around a robot's twin rollout (``connect_twin``)."""

    def _build(robot: str) -> tuple[SimAttachedHAL, MujocoTwinRollout]:
        desc = robot_description(robot)
        twin = MujocoArmHAL.from_description(desc)._twin
        hal = connect_twin(lambda: SimAttachedHAL(twin, desc, env_action_dim=twin.action_dim))
        assert type(hal) is SimAttachedHAL
        return hal, twin

    return _build


def test_plain_shell_serves_the_twins_joint_state_in_public_gripper_units(
    plain_shell: PlainShell,
) -> None:
    hal, twin = plain_shell("so101_follower")
    assert twin.data is not None
    assert hal._sim_dt_per_tick_s == twin.sim_dt_per_tick_s  # time base adopted at connect
    idx = twin.joint_names.index("gripper")
    row = [0.2, -0.3, 0.4, 0.1, 0.0, 1.0]
    for _ in range(60):
        twin.step(row)

    state = hal.read_state()
    assert state.name == twin.joint_names
    assert state.position == pytest.approx(twin.read_joint_state().position)
    raw = float(twin.data.qpos[twin.grippers_by_joint["gripper"].qpos_addrs[0]])
    # The shell's MJCF-by-name walk would publish ``raw`` (~1.74 rad).
    assert raw > 1.0
    assert 0.0 <= state.position[idx] <= 1.0
    assert state.position[idx] == pytest.approx(1.0, abs=0.02)
    assert not hasattr(hal, "reset_to_pose")  # a twin operation, not the shell's
    assert callable(twin.reset_to_pose)
    hal.disconnect()


@requires_renderer
def test_plain_shell_renders_lazily_through_the_twin(plain_shell: PlainShell) -> None:
    hal, twin = plain_shell("so101_follower")
    frames = hal.read_images()
    if not frames:
        pytest.skip("off-screen MuJoCo render produced no frames on this host")
    assert set(frames) == {"top", "wrist"}
    # Nothing was cached from a step: the frames came from the hook, on demand.
    assert "images" not in (hal._last_obs or {})
    assert hal.read_image_times() == {}
    before = frames["wrist"].copy()
    for _ in range(20):
        twin.step([0.6, -0.5, 0.6, 0.4, 0.8, 1.0])
    assert not np.array_equal(hal.read_images()["wrist"], before)  # the current state
    hal.disconnect()
    twin.close()


def test_plain_shell_idle_is_a_wall_slice_with_one_step_boundary(plain_shell: PlainShell) -> None:
    hal, twin = plain_shell("ur5e")
    assert twin.data is not None and twin.model is not None
    fired: list[int] = []
    hal.add_post_step_observer(lambda: fired.append(1))
    dt = float(twin.model.opt.timestep)
    obs = hal._last_obs
    ctrl = twin.data.ctrl.copy()

    for n in range(1, 4):
        t0 = hal.sim_time_ns()
        assert t0 is not None
        assert hal.idle_step(0.1) is True
        assert len(fired) == n
        # A wall slice (0.1 s), not one control period (1/30 s) of an idle_action step.
        assert hal.sim_time_ns() == pytest.approx(t0 + round(0.1 / dt) * dt * 1e9, abs=10)
    np.testing.assert_array_equal(twin.data.ctrl, ctrl)
    assert hal._last_obs is obs  # no env.step: nothing re-cached

    # A command: one step boundary, one control period (the twin's step_action_group).
    tick = hal._sim_dt_per_tick_s
    assert tick is not None
    t0 = hal.sim_time_ns()
    assert t0 is not None
    row = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]
    hal.send_action(Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[row]))
    assert len(fired) == 4
    assert hal.sim_time_ns() == pytest.approx(t0 + tick * 1e9, abs=10)
    assert [float(twin.data.ctrl[twin.actuator_index[j]]) for j in twin.joint_names] == (
        pytest.approx(row)
    )
    hal.disconnect()
    twin.close()


@requires_renderer
def test_hooks_are_opt_in_a_backend_without_them_keeps_the_previous_paths(
    _build_so101_hal: Callable[[], object],
) -> None:
    """``so101_box`` (native MuJoCo, ``openral_sim``) defines none of the three hooks."""
    pytest.importorskip("openral_sim")
    import mujoco

    hal = _build_so101_hal()
    assert type(hal) is SimAttachedHAL
    env = hal.env
    for name in _HOOKS:
        assert getattr(env, name, None) is None, name

    # Images: the frames cached from the last env step, served as-is.
    cached = (hal._last_obs or {}).get("images")
    assert isinstance(cached, dict) and cached
    frames = hal.read_images()
    assert frames.keys() == cached.keys()
    assert all(frames[k] is cached[k] for k in frames)

    # State: the MJCF-by-name walk.
    handles = hal.mujoco_handles()
    assert handles is not None
    model, data = handles
    expected = [
        float(
            data.qpos[
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.sim_joint_name or j.name)
                ]
            ]
        )
        for j in hal.description.joints
    ]
    assert hal.read_state().position == pytest.approx(expected)

    # Idle: an env.step of the idle action, whose observation is re-cached.
    fired: list[int] = []
    hal.add_post_step_observer(lambda: fired.append(1))
    obs = hal._last_obs
    assert hal.idle_step(0.5) is True
    assert len(fired) == 1
    assert hal._last_obs is not obs
    hal.disconnect()


@requires_renderer
def test_so101_box_deploy_scene_builds_a_working_bare_twin(repo_root: Path) -> None:
    """``scenes/deploy/so101_box.yaml`` through the shell: the five gaps #682 measured.

    And the composed scene gets the scene-attached surface (#683): the attachment API
    resolves the arena's ``tube`` body, and ``reset_episode`` reloads the arena.

    Builds the HAL the way ``_ManifestHALLifecycleNode._create_hal`` does for a
    bare-twin scene (``_compose_scene_mjcf``: run the scene's composer, write the
    XML beside ``meshdir``, thread it as ``transport["mjcf_path"]`` into
    ``build_hal``). The node itself needs ``rclpy``; this is its body, step for step.
    """
    pytest.importorskip("openral_sim")  # the composer lives in openral_sim
    scene = DeployScene.from_yaml(str(repo_root / "scenes/deploy/so101_box.yaml"))
    assert scene.composition is not None and scene.robot_id is not None
    desc = RobotDescription.from_yaml(str(repo_root / "robots" / scene.robot_id / "robot.yaml"))
    composer = import_code_ref(scene.composition.composer, setting="composition.composer")
    xml, meshdir = composer(**scene.composition.params)  # type: ignore[operator]  # reason: import_code_ref returns object; the composer contract is (xml, meshdir)
    scene_path = meshdir.parent / f"{desc.name}_composed_scene.xml"
    scene_path.write_text(xml)

    hal = build_hal(desc, mode="sim", transport={"mjcf_path": str(scene_path)})
    assert isinstance(hal, MujocoArmHAL)
    assert isinstance(hal, SimAttachedHAL)
    hal.connect()
    try:
        tube = AttachedCollisionObject(
            object_id="tube",
            attach_link="gripper",
            touch_links=["moving_jaw_so101_v1"],
            primitives=[
                AttachedCollisionPrimitive(
                    shape=BoxShape(half_extents_m=(0.01, 0.01, 0.04)),
                    pose_in_object=Pose6D(
                        xyz=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="tube"
                    ),
                )
            ],
            pose_in_link=Pose6D(
                xyz=(0.0, 0.0, 0.05), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="gripper"
            ),
            confidence=1.0,
            evidence_kind=AttachmentEvidenceKind.SIM_CONTACT,
            evidence_ref="mujoco_body:tube",
            stamp_ns=1,
        )
        hal.update_attached_objects([tube])
        assert [o.object_id for o in hal.read_attached_objects()] == ["tube"]
        assert hal.read_attached_body_ids()
        home = list(hal.read_state().position)
        names = [j.name for j in desc.joints]
        row = [0.2, -0.3, 0.4, 0.1, 0.0, 0.8]
        assert len(row) == len(names) == 6
        action = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[row])
        for _ in range(30):
            hal.send_action(action)  # the 6-wide arm + gripper row is accepted
        held = hal.read_state().position

        for _ in range(60):
            assert hal.idle_step(1 / 30) is True
        idle = hal.read_state().position
        arm = [i for i, n in enumerate(names) if n != "gripper"]
        assert [idle[i] for i in arm] == pytest.approx([held[i] for i in arm], abs=0.02)
        assert max(abs(idle[i]) for i in arm) > 0.2  # not relaxed back to 0

        g = idle[names.index("gripper")]
        assert 0.0 <= g <= 1.0
        assert g == pytest.approx(0.8, abs=0.05)

        frames = hal.read_images()
        if not frames:
            pytest.skip("off-screen MuJoCo render produced no frames on this host")
        top = next(s for s in desc.sensors if s.name == "top")
        assert top.intrinsics is not None
        assert frames["top"].shape == (top.intrinsics.height, top.intrinsics.width, 3)
        assert frames["top"].std() > 1.0

        assert hal.reset_episode() == 1
        assert hal.read_state().position == pytest.approx(home, abs=1e-6)
        # The tube is back at its spawn pose: no longer held, no longer masked from depth.
        assert hal.read_attached_objects() == []
        assert hal.read_attached_body_ids() == frozenset()
    finally:
        hal.disconnect()


def test_plain_shell_drives_the_twin_through_step_action_group(plain_shell: PlainShell) -> None:
    """The shell's send path drives a twin with no MujocoArmHAL around it (#683).

    The twin composes and validates its own commands (``step_action_group``): a 6-wide
    SO-101 row moves the arm, and a mode the arm does not take is refused before
    anything moves — never packed into the wrong ctrl slots (a gripper command zeroing
    the arm) or written into arm-joint qpos (a twist on a robot without a base).
    """
    hal, twin = plain_shell("so101_follower")
    assert twin.data is not None
    row = [0.2, -0.3, 0.4, 0.1, 0.0, 0.8]
    t0 = hal.sim_time_ns()
    hal.send_action(Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[row]))
    assert hal.sim_time_ns() != t0
    arm_ctrl = twin.data.ctrl.copy()
    qpos = twin.data.qpos.copy()
    with pytest.raises(ROSConfigError):
        hal.send_action(
            Action(control_mode=ControlMode.GRIPPER_POSITION, gripper=[0.5], ee_name="gripper")
        )
    with pytest.raises(ROSConfigError):
        hal.send_action(
            Action(control_mode=ControlMode.BODY_TWIST, body_twist=[(0.1, 0, 0, 0, 0, 0)])
        )
    np.testing.assert_array_equal(twin.data.ctrl, arm_ctrl)
    np.testing.assert_array_equal(twin.data.qpos, qpos)
    assert hal.last_applied_action is not None
    assert hal.last_applied_action.joint_targets == [row]
    hal.disconnect()


def test_a_twist_on_a_scene_without_a_planar_base_is_refused(
    _build_so101_hal: Callable[[], object],
) -> None:
    """The qpos base path needs ``base_joints [x, y, yaw]``; it never moves arm joints.

    Before #683's review it fell back to the first three manifest joints, so a twist on
    the SO-101 scene wrote velocity * dt into its shoulder and elbow qpos.
    """
    pytest.importorskip("openral_sim")
    hal = _build_so101_hal()
    assert isinstance(hal, SimAttachedHAL)
    handles = hal.mujoco_handles()
    assert handles is not None
    qpos = handles[1].qpos.copy()
    with pytest.raises(ROSConfigError, match="declares no planar base"):
        hal.send_action(
            Action(control_mode=ControlMode.BODY_TWIST, body_twist=[(0.1, 0, 0, 0, 0, 0)])
        )
    np.testing.assert_array_equal(handles[1].qpos, qpos)
    assert hal.last_action_ns == 0
    hal.disconnect()
