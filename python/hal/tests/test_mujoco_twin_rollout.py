"""``MujocoTwinRollout`` and the ``MujocoArmHAL`` shell around it (issue #683).

``MujocoArmHAL`` is a ``SimAttachedHAL`` (the one sim HAL shell) whose env is a
``MujocoTwinRollout``, the one MuJoCo driver of the bare twins.
These tests pin the rollout's own contract (time base, reset, gripper units, idle
slice, lazy renders, teardown, no ``openral_sim``), the shell surface a twin
offers, and each robot's twin (PD, glide, walking, GR00T WBC, differential
base) driven through the shell.

Real manifests from ``robots/``, real MuJoCo, no mocks (CLAUDE.md §1.11). A host
without ``mujoco`` or with an unfetchable MJCF skips; a render test skips only
when no off-screen GL backend exists (``_renderer_probe``).
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _renderer_probe import requires_renderer
from _twin_assets import expect_refusal, skip_if_unavailable
from openral_core import Action, ControlMode, RobotDescription
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSRuntimeError,
    ROSSafetyViolation,
)
from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal.mujoco_twin import _IDLE_STEP_CAP, MujocoTwinRollout
from openral_hal.openamrobot_v2 import OpenAMRobotV2MujocoHAL
from openral_hal.sim_attached import SimAttachedHAL

pytest.importorskip("mujoco")

Describe = Callable[[str], RobotDescription]
ConnectTwin = Callable[..., Any]


@pytest.fixture
def rollout(robot_description: Describe) -> Iterator[Callable[[str], MujocoTwinRollout]]:
    """Load the robot's twin rollout, built by ``MujocoArmHAL`` itself; closed at teardown."""
    loaded: list[MujocoTwinRollout] = []

    def _load(robot: str) -> MujocoTwinRollout:
        try:
            r = MujocoArmHAL.from_description(robot_description(robot))._twin
            r.load()
        except ROSConfigError as exc:
            skip_if_unavailable(exc, f"{robot} MJCF")
        loaded.append(r)
        return r

    yield _load
    for r in loaded:
        r.close()


def _joint_action(row: list[float], **ticked: int) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[row],
        **ticked,
    )


def _twist(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, **ticked: int) -> Action:
    return Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[(vx, vy, 0.0, 0.0, 0.0, wz)],
        **ticked,
    )


def _g1(s0_controller: str | None = None) -> Callable[[], Any]:
    def build() -> Any:  # reason: a G1MujocoHAL
        from openral_hal.g1 import G1MujocoHAL

        return G1MujocoHAL(s0_controller=s0_controller)  # type: ignore[arg-type]  # reason: a G1S0Controller literal

    return build


def _h1(robot_description: Describe) -> Callable[[], Any]:
    def build() -> Any:  # reason: an H1MujocoHAL
        from openral_hal.h1 import H1MujocoHAL

        return H1MujocoHAL(gravity_enabled=False, description=robot_description("h1"))

    return build


_TWINS: dict[str, Callable[[], Any] | None] = {
    "g1_walk": _g1("playground_walk"),
    "g1_gr00t": _g1("gr00t_wbc"),
}


# ── MujocoTwinRollout, driven directly ────────────────────────────────────────


def test_reset_reloads_the_loaded_state_and_time_base(
    rollout: Callable[[str], MujocoTwinRollout],
) -> None:
    r = rollout("so101_follower")
    assert r.data is not None and r.model is not None
    qpos0, ctrl0 = r.data.qpos.copy(), r.data.ctrl.copy()
    steps0, tick0, timestep0 = r.settle_steps, r.sim_dt_per_tick_s, float(r.model.opt.timestep)
    row = [0.3, -0.4, 0.5, 0.2, 0.1, 1.0]
    for _ in range(10):
        r.step(row)
    assert float(r.data.time) > 0.0
    model = r.model
    assert r.reset() == {}
    assert r.model is model  # in place: the bridge's handles stay valid
    assert r.data is not None
    np.testing.assert_array_equal(r.data.qpos, qpos0)
    np.testing.assert_array_equal(r.data.ctrl, ctrl0)
    assert float(r.data.time) == 0.0
    assert (r.settle_steps, r.sim_dt_per_tick_s, float(r.model.opt.timestep)) == (
        steps0,
        tick0,
        timestep0,
    )


@pytest.mark.parametrize("robot", ["so101_follower", "ur5e"])
def test_one_step_is_one_control_period_and_moves_toward_the_target(
    robot: str, rollout: Callable[[str], MujocoTwinRollout]
) -> None:
    r = rollout(robot)
    assert r.data is not None and r.model is not None
    rate = r.description.control_rate_hz
    assert rate is not None and r.sim_dt_per_tick_s is not None
    assert r.sim_dt_per_tick_s == pytest.approx(1.0 / rate, rel=1e-9)
    assert r.settle_steps * float(r.model.opt.timestep) == pytest.approx(r.sim_dt_per_tick_s)

    row = list(r.read_joint_state().position)
    row[1] += 0.3
    target = row[1]
    gap0 = abs(target - float(r.data.qpos[r.joint_qpos_addr[r.joint_names[1]]]))
    t0 = float(r.data.time)
    r.step(row)
    assert float(r.data.time) - t0 == pytest.approx(r.sim_dt_per_tick_s, abs=1e-12)
    assert float(r.data.ctrl[r.actuator_index[r.joint_names[1]]]) == target
    gap1 = abs(target - r.read_joint_state().position[1])
    assert gap1 < gap0
    for _ in range(60):
        r.step(row)
    assert r.read_joint_state().position[1] == pytest.approx(target, abs=0.05)


def test_step_is_step_action_group_of_one_row(rollout: Callable[[str], MujocoTwinRollout]) -> None:
    """The plain ``step`` face has no command path of its own (#683): it delegates."""
    a, b = rollout("ur5e"), rollout("ur5e")
    row = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]
    result = a.step(row)
    b.step_action_group([_joint_action(row)])
    assert a.data is not None and b.data is not None
    np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    np.testing.assert_array_equal(a.data.ctrl, b.data.ctrl)
    applied = result.info["applied_action"]
    assert isinstance(applied, Action) and applied.joint_targets == [row]


def test_normalised_gripper_round_trips_in_public_units(
    rollout: Callable[[str], MujocoTwinRollout],
) -> None:
    """SO-101 (affine_low_high): command 1.0 reads back ~1.0, never raw radians (#683)."""
    r = rollout("so101_follower")
    assert r.data is not None
    gripper = r.grippers_by_joint["gripper"]
    low, high = gripper.ctrl_range
    idx = r.joint_names.index("gripper")
    act = r.actuator_index["gripper"]

    row = list(r.read_joint_state().position)
    row[idx] = 1.0
    for _ in range(60):
        r.step(row)
    state = r.read_joint_state()
    raw = float(r.data.qpos[gripper.qpos_addrs[0]])
    assert float(r.data.ctrl[act]) == pytest.approx(high)
    assert raw > 1.0  # the jaw really is ~1.74 rad open
    assert 0.0 <= state.position[idx] <= 1.0
    assert state.position[idx] == pytest.approx(1.0, abs=0.02)
    assert state.velocity[idx] == 0.0  # a normalised gripper reports no qvel

    row[idx] = 0.0
    r.step(row)
    assert float(r.data.ctrl[act]) == pytest.approx(low)
    row[idx] = 1.7  # out of the public range: clipped, not passed through as radians
    r.step(row)
    assert float(r.data.ctrl[act]) == pytest.approx(high)


def test_reset_to_pose_holds_the_pose_with_grippers_in_public_units(
    rollout: Callable[[str], MujocoTwinRollout],
) -> None:
    """The snap re-seeds ctrl through the one gripper writer and the one qpos seeder."""
    r = rollout("so101_follower")
    assert r.data is not None
    pose = [0.2, -0.3, 0.4, 0.1, 0.0, 1.0]
    r.reset_to_pose(pose)
    gripper = r.grippers_by_joint["gripper"]
    assert float(r.data.ctrl[r.actuator_index["gripper"]]) == pytest.approx(gripper.ctrl_range[1])
    for name, value in zip(r.joint_names[:5], pose[:5], strict=True):
        assert float(r.data.ctrl[r.actuator_index[name]]) == pytest.approx(value)


def test_idle_step_advances_a_capped_wall_slice_and_leaves_ctrl_alone(
    rollout: Callable[[str], MujocoTwinRollout],
) -> None:
    r = rollout("ur5e")
    assert r.data is not None and r.model is not None
    r.step([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    ctrl = r.data.ctrl.copy()
    dt = float(r.model.opt.timestep)

    for wall_dt_s, steps in ((0.1, round(0.1 / dt)), (None, 1), (0.0, 1), (60.0, _IDLE_STEP_CAP)):
        t0 = float(r.data.time)
        assert r.idle_step(wall_dt_s) is True
        assert float(r.data.time) - t0 == pytest.approx(steps * dt, abs=1e-9), wall_dt_s
        np.testing.assert_array_equal(r.data.ctrl, ctrl)
    r.close()
    assert r.idle_step(0.1) is False


@requires_renderer
def test_read_images_renders_each_camera_at_its_own_resolution(
    rollout: Callable[[str], MujocoTwinRollout],
) -> None:
    """OpenArm: a 672x376 head ZED and two 960x600 wrists, all from one twin."""
    r = rollout("openarm")
    frames = r.read_images()
    rgb = {s.name: s.intrinsics for s in r.description.sensors if s.modality == "rgb"}
    assert set(frames) == set(rgb)
    for name, frame in frames.items():
        intr = rgb[name]
        assert intr is not None
        assert frame.shape == (intr.height, intr.width, 3), name
        assert frame.dtype == np.uint8
    assert {(f.shape[0], f.shape[1]) for f in frames.values()} == {(376, 672), (600, 960)}
    r.close()
    assert r.read_images() == {}


@requires_renderer
def test_a_reload_renders_again_after_a_render_failure(
    rollout: Callable[[str], MujocoTwinRollout],
) -> None:
    """The render-failure latch belongs to one load: ``close`` clears it with the model."""
    r = rollout("openarm")
    r._render_failed = True  # what a renderer that failed to build leaves
    assert r.read_images() == {}
    r.close()
    r.load()
    assert set(r.read_images()) == {s.name for s in r.description.sensors if s.modality == "rgb"}


def test_load_leaves_no_camera_rig_file_beside_the_mjcf(
    rollout: Callable[[str], MujocoTwinRollout],
) -> None:
    """Each load rigs its cameras into a file of its own and removes it once compiled."""
    r = rollout("openarm")
    assert list(Path(r.mjcf_path).parent.glob(f"{r.description.name}_camrig_*.xml")) == []


def test_close_releases_the_model_and_is_idempotent(
    rollout: Callable[[str], MujocoTwinRollout],
) -> None:
    r = rollout("ur5e")
    assert r.mujoco_handles() is not None
    assert r.sim_time_ns() == 0
    r.close()
    assert r.model is None and r.data is None
    assert r.mujoco_handles() is None
    assert r.sim_time_ns() is None
    assert r.renderers == {}
    r.close()


def test_the_twin_runs_without_openral_sim(repo_root: Path) -> None:
    """A fresh interpreter whose ``openral_sim`` import is blocked drives a twin end to end.

    ``sys.modules["openral_sim"] = None`` makes every ``import openral_sim[.x]`` raise
    ``ImportError`` in that process; the probe also asserts nothing imported it. It
    covers the rollout and the ``MujocoArmHAL`` shell (connect, send, idle, state).
    """
    probe = textwrap.dedent(
        """
        import sys
        sys.modules["openral_sim"] = None
        from openral_core import Action, ControlMode, RobotDescription
        from openral_hal._mujoco_arm import MujocoArmHAL
        from openral_hal.mujoco_twin import MujocoTwinRollout  # noqa: F401
        desc = RobotDescription.from_yaml(sys.argv[1])
        hal = MujocoArmHAL.from_description(desc)
        hal.connect()
        row = [0.1, -0.2, 0.3, 0.0, 0.0, 0.5]
        hal.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[row])
        )
        assert hal.idle_step(0.05)
        assert len(hal.read_state().position) == 6
        hal.disconnect()
        assert sys.modules["openral_sim"] is None
        assert not [m for m in sys.modules if m.startswith("openral_sim.")]
        print("TWIN_OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe, str(repo_root / "robots/so101_follower/robot.yaml")],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")},
        check=False,
    )
    if "AssetFetchError" in proc.stderr:  # the pinned MJCF could not be fetched here
        pytest.skip(f"so101 MJCF unavailable: {proc.stderr.strip().splitlines()[-1]}")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "TWIN_OK" in proc.stdout


# ── The MujocoArmHAL shell ────────────────────────────────────────────────────


@pytest.mark.parametrize("robot", ["so101_follower", "ur5e"])
def test_twin_offers_the_scene_attached_surface(robot: str, connect_twin: ConnectTwin) -> None:
    """A twin is a full ``SimAttachedHAL``: one sim HAL surface for every deploy scene (#683).

    The lifecycle node and the sensor bridge switch the attachment staging, the
    ``reset_episode`` service and the ``/openral/reward/active_task`` subscription on
    by ``getattr``, so a composed (bare-twin) scene gets the same deploy graph as a
    scene-attached one.
    """
    hal = connect_twin(robot)
    assert isinstance(hal, SimAttachedHAL)
    for name in (
        "update_attached_objects",
        "read_attached_objects",
        "read_attached_body_ids",
        "reset_episode",
        "set_active_task",
    ):
        assert callable(getattr(hal, name, None)), name
    # The exact expression SimSensorBridge evaluates: nothing attached yet.
    assert getattr(hal, "read_attached_objects", list)() == []
    hal.set_active_task("pick the cube")  # a twin grades nothing: a logged no-op
    home = list(hal.read_state().position)
    handles = hal.mujoco_handles()
    assert handles is not None
    hal.send_action(_joint_action([0.3] * len(home)))
    assert hal.read_state().position != pytest.approx(home, abs=1e-3)
    assert hal.reset_episode() == 1
    assert hal.read_state().position == pytest.approx(home, abs=1e-6)
    after = hal.mujoco_handles()
    assert after is not None and after[0] is handles[0]  # the bridge's model stays live
    assert callable(getattr(hal, "reset_to_pose", None))
    assert not hasattr(SimAttachedHAL, "reset_to_pose")


def test_fixed_base_twin_reports_the_origin_not_its_first_three_joints(
    connect_twin: ConnectTwin,
) -> None:
    hal = connect_twin("ur5e")
    for _ in range(20):
        hal.send_action(_joint_action([0.4, -0.5, 0.6, 0.0, 0.0, 0.0]))
    # The shell reads no base off a manifest without base_joints (its first three
    # joints are arm joints, nowhere near the origin after these actions).
    assert hal.base_pose == (0.0, 0.0, 0.0)
    assert hal.read_state().position[:3] != pytest.approx([0.0, 0.0, 0.0], abs=0.1)


@pytest.mark.parametrize(
    "twin", ["so101_follower", "openamrobot_v2", "h1", "h2", "g1", "g1_walk", "g1_gr00t"]
)
def test_every_twin_is_driven_through_the_shell(twin: str, connect_twin: ConnectTwin) -> None:
    """Each twin's command and idle slice run the shell's step boundary exactly once.

    One ``send_action`` and one ``idle_step`` each advance the sim clock and fire the
    post-step observers once; the command stamps ``last_action_ns``; the scene export,
    the staged-slot surface and ``base_pose`` answer — the shell's, on every twin.
    """
    hal = connect_twin(_TWINS.get(twin) or twin)
    names = [j.name for j in hal.description.joints]
    fired: list[int] = []
    hal.add_post_step_observer(lambda: fired.append(1))
    t0 = hal.sim_time_ns()
    row = list(hal.read_state().position)
    if twin == "g1_gr00t":  # legs + waist belong to the S0: command the arms by name
        command = _joint_action(row).model_copy(update={"joint_names": names[15:]})
    else:
        command = _joint_action(row)
    hal.send_action(command)
    assert len(fired) == 1
    assert hal.last_action_ns > 0
    t1 = hal.sim_time_ns()
    assert t0 is not None and t1 is not None and t1 > t0
    assert hal.idle_step(0.02) is True
    assert len(fired) == 2
    assert hal.export_sim_scene() is not None
    assert hal.discard_staged_slots() == 0
    assert all(isinstance(v, float) for v in hal.base_pose)


def test_one_send_is_one_step_boundary_and_estop_releases_the_model(
    connect_twin: ConnectTwin,
) -> None:
    hal = connect_twin("so101_follower")
    fired: list[int] = []
    hal.add_post_step_observer(lambda: fired.append(1))
    tick = hal.sim_dt_per_tick_s
    assert tick is not None

    t0 = hal.sim_time_ns()
    assert t0 is not None
    hal.send_action(_joint_action([0.1, -0.2, 0.3, 0.0, 0.0, 1.0]))
    assert len(fired) == 1
    assert hal.sim_time_ns() == pytest.approx(t0 + tick * 1e9, abs=10)
    assert hal.last_action_ns > 0
    hal.idle_step(0.05)
    assert len(fired) == 2
    assert hal.export_sim_scene() is not None

    with pytest.raises(ROSEStopRequested):
        hal.estop()
    assert hal._connected is False
    assert hal.mujoco_handles() is None
    assert hal._twin.model is None and hal._twin.data is None
    assert hal.sim_time_ns() is None
    assert hal.idle_step(0.05) is False
    with pytest.raises(ROSRuntimeError):
        hal.read_state()
    assert len(fired) == 2


@pytest.fixture
def openamr(connect_twin: ConnectTwin) -> OpenAMRobotV2MujocoHAL:
    hal = connect_twin("openamrobot_v2")
    assert isinstance(hal, OpenAMRobotV2MujocoHAL)
    return hal


def test_an_ungrouped_command_is_its_own_record(openamr: OpenAMRobotV2MujocoHAL) -> None:
    """An ungrouped row naming a subset of the joints (a ramp) is committed as sent.

    The twin composes it by name (the base and lift it leaves out hold their ctrl
    targets) to drive the arms, but ``last_applied_action`` and the watermark take the
    action itself, so no tick field is copied around.
    """
    names = [j.name for j in openamr.description.joints]
    arm = [n for n in names if not n.startswith(("base", "lift"))]
    row = list(openamr.read_state().position)
    row[names.index(arm[1])] -= 0.2
    ramp = _joint_action(row, tick_index=7, runner_session_id=0x683).model_copy(
        update={"joint_names": arm}
    )
    openamr.send_action(ramp)
    assert openamr.last_applied_action is ramp
    assert openamr.last_committed_tick == 7
    handles = openamr.mujoco_handles()
    assert handles is not None
    ctrl = float(handles[1].ctrl[openamr._twin.actuator_index[arm[1]]])
    assert ctrl == pytest.approx(row[names.index(arm[1])])


def test_mobile_twin_base_pose_follows_a_body_twist(openamr: OpenAMRobotV2MujocoHAL) -> None:
    before = openamr.base_pose
    openamr.send_action(_twist(0.2, wz=0.5))
    after = openamr.base_pose
    assert after != pytest.approx(before, abs=1e-4)
    period = openamr.sim_dt_per_tick_s
    assert period is not None
    # One command integrates one control period: the commanded rate on the sim clock.
    assert after[2] - before[2] == pytest.approx(0.5 * period, abs=1e-6)


def test_mobile_twin_reads_its_base_pose_only_while_connected(
    robot_description: Describe,
) -> None:
    """The OpenAMRobot twin reads its three base joints itself; before connect it refuses."""
    hal = OpenAMRobotV2MujocoHAL(description=robot_description("openamrobot_v2"))
    with pytest.raises(ROSRuntimeError, match="not connected"):
        _ = hal.base_pose


def test_mobile_twin_body_twist_is_one_step_boundary(openamr: OpenAMRobotV2MujocoHAL) -> None:
    fired: list[int] = []
    openamr.add_post_step_observer(lambda: fired.append(1))
    t0 = openamr.sim_time_ns()
    openamr.send_action(_twist(0.2, wz=0.5))
    assert openamr.sim_time_ns() != t0  # the physics did advance
    assert len(fired) == 1


def test_an_explicit_twist_interval_off_the_period_is_refused_at_connect(
    robot_description: Describe,
) -> None:
    """A timing-faithful twin integrates exactly one period per twist (#683)."""
    hal = OpenAMRobotV2MujocoHAL(
        description=robot_description("openamrobot_v2"), body_twist_dt_s=0.05
    )
    with pytest.raises(ROSConfigError) as info:
        hal.connect()
    if "body_twist_dt_s" not in str(info.value):  # an offline host refuses the MJCF first
        skip_if_unavailable(info.value, "openamrobot_v2 twin")
    assert hal._connected is False and hal.mujoco_handles() is None


@pytest.mark.parametrize("twin", ["g1", "g1_walk", "g1_gr00t", "h2"])
def test_floating_base_twin_twist_is_one_committed_step(
    twin: str, connect_twin: ConnectTwin
) -> None:
    """A humanoid's ticked twist runs one step boundary and is committed as sent."""
    hal = connect_twin(_TWINS.get(twin) or twin)
    fired: list[int] = []
    hal.add_post_step_observer(lambda: fired.append(1))
    command = _twist(0.1, tick_index=3, runner_session_id=0x683)
    hal.send_action(command)
    assert len(fired) == 1
    assert hal.last_committed_tick == 3
    assert hal.last_applied_action is command
    assert hal.base_twist == pytest.approx((0.1, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert hal.idle_step(0.02) is True
    assert len(fired) == 2


@pytest.mark.parametrize("robot", ["g1", "h2"])
def test_a_glide_slot_group_moves_the_base_and_the_joints_in_one_period(
    robot: str, connect_twin: ConnectTwin
) -> None:
    """A twist slot beside joint slots: one step, both applied, the composed row recorded."""
    hal = connect_twin(robot)
    names = [j.name for j in hal.description.joints]
    arm = [n for n in names if "shoulder" in n or "elbow" in n]
    rest = [n for n in names if n not in arm]
    row = list(hal.read_state().position)
    for n in arm:
        row[names.index(n)] += 0.1
    ticked = {"tick_index": 1, "tick_group_size": 3, "runner_session_id": 0x683}

    def joint_slot(joints: list[str]) -> Action:
        padded = [row[i] if names[i] in joints else 0.0 for i in range(len(names))]
        return _joint_action(padded).model_copy(update={"joint_names": joints, **ticked})

    x0 = hal.base_pose[0]
    fired: list[int] = []
    hal.add_post_step_observer(lambda: fired.append(1))
    for slot in (joint_slot(arm), joint_slot(rest), _twist(0.2, **ticked)):
        hal.send_action(slot)
    assert len(fired) == 1
    assert hal.last_committed_tick == 1
    assert hal.base_pose[0] > x0
    applied = hal.last_applied_action
    assert applied is not None and applied.joint_targets is not None
    assert applied.joint_targets[-1] == pytest.approx(row)


def test_g1_walking_refuses_a_tick_with_a_twist_and_joint_targets(
    connect_twin: ConnectTwin,
) -> None:
    """The walker drives all 29 actuators: acknowledging the joint slots would be false."""
    hal = connect_twin(_g1("playground_walk"))
    names = [j.name for j in hal.description.joints]
    arm = [n for n in names if "shoulder" in n]
    row = list(hal.read_state().position)
    handles = hal.mujoco_handles()
    assert handles is not None
    ctrl = handles[1].ctrl.copy()
    t0 = hal.sim_time_ns()
    ticked = {"tick_index": 1, "tick_group_size": 2, "runner_session_id": 7}
    padded = [row[i] + 0.4 if names[i] in arm else 0.0 for i in range(len(names))]
    hal.send_action(_joint_action(padded).model_copy(update={"joint_names": arm, **ticked}))
    with pytest.raises(ROSConfigError, match="cannot carry a BODY_TWIST and joint targets"):
        hal.send_action(_twist(0.3, **ticked))
    assert hal.last_committed_tick == 0
    assert hal.sim_time_ns() == t0  # nothing moved
    np.testing.assert_array_equal(handles[1].ctrl, ctrl)


def test_gr00t_fall_inside_a_control_period_is_raised_and_not_committed(
    connect_twin: ConnectTwin,
) -> None:
    """The fall detector latching DURING the command's period raises after it (#683)."""
    import mujoco

    hal = connect_twin(_g1("gr00t_wbc"))
    names = [j.name for j in hal.description.joints]
    handles = hal.mujoco_handles()
    assert handles is not None
    model, data = handles
    data.qpos[2] = 0.3  # the pelvis on the floor: the period's first policy tick sees a fall
    mujoco.mj_forward(model, data)
    command = _joint_action(
        list(hal.read_state().position), tick_index=1, runner_session_id=0x683
    ).model_copy(update={"joint_names": names[15:]})
    assert hal.s0_fallen is False  # not latched before the command: the second raise fires
    with pytest.raises(ROSSafetyViolation, match="fell"):
        hal.send_action(command)
    assert hal.s0_fallen is True
    assert hal.last_committed_tick == 0
    assert hal.last_applied_action is None


# ── One stepping loop, driven by the shell (#683) ────────────────────────────


def test_g1_glide_idle_keeps_the_pelvis_pinned(connect_twin: ConnectTwin) -> None:
    """Idle slices run the per-step hook: the glide pin holds every physics step."""
    hal = connect_twin("g1")
    handles = hal.mujoco_handles()
    assert handles is not None
    _model, data = handles
    z0 = float(data.qpos[2])
    for _ in range(10):
        assert hal.idle_step(0.1) is True
    # Each step starts from the pin (only one step's fall, ~4e-5 m, shows).
    assert float(data.qpos[2]) == pytest.approx(z0, abs=1e-3)


def test_post_step_observers_run_after_the_commit(connect_twin: ConnectTwin) -> None:
    """Observers read the tick just applied; a failing one never un-applies it.

    They used to run inside the step boundary, before the commit: they saw the previous
    tick, and a raising observer left a stepped tick uncommitted, so a resend stepped
    the same tick again. Raising after the commit would still report an applied tick as
    unapplied (no ``/openral/action_applied`` ack, a resend refused as stale), so an
    observer's error is logged and the send completes; a safety violation propagates.
    """
    from structlog.testing import capture_logs

    hal = connect_twin("so101_follower")
    row = list(hal.read_state().position)
    seen: list[int] = []
    hal.add_post_step_observer(lambda: seen.append(hal.last_committed_tick))
    hal.send_action(_joint_action(row, tick_index=1, runner_session_id=0x683))
    assert seen == [1]

    def boom() -> None:
        raise ROSRuntimeError("observer failed")

    hal.add_post_step_observer(boom)
    with capture_logs() as logs:
        hal.send_action(_joint_action(row, tick_index=2, runner_session_id=0x683))
    assert hal.last_committed_tick == 2
    assert seen == [1, 2]
    failed = [e for e in logs if e["event"] == "hal.post_step_observer_failed"]
    assert len(failed) == 1 and failed[0]["tick"] == 2
    steps = hal._step_count
    with pytest.raises(ROSRuntimeError):
        hal.send_action(_joint_action(row, tick_index=2, runner_session_id=0x683))
    assert hal._step_count == steps  # the replay is refused, nothing stepped again

    def stop() -> None:
        raise ROSSafetyViolation("observer saw a violation")

    hal.add_post_step_observer(stop)
    with pytest.raises(ROSSafetyViolation, match="observer saw a violation"):
        hal.send_action(_joint_action(row, tick_index=3, runner_session_id=0x683))
    assert hal.last_committed_tick == 3


def test_g1_walk_idle_slices_keep_one_pin(connect_twin: ConnectTwin) -> None:
    """Consecutive idle slices of the walking twin hold one pin: the stand never sinks.

    Re-capturing the pin each slice read qpos one physics step's sag below it, so every
    slice lowered the stand a little more.
    """
    hal = connect_twin(_g1("playground_walk"))
    handles = hal.mujoco_handles()
    assert handles is not None
    _model, data = handles
    heights = []
    for _ in range(60):
        assert hal.idle_step(0.1) is True
        heights.append(float(data.qpos[2]))
    # Held: within the leg PD's jitter (~1e-5 m). Re-capturing sank ~1.5 mm over 60.
    assert min(heights) == pytest.approx(heights[0], abs=1e-4)


@pytest.mark.parametrize("robot", ["g1", "h2"])
def test_the_glide_pin_is_kept_across_reset_to_pose(robot: str, connect_twin: ConnectTwin) -> None:
    """The snap moves no free joint, so the pin holds: same pose, same height (#683).

    A pin recaptured from qpos would read the free joint one physics step after it
    was written — lower by that step's fall — so every snap would sink the base.
    """
    hal = connect_twin(robot)
    hal.send_action(_twist(0.3, wz=0.2))
    pin = hal._twin._base_pin
    hal.reset_to_pose(list(hal.read_state().position))
    assert hal._twin._base_pin == pin
    hal.idle_step(0.05)
    assert hal._twin._base_pin == pin
    assert hal.base_pose == pytest.approx(pin[:3], abs=1e-12)


def test_h1_idle_keeps_tracking_the_last_target(
    connect_twin: ConnectTwin, robot_description: Describe
) -> None:
    """The H1's software PD runs during idle slices, holding the last commanded pose."""
    hal = connect_twin(_h1(robot_description))
    target = [0.0] * len(hal.description.joints)
    target[11] = 0.4  # torso
    for _ in range(30):
        hal.send_action(_joint_action(target))
    for _ in range(20):
        hal.idle_step(0.1)
    assert hal.read_state().position[11] == pytest.approx(0.4, abs=0.02)


def test_h1_reset_to_pose_and_reconnect_drop_the_previous_target(
    connect_twin: ConnectTwin, robot_description: Describe
) -> None:
    """The snapped pose is the PD target; a reload applies no torque before a command.

    The previous target survived both: idle PD drove the twin back to the old command
    (0.2 rad within 1 s), and the snap's ctrl re-seed applied joint angles as N·m on
    the torque motors.
    """
    hal = connect_twin(_h1(robot_description))
    n = len(hal.description.joints)
    for _ in range(25):
        hal.send_action(_joint_action([0.2] * n))
    hal.reset_to_pose([0.0] * n)
    handles = hal.mujoco_handles()
    assert handles is not None
    assert float(np.abs(handles[1].ctrl).max()) == 0.0  # no joint angle applied as torque
    for _ in range(20):
        hal.idle_step(0.05)
    assert max(abs(q) for q in hal.read_state().position) < 0.01

    for _ in range(25):
        hal.send_action(_joint_action([0.2] * n))
    hal.disconnect()
    hal.connect()
    assert hal._twin._pd_targets is None
    for _ in range(20):
        hal.idle_step(0.05)
    assert max(abs(q) for q in hal.read_state().position) < 0.01


@pytest.mark.parametrize("robot", ["h1", "h2"])
def test_software_pd_writes_the_clamped_pd_law(
    robot: str, connect_twin: ConnectTwin, robot_description: Describe
) -> None:
    """The precomputed PD (index arrays, gains, ctrlrange) is the scalar law, every joint."""
    hal = connect_twin(_h1(robot_description) if robot == "h1" else robot)
    twin = hal._twin
    hal.send_action(_joint_action([0.1] * len(hal.description.joints)))
    model, data = twin.model, twin.data
    twin._per_step()
    for name in twin.joint_names:
        act = twin.actuator_index.get(name)
        if act is None:
            continue
        kp, kd = twin._pd_gains[name]
        q = float(data.qpos[twin.joint_qpos_addr[name]])
        dq = float(data.qvel[twin.joint_qvel_addr[name]])
        low, high = (float(v) for v in model.actuator_ctrlrange[act])
        expected = max(low, min(high, kp * (0.1 - q) - kd * dq))
        assert float(data.ctrl[act]) == pytest.approx(expected, abs=1e-9), name


@pytest.mark.parametrize(
    "twin", ["so101_follower", "openamrobot_v2", "h1", "h2", "g1", "g1_walk", "g1_gr00t"]
)
def test_a_dropped_connected_twin_is_freed_by_refcount(
    twin: str, robot_description: Describe
) -> None:
    """No HAL <-> rollout reference cycle, on every rollout subclass."""
    from openral_hal import build_hal

    build: Callable[[], Any] = (
        _h1(robot_description)
        if twin == "h1"
        else _TWINS.get(twin) or (lambda: build_hal(robot_description(twin), mode="sim"))
    )
    hal = build()
    try:
        hal.connect()
    except ROSConfigError as exc:
        skip_if_unavailable(exc, f"{twin} twin")
    assert hal.idle_step(0.02) is True
    hal.read_images()  # creates the renderers when a GL backend exists
    rollout_ref = weakref.ref(hal._twin)
    renderers = [weakref.ref(r) for r in hal._twin.renderers.values()]
    gc.disable()
    try:
        del hal
        assert rollout_ref() is None
        assert all(r() is None for r in renderers)
    finally:
        gc.enable()


def test_mobile_twin_arm_and_twist_tick_is_one_control_period(
    openamr: OpenAMRobotV2MujocoHAL,
) -> None:
    """A slot tick carrying arm slots and a twist moves both in ONE step (one period)."""
    names = [j.name for j in openamr.description.joints]
    arm = [n for n in names if n.startswith("left_joint")]
    right_arm = [n for n in names if n.startswith("right_joint")]
    state = openamr.read_state()
    row = list(state.position)
    for n in arm:
        row[names.index(n)] += 0.05
    ticked = {
        "tick_index": openamr.last_committed_tick + 1,
        "tick_group_size": 5,
        "runner_session_id": 0x683,
    }

    def arm_slot(joints: list[str]) -> Action:
        padded = [row[i] if names[i] in joints else 0.0 for i in range(len(names))]
        return _joint_action(padded).model_copy(update={"joint_names": joints, **ticked})

    def grip_slot(ee: str) -> Action:
        return Action(
            control_mode=ControlMode.GRIPPER_POSITION,
            horizon=1,
            gripper=[state.position[names.index(ee)]],
            ee_name=ee,
            **ticked,
        )

    slots = [
        arm_slot(arm),
        grip_slot("left_gripper"),
        grip_slot("right_gripper"),
        _twist(0.2, **ticked),
        arm_slot(right_arm),
    ]
    fired: list[int] = []
    openamr.add_post_step_observer(lambda: fired.append(1))
    x0, _, _ = openamr.base_pose
    t0 = openamr.sim_time_ns()
    assert t0 is not None
    for slot in slots:
        openamr.send_action(slot)
    assert openamr.last_committed_tick == ticked["tick_index"]
    assert len(fired) == 1
    tick_s = openamr.sim_dt_per_tick_s
    assert tick_s is not None
    assert openamr.sim_time_ns() == pytest.approx(t0 + tick_s * 1e9, abs=10)
    # The base moves at the commanded speed: one period of 0.2 m/s.
    assert openamr.base_pose[0] - x0 == pytest.approx(0.2 * tick_s, abs=1e-3)
    handles = openamr.mujoco_handles()
    assert handles is not None
    _model, data = handles
    for n in arm:
        assert float(data.ctrl[openamr._twin.actuator_index[n]]) == pytest.approx(
            row[names.index(n)]
        )


# ── One time base, one clock, one estop; a refused connect leaves nothing (#683) ──


@pytest.mark.parametrize("keyframe", [99, -1])
def test_a_refused_load_leaves_the_twin_disconnected_and_clockless(
    robot_description: Describe, keyframe: int
) -> None:
    """An out-of-range keyframe refuses the connect; no reader sees a half-loaded twin.

    ``-1`` too: MuJoCo would silently reset to the defaults instead of a keyframe.
    """
    from openral_core import ClockOrigin

    desc = robot_description("so101_follower")
    try:
        kw = MujocoArmHAL._sim_kwargs_for(desc)
    except ROSConfigError as exc:
        skip_if_unavailable(exc, "so101_follower MJCF")
    kw["keyframe_index"] = keyframe
    hal = MujocoArmHAL(desc, **kw)
    with pytest.raises(ROSConfigError, match=f"keyframe_index={keyframe}"):
        hal.connect()
    assert hal._connected is False
    assert hal._twin.model is None and hal._twin.data is None
    assert hal.mujoco_handles() is None
    assert hal.sim_time_ns() is None
    assert hal.clock_authority().origin is ClockOrigin.HOST_WALL
    assert hal.read_images() == {}
    assert hal.idle_step(0.05) is False
    with pytest.raises(ROSRuntimeError, match="not connected"):
        hal.read_state()
    hal.disconnect()


def test_the_sim_clock_is_monotonic_across_a_reconnect(connect_twin: ConnectTwin) -> None:
    hal = connect_twin("so101_follower")
    for _ in range(5):
        hal.send_action(_joint_action([0.1, -0.2, 0.3, 0.0, 0.0, 1.0]))
    before = hal.sim_time_ns()
    assert before is not None and before > 0
    hal.disconnect()
    hal.connect()
    after = hal.sim_time_ns()
    assert after is not None and after >= before


def test_estop_stays_latched_across_a_reconnect_until_reset(connect_twin: ConnectTwin) -> None:
    """The latch drops commands — saying so once per latch episode — until reset_estop."""
    from structlog.testing import capture_logs

    hal = connect_twin("so101_follower")
    with pytest.raises(ROSEStopRequested):
        hal.estop()
    assert hal.estop_latched is True
    assert hal.mujoco_handles() is None
    hal.connect()
    assert hal.estop_latched is True
    t0 = hal.sim_time_ns()
    assert t0 is not None
    stamp = hal.last_action_ns
    with capture_logs() as logs:
        for _ in range(3):
            hal.send_action(_joint_action([0.1, -0.2, 0.3, 0.0, 0.0, 1.0]))
    drops = [e for e in logs if e["event"] == "hal.send_action.dropped_estop_latched"]
    assert len(drops) == 1 and drops[0]["log_level"] == "warning"
    assert hal.last_action_ns == stamp  # a dropped command does not pause the idle stepper
    assert hal.idle_step(0.1) is False
    assert hal.sim_time_ns() == t0  # nothing stepped while latched
    hal.reset_estop()
    hal.send_action(_joint_action([0.1, -0.2, 0.3, 0.0, 0.0, 1.0]))
    after = hal.sim_time_ns()
    assert after is not None and after > t0


def test_the_time_base_is_checked_and_logged_once_per_connect(
    robot_description: Describe,
) -> None:
    """The shell alone holds the twin to its rate and logs ``sim.time_base``."""
    import logging

    import structlog
    from openral_hal import build_hal
    from structlog.testing import capture_logs

    hal = build_hal(robot_description("so101_follower"), mode="sim")
    assert isinstance(hal, MujocoArmHAL)
    # The suite filters below WARNING (root conftest); this line is INFO.
    previous = structlog.get_config()["wrapper_class"]
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
    try:
        with capture_logs() as logs:
            hal.connect()
    except ROSConfigError as exc:
        skip_if_unavailable(exc, "so101_follower twin")
    finally:
        structlog.configure(wrapper_class=previous)
    try:
        lines = [e for e in logs if e["event"] == "sim.time_base"]
        assert len(lines) == 1
        assert lines[0]["backend"] == "MujocoArmHAL"  # the HAL class, as before #683
        assert lines[0]["physics_steps"] == hal._twin.settle_steps
        assert hal.clock_authority().clock_id == "mujoco"
    finally:
        hal.disconnect()


def test_a_refused_slot_keeps_neither_the_stale_group_timer_nor_the_idle_hold(
    connect_twin: ConnectTwin,
) -> None:
    """A replayed slot arriving while a dead tick is staged must not keep it alive (#683).

    Only a slot the stager takes restarts the 5 s stale-group timer and stamps
    ``last_action_ns``; a refused one leaves both, so the dead tick still expires.
    """
    hal = connect_twin("so101_follower")
    names = [j.name for j in hal.description.joints]
    row = list(hal.read_state().position)

    def slot(tick: int, joints: list[str]) -> Action:
        return _joint_action(row, tick_index=tick, runner_session_id=10).model_copy(
            update={"joint_names": joints, "tick_group_size": 2}
        )

    hal.send_action(slot(1, names[:3]))
    hal.send_action(slot(1, names[3:]))
    assert hal.last_committed_tick == 1
    hal.send_action(slot(2, names[:3]))  # tick 2 never completes
    since, stamp = hal._pending_since_ns, hal.last_action_ns
    with pytest.raises(ROSRuntimeError, match="stale slot group"):
        hal.send_action(slot(1, names[:3]))  # a replay: refused
    assert (hal._pending_since_ns, hal.last_action_ns) == (since, stamp)


def test_twists_off_the_plane_are_refused_with_one_tolerance(connect_twin: ConnectTwin) -> None:
    """The shell and the twin apply the same rule and tolerance (``PLANAR_TWIST_EPS``)."""
    from openral_hal._floating_base import PLANAR_TWIST_EPS

    hal = connect_twin("g1")
    tiny = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[(0.1, 0.0, 10 * PLANAR_TWIST_EPS, 0.0, 0.0, 0.0)],
    )
    with pytest.raises(ROSConfigError, match="linear-z"):
        hal.send_action(tiny)
    with pytest.raises(ROSConfigError, match="linear-z"):
        hal._twin.step_action_group([tiny])


# ── CodeRabbit review of #696: refusals and resets ───────────────────────────


def test_an_empty_action_group_is_refused(rollout: Callable[[str], MujocoTwinRollout]) -> None:
    """A direct ``step_action_group([])`` is a typed refusal, not an ``IndexError``."""
    r = rollout("so101_follower")
    with pytest.raises(ROSConfigError, match="empty action group"):
        r.step_action_group([])


def test_a_non_finite_twist_interval_is_refused(robot_description: Describe) -> None:
    """``body_twist_dt_s=nan`` passed both comparisons and put NaN into the base pose."""
    hal = OpenAMRobotV2MujocoHAL(
        description=robot_description("openamrobot_v2"), body_twist_dt_s=float("nan")
    )
    expect_refusal(hal.connect, "body_twist_dt_s=nan", "openamrobot_v2 twin")
    assert hal._connected is False


def test_a_twist_on_a_baseless_twin_leaves_the_arm_targets(connect_twin: ConnectTwin) -> None:
    """A refused mixed group writes no ``ctrl``: the base refusal comes first."""
    hal = connect_twin("so101_follower")
    twin = hal._twin
    assert twin.data is not None
    ctrl = twin.data.ctrl.copy()
    row = [0.4] * len(hal.description.joints)
    with pytest.raises(ROSConfigError, match="has no base"):
        twin._command(row, (0.1, 0.0, 0.0))
    np.testing.assert_array_equal(twin.data.ctrl, ctrl)


def test_a_snap_reports_effort_against_the_new_ctrl(connect_twin: ConnectTwin) -> None:
    """``reset_to_pose`` refreshes the derived state after its last ``ctrl`` write.

    Position actuators then hold the snapped pose: zero effort, not ``kp`` times the
    distance to the previous target.
    """
    hal = connect_twin("so101_follower")
    pose = [0.5, -0.3, 0.4, 0.2, 0.1, 0.5]
    hal.reset_to_pose(pose)
    effort = hal.read_state().effort
    assert effort is not None
    assert max(abs(e) for e in effort[:-1]) == pytest.approx(0.0, abs=1e-6)


def test_a_snap_zeroes_every_finger_coordinate_velocity(connect_twin: ConnectTwin) -> None:
    """The Franka finger pair is two qpos slots: the snap zeroes both velocities."""
    hal = connect_twin("franka_panda")
    twin = hal._twin
    assert twin.data is not None
    gripper = next(iter(twin.grippers_by_joint.values()))
    dofs = [twin._dof_of_qpos(a) for a in gripper.qpos_addrs]
    assert len(dofs) == 2
    twin.data.qvel[dofs] = 0.3
    hal.reset_to_pose(list(hal.read_state().position))
    assert [float(twin.data.qvel[d]) for d in dofs] == [0.0, 0.0]


def test_dof_addresses_follow_the_free_joint(connect_twin: ConnectTwin) -> None:
    """After a free joint (7 qpos, 6 dofs) the qpos and DOF addresses differ by one."""
    hal = connect_twin("h2")
    model = hal._twin.model
    assert model is not None
    first = int(model.jnt_qposadr[1])
    assert hal._twin._dof_of_qpos(first) == int(model.jnt_dofadr[1]) == first - 1


def test_an_affine_reset_round_trips_whatever_the_write_mode(connect_twin: ConnectTwin) -> None:
    """``AFFINE_LOW_HIGH`` resets invert the read mapping, not the write mode."""
    from openral_core.schemas import GripperWriteMode

    hal = connect_twin("makermods_metal_station")
    twin = hal._twin
    for name, gripper in list(twin.grippers_by_joint.items()):
        twin.grippers_by_joint[name] = gripper.model_copy(
            update={"write_mode": GripperWriteMode.PASSTHROUGH}
        )
    gripper = next(iter(twin.grippers_by_joint.values()))
    twin._reset_gripper_qpos(gripper, 0.5)
    assert twin._read_gripper_value(gripper) == pytest.approx(0.5, abs=1e-9)


def test_a_pd_twin_loads_with_no_torque(
    robot_description: Describe, connect_twin: ConnectTwin
) -> None:
    """H1 (``hold_pose_on_load=False``): ``ctrl`` is torque, so nothing is seeded into it."""
    hal = connect_twin(_h1(robot_description))
    twin = hal._twin
    assert twin.data is not None
    assert not np.any(twin.data.ctrl[twin._pd_act])


def test_a_gr00t_snap_leaves_no_joint_angle_as_torque(connect_twin: ConnectTwin) -> None:
    """The base reset seeds ``ctrl`` from joint angles; GR00T's motors take torque."""
    hal = connect_twin(_g1("gr00t_wbc"))
    hal.reset_to_pose(list(hal.read_state().position))
    assert hal._twin.data is not None
    assert not np.any(hal._twin.data.ctrl)


def test_gr00t_refuses_a_manifest_joint_order_it_cannot_slice(
    robot_description: Describe,
) -> None:
    """Commands are composed in manifest order and sliced by position: the order is checked."""
    from openral_hal.g1 import G1MujocoHAL

    desc = robot_description("g1")
    joints = list(desc.joints)
    joints[0], joints[1] = joints[1], joints[0]
    hal = G1MujocoHAL(
        s0_controller="gr00t_wbc", description=desc.model_copy(update={"joints": joints})
    )
    expect_refusal(hal.connect, "G1_DESCRIPTION order", "g1 gr00t_wbc twin")


def test_the_walking_twin_refuses_a_timestep_off_the_policy_cadence(
    robot_description: Describe,
) -> None:
    """At 30 Hz the time base shrinks the timestep: ten steps are no longer 20 ms."""
    from openral_hal.g1 import G1MujocoHAL

    desc = robot_description("g1")
    assert desc.action_spec is not None
    spec = desc.action_spec.model_copy(update={"control_freq_hz": 30.0})
    hal = G1MujocoHAL(
        s0_controller="playground_walk", description=desc.model_copy(update={"action_spec": spec})
    )
    expect_refusal(hal.connect, "trained at 20 ms", "g1 walking twin")


def test_a_mixed_amr_tick_records_the_base_it_moved(connect_twin: ConnectTwin) -> None:
    """The applied record carries the base targets the twist wrote, not the pre-twist ones."""
    hal = connect_twin("openamrobot_v2")
    names = [j.name for j in hal.description.joints]
    arms = [j.name for j in hal.description.joints if j.role not in ("base", "torso")]
    position = dict(zip(names, hal.read_state().position, strict=True))
    slot = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[position[n] for n in names]],  # full-dof row; base and lift held
        joint_names=arms,
        tick_index=1,
        tick_group_size=2,
        runner_session_id=0x683,
    )
    hal.send_action(slot)
    hal.send_action(_twist(vx=0.5, tick_index=1, tick_group_size=2, runner_session_id=0x683))
    applied = hal.last_applied_action
    assert applied is not None and applied.joint_targets is not None
    record = dict(zip(applied.joint_names or names, applied.joint_targets[-1], strict=True))
    base_x = hal.description.base_joints[0]
    assert record[base_x] == pytest.approx(hal.base_pose[0], abs=1e-9)
    assert record[base_x] > 0.0


def test_reset_episode_needs_a_connection(robot_description: Describe) -> None:
    """Before ``connect`` it is refused: it would skip the width probe and the time base."""
    desc = robot_description("so101_follower")
    try:
        hal = MujocoArmHAL.from_description(desc)
    except ROSConfigError as exc:
        skip_if_unavailable(exc, "so101_follower MJCF")
    with pytest.raises(ROSRuntimeError, match="not connected"):
        hal.reset_episode()
    assert hal._connected is False


def test_a_failed_load_leaves_no_camera_rig_file(
    robot_description: Describe, tmp_path: Path
) -> None:
    """The rig file is written and removed inside one cleanup scope, also on failure.

    An MJCF that rigs the manifest's cameras but does not compile (a mesh it never
    declares) refuses the load; its temporary rig must not stay beside it.
    """
    desc = robot_description("so101_follower")
    mjcf = tmp_path / "broken.xml"
    mjcf.write_text(
        '<mujoco><worldbody><body name="gripper"><geom type="mesh" mesh="absent"/>'
        "</body></worldbody></mujoco>"
    )
    names = [j.name for j in desc.joints]
    twin = MujocoTwinRollout(
        desc,
        mjcf_path=str(mjcf),
        joint_qpos_addr={n: i for i, n in enumerate(names)},
        actuator_index={n: i for i, n in enumerate(names)},
    )
    # The message names the rig file: it was written, then removed.
    with pytest.raises(ROSConfigError, match=r"Could not load MJCF '.*_camrig_.*\.xml'"):
        twin.load()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["broken.xml"]
