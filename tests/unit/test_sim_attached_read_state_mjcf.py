# SPDX-License-Identifier: Apache-2.0
"""``SimAttachedHAL.read_state`` over an MJCF it does not own: indices and gripper units.

Real ALOHA manifest (``robots/aloha_bimanual/robot.yaml``) on the real gym-aloha MJCF. The
gripper case runs the real gym-aloha env (``scenes/sim/aloha_transfer_cube.yaml``). The index
case needs an unnamed joint ahead of both arms, which no shipped scene has, so it compiles
the gym-aloha MJCF with one and serves it through ``FakeSimEnv`` (the repo's ``SimRollout``
boundary double, ``tests/unit/fakes/``), whose only role here is to hand over the real
``MjModel`` / ``MjData``. Two defects found in review (#754):

* the joint-name index skipped unnamed joints, so every joint after one read another
  joint's ``qpos`` (here: an unnamed joint ahead of both arms shifted every arm joint);
* an ``affine_low_high`` gripper published its finger travel (gym-aloha: 0.021-0.057 m)
  on a channel the manifest defines as the ``[0, 1]`` opening.

Skips without ``mujoco`` or ``gym_aloha``.
"""

from __future__ import annotations

from typing import Any

import pytest
from openral_core import RobotDescription
from openral_core.assets import AssetFetchError, resolve_asset
from openral_hal.sim_attached import SimAttachedHAL

from tests.unit.fakes.fake_sim_env import FakeSimEnv

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gym_aloha", reason="the ALOHA twin ships with gym-aloha")


def _aloha(*, unnamed_joint: bool = False) -> tuple[RobotDescription, Any, Any]:
    desc = RobotDescription.from_yaml("robots/aloha_bimanual/robot.yaml")
    assert desc.assets.mjcf is not None
    try:
        path = resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetFetchError as exc:
        pytest.skip(f"gym-aloha MJCF not fetchable: {exc}")
    spec = mujoco.MjSpec.from_file(str(path))
    if unnamed_joint:
        # gym-aloha's jointless ``midair`` body precedes both arms: an unnamed slide on it
        # is joint 0, ahead of every manifest joint. Group 4 counts toward inertia there.
        midair = spec.body("midair")
        midair.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
        midair.add_geom(size=[0.01, 0, 0], mass=0.01, group=4, contype=0, conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    return desc, model, data


def _hal(desc: RobotDescription, model: Any, data: Any) -> SimAttachedHAL:
    hal = SimAttachedHAL(FakeSimEnv(action_dim=14, handles=(model, data)), desc)
    hal.connect()
    return hal


def test_an_unnamed_joint_does_not_shift_the_joints_after_it() -> None:
    desc, model, data = _aloha(unnamed_joint=True)
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 0) is None
    for jid in range(int(model.njnt)):
        data.qpos[int(model.jnt_qposadr[jid])] = 0.1 * (jid + 1)
    mujoco.mj_forward(model, data)
    state = _hal(desc, model, data).read_state()
    name_id = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(model.njnt)}
    for joint in ("left_waist", "right_waist", "right_wrist_rotate"):
        spec = next(j for j in desc.joints if j.name == joint)
        expected = float(data.qpos[int(model.jnt_qposadr[name_id[spec.sim_joint_name]])])
        assert state.position[state.name.index(joint)] == pytest.approx(expected), joint


@pytest.mark.parametrize(("finger_m", "opening"), [(0.057, 1.0), (0.021, 0.0), (0.039, 0.5)])
def test_an_affine_gripper_reads_its_opening_not_its_finger_travel(
    finger_m: float, opening: float
) -> None:
    pytest.importorskip("openral_sim")
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/aloha_transfer_cube.yaml", robot_id_fallback="aloha_bimanual"
    )
    try:
        hal = SimAttachedHAL(
            env,
            RobotDescription.from_yaml("robots/aloha_bimanual/robot.yaml"),
            env_reset_seed=seed,
        )
        hal.connect()
        handles = hal.mujoco_handles()
        assert handles is not None
        model, data = handles
        finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "vx300s_left/left_finger")
        data.qpos[int(model.jnt_qposadr[finger])] = finger_m
        mujoco.mj_forward(model, data)
        state = hal.read_state()
        assert state.position[state.name.index("left_gripper")] == pytest.approx(opening)
    finally:
        env.close()
