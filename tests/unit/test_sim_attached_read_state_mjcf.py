# SPDX-License-Identifier: Apache-2.0
"""``SimAttachedHAL.read_state`` over an MJCF it does not own: indices and gripper units.

Real ALOHA manifest (``robots/aloha_bimanual/robot.yaml``) on the real gym-aloha MJCF,
behind ``FakeSimEnv`` (the ``SimRollout`` boundary double, CLAUDE.md §1.11). Found in
review (#754): the joint-name index skipped unnamed joints, so every joint after one read
another joint's ``qpos`` (here: an unnamed joint ahead of both arms shifted every arm joint).

Skips without ``mujoco`` or ``gym_aloha``.
"""

from __future__ import annotations

from typing import Any

import pytest
from openral_core import RobotDescription
from openral_core.assets import AssetFetchError, AssetRefError, resolve_asset
from openral_hal.sim_attached import SimAttachedHAL

from tests.unit.fakes.fake_sim_env import FakeSimEnv

mujoco = pytest.importorskip("mujoco")


def _aloha(*, unnamed_joint: bool = False) -> tuple[RobotDescription, Any, Any]:
    desc = RobotDescription.from_yaml("robots/aloha_bimanual/robot.yaml")
    assert desc.assets.mjcf is not None
    try:
        path = resolve_asset(desc.assets.mjcf, "mjcf")
    except (AssetFetchError, AssetRefError) as exc:
        pytest.skip(f"gym-aloha MJCF unavailable: {exc}")
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
