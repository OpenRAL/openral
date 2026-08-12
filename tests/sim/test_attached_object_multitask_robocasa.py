"""Multi-primitive attached payloads across two real RoboCasa tasks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    PhysicsBackend,
    Pose6D,
    RobotDescription,
    SceneSpec,
    SimEnvironment,
    SphereShape,
    TaskSpec,
    VLASpec,
)
from openral_hal._sim_attachment_evidence import (
    SimAttachmentEvidenceTracker,
    SimObjectMobility,
    classify_body_mobility,
    extract_body_primitives,
)
from openral_hal.depth_cloud import depth_synth_kwargs, robot_self_body_ids
from openral_hal.sim_attached import SimAttachedHAL
from openral_sim._deps import _has_robocasa_kitchen
from openral_sim.backends.depth_camera import synthesize_depth_pointcloud
from openral_sim.backends.robocasa import _build_robocasa_sim

from tests.sim.conftest import mujoco_renderer_probe_error

_ROBOCASA_INACTIVE = not _has_robocasa_kitchen()
_RENDERER_ERROR = mujoco_renderer_probe_error() if not _ROBOCASA_INACTIVE else ""

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        _ROBOCASA_INACTIVE,
        reason="RoboCasa kitchen fork is not active",
    ),
    pytest.mark.skipif(
        bool(_RENDERER_ERROR),
        reason=_RENDERER_ERROR or "no MuJoCo offscreen renderer",
    ),
]


@contextmanager
def _build_hal(task_name: str) -> Iterator[SimAttachedHAL]:
    scene_id = f"robocasa/{task_name}"
    config = SimEnvironment(
        robot_id="panda_mobile",
        scene=SceneSpec(
            id=scene_id,
            backend=PhysicsBackend.MUJOCO,
            observation_height=256,
            observation_width=256,
            backend_options={
                "mode": "prebuilt",
                "prebuilt_task": task_name,
                "robots": ["PandaMobile"],
                "controller": "OSC_POSE",
            },
        ),
        task=TaskSpec(
            id=f"{scene_id}/attachment-contract",
            scene_id=scene_id,
            instruction="validate attached payload collision geometry",
        ),
        vla=VLASpec(id="zero", weights_uri="mock://zero"),
        seed=1,
    )
    rollout = _build_robocasa_sim(config)
    description = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    hal = SimAttachedHAL(rollout, description, env_reset_seed=1)
    hal.connect()
    try:
        yield hal
    finally:
        hal.disconnect()
        rollout.close()


def _attachment(object_id: str, body_name: str) -> AttachedCollisionObject:
    return AttachedCollisionObject(
        object_id=object_id,
        attach_link="panda_link7",
        touch_links=["panda_finger_pair"],
        primitives=[
            AttachedCollisionPrimitive(
                shape=BoxShape(half_extents_m=(0.08, 0.03, 0.02)),
                pose_in_object=Pose6D(
                    xyz=(0.0, 0.0, 0.0),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id=object_id,
                ),
            ),
            AttachedCollisionPrimitive(
                shape=SphereShape(radius_m=0.02),
                pose_in_object=Pose6D(
                    xyz=(0.06, 0.0, 0.0),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id=object_id,
                ),
            ),
        ],
        pose_in_link=Pose6D(
            xyz=(0.0, 0.0, 0.16),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="panda_link7",
        ),
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.SIM_CONTACT,
        evidence_ref=f"mujoco_body:{body_name}",
        stamp_ns=1,
    )


@pytest.mark.parametrize(
    "task_name",
    ["PickPlaceCounterToCabinet", "PickPlaceCounterToSink"],
)
def test_two_unknown_payloads_mask_and_restore_across_tasks(task_name: str) -> None:
    import mujoco

    with _build_hal(task_name) as hal:
        handles = hal.mujoco_handles()
        assert handles is not None
        model, data = handles
        free_bodies = [
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.jnt_bodyid[joint_id]),
            )
            for joint_id in range(int(model.njnt))
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        object_bodies = [name for name in free_bodies if name and not name.startswith("robot")]
        assert len(object_bodies) >= 2

        spec = next(sensor for sensor in hal.description.sensors if sensor.name == "front_depth")
        kwargs = depth_synth_kwargs(
            spec,
            max_range_default=5.0,
            render_size=(256, 256),
        )
        robot_ids = robot_self_body_ids(
            model,
            [joint.sim_joint_name for joint in hal.description.joints],
        )

        def read_points() -> NDArray[np.float32]:
            excluded = robot_ids | hal.read_attached_body_ids()
            return synthesize_depth_pointcloud(
                model=model,
                data=data,
                stride=4,
                exclude_body_ids=excluded,
                **kwargs,
            )

        baseline = read_points()
        attachments = [
            _attachment(f"{task_name}:payload:{index}", body_name)
            for index, body_name in enumerate(object_bodies[:2])
        ]
        assert all(len(obj.primitives) == 2 for obj in attachments)
        hal.update_attached_objects(attachments)
        transparent = read_points()
        assert transparent.shape == baseline.shape
        assert not np.allclose(transparent, baseline)
        assert np.count_nonzero(transparent[:, 2] > baseline[:, 2] + 1e-4) > 0

        masked_body_ids = hal.read_attached_body_ids()
        assert {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for body_name in object_bodies[:2]
        } <= masked_body_ids

        hal.update_attached_objects([])
        assert np.allclose(read_points(), baseline)


@pytest.mark.parametrize(
    "task_name",
    ["PickPlaceCounterToCabinet", "PickPlaceCounterToSink"],
)
def test_runtime_evidence_resolves_unknown_free_objects(task_name: str) -> None:
    import mujoco

    with _build_hal(task_name) as hal:
        handles = hal.mujoco_handles()
        assert handles is not None
        model, data = handles
        tracker = SimAttachmentEvidenceTracker(model, hal.description)
        assert tracker.update(data, stamp_ns=1) is None

        free_roots = [
            int(model.jnt_bodyid[joint_id])
            for joint_id in range(int(model.njnt))
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
            and (
                name := mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.jnt_bodyid[joint_id]),
                )
            )
            and not name.startswith("robot")
        ]
        assert len(free_roots) >= 2
        for root in free_roots[:2]:
            assert classify_body_mobility(model, root) is SimObjectMobility.FREE
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root)
            primitives = extract_body_primitives(
                model,
                data,
                root_body_id=root,
                object_id=f"sim:{body_name}",
            )
            assert primitives
