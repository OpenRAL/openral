"""Ground-truth MuJoCo attachment evidence for free and articulated objects."""

from __future__ import annotations

import pytest
from openral_core import (
    BoxShape,
    EndEffectorSpec,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_hal._sim_attachment_evidence import (
    SimAttachmentEvidenceTracker,
    SimObjectMobility,
    classify_body_mobility,
    extract_body_primitives,
)

mujoco = pytest.importorskip("mujoco")

_MJCF = """
<mujoco model="attachment_evidence">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="robot0_link7">
      <joint name="robot0_joint7" type="hinge" axis="0 0 1"/>
      <geom name="wrist" type="sphere" size="0.02"/>
      <body name="gripper0_leftfinger" pos="-0.045 0 0">
        <joint name="gripper0_left_joint" type="slide" axis="-1 0 0"/>
        <geom name="left_pad" type="box" size="0.02 0.04 0.04"/>
      </body>
      <body name="gripper0_rightfinger" pos="0.045 0 0">
        <joint name="gripper0_right_joint" type="slide" axis="1 0 0"/>
        <geom name="right_pad" type="box" size="0.02 0.04 0.04"/>
      </body>
    </body>
    <body name="unknown_payload" pos="0 0 0">
      <freejoint name="unknown_payload_joint"/>
      <geom name="payload_core" type="sphere" size="0.04"/>
      <geom name="payload_tail" type="box" pos="0 0.06 0" size="0.02 0.04 0.02"/>
    </body>
    <body name="fridge_door" pos="0 0 0">
      <joint name="fridge_hinge" type="hinge" axis="0 0 1"/>
      <geom name="door_handle" type="sphere" size="0.04"/>
    </body>
    <body name="drawer" pos="1 0 0">
      <joint name="drawer_slide" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
    <body name="complex_payload" pos="2 0 0">
      <freejoint name="complex_payload_joint"/>
      <geom type="sphere" pos="-0.45 0 0" size="0.03"/>
      <geom type="sphere" pos="-0.35 0 0" size="0.03"/>
      <geom type="sphere" pos="-0.25 0 0" size="0.03"/>
      <geom type="sphere" pos="-0.15 0 0" size="0.03"/>
      <geom type="sphere" pos="-0.05 0 0" size="0.03"/>
      <geom type="sphere" pos="0.05 0 0" size="0.03"/>
      <geom type="sphere" pos="0.15 0 0" size="0.03"/>
      <geom type="sphere" pos="0.25 0 0" size="0.03"/>
      <geom type="sphere" pos="0.35 0 0" size="0.03"/>
      <geom type="sphere" pos="0.45 0 0" size="0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


def _description(*, logical_finger_pair: bool = False) -> RobotDescription:
    gripper_joints = [
        JointSpec(
            name="left_finger",
            joint_type=JointType.PRISMATIC,
            parent_link="panda_link7",
            child_link="left_finger",
            sim_joint_name="gripper0_left_joint",
            role="gripper",
        )
    ]
    if not logical_finger_pair:
        gripper_joints.append(
            JointSpec(
                name="right_finger",
                joint_type=JointType.PRISMATIC,
                parent_link="panda_link7",
                child_link="right_finger",
                sim_joint_name="gripper0_right_joint",
                role="gripper",
            )
        )
    return RobotDescription(
        name="attachment_test_robot",
        embodiment_kind="manipulator",
        joints=[
            JointSpec(
                name="joint7",
                joint_type=JointType.REVOLUTE,
                parent_link="base_link",
                child_link="panda_link7",
                sim_joint_name="robot0_joint7",
                role="arm",
            ),
            *gripper_joints,
        ],
        end_effectors=[
            EndEffectorSpec(
                name="parallel_gripper",
                kind="parallel_gripper",
                parent_link="panda_link7",
                actuated=True,
            )
        ],
        capabilities=RobotCapabilities(),
        safety=SafetyEnvelope(),
    )


@pytest.fixture
def sim() -> tuple[object, object]:
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _body_id(model: object, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def test_classifies_free_hinge_and_slide_bodies(sim: tuple[object, object]) -> None:
    model, _data = sim

    assert (
        classify_body_mobility(model, _body_id(model, "unknown_payload")) is SimObjectMobility.FREE
    )
    assert classify_body_mobility(model, _body_id(model, "fridge_door")) is SimObjectMobility.HINGE
    assert classify_body_mobility(model, _body_id(model, "drawer")) is SimObjectMobility.SLIDE


def test_extracts_multiple_runtime_primitives(sim: tuple[object, object]) -> None:
    model, data = sim
    primitives = extract_body_primitives(
        model,
        data,
        root_body_id=_body_id(model, "unknown_payload"),
        object_id="sim:unknown_payload",
    )

    assert len(primitives) == 2
    assert all(
        primitive.pose_in_object.frame_id == "sim:unknown_payload" for primitive in primitives
    )


def test_reduces_many_runtime_geoms_to_bounded_conservative_boxes(
    sim: tuple[object, object],
) -> None:
    model, data = sim
    primitives = extract_body_primitives(
        model,
        data,
        root_body_id=_body_id(model, "complex_payload"),
        object_id="sim:complex_payload",
        max_primitives=8,
    )

    assert len(primitives) == 8
    assert all(isinstance(primitive.shape, BoxShape) for primitive in primitives)
    covered_lower = min(
        primitive.pose_in_object.xyz[0] - primitive.shape.half_extents_m[0]
        for primitive in primitives
        if isinstance(primitive.shape, BoxShape)
    )
    covered_upper = max(
        primitive.pose_in_object.xyz[0] + primitive.shape.half_extents_m[0]
        for primitive in primitives
        if isinstance(primitive.shape, BoxShape)
    )
    assert covered_lower <= -0.48
    assert covered_upper >= 0.48


def test_sustained_two_finger_contact_attaches_then_releases(
    sim: tuple[object, object],
) -> None:
    model, data = sim
    tracker = SimAttachmentEvidenceTracker(model, _description())

    assert tracker.update(data, stamp_ns=1) is None
    assert tracker.update(data, stamp_ns=2) is None
    attached = tracker.update(data, stamp_ns=3)
    assert attached is not None
    assert len(attached) == 1
    assert attached[0].object_id == "sim:unknown_payload"
    assert len(attached[0].primitives) == 2
    assert attached[0].touch_links == ["left_finger", "right_finger"]

    free_joint = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "unknown_payload_joint",
    )
    # Contact loss alone is insufficient while the object remains rigidly
    # coupled to the hand frame.
    for joint_name in ("gripper0_left_joint", "gripper0_right_joint"):
        finger_joint = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        data.qpos[int(model.jnt_qposadr[finger_joint])] = 0.05
    mujoco.mj_forward(model, data)
    assert tracker.update(data, stamp_ns=4) is None

    data.qpos[int(model.jnt_qposadr[free_joint])] = 1.0
    mujoco.mj_forward(model, data)
    assert tracker.update(data, stamp_ns=5) is None
    assert tracker.update(data, stamp_ns=6) is None
    assert tracker.update(data, stamp_ns=7) == []


def test_logical_finger_pair_resolves_both_physical_jaws(
    sim: tuple[object, object],
) -> None:
    model, data = sim
    tracker = SimAttachmentEvidenceTracker(
        model,
        _description(logical_finger_pair=True),
        stable_ticks=1,
    )

    attached = tracker.update(data, stamp_ns=1)

    assert attached is not None
    assert [obj.object_id for obj in attached] == ["sim:unknown_payload"]
    assert attached[0].touch_links == ["left_finger"]


def test_articulated_handle_contact_never_becomes_payload(
    sim: tuple[object, object],
) -> None:
    model, data = sim
    payload_joint = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "unknown_payload_joint",
    )
    data.qpos[int(model.jnt_qposadr[payload_joint])] = 1.0
    mujoco.mj_forward(model, data)
    tracker = SimAttachmentEvidenceTracker(model, _description())

    for stamp in range(1, 8):
        assert tracker.update(data, stamp_ns=stamp) is None
