"""WorldStateAggregator.update_detected_objects (ADR-0035)."""

from __future__ import annotations

from openral_core import (
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_core.schemas import DetectedObject, Pose6D
from openral_world_state import WorldStateAggregator


def _desc():
    return RobotDescription(
        name="t",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name="j0",
                joint_type=JointType.REVOLUTE,
                parent_link="base_link",
                child_link="link_0",
            )
        ],
        capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        safety=SafetyEnvelope(),
    )


def _obj():
    return DetectedObject(
        label="cup",
        confidence=0.9,
        pose=Pose6D(xyz=(1.0, 2.0, 3.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        bbox_3d=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        track_id=0,
    )


def test_detected_objects_default_empty():
    agg = WorldStateAggregator(_desc())
    assert agg.snapshot().detected_objects == []


def test_update_detected_objects_surfaces_in_snapshot():
    agg = WorldStateAggregator(_desc())
    agg.update_detected_objects([_obj()])
    snap = agg.snapshot()
    assert len(snap.detected_objects) == 1
    assert snap.detected_objects[0].pose.xyz == (1.0, 2.0, 3.0)


def test_update_detected_objects_replaces_and_copies():
    agg = WorldStateAggregator(_desc())
    src = [_obj()]
    agg.update_detected_objects(src)
    src.clear()
    assert len(agg.snapshot().detected_objects) == 1
    agg.update_detected_objects([])
    assert agg.snapshot().detected_objects == []
