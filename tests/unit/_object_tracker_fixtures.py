"""Shared fixtures of the #418 object-tracker binding/release tests (real manifests)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from openral_core import (
    FitKind,
    ObjectRole,
    PlaceRegion,
    Pose6D,
    RobotDescription,
    TrackedObject,
    TrackStatus,
)
from openral_world_state.object_tracker.binding import GraspBinder
from openral_world_state.object_tracker.pinch import HandModel

CELL = 0.01
ROBOTS = {
    # robot dir: (gripper joint, arm pose, jaw at the hold: inside each manifest's stall band)
    "so101_follower": (
        "gripper",
        {
            "shoulder_pan": 0.2,
            "shoulder_lift": -0.3,
            "elbow_flex": 0.5,
            "wrist_flex": 1.0,
            "wrist_roll": 0.3,
        },
        0.28,
    ),
    "droid_franka_robotiq": (
        "finger_joint",
        {
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,
            "panda_joint5": 0.0,
            "panda_joint6": 1.571,
            "panda_joint7": 0.785,
        },
        0.45,
    ),
}


class DictStore:
    """``TrackStoreLike`` over a dict: the smallest real track store."""

    def __init__(self, tracks: list[TrackedObject]) -> None:
        self.tracks = {t.object_id: t for t in tracks}
        self.revision = 0

    def get(self, object_id: str) -> TrackedObject | None:
        return self.tracks.get(object_id)

    def all(self) -> list[TrackedObject]:
        return list(self.tracks.values())

    def replace(self, obj: TrackedObject) -> None:
        self.tracks[obj.object_id] = obj
        self.revision += 1


def track(
    oid: str,
    frame: str,
    xyz: NDArray[np.float64],
    half: tuple[float, float, float],
    status: TrackStatus = TrackStatus.CONFIRMED,
    quat: tuple[float, ...] = (0, 0, 0, 1),
    role: ObjectRole = ObjectRole.TARGET,
) -> TrackedObject:
    pose = Pose6D(xyz=tuple(float(v) for v in xyz), quat_xyzw=quat, frame_id=frame)
    return TrackedObject(
        object_id=oid,
        epoch=0,
        label="orange",
        role=role,
        declaration_key="obj:orange@1",
        status=status,
        box=PlaceRegion(frame_id=frame, pose=pose, half_extents=half),
        fit_kind=FitKind.SILHOUETTE_CONE,
        support_z=float(xyz[2]) - half[2] - 0.001,
        support_ref="map",
        floor_z=float(xyz[2]) - half[2],
        visible_cells=6,
        occupied_cells=5,
        last_fit_stamp_ns=1,
        last_confirmed_stamp_ns=1,
    )


def arm_at_hold(robot: str) -> tuple[GraspBinder, dict[str, float], NDArray[np.float64]]:
    d = RobotDescription.from_yaml(f"robots/{robot}/robot.yaml")
    joint, arm, jaw = ROBOTS[robot]
    binder = GraspBinder(
        HandModel(d, joint), frame_id=d.base_frame, cell_m=CELL, max_track_fit_age_s=2.0
    )
    positions = {**arm, joint: jaw}
    pinch = binder.hand.pinch_in_link(jaw)
    t = binder.hand.t_frame_from_attach(d.base_frame, positions)
    return binder, positions, t[:3, :3] @ pinch.centre + t[:3, 3]
