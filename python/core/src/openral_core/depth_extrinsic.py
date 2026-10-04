"""The depth-camera extrinsic a real world-voxel deploy runs on: what it must be, who declares it.

The kernel's world-voxel check places every obstacle through the depth camera's
``parent_frame -> frame_id`` mount, absolutely. OpenRAL does not measure that mount: the
operator calibrates it with the tool of their choice (a hand-eye calibration against a
printed board is the usual one) and declares the result as ``static_transform_xyz_rpy`` in
the robot unit's overlay, ``robots/<id>/units/<unit>.yaml``. This module states the accuracy
such a calibration needs (``MAX_HEIGHT_ERR_M`` .. ``MAX_YAW_DEG``, derived from the real
world-voxel margin) and
:func:`depth_extrinsic_problems`, the one gate ``openral deploy run`` applies before a real
launch with the world-voxel check on: every cloud source must have a declared mount, and a
robot-mounted one must come from the unit's overlay, never from the manifest's nominal value.

Kept out of ``openral_core.__init__`` and free of numpy, so the launch file can read
:data:`REAL_WORLD_VOXEL_MARGIN_M` without loading anything heavy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from openral_core.schemas import RobotDescription, SensorOverlay, SensorSpec

#: Clearance the kernel's world-voxel check keeps around the robot on a real deploy
#: (``world_voxel_margin_m``; sim runs at 0). The launch passes this value, and the accuracy
#: an extrinsic needs below is derived from it, so tightening the margin tightens that too.
REAL_WORLD_VOXEL_MARGIN_M: Final[float] = 0.02
#: Range at which an angular error is converted to a position error. A head camera sees
#: its workspace within 1 m; a closer (wrist) camera only makes the same angle smaller.
CHECKED_RANGE_M: Final[float] = 1.0
#: Height error and the tilt error at ``CHECKED_RANGE_M`` each get half the margin, so
#: together they never exceed it.
MAX_HEIGHT_ERR_M: Final[float] = REAL_WORLD_VOXEL_MARGIN_M / 2.0
MAX_TILT_DEG: Final[float] = math.degrees(math.atan(MAX_HEIGHT_ERR_M / CHECKED_RANGE_M))
#: Planar (x, y) error of the camera, and the yaw error that moves a point as far at
#: ``CHECKED_RANGE_M``.
MAX_PLANAR_ERR_M: Final[float] = 0.75 * REAL_WORLD_VOXEL_MARGIN_M
MAX_YAW_DEG: Final[float] = math.degrees(math.atan(MAX_PLANAR_ERR_M / CHECKED_RANGE_M))


def depth_extrinsic_problems(
    description: RobotDescription,
    overlays: Sequence[SensorOverlay],
    scene_sensors: Sequence[SensorSpec] = (),
) -> list[str]:
    """Every cloud source whose mount a real world-voxel deploy may not trust (empty = go).

    ``description`` is the robot as the selected unit publishes it (its ``SensorOverlay``
    s already applied); ``overlays`` are those overlays. A robot-mounted cloud source
    (``SensorSpec.is_cloud_source``: a depth camera, a 3D lidar) must have its
    ``static_transform_xyz_rpy`` set by one of them: every manifest ships a nominal mount
    from CAD, so a value being present proves nothing, while a unit overlay only exists
    because someone set that cell up and measured it. A scene-mounted cloud source is
    declared per deployment already; it must carry ``parent_frame`` and
    ``static_transform_xyz_rpy``.

    Example:
        >>> from openral_core import RobotDescription, apply_sensor_overlays, load_robot_unit
        >>> robot = "robots/openarm/robot.yaml"
        >>> desc = RobotDescription.from_yaml(robot)
        >>> depth_extrinsic_problems(desc, [])[0]
        "head_zed: mount is the manifest's nominal value, not declared in units/<unit>.yaml"
        >>> thor = load_robot_unit(robot, "thor").sensors
        >>> unit = desc.model_copy(update={"sensors": apply_sensor_overlays(desc.sensors, thor)})
        >>> depth_extrinsic_problems(unit, thor)
        []
    """
    declared = {o.name for o in overlays if o.static_transform_xyz_rpy is not None}
    problems = []
    for spec in description.sensors:
        if not spec.is_cloud_source:
            continue
        if spec.parent_frame is None or spec.static_transform_xyz_rpy is None:
            problems.append(
                f"{spec.name}: no parent_frame + static_transform_xyz_rpy declares where this "
                "cloud source is mounted"
            )
        elif spec.name not in declared:
            problems.append(
                f"{spec.name}: mount is the manifest's nominal value, not declared in "
                "units/<unit>.yaml"
            )
    robot_names = {s.name for s in description.sensors}
    for spec in scene_sensors:
        if spec.name in robot_names or not spec.is_cloud_source:
            continue
        if spec.parent_frame is None or spec.static_transform_xyz_rpy is None:
            problems.append(
                f"{spec.name}: a scene {spec.modality} sensor can feed octomap, but the scene "
                "declares no parent_frame + static_transform_xyz_rpy for it"
            )
    return problems
