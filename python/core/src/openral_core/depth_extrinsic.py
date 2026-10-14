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
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Final, NamedTuple

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
    *,
    estimated_camera: str | None = None,
) -> list[str]:
    """Every cloud source whose mount a real world-voxel deploy may not trust (empty = go).

    ``description`` is the robot as the selected unit publishes it (its ``SensorOverlay``
    s already applied); ``overlays`` are those overlays. A robot-mounted cloud source
    (``SensorSpec.is_cloud_source``: a depth camera, a 3D lidar) must have its
    ``static_transform_xyz_rpy`` set by one of them: every manifest ships a nominal mount
    from CAD, so a value being present proves nothing, while a unit overlay only exists
    because someone set that cell up and measured it. A scene-mounted cloud source is
    declared per deployment already; it must carry ``parent_frame`` and
    ``static_transform_xyz_rpy``. ``estimated_camera`` (``DeployRuntime.
    estimated_depth_camera``) is an RGB camera that feeds octomap DA3's cloud, so its
    mount (its own, or the one it ``shares_mount_with``) is a cloud source's here too.

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
    if estimated_camera is not None:
        # The camera sits at its own mount or the one it shares; that mount is checked.
        spec = next((s for s in description.sensors if s.name == estimated_camera), None)
        if spec is not None and spec.shares_mount_with:
            estimated_camera = spec.shares_mount_with
    problems = []
    for spec in description.sensors:
        if not spec.is_cloud_source and spec.name != estimated_camera:
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
        if spec.name in robot_names or not (spec.is_cloud_source or spec.name == estimated_camera):
            continue
        if spec.parent_frame is None or spec.static_transform_xyz_rpy is None:
            problems.append(
                f"{spec.name}: a scene {spec.modality} sensor can feed octomap, but the scene "
                "declares no parent_frame + static_transform_xyz_rpy for it"
            )
    return problems


class EstimatedDepthMount(NamedTuple):
    """Where ``DeployRuntime.estimated_depth_camera``'s DA3 cloud is placed, or why it cannot be.

    Attributes:
        cloud_frame: The REP-103 optical frame the depth, ``CameraInfo`` and cloud are
            stamped in (``""`` when ``problems``).
        optical_parent: The body-convention mount frame the launch must hang
            ``cloud_frame`` under (``BODY_TO_OPTICAL``), or ``None`` when ``cloud_frame``
            is already on TF (a declared optical mount, or the sim bridge's live one).
        mount_sensor: The sensor whose mount places the camera (itself, or the one it
            ``shares_mount_with``); a real deploy's extrinsic gate checks that one.
        problems: Why the camera cannot feed the world map (empty = go).
    """

    cloud_frame: str
    optical_parent: str | None
    mount_sensor: str
    problems: list[str]


def resolve_estimated_depth_camera(
    sensors: Sequence[SensorSpec],
    camera: str,
    *,
    hal_mode: str,
    scene_sensor_names: Collection[str] = (),
) -> EstimatedDepthMount:
    """Resolve ``camera``'s mount for DA3 world-collision depth: any RGB camera tied to the robot.

    A monocular cloud is only useful where TF can place it, so the camera (top, context
    or wrist alike) must be *extrinsically linked to the robot*:

    1. its mount is declared — ``parent_frame`` + ``static_transform_xyz_rpy`` on the
       camera, or on the sensor it ``shares_mount_with`` (the same physical device, e.g.
       a ZED's colour stream and its depth). The launch publishes that mount; a
       body-convention mount frame gets its ``<frame>_optical_frame`` child. A wrist
       camera's parent is a robot link, so TF moves it with the arm. Works in sim and
       on real hardware (where the mount must also be unit-calibrated,
       :func:`depth_extrinsic_problems`).
    2. in sim only, its ``frame_id`` is already an ``*_optical_frame`` — the sim bridge
       publishes that camera's live pose from the MJCF.

    ``world`` (a camera whose pose relative to the robot nobody declared) and a bare
    robot link (``panda_hand``: the camera is not at the link's origin) are refused.
    The camera must also be RGB (a depth sensor publishes its own depth) with
    ``intrinsics`` (the real sensor leg publishes ``CameraInfo`` only then).
    ``sensors`` are the deploy's manifest (unit-overlaid) + scene sensors.

    An estimate never replaces a measurement: when any of ``sensors``
    :attr:`~openral_core.schemas.SensorSpec.is_cloud_source` (a depth camera, a 3D lidar),
    the camera is refused and the world map must use that sensor's cloud
    (``octomap_cloud_topic``). There is no override (Safety-WG, hazard HZ-0382-6).

    Example:
        >>> from openral_core import RobotDescription
        >>> from openral_core.schemas import SensorSpec
        >>> arm = RobotDescription.from_yaml("robots/openarm/robot.yaml").sensors
        >>> (refusal,) = resolve_estimated_depth_camera(arm, "top", hal_mode="real").problems
        >>> refusal.startswith("top: REFUSED"), "robot sensor 'head_zed' (depth)" in refusal
        (True, True)
        >>> k = arm[1].intrinsics  # top's K; an RGB camera riding an RGB sensor's mount:
        >>> rig = [
        ...     SensorSpec(
        ...         name="color",
        ...         modality="rgb",
        ...         frame_id="cam_link",
        ...         rate_hz=30.0,
        ...         intrinsics=k,
        ...         shares_mount_with="rig",
        ...     ),
        ...     SensorSpec(
        ...         name="rig",
        ...         modality="rgb",
        ...         frame_id="cam_link",
        ...         rate_hz=30.0,
        ...         intrinsics=k,
        ...         parent_frame="openarm_base",
        ...         static_transform_xyz_rpy=(0.1, 0.0, 0.6, 0.0, 0.5, 0.0),
        ...     ),
        ... ]
        >>> resolve_estimated_depth_camera(rig, "color", hal_mode="real")[:3]
        ('cam_link_optical_frame', 'cam_link', 'rig')
        >>> resolve_estimated_depth_camera(rig, "color", hal_mode="real").problems
        []
        >>> mobile = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml").sensors
        >>> rgb_only = [s for s in mobile if not s.is_cloud_source]  # drop front_depth
        >>> resolve_estimated_depth_camera(rgb_only, "shoulder_left", hal_mode="sim")[:3]
        ('shoulder_left_optical_frame', None, 'shoulder_left')
        >>> real = resolve_estimated_depth_camera(rgb_only, "shoulder_left", hal_mode="real")
        >>> "no mount ties it to the robot" in real.problems[0]
        True
    """
    by_name = {s.name: s for s in sensors}
    spec = by_name.get(camera)
    if spec is None:
        return EstimatedDepthMount("", None, camera, [f"{camera}: no such sensor"])
    if spec.modality != "rgb":
        what = (
            "a depth sensor, which publishes its own depth"
            if spec.is_cloud_source
            else f"a {spec.modality} sensor"
        )
        return EstimatedDepthMount("", None, camera, [f"{camera}: {what}; name an RGB camera"])
    problems = []
    measured = [
        f"{'scene' if s.name in scene_sensor_names else 'robot'} sensor {s.name!r} ({s.modality})"
        for s in sensors
        if s.is_cloud_source
    ]
    if measured:
        # Safety-WG (hazard Entry 059, HZ-0382-6): an estimate never replaces a measurement,
        # in sim or on real hardware, and no flag overrides it.
        problems.append(
            f"{camera}: REFUSED — estimated depth never replaces a measured depth or "
            f"point-cloud sensor, and this deploy has {', '.join(measured)}. Map that "
            "sensor's cloud (runtime.octomap_cloud_topic) or remove "
            "runtime.estimated_depth_camera; there is no override"
        )
    if spec.intrinsics is None:
        problems.append(f"{camera}: no intrinsics, so no CameraInfo gives DA3 its focal")
    mount = by_name.get(spec.shares_mount_with or camera, spec)
    if mount.parent_frame is not None and mount.static_transform_xyz_rpy is not None:
        frame = str(mount.frame_id)
        optical = frame.endswith("_optical_frame")
        return EstimatedDepthMount(
            frame if optical else f"{frame}_optical_frame",
            None if optical else frame,
            mount.name,
            problems,
        )
    if hal_mode == "sim" and str(spec.frame_id).endswith("_optical_frame"):
        return EstimatedDepthMount(str(spec.frame_id), None, camera, problems)
    problems.append(
        f"{camera}: no mount ties it to the robot: declare parent_frame + "
        "static_transform_xyz_rpy (or shares_mount_with) for it"
    )
    return EstimatedDepthMount("", None, mount.name, problems)
