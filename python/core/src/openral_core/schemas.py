"""openral schema v0 — normative Pydantic v2 contracts for all layers.

This is the single source of truth for the data contracts between layers.
Anything imported from openral_core.__init__ is public API.
Breaking changes require a SemVer MAJOR bump (pre-1.0: MINOR) and a migration entry.
"""

from __future__ import annotations

import base64
import binascii
import math
from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

# ROSConfigError lives in the sibling `exceptions` module and is the
# canonical exception family for any configuration-level failure (see
# CLAUDE.md §10).
from openral_core.exceptions import ROSConfigError

# ─── Enums ─────────────────────────────────────────────────────────────────────


class EmbodimentKind(str, Enum):
    """Top-level kinematic class of a robot body."""

    HUMANOID = "humanoid"
    MANIPULATOR = "manipulator"
    BIMANUAL = "bimanual"
    QUADRUPED = "quadruped"
    MOBILE_BASE = "mobile_base"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    DRONE = "drone"


class JointType(str, Enum):
    """Kinematic joint type following URDF conventions."""

    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    CONTINUOUS = "continuous"
    FIXED = "fixed"
    FLOATING = "floating"
    PLANAR = "planar"


JointRole: TypeAlias = Literal[
    "arm",
    "base",
    "gripper",
    "torso",
    "leg",
    "head",
    "neck",
    "wheel",
    "unknown",
]
"""Structural classification of a :class:`JointSpec` (ADR-0028a).

Carries the joint's *purpose* in the embodiment's morphology — what
the runner, safety kernel, and dataset bridge need to identify a
channel without relying on name-substring heuristics (e.g.
``"gripper" in name.lower()`` in ``rskill_runner_node._build_joint_permutation``
which silently misclassifies any joint with ``"gripper"`` in the name).

``"unknown"`` is the default so legacy manifests load unchanged; the
fleet annotates incrementally as ADR-0028 sub-PRs land.
"""


class ControlMode(str, Enum):
    """Action space / control interface exposed by a robot or skill."""

    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    JOINT_TRAJECTORY = "joint_trajectory"
    CARTESIAN_POSE = "cartesian_pose"  # 6D EE pose absolute
    CARTESIAN_DELTA = "cartesian_delta"  # 6D EE delta
    CARTESIAN_TWIST = "cartesian_twist"  # 6D velocity
    BODY_TWIST = "body_twist"  # base linear/angular velocity
    FOOT_PLACEMENT = "foot_placement"  # discrete footsteps
    GRIPPER_BINARY = "gripper_binary"
    GRIPPER_POSITION = "gripper_position"
    DEX_HAND_JOINT = "dex_hand_joint"  # multi-DoF fingers
    # ADR-0028d — sim-only robosuite-composite multiplexer flag (e.g.
    # ``HybridMobileBase.set_goal`` reads ``action[-1]`` to switch the
    # arm controller between "achieved" (frozen) and "desired"
    # (responds to delta) modes). 1-D value in ``[-1, +1]``. Real-HW
    # adapters with independent arm + base controllers ignore this.
    COMPOSITE_MODE = "composite_mode"


# Single source of truth for the `openral_msgs/ActionChunk.control_mode`
# uint8 wire encoding. Both producers (`openral_runner.ros_publishing_hal`)
# and consumers (`openral_hal.lifecycle.decode_action_chunk`, the base
# `_on_safe_action` decoder every robot now shares) import this so a
# wire-format change happens in one place. The order
# matches the enum declaration above and the C++ kernel's
# ``cpp/openral_safety_kernel/include/openral_safety_kernel/validator.hpp::ControlMode``.
CONTROL_MODE_TO_UINT8: dict[ControlMode, int] = {
    ControlMode.JOINT_POSITION: 0,
    ControlMode.JOINT_VELOCITY: 1,
    ControlMode.JOINT_TORQUE: 2,
    ControlMode.JOINT_TRAJECTORY: 3,
    ControlMode.CARTESIAN_POSE: 4,
    ControlMode.CARTESIAN_DELTA: 5,
    ControlMode.CARTESIAN_TWIST: 6,
    ControlMode.BODY_TWIST: 7,
    ControlMode.FOOT_PLACEMENT: 8,
    ControlMode.GRIPPER_BINARY: 9,
    ControlMode.GRIPPER_POSITION: 10,
    ControlMode.DEX_HAND_JOINT: 11,
    ControlMode.COMPOSITE_MODE: 12,
}

# Inverse mapping for consumers that decode `ActionChunk.control_mode`
# back to the ControlMode enum.
UINT8_TO_CONTROL_MODE: dict[int, ControlMode] = {v: k for k, v in CONTROL_MODE_TO_UINT8.items()}

# Width of a BODY_TWIST / CARTESIAN_* twist row: (vx, vy, vz, wx, wy, wz).
# Single source for the HAL packers + safety supervisor that validate
# 6-vec twist payloads; matches ``Action.body_twist`` (a 6-tuple).
BODY_TWIST_DIM = 6

# Obstacle-clearance buffer added to a circular base's ``footprint_radius``
# to derive the Nav2 costmap ``inflation_radius`` (see
# ``RobotDescription.nav2_param_overrides``). Nav2 requires
# ``inflation_radius >= the costmap-discretised circumscribed radius``,
# which exceeds ``footprint_radius`` by up to one cell; a small buffer
# clears that and gives a thin obstacle-avoidance halo without blocking
# tight aisles. 0.05 m = one cell at the default 0.05 m costmap
# resolution.
NAV2_INFLATION_CLEARANCE_M = 0.05


class SensorModality(str, Enum):
    """Physical sensing modality."""

    RGB = "rgb"
    DEPTH = "depth"
    STEREO = "stereo"
    IR = "ir"
    POINT_CLOUD = "point_cloud"
    LIDAR_2D = "lidar_2d"
    IMU = "imu"
    FORCE_TORQUE = "force_torque"
    JOINT_STATE = "joint_state"
    TACTILE_VISION = "tactile_vision"
    TACTILE_ARRAY = "tactile_array"
    AUDIO = "audio"
    GPS = "gps"
    BATTERY = "battery"


class Hand(str, Enum):
    """Laterality of an end-effector."""

    LEFT = "left"
    RIGHT = "right"
    NA = "na"


# ─── Sensors ───────────────────────────────────────────────────────────────────


class IntrinsicsPinhole(BaseModel):
    """Pinhole camera intrinsics.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Focal length in x (pixels).
        fy: Focal length in y (pixels).
        cx: Principal point x (pixels).
        cy: Principal point y (pixels).
        distortion_model: Distortion model name.
        distortion_coeffs: Distortion coefficients.

    Example:
        >>> IntrinsicsPinhole(width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0)
        IntrinsicsPinhole(width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0, ...)
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: Literal["plumb_bob", "equidistant", "none"] = "plumb_bob"
    distortion_coeffs: list[float] = Field(default_factory=list)


def scale_intrinsics_to(base: IntrinsicsPinhole, width: int, height: int) -> IntrinsicsPinhole:
    """Linearly rescale pinhole intrinsics to a new render resolution.

    For a camera with a *fixed* field of view, ``fx``/``fy``/``cx``/``cy`` scale
    linearly with the image dimensions: a frame rendered at twice the width has
    twice the focal length (px) and twice the principal-point x. This is the
    consistency rule deploy-sim needs — the canonical manifest pins one nominal
    resolution, but a scene may render the same MuJoCo camera at another
    (``scene.observation_width``/``height``). Publishing the manifest's nominal
    intrinsics on a different-resolution render would back-project depth pixels
    and project occupancy voxels with the wrong focal length, corrupting the
    OctoMap voxels and the 2D→3D object lift. Scaling keeps the published camera
    model matched to whatever was actually rendered.

    The distortion model and coefficients are preserved unchanged (coefficients
    are resolution-independent for the normalised plumb-bob model). When the
    target already equals ``base``'s resolution the input is returned as-is.

    Args:
        base: Nominal pinhole intrinsics (e.g. from a ``SensorSpec``).
        width: Target render width in pixels (``> 0``).
        height: Target render height in pixels (``> 0``).

    Returns:
        Intrinsics scaled so ``(fx, fy, cx, cy)`` correspond to ``(width,
        height)`` at the same field of view as ``base``.

    Raises:
        ValueError: If ``width`` or ``height`` is not strictly positive.

    Example:
        >>> base = IntrinsicsPinhole(width=256, height=256, fx=256.0, fy=256.0, cx=128.0, cy=128.0)
        >>> hi = scale_intrinsics_to(base, 640, 640)
        >>> (hi.width, hi.fx, hi.cx)
        (640, 640.0, 320.0)
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"target resolution must be positive; got ({width}, {height})")
    if width == base.width and height == base.height:
        return base
    sx = width / base.width
    sy = height / base.height
    return IntrinsicsPinhole(
        width=width,
        height=height,
        fx=base.fx * sx,
        fy=base.fy * sy,
        cx=base.cx * sx,
        cy=base.cy * sy,
        distortion_model=base.distortion_model,
        distortion_coeffs=list(base.distortion_coeffs),
    )


class SensorSpec(BaseModel):
    """Generalizable sensor descriptor — covers all modalities.

    Attributes:
        name: Human-readable sensor name, e.g. "head_rgb", "left_wrist_depth".
        modality: Physical sensing modality.
        frame_id: tf2 frame name.
        parent_frame: tf2 parent frame (for static transform).
        static_transform_xyz_rpy: Static transform from parent to this sensor.
        rate_hz: Expected publishing rate in Hz.
        intrinsics: Pinhole camera intrinsics (if applicable).
        encoding: Image encoding, e.g. "rgb8", "16UC1".
        vla_feature_key: VLA observation dict key this sensor maps to, e.g.
            'observation.images.camera1'. Used by skill loaders to auto-wire
            sensors to VLA input_features.
        ros2_topic: ROS 2 topic name. None for non-ROS robots (USB, sim-only).
        ros2_msg_type: ROS 2 message type, e.g. "sensor_msgs/Image". None for
            non-ROS robots (USB, sim-only).
        qos_profile: QoS profile key.
        vendor: Sensor vendor name.
        model: Sensor model, e.g. "RealSense D455".
        driver_pkg: ROS 2 driver package name.
        metadata: Additional key-value metadata.
    """

    model_config = ConfigDict(use_enum_values=True)

    name: str
    modality: SensorModality
    frame_id: str
    parent_frame: str | None = None
    static_transform_xyz_rpy: tuple[float, float, float, float, float, float] | None = None
    rate_hz: float
    # Image / depth / stereo
    intrinsics: IntrinsicsPinhole | None = None
    encoding: str | None = None
    fov_h_deg: float | None = None
    fov_v_deg: float | None = None
    # Optional override carrying the MJCF/MuJoCo camera name when it differs
    # from this sensor's logical ``name`` (issue #191 Phase 3b — mirrors
    # ``JointSpec.sim_joint_name``). ``MujocoArmHAL.read_images`` renders
    # ``sim_camera_name or name`` for each RGB sensor, keying the frame by the
    # sensor ``name`` so ``SimSensorBridge`` finds it. ``None`` = "MJCF camera
    # name matches ``name``" (the common case). E.g. openarm's ``base`` sensor
    # renders the MJCF ``top`` camera.
    sim_camera_name: str | None = None
    # LiDAR / point cloud
    n_channels: int | None = None
    range_min_m: float | None = None
    range_max_m: float | None = None
    # IMU
    accel_noise_density: float | None = None
    gyro_noise_density: float | None = None
    # F/T
    n_axes: int | None = None
    # Tactile
    tactile_grid: tuple[int, int] | None = None
    vla_feature_key: str | None = None
    # ROS 2 wiring
    ros2_topic: str | None = None
    ros2_msg_type: str | None = None
    qos_profile: Literal["sensor_data", "reliable", "transient_local", "parameters"] = "sensor_data"
    # Driver / vendor
    vendor: str | None = None
    model: str | None = None
    driver_pkg: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class SensorBundle(BaseModel):
    """Multi-modal sensor group, e.g. RealSense D455 = (RGB, depth, IMU).

    Attributes:
        bundle_name: Unique name for this bundle.
        sensors: List of constituent sensor specs.
        sync: Synchronization strategy.
        sync_tolerance_ms: Tolerance for approximate synchronization.
    """

    bundle_name: str
    sensors: list[SensorSpec]
    sync: Literal["hardware", "approximate", "none"] = "approximate"
    sync_tolerance_ms: float = 30.0


# ─── Joints / Actuation ────────────────────────────────────────────────────────


class JointSpec(BaseModel):
    """URDF-derived joint specification.

    Attributes:
        name: Joint name matching the URDF.
        joint_type: Kinematic joint type.
        parent_link: Parent link name.
        child_link: Child link name.
        axis_xyz: Rotation/translation axis unit vector.
        position_limits: (min, max) in radians or meters.
        velocity_limit: Maximum velocity.
        effort_limit: Maximum effort (N or Nm).
        has_position_sensor: Whether position feedback is available.
        has_velocity_sensor: Whether velocity feedback is available.
        has_torque_sensor: Whether torque feedback is available.
        backlash_estimate: Estimated backlash in radians.
        actuator_kind: Type of actuator.
        sim_joint_name: Optional override carrying the **MJCF/MuJoCo
            joint name** as it appears in the simulator's compiled
            model when that name differs from :attr:`name`. The HAL
            uses :attr:`name` for the world-state contract (it's the
            URDF-shaped logical identifier the safety supervisor sees
            on every chunk); a separate ``sim_joint_name`` lets a
            sim-adapter look up ``mj_name2id`` without hardcoding
            robosuite / robocasa / MuJoCo naming conventions in
            backend modules. ``None`` means "the MJCF joint name
            matches :attr:`name`" — true for every fixed-base
            manipulator we ship; only mobile bases and humanoids
            whose MJCF auto-prefixes joints (robosuite's
            ``mobilebase0_…`` namespace, GR-1's ``robot0_…``) need
            it. See ADR-0025.
        role: Structural classification (ADR-0028a). The downstream
            runner / safety / dataset-bridge code identifies grippers
            and base DoFs by this tag instead of substring-matching the
            joint name (which silently misclassifies any joint
            containing ``"gripper"``, e.g. ``"gripper_pose"``). Default
            ``"unknown"`` keeps legacy manifests loadable; the fleet
            annotates incrementally per ADR-0028a.
        origin_xyz: Fixed translation (metres) of this joint's frame in
            its ``parent_link`` frame — the URDF ``<joint><origin xyz>``.
            With :attr:`origin_rpy` and :attr:`axis_xyz` it gives the
            kernel everything it needs for forward kinematics (ADR-0030).
            Default ``(0, 0, 0)``; populated by the offline lowering tool
            (from MJCF/URDF) only for robots that enable self-collision
            checking — legacy manifests are unaffected.
        origin_rpy: Fixed orientation (roll, pitch, yaw, radians) of this
            joint's frame in its ``parent_link`` frame — the URDF
            ``<joint><origin rpy>``. Default ``(0, 0, 0)``. ADR-0030.
    """

    name: str
    joint_type: JointType
    parent_link: str
    child_link: str
    axis_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_limits: tuple[float, float] | None = None
    velocity_limit: float | None = None
    effort_limit: float | None = None
    has_position_sensor: bool = True
    has_velocity_sensor: bool = True
    has_torque_sensor: bool = False
    backlash_estimate: float | None = None
    actuator_kind: (
        Literal["dc", "bldc", "stepper", "servo", "tendon", "hydraulic", "pneumatic"] | None
    ) = None
    sim_joint_name: str | None = None
    role: JointRole = "unknown"


class EndEffectorSpec(BaseModel):
    """End-effector specification.

    Attributes:
        name: End-effector name.
        kind: Type of end-effector.
        hand: Laterality.
        n_dof: Number of controllable degrees of freedom.
        max_grip_force_n: Maximum grip force in Newtons.
        max_payload_kg: Maximum payload in kg.
        workspace_radius_m: Reach radius in meters.
        tactile_sensors: List of tactile SensorSpec names attached to this EE.
        actuated: Whether the end-effector is driven by an actuator
            (ADR-0028a). False for passive tools (inert flanges,
            magnetic plates without electromagnet, kinematic-only
            mounts). When False, the safety kernel rejects any
            chunk addressed at this EE — the chunk routes to a
            no-op rather than risking unintended motion. Default
            True: every actuated gripper / dexterous hand / suction
            cup we ship is driven.
    """

    name: str
    kind: Literal["parallel_gripper", "suction", "dexterous_hand", "tool", "none"]
    hand: Hand = Hand.NA
    n_dof: int = 1
    max_grip_force_n: float | None = None
    max_payload_kg: float | None = None
    workspace_radius_m: float | None = None
    tactile_sensors: list[str] = Field(default_factory=list)
    actuated: bool = True


# ─── Capabilities ──────────────────────────────────────────────────────────────

LocomotionKind: TypeAlias = Literal["bipedal", "quadruped", "wheeled", "tracked", "none"]


class RobotCapabilities(BaseModel):
    """Capability flags used for skill compatibility checking.

    Attributes:
        locomotion: Locomotion types available.
        can_lift_kg: Maximum payload in kg.
        has_dexterous_hands: Whether dexterous hands are present.
        has_tactile: Whether tactile sensing is available.
        has_force_control: Whether force/impedance control is supported.
        has_vision: Whether camera(s) are present.
        has_lidar: Whether LiDAR is present.
        has_audio: Whether audio I/O is present.
        bimanual: Whether the robot has two arms.
        onboard_compute_tops: Peak onboard compute in TOPS.
        onboard_memory_gb: Onboard RAM in GB.
        gpu_vram_gb: Largest single-GPU VRAM in GB (0 when no discrete GPU).
        cuda_compute_capability: Highest CUDA compute capability (major, minor),
            e.g. (8, 9) for Ada Lovelace, (10, 0) for Blackwell.  ``None`` on
            non-CUDA hosts.
        cuda_toolkit_version: ``nvcc`` version string when present.
        tensorrt_version: TensorRT runtime version string when importable.
        gpu_supported_runtimes: Inference runtimes the host can execute, used
            by ``rSkill.check_capabilities`` to match ``RSkillManifest.runtime``.
        gpu_supported_dtypes: Quantization dtypes the host accelerator
            supports (derived from CUDA compute capability or platform).
        supported_control_modes: List of supported ControlMode values.
        supported_vla_embodiments: VLA embodiment IDs this robot can run.
        embodiment_tags: Short tags mapping to VLA heads / dataset splits.
        nvmm_available: Whether ``libnvbufsurface.so`` is present on the host
            (Tegra L4T multimedia stack).  ``True`` enables the NVMM
            zero-copy sensor-ingest path on Jetson; always ``False`` on x86
            and on stripped L4T images.  Populated by
            :func:`openral_detect.probes.gpu._probe_nvmm_available`.
            Added in v0.5 per ADR-0013.
    """

    locomotion: list[LocomotionKind] = Field(
        default_factory=lambda: ["none"]  # type: ignore[arg-type]  # reason: ["none"] satisfies LocomotionKind at runtime; cast would be noisier
    )
    can_lift_kg: float = 0.0
    has_dexterous_hands: bool = False
    has_tactile: bool = False
    has_force_control: bool = False
    has_vision: bool = True
    has_lidar: bool = False
    has_audio: bool = False
    bimanual: bool = False
    onboard_compute_tops: float = 0.0
    onboard_memory_gb: float = 0.0
    gpu_vram_gb: float = 0.0
    cuda_compute_capability: tuple[int, int] | None = None
    cuda_toolkit_version: str | None = None
    tensorrt_version: str | None = None
    gpu_supported_runtimes: list[RSkillRuntime] = Field(default_factory=list)
    gpu_supported_dtypes: list[QuantizationDtype] = Field(default_factory=list)
    supported_control_modes: list[ControlMode] = Field(default_factory=list)
    supported_vla_embodiments: list[str] = Field(default_factory=list)
    embodiment_tags: list[str] = Field(default_factory=list)
    nvmm_available: bool = False


# ─── Safety ────────────────────────────────────────────────────────────────────


class SafetyEnvelope(BaseModel):
    """Safety constraints enforced by the C++ safety kernel.

    Attributes:
        workspace_box_min_xyz: Lower corner of allowed workspace (m).
        workspace_box_max_xyz: Upper corner of allowed workspace (m).
        no_go_zones: List of polygon definitions (dicts with 'vertices').
        max_ee_speed_m_s: Maximum end-effector linear speed in m/s.
            Also used by ADR-0028b's per-mode supervisor as the
            CARTESIAN_TWIST linear bound.
        max_ee_accel_m_s2: Maximum end-effector acceleration in m/s².
        max_joint_speed_factor: Fraction of joint velocity_limit allowed.
        max_force_n: Maximum contact force in Newtons.
        max_torque_nm: Maximum joint torque in Nm.
        deadman_required: Whether a deadman switch is required.
        e_stop_topic: ROS 2 topic for E-stop commands.
        e_stop_qos: QoS profile for E-stop topic.
        contact_force_threshold_n: Force threshold for contact detection.
        cycle_time_violation_threshold_ms: Control cycle time violation threshold.
        human_in_loop_required: rSkill names requiring human supervision.
        max_cartesian_step_m: ADR-0028b — per-step magnitude bound on
            CARTESIAN_DELTA's xyz triplet (Euclidean). ``None`` means
            "no per-mode check declared, skip"; today's behaviour
            preserved. Robots that host OSC-trained checkpoints
            (panda_mobile, future Franka + π0.7) declare this so the
            supervisor rejects out-of-distribution arm deltas before
            they reach the controller.
        max_cartesian_step_rad: ADR-0028b — per-step magnitude bound on
            CARTESIAN_DELTA's axis-angle triplet (Euclidean). ``None``
            skips the check.
        max_ee_angular_speed_rad_s: ADR-0028b — angular component bound
            for CARTESIAN_TWIST (the linear bound reuses
            :attr:`max_ee_speed_m_s`). ``None`` skips the check.
        max_base_linear_speed_m_s: ADR-0028b — BODY_TWIST linear bound
            (Euclidean over vx,vy,vz). ``None`` skips the check;
            mobile manipulators / wheeled bases declare it.
        max_base_angular_speed_rad_s: ADR-0028b — BODY_TWIST angular
            bound (Euclidean over wx,wy,wz; for planar bases only
            wz is non-zero). ``None`` skips the check.
    """

    workspace_box_min_xyz: tuple[float, float, float] | None = None
    workspace_box_max_xyz: tuple[float, float, float] | None = None
    no_go_zones: list[dict[str, object]] = Field(default_factory=list)
    max_ee_speed_m_s: float = 0.5
    max_ee_accel_m_s2: float = 1.0
    max_joint_speed_factor: float = 0.7
    max_force_n: float = 50.0
    max_torque_nm: float = 10.0
    deadman_required: bool = True
    e_stop_topic: str = "/safety/e_stop"
    e_stop_qos: Literal["reliable"] = "reliable"
    contact_force_threshold_n: float = 30.0
    cycle_time_violation_threshold_ms: float = 5.0
    human_in_loop_required: list[str] = Field(default_factory=list)
    # ADR-0028b — per-control-mode bounds for the supervisor dispatch.
    # All default to None so legacy behaviour is preserved: a robot
    # that doesn't declare these gets its chunks passed through the
    # per-mode check (cartesian / twist / gripper / etc.) verbatim, the
    # same as today.
    max_cartesian_step_m: float | None = None
    max_cartesian_step_rad: float | None = None
    max_ee_angular_speed_rad_s: float | None = None
    max_base_linear_speed_m_s: float | None = None
    max_base_angular_speed_rad_s: float | None = None


# ─── VLA observation / action specs ────────────────────────────────────────────


class StateRepresentation(str, Enum):
    """State vector representation format."""

    JOINT_POSITIONS = "joint_positions"
    EEF_POS_AXISANGLE = "eef_pos_axisangle"
    EEF_POS_EULER = "eef_pos_euler"
    EEF_POS_QUAT = "eef_pos_quat"
    EEF_POS_AXISANGLE_GRIPPER = "eef_pos_axisangle_gripper"


class ActionRepresentation(str, Enum):
    """Action vector representation format."""

    JOINT_POSITIONS = "joint_positions"
    JOINT_VELOCITIES = "joint_velocities"
    DELTA_EE_6D_PLUS_GRIPPER = "delta_ee_6d_plus_gripper"
    DELTA_EE_6D = "delta_ee_6d"
    CARTESIAN_POSE = "cartesian_pose"


class RSkillAction(str, Enum):
    """Closed vocabulary of high-level action verbs an rSkill can perform.

    Declared on :attr:`RSkillManifest.actions` so the reasoner's LLM tool
    palette can present each skill with a structured "what does it do"
    label, in addition to the free-form ``description`` and the slug.
    The LLM scores tools primarily on natural-language description, but
    a closed verb vocabulary lets the palette pre-filter and lets the
    schema be unit-testable. New entries are additive.

    Categories (descriptive only — not part of the wire format):

    - Manipulation primitives: ``PICK``, ``PLACE``, ``PICK_AND_PLACE``,
      ``TRANSFER``, ``GRASP``, ``RELEASE``.
    - Articulated / contact-rich: ``OPEN``, ``CLOSE``, ``PUSH``, ``PULL``,
      ``SLIDE``, ``INSERT``, ``POUR``, ``WIPE``, ``ROTATE``.
    - Motion: ``REACH``; ``LOOK`` (aim a camera at a point, ADR-0044).
    - Mobile (for mobile-manipulator embodiments): ``NAVIGATE``.
    - Social / expressive: ``WAVE``, ``SHAKE``.
    - Generalist marker: ``GENERALIST`` for foundation / multi-task
      checkpoints (e.g. RoboCasa-365, DROID, MetaWorld-MT50). The palette
      surfaces a generalist skill for goals that don't match a specific
      verb.
    """

    PICK = "pick"
    PLACE = "place"
    PICK_AND_PLACE = "pick_and_place"
    TRANSFER = "transfer"
    GRASP = "grasp"
    RELEASE = "release"
    OPEN = "open"
    CLOSE = "close"
    PUSH = "push"
    PULL = "pull"
    SLIDE = "slide"
    INSERT = "insert"
    POUR = "pour"
    WIPE = "wipe"
    ROTATE = "rotate"
    REACH = "reach"
    LOOK = "look"
    NAVIGATE = "navigate"
    WAVE = "wave"
    SHAKE = "shake"
    GENERALIST = "generalist"
    DETECT = "detect"
    QUERY = "query"
    MONITOR = "monitor"


class ObservationSpec(BaseModel):
    """VLA observation configuration for this robot.

    Attributes:
        state_key: Observation dict key for the state vector.
        state_shape: Shape of the state tensor, e.g. ``(6,)`` for 6-D EEF.
        state_representation: How the state vector is encoded.
        image_flip_180: Whether camera images need 180° rotation before
            feeding to the VLA (common for wrist-mounted cameras).
    """

    state_key: str = "observation.state"
    state_shape: tuple[int, ...] = ()
    state_representation: StateRepresentation | None = None
    image_flip_180: bool = False


class ActionSpec(BaseModel):
    """VLA action configuration for this robot.

    Attributes:
        dim: Dimensionality of the action vector.
        representation: How the action vector is encoded.
        control_freq_hz: Control frequency the actions are executed at.
        chunk_size: Number of action steps per inference call (chunk size H).
    """

    dim: int
    representation: ActionRepresentation | None = None
    control_freq_hz: float | None = None
    chunk_size: int | None = None


# ─── Robot description (top-level) ─────────────────────────────────────────────


class GripperReadMode(str, Enum):
    """How :class:`openral_hal.MujocoArmHAL` reports the gripper qpos.

    ``SUM_OVER_SCALE`` (default) — ``clip(sum(qpos[addrs]) / scale, 0, 1)``.
      Matches the Franka parallel gripper where two finger qpos are summed
      and divided by ``2 * max_finger_extent`` (=0.08 m for Panda).  Public
      surface is normalised ``[0, 1]``.
    ``AFFINE_LOW_HIGH`` — ``(qpos[addrs[0]] - ctrl_range[0]) / (ctrl_range[1] - ctrl_range[0])``.
      Used by the SO-100 menagerie ``Jaw`` joint, a 1-DoF revolute with a
      non-zero closed position (``-0.174`` rad).  Public surface is
      normalised ``[0, 1]``.
    ``PASSTHROUGH`` — ``qpos[addrs[0]]`` reported verbatim, in the same
      physical units as the MJCF (metres for Aloha prismatic fingers,
      radians for OpenArm revolute jaws).  The public surface is **not**
      ``[0, 1]`` — Skills must accept the raw range.
    """

    SUM_OVER_SCALE = "sum_over_scale"
    AFFINE_LOW_HIGH = "affine_low_high"
    PASSTHROUGH = "passthrough"


class GripperWriteMode(str, Enum):
    """How :class:`openral_hal.MujocoArmHAL` maps an Action's gripper value to ``ctrl``.

    ``NORMALISED`` (default) — input is ``[0, 1]``, mapped affinely to
      ``ctrl_range``.  ``low`` ↔ closed (Action.gripper = 0), ``high`` ↔
      open (Action.gripper = 1).
    ``PASSTHROUGH`` — input is in the same physical units as ``ctrl_range``
      (metres / radians), written directly to ``ctrl``.  MuJoCo's actuator
      ``ctrlrange`` does the clipping.  Used by Aloha (positive-finger
      metres) and OpenArm (jaw radians).
    """

    NORMALISED = "normalised"
    PASSTHROUGH = "passthrough"


class SimGripperDescription(BaseModel):
    """Gripper wiring inside a MuJoCo MJCF.

    Attributes:
        joint: Name of the gripper joint as it appears in
            :attr:`RobotDescription.joints` (the public, lerobot-style name —
            not the menagerie MJCF name).
        ctrl_range: ``(low, high)`` raw control range for the gripper
            actuator.  In ``NORMALISED`` write mode, ``low`` ↔ closed
            (Action.gripper = 0), ``high`` ↔ open (Action.gripper = 1).  In
            ``PASSTHROUGH`` write mode this is informational — MuJoCo clips
            on its own.
        qpos_addrs: ``qpos`` indices used to compute the reported gripper
            position.  When the gripper has multiple finger joints (Franka),
            list every finger's qpos index.
        qpos_scale: Span used to normalise the summed/raw gripper qpos to
            ``[0, 1]``.  For Franka with two fingers each in ``[0, 0.04]`` m,
            ``qpos_scale = 0.08``.  Ignored by ``AFFINE_LOW_HIGH`` and
            ``PASSTHROUGH`` read modes but kept for symmetry with the
            base-class invariant.
        read_mode: How to report the qpos — see :class:`GripperReadMode`.
        write_mode: How to map an Action's gripper value to ``ctrl`` — see
            :class:`GripperWriteMode`.
        actuator_index: Explicit MJCF actuator index that receives the
            (mapped) gripper command.  When omitted, defaults to the
            position of ``joint`` in :attr:`RobotDescription.joints`
            (i.e. the 1:1 mapping derived by ``SimDescription`` for arm
            joints).
        mirror_actuator_index: Optional second actuator that receives the
            **negation** of the (mapped) gripper command.  Models the
            Aloha parallel jaws where one motor drives ``+x`` and the
            other ``-x`` to keep the fingers symmetric.  ``None`` for
            single-actuator grippers.
    """

    model_config = ConfigDict(extra="forbid")

    joint: str
    ctrl_range: tuple[float, float]
    qpos_addrs: tuple[int, ...]
    qpos_scale: float
    read_mode: GripperReadMode = GripperReadMode.SUM_OVER_SCALE
    write_mode: GripperWriteMode = GripperWriteMode.NORMALISED
    actuator_index: int | None = None
    mirror_actuator_index: int | None = None


# ADR-0058 — the single description-asset ref grammar. The schema validator
# below only checks the ref *string* shape (cheap, no I/O); the resolver in
# ``openral_core.assets`` does the file resolution. Both must accept the same
# schemes, so keep these in lock-step with ``openral_core.assets``.
_ASSET_SCHEMES = ("rd:", "file:", "gym_aloha:", "openarm:", "menagerie:")
_ROS2_DYNAMIC = "ros2://robot_description"


def _validate_ref(v: str) -> str:
    """Reject any asset ref that is neither the dynamic marker nor a known scheme."""
    if v == _ROS2_DYNAMIC or v.startswith(_ASSET_SCHEMES):
        return v
    raise ValueError(
        f"asset ref {v!r} must start with one of {_ASSET_SCHEMES} or be {_ROS2_DYNAMIC!r}"
    )


class UrdfAsset(BaseModel):
    """A URDF asset reference plus its ``robot_state_publisher`` wiring.

    ADR-0027 / ADR-0058. The ``ref`` is resolved by
    :func:`openral_core.assets.resolve_asset`; ``root_frame`` and
    ``base_to_root_xyz_rpy`` carry the static transform that bridges a URDF
    whose root link differs from the robot's ``base_frame`` (e.g. Franka's
    ``panda_link0`` mounted onto a ``base_link`` mobile platform).

    Attributes:
        ref: Asset reference (``rd:<module>``, ``file:<relpath>``, or
            ``ros2://robot_description`` for runtime topic-supplied URDFs).
        root_frame: The URDF's root link name when it differs from
            :attr:`RobotDescription.base_frame`. ``None`` → the URDF root
            equals ``base_frame`` (no static transform needed).
        base_to_root_xyz_rpy: The 6-DoF transform ``[x, y, z, roll, pitch,
            yaw]`` (metres + radians) published via
            ``static_transform_publisher`` to bridge ``base_frame`` to
            :attr:`root_frame`. ``None`` when :attr:`root_frame` is ``None``.

    Example:
        >>> UrdfAsset(ref="rd:panda_description", root_frame="panda_link0").root_frame
        'panda_link0'
    """

    model_config = ConfigDict(extra="forbid")

    ref: str
    root_frame: str | None = None
    base_to_root_xyz_rpy: tuple[float, float, float, float, float, float] | None = None

    @field_validator("ref")
    @classmethod
    def _validate_urdf_ref(cls, v: str) -> str:
        return _validate_ref(v)


class AssetRefs(BaseModel):
    """The unified description-asset block on :class:`RobotDescription`.

    ADR-0058 §4. Replaces the scattered ``urdf_path`` / ``mjcf_uri`` /
    ``srdf_path`` (+ ADR-0027 URDF-root fields) with one block whose refs
    share the :func:`openral_core.assets.resolve_asset` grammar.

    Attributes:
        urdf: URDF asset (with optional ``robot_state_publisher`` wiring),
            or ``None`` when the robot ships no URDF.
        mjcf: MJCF asset ref (MuJoCo wiring), or ``None``.
        srdf: SRDF asset ref whose ``disable_collisions`` block seeds
            :attr:`RobotDescription.allowed_collision_pairs`, or ``None``.

    Example:
        >>> AssetRefs(urdf=UrdfAsset(ref="rd:panda_description")).urdf.ref
        'rd:panda_description'
    """

    model_config = ConfigDict(extra="forbid")

    urdf: UrdfAsset | None = None
    mjcf: str | None = None
    srdf: str | None = None

    @field_validator("mjcf", "srdf")
    @classmethod
    def _validate_optional_ref(cls, v: str | None) -> str | None:
        return None if v is None else _validate_ref(v)


class SimDescription(BaseModel):
    """MuJoCo wiring for a single-arm robot, consumed by ``MujocoArmHAL``.

    The MJCF itself is named by :attr:`RobotDescription.assets.mjcf`
    (ADR-0058); this block carries only the joint↔qpos/qvel/actuator
    plumbing. All fields are optional with defaults derived from
    :attr:`RobotDescription.joints`.  The default mapping is "1:1 in joint
    order, offset by 7 (qpos) / 6 (qvel) if ``floating_base`` is True" —
    which is correct for every robot in the open core today bar minor
    gripper bookkeeping.

    Attributes:
        floating_base: If True, the robot's MJCF has a 6-DoF free joint
            before the actuated joints (humanoids).  In that case the
            default ``joint_qpos_addr[joints[i].name] = 7 + i`` and
            ``joint_qvel_addr[joints[i].name] = 6 + i``.
        joint_qpos_addr: Optional override of the joint→qpos-index map.
            When omitted, the default 1:1 mapping is used.
        joint_qvel_addr: Optional override of the joint→qvel-index map.
            When omitted, defaults to ``joint_qpos_addr`` (or its default).
        actuator_index: Optional override of the joint→actuator-index map.
            When omitted, defaults to the 1:1 mapping ``joints[i].name → i``.
        grippers: Optional list of :class:`SimGripperDescription` entries.
            Each must reference a joint by name that is also present in
            :attr:`RobotDescription.joints`.  Single-arm robots have one
            entry (or none); bimanual robots have two (left + right).
        settle_steps_default: Default number of ``mj_step`` calls executed
            per :meth:`MujocoArmHAL.send_action`.  Defaults to 1.
        keyframe_index: When set, :meth:`MujocoArmHAL.connect` calls
            ``mj_resetDataKeyframe(model, data, keyframe_index)`` before
            ``mj_forward``.  Required for MJCFs whose default
            ``MjData.qpos`` (zeros) sits outside the actuator
            ``ctrlrange`` — e.g. the gym-aloha parallel jaws, whose
            fingers have ``ctrlrange=[0.021, 0.057]`` and never recover
            from qpos = 0.
        seed_ctrl_from_qpos: When True, :meth:`MujocoArmHAL.connect`
            seeds ``data.ctrl[actuator] = data.qpos[joint_qpos_addr]``
            for every controllable joint so position actuators hold the
            initial pose on the first ``mj_step``.  Required by the
            OpenArm v2 MJCF (its position actuators with per-class PD
            gains will drive ``qpos`` to ``ctrl == 0`` otherwise).

    Example:
        >>> SimDescription(floating_base=True).floating_base
        True
    """

    model_config = ConfigDict(extra="forbid")

    floating_base: bool = False
    joint_qpos_addr: dict[str, int] | None = None
    joint_qvel_addr: dict[str, int] | None = None
    actuator_index: dict[str, int] | None = None
    grippers: list[SimGripperDescription] = Field(default_factory=list)
    settle_steps_default: int = Field(default=1, ge=1)
    keyframe_index: int | None = None
    seed_ctrl_from_qpos: bool = False


class TopCameraDefaults(BaseModel):
    """Default placement for the scene-level "top" (a.k.a. "base") camera.

    Per-robot scene defaults consumed by sim backends that render an
    overview camera (today: the ``openarm_tabletop_pnp`` scene). The
    values describe a look-at camera pointed from ``pos`` toward
    ``target`` with vertical field-of-view ``fovy`` in degrees.

    Backend YAML overrides (``scene.backend_options.top_camera_*``) still
    win — this submodel is the *default* fed to the composer when no
    override is set, replacing the previous module-level constants in
    ``openral_sim.backends.openarm_robosuite._assets``.

    Attributes:
        pos: ``(x, y, z)`` world-frame camera position in metres.
        target: ``(x, y, z)`` world-frame look-at point in metres.
        fovy: Vertical field-of-view in degrees.

    Example:
        >>> TopCameraDefaults(pos=(0.2, 0.0, 0.95), target=(0.65, 0.0, 0.05), fovy=65.0).fovy
        65.0
    """

    model_config = ConfigDict(extra="forbid")

    pos: tuple[float, float, float]
    target: tuple[float, float, float]
    fovy: float = Field(gt=0.0, lt=180.0)


class SceneComposition(BaseModel):
    """Declarative MJCF scene composition for a manifest-driven HAL (ADR-0029).

    Lets a robot whose sim HAL needs a *composed* MJCF (a bare arm spliced onto
    a tabletop + props) declare the composer in its manifest instead of a
    bespoke ``_create_hal`` lifecycle subclass (issue #191 Phase 3b). The
    manifest-driven node calls :attr:`composer` before constructing the HAL and
    threads the resulting MJCF path in as the HAL's ``mjcf_path``.

    The composer is a ``"module.path:function"`` import string. The function is
    called with ``**params`` and MUST return ``(xml: str, meshdir: Path)`` — the
    composed MJCF XML and the mesh directory it references (the node writes the
    XML next to ``meshdir`` so relative mesh paths resolve). Today's only
    composer is ``openral_sim.backends.openarm_robosuite._assets:compose_openarm_tabletop_mjcf``.

    Attributes:
        composer: ``"module.path:function"`` returning ``(xml, meshdir)``.
        params: Keyword arguments for the composer (e.g. ``robot_lift_z``,
            ``white_background``).

    Example:
        >>> sc = SceneComposition(
        ...     composer="pkg.scenes:compose_tabletop",
        ...     params={"robot_lift_z": 0.36, "white_background": True},
        ... )
        >>> sc.params["robot_lift_z"]
        0.36
    """

    model_config = ConfigDict(extra="forbid")

    composer: str
    params: dict[str, object] = Field(default_factory=dict)


class SceneDefaults(BaseModel):
    """Per-robot scene rendering defaults.

    These are values that scene composers may consult when the
    :class:`SimScene` YAML does not override them. Today the
    only field is :attr:`top_camera`, which the
    ``openarm_tabletop_pnp`` backend consumes; future scenes can extend
    this submodel as new defaults are pulled out of backend hardcodes.

    Attributes:
        top_camera: Default placement for the scene-overview camera.

    Example:
        >>> sd = SceneDefaults(
        ...     top_camera=TopCameraDefaults(
        ...         pos=(0.2, 0.0, 0.95),
        ...         target=(0.65, 0.0, 0.05),
        ...         fovy=65.0,
        ...     ),
        ... )
        >>> sd.top_camera.pos
        (0.2, 0.0, 0.95)
    """

    model_config = ConfigDict(extra="forbid")

    top_camera: TopCameraDefaults | None = None
    # issue #191 Phase 3b — declarative MJCF scene composition (openarm tabletop).
    # The manifest-driven node composes the MJCF before building the HAL.
    composition: SceneComposition | None = None


# ─── Collision geometry (ADR-0030) ─────────────────────────────────────────────


class SphereShape(BaseModel):
    """Sphere collision primitive — the simplest convex link/obstacle volume.

    Attributes:
        shape: Discriminator (always ``"sphere"``).
        radius_m: Sphere radius in metres.

    Example:
        >>> SphereShape(radius_m=0.05).shape
        'sphere'
    """

    model_config = ConfigDict(extra="forbid")

    shape: Literal["sphere"] = "sphere"
    radius_m: float = Field(gt=0.0)


class CapsuleShape(BaseModel):
    """Capsule collision primitive — a segment swept by a radius.

    The central segment runs along the local +Z axis from ``-length_m / 2``
    to ``+length_m / 2`` (the MJCF / URDF capsule convention); it is placed
    and oriented by the owning frame (:attr:`LinkCollisionGeometry.origin_xyz_rpy`
    for a link, :attr:`WorldCollisionPrimitive.pose` for an obstacle).
    Capsules bound most robot links tightly, so the safety check stays
    conservative (ADR-0030 §2).

    Attributes:
        shape: Discriminator (always ``"capsule"``).
        radius_m: Capsule radius in metres.
        length_m: Length of the central segment in metres (the cylinder
            portion; the total span is ``length_m + 2 * radius_m``). ``0.0``
            degenerates to a sphere.

    Example:
        >>> CapsuleShape(radius_m=0.04, length_m=0.3).length_m
        0.3
    """

    model_config = ConfigDict(extra="forbid")

    shape: Literal["capsule"] = "capsule"
    radius_m: float = Field(gt=0.0)
    length_m: float = Field(ge=0.0)


CollisionShape: TypeAlias = CapsuleShape | SphereShape
"""Discriminated union of convex collision primitives (ADR-0030).

The discriminator field is ``shape``. Used by
:class:`LinkCollisionGeometry` (robot links) and
:class:`WorldCollisionPrimitive` (world obstacles). Mesh primitives are
intentionally excluded — the allocation-free safety kernel checks only
convex analytic shapes; mesh-accurate collision stays a planning-layer
concern.
"""


class LinkCollisionGeometry(BaseModel):
    """One convex collision volume rigidly attached to a robot link (ADR-0030).

    The lowered, kernel-facing form of a link's collision geometry. Authored
    by hand, or emitted by the offline lowering tool from a robot's MJCF or
    URDF + SRDF source; the kernel loads these into pre-sized buffers at
    ``on_configure`` and never parses the source geometry on the hot path.
    ``link_name`` is a :class:`JointSpec` link (``joints`` stays normative for
    the kinematic chain — this adds geometry only).

    Attributes:
        link_name: The :attr:`JointSpec.child_link` (or
            :attr:`JointSpec.parent_link`) this volume is attached to.
        shape: The convex primitive (capsule or sphere).
        origin_xyz_rpy: Pose of the primitive in the link frame —
            ``(x, y, z, roll, pitch, yaw)`` in metres and radians.

    Example:
        >>> g = LinkCollisionGeometry(
        ...     link_name="link_1",
        ...     shape=CapsuleShape(radius_m=0.04, length_m=0.3),
        ... )
        >>> g.origin_xyz_rpy
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """

    model_config = ConfigDict(extra="forbid")

    link_name: str
    shape: CollisionShape
    origin_xyz_rpy: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


class HalParameters(BaseModel):
    """Per-robot HAL construction defaults declared in the manifest (ADR-0029).

    Carries the transport / constructor keyword arguments a robot's HAL needs
    — the SO-100's serial ``port`` + ``baud``, a ros2_control arm's
    ``robot_ip`` / ``fci_ip`` — so the manifest is the single source of those
    defaults instead of a per-robot lifecycle-node subclass. This is the
    schema seam that lets the unified, ``robot.yaml``-driven
    ``ManifestHALLifecycleNode`` (ADR-0032) serve a parameterised HAL without
    a bespoke ``_create_hal``.

    :func:`openral_hal.build_hal` merges :attr:`defaults` **underneath** any
    explicit ``transport`` kwargs (so a ``deploy run`` override wins) and then
    drops every key the target HAL constructor does not accept — exactly the
    filtering it already applies to ``transport``. Empty by default, so robots
    that need no construction kwargs (the derived ``MujocoArmHAL`` arms) are
    unaffected.

    Attributes:
        defaults: HAL constructor / transport keyword defaults, e.g.
            ``{"port": "/dev/ttyACM0", "baud": 1_000_000}`` for the SO-100 or
            ``{"robot_ip": "192.168.1.10"}`` for a real UR arm.

    Example:
        >>> HalParameters(defaults={"port": "/dev/ttyACM0"}).defaults["port"]
        '/dev/ttyACM0'
    """

    model_config = ConfigDict(extra="forbid")

    defaults: dict[str, object] = Field(default_factory=dict)


class HalEntrypoints(BaseModel):
    """Per-robot simulation and real-hardware HAL import strings.

    The two HALs a robot can expose, declared independently so the choice of
    HAL *type* lives in the manifest (never in environment config or runtime
    params). Each value is a ``"module:Attr"`` import string resolved by
    :func:`openral_hal.build_hal`, or ``None`` when the robot has no HAL of
    that kind (sim-only / real-only / scene-only). ADR-0031.

    Attributes:
        sim: Import string for the simulation HAL. When ``None`` **and**
            :attr:`RobotDescription.sim` is populated, the resolver derives
            ``MujocoArmHAL.from_description`` (ADR-0023) — so every plain arm
            leaves this null. Set it explicitly only for a non-generic sim
            HAL (e.g. ``"openral_hal.panda_mobile:PandaMobileHAL"``, which has
            no ``sim:`` block to derive from).
        real: Import string for the real-hardware HAL (e.g.
            ``"openral_hal.ur_real:UR5eRealHAL"``). ``None`` for sim-only
            robots; the resolver raises ``ROSCapabilityMismatch`` if
            ``mode="real"`` is requested on a robot whose ``real`` is ``None``.
        parameters: Per-robot HAL construction defaults (serial ``port``,
            ``robot_ip``, …) merged into the constructor by
            :func:`openral_hal.build_hal`. ADR-0029. Empty by default.

    Example:
        >>> HalEntrypoints(real="openral_hal.ur_real:UR5eRealHAL").sim is None
        True
    """

    model_config = ConfigDict(extra="forbid")

    sim: str | None = None
    real: str | None = None
    parameters: HalParameters = Field(default_factory=HalParameters)


class RobotDescription(BaseModel):
    """Top-level robot manifest — one per robot, published to HuggingFace Hub.

    Attributes:
        name: Robot name, e.g. "so100_follower".
        embodiment_kind: Top-level kinematic class.
        assets: Unified URDF / MJCF / SRDF reference block (ADR-0058).
            Refs share the :func:`openral_core.assets.resolve_asset`
            grammar; the URDF's ``robot_state_publisher`` wiring lives on
            :attr:`AssetRefs.urdf` (ADR-0027). Empty by default.
        base_frame: Base link tf2 frame name.
        odom_frame: Odometry tf2 frame name.
        map_frame: Map tf2 frame name.
        joints: List of joint specifications.
        end_effectors: List of end-effector specifications.
        sensors: List of individual sensor specs.
        sensor_bundles: List of multi-modal sensor bundles.
        capabilities: Capability flags for skill matching.
        safety: Safety envelope constraints.
        ros2_namespace: ROS 2 namespace prefix.
        middleware: ROS 2 middleware selection.
        onboard_compute: Onboard compute descriptors.
        sdk_kind: Whether the SDK is open or closed.
        hal: Simulation + real-hardware HAL import strings. ``deploy sim``
            constructs ``hal.sim`` (or derives ``MujocoArmHAL`` from
            :attr:`sim`); ``deploy run`` constructs ``hal.real``. ADR-0031.
        observation_spec: VLA observation configuration.
        action_spec: VLA action configuration.
        sim: Optional MuJoCo wiring consumed by
            :class:`openral_hal.MujocoArmHAL`.  When set, the HAL can be
            constructed entirely from the manifest via
            :meth:`MujocoArmHAL.from_description`; no per-robot Python
            subclass is required.
        scene_defaults: Optional per-robot scene rendering defaults.
            Today carries :class:`TopCameraDefaults` consumed by the
            ``openarm_tabletop_pnp`` MJCF composer (see
            :mod:`openral_sim.backends.openarm_robosuite._assets`).
            Added in openral-core 0.6.0 to replace the dataset-specific
            module-level constants that previously lived in the sim
            backend.
        base_joints: Optional ordered list of :attr:`JointSpec.name`
            references identifying the **planar mobile base** joints.
            When the robot has a holonomic / differential / omnidirectional
            base, the three entries are conventionally ``[forward_axis,
            side_axis, yaw_axis]``. Consumed by the generic ray-cast
            helpers in :mod:`openral_sim.backends.robocasa` and the
            ROS HAL lifecycle nodes to resolve MJCF joint names without
            hardcoding robot-specific conventions. ``None`` for
            fixed-base manipulators. ADR-0025.
        collision_geometry: Per-link convex collision primitives
            (capsules / spheres) the safety kernel uses for self- and
            world-collision checking. Empty by default. ADR-0030.
        allowed_collision_pairs: Link-name pairs excluded from
            self-collision (adjacent links touch by design); the
            allowed-collision matrix. On real robots this is sourced from
            the SRDF ``disable_collisions`` block (named by
            :attr:`AssetRefs.srdf`). ADR-0030.

    Example:
        >>> desc = RobotDescription(
        ...     name="smoke_robot",
        ...     embodiment_kind=EmbodimentKind.MANIPULATOR,
        ...     joints=[
        ...         JointSpec(
        ...             name="j1",
        ...             joint_type=JointType.REVOLUTE,
        ...             parent_link="base_link",
        ...             child_link="link_1",
        ...         )
        ...     ],
        ...     capabilities=RobotCapabilities(
        ...         supported_control_modes=[ControlMode.JOINT_POSITION],
        ...         embodiment_tags=["smoke"],
        ...     ),
        ...     safety=SafetyEnvelope(),
        ... )
        >>> assert desc.name == "smoke_robot"
    """

    name: str
    embodiment_kind: EmbodimentKind
    assets: AssetRefs = Field(default_factory=AssetRefs)
    base_frame: str = "base_link"
    odom_frame: str = "odom"
    map_frame: str = "map"
    joints: list[JointSpec]
    end_effectors: list[EndEffectorSpec] = Field(default_factory=list)
    sensors: list[SensorSpec] = Field(default_factory=list)
    sensor_bundles: list[SensorBundle] = Field(default_factory=list)
    capabilities: RobotCapabilities
    safety: SafetyEnvelope
    ros2_namespace: str = ""
    middleware: Literal["fastdds", "cyclonedds", "zenoh"] = "cyclonedds"
    onboard_compute: dict[str, object] = Field(default_factory=dict)
    sdk_kind: Literal["open", "closed_with_api", "closed"] = "open"
    hal: HalEntrypoints = Field(default_factory=HalEntrypoints)
    observation_spec: ObservationSpec | None = None
    action_spec: ActionSpec | None = None
    sim: SimDescription | None = None
    scene_defaults: SceneDefaults | None = None
    base_joints: list[str] | None = None
    # ADR-0027 robot_state_publisher wiring now lives on ``assets.urdf``
    # (``root_frame`` + ``base_to_root_xyz_rpy``) — see ``UrdfAsset``.
    # ADR-0025 / Nav2 — generic mobile-base properties so a per-robot
    # Nav2 param file need not be hand-vendored. ``footprint_radius``
    # feeds Nav2's ``robot_radius`` (collision envelope, metres) and
    # ``base_kinematics`` selects the MPPI ``motion_model`` ("omni" /
    # "holonomic" → holonomic + symmetric lateral bound;
    # "differential" / "ackermann" → the matching upstream model).
    # Both ``None`` on fixed-base arms (no Nav2).
    footprint_radius: float | None = Field(default=None, gt=0.0)
    base_kinematics: Literal["differential", "holonomic", "omni", "ackermann"] | None = None
    # ADR-0030 — geometric safety. ``collision_geometry`` is the lowered,
    # kernel-facing set of per-link convex primitives; ``allowed_collision_pairs``
    # is the self-collision exclusion matrix (adjacent links touch by design).
    # Both are authored by hand or emitted by the offline lowering tool from
    # this robot's MJCF or URDF + SRDF. ``assets.srdf`` points at an SRDF whose
    # ``disable_collisions`` block is the canonical source for
    # ``allowed_collision_pairs`` on real robots. All empty / ``None`` keeps
    # existing manifests loadable; ``joints`` stays normative for the chain, so
    # URDF/SRDF contribute geometry + ACM only (no dual source of truth).
    collision_geometry: list[LinkCollisionGeometry] = Field(default_factory=list)
    allowed_collision_pairs: list[tuple[str, str]] = Field(default_factory=list)
    # ADR-0025 / dashboard overlay — optional base footprint as a list of
    # base-frame ``(x, y)`` vertices in metres (CCW by convention). Used to
    # draw the robot's true outline on the SLAM occupancy grid; ``None``
    # falls back to the ``footprint_radius`` circle. Independent of
    # ``footprint_radius`` (a robot may declare either, both, or neither).
    footprint_polygon: list[tuple[float, float]] | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_footprint_polygon(self) -> RobotDescription:
        """A declared footprint polygon needs >= 3 vertices with finite coords."""
        min_polygon_vertices = 3
        poly = self.footprint_polygon
        if poly is None:
            return self
        if len(poly) < min_polygon_vertices:
            raise ValueError(
                f"footprint_polygon needs >= {min_polygon_vertices} vertices; got {len(poly)}"
            )
        if any(not math.isfinite(c) for pt in poly for c in pt):
            raise ValueError("footprint_polygon vertices must be finite (no NaN/inf)")
        return self

    @model_validator(mode="after")
    def _validate_base_joints_against_joints(self) -> RobotDescription:
        """Ensure ``base_joints`` is well-formed: ≥3 entries, all real joint names.

        ADR-0025 — `extract_base_sim_joint_names` specialises to the
        planar-base case (3 entries: forward / side / yaw). A robot
        manifest declaring `base_joints: [base_x]` would silently
        miss the helper's gate and fall back to module defaults.
        Reject the shape up front so the misconfiguration is loud.
        """
        if self.base_joints is None:
            return self
        planar_base_dof = 3
        if len(self.base_joints) < planar_base_dof:
            raise ValueError(
                f"base_joints must declare at least {planar_base_dof} entries "
                f"(forward, side, yaw) for the planar-base helper to engage; "
                f"got {len(self.base_joints)}."
            )
        joint_names = {j.name for j in self.joints}
        for ref in self.base_joints:
            if ref not in joint_names:
                raise ValueError(
                    f"base_joints[*]={ref!r} is not present in joints (have: {sorted(joint_names)})"
                )
        return self

    @property
    def lidar_sensor(self) -> SensorSpec | None:
        """The first declared 2-D LiDAR / scan sensor, or ``None``.

        ADR-0025 — the panda_mobile HAL synthesises a ``sensor_msgs/
        LaserScan`` from MuJoCo ray-casts; its beam count
        (:attr:`SensorSpec.n_channels`), range
        (:attr:`SensorSpec.range_min_m` / :attr:`SensorSpec.range_max_m`)
        and rate (:attr:`SensorSpec.rate_hz`) live on this descriptor so
        ``openral deploy sim`` and the HAL lifecycle node read one source of
        truth from ``robot.yaml`` instead of each hardcoding scan params.
        """
        return next(
            (s for s in self.sensors if s.modality == SensorModality.LIDAR_2D.value),
            None,
        )

    def nav2_param_overrides(self) -> dict[str, str]:
        """Nav2 param substitutions derived from this robot's base props.

        ADR-0025 — lets the Nav2 bringup stay generic: instead of
        hand-vendoring a per-robot ``nav2_<robot>.yaml``, the launch
        rewrites a shared base param file with these key→value
        substitutions. Maps :attr:`footprint_radius` → ``robot_radius``
        (collision envelope) **and** the costmap ``inflation_radius``
        (``footprint_radius`` + :data:`NAV2_INFLATION_CLEARANCE_M`, so it
        stays ≥ the inscribed/circumscribed radius Nav2 derives from the
        footprint — otherwise Nav2 errors and falls back to slow
        full-footprint collision checks), and :attr:`base_kinematics` →
        the MPPI ``motion_model``. Returns an empty dict for fixed-base
        arms (no mobile base → no Nav2). Velocity bounds remain Nav2
        tuning in the base param file, not robot identity.
        """
        overrides: dict[str, str] = {}
        if self.footprint_radius is not None:
            overrides["robot_radius"] = str(self.footprint_radius)
            overrides["inflation_radius"] = (
                f"{self.footprint_radius + NAV2_INFLATION_CLEARANCE_M:.3f}"
            )
        if self.base_kinematics is not None:
            overrides["motion_model"] = {
                "omni": "Omni",
                "holonomic": "Omni",
                "differential": "DiffDrive",
                "ackermann": "Ackermann",
            }[self.base_kinematics]
        return overrides

    @model_validator(mode="after")
    def _validate_sim_against_joints(self) -> RobotDescription:
        """Ensure every ``sim.grippers[*].joint`` is also in ``joints``."""
        if self.sim is None:
            return self
        joint_names = {j.name for j in self.joints}
        for gripper in self.sim.grippers:
            if gripper.joint not in joint_names:
                raise ValueError(
                    f"sim.grippers[].joint={gripper.joint!r} is not present "
                    f"in joints (have: {sorted(joint_names)})"
                )
        # Duplicate-joint guard — a gripper can't be listed twice (would
        # double-apply the write in send_action).
        gripper_joints = [g.joint for g in self.sim.grippers]
        if len(gripper_joints) != len(set(gripper_joints)):
            raise ValueError(f"sim.grippers contains duplicate joint names: {gripper_joints}")
        return self

    @classmethod
    def from_yaml(cls, path: str) -> RobotDescription:
        """Load and validate a ``RobotDescription`` YAML manifest from disk.

        Args:
            path: Filesystem path to the ``robot.yaml`` file.

        Returns:
            A validated :class:`RobotDescription`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            pydantic.ValidationError: If the YAML fails schema validation.

        Example:
            >>> # RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
        """
        import yaml  # noqa: PLC0415  # reason: deferred to avoid import-time cost

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def validate_for_e2e_pipeline(self) -> None:
        """Assert this manifest carries every field the e2e ROS graph needs.

        The C++ safety kernel (``cpp/openral_safety_kernel``, ADR-0020)
        reads per-joint ``position_limits`` / ``velocity_limit`` /
        ``effort_limit`` + the global ``safety:`` block. Per-joint
        limit fields are *optional* on :class:`JointSpec` for sim-only
        robots, but they become mandatory the moment you ask
        ``openral deploy sim`` to bring up the kernel against this robot.

        This method makes the e2e contract explicit and fails loud
        listing every missing field. Pair it with
        ``openral_safety.envelope_loader.compute_intersection(robot,
        skill=None)`` for the actual envelope synthesis — this method
        only validates.

        Raises:
            ROSConfigError: If any actuated joint is missing
                ``position_limits``, ``velocity_limit``, or
                ``effort_limit``. The error lists every missing field
                at once so the operator does not have to fix one,
                re-run, fix the next.
        """
        missing: list[str] = []
        for joint in self.joints:
            if joint.joint_type not in (
                JointType.REVOLUTE,
                JointType.PRISMATIC,
                JointType.CONTINUOUS,
            ):
                continue
            if joint.position_limits is None:
                missing.append(f"joints[{joint.name}].position_limits")
            if joint.velocity_limit is None:
                missing.append(f"joints[{joint.name}].velocity_limit")
            if joint.effort_limit is None:
                missing.append(f"joints[{joint.name}].effort_limit")
        if missing:
            raise ROSConfigError(
                f"RobotDescription({self.name!r}) is missing fields required by "
                f"the e2e safety kernel: {missing}. Set them on the matching "
                "JointSpec(s) in robots/<robot_id>/robot.yaml."
            )


def extract_base_sim_joint_names(
    description: RobotDescription,
) -> tuple[str, str, str] | None:
    """Return ``(forward, side, yaw)`` MJCF joint names from any mobile-base description.

    ADR-0025 — generic, robot-agnostic helper. Consumes the
    :attr:`RobotDescription.base_joints` declaration + each referenced
    :class:`JointSpec`'s :attr:`~JointSpec.sim_joint_name` override.
    Works for any robot whose ``robot.yaml`` declares both fields:

    * ``base_joints: [<forward>, <side>, <yaw>]`` at the top level.
    * Each of the three referenced joints carries a
      ``sim_joint_name: "..."`` mapping its URDF-shape ``name`` to the
      MJCF/MuJoCo joint name the simulator emits (typically
      auto-prefixed under a composed scene).

    Returns ``None`` when the description has no ``base_joints``
    block, fewer than three entries (the schema permits arbitrary
    list length so future non-planar bases can declare more — this
    helper specialises to the planar-base case), or any referenced
    joint lacks ``sim_joint_name``. Callers should treat ``None`` as
    "fall back to module defaults" — the sim-side ray-cast helpers
    in :mod:`openral_sim.backends.robocasa` accept this contract.

    Args:
        description: The robot's manifest, loaded via
            :meth:`RobotDescription.from_yaml`.

    Returns:
        ``(forward, side, yaw)`` MJCF joint names, or ``None`` when
        the description doesn't carry a complete mobile-base block.

    Example:
        >>> # See `robots/panda_mobile/robot.yaml` for a real fixture.
        >>> desc = RobotDescription(
        ...     name="ex",
        ...     embodiment_kind=EmbodimentKind.MOBILE_MANIPULATOR,
        ...     joints=[
        ...         JointSpec(
        ...             name="base_x",
        ...             joint_type=JointType.PRISMATIC,
        ...             parent_link="world",
        ...             child_link="base_x_link",
        ...             sim_joint_name="mobilebase0_joint_mobile_forward",
        ...         ),
        ...         JointSpec(
        ...             name="base_y",
        ...             joint_type=JointType.PRISMATIC,
        ...             parent_link="base_x_link",
        ...             child_link="base_y_link",
        ...             sim_joint_name="mobilebase0_joint_mobile_side",
        ...         ),
        ...         JointSpec(
        ...             name="base_yaw",
        ...             joint_type=JointType.REVOLUTE,
        ...             parent_link="base_y_link",
        ...             child_link="base_link",
        ...             sim_joint_name="mobilebase0_joint_mobile_yaw",
        ...         ),
        ...     ],
        ...     capabilities=RobotCapabilities(embodiment_tags=["ex"]),
        ...     safety=SafetyEnvelope(),
        ...     base_joints=["base_x", "base_y", "base_yaw"],
        ... )
        >>> extract_base_sim_joint_names(desc)[0]
        'mobilebase0_joint_mobile_forward'
    """
    # Planar mobile bases carry exactly three holonomic axes (forward,
    # side, yaw); the schema doesn't constrain `base_joints` length so
    # future non-planar bases can declare richer surfaces, but this
    # helper specialises here.
    planar_base_dof = 3
    if description.base_joints is None or len(description.base_joints) < planar_base_dof:
        return None
    by_name = {j.name: j for j in description.joints}
    sim_names: list[str | None] = []
    for ref in description.base_joints[:planar_base_dof]:
        spec = by_name.get(ref)
        if spec is None:
            return None
        sim_names.append(spec.sim_joint_name)
    if any(n is None for n in sim_names):
        return None
    forward, side, yaw = sim_names
    # Narrow tuple[str | None, ...] → tuple[str, str, str] for type checkers.
    assert isinstance(forward, str)
    assert isinstance(side, str)
    assert isinstance(yaw, str)
    return (forward, side, yaw)


# ─── World state ───────────────────────────────────────────────────────────────


class JointState(BaseModel):
    """Real-time joint state snapshot.

    Attributes:
        name: Ordered list of joint names.
        position: Joint positions (rad or m).
        velocity: Joint velocities (rad/s or m/s).
        effort: Joint efforts (Nm or N).
        stamp_ns: ROS 2 timestamp in nanoseconds.
    """

    name: list[str]
    position: list[float]
    velocity: list[float] = Field(default_factory=list)
    effort: list[float] = Field(default_factory=list)
    stamp_ns: int


class Pose6D(BaseModel):
    """6D pose: position + quaternion.

    Attributes:
        xyz: Position (x, y, z) in meters.
        quat_xyzw: Quaternion (x, y, z, w).
        frame_id: tf2 reference frame.
    """

    xyz: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]
    frame_id: str


class DetectedObject(BaseModel):
    """A detected object in the scene.

    Attributes:
        label: Semantic class label.
        confidence: Detection confidence in [0, 1].
        pose: 6D pose in the world or camera frame.
        bbox_3d: 3D bounding box (x_min, y_min, z_min, x_max, y_max, z_max).
        track_id: Persistent track ID across frames.
    """

    label: str
    confidence: float
    pose: Pose6D
    bbox_3d: tuple[float, float, float, float, float, float] | None = None
    track_id: int | None = None


class WorldCollisionPrimitive(BaseModel):
    """A placed convex obstacle volume in the world (ADR-0030).

    The world-frame analogue of :class:`LinkCollisionGeometry`: a convex
    primitive plus the pose that places it. Populated by perception / SLAM and
    consumed by the kernel's world-collision phase against the robot's link
    capsules. A bounded, capped set is the kernel's world model (mesh
    obstacles are out of scope for the allocation-free check).

    Attributes:
        shape: The convex primitive (capsule or sphere).
        pose: Pose of the primitive's local origin in the world frame.
        object_id: Optional stable identifier (e.g. a
            :attr:`DetectedObject.track_id` rendered as text) surfaced in
            :attr:`CollisionEvidence.link_b_or_object`.
    """

    model_config = ConfigDict(extra="forbid")

    shape: CollisionShape
    pose: Pose6D
    object_id: str | None = None


class OccupancyGridRef(BaseModel):
    """Reference to a 2D occupancy grid for mobile-base world-collision (ADR-0030).

    Mirrors the ``nav_msgs/OccupancyGrid`` metadata that
    :mod:`openral_runner.slam_bridge` already decodes. The kernel consumes a
    bounded, fixed-capacity copy; a grid exceeding the configured capacity or
    older than the staleness deadline is treated as unavailable
    (fail-closed). The occupancy bytes are referenced by topic, not inlined.

    Attributes:
        frame_id: tf2 frame the grid origin is expressed in (e.g. ``"map"``).
        resolution_m: Edge length of one cell, in metres.
        width: Grid width in cells.
        height: Grid height in cells.
        origin: Pose of cell ``(0, 0)``'s lower-left corner.
        data_topic: ROS 2 topic carrying the ``nav_msgs/OccupancyGrid``.
    """

    model_config = ConfigDict(extra="forbid")

    frame_id: str
    resolution_m: float = Field(gt=0.0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    origin: Pose6D
    data_topic: str


class FrameEncoding(str, Enum):
    """How the bytes inside a :class:`SensorFrame` are interpreted.

    The first four values are raw per-pixel layouts. ``JPEG`` / ``PNG`` are
    compressed forms; the runner decodes lazily. ``CUDA_NV12`` and ``RAW``
    mark frames whose payload is an opaque handle (NVMM pointer, DMA-BUF fd)
    — the ``data`` field is empty and the consumer must read via ``handle``.
    """

    BGR8 = "bgr8"
    RGB8 = "rgb8"
    MONO8 = "mono8"
    DEPTH16 = "depth16"
    JPEG = "jpeg"
    PNG = "png"
    CUDA_NV12 = "cuda_nv12"
    RAW = "raw"


class SensorFrame(BaseModel):
    """A single sensor frame — metadata plus an optional inline payload.

    A :class:`SensorFrame` is the carrier passed from a ``SensorReader`` into
    :class:`WorldState.image_frames`. Exactly one of ``data``,
    ``topic``, or ``handle`` is populated:

    * ``data`` carries the pixel bytes for in-process delivery and for trace
      capture. JSON-serialized as base64 by Pydantic, so payloads round-trip
      through ``model_dump_json`` / ``model_validate_json``.
    * ``topic`` points at a ROS 2 topic — the bytes live on the ROS 2 bus and
      the consumer subscribes for them. This is the path the
      ``Ros2ImageSensorReader`` uses and what :attr:`WorldState.images`
      already carries.
    * ``handle`` is an opaque integer (e.g. CUDA NVMM pointer, DMA-BUF file
      descriptor) — in-process only, never serialized. Set by the
      ``GStreamerSensorReader`` when frames stay on the GPU.

    Attributes:
        sensor_id: Sensor name; matches :attr:`SensorSpec.name`.
        stamp_monotonic_ns: Capture-time monotonic timestamp in nanoseconds.
        stamp_wall_ns: Capture-time wall-clock timestamp in nanoseconds.
        encoding: How to interpret the bytes.
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of channels (3 for RGB/BGR, 1 for MONO/DEPTH).
        data: Optional inline pixel payload (or compressed bytes for
            ``JPEG`` / ``PNG``). Mutually exclusive with ``topic`` /
            ``handle``.
        topic: Optional ROS 2 topic reference. Mutually exclusive with
            ``data`` / ``handle``.
        handle: Optional opaque in-process handle (NVMM / DMA-BUF). Mutually
            exclusive with ``data`` / ``topic``.
        metadata: Free-form per-frame metadata (gain, exposure, …).

    Example:
        >>> SensorFrame(
        ...     sensor_id="wrist_rgb",
        ...     stamp_monotonic_ns=1,
        ...     stamp_wall_ns=2,
        ...     encoding=FrameEncoding.RGB8,
        ...     width=640,
        ...     height=480,
        ...     topic="/cameras/wrist_rgb/image_raw",
        ... ).channels
        3
    """

    model_config = ConfigDict(extra="forbid")

    sensor_id: str
    stamp_monotonic_ns: int = Field(ge=0)
    stamp_wall_ns: int = Field(ge=0)
    encoding: FrameEncoding
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: int = Field(default=3, gt=0)
    data: bytes | None = None
    topic: str | None = None
    handle: int | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("data", mode="before")
    @classmethod
    def _decode_data(cls, value: Any) -> bytes | None:  # noqa: ANN401  # reason: Pydantic field_validator passes raw input
        """Accept raw ``bytes`` or a base64-encoded ``str`` (from JSON)."""
        if value is None or isinstance(value, bytes):
            return value
        if isinstance(value, str):
            try:
                return base64.b64decode(value, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError(
                    f"SensorFrame.data string must be valid base64; got {value!r}"
                ) from exc
        raise TypeError(
            f"SensorFrame.data must be bytes or a base64 string; got {type(value).__name__}"
        )

    @field_serializer("data", when_used="json")
    def _encode_data(self, value: bytes | None) -> str | None:
        """JSON-serialize the binary payload as base64 (preserves arbitrary bytes)."""
        return None if value is None else base64.b64encode(value).decode("ascii")

    def model_post_init(self, _context: object) -> None:
        """Exactly one of (data, topic, handle) must be set."""
        populated = [bool(self.data), self.topic is not None, self.handle is not None]
        n = sum(populated)
        if n != 1:
            raise ValueError(
                f"SensorFrame({self.sensor_id!r}): exactly one of "
                f"(data, topic, handle) must be set; got {n}."
            )


class WorldState(BaseModel):
    """Snapshot consumed by Reasoner and Skills.

    Attributes:
        stamp_ns: Snapshot timestamp in nanoseconds.
        joint_state: Current joint state.
        base_pose: Base link pose (mobile robots).
        base_twist: Base link twist (vx, vy, vz, wx, wy, wz).
        ee_poses: End-effector poses keyed by EE name.
        contact_forces: Contact forces keyed by contact name.
        images: Sensor name → ROS 2 topic reference (not the raw image).
        image_frames: Optional per-sensor :class:`SensorFrame` snapshot for
            no-ROS / in-process deployments. When ``None`` the
            consumer reads frames via :attr:`images` topic refs as before.
        point_clouds: Sensor name → ROS 2 topic reference.
        tactile: Sensor name → ROS 2 topic reference.
        detected_objects: List of detected objects.
        battery_pct: Battery percentage in [0, 100].
        diagnostics: Per-component diagnostic status.
        collision_primitives: Bounded set of placed convex obstacle volumes
            the kernel checks robot links against (world-collision). Empty
            until a perception / SLAM source populates it. ADR-0030.
        occupancy_grid: Optional 2D occupancy grid reference for mobile-base
            footprint checks. ``None`` until populated; an absent or stale
            grid is treated as unavailable (fail-closed). ADR-0030.
    """

    stamp_ns: int
    joint_state: JointState
    base_pose: Pose6D | None = None
    base_twist: tuple[float, float, float, float, float, float] | None = None
    ee_poses: dict[str, Pose6D] = Field(default_factory=dict)
    contact_forces: dict[str, tuple[float, ...]] = Field(default_factory=dict)
    images: dict[str, str] = Field(default_factory=dict)
    image_frames: dict[str, SensorFrame] | None = None
    point_clouds: dict[str, str] = Field(default_factory=dict)
    tactile: dict[str, str] = Field(default_factory=dict)
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    battery_pct: float | None = None
    diagnostics: dict[str, Literal["ok", "warn", "error", "stale"]] = Field(default_factory=dict)
    # ADR-0030 — bounded world surface for kernel world-collision checking.
    collision_primitives: list[WorldCollisionPrimitive] = Field(default_factory=list)
    occupancy_grid: OccupancyGridRef | None = None


# ─── Spatial memory — persistent scene graph (ADR-0038) ──────────────────────────


class SpatialNodeKind(str, Enum):
    """Kind of node in the persistent scene-graph spatial memory (ADR-0038).

    ``OBJECT`` is the foundation (an accumulated :class:`DetectedObject`);
    ``PLACE`` is a standable navigation waypoint; ``ROOM`` is a semantic area
    grouping places/objects; ``AGENT`` is a person or robot with a pose — the
    requester of a task is an ``AGENT`` so "bring it back to me" resolves to a
    concrete goal.
    """

    OBJECT = "object"
    PLACE = "place"
    ROOM = "room"
    AGENT = "agent"


class SpatialRelationKind(str, Enum):
    """Kind of directed edge between scene-graph nodes (ADR-0038).

    ``CONTAINS`` links a room/container to what is inside it (a fridge
    ``CONTAINS`` a wine bottle); ``AT_PLACE`` links an object/agent to the
    waypoint to stand at to reach it; ``TRAVERSABLE_TO`` is the topological
    navigation graph between places/rooms; ``ON`` / ``NEAR`` are incidental
    object-to-object spatial relations.
    """

    CONTAINS = "contains"
    AT_PLACE = "at_place"
    TRAVERSABLE_TO = "traversable_to"
    ON = "on"
    NEAR = "near"


class SpatialNode(BaseModel):
    """A persistent, typed node in the scene-graph spatial memory (ADR-0038).

    A superset of :class:`DetectedObject` for ``kind == OBJECT``; also used for
    places, rooms, and agents. The pose is anchored in a durable, drift-corrected
    frame (typically the tf2 ``map`` frame); consumers resolve it to the live
    base frame at query time via tf2 — the node never stores a raw transform.

    This is **advisory** world-model state consumed by the S2 Reasoner. It is
    never a safety input (ADR-0038 §1, CLAUDE.md §1.1): the safety kernel gates
    only on the live, bounded ADR-0030 geometric world.

    Attributes:
        node_id: Stable identifier, unique within a :class:`SceneGraph`.
        kind: Node kind (object / place / room / agent).
        pose: 6D pose, anchored in the durable map frame.
        label: Semantic label or name (e.g. ``"fridge"``, ``"kitchen"``).
        confidence: Confidence in [0, 1] for a perceived node.
        bbox_3d: Optional 3D bounding box
            (x_min, y_min, z_min, x_max, y_max, z_max).
        embedding_ref: Optional handle into the vector store for
            open-vocabulary matching (ADR-0038 §5); ``None`` → label-only.
        is_container: Whether the node can hold other nodes (fridge, cabinet).
        occludes_contents: Whether contents are unobservable until the
            container is opened. Requires ``is_container``.
        first_seen_ns: Timestamp of first observation, in nanoseconds.
        last_seen_ns: Timestamp of the most recent observation, in nanoseconds;
            must be ``>= first_seen_ns``.
        observation_count: Number of times this node has been observed.

    Example:
        >>> node = SpatialNode(
        ...     node_id="fridge",
        ...     kind=SpatialNodeKind.OBJECT,
        ...     pose=Pose6D(xyz=(3.0, 1.0, 0.9), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map"),
        ...     label="fridge",
        ...     is_container=True,
        ...     occludes_contents=True,
        ...     first_seen_ns=1,
        ...     last_seen_ns=2,
        ... )
        >>> node.is_container
        True
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    kind: SpatialNodeKind
    pose: Pose6D
    label: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    bbox_3d: tuple[float, float, float, float, float, float] | None = None
    embedding_ref: str | None = None
    is_container: bool = False
    occludes_contents: bool = False
    first_seen_ns: int = Field(ge=0)
    last_seen_ns: int = Field(ge=0)
    observation_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_node_invariants(self) -> SpatialNode:
        """Enforce temporal ordering and the container/occlusion relationship."""
        if self.last_seen_ns < self.first_seen_ns:
            raise ValueError(
                f"SpatialNode({self.node_id!r}): last_seen_ns ({self.last_seen_ns}) "
                f"must be >= first_seen_ns ({self.first_seen_ns})."
            )
        if self.occludes_contents and not self.is_container:
            raise ValueError(
                f"SpatialNode({self.node_id!r}): occludes_contents requires is_container."
            )
        return self


class SpatialEdge(BaseModel):
    """A directed relation between two scene-graph nodes (ADR-0038).

    Attributes:
        src: ``node_id`` of the source node.
        dst: ``node_id`` of the destination node.
        kind: Relation kind.
    """

    model_config = ConfigDict(extra="forbid")

    src: str = Field(min_length=1)
    dst: str = Field(min_length=1)
    kind: SpatialRelationKind


class SceneGraph(BaseModel):
    """Persistent hierarchical scene-graph spatial memory (ADR-0038).

    The durable, queryable world model the S2 Reasoner consults to recall where
    objects/places/agents are and how to navigate to them. Distinct from the
    ephemeral ADR-0030 collision grid and **advisory only** — never a safety
    input (CLAUDE.md §1.1).

    Invariants (enforced): node ids are unique; every edge references existing
    nodes.

    Attributes:
        schema_version: On-disk schema version (``"0.1"``; no
            backward-incompatible change yet). Now the repo is published it
            is versioned for real (CLAUDE.md §1.6): an incompatible change
            bumps it and ships a migrator.
        nodes: All scene-graph nodes.
        edges: All directed relations between nodes.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    nodes: list[SpatialNode] = Field(default_factory=list)
    edges: list[SpatialEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_graph_integrity(self) -> SceneGraph:
        """Node ids are unique and every edge references existing nodes."""
        ids = [node.node_id for node in self.nodes]
        known = set(ids)
        if len(ids) != len(known):
            raise ValueError("SceneGraph node_id values must be unique.")
        for edge in self.edges:
            if edge.src not in known:
                raise ValueError(f"SceneGraph edge references unknown src node: {edge.src!r}")
            if edge.dst not in known:
                raise ValueError(f"SceneGraph edge references unknown dst node: {edge.dst!r}")
        return self


class RecallObjectQuery(BaseModel):
    """Read-only query to recall a remembered object (ADR-0038 §6).

    At least one of ``text`` / ``label`` must be non-empty. ``text`` is matched
    against node embeddings when an embedder is configured (ADR-0038 §5),
    otherwise matching falls back to ``label``.

    Attributes:
        text: Free-text query (open-vocabulary match).
        label: Exact label match.
        near: Optional pose to bias results toward (proximity).
        max_age_ns: Optional recency filter — drop nodes whose ``last_seen_ns``
            is older than this many nanoseconds before "now".
        limit: Maximum number of matches to return.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    label: str = ""
    near: Pose6D | None = None
    max_age_ns: int | None = Field(default=None, ge=0)
    limit: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def _require_query_term(self) -> RecallObjectQuery:
        """Reject an empty query — a recall must name what it is looking for."""
        if not self.text and not self.label:
            raise ValueError("RecallObjectQuery requires a non-empty text or label.")
        return self


class ApproachViewpoint(BaseModel):
    """A camera-facing standoff pose for viewing/manipulating an object (ADR-0038 §6).

    Attributes:
        pose: Base/EE goal pose (map frame) at a standoff from the object,
            oriented so the gripper-mounted camera faces it.
        standoff_m: Standoff distance from the object, in metres.
        camera_frame_id: tf2 frame of the camera the viewpoint orients toward.
    """

    model_config = ConfigDict(extra="forbid")

    pose: Pose6D
    standoff_m: float = Field(gt=0.0)
    camera_frame_id: str = Field(min_length=1)


class RecallObjectMatch(BaseModel):
    """One ranked match from a :class:`RecallObjectQuery` (ADR-0038 §6).

    Attributes:
        node_id: The matched node's id.
        label: The matched node's label.
        pose: The object's recalled pose (map frame).
        score: Match score in [0, 1].
        last_seen_ns: When the object was last observed, in nanoseconds.
        approach: Optional computed camera-facing approach viewpoint.
        inside_container_id: Set when the object is inside an occluding container
            that must be opened first; ``None`` otherwise.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    label: str = ""
    pose: Pose6D
    score: float = Field(ge=0.0, le=1.0)
    last_seen_ns: int = Field(ge=0)
    approach: ApproachViewpoint | None = None
    inside_container_id: str | None = None


class RecallObjectResult(BaseModel):
    """Result of a :class:`RecallObjectQuery` (ADR-0038 §6).

    Attributes:
        matches: Ranked matches (possibly empty — an empty result is how the
            query reports "unknown"; callers raise / handle
            :class:`~openral_core.exceptions.ROSObjectNotInMemory` rather than
            fabricating a pose).
    """

    model_config = ConfigDict(extra="forbid")

    matches: list[RecallObjectMatch] = Field(default_factory=list)


class ResolvePlaceQuery(BaseModel):
    """Read-only query to resolve a place/room/agent reference to a goal (ADR-0038 §6).

    Attributes:
        reference: Free-text, id, or label of the target (e.g. ``"kitchen"``,
            ``"where I was standing"``, a node id).
        kind: Optional node-kind filter (room / place / agent).
    """

    model_config = ConfigDict(extra="forbid")

    reference: str = Field(min_length=1)
    kind: SpatialNodeKind | None = None


class ResolvePlaceResult(BaseModel):
    """Result of a :class:`ResolvePlaceQuery` (ADR-0038 §6).

    Attributes:
        node_id: The resolved node's id.
        goal: Navigation goal pose (map frame).
        path_node_ids: Ordered ``traversable_to`` path of node ids from the
            robot's current place to the goal (empty when no path is needed or
            known).
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    goal: Pose6D
    path_node_ids: list[str] = Field(default_factory=list)


# ─── Action ────────────────────────────────────────────────────────────────────


class Action(BaseModel):
    """A single action step or chunk produced by a Skill.

    Attributes:
        control_mode: Target action space.
        horizon: Number of steps (1 = single step, H = chunk).
        joint_targets: Joint position targets, shape ``(H, N)``.
        joint_velocities: Joint velocity targets, shape ``(H, N)``.
        joint_torques: Joint torque targets, shape ``(H, N)``.
        cartesian_pose: EE pose targets.
        cartesian_delta: EE pose deltas.
        cartesian_twist: EE velocity targets.
        body_twist: Base twist targets.
        foot_placements: Discrete footstep targets.
        gripper: Gripper commands in [0, 1].
        dex_hand_joints: Dexterous hand joint targets.
        confidence: rSkill confidence in [0, 1].
        stamp_ns: Action timestamp in nanoseconds.
        ee_name: Target end-effector name.
        frame_id: Reference frame for Cartesian actions.
        safety_overrides: Operator-approved safety override tokens.
    """

    control_mode: ControlMode
    horizon: int = 1
    # one of the following is populated:
    joint_targets: list[list[float]] | None = None
    joint_velocities: list[list[float]] | None = None
    joint_torques: list[list[float]] | None = None
    cartesian_pose: list[Pose6D] | None = None
    cartesian_delta: list[tuple[float, ...]] | None = None
    cartesian_twist: list[tuple[float, ...]] | None = None
    body_twist: list[tuple[float, float, float, float, float, float]] | None = None
    foot_placements: list[dict[str, object]] | None = None
    gripper: list[float] | None = None
    dex_hand_joints: list[list[float]] | None = None
    # ADR-0028d — sim-only robosuite-composite multiplexer flag, 1-D
    # value per horizon step in [-1, +1].
    composite_mode: list[float] | None = None
    # metadata
    confidence: float = 1.0
    stamp_ns: int = 0
    ee_name: str | None = None
    frame_id: str | None = None
    safety_overrides: dict[str, object] = Field(default_factory=dict)


# ─── Compute / Quantization ────────────────────────────────────────────────────


class QuantizationDtype(str, Enum):
    """Numeric format used to represent model weights and activations.

    Attributes:
        FP32: 32-bit float (full precision, reference).
        FP16: 16-bit float (GPU-native, common default).
        BF16: Brain-float16 (Ampere+ GPUs, TPUs; better dynamic range than FP16).
        INT8: 8-bit integer (CPU/GPU; good accuracy/speed trade-off).
        INT4: 4-bit integer (edge / memory-constrained; some accuracy loss).
        FP4_NVFP4: NVIDIA FP4 format (Hopper+; highest throughput).

    Example:
        >>> QuantizationDtype.INT8.value
        'int8'
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    FP4_NVFP4 = "fp4_nvfp4"


class QuantizationBackend(str, Enum):
    """Inference backend that will execute the quantized model.

    Attributes:
        PYTORCH: ``torch.quantization`` / ``bitsandbytes`` / ``torchao``.
        ONNX: ``onnxruntime`` with quantization pre-applied at export.
        TENSORRT: NVIDIA TensorRT engine (INT8 / FP8 calibrated).
        GGUF: llama.cpp / ggml GGUF format (CPU and Metal).
        MLX: Apple MLX framework (Apple Silicon only).

    Example:
        >>> QuantizationBackend.PYTORCH.value
        'pytorch'
    """

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    GGUF = "gguf"
    MLX = "mlx"


class QuantizationConfig(BaseModel):
    """Full specification of how to quantize a skill's model weights.

    Attributes:
        dtype: Target numeric format.
        backend: Inference backend that will execute the model.
        per_channel: Use per-channel (vs per-tensor) quantization.
        calibration_dataset: HuggingFace dataset ID or local path used for
            post-training calibration (INT8 / TensorRT only).
        extra: Backend-specific overrides (e.g. ``{"calibration_steps": 128}``).

    Example:
        >>> cfg = QuantizationConfig(dtype=QuantizationDtype.INT8)
        >>> cfg.backend
        <QuantizationBackend.PYTORCH: 'pytorch'>
    """

    dtype: QuantizationDtype = QuantizationDtype.FP32
    backend: QuantizationBackend = QuantizationBackend.PYTORCH
    per_channel: bool = False
    calibration_dataset: str | None = None
    extra: dict[str, object] = Field(default_factory=dict)


class DeviceInfo(BaseModel):
    """Snapshot of the host compute capabilities used for runtime selection.

    Attributes:
        device_str: PyTorch-style device string (``"cpu"``, ``"cuda:0"``, ``"mps"``).
        gpu_memory_bytes: Total GPU VRAM in bytes (0 if CPU-only).
        cuda_compute_capability: CUDA compute capability major/minor pair.
        cpu_count: Logical CPU count.
        arch: CPU architecture string.

    Example:
        >>> info = DeviceInfo(device_str="cpu")
        >>> info.gpu_memory_bytes
        0
    """

    device_str: str = "cpu"
    gpu_memory_bytes: int = 0
    cuda_compute_capability: tuple[int, int] | None = None
    cpu_count: int = 1
    arch: str = "x86_64"


# ─── Skill lifecycle ────────────────────────────────────────────────────────────


class RSkillState(str, Enum):
    """Primary lifecycle states for a Skill node.

    Matches the ROS 2 Managed Node state machine with openral extensions.

    States
    ------
    ``unconfigured``
        Initial state.  Weights are not loaded.
    ``inactive``
        Configured and weights loaded; not yet warmed up.  Accepts no actions.
    ``active``
        Ready to execute.  ``step()`` may be called.
    ``finalized``
        Terminal state after ``shutdown()``.  Cannot transition further.
    ``error``
        Unrecoverable failure.  Requires external intervention.
    """

    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"
    ERROR = "error"


class RSkillInfo(BaseModel):
    """Snapshot of a Skill's runtime state — published on ``/skill/<name>/info``.

    Attributes:
        name: rSkill name (e.g. ``"noop_skill"``).
        version: SemVer string.
        state: Current primary lifecycle state.
        weights_loaded: Whether model weights have been loaded into memory.
        quantized: Whether weights have been quantized for the target runtime.
        warmed_up: Whether the model has been warmed up (first inference run).
        embodiment_tags: Embodiment tags from the skill manifest.
        role: rSkill role: ``"s0"`` (cerebellar), ``"s1"`` (fast policy),
            ``"s2"`` (slow reasoning).
        latency_budget_ms: Maximum allowed inference latency in milliseconds.
        last_inference_ms: Actual latency of the most recent ``step()`` call.
        error_msg: Human-readable error description when ``state == "error"``.
        stamp_ns: Timestamp of this snapshot in nanoseconds.

    Example:
        >>> info = RSkillInfo(name="hello", version="0.1.0", state=RSkillState.UNCONFIGURED)
        >>> info.weights_loaded
        False
        >>> info.state
        <RSkillState.UNCONFIGURED: 'unconfigured'>
    """

    name: str
    version: str = "0.1.0"
    state: RSkillState = RSkillState.UNCONFIGURED
    weights_loaded: bool = False
    quantized: bool = False
    warmed_up: bool = False
    embodiment_tags: list[str] = Field(default_factory=list)
    role: Literal["s0", "s1", "s2"] = "s1"
    latency_budget_ms: float | None = None
    last_inference_ms: float | None = None
    error_msg: str | None = None
    stamp_ns: int = 0


# ─── rSkill package manifest ──────────────────────────────────────────────────
#
# An ``rSkill`` is the *packaged, signed, capability-tagged distribution
# format* for a robot skill (CLAUDE.md §6.4 / RFC §1.4, §8.7).  One HF Hub
# repo per rSkill, containing weights + ``rskill.yaml`` + optional engine
# files + ``README.md`` with a runnable example.
#
# This is **distinct from** the runtime ``Skill`` ABC (see
# ``openral_rskill.Skill``) and the ``RSkillInfo`` runtime snapshot above:
#   - ``Skill``         : in-process lifecycle node (S0/S1/S2 ABC).
#   - ``RSkillInfo``     : runtime state snapshot of a live ``Skill``.
#   - ``RSkillManifest``: on-disk / Hub-side package descriptor.


class RSkillLicensePosture(str, Enum):
    """License posture surfaced at install time (CLAUDE.md §7.4)."""

    APACHE_2_0 = "apache-2.0"
    MIT = "mit"
    BSD = "bsd"
    PERMISSIVE_RESEARCH = "permissive_research"  # e.g. pi0 weights
    NVIDIA_NON_COMMERCIAL = "nvidia_non_commercial"  # GR00T N1 / N1.5 / N1.6 weights
    NVIDIA_OPEN_MODEL = "nvidia_open_model"  # GR00T N1.7+ — Open Model License, commercial OK
    RLWRLD_NON_COMMERCIAL = "rlwrld_non_commercial"  # RLDX-1 weights
    PROPRIETARY = "proprietary"  # Helix, Skild, Gemini Robotics
    UNKNOWN = "unknown"


class RSkillRuntime(str, Enum):
    """Inference runtime hint declared in the manifest (RFC §7.3)."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TRT_LLM = "trt_llm"
    VLLM = "vllm"
    GGUF = "gguf"
    MLX = "mlx"
    JAX = "jax"


class RSkillLatencyBudget(BaseModel):
    """Per-stage latency budget declared in ``rskill.yaml``.

    Attributes:
        per_chunk_ms: End-to-end ``step()`` budget.  CI fails if exceeded on
            the reference host (CLAUDE.md §7.4).
        warmup_ms: Maximum allowed warm-up time during ``activate()``.
        load_ms: Maximum allowed weight-load time during ``configure()``.
    """

    per_chunk_ms: float = Field(gt=0)
    warmup_ms: float | None = Field(default=None, gt=0)
    load_ms: float | None = Field(default=None, gt=0)


class SensorRequirement(BaseModel):
    """One sensor an rSkill needs the robot to provide.

    Used by :class:`RSkillManifest.sensors_required` to declare the inputs
    the policy expects. The compatibility check resolves each entry against
    a :class:`RobotDescription`'s ``sensors`` list and rejects the pairing
    if no robot sensor satisfies the requirement.

    Resolution rules (in order):

    1. If ``vla_feature_key`` is set, the robot MUST expose exactly one
       sensor with that ``vla_feature_key``. The check then verifies
       ``modality`` matches and (if specified) the sensor's intrinsics meet
       ``min_width`` / ``min_height``.
    2. Otherwise, the robot must expose at least ``count`` sensors of the
       requested ``modality`` (each meeting any specified resolution
       minimum).

    Attributes:
        modality: Required physical modality (``rgb``, ``depth``, ``imu``…).
        vla_feature_key: Optional exact key the VLA expects, e.g.
            ``"observation.images.camera1"``. When set, the robot's matching
            ``SensorSpec.vla_feature_key`` must be identical.
        min_width: Minimum image width in pixels (RGB/depth/IR/stereo only).
        min_height: Minimum image height in pixels.
        count: Number of robot sensors of this modality required when
            ``vla_feature_key`` is unset. Ignored when ``vla_feature_key``
            is provided (a key uniquely identifies one sensor).

    Example:
        >>> SensorRequirement(
        ...     modality=SensorModality.RGB,
        ...     vla_feature_key="observation.images.camera1",
        ...     min_width=224,
        ...     min_height=224,
        ... )  # doctest: +ELLIPSIS
        SensorRequirement(modality=<SensorModality.RGB: 'rgb'>, ...)
    """

    model_config = ConfigDict(extra="forbid")

    modality: SensorModality
    vla_feature_key: str | None = None
    min_width: int | None = Field(default=None, gt=0)
    min_height: int | None = Field(default=None, gt=0)
    count: int = Field(default=1, ge=1)


class ImagePreprocessing(BaseModel):
    """Per-rSkill image preprocessing contract.

    Properties of how the *checkpoint* was trained against image frames,
    surfaced on the manifest so the sim adapter does not have to learn
    them from a YAML override. ``vla.extra`` overrides on the eval
    config still win — see ``openral_rskill._vla_core.resolve_image_preprocessing``.

    Attributes:
        flip_180: Apply ``torch.flip(t, dims=[1, 2])`` (H+W reversal) to
            every camera frame before the policy forward. SmolVLA / pi05
            LIBERO checkpoints want this on; the RoboCasa checkpoints
            published by RoMALab / DAVIAN-Robotics want it off.
        flip_vertical: Apply ``img[::-1, :, :]`` (vertical-only, H-axis
            reversal) BEFORE any 180° rotation. Mirrors
            ``robocasa.wrappers.gym_wrapper.RoboCasaGymEnv.process_img``,
            which the canonical openpi-robocasa eval applies before
            feeding frames to the policy. Required for
            ``robocasa/robocasa365_checkpoints/pi05_pretrain_human300``
            and its lerobot-converted siblings; the RoMALab MG_300
            checkpoint does NOT want this flip.
        input_template: Format string used to name image tensors in the
            policy input batch — for example ``"observation.images.{cam}"``
            for SmolVLA / pi05 RoMALab vs ``"observation.image.{cam}"``
            for ruiname/pi05-robocasa-10tasks-200k.
        aliases: Per-checkpoint rename map from the *scene* / *robot* raw
            camera key (e.g. robosuite's ``robot0_agentview_left_image``)
            to the *model*'s expected input feature name (e.g.
            ``agentview``). Empty means pass through unchanged. SmolVLA
            LIBERO usually wants ``{"camera1": "image", "camera2": "image2"}``.
        norm_tag: Normalization statistics tag for policies that use
            multiple checkpoints or training distributions (e.g. MolmoAct2
            with ``norm_tag="so100_so101_molmoact2"`` for SO-100/101
            finetuned weights). Optional; ``None`` means the adapter's
            default tag applies. Used by ``predict_action`` methods to
            select the right norm_stats entry at inference time.
        image_max_crops: Cap on the number of image tiles a multi-crop
            image processor (Molmo / MolmoAct2 family) produces per camera
            frame. Each extra 378px crop adds ~182 pooled image tokens and
            attention cost is quadratic in the token count, so this is the
            primary *activation*-memory lever (the NF4 weights are fixed at
            ~3.5 GiB; what overflows an 8 GiB card is the crop activations).
            Optional; ``None`` keeps the checkpoint's own default (8 for
            MolmoAct2). A per-rollout ``vla.extra["image_max_crops"]`` (or
            the ``OPENRAL_MOLMOACT2_MAX_CROPS`` env) still overrides this
            per-checkpoint default. Must be ``>= 1`` when set.
    """

    model_config = ConfigDict(extra="forbid")

    flip_180: bool = False
    flip_vertical: bool = False
    input_template: str = "observation.images.{cam}"
    aliases: dict[str, str] = Field(default_factory=dict)
    norm_tag: str | None = None
    image_max_crops: int | None = Field(default=None, ge=1)


StateLayout: TypeAlias = Literal[
    "smolvla_9d",
    "human300_16d",
    "gr1",
    "rc365",
    # RLDX-1 SimplerEnv layouts (ADR-0014 amendment 2026-05-22).
    # ``simpler_widowx`` matches RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX's
    # ``bridge_orig`` modality config (8 scalar state keys, single
    # ``video.image_0`` camera, Bridge-data orientation rotation).
    # ``simpler_google`` matches RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE's
    # ``fractal20220817_data`` modality config (8 scalar state keys
    # including a 4-D quaternion in ``state.r{x,y,z,w}``, single
    # ``video.image`` camera, sticky-gripper postprocessing).
    "simpler_widowx",
    "simpler_google",
]
"""Closed set of per-checkpoint proprioception layouts. ADR-0014 + ADR-0027.

A layout names the SHAPE the checkpoint was trained on — field order,
frame convention, gripper encoding, quaternion handedness. The per-robot
SOURCE bindings (which TF frame is "the EE", which joint names are "the
gripper") live on :class:`StateContractBindings`. The
``openral_state_adapter`` registry maps each literal to an assembler
function that joins shape + bindings + live JointState + live TF.
"""


WRAPPED_TASK_SPACE_LAYOUTS: frozenset[StateLayout] = frozenset(
    {"rc365", "human300_16d"},
)
"""Layouts that are TASK-space composites (Cartesian poses + gripper widths),
NOT one-scalar-per-joint. These layouts REQUIRE
:attr:`StateContract.bindings` to name the source TF frames + JointState
entries — the cross-validator on :class:`StateContract` enforces this at
manifest load. The remaining layouts (``smolvla_9d``, ``gr1``,
``simpler_*``) are joint-space slices (potentially across multiple
controller groups, as in GR1's 29-D waist+arms+hands composite) that the
runner serves verbatim from ``observation.joint_state.position``; a
robot.yaml with the matching joint count dispatches them without an
assembler.
"""


class StateContractBindings(BaseModel):
    """Per-robot source bindings for an rSkill's `state_contract.layout`. ADR-0027.

    Symmetric to :class:`ControlModeSemantics` on the action side
    (``joint_order`` + ``reference_frame`` + ``gripper_convention``):
    the rSkill manifest names the *shape* via :attr:`StateContract.layout`;
    these bindings name the *sources* on the deploying robot. The
    layout-adapter registry (``openral_state_adapter``) joins shape +
    bindings + live :class:`sensor_msgs/JointState` + live ``/tf`` into
    the per-checkpoint state vector.

    Bindings are required for layouts in :data:`WRAPPED_TASK_SPACE_LAYOUTS`
    and forbidden otherwise (the joint-space layouts have no source to
    parameterise).

    Attributes:
        eef_frame: tf2 link name of the end effector
            (e.g. ``"panda_hand"``). Required for any layout reading
            EE position / orientation.
        base_frame: tf2 link name of the mobile base
            (e.g. ``"base_link"``). Required for any layout reading
            base position / orientation OR base-relative EE poses.
        world_frame: tf2 root frame the base pose is expressed in
            (default ``"map"`` — slam_toolbox publishes ``map → odom →
            base_link``). Set to ``"odom"`` for deployments without SLAM.
        gripper_qpos_joints: ``JointState.name`` entries whose positions
            populate the gripper-width slot(s) of the layout, in the
            order the policy expects (e.g. ``["panda_finger_joint1",
            "panda_finger_joint2"]``). Empty for grasper-less robots
            or layouts without a gripper slot.
        quaternion_convention: Component order of any quaternion in the
            assembled vector. ROS / TF2 default is ``"xyzw"`` (the
            ``geometry_msgs/Quaternion`` field order). Some upstream
            checkpoints expect ``"wxyz"`` — declare it here so the
            assembler permutes once, at the boundary.
    """

    model_config = ConfigDict(extra="forbid")

    eef_frame: str | None = None
    base_frame: str | None = None
    world_frame: str | None = "map"
    gripper_qpos_joints: list[str] = Field(default_factory=list)
    quaternion_convention: Literal["xyzw", "wxyz"] = "xyzw"


class StateContract(BaseModel):
    """Per-rSkill state-vector contract.

    Surfaces the proprioception layout the *checkpoint* was trained
    against so the runtime adapter does not have to learn it from a
    YAML override. ADR-0014 + ADR-0027.

    Attributes:
        layout: Named proprioception layout — see :data:`StateLayout`.
            LIBERO / MetaWorld / pusht / aloha leave this ``None`` and
            consume the raw joint-position vector directly.
        dim: Explicit state dimension override. The runtime adapter
            clips or pads the env state vector to this width before
            handing it to the policy.
        bindings: Per-robot source bindings — TF frame names + JointState
            entries the runtime adapter pulls values from. REQUIRED when
            ``layout`` is in :data:`WRAPPED_TASK_SPACE_LAYOUTS`,
            FORBIDDEN otherwise (joint-space layouts have no source to
            parameterise).
    """

    model_config = ConfigDict(extra="forbid")

    layout: StateLayout | None = None
    dim: int | None = Field(default=None, gt=0)
    bindings: StateContractBindings | None = None

    @model_validator(mode="after")
    def _validate_bindings(self) -> StateContract:
        if self.layout in WRAPPED_TASK_SPACE_LAYOUTS:
            if self.bindings is None:
                raise ValueError(
                    f"StateContract.layout={self.layout!r} is a wrapped task-space "
                    "layout and REQUIRES `bindings` to name the per-robot TF "
                    "frames + JointState entries. See ADR-0027.",
                )
            # Layout-specific binding requirements (the registry's
            # assemblers read these; if absent the assembler would raise
            # at runtime — surface the missing-field at manifest load).
            if self.layout in {"human300_16d", "rc365"}:
                missing = [
                    name
                    for name in ("eef_frame", "base_frame")
                    if getattr(self.bindings, name) is None
                ]
                if missing:
                    raise ValueError(
                        f"StateContract.layout={self.layout!r} requires "
                        f"bindings.{', bindings.'.join(missing)} — these "
                        f"layouts include EE and base poses.",
                    )
        elif self.bindings is not None:
            raise ValueError(
                f"StateContract.layout={self.layout!r} is a joint-space "
                "layout; `bindings` must be omitted (it would have no effect).",
            )
        return self


class ActionSlot(BaseModel):
    """One contiguous slice of an rSkill's action vector (ADR-0028b).

    The skill_runner reads ``ActionContract.slots`` and emits one
    typed :class:`Action` per non-discard slot per step. All actions
    inherit the parent step's ``trace_id`` so the safety supervisor
    and downstream telemetry can join them post-hoc.

    Attributes:
        range: Inclusive ``[start, end]`` indices into the flat policy
            action vector. ``range[0]`` must be ≤ ``range[1]``; both
            must fall within ``[0, ActionContract.dim)``.
        control_mode: The :class:`ControlMode` the slice is routed to.
            The HAL whitelist on the target robot must include this
            mode (the palette filter rejects the rSkill at install
            time otherwise). ``None`` only when :attr:`discard` is
            ``True``.
        discard: When ``True`` the slice is dropped silently — used
            for dataset artefacts like RoboCasa365's torso
            placeholder dim or paired gripper channels. The slot
            still occupies its range so coverage validation works;
            no :class:`Action` is emitted.
        ee: End-effector name from the robot's
            :attr:`RobotDescription.end_effectors` /
            :attr:`RobotDescription.joints`. REQUIRED for
            ``CARTESIAN_*`` modes (the pose is computed in the named
            EE's frame) and for ``GRIPPER_*`` modes (names the
            actuator). FORBIDDEN for ``BODY_TWIST`` and
            ``JOINT_POSITION``.
        frame: tf2 frame name. REQUIRED for cartesian + body-twist
            modes (the slice's bytes are expressed in this frame).
            FORBIDDEN for joint-position and gripper modes.
        joint_names: Robot joint names this slice targets, in slot
            order. REQUIRED for ``JOINT_POSITION`` / ``JOINT_VELOCITY``
            / ``JOINT_TORQUE`` when the slot covers fewer than all
            robot joints; FORBIDDEN for non-joint modes. Length must
            equal ``range[1] - range[0] + 1``.
    """

    model_config = ConfigDict(extra="forbid")

    range: tuple[int, int]
    control_mode: ControlMode | None = None
    discard: bool = False
    ee: str | None = None
    frame: str | None = None
    joint_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_slot(self) -> ActionSlot:  # noqa: PLR0912, PLR0915  # reason: validates each control-mode slot's field requirements; one branch per mode family
        lo, hi = self.range
        if lo > hi:
            raise ValueError(f"ActionSlot.range must satisfy start <= end; got [{lo}, {hi}]")
        if lo < 0:
            raise ValueError(f"ActionSlot.range start must be >= 0; got {lo}")
        if self.discard:
            if self.control_mode is not None:
                raise ValueError(
                    "ActionSlot: discard=True is mutually exclusive with control_mode "
                    f"(got {self.control_mode!r})"
                )
            if self.ee is not None or self.frame is not None or self.joint_names:
                raise ValueError(
                    "ActionSlot: discard=True forbids ee / frame / joint_names "
                    "(no routing target needed for a discarded slice)"
                )
            return self
        if self.control_mode is None:
            raise ValueError("ActionSlot: control_mode is required when discard is False")
        # Per-mode field requirements (ADR-0028b).
        mode = self.control_mode
        width = hi - lo + 1
        if mode in _JOINT_MODES:
            if self.ee is not None:
                raise ValueError(f"ActionSlot[{mode.value}]: ee is forbidden")
            if self.frame is not None:
                raise ValueError(f"ActionSlot[{mode.value}]: frame is forbidden")
            if self.joint_names and len(self.joint_names) != width:
                raise ValueError(
                    f"ActionSlot[{mode.value}]: joint_names length "
                    f"({len(self.joint_names)}) must equal slot width ({width})"
                )
        elif mode in _CARTESIAN_MODES:
            if self.ee is None:
                raise ValueError(f"ActionSlot[{mode.value}]: ee is required")
            if self.frame is None:
                raise ValueError(f"ActionSlot[{mode.value}]: frame is required")
            if self.joint_names:
                raise ValueError(f"ActionSlot[{mode.value}]: joint_names is forbidden")
        elif mode is ControlMode.BODY_TWIST:
            if self.ee is not None:
                raise ValueError("ActionSlot[body_twist]: ee is forbidden")
            if self.frame is None:
                raise ValueError("ActionSlot[body_twist]: frame is required")
            if self.joint_names:
                raise ValueError("ActionSlot[body_twist]: joint_names is forbidden")
        elif mode in _GRIPPER_MODES:
            if self.ee is None:
                raise ValueError(f"ActionSlot[{mode.value}]: ee is required")
            if self.frame is not None:
                raise ValueError(f"ActionSlot[{mode.value}]: frame is forbidden")
            if self.joint_names:
                raise ValueError(f"ActionSlot[{mode.value}]: joint_names is forbidden")
        elif mode is ControlMode.COMPOSITE_MODE:
            # ADR-0028d — sim-only multiplexer flag, 1-D, no ee/frame/joints.
            if width != 1:
                raise ValueError(f"ActionSlot[composite_mode]: slot width must be 1; got {width}")
            if self.ee is not None:
                raise ValueError("ActionSlot[composite_mode]: ee is forbidden")
            if self.frame is not None:
                raise ValueError("ActionSlot[composite_mode]: frame is forbidden")
            if self.joint_names:
                raise ValueError("ActionSlot[composite_mode]: joint_names is forbidden")
        return self


_JOINT_MODES: frozenset[ControlMode] = frozenset(
    {ControlMode.JOINT_POSITION, ControlMode.JOINT_VELOCITY, ControlMode.JOINT_TORQUE}
)
_CARTESIAN_MODES: frozenset[ControlMode] = frozenset(
    {ControlMode.CARTESIAN_POSE, ControlMode.CARTESIAN_DELTA, ControlMode.CARTESIAN_TWIST}
)
_GRIPPER_MODES: frozenset[ControlMode] = frozenset(
    {ControlMode.GRIPPER_BINARY, ControlMode.GRIPPER_POSITION}
)


class ActionContract(BaseModel):
    """Per-rSkill action-vector contract (ADR-0019 PR-revert).

    Mirrors :class:`StateContract` for the action side. Carries the
    output dimensionality the checkpoint emits so the dataset bridge
    (and any downstream consumer) can bind the LeRobot v3 ``action``
    feature shape without consulting the sim or hardware adapter.

    Per ADR-0007, the sim-specific action contract belongs on the
    per-checkpoint rSkill manifest, not on the physical
    :class:`RobotDescription` (the same Franka emits 7-D delta-EEF on
    LIBERO vs 8-D joint pos on a hardware deploy).

    Attributes:
        dim: Dimensionality of the action vector emitted by
            ``Skill.step()`` / ``PolicyAdapter.step()``. Required when
            this contract is set.
        representation: Optional named representation (mirrors
            :attr:`ActionSpec.representation`). When set, downstream
            consumers can map between equivalent representations
            (e.g. ``joint_positions`` → ``delta_ee_6d_plus_gripper``).
        slots: ADR-0028b — declarative slot layout. When set, every
            index in ``[0, dim)`` must be covered by exactly one
            :class:`ActionSlot` (no gaps, no overlaps). The
            skill_runner reads this to dispatch slices of the policy
            vector onto typed :class:`Action` objects. When ``None``,
            the runner falls back to the legacy single-Action path
            (one implicit ``JOINT_POSITION`` slot covering the whole
            vector). Manifests carrying ``slots`` are exempt from the
            ADR-0028a ``dim <= len(robot.joints)`` invariant because
            the slot decoder gives a typed contract per slice instead.
    """

    model_config = ConfigDict(extra="forbid")

    dim: int = Field(gt=0)
    representation: ActionRepresentation | None = None
    slots: list[ActionSlot] | None = None

    @model_validator(mode="after")
    def _validate_slots_cover_dim(self) -> ActionContract:
        if self.slots is None:
            return self
        if not self.slots:
            raise ValueError(
                "ActionContract.slots: must be omitted (None) or a non-empty list; "
                "empty lists silently lose the whole policy vector"
            )
        # Range bounds vs dim.
        for slot in self.slots:
            lo, hi = slot.range
            if hi >= self.dim:
                raise ValueError(
                    f"ActionContract.slots: slot range [{lo}, {hi}] exceeds "
                    f"action_contract.dim={self.dim}"
                )
        # Coverage: every index in [0, dim) appears in exactly one slot.
        covered: list[int] = [0] * self.dim
        for slot in self.slots:
            lo, hi = slot.range
            for i in range(lo, hi + 1):
                covered[i] += 1
        missing = [i for i, n in enumerate(covered) if n == 0]
        overlapping = [i for i, n in enumerate(covered) if n > 1]
        if missing:
            raise ValueError(
                f"ActionContract.slots: indices {missing!r} are not covered by any "
                f"slot (dim={self.dim}). Declare a discard slot for unused channels."
            )
        if overlapping:
            raise ValueError(
                f"ActionContract.slots: indices {overlapping!r} are covered by "
                "multiple slots; ranges must be disjoint."
            )
        return self


# ADR-0036 — representation → ControlMode + canonical slot layout. The
# single source of truth shared by the skill_runner (action dispatch)
# and the reasoner (deploy-path palette gate): given a VLA's declared
# ``ActionRepresentation`` we derive (a) which ``ControlMode`` s the
# target robot must advertise, and (b) the typed ``ActionSlot`` layout
# the runner dispatches the flat policy vector through.
_EE_6D_WIDTH = 6  # (dx, dy, dz, drx, dry, drz) — 6-DoF cartesian slice.


def control_modes_for_representation(rep: ActionRepresentation) -> set[ControlMode]:
    """Map an :class:`ActionRepresentation` to the :class:`ControlMode` s it drives.

    ADR-0036. Used by the reasoner's deploy-path palette gate: a skill is
    only offered when the target robot advertises *every* mode in the
    returned set.

    Args:
        rep: The VLA's declared action-vector representation.

    Returns:
        The set of :class:`ControlMode` s the representation maps onto.

    Example:
        >>> control_modes_for_representation(ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER) == {
        ...     ControlMode.CARTESIAN_DELTA,
        ...     ControlMode.GRIPPER_POSITION,
        ... }
        True
    """
    if rep is ActionRepresentation.JOINT_POSITIONS:
        return {ControlMode.JOINT_POSITION}
    if rep is ActionRepresentation.JOINT_VELOCITIES:
        return {ControlMode.JOINT_VELOCITY}
    if rep is ActionRepresentation.DELTA_EE_6D:
        return {ControlMode.CARTESIAN_DELTA}
    if rep is ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER:
        return {ControlMode.CARTESIAN_DELTA, ControlMode.GRIPPER_POSITION}
    # CARTESIAN_POSE — the only remaining enum member.
    return {ControlMode.CARTESIAN_POSE}


# ADR-0036 (amended 2026-06-04) — the canonical set of ControlModes the
# DEFAULT sim HAL action-packers can actually execute, the single source
# of truth for the reasoner's ``hal_mode="sim"`` palette gate (see
# ``openral_reasoner_ros.reasoner_node._action_executable``).
#
# This is the *default-sim-packer contract*. The packers in
# ``python/hal/src/openral_hal/sim_attached.py`` are pinned to this exact
# set — every member here is handled by at least one packer path, and no
# packer handles a mode that is not here. The lockstep is enforced (both
# directions) by ``tests/unit/test_sim_executable_modes_match_packers.py``.
#
# Provenance of each member:
#   * JOINT_POSITION  — ``pack_action_for_env`` + ``_pack_with_composite_split``.
#   * JOINT_VELOCITY  — ``_pack_with_composite_split`` (HybridMobileBase base part).
#   * CARTESIAN_DELTA — both packers (robosuite OSC arm slot).
#   * GRIPPER_POSITION— both packers (gripper slot).
#   * COMPOSITE_MODE  — ``_pack_with_composite_split`` (HybridMobileBase multiplexer flag).
#   * BODY_TWIST      — intercepted in ``SimAttachedHAL.send_action`` and applied
#                       via ``_apply_body_twist_to_qpos`` (direct base-qpos write,
#                       NOT through a packer else-branch).
#
# Modes deliberately EXCLUDED (no sim packer path → would E-stop mid-run):
# JOINT_TORQUE, JOINT_TRAJECTORY, GRIPPER_BINARY (decoded by
# ``openral_hal.lifecycle.decode_action_chunk`` but never pack-executed);
# CARTESIAN_POSE (not even decoded — ``decode_action_chunk`` returns None for
# it, as it carries a Pose6D rather than a flat row); and CARTESIAN_TWIST,
# FOOT_PLACEMENT, DEX_HAND_JOINT (no sim controller at all).
SIM_EXECUTABLE_CONTROL_MODES: frozenset[ControlMode] = frozenset(
    {
        ControlMode.JOINT_POSITION,
        ControlMode.JOINT_VELOCITY,
        ControlMode.CARTESIAN_DELTA,
        ControlMode.GRIPPER_POSITION,
        ControlMode.BODY_TWIST,
        ControlMode.COMPOSITE_MODE,
    }
)


def canonical_slots_for_representation(
    rep: ActionRepresentation,
    *,
    dim: int,
    description: RobotDescription,
) -> list[ActionSlot] | None:
    """Build the canonical :class:`ActionSlot` layout for a representation.

    ADR-0036. The skill_runner calls this to expand a skill that declares
    only ``ActionContract.representation`` (no explicit ``slots``) into a
    typed slot layout it can dispatch. Joint representations return
    ``None`` so the caller keeps the legacy whole-vector ``JOINT_POSITION``
    path; cartesian / gripper representations get one slot per control
    mode, addressed at the robot's primary end-effector
    (``description.end_effectors[0]``).

    :class:`EndEffectorSpec` carries no explicit tf-frame field, so the
    end-effector's ``name`` is used as the slot ``frame`` (the EE name is
    the tf frame the cartesian pose/delta is expressed in for the
    canonical embodiments).

    Args:
        rep: The VLA's declared action-vector representation.
        dim: Dimensionality of the policy's flat action vector.
        description: The target robot — its primary end-effector names
            the cartesian/gripper slots.

    Returns:
        For joint representations: ``None`` (caller keeps the legacy
        whole-vector ``JOINT_POSITION`` path). Otherwise a list of typed
        :class:`ActionSlot` s that satisfy the per-slot validators.

    Raises:
        ROSConfigError: If the representation needs an end-effector but
            ``description.end_effectors`` is empty, or if ``dim`` is too
            small for the layout (``DELTA_EE_6D`` / ``CARTESIAN_POSE``
            need ``dim >= 6``; ``DELTA_EE_6D_PLUS_GRIPPER`` needs
            ``dim >= 7``).

    Example:
        >>> desc = RobotDescription(
        ...     name="ex_robot",
        ...     embodiment_kind=EmbodimentKind.MANIPULATOR,
        ...     joints=[
        ...         JointSpec(
        ...             name="j1",
        ...             joint_type=JointType.REVOLUTE,
        ...             parent_link="base_link",
        ...             child_link="link_1",
        ...         )
        ...     ],
        ...     end_effectors=[EndEffectorSpec(name="ee0", kind="parallel_gripper")],
        ...     capabilities=RobotCapabilities(),
        ...     safety=SafetyEnvelope(),
        ... )
        >>> slots = canonical_slots_for_representation(
        ...     ActionRepresentation.DELTA_EE_6D, dim=6, description=desc
        ... )
        >>> slots[0].control_mode is ControlMode.CARTESIAN_DELTA
        True
    """
    if rep in (ActionRepresentation.JOINT_POSITIONS, ActionRepresentation.JOINT_VELOCITIES):
        # Legacy whole-vector JOINT path — the caller emits one implicit
        # JOINT_POSITION / JOINT_VELOCITY slot covering the whole vector.
        return None

    if not description.end_effectors:
        raise ROSConfigError(
            f"canonical_slots_for_representation: representation {rep.value!r} addresses "
            "an end-effector but the robot description declares no end_effectors; "
            "cannot build a cartesian/gripper slot layout."
        )
    primary = description.end_effectors[0]
    ee_name = primary.name
    # EndEffectorSpec has no explicit tf-frame field; the EE name is the
    # tf frame the cartesian pose/delta is expressed in.
    ee_frame = ee_name

    has_gripper = rep is ActionRepresentation.DELTA_EE_6D_PLUS_GRIPPER
    min_dim = _EE_6D_WIDTH + 1 if has_gripper else _EE_6D_WIDTH
    if dim < min_dim:
        raise ROSConfigError(
            f"canonical_slots_for_representation: representation {rep.value!r} requires "
            f"dim >= {min_dim} but action_contract.dim={dim} is too small for the layout."
        )

    cart_mode = (
        ControlMode.CARTESIAN_POSE
        if rep is ActionRepresentation.CARTESIAN_POSE
        else ControlMode.CARTESIAN_DELTA
    )
    slots: list[ActionSlot] = [
        ActionSlot(
            range=(0, _EE_6D_WIDTH - 1),
            control_mode=cart_mode,
            ee=ee_name,
            frame=ee_frame,
        )
    ]
    if has_gripper:
        slots.append(
            ActionSlot(
                range=(_EE_6D_WIDTH, dim - 1),
                control_mode=ControlMode.GRIPPER_POSITION,
                ee=ee_name,
            )
        )
    return slots


GripperConvention: TypeAlias = Literal[
    "normalized_open_unit",
    "normalized_open_symmetric",
    "binary_close_one",
    "raw_joint_rad",
    "width_meters",
]
"""Per-skill gripper action encoding (Gap 2 of the rSkill self-containment audit).

Required on :class:`ControlModeSemantics` whenever the parent
:class:`ActuatorRequirement.kind` is :attr:`ControlMode.GRIPPER_BINARY` or
:attr:`ControlMode.GRIPPER_POSITION` — silently mis-encoding the gripper
slot between scenes / robots is a top observed mis-actuation failure.

Conventions:

* ``normalized_open_unit`` — ``0.0 = fully closed``, ``1.0 = fully open``.
  Most lerobot / SmolVLA / pi0.5 LIBERO checkpoints.
* ``normalized_open_symmetric`` — ``-1.0 = fully closed``, ``+1.0 = fully open``.
  Some MetaWorld checkpoints.
* ``binary_close_one`` — ``0.0 = open``, ``1.0 = close`` (single bit, RoboCasa-style).
* ``raw_joint_rad`` — raw per-finger joint angle in radians (Fourier dexhands,
  some humanoid skills).
* ``width_meters`` — physical gripper width in metres (Franka FCI native).
"""


class ControlModeSemantics(BaseModel):
    """Action-space semantics declared on each :class:`ActuatorRequirement`.

    Closes Gap 2 of the rSkill self-containment audit: ``kind`` alone
    ("joint_position") doesn't say whether the policy emits absolute targets
    or per-step deltas, in which joint order, in which reference frame, or
    how it encodes the gripper. Today these are assumed at the adapter
    level and silently broken when porting a skill between embodiments.

    The block is REQUIRED on every actuator entry of a manifest. For
    the canonical embodiments the loader still auto-fills ``n_dof`` /
    ``vla_action_key`` from the robot YAML; semantics must be declared
    explicitly on the manifest because they are a property of the *trained
    checkpoint*, not the robot.

    Attributes:
        mode: Whether the action vector represents absolute targets
            (joint positions, end-effector pose) or deltas from the current
            state. Required.
        gripper_convention: Encoding of the gripper slot in the action
            vector. Required when the parent :attr:`ActuatorRequirement.kind`
            is :attr:`ControlMode.GRIPPER_BINARY` or
            :attr:`ControlMode.GRIPPER_POSITION`; forbidden otherwise.
        joint_order: Ordered list of joint names matching the channels
            the policy emits. Optional for canonical embodiments
            (the robot YAML's :attr:`ActionSpec.joint_names` is the
            source of truth); REQUIRED for ``"custom"`` embodiments.
        reference_frame: tf2 frame the action is expressed in. Required
            when the parent :attr:`ActuatorRequirement.kind` is one of
            :attr:`ControlMode.CARTESIAN_POSE`,
            :attr:`ControlMode.CARTESIAN_DELTA`, or
            :attr:`ControlMode.CARTESIAN_TWIST`; forbidden otherwise.

    Example:
        >>> sem = ControlModeSemantics(mode="absolute")
        >>> sem.mode
        'absolute'
        >>> sem.gripper_convention is None and sem.reference_frame is None
        True
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["absolute", "delta"]
    gripper_convention: GripperConvention | None = None
    joint_order: list[str] | None = None
    reference_frame: str | None = None


_GRIPPER_KINDS: frozenset[ControlMode] = frozenset(
    {ControlMode.GRIPPER_BINARY, ControlMode.GRIPPER_POSITION}
)
_CARTESIAN_KINDS: frozenset[ControlMode] = frozenset(
    {ControlMode.CARTESIAN_POSE, ControlMode.CARTESIAN_DELTA, ControlMode.CARTESIAN_TWIST}
)


class ActuatorRequirement(BaseModel):
    """One actuator slot an rSkill emits actions for.

    Symmetric with :class:`SensorRequirement` on the action side.
    The compatibility check resolves each entry against a
    :class:`RobotDescription`'s ``action_spec`` and rejects pairings
    whose declared :class:`ControlMode` is not advertised by the robot.

    For predefined embodiments (one of the 9 canonical
    ``robots/<id>/robot.yaml`` slugs), ``n_dof`` and ``vla_action_key``
    are optional — the loader auto-fills them from the robot YAML at
    compatibility-check time. For the ``"custom"`` embodiment escape
    hatch they MUST be set on the manifest; the
    :class:`RSkillManifest` cross-validator enforces this.

    Attributes:
        kind: The action interface the skill emits (e.g.
            ``ControlMode.JOINT_POSITION``,
            ``ControlMode.CARTESIAN_DELTA``). Reuses the canonical
            ``ControlMode`` enum so the manifest, the safety kernel,
            ``Action.control_mode``, and ``RobotDescription.action_spec``
            all share one source of truth.
        n_dof: Optional degrees of freedom the skill emits. Auto-filled
            from the robot YAML for predefined embodiments; REQUIRED on
            the manifest when ``"custom"`` is in
            ``RSkillManifest.embodiment_tags``.
        vla_action_key: Optional slot name the policy emits, e.g.
            ``"action.joints.arm_left"``. Auto-filled for predefined
            embodiments; REQUIRED on the manifest when ``"custom"`` is
            in ``RSkillManifest.embodiment_tags``.
        control_mode_semantics: Action-space semantics for this slot.
            REQUIRED — closes Gap 2 of the rSkill self-containment audit.
            Cross-validators enforce: gripper kinds require
            ``gripper_convention``; cartesian kinds require
            ``reference_frame``; other kinds forbid both.

    Example:
        >>> a = ActuatorRequirement(
        ...     kind=ControlMode.JOINT_POSITION,
        ...     control_mode_semantics=ControlModeSemantics(mode="absolute"),
        ... )
        >>> a.kind is ControlMode.JOINT_POSITION
        True
        >>> a.n_dof is None and a.vla_action_key is None
        True
    """

    model_config = ConfigDict(extra="forbid")

    kind: ControlMode
    n_dof: int | None = Field(default=None, gt=0)
    vla_action_key: str | None = None
    control_mode_semantics: ControlModeSemantics

    @model_validator(mode="after")
    def _check_semantics_per_kind(self) -> ActuatorRequirement:
        """Enforce per-kind semantics rules (Gap 2).

        Three rules:

        1. Gripper kinds (``GRIPPER_BINARY`` / ``GRIPPER_POSITION``) require
           ``gripper_convention`` to be set; non-gripper kinds forbid it.
        2. Cartesian kinds (``CARTESIAN_POSE`` / ``CARTESIAN_DELTA`` /
           ``CARTESIAN_TWIST``) require ``reference_frame``; non-cartesian
           kinds forbid it.
        3. ``mode`` itself is always required (enforced by typing).
        """
        sem = self.control_mode_semantics
        is_gripper = self.kind in _GRIPPER_KINDS
        is_cartesian = self.kind in _CARTESIAN_KINDS

        if is_gripper and sem.gripper_convention is None:
            raise ValueError(
                f"ActuatorRequirement(kind={self.kind.value!r}) requires "
                "control_mode_semantics.gripper_convention to be set "
                "(gripper kinds need an explicit encoding declaration)."
            )
        if not is_gripper and sem.gripper_convention is not None:
            raise ValueError(
                f"ActuatorRequirement(kind={self.kind.value!r}) must not declare "
                "control_mode_semantics.gripper_convention — only GRIPPER_* "
                "kinds carry a gripper encoding."
            )
        if is_cartesian and sem.reference_frame is None:
            raise ValueError(
                f"ActuatorRequirement(kind={self.kind.value!r}) requires "
                "control_mode_semantics.reference_frame to be set "
                "(cartesian kinds need an explicit tf2 frame)."
            )
        if not is_cartesian and sem.reference_frame is not None:
            raise ValueError(
                f"ActuatorRequirement(kind={self.kind.value!r}) must not declare "
                "control_mode_semantics.reference_frame — only CARTESIAN_* "
                "kinds carry a reference frame."
            )
        return self


class EmbodimentExtra(BaseModel):
    """Sensor + actuator surface for a ``"custom"`` embodiment.

    Required when ``"custom"`` appears in
    :attr:`RSkillManifest.embodiment_tags`; forbidden otherwise. This
    is the explicit "I know what I'm doing" hatch for skill manifests
    targeting embodiments that do not have a canonical
    ``robots/<id>/robot.yaml`` in tree.

    Reuses :class:`SensorRequirement` and :class:`ActuatorRequirement`
    on the assumption that the natural shape of "what this embodiment
    offers" mirrors "what the skill needs" — the loader just runs the
    same compat check it would run against a real robot YAML.

    Attributes:
        sensors: Sensor slots this custom embodiment exposes (≥1).
        actuators: Actuator slots this custom embodiment exposes (≥1).

    Example:
        >>> from openral_core import SensorModality, ControlMode
        >>> EmbodimentExtra(
        ...     sensors=[SensorRequirement(modality=SensorModality.RGB)],
        ...     actuators=[
        ...         ActuatorRequirement(
        ...             kind=ControlMode.JOINT_POSITION,
        ...             n_dof=6,
        ...             vla_action_key="action.joints.arm",
        ...             control_mode_semantics=ControlModeSemantics(mode="absolute"),
        ...         )
        ...     ],
        ... )  # doctest: +ELLIPSIS
        EmbodimentExtra(sensors=[...], actuators=[...])
    """

    model_config = ConfigDict(extra="forbid")

    sensors: list[SensorRequirement] = Field(min_length=1)
    actuators: list[ActuatorRequirement] = Field(min_length=1)


# ─── rSkill V1 enumerated sets ──────────────────────────────────────────────
#
# Closed Literals enforced by RSkillManifest. Adding a new robot HAL,
# benchmark suite, or VLA family requires editing the matching alias here
# (and shipping the supporting code in the same PR per CLAUDE.md §1.6).

EmbodimentTag: TypeAlias = Literal[
    "aloha",
    "custom",
    "franka_panda",
    "g1",
    "google_robot",
    "gr1",
    "h1",
    "mobile_base",
    "openarm",
    "panda_mobile",
    "pusht",
    "rizon4",
    "sawyer",
    "so100_follower",
    "so101_follower",
    "ur10e",
    "ur5e",
    "widowx",
]
"""Canonical embodiment tags — one per ``robots/<id>/robot.yaml`` shipped in tree,
plus ``"custom"`` as the explicit "I know what I'm doing" escape hatch.

``"mobile_base"`` is a CLASS tag (not a specific robot): any robot with a planar
base + ``body_twist`` actuator declares it so base-only rSkills (Nav2
NavigateToPose, etc.) can target the whole class without naming each specific
mobile platform. Robot-specific tags (e.g. ``"panda_mobile"``) coexist on the
same ``RobotDescription`` for skills that DO depend on the specific composition.

The ``RSkillManifest.embodiment_tags`` field is restricted to this set so a
typo or framework hint (``lerobot``, ``libero``) cannot land in a manifest
where the loader's compat check would silently never match. When
``"custom"`` is used, the manifest MUST also populate
``embodiment_extra`` (see :class:`EmbodimentExtra`); the cross-validator
on :class:`RSkillManifest` enforces this. ADR-0013.
"""

BenchmarkName: TypeAlias = Literal[
    "aloha",
    "aloha_insertion",
    "aloha_transfer_cube",
    "gr1_tabletop",
    "libero_10",
    "libero_goal",
    "libero_object",
    "libero_spatial",
    "maniskill3_franka_pick_cube",
    "maniskill3_panda",
    "maniskill3_pick_place",
    "metaworld_mt50",
    "pusht",
    "robocasa_pnp",
    "simpler_env_widowx",
]
"""Canonical benchmark ids — one per ``benchmarks/<id>.yaml`` suite in tree.

Used as keys in ``RSkillManifest.benchmarks``. Each value is the headline
success rate ``[0.0, 1.0]`` the skill achieves on that suite. The full
breakdown lives in the matching ``rskills/<id>/eval/<key>.json``.

``aloha_insertion`` / ``aloha_transfer_cube`` are retained as task-level ids
(the act-aloha* manifests cite their per-task paper numbers) even though the
two single-task suites were unified into ``aloha.yaml`` — the unified suite
auto-filters per rSkill so a single run scores one ACT checkpoint's one task.
"""

ModelFamily: TypeAlias = Literal[
    "smolvla", "pi05", "xvla", "act", "diffusion", "rldx", "molmoact2", "gr00t", "openvla"
]
"""VLA / policy family the skill belongs to.

Used by the eval / runner adapters to dispatch to the right
``openral_sim.adapters.<family>`` policy adapter without
string-matching the skill name. Adding a family here means landing the
matching adapter under ``python/sim/src/openral_sim/adapters/``.

``gr00t`` (NVIDIA Isaac GR00T N1.x / N2) runs out-of-process via a ZMQ
sidecar in an isolated Python 3.10 venv, sharing the architecture of the
``rldx`` adapter (RLDX-1 is itself a GR00T-N1.5 finetune) — see ADR-0046.

``openvla`` (OpenVLA / OpenVLA-OFT) is a transformers *custom-code* model
loaded in-process (``trust_remote_code``, gated by
``OPENRAL_ALLOW_REMOTE_CODE=1``); the adapter de-normalizes the policy's
discrete action tokens with the checkpoint's embedded ``unnorm_key`` stats
and replays the action chunk closed-loop — see ADR-0061.
"""

# Regexes pinned at module scope so error messages stay consistent and
# the patterns stay greppable.
_HF_HUB_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9._-]+$"
_SEMVER_PATTERN = r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
_WEIGHTS_URI_PATTERN = (
    r"^(?:hf:\/\/[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9._-]+(?:@[A-Za-z0-9._-]+)?"
    r"|local:\/\/[A-Za-z0-9._\/-]+)$"
)
_HF_DATASET_URI_PATTERN = (
    r"^hf:\/\/[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9._-]+(?:@[A-Za-z0-9._-]+)?$"
)
_HTTPS_URL_PATTERN = r"^https?:\/\/[^\s]+$"
# Per-file URI pattern accepted by :class:`RSkillProcessors`. Requires a
# file tail (``/path/to/file.ext``) so the implicit-snapshot shape
# ``hf://owner/repo`` is rejected. The whole point of the processors
# block is to name the artefact, not the repo.
_PROCESSOR_URI_PATTERN = (
    r"^hf:\/\/[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9._-]+(?:@[A-Za-z0-9._-]+)?"
    r"\/[A-Za-z0-9._\/-]+\.[A-Za-z0-9]+$"
)

_LICENSES_ALLOWING_COMMERCIAL: frozenset[RSkillLicensePosture] = frozenset(
    {
        RSkillLicensePosture.APACHE_2_0,
        RSkillLicensePosture.MIT,
        RSkillLicensePosture.BSD,
        # NVIDIA Open Model License (GR00T N1.7+) permits commercial use,
        # unlike the OneWay Noncommercial License on N1 / N1.5 / N1.6.
        RSkillLicensePosture.NVIDIA_OPEN_MODEL,
    }
)

# Model families whose adapters consume the modern lerobot
# ``PolicyProcessorPipeline``. Manifests of these families MUST declare a
# :class:`RSkillProcessors` block; ``act`` may omit it to use the legacy
# norm-stats-in-safetensors path (e.g. ``rskills/act-aloha``).
_MODERN_PROCESSOR_FAMILIES: frozenset[str] = frozenset(
    {"smolvla", "pi05", "xvla", "diffusion", "rldx", "molmoact2"}
)


class RSkillProcessors(BaseModel):
    """Explicit lerobot ``PolicyProcessorPipeline`` artefact pointers.

    Closes Gap 1 + Gap 3 of the rSkill self-containment audit. Replaces the
    implicit ``snapshot_download(repo_id)`` →
    ``make_pre_post_processors(pretrained_path=...)`` fetch path used today
    by the SmolVLA / pi05 / xVLA / Diffusion / modern-ACT adapters. Two
    files are required because lerobot's pipeline ships them as a pair: a
    preprocessor that normalises the policy input batch and a
    postprocessor that un-normalises the action chunk.

    Each URI must point at a specific file (the implicit-snapshot shape
    ``hf://owner/repo`` with no file tail is rejected). Two URI schemes
    accepted:

    * ``hf://owner/repo[@rev]/path/to/file.json`` — file inside an HF Hub
      repo, revision-pinnable.

    Attributes:
        preprocessor_uri: Per-file URI to the policy preprocessor JSON.
        postprocessor_uri: Per-file URI to the policy postprocessor JSON.

    Example:
        >>> RSkillProcessors(
        ...     preprocessor_uri="hf://lerobot/smolvla_libero/policy_preprocessor.json",
        ...     postprocessor_uri="hf://lerobot/smolvla_libero/policy_postprocessor.json",
        ... )  # doctest: +ELLIPSIS
        RSkillProcessors(preprocessor_uri='hf://...preprocessor.json', postprocessor_uri='hf://...postprocessor.json')
    """

    model_config = ConfigDict(extra="forbid")

    preprocessor_uri: str = Field(pattern=_PROCESSOR_URI_PATTERN)
    postprocessor_uri: str = Field(pattern=_PROCESSOR_URI_PATTERN)

    @model_validator(mode="after")
    def _check_distinct(self) -> RSkillProcessors:
        """Reject when both URIs point at the same file."""
        if self.preprocessor_uri == self.postprocessor_uri:
            raise ValueError(
                "RSkillProcessors: preprocessor_uri and postprocessor_uri "
                "must point to different files (got "
                f"{self.preprocessor_uri!r} twice)."
            )
        return self


RSkillKind: TypeAlias = Literal[
    "vla", "wam", "ros_action", "ros_service", "detector", "vlm", "reward"
]
"""Discriminator selecting how an rSkill is instantiated at the loader.

* ``"vla"`` — learnable Vision-Language-Action policy. Requires
  :attr:`RSkillManifest.model_family` and :attr:`RSkillManifest.weights_uri`;
  resolved by the policy adapter dispatch in ``openral_rskill`` (the
  pre-existing path used by every in-tree rSkill prior to this discriminator
  landing).
* ``"wam"`` — World Action Model (planning-layer mental-simulation /
  failure-anticipation component per CLAUDE.md §3). Reserved so the
  discriminator is forward-compatible; the loader / runner branch is not
  implemented yet and ``kind: wam`` manifests are rejected at resolve time
  with :class:`~openral_core.exceptions.ROSConfigError`.
* ``"ros_action"`` — wraps an existing ROS 2 action server. Requires a
  :class:`RosIntegration` block. ``model_family`` and ``weights_uri`` are
  forbidden. ``chunk_size`` is pinned to ``1`` so the safety supervisor's
  per-row check sees every commanded position.
* ``"ros_service"`` — wraps an existing ROS 2 service. Same constraints as
  ``"ros_action"``.
* ``"detector"`` — perception producer that runs an exported detection model
  (RT-DETR / D-FINE ONNX) on the camera tee and publishes
  :class:`~openral_core.schemas.ObjectsMetadata`; emits no
  :class:`~openral_core.schemas.Action`. Requires a
  :class:`DetectorContract` block and :attr:`RSkillManifest.weights_uri`
  (the exported ONNX / TensorRT engine). ``model_family`` and
  ``action_contract`` / ``state_contract`` are forbidden. ADR-0037.
* ``"vlm"`` — vision/video-language model used as a scene-understanding
  perception component (e.g. Qwen3.5-4B NF4). Accepts RGB image or video
  frames and a natural-language query; returns a text answer. Emits no
  actions or bounding boxes. Runs at S2 rate (``role: "s2"``), so it is
  surfaced to the reasoner as a read-only scene-query tool, never as an
  ``ExecuteSkill`` policy. ``weights_uri`` REQUIRED; ``actuators_required``
  MUST be empty; ``action_contract``, ``state_contract``, ``detector``,
  ``ros_integration``, ``processors``, ``image_preprocessing``,
  ``n_action_steps``, and ``starting_pose`` are FORBIDDEN. ``model_family``
  is OPTIONAL metadata. ADR-0047.
"""

_ROS_WRAPPER_KINDS: frozenset[str] = frozenset({"ros_action", "ros_service"})

# Perception rSkills are embodiment-agnostic: they consume camera frames and
# emit detections / scene text with no action contract, so a robot's embodiment
# is not a meaningful match axis. They MAY declare an empty ``embodiment_tags``
# (match-any) — see ``_check_embodiment_tags_present`` below — and the
# rSkill↔robot gate exempts them (``openral_rskill.loader._EMBODIMENT_AGNOSTIC_KINDS``).
_PERCEPTION_KINDS: frozenset[str] = frozenset({"detector", "vlm", "reward"})


class RosIntegration(BaseModel):
    """Wiring for an rSkill that wraps an existing ROS 2 action or service.

    Populated when :attr:`RSkillManifest.kind` is ``"ros_action"`` or
    ``"ros_service"``. The
    :class:`~openral_rskill.ros_action_rskill.ROSActionRskill` adapter reads
    this block at configure time to build the right action / service client
    on the host ``LifecycleNode``.

    Attributes:
        package: IDL package name (e.g. ``"moveit_msgs"``, ``"nav2_msgs"``).
            Imported lazily so manifests for un-installed ROS packages
            still parse — the loader raises
            :class:`~openral_core.exceptions.ROSConfigError` only when the
            skill is actually resolved.
        interface_type: Action / service type name inside ``package`` (e.g.
            ``"MoveGroup"``, ``"NavigateToPose"``).
        interface_name: Fully-qualified ROS path of the running server
            (e.g. ``"/move_action"``, ``"/navigate_to_pose"``).
        result_trajectory_field: Dotted accessor pointing at a
            ``trajectory_msgs/JointTrajectory`` inside the action result
            (e.g. ``"planned_trajectory.joint_trajectory"`` for MoveIt's
            ``MoveGroup``). When set, the adapter replays one waypoint per
            :meth:`~openral_rskill.base.rSkillBase.step` call onto
            ``/openral/candidate_action`` (so the safety supervisor + HAL
            see every position). When ``None`` the action is treated as
            result-only: the adapter awaits the action result, raises
            :class:`~openral_core.exceptions.ROSRskillGoalSatisfied` on
            success, and never emits an
            :class:`~openral_core.schemas.Action` chunk. This second mode
            covers wrapped ROS packages that drive actuators on their own
            (Nav2's behaviour tree publishes ``cmd_vel`` directly).
        default_goal_json: JSON dict literal used to construct the goal
            message. v1 hard-codes the target here — the structured-prompt
            path that lowers LLM-emitted JSON into
            ``ExecuteSkill.Goal.prompt_metadata_json`` is a follow-up.
            REQUIRED so that a wrapped skill is always invocable end-to-end
            without per-call schema work.
        ros_dependencies: Apt / colcon packages the operator must have
            installed for the wrapped server to be reachable (e.g.
            ``"ros-${ROS_DISTRO}-moveit"``). Surfaced by ``ral skill check``
            and quoted in :class:`~openral_core.exceptions.ROSConfigError`
            messages when the action client fails to connect.

    Example:
        >>> ri = RosIntegration(
        ...     package="moveit_msgs",
        ...     interface_type="MoveGroup",
        ...     interface_name="/move_action",
        ...     result_trajectory_field="planned_trajectory.joint_trajectory",
        ...     default_goal_json='{"target_joint_positions": [0, 0, 0, 0, 0, 0, 0]}',
        ...     ros_dependencies=["ros-${ROS_DISTRO}-moveit"],
        ... )
        >>> ri.package
        'moveit_msgs'
    """

    model_config = ConfigDict(extra="forbid")

    package: str = Field(min_length=1, max_length=200)
    interface_type: str = Field(min_length=1, max_length=200)
    interface_name: str = Field(min_length=1, max_length=200)
    result_trajectory_field: str | None = Field(default=None, max_length=200)
    default_goal_json: str = Field(min_length=2, max_length=10_000)
    ros_dependencies: list[str] = Field(default_factory=list)
    goal_builder: Literal["joint", "pose", "look_at"] | None = None
    """ADR-0044 / ADR-0054 — optional goal-lowering adapter over the shared
    ``ROSActionRskill`` MoveGroup engine. ``None`` (the default) sends
    ``default_goal_json`` + LLM overrides verbatim (the raw-IDL escape hatch).
    The named builders consume a typed block from the merged goal and lower it
    into MoveGroup constraints:

    * ``"joint"`` — a ``joint`` block (``positions``, ``joint_names``) →
      ``joint_constraints`` (:class:`~openral_rskill.joint_goal_rskill.JointGoalRskill`).
    * ``"pose"`` — a ``pose`` block (``position``, ``orientation`` as a
      quaternion array with ``quaternion_order``, ``link_name``) → MoveGroup
      position + orientation constraints for a Cartesian end-effector goal
      (:class:`~openral_rskill.pose_goal_rskill.PoseGoalRskill`).
    * ``"look_at"`` — a ``look_at`` block (``target_xyz``, ``camera``, …) → the
      same pose-constraint lowering, with the gaze pose computed from the
      camera's live TF pose (:class:`~openral_rskill.look_at_rskill.LookAtRskill`,
      a specialisation of the ``pose`` builder)."""

    @field_validator("interface_name")
    @classmethod
    def _check_interface_name_is_ros_path(cls, v: str) -> str:
        """``interface_name`` is the running server's ROS topic path."""
        if not v.startswith("/"):
            raise ValueError(
                f"RosIntegration.interface_name must be a fully-qualified ROS "
                f"path starting with '/', got {v!r}."
            )
        return v

    @field_validator("default_goal_json")
    @classmethod
    def _check_default_goal_json_parses(cls, v: str) -> str:
        """Reject manifests that ship un-parseable goal JSON."""
        import json  # noqa: PLC0415  # reason: stdlib, defer to keep import-time cheap

        try:
            parsed = json.loads(v)
        except json.JSONDecodeError as exc:
            raise ValueError(f"RosIntegration.default_goal_json is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                "RosIntegration.default_goal_json must encode a JSON object, "
                f"got {type(parsed).__name__}."
            )
        return v


class DetectorEngine(str, Enum):
    """Backend that executes a ``kind: "detector"`` rSkill (ADR-0037).

    Selects which runtime detector class :func:`build_manifest_detector`
    constructs. ``None`` (the default on :class:`DetectorContract`) preserves
    the legacy ``runtime``-keyed dispatch: ``runtime: onnx``/``tensorrt`` →
    RT-DETR ONNX, ``runtime: pytorch`` → the LocateAnything VLM sidecar. Set it
    explicitly to opt into a backend that the ``runtime`` value alone cannot
    disambiguate (e.g. an in-process Transformers open-vocabulary detector,
    which is also ``runtime: pytorch``).

    Attributes:
        RTDETR_ONNX: Fixed-label RT-DETR / D-FINE ONNX export (CPU / NVMM
            tiers). Equivalent to leaving ``engine`` unset with an
            ``onnx``/``tensorrt`` runtime.
        VLM_SIDECAR: Out-of-process open-vocabulary visual-grounding VLM
            (LocateAnything-3B). Equivalent to leaving ``engine`` unset with a
            ``pytorch`` runtime. Query-driven (prompted).
        ZEROSHOT_HF: In-process Transformers open-vocabulary detector
            (``AutoModelForZeroShotObjectDetection`` — e.g. OmDet-Turbo). Runs
            against a **fixed** class vocabulary (the manifest ``labels``)
            every frame, so it behaves like a large closed-vocabulary detector
            that needs no prompting — an unprompted background producer that
            populates the world object list with far more than the 80 COCO
            classes (ADR-0037 2026-06-12 amendment).

    Example:
        >>> DetectorEngine.ZEROSHOT_HF.value
        'zeroshot_hf'
    """

    RTDETR_ONNX = "rtdetr_onnx"
    VLM_SIDECAR = "vlm_sidecar"
    ZEROSHOT_HF = "zeroshot_hf"


class DetectorMode(str, Enum):
    """Invocation mode of a ``kind: "detector"`` rSkill (ADR-0051).

    The axis **orthogonal** to :class:`DetectorEngine`: where ``engine`` says
    *how* the model runs, ``mode`` says *when the reasoner invokes it* and
    therefore how its output reaches the LLM.

    Attributes:
        CONTINUOUS: An always-on background producer. Runs on the camera tee
            every frame and streams ``ObjectsMetadata`` into
            ``WorldState.detected_objects``; the reasoner reads it **passively**
            (via world state / ``recall_object``) and never prompts it. It is
            **not** an ExecuteSkill-dispatchable tool and carries no actuation
            authority. RT-DETR (closed vocab) and OmDet-Turbo (frozen open
            vocab) are continuous. The reasoner may still toggle it via
            ``LifecycleTransitionTool`` to free VRAM (ADR-0050).
        ON_DEMAND: A prompted locator the reasoner invokes only when it needs
            to find a specific object **right now**. Surfaces the read-only
            ``locate_in_view`` tool (ADR-0043) backed by an open-vocabulary
            detector; it is not run continuously. LocateAnything is on-demand.

    The two modes cleanly separate "open-vocabulary" from "prompting":
    continuous detectors cover a fixed bank of classes the reasoner reads for
    free; the on-demand locator handles the long tail that bank does not cover.

    Example:
        >>> DetectorMode.CONTINUOUS.value
        'continuous'
        >>> DetectorMode.ON_DEMAND.value
        'on_demand'
    """

    CONTINUOUS = "continuous"
    ON_DEMAND = "on_demand"


class DetectorContract(BaseModel):
    """Manifest contract for ``kind: "detector"`` rSkills (ADR-0037).

    Carries the configuration the runtime
    :class:`~openral_core.schemas.ObjectsDetector` needs to instantiate an
    exported detection model (RT-DETR / D-FINE ONNX) and match its output
    indices to semantic class labels.

    Required when :attr:`RSkillManifest.kind` is ``"detector"``; forbidden
    for all other kinds (enforced by
    :meth:`RSkillManifest._check_kind_consistency`).

    Attributes:
        labels: Ordered class-label list; the integer class-id the model
            emits is used as an index into this list.  At least one label
            is required.
        input_size: Model input resolution as ``(width, height)`` in pixels.
            Both dimensions must be > 0.  Default ``(640, 640)`` matches the
            RT-DETR / D-FINE default export resolution.
        score_threshold: Detections with confidence below this value are
            discarded before publishing.  Must be in ``[0.0, 1.0]``.
            Default ``0.5``.
        engine: Optional explicit backend selector (:class:`DetectorEngine`).
            ``None`` (default) keeps the legacy ``runtime``-keyed dispatch.
            Set it to disambiguate backends that share a ``runtime`` — e.g.
            ``zeroshot_hf`` for an in-process Transformers open-vocabulary
            detector, which is ``runtime: pytorch`` like the VLM sidecar.
        mode: Invocation mode (:class:`DetectorMode`; default ``continuous``).
            Declares whether the detector is an always-on background producer
            (output reaches the reasoner via world state) or an on-demand
            prompted locator (surfaces the ``locate_in_view`` tool). ADR-0051.

    Example:
        >>> c = DetectorContract(
        ...     labels=["person", "bicycle", "car"],
        ...     input_size=(640, 640),
        ...     score_threshold=0.4,
        ... )
        >>> c.labels[0]
        'person'
        >>> c.mode
        <DetectorMode.CONTINUOUS: 'continuous'>
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    labels: list[str] = Field(min_length=1)
    input_size: tuple[int, int] = (640, 640)
    score_threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    engine: DetectorEngine | None = None
    mode: DetectorMode = DetectorMode.CONTINUOUS

    @field_validator("input_size")
    @classmethod
    def _check_input_size_positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Both width and height must be strictly positive."""
        w, h = v
        if w <= 0 or h <= 0:
            raise ValueError(
                f"DetectorContract.input_size must have both dimensions > 0, got {v!r}."
            )
        return v


class RewardContract(BaseModel):
    """Manifest contract for ``kind: "reward"`` rSkills (ADR-0057).

    Carries the configuration a robotic **reward / progress-monitor** model
    (e.g. Robometer-4B, a Qwen3-VL-4B reward foundation model) needs to score
    a rollout. A reward skill runs in parallel with a ``kind: "vla"`` policy,
    continuously ingesting the VLA's camera frames into a rolling window, and
    emits per-frame **progress** ∈ ``progress_range`` and per-frame **success**
    probability. The Reasoner queries it on demand (``QueryTaskProgressTool``)
    to decide whether to continue, escalate to a scene VLM, advance, or replan.
    The signal is **advisory only** — it never actuates and never gates motors
    (CLAUDE.md §1.1).

    Required when :attr:`RSkillManifest.kind` is ``"reward"``; forbidden for all
    other kinds (enforced by :meth:`RSkillManifest._check_kind_consistency`).
    Like ``detector`` / ``vlm``, a reward skill is a pure perception consumer:
    it emits no Action chunks and requires no actuators.

    Attributes:
        progress_range: ``(min, max)`` of the normalized per-frame progress
            scalar. Default ``(0.0, 1.0)`` — Robometer's discrete/binned mode
            (see :attr:`num_bins`) emits progress already in ``[0, 1]``.
        success_threshold: Per-frame success probability at/above which the
            frame is considered a task success. In ``[0.0, 1.0]``; default
            ``0.5``.
        preference: Whether the model also exposes a trajectory-preference
            head (Robometer does). Default ``False`` — the progress/success
            path is the Reasoner-facing contract; preference is future work.
        frame_window_s: Length of the rolling frame buffer in seconds. The
            sidecar evicts frames older than this relative to the newest.
            Must be > 0.
        target_fps: Frame-sampling rate fed to the model (Robometer's example
            uses 3 fps). Must be > 0. This is an S2-cadence monitor, not a
            per-control-step signal.
        num_bins: Discrete-mode bin count for the progress head. Robometer's
            discrete mode yields per-frame normalized progress in ``[0, 1]``;
            continuous mode yields raw regression values. Must be > 0;
            default ``100``.
        instruction_required: Whether a natural-language task instruction must
            accompany the frames (Robometer requires one). Default ``True``.

    Example:
        >>> c = RewardContract(frame_window_s=8.0, target_fps=3.0)
        >>> c.progress_range
        (0.0, 1.0)
        >>> c.success_threshold
        0.5
        >>> c.num_bins
        100
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    progress_range: tuple[float, float] = (0.0, 1.0)
    success_threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    preference: bool = False
    frame_window_s: float = Field(gt=0.0)
    target_fps: float = Field(gt=0.0)
    num_bins: int = Field(gt=0, default=100)
    instruction_required: bool = True

    @field_validator("progress_range")
    @classmethod
    def _check_progress_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """``progress_range`` must be a non-degenerate ``(min, max)`` interval."""
        lo, hi = v
        if hi <= lo:
            raise ValueError(f"RewardContract.progress_range must have max > min, got {v!r}.")
        return v


class RSkillManifest(BaseModel):
    """Pydantic model of the ``rskill.yaml`` package manifest (V1).

    This is the on-disk schema for an rSkill HF Hub repo. An rSkill is
    loaded by capability-checking against a :class:`RobotDescription`,
    selecting a runtime + quantization, then constructing a runtime
    :class:`~openral_rskill.Skill` instance.

    ``schema_version`` is ``"0.1"``: the manifest surface has had no
    backward-incompatible change. Now the repo is published it is
    versioned for real (CLAUDE.md §1.6) — a backward-incompatible change
    bumps it and ships a migrator, while backward-compatible additions
    (ADR-0013/0022/0024) evolve the surface in place.

    ADR-0013 added two symmetric guards on top of the initial V1 shape:

    1. **``actuators_required``** mirrors ``sensors_required`` on the
       output side. Every skill declares at least one
       :class:`ActuatorRequirement`; the loader validates it against
       :attr:`RobotDescription.action_spec`. ``n_dof`` and
       ``vla_action_key`` are auto-filled from the robot YAML for the 9
       canonical embodiments; for ``"custom"`` they must be set on the
       manifest.

    2. **``embodiment_extra``** is the explicit escape hatch for
       embodiments that do not have a canonical
       ``robots/<id>/robot.yaml``. Required iff ``"custom"`` appears in
       :attr:`embodiment_tags`; forbidden otherwise.

    V1 already tightened: ``name`` / ``fallback_skill_id`` must be
    HF-Hub-shaped, ``version`` is SemVer, ``weights_uri`` is restricted
    to ``hf://`` or ``local://``, ``embodiment_tags`` is closed to the
    set of in-tree-supported robots (now including ``"custom"``), and
    ``benchmarks`` replaces the old free-form ``metadata`` blob with a
    typed dict of canonical suite ids → success rate. Commercial-use
    posture is derived from :attr:`license` via
    :attr:`is_commercial_use_allowed`.

    Attributes:
        schema_version: On-disk format version. ``"0.1"`` today.
            Backward-compatible extensions (ADR-0013/0022/0024) evolved
            the surface in place; now the repo is published, a
            backward-incompatible change bumps this and ships a migrator
            (CLAUDE.md §1.6).
        name: HF Hub identifier, e.g. ``"openral/rskill-pick-cube-so100"``.
            Must match ``<owner>/<repo>``.
        version: SemVer string of the rSkill package itself (not the
            wrapped weights).
        license: License posture (surfaced at install time).
            Drives :attr:`is_commercial_use_allowed`.
        role: rSkill slot. ``"s1"`` for fast policies (the only slot
            currently loaded via :func:`rSkill.from_yaml` — 100% of
            in-tree skills); ``"s0"`` reserved for cerebellar realtime
            (humanoid balance, C++ only); ``"s2"`` reserved for slow
            reasoning (LLM-emitted BehaviorTree XML).
        model_family: Closed VLA / policy family. Drives the runner's
            adapter dispatch.
        embodiment_tags: Robot embodiments this rSkill targets. Must
            intersect a robot's ``RobotCapabilities.embodiment_tags`` for
            the loader to accept. Restricted to the canonical set
            (:data:`EmbodimentTag`) so typos / framework hints cannot
            silently always-miss. ``"custom"`` is the explicit hatch
            (see :attr:`embodiment_extra`).
        embodiment_extra: ADR-0013. When ``"custom"`` is in
            :attr:`embodiment_tags`, declares the sensor + actuator
            surface of the custom rig so the loader's compat check
            still has something to match against. MUST be ``None`` when
            ``"custom"`` is not present.
        capabilities_required: Per-flag capability requirements (subset of
            :class:`RobotCapabilities` boolean fields). The loader rejects
            installation on a robot that does not satisfy every flag.
        sensors_required: Sensor inputs the policy expects from the
            robot. See :class:`SensorRequirement`.
        actuators_required: ADR-0013. Symmetric output-side
            contract: at least one :class:`ActuatorRequirement`
            entry. The loader matches each entry's :attr:`kind` against
            :attr:`RobotDescription.action_spec.control_mode` and
            auto-fills :attr:`ActuatorRequirement.n_dof` /
            :attr:`vla_action_key` from the canonical robot YAML
            (predefined embodiments only — for ``"custom"`` they must
            be set on the manifest).
        runtime: Preferred inference runtime.
        quantization: Default :class:`QuantizationConfig` for this rSkill.
        weights_uri: HF Hub revision-pinned URI to weights or local
            path reference. Production deployments pin a SHA
            (CLAUDE.md operating principle 8).
        chunk_size: Action-chunk length the policy emits per
            :meth:`Skill.step`. Drives the ChunkedExecutor's overlap
            schedule and shows up in the trace surface.
        latency_budget: Latency contract enforced by CI on the reference
            host (CLAUDE.md §7.4).
        min_vram_gb: Optional minimum VRAM (GB) the skill needs per
            quantization dtype. Informational — the loader uses it for
            ``ral skill check`` / ``openral doctor``; the actually-applied
            dtype is still pinned by ``quantization.dtype`` (CLAUDE.md
            §11 forbids silent downcast).
        fallback_skill_id: rSkill id to substitute if this one fails
            (RFC §8.3 replanning ladder). HF Hub shape.
        benchmarks: Map of canonical benchmark suite id → success rate
            (``[0.0, 1.0]``) as measured for this skill. Keys
            constrained to :data:`BenchmarkName`. The full per-task
            breakdown lives in the matching
            ``rskills/<id>/eval/<key>.json`` validated against
            :class:`RSkillEvalResult`.
        evaluated_tasks: Benchmark task ids / families this checkpoint was
            trained or validated for (e.g. ``["libero_spatial"]`` covering
            ``libero_spatial/0..9``, or ``["maniskill3/PickCube-v1"]``). The
            benchmark runner gates a scene's ``task.id`` against this list: a
            non-empty list that does not cover the scene's task is refused with
            :class:`ROSCapabilityMismatch` (prevents running a checkpoint on a
            task it was not trained for — e.g. a LiftCube policy on PickCube).
            Empty (default) is permissive: legacy rSkills run with a warning.
            Matching: exact ``task.id``, a ``"<scene>/<...>"`` prefix family,
            or the bare ``scene.id``. See ADR-0060.
        sim_env_control_mode: Optional simulator controller mode this policy
            expects the env to run in, when the scene itself does not pin one.
            Currently consumed by the LIBERO backend (``"relative"`` = OSC
            delta-EE, the default for SmolVLA / π0.5 / rldx1 / molmoact2 /
            GR00T; ``"absolute"`` = OSC absolute-EE, required by xVLA which
            emits absolute end-effector targets). Lets an absolute-control
            policy run on the canonical ``libero_spatial.yaml`` without a
            duplicate per-policy scene; ``scene.backend_options.control_mode``
            still overrides it. ``None`` (default) → backend default.
        policy_extras: Adapter-owned runtime knobs copied from the rSkill
            manifest into :class:`VLASpec.extra` during CLI composition.
            Used for family-specific generation, sampling, replay, or
            transform settings that are part of the checkpoint contract but
            should not become top-level manifest schema fields.
        paper_url: Canonical paper URL for this skill / family.
        dataset_uri: HF Hub URI for the training dataset.
        source_repo: HF Hub URI for the upstream weights repo (often
            the same as :attr:`weights_uri`'s prefix).
        description: REQUIRED. Short (1-500 char) human-readable summary
            surfaced by ``ral skill list`` and, more importantly, by the
            reasoner's per-skill LLM tool description (see
            :func:`openral_reasoner.palette.build_tool_palette`). The LLM
            scores tools primarily on this text; keep it specific to what
            the skill does (objects, scenes, task type), not how it was
            trained.
        actions: REQUIRED. Closed-vocabulary list (≥1) of high-level
            action verbs this skill performs (see :class:`RSkillAction`).
            Generalist / foundation checkpoints declare ``[GENERALIST]``;
            specialist checkpoints list the verbs they were trained on
            (e.g. ``[PICK, PLACE]``). Surfaced to the reasoner LLM
            alongside :attr:`description` so it can pick the right skill
            for a given goal.
        objects: Optional free-form keywords for the objects this skill
            manipulates (e.g. ``["cube"]``, ``["pipe"]``, ``["drawer"]``).
            Discriminative hints for the LLM; the long tail (RoboCasa-365
            has hundreds of categories) makes a closed enum impractical.
        scenes: Optional free-form keywords for the scenes / environments
            this skill targets (e.g. ``["tabletop"]``, ``["kitchen"]``).
            Same rationale as :attr:`objects`.
        processors: Per-file URIs for the lerobot ``PolicyProcessorPipeline``
            artefacts (Gap 1 + Gap 3 of the rSkill self-containment audit).
            REQUIRED when ``model_family`` is one of ``smolvla``, ``pi05``,
            ``xvla``, ``diffusion``, or ``rldx``; OPTIONAL for ``act`` (the
            legacy norm-stats-in-safetensors path is allowed there).
        image_preprocessing: Per-checkpoint image-side knobs that cannot be
            encoded in the processor JSONs (flip, camera renames).
            Optional; ``None`` means schema defaults apply.
        state_contract: Per-checkpoint proprioception layout (named layouts
            for RoboCasa; explicit ``dim`` for everything else).
        action_contract: Per-checkpoint action vector contract (dim +
            optional representation). ADR-0019: consumed by the dataset
            bridge to bind the LeRobot v3 ``action`` feature shape
            without consulting the runtime adapter.
        n_action_steps: Replay cadence (how many actions to consume from a
            chunk before re-inferring). Omit when equal to ``chunk_size``;
            the adapter falls through to its family default.

    Example:
        >>> m = RSkillManifest(
        ...     name="openral/rskill-pick-cube-so100",
        ...     version="0.1.0",
        ...     license=RSkillLicensePosture.APACHE_2_0,
        ...     role="s1",
        ...     kind="vla",
        ...     model_family="smolvla",
        ...     embodiment_tags=["so100_follower"],
        ...     runtime=RSkillRuntime.PYTORCH,
        ...     weights_uri="hf://lerobot/smolvla_base",
        ...     chunk_size=16,
        ...     latency_budget=RSkillLatencyBudget(per_chunk_ms=100.0),
        ...     description="Pick a cube on the SO-100 follower arm.",
        ...     actions=[RSkillAction.PICK],
        ...     objects=["cube"],
        ...     scenes=["tabletop"],
        ...     actuators_required=[
        ...         ActuatorRequirement(
        ...             kind=ControlMode.JOINT_POSITION,
        ...             control_mode_semantics=ControlModeSemantics(mode="absolute"),
        ...         ),
        ...     ],
        ...     processors=RSkillProcessors(
        ...         preprocessor_uri="hf://lerobot/smolvla_base/policy_preprocessor.json",
        ...         postprocessor_uri="hf://lerobot/smolvla_base/policy_postprocessor.json",
        ...     ),
        ... )
        >>> m.is_commercial_use_allowed
        True
        >>> m.schema_version
        '0.1'
    """

    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    name: str = Field(pattern=_HF_HUB_ID_PATTERN)
    version: str = Field(pattern=_SEMVER_PATTERN)
    license: RSkillLicensePosture = RSkillLicensePosture.UNKNOWN
    role: Literal["s0", "s1", "s2"] = "s1"
    # Discriminator selecting the loader / runner branch. Required (no
    # default) so every manifest declares it explicitly — see
    # :data:`RSkillKind` for the semantics of each value and which other
    # fields are required / forbidden per kind.
    kind: RSkillKind
    # Required when `kind == "vla"`, forbidden otherwise (enforced by
    # :meth:`_check_kind_consistency`).
    model_family: ModelFamily | None = None
    # Non-perception kinds must declare >=1 tag; perception kinds (detector /
    # vlm) MAY be empty (match-any) — enforced by `_check_embodiment_tags_present`.
    embodiment_tags: list[EmbodimentTag] = Field(default_factory=list)
    embodiment_extra: EmbodimentExtra | None = None
    capabilities_required: dict[str, bool | float | int | str] = Field(default_factory=dict)
    sensors_required: list[SensorRequirement] = Field(default_factory=list)
    actuators_required: list[ActuatorRequirement] = Field(default_factory=list)
    runtime: RSkillRuntime = RSkillRuntime.PYTORCH
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    # Required when `kind == "vla"`, forbidden otherwise (a wrapped ROS
    # action has no weights to download). Enforced by
    # :meth:`_check_kind_consistency`.
    weights_uri: str | None = Field(default=None, pattern=_WEIGHTS_URI_PATTERN)
    chunk_size: int = Field(gt=0)
    latency_budget: RSkillLatencyBudget
    min_vram_gb: dict[QuantizationDtype, float] | None = None
    fallback_skill_id: str | None = Field(default=None, pattern=_HF_HUB_ID_PATTERN)
    benchmarks: dict[BenchmarkName, float] = Field(default_factory=dict)
    evaluated_tasks: list[str] = Field(default_factory=list)
    sim_env_control_mode: str | None = None
    paper_url: str | None = Field(default=None, pattern=_HTTPS_URL_PATTERN)
    dataset_uri: str | None = Field(default=None, pattern=_HF_DATASET_URI_PATTERN)
    source_repo: str | None = Field(default=None, pattern=_HF_DATASET_URI_PATTERN)
    description: str = Field(min_length=1, max_length=500)
    # ADR-0022 — per-skill action vocabulary surfaced to the reasoner LLM
    # tool palette so it can pick the right skill for a given goal.
    actions: list[RSkillAction] = Field(min_length=1)
    objects: list[str] = Field(default_factory=list)
    scenes: list[str] = Field(default_factory=list)
    # ADR-0018 §5 / ADR-0020 (C++ safety kernel). Optional per-skill safety
    # envelope. When set, the kernel enforces the *intersection* with the
    # robot ceiling at goal acceptance (envelope_loader.py): every field that
    # is tighter than the robot's wins; any field that LOOSENS the robot
    # ceiling causes the loader to reject the skill with ROSConfigError
    # (never silently honored — CLAUDE.md §1.1 / §1.4). Pre-existing skill
    # manifests without this field continue to load unchanged; they inherit
    # the full robot ceiling.
    envelope: SafetyEnvelope | None = None

    @field_validator("benchmarks")
    @classmethod
    def _validate_benchmark_scores(cls, v: dict[str, float]) -> dict[str, float]:
        """Reject benchmark scores outside ``[0.0, 1.0]``."""
        for key, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"benchmarks[{key!r}] = {score!r} is out of range; success "
                    "rates must satisfy 0.0 <= score <= 1.0"
                )
        return v

    @field_validator("min_vram_gb")
    @classmethod
    def _validate_vram_positive(
        cls, v: dict[QuantizationDtype, float] | None
    ) -> dict[QuantizationDtype, float] | None:
        """Reject zero / negative VRAM entries."""
        if v is None:
            return v
        for dtype, gb in v.items():
            if gb <= 0:
                raise ValueError(f"min_vram_gb[{dtype.value!r}] = {gb!r} must be > 0")
        return v

    @model_validator(mode="after")
    def _check_self_referential_fallback(self) -> RSkillManifest:
        """A skill cannot list itself as its own fallback."""
        if self.fallback_skill_id is not None and self.fallback_skill_id == self.name:
            raise ValueError(
                f"fallback_skill_id ({self.fallback_skill_id!r}) cannot equal the "
                "skill's own name; pick a different rSkill id or leave it null."
            )
        return self

    @model_validator(mode="after")
    def _check_embodiment_tags_present(self) -> RSkillManifest:
        """Non-perception rSkills must declare at least one embodiment tag.

        Perception kinds (``detector`` / ``vlm``, see :data:`_PERCEPTION_KINDS`)
        are embodiment-agnostic — camera-in → detections/text-out, no action
        contract — so they MAY ship an empty ``embodiment_tags`` (match-any; the
        rSkill↔robot gate exempts them). Every other kind (``vla`` / ``wam`` /
        ``ros_action`` / ``ros_service``) still actuates a specific embodiment
        and must target one, preserving the prior ``min_length=1`` guarantee.
        """
        if self.kind not in _PERCEPTION_KINDS and not self.embodiment_tags:
            raise ValueError(
                f"RSkillManifest({self.name!r}): kind={self.kind!r} requires at "
                "least one embodiment_tag. Only perception kinds "
                f"{sorted(_PERCEPTION_KINDS)} may be embodiment-agnostic (empty tags)."
            )
        return self

    @model_validator(mode="after")
    def _check_custom_embodiment_extra(self) -> RSkillManifest:
        """Enforce the ADR-0013 ``"custom"`` ↔ ``embodiment_extra`` contract.

        Three rules:

        1. ``"custom"`` in :attr:`embodiment_tags` → :attr:`embodiment_extra`
           MUST be set.
        2. :attr:`embodiment_extra` set → ``"custom"`` MUST be in
           :attr:`embodiment_tags` (otherwise the extra block is dead
           weight; reject loudly).
        3. When ``"custom"`` is in :attr:`embodiment_tags`, every entry in
           :attr:`actuators_required` MUST have both ``n_dof`` and
           ``vla_action_key`` populated — the loader has no canonical
           robot YAML to auto-fill them from.
        """
        is_custom = "custom" in self.embodiment_tags
        has_extra = self.embodiment_extra is not None
        if is_custom and not has_extra:
            raise ValueError(
                "embodiment_tags contains 'custom' but embodiment_extra is not set; "
                "custom embodiments must declare their sensor + actuator surface "
                "(ADR-0013). Either drop 'custom' or populate embodiment_extra."
            )
        if has_extra and not is_custom:
            raise ValueError(
                "embodiment_extra is set but 'custom' is not in embodiment_tags; "
                "the extra block is only meaningful for the custom-embodiment hatch "
                "(ADR-0013). Either add 'custom' to embodiment_tags or drop "
                "embodiment_extra."
            )
        if is_custom:
            for i, act in enumerate(self.actuators_required):
                if act.n_dof is None or act.vla_action_key is None:
                    raise ValueError(
                        f"actuators_required[{i}] is missing n_dof or "
                        "vla_action_key. These are auto-filled from the robot "
                        "YAML for canonical embodiments, but 'custom' has no "
                        "canonical YAML — set both fields explicitly on the "
                        "manifest."
                    )
        return self

    @property
    def is_commercial_use_allowed(self) -> bool:
        """Derive commercial-use posture from :attr:`license`.

        ``apache-2.0`` / ``mit`` / ``bsd`` → True. Every other posture
        (including ``unknown``) → False, conservatively. The loader
        consults this in :meth:`rSkill._check_license`; users wanting to
        deploy a non-commercial skill in a research context set
        ``OPENRAL_ALLOW_NONCOMMERCIAL=1`` per CLAUDE.md §7.4.
        """
        return self.license in _LICENSES_ALLOWING_COMMERCIAL

    # ── Preprocessing block ───────────────────────────────────────────────
    # Knobs the trained checkpoint needs to interpret IO. Grouped here so
    # manifest readers see them together. ``processors`` is the explicit
    # per-file URI block (Gap 1 + Gap 3); ``image_preprocessing`` /
    # ``state_contract`` carry the small bits of checkpoint metadata that
    # can't be encoded in the processor JSONs. Precedence at adapter
    # construction: ``spec_extra`` > manifest > schema default. See
    # ``openral_rskill._vla_core.resolve_image_preprocessing``,
    # ``resolve_state_dim``, ``resolve_camera_keys``, and ``apply_chunk_replay``.
    policy_extras: dict[str, object] = Field(default_factory=dict)
    processors: RSkillProcessors | None = None
    image_preprocessing: ImagePreprocessing | None = None
    state_contract: StateContract | None = None
    # ADR-0019 PR-revert: action contract mirrors state_contract for the
    # bridge's LeRobot v3 feature binding. Optional today (backward-compat
    # with checkpoints that pre-date the bridge); the dataset bridge
    # requires either this OR RobotDescription.action_spec.dim, raising
    # ROSConfigError when both are missing.
    action_contract: ActionContract | None = None
    n_action_steps: int | None = Field(default=None, gt=0)
    # Optional initial joint pose the policy expects an episode to
    # start from. The list is ``state_contract.dim``-long and uses the
    # checkpoint's own ``action_feature_names`` order (i.e. the same
    # order the policy emits actions in — *not* the robot.yaml URDF
    # order). Units are radians. Sim adapters apply this as the qpos
    # at ``reset()`` and the live HAL can use it as the calibration
    # target.
    #
    # When omitted, sim adapters fall back to a kinematic default
    # (e.g. elbows at +π/2). There is no implicit "centre-of-training"
    # discovery — if a specific start pose matters, declare it here.
    starting_pose: list[float] | None = None

    # Wiring for wrapped ROS 2 actions / services. REQUIRED when
    # ``kind in {"ros_action", "ros_service"}``; FORBIDDEN otherwise. The
    # ``_check_kind_consistency`` validator enforces this so that a VLA
    # manifest cannot accidentally carry stale wrapper config.
    ros_integration: RosIntegration | None = None

    # Detector model contract (ADR-0037). REQUIRED when ``kind == "detector"``;
    # FORBIDDEN otherwise. Carries the class-label list, input resolution, and
    # score threshold the runtime ObjectsDetector reads at configure time.
    # A detector emits no Action chunks and requires no actuators — it is a
    # pure perception producer.
    detector: DetectorContract | None = None

    # Reward / progress-monitor model contract (ADR-0057). REQUIRED when
    # ``kind == "reward"``; FORBIDDEN otherwise. Carries the rolling-window +
    # sampling-rate + progress-range config a robotic reward model (Robometer)
    # needs. A reward skill is a pure perception consumer — it emits no Action
    # chunks, requires no actuators, and its progress/success signal is
    # advisory-only (never gates motors).
    reward: RewardContract | None = None

    # ADR-0026 — optional JSON-Schema (OpenAPI / JSON-Schema 7 shape)
    # describing the per-skill ``goal_params_json`` payload the LLM may
    # attach to an ``ExecuteRskillTool`` dispatch. The reasoner's
    # ``build_tool_palette`` surfaces this verbatim as the per-skill
    # tool ``parameters`` block in the LLM's tool definition, so the
    # provider's structured-output / tool-use path generates well-formed
    # JSON the wrapped-ROS adapter can merge onto its ``default_goal_json``.
    # ``None`` (default) means "no structured params for this skill" —
    # backward-compat for every existing VLA / wrapped-ROS manifest. The
    # field is intentionally a free ``dict[str, Any]`` (not a typed
    # JSONSchema Pydantic model) because the schema must support the full
    # JSON-Schema vocabulary the LLM provider expects.
    goal_params_schema: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_processors_required_for_modern_families(self) -> RSkillManifest:
        """Modern lerobot families MUST declare a :attr:`processors` block.

        Closes Gap 1 + Gap 3: SmolVLA / pi05 / xVLA / Diffusion / RLDX
        adapters consume the modern ``PolicyProcessorPipeline`` and must
        be able to download exactly the preprocessor + postprocessor
        artefacts from per-file URIs. Only the ``act`` family may omit
        the block — its legacy checkpoints carry norm stats inside
        ``model.safetensors`` and the ACT adapter dispatches on
        ``manifest.processors is not None``.
        """
        # model_family is a closed Literal (typed as ``ModelFamily``) or None
        # for wrapped-ROS kinds; the `in` check is safe against None.
        if self.model_family in _MODERN_PROCESSOR_FAMILIES and self.processors is None:
            raise ValueError(
                f"RSkillManifest({self.name!r}): model_family={self.model_family!r} "
                "requires a `processors` block (preprocessor_uri + postprocessor_uri). "
                "Only `act` may omit it (legacy norm-stats-in-safetensors path)."
            )
        return self

    @model_validator(mode="after")
    def _check_kind_consistency(self) -> RSkillManifest:  # noqa: PLR0912, PLR0915  # reason: each branch is a separate kind — splitting would obscure the per-kind contract table
        """Enforce the per-:attr:`kind` field shape for VLA vs ROS-wrapper vs detector.

        Rules:

        * ``kind == "vla"`` → :attr:`model_family` REQUIRED,
          :attr:`weights_uri` REQUIRED, :attr:`ros_integration` FORBIDDEN,
          :attr:`detector` FORBIDDEN.
          :attr:`actuators_required` REQUIRED (≥1 entry).
          The existing VLA adapter dispatch path consumes both required
          fields; without them the loader cannot build a runtime skill.
        * ``kind in {"ros_action", "ros_service"}`` →
          :attr:`ros_integration` REQUIRED; :attr:`model_family`,
          :attr:`weights_uri`, :attr:`processors`, :attr:`state_contract`,
          :attr:`action_contract`, :attr:`n_action_steps`,
          :attr:`image_preprocessing`, :attr:`starting_pose`,
          :attr:`detector` all FORBIDDEN
          (none of them have meaning for a wrapped server). :attr:`chunk_size`
          is pinned to ``1`` so that each waypoint of a planner trajectory
          is its own ``Action`` chunk — the safety supervisor only checks
          row 0 of every chunk today (``supervisor_node.py``), and we MUST
          NOT let rows 1..N actuate unchecked.
          :attr:`actuators_required` REQUIRED (≥1 entry).
        * ``kind == "detector"`` → :attr:`detector` REQUIRED;
          :attr:`weights_uri` REQUIRED (the exported ONNX / TensorRT engine);
          :attr:`model_family`, :attr:`ros_integration`,
          :attr:`action_contract`, :attr:`state_contract`,
          :attr:`processors`, :attr:`n_action_steps`, :attr:`starting_pose`
          all FORBIDDEN (a detector has no VLA policy family, no ROS wrapper,
          and no VLA inference lifecycle);
          :attr:`actuators_required` MUST be empty (a detector actuates
          nothing). ADR-0037.
        * ``kind == "wam"`` → schema-side this is unconstrained beyond the
          base VLA shape; the loader's resolver branch raises
          :class:`~openral_core.exceptions.ROSConfigError` at resolve time
          because the WAM dispatch path is not implemented in this PR
          (tracked separately).
        """
        if self.kind == "vla":
            if self.model_family is None:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='vla' requires `model_family` to be set."
                )
            if self.weights_uri is None:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='vla' requires `weights_uri` to be set."
                )
            if self.ros_integration is not None:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='vla' forbids "
                    "`ros_integration` (it is for wrapped ROS 2 servers only)."
                )
            if self.detector is not None:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='vla' forbids "
                    "`detector` (it is for kind='detector' perception producers only)."
                )
            if self.reward is not None:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='vla' forbids "
                    "`reward` (it is for kind='reward' progress monitors only)."
                )
            if not self.actuators_required:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='vla' requires at least one "
                    "`actuators_required` entry."
                )
            return self

        if self.kind in _ROS_WRAPPER_KINDS:
            if self.ros_integration is None:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind={self.kind!r} requires "
                    "a `ros_integration` block."
                )
            forbidden = {
                "model_family": self.model_family,
                "weights_uri": self.weights_uri,
                "processors": self.processors,
                "state_contract": self.state_contract,
                "action_contract": self.action_contract,
                "n_action_steps": self.n_action_steps,
                "image_preprocessing": self.image_preprocessing,
                "starting_pose": self.starting_pose,
                "detector": self.detector,
                "reward": self.reward,
            }
            set_fields = sorted(name for name, value in forbidden.items() if value is not None)
            if set_fields:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind={self.kind!r} forbids "
                    f"these VLA-only fields: {set_fields!r}. Drop them from the "
                    "manifest (wrapped ROS skills have no weights or policy "
                    "preprocessing)."
                )
            if self.chunk_size != 1:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind={self.kind!r} requires "
                    f"chunk_size=1, got {self.chunk_size}. Wrapped trajectories "
                    "must be emitted one waypoint per chunk so the safety "
                    "supervisor's per-row check sees every commanded position."
                )
            if not self.actuators_required:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind={self.kind!r} requires at least one "
                    "`actuators_required` entry."
                )
            return self

        if self.kind == "detector":
            if self.detector is None:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='detector' requires a "
                    "`detector` block (labels, input_size, score_threshold)."
                )
            if self.weights_uri is None:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='detector' requires "
                    "`weights_uri` (the exported ONNX / TensorRT engine path)."
                )
            forbidden_detector = {
                "model_family": self.model_family,
                "ros_integration": self.ros_integration,
                "action_contract": self.action_contract,
                "state_contract": self.state_contract,
                "processors": self.processors,
                "n_action_steps": self.n_action_steps,
                "starting_pose": self.starting_pose,
                "reward": self.reward,
            }
            set_detector_forbidden = sorted(
                name for name, value in forbidden_detector.items() if value is not None
            )
            if set_detector_forbidden:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='detector' forbids "
                    f"these fields: {set_detector_forbidden!r}. A detector is a "
                    "pure perception producer — it has no VLA policy family, no "
                    "ROS wrapper, no VLA inference lifecycle (processors / "
                    "n_action_steps / starting_pose), and emits no actions or "
                    "proprioception."
                )
            if self.actuators_required:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='detector' requires "
                    f"`actuators_required` to be empty (got {len(self.actuators_required)} "
                    "entries). A detector actuates nothing."
                )
            return self

        if self.kind == "vlm":
            if self.weights_uri is None:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='vlm' requires "
                    "`weights_uri` (the Hugging Face model repository)."
                )
            forbidden_vlm = {
                "detector": self.detector,
                "reward": self.reward,
                "ros_integration": self.ros_integration,
                "action_contract": self.action_contract,
                "state_contract": self.state_contract,
                "processors": self.processors,
                "n_action_steps": self.n_action_steps,
                "image_preprocessing": self.image_preprocessing,
                "starting_pose": self.starting_pose,
            }
            set_vlm_forbidden = sorted(
                name for name, value in forbidden_vlm.items() if value is not None
            )
            if set_vlm_forbidden:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='vlm' forbids "
                    f"these fields: {set_vlm_forbidden!r}. A scene VLM is a "
                    "pure perception component — it has no action contract, no "
                    "detector block, no ROS wrapper, and no VLA policy "
                    "preprocessing."
                )
            if self.actuators_required:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='vlm' requires "
                    f"`actuators_required` to be empty (got {len(self.actuators_required)} "
                    "entries). A scene VLM actuates nothing."
                )
            return self

        if self.kind == "reward":
            if self.reward is None:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='reward' requires a "
                    "`reward` block (frame_window_s, target_fps, progress_range)."
                )
            if self.weights_uri is None:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='reward' requires "
                    "`weights_uri` (the Hugging Face reward-model repository)."
                )
            forbidden_reward = {
                "detector": self.detector,
                "model_family": self.model_family,
                "ros_integration": self.ros_integration,
                "action_contract": self.action_contract,
                "state_contract": self.state_contract,
                "processors": self.processors,
                "n_action_steps": self.n_action_steps,
                "image_preprocessing": self.image_preprocessing,
                "starting_pose": self.starting_pose,
            }
            set_reward_forbidden = sorted(
                name for name, value in forbidden_reward.items() if value is not None
            )
            if set_reward_forbidden:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='reward' forbids "
                    f"these fields: {set_reward_forbidden!r}. A reward monitor is a "
                    "pure perception consumer — it has no action contract, no "
                    "detector block, no ROS wrapper, and no VLA policy "
                    "preprocessing."
                )
            if self.actuators_required:
                raise ValueError(
                    f"RSkillManifest({self.name!r}): kind='reward' requires "
                    f"`actuators_required` to be empty (got {len(self.actuators_required)} "
                    "entries). A reward monitor actuates nothing."
                )
            return self

        # kind == "wam": no extra schema constraint here; loader rejects at
        # resolve time.
        return self

    @classmethod
    def from_yaml(cls, path: str) -> RSkillManifest:
        """Load and validate an ``rskill.yaml`` from disk.

        Args:
            path: Filesystem path to the manifest YAML.

        Returns:
            A validated :class:`RSkillManifest`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            pydantic.ValidationError: If the YAML fails schema validation.
        """
        import yaml  # noqa: PLC0415  # reason: yaml is a runtime dep; deferred to avoid import-time cost

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)


# ─── Skill evaluation results (rskills/<id>/eval/<benchmark>.json) ───────────
#
# Every benchmarked rSkill ships one ``eval/<benchmark>.json`` file per
# benchmark suite it has been (or will be) evaluated on.  Four shapes are
# already in tree (``rskills/{smolvla-libero, smolvla-metaworld, pi05-libero-nf4,
# xvla-libero}/eval/*.json``); CLAUDE.md §6.4 lists ``eval/`` as required
# packaging.  This schema pins the format so ``rSkill.from_yaml`` can
# validate every JSON it finds and ``openral benchmark report`` can aggregate
# across skills.


class RSkillEvalSource(BaseModel):
    """Provenance for a benchmark result block.

    Used inside :class:`RSkillEvalResult` so consumers know where the numbers
    came from (paper vs local reproduction) without having to read the
    ``_comment`` line.

    Attributes:
        paper: Plain-text title of the source paper / report.
        arxiv: Optional arxiv URL or id.
        model_variant: Which checkpoint variant the numbers describe
            (e.g. ``"SmolVLA (0.45B)"``).
        evaluated_by: Free-text — ``"upstream authors"`` or a contributor.
        reproduced_locally: ``True`` only when OpenRAL CI / contributor
            re-ran the benchmark and the listed numbers match.
        reproduction_planned: Optional plan for closing the gap when
            ``reproduced_locally`` is ``False``.
        reproduction_cli: Either a single command string or a structured
            ``{description, single_suite_example, all_suites, suite_max_steps,
            notes}`` block (matches the LIBERO eval JSON shape).
        table: Optional pointer at a specific table in the source paper.
        status: Optional state marker — ``"in_progress"``, ``"deferred"``,
            ``"reproduced"`` — for benchmarks that are not yet final.
    """

    model_config = ConfigDict(extra="allow")

    paper: str
    arxiv: str | None = None
    model_variant: str
    evaluated_by: str
    reproduced_locally: bool
    reproduction_planned: str | None = None
    reproduction_cli: dict[str, object] | str | None = None
    table: str | None = None
    status: str | None = None


class RSkillEvalBenchmark(BaseModel):
    """The benchmark suite a :class:`RSkillEvalResult` was measured against.

    Attributes:
        name: Canonical suite name (``"LIBERO"``, ``"MetaWorld"``,
            ``"ALOHA"``, ``"PushT"``, ``"RoboCasa"`` …).  Used by
            ``openral benchmark report`` to group rows.
        dataset: Optional HF dataset identifier.
        protocol: Free-text description of the eval protocol.
        robot: ``robot_id`` the benchmark was evaluated on.
        simulator: Free-text simulator description.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    dataset: str | None = None
    protocol: str
    robot: str
    simulator: str


class RSkillEvalResult(BaseModel):
    """Pydantic model of a ``rskills/<id>/eval/<benchmark>.json`` file.

    The ``results`` and ``baselines`` blocks are intentionally
    benchmark-specific — LIBERO has four sub-suites with success rates,
    MetaWorld carries an MT50 average, ALOHA has a single cube-transfer
    rate.  We require the metadata blocks (``source``, ``benchmark``,
    ``eval_config``) but leave ``results`` as a free-form dict so each
    benchmark can declare its own keys.

    Attributes:
        schema_version: Semver-ish version string for this on-disk format.
        source: Provenance — paper / reproduction state.
        benchmark: Suite identity (name, robot, simulator).
        eval_config: Free-form configuration the benchmark was run with
            (chunk size, image size, denoising steps, …).
        results: Free-form per-task / per-suite success rates.
        baselines: Optional free-form comparison numbers from prior work.
        trace_id: Hex OTel trace id (32 chars) for the rollout that
            produced this result. Set by ``openral benchmark run`` from the
            ``cli.command`` root span so reviewers can deep-link from
            ``rskills/<id>/eval/<benchmark>.json`` straight to the
            trace tree in Jaeger / Tempo. Optional — paper-cited
            numbers (``reproduced_locally: false``) leave it ``None``.

    Example:
        >>> # RSkillEvalResult.model_validate_json(
        >>> #     '{"schema_version": "0.1", "source": {...}, ...}')
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str = "0.1"
    source: RSkillEvalSource
    benchmark: RSkillEvalBenchmark
    eval_config: dict[str, object] = Field(default_factory=dict)
    results: dict[str, object]
    baselines: dict[str, object] = Field(default_factory=dict)
    # OTel design doc §6 P2: link the eval JSON to the OTel trace tree
    # that produced it. Hex trace id (32 chars), no ``traceparent``
    # — the dataset / trace cross-reference is a one-way pointer at
    # the top of the run, not a propagation seam.
    trace_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")

    @classmethod
    def from_json(cls, path: str) -> RSkillEvalResult:
        """Load and validate a ``rskills/<id>/eval/<benchmark>.json`` file.

        Args:
            path: Filesystem path to the JSON file.

        Returns:
            A validated :class:`RSkillEvalResult`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            pydantic.ValidationError: If the JSON fails schema validation.
        """
        import json as _json  # noqa: PLC0415  # reason: deferred import

        with open(path, encoding="utf-8") as fh:
            data = _json.load(fh)
        return cls.model_validate(data)


# ─── Sim environment specs (scene x task x VLA composition) ──────────────────
#
# These three models compose into a :class:`SimEnvironment`, the swappable
# triple consumed by ``openral_sim`` to validate rSkills before hardware
# deployment (CLAUDE.md §6 "WAMs / sim eval", ADR-0002).
#
# Design notes:
#   - The registry pattern (string ids ↔ Python factories) keeps these specs
#     fully serialisable and YAML-friendly while letting backend code live in
#     ``openral_sim.{policies,backends}``.
#   - Backends (LIBERO, MetaWorld, ...) are imported lazily by the factory;
#     the schemas have no hard dependency on physics packages.
#   - ``SimEnvironment`` itself does NOT bake in a specific physics engine —
#     the engine is declared on :class:`SceneSpec`.


class PhysicsBackend(str, Enum):
    """Physics / scene backend used to instantiate a :class:`SceneSpec`.

    Attributes:
        MUJOCO: Vanilla MuJoCo (CPU / single-env). Default for LIBERO, MetaWorld.
        MUJOCO_MJX: MuJoCo MJX (XLA, GPU-batched headless rollouts).
        PYBULLET: PyBullet (legacy adapters, contact-rich tabletop).
        ISAACSIM: NVIDIA Isaac Sim (Omniverse, GPU). Future.
        GENESIS: Genesis (physics-language unification). Future.
        MOCK: In-process mock with no physics — used for wiring smoketests.
    """

    MUJOCO = "mujoco"
    MUJOCO_MJX = "mujoco_mjx"
    PYBULLET = "pybullet"
    ISAACSIM = "isaacsim"
    GENESIS = "genesis"
    MOCK = "mock"


class SceneSpec(BaseModel):
    """A physics scene declaration — the WORLD the robot acts in.

    A scene is a deterministic, reproducible MuJoCo / MJX / etc. world: assets,
    lighting, cameras, fixed objects.  Tasks (:class:`TaskSpec`) are evaluated
    INSIDE a scene; multiple tasks can share one scene (e.g. ``libero_spatial``
    has 10 tasks per suite).

    Attributes:
        id: Stable scene identifier used by the eval registry, e.g.
            ``"libero_spatial"``, ``"metaworld_mt50"``, ``"so100_tabletop"``.
        backend: Physics backend used to instantiate the scene.
        assets_uri: Optional URI (file:// or hf://) pointing at scene assets
            (XML / MJCF / asset bundle).  When ``None``, the registered
            adapter resolves assets internally (LIBERO / MetaWorld pull theirs
            from their own packages).
        observation_height: Default render height in pixels for camera obs.
        observation_width: Default render width in pixels for camera obs.
        cameras: List of camera names the scene exposes.  Adapters use this
            both to render and to map sensors to VLA feature keys.
        backend_options: Backend-specific overrides
            (e.g. ``{"render_modes": ["rgb_array"]}``).  Opaque to the eval
            layer; passed through to the adapter.

    Example:
        >>> s = SceneSpec(id="libero_spatial", backend=PhysicsBackend.MUJOCO)
        >>> s.observation_width
        256
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    backend: PhysicsBackend = PhysicsBackend.MUJOCO
    assets_uri: str | None = None
    observation_height: int = Field(default=256, gt=0)
    observation_width: int = Field(default=256, gt=0)
    cameras: list[str] = Field(default_factory=list)
    backend_options: dict[str, object] = Field(default_factory=dict)


class RoboCasaBackendOptions(BaseModel):
    """Typed validator helper for ``SceneSpec.backend_options`` under RoboCasa.

    RoboCasa exposes two scenario modes:

    * **prebuilt** — pick one of the ~100 atomic-PnP / door / drawer /
      navigation tasks shipped with the package by name (e.g.
      ``"PnPCounterToCab"``). Set :attr:`prebuilt_task`; leave all
      procedural keys at their defaults.
    * **procedural** -- author a kitchen by composing
      :attr:`kitchen_style` x :attr:`layout_id` x :attr:`fixtures` x
      :attr:`spawn_objects` x :attr:`task_verb`. Set :attr:`mode` to
      ``"procedural"`` and leave :attr:`prebuilt_task` ``None``.

    The model is **purely additive**: it lives alongside
    :class:`SceneSpec` and is constructed by the RoboCasa scene adapter
    at factory time via ``RoboCasaBackendOptions.model_validate(
    scene.backend_options)``. The parent ``SceneSpec.backend_options:
    dict[str, object]`` field is unchanged, so this class does not
    constitute a schema migration (ADR-0015, CLAUDE.md §1.6).

    See :doc:`ADR-0015 </adr/0015-robocasa-isolated-backend-lazy-assets>`
    for the full backend rationale.

    Attributes:
        mode: ``"prebuilt"`` (default) or ``"procedural"``.
        prebuilt_task: One of RoboCasa's ~100 atomic task names, e.g.
            ``"PnPCounterToCab"``. Only valid when ``mode="prebuilt"``.
        kitchen_style: 0..9, picks one of RoboCasa's 10 kitchen aesthetic
            packs. Only valid when ``mode="procedural"``.
        layout_id: 0..9, picks one of RoboCasa's 10 floor plans. Only
            valid when ``mode="procedural"``.
        fixtures: Subset of RoboCasa fixture names to spawn. Empty list
            keeps the layout's default fixtures.
        spawn_objects: Subset of RoboCasa object asset names to spawn on
            counters / inside cabinets.
        task_verb: Coarse procedural-mode goal: pick-and-place, open,
            close, press, navigate. Required when ``mode="procedural"``.
        robots: RoboCasa robot composition names, e.g.
            ``["PandaMobile"]`` (default) or ``["GR1"]``. Must match
            entries the upstream RoboCasa env factory accepts.
        controller: RoboCasa controller name. ``"OSC_POSE"`` works for
            arm-style robots; the adapter validates against the
            controller config exposed by RoboCasa at import time.
        horizon: Maximum step budget for the underlying RoboCasa env.

    Example:
        >>> opts = RoboCasaBackendOptions(mode="prebuilt", prebuilt_task="PnPCounterToCab")
        >>> opts.task_verb is None
        True
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["prebuilt", "procedural"] = "prebuilt"
    prebuilt_task: str | None = None
    kitchen_style: int | None = Field(default=None, ge=0, le=9)
    layout_id: int | None = Field(default=None, ge=0, le=9)
    fixtures: list[str] = Field(default_factory=list)
    spawn_objects: list[str] = Field(default_factory=list)
    task_verb: Literal["pnp", "open", "close", "press", "navigate"] | None = None
    robots: list[str] = Field(default_factory=lambda: ["PandaMobile"])
    controller: str = "OSC_POSE"
    horizon: int = Field(default=500, gt=0)
    # Proprioception layout the wrapped policy expects. Each option
    # concatenates a different subset of `robot0_*` robosuite obs keys
    # into the `observation.state` vector emitted by `_RoboCasaSim`:
    #   "human300_16d"   base_to_eef_pos(3) + base_to_eef_quat(4)
    #                    + base_pos(3) + base_quat(4) + gripper_qpos(2)
    #                    -> 16-D, robocasa-benchmark/openpi
    #                    `pi05_pretrain_human300` schema.
    #   "smolvla_9d"      eef_pos(3) + eef_quat(4) + gripper_qpos(2)
    #                    -> 9-D, the older smolvla / pre-mg_300 layout.
    #   "gr1" robot0_joint_pos(17) + robot0_right_gripper_qpos(11)
    #                    + robot0_left_gripper_qpos(11) -> 39-D. The 17 joint
    #                    slots are waist(3) + right arm(7) + left arm(7) in
    #                    MJCF order; the per-part views the upstream GR1
    #                    fork's GR1ArmsAndWaistKeyConverter exposes
    #                    (`hand.right_hand`, `body.right_arm`, etc.) are
    #                    sub-slices into this same 39-D vector.
    state_layout: Literal[
        "smolvla_9d",
        "human300_16d",
        "gr1",
    ] = "human300_16d"

    # Scene-pool restrictors mirroring `robocasa.environments.kitchen.kitchen.Kitchen`
    # constructor kwargs. Independent of `mode` -- they apply equally to
    # prebuilt and procedural authoring (the upstream Kitchen base class
    # consumes them in both paths). Defaults are `None` so existing
    # configs remain unchanged: when the user leaves these empty, the
    # robocasa scene factory falls through to its own (uniform across
    # all 60 layouts x 60 styles) sampling. To match the canonical
    # benchmark eval split, pin
    #   obj_instance_split="B"
    #   layout_and_style_ids=[[1,1],[2,2],[4,4],[6,9],[7,10]]
    # (see robocasa/utils/eval_utils.py::create_eval_env). To match the
    # training-data collection distribution
    # (DAVIAN-Robotics/robocasa-MG_*) pin
    #   obj_instance_split="pretrain"
    #   layout_ids=[-2]   # robocasa shorthand for "all train layouts (11..60)"
    #   style_ids=[-2]
    # See robocasa/scripts/collect_demos.py for the per-split semantics.
    #
    # ADR-amendment note: this is a purely additive Pydantic field. Old
    # SimEnvironment YAMLs (which don't carry these keys) load
    # unchanged because each field defaults to None.
    obj_instance_split: str | None = None
    layout_and_style_ids: list[list[int]] | str | None = None
    layout_ids: list[int] | int | None = None
    style_ids: list[int] | int | None = None
    obj_groups: str | None = None
    """Pin the PnP target object to a specific RoboCasa object group / category
    (e.g. ``"baguette"``, ``"vegetable"``). Forwarded to the prebuilt PnP task's
    ``obj_groups`` kwarg so the sampled object is deterministic instead of the
    default ``"all"`` random draw. Only valid for prebuilt PickPlace tasks;
    leave ``None`` for door / drawer / navigation tasks."""
    # ``False`` (default) matches ``openral sim run`` semantics: the env
    # terminates at ``horizon`` so the runner can score the episode.
    # ``True`` is for continuous-mode consumers (``openral deploy sim``)
    # where the operator drives an indefinite stream of commands and we
    # never want robosuite to refuse a follow-up step with
    # ``ValueError: executing action in terminated episode``. The HAL
    # bringup forces it on automatically via
    # ``openral_hal.sim_bringup.build_sim_env_from_yaml`` regardless of
    # what the YAML declares.
    ignore_done: bool = False

    @model_validator(mode="after")
    def _xor_prebuilt_vs_procedural(self) -> RoboCasaBackendOptions:
        """Forbid mixing the two scenario authoring modes.

        ``mode="prebuilt"`` requires :attr:`prebuilt_task` and forbids
        any of the procedural-only keys (:attr:`kitchen_style`,
        :attr:`layout_id`, :attr:`fixtures`, :attr:`spawn_objects`,
        :attr:`task_verb`).

        ``mode="procedural"`` forbids :attr:`prebuilt_task` and requires
        at least one of the procedural-only keys to be set (otherwise
        the user has just selected the upstream RoboCasa procedural
        defaults, which is fine but ambiguous in a YAML — we force them
        to set :attr:`task_verb` at minimum so the intent is explicit).
        """
        procedural_keys_set = any(
            [
                self.kitchen_style is not None,
                self.layout_id is not None,
                bool(self.fixtures),
                bool(self.spawn_objects),
                self.task_verb is not None,
            ]
        )
        if self.mode == "prebuilt":
            if self.prebuilt_task is None:
                raise ValueError(
                    "RoboCasaBackendOptions(mode='prebuilt') requires "
                    "prebuilt_task to name one of the ~100 atomic tasks "
                    "RoboCasa ships (e.g. 'PnPCounterToCab')."
                )
            if procedural_keys_set:
                raise ValueError(
                    "RoboCasaBackendOptions: procedural keys (kitchen_style, "
                    "layout_id, fixtures, spawn_objects, task_verb) are not "
                    "valid when mode='prebuilt'."
                )
        else:  # mode == "procedural"
            if self.prebuilt_task is not None:
                raise ValueError(
                    "RoboCasaBackendOptions: prebuilt_task is not valid when mode='procedural'."
                )
            if not procedural_keys_set:
                raise ValueError(
                    "RoboCasaBackendOptions(mode='procedural') requires at "
                    "least one procedural key — set task_verb (and "
                    "kitchen_style / layout_id / fixtures / spawn_objects "
                    "to taste)."
                )
        return self


class TaskSpec(BaseModel):
    """A task declaration — WHAT the robot must achieve inside a scene.

    Tasks decouple scene assets (``SceneSpec``) from goal-conditioning. The
    same scene can host many tasks; the same task can occasionally be run in
    multiple compatible scenes (rare; usually 1:1 with ``scene_id``).

    Success is evaluated by the scene adapter — the reward / success signal
    comes from the underlying gym env (``info['is_success']``,
    ``info['success']``, terminal reward, …).  ``success_key`` lets manifest
    authors override which info field the runner reads.

    Attributes:
        id: Stable task identifier, e.g. ``"libero_spatial/task_0"``,
            ``"metaworld/push-v3"``.  Adapters split on ``/`` to resolve.
        scene_id: ID of the :class:`SceneSpec` this task runs in.
        instruction: Natural-language goal handed to the VLA as the
            ``"task"`` text input.  Some adapters override this with a
            description baked into the underlying suite (LIBERO, MetaWorld).
        max_steps: Episode budget. Adapters may clip to scene-internal limits.
            ``None`` means unset; ``BenchmarkScene`` enforces a concrete value
            via its model validator.
        success_key: Key inside ``info`` returned by ``env.step()`` whose
            truthy value marks task success. ``None`` means unset;
            :class:`BenchmarkScene` enforces a concrete value via its model
            validator (required for paper-comparison benchmarks).
        metadata: Free-form per-task notes (paper reference, dataset split, …).

    Example:
        >>> t = TaskSpec(
        ...     id="libero_spatial/task_0",
        ...     scene_id="libero_spatial",
        ...     instruction="pick up the black bowl",
        ...     max_steps=200,
        ...     success_key="is_success",
        ... )
        >>> t.max_steps
        200
        >>> t.success_key
        'is_success'
        >>> TaskSpec(id="libero_spatial/task_0", scene_id="libero_spatial").max_steps is None
        True
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    scene_id: str
    instruction: str = ""
    max_steps: int | None = Field(default=None, gt=0)
    success_key: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class VLASpec(BaseModel):
    """A VLA / policy declaration — the BRAIN driving the robot.

    Lightweight pointer to a policy: either an installed rSkill (referenced
    by manifest name) or a raw HF Hub repo. The eval registry resolves this
    to a runtime adapter that returns ``Action`` objects.

    Attributes:
        id: Adapter id in the eval registry, e.g. ``"smolvla"``, ``"pi05"``,
            ``"xvla"``, ``"random"``, ``"zero"``. Picks the loader.
        weights_uri: Where to fetch weights from. Pass a bare rSkill
            reference: a name (``smolvla-libero``), a path (``rskills/smolvla-libero``),
            or a bare HF repo ID (``OpenRAL/rskill-smolvla-libero``). The
            :class:`openral_sim.SimRunner` requires a locally-resolvable
            reference — raw ``"hf://"`` URIs are rejected. Other URI shapes
            (e.g. ``"mock://"``) are still parsed by the schema so unit tests
            of the eval registries can run without an rSkill, but they will
            fail-fast if used with the runner.
        device: Torch device override. ``"auto"`` picks ``cuda:0`` if
            available, otherwise ``cpu``.
        runtime: Optional runtime override; ``None`` means "use whatever the
            policy/manifest declares".
        quantization: Optional quantization override for this run.
        deterministic: When True, set ``torch.use_deterministic_algorithms``
            and disable cuDNN benchmarking.
        extra: Adapter-specific options, e.g. ``{"chunk_size": 16}``.

    Example:
        >>> v = VLASpec(id="smolvla", weights_uri="rskills/smolvla-libero")
        >>> v.device
        'auto'
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    weights_uri: str
    device: str = "auto"
    runtime: RSkillRuntime | None = None
    quantization: QuantizationConfig | None = None
    deterministic: bool = False
    extra: dict[str, object] = Field(default_factory=dict)


class SimEnvironment(BaseModel):
    """A full sim configuration: the swappable (robot x scene x task x VLA) tuple.

    The composed runtime form of a :class:`SimScene` plus an
    :class:`RSkillManifest`. ``openral_sim``'s CLI builds it; adapters
    consume it. Not loaded from YAML directly — ``from_yaml`` raises.

    Attributes:
        robot_id: ID into the eval ``ROBOTS`` registry. Matches a robot's
            ``RobotDescription.name`` or an embodiment shortcut (e.g.
            ``"so100_follower"``, ``"franka_panda"``).
        scene: Scene the robot acts in.
        task: Task to evaluate. ``task.scene_id`` MUST equal ``scene.id``.
        vla: Policy that drives the robot.
        base_pose: Optional per-episode mounting pose for the robot in the
            scene's world frame. ``None`` means "use the scene adapter's
            default" (e.g. an URDF's identity placement). Only honoured by
            **free-axis** scene adapters; setting it on a scene that
            registers with a ``fixed_robot=`` constraint (LIBERO,
            MetaWorld, RoboCasa, PushT, ALOHA) is rejected at CLI
            compose-time. The pose's ``frame_id`` is expected to be
            ``"world"``; adapters anchor on the robot's
            :class:`RobotDescription.base_frame`.
        seed: Global random seed (env reset, action sampling, torch RNG).
        n_episodes: Number of episodes to run.  ``1`` for a smoketest;
            ≥ 50 for an honest single-task success rate.
        record_video: Whether to ask the runner to record video frames.
        save_dir: Optional directory to write artefacts (video, traces, json
            summary). ``None`` means "discard outputs".
        metadata: Free-form notes (commit SHA, run owner, etc.).

    Example:
        >>> env = SimEnvironment(
        ...     robot_id="franka_panda",
        ...     scene=SceneSpec(id="libero_spatial"),
        ...     task=TaskSpec(
        ...         id="libero_spatial/task_0",
        ...         scene_id="libero_spatial",
        ...         instruction="pick up the cube",
        ...     ),
        ...     vla=VLASpec(id="smolvla", weights_uri="hf://lerobot/smolvla_libero"),
        ... )
        >>> env.task.scene_id == env.scene.id
        True
    """

    model_config = ConfigDict(extra="forbid")

    robot_id: str
    scene: SceneSpec
    task: TaskSpec
    vla: VLASpec
    base_pose: Pose6D | None = None
    seed: int = 0
    n_episodes: int = Field(default=1, gt=0)
    record_video: bool = False
    save_dir: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    def model_post_init(self, _context: object) -> None:
        """Cross-field validation: task.scene_id must match scene.id."""
        if self.task.scene_id != self.scene.id:
            raise ValueError(
                f"SimEnvironment.task.scene_id ({self.task.scene_id!r}) does not "
                f"match scene.id ({self.scene.id!r}); a task can only run in its "
                f"declared scene."
            )

    @classmethod
    def from_yaml(cls, path: str) -> SimEnvironment:  # pragma: no cover - removed
        """Removed: pass ``--rskill`` on the CLI; YAML carries only the scene.

        After the ``feat(core,sim): SceneEnvironment + openral sim run --rskill,
        no legacy`` commit, the YAML shape for ``openral sim run`` is
        :class:`SimScene` (scene + task only; ADR-0041 renamed
        ``SceneEnvironment`` → ``SimScene``) and the policy is
        supplied via the ``--rskill`` CLI flag. ``SimEnvironment`` is now
        the *composed runtime form* and is built by the CLI; loading it
        directly from YAML is no longer supported.
        """
        raise ROSConfigError(
            "SimEnvironment.from_yaml has been removed. "
            "Load the scene + task via SimScene.from_yaml(path) (or "
            "load_scene_strict(path, SimScene)) and supply the rSkill via "
            "--rskill rskills/<id> on the CLI; the CLI composes the "
            "SimEnvironment from those two artefacts."
        )


class BenchmarkMetadata(BaseModel):
    """Provenance block required on every BenchmarkScene.

    ``paper`` / ``honest_scope`` are the published-protocol citation +
    a one-sentence "what this eval actually measured" statement.

    ``display_name`` / ``simulator`` are optional paper-comparison
    labels (ADR-0042) that surface into ``RSkillEvalResult.benchmark``
    when present — ``display_name`` becomes ``benchmark.name`` and
    ``simulator`` becomes ``benchmark.simulator``. Pre-ADR-0042 these
    lived in a free-form dict on the deleted ``BenchmarkSpec``; moving
    them per-scene keeps them with their provenance and lets the
    aggregator emit identical JSON whether driven by ``run_benchmark``
    (suite) or ``run_benchmark_scene`` (single scene).
    """

    model_config = ConfigDict(extra="forbid")

    paper: str
    honest_scope: str
    display_name: str | None = None
    simulator: str | None = None


class DeployScene(BaseModel):
    """Environment-only scene for ``openral deploy run``.

    Carries the physics world and optional robot mount. No task, no eval
    config — this is a playground for the full OpenRAL stack.
    """

    model_config = ConfigDict(extra="forbid")

    scene: SceneSpec
    robot_id: str | None = None
    base_pose: Pose6D | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_vla_block(cls, v: object) -> object:
        if isinstance(v, dict) and "vla" in v:
            raise ROSConfigError(
                "'vla:' block is not accepted in scene configs. Pass --rskill on the CLI instead."
            )
        return v

    @classmethod
    def from_yaml(cls, path: str) -> DeployScene:
        """Load and validate a ``DeployScene`` YAML from disk."""
        import yaml  # noqa: PLC0415

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)


class SimScene(DeployScene):
    """Scene + task for ``openral sim run``.

    All task fields are overridable at the CLI level. ``max_steps`` and
    ``success_key`` are optional — omit them for open-ended experiments.
    Accepts a BenchmarkScene YAML transparently (the eval-specific fields
    ``n_episodes``, ``seed``, and ``metadata`` simply fill the defaults).
    """

    model_config = ConfigDict(extra="forbid")

    task: TaskSpec
    seed: int = 0
    n_episodes: int = Field(default=1, gt=0)
    record_video: bool = False
    save_dir: str | None = None
    metadata: dict[str, object] | BenchmarkMetadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _task_scene_id_matches(self) -> SimScene:
        if self.task.scene_id != self.scene.id:
            raise ValueError(
                f"task.scene_id={self.task.scene_id!r} does not match scene.id={self.scene.id!r}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str) -> SimScene:
        """Load and validate a ``SimScene`` YAML from disk."""
        import yaml  # noqa: PLC0415

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)


class BenchmarkScene(SimScene):
    """Full benchmark eval spec for ``openral benchmark``.

    ``n_episodes``, ``seed``, and ``metadata`` are required with no
    defaults — they must match the published evaluation protocol.
    The task must supply ``success_key`` and ``max_steps``.
    """

    model_config = ConfigDict(extra="forbid")

    n_episodes: int = Field(gt=0)  # required — no default
    seed: int  # required — no default
    metadata: BenchmarkMetadata  # typed, required

    @model_validator(mode="after")
    def _require_task_eval_fields(self) -> BenchmarkScene:
        if self.task.success_key is None:
            raise ValueError(
                "BenchmarkScene.task.success_key is required. "
                "Set it to the env info[] key that signals success (e.g. 'is_success')."
            )
        if self.task.max_steps is None:
            raise ValueError(
                "BenchmarkScene.task.max_steps is required. "
                "Set it to the paper's canonical step budget."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str) -> BenchmarkScene:
        """Load and validate a ``BenchmarkScene`` YAML from disk."""
        import yaml  # noqa: PLC0415

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)


# ─── Standalone protocol descriptor (eval suites are bare lists now) ─────────
#
# ADR-0009 originally split the "eval" responsibility into two named
# subsystems:
#   * SimEnvironment (above) — free-axis single rollouts. Every axis is a
#     field on the spec; the user composes the rollout they want.
#   * BenchmarkSpec — a Pydantic wrapper around a list of BenchmarkScenes
#     pinning a fixed (robot x scenes x tasks x protocol); only the VLA
#     varied. Loaded from ``benchmarks/<id>.yaml`` via
#     ``BenchmarkSpec.from_yaml``.
#
# ADR-0042 (June 2026) deleted the wrapper. A benchmark suite is now a
# bare ``list[BenchmarkScene]`` on disk and in memory; the suite-id is
# the filename stem. Load via :func:`openral_core.load_benchmark_suite`
# and validate the suite-level invariants (uniformity of robot_id,
# n_episodes, seed, metadata; unique task ids; non-empty list) via
# :func:`openral_core.raise_on_invalid_suite`. The output is still a
# validated :class:`RSkillEvalResult` JSON dropped into
# ``rskills/<vla>/eval/<benchmark_id>.json`` with
# ``reproduced_locally=true``.
#
# Task 10 of ADR-0041 (the precursor to ADR-0042) had already flattened
# the per-scene payload: each entry carries its own ``robot_id``, ``task``,
# ``n_episodes``, ``seed``, and :class:`BenchmarkMetadata` block.
# :class:`ProtocolSpec` survives as a standalone schema for ADR drafts
# and benchmark-report tooling that wants to describe a protocol outside
# a suite context.


class ProtocolSpec(BaseModel):
    """Stand-alone eval-protocol descriptor.

    Historically the ``protocol`` block on the deleted ``BenchmarkSpec``
    (ADR-0009). After the Task-10 scene-hierarchy convergence (ADR-0041,
    June 2026) a benchmark became a list of :class:`BenchmarkScene`s —
    each scene carries its own ``n_episodes`` / ``seed`` /
    ``task.success_key`` / ``task.max_steps`` — and ADR-0042 then
    deleted the ``BenchmarkSpec`` wrapper altogether. ``ProtocolSpec``
    is retained as a public schema for callers that want to describe a
    protocol independently (e.g. ADR drafts, benchmark-report tooling).

    Pins the methodology so two rSkills evaluated under the same benchmark
    produce apples-to-apples numbers. Authors of a benchmark should set
    these to match the published protocol of the suite they are reproducing
    (e.g. LIBERO: 10 episodes per task, fixed seed range, ``is_success``).

    Attributes:
        n_episodes: Number of independent episodes per task. Honest
            success rates need >= 10 per task for the published LIBERO /
            MetaWorld protocols.
        seeds: Seed list applied per task (paired index-wise with the
            episode index). Length MUST be >= ``n_episodes``; runners
            slice ``seeds[:n_episodes]`` so re-runs are reproducible.
        success_key: ``info`` key on the gym ``step()`` return whose truthy
            value marks task success. Default ``"is_success"`` (LIBERO
            convention); MetaWorld uses ``"success"``.
        max_steps: Per-task episode budget. Adapters may clip to
            scene-internal limits; the value here is the protocol ceiling.
        min_reps: Minimum number of completed episodes per task before the
            result JSON is written. Guards against a crash mid-suite
            producing a misleadingly partial roll-up. ``None`` means
            "require every task x episode".

    Example:
        >>> p = ProtocolSpec(n_episodes=10, seeds=list(range(10)), max_steps=280)
        >>> p.success_key
        'is_success'
    """

    model_config = ConfigDict(extra="forbid")

    n_episodes: int = Field(default=10, gt=0)
    seeds: list[int] = Field(default_factory=lambda: list(range(10)))
    success_key: str = "is_success"
    max_steps: int = Field(default=280, gt=0)
    min_reps: int | None = Field(default=None, ge=1)

    def model_post_init(self, _context: object) -> None:
        """Cross-field validation: seeds list must cover n_episodes."""
        if len(self.seeds) < self.n_episodes:
            raise ValueError(
                f"ProtocolSpec.seeds has {len(self.seeds)} entries but "
                f"n_episodes={self.n_episodes}; provide at least n_episodes "
                f"seeds so re-runs are reproducible."
            )
        if self.min_reps is not None and self.min_reps > self.n_episodes:
            raise ValueError(
                f"ProtocolSpec.min_reps ({self.min_reps}) exceeds "
                f"n_episodes ({self.n_episodes}); a benchmark cannot require "
                f"more completed reps than it schedules."
            )


# ─── Inference runner ─────────────────────────────────────────────
#
# ADR-0010 introduces a hardware inference runner: the loop that closes
# ``WorldState → Skill.step → HAL.send_action`` at a cadence, mirroring the
# sim ``SimRunner`` for hardware. The schemas below are the on-disk
# contract for ``openral deploy --config <yaml>`` (sibling of ``openral sim run``) and
# the in-process record returned by the runner.
#
# These schemas are additive — they do not change ``SimEnvironment``,
# ``RSkillEvalResult``, or any pre-existing model. ``SensorFrame`` and
# ``FrameEncoding`` are declared earlier in this file because
# :attr:`WorldState.image_frames` references them.


class SensorReaderBackend(str, Enum):
    """Which :class:`SensorReader` implementation to instantiate.

    ADR-0010 §SensorReader — four backends are reserved. ``opencv_thread``
    is the default and mirrors lerobot's per-camera background-thread
    pattern. ``ros2_image`` subscribes to a ROS 2 image topic published by
    a vendor driver. ``gstreamer`` runs a GStreamer pipeline whose appsink
    delivers frames (NVMM / DMA-BUF on Jetson; CPU bytes on x86).

    ``holoscan`` is **reserved-but-unimplemented**: ADR-0010 Amendment
    2026-05-12 evaluated NVIDIA Holoscan SDK as a parallel ingest backbone
    and deferred adoption (lean GStreamer with custom NvBufSurface glue
    won the comparison). The enum value exists so a future PR can add
    the backend additively without bumping the schema again; configs
    that select it today raise ``ROSConfigError`` at factory time.
    """

    OPENCV_THREAD = "opencv_thread"
    ROS2_IMAGE = "ros2_image"
    GSTREAMER = "gstreamer"
    HOLOSCAN = "holoscan"


class DeadlineOverrunPolicy(str, Enum):
    """What the inference runner does when a tick exceeds the deadline.

    ``warn`` logs + records an OTel attribute but still sends the action
    (mirrors lerobot's record loop). ``drop`` skips the action for this
    tick — used when stale actions are worse than no action (e.g. velocity
    control). ``raise`` is test-only: raises :class:`ROSDeadlineMissed`.
    """

    WARN = "warn"
    DROP = "drop"
    RAISE = "raise"


class SensorReaderConfig(BaseModel):
    """Per-sensor backend configuration for the inference runner.

    Picks which :class:`SensorReader` backend services a sensor and how the
    pipeline is parameterised. The optional ``publish_to_ros`` tee lets a
    GStreamer pipeline publish a downsampled stream to a ROS 2 topic for
    observability (rosbag2 / rqt_image_view) without putting the hot path
    through ROS.

    Attributes:
        sensor_id: Sensor name; MUST match a :attr:`SensorSpec.name` in the
            target :class:`RobotDescription`.
        backend: Which :class:`SensorReaderBackend` to instantiate.
        backend_params: Backend-specific keyword arguments forwarded to the
            reader constructor. For ``gstreamer`` typically a single
            ``"pipeline"`` string. For ``opencv_thread`` typically
            ``{"device": "/dev/video0", "fps": 30}``.
        max_age_ms: How stale a frame may be before
            ``SensorReader.read_latest`` raises. Defaults to ~3 frames at
            30 Hz.
        publish_to_ros: If True, the reader tees a downsampled stream to
            ``publish_topic`` at ``publish_rate_hz``.
        publish_topic: ROS 2 topic to publish to when ``publish_to_ros`` is
            True. Required iff ``publish_to_ros``.
        publish_rate_hz: Downsample rate for the ROS tee.

    Example:
        >>> SensorReaderConfig(
        ...     sensor_id="wrist_rgb",
        ...     backend=SensorReaderBackend.GSTREAMER,
        ...     backend_params={
        ...         "pipeline": "v4l2src device=/dev/video0 ! "
        ...         "nvv4l2decoder ! nvvideoconvert ! appsink"
        ...     },
        ...     publish_to_ros=True,
        ...     publish_topic="/cameras/wrist_rgb/image_raw",
        ...     publish_rate_hz=5.0,
        ... ).backend.value
        'gstreamer'
    """

    model_config = ConfigDict(extra="forbid")

    sensor_id: str
    backend: SensorReaderBackend = SensorReaderBackend.OPENCV_THREAD
    backend_params: dict[str, object] = Field(default_factory=dict)
    max_age_ms: int = Field(default=100, gt=0)
    publish_to_ros: bool = False
    publish_topic: str | None = None
    publish_rate_hz: float | None = Field(default=None, gt=0)

    def model_post_init(self, _context: object) -> None:
        """Cross-field validation for the ROS tee."""
        if self.publish_to_ros and self.publish_topic is None:
            raise ValueError(
                f"SensorReaderConfig({self.sensor_id!r}): publish_to_ros is "
                f"True but publish_topic is unset; a ROS tee needs a topic."
            )
        if self.publish_topic is not None and not self.publish_to_ros:
            raise ValueError(
                f"SensorReaderConfig({self.sensor_id!r}): publish_topic is "
                f"set but publish_to_ros is False; enable publish_to_ros "
                f"explicitly or drop the topic."
            )


class HalConfig(BaseModel):
    """Configuration for a HAL adapter instantiation by the inference runner.

    Picks which HAL adapter class to use (sim digital twin vs real hardware)
    and how to reach the robot (serial port, FCI URI, ROS 2 namespace).
    Adapter-specific fields are forwarded as ``params``.

    Attributes:
        adapter: HAL adapter id, e.g. ``"so100_follower"``,
            ``"so100_digital_twin"``, ``"franka_panda_real"``, ``"ur5e_real"``,
            ``"ros_control"``. Resolved against the HAL registry.
        transport: Transport-layer parameters. Examples:
            ``{"port": "/dev/ttyACM0", "baud": 1_000_000}`` for SO-100;
            ``{"fci_uri": "172.16.0.2"}`` for Franka;
            ``{"namespace": "/ur5e"}`` for ros2_control.
        params: Adapter-specific keyword arguments forwarded to the
            constructor (e.g. calibration overrides).

    Example:
        >>> HalConfig(
        ...     adapter="so100_follower",
        ...     transport={"port": "/dev/ttyACM0", "baud": 1_000_000},
        ... ).adapter
        'so100_follower'
    """

    model_config = ConfigDict(extra="forbid")

    adapter: str
    transport: dict[str, object] = Field(default_factory=dict)
    params: dict[str, object] = Field(default_factory=dict)


class RobotEnvironment(BaseModel):
    """A full hardware deployment configuration — the ``openral deploy`` artefact.

    Sibling of :class:`SimEnvironment` for real hardware. Loaded from YAML
    by ``openral_runner`` (planned, ADR-0010) and consumed by the
    hardware ``InferenceRunner``. The runner instantiates the HAL adapter,
    opens every :class:`SensorReader`, wires the
    :class:`~openral_world_state.aggregator.WorldStateAggregator`,
    constructs the :class:`~openral_rskill.Skill` from
    ``vla.weights_uri`` (a bare rSkill reference — name, path, or HF repo ID)
    and ticks at ``rate_hz``.

    Cross-field invariants enforced in :meth:`model_post_init`:

    * Every :attr:`SensorReaderConfig.sensor_id` MUST be unique within
      :attr:`sensors`.
    * :attr:`vla.weights_uri` MUST be a bare rSkill reference (no URI scheme)
      — the rSkill manifest is the contract between robot/sensors/preprocessing
      and policy weights (CLAUDE.md §6.4).

    Attributes:
        robot_id: ID into the ROBOTS registry — matches a
            :attr:`RobotDescription.name`. Examples: ``"so100_follower"``,
            ``"franka_panda"``, ``"ur5e"``.
        hal: HAL adapter + transport configuration.
        sensors: Per-sensor reader backend choices. The runner opens one
            :class:`SensorReader` per entry and feeds the
            :class:`WorldStateAggregator`.
        task: What the robot should achieve. Reused from
            :class:`SimEnvironment` so :attr:`TaskSpec.instruction` becomes
            the language prompt handed to the VLA.
        vla: Policy that drives the robot. ``vla.weights_uri`` MUST be a
            bare rSkill reference (name, path, or HF repo ID).
        safety: Optional :class:`SafetyEnvelope` override; falls back to the
            robot's :attr:`RobotDescription.safety` when ``None``.
        rate_hz: Foreground tick rate. Default 30 Hz (matches the
            :class:`WorldStateAggregator` publish rate).
        thumbnail_hz: Per-camera rate at which the runner encodes a JPEG
            thumbnail onto the ``sensors.read_latest`` span for the dashboard.
            Default 25 Hz; decoupled from ``rate_hz``. ``0`` disables thumbnails.
            End-to-end dashboard refresh is ``min(thumbnail_hz, span export
            flush rate)`` — see ``OPENRAL_OTEL_SPAN_SCHEDULE_DELAY_MS``.
        deadline_overrun_policy: Behavior when tick wall-time exceeds
            ``1 / rate_hz``.
        max_ticks: Optional cap — the runner exits cleanly after this many
            ticks. ``None`` means "run until ``task.max_steps`` or external
            stop".
        save_dir: Optional directory for traces / video / JSON summaries.
        metadata: Free-form notes (operator, run id, …).

    Example:
        >>> env = RobotEnvironment(
        ...     robot_id="so100_follower",
        ...     hal=HalConfig(adapter="so100_follower"),
        ...     sensors=[SensorReaderConfig(sensor_id="wrist_rgb")],
        ...     task=TaskSpec(
        ...         id="pick_cube/red",
        ...         scene_id="pick_cube/red",
        ...         instruction="pick up the red cube",
        ...     ),
        ...     vla=VLASpec(
        ...         id="smolvla",
        ...         weights_uri="rskills/smolvla-so100",
        ...     ),
        ... )
        >>> env.rate_hz
        30.0
    """

    model_config = ConfigDict(extra="forbid")

    robot_id: str
    hal: HalConfig
    sensors: list[SensorReaderConfig] = Field(default_factory=list)
    task: TaskSpec
    vla: VLASpec
    safety: SafetyEnvelope | None = None
    rate_hz: float = Field(default=30.0, gt=0)
    thumbnail_hz: float = Field(default=25.0, ge=0)
    deadline_overrun_policy: DeadlineOverrunPolicy = DeadlineOverrunPolicy.WARN
    max_ticks: int | None = Field(default=None, gt=0)
    save_dir: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    def model_post_init(self, _context: object) -> None:
        """Cross-field validation: unique sensors, bare rSkill weights_uri."""
        seen: set[str] = set()
        for sensor in self.sensors:
            if sensor.sensor_id in seen:
                raise ValueError(
                    f"RobotEnvironment({self.robot_id!r}) has duplicate "
                    f"sensor_id={sensor.sensor_id!r} in sensors; each "
                    f"sensor must be configured at most once."
                )
            seen.add(sensor.sensor_id)
        for bad in ("hf://", "local://", "file://", "http://", "https://"):
            if self.vla.weights_uri.startswith(bad):
                raise ValueError(
                    f"RobotEnvironment({self.robot_id!r}).vla.weights_uri must "
                    f"be a bare rSkill reference (name, path, or HF repo ID), "
                    f"got {self.vla.weights_uri!r}; hardware deployments resolve "
                    f"weights through an rSkill manifest for reproducibility "
                    f"(CLAUDE.md §6.4)."
                )

    @classmethod
    def from_yaml(cls, path: str) -> RobotEnvironment:
        """Load and validate a ``RobotEnvironment`` YAML from disk.

        Args:
            path: Filesystem path to the YAML config file.

        Returns:
            A validated :class:`RobotEnvironment`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            pydantic.ValidationError: If the YAML fails schema validation.
        """
        import yaml  # noqa: PLC0415  # reason: deferred to avoid import-time cost

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)


class TickResult(BaseModel):
    """One tick's record returned by :meth:`InferenceRunner.tick`.

    Carries the timing breakdown so the parent OTel span can attach exact
    sub-stage durations and the latency budget enforcement can flag
    violations without re-instrumenting.

    The hardware fields (``sensors_ms``..``hal_ms``, ``safety_violations``,
    ``action_applied``) are the original (v1) surface used by
    :class:`HardwareRunner`. The sim-specific fields
    (``step_idx``..``truncated``) were added by ADR-0010 amendment 1 when
    :class:`SimRunner` adopted per-step tick semantics; hardware leaves
    them at their defaults (``None``), so a hardware tick serialises
    identically to v1 under ``model_dump(exclude_none=True)``.

    Attributes:
        stamp_ns: Tick wall-clock timestamp in nanoseconds (tick start).
        tick_idx: 0-indexed tick counter within a run.
        sensors_ms: ``SensorReader.read_latest`` total wall-time.
        world_state_ms: ``WorldStateAggregator.snapshot`` wall-time.
        inference_ms: ``Skill.step`` wall-time (the chunk dispatch cost,
            not the full chunked inference — see :class:`ChunkedExecutor`).
        safety_ms: Safety check wall-time.
        hal_ms: ``HAL.send_action`` wall-time.
        tick_ms: End-to-end tick wall-time including the rate-limiter
            overhead.
        chunk_index: Index of the action played out from a chunked
            executor. ``None`` for non-chunked skills.
        safety_violations: List of safety-violation reason strings emitted
            during this tick. Non-empty implies the action was either
            clamped or dropped per the safety policy.
        action_applied: ``False`` when the
            :class:`DeadlineOverrunPolicy` was ``drop`` and the runner
            elected not to publish this tick's action, **or** when a sim
            tick is the reset-tick between episodes (no inference / env
            step happened).
        step_idx: 0-indexed step within the current episode. Set by
            :class:`SimRunner` on step-ticks; ``None`` on hardware ticks
            and on sim reset-ticks.
        episode_idx: 0-indexed episode within the current run. Set by
            :class:`SimRunner` on every tick (including reset-ticks);
            ``None`` on hardware ticks.
        reward: Env step reward for this tick. Set by :class:`SimRunner`
            on step-ticks; ``None`` on hardware ticks and reset-ticks.
        terminated: Whether the env signalled natural termination this
            tick. Set by :class:`SimRunner`; ``None`` elsewhere.
        truncated: Whether the env hit its step budget this tick. Set by
            :class:`SimRunner`; ``None`` elsewhere.
        trace_context: Full W3C ``traceparent`` for this tick's
            ``rskill.tick`` span, in the form
            ``00-<trace_id_hex>-<span_id_hex>-<flags_hex>``. Optional —
            set by the runner when an OTel context is active so offline
            consumers (dataset writers, post-hoc analysers) can resume
            the trace without re-deriving it from the live span. Default
            ``None`` for byte-identical v1 JSON under
            ``model_dump(exclude_none=True)``.
    """

    model_config = ConfigDict(extra="forbid")

    stamp_ns: int = Field(ge=0)
    tick_idx: int = Field(ge=0)
    sensors_ms: float = Field(default=0.0, ge=0)
    world_state_ms: float = Field(default=0.0, ge=0)
    inference_ms: float = Field(default=0.0, ge=0)
    safety_ms: float = Field(default=0.0, ge=0)
    hal_ms: float = Field(default=0.0, ge=0)
    tick_ms: float = Field(ge=0)
    chunk_index: int | None = None
    safety_violations: list[str] = Field(default_factory=list)
    action_applied: bool = True
    # ADR-0010 amendment 1: sim-only fields. Default None so hardware ticks
    # round-trip byte-identically with v1 JSON under exclude_none=True.
    step_idx: int | None = Field(default=None, ge=0)
    episode_idx: int | None = Field(default=None, ge=0)
    reward: float | None = None
    terminated: bool | None = None
    truncated: bool | None = None
    # OTel design doc §7 P1: persist the W3C traceparent for offline
    # consumers. The value at runtime is the live span's parent — the
    # field is purely a serialised escape hatch for dataset writers /
    # openral replay. Always optional so existing JSON round-trips under
    # exclude_none=True.
    trace_context: str | None = None


class RunResult(BaseModel):
    """Aggregated summary returned by :meth:`InferenceRunner.run`.

    Attributes:
        n_ticks: Total ticks executed.
        success: Task success when the runner has a success signal (sim,
            scripted goal). ``None`` for open-ended hardware runs.
        budget_violations: Count of ticks whose ``tick_ms`` exceeded the
            rSkill manifest's :attr:`RSkillLatencyBudget.per_chunk_ms`.
        avg_inference_ms: Mean of :attr:`TickResult.inference_ms`.
        p99_inference_ms: 99th-percentile of :attr:`TickResult.inference_ms`.
        avg_tick_ms: Mean of :attr:`TickResult.tick_ms`.
        p99_tick_ms: 99th-percentile of :attr:`TickResult.tick_ms`.
        trace_id: OTel trace id (hex) of the run's root span. ``None`` when
            tracing is not configured.
        save_dir: Directory where artefacts were written, when configured.
        metadata: Free-form metadata (runner id, host, git SHA, …).
    """

    model_config = ConfigDict(extra="forbid")

    n_ticks: int = Field(ge=0)
    success: bool | None = None
    budget_violations: int = Field(default=0, ge=0)
    avg_inference_ms: float = Field(default=0.0, ge=0)
    p99_inference_ms: float = Field(default=0.0, ge=0)
    avg_tick_ms: float = Field(default=0.0, ge=0)
    p99_tick_ms: float = Field(default=0.0, ge=0)
    trace_id: str | None = None
    save_dir: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


# ─── Failure evidence (ADR-0018 F3) ────────────────────────────────────────────


class _FailureEvidenceBase(BaseModel):
    """Common base for every :data:`FailureEvidence` variant.

    Each variant declares ``kind: Literal["..."] = "..."`` as the
    discriminator field. Subclasses must set ``kind`` to the exact
    string the union dispatcher matches on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class TimeoutEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_TIMEOUT`` — an operation missed its deadline.

    Attributes:
        kind: Discriminator (always ``"timeout"``).
        operation: Short name of the operation that timed out (e.g.
            ``"skill.step"``, ``"hal.read_state"``, ``"reasoner.tick"``).
        deadline_s: Configured deadline in seconds.
        elapsed_s: Actual elapsed wall-clock time in seconds.
    """

    kind: Literal["timeout"] = "timeout"
    operation: str
    deadline_s: float = Field(gt=0)
    elapsed_s: float = Field(ge=0)


class ForceEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_FORCE`` — measured force exceeded a safety limit.

    Attributes:
        kind: Discriminator (always ``"force"``).
        joint_or_ee: Name of the joint or end-effector that tripped the limit.
        measured_n: Measured force in newtons.
        limit_n: Configured ceiling in newtons.
    """

    kind: Literal["force"] = "force"
    joint_or_ee: str
    measured_n: float
    limit_n: float = Field(gt=0)


class WorkspaceEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_WORKSPACE`` — EE pose left the safety AABB.

    Attributes:
        kind: Discriminator (always ``"workspace"``).
        ee_name: End-effector that violated the box.
        measured_xyz: Measured position in metres.
        box_min: Box minimum corner (``x_min, y_min, z_min``).
        box_max: Box maximum corner (``x_max, y_max, z_max``).
    """

    kind: Literal["workspace"] = "workspace"
    ee_name: str
    measured_xyz: tuple[float, float, float]
    box_min: tuple[float, float, float]
    box_max: tuple[float, float, float]


class PerceptionStaleEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_PERCEPTION`` — a sensor frame went stale.

    Attributes:
        kind: Discriminator (always ``"perception"``).
        sensor_id: ``SensorSpec.name`` of the stale sensor.
        staleness_ms: Observed age of the last frame, in milliseconds.
        threshold_ms: Staleness threshold that was crossed, in milliseconds.
    """

    kind: Literal["perception"] = "perception"
    sensor_id: str
    staleness_ms: float = Field(ge=0)
    threshold_ms: float = Field(gt=0)


class CriticEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_CRITIC`` — a critic flagged an action below threshold.

    Attributes:
        kind: Discriminator (always ``"critic"``).
        critic_id: Identifier of the critic that fired (model id or hand-rolled).
        score: Critic output in the critic's native range.
        threshold: Configured pass threshold.
    """

    kind: Literal["critic"] = "critic"
    critic_id: str
    score: float
    threshold: float


class ControllerEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_CONTROLLER`` — a ros2_control controller faulted.

    Attributes:
        kind: Discriminator (always ``"controller"``).
        controller_name: Failing controller (e.g. ``"joint_trajectory_controller"``).
        state: Reported controller state (e.g. ``"inactive"``, ``"error"``).
        detail: Free-form detail from the controller manager.
    """

    kind: Literal["controller"] = "controller"
    controller_name: str
    state: str
    detail: str = ""


class SelfVerifyEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_SELFVERIFY`` — a self-check failed.

    Attributes:
        kind: Discriminator (always ``"selfverify"``).
        check: Short check identifier (e.g. ``"action_chunk.shape"``).
        expected: Expected value as a string.
        observed: Observed value as a string.
    """

    kind: Literal["selfverify"] = "selfverify"
    check: str
    expected: str
    observed: str


class HumanEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_HUMAN`` — a human triggered an intervention.

    Attributes:
        kind: Discriminator (always ``"human"``).
        actor: Identifier of the operator (e.g. ``"slack:alice"``, ``"gui"``).
        reason: Free-form reason given by the actor.
    """

    kind: Literal["human"] = "human"
    actor: str
    reason: str = ""


class WamEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_WAM`` — a world-action-model discrepancy.

    Attributes:
        kind: Discriminator (always ``"wam"``).
        horizon: Rollout horizon at which the discrepancy was measured.
        discrepancy: Scalar discrepancy in the WAM's native units.
        wam_id: Identifier of the WAM that fired.
    """

    kind: Literal["wam"] = "wam"
    horizon: int = Field(gt=0)
    discrepancy: float
    wam_id: str


class ReasonerTimeoutEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_REASONER_TIMEOUT`` — the LLM call missed its deadline.

    Attributes:
        kind: Discriminator (always ``"reasoner_timeout"``).
        model: Model identifier (e.g. ``"claude-opus-4-7"``).
        deadline_s: Configured deadline in seconds.
        elapsed_s: Actual elapsed wall-clock time in seconds.
    """

    kind: Literal["reasoner_timeout"] = "reasoner_timeout"
    model: str
    deadline_s: float = Field(gt=0)
    elapsed_s: float = Field(ge=0)


class CollisionEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_COLLISION`` — a proposed motion would collide (ADR-0030).

    Attributes:
        kind: Discriminator (always ``"collision"``).
        collision_kind: ``"self"`` (link vs link) or ``"world"`` (link vs
            obstacle).
        link_a: Robot link whose collision volume tripped the check.
        link_b_or_object: The other robot link (self-collision) or the world
            object / occupancy region (world-collision).
        horizon_step: Chunk step (horizon index) where the collision was first
            detected.
        min_distance_m: Signed minimum distance at detection, in metres
            (negative means interpenetration).
    """

    kind: Literal["collision"] = "collision"
    collision_kind: Literal["self", "world"]
    link_a: str
    link_b_or_object: str
    horizon_step: int = Field(ge=0)
    min_distance_m: float


class SuppressedSummaryEvidence(_FailureEvidenceBase):
    """Evidence for ``KIND_SUPPRESSED_SUMMARY`` — rolling rate-limit roll-up.

    Emitted at ~1 Hz by :class:`openral_observability.FailureBusPublisher`
    when one or more ``(kind, severity)`` buckets dropped events during
    the past window.

    Attributes:
        kind: Discriminator (always ``"suppressed_summary"``).
        window_s: Length of the summarized window, in seconds.
        kinds: Parallel array of suppressed ``KIND_*`` values.
        severities: Parallel array of suppressed ``SEVERITY_*`` values.
        counts: Parallel array of dropped-event counts per bucket.
    """

    kind: Literal["suppressed_summary"] = "suppressed_summary"
    window_s: float = Field(gt=0)
    kinds: list[int] = Field(default_factory=list)
    severities: list[int] = Field(default_factory=list)
    counts: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_parallel_arrays(self) -> SuppressedSummaryEvidence:
        """Enforce that ``kinds`` / ``severities`` / ``counts`` are parallel."""
        if not (len(self.kinds) == len(self.severities) == len(self.counts)):
            msg = (
                "SuppressedSummaryEvidence requires parallel arrays "
                f"(kinds={len(self.kinds)}, severities={len(self.severities)}, "
                f"counts={len(self.counts)})"
            )
            raise ROSConfigError(msg)
        return self


FailureEvidence: TypeAlias = (
    TimeoutEvidence
    | ForceEvidence
    | WorkspaceEvidence
    | PerceptionStaleEvidence
    | CriticEvidence
    | ControllerEvidence
    | SelfVerifyEvidence
    | HumanEvidence
    | WamEvidence
    | ReasonerTimeoutEvidence
    | CollisionEvidence
    | SuppressedSummaryEvidence
)
"""Discriminated union for ``FailureTrigger.evidence_json`` payloads.

The discriminator field is ``kind`` (a string Literal on each variant).
Consumers decode an incoming ``evidence_json`` string with::

    from pydantic import TypeAdapter
    from openral_core import FailureEvidence

    evidence = TypeAdapter(FailureEvidence).validate_json(msg.evidence_json)

Producers serialize via ``evidence.model_dump_json()``.

The union mirrors the ``KIND_*`` constants on
``openral_msgs/msg/FailureTrigger`` (ADR-0018 F3). Adding a new
variant requires adding a new ``KIND_*`` constant to the IDL.
"""


# ─── Perception event metadata (ADR-0018 F6) ───────────────────────────────────


class _PerceptionEventBase(BaseModel):
    """Common base for every :data:`PerceptionEventMetadata` variant.

    Each variant declares ``kind: Literal["..."] = "..."`` as the
    discriminator field. The discriminator is identical to the
    ``/openral/perception/<kind>`` ROS 2 topic suffix the
    :class:`openral_runner.backends.gstreamer.perception_tee.PerceptionEventPublisher`
    publishes onto (ADR-0018 §3, "Topology of /openral/perception/events").
    Producers serialise via ``model_dump_json()`` and stuff the result
    into ``PromptStamped.metadata_json``; consumers decode with
    ``TypeAdapter(PerceptionEventMetadata).validate_json(...)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sensor_id: str
    """``SensorSpec.name`` of the camera that emitted the event."""


class ObjectDetection2D(BaseModel):
    """A single 2D detection inside an :class:`ObjectsMetadata` event.

    The 3D :class:`DetectedObject` lives in :class:`WorldState`; the 2D
    form here is what a per-camera detector (``nvinfer``, ``tflite``,
    ``cv2``-CPU) emits before any pose lift / fusion.

    Attributes:
        label: Semantic class label as produced by the model.
        confidence: Detection confidence in ``[0, 1]``.
        bbox_xyxy: Axis-aligned bounding box in pixels
            ``(x_min, y_min, x_max, y_max)``; image origin top-left.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[int, int, int, int]


class MotionMetadata(_PerceptionEventBase):
    """Perception event for ``/openral/perception/motion``.

    Emitted when a frame-difference detector measures sub-frame motion
    above a configured threshold. ``region_bbox`` is set when the
    detector localises the moving pixels (top-left origin in pixels);
    ``None`` means "motion magnitude only, no localisation."

    Attributes:
        kind: Discriminator (always ``"motion"``).
        magnitude: Mean absolute per-pixel difference, normalised to
            ``[0, 1]`` against the encoding's full-scale range.
        threshold: Magnitude threshold that fired the event.
        region_bbox: Optional bounding box ``(x_min, y_min, x_max, y_max)``
            of the moving region in pixels.
    """

    kind: Literal["motion"] = "motion"
    magnitude: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    region_bbox: tuple[int, int, int, int] | None = None


class ObjectsMetadata(_PerceptionEventBase):
    """Perception event for ``/openral/perception/objects``.

    Emitted by the event leg's detector element (``nvinfer`` on Jetson,
    ``tflite`` on CPU, etc.) when one or more objects are detected.

    Attributes:
        kind: Discriminator (always ``"objects"``).
        detections: Per-object 2D detections, ordered by descending
            ``confidence``.
        model_id: Identifier of the detector that fired (e.g.
            ``"yolov8n"``, ``"nvinfer:resnet50"``).
        frame_width: Pixel width of the frame the detector ran on; the
            ``bbox_xyxy`` of each detection is in this pixel space (ADR-0035
            lift scales it to the sensor's intrinsics resolution).
        frame_height: Pixel height of that frame.
    """

    kind: Literal["objects"] = "objects"
    detections: list[ObjectDetection2D]
    model_id: str
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)


class OcrMetadata(_PerceptionEventBase):
    """Perception event for ``/openral/perception/ocr``.

    Attributes:
        kind: Discriminator (always ``"ocr"``).
        text: Recognised text (already stripped of leading / trailing
            whitespace by the detector).
        confidence: Recogniser confidence in ``[0, 1]``.
        region_bbox: Optional bounding box ``(x_min, y_min, x_max, y_max)``
            of the recognised region in pixels.
    """

    kind: Literal["ocr"] = "ocr"
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    region_bbox: tuple[int, int, int, int] | None = None


class SceneChangeMetadata(_PerceptionEventBase):
    """Perception event for ``/openral/perception/scene_change``.

    Emitted when a histogram / structural-similarity detector measures a
    frame-to-frame distance above a configured threshold — typically
    used to wake the reasoner when the scene has changed enough to
    warrant a re-plan.

    Attributes:
        kind: Discriminator (always ``"scene_change"``).
        distance: Frame-to-frame distance in the detector's native
            metric (e.g. ``cv2.HISTCMP_CHISQR_ALT`` distance, ``1 - ssim``).
        threshold: Distance threshold that fired the event.
        metric: Identifier of the distance metric (e.g. ``"chisqr_alt"``,
            ``"1-ssim"``, ``"hellinger"``).
    """

    kind: Literal["scene_change"] = "scene_change"
    distance: float = Field(ge=0.0)
    threshold: float = Field(ge=0.0)
    metric: str


PerceptionEventMetadata: TypeAlias = (
    MotionMetadata | ObjectsMetadata | OcrMetadata | SceneChangeMetadata
)
"""Discriminated union for ``PromptStamped.metadata_json`` payloads on the
``/openral/perception/<kind>`` topics (ADR-0018 F6).

The discriminator field is ``kind`` (a string ``Literal`` on each
variant). Consumers decode an incoming ``metadata_json`` string with::

    from pydantic import TypeAdapter
    from openral_core import PerceptionEventMetadata

    metadata = TypeAdapter(PerceptionEventMetadata).validate_json(msg.metadata_json)

Producers serialise via ``metadata.model_dump_json()``.

The four variants map 1:1 onto the four per-kind topics fixed in
ADR-0018 §3 (``motion``, ``objects``, ``ocr``, ``scene_change``).
Adding a new variant requires adding a new ``/openral/perception/<kind>``
topic to the contract — by design, new kinds get new topics, not a
schema bump, so subscribers can subscribe to exactly the kinds they
care about.
"""


# ─── Reasoner tool calls (ADR-0018 F4) ─────────────────────────────────────────


class _ReasonerToolBase(BaseModel):
    """Common base for every :data:`ReasonerToolCall` variant.

    Each variant declares ``tool: Literal["..."] = "..."`` as the
    discriminator field. The reasoner (ADR-0018 F4) emits exactly one
    of these per tick via the LLM's structured-output / tool-use mode;
    the value is then dispatched onto the ROS graph (action client,
    service client, or publisher depending on the variant).

    Variants intentionally hold **no authority over actuation**: the
    reasoner never publishes ``ActionChunk`` itself (see ADR-0018 §4
    "Holds no authority over actuation"). :class:`ExecuteRskillTool` is
    indirect — it sends an action goal to ``rskill_runner_node`` which
    in turn produces the chunk and gates it through ``safety_node``.

    All variants are ``frozen=True`` and ``extra="forbid"`` so the LLM
    cannot smuggle ad-hoc fields onto the wire.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    rationale: str = Field(
        default="",
        validation_alias=AliasChoices("rationale", "rational"),
    )
    """Optional one-line LLM rationale recorded on the reasoner span and trace.

    Accepts both ``rationale`` (canonical) and ``rational`` (a common LLM
    mis-spelling — gemma/qwen tool-use models emit it often enough that
    rejecting the whole call as ``extra_forbidden`` would make the
    reasoner unusable). Both spellings populate the same field; the
    canonical name is used on the OTel span.
    """


class ExecuteRskillTool(_ReasonerToolBase):
    """Tool variant — invoke an installed, capability-matched rSkill.

    Dispatch: action goal on ``/openral/execute_skill`` (the
    ``openral_msgs/action/ExecuteSkill`` action server in F1's
    ``rskill_runner_node``). The chunk path that follows is
    ``Skill → /openral/candidate_action → safety_node →
    /openral/safe_action → HAL`` per ADR-0018 §3.

    Attributes:
        tool: Discriminator (always ``"execute_rskill"``).
        rskill_id: ``RSkillManifest.name`` of an installed, capable skill.
            Validated against the local registry by the reasoner at
            palette-build time; an unknown id raises
            :class:`ROSReasonerInvalidPlan`.
        prompt: Natural-language prompt forwarded to the skill's
            ``ExecuteSkill`` goal alongside the rskill_id. For VLA
            skills this is the policy's task-conditioning signal
            (SmolVLA writes it into ``observation["task"]``); for
            wrapped-ROS skills it's carried for trace / log context
            but the actual goal is built from ``goal_params_json``
            merged over the manifest's ``default_goal_json``.
        goal_params_json: ADR-0026 — serialised JSON object carrying
            per-skill typed parameters the LLM produces against the
            skill's :attr:`RSkillManifest.goal_params_schema`. Empty
            string disables the merge (today's behaviour). Wrapped-ROS
            skills (``kind: ros_action`` / ``ros_service``) deep-merge
            it over ``ros_integration.default_goal_json`` at
            configure-time; VLA skills accept the field and ignore it
            (their prompt is already the structured signal).
        deadline_s: Hard deadline in seconds for the action server to
            complete the goal. ``0`` means "use the skill manifest's
            default latency budget".
    """

    tool: Literal["execute_rskill"] = "execute_rskill"
    rskill_id: str = Field(min_length=1)
    prompt: str = ""
    goal_params_json: str = ""
    deadline_s: float = Field(default=0.0, ge=0.0)


class ReloadGstPipelineTool(_ReasonerToolBase):
    """Tool variant — swap a sensor's GStreamer pipeline at runtime.

    Dispatch: service call on
    ``/openral/sensors/<sensor_id>/reload_pipeline`` with the
    pipeline YAML payload. Lets the reasoner tune perception (switch
    resolution, enable an ``nvinfer`` leg, swap an RTSP source) without
    redeploying the runtime.

    Attributes:
        tool: Discriminator (always ``"reload_gst_pipeline"``).
        sensor_id: ``SensorSpec.name`` of the camera whose pipeline is
            being reloaded. Validated against the active runtime's
            sensor catalog at dispatch time.
        pipeline_yaml: Full YAML body of the new
            :class:`SensorReaderConfig`. The sensor node validates this
            against the Pydantic schema before accepting the swap.
    """

    tool: Literal["reload_gst_pipeline"] = "reload_gst_pipeline"
    sensor_id: str = Field(min_length=1)
    pipeline_yaml: str = Field(min_length=1)


class LifecycleTransitionTool(_ReasonerToolBase):
    """Tool variant — drive a ROS 2 lifecycle transition on a peer node.

    Dispatch: service call on ``<node>/change_state`` with the matching
    ``Transition`` id (``configure`` / ``activate`` / ``deactivate`` /
    ``cleanup``). Used to bring a HAL back online, restart a faulted
    node, or stage a controlled shutdown.

    Attributes:
        tool: Discriminator (always ``"lifecycle_transition"``).
        node: Fully-qualified ROS node name (e.g. ``"/openral/hal/so100"``).
        transition: One of ``"configure"``, ``"activate"``,
            ``"deactivate"``, ``"cleanup"``. Other transitions
            (``"shutdown"``, error-recovery) are deliberately omitted
            from the open-core palette per CLAUDE.md §6 Layer 6 —
            shutdown is owned by the safety supervisor, not the
            reasoner.
    """

    tool: Literal["lifecycle_transition"] = "lifecycle_transition"
    node: str = Field(min_length=1)
    transition: Literal["configure", "activate", "deactivate", "cleanup"]


class EmitPromptTool(_ReasonerToolBase):
    """Tool variant — republish a ``PromptStamped`` onto another topic.

    Dispatch: publish on ``target_topic`` (typically ``/openral/prompt``
    for self-cascading, but any ``PromptStamped`` topic is valid).
    Lets the reasoner stage multi-step plans, talk to a peer reasoner,
    or feed a downstream prompt-aware skill without going through an
    ExecuteSkill goal.

    Attributes:
        tool: Discriminator (always ``"emit_prompt"``).
        target_topic: Absolute ROS topic name (must start with ``"/"``).
            The reasoner's own subscription on ``/openral/prompt`` plus
            the prompt-router's FIFO queue (ADR-0018 F10) handle the
            cascade.
        text: Human-readable prompt body forwarded as
            ``PromptStamped.text``.
        metadata_json: Free-form JSON forwarded as
            ``PromptStamped.metadata_json``. Empty string when no
            structured metadata is needed.
    """

    tool: Literal["emit_prompt"] = "emit_prompt"
    target_topic: str = Field(min_length=2, pattern=r"^/")
    text: str = Field(min_length=1)
    metadata_json: str = ""


class RecallObjectTool(_ReasonerToolBase):
    """Tool variant (**read-only**) — recall a remembered object (ADR-0039).

    Queries the ADR-0038 scene-graph spatial memory and returns the object's
    ``map``-frame pose plus a camera-facing approach viewpoint and any occluding
    container to the reasoner's next reasoning step. Like every variant it
    **holds no authority over actuation** (ADR-0018 §4) — it only *reads*
    memory. The dispatch that runs the query and feeds the result back to the
    LLM is wired in ADR-0039 Phase 2 (this is the typed contract).

    Attributes:
        tool: Discriminator (always ``"recall_object"``).
        query: Free-text or label naming the object to recall (e.g.
            ``"the red mug"``). Mapped to a :class:`RecallObjectQuery` at dispatch.
        limit: Maximum number of ranked matches to return.
    """

    tool: Literal["recall_object"] = "recall_object"
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)


class ResolvePlaceTool(_ReasonerToolBase):
    """Tool variant (**read-only**) — resolve a place/room/agent to a goal (ADR-0039).

    Queries the ADR-0038 scene-graph memory for a navigation goal pose plus a
    ``traversable_to`` path. Read-only; **holds no authority over actuation**.
    Dispatch + result-return are ADR-0039 Phase 2.

    Attributes:
        tool: Discriminator (always ``"resolve_place"``).
        reference: Free-text, id, or label of the target (e.g. ``"the kitchen"``,
            ``"where I was standing"``). Mapped to a :class:`ResolvePlaceQuery`.
    """

    tool: Literal["resolve_place"] = "resolve_place"
    reference: str = Field(min_length=1)


class LocateInViewTool(_ReasonerToolBase):
    """Tool variant (**read-only**) — check if an object is in a *live* camera view (ADR-0043).

    The complement to :class:`RecallObjectTool`: where ``recall_object`` recalls a
    *remembered* object from the ADR-0038 scene-graph memory, ``locate_in_view``
    asks a live camera-mounted VLM detector (e.g. LocateAnything, ADR-0037) to
    look at the current frame *right now* and report whether the queried object is
    visible — and where. It runs the detector's open-vocabulary query on demand
    via the ``/openral/perception/locate_in_view`` ROS service and feeds the
    answer back to the LLM as a re-prompt (the prompt cascade). Like every variant
    it **holds no authority over actuation** (ADR-0018 §4) — it only *reads* a
    frame; the dispatch never gates the safety kernel.

    Attributes:
        tool: Discriminator (always ``"locate_in_view"``).
        query: Free-text object/description to look for (e.g. ``"the red mug"``).
            Sent verbatim as the detector's open-vocabulary query.
        camera: Optional camera selector. Empty (default) uses the detector's
            primary camera; otherwise names one of the detector's configured
            cameras so the reasoner can pick a viewpoint. **Not a hardcoded
            name** — the detector is camera-agnostic and maps the id to a topic.
        detector: Optional on-demand locator selector (ADR-0056). Empty (default)
            uses the deployment's default locator; otherwise an rSkill id / short
            alias of one of the on-demand locators in the graph (e.g.
            ``"omdet-turbo-locator"`` for fast simple "find X",
            ``"locateanything-3b"`` for complex referring expressions). The
            reasoner routes to ``/openral/perception/<detector>/locate_in_view``.
            Still **read-only** — choosing a model does not grant actuation.
    """

    tool: Literal["locate_in_view"] = "locate_in_view"
    query: str = Field(min_length=1)
    camera: str = ""
    detector: str = ""


class QuerySceneTool(_ReasonerToolBase):
    """Tool variant (**read-only**) — ask a scene VLM a question about the live view (ADR-0047).

    Backed by a ``kind: "vlm"`` rSkill (e.g. Qwen3.5-4B NF4) running in an
    out-of-process ZMQ sidecar. Where :class:`LocateInViewTool` answers *where*
    an object is (open-vocabulary localization via the detector), ``query_scene``
    answers *open-ended questions about the scene's state* — task-progress and
    success/failure verification the reasoner needs for its replanning ladder:
    "has the robot grasped the mug?", "is the bowl on the shelf?", "did we drop
    the object?", "is the table clear?".

    It captures the current frame of the requested camera, sends it plus the
    question to the VLM over the ``/openral/perception/query_scene`` ROS service,
    and feeds the free-text answer back to the LLM as a re-prompt (the prompt
    cascade). Like every variant it **holds no authority over actuation**
    (ADR-0018 §4) — it only *reads* a frame; the dispatch never gates the safety
    kernel. It is not a localizer: use :class:`LocateInViewTool` to find objects.

    Attributes:
        tool: Discriminator (always ``"query_scene"``).
        question: Natural-language question about the current scene
            (e.g. ``"Has the robot grasped the red mug?"``). Sent verbatim to
            the VLM as the textual prompt alongside the frame.
        camera: Optional camera selector. Empty (default) uses the perception
            node's primary camera; otherwise names one of its configured cameras
            so the reasoner can pick a viewpoint. **Not a hardcoded name.**
    """

    tool: Literal["query_scene"] = "query_scene"
    question: str = Field(min_length=1)
    camera: str = ""


class QueryTaskProgressTool(_ReasonerToolBase):
    """Tool variant (**read-only**) — ask the reward monitor how the task is going (ADR-0057).

    Backed by a ``kind: "reward"`` rSkill (Robometer-4B NF4) running in parallel
    with the active VLA in an out-of-process ZMQ sidecar. Where
    :class:`QuerySceneTool` answers *open-ended* scene questions as free text,
    ``query_task_progress`` returns a **quantitative** windowed assessment of the
    *current task*: normalized progress and success over the last
    :attr:`window_s` seconds, plus their trends and a ``stalled`` flag.

    It calls the ``/openral/perception/query_task_progress`` ROS service, which
    scores the monitor's buffered camera frames against the task instruction and
    returns ``progress_now`` / ``success_now`` / ``progress_trend`` /
    ``success_trend`` / ``stalled`` / ``succeeded``. The reasoner uses it to
    decide whether to continue, escalate to :class:`QuerySceneTool`, advance, or
    enter the replanning ladder. Like every variant it **holds no authority over
    actuation** (ADR-0018 §4) — the reward signal is advisory; the dispatch never
    gates the safety kernel.

    Attributes:
        tool: Discriminator (always ``"query_task_progress"``).
        window_s: How many seconds of recent frames to assess. Must be > 0;
            clamped to the monitor's configured ``frame_window_s``.
        task: Optional task-instruction override. Empty (default) reuses the
            instruction the monitor was co-activated with (the active VLA's goal).
    """

    tool: Literal["query_task_progress"] = "query_task_progress"
    window_s: float = Field(gt=0.0, default=8.0)
    task: str = ""


ReasonerToolCall: TypeAlias = (
    ExecuteRskillTool
    | ReloadGstPipelineTool
    | LifecycleTransitionTool
    | EmitPromptTool
    | RecallObjectTool
    | ResolvePlaceTool
    | LocateInViewTool
    | QuerySceneTool
    | QueryTaskProgressTool
)
"""Discriminated union over the reasoner tool variants (ADR-0018 §4; ADR-0039).

The discriminator field is ``tool`` (a string ``Literal`` on each
variant). Consumers decode an LLM tool-use payload with::

    from pydantic import TypeAdapter
    from openral_core import ReasonerToolCall

    call = TypeAdapter(ReasonerToolCall).validate_json(payload)

Producers (LLM clients) serialise via ``call.model_dump_json()``.

The first four variants are the actuation/effect palette ADR-0018 §4 commits
to. ADR-0039 adds two **read-only query** variants — :class:`RecallObjectTool`
and :class:`ResolvePlaceTool` — that only *read* the ADR-0038 spatial memory
(no actuation authority). Extending the palette requires (a) a new variant
here, (b) the corresponding ROS-side dispatch in
``openral_reasoner_ros.reasoner_node``, (c) a CLAUDE.md §6.2 / §7.6 amendment if
the new tool shifts the reasoner's authority surface. The two query variants'
dispatch + result-return path is ADR-0039 Phase 2; until then they are a typed
contract not yet exposed in the live provider palette.
"""
