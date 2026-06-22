"""Hypothesis fuzz tests for all openral_core Pydantic models.

Property tested: round-trip serialization guarantee.
  For any valid model instance:
    model → model_dump_json() → model_validate_json() == model
  And the serialized JSON must validate against the model's own JSON Schema.

Run with:
    uv run pytest tests/unit/test_schemas_fuzz.py -v
"""

from __future__ import annotations

import json
from typing import get_args

import jsonschema
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from openral_core.schemas import (
    _MODERN_PROCESSOR_FAMILIES,
    Action,
    ActuatorRequirement,
    ApproachViewpoint,
    BenchmarkName,
    CameraSimPlacement,
    CapsuleShape,
    CollisionEvidence,
    ControlMode,
    ControlModeSemantics,
    DeadlineOverrunPolicy,
    DetectedObject,
    DeviceInfo,
    EmbodimentKind,
    EmbodimentTag,
    EndEffectorSpec,
    FrameEncoding,
    GripperConvention,
    HalConfig,
    HalEntrypoints,
    Hand,
    IntrinsicsPinhole,
    JointSpec,
    JointState,
    JointType,
    LinkCollisionGeometry,
    ModelFamily,
    OccupancyGridRef,
    PhysicsBackend,
    Pose6D,
    QuantizationBackend,
    QuantizationConfig,
    QuantizationDtype,
    RecallObjectMatch,
    RecallObjectQuery,
    RecallObjectResult,
    RecallObjectTool,
    ResolvePlaceQuery,
    ResolvePlaceResult,
    ResolvePlaceTool,
    RobotCapabilities,
    RobotDescription,
    RobotEnvironment,
    RSkillAction,
    RSkillLatencyBudget,
    RSkillLicensePosture,
    RSkillManifest,
    RSkillProcessors,
    RSkillRuntime,
    RunResult,
    SafetyEnvelope,
    SceneGraph,
    SceneSpec,
    SensorBundle,
    SensorFrame,
    SensorModality,
    SensorReaderBackend,
    SensorReaderConfig,
    SensorSpec,
    SimEnvironment,
    SpatialEdge,
    SpatialNode,
    SpatialNodeKind,
    SpatialRelationKind,
    SphereShape,
    TaskSpec,
    TickResult,
    VLASpec,
    WorldCollisionPrimitive,
    WorldState,
)

# ─── Shared primitive strategies ──────────────────────────────────────────────

_name = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-/"),
    min_size=1,
    max_size=64,
)
_topic = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_/"),
    min_size=1,
    max_size=128,
)
_safe_float = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
_pos_float = st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e4)
_ns = st.integers(min_value=0, max_value=10**18)
_prob = st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1.0)

# ─── Model strategies ─────────────────────────────────────────────────────────

_intrinsics_st = st.builds(
    IntrinsicsPinhole,
    width=st.integers(min_value=1, max_value=4096),
    height=st.integers(min_value=1, max_value=4096),
    fx=_pos_float,
    fy=_pos_float,
    cx=_pos_float,
    cy=_pos_float,
)

_xyz = st.tuples(_safe_float, _safe_float, _safe_float)
_camera_sim_placement_st = st.builds(
    CameraSimPlacement,
    parent_body=st.none() | _name,
    pos=_xyz,
    target=_xyz,
    fovy_deg=st.none()
    | st.floats(allow_nan=False, allow_infinity=False, min_value=1.0, max_value=179.0),
)

_sensor_spec_st = st.builds(
    SensorSpec,
    name=_name,
    modality=st.sampled_from(list(SensorModality)),
    frame_id=_name,
    rate_hz=_pos_float,
    ros2_topic=_topic,
    ros2_msg_type=_name,
    catalog_id=st.none()
    | st.sampled_from(["generic/usb_uvc_rgb", "intel/realsense_d435i", "luxonis/oak_d_pro"]),
    sim_placement=st.none() | _camera_sim_placement_st,
)

_sensor_bundle_st = st.builds(
    SensorBundle,
    bundle_name=_name,
    sensors=st.lists(_sensor_spec_st, min_size=1, max_size=3),
)

_joint_spec_st = st.builds(
    JointSpec,
    name=_name,
    joint_type=st.sampled_from(list(JointType)),
    parent_link=_name,
    child_link=_name,
)

_end_effector_st = st.builds(
    EndEffectorSpec,
    name=_name,
    kind=st.sampled_from(["parallel_gripper", "suction", "dexterous_hand", "tool", "none"]),
    hand=st.sampled_from(list(Hand)),
    n_dof=st.integers(min_value=1, max_value=16),
)

_capabilities_st = st.builds(
    RobotCapabilities,
    can_lift_kg=_pos_float,
    onboard_compute_tops=_pos_float,
    onboard_memory_gb=_pos_float,
    embodiment_tags=st.lists(_name, max_size=5),
    supported_control_modes=st.lists(st.sampled_from(list(ControlMode)), max_size=4),
)

_safety_st = st.builds(
    SafetyEnvelope,
    max_ee_speed_m_s=_pos_float,
    max_ee_accel_m_s2=_pos_float,
    max_joint_speed_factor=st.floats(
        allow_nan=False, allow_infinity=False, min_value=0.01, max_value=1.0
    ),
    max_force_n=_pos_float,
    max_torque_nm=_pos_float,
    contact_force_threshold_n=_pos_float,
    cycle_time_violation_threshold_ms=_pos_float,
)

_hal_entrypoints_st = st.builds(
    HalEntrypoints,
    sim=st.none() | _name,
    real=st.none() | _name,
)

_robot_description_st = st.builds(
    RobotDescription,
    name=_name,
    embodiment_kind=st.sampled_from(list(EmbodimentKind)),
    joints=st.lists(_joint_spec_st, min_size=1, max_size=6),
    end_effectors=st.lists(_end_effector_st, max_size=2),
    sensors=st.lists(_sensor_spec_st, max_size=3),
    capabilities=_capabilities_st,
    safety=_safety_st,
    sdk_kind=st.sampled_from(["open", "closed_with_api", "closed"]),
    hal=_hal_entrypoints_st,
)

_joint_state_st = st.builds(
    JointState,
    name=st.lists(_name, min_size=1, max_size=12),
    position=st.lists(_safe_float, min_size=1, max_size=12),
    stamp_ns=_ns,
)

_pose6d_st = st.builds(
    Pose6D,
    xyz=st.tuples(_safe_float, _safe_float, _safe_float),
    quat_xyzw=st.tuples(_safe_float, _safe_float, _safe_float, _safe_float),
    frame_id=_name,
)

_detected_object_st = st.builds(
    DetectedObject,
    label=_name,
    confidence=_prob,
    pose=_pose6d_st,
)

_world_state_st = st.builds(
    WorldState,
    stamp_ns=_ns,
    joint_state=_joint_state_st,
    detected_objects=st.lists(_detected_object_st, max_size=4),
)

_action_st = st.builds(
    Action,
    control_mode=st.sampled_from(list(ControlMode)),
    horizon=st.integers(min_value=1, max_value=64),
    confidence=_prob,
    stamp_ns=_ns,
)

# ─── Collision geometry (ADR-0030) ──────────────────────────────────────────────

_radius = st.floats(allow_nan=False, allow_infinity=False, min_value=1e-3, max_value=1.0)
_capsule_shape_st = st.builds(
    CapsuleShape,
    radius_m=_radius,
    length_m=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=2.0),
)
_sphere_shape_st = st.builds(SphereShape, radius_m=_radius)
_collision_shape_st = st.one_of(_capsule_shape_st, _sphere_shape_st)

_link_collision_geometry_st = st.builds(
    LinkCollisionGeometry,
    link_name=_name,
    shape=_collision_shape_st,
    origin_xyz_rpy=st.tuples(*([_safe_float] * 6)),
)

_world_collision_primitive_st = st.builds(
    WorldCollisionPrimitive,
    shape=_collision_shape_st,
    pose=_pose6d_st,
    object_id=st.one_of(st.none(), _name),
)

_occupancy_grid_ref_st = st.builds(
    OccupancyGridRef,
    frame_id=_name,
    resolution_m=_pos_float.filter(lambda x: x > 0.0),
    width=st.integers(min_value=0, max_value=4096),
    height=st.integers(min_value=0, max_value=4096),
    origin=_pose6d_st,
    data_topic=_topic,
)

_collision_evidence_st = st.builds(
    CollisionEvidence,
    collision_kind=st.sampled_from(["self", "world"]),
    link_a=_name,
    link_b_or_object=_name,
    horizon_step=st.integers(min_value=0, max_value=64),
    min_distance_m=_safe_float,
)

# ─── Round-trip helper ────────────────────────────────────────────────────────

_FUZZ_SETTINGS = settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    # reason: pytest --cov instrumentation pushes per-example past the 200ms default on CI
    deadline=None,
)


def _round_trip_and_validate(cls: type, instance: object) -> None:  # type: ignore[type-arg]
    """Serialize → deserialize and JSON Schema validate."""
    serialized = instance.model_dump_json()  # type: ignore[attr-defined]
    reloaded = cls.model_validate_json(serialized)  # type: ignore[attr-defined]
    assert reloaded == instance, f"Round-trip failed for {cls.__name__}"

    data = json.loads(serialized)
    schema = cls.model_json_schema()  # type: ignore[attr-defined]
    jsonschema.validate(data, schema)


# ─── Fuzz tests ───────────────────────────────────────────────────────────────


@_FUZZ_SETTINGS
@given(_intrinsics_st)
def test_fuzz_intrinsics_pinhole(instance: IntrinsicsPinhole) -> None:
    """IntrinsicsPinhole round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(IntrinsicsPinhole, instance)


@_FUZZ_SETTINGS
@given(_camera_sim_placement_st)
def test_fuzz_camera_sim_placement(instance: CameraSimPlacement) -> None:
    """CameraSimPlacement round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(CameraSimPlacement, instance)


@_FUZZ_SETTINGS
@given(_sensor_spec_st)
def test_fuzz_sensor_spec(instance: SensorSpec) -> None:
    """SensorSpec round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(SensorSpec, instance)


@_FUZZ_SETTINGS
@given(_sensor_bundle_st)
def test_fuzz_sensor_bundle(instance: SensorBundle) -> None:
    """SensorBundle round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(SensorBundle, instance)


@_FUZZ_SETTINGS
@given(_joint_spec_st)
def test_fuzz_joint_spec(instance: JointSpec) -> None:
    """JointSpec round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(JointSpec, instance)


@_FUZZ_SETTINGS
@given(_end_effector_st)
def test_fuzz_end_effector_spec(instance: EndEffectorSpec) -> None:
    """EndEffectorSpec round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(EndEffectorSpec, instance)


@_FUZZ_SETTINGS
@given(_capabilities_st)
def test_fuzz_robot_capabilities(instance: RobotCapabilities) -> None:
    """RobotCapabilities round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(RobotCapabilities, instance)


@_FUZZ_SETTINGS
@given(_safety_st)
def test_fuzz_safety_envelope(instance: SafetyEnvelope) -> None:
    """SafetyEnvelope round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(SafetyEnvelope, instance)


@_FUZZ_SETTINGS
@given(_hal_entrypoints_st)
def test_fuzz_hal_entrypoints(instance: HalEntrypoints) -> None:
    """HalEntrypoints round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(HalEntrypoints, instance)


@_FUZZ_SETTINGS
@given(_robot_description_st)
def test_fuzz_robot_description(instance: RobotDescription) -> None:
    """RobotDescription round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(RobotDescription, instance)


@_FUZZ_SETTINGS
@given(_joint_state_st)
def test_fuzz_joint_state(instance: JointState) -> None:
    """JointState round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(JointState, instance)


@_FUZZ_SETTINGS
@given(_pose6d_st)
def test_fuzz_pose6d(instance: Pose6D) -> None:
    """Pose6D round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(Pose6D, instance)


@_FUZZ_SETTINGS
@given(_detected_object_st)
def test_fuzz_detected_object(instance: DetectedObject) -> None:
    """DetectedObject round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(DetectedObject, instance)


@_FUZZ_SETTINGS
@given(_world_state_st)
def test_fuzz_world_state(instance: WorldState) -> None:
    """WorldState round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(WorldState, instance)


@_FUZZ_SETTINGS
@given(_capsule_shape_st)
def test_fuzz_capsule_shape(instance: CapsuleShape) -> None:
    """CapsuleShape round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(CapsuleShape, instance)


@_FUZZ_SETTINGS
@given(_sphere_shape_st)
def test_fuzz_sphere_shape(instance: SphereShape) -> None:
    """SphereShape round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(SphereShape, instance)


@_FUZZ_SETTINGS
@given(_link_collision_geometry_st)
def test_fuzz_link_collision_geometry(instance: LinkCollisionGeometry) -> None:
    """LinkCollisionGeometry round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(LinkCollisionGeometry, instance)


@_FUZZ_SETTINGS
@given(_world_collision_primitive_st)
def test_fuzz_world_collision_primitive(instance: WorldCollisionPrimitive) -> None:
    """WorldCollisionPrimitive round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(WorldCollisionPrimitive, instance)


@_FUZZ_SETTINGS
@given(_occupancy_grid_ref_st)
def test_fuzz_occupancy_grid_ref(instance: OccupancyGridRef) -> None:
    """OccupancyGridRef round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(OccupancyGridRef, instance)


@_FUZZ_SETTINGS
@given(_collision_evidence_st)
def test_fuzz_collision_evidence(instance: CollisionEvidence) -> None:
    """CollisionEvidence round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(CollisionEvidence, instance)


@_FUZZ_SETTINGS
@given(_action_st)
def test_fuzz_action(instance: Action) -> None:
    """Action round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(Action, instance)


# ─── Quantization / device strategies ────────────────────────────────────────

_quant_config_st = st.builds(
    QuantizationConfig,
    dtype=st.sampled_from(list(QuantizationDtype)),
    backend=st.sampled_from(list(QuantizationBackend)),
    per_channel=st.booleans(),
    calibration_dataset=st.one_of(st.none(), _name),
)

_device_info_st = st.builds(
    DeviceInfo,
    device_str=st.sampled_from(["cpu", "cuda:0", "cuda:1", "mps"]),
    gpu_memory_bytes=st.integers(min_value=0, max_value=128 * (1 << 30)),
    cuda_compute_capability=st.one_of(
        st.none(),
        st.tuples(st.integers(min_value=3, max_value=12), st.integers(min_value=0, max_value=9)),
    ),
    cpu_count=st.integers(min_value=1, max_value=256),
    arch=st.sampled_from(["x86_64", "aarch64", "apple_silicon"]),
)


@_FUZZ_SETTINGS
@given(_quant_config_st)
def test_fuzz_quantization_config(instance: QuantizationConfig) -> None:
    """QuantizationConfig round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(QuantizationConfig, instance)


@_FUZZ_SETTINGS
@given(_device_info_st)
def test_fuzz_device_info(instance: DeviceInfo) -> None:
    """DeviceInfo round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(DeviceInfo, instance)


# ─── rSkill strategies ───────────────────────────────────────────────────────

# Latency budget fields are constrained to ``> 0`` (Field(gt=0)); use a strict
# positive float strategy so hypothesis doesn't generate invalid examples.
_strict_pos_float = st.floats(allow_nan=False, allow_infinity=False, min_value=1e-3, max_value=1e6)

_rskill_latency_budget_st = st.builds(
    RSkillLatencyBudget,
    per_chunk_ms=_strict_pos_float,
    warmup_ms=st.one_of(st.none(), _strict_pos_float),
    load_ms=st.one_of(st.none(), _strict_pos_float),
)

# V1: name + fallback_skill_id are HF-Hub regex-validated; build owner/repo
# pairs from the ASCII alnum + dot/underscore/hyphen alphabet (the regex
# only allows these characters; broader Unicode letters fail validation).
_HUB_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
_hub_segment = st.text(alphabet=_HUB_ALPHABET, min_size=1, max_size=12).filter(
    lambda s: s[0].isalnum()
)
_hub_id = st.builds(lambda owner, repo: f"{owner}/{repo}", _hub_segment, _hub_segment)

# V1: SemVer-validated.
_semver = st.builds(
    lambda a, b, c: f"{a}.{b}.{c}",
    st.integers(min_value=0, max_value=999),
    st.integers(min_value=0, max_value=999),
    st.integers(min_value=0, max_value=999),
)

# V1/V2: closed Literal sets — sample directly from get_args. Exclude
# "custom" from the embodiment-tag fuzz: it triggers the
# embodiment_extra cross-validator + per-actuator n_dof / vla_action_key
# requirement (ADR-0013), which has its own coverage in
# test_rskill_manifest.py. Fuzzing it here would degenerate into a
# filter against the cross-validator.
_NON_CUSTOM_EMBODIMENT_TAGS = [t for t in get_args(EmbodimentTag) if t != "custom"]
_embodiment_tag = st.sampled_from(_NON_CUSTOM_EMBODIMENT_TAGS)
_benchmark_name = st.sampled_from(list(get_args(BenchmarkName)))
_model_family = st.sampled_from(list(get_args(ModelFamily)))

# V1: weights_uri is a bare rSkill ref (name, path, or HF repo ID). Build bare names for fuzz.
_weights_uri = st.builds(lambda owner, repo: f"hf://{owner}/{repo}", _hub_segment, _hub_segment)

# V2: actuators_required is required, min_length=1. For non-"custom"
# embodiments the manifest may leave n_dof / vla_action_key as None
# (loader-side auto-fill); fuzz both branches.
_action_key_chars = st.characters(
    whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="._"
)

_GRIPPER_KINDS = (ControlMode.GRIPPER_BINARY, ControlMode.GRIPPER_POSITION)
_CARTESIAN_KINDS = (
    ControlMode.CARTESIAN_POSE,
    ControlMode.CARTESIAN_DELTA,
    ControlMode.CARTESIAN_TWIST,
)
_OTHER_KINDS = tuple(
    k for k in ControlMode if k not in _GRIPPER_KINDS and k not in _CARTESIAN_KINDS
)


def _semantics_for_kind(kind: ControlMode) -> st.SearchStrategy[ControlModeSemantics]:
    """Build kind-appropriate ControlModeSemantics (rSkill self-containment audit Gap 2).

    Gripper kinds require ``gripper_convention``; cartesian kinds require
    ``reference_frame``; other kinds forbid both.
    """
    mode = st.sampled_from(["absolute", "delta"])
    if kind in _GRIPPER_KINDS:
        return st.builds(
            ControlModeSemantics,
            mode=mode,
            gripper_convention=st.sampled_from(list(get_args(GripperConvention))),
        )
    if kind in _CARTESIAN_KINDS:
        return st.builds(
            ControlModeSemantics,
            mode=mode,
            reference_frame=st.text(alphabet=_action_key_chars, min_size=1, max_size=32),
        )
    return st.builds(ControlModeSemantics, mode=mode)


@st.composite
def _build_actuator_requirement(draw: st.DrawFn) -> ActuatorRequirement:
    kind = draw(st.sampled_from(list(ControlMode)))
    return ActuatorRequirement(
        kind=kind,
        n_dof=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=64))),
        vla_action_key=draw(
            st.one_of(
                st.none(),
                st.text(alphabet=_action_key_chars, min_size=1, max_size=32),
            )
        ),
        control_mode_semantics=draw(_semantics_for_kind(kind)),
    )


_actuator_requirement_st = _build_actuator_requirement()

# V1 + audit: modern model_family values require a processors block.
# Generate URIs that match _PROCESSOR_URI_PATTERN (file tail required).
_hub_processor_uri = st.builds(
    lambda owner, repo, fname: f"hf://{owner}/{repo}/{fname}.json",
    _hub_segment,
    _hub_segment,
    st.sampled_from(["policy_preprocessor", "policy_postprocessor", "stats", "normalize_buf"]),
)


@st.composite
def _build_processors(draw: st.DrawFn) -> RSkillProcessors:
    pre = draw(_hub_processor_uri)
    post = draw(_hub_processor_uri.filter(lambda u: u != pre))
    return RSkillProcessors(preprocessor_uri=pre, postprocessor_uri=post)


_processors_st = _build_processors()

# Source the modern-family set from the schema module itself so a new
# `model_family` (e.g. molmoact2) doesn't drift the fuzz strategy out of sync
# with the manifest validator it exercises.
_MODERN_FAMILIES = _MODERN_PROCESSOR_FAMILIES


@st.composite
def _build_rskill_manifest(draw: st.DrawFn) -> RSkillManifest:
    family = draw(_model_family)
    # Modern families require processors; act may omit (legacy path).
    if family in _MODERN_FAMILIES:
        processors: RSkillProcessors | None = draw(_processors_st)
    else:
        processors = draw(st.one_of(st.none(), _processors_st))
    return RSkillManifest(
        name=draw(_hub_id),
        version=draw(_semver),
        license=draw(st.sampled_from(list(RSkillLicensePosture))),
        role=draw(st.sampled_from(["s0", "s1", "s2"])),
        kind="vla",
        model_family=family,
        embodiment_tags=draw(st.lists(_embodiment_tag, min_size=1, max_size=3, unique=True)),
        runtime=draw(st.sampled_from(list(RSkillRuntime))),
        quantization=draw(_quant_config_st),
        weights_uri=draw(_weights_uri),
        chunk_size=draw(st.integers(min_value=1, max_value=1000)),
        latency_budget=draw(_rskill_latency_budget_st),
        actuators_required=draw(st.lists(_actuator_requirement_st, min_size=1, max_size=3)),
        min_vram_gb=draw(
            st.one_of(
                st.none(),
                st.dictionaries(
                    keys=st.sampled_from(list(QuantizationDtype)),
                    values=st.floats(
                        min_value=0.1,
                        max_value=128.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=1,
                    max_size=4,
                ),
            )
        ),
        # V1 forbids self-referential fallback; sample from a disjoint id space
        # by prefixing "fb-" so the equality check never trips.
        fallback_skill_id=draw(
            st.one_of(
                st.none(),
                st.builds(lambda h: f"fb-{h}", _hub_id),
            )
        ),
        benchmarks=draw(st.dictionaries(keys=_benchmark_name, values=_prob, max_size=4)),
        description=draw(st.text(min_size=1, max_size=500)),
        actions=draw(
            st.lists(
                st.sampled_from(list(RSkillAction)),
                min_size=1,
                max_size=4,
                unique=True,
            ),
        ),
        objects=draw(st.lists(st.text(min_size=1, max_size=16), max_size=4)),
        scenes=draw(st.lists(st.text(min_size=1, max_size=16), max_size=4)),
        processors=processors,
    )


_rskill_manifest_st = _build_rskill_manifest()


@_FUZZ_SETTINGS
@given(_rskill_latency_budget_st)
def test_fuzz_rskill_latency_budget(instance: RSkillLatencyBudget) -> None:
    """RSkillLatencyBudget round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(RSkillLatencyBudget, instance)


@_FUZZ_SETTINGS
@given(_rskill_manifest_st)
def test_fuzz_rskill_manifest(instance: RSkillManifest) -> None:
    """RSkillManifest round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(RSkillManifest, instance)


# ─── SimEnvironment strategies ───────────────────────────────────────────────

# ``SimEnvironment`` enforces ``task.scene_id == scene.id`` cross-field, so we
# build the SceneSpec / TaskSpec / VLASpec independently and then compose
# inside the SimEnvironment strategy with that invariant respected.

_scene_spec_st = st.builds(
    SceneSpec,
    id=_name,
    backend=st.sampled_from(list(PhysicsBackend)),
    assets_uri=st.one_of(st.none(), _name),
    observation_height=st.integers(min_value=1, max_value=4096),
    observation_width=st.integers(min_value=1, max_value=4096),
    cameras=st.lists(_name, max_size=4),
)


@st.composite
def _task_spec_for_scene(draw: st.DrawFn, scene_id: str) -> TaskSpec:
    return TaskSpec(
        id=draw(_name),
        scene_id=scene_id,
        instruction=draw(st.text(max_size=128)),
        max_steps=draw(st.integers(min_value=1, max_value=10_000)),
        success_key=draw(_name),
    )


_vla_spec_st = st.builds(
    VLASpec,
    id=_name,
    weights_uri=_name,
    device=st.sampled_from(["auto", "cpu", "cuda:0", "mps"]),
    runtime=st.one_of(st.none(), st.sampled_from(list(RSkillRuntime))),
    quantization=st.one_of(st.none(), _quant_config_st),
    deterministic=st.booleans(),
)


@st.composite
def _sim_environment_st(draw: st.DrawFn) -> SimEnvironment:
    scene = draw(_scene_spec_st)
    task = draw(_task_spec_for_scene(scene.id))
    vla = draw(_vla_spec_st)
    return SimEnvironment(
        robot_id=draw(_name),
        scene=scene,
        task=task,
        vla=vla,
        seed=draw(st.integers(min_value=0, max_value=10_000)),
        n_episodes=draw(st.integers(min_value=1, max_value=1000)),
        record_video=draw(st.booleans()),
    )


@_FUZZ_SETTINGS
@given(_scene_spec_st)
def test_fuzz_scene_spec(instance: SceneSpec) -> None:
    """SceneSpec round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(SceneSpec, instance)


@_FUZZ_SETTINGS
@given(_vla_spec_st)
def test_fuzz_vla_spec(instance: VLASpec) -> None:
    """VLASpec round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(VLASpec, instance)


@_FUZZ_SETTINGS
@given(_task_spec_for_scene("any_scene"))
def test_fuzz_task_spec(instance: TaskSpec) -> None:
    """TaskSpec round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(TaskSpec, instance)


@_FUZZ_SETTINGS
@given(_sim_environment_st())
def test_fuzz_sim_environment(instance: SimEnvironment) -> None:
    """SimEnvironment round-trips while preserving the ``task.scene_id == scene.id`` invariant."""
    _round_trip_and_validate(SimEnvironment, instance)
    assert instance.task.scene_id == instance.scene.id


# ─── Inference runner schemas (ADR-0010) ─────────────────────────────────────
#
# SensorFrame has a mutual-exclusion invariant on (data | topic | handle), so
# the strategy below splits across the three valid carry-modes. RobotEnvironment
# requires a bare rSkill reference for weights_uri.


@st.composite
def _sensor_frame_data_st(draw: st.DrawFn) -> SensorFrame:
    """SensorFrame instance carrying inline bytes."""
    return SensorFrame(
        sensor_id=draw(_name),
        stamp_monotonic_ns=draw(_ns),
        stamp_wall_ns=draw(_ns),
        encoding=draw(st.sampled_from(list(FrameEncoding))),
        width=draw(st.integers(min_value=1, max_value=4096)),
        height=draw(st.integers(min_value=1, max_value=4096)),
        channels=draw(st.integers(min_value=1, max_value=4)),
        data=draw(st.binary(min_size=1, max_size=64)),
    )


@st.composite
def _sensor_frame_topic_st(draw: st.DrawFn) -> SensorFrame:
    """SensorFrame instance carrying a ROS topic ref."""
    return SensorFrame(
        sensor_id=draw(_name),
        stamp_monotonic_ns=draw(_ns),
        stamp_wall_ns=draw(_ns),
        encoding=draw(st.sampled_from(list(FrameEncoding))),
        width=draw(st.integers(min_value=1, max_value=4096)),
        height=draw(st.integers(min_value=1, max_value=4096)),
        channels=draw(st.integers(min_value=1, max_value=4)),
        topic=draw(_topic),
    )


@st.composite
def _sensor_frame_handle_st(draw: st.DrawFn) -> SensorFrame:
    """SensorFrame instance carrying an opaque in-process handle."""
    return SensorFrame(
        sensor_id=draw(_name),
        stamp_monotonic_ns=draw(_ns),
        stamp_wall_ns=draw(_ns),
        encoding=draw(st.sampled_from(list(FrameEncoding))),
        width=draw(st.integers(min_value=1, max_value=4096)),
        height=draw(st.integers(min_value=1, max_value=4096)),
        channels=draw(st.integers(min_value=1, max_value=4)),
        handle=draw(st.integers(min_value=0, max_value=2**63 - 1)),
    )


_sensor_frame_st = st.one_of(
    _sensor_frame_data_st(),
    _sensor_frame_topic_st(),
    _sensor_frame_handle_st(),
)


@_FUZZ_SETTINGS
@given(_sensor_frame_st)
def test_fuzz_sensor_frame(instance: SensorFrame) -> None:
    """SensorFrame round-trips through JSON for every carry-mode (data | topic | handle)."""
    _round_trip_and_validate(SensorFrame, instance)
    populated = [bool(instance.data), instance.topic is not None, instance.handle is not None]
    assert sum(populated) == 1


@st.composite
def _sensor_reader_config_st(draw: st.DrawFn) -> SensorReaderConfig:
    """SensorReaderConfig respecting the publish_to_ros ↔ publish_topic invariant."""
    publish = draw(st.booleans())
    topic = draw(_topic) if publish else None
    rate = draw(_pos_float.filter(lambda x: x > 0)) if publish else None
    return SensorReaderConfig(
        sensor_id=draw(_name),
        backend=draw(st.sampled_from(list(SensorReaderBackend))),
        backend_params=draw(st.dictionaries(_name, st.text(max_size=32), max_size=4)),
        max_age_ms=draw(st.integers(min_value=1, max_value=10_000)),
        publish_to_ros=publish,
        publish_topic=topic,
        publish_rate_hz=rate,
    )


@_FUZZ_SETTINGS
@given(_sensor_reader_config_st())
def test_fuzz_sensor_reader_config(instance: SensorReaderConfig) -> None:
    """SensorReaderConfig round-trips and enforces publish_to_ros ↔ publish_topic."""
    _round_trip_and_validate(SensorReaderConfig, instance)
    if instance.publish_to_ros:
        assert instance.publish_topic is not None
    else:
        assert instance.publish_topic is None


_hal_config_st = st.builds(
    HalConfig,
    adapter=_name,
    transport=st.dictionaries(_name, st.text(max_size=64), max_size=4),
    params=st.dictionaries(_name, st.text(max_size=64), max_size=4),
)


@_FUZZ_SETTINGS
@given(_hal_config_st)
def test_fuzz_hal_config(instance: HalConfig) -> None:
    """HalConfig round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(HalConfig, instance)


_rskill_uri = _name
_rskill_vla_st = st.builds(
    VLASpec,
    id=_name,
    weights_uri=_rskill_uri,
    device=st.sampled_from(["auto", "cpu", "cuda:0", "mps"]),
    runtime=st.one_of(st.none(), st.sampled_from(list(RSkillRuntime))),
    deterministic=st.booleans(),
)


@st.composite
def _robot_environment_st(draw: st.DrawFn) -> RobotEnvironment:
    """RobotEnvironment with unique sensor ids and a bare-ref weights_uri."""
    n_sensors = draw(st.integers(min_value=0, max_value=4))
    ids = draw(st.lists(_name, min_size=n_sensors, max_size=n_sensors, unique=True))
    sensors = [
        SensorReaderConfig(sensor_id=sid, backend=SensorReaderBackend.OPENCV_THREAD) for sid in ids
    ]
    task = draw(_task_spec_for_scene("any_scene"))
    return RobotEnvironment(
        robot_id=draw(_name),
        hal=draw(_hal_config_st),
        sensors=sensors,
        task=task,
        vla=draw(_rskill_vla_st),
        rate_hz=draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False)),
        thumbnail_hz=draw(st.floats(min_value=0.0, max_value=60.0, allow_nan=False)),
        deadline_overrun_policy=draw(st.sampled_from(list(DeadlineOverrunPolicy))),
        max_ticks=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10_000))),
    )


@_FUZZ_SETTINGS
@given(_robot_environment_st())
def test_fuzz_robot_environment(instance: RobotEnvironment) -> None:
    """RobotEnvironment round-trips, sensor ids stay unique, weights_uri is a bare ref."""
    _round_trip_and_validate(RobotEnvironment, instance)
    sensor_ids = [s.sensor_id for s in instance.sensors]
    assert len(sensor_ids) == len(set(sensor_ids))
    assert not instance.vla.weights_uri.startswith(("rskill://", "hf://", "local://"))


_tick_result_st = st.builds(
    TickResult,
    stamp_ns=_ns,
    tick_idx=st.integers(min_value=0, max_value=10**9),
    sensors_ms=_pos_float,
    world_state_ms=_pos_float,
    inference_ms=_pos_float,
    safety_ms=_pos_float,
    hal_ms=_pos_float,
    tick_ms=_pos_float,
    chunk_index=st.one_of(st.none(), st.integers(min_value=0, max_value=1024)),
    safety_violations=st.lists(_name, max_size=3),
    action_applied=st.booleans(),
)


@_FUZZ_SETTINGS
@given(_tick_result_st)
def test_fuzz_tick_result(instance: TickResult) -> None:
    """TickResult round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(TickResult, instance)


_run_result_st = st.builds(
    RunResult,
    n_ticks=st.integers(min_value=0, max_value=10**6),
    success=st.one_of(st.none(), st.booleans()),
    budget_violations=st.integers(min_value=0, max_value=10**6),
    avg_inference_ms=_pos_float,
    p99_inference_ms=_pos_float,
    avg_tick_ms=_pos_float,
    p99_tick_ms=_pos_float,
    trace_id=st.one_of(st.none(), _name),
    save_dir=st.one_of(st.none(), _name),
)


@_FUZZ_SETTINGS
@given(_run_result_st)
def test_fuzz_run_result(instance: RunResult) -> None:
    """RunResult round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(RunResult, instance)


# ─── Spatial memory — scene graph (ADR-0038) ─────────────────────────────────────


@st.composite
def _build_spatial_node(draw: st.DrawFn, node_id: str | None = None) -> SpatialNode:
    is_container = draw(st.booleans())
    occludes = is_container and draw(st.booleans())
    first = draw(_ns)
    last = first + draw(st.integers(min_value=0, max_value=10**9))
    return SpatialNode(
        node_id=node_id if node_id is not None else draw(_name),
        kind=draw(st.sampled_from(list(SpatialNodeKind))),
        pose=draw(_pose6d_st),
        label=draw(st.text(max_size=16)),
        confidence=draw(_prob),
        bbox_3d=draw(st.one_of(st.none(), st.tuples(*([_safe_float] * 6)))),
        embedding_ref=draw(st.one_of(st.none(), _name)),
        is_container=is_container,
        occludes_contents=occludes,
        first_seen_ns=first,
        last_seen_ns=last,
        observation_count=draw(st.integers(min_value=1, max_value=1000)),
    )


_spatial_node_st = _build_spatial_node()

_spatial_edge_st = st.builds(
    SpatialEdge,
    src=_name,
    dst=_name,
    kind=st.sampled_from(list(SpatialRelationKind)),
)


@st.composite
def _build_scene_graph(draw: st.DrawFn) -> SceneGraph:
    ids = draw(st.lists(_name, min_size=1, max_size=5, unique=True))
    nodes = [draw(_build_spatial_node(node_id=node_id)) for node_id in ids]
    n_edges = draw(st.integers(min_value=0, max_value=5))
    edges = [
        SpatialEdge(
            src=draw(st.sampled_from(ids)),
            dst=draw(st.sampled_from(ids)),
            kind=draw(st.sampled_from(list(SpatialRelationKind))),
        )
        for _ in range(n_edges)
    ]
    return SceneGraph(nodes=nodes, edges=edges)


_scene_graph_st = _build_scene_graph()


@st.composite
def _build_recall_object_query(draw: st.DrawFn) -> RecallObjectQuery:
    text = draw(st.text(max_size=16))
    # At least one of text/label must be non-empty (RecallObjectQuery validator).
    label = draw(st.text(min_size=0 if text else 1, max_size=16))
    return RecallObjectQuery(
        text=text,
        label=label,
        near=draw(st.one_of(st.none(), _pose6d_st)),
        max_age_ns=draw(st.one_of(st.none(), _ns)),
        limit=draw(st.integers(min_value=1, max_value=100)),
    )


_recall_object_query_st = _build_recall_object_query()

_approach_viewpoint_st = st.builds(
    ApproachViewpoint,
    pose=_pose6d_st,
    standoff_m=st.floats(allow_nan=False, allow_infinity=False, min_value=1e-3, max_value=5.0),
    camera_frame_id=_name,
)

_recall_object_match_st = st.builds(
    RecallObjectMatch,
    node_id=_name,
    label=st.text(max_size=16),
    pose=_pose6d_st,
    score=_prob,
    last_seen_ns=_ns,
    approach=st.one_of(st.none(), _approach_viewpoint_st),
    inside_container_id=st.one_of(st.none(), _name),
)

_recall_object_result_st = st.builds(
    RecallObjectResult,
    matches=st.lists(_recall_object_match_st, max_size=4),
)

_resolve_place_query_st = st.builds(
    ResolvePlaceQuery,
    reference=_name,
    kind=st.one_of(st.none(), st.sampled_from(list(SpatialNodeKind))),
)

_resolve_place_result_st = st.builds(
    ResolvePlaceResult,
    node_id=_name,
    goal=_pose6d_st,
    path_node_ids=st.lists(_name, max_size=5),
)


@_FUZZ_SETTINGS
@given(_spatial_node_st)
def test_fuzz_spatial_node(instance: SpatialNode) -> None:
    """SpatialNode round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(SpatialNode, instance)


@_FUZZ_SETTINGS
@given(_spatial_edge_st)
def test_fuzz_spatial_edge(instance: SpatialEdge) -> None:
    """SpatialEdge round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(SpatialEdge, instance)


@_FUZZ_SETTINGS
@given(_scene_graph_st)
def test_fuzz_scene_graph(instance: SceneGraph) -> None:
    """SceneGraph round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(SceneGraph, instance)


@_FUZZ_SETTINGS
@given(_recall_object_query_st)
def test_fuzz_recall_object_query(instance: RecallObjectQuery) -> None:
    """RecallObjectQuery round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(RecallObjectQuery, instance)


@_FUZZ_SETTINGS
@given(_approach_viewpoint_st)
def test_fuzz_approach_viewpoint(instance: ApproachViewpoint) -> None:
    """ApproachViewpoint round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(ApproachViewpoint, instance)


@_FUZZ_SETTINGS
@given(_recall_object_match_st)
def test_fuzz_recall_object_match(instance: RecallObjectMatch) -> None:
    """RecallObjectMatch round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(RecallObjectMatch, instance)


@_FUZZ_SETTINGS
@given(_recall_object_result_st)
def test_fuzz_recall_object_result(instance: RecallObjectResult) -> None:
    """RecallObjectResult round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(RecallObjectResult, instance)


@_FUZZ_SETTINGS
@given(_resolve_place_query_st)
def test_fuzz_resolve_place_query(instance: ResolvePlaceQuery) -> None:
    """ResolvePlaceQuery round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(ResolvePlaceQuery, instance)


@_FUZZ_SETTINGS
@given(_resolve_place_result_st)
def test_fuzz_resolve_place_result(instance: ResolvePlaceResult) -> None:
    """ResolvePlaceResult round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(ResolvePlaceResult, instance)


# ─── Reasoner read-only query tools (ADR-0039) ───────────────────────────────────

_recall_object_tool_st = st.builds(
    RecallObjectTool,
    query=_name,
    limit=st.integers(min_value=1, max_value=100),
    rationale=st.text(max_size=24),
)

_resolve_place_tool_st = st.builds(
    ResolvePlaceTool,
    reference=_name,
    rationale=st.text(max_size=24),
)


@_FUZZ_SETTINGS
@given(_recall_object_tool_st)
def test_fuzz_recall_object_tool(instance: RecallObjectTool) -> None:
    """RecallObjectTool round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(RecallObjectTool, instance)


@_FUZZ_SETTINGS
@given(_resolve_place_tool_st)
def test_fuzz_resolve_place_tool(instance: ResolvePlaceTool) -> None:
    """ResolvePlaceTool round-trips through JSON and validates against its schema."""
    _round_trip_and_validate(ResolvePlaceTool, instance)
