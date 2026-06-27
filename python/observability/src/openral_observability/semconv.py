"""Semantic-convention constants for OpenRAL OpenTelemetry attributes.

Every OpenRAL span attribute, span-event name, and metric name lives here as
a ``Final[str]`` constant. Call sites import the constant rather than
hardcoding the string; this keeps the on-the-wire vocabulary refactorable
and lets ``mypy`` catch typos.

Namespace scheme (top-level prefix ``openral.``):

* ``openral.run.*`` — whole-CLI invocation (``run_id``, ``mode``, ``git_sha``)
* ``openral.tick.*`` — per-tick attributes (``idx``, ``rate_hz``, ``deadline_ms``)
* ``openral.rskill.*`` — rSkill identity & manifest fields
* ``openral.hal.*`` — HAL adapter identity & control mode
* ``openral.sensors.*`` — sensor reader stage
* ``openral.world_state.*`` — world-state aggregator snapshot
* ``openral.dataset.*`` — LeRobotDataset linkage
* ``openral.event.*`` — span-event names

The short-prefix ``rskill.*`` / ``inference.*`` / ``safety.*`` attributes
ship alongside the namespaced ``openral.rskill.*`` form for dashboard /
metric-label compatibility.
"""

from __future__ import annotations

from typing import Final

# ── rskill.* — short-prefix span/metric attributes ─────────────────────────

RSKILL_ID: Final[str] = "rskill.id"
RSKILL_ROLE: Final[str] = "rskill.role"

RSKILL_TICK_MS: Final[str] = "rskill.tick_ms"
RSKILL_INFERENCE_MS: Final[str] = "rskill.inference_ms"
RSKILL_SENSORS_MS: Final[str] = "rskill.sensors_ms"
RSKILL_WORLD_STATE_MS: Final[str] = "rskill.world_state_ms"
RSKILL_SAFETY_MS: Final[str] = "rskill.safety_ms"
RSKILL_HAL_MS: Final[str] = "rskill.hal_ms"
RSKILL_ACTION_APPLIED: Final[str] = "rskill.action_applied"
RSKILL_SAFETY_VIOLATIONS: Final[str] = "rskill.safety_violations"
RSKILL_EPISODE_IDX: Final[str] = "rskill.episode_idx"
RSKILL_STEP_IDX: Final[str] = "rskill.step_idx"
RSKILL_REWARD: Final[str] = "rskill.reward"
RSKILL_TERMINATED: Final[str] = "rskill.terminated"
RSKILL_TRUNCATED: Final[str] = "rskill.truncated"

INFERENCE_KIND: Final[str] = "inference.kind"
INFERENCE_CHUNK_INDEX: Final[str] = "inference.chunk_index"
INFERENCE_CHUNK_SIZE: Final[str] = "inference.chunk_size"
INFERENCE_ENGINE: Final[str] = "inference.engine"
INFERENCE_DEVICE: Final[str] = "inference.device"

SAFETY_SEVERITY: Final[str] = "safety.severity"
SAFETY_CHECK_NAME: Final[str] = "safety.check_name"
SAFETY_KERNEL: Final[str] = "safety.kernel"
SAFETY_CLAMPED: Final[str] = "safety.clamped"

# ── openral.run.* — CLI invocation ─────────────────────────────────────────

RUN_ID: Final[str] = "openral.run.id"
RUN_MODE: Final[str] = "openral.run.mode"
RUN_GIT_SHA: Final[str] = "openral.run.git_sha"

# ── openral.tick.* — runner cadence ────────────────────────────────────────

TICK_IDX: Final[str] = "openral.tick.idx"
TICK_RATE_HZ: Final[str] = "openral.tick.rate_hz"
TICK_DEADLINE_MS: Final[str] = "openral.tick.deadline_ms"

# ── openral.rskill.* — rSkill identity (namespaced) ────────────────────────

RSKILL_ID_NS: Final[str] = "openral.rskill.id"
RSKILL_REVISION_NS: Final[str] = "openral.rskill.revision"
RSKILL_ROLE_NS: Final[str] = "openral.rskill.role"
RSKILL_ACTION_HORIZON: Final[str] = "openral.rskill.action_horizon"

# ── openral.hal.* ──────────────────────────────────────────────────────────

HAL_ADAPTER: Final[str] = "openral.hal.adapter"
HAL_ROBOT_MODEL: Final[str] = "openral.hal.robot.model"
HAL_CONTROL_MODE: Final[str] = "openral.hal.control_mode"

# Robot-state vector attributes (recorded on `hal.read_state` spans).
# Lists are short (≤32 joints typical) so they live on spans, not metrics
# — cardinality discipline (design §9): no per-joint metric labels.
HAL_JOINT_NAMES: Final[str] = "openral.hal.joint.names"
HAL_JOINT_POSITIONS: Final[str] = "openral.hal.joint.positions"
HAL_JOINT_VELOCITIES: Final[str] = "openral.hal.joint.velocities"
HAL_JOINT_EFFORTS: Final[str] = "openral.hal.joint.efforts"
HAL_JOINT_POSITION_LIMITS_LO: Final[str] = "openral.hal.joint.position_limits_lo"
HAL_JOINT_POSITION_LIMITS_HI: Final[str] = "openral.hal.joint.position_limits_hi"
HAL_JOINT_VELOCITY_LIMITS: Final[str] = "openral.hal.joint.velocity_limits"
HAL_JOINT_EFFORT_LIMITS: Final[str] = "openral.hal.joint.effort_limits"
HAL_JOINT_STAMP_NS: Final[str] = "openral.hal.joint.stamp_ns"

# Commanded-action vector (recorded on `hal.send_action` spans). The
# action shape mirrors the policy's action_horizon * action_dim; we
# record the action chunk's first row (the row about to be applied)
# so the dashboard can overlay command-vs-reality without an explosion
# of attribute bytes.
HAL_ACTION_NEXT: Final[str] = "openral.hal.action.next"
HAL_ACTION_DIM: Final[str] = "openral.hal.action.dim"
HAL_ACTION_HORIZON: Final[str] = "openral.hal.action.horizon"
HAL_ACTION_APPLIED: Final[str] = "openral.hal.action.applied"

# End-effector pose + gripper (recorded on `world_state.snapshot` spans;
# the aggregator already owns `Pose6D` per end effector name).
HAL_EE_NAMES: Final[str] = "openral.hal.ee.names"
HAL_EE_POSE_PREFIX: Final[str] = "openral.hal.ee.pose"  # + "." + <ee_name>
HAL_GRIPPER_POSITION: Final[str] = "openral.hal.gripper.position"
HAL_GRIPPER_FORCE_N: Final[str] = "openral.hal.gripper.force_n"

# ── openral.sensors.* ──────────────────────────────────────────────────────

SENSORS_MODALITY: Final[str] = "openral.sensors.modality"
SENSORS_SOURCE: Final[str] = "openral.sensors.source"
SENSORS_AGE_MS: Final[str] = "openral.sensors.age_ms"
SENSORS_WIDTH: Final[str] = "openral.sensors.width"
SENSORS_HEIGHT: Final[str] = "openral.sensors.height"
SENSORS_CHANNELS: Final[str] = "openral.sensors.channels"
SENSORS_ENCODING: Final[str] = "openral.sensors.encoding"
# Throttled thumbnail (DeployRunner.thumbnail_hz, default 25 Hz per
# camera): a base64-encoded JPEG capped at 640x480 / q90 so the dashboard
# can show "what is the robot seeing" without competing with Foxglove's
# image pipeline.
SENSORS_THUMBNAIL_JPEG_B64: Final[str] = "openral.sensors.thumbnail_jpeg_b64"

# ── openral.system.* — host / GPU / CPU health ─────────────────────────────

SYSTEM_GPU_INDEX: Final[str] = "openral.system.gpu.index"
SYSTEM_GPU_NAME: Final[str] = "openral.system.gpu.name"

# ── openral.world_state.* ──────────────────────────────────────────────────

WORLD_STATE_STALENESS_MS: Final[str] = "openral.world_state.staleness_ms"
WORLD_STATE_COMPONENTS_STALE: Final[str] = "openral.world_state.components_stale"
WORLD_STATE_HAS_LATCHED_ERROR: Final[str] = "openral.world_state.has_latched_error"
WORLD_STATE_COMPONENT: Final[str] = "openral.world_state.component"

# ── openral.world_state.scene_objects.* (ADR-0038 durable spatial memory) ───
# Carried on the ``world.scene_objects`` span so the dashboard can show the
# remembered object nodes (a table + map overlay). ``LIST`` is a JSON array of
# ``{id,label,x,y,z,frame_id,confidence,last_seen_ns,observation_count,is_container}``.
WORLD_SCENE_OBJECTS_LIST: Final[str] = "openral.world_state.scene_objects.list"
WORLD_SCENE_OBJECTS_COUNT: Final[str] = "openral.world_state.scene_objects.count"
WORLD_SCENE_OBJECTS_FRAME: Final[str] = "openral.world_state.scene_objects.frame_id"
WORLD_SCENE_OBJECTS_SOURCE_NODE: Final[str] = "openral.world_state.scene_objects.source_node"

# ── openral.dataset.* ──────────────────────────────────────────────────────

DATASET_REPO_ID: Final[str] = "openral.dataset.repo_id"
DATASET_EPISODE_IDX: Final[str] = "openral.dataset.episode_idx"
DATASET_FRAME_IDX: Final[str] = "openral.dataset.frame_idx"
# ADR-0019: written by Rosbag2Sink / LeRobotDatasetSink on episode_end()
# so the Jaeger trace can be filtered to successful vs failed rollouts
# without correlating against the dataset filesystem.
DATASET_EPISODE_SUCCESS: Final[str] = "openral.dataset.episode.success"

# ── Span names ─────────────────────────────────────────────────────────────

SPAN_CLI_COMMAND: Final[str] = "cli.command"
SPAN_RSKILL_TICK: Final[str] = "rskill.tick"
SPAN_RSKILL_CONFIGURE: Final[str] = "rskill.configure"
SPAN_RSKILL_ACTIVATE: Final[str] = "rskill.activate"
SPAN_RSKILL_EXECUTE: Final[str] = "rskill.execute"
SPAN_RSKILL_CHUNK_INFERENCE: Final[str] = "rskill.chunk_inference"
SPAN_HAL_READ_STATE: Final[str] = "hal.read_state"
SPAN_HAL_SEND_ACTION: Final[str] = "hal.send_action"
SPAN_SENSORS_READ_LATEST: Final[str] = "sensors.read_latest"
SPAN_WORLD_STATE_SNAPSHOT: Final[str] = "world_state.snapshot"
SPAN_WORLD_SCENE_OBJECTS: Final[str] = "world.scene_objects"
SPAN_SAFETY_CHECK: Final[str] = "safety.check"
SPAN_REASONER_TICK: Final[str] = "reasoner.tick"
"""ADR-0018 F4 — one span per :meth:`openral_reasoner.ReasonerCore.tick`."""

# Reasoner span attributes (rendered with the ``reasoner.`` prefix).
REASONER_MODEL: Final[str] = "reasoner.model"
REASONER_TICK_IDX: Final[str] = "reasoner.tick.idx"
REASONER_TOOL: Final[str] = "reasoner.tool"
REASONER_RSKILL_ID: Final[str] = "reasoner.rskill_id"
REASONER_SUPPRESSED_REASON: Final[str] = "reasoner.suppressed_reason"
REASONER_ERROR_KIND: Final[str] = "reasoner.error_kind"
REASONER_FORCE: Final[str] = "reasoner.force"
# ADR-0073 — the active MissionState queue, serialized as JSON
# (``MissionState.to_summary``) so the live dashboard renders the task
# checklist (status / attempts / verdict) rather than only the text ledger.
REASONER_MISSION_JSON: Final[str] = "reasoner.mission_json"
# ADR-0018 2026-05-25 amendment — trigger taxonomy. Records which
# tier drove the LLM call: "A" (safety), "B" (replan), "C" (critic),
# "D" (operator / perception), or "heartbeat" (deadlock-insurance
# periodic fallback).
REASONER_TIER: Final[str] = "reasoner.tier"

SPAN_SIM_RUN: Final[str] = "sim.run"
SPAN_SIM_STEP: Final[str] = "sim.step"
SPAN_PHYSICS_STEP: Final[str] = "physics.step"

# ── Span-event names ───────────────────────────────────────────────────────

EVENT_ESTOP_REQUESTED: Final[str] = "openral.event.estop_requested"
EVENT_SENSOR_STALE: Final[str] = "openral.event.sensor_stale"
EVENT_STALENESS_LATCHED: Final[str] = "openral.event.staleness_latched"
EVENT_ERROR_LATCHED: Final[str] = "openral.event.error_latched"
EVENT_SAFETY_VIOLATION: Final[str] = "openral.event.safety_violation"
EVENT_ACTION_DROPPED: Final[str] = "openral.event.action_dropped"
EVENT_DEADLINE_MISSED: Final[str] = "openral.event.deadline_missed"
EVENT_CHUNK_PREFETCH_HIT: Final[str] = "openral.event.chunk_prefetch_hit"
EVENT_CHUNK_PREFETCH_MISS: Final[str] = "openral.event.chunk_prefetch_miss"
# ADR-0019: emitted by RolloutRecorder.episode_end() so a Jaeger query
# can pivot from a successful skill execution to the produced dataset row.
EVENT_EPISODE_CLOSED: Final[str] = "openral.event.episode_closed"

# ── Metric instrument names ────────────────────────────────────────────────

METRIC_TICK_DURATION: Final[str] = "openral.tick.duration"
METRIC_INFERENCE_DURATION: Final[str] = "openral.inference.duration"
METRIC_TICK_BUDGET_VIOLATIONS: Final[str] = "openral.tick.budget_violations"
METRIC_TICK_DEADLINE_MISSES: Final[str] = "openral.tick.deadline_misses"
METRIC_INFERENCE_TIMEOUTS: Final[str] = "openral.inference.timeouts"
METRIC_SAFETY_VIOLATIONS: Final[str] = "openral.safety.violations"
METRIC_SAFETY_CLAMPS: Final[str] = "openral.safety.clamps"
METRIC_HAL_READ_STATE_DURATION: Final[str] = "openral.hal.read_state.duration"
METRIC_HAL_SEND_ACTION_DURATION: Final[str] = "openral.hal.send_action.duration"
METRIC_HAL_ESTOP_COUNT: Final[str] = "openral.hal.estop.count"
METRIC_SENSORS_AGE_MS: Final[str] = "openral.sensors.age_ms"
METRIC_SENSORS_STALE_READS: Final[str] = "openral.sensors.stale_reads"
METRIC_WORLD_STATE_STALENESS_MS: Final[str] = "openral.world_state.staleness_ms"
METRIC_WORLD_STATE_COMPONENTS_STALE: Final[str] = "openral.world_state.components_stale"
METRIC_OBSERVABILITY_EXPORT_FAILURES: Final[str] = "openral.observability.export_failures"
METRIC_SIM_EPISODE_COUNT: Final[str] = "openral.sim.episode.count"

# System-health gauges (sampled at low frequency by the system_metrics
# collector). Labelled with `openral.system.gpu.index` for multi-GPU.
METRIC_SYSTEM_GPU_MEMORY_USED_MB: Final[str] = "openral.system.gpu.memory_used_mb"
METRIC_SYSTEM_GPU_MEMORY_TOTAL_MB: Final[str] = "openral.system.gpu.memory_total_mb"
METRIC_SYSTEM_GPU_UTIL_PCT: Final[str] = "openral.system.gpu.utilization_pct"
METRIC_SYSTEM_CPU_UTIL_PCT: Final[str] = "openral.system.cpu.utilization_pct"
METRIC_SYSTEM_RAM_USED_MB: Final[str] = "openral.system.ram.used_mb"
METRIC_SYSTEM_RAM_TOTAL_MB: Final[str] = "openral.system.ram.total_mb"
METRIC_SIM_EPISODE_SUCCESS: Final[str] = "openral.sim.episode.success"

# ── Closed-set label vocabularies (for metric labels) ──────────────────────

# Strict whitelists per design §9: only these label keys appear on metrics.
# Values are also closed sets (per metric) — see individual call sites.

LABEL_RSKILL_ID: Final[str] = "rskill.id"
LABEL_RSKILL_REVISION: Final[str] = "rskill.revision"
LABEL_HAL_ADAPTER: Final[str] = "hal.adapter"
LABEL_ROBOT_MODEL: Final[str] = "robot.model"
LABEL_CONTROL_MODE: Final[str] = "control_mode"
LABEL_ENGINE: Final[str] = "engine"
LABEL_DEVICE: Final[str] = "device"
LABEL_KIND: Final[str] = "kind"
LABEL_MODALITY: Final[str] = "modality"
LABEL_CHECK_NAME: Final[str] = "check_name"
LABEL_SEVERITY: Final[str] = "severity"
LABEL_SIGNAL_KIND: Final[str] = "signal_kind"
LABEL_COMPONENT: Final[str] = "component"
LABEL_REASON: Final[str] = "reason"

# Generic per-data-point threshold (ms): the contractual budget/deadline that
# governs this metric (e.g. a runner latency budget, a world-state staleness
# deadline). The dashboard promotes it to a threshold line + breach coloring on
# the metric's sparkline. Constant per series, so it does not add cardinality.
METRIC_THRESHOLD_MS: Final[str] = "openral.metric.threshold_ms"

# Direction of a threshold breach: ``"upper"`` (default — breach when the value
# rises ABOVE the threshold, e.g. a latency budget or staleness deadline) or
# ``"lower"`` (breach when it falls BELOW, e.g. a control-rate floor or a
# success-score minimum). Absent ⇒ ``"upper"``. Constant per series.
METRIC_THRESHOLD_DIR: Final[str] = "openral.metric.threshold_dir"
THRESHOLD_DIR_UPPER: Final[str] = "upper"
THRESHOLD_DIR_LOWER: Final[str] = "lower"

# ── Run-mode enum (closed set for openral.run.mode) ────────────────────────

RUN_MODE_SIM: Final[str] = "sim"
RUN_MODE_HARDWARE: Final[str] = "hardware"
RUN_MODE_BENCHMARK: Final[str] = "benchmark"

# ── Safety kernel enum (closed set for safety.kernel — see SAFETY_KERNEL) ──

SAFETY_KERNEL_NULL: Final[str] = "null"
SAFETY_KERNEL_CPP: Final[str] = "cpp"

__all__ = [
    "DATASET_EPISODE_IDX",
    "DATASET_EPISODE_SUCCESS",
    "DATASET_FRAME_IDX",
    "DATASET_REPO_ID",
    "EVENT_ACTION_DROPPED",
    "EVENT_CHUNK_PREFETCH_HIT",
    "EVENT_CHUNK_PREFETCH_MISS",
    "EVENT_DEADLINE_MISSED",
    "EVENT_EPISODE_CLOSED",
    "EVENT_ERROR_LATCHED",
    "EVENT_ESTOP_REQUESTED",
    "EVENT_SAFETY_VIOLATION",
    "EVENT_SENSOR_STALE",
    "EVENT_STALENESS_LATCHED",
    "HAL_ACTION_APPLIED",
    "HAL_ACTION_DIM",
    "HAL_ACTION_HORIZON",
    "HAL_ACTION_NEXT",
    "HAL_ADAPTER",
    "HAL_CONTROL_MODE",
    "HAL_EE_NAMES",
    "HAL_EE_POSE_PREFIX",
    "HAL_GRIPPER_FORCE_N",
    "HAL_GRIPPER_POSITION",
    "HAL_JOINT_EFFORTS",
    "HAL_JOINT_EFFORT_LIMITS",
    "HAL_JOINT_NAMES",
    "HAL_JOINT_POSITIONS",
    "HAL_JOINT_POSITION_LIMITS_HI",
    "HAL_JOINT_POSITION_LIMITS_LO",
    "HAL_JOINT_STAMP_NS",
    "HAL_JOINT_VELOCITIES",
    "HAL_JOINT_VELOCITY_LIMITS",
    "HAL_ROBOT_MODEL",
    "INFERENCE_CHUNK_INDEX",
    "INFERENCE_CHUNK_SIZE",
    "INFERENCE_DEVICE",
    "INFERENCE_ENGINE",
    "INFERENCE_KIND",
    "LABEL_CHECK_NAME",
    "LABEL_COMPONENT",
    "LABEL_CONTROL_MODE",
    "LABEL_DEVICE",
    "LABEL_ENGINE",
    "LABEL_HAL_ADAPTER",
    "LABEL_KIND",
    "LABEL_MODALITY",
    "LABEL_REASON",
    "LABEL_ROBOT_MODEL",
    "LABEL_RSKILL_ID",
    "LABEL_RSKILL_REVISION",
    "LABEL_SEVERITY",
    "LABEL_SIGNAL_KIND",
    "METRIC_HAL_ESTOP_COUNT",
    "METRIC_HAL_READ_STATE_DURATION",
    "METRIC_HAL_SEND_ACTION_DURATION",
    "METRIC_INFERENCE_DURATION",
    "METRIC_INFERENCE_TIMEOUTS",
    "METRIC_OBSERVABILITY_EXPORT_FAILURES",
    "METRIC_SAFETY_CLAMPS",
    "METRIC_SAFETY_VIOLATIONS",
    "METRIC_SENSORS_AGE_MS",
    "METRIC_SENSORS_STALE_READS",
    "METRIC_SIM_EPISODE_COUNT",
    "METRIC_SIM_EPISODE_SUCCESS",
    "METRIC_SYSTEM_CPU_UTIL_PCT",
    "METRIC_SYSTEM_GPU_MEMORY_TOTAL_MB",
    "METRIC_SYSTEM_GPU_MEMORY_USED_MB",
    "METRIC_SYSTEM_GPU_UTIL_PCT",
    "METRIC_SYSTEM_RAM_TOTAL_MB",
    "METRIC_SYSTEM_RAM_USED_MB",
    "METRIC_THRESHOLD_DIR",
    "METRIC_THRESHOLD_MS",
    "METRIC_TICK_BUDGET_VIOLATIONS",
    "METRIC_TICK_DEADLINE_MISSES",
    "METRIC_TICK_DURATION",
    "METRIC_WORLD_STATE_COMPONENTS_STALE",
    "METRIC_WORLD_STATE_STALENESS_MS",
    "REASONER_ERROR_KIND",
    "REASONER_FORCE",
    "REASONER_MISSION_JSON",
    "REASONER_MODEL",
    "REASONER_RSKILL_ID",
    "REASONER_SUPPRESSED_REASON",
    "REASONER_TICK_IDX",
    "REASONER_TIER",
    "REASONER_TOOL",
    "RSKILL_ACTION_APPLIED",
    "RSKILL_ACTION_HORIZON",
    "RSKILL_EPISODE_IDX",
    "RSKILL_HAL_MS",
    "RSKILL_ID",
    "RSKILL_ID_NS",
    "RSKILL_INFERENCE_MS",
    "RSKILL_REVISION_NS",
    "RSKILL_REWARD",
    "RSKILL_ROLE",
    "RSKILL_ROLE_NS",
    "RSKILL_SAFETY_MS",
    "RSKILL_SAFETY_VIOLATIONS",
    "RSKILL_SENSORS_MS",
    "RSKILL_STEP_IDX",
    "RSKILL_TERMINATED",
    "RSKILL_TICK_MS",
    "RSKILL_TRUNCATED",
    "RSKILL_WORLD_STATE_MS",
    "RUN_GIT_SHA",
    "RUN_ID",
    "RUN_MODE",
    "RUN_MODE_BENCHMARK",
    "RUN_MODE_HARDWARE",
    "RUN_MODE_SIM",
    "SAFETY_CHECK_NAME",
    "SAFETY_CLAMPED",
    "SAFETY_KERNEL",
    "SAFETY_KERNEL_CPP",
    "SAFETY_KERNEL_NULL",
    "SAFETY_SEVERITY",
    "SENSORS_AGE_MS",
    "SENSORS_CHANNELS",
    "SENSORS_ENCODING",
    "SENSORS_HEIGHT",
    "SENSORS_MODALITY",
    "SENSORS_SOURCE",
    "SENSORS_THUMBNAIL_JPEG_B64",
    "SENSORS_WIDTH",
    "SPAN_CLI_COMMAND",
    "SPAN_HAL_READ_STATE",
    "SPAN_HAL_SEND_ACTION",
    "SPAN_PHYSICS_STEP",
    "SPAN_REASONER_TICK",
    "SPAN_RSKILL_ACTIVATE",
    "SPAN_RSKILL_CHUNK_INFERENCE",
    "SPAN_RSKILL_CONFIGURE",
    "SPAN_RSKILL_EXECUTE",
    "SPAN_RSKILL_TICK",
    "SPAN_SAFETY_CHECK",
    "SPAN_SENSORS_READ_LATEST",
    "SPAN_SIM_RUN",
    "SPAN_SIM_STEP",
    "SPAN_WORLD_SCENE_OBJECTS",
    "SPAN_WORLD_STATE_SNAPSHOT",
    "SYSTEM_GPU_INDEX",
    "SYSTEM_GPU_NAME",
    "THRESHOLD_DIR_LOWER",
    "THRESHOLD_DIR_UPPER",
    "TICK_DEADLINE_MS",
    "TICK_IDX",
    "TICK_RATE_HZ",
    "WORLD_SCENE_OBJECTS_COUNT",
    "WORLD_SCENE_OBJECTS_FRAME",
    "WORLD_SCENE_OBJECTS_LIST",
    "WORLD_SCENE_OBJECTS_SOURCE_NODE",
    "WORLD_STATE_COMPONENT",
    "WORLD_STATE_COMPONENTS_STALE",
    "WORLD_STATE_HAS_LATCHED_ERROR",
    "WORLD_STATE_STALENESS_MS",
]
