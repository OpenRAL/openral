"""The object-tracking and pick-and-place thresholds, loaded from one file.

``openral_core/data/tracking_thresholds.yaml`` (package data, so it ships in the wheel) is the
single home of these numbers; modules read them through :func:`load_tracking_thresholds`
instead of hard-coding them.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openral_core._safe_yaml import safe_load
from openral_core.exceptions import ROSConfigError

__all__ = ["DeployModeThresholds", "TrackingThresholds", "load_tracking_thresholds"]

# Package data, not a repo-root file: several modules load it at import time, so it must
# resolve in a non-editable wheel install too.
DEFAULT_PATH = Path(str(files("openral_core") / "data" / "tracking_thresholds.yaml"))


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _FitThresholds(_Section):
    """Per-camera surface fit."""

    k_px: int = Field(ge=1)
    w_occ_px: int = Field(ge=0)
    eps_m: float = Field(ge=0.0)
    occluder_tol_m: float = Field(gt=0.0)
    min_points: int = Field(ge=4)
    min_depth_valid: float = Field(ge=0.0, le=1.0)
    min_depth_m: float = Field(ge=0.0)
    max_depth_m: float = Field(gt=0.0)
    extent: Literal["measured", "containing"]
    yaw_step_deg: float = Field(gt=0.0)


class _TrackerThresholds(_Section):
    """Tracker lifecycle, refit gates and prompts."""

    theta_vis: float = Field(gt=0.0, le=1.0)
    theta_vis_provisional: float = Field(gt=0.0, le=1.0)
    continuous_locate_s: float = Field(gt=0.0)
    locate_min_score: float = Field(ge=0.0, le=1.0)
    low_score_confirm_grids: int = Field(ge=1)
    static_pad_cells: int = Field(ge=0)
    provisional_oversize_factor: float = Field(ge=1.0)
    refit_growth_factor: float = Field(ge=1.0)
    max_age_s: float = Field(gt=0.0)
    lost_s: float = Field(gt=0.0)
    keep_lost_s: float = Field(gt=0.0)
    disagree_grids: int = Field(ge=1)
    segment_hz: float = Field(gt=0.0)
    segment_budget_wall_s: float = Field(gt=0.0)
    search_start_s: float = Field(ge=0.0)
    search_max_s: float = Field(gt=0.0)
    support_min_cells: int = Field(ge=1)
    support_probe_m: float = Field(gt=0.0)
    moved_min_points: int = Field(ge=1)
    moved_min_fraction: float = Field(ge=0.0, le=1.0)
    redate_inside_fraction: float = Field(gt=0.0, le=1.0)
    redate_spill_fraction: float = Field(gt=0.0, le=1.0)
    growth_votes: int = Field(ge=2)
    growth_agree_m: float = Field(gt=0.0)
    growth_window_s: float = Field(gt=0.0)
    context_defer_s: float = Field(ge=0.0)
    prior_margin_m: float = Field(ge=0.0)
    vis_hysteresis: float = Field(ge=0.0, le=1.0)
    duplicate_mint_iou: float = Field(gt=0.0, le=1.0)
    view_window_s: float = Field(gt=0.0)
    locate_roi_pad: float = Field(ge=0.0)
    robot_box_fraction: float = Field(gt=0.0, le=1.0)
    low_prompt_score: float = Field(ge=0.0, le=1.0)
    reanchor_max_ratio: float = Field(ge=1.0)
    refusal_log_period_s: float = Field(gt=0.0)
    robot_box_negatives: int = Field(ge=0)
    support_seen_tol_m: float = Field(gt=0.0)
    beside_px: int = Field(ge=1)
    beside_min_px: int = Field(ge=1)
    support_flatness_res_fraction: float = Field(gt=0.0)
    plausible_extent_factor: float = Field(ge=1.0)
    recent_s: float = Field(gt=0.0)
    locate_contrast_labels: tuple[str, ...]


class _VisionThresholds(_Section):
    """EXPERIMENT: vision + voxel fusion (instance mask propagator, ``apply_instance_cells``)."""

    min_cells: int = Field(ge=1)
    need_fraction: float = Field(gt=0.0, le=1.0)
    max_extent_factor: float = Field(ge=1.0)
    erode_px: int = Field(ge=0)
    memory_s: float = Field(gt=0.0)
    orphan_s: float = Field(ge=0.0)
    reseed_frames: int = Field(ge=1)
    empty_reseed_frames: int = Field(ge=1)
    drift_ratio: float = Field(gt=1.0)
    drift_frames: int = Field(ge=1)


class _CoverThresholds(_Section):
    """Own-cover rules."""

    min_split_cells: int = Field(ge=1)
    need_fraction: float = Field(gt=0.0, le=1.0)
    need_min_cells: int = Field(ge=1)


class _BindingThresholds(_Section):
    """Grasp-time binding (the map never grows a payload since 2026-11-02)."""

    support_near_m: float = Field(gt=0.0)
    tracked_confidence: float = Field(gt=0.0, le=1.0)
    fallback_confidence: float = Field(gt=0.0, le=1.0)


class _LateBindThresholds(_Section):
    """Late ATTACH window."""

    min_s: float = Field(ge=0.0)
    max_s: float = Field(gt=0.0)
    max_travel_m: float = Field(gt=0.0)
    lookback_s: float = Field(gt=0.0)
    confirm_s: float = Field(gt=0.0)
    joint_history_len: int = Field(ge=1)


class _ReleaseThresholds(_Section):
    """Release window; kernel, bridge and HAL values must follow these."""

    dwell_s: float = Field(gt=0.0)
    clear_timeout_s: float = Field(gt=0.0, le=10.0)
    clear_m: float = Field(gt=0.0, le=0.1)
    judge_clear_m: float = Field(gt=0.0)
    drop_fits: int = Field(ge=1)
    joint_sample_max_age_s: float = Field(gt=0.0)


class _StallThresholds(_Section):
    """Jaw stall detector defaults."""

    window_s: float = Field(gt=0.0)
    dwell_s: float = Field(gt=0.0)
    min_samples: int = Field(ge=1)
    approach_s: float = Field(gt=0.0)
    min_dt_window_fraction: float = Field(gt=0.0)


class _CapsThresholds(_Section):
    """Mirrored safety invariants: the kernel's C++ ``constexpr`` caps and the schema bounds.

    C++ cannot read this file, so the literals stay in
    ``cpp/openral_safety_kernel`` and a test pins each equal to its entry here. The schema
    constants (``GraspDeclaration.MAX_*``, ``PlaceRegion.MAX_*``, ``PlaceDeclaration.MAX_*``)
    stay literals too (``openral_core.schemas`` must import without the repo's config
    directory) and the same test pins them.

    Each bound is one-directional at the value the kernel was reviewed with: a ceiling
    (``le``) on every allowance, exemption size, timeout, age and capacity, a floor (``ge``)
    where a larger number is the stricter one. This file can tighten a cap, never loosen it;
    loosening needs a reviewed code change to the bound (and the C++ literal) itself.
    """

    grasp_region_half_extent_m: float = Field(gt=0.0, le=0.20)
    grasp_region_volume_m3: float = Field(gt=0.0, le=0.03)
    workspace_half_extent_m: float = Field(gt=0.0, le=0.50)
    workspace_volume_m3: float = Field(gt=0.0, le=0.25)
    workspace_contact_lead_m: float = Field(gt=0.0, le=0.10)
    place_region_half_extent_m: float = Field(gt=0.0, le=1.5)
    place_region_volume_m3: float = Field(gt=0.0, le=8.0)
    place_target_primitives: int = Field(ge=1, le=64)
    grasp_timeout_s: float = Field(gt=0.0, le=120.0)
    place_timeout_s: float = Field(gt=0.0, le=600.0)
    contact_force_n: float = Field(gt=0.0, le=140.0)
    place_approach_allowance_voxels: float = Field(gt=0.0, le=1.5)
    place_approach_allowance_max_m: float = Field(gt=0.0, le=0.04)
    place_advisory_depth_voxels: float = Field(gt=0.0, le=0.2)
    place_advisory_depth_max_m: float = Field(gt=0.0, le=0.005)
    attach_residue_voxels: float = Field(ge=0.5)  # deeper = fewer exempt cells: a floor
    tracked_objects: int = Field(ge=1, le=32)
    tracked_hull_vertices: int = Field(ge=1, le=64)
    bound_primitives: int = Field(ge=1, le=16)
    attached_max_objects: int = Field(ge=1, le=8)
    attached_max_primitives: int = Field(ge=1, le=16)
    attached_max_touch_links: int = Field(ge=1, le=32)
    region_measurement_age_s: float = Field(gt=0.0, le=4.0)
    explained_static_pad_cells: int = Field(ge=0, le=1)
    support_witness_max_patch_radius_m: float = Field(gt=0.0, le=0.5)
    support_witness_max_penetration_m: float = Field(gt=0.0, le=0.01)


class _HalThresholds(_Section):
    """HAL vision leg: ``VisionAttachmentConfig``, ``VisionGateConfig`` and the legs' bounds.

    The attachment leg's safety gates are one-directional at their reviewed value, as in
    ``caps``: a ceiling (``le``) on every age, timeout, deadline, clearance, payload size and
    hand/support reach (a larger one accepts more), a floor (``ge``) on the depth-valid
    fraction, the grasp-target cell/cover minima and the conservative payload box's jaw span
    and view-ray inflation (a smaller one accepts more). Rates, depth range, probe margins,
    stall, gap and the pre-measurement knobs are tuning.
    """

    grasp_target_rate_hz: float = Field(gt=0.0)
    grasp_target_min_cells: int = Field(ge=8)
    grasp_target_min_cover: float = Field(ge=0.5, le=1.0)
    grasp_target_support_search_below_m: float = Field(gt=0.0)
    grasp_target_support_probe_margin_m: float = Field(ge=0.0)
    grasp_target_occluder_margin_m: float = Field(ge=0.0)
    place_target_rate_hz: float = Field(gt=0.0)
    place_target_search_depth_m: float = Field(gt=0.0)
    grid_max_age_s: float = Field(gt=0.0, le=1.0)
    evidence_timeout_s: float = Field(gt=0.0, le=0.5)
    deadline_s: float = Field(gt=0.0, le=0.25)
    max_release_clear_m: float = Field(gt=0.0, le=0.1)
    stall_rad: float = Field(gt=0.0)
    stall_ticks: int = Field(ge=1)
    gap_periods: float = Field(gt=0.0)
    containment_radius_m: float = Field(gt=0.0, le=0.12)
    max_payload_extents_m: tuple[
        Annotated[float, Field(gt=0.0, le=0.10)],
        Annotated[float, Field(gt=0.0, le=0.25)],
        Annotated[float, Field(gt=0.0, le=0.25)],
    ]
    max_payload_volume_m3: float = Field(gt=0.0, le=0.004)
    min_depth_valid_fraction: float = Field(ge=0.3, le=1.0)
    min_depth_m: float = Field(ge=0.0)
    max_depth_m: float = Field(gt=0.0)
    # ge: these size the conservative payload box; a smaller one covers less of the object.
    view_ray_inflation_m: float = Field(ge=0.02)
    jaw_span_m: float = Field(ge=0.05)
    gate_max_primitives: int = Field(ge=1)
    premeasure_max_hz: float = Field(gt=0.0)
    premeasure_reach_approaches: float = Field(gt=0.0)
    max_occluder_margin_m: float = Field(gt=0.0, le=0.1)
    max_support_search_below_m: float = Field(gt=0.0, le=0.5)
    max_freeze_grid_ages: float = Field(gt=0.0, le=2.0)


class DeployModeThresholds(_Section):
    """The launch values that differ between a simulated and a real deploy."""

    octomap_resolution_m: float = Field(ge=0.001, le=0.5)
    occupancy_thres: float = Field(gt=0.0, lt=1.0)
    clamping_max: float = Field(gt=0.0, lt=1.0)
    world_voxel_margin_m: float = Field(ge=0.0)
    self_filter_trailing_sweep_s: float = Field(ge=0.0)
    attached_collision_deadline_ms: float = Field(gt=0.0)
    safe_action_deadline_s: float = Field(gt=0.0)


class _SimDeployModeThresholds(DeployModeThresholds):
    """``deploy.sim``: the safety-relevant values may only tighten (see the real twin)."""

    # ge: a finer octree cell is the less conservative one (the launch's quantisation term).
    octomap_resolution_m: float = Field(ge=0.015, le=0.5)
    occupancy_thres: float = Field(gt=0.0, le=0.8)
    clamping_max: float = Field(ge=0.85, lt=1.0)
    self_filter_trailing_sweep_s: float = Field(ge=0.0, le=0.2)
    attached_collision_deadline_ms: float = Field(gt=0.0, le=5000.0)
    safe_action_deadline_s: float = Field(gt=0.0, le=8.0)


class _RealDeployModeThresholds(DeployModeThresholds):
    """``deploy.real``: every value the kernel's hardware posture rests on may only tighten.

    Each bound is the value that used to be hard-coded in ``deploy_e2e.launch.py``: a floor
    on the world-voxel margin and occupancy clamping, a ceiling on the occupancy threshold,
    the self-filter sweep (off until Safety-WG review) and the deadlines. Loosening one is a
    reviewed code change to the bound, never an edit of the YAML (CLAUDE.md §1.1).
    """

    octomap_resolution_m: float = Field(ge=0.02, le=0.5)
    occupancy_thres: float = Field(gt=0.0, le=0.6)
    clamping_max: float = Field(ge=0.97, lt=1.0)
    world_voxel_margin_m: float = Field(ge=0.02)
    self_filter_trailing_sweep_s: float = Field(ge=0.0, le=0.0)
    attached_collision_deadline_ms: float = Field(gt=0.0, le=1000.0)
    safe_action_deadline_s: float = Field(gt=0.0, le=5.0)


class _DeployThresholds(_Section):
    """``deploy_e2e.launch.py``: per-``hal_mode`` values plus the ones both modes share."""

    sim: _SimDeployModeThresholds
    real: _RealDeployModeThresholds
    # ge: a weaker miss clears an occupied cell in fewer rays.
    sensor_model_miss: float = Field(ge=0.4, lt=1.0)
    collision_state_deadline_ms: float = Field(gt=0.0, le=1000.0)
    segmenter_camera_deadline_s: float = Field(gt=0.0)
    segmenter_max_frame_skew_s: float = Field(gt=0.0)
    # ge: a smaller coverage ball leaves world occupancy near the reach unmapped.
    coverage_margin_m: float = Field(ge=0.05)
    bridge_max_cells: int = Field(ge=1)


class TrackingThresholds(_Section):
    """Every object-tracking / pick-and-place threshold, one section per module.

    Example:
        >>> load_tracking_thresholds().release.clear_timeout_s
        3.0
    """

    schema_version: Literal[1]
    fit: _FitThresholds
    tracker: _TrackerThresholds
    vision: _VisionThresholds
    cover: _CoverThresholds
    binding: _BindingThresholds
    late_bind: _LateBindThresholds
    release: _ReleaseThresholds
    stall: _StallThresholds
    caps: _CapsThresholds
    hal: _HalThresholds
    deploy: _DeployThresholds


@cache
def load_tracking_thresholds(path: Path | None = None) -> TrackingThresholds:
    """Load and validate ``openral_core/data/tracking_thresholds.yaml`` (cached per path).

    Raises:
        ROSConfigError: the file is missing, is not YAML, or a key is missing, unknown or out
            of range. There are no defaults: the tracker never runs on a guessed threshold.
    """
    target = path or DEFAULT_PATH
    try:
        return TrackingThresholds.model_validate(safe_load(target.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ROSConfigError(f"tracking thresholds {target}: {exc}") from exc
