"""``openral_core/data/tracking_thresholds.yaml`` is the one home of the tracking thresholds.

Pins: the real file validates, the model round-trips, every tracker constant reads the file,
the C++ defaults that mirror a release value equal it, and no module re-hardcodes a number.
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from openral_core import TrackingThresholds, load_tracking_thresholds
from openral_core.exceptions import ROSConfigError
from openral_core.tracking_thresholds import DEFAULT_PATH
from openral_world_state.object_tracker import binding, cover, fit, session, stall, tracker
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TH = load_tracking_thresholds()
_NS = 1_000_000_000


def test_the_real_file_validates_and_ships_inside_the_openral_core_package() -> None:
    """Package data, not a repo-root file: modules load it at import time, so a non-editable
    wheel install (which ships only ``src/openral_core``) must still find it."""
    packaged = files("openral_core") / "data" / "tracking_thresholds.yaml"
    assert packaged.is_file()
    assert Path(str(packaged)) == DEFAULT_PATH
    assert DEFAULT_PATH.is_relative_to(Path(str(files("openral_core"))))
    assert not (_REPO_ROOT / "config" / "tracking_thresholds.yaml").exists(), "one copy only"
    assert _TH.schema_version == 1


def test_an_unknown_key_or_out_of_range_value_is_refused() -> None:
    raw = _TH.model_dump()
    with pytest.raises(ValidationError):
        TrackingThresholds.model_validate({**raw, "extra": 1})
    with pytest.raises(ValidationError):
        TrackingThresholds.model_validate({**raw, "release": {**raw["release"], "clear_m": 0.5}})


def test_the_loader_fails_closed_on_a_missing_key_bad_yaml_or_no_file(tmp_path: Path) -> None:
    """No defaults: a copy of the real file with one key dropped, a non-YAML file and a missing
    path each raise ``ROSConfigError`` naming the file, never load with a guessed value."""
    real = DEFAULT_PATH.read_text(encoding="utf-8")
    dropped = tmp_path / "dropped.yaml"
    dropped.write_text(re.sub(r"^  clear_m: .*$\n", "", real, flags=re.M), encoding="utf-8")
    assert dropped.read_text(encoding="utf-8") != real
    broken = tmp_path / "broken.yaml"
    broken.write_text("release: [unclosed", encoding="utf-8")
    for path in (dropped, broken, tmp_path / "absent.yaml"):
        with pytest.raises(ROSConfigError, match=str(path)):
            load_tracking_thresholds(path)


@settings(max_examples=25, deadline=None)
@given(st.floats(min_value=0.001, max_value=0.1), st.floats(min_value=0.01, max_value=10.0))
def test_the_model_round_trips(clear_m: float, timeout_s: float) -> None:
    raw = _TH.model_dump()
    raw["release"] = {**raw["release"], "clear_m": clear_m, "clear_timeout_s": timeout_s}
    model = TrackingThresholds.model_validate(raw)
    assert TrackingThresholds.model_validate_json(model.model_dump_json()) == model


def test_tracker_params_and_fit_params_defaults_are_the_file() -> None:
    for name, value in _TH.tracker:
        if hasattr(tracker.TrackerParams(), name):
            assert getattr(tracker.TrackerParams(), name) == value, name
    for name, value in _TH.fit:
        if hasattr(fit.FitParams(), name):
            assert getattr(fit.FitParams(), name) == value, name
    assert tracker.TrackerParams().fit == fit.FitParams()


def test_module_constants_equal_the_file() -> None:
    t, b, r, late = _TH.tracker, _TH.binding, _TH.release, _TH.late_bind
    pairs = [
        (tracker._MOVED_MIN_POINTS, t.moved_min_points),
        (tracker._MOVED_MIN_FRACTION, t.moved_min_fraction),
        (tracker._REDATE_INSIDE_FRACTION, t.redate_inside_fraction),
        (tracker._REDATE_SPILL_FRACTION, t.redate_spill_fraction),
        (tracker._GROWTH_VOTES, t.growth_votes),
        (tracker._GROWTH_AGREE_M, t.growth_agree_m),
        (tracker._GROWTH_WINDOW_NS, round(t.growth_window_s * _NS)),
        (tracker._CONTEXT_DEFER_NS, round(t.context_defer_s * _NS)),
        (tracker.PRIOR_MARGIN_M, t.prior_margin_m),
        (tracker._VIS_HYSTERESIS, t.vis_hysteresis),
        (tracker._DUPLICATE_MINT_IOU, t.duplicate_mint_iou),
        (tracker.VIEW_WINDOW_S, t.view_window_s),
        (tracker.LOCATE_ROI_PAD, t.locate_roi_pad),
        (tracker.ROBOT_BOX_FRACTION, t.robot_box_fraction),
        (tracker.LOW_PROMPT_SCORE, t.low_prompt_score),
        (tracker._REANCHOR_MAX_RATIO, t.reanchor_max_ratio),
        (cover.MIN_SPLIT_CELLS, _TH.cover.min_split_cells),
        (binding._SUPPORT_NEAR_M, b.support_near_m),
        (binding.TRACKED_CONFIDENCE, b.tracked_confidence),
        (binding.FALLBACK_CONFIDENCE, b.fallback_confidence),
        (session.RELEASE_DWELL_S, r.dwell_s),
        (session.RELEASED_CLEAR_TIMEOUT_S, r.clear_timeout_s),
        (session._RELEASE_CLEAR_M, r.clear_m),
        (session._JUDGE_CLEAR_M, r.judge_clear_m),
        (session._DROP_FITS, r.drop_fits),
        (session._JOINT_SAMPLE_MAX_AGE_NS, round(r.joint_sample_max_age_s * _NS)),
        (session._LATE_MIN_NS, round(late.min_s * _NS)),
        (session._LATE_MAX_NS, round(late.max_s * _NS)),
        (session._LATE_MAX_TRAVEL_M, late.max_travel_m),
        (session._LATE_LOOKBACK_NS, round(late.lookback_s * _NS)),
        (session._LATE_CONFIRM_NS, round(late.confirm_s * _NS)),
    ]
    for got, want in pairs:
        assert got == want
    assert cover.cover_need(1) == 2
    assert cover.cover_need(10) == 5


def test_the_stall_detector_defaults_are_the_file() -> None:
    d = stall.JawStallDetector(closed=0.0, gap=0.01, settle=0.0)
    st_ = _TH.stall
    assert (d.window_ns, d.dwell_ns, d.approach_ns) == (
        int(st_.window_s * _NS),
        int(st_.dwell_s * _NS),
        int(st_.approach_s * _NS),
    )
    assert d.min_samples == st_.min_samples


def _cpp_default(path: str, pattern: str) -> float:
    match = re.search(pattern, (_REPO_ROOT / path).read_text(encoding="utf-8"))
    assert match, f"{pattern} not found in {path}"
    return float(match.group(1))


def test_cpp_defaults_that_mirror_a_release_value_equal_the_file() -> None:
    kernel = "cpp/openral_safety_kernel/src/lifecycle_kernel.cpp"
    assert _cpp_default(kernel, r'"tracked_release_clear_m",\s*([0-9.]+)\)') == _TH.release.clear_m
    assert (
        _cpp_default(kernel, r'"tracked_release_timeout_s",\s*([0-9.]+)\)')
        == _TH.release.clear_timeout_s
    )
    bridge = "packages/openral_octomap_bridge/include/openral_octomap_bridge/identity_clearing.hpp"
    assert (
        _cpp_default(bridge, r"kReleasedClearTimeoutS = ([0-9.]+);") == _TH.release.clear_timeout_s
    )


@pytest.mark.parametrize(
    ("module", "names"),
    [
        (
            tracker,
            [
                "_MOVED_MIN_POINTS",
                "PRIOR_MARGIN_M",
                "VIEW_WINDOW_S",
                "_REANCHOR_MAX_RATIO",
                "_GROWTH_VOTES",
            ],
        ),
        (session, ["RELEASE_DWELL_S", "_RELEASE_CLEAR_M", "_LATE_MAX_TRAVEL_M", "_DROP_FITS"]),
        (binding, ["_SUPPORT_NEAR_M", "TRACKED_CONFIDENCE", "FALLBACK_CONFIDENCE"]),
        (cover, ["MIN_SPLIT_CELLS"]),
    ],
)
def test_no_module_constant_is_a_literal_again(module: object, names: list[str]) -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[attr-defined]  # reason: module object
    for name in names:
        line = re.search(rf"^{name}\s*=\s*(.+)$", source, re.M)
        assert line, name
        assert re.fullmatch(r"[0-9_.\s*/]+", line.group(1)) is None, f"{name} is hard-coded again"


def test_the_hal_release_window_uses_the_one_release_clearance_and_lifetime() -> None:
    """User decision 2026-11-01: one release clearance everywhere. The HAL vision leg's defaults
    are the config's (it used to be derived in sim as margin + one cell, 1.5 cm vs 4 cm)."""
    hal = pytest.importorskip("openral_hal.vision_attachment_bridge")
    t = load_tracking_thresholds().release
    cfg = hal.VisionAttachmentConfig.__dataclass_fields__
    assert cfg["release_clear_m"].default == t.clear_m
    assert cfg["release_timeout_s"].default == t.clear_timeout_s


def test_the_deploy_launch_hands_the_hal_the_one_release_clearance() -> None:
    """The launch passes the config value, not a margin + cell derivation."""
    src = (_REPO_ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py").read_text()
    assert '"vision_attachment_release_clear_m": load_tracking_thresholds().release.clear_m' in src


# --- phase 2: caps (C++ and schema mirrors), HAL leg, deploy launch ------------------------

_CPP = "cpp/openral_safety_kernel"
_COLLISION_HPP = f"{_CPP}/include/openral_safety_kernel/collision.hpp"
_LIFECYCLE_HPP = f"{_CPP}/include/openral_safety_kernel/lifecycle_kernel.hpp"


def _constexpr(path: str, name: str) -> float:
    """Value of ``<type> name = <number>;`` (``k``-prefixed constexpr or static member)."""
    return _cpp_default(path, rf"\b{name}\s*(?:\{{|=)\s*([0-9.]+)")


def test_the_kernel_constexpr_caps_equal_the_file() -> None:
    c = _TH.caps
    pairs = {
        "kMaxGraspRegionHalfExtentM": c.grasp_region_half_extent_m,
        "kMaxGraspRegionVolumeM3": c.grasp_region_volume_m3,
        "kMaxWorkspaceRegionHalfExtentM": c.workspace_half_extent_m,
        "kMaxWorkspaceRegionVolumeM3": c.workspace_volume_m3,
        "kMaxWorkspaceContactLeadM": c.workspace_contact_lead_m,
        "kMaxPlaceRegionHalfExtentM": c.place_region_half_extent_m,
        "kMaxPlaceRegionVolumeM3": c.place_region_volume_m3,
        "kMaxPlaceTargetPrimitives": c.place_target_primitives,
        "kMaxGraspDeclarationTimeoutS": c.grasp_timeout_s,
        "kMaxPlaceForceGateTimeoutS": c.place_timeout_s,
        "kMaxContactForceThresholdN": c.contact_force_n,
        "kPlaceApproachAllowanceVoxels": c.place_approach_allowance_voxels,
        "kMaxPlaceApproachAllowanceM": c.place_approach_allowance_max_m,
        "kPlaceAdvisoryDepthVoxels": c.place_advisory_depth_voxels,
        "kMaxPlaceAdvisoryDepthM": c.place_advisory_depth_max_m,
        "kMaxTrackedObjects": c.tracked_objects,
        "kMaxTrackedHullVertices": c.tracked_hull_vertices,
        "kExplainedStaticPadCells": c.explained_static_pad_cells,
    }
    for name, want in pairs.items():
        assert _constexpr(_COLLISION_HPP, name) == want, name
    assert _constexpr(_LIFECYCLE_HPP, "kMaxBoundPrimitives") == c.bound_primitives
    # kMaxRegionMeasurementAgeS = 2.0 * kMaxWorldVoxelDeadlineMs / 1000.0
    deadline_ms = _constexpr(_LIFECYCLE_HPP, "kMaxWorldVoxelDeadlineMs")
    assert 2.0 * deadline_ms / 1000.0 == c.region_measurement_age_s
    assert "kMaxRegionMeasurementAgeS = 2.0 * kMaxWorldVoxelDeadlineMs / 1000.0" in (
        _REPO_ROOT / _LIFECYCLE_HPP
    ).read_text(encoding="utf-8")
    # the attach residue is the inline `-0.5 * grid.resolution` in collision.cpp
    cpp = (_REPO_ROOT / f"{_CPP}/src/collision.cpp").read_text(encoding="utf-8")
    assert f"-{c.attach_residue_voxels} * grid.resolution" in cpp


def test_the_kernel_parameter_defaults_that_duplicate_a_cap_equal_the_file() -> None:
    kernel = f"{_CPP}/src/lifecycle_kernel.cpp"
    c = _TH.caps
    for name, want in {
        "support_witness_max_patch_radius_m": c.support_witness_max_patch_radius_m,
        "support_witness_max_penetration_m": c.support_witness_max_penetration_m,
        "attached_max_objects": c.attached_max_objects,
        "attached_max_primitives": c.attached_max_primitives,
        "attached_max_touch_links": c.attached_max_touch_links,
    }.items():
        assert _cpp_default(kernel, rf'"{name}",\s*([0-9.]+)\)') == want, name


def test_the_schema_bounds_equal_the_file() -> None:
    """The schema constants stay literals (openral_core.schemas imports without the config
    directory); this pins each equal to the caps entry it mirrors."""
    from openral_core.schemas import (
        GraspDeclaration,
        PlaceDeclaration,
        PlaceRegion,
        TrackedObjectSet,
    )

    c = _TH.caps
    assert c.grasp_region_half_extent_m == GraspDeclaration.MAX_HALF_EXTENT_M
    assert c.grasp_region_volume_m3 == GraspDeclaration.MAX_VOLUME_M3
    assert c.workspace_half_extent_m == GraspDeclaration.MAX_WORKSPACE_HALF_EXTENT_M
    assert c.workspace_volume_m3 == GraspDeclaration.MAX_WORKSPACE_VOLUME_M3
    assert c.grasp_timeout_s == GraspDeclaration.MAX_TIMEOUT_S
    assert c.place_region_half_extent_m == PlaceRegion.MAX_HALF_EXTENT_M
    assert c.place_region_volume_m3 == PlaceRegion.MAX_VOLUME_M3
    assert c.place_target_primitives == PlaceRegion.MAX_GEOMETRY_PRIMITIVES
    assert c.place_timeout_s == PlaceDeclaration.MAX_TIMEOUT_S
    assert c.contact_force_n == PlaceDeclaration.MAX_CONTACT_FORCE_THRESHOLD_N
    assert c.tracked_objects == TrackedObjectSet.MAX_OBJECTS
    assert c.bound_primitives == binding.MAX_PRIMITIVES
    assert _TH.tracker.static_pad_cells == c.explained_static_pad_cells


def test_the_octomap_bridge_allocation_guard_equals_the_file() -> None:
    # (the bridge's released_clear_timeout_s default is pinned above, via identity_clearing.hpp)
    path = "packages/openral_octomap_bridge/src/octree_to_grid.cpp"
    assert _cpp_default(path, r"kMaxCells = ([0-9]+);") == _TH.deploy.bridge_max_cells


def test_the_hal_vision_leg_defaults_are_the_file() -> None:
    hal = pytest.importorskip("openral_hal.vision_attachment_bridge")
    pytest.importorskip("openral_hal._vision_attachment_evidence")
    from openral_hal import _grasp_target_leg, _grasp_trigger, _vision_attachment_evidence
    from openral_hal._place_target_leg import _MAX_PATCH_RADIUS_M, _MAX_PENETRATION_M

    h, cfg = _TH.hal, hal.VisionAttachmentConfig()
    for name in (
        "grasp_target_rate_hz",
        "grasp_target_min_cells",
        "grasp_target_min_cover",
        "grasp_target_support_search_below_m",
        "grasp_target_support_probe_margin_m",
        "grasp_target_occluder_margin_m",
        "place_target_rate_hz",
        "place_target_search_depth_m",
        "grid_max_age_s",
        "evidence_timeout_s",
        "deadline_s",
    ):
        assert getattr(cfg, name) == getattr(h, name), name
    assert h.max_release_clear_m == hal.MAX_RELEASE_CLEAR_M
    assert (h.stall_rad, h.stall_ticks) == (hal._STALL_RAD, hal._STALL_TICKS)
    assert h.gap_periods == _grasp_trigger.GAP_PERIODS
    gate = _vision_attachment_evidence.VisionGateConfig()
    for name in (
        "containment_radius_m",
        "max_payload_volume_m3",
        "min_depth_valid_fraction",
        "min_depth_m",
        "max_depth_m",
        "view_ray_inflation_m",
        "jaw_span_m",
    ):
        assert getattr(gate, name) == getattr(h, name), name
    assert gate.max_payload_extents_m == h.max_payload_extents_m
    assert gate.max_primitives == h.gate_max_primitives == _TH.caps.bound_primitives
    assert h.max_occluder_margin_m == _grasp_target_leg._MAX_OCCLUDER_MARGIN_M
    assert h.max_support_search_below_m == _grasp_target_leg._MAX_SUPPORT_SEARCH_BELOW_M
    assert h.max_freeze_grid_ages == _grasp_target_leg._MAX_FREEZE_GRID_AGES
    assert h.premeasure_max_hz == _grasp_target_leg._PREMEASURE_MAX_HZ
    assert h.premeasure_reach_approaches == _grasp_target_leg._PREMEASURE_REACH_APPROACHES
    assert _TH.caps.support_witness_max_patch_radius_m == _MAX_PATCH_RADIUS_M
    assert _TH.caps.support_witness_max_penetration_m == _MAX_PENETRATION_M
    # the schema's runtime defaults mirror the same entries
    from openral_core import VisionAttachmentRuntime

    runtime = VisionAttachmentRuntime.model_fields
    assert runtime["deadline_s"].default == h.deadline_s
    assert runtime["evidence_timeout_s"].default == h.evidence_timeout_s


def test_remaining_tracker_constants_are_the_file() -> None:
    from openral_world_state.object_tracker import fit as fit_mod

    t = _TH.tracker
    assert t.refusal_log_period_s == tracker.REFUSAL_LOG_PERIOD_S
    assert t.robot_box_negatives == tracker.ROBOT_BOX_NEGATIVES
    assert t.support_seen_tol_m == tracker._SUPPORT_SEEN_TOL_M
    assert (t.beside_px, t.beside_min_px) == (tracker._BESIDE_PX, tracker._BESIDE_MIN_PX)
    assert _TH.fit.yaw_step_deg == fit_mod._YAW_STEP_DEG


def test_the_launch_per_mode_values_are_the_file() -> None:
    d = _TH.deploy
    src = (_REPO_ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py").read_text()
    assert "_deploy_mode(hal_mode)" in src
    assert (d.sim.octomap_resolution_m, d.real.octomap_resolution_m) == (0.015, 0.02)
    assert (d.sim.occupancy_thres, d.real.occupancy_thres) == (0.8, 0.6)
    assert (d.sim.clamping_max, d.real.clamping_max) == (0.85, 0.97)
    assert (d.sim.world_voxel_margin_m, d.real.world_voxel_margin_m) == (0.0, 0.02)
    assert (d.sim.self_filter_trailing_sweep_s, d.real.self_filter_trailing_sweep_s) == (0.2, 0.0)
    assert (
        d.sim.attached_collision_deadline_ms,
        d.real.attached_collision_deadline_ms,
    ) == (5000.0, 1000.0)
    assert (d.sim.safe_action_deadline_s, d.real.safe_action_deadline_s) == (8.0, 5.0)
    # the values both modes share: kernel joint-state deadline and the octomap/coverage pair
    assert d.collision_state_deadline_ms == 1000.0
    assert (d.sensor_model_miss, d.coverage_margin_m) == (0.4, 0.05)
    from openral_core.depth_extrinsic import REAL_WORLD_VOXEL_MARGIN_M

    assert d.real.world_voxel_margin_m == REAL_WORLD_VOXEL_MARGIN_M


def test_the_hal_safety_gates_are_their_reviewed_values() -> None:
    """The one-way bounds below sit at these values; changing one is a reviewed code change."""
    h = _TH.hal
    assert (h.grid_max_age_s, h.evidence_timeout_s, h.deadline_s) == (1.0, 0.5, 0.25)
    assert (h.max_freeze_grid_ages, h.max_release_clear_m, h.containment_radius_m) == (
        2.0,
        0.1,
        0.12,
    )
    assert (h.max_payload_extents_m, h.max_payload_volume_m3) == ((0.1, 0.25, 0.25), 0.004)
    assert (h.max_occluder_margin_m, h.max_support_search_below_m) == (0.1, 0.5)
    assert (h.min_depth_valid_fraction, h.grasp_target_min_cells) == (0.3, 8)
    assert h.grasp_target_min_cover == 0.5


# (section, key, a looser value, a tighter value): the file may tighten, never loosen.
_ONE_WAY = [
    ("real", "world_voxel_margin_m", 0.0, 0.03),
    ("real", "occupancy_thres", 0.7, 0.5),
    ("real", "clamping_max", 0.9, 0.98),
    ("real", "self_filter_trailing_sweep_s", 0.1, 0.0),
    ("real", "attached_collision_deadline_ms", 600_000.0, 500.0),
    ("real", "safe_action_deadline_s", 600.0, 2.0),
    ("sim", "attached_collision_deadline_ms", 6000.0, 1000.0),
    ("sim", "safe_action_deadline_s", 9.0, 5.0),
    ("deploy", "collision_state_deadline_ms", 600_000.0, 500.0),
    ("deploy", "sensor_model_miss", 0.2, 0.45),
    ("deploy", "coverage_margin_m", 0.0, 0.1),
    ("caps", "grasp_region_half_extent_m", 0.5, 0.1),
    ("caps", "contact_force_n", 500.0, 100.0),
    ("caps", "place_timeout_s", 6000.0, 300.0),
    ("caps", "place_approach_allowance_max_m", 0.1, 0.02),
    ("caps", "attach_residue_voxels", 0.1, 1.0),
    ("caps", "explained_static_pad_cells", 3, 0),
    ("caps", "region_measurement_age_s", 60.0, 2.0),
    ("caps", "support_witness_max_penetration_m", 0.05, 0.005),
    ("hal", "grid_max_age_s", 2.0, 0.5),
    ("hal", "evidence_timeout_s", 1.0, 0.3),
    ("hal", "deadline_s", 1.0, 0.2),
    ("hal", "max_freeze_grid_ages", 4.0, 1.0),
    ("hal", "max_release_clear_m", 0.2, 0.05),
    ("hal", "containment_radius_m", 0.3, 0.08),
    ("hal", "max_payload_extents_m", (0.2, 0.25, 0.25), (0.08, 0.2, 0.2)),
    ("hal", "max_payload_extents_m", (0.1, 0.5, 0.25), (0.1, 0.25, 0.2)),
    ("hal", "max_payload_extents_m", (0.1, 0.25, 0.5), (0.1, 0.2, 0.25)),
    ("hal", "max_payload_volume_m3", 0.01, 0.002),
    ("hal", "max_occluder_margin_m", 0.2, 0.05),
    ("hal", "max_support_search_below_m", 1.0, 0.3),
    ("hal", "min_depth_valid_fraction", 0.1, 0.5),
    ("hal", "grasp_target_min_cells", 4, 12),
    ("hal", "grasp_target_min_cover", 0.3, 0.7),
    # Review of #483: the conservative payload box and the octree cell can only grow.
    ("hal", "jaw_span_m", 0.01, 0.06),
    ("hal", "view_ray_inflation_m", 0.0, 0.03),
    ("sim", "octomap_resolution_m", 0.005, 0.02),
    ("real", "octomap_resolution_m", 0.005, 0.025),
]


@pytest.mark.parametrize(("section", "key", "looser", "tighter"), _ONE_WAY)
def test_a_safety_value_can_only_be_tightened_in_the_file(
    section: str, key: str, looser: object, tighter: object
) -> None:
    """Deploy, kernel-cap and HAL-gate bounds encode the reviewed value as the loosest allowed:
    an edit to the YAML that relaxes the kernel or the vision leg is refused at load, not run."""
    raw = _TH.model_dump()

    def with_value(value: object) -> dict[str, object]:
        out = {**raw, "deploy": {**raw["deploy"]}, "caps": {**raw["caps"]}, "hal": {**raw["hal"]}}
        target = out[section] if section in ("caps", "hal") else out["deploy"]
        if section in ("sim", "real"):
            target[section] = {**raw["deploy"][section], key: value}
        else:
            target[key] = value
        return out

    with pytest.raises(ValidationError):
        TrackingThresholds.model_validate(with_value(looser))
    TrackingThresholds.model_validate(with_value(tighter))


@settings(max_examples=25, deadline=None)
@given(st.floats(min_value=0.01, max_value=0.20), st.integers(min_value=1, max_value=16))
def test_the_caps_and_hal_sections_round_trip(extent: float, primitives: int) -> None:
    raw = _TH.model_dump()
    raw["caps"] = {
        **raw["caps"],
        "grasp_region_half_extent_m": extent,
        "bound_primitives": primitives,
    }
    model = TrackingThresholds.model_validate(raw)
    assert TrackingThresholds.model_validate_json(model.model_dump_json()) == model
    with pytest.raises(ValidationError):
        TrackingThresholds.model_validate({**raw, "caps": {**raw["caps"], "bound_primitives": 0}})
