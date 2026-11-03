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
        assert getattr(fit.FitParams(), name) == value, name
    assert tracker.TrackerParams().fit == fit.FitParams()


def test_module_constants_equal_the_file() -> None:
    t, b, r, late = _TH.tracker, _TH.binding, _TH.release, _TH.late_bind
    pairs = [
        (tracker._MOVED_MIN_POINTS, t.moved_min_points),
        (tracker._MOVED_MIN_FRACTION, t.moved_min_fraction),
        (tracker._REDATE_INSIDE_FRACTION, t.redate_inside_fraction),
        (tracker._REDATE_SPILL_FRACTION, t.redate_spill_fraction),
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
        (tracker, ["_MOVED_MIN_POINTS", "PRIOR_MARGIN_M", "VIEW_WINDOW_S", "_REANCHOR_MAX_RATIO"]),
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
