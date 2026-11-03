"""The object-tracking and pick-and-place thresholds, loaded from one file.

``openral_core/data/tracking_thresholds.yaml`` (package data, so it ships in the wheel) is the
single home of these numbers; modules read them through :func:`load_tracking_thresholds`
instead of hard-coding them.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openral_core._safe_yaml import safe_load
from openral_core.exceptions import ROSConfigError

__all__ = ["TrackingThresholds", "load_tracking_thresholds"]

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
    context_defer_s: float = Field(ge=0.0)
    prior_margin_m: float = Field(ge=0.0)
    vis_hysteresis: float = Field(ge=0.0, le=1.0)
    duplicate_mint_iou: float = Field(gt=0.0, le=1.0)
    view_window_s: float = Field(gt=0.0)
    locate_roi_pad: float = Field(ge=0.0)
    robot_box_fraction: float = Field(gt=0.0, le=1.0)
    low_prompt_score: float = Field(ge=0.0, le=1.0)
    reanchor_max_ratio: float = Field(ge=1.0)


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


class TrackingThresholds(_Section):
    """Every object-tracking / pick-and-place threshold, one section per module.

    Example:
        >>> load_tracking_thresholds().release.clear_timeout_s
        3.0
    """

    schema_version: Literal[1]
    fit: _FitThresholds
    tracker: _TrackerThresholds
    cover: _CoverThresholds
    binding: _BindingThresholds
    late_bind: _LateBindThresholds
    release: _ReleaseThresholds
    stall: _StallThresholds


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
