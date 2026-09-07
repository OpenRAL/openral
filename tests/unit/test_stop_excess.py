"""The decomposition that decides which collision levers are worth pulling.

`PLAN.md` §5 struck two levers and promoted a third on one quantity: a stop
class's median excess *beyond* the voxel half-diagonal. A class at or below zero
has no geometry headroom, so no tighter envelope can recover anything there.
Getting that arithmetic or its guards wrong redirects the whole programme.

CLAUDE.md §1.11 — the round is a real recorded artifact
(`tests/unit/fixtures/validation_matrix/2026-09-07-adr0101-live-1`, provenance
in that directory's `SOURCE.txt`).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "validation_matrix"
ROUND_LIVE = FIXTURES / "2026-09-07-adr0101-live-1"

_spec = importlib.util.spec_from_file_location(
    "stop_excess", REPO_ROOT / "tools" / "stop_excess.py"
)
assert _spec is not None and _spec.loader is not None
stop_excess = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stop_excess
_spec.loader.exec_module(stop_excess)


def test_the_decomposition_reproduces_the_recorded_stop() -> None:
    """Reported −4.05 mm against a certified +24.86 mm is 28.90 mm of excess."""
    stops, skipped = stop_excess.collect([ROUND_LIVE])

    assert skipped == []
    assert len(stops) == 1
    (stop,) = stops
    assert stop.is_payload
    assert round(stop.excess_m * 1e3, 2) == 28.90
    # 25 mm cells → 21.65 mm half-diagonal, leaving 7.25 mm to the collision model.
    assert round(stop.voxel_half_diagonal_m * 1e3, 2) == 21.65
    assert round(stop.beyond_voxel_m * 1e3, 2) == 7.25


def test_the_half_diagonal_comes_from_the_round_not_from_a_constant() -> None:
    """A round at a different resolution must decompose against *its* grid.

    The half-diagonal is the entire quantity being subtracted, so assuming
    25 mm for a round taken at another resolution would not perturb the answer
    — it would replace it.
    """
    assert stop_excess.half_diagonal(0.025) == 0.025 * math.sqrt(3.0) / 2.0
    assert stop_excess.half_diagonal(0.015) < stop_excess.half_diagonal(0.025)


def test_a_stop_with_no_recorded_resolution_is_skipped_not_assumed() -> None:
    """Missing `grid_resolution_m` must skip the stop, never default to 25 mm."""
    import copy
    import json
    import tempfile

    payload = json.loads((ROUND_LIVE / "verdicts.json").read_text())
    stripped = copy.deepcopy(payload)
    for scene in stripped["scenes"]:
        if isinstance(scene.get("ground_truth"), dict):
            scene["ground_truth"].pop("grid_resolution_m", None)

    with tempfile.TemporaryDirectory() as tmp:
        room = Path(tmp) / "round"
        room.mkdir()
        (room / "verdicts.json").write_text(json.dumps(stripped))
        stops, skipped = stop_excess.collect([room])

    assert stops == []
    assert any("no recorded grid_resolution_m" in reason for reason in skipped), skipped


def test_headroom_is_reported_per_class_and_zero_counts_as_none() -> None:
    """`has_geometry_headroom` is strict: a class exactly at the voxel term has none.

    The flag drives a spend/don't-spend decision. At exactly zero the class's
    entire over-approximation is already the grid, so calling that "headroom"
    would license work that cannot recover a millimetre.
    """
    summary = stop_excess.summarise(
        [
            stop_excess.Excess(
                round_id="r",
                scene="s",
                party="panda_link1",
                is_payload=False,
                reported_depth_m=-0.021650635,
                certified_gap_m=0.0,
                voxel_half_diagonal_m=0.021650635,
                verdict="within-quantization",
            )
        ],
        [],
    )
    assert summary["by_class"]["link"]["has_geometry_headroom"] is False
    assert abs(summary["by_class"]["link"]["median_beyond_voxel_mm"]) < 1e-9
