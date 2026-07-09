"""``openral benchmark scene`` manifest-writeback eligibility guard.

``benchmark scene`` writes ``rskill.yaml``'s ``benchmarks:`` block (the
suite-headline map, keyed by the ``BenchmarkName`` literal) only when the
scene's id IS a canonical suite id. A single-scene config whose ``scene.id``
is not a suite id (e.g. ``scenes/benchmark/metaworld_push.yaml`` →
``scene.id == "metaworld"``, suite is ``"metaworld_mt50"``) used to crash
``update_rskill_benchmarks`` with ``ROSConfigError`` ("merged manifest failed
validation"). The guard now skips the manifest write for those scenes; the
per-scene eval JSON still records the result.

CLAUDE.md §1.11 — real schema (``BenchmarkName``) and real shipped scene ids,
no placeholders.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_cli.main import _scene_id_is_benchmark_suite

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("scene_id", "expected"),
    [
        # Canonical suite ids — eligible for headline writeback.
        ("pusht", True),
        ("libero_spatial", True),
        ("metaworld_mt50", True),
        ("robocasa_pnp", True),
        ("simpler_env_widowx", True),
        # Single-scene ids that are NOT suite ids — must be skipped, not crash.
        ("metaworld", False),  # the reported bug (suite is metaworld_mt50)
        ("robocasa/PickPlaceCounterToCabinet", False),
        ("maniskill3", False),
        ("simpler_env", False),
        ("tabletop_push", False),
        ("so101_box", False),
    ],
)
def test_scene_id_is_benchmark_suite(scene_id: str, expected: bool) -> None:
    assert _scene_id_is_benchmark_suite(scene_id) is expected


def test_metaworld_push_scene_is_not_writeback_eligible() -> None:
    """The exact shipped scene that crashed: its id is not a suite id."""
    cfg = _REPO_ROOT / "scenes" / "benchmark" / "metaworld_push.yaml"
    if not cfg.exists():
        pytest.skip(f"shipped scene not present: {cfg}")
    scene_id = yaml.safe_load(cfg.read_text())["scene"]["id"]
    assert scene_id == "metaworld"
    assert _scene_id_is_benchmark_suite(scene_id) is False
