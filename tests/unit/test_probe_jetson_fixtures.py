"""_probe_jetson fallback path — exercised against recorded real-device strings.

ADR-0013 PR 2/3 — replaces the legacy ``(8, 7) if "Orin" in board else (7, 2)``
heuristic with an explicit per-SoC table and pins the mapping with
fixtures captured from real boards (no mocks per CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_detect.probes.gpu import _probe_jetson

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "jetson"


@pytest.mark.parametrize(
    "board_dir,expected_board_keyword,expected_cc",
    [
        ("orin_agx", "Orin", (8, 7)),
        ("orin_nx", "Orin NX", (8, 7)),
        ("orin_nano", "Orin Nano", (8, 7)),
        ("xavier_agx", "Xavier", (7, 2)),
        ("xavier_nx", "Xavier NX", (7, 2)),
        ("maxwell_nano", "Nano", (5, 3)),
    ],
)
def test_probe_jetson_classifies_board(
    board_dir: str,
    expected_board_keyword: str,
    expected_cc: tuple[int, int],
) -> None:
    warnings: list[str] = []
    info = _probe_jetson(
        warnings,
        model_path=FIXTURE_ROOT / board_dir / "model",
        release_path=FIXTURE_ROOT / board_dir / "nv_tegra_release",
    )
    assert info is not None
    assert expected_board_keyword in info.board
    assert info.cuda_compute_capability == expected_cc
    assert warnings == []


def test_probe_jetson_unknown_board_returns_none_with_warning(tmp_path: Path) -> None:
    model = tmp_path / "model"
    release = tmp_path / "nv_tegra_release"
    model.write_text("Some Weird Custom Carrier Board\x00")
    release.write_text("# R36 (release), REVISION: 4.0\n")
    warnings: list[str] = []
    info = _probe_jetson(warnings, model_path=model, release_path=release)
    assert info is None
    assert any("unknown" in w.lower() for w in warnings), warnings


def test_probe_jetson_returns_none_when_both_paths_missing(tmp_path: Path) -> None:
    warnings: list[str] = []
    info = _probe_jetson(
        warnings,
        model_path=tmp_path / "absent_model",
        release_path=tmp_path / "absent_release",
    )
    assert info is None
    assert warnings == []
