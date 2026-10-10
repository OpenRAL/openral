"""Tests for the repo-state-map drift checker (``tools/check_repo_state_map.py``).

Two jobs. ``test_repo_state_map_has_no_drift`` runs the checker against the
*real* map and the *real* tree (CLAUDE.md §1.11) and is the regression guard the
whole tool exists for. The rest prove it is not a rubber stamp: a checker that
never fires would pass that first test forever, so each defect class the map has
actually shipped — a dead ``pkg:`` pointer, a rotted ``desc`` count, and an
unescaped quote that blanked the whole page — is
re-seeded here and must be caught.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "check_repo_state_map", REPO_ROOT / "tools" / "check_repo_state_map.py"
)
assert _spec is not None and _spec.loader is not None
check_repo_state_map = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check_repo_state_map
_spec.loader.exec_module(check_repo_state_map)


def _card(pkg: str, desc: str = "") -> str:
    """One card's worth of the map's JS literal — the shape the regexes parse."""
    return (
        f'      {{\n        title: "T",\n        pkg: "{pkg}",\n        desc: "{desc}",\n      }},'
    )


class TestPaths:
    def test_real_directory_resolves(self) -> None:
        cards = check_repo_state_map.iter_cards(_card("packages/openral_hal_node"))
        assert check_repo_state_map.check_paths(cards) == []

    def test_module_relative_to_its_package_head_resolves(self) -> None:
        # "policies/pi05.py" lives at python/sim/src/openral_sim/policies/pi05.py;
        # read from the repo root it does not exist, and that is not drift.
        cards = check_repo_state_map.iter_cards(_card("python/sim · policies/pi05.py"))
        assert check_repo_state_map.check_paths(cards) == []

    def test_bare_module_resolves(self) -> None:
        cards = check_repo_state_map.iter_cards(_card("python/hal · openarm.py"))
        assert check_repo_state_map.check_paths(cards) == []

    def test_planned_directory_is_not_drift(self) -> None:
        cards = check_repo_state_map.iter_cards(_card("cpp/rt_bridge (planned)"))
        assert check_repo_state_map.check_paths(cards) == []

    def test_dead_directory_is_caught(self) -> None:
        # The real defect: an "Examples" card sat green against examples/, a
        # directory with no git history in this repo.
        problems = check_repo_state_map.check_paths(
            check_repo_state_map.iter_cards(_card("examples/"))
        )
        assert len(problems) == 1
        assert "examples/" in problems[0]

    def test_dead_module_is_caught(self) -> None:
        cards = check_repo_state_map.iter_cards(_card("examples/so100_robosuite_lift.py"))
        assert len(check_repo_state_map.check_paths(cards)) == 1

    def test_missing_src_segment_is_caught(self) -> None:
        # The replay card omitted `src/`, so its path did not resolve.
        cards = check_repo_state_map.iter_cards(
            _card("python/observability/openral_observability/replay")
        )
        assert len(check_repo_state_map.check_paths(cards)) == 1


class TestCounts:
    def test_truthful_count_passes(self) -> None:
        actual = len(list((REPO_ROOT / "tests" / "hil").glob("test_*.py")))
        cards = check_repo_state_map.iter_cards(_card("tests/hil", f"{actual} files. Whatever."))
        assert check_repo_state_map.check_counts(cards) == []

    def test_rotted_count_is_caught(self) -> None:
        # The shape that shipped: tests/unit claimed 190 against a real 379.
        actual = len(list((REPO_ROOT / "tests" / "hil").glob("test_*.py")))
        cards = check_repo_state_map.iter_cards(_card("tests/hil", f"{actual * 2} files. Stale."))
        problems = check_repo_state_map.check_counts(cards)
        assert len(problems) == 1
        assert f"found {actual}" in problems[0]

    def test_count_within_tolerance_passes(self) -> None:
        # Adding one test file must not turn the hook red — that is how a hook
        # gets disabled before it catches anything that matters.
        actual = len(list((REPO_ROOT / "tests" / "unit").glob("test_*.py")))
        cards = check_repo_state_map.iter_cards(_card("tests/unit", f"{actual - 1} files. Close."))
        assert check_repo_state_map.check_counts(cards) == []

    def test_manifest_count_is_checked(self) -> None:
        actual = len(list((REPO_ROOT / "rskills").glob("*/rskill.yaml")))
        cards = check_repo_state_map.iter_cards(_card("rskills/", f"{actual * 3} manifests: x."))
        assert len(check_repo_state_map.check_counts(cards)) == 1

    def test_desc_without_a_count_is_ignored(self) -> None:
        cards = check_repo_state_map.iter_cards(_card("tests/hil", "Prose with no leading count."))
        assert check_repo_state_map.check_counts(cards) == []


class TestLiterals:
    def test_escaped_quote_is_well_formed(self) -> None:
        assert check_repo_state_map.check_literals(_card("tests/hil", 'a \\"b\\" c')) == []

    def test_unescaped_quote_is_caught(self) -> None:
        # The #289 regression: `image_topic (default "" — ...)` inside a desc.
        card = _card("tests/hil", 'image_topic (default "")')
        problems = check_repo_state_map.check_literals(card)
        assert len(problems) == 1 and "malformed string literal" in problems[0]

    def test_missing_opening_quote_is_caught(self) -> None:
        # Matching on the opening quote skipped this line entirely.
        card = _card("tests/hil").replace('desc: "",', 'desc: image_topic (default ""),')
        problems = check_repo_state_map.check_literals(card)
        assert len(problems) == 1 and "malformed string literal" in problems[0]

    def test_unescaped_quote_in_a_card_array_is_caught(self) -> None:
        card = _card("tests/hil") + '\n        outputs: ["Image on /x (default "")", "y"],'
        problems = check_repo_state_map.check_literals(card)
        assert len(problems) == 1 and "malformed string literal" in problems[0]

    def test_unescaped_quote_in_a_schemas_row_is_caught(self) -> None:
        good = '  { name: "ClockOrigin", where: "core", kind: "enum", note: "a \\"b\\"" },'
        bad = '  { name: "ClockOrigin", where: "core", kind: "enum", note: "a "b"" },'
        assert check_repo_state_map.check_literals(good) == []
        assert len(check_repo_state_map.check_literals(bad)) == 1

    def test_css_content_lines_are_not_cards(self) -> None:
        assert check_repo_state_map.check_literals('      content: ""; width: 10px;') == []


def test_repo_state_map_has_no_drift() -> None:
    """The real map, against the real tree. Goes red when either side moves."""
    assert check_repo_state_map.main(["--quiet"]) == 0
