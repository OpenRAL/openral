"""Guard the "declared but never emitted" telemetry inventory.

``semconv`` is read as the inventory of what OpenRAL reports, but four
span-event names and one metric instrument have no producer anywhere in
the tree. A name that exists and never fires is worse than a missing one:
the dashboard's ``e-stops`` counter was wired to
``openral.event.estop_requested`` and read 0 no matter how many e-stops
fired, until the HAL lifecycle node was made to emit it.

This test pins the set both ways, by scanning the real source tree:

* Wiring one up without removing it from the expected set fails here, so
  the annotation in ``semconv`` / ``metrics`` and the table in
  ``docs/reference/telemetry.md`` get updated in the same PR.
* Adding a *new* never-emitted constant also fails here, so the inventory
  cannot quietly grow.

Source-scanning rather than behavioural, matching
``tests/unit/test_adapters_no_snapshot_download.py``: proving a negative
("nothing anywhere emits this") is not something a runtime fixture can do.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from openral_observability import metrics, semconv

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_DIRS = ("python", "packages", "cpp", "tools")

# Files that define or merely *consume* a name rather than emit it. The
# dashboard store maps incoming event names to counters, so it mentions
# events it never produces.
_NON_EMITTER_FILES = {
    "semconv.py",
    "metrics.py",
    "store.py",
}

_EXPECTED_UNEMITTED_EVENTS = {
    "EVENT_ACTION_DROPPED",
    "EVENT_CHUNK_PREFETCH_HIT",
    "EVENT_CHUNK_PREFETCH_MISS",
    "EVENT_EPISODE_CLOSED",
}

_EXPECTED_UNRECORDED_METRICS: set[str] = set()


def _source_files() -> list[Path]:
    """Every tracked Python/C++ source file outside tests."""
    found: list[Path] = []
    for top in _SOURCE_DIRS:
        root = _REPO_ROOT / top
        if not root.is_dir():
            continue
        for pattern in ("*.py", "*.cpp"):
            found.extend(
                path
                for path in root.rglob(pattern)
                if "test" not in path.parts
                and not path.name.startswith("test_")
                and ".venv" not in path.parts
                and "site-packages" not in path.parts
            )
    return found


@pytest.fixture(scope="module")
def sources() -> list[tuple[Path, str]]:
    """Read every source file once; the scan is O(files), not O(files x names)."""
    out: list[tuple[Path, str]] = []
    for path in _source_files():
        try:
            out.append((path, path.read_text(encoding="utf-8", errors="ignore")))
        except OSError:  # pragma: no cover - unreadable file is not this test's problem
            continue
    return out


def _names_with_producers(
    sources: list[tuple[Path, str]], candidates: set[str], *, literal_by_name: dict[str, str]
) -> set[str]:
    """Return the subset of ``candidates`` some non-defining file references."""
    produced: set[str] = set()
    for path, text in sources:
        if path.name in _NON_EMITTER_FILES:
            continue
        for name in candidates:
            if name in produced:
                continue
            literal = literal_by_name.get(name)
            if re.search(rf"\b{re.escape(name)}\b", text) or (
                literal is not None and literal in text
            ):
                produced.add(name)
    return produced


def test_unemitted_span_events_are_exactly_the_documented_set(
    sources: list[tuple[Path, str]],
) -> None:
    """No producer for these four; a producer for anything else."""
    all_events = {n for n in dir(semconv) if n.startswith("EVENT_")}
    literals = {n: getattr(semconv, n) for n in all_events}

    produced = _names_with_producers(sources, all_events, literal_by_name=literals)
    unemitted = all_events - produced

    assert unemitted == _EXPECTED_UNEMITTED_EVENTS, (
        "The declared-but-never-emitted span-event set changed.\n"
        f"  newly unemitted: {sorted(unemitted - _EXPECTED_UNEMITTED_EVENTS)}\n"
        f"  now emitted:     {sorted(_EXPECTED_UNEMITTED_EVENTS - unemitted)}\n"
        "Update the annotation in semconv.py, the table in "
        "docs/reference/telemetry.md, and this expected set together."
    )


def test_unrecorded_metrics_are_exactly_the_documented_set(
    sources: list[tuple[Path, str]],
) -> None:
    """This instrument exists but nothing ever calls ``.add()`` on it."""
    all_getters = {n for n in metrics.__all__ if n.startswith("get_") and n != "get_meter"}

    produced = _names_with_producers(sources, all_getters, literal_by_name={})
    unrecorded = all_getters - produced

    assert unrecorded == _EXPECTED_UNRECORDED_METRICS, (
        "The declared-but-never-recorded metric set changed.\n"
        f"  newly unrecorded: {sorted(unrecorded - _EXPECTED_UNRECORDED_METRICS)}\n"
        f"  now recorded:     {sorted(_EXPECTED_UNRECORDED_METRICS - unrecorded)}\n"
        "Update the annotation in metrics.py, the table in "
        "docs/reference/telemetry.md, and this expected set together."
    )
