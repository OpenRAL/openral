"""ADR-0101's recovery rate must be derivable, and must fail closed.

The ADR cites "48 of 51 payload-vs-``voxel_`` stops (94 %) recovered" as the
measurement that justifies building a **fail-open** mechanism. A number in that
position needs a producer, and the producer needs a test — specifically a test
that the *denominator* cannot be quietly inflated, because a recovery rate is
only as honest as the set it divides by.

CLAUDE.md §1.11 — the round here is a real recorded battery artifact
(``tests/unit/fixtures/validation_matrix/2026-08-23-master-s1``, provenance in
that directory's ``SOURCE.txt``), copied verbatim, not a synthesized log.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "validation_matrix"
ROUND_0823 = FIXTURES / "2026-08-23-master-s1"
ROUND_LIVE = FIXTURES / "2026-09-07-adr0101-live-1"

# tools/ is not an installed package — load by path, as test_validation_matrix.py does.
_spec = importlib.util.spec_from_file_location(
    "adr0101_recovery", REPO_ROOT / "tools" / "adr0101_recovery.py"
)
assert _spec is not None and _spec.loader is not None
adr0101_recovery = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = adr0101_recovery
_spec.loader.exec_module(adr0101_recovery)

_vm_spec = importlib.util.spec_from_file_location(
    "validation_matrix", REPO_ROOT / "tools" / "validation_matrix.py"
)
assert _vm_spec is not None and _vm_spec.loader is not None
validation_matrix = importlib.util.module_from_spec(_vm_spec)
sys.modules[_vm_spec.name] = validation_matrix
_vm_spec.loader.exec_module(validation_matrix)


def _derived(round_dir: Path, tmp_path: Path) -> Path:
    """Copy a fixture round into ``tmp_path`` and derive its ``verdicts.json``."""
    work = tmp_path / round_dir.name
    shutil.copytree(round_dir, work)
    assert validation_matrix.cmd_verdicts(work) == 0
    return work


def test_an_uncertified_probe_is_excluded_and_never_counted_as_a_recovery(
    tmp_path: Path,
) -> None:
    """The 2026-08-23 payload stop is real, and must not reach the numerator.

    That round's ``baguette`` scene recorded exactly the stop shape this tool
    selects — ``kind=world``, ``party_a=attached:sim:obj_main``,
    ``party_b=voxel_100273`` — but it predates the probe's distance
    attestation (standing caveat 8 of the evidence ledger), so its
    ``nearest_tripping_party_m`` is not certified.

    Counting it either way would be wrong, and the two errors are not
    symmetric: silently dropping it shrinks the denominator and *inflates* the
    rate, which is the direction that would overstate the case for a fail-open
    mechanism. So it must appear in ``excluded``, by name and with a reason.
    """
    round_dir = _derived(ROUND_0823, tmp_path)
    stops, excluded = adr0101_recovery.collect([round_dir])

    assert stops == [], "an uncertified probe must never become an adjudicable stop"
    reasons = [e.reason for e in excluded if e.scene == "baguette"]
    assert reasons == ["probe distances not certified"], excluded

    summary = adr0101_recovery.summarise(stops, excluded)
    assert summary["adjudicable_stops"] == 0
    assert summary["recovery_rate"] is None, "no rate may be published over an empty set"


def test_an_empty_denominator_renders_as_a_refusal_not_as_a_rate(tmp_path: Path) -> None:
    """With nothing adjudicable, the report says so rather than printing 0 % or 100 %."""
    round_dir = _derived(ROUND_0823, tmp_path)
    stops, excluded = adr0101_recovery.collect([round_dir])
    text = adr0101_recovery.render(adr0101_recovery.summarise(stops, excluded))

    assert "not a result" in text
    assert "%" not in text.split("excluded")[0], text


def test_touching_is_contact_not_clearance() -> None:
    """A payload at exactly 0 m is stopped, not recovered.

    The boundary has a safety direction: a modeled fixture stops a payload that
    is touching it, exactly as the cube did. Putting 0.0 on the clearance side
    would count real contacts as recoveries — the one class ADR-0101 must never
    suppress.
    """
    at_contact = adr0101_recovery.Stop(
        round_id="r",
        scene="s",
        payload="p",
        cell="voxel_1",
        reported_depth_m=-0.02,
        certified_gap_m=0.0,
        nearest_body="counter_1_right",
    )
    penetrating = at_contact._replace(certified_gap_m=-0.00276)
    clear = at_contact._replace(certified_gap_m=0.0162)

    assert not at_contact.recovered
    assert not penetrating.recovered
    assert clear.recovered


def test_the_fixture_is_the_body_the_payload_was_near_not_the_nearest_pair_of_any_kind() -> None:
    """The by-fixture table must not name a robot link.

    This is a regression test for a real defect in the first cut of this tool.
    ``ground_truth.nearest_pair`` records the closest probed pair *of any kind*,
    and for a carried payload that is routinely two robot links: on this very
    round it reads ``robot0_link3`` vs ``robot0_link4`` at −36.3 mm, while the
    payload itself sat 24.9 mm clear of a counter. Reading the fixture off that
    field put ``robot0_link4`` into a table of kitchen fixtures — a robot link
    presented as a static world body, in the record that argues for modelling
    static world bodies.

    The attribution must instead come from the probe's payload-vs-world pair
    list, and it must agree with the gap the stop was adjudicated on.
    """
    stops, excluded = adr0101_recovery.collect([ROUND_LIVE])

    assert excluded == []
    assert len(stops) == 1, stops
    (stop,) = stops
    assert stop.nearest_body == "counter_1_right_group_main"
    assert not stop.nearest_body.startswith("robot0_"), (
        "a robot link must never be attributed as a static fixture"
    )
    # The gap the attribution was matched against is the one that decides
    # recovery, so the two cannot drift apart silently.
    assert stop.certified_gap_m == 0.024858349
    assert stop.recovered


def test_attribution_is_withheld_when_the_snapshot_does_not_match_the_stop() -> None:
    """A snapshot describing a different stop yields no attribution, not a wrong one.

    The recovery count never depends on the fixture name, so the safe failure is
    to say nothing rather than to name a body the probe did not measure at this
    stop.
    """
    assert (
        adr0101_recovery.fixture_at_stop(ROUND_LIVE / "utensil", certified_gap_m=0.999)
        == adr0101_recovery.UNATTRIBUTED
    )


def test_a_missing_deploy_log_degrades_to_no_attribution(tmp_path: Path) -> None:
    """Without the raw snapshot the stop still counts; only its fixture is unknown."""
    assert (
        adr0101_recovery.fixture_at_stop(tmp_path, certified_gap_m=0.0249)
        == adr0101_recovery.UNATTRIBUTED
    )
