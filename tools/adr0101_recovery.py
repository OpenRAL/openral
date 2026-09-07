"""How many payload stops would a modeled static fixture have recovered?

[ADR-0101](../docs/decisions.md) proposes adjudicating a carried payload
against the exact primitives of the nearest static world bodies instead of
the occupancy cells that represent the same surfaces. Its headline number —
**48 of 51 payload-vs-``voxel_`` stops (94%) recovered** — had no producer in
the repo; this is that producer (see `docs/reference/collision-validation-
evidence.md`). A stop is "recovered" if the probe's certified payload-to-real-
body distance at the moment of the stop is > 0.

Not claimed: a stop removed mid-carry is a run that continues, not a
completion — the ceiling run bounds completion separately at 29 points. The
minimum recovered clearance matters as much as the median (ADR-0101's
suppression-off first landing rests on it).

A stop counts only when all of:

* it is a **world** stop (``kind == "world"``) — self stops are Entry 026's;
* the tripping party is the **carried payload** (``party_a`` = ``attached:<id>``);
* the other party is an **anonymous cell** (``party_b`` = ``voxel_<n>``);
* the probe **certified** its distances (``probe_distance_certified``) and
  produced ``nearest_tripping_party_m``. Uncertified/truncated -> ``excluded``,
  never a recovery.

Run::

    uv run python tools/adr0101_recovery.py outputs/validation-matrix/<round>...
    uv run python tools/adr0101_recovery.py --json outputs/validation-matrix/*

Pure, offline, stdlib-only (like ``tools/round_power.py``) — reads recorded
artifacts, no GPU or simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final, NamedTuple

#: A stop is "recovered" when the payload was certifiably clear of the real
#: surface. Zero is the boundary and belongs to *contact*, not to clearance:
#: at exactly 0 m the payload is touching, which the modeled body stops just as
#: the cube did. Erring the other way would inflate the recovery rate with the
#: very cases the mechanism must not let through.
CLEAR_THRESHOLD_M: Final[float] = 0.0

PAYLOAD_PREFIX: Final[str] = "attached:"
CELL_PREFIX: Final[str] = "voxel_"


class Stop(NamedTuple):
    """One payload-vs-cell stop, with the certified truth behind it."""

    round_id: str
    scene: str
    payload: str
    cell: str
    reported_depth_m: float
    certified_gap_m: float
    nearest_body: str

    @property
    def recovered(self) -> bool:
        """Would a modeled fixture have let this stop through?"""
        return self.certified_gap_m > CLEAR_THRESHOLD_M


class Excluded(NamedTuple):
    """A payload-vs-cell stop that could not be adjudicated, and why."""

    round_id: str
    scene: str
    reason: str


#: Tolerance for matching a recorded snapshot to the stop being adjudicated.
#: Both numbers come from the same probe call, so they agree exactly or the
#: snapshot describes a different stop.
_GAP_MATCH_TOL_M: Final[float] = 1e-9

UNATTRIBUTED: Final[str] = "<unattributed>"


def fixture_at_stop(scene_dir: Path, certified_gap_m: float) -> str:
    """Name the world body the payload was nearest, for the by-fixture table.

    Not ``ground_truth.nearest_pair`` — that is the closest probed pair of ANY
    kind (often two robot links: one round read ``robot0_link3``/``link4`` at
    -36 mm while the payload sat 24.9 mm clear of a counter). The body must
    come from the probe's payload-vs-world pair list in the raw
    ``sim.estop_ground_truth_snapshot`` line instead.

    Verified, not assumed: the list's minimum certified distance must equal
    this stop's ``nearest_tripping_party_m`` (same probe call), else the
    attribution is ``UNATTRIBUTED``. Only the by-fixture breakdown depends on
    this — the recovery count does not.
    """
    log_path = scene_dir / "run_deploy.log"
    if not log_path.is_file():
        return UNATTRIBUTED
    snapshot: dict[str, Any] | None = None
    with log_path.open(errors="replace") as handle:
        for line in handle:
            if "sim.estop_ground_truth_snapshot" not in line:
                continue
            start, end = line.find("{"), line.rfind("}")
            if start < 0 or end <= start:
                continue
            try:
                snapshot = json.loads(line[start : end + 1])
            except json.JSONDecodeError:
                continue
            break
    if snapshot is None:
        return UNATTRIBUTED
    pairs = snapshot.get("nearest_payload_world_pairs")
    if not isinstance(pairs, list):
        return UNATTRIBUTED
    certified = [
        pair
        for pair in pairs
        if isinstance(pair, dict)
        and pair.get("distance_certified") is True
        and isinstance(pair.get("distance_m"), (int, float))
    ]
    if not certified:
        return UNATTRIBUTED
    nearest = min(certified, key=lambda pair: float(pair["distance_m"]))
    if abs(float(nearest["distance_m"]) - certified_gap_m) > _GAP_MATCH_TOL_M:
        return UNATTRIBUTED
    body = nearest.get("body_b")
    return body if isinstance(body, str) and body else UNATTRIBUTED


def collect(round_dirs: list[Path]) -> tuple[list[Stop], list[Excluded]]:
    """Read every round's ``verdicts.json`` into adjudicable stops + exclusions."""
    stops: list[Stop] = []
    excluded: list[Excluded] = []
    for round_dir in round_dirs:
        verdicts_path = round_dir / "verdicts.json"
        if not verdicts_path.is_file():
            excluded.append(Excluded(round_dir.name, "-", "no verdicts.json"))
            continue
        payload = json.loads(verdicts_path.read_text())
        round_id = str(payload.get("metadata", {}).get("round_id", round_dir.name))
        for scene in payload.get("scenes", []):
            name = str(scene.get("scene", "?"))
            stop = scene.get("stop")
            if not isinstance(stop, dict):
                continue  # No stop at all: not an exclusion, just not a stop.
            party_a = str(stop.get("party_a", ""))
            party_b = str(stop.get("party_b", ""))
            if str(stop.get("kind", "")) != "world":
                continue
            if not party_a.startswith(PAYLOAD_PREFIX) or not party_b.startswith(CELL_PREFIX):
                continue
            truth = scene.get("ground_truth")
            if not isinstance(truth, dict):
                excluded.append(Excluded(round_id, name, "no ground-truth adjudication"))
                continue
            if truth.get("probe_distance_certified") is not True:
                excluded.append(Excluded(round_id, name, "probe distances not certified"))
                continue
            gap = truth.get("nearest_tripping_party_m")
            if not isinstance(gap, (int, float)):
                excluded.append(Excluded(round_id, name, "no nearest_tripping_party_m"))
                continue
            stops.append(
                Stop(
                    round_id=round_id,
                    scene=name,
                    payload=party_a[len(PAYLOAD_PREFIX) :],
                    cell=party_b,
                    reported_depth_m=float(stop.get("min_distance_m", 0.0)),
                    certified_gap_m=float(gap),
                    nearest_body=fixture_at_stop(round_dir / name, float(gap)),
                )
            )
    return stops, excluded


def _median(values: list[float]) -> float:
    """Median of a non-empty list; the mid-average on an even count."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def summarise(stops: list[Stop], excluded: list[Excluded]) -> dict[str, Any]:
    """The ADR's table, as data."""
    recovered = [s for s in stops if s.recovered]
    still = [s for s in stops if not s.recovered]
    by_fixture: dict[str, int] = {}
    for stop in recovered:
        by_fixture[stop.nearest_body] = by_fixture.get(stop.nearest_body, 0) + 1
    gaps = [s.certified_gap_m for s in recovered]
    return {
        "adjudicable_stops": len(stops),
        "recovered": len(recovered),
        "still_stop": len(still),
        "recovery_rate": (len(recovered) / len(stops)) if stops else None,
        "median_recovered_clearance_mm": (_median(gaps) * 1e3) if gaps else None,
        "min_recovered_clearance_mm": (min(gaps) * 1e3) if gaps else None,
        "still_stop_depths_mm": sorted(s.certified_gap_m * 1e3 for s in still),
        "still_stop_bodies": sorted({s.nearest_body for s in still}),
        "recovered_by_fixture": dict(sorted(by_fixture.items(), key=lambda kv: (-kv[1], kv[0]))),
        "excluded": [e._asdict() for e in excluded],
    }


def render(summary: dict[str, Any]) -> str:
    """Human-readable report, shaped like the ADR's Consequences table."""
    n = summary["adjudicable_stops"]
    lines = ["ADR-0101 — modeled-fixture recovery, from certified ground truth", ""]
    if not n:
        lines.append("  No adjudicable payload-vs-voxel_ stops in these rounds.")
        lines.append("  A recovery rate over an empty denominator is not a result.")
    else:
        rate = summary["recovery_rate"] or 0.0
        lines += [
            f"  adjudicable payload-vs-voxel_ stops : {n}",
            f"  would be recovered                  : {summary['recovered']} ({rate:.0%})",
            f"  would still stop (real contact)     : {summary['still_stop']}",
            "",
            f"  median recovered clearance : {summary['median_recovered_clearance_mm']:.2f} mm"
            if summary["median_recovered_clearance_mm"] is not None
            else "  median recovered clearance : n/a",
        ]
        if summary["min_recovered_clearance_mm"] is not None:
            lines.append(
                f"  MINIMUM recovered clearance: {summary['min_recovered_clearance_mm']:.2f} mm"
                "   <- the modelling-error budget"
            )
        if summary["still_stop_depths_mm"]:
            depths = ", ".join(f"{d:.2f}" for d in summary["still_stop_depths_mm"])
            bodies = ", ".join(summary["still_stop_bodies"])
            lines += ["", f"  correctly still stopping: {depths} mm against {bodies}"]
        if summary["recovered_by_fixture"]:
            lines += ["", "  recovered by fixture:"]
            lines += [
                f"    {body:<40} {count}" for body, count in summary["recovered_by_fixture"].items()
            ]
    if summary["excluded"]:
        lines += ["", f"  excluded (never counted as recoveries): {len(summary['excluded'])}"]
        lines += [f"    {e['round_id']}/{e['scene']}: {e['reason']}" for e in summary["excluded"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("rounds", nargs="+", type=Path, help="Round directories.")
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON.")
    args = parser.parse_args(argv)

    stops, excluded = collect([p for p in args.rounds if p.is_dir()])
    summary = summarise(stops, excluded)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
