"""How much of a kernel stop's over-approximation is the voxel grid, and how much is geometry?

`PLAN.md` §5 turns on one table: kernel-reported depth vs. certified mesh gap,
with the 25 mm grid's 21.65 mm half-diagonal subtracted. That decomposition
struck two levers and promoted a third; like ADR-0101's 94%, it had no
producer in the repo until now::

    excess = certified_gap - reported_depth
    beyond_voxel = excess - voxel_half_diagonal

``excess`` is how much further from the surface the robot really was than the
kernel believed; subtracting the half-diagonal leaves what the collision model
itself contributes. A class whose median ``beyond_voxel`` is <= 0 has no
geometry headroom left: every millimetre of over-approximation is the grid.

Two guards: only probe-**certified** distances are counted (an uncertified
distance can be wrong by 15-108 mm either way — `collision-validation-
evidence.md` caveat 8, larger than the quantity being measured); the voxel
half-diagonal is read from each round's own recorded ``grid_resolution_m``,
never assumed, so rounds at different resolutions cannot be silently pooled.

Run::

    uv run python tools/stop_excess.py outputs/validation-matrix/<round>...
    uv run python tools/stop_excess.py --json outputs/validation-matrix/*

Pure, offline, stdlib-only. Reads recorded artifacts; no GPU, no simulator.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, NamedTuple

PAYLOAD_PREFIX = "attached:"


class Excess(NamedTuple):
    """One stop's decomposition."""

    round_id: str
    scene: str
    party: str
    is_payload: bool
    reported_depth_m: float
    certified_gap_m: float
    voxel_half_diagonal_m: float
    verdict: str

    @property
    def excess_m(self) -> float:
        """How much further from the surface the robot really was."""
        return self.certified_gap_m - self.reported_depth_m

    @property
    def beyond_voxel_m(self) -> float:
        """The share of that excess the collision model, not the grid, accounts for."""
        return self.excess_m - self.voxel_half_diagonal_m


def half_diagonal(resolution_m: float) -> float:
    """Half the body diagonal of a cubic cell — the grid's worst-case error."""
    return resolution_m * math.sqrt(3.0) / 2.0


def collect(round_dirs: list[Path]) -> tuple[list[Excess], list[str]]:
    """Read every round's certified stops; return them plus the reasons for skips."""
    out: list[Excess] = []
    skipped: list[str] = []
    for round_dir in round_dirs:
        verdicts = round_dir / "verdicts.json"
        if not verdicts.is_file():
            continue
        payload = json.loads(verdicts.read_text())
        round_id = str(payload.get("metadata", {}).get("round_id", round_dir.name))
        for scene in payload.get("scenes", []):
            name = str(scene.get("scene", "?"))
            stop, truth = scene.get("stop"), scene.get("ground_truth")
            if not isinstance(stop, dict):
                continue
            if not isinstance(truth, dict):
                skipped.append(f"{round_id}/{name}: no ground-truth adjudication")
                continue
            if truth.get("probe_distance_certified") is not True:
                skipped.append(f"{round_id}/{name}: probe distances not certified")
                continue
            gap = truth.get("nearest_tripping_party_m")
            resolution = truth.get("grid_resolution_m")
            if not isinstance(gap, (int, float)):
                skipped.append(f"{round_id}/{name}: no nearest_tripping_party_m")
                continue
            if not isinstance(resolution, (int, float)):
                # Never fall back to 25 mm: a wrong half-diagonal moves the whole
                # result, and assuming one would hide exactly that.
                skipped.append(f"{round_id}/{name}: no recorded grid_resolution_m")
                continue
            party = str(stop.get("party_a", ""))
            out.append(
                Excess(
                    round_id=round_id,
                    scene=name,
                    party=party,
                    is_payload=party.startswith(PAYLOAD_PREFIX),
                    reported_depth_m=float(stop.get("min_distance_m", 0.0)),
                    certified_gap_m=float(gap),
                    voxel_half_diagonal_m=half_diagonal(float(resolution)),
                    verdict=str(truth.get("verdict", "")),
                )
            )
    return out, skipped


def summarise(stops: list[Excess], skipped: list[str]) -> dict[str, Any]:
    """Per-class medians, which is what the lever decisions rest on."""
    classes: dict[str, list[Excess]] = {"payload": [], "link": []}
    for stop in stops:
        classes["payload" if stop.is_payload else "link"].append(stop)
    per_class: dict[str, Any] = {}
    for name, group in classes.items():
        if not group:
            continue
        per_class[name] = {
            "n": len(group),
            "median_excess_mm": statistics.median(s.excess_m for s in group) * 1e3,
            "median_beyond_voxel_mm": statistics.median(s.beyond_voxel_m for s in group) * 1e3,
            "has_geometry_headroom": statistics.median(s.beyond_voxel_m for s in group) > 0.0,
        }
    return {
        "stops": len(stops),
        "by_class": per_class,
        "skipped": skipped,
    }


def render(stops: list[Excess], summary: dict[str, Any]) -> str:
    lines = ["Stop decomposition — reported depth vs certified truth, minus the grid", ""]
    if not stops:
        lines.append("  No certified stops in these rounds. Nothing to decompose.")
    else:
        lines.append(
            f"  {'round':<13}{'party':<14}{'reported':>10}{'true':>9}"
            f"{'excess':>9}{'beyond vox':>11}  verdict"
        )
        for s in stops:
            party = "payload" if s.is_payload else s.party
            lines.append(
                f"  {s.round_id[-11:]:<13}{party:<14}{s.reported_depth_m * 1e3:>9.2f}"
                f"{s.certified_gap_m * 1e3:>9.2f}{s.excess_m * 1e3:>9.2f}"
                f"{s.beyond_voxel_m * 1e3:>11.2f}  {s.verdict}"
            )
        lines.append("")
        for name, block in summary["by_class"].items():
            verdict = (
                "geometry headroom remains"
                if block["has_geometry_headroom"]
                else "NO geometry headroom — the residual is the grid"
            )
            lines.append(
                f"  {name:<8} n={block['n']:<3} median excess {block['median_excess_mm']:+7.2f} mm"
                f"   beyond voxel {block['median_beyond_voxel_mm']:+7.2f} mm   {verdict}"
            )
    if summary["skipped"]:
        lines += ["", f"  skipped (never counted): {len(summary['skipped'])}"]
        lines += [f"    {reason}" for reason in summary["skipped"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("rounds", nargs="+", type=Path, help="Round directories.")
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON.")
    args = parser.parse_args(argv)

    stops, skipped = collect([p for p in args.rounds if p.is_dir()])
    summary = summarise(stops, skipped)
    print(json.dumps(summary, indent=2) if args.json else render(stops, summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
