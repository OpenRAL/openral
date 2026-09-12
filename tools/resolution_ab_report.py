#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score the 25 mm vs 15 mm world-voxel A/B (#253).

Pure and offline: reads only the artifacts `tools/resolution_ab.sh` wrote, so a
battery can be re-scored later, on another checkout, without re-running it.

**The primary endpoint is the over-approximation, not the completion rate.**
For each stop the kernel reports a depth; the same run's
`sim.estop_ground_truth_snapshot` carries the certified GJK distance to real
geometry. Their difference is how much the map over-stated the obstacle, and it
is what a finer cell is supposed to shrink: the half-diagonal falls from
21.65 mm to 12.99 mm, so the prediction is a **-8.66 mm shift**.

Completion and stop counts are reported too, and labelled under-powered,
because they are: against the predicted 2.7% -> 10.8% completion effect, 80%
power needs 200 runs per arm (`tools/round_power.py`). Reporting them without
that caveat is how a null gets mistaken for a refutation.

Usage:
    python tools/resolution_ab_report.py outputs/resolution-ab/<date>
"""

from __future__ import annotations

import contextlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validation_matrix as vm  # type: ignore[import-not-found]  # reason: sibling script, no stub

#: Cell half-diagonal at each arm's resolution, `res * sqrt(3) / 2`.
HALF_DIAGONAL_MM = {"0.025": 21.65, "0.015": 12.99}


def _stops(arm_dir: Path) -> list[dict[str, Any]]:
    """Every stop under one arm, with its certified truth when the run recorded one."""
    rows: list[dict[str, Any]] = []
    for records in sorted(arm_dir.glob("*/records.json")):
        scene = records.parent.name.rsplit("-", 1)[0]
        for record in json.loads(records.read_text(encoding="utf-8")):
            run_dir = records.parent / f"r{int(record['round']):02d}"
            log = run_dir / "run_deploy.log"
            if not log.is_file():
                continue
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            row: dict[str, Any] = {
                "scene": scene,
                "round": record["round"],
                "outcome": record.get("outcome"),
                "success": record.get("success"),
                "chunks": record.get("chunks"),
                "bond_teardown": record.get("nav2_bond_teardown"),
                "depth_mm": None,
                "certified_mm": None,
                "excess_mm": None,
                "resolution_m": None,
                "not_ready_attempts": record.get("dispatch_not_ready_attempts") or [],
                "started_at": record.get("started_at"),
                "wall_s": record.get("wall_s"),
            }
            # Prove this run really ran at its arm's resolution. The HAL echoes
            # the grid's own `resolution_m` in the voxel backing record, so an
            # arm that silently fell back to the 0.025 default cannot pass
            # unnoticed. Only stopped runs carry one; that is enough to attest
            # the arm.
            for line in lines:
                if '"resolution_m"' in line:
                    with contextlib.suppress(IndexError, ValueError):
                        row["resolution_m"] = float(
                            line.split('"resolution_m"')[1].split(":")[1].split(",")[0].strip(" }")
                        )
                    break
            stop = vm.parse_kernel_collision(lines)
            if stop is not None:
                row["depth_mm"] = stop.min_distance_m * 1000.0
                row["party"] = stop.party_a
                snap = vm.parse_json_log_line(lines, "sim.estop_ground_truth_snapshot")
                if snap is not None:
                    key = (
                        "nearest_payload_world_pairs"
                        if snap.get("stop_class") == "attached_payload"
                        else "nearest_robot_world_pairs"
                    )
                    pairs = [p for p in (snap.get(key) or []) if p.get("distance_certified")]
                    if pairs:
                        best = min(pairs, key=lambda p: float(p["distance_m"]))
                        row["certified_mm"] = float(best["distance_m"]) * 1000.0
                        row["excess_mm"] = row["certified_mm"] - row["depth_mm"]
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(__doc__)
        return 2
    root = Path(args[0])
    arms = {d.name: _stops(d) for d in sorted(root.iterdir()) if d.is_dir()}
    if not arms:
        print(f"no arms under {root}")
        return 2

    print(f"=== 25 mm vs 15 mm world-voxel A/B — {root}\n")
    print("PRIMARY — over-approximation (certified gap minus reported depth), per stop")
    print(f"{'arm':>8} {'half-diag':>10} {'stops':>6} {'median excess':>14} {'range':>18}")
    medians: dict[str, float] = {}
    for arm, rows in sorted(arms.items(), reverse=True):
        excess = [r["excess_mm"] for r in rows if r["excess_mm"] is not None]
        if not excess:
            print(f"{arm:>8} {'':>10} {0:>6}   no adjudicable stops")
            continue
        medians[arm] = statistics.median(excess)
        print(
            f"{arm:>8} {HALF_DIAGONAL_MM.get(arm, float('nan')):>9.2f}mm {len(excess):>6} "
            f"{medians[arm]:>13.1f}mm {min(excess):>8.1f} .. {max(excess):<7.1f}"
        )
    if len(medians) == 2:
        coarse, fine = "0.025", "0.015"
        if coarse in medians and fine in medians:
            shift = medians[fine] - medians[coarse]
            predicted = HALF_DIAGONAL_MM[fine] - HALF_DIAGONAL_MM[coarse]
            print(f"\n  measured shift {shift:+.1f} mm   predicted {predicted:+.2f} mm")

    print("\nINTEGRITY — the resolution each arm's graph actually ran at, and re-dispatches")
    for arm, rows in sorted(arms.items(), reverse=True):
        seen = sorted({r["resolution_m"] for r in rows if r["resolution_m"] is not None})
        retried = sum(1 for r in rows if r["not_ready_attempts"])
        gave_up = sum(1 for r in rows if any("GAVE UP" in a for a in r["not_ready_attempts"]))
        flag = "" if seen == [float(arm)] else "   <-- MISMATCH, arm did not run at its resolution"
        print(
            f"{arm:>8}  observed resolution_m={seen or 'none recorded'}"
            f"   re-dispatched={retried}  gave-up={gave_up}{flag}"
        )

    print("\nPAIRING — was each round's partner the very next thing the host did?")
    print("Not `did the arms run the same evening`: the endpoint is paired per round,")
    print("so what matters is that nothing came between round n of one arm and round n")
    print("of the other. Measured end-to-start, so a long round separates nobody.")
    spans: dict[tuple[str, str, int], tuple[float, float]] = {}
    walls: list[float] = []
    for arm, rows in arms.items():
        for r in rows:
            if r.get("started_at") and r.get("wall_s") is not None:
                begin = float(r["started_at"])
                spans[(r["scene"], arm, int(r["round"]))] = (begin, begin + float(r["wall_s"]))
                walls.append(float(r["wall_s"]))
    if not spans:
        print("  no run recorded `started_at` — pre-2026-09-10 data, pairing unverifiable")
    # One typical round of slack. Arms that alternate are adjacent by construction
    # and separate by seconds; arms run as back-to-back lanes separate by half a
    # lane, which is orders of magnitude over this however it is set.
    budget = statistics.median(walls) if walls else 0.0
    for scene in sorted({scene for scene, _, _ in spans}):
        seps = []
        for n in sorted({n for sc, _a, n in spans if sc == scene}):
            coarse_span = spans.get((scene, "0.025", n))
            fine_span = spans.get((scene, "0.015", n))
            if coarse_span and fine_span:
                first, second = sorted((coarse_span, fine_span))
                seps.append(max(0.0, second[0] - first[1]))
        if not seps:
            print(f"{scene:>10}  no round has both arms — not a pair")
            continue
        unpaired = sum(1 for gap in seps if gap > budget)
        flag = "" if not unpaired else "   <-- NOT PAIRED, do not read this scene's shift"
        print(
            f"{scene:>10}  {len(seps):>3} paired rounds   median separation "
            f"{statistics.median(seps):>6.0f}s   worst {max(seps):>6.0f}s "
            f"(budget {budget:.0f}s, {unpaired} over){flag}"
        )

    print("\nSECONDARY — counts. UNDER-POWERED: 80% power on completion needs 200 runs/arm")
    print("(tools/round_power.py --baseline 0.027 --alternative 0.108). Do not read a")
    print("null here as a refutation.")
    print(f"{'arm':>8} {'valid':>6} {'completed':>10} {'stopped':>8} {'teardowns':>10}")
    for arm, rows in sorted(arms.items(), reverse=True):
        # A run whose graph never assembled delivered no action chunks and is not
        # a policy result (#263). Excluding it is the correction that moved the
        # ceiling battery from 62.5/2.7 to 80.0/4.3.
        valid = [
            r for r in rows if r["outcome"] != "harness-error" and r["chunks"] not in (0, None)
        ]
        print(
            f"{arm:>8} {len(valid):>6} {sum(1 for r in valid if r['success']):>10} "
            f"{sum(1 for r in valid if r['depth_mm'] is not None):>8} "
            f"{sum(1 for r in rows if r['bond_teardown']):>10}"
        )
    out = root / "report.json"
    out.write_text(json.dumps(arms, indent=1), encoding="utf-8")
    print(f"\nper-stop rows: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
