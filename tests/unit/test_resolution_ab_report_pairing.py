# SPDX-License-Identifier: Apache-2.0
"""The report has to say whether a scene's two arms actually overlapped in time.

`tools/resolution_ab.sh` now schedules them together, but a report is read long
after the run and often against data some other version produced. The 2026-09-10
run's `sink_cup` arms were 29 minutes apart and nothing in the output said so —
the 7/10-vs-0/10 gap it produced was taken for a resolution effect until the
directory mtimes were checked by hand. This is that check, made from the data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "tools" / "resolution_ab_report.py"


def _lane(root: Path, arm: str, scene: str, *, first_start: float, rounds: int = 3) -> None:
    """A real arm directory: records.json plus the run_deploy.log each row needs."""
    lane = root / arm / f"{scene}-on"
    records = []
    for n in range(1, rounds + 1):
        run = lane / f"r{n:02d}"
        run.mkdir(parents=True, exist_ok=True)
        # No kernel stop line: the row is still built, which is all pairing needs.
        (run / "run_deploy.log").write_text(
            f'[hal] {{"event": "sim.voxel_backing", "resolution_m": {float(arm)}}}\n',
            encoding="utf-8",
        )
        records.append(
            {
                "outcome": "ran",
                "success": False,
                "chunks": 200,
                "round": n,
                "scene": scene,
                "started_at": first_start + (n - 1) * 100.0,
                "wall_s": 100.0,
                "nav2_bond_teardown": None,
                "dispatch_not_ready_attempts": [],
            }
        )
    (lane / "records.json").write_text(json.dumps(records), encoding="utf-8")


def _run(root: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(REPORT), str(root)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def test_arms_that_ran_back_to_back_are_flagged_not_paired(tmp_path: Path) -> None:
    """The sink_cup shape: one arm runs all its rounds, then the other starts."""
    _lane(tmp_path, "0.025", "sink_cup", first_start=1_000_000.0)
    _lane(tmp_path, "0.015", "sink_cup", first_start=1_000_400.0)  # a whole lane later

    out = _run(tmp_path)
    assert "PAIRING" in out
    assert "NOT PAIRED" in out, out
    assert "sink_cup" in out


def test_arms_that_ran_together_are_not_flagged(tmp_path: Path) -> None:
    """The alternating shape: round n of each arm is minutes from its partner."""
    # Alternating: each arm's round n sits one round away from its partner's.
    _lane(tmp_path, "0.025", "baguette", first_start=1_000_000.0)
    _lane(tmp_path, "0.015", "baguette", first_start=1_000_100.0)

    out = _run(tmp_path)
    assert "PAIRING" in out
    assert "NOT PAIRED" not in out, out
    assert "3 paired rounds" in out, out


def test_data_without_a_timestamp_says_so_instead_of_passing(tmp_path: Path) -> None:
    """Runs predating `started_at` must not read as verified pairing."""
    _lane(tmp_path, "0.025", "fridge", first_start=1_000_000.0)
    _lane(tmp_path, "0.015", "fridge", first_start=1_000_005.0)
    for lane in tmp_path.glob("*/fridge-on/records.json"):
        rows = json.loads(lane.read_text(encoding="utf-8"))
        for row in rows:
            del row["started_at"]
        lane.write_text(json.dumps(rows), encoding="utf-8")

    out = _run(tmp_path)
    assert "pairing unverifiable" in out, out
    assert "NOT PAIRED" not in out
