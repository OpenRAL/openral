"""The checked-in visual-SLAM Nav2 profile must equal a fresh render of the base profile.

``tools/gen_nav2_visual.py`` derives ``nav2_visual.yaml`` from ``nav2_panda_mobile.yaml``;
editing the base without re-running it leaves the two profiles disagreeing on geometry
or planner tuning. Real files, no fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import gen_nav2_visual  # noqa: E402  # reason: needs the sys.path insert above


def test_checked_in_visual_profile_matches_a_fresh_render() -> None:
    base = gen_nav2_visual._BASE.read_text()
    assert gen_nav2_visual._OUT.read_text() == gen_nav2_visual.render(base)


def test_check_mode_reports_the_checked_in_state() -> None:
    assert gen_nav2_visual.main(["--check"]) == 0
