# SPDX-License-Identifier: Apache-2.0
"""A self-pair whose link has no stage-2 hull is a box measurement, not an unadjudicable one.

Issue #260. ``hull_hull_distance`` returns the OBB bound untouched when either
side carries no hull vertices, and deliberately leaves ``depth_is_box_bound``
clear — the flag's contract is "a hull refinement was *attempted* and fell
back", and on that path none was attempted
(``SelfCollisionHull.TheFlagIsClearedWhenNoHullRefinementHappened``).

The adjudicator read the clear flag as "the kernel measured at hull fidelity",
charged the pair a hull budget, then found the link had no measured overhang
and gave up. `panda_link1` ships no stage-2 hull **by decision** (#191 — its
refined envelope moved its own stops by 0.0003 mm and was withdrawn), so every
self-pair naming it was permanently ``unadjudicated``: two of the seven self
stops in the 2026-09-10 ceiling battery.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from openral_core import RobotDescription, ValidationStopEvidence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import validation_matrix as vm  # type: ignore[import-not-found]  # reason: sibling script, no stub

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def panda() -> RobotDescription:
    return RobotDescription.from_yaml(REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml")


def test_the_shipped_manifest_still_has_a_link_with_no_stage2_hull(
    panda: RobotDescription,
) -> None:
    """The premise. If `panda_link1` ever gains a hull, this whole path is moot."""
    link1 = next(g for g in panda.collision_geometry if g.link_name == "panda_link1")
    assert link1.tight_geometry is not None
    assert not (link1.tight_geometry.hull_vertices_m or ()), (
        "panda_link1 is expected to ship a stage-1-only tight_geometry block"
    )
    others = [
        g
        for g in panda.collision_geometry
        if g.link_name != "panda_link1" and g.tight_geometry is not None
    ]
    assert others, "the manifest must still carry hulls to contrast against"
    assert all(g.tight_geometry is not None and g.tight_geometry.hull_vertices_m for g in others)


def _budget(link_a: dict[str, Any], link_b: dict[str, Any], *, box_m: float) -> float | None:
    stop = ValidationStopEvidence(
        kind="self",
        party_a="panda_link1",
        party_b="panda_link7",
        horizon_step=0,
        min_distance_m=-0.0017,
        sweep_min_distance_m=-0.0017,
        place_allowance_active=False,
        depth_is_box_bound=False,
        place_target="",
    )
    # Shape mirrors a real `sim.estop_ground_truth_snapshot`: the box term lives
    # inside `link_link`, beside the rule that describes it.
    snapshot = {
        "adjudication_budget": {
            "link_link": {"admissible_gap_box_m": box_m, "max_corner_slop_m": box_m / 2},
            "collision_model_slop": {"links": {"panda_link1": link_a, "panda_link7": link_b}},
        }
    }
    return vm.hal_admissible_gap_m(snapshot, stop)


def test_a_link_with_no_stage2_hull_is_charged_the_box_budget() -> None:
    """The fix: no hull means the kernel measured with boxes, so budget with boxes."""
    got = _budget(
        {"hull_overhang_m": None, "has_stage2_hull": False},
        {"hull_overhang_m": 8.9e-05, "has_stage2_hull": True},
        box_m=0.1764,
    )
    assert got == pytest.approx(0.1764)


def test_a_hull_whose_overhang_was_never_measured_stays_unadjudicable() -> None:
    """The other half of `None`, and it must NOT get the box budget.

    Here the kernel really did refine to hull fidelity, so nothing bounds the
    hull-to-mesh gap and the honest answer is still "no budget".
    """
    got = _budget(
        {"hull_overhang_m": None, "has_stage2_hull": True},
        {"hull_overhang_m": 8.9e-05, "has_stage2_hull": True},
        box_m=0.1764,
    )
    assert got is None


def test_a_snapshot_predating_the_field_keeps_the_old_behaviour() -> None:
    """Absent is "unknown", never "no hull".

    Reading absence as "no hull" would hand the box budget to every genuine
    hull measurement on every historical round, turning correct stops into
    `within-quantization` — the exact failure `hal_admissible_gap_m` exists to
    prevent, in the other direction.
    """
    got = _budget(
        {"hull_overhang_m": None},
        {"hull_overhang_m": 8.9e-05},
        box_m=0.1764,
    )
    assert got is None


def test_two_hulled_links_still_sum_their_overhangs() -> None:
    """The #221 path is untouched."""
    got = _budget(
        {"hull_overhang_m": 0.000259, "has_stage2_hull": True},
        {"hull_overhang_m": 8.9e-05, "has_stage2_hull": True},
        box_m=0.1764,
    )
    assert got == pytest.approx(0.000348)
