"""``fold_robot_contacts``: per-episode robot-vs-world contact statistics (pure)."""

from __future__ import annotations

from typing import Any

from openral_sim.rollout import EpisodeResult, fold_robot_contacts


def _c(link: str, other: str, sep: float, imp: float = 0.0) -> dict[str, Any]:
    return {"link": link, "other_prim": other, "min_separation_m": sep, "impulse_ns": imp}


def test_pairs_are_kept_apart_and_extremes_are_tracked() -> None:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    fold_robot_contacts(stats, [_c("finger", "/World/tote/wall", 0.003, 0.1)], step=10)
    fold_robot_contacts(
        stats,
        [_c("finger", "/World/tote/wall", -0.002, 0.4), _c("finger", "/World/carton", -0.006, 0.2)],
        step=11,
    )
    fold_robot_contacts(stats, [_c("finger", "/World/tote/wall", 0.001, 0.05)], step=15)
    wall = stats[("finger", "/World/tote/wall")]
    assert (wall["steps"], wall["first_step"], wall["last_step"]) == (3, 10, 15)
    assert wall["min_separation_m"] == -0.002
    assert wall["max_penetration_m"] == 0.002
    assert wall["max_impulse_ns"] == 0.4
    carton = stats[("finger", "/World/carton")]
    assert (carton["steps"], carton["max_penetration_m"]) == (1, 0.006)


def test_a_contact_only_inside_the_offset_band_records_no_penetration() -> None:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    fold_robot_contacts(stats, [_c("wrist", "/World/shelf", 0.004)], step=1)
    assert stats[("wrist", "/World/shelf")]["max_penetration_m"] == 0.0


def test_episode_result_defaults_to_no_contacts() -> None:
    assert EpisodeResult().contact_stats == []
