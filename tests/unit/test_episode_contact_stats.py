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


def test_onset_speed_is_the_fastest_start_of_a_contact_run_and_max_is_the_peak() -> None:
    stats: dict[tuple[str, str], dict[str, Any]] = {}

    def at(step: int, speed: float) -> None:
        c = _c("finger", "/World/tote/wall", 0.0) | {"link_speed_mps": speed}
        fold_robot_contacts(stats, [c], step=step)

    at(5, 0.04)  # run 1 starts at 0.04 m/s
    at(6, 0.30)  # still run 1: a peak, not an onset
    at(9, 0.12)  # run 2 starts at 0.12 m/s
    rec = stats[("finger", "/World/tote/wall")]
    assert rec["onset_speed_mps"] == 0.12
    assert rec["max_speed_mps"] == 0.30


def test_contacts_without_a_speed_record_no_speed_fields() -> None:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    fold_robot_contacts(stats, [_c("finger", "/World/carton", -0.001)], step=1)
    assert "onset_speed_mps" not in stats[("finger", "/World/carton")]


def test_a_run_whose_first_sample_has_no_speed_takes_its_first_speed_as_onset() -> None:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    fold_robot_contacts(stats, [_c("finger", "/World/tote/wall", 0.0)], step=0)  # after reset
    c = _c("finger", "/World/tote/wall", 0.0) | {"link_speed_mps": 0.2}
    fold_robot_contacts(stats, [c], step=1)
    fold_robot_contacts(stats, [c | {"link_speed_mps": 0.5}], step=2)  # same run: not an onset
    assert stats[("finger", "/World/tote/wall")]["onset_speed_mps"] == 0.2
