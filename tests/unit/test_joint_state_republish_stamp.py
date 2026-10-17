"""The HAL's ``~/joint_states`` republish carries the sample's age, not the read time.

The robot self-filter pairs each depth cloud with the joint state nearest its capture
stamp and trusts a 0.1 s skew bound (``max_joint_state_skew_s``). Stamping the republish
with the node's now made a sample up to the staleness limit old look fresh, so the filter
posed the arm where it had been, not where it was.
"""

from __future__ import annotations

from openral_hal.lifecycle import joint_state_republish_stamp_ns

_S = 1_000_000_000


def test_the_samples_wall_clock_age_is_carried_into_the_node_clock() -> None:
    # Node on sim time (1000 s), sample 0.3 s old on the wall clock.
    stamp = joint_state_republish_stamp_ns(1000 * _S, 50 * _S, wall_now_ns=50 * _S + 300_000_000)
    assert stamp == 1000 * _S - 300_000_000


def test_a_sample_from_another_clock_domain_keeps_the_node_now() -> None:
    now = 1000 * _S
    # Stamp ahead of the wall clock, far behind it, or absent: no age can be carried.
    assert joint_state_republish_stamp_ns(now, 60 * _S, wall_now_ns=50 * _S) == now
    assert joint_state_republish_stamp_ns(now, 1 * _S, wall_now_ns=50 * _S) == now
    assert joint_state_republish_stamp_ns(now, 0, wall_now_ns=50 * _S) == now


def test_a_frame_on_the_nodes_own_sim_clock_is_stamped_with_its_sim_time() -> None:
    # sf_dr2 (issue #418): Isaac at ~0.13x real time, one post-step frame at sim 27.283 s
    # republished six times at 15 Hz wall. The wall-age rule stamped it 27.159 s down to
    # 26.826 s (the self-filter then posed the arm at the wrong instant). Each republish
    # of that frame now carries its sim time, whatever its wall age.
    frame_sim_ns = 27_283_333_333
    sample_wall_ns = 50 * _S
    stamps = {
        joint_state_republish_stamp_ns(
            frame_sim_ns,
            sample_wall_ns,
            wall_now_ns=sample_wall_ns + k * 66_666_667,
            sim_time_ns=frame_sim_ns,
        )
        for k in range(7)
    }
    assert stamps == {frame_sim_ns}


def test_no_sim_time_keeps_the_wall_age_rule() -> None:
    stamp = joint_state_republish_stamp_ns(
        1000 * _S, 50 * _S, wall_now_ns=50 * _S + 300_000_000, sim_time_ns=None
    )
    assert stamp == 1000 * _S - 300_000_000
