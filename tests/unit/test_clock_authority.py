"""Clock authority contract tests.

The question this pins: `/clock` is not the source of truth. It is the ROS
projection of a named OpenRAL clock authority. The same authority labels
timestamps for sim, rSkills, real robots, and eventually hardware-synced clocks.
"""

from __future__ import annotations

import pytest
from openral_core import ClockAuthority, ClockEpoch, ClockOrigin
from pydantic import ValidationError


def test_simulation_authority_projects_to_ros_clock() -> None:
    authority = ClockAuthority.simulation("robocasa", timestep_s=0.05)

    assert authority.origin is ClockOrigin.SIMULATION
    assert authority.epoch is ClockEpoch.SIMULATION_ELAPSED
    assert authority.publishes_ros_clock is True
    assert authority.timestep_s == pytest.approx(0.05)


def test_host_wall_authority_is_default_real_deployment_origin() -> None:
    authority = ClockAuthority.host_wall()

    assert authority.origin is ClockOrigin.HOST_WALL
    assert authority.epoch is ClockEpoch.UNIX
    assert authority.publishes_ros_clock is False


def test_invalid_origin_epoch_pair_is_rejected() -> None:
    with pytest.raises(ValidationError, match="simulation_elapsed"):
        ClockAuthority(
            origin=ClockOrigin.SIMULATION,
            epoch=ClockEpoch.UNIX,
            clock_id="bad_sim_clock",
        )
