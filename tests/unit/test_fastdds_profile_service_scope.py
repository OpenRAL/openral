"""The deploy Fast DDS profile's longer reply wait reaches Nav2 lifecycle replies only.

``rmw_fastrtps`` waits up to a reply writer's ``max_blocking_time`` for the client's
reply reader to be discovered, on the server's executor thread. Nav2's lifecycle
manager needs that wait longer than 100 ms or its change_state / get_state replies are
dropped and Nav2 never activates. The safety kernel serves its own lifecycle and
estop_reset services from a single-threaded executor, so a catch-all ``service``
profile (or one naming the kernel) could hold its only thread for the full wait.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from openral_cli.deploy_sim import FASTDDS_LARGE_DATA_PROFILE

_NS = "{http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles}"
_NAV2_MANAGED = {
    "behavior_server",
    "bt_navigator",
    "collision_monitor",
    "controller_server",
    "docking_server",
    "planner_server",
    "route_server",
    "smoother_server",
    "velocity_smoother",
    "waypoint_follower",
}


def _writer_profiles() -> list[str]:
    root = ET.parse(FASTDDS_LARGE_DATA_PROFILE).getroot()
    return [w.attrib["profile_name"] for w in root.iter(f"{_NS}data_writer")]


def test_no_catch_all_service_or_default_writer_profile() -> None:
    root = ET.parse(FASTDDS_LARGE_DATA_PROFILE).getroot()
    for writer in root.iter(f"{_NS}data_writer"):
        assert writer.attrib["profile_name"] != "service"
        assert writer.attrib.get("is_default_profile") != "true"


def test_only_nav2_lifecycle_replies_get_the_longer_wait() -> None:
    names = _writer_profiles()
    expected = {
        f"rr/{node}/{srv}Reply" for node in _NAV2_MANAGED for srv in ("change_state", "get_state")
    }
    assert set(names) == expected
    assert not any("safety" in n or "kernel" in n or "estop" in n for n in names)
