"""The deploy launch environment carries the Fast DDS large-data profile, explicitly.

Under Jazzy's default 512 KiB shared-memory segment a same-host BEST_EFFORT sample
above ~512 KB is lost almost every time, so the deploy graph's megabyte depth clouds
mostly never reached the robot self-filter (``docs/reference/dds-large-messages.md``).
``_apply_rmw_default`` now exports the shipped profile and names it on the
``dds_transport_ready:`` line. Delivery itself is proven live, through the real
self-filter, in ``tests/integration/test_large_cloud_transport_live.py``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from openral_cli.deploy_sim import (
    DDS_TRANSPORT_READY_MARKER,
    FASTDDS_LARGE_DATA_PROFILE,
    FASTDDS_PROFILES_ENV,
    FASTDDS_SHM_CLEAN_ENV,
    _apply_rmw_default,
)

_NS = {"f": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
#: An Isaac 512x384 depth camera's base-frame cloud, the largest sample measured lost.
_SIM_CLOUD_BYTES = 2_800_000


def test_default_fastdds_gets_the_shipped_profile(capsys: pytest.CaptureFixture[str]) -> None:
    env = {FASTDDS_SHM_CLEAN_ENV: "0"}
    _apply_rmw_default(env)
    assert env[FASTDDS_PROFILES_ENV] == str(FASTDDS_LARGE_DATA_PROFILE)
    line = capsys.readouterr().out
    assert DDS_TRANSPORT_READY_MARKER in line
    assert f"fastdds_profile={FASTDDS_LARGE_DATA_PROFILE}" in line


def test_an_operator_profile_wins_and_is_named(capsys: pytest.CaptureFixture[str]) -> None:
    env = {FASTDDS_SHM_CLEAN_ENV: "0", FASTDDS_PROFILES_ENV: "/etc/openral/site_fastdds.xml"}
    _apply_rmw_default(env)
    assert env[FASTDDS_PROFILES_ENV] == "/etc/openral/site_fastdds.xml"
    assert "fastdds_profile=/etc/openral/site_fastdds.xml" in capsys.readouterr().out


def test_an_empty_profile_variable_is_replaced() -> None:
    """Fast DDS logs ``XMLPARSER Error ... realpath failed`` for an empty path."""
    env = {FASTDDS_SHM_CLEAN_ENV: "0", FASTDDS_PROFILES_ENV: ""}
    _apply_rmw_default(env)
    assert env[FASTDDS_PROFILES_ENV] == str(FASTDDS_LARGE_DATA_PROFILE)


def test_cyclone_is_left_alone(capsys: pytest.CaptureFixture[str]) -> None:
    env = {"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"}
    _apply_rmw_default(env)
    assert FASTDDS_PROFILES_ENV not in env
    assert "fastdds_profile=n/a" in capsys.readouterr().out


def test_profile_enlarges_only_the_shm_segment_and_keeps_udp() -> None:
    root = ET.parse(FASTDDS_LARGE_DATA_PROFILE).getroot()
    kinds = {
        d.findtext("f:transport_id", namespaces=_NS): d
        for d in root.iterfind("f:transport_descriptors/f:transport_descriptor", _NS)
    }
    (participant,) = root.iterfind("f:participant", _NS)
    assert participant.get("is_default_profile") == "true"
    used = [t.text for t in participant.iterfind("f:rtps/f:userTransports/f:transport_id", _NS)]
    assert sorted(kinds[t].findtext("f:type", namespaces=_NS) or "" for t in used) == [
        "SHM",
        "UDPv4",
    ]
    (shm,) = (kinds[t] for t in used if kinds[t].findtext("f:type", namespaces=_NS) == "SHM")
    # Room for a cloud, its depth and colour frames, and one more in flight.
    assert int(shm.findtext("f:segment_size", namespaces=_NS) or 0) >= 4 * _SIM_CLOUD_BYTES
