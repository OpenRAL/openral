# SPDX-License-Identifier: Apache-2.0
"""Every in-tree vendor ``real_bringup`` launch refuses a networked ROS graph (issue #556).

``deploy_e2e.launch.py hal_mode:=real`` refused a networked graph, but the vendor bringups
it includes are also run by hand, and a hand-run one on ROS's default range (``SUBNET``,
which ``setup.bash`` exports) joined the LAN graph. Each launch now runs
``openral_core.ros_graph_scope.require_host_confined_real_bringup`` on its own environment
before composing any node. The launches are discovered from ``robots/*/robot.yaml``
(resolved through each package's ``package.xml``), so a new robot's bringup is covered too.

Each launch description is walked the way ``LaunchService`` visits it (``walk_launch`` from
``openral_rskill_ros``'s launch-test helpers): a process is recorded instead of spawned.
Other actions really run, so on a host with the vendor packages installed a walk that gets
past the gate (the LOCALHOST control, or a regressed gate) executes the vendor launch's
non-process actions; no process is ever started.
"""

from __future__ import annotations

import functools
import importlib.util
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import structlog
import yaml
from openral_core.exceptions import ROSConfigError

_REPO = Path(__file__).resolve().parents[2]
_SCOPE_MODULE = "openral_core.ros_graph_scope"
#: Variables that make an env host-confined, opted in, secured, or networked through
#: another branch of ``networked_ros_scope`` than the discovery range under test.
_SCOPE_VARS = (
    "OPENRAL_ALLOW_NETWORK_ROS",
    "ROS_LOCALHOST_ONLY",
    "ROS_AUTOMATIC_DISCOVERY_RANGE",
    "ROS_STATIC_PEERS",
    "ROS_DISCOVERY_SERVER",
    "RMW_IMPLEMENTATION",
    "CYCLONEDDS_URI",
    "FASTRTPS_DEFAULT_PROFILES_FILE",
    "FASTDDS_DEFAULT_PROFILES_FILE",
    "ROS_SECURITY_ENABLE",
    "ROS_SECURITY_STRATEGY",
    "ROS_SECURITY_KEYSTORE",
)


def _in_tree_real_bringups() -> list[Path]:
    packages = {
        ET.parse(xml).getroot().findtext("name"): xml.parent
        for xml in (_REPO / "packages").rglob("package.xml")
    }
    launches = set()
    for manifest in sorted((_REPO / "robots").glob("*/robot.yaml")):
        ref = (yaml.safe_load(manifest.read_text()).get("hal") or {}).get("real_bringup")
        if ref:
            package, launch_file = ref.split(":")
            # An upstream vendor launch cannot carry our gate; deploy_e2e's covers it.
            if package in packages:
                launches.add(packages[package] / "launch" / launch_file)
    return sorted(launches)


_LAUNCHES = _in_tree_real_bringups()
_IDS = [p.parent.parent.name for p in _LAUNCHES]


def _requires_launch() -> None:
    pytest.importorskip("launch", reason="`launch` not importable — source ROS 2")
    pytest.importorskip("launch_ros", reason="`launch_ros` not importable — source ROS 2")


@functools.cache
def _helpers() -> ModuleType:
    path = _REPO / "packages" / "openral_rskill_ros" / "test" / "_launch_test_common.py"
    spec = importlib.util.spec_from_file_location("openral_launch_test_common", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def _launch_module(launch: Path) -> ModuleType:
    return _helpers().import_launch_module(launch)  # type: ignore[no-any-return] # reason: helper module loaded by path


def _set_scope(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for var in _SCOPE_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)


def _walk(description: Any, started: list[Any]) -> None:  # reason: LaunchDescription
    from launch import LaunchContext

    _helpers().walk_launch(description, LaunchContext(), started)


def _assert_past_the_gate(exc: BaseException) -> None:
    """``exc`` came from past the scope gate: from what this host lacks, never the gate.

    Without the vendor packages the walk stops resolving one (``not found``); with them it
    stops at an artifact built on the robot's host, such as the G1's LEAPP bundle
    (``does not exist``).
    """
    frames = [f.filename for f in traceback.extract_tb(exc.__traceback__)]
    assert not any(f.endswith("ros_graph_scope.py") for f in frames), exc
    assert re.search("not found|does not exist", str(exc)), exc


def _included(launch: Path) -> Any:  # reason: LaunchDescription
    """``launch`` as ``deploy_e2e`` includes it: under a set ``hal_mode``."""
    from launch import LaunchDescription
    from launch.actions import IncludeLaunchDescription, SetLaunchConfiguration
    from launch.launch_description_sources import PythonLaunchDescriptionSource

    return LaunchDescription(
        [
            SetLaunchConfiguration("hal_mode", "real"),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(str(launch))),
        ]
    )


def test_in_tree_bringups_are_found() -> None:
    assert _LAUNCHES, "no robots/*/robot.yaml declares an in-tree hal.real_bringup"
    for launch in _LAUNCHES:
        assert launch.is_file(), launch


@pytest.mark.parametrize(
    ("scope_env", "scope"),
    [
        ({"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"}, "SUBNET"),  # what setup.bash exports
        ({}, "(unset = SUBNET)"),
        ({"ROS_AUTOMATIC_DISCOVERY_RANGE": ""}, "(unset = SUBNET)"),
        (
            {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ROS_STATIC_PEERS": "10.0.0.2"},
            "LOCALHOST ROS_STATIC_PEERS=10.0.0.2",
        ),
    ],
    ids=["subnet", "unset", "empty", "static-peers"],
)
@pytest.mark.parametrize("launch", _LAUNCHES, ids=_IDS)
def test_a_networked_real_bringup_starts_no_process(
    launch: Path, scope_env: dict[str, str], scope: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _requires_launch()
    _set_scope(monkeypatch, scope_env)
    started: list[Any] = []
    with pytest.raises(
        ROSConfigError,
        match=re.escape(f"reaches other hosts (ROS_AUTOMATIC_DISCOVERY_RANGE={scope}"),
    ):
        _walk(_launch_module(launch).generate_launch_description(), started)
    assert started == [], "a process was reached before the gate"


@pytest.mark.parametrize("launch", _LAUNCHES, ids=_IDS)
def test_a_host_confined_real_bringup_passes_the_gate(
    launch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: on LOCALHOST nothing raised comes from the scope gate.

    With the vendor packages and their host-built artifacts the walk records the bringup's
    processes (never spawned); without them it stops past the gate.
    """
    _requires_launch()
    _set_scope(monkeypatch, {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"})
    try:
        _walk(_launch_module(launch).generate_launch_description(), [])
    except Exception as exc:  # reason: package lookup raises launch/ament-specific types
        _assert_past_the_gate(exc)


@pytest.mark.parametrize("launch", _LAUNCHES, ids=_IDS)
def test_a_bringup_without_openral_core_fails_closed(
    launch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``/opt/ros``'s system Python the gate cannot run, so the launch must not either."""
    _requires_launch()
    _set_scope(monkeypatch, {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"})
    monkeypatch.setitem(sys.modules, _SCOPE_MODULE, None)  # import raises ImportError
    started: list[Any] = []
    with pytest.raises(ImportError, match="openral_core"):
        _walk(_helpers().import_launch_module(launch).generate_launch_description(), started)
    assert started == []


@pytest.mark.parametrize("launch", _LAUNCHES, ids=_IDS)
def test_an_opted_in_bringup_logs_the_hazard_only_when_run_by_hand(
    launch: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#558: under ``deploy_e2e`` its gate is the one emitter; a hand run logs it itself.

    An opted-in networked graph runs only under SROS2 Enforce (ADR-0130), so this builds
    a real keystore.
    """
    _requires_launch()
    security = pytest.importorskip("openral_cli.security")
    from openral_core.ros_security import render_sros2_policy

    if security.shutil.which("ros2") is None:
        pytest.skip("SROS2 keystore needs `ros2` on PATH")
    security.generate_keystore(render_sros2_policy({}), tmp_path / "ks", domain="77")
    _set_scope(
        monkeypatch,
        {
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET",
            "OPENRAL_ALLOW_NETWORK_ROS": "1",
            "ROS_DOMAIN_ID": "77",
            "ROS_SECURITY_ENABLE": "true",
            "ROS_SECURITY_STRATEGY": "Enforce",
            "ROS_SECURITY_KEYSTORE": str(tmp_path / "ks"),
        },
    )
    events = {}
    for name, description in (
        ("hand", _launch_module(launch).generate_launch_description()),
        ("deploy", _included(launch)),
    ):
        with structlog.testing.capture_logs() as logs:
            try:
                _walk(description, [])
            except Exception as exc:  # reason: past the gate, the vendor package lookup
                _assert_past_the_gate(exc)
        events[name] = [log["event"] for log in logs if log["log_level"] == "warning"]
    assert events == {"hand": ["ros_graph_scope.real_bringup_networked"], "deploy": []}
