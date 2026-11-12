# SPDX-License-Identifier: Apache-2.0
"""Every in-tree vendor ``real_bringup`` launch refuses a networked ROS graph (issue #556).

``deploy_e2e.launch.py hal_mode:=real`` refused a networked graph, but the vendor bringups
it includes are also run by hand, and a hand-run one on ROS's default range (``SUBNET``,
which ``setup.bash`` exports) joined the LAN graph. Each launch now applies
``openral_core.ros_graph_scope.require_host_confined_ros_scope`` to its own environment
before composing any node. The launches are discovered from ``robots/*/robot.yaml`` so a
new robot's bringup is covered too.

Each launch description is walked the way ``LaunchService`` visits it
(``walk_launch`` from ``openral_rskill_ros``'s launch-test helpers): every action really
runs, but a process is recorded instead of spawned, so no hardware can start here even if
the gate regresses.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from openral_core.exceptions import ROSConfigError

_REPO = Path(__file__).resolve().parents[2]
#: Variables that make an env host-confined, opted in, or networked through another
#: branch of ``networked_ros_scope`` than the discovery range under test.
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
)


def _in_tree_real_bringups() -> list[Path]:
    launches = set()
    for manifest in sorted((_REPO / "robots").glob("*/robot.yaml")):
        ref = (yaml.safe_load(manifest.read_text()).get("hal") or {}).get("real_bringup")
        if ref:
            package, launch_file = ref.split(":")
            # An upstream vendor launch cannot carry our gate; deploy_e2e's covers it.
            if (_REPO / "packages" / package).is_dir():
                launches.add(_REPO / "packages" / package / "launch" / launch_file)
    return sorted(launches)


_LAUNCHES = _in_tree_real_bringups()


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _walk(
    launch: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str], started: list[Any]
) -> None:
    """Walk ``launch`` under ``env``, appending the processes it would start to ``started``."""
    pytest.importorskip("launch", reason="`launch` not importable — source ROS 2")
    pytest.importorskip("launch_ros", reason="`launch_ros` not importable — source ROS 2")
    from launch import LaunchContext

    walk_launch = _load(
        _REPO / "packages" / "openral_rskill_ros" / "test" / "_launch_test_common.py",
        "openral_launch_test_common",
    ).walk_launch
    module = _load(launch, f"real_bringup_{launch.parent.parent.name}")
    for var in _SCOPE_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    walk_launch(module.generate_launch_description(), LaunchContext(), started)


def test_in_tree_bringups_are_found() -> None:
    assert _LAUNCHES, "no robots/*/robot.yaml declares an in-tree hal.real_bringup"
    for launch in _LAUNCHES:
        assert launch.is_file(), launch


@pytest.mark.parametrize(
    "scope_env",
    [
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET"},  # what setup.bash exports
        {},  # absent = SUBNET to rcl
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": ""},
        {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ROS_STATIC_PEERS": "10.0.0.2"},
    ],
    ids=["subnet", "unset", "empty", "static-peers"],
)
@pytest.mark.parametrize("launch", _LAUNCHES, ids=lambda p: p.parent.parent.name)
def test_a_networked_real_bringup_starts_no_process(
    launch: Path, scope_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[Any] = []
    with pytest.raises(ROSConfigError, match="ROS_AUTOMATIC_DISCOVERY_RANGE="):
        _walk(launch, monkeypatch, scope_env, started)
    assert started == [], "a process was reached before the gate"


@pytest.mark.parametrize("launch", _LAUNCHES, ids=lambda p: p.parent.parent.name)
def test_a_host_confined_real_bringup_passes_the_gate(
    launch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: on LOCALHOST the walk gets past the gate (processes recorded, never spawned).

    Without the vendor packages installed it stops at resolving them instead; either way
    the scope is not what stops it.
    """
    try:
        _walk(launch, monkeypatch, {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}, [])
    except ROSConfigError as exc:
        assert "ROS graph" not in str(exc), exc
    except Exception as exc:  # reason: package lookup raises launch/ament-specific types
        assert "not found" in str(exc), exc
