# SPDX-License-Identifier: Apache-2.0
"""Foxglove can fetch the meshes of a URDF whose package is a fetched public clone.

On Spark (Isaac deploy sim, OpenArm) the 3D panel drew the TF axes and the voxel
cloud but no robot. `/robot_description` was live and the layout's URDF layer
subscribed it; every mesh fetch then failed in the bridge:

    Failed to retrieve asset 'package://openarm_description/.../body_link0.dae':
    ... Package [openarm_description] does not exist

`foxglove_bridge` resolves `package://` through `resource_retriever`, i.e. the
ament index on `AMENT_PREFIX_PATH`, and OpenArm's `openarm_description` is a pinned
public clone in the openral cache that no workspace indexes. These tests resolve
every mesh of the real `robots/openarm/openarm.urdf` the way the bridge does.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest
from openral_core.exceptions import ROSConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
URDF = REPO_ROOT / "robots" / "openarm" / "openarm.urdf"
LAUNCH = REPO_ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"


@pytest.fixture
def overlay_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh openral cache holding the real pinned `openarm_description` clone."""
    from openral_hal._openarm_description_assets import ensure_openarm_description

    try:
        clone = ensure_openarm_description()  # the real fetch (cached after the first run)
    except ROSConfigError as exc:
        pytest.skip(f"openarm_description clone unavailable: {exc}")
    cache = tmp_path / "cache"
    (cache / "openarm_description").mkdir(parents=True)
    (cache / "openarm_description" / clone.name).symlink_to(clone, target_is_directory=True)
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(cache))
    # The bridge's own env: the distro, with no workspace that builds the package.
    monkeypatch.setenv("AMENT_PREFIX_PATH", "/opt/ros/jazzy")
    return cache


def test_every_openarm_mesh_resolves_through_the_overlay(
    overlay_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ament = pytest.importorskip("ament_index_python.packages")
    from openral_hal.ros_package_overlay import public_package_overlay

    xml = URDF.read_text(encoding="utf-8")
    with pytest.raises(ament.PackageNotFoundError):
        ament.get_package_share_directory("openarm_description")

    overlay = public_package_overlay(xml)
    assert overlay == overlay_cache / "ament_overlay"
    monkeypatch.setenv("AMENT_PREFIX_PATH", f"{overlay}{os.pathsep}/opt/ros/jazzy")
    share = Path(ament.get_package_share_directory("openarm_description"))
    meshes = re.findall(r'filename="package://openarm_description/([^"]+)"', xml)
    assert meshes, "the OpenArm URDF names its meshes by package://"
    missing = [m for m in meshes if not (share / m).is_file()]
    assert not missing, f"meshes the bridge could not serve: {missing}"

    # Idempotent: the next launch (same bridge env, same cache) reuses the overlay.
    monkeypatch.setenv("AMENT_PREFIX_PATH", "/opt/ros/jazzy")
    assert public_package_overlay(xml) == overlay
    assert (share / meshes[0]).is_file()


def test_a_workspace_that_indexes_the_package_wins(
    overlay_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openral_hal.ros_package_overlay import public_package_overlay

    ws = tmp_path / "ws"
    (ws / "share" / "ament_index" / "resource_index" / "packages").mkdir(parents=True)
    (ws / "share" / "ament_index" / "resource_index" / "packages" / "openarm_description").touch()
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(ws))
    assert public_package_overlay(URDF.read_text(encoding="utf-8")) is None
    assert not (overlay_cache / "ament_overlay").exists()


def test_the_deploy_launch_gives_the_bridge_the_overlay(overlay_cache: Path) -> None:
    """`deploy_e2e` hands `foxglove_bridge` an `AMENT_PREFIX_PATH` that indexes the meshes."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_fox_assets", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_fox_assets"] = module
    spec.loader.exec_module(module)

    env = module._foxglove_asset_env(URDF.read_text(encoding="utf-8"))
    assert env["AMENT_PREFIX_PATH"].split(os.pathsep) == [
        str(overlay_cache / "ament_overlay"),
        "/opt/ros/jazzy",
    ]
    assert module._foxglove_asset_env(None) == {}
    text = LAUNCH.read_text(encoding="utf-8")
    assert "additional_env=_foxglove_asset_env(robot_description_xml)" in text
