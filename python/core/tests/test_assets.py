"""Tests for the single description-asset resolver :func:`resolve_asset`."""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.assets import AssetRefError, resolve_asset


def test_rd_urdf_resolves_to_existing_file() -> None:
    p = resolve_asset("rd:panda_description", "urdf")
    assert p is not None and p.is_file() and p.suffix == ".urdf"


def test_rd_mjcf_resolves_to_existing_file() -> None:
    p = resolve_asset("rd:panda_mj_description", "mjcf")
    assert p is not None and p.is_file() and p.suffix in {".xml", ".mjcf"}


def test_rd_urdf_on_xacro_only_module_raises_with_guidance() -> None:
    with pytest.raises(AssetRefError) as e:
        resolve_asset("rd:ur5e_description", "urdf")
    assert "vendor-urdf" in str(e.value)


def test_file_resolves_against_manifest_dir(tmp_path: Path) -> None:
    (tmp_path / "robot.srdf").write_text("<robot/>")
    p = resolve_asset("file:robot.srdf", "srdf", manifest_dir=tmp_path)
    assert p == tmp_path / "robot.srdf"


def test_unknown_scheme_raises() -> None:
    with pytest.raises(AssetRefError):
        resolve_asset("python:robot_descriptions.panda_description:URDF_PATH", "urdf")


def test_unknown_openarm_variant_raises() -> None:
    """``openarm:<variant>`` dispatches explicitly; a typo'd variant must not
    silently resolve to some other arm's MJCF (the pre-Anvil resolver ignored
    the variant string entirely — this pins the stricter behaviour)."""
    with pytest.raises(AssetRefError, match="unknown openarm variant"):
        resolve_asset("openarm:anvil_bimanual", "mjcf")


def test_openarm_variants_are_mjcf_only() -> None:
    """Both wired variants reject non-mjcf kinds before any fetch happens."""
    for ref in ("openarm:bimanual", "openarm:anvil_v2_bimanual"):
        with pytest.raises(AssetRefError, match="mjcf-only"):
            resolve_asset(ref, "urdf")


def test_ros2_marker_is_passthrough_not_a_file() -> None:
    assert resolve_asset("ros2://robot_description", "urdf") is None
