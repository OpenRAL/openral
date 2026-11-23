"""Tests for the single description-asset resolver ``resolve_asset``."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from openral_core.assets import AssetFetchError, AssetKind, AssetRefError, resolve_asset


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


def test_unknown_unitree_model_raises() -> None:
    """``unitree:<model>`` dispatches explicitly; only ``h2`` is wired."""
    with pytest.raises(AssetRefError, match="unknown unitree model"):
        resolve_asset("unitree:h3", "mjcf")


def test_unitree_h2_has_no_srdf() -> None:
    """``unitree:h2`` ships a URDF and an MJCF; an SRDF request fails before any fetch."""
    with pytest.raises(AssetRefError, match="not an srdf"):
        resolve_asset("unitree:h2", "srdf")


def test_unknown_makermods_variant_raises() -> None:
    """A typo'd ``makermods:<variant>`` is a broken ref, rejected before any fetch."""
    for kind in ("urdf", "mjcf"):
        with pytest.raises(AssetRefError, match="unknown makermods variant") as info:
            resolve_asset("makermods:metal_arm", kind)
        assert not isinstance(info.value, AssetFetchError)


def test_makermods_has_no_srdf() -> None:
    """Both wired variants reject ``srdf`` before any fetch happens."""
    for ref in ("makermods:metal", "makermods:metal_station"):
        with pytest.raises(AssetRefError, match="no srdf") as info:
            resolve_asset(ref, "srdf")
        assert not isinstance(info.value, AssetFetchError)


@pytest.mark.parametrize("ref", ["makermods:metal", "makermods:metal_station"])
@pytest.mark.parametrize("kind", ["urdf", "mjcf"])
def test_makermods_resolves_to_existing_file(ref: str, kind: AssetKind) -> None:
    """``makermods:`` derives a URDF / MJCF from the pinned upstream clone."""
    if kind == "mjcf":
        pytest.importorskip("mujoco")
    try:
        path = resolve_asset(ref, kind)
    except AssetFetchError as exc:
        pytest.skip(f"pinned MakerMods Metal clone not fetchable on this host: {exc}")
    assert path is not None and path.is_file()
    assert path.suffix == (".urdf" if kind == "urdf" else ".xml")
    assert (
        path.read_text(encoding="utf-8")
        .lstrip()
        .startswith("<robot" if kind == "urdf" else "<mujoco")
    )


def test_ros2_marker_is_passthrough_not_a_file() -> None:
    assert resolve_asset("ros2://robot_description", "urdf") is None


def test_behavior_rejects_unknown_model_and_kind() -> None:
    """``behavior:`` serves only the R1 Pro URDF; anything else fails before any fetch."""
    for ref, kind in (("behavior:r1", "urdf"), ("behavior:r1pro", "mjcf")):
        with pytest.raises(AssetRefError, match="only behavior:r1pro") as info:
            resolve_asset(ref, kind)  # type: ignore[arg-type]  # reason: AssetKind literal from a loop
        assert not isinstance(info.value, AssetFetchError)


def test_behavior_r1pro_urdf_loads_in_mujoco_with_visuals() -> None:
    """The derived URDF opens with its visual meshes: no unshipped collision ref left."""
    mujoco = pytest.importorskip("mujoco")
    try:
        path = resolve_asset("behavior:r1pro", "urdf")
    except AssetFetchError as exc:
        pytest.skip(f"BEHAVIOR-1K robot assets not fetchable on this host: {exc}")
    assert path is not None
    xml = path.read_text(encoding="utf-8").replace(
        "<link", '<mujoco><compiler strippath="false" discardvisual="false"/></mujoco><link', 1
    )
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.ngeom > 0
    assert "torso_joint1" in {model.joint(i).name for i in range(model.njnt)}


def test_galaxea_rejects_unknown_model_and_kind() -> None:
    """``galaxea:`` serves only the A1 URDF; anything else fails before any fetch."""
    for ref, kind in (("galaxea:r1pro", "urdf"), ("galaxea:a1", "mjcf")):
        with pytest.raises(AssetRefError, match="only galaxea:a1") as info:
            resolve_asset(ref, kind)  # type: ignore[arg-type]  # reason: AssetKind literal from a loop
        assert not isinstance(info.value, AssetFetchError)


def test_galaxea_a1_urdf_meshes_resolve_beside_it() -> None:
    """The derived A1 URDF names its meshes relative to itself, and every one is on disk."""
    try:
        path = resolve_asset("galaxea:a1", "urdf")
    except AssetFetchError as exc:
        pytest.skip(f"Galaxea A1 description not fetchable on this host: {exc}")
    assert path is not None
    meshes = re.findall(r'filename="([^"]+)"', path.read_text(encoding="utf-8"))
    assert meshes
    assert all((path.parent / m).is_file() for m in meshes)


def test_robotwin_rejects_unknown_model_and_kind() -> None:
    """``robotwin:`` serves only aloha-agilex as a URDF / MJCF; anything else fails pre-fetch."""
    for ref, kind in (("robotwin:piper", "urdf"), ("robotwin:aloha_agilex", "srdf")):
        with pytest.raises(AssetRefError, match="only robotwin:aloha_agilex") as info:
            resolve_asset(ref, kind)  # type: ignore[arg-type]  # reason: AssetKind literal from a loop
        assert not isinstance(info.value, AssetFetchError)


def test_robotwin_aloha_agilex_urdf_meshes_resolve_beside_it() -> None:
    """Every mesh the fetched URDF names (bar its commented-out xacro includes) is on disk."""
    try:
        path = resolve_asset("robotwin:aloha_agilex", "urdf")
    except AssetFetchError as exc:
        pytest.skip(f"RoboTwin embodiments archive not readable on this host: {exc}")
    assert path is not None
    xml = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
    meshes = re.findall(r'filename="([^"]+)"', xml)
    assert meshes
    assert all((path.parent / m).is_file() for m in meshes)


def test_robotwin_aloha_agilex_mjcf_actuates_the_manifest_joints() -> None:
    """The derived twin has one actuator per manifest joint, in order, at the manifest's range.

    Arm ranges come from the manifest (the URDF says ±10 rad); each gripper
    actuator spans the manifest ``sim.grippers`` ``ctrl_range`` (RoboTwin's
    ``gripper_scale``), with ``joint8`` mirroring ``joint7``. The ``home``
    keyframe the manifest starts from sits strictly inside every arm range.
    """
    mujoco = pytest.importorskip("mujoco")
    from openral_core import RobotDescription

    try:
        path = resolve_asset("robotwin:aloha_agilex", "mjcf")
    except AssetFetchError as exc:
        pytest.skip(f"RoboTwin embodiments archive not readable on this host: {exc}")
    assert path is not None
    model = mujoco.MjModel.from_xml_path(str(path))
    manifest = Path(__file__).resolve().parents[3] / "robots" / "aloha_agilex" / "robot.yaml"
    desc = RobotDescription.from_yaml(str(manifest))
    joints = desc.joints
    assert desc.sim is not None
    gripper_ranges = {g.joint: g.ctrl_range for g in desc.sim.grippers}
    assert model.nu == len(joints) == 14
    assert model.nq == 16  # 2 x (6 arm + 2 fingers); base, wheels and rear arms welded
    for i, joint in enumerate(joints):
        jid = model.actuator_trnid[i, 0]
        assert model.joint(jid).name == joint.sim_joint_name
        lo, hi = model.actuator_ctrlrange[i]
        if joint.role == "gripper":
            assert (lo, hi) == pytest.approx(gripper_ranges[joint.name])
        else:
            assert (lo, hi) == pytest.approx(joint.position_limits)
            assert tuple(model.jnt_range[jid]) == pytest.approx(joint.position_limits)
    assert desc.sim.keyframe_index is not None
    home = model.key_qpos[desc.sim.keyframe_index]
    for joint in joints:
        if joint.role != "gripper":
            jid = model.actuator_trnid[joints.index(joint), 0]
            q = home[model.jnt_qposadr[jid]]
            assert joint.position_limits is not None
            lo, hi = joint.position_limits
            assert lo < q < hi, f"{joint.name} starts at {q}, not inside [{lo}, {hi}]"
    mimics = {
        (model.joint(model.eq_obj1id[e]).name, model.joint(model.eq_obj2id[e]).name)
        for e in range(model.neq)
    }
    assert mimics == {("fl_joint8", "fl_joint7"), ("fr_joint8", "fr_joint7")}


def test_robotwin_mjcf_cache_write_failure_is_an_asset_ref_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable cache surfaces as ``AssetRefError``, not a bare ``OSError``."""
    import os

    pytest.importorskip("mujoco")
    from openral_hal import _robotwin_aloha_agilex_assets as robotwin

    if os.geteuid() == 0:
        pytest.skip("root ignores directory write permissions")
    try:
        urdf = resolve_asset("robotwin:aloha_agilex", "urdf")
    except AssetFetchError as exc:
        pytest.skip(f"RoboTwin embodiments archive not readable on this host: {exc}")
    assert urdf is not None
    # A verified, read-only copy of the cache (symlinks, so nothing is re-downloaded).
    root = tmp_path / "robotwin_aloha_agilex" / robotwin.ROBOTWIN_ASSETS_REVISION[:12]
    root.mkdir(parents=True)
    for entry in urdf.parent.iterdir():
        if not entry.name.startswith("derived-"):
            (root / entry.name).symlink_to(entry)
    root.chmod(0o555)
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path))
    try:
        with pytest.raises(AssetRefError, match="Permission denied"):
            resolve_asset("robotwin:aloha_agilex", "mjcf")
    finally:
        root.chmod(0o755)


def test_robosuite_is_mjcf_only() -> None:
    """``robosuite:`` composes an MJCF; a URDF request fails before robosuite is touched."""
    with pytest.raises(AssetRefError, match="is an mjcf"):
        resolve_asset("robosuite:PandaOmron", "urdf")


def test_robosuite_panda_omron_has_the_manifest_sim_joints() -> None:
    """The composed PandaOmron carries every ``sim_joint_name`` panda_mobile declares."""
    mujoco = pytest.importorskip("mujoco")
    from openral_core import RobotDescription

    try:
        path = resolve_asset("robosuite:PandaOmron", "mjcf")
    except AssetRefError as exc:
        pytest.skip(f"robosuite >= 1.5 not installed: {exc}")
    assert path is not None
    model = mujoco.MjModel.from_xml_path(str(path))
    joints = {model.joint(i).name for i in range(model.njnt)}
    manifest = Path(__file__).resolve().parents[3] / "robots" / "panda_mobile" / "robot.yaml"
    declared = {j.sim_joint_name for j in RobotDescription.from_yaml(str(manifest)).joints}
    assert declared - {None} <= joints


def test_robotwin_cache_write_failure_is_not_a_fetch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable cache is a local error (``AssetRefError``), never the
    skippable ``AssetFetchError`` an unreachable upstream raises."""
    pytest.importorskip("huggingface_hub")
    if os.geteuid() == 0:
        pytest.skip("root ignores the read-only mode this test relies on")
    cache = tmp_path / "cache"
    cache.mkdir()
    cache.chmod(0o500)  # read-only: the fetcher cannot create its cache root
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(cache))
    try:
        with pytest.raises(AssetRefError) as info:
            resolve_asset("robotwin:aloha_agilex", "urdf")
        assert not isinstance(info.value, AssetFetchError)
    finally:
        cache.chmod(0o700)
