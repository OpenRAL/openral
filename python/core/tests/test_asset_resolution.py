"""All-robots asset resolution + URDF/MJCF/SRDF validity (ADR-0058 §6).

The user's explicit "test everything for all robots": every ``robots/*/robot.yaml``
is parametrized through :func:`openral_core.assets.resolve_asset` and its declared
assets are loaded with the real parser for their kind (``yourdfpy`` for URDF,
``mujoco`` for MJCF, the safety-kernel SRDF parser for SRDF).

Principled skips/xfails only — no faked passes:

* MJCF that needs an absent optional sim dep → ``pytest.skip`` (never faked).
* ``menagerie:`` refs are not yet wired (Task 1 YAGNI); ``widowx``'s MJCF
  therefore *must* raise :class:`AssetRefError`, which the test asserts rather
  than skipping (the honest outcome).
* h1, so100_follower, so101_follower left this table under ADR-0058 — each ships
  a vendored, joint-name-patched URDF (``robots/<id>/<id>.urdf``) whose joints
  match the manifest, so all three PASS the cross-check with no safety-lowering
  drift. h1's ``package://h1_description`` meshes resolve location-independently;
  so100/so101 use *relative* mesh paths, so their Apache-2.0 mesh assets are
  vendored alongside under ``robots/<id>/assets/`` (with the upstream LICENSE) —
  the lowering re-fits identically from the vendored meshes.
* gr1 still :data:`xfail`: its upstream (Wiki-GRx-Models) URDF is **GPL-3.0**, a
  copy-left license that CLAUDE.md §1.9 rejects from open-core without TSC
  review, so it cannot be vendored into the repo. It keeps its ``_joint``-suffix
  xfail and its ``rd:`` ref.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_core.assets import AssetRefError, resolve_asset
from openral_core.schemas import RobotDescription

# Anchor ``robots/`` to the repo root so the suite is cwd-independent. This file
# is python/core/tests/test_asset_resolution.py → parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = sorted((_REPO_ROOT / "robots").glob("*/robot.yaml"))

# Robots whose declared URDF uses joint names that diverge from the manifest's
# HAL/control-contract names. h1, so100_follower and so101_follower left this
# table under ADR-0058 — each ships a vendored, joint-name-patched URDF whose
# joints match the manifest (so100/so101 vendor their Apache-2.0 meshes too).
# Only gr1 remains, for a documented, auditable reason:
#  * gr1 — upstream URDF is GPL-3.0, copy-left, rejected from open-core (§1.9).
_URDF_JOINT_NAME_MISMATCH: dict[str, str] = {
    "gr1": "rd:gr1_description URDF suffixes every joint with '_joint' "
    "(waist_yaw_joint); manifest drops the suffix (waist_yaw). Not vendorable: "
    "the Wiki-GRx-Models upstream is GPL-3.0, copy-left (CLAUDE.md §1.9).",
}


def _load(mf: Path) -> RobotDescription:
    return RobotDescription.model_validate(yaml.safe_load(mf.read_text()))


def test_manifest_glob_is_nonempty() -> None:
    """Guard against a silent zero-parametrization (wrong cwd / moved robots/)."""
    assert MANIFESTS, f"no robots/*/robot.yaml under {_REPO_ROOT}"
    assert len(MANIFESTS) == 17, f"expected 17 robots, found {len(MANIFESTS)}"


@pytest.mark.parametrize("mf", MANIFESTS, ids=lambda p: p.parent.name)
def test_declared_assets_resolve_to_files(mf: Path) -> None:
    """Every declared static asset ref resolves to an on-disk file.

    ``ros2://robot_description`` is the runtime-supplied marker (no file), and
    ``widowx``'s ``menagerie:`` MJCF is intentionally not wired yet — both are
    handled explicitly rather than asserted to be files.
    """
    a = _load(mf).assets
    if a.urdf and a.urdf.ref != "ros2://robot_description":
        p = resolve_asset(a.urdf.ref, "urdf", manifest_dir=mf.parent)
        assert p is not None and p.is_file()
    if a.mjcf:
        if a.mjcf.startswith("menagerie:"):
            # Task 1 YAGNI: menagerie wiring is deferred; the resolver raises.
            with pytest.raises(AssetRefError):
                resolve_asset(a.mjcf, "mjcf", manifest_dir=mf.parent)
        else:
            try:
                p = resolve_asset(a.mjcf, "mjcf", manifest_dir=mf.parent)
            except AssetRefError as exc:
                pytest.skip(f"optional sim dep absent for {a.mjcf}: {exc}")
            assert p is not None and p.is_file()
    if a.srdf:
        p = resolve_asset(a.srdf, "srdf", manifest_dir=mf.parent)
        assert p is not None and p.is_file()


@pytest.mark.parametrize("mf", MANIFESTS, ids=lambda p: p.parent.name)
def test_declared_urdf_parses_and_matches_hal_joints(mf: Path) -> None:
    """A declared static URDF parses and contains the manifest's actuated joints.

    The check is narrowed to non-gripper / non-base joints: gripper and virtual
    base DoFs (``base_x``/``base_y``/``base_yaw``) are part of the HAL contract
    but are deliberately absent from the arm URDF. Robots whose upstream URDF
    uses an entirely different joint-naming convention are :data:`xfail`-ed with
    a documented reason (see :data:`_URDF_JOINT_NAME_MISMATCH`).
    """
    pytest.importorskip("yourdfpy")
    import yourdfpy

    d = _load(mf)
    if not d.assets.urdf or d.assets.urdf.ref == "ros2://robot_description":
        pytest.skip("no static urdf")

    reason = _URDF_JOINT_NAME_MISMATCH.get(mf.parent.name)
    if reason is not None:
        pytest.xfail(reason)

    p = resolve_asset(d.assets.urdf.ref, "urdf", manifest_dir=mf.parent)
    assert p is not None
    model = yourdfpy.URDF.load(str(p))
    urdf_joints = set(model.joint_map)
    # Grippers/base DoFs are HAL-contract joints, not arm-URDF joints.
    contract_joints = {j.name for j in d.joints if j.role not in ("gripper", "base")}
    missing = contract_joints - urdf_joints
    assert not missing, f"{mf.parent.name}: manifest joints {missing} not in URDF"


@pytest.mark.parametrize("mf", MANIFESTS, ids=lambda p: p.parent.name)
def test_declared_mjcf_loads(mf: Path) -> None:
    """A declared MJCF compiles under MuJoCo (raises on malformed XML).

    ``widowx``'s ``menagerie:`` ref is not yet wired (Task 1 YAGNI), so the
    resolver raises before MuJoCo is ever invoked — asserted, not skipped.
    Other MJCFs whose optional sim package is absent are skipped honestly.
    """
    pytest.importorskip("mujoco")
    import mujoco

    a = _load(mf).assets
    if not a.mjcf:
        pytest.skip("no mjcf")

    if a.mjcf.startswith("menagerie:"):
        with pytest.raises(AssetRefError):
            resolve_asset(a.mjcf, "mjcf", manifest_dir=mf.parent)
        return

    try:
        path = resolve_asset(a.mjcf, "mjcf", manifest_dir=mf.parent)
    except AssetRefError as exc:
        pytest.skip(f"optional sim dep absent for {a.mjcf}: {exc}")
    assert path is not None
    mujoco.MjModel.from_xml_path(str(path))  # raises on malformed


@pytest.mark.parametrize("mf", MANIFESTS, ids=lambda p: p.parent.name)
def test_declared_srdf_parses(mf: Path) -> None:
    """A declared SRDF parses into a set of disabled-collision link pairs."""
    a = _load(mf).assets
    if not a.srdf:
        pytest.skip("no srdf")
    from openral_safety.urdf_lowering import parse_srdf_disabled_pairs

    p = resolve_asset(a.srdf, "srdf", manifest_dir=mf.parent)
    assert p is not None
    pairs = parse_srdf_disabled_pairs(str(p))
    assert isinstance(pairs, (list, set))
    assert pairs, f"{mf.parent.name}: SRDF declared but yielded no disabled pairs"


def test_urdf_less_robots_declare_no_urdf() -> None:
    """The sim-only robots ship no URDF — derived from manifests, not hardcoded."""
    less = {mf.parent.name for mf in MANIFESTS if _load(mf).assets.urdf is None}
    assert {
        "aloha_bimanual",
        "sawyer",
        "widowx",
        "google_robot",
        "pusht_2d",
    } <= less


def test_grammar_validator_rejects_legacy_form() -> None:
    """The pre-ADR-0058 ``robot_descriptions:`` ref form is rejected at validation."""
    from openral_core.schemas import UrdfAsset
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UrdfAsset(ref="robot_descriptions:ur5e_description")
