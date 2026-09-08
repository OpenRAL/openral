"""All-robots asset resolution + URDF/MJCF/SRDF validity.

Every ``robots/*/robot.yaml`` is parametrized through
``openral_core.assets.resolve_asset``; declared assets are loaded with the
real parser for their kind (``yourdfpy`` URDF, ``mujoco`` MJCF, the
safety-kernel SRDF parser).

Skip/xfail policy — no faked passes:

* MJCF needing an absent optional sim dep → ``pytest.skip``.
* ``menagerie:`` refs are unwired (Task 1 YAGNI); ``widowx``'s MJCF asserts
  ``AssetRefError`` rather than skipping.
* h1, so100_follower, so101_follower ship a vendored, joint-name-patched URDF
  matching the manifest (so100/so101 also vendor their Apache-2.0 meshes under
  ``robots/<id>/assets/`` with the upstream LICENSE) — no xfail needed.
* gr1 stays ``xfail``: upstream (Wiki-GRx-Models) URDF is GPL-3.0,
  copy-left, rejected from open-core without TSC review (CLAUDE.md §1.9).
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

# Robots whose URDF joint names diverge from the manifest's HAL contract.
# h1/so100/so101 left this table (vendored joint-patched URDF matches manifest).
# gr1 remains: upstream URDF is GPL-3.0, copy-left, rejected from open-core (§1.9).
_URDF_JOINT_NAME_MISMATCH: dict[str, str] = {
    "gr1": "rd:gr1_description URDF suffixes every joint with '_joint' "
    "(waist_yaw_joint); manifest drops the suffix (waist_yaw). Not vendorable: "
    "the Wiki-GRx-Models upstream is GPL-3.0, copy-left (CLAUDE.md §1.9).",
}


def _load(mf: Path) -> RobotDescription:
    return RobotDescription.model_validate(yaml.safe_load(mf.read_text()))


#: Lower bound, not an exact count — guards against the glob silently matching
#: zero manifests (wrong cwd, moved/renamed ``robots/``). Pinning an exact
#: count caused red CI on unrelated PRs when a robot was added; raise this
#: when a batch of robots lands.
_MIN_ROBOT_MANIFESTS = 21


def test_manifest_glob_is_nonempty() -> None:
    """Guard against a silent zero-parametrization (wrong cwd / moved robots/)."""
    assert MANIFESTS, f"no robots/*/robot.yaml under {_REPO_ROOT}"
    assert len(MANIFESTS) >= _MIN_ROBOT_MANIFESTS, (
        f"expected at least {_MIN_ROBOT_MANIFESTS} robots, found {len(MANIFESTS)} "
        f"under {_REPO_ROOT / 'robots'} — the glob is matching too few manifests."
    )


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
    uses an entirely different joint-naming convention are ``xfail``-ed with
    a documented reason (see ``_URDF_JOINT_NAME_MISMATCH``).
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
    """The legacy ``robot_descriptions:`` ref form is rejected at validation."""
    from openral_core.schemas import UrdfAsset
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UrdfAsset(ref="robot_descriptions:ur5e_description")
