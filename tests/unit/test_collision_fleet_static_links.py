"""Fleet audit: every URDF collision body is in the kernel model or listed here with a reason.

Issue #356: the OpenArm manifest lowered the arms and not the torso they bolt
to, so the C++ safety kernel could not see a hand swung into the body. The
torso is in the model now; this keeps the gap from coming back on any robot,
and names the robots that still have one.

For every manifest whose URDF resolves on this host, the URDF links carrying a
``<collision>`` element are diffed against the links the manifest's collision
tree names (``joints`` and ``fixed_attachments``). A link the tree never names
cannot carry ``collision_geometry`` (the loader refuses orphan geometry), so
the kernel never checks it. Every such link must have a row in ``KNOWN_GAPS``
with its reason; a new one fails, and so does a stale row once the link joins
the tree (the remedy on both lowering paths is one ``fixed_attachments`` row,
then ``openral collision lower``). Naming aliases (the OpenArm manifest uses
its MJCF's link names) and gripper-only links are rows too, so the table is
the audit itself. Real manifests and URDFs only (CLAUDE.md §1.11); a robot
whose URDF cannot be fetched here skips.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.assets import AssetRefError, resolve_asset

_REPO = Path(__file__).resolve().parents[2]
_MANIFESTS = sorted(_REPO.glob("robots/*/robot.yaml"))

# URDF links with <collision> that the manifest's collision tree does not name,
# with the reason each is (still) outside the kernel model. Every entry is a
# known hole or an alias; adding one needs the same safety-WG eye as an ACM row.
KNOWN_GAPS: dict[str, dict[str, str]] = {
    "openarm": {
        # The manifest is MJCF-named: *_base_link is *_link0, the end-effector
        # base is link7 and the two finger links are the swept finger_pair.
        "openarm_left_base_link": "alias: manifest openarm_left_link0 (MJCF naming)",
        "openarm_right_base_link": "alias: manifest openarm_right_link0 (MJCF naming)",
        "openarm_left_ee_base_link": "alias: manifest openarm_left_link7",
        "openarm_right_ee_base_link": "alias: manifest openarm_right_link7",
        "openarm_left_ee_link1": "alias: swept into openarm_left_finger_pair",
        "openarm_left_ee_link2": "alias: swept into openarm_left_finger_pair",
        "openarm_right_ee_link1": "alias: swept into openarm_right_finger_pair",
        "openarm_right_ee_link2": "alias: swept into openarm_right_finger_pair",
    },
    "ur5e": {
        "base_link_inertia": "UR base body: not in the kernel model (#356 follow-up)",
    },
    "ur10e": {
        "base_link_inertia": "UR base body: not in the kernel model (#356 follow-up)",
    },
    "panda_mobile": {
        "panda_link0": "arm pedestal on the mobile base: not in the kernel model (#356 follow-up)",
        "panda_hand": "gripper-only link: the manifest models the arm (panda_link1-7)",
        "panda_leftfinger": "gripper-only link: the manifest models the arm (panda_link1-7)",
        "panda_rightfinger": "gripper-only link: the manifest models the arm (panda_link1-7)",
    },
    "panda_mobile_vslam": {
        "panda_link0": "arm pedestal on the mobile base: not in the kernel model (#356 follow-up)",
        "panda_hand": "gripper-only link: the manifest models the arm (panda_link1-7)",
        "panda_leftfinger": "gripper-only link: the manifest models the arm (panda_link1-7)",
        "panda_rightfinger": "gripper-only link: the manifest models the arm (panda_link1-7)",
    },
    "franka_panda": {
        "panda_leftfinger": "swept into panda_finger_pair",
        "panda_rightfinger": "swept into panda_finger_pair",
    },
    "g1": {
        "pelvis_contour_link": "pelvis body (the pelvis frame has no mesh): not in the model",
        "waist_support_link": "waist support: not in the kernel model (#356 follow-up)",
        "head_link": "head: not in the kernel model (#356 follow-up)",
        "logo_link": "decorative badge on the torso",
    },
    "h1": {
        "d435_left_imager_link": "torso-mounted sensor housing: not in the kernel model",
        "d435_rgb_module_link": "torso-mounted sensor housing: not in the kernel model",
        "mid360_link": "torso-mounted lidar housing: not in the kernel model",
    },
    "so101_follower": {
        "gripper": "alias: manifest gripper_base (hand-authored manifest, no geometry on it)",
        "jaw": "alias: manifest moving_jaw (hand-authored manifest, no geometry on it)",
    },
}


def _urdf_collision_links(urdf: Path) -> set[str]:
    root = ET.parse(urdf).getroot()
    return {
        link.attrib["name"]
        for link in root.iter("link")
        if link.find("collision") is not None and "name" in link.attrib
    }


@pytest.mark.parametrize("manifest", _MANIFESTS, ids=lambda p: p.parent.name)
def test_every_urdf_collision_body_is_in_the_kernel_model_or_listed(manifest: Path) -> None:
    robot = RobotDescription.from_yaml(str(manifest))
    name = manifest.parent.name
    if not robot.collision_geometry:
        # A robot with no collision model at all is a different gap (the kernel
        # runs its scalar envelope only); this audit is about links missing
        # from a model that exists.
        pytest.skip(f"{name}: no collision model to audit")
    if robot.assets.urdf is None:
        pytest.skip(f"{name}: no URDF asset to audit against")
    try:
        urdf = resolve_asset(robot.assets.urdf.ref, "urdf", manifest_dir=manifest.parent)
    except AssetRefError as exc:
        pytest.skip(f"{name}: URDF unavailable on this host: {exc}")
    if urdf is None:
        pytest.skip(f"{name}: URDF comes over a topic at runtime, nothing to audit offline")
    tree_links = (
        {j.parent_link for j in robot.joints}
        | {j.child_link for j in robot.joints}
        | {a.parent_link for a in robot.fixed_attachments}
        | {a.child_link for a in robot.fixed_attachments}
    )
    outside = _urdf_collision_links(urdf) - tree_links
    listed = KNOWN_GAPS.get(name, {})
    unlisted = sorted(outside - set(listed))
    assert not unlisted, (
        f"{name}: URDF links with <collision> that the kernel model cannot see: {unlisted}. "
        "Add a fixed_attachments row and re-run `openral collision lower`, or record the "
        "hole with its reason in KNOWN_GAPS."
    )
    stale = sorted(set(listed) - outside)
    assert not stale, f"{name}: KNOWN_GAPS rows no longer outside the model, remove them: {stale}"


def test_the_openarm_torso_is_no_longer_a_gap() -> None:
    """The concrete #356 fix: the body is in the tree, and no row excuses it."""
    robot = RobotDescription.from_yaml(str(_REPO / "robots" / "openarm" / "robot.yaml"))
    assert any(a.child_link == "openarm_body_link0" for a in robot.fixed_attachments)
    assert "openarm_body_link0" not in KNOWN_GAPS["openarm"]
    assert sum(g.link_name == "openarm_body_link0" for g in robot.collision_geometry) == 3
