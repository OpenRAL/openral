# SPDX-License-Identifier: Apache-2.0
"""Every in-tree URDF must be reachable from its manifest's ``base_frame``.

``robot_state_publisher`` publishes the URDF's own link names. When a manifest's
``base_frame`` is not the URDF's root link, nothing on ``/tf`` ever publishes
``base_frame`` — and every consumer that looks it up fails *silently*:
``_sensor_wiring`` defaults each sensor's ``parent_frame`` to it, ``SlamMapBridge``
and ``WorldCloudBridge`` transform into it, and ``octomap_server`` /
``octomap_voxel_bridge`` map against it. The nodes stay up and report healthy
while dropping every frame, which is the worst shape this failure can take.

:class:`~openral_core.schemas.UrdfAsset` already carries the fix
(``root_frame`` + ``base_to_root_xyz_rpy``). These tests pin that a manifest
which *needs* it *declares* it, and that what it declares matches the URDF on
disk rather than a remembered number.

``root_frame`` names the link ``base_frame`` ATTACHES TO — not necessarily the
tree's root. That distinction is load-bearing: ``derive_robot_relative_height_band``
reads the same pair to place collision geometry, and the UR manifests point it
at ``base_link`` because ``joints`` lists only movable joints, so ``base_link``
is no joint's child and the ``base_frame`` chain alone cannot reach the arm.
Retargeting those to the URDF's ``world`` root made every UR collision volume
unplaceable and the height band refuse outright — verified by doing it and
watching `packages/openral_slam_bringup` fail, so this file asserts
reachability, NOT rootness.

A consequence worth knowing rather than enforcing: when ``root_frame`` is a
link the URDF already parents, that link ends up with two parents on /tf. For
the UR manifests both hops are identity and nothing consumes the URDF's
``world``, so the ambiguity is inert — and not worth trading a working height
band for.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from openral_core import RobotDescription

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOTS = _REPO_ROOT / "robots"


def _file_urdf_manifests() -> list[tuple[str, Path, Path]]:
    """Every manifest whose ``assets.urdf.ref`` is an in-tree ``file:`` URDF."""
    found: list[tuple[str, Path, Path]] = []
    for manifest in sorted(_ROBOTS.glob("*/robot.yaml")):
        raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        ref = ((raw.get("assets") or {}).get("urdf") or {}).get("ref", "")
        if not isinstance(ref, str) or not ref.startswith("file:"):
            continue
        urdf = manifest.parent / ref[len("file:") :]
        if urdf.is_file():
            found.append((manifest.parent.name, manifest, urdf))
    return found


def _urdf_links_and_root(urdf: Path) -> tuple[set[str], str]:
    """Every link name in the URDF, plus the one that is never a joint's child."""
    root = ET.parse(urdf).getroot()
    links = {name for el in root.findall("link") if (name := el.get("name"))}
    children = {
        child.get("link")
        for joint in root.findall("joint")
        if (child := joint.find("child")) is not None
    }
    roots = links - children
    assert len(roots) == 1, f"{urdf.name}: expected exactly one root link, got {roots}"
    return links, next(iter(roots))


_MANIFESTS = _file_urdf_manifests()


@pytest.mark.parametrize(
    ("robot_id", "manifest", "urdf"),
    _MANIFESTS,
    ids=[m[0] for m in _MANIFESTS],
)
def test_base_frame_reaches_the_urdf_root(robot_id: str, manifest: Path, urdf: Path) -> None:
    """`base_frame` is the URDF root, or a declared bridge connects it to one."""
    description = RobotDescription.model_validate(
        yaml.safe_load(manifest.read_text(encoding="utf-8"))
    )
    urdf_links, urdf_root = _urdf_links_and_root(urdf)
    asset = description.assets.urdf
    assert asset is not None  # reason: _file_urdf_manifests filtered on assets.urdf.ref

    if description.base_frame in urdf_links:
        # robot_state_publisher already emits this frame — it need not be the
        # ROOT (e.g. rizon4's `base_link` hangs off a `world` root). A bridge
        # here would hand an already-parented link a second parent.
        assert asset.root_frame is None, (
            f"{robot_id}: base_frame {description.base_frame!r} is already a link in "
            f"{urdf.name}, so robot_state_publisher publishes it; declaring root_frame "
            f"{asset.root_frame!r} on top gives that link two parents"
        )
        return

    assert asset.root_frame is not None, (
        f"{robot_id}: base_frame {description.base_frame!r} is not a link in "
        f"{urdf.name} and no root_frame bridge is declared — nothing will ever "
        f"publish {description.base_frame!r} on /tf"
    )
    assert asset.root_frame in urdf_links, (
        f"{robot_id}: root_frame {asset.root_frame!r} is not a link in {urdf.name} "
        f"(links include {urdf_root!r}). The bridge must land on a real link, or "
        f"nothing connects base_frame to the robot's geometry."
    )
    assert asset.base_to_root_xyz_rpy is not None, (
        f"{robot_id}: root_frame is declared without base_to_root_xyz_rpy, so the "
        f"static_transform_publisher is never spawned"
    )


def test_openarm_bridge_matches_the_offset_in_its_urdf() -> None:
    """The OpenArm bridge's 0.698 m is the URDF's own pedestal height, not a memory.

    `openarm_base` (this manifest's frame, where the arm bases sit at z=0) is the
    URDF's `left_base_link` height; the URDF puts that 0.698 m above `body_link0`,
    which is identity to the root `world`. So the root expressed in `openarm_base`
    is -0.698 in z.
    """
    manifest = _ROBOTS / "openarm" / "robot.yaml"
    description = RobotDescription.model_validate(
        yaml.safe_load(manifest.read_text(encoding="utf-8"))
    )
    root = ET.parse(_ROBOTS / "openarm" / "openarm.urdf").getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}

    arm_mount = joints["left_base_link_mount_joint"]
    assert arm_mount.find("parent").get("link") == "body_link0"
    pedestal_z = float(arm_mount.find("origin").get("xyz").split()[2])

    body_mount = joints["body_link0_mount_joint"]
    assert body_mount.find("parent").get("link") == "world"
    assert [float(v) for v in body_mount.find("origin").get("xyz").split()] == [0.0, 0.0, 0.0]

    asset = description.assets.urdf
    assert asset is not None
    assert asset.base_to_root_xyz_rpy is not None
    x, y, z, roll, pitch, yaw = asset.base_to_root_xyz_rpy
    assert (x, y, roll, pitch, yaw) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert z == pytest.approx(-pedestal_z), (
        f"bridge z {z} should be the negative of the URDF pedestal height {pedestal_z}"
    )
