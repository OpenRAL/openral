# SPDX-License-Identifier: Apache-2.0
"""One wire naming convention for OpenArm: upstream's ``openarm_`` spelling.

The arm's own ros2_control graph, its ``/joint_states``, its MJCF and
``OpenArmRealHAL.ros2_control_joint_names()`` all say ``openarm_left_joint1``.
The vendored URDF used to say ``left_joint1`` — `openral robot vendor-urdf`
stripped the prefix to match the manifest's *logical* joint names — which made
it the only artefact in the repo with a different spelling for the same robot.

Two spellings break two things at once, both silently:

* ``joint_state_broadcaster``'s ``use_urdf_to_filter`` publishes only joints the
  robot description also names. With disjoint spellings the intersection is
  empty, so ``/joint_states`` carries ``name: []`` at the full controller rate
  while every controller reports active and the CAN bus is healthy.
* ``robot_state_publisher`` animates its tree by matching ``/joint_states``
  names against its URDF. No match means the whole arm sits at the rest pose in
  ``/tf`` — so anything mounted on an arm link is posed as if it never moved.

The manifest's logical joint names (``left_joint1``) are deliberately *not*
changed: they are the action-vector contract, shared with other robots and
pinned by published rSkill manifests. ``sim_joint_name`` and
``ros2_control_joint_names()`` are the declared bridges between the two
namespaces, and this test pins that they still line up.
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_URDF = _ROOT / "robots" / "openarm" / "openarm.urdf"
_MANIFEST = _ROOT / "robots" / "openarm" / "robot.yaml"


def _urdf_joint_names() -> set[str]:
    root = ET.parse(_URDF).getroot()
    return {j.get("name", "") for j in root.findall("joint")}


def test_every_urdf_joint_carries_the_upstream_prefix() -> None:
    """No `left_*` / `right_*` joint may survive in the vendored URDF."""
    stripped = sorted(n for n in _urdf_joint_names() if n.startswith(("left_", "right_")))
    assert not stripped, (
        f"{len(stripped)} joint(s) still use the stripped spelling, e.g. {stripped[:3]} — "
        "the arm's /joint_states says openarm_*, so these would never match"
    )


def test_the_urdf_names_the_joints_the_real_hal_reads() -> None:
    """The URDF must name every joint the transport matches on `/joint_states`.

    This is the pairing that actually failed on the cell: the HAL was reading
    `openarm_*` while the URDF declared `left_*`.
    """
    pytest.importorskip("openral_hal.openarm_real")
    from openral_hal.openarm_real import OpenArmRealHAL

    hal_names = set(OpenArmRealHAL(require_can_links=False).ros2_control_joint_names())
    missing = sorted(hal_names - _urdf_joint_names())
    assert not missing, f"URDF does not declare {missing} — robot_state_publisher cannot pose them"


def test_ros2_control_joints_match_the_urdf_joints() -> None:
    """`use_urdf_to_filter` intersects these two sets; an empty result is silent."""
    root = ET.parse(_URDF).getroot()
    control_joints = {
        j.get("name", "") for block in root.findall("ros2_control") for j in block.findall("joint")
    }
    assert control_joints, "no <ros2_control> joints found — the fixture moved"
    assert control_joints <= _urdf_joint_names(), (
        f"ros2_control declares joints the URDF does not: "
        f"{sorted(control_joints - _urdf_joint_names())}"
    )


def test_the_manifest_bridges_its_logical_names_to_the_wire_names() -> None:
    """`sim_joint_name` is the declared bridge and must reach real URDF joints.

    The logical names stay `left_joint1` on purpose (rSkill action contract), so
    the bridge is what has to hold — if it drifts, the two namespaces silently
    stop corresponding and nothing else would notice.
    """
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    urdf_joints = _urdf_joint_names()
    bridged = {
        j["name"]: j["sim_joint_name"] for j in manifest["joints"] if j.get("sim_joint_name")
    }
    assert bridged, "manifest declares no sim_joint_name bridges — fixture moved"
    unreachable = sorted(w for w in bridged.values() if w not in urdf_joints)
    assert not unreachable, f"sim_joint_name points at joints the URDF lacks: {unreachable}"
