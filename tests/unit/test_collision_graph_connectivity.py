"""Every shipped robot's collision links must form exactly one connected tree.

``RobotDescription.joints`` enumerates only *movable* joints, so a robot with a
rigid mount — a Franka hand bolted to the flange, a bimanual rig's two arm
pedestals — has links that no joint reaches. The envelope loader used to treat
each such link as a second base and place it, and its whole subtree, at the
robot's origin. That is silently wrong in both directions: it fabricates
contacts that cannot happen, and (far worse) it leaves the subtree's real swept
volume completely unmodelled, so a genuine self-collision goes undetected.

These tests pin the invariant that replaced it — one root, or no collision model
at all — across every manifest in ``robots/``, so a future robot cannot
reintroduce the defect by omitting its rigid mounts.

CLAUDE.md §1.11 — real manifests from ``robots/``, real schemas, no mocks.
"""

from __future__ import annotations

import pathlib

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_safety.envelope_loader import collision_params_from_description

_ROBOTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "robots"


def _manifest_paths() -> list[pathlib.Path]:
    return sorted(p for p in _ROBOTS_DIR.glob("*/robot.yaml"))


def _ids() -> list[str]:
    return [p.parent.name for p in _manifest_paths()]


@pytest.mark.parametrize("manifest", _manifest_paths(), ids=_ids())
def test_shipped_manifest_lowers_to_a_single_rooted_tree(manifest: pathlib.Path) -> None:
    """No shipped manifest yields a disconnected collision graph.

    Either the robot declares no collision geometry (the kernel runs the scalar
    envelope check only), or its links form one tree with exactly one root.
    """
    robot = RobotDescription.from_yaml(str(manifest))
    params = collision_params_from_description(robot)
    if not params.get("self_collision_enabled"):
        return

    parent = params["collision_parent"]
    names = params["collision_link_names"]
    assert isinstance(parent, list)
    assert isinstance(names, list)
    roots = [names[i] for i, p in enumerate(parent) if p == -1]
    assert len(roots) == 1, (
        f"{manifest.parent.name}: collision links form {len(roots)} disconnected "
        f"trees (roots: {roots}); every link must be placeable relative to the "
        "base. Declare the missing rigid mounts in 'fixed_attachments'."
    )
    # Topological invariant the kernel's forward kinematics depends on.
    for child_idx, parent_idx in enumerate(parent):
        assert parent_idx < child_idx


@pytest.mark.parametrize("manifest", _manifest_paths(), ids=_ids())
def test_geometry_free_manifests_cannot_go_live_disconnected(manifest: pathlib.Path) -> None:
    """A manifest we cannot connect yet must stay geometry-free.

    ``r1pro`` ships no ``assets`` block at all, so there is no URDF or MJCF on
    disk to source its four missing mounts from — and inventing them is not an
    option. It is safe only because it declares no collision geometry, so the
    loader never builds a model for it. This asserts that pairing holds: the day
    someone adds geometry to a still-disconnected manifest, the loader refuses
    and this suite says why, rather than a wrong envelope shipping quietly.
    """
    robot = RobotDescription.from_yaml(str(manifest))
    if not robot.collision_geometry:
        return
    # Geometry is present → the tree must be complete. Raises otherwise.
    params = collision_params_from_description(robot)
    assert params["self_collision_enabled"] is True


def test_disconnected_graph_is_refused_not_guessed() -> None:
    """Dropping a real robot's rigid mount makes the loader refuse, loudly.

    Uses the real Franka manifest with its one ``fixed_attachments`` entry
    removed — exactly the state every shipped Franka manifest was in before this
    fix. The old loader lowered this silently, placing ``panda_hand`` inside the
    base.
    """
    robot = RobotDescription.from_yaml(str(_ROBOTS_DIR / "franka_panda" / "robot.yaml"))
    assert robot.fixed_attachments, "fixture precondition: the manifest declares a mount"
    orphaned = robot.model_copy(update={"fixed_attachments": []})

    with pytest.raises(ROSConfigError) as excinfo:
        collision_params_from_description(orphaned)

    message = str(excinfo.value)
    assert "panda_hand" in message, "the error must name the disconnected root"
    assert "panda_link0" in message
    assert "fixed_attachments" in message, "the error must say how to fix it"


def test_fixed_attachment_cannot_redefine_a_joint_child() -> None:
    """Two edges claiming one link make its pose ambiguous — rejected at load."""
    robot = RobotDescription.from_yaml(str(_ROBOTS_DIR / "franka_panda" / "robot.yaml"))
    clash = {
        "name": "bogus_remount",
        "parent_link": "panda_link0",
        "child_link": "panda_link1",  # already defined by panda_joint1
        "origin_xyz": (0.0, 0.0, 0.0),
        "origin_rpy": (0.0, 0.0, 0.0),
    }
    payload = robot.model_dump()
    payload["fixed_attachments"] = [*payload["fixed_attachments"], clash]

    with pytest.raises(ValueError, match="re-defines child_link"):
        RobotDescription.model_validate(payload)


def test_cyclic_chain_is_refused() -> None:
    """A cycle leaves links unreachable from the root; the loader says so."""
    robot = RobotDescription.from_yaml(str(_ROBOTS_DIR / "franka_panda" / "robot.yaml"))
    payload = robot.model_dump()
    # Re-parent the base onto a link further down the arm, closing a loop.
    payload["fixed_attachments"] = [
        *payload["fixed_attachments"],
        {
            "name": "loop_closure",
            "parent_link": "panda_link3",
            "child_link": "panda_link0",
            "origin_xyz": (0.0, 0.0, 0.0),
            "origin_rpy": (0.0, 0.0, 0.0),
        },
    ]
    looped = RobotDescription.model_validate(payload)

    with pytest.raises(ROSConfigError, match="cycle"):
        collision_params_from_description(looped)


def test_franka_hand_is_placed_on_the_flange_not_the_base() -> None:
    """The regression in numbers: the hand hangs off ``panda_link7``.

    Before the fix ``panda_hand``'s parent index was ``-1`` (a phantom second
    base at the origin). The gripper capsule therefore sat inside the robot's
    own pedestal instead of ~0.93 m up at the flange.
    """
    robot = RobotDescription.from_yaml(str(_ROBOTS_DIR / "franka_panda" / "robot.yaml"))
    params = collision_params_from_description(robot)
    names = params["collision_link_names"]
    parent = params["collision_parent"]
    assert isinstance(names, list)
    assert isinstance(parent, list)

    hand = names.index("panda_hand")
    assert parent[hand] == names.index("panda_link7")
    assert parent[hand] != -1

    origin = params["collision_origin_xyzrpy"]
    assert isinstance(origin, list)
    assert origin[6 * hand : 6 * hand + 3] == pytest.approx([0.0, 0.0, 0.107])
    assert origin[6 * hand + 3 : 6 * hand + 6] == pytest.approx([0.0, 0.0, -0.7853981633974483])


def test_openarm_arms_are_not_superimposed() -> None:
    """Both openarm pedestals hang off the base, 62 mm apart — not on top of it."""
    robot = RobotDescription.from_yaml(str(_ROBOTS_DIR / "openarm" / "robot.yaml"))
    params = collision_params_from_description(robot)
    names = params["collision_link_names"]
    parent = params["collision_parent"]
    origin = params["collision_origin_xyzrpy"]
    assert isinstance(names, list)
    assert isinstance(parent, list)
    assert isinstance(origin, list)

    base = names.index("openarm_base")
    left = names.index("openarm_left_link0")
    right = names.index("openarm_right_link0")
    assert parent[left] == base
    assert parent[right] == base
    assert origin[6 * left : 6 * left + 3] == pytest.approx([0.0, 0.031, 0.0])
    assert origin[6 * right : 6 * right + 3] == pytest.approx([0.0, -0.031, 0.0])


def test_g1_upper_body_hangs_off_the_waist_not_the_pelvis() -> None:
    """``torso_link`` is the waist-pitch child; the G1 has no ``waist_pitch_link``.

    The manifest used to name a link that exists in no real model file, which
    orphaned the torso and both arms — 15 links, 15 collision volumes — onto the
    pelvis origin.
    """
    robot = RobotDescription.from_yaml(str(_ROBOTS_DIR / "g1" / "robot.yaml"))
    link_names = {j.child_link for j in robot.joints} | {j.parent_link for j in robot.joints}
    assert "waist_pitch_link" not in link_names, "that link does not exist on a real G1"

    params = collision_params_from_description(robot)
    names = params["collision_link_names"]
    parent = params["collision_parent"]
    assert isinstance(names, list)
    assert isinstance(parent, list)
    assert parent[names.index("torso_link")] == names.index("waist_roll_link")
    # The arms reach the pelvis only through the waist, never directly.
    assert parent[names.index("left_shoulder_pitch_link")] == names.index("torso_link")
    assert parent[names.index("right_shoulder_pitch_link")] == names.index("torso_link")
