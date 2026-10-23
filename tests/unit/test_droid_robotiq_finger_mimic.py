"""DROID Robotiq 2F-85 linkage: mimic links posed from the jaw fraction, fitted by the tool.

``robots/droid_franka_robotiq`` models the eight linkage links (outer knuckle, outer
finger ``bar``, inner knuckle ``proximal_phalanx``, inner finger + pad
``distal_phalanx``, per side) as ``mimic_joints`` rows on the normalised
``finger_joint`` fraction f, composing the Isaac scenes' ``gripper_joint_curve``
(URDF ``finger_joint = pi/4 * (1 - f)``) with each URDF ``<mimic>`` multiplier. The
URDF lowering does not emit mimic links, so the manifest's geometry is hand-spliced
and the fleet drift guard checks its ACM only; this file pins the rest:

- every ``collision_geometry`` entry is byte-identical to the URDF fitter's output
  (``lower_link_geometry``, box + exact hull) for its link;
- the kernel's forward kinematics (``collision_params_from_description``) places
  every link where the URDF does at the curve's ``finger_joint``, over the stroke;
- the curve the rows compose is the deploy scene's;
- no non-allowed pair trips at the scene start pose, closed, half and open;
- ``grasp_contact_links`` carries the mimic links.

Real manifest, URDF and meshes (CLAUDE.md §1.11).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml

pytest.importorskip("yourdfpy")
pytest.importorskip("trimesh")
pytest.importorskip("robot_descriptions")

from openral_cli.collision import render_blocks
from openral_core import RobotDescription
from openral_core.schemas import LinkCollisionGeometry
from openral_safety.envelope_loader import collision_params_from_description
from openral_safety.urdf_lowering import LoweredCollisionModel, _load_urdf, lower_link_geometry

from tests.unit.test_collision_contact_sweep import _kernel_fk
from tests.unit.test_collision_geometry_zero_pose import _tripping_pairs

_REPO = Path(__file__).resolve().parents[2]
_DIR = _REPO / "robots" / "droid_franka_robotiq"
_URDF = str(_DIR / "droid_franka_robotiq.urdf")
_SCENE = _REPO / "scenes" / "deploy" / "isaac_arena_droid_banana_in_bowl.yaml"
_CLOSED_RAD = math.pi / 4  # gripper_joint_curve [[0, pi/4], [1, 0]]
_START = (0.0, -math.pi / 5, 0.0, -4 * math.pi / 5, 0.0, 3 * math.pi / 5, 0.0)


@pytest.fixture(scope="module")
def robot() -> RobotDescription:
    return RobotDescription.from_yaml(str(_DIR / "robot.yaml"))


def _render(geoms: list[LinkCollisionGeometry]) -> str:
    return render_blocks(LoweredCollisionModel(collision_geometry=geoms))[0]


def test_every_geometry_entry_is_the_fitters_output(robot: RobotDescription) -> None:
    links = [g.link_name for g in robot.collision_geometry]
    fitted = lower_link_geometry(_URDF, tight_links=sorted(set(links)))
    by_link: dict[str, list[LinkCollisionGeometry]] = {}
    for g in fitted:
        by_link.setdefault(g.link_name, []).append(g)
    expected = [g for link in dict.fromkeys(links) for g in by_link[link]]
    assert _render(list(robot.collision_geometry)) == _render(expected)


def test_the_rows_compose_the_deploy_scenes_curve(robot: RobotDescription) -> None:
    curve = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))["scene"]["backend_options"][
        "gripper_joint_curve"
    ]["finger_joint"]
    assert sorted(map(tuple, curve)) == [(0.0, pytest.approx(_CLOSED_RAD)), (1.0, 0.0)]
    for m in robot.mimic_joints:
        sign = 1.0 if "distal" in m.child_link else -1.0  # URDF <mimic> multiplier -1 / +1
        assert m.value_at(0.0) == pytest.approx(-sign * _CLOSED_RAD, abs=1e-7)
        assert m.value_at(1.0) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_kernel_fk_matches_the_urdf_over_the_stroke(
    robot: RobotDescription, fraction: float
) -> None:
    params = collision_params_from_description(robot)
    names = cast("list[str]", params["collision_link_names"])
    kernel = _kernel_fk(params, np.array([*_START, fraction]))
    model = _load_urdf(_URDF)
    cfg = {f"panda_joint{i + 1}": float(v) for i, v in enumerate(_START)}
    cfg["finger_joint"] = _CLOSED_RAD * (1.0 - fraction)
    model.update_cfg(cfg)
    base = np.linalg.inv(model.get_transform("panda_link0"))
    geom_links = {g.link_name for g in robot.collision_geometry}
    for i, name in enumerate(names):
        if name in geom_links:
            err = float(np.abs(base @ model.get_transform(name) - kernel[i]).max())
            assert err < 1e-6, f"{name} at f={fraction}: kernel vs URDF {err:.2e}"


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_nothing_trips_at_the_start_pose(robot: RobotDescription, fraction: float) -> None:
    q = {f"panda_joint{i + 1}": v for i, v in enumerate(_START)} | {"finger_joint": fraction}
    assert _tripping_pairs(robot, q) == []


def test_grasp_contact_links_carry_the_linkage(robot: RobotDescription) -> None:
    contact = set(robot.grasp_contact_links())
    assert {m.child_link for m in robot.mimic_joints} <= contact
    assert len(robot.mimic_joints) == 6


def test_the_outer_fingers_are_grasp_contact_and_hand_links(robot: RobotDescription) -> None:
    """The bars are rigid on the outer knuckles (mimic links), so they are the hand's too."""
    contact = robot.grasp_contact_links()
    assert contact[-2:] == ("left_bar", "right_bar")
    assert {"left_bar", "right_bar"} <= set(robot.hand_links())
    # The Robotiq base is welded to panda_link7 through fixed attachments (link8): that
    # flange link is the wrist (Spark R15, refused as `not_hand_link` 2.4 mm from the banana).
    assert robot.hand_links()[-2:] == ("robotiq_arg2f_base_link", "panda_link7")


@pytest.mark.parametrize(
    ("robot_id", "contact", "hand"),
    [
        (
            "openarm",
            (
                "openarm_left_finger_pair",
                "openarm_right_finger_pair",
                "openarm_left_finger_follower",
                "openarm_right_finger_follower",
            ),
            (
                "openarm_left_finger_pair",
                "openarm_right_finger_pair",
                "openarm_left_finger_follower",
                "openarm_right_finger_follower",
                "openarm_left_link7",
                "openarm_right_link7",
                "openarm_left_link6",
                "openarm_right_link6",
            ),
        ),
        # panda_hand is welded to panda_link7 (fixed attachments): that is its wrist.
        (
            "franka_panda",
            ("panda_finger_pair",),
            ("panda_finger_pair", "panda_hand", "panda_link7"),
        ),
        # The SO-101 jaw is a mimic link (no fixed child), its link `gripper` (URDF name).
        ("so101_follower", ("moving_jaw", "jaw"), ("moving_jaw", "jaw", "gripper", "wrist")),
    ],
)
def test_robots_without_a_fixed_child_on_a_mimic_finger_are_unchanged(
    robot_id: str, contact: tuple[str, ...], hand: tuple[str, ...]
) -> None:
    other = RobotDescription.from_yaml(str(_REPO / "robots" / robot_id / "robot.yaml"))
    assert other.grasp_contact_links() == contact
    assert other.hand_links() == hand
