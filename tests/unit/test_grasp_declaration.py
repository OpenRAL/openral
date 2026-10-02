"""``GraspDeclaration`` — the wire for the grasp-target exemption (design §2.1).

Field-for-field mirror of ``PlaceDeclaration``: dispatch names the target and
the gripper contact links, a producer measures the region, the kernel bounds it
(that consumer lands separately and defaults off). These tests pin the schema
bounds, the liveness rule on both clock domains, the scene-may-not-supply-a-
region rule against the real OpenArm cell scene, and the aggregator's atomic
storage beside the attachment set.

Real fixtures throughout (``scenes/deploy/``, ``robots/openarm/``) — CLAUDE.md
§1.11.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from openral_core import (
    DeployScene,
    GraspDeclaration,
    PlaceRegion,
    Pose6D,
    RobotDescription,
)
from openral_core.schemas import AttachedCollisionPrimitive, BoxShape
from openral_world_state import WorldStateAggregator
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPENARM_SCENE = _REPO_ROOT / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"
_OPENARM_ROBOT = _REPO_ROOT / "robots" / "openarm" / "robot.yaml"
_FINGERS = ("openarm_left_finger_pair", "openarm_right_finger_pair")
# Simulator time, the domain the runner stamps in under `use_sim_time`.
_SIM_NS = 1_240_000_000


def _region(*, half: tuple[float, float, float] = (0.06, 0.05, 0.04)) -> PlaceRegion:
    return PlaceRegion(
        frame_id="openarm_base",
        pose=Pose6D(xyz=(0.45, 0.0, 0.12), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="openarm_base"),
        half_extents=half,
        evidence_ref="zed_sam2_obb:restock_box",
        stamp_ns=_SIM_NS,
    )


def _declaration(stamp_ns: int = _SIM_NS, **update: Any) -> GraspDeclaration:
    fields: dict[str, Any] = {
        "target_id": "cell:restock_box",
        "contact_links": _FINGERS,
        "rskill_id": "openral/pi05-openarm-restock",
        "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
        "timeout_s": 70.0,
        "stamp_ns": stamp_ns,
    }
    fields.update(update)
    return GraspDeclaration(**fields)


# -- liveness ------------------------------------------------------------------


@pytest.mark.parametrize("stamp_ns", [_SIM_NS, time.time_ns()], ids=["sim_clock", "wall_clock"])
def test_is_live_inside_the_backstop_on_either_clock(stamp_ns: int) -> None:
    declaration = _declaration(stamp_ns)
    assert declaration.is_live(now_ns=stamp_ns)
    assert declaration.is_live(now_ns=stamp_ns + 70_000_000_000)
    assert not declaration.is_live(now_ns=stamp_ns + 70_000_000_001)  # expired
    assert not declaration.is_live(now_ns=stamp_ns - 1)  # future-stamped
    assert not declaration.model_copy(update={"active": False}).is_live(now_ns=stamp_ns)


def test_a_sim_stamped_declaration_is_dead_against_wall_time() -> None:
    """The clock-domain contract: wall `now` puts a sim stamp ~57 years past its backstop."""
    assert not _declaration(_SIM_NS).is_live(now_ns=time.time_ns())


# -- bounds --------------------------------------------------------------------


def test_timeout_is_capped() -> None:
    _declaration(timeout_s=GraspDeclaration.MAX_TIMEOUT_S)
    with pytest.raises(ValidationError, match="backstop ceiling"):
        _declaration(timeout_s=GraspDeclaration.MAX_TIMEOUT_S + 0.1)


def test_an_active_declaration_needs_target_and_contact_links() -> None:
    with pytest.raises(ValidationError, match="target_id"):
        _declaration(target_id="")
    with pytest.raises(ValidationError, match="contact_links"):
        _declaration(contact_links=())
    with pytest.raises(ValidationError, match="empty name"):
        _declaration(contact_links=("openarm_left_finger_pair", ""))
    # A retraction may be bare.
    assert not _declaration(target_id="", contact_links=(), active=False).active


def test_region_within_the_caps_is_accepted() -> None:
    assert _declaration(region=_region()).region is not None


def test_region_half_extent_over_the_cap_is_refused() -> None:
    with pytest.raises(ValidationError, match="half-extent"):
        _declaration(region=_region(half=(0.21, 0.05, 0.04)))


def test_region_volume_over_the_cap_is_refused() -> None:
    # Every side under 0.20 m, product 8 * 0.2 * 0.2 * 0.1 = 0.032 m^3 > 0.03.
    with pytest.raises(ValidationError, match="volume"):
        _declaration(region=_region(half=(0.20, 0.20, 0.10)))


def test_region_geometry_must_be_empty() -> None:
    region = _region().model_copy(
        update={
            "geometry": (
                AttachedCollisionPrimitive(
                    shape=BoxShape(half_extents_m=(0.05, 0.04, 0.03)),
                    pose_in_object=Pose6D(
                        xyz=(0.45, 0.0, 0.12),
                        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                        frame_id="openarm_base",
                    ),
                ),
            )
        }
    )
    with pytest.raises(ValidationError, match="geometry must be empty"):
        GraspDeclaration.model_validate(_declaration().model_dump() | {"region": region})


# -- the real scene fixture ----------------------------------------------------


def _scene_dict() -> dict[str, Any]:
    loaded = yaml.safe_load(_OPENARM_SCENE.read_text())
    assert isinstance(loaded, dict)
    return copy.deepcopy(loaded)


def test_the_openarm_cell_scene_declares_its_grasp_target() -> None:
    scene = DeployScene.from_yaml(str(_OPENARM_SCENE))
    declaration = scene.grasp_declaration
    assert declaration is not None
    assert declaration.target_id == "cell:restock_box"
    assert declaration.contact_links == _FINGERS
    assert declaration.timeout_s == 70.0
    assert declaration.region is None
    # Every named contact link is a real link of the robot this scene drives.
    robot = RobotDescription.from_yaml(str(_OPENARM_ROBOT))
    links = {j.child_link for j in robot.joints} | {j.parent_link for j in robot.joints}
    assert set(declaration.contact_links) <= links
    # The scene round-trips with the field intact.
    assert DeployScene.model_validate_json(scene.model_dump_json()) == scene


def test_a_scene_may_not_supply_the_grasp_region() -> None:
    raw = _scene_dict()
    raw["grasp_declaration"]["region"] = _region().model_dump(mode="json")
    with pytest.raises(ValidationError, match=r"grasp_declaration\.region is producer-supplied"):
        DeployScene.model_validate(raw)


def test_a_scene_may_supply_a_search_box_which_only_seeds_perception() -> None:
    raw = _scene_dict()
    raw["grasp_declaration"]["search_box"] = _region(half=(0.15, 0.15, 0.10)).model_dump(
        mode="json"
    )
    scene = DeployScene.model_validate(raw)
    assert scene.grasp_declaration is not None
    assert scene.grasp_declaration.search_box is not None
    assert scene.grasp_declaration.region is None


def test_search_box_geometry_must_be_empty() -> None:
    box = _region().model_copy(
        update={
            "geometry": (
                AttachedCollisionPrimitive(
                    shape=BoxShape(half_extents_m=(0.05, 0.04, 0.03)),
                    pose_in_object=Pose6D(
                        xyz=(0.45, 0.0, 0.12),
                        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                        frame_id="openarm_base",
                    ),
                ),
            )
        }
    )
    with pytest.raises(ValidationError, match=r"search_box\.geometry must be empty"):
        GraspDeclaration.model_validate(_declaration().model_dump() | {"search_box": box})


# -- aggregator ----------------------------------------------------------------


def _aggregator() -> WorldStateAggregator:
    # Production construction: no clock_fn, so the aggregator's own clock is wall
    # time while the stream is stamped in sim time — the mismatch that matters.
    return WorldStateAggregator(RobotDescription.from_yaml(str(_OPENARM_ROBOT)))


def test_aggregator_stores_the_grasp_declaration_with_the_set() -> None:
    aggregator = _aggregator()
    declaration = _declaration(region=_region())
    aggregator.update_attached_objects(
        [], revision=1, stamp_ns=_SIM_NS, grasp_declaration=declaration
    )
    snapshot = aggregator.snapshot()
    assert snapshot.grasp_declaration == declaration
    assert snapshot.attachment_stamp_ns == _SIM_NS


def test_aggregator_rejects_a_dead_grasp_declaration() -> None:
    aggregator = _aggregator()
    retracted = _declaration(active=False)
    aggregator.update_attached_objects(
        [], revision=1, stamp_ns=_SIM_NS, grasp_declaration=retracted
    )
    assert aggregator.snapshot().grasp_declaration is None

    expired = _declaration(0)  # 70 s backstop, judged 70 s + 1 ns later
    aggregator.update_attached_objects(
        [], revision=2, stamp_ns=70_000_000_001, grasp_declaration=expired
    )
    assert aggregator.snapshot().grasp_declaration is None

    future = _declaration(_SIM_NS + 1)
    aggregator.update_attached_objects([], revision=3, stamp_ns=_SIM_NS, grasp_declaration=future)
    assert aggregator.snapshot().grasp_declaration is None


def test_aggregator_clears_on_a_set_that_carries_none() -> None:
    aggregator = _aggregator()
    aggregator.update_attached_objects(
        [], revision=1, stamp_ns=_SIM_NS, grasp_declaration=_declaration()
    )
    assert aggregator.snapshot().grasp_declaration is not None
    aggregator.update_attached_objects([], revision=2, stamp_ns=_SIM_NS + 1)
    assert aggregator.snapshot().grasp_declaration is None


def test_the_cli_forwards_the_scenes_grasp_declaration_to_the_runner() -> None:
    """`deploy sim` (the twin pass of this scene) hands the runner the scene's block."""
    from openral_cli.deploy_sim import resolve_launch_invocation

    inv = resolve_launch_invocation(
        config=_OPENARM_SCENE,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="sim",
    )
    forwarded = [a for a in inv.argv_template if a.startswith("grasp_declaration_json:=")]
    assert len(forwarded) == 1
    payload = forwarded[0].removeprefix("grasp_declaration_json:=")
    assert GraspDeclaration.model_validate_json(payload) == (
        DeployScene.from_yaml(str(_OPENARM_SCENE)).grasp_declaration
    )
