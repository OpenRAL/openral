"""Station-scoped place-zone choice on the real OpenAMRobot 2.0 manifest.

The aisle declares one slot at six shelves in ``map`` (``s<k>_lower_slot_1``). The runner
names every zone the prompt matches as a candidate set (``zone_candidates_target_id``); the
tracker keeps the one whose box centre, re-expressed in the grid frame at the base's pose
(``zone_transform`` + ``rebase_box``), lies inside the robot's reach: the world map's coverage
ball the launch measures from the kernel's collision model (``_coverage_ball``). Docked at
shelf 3 -> ``s3``; midway between two shelves -> none; a fixed-base unique zone -> unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

_ROOT = Path(__file__).resolve().parents[2]
_LAUNCH = _ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"
_S = 1_000_000_000
_SPACING_M = 2.0  # shelf pitch along map x in this fixture


@pytest.fixture(scope="module")
def reach() -> tuple[tuple[float, float, float], float]:
    """The real coverage ball for openamrobot_v2 (what the launch hands the tracker)."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    pytest.importorskip("openral_safety", reason="the collision lowering lives in openral_safety")
    from openral_core import RobotDescription
    from openral_safety.envelope_loader import collision_params_from_description

    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_zone_choice", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_zone_choice"] = module
    spec.loader.exec_module(module)
    desc = RobotDescription.from_yaml(str(_ROOT / "robots/openamrobot_v2/robot.yaml"))
    base = set(desc.base_joints or [])
    ball = module._coverage_ball(
        collision_params_from_description(desc),
        [j.position_limits for j in desc.joints],
        [i for i, j in enumerate(desc.joints) if j.name in base],
        0.02,
    )
    assert ball.radius < _SPACING_M, "the fixture's shelves must be farther apart than reach"
    return (tuple(float(v) for v in ball.centre), float(ball.radius))  # type: ignore[return-value]


def _session(reach: tuple[tuple[float, float, float], float]) -> object:
    from openral_core import PlaceRegion, PlaceZone, Pose6D, RobotDescription
    from openral_world_state.object_tracker import TrackerSession

    s = TrackerSession(
        RobotDescription.from_yaml(str(_ROOT / "robots/openamrobot_v2/robot.yaml")),
        cell_m=0.02,
        grounding_wait_s=5.0,
    )
    (_, _, cz), _ = reach
    zones = []
    for k in range(1, 7):
        pose = Pose6D(xyz=(_SPACING_M * k, 0.0, cz), quat_xyzw=(0, 0, 0, 1), frame_id="map")
        box = PlaceRegion(frame_id="map", pose=pose, half_extents=(0.188, 0.062, 0.074))
        zones.append(PlaceZone(name=f"s{k}_lower_slot_1", aliases=("lower_1",), region=box))
    s.place_zones = {z.target_id: z for z in zones}
    s.zone_reach = reach
    s.grid_stamp_ns = 10 * _S
    return s


def _dock(s: object, base_xy: tuple[float, float]) -> None:
    from openral_core.geometry import homogeneous_from_quat_xyz

    base_in_map = homogeneous_from_quat_xyz((base_xy[0], base_xy[1], 0.0), (0.0, 0.0, 0.0, 1.0))

    def grid_from(frame: str, stamp_ns: int) -> NDArray[np.float64]:
        assert frame == "map"
        return np.asarray(np.linalg.inv(base_in_map), dtype=np.float64)

    s.zone_transform = grid_from  # type: ignore[attr-defined]


def _choose(s: object) -> str | None:
    from openral_world_state.object_tracker.session import zone_candidates_target_id

    zones = list(s.place_zones.values())  # type: ignore[attr-defined]
    target = zone_candidates_target_id(zones)
    region = s._zone_region(target, s.frame_id)  # type: ignore[attr-defined]
    return None if region is None else str(region.evidence_ref)


def test_docked_at_shelf_three_the_s3_slot_is_chosen(
    reach: tuple[tuple[float, float, float], float],
) -> None:
    s = _session(reach)
    (cx, cy, _), _ = reach
    _dock(s, (3 * _SPACING_M - cx, -cy))  # s3's slot sits at the reach ball's centre
    assert _choose(s) == "declared:s3_lower_slot_1"


def test_midway_between_two_shelves_no_slot_is_chosen(
    reach: tuple[tuple[float, float, float], float],
) -> None:
    s = _session(reach)
    (cx, cy, _), _ = reach
    _dock(s, (3.5 * _SPACING_M - cx, -cy))  # s3 and s4 both a half pitch from the centre
    assert _choose(s) is None  # both in reach (ambiguous) or both out: none, fail closed


def test_without_a_reach_no_candidate_set_resolves(
    reach: tuple[tuple[float, float, float], float],
) -> None:
    s = _session(reach)
    s.zone_reach = None  # type: ignore[attr-defined]
    (cx, cy, _), _ = reach
    _dock(s, (3 * _SPACING_M - cx, -cy))
    assert _choose(s) is None


def test_a_fixed_base_unique_zone_resolves_exactly_as_before() -> None:
    """``openarm_restock_shelf``'s base-frame zone, one name: no reach, no tf2, unchanged."""
    import yaml
    from openral_core import PlaceZone, RobotDescription
    from openral_world_state.object_tracker import TrackerSession
    from pydantic import TypeAdapter

    raw = yaml.safe_load(
        (_ROOT / "scenes/deploy/place_zones/openarm_restock_shelf.yaml").read_text()
    )
    zones = TypeAdapter(list[PlaceZone]).validate_python(raw["place_zones"])
    s = TrackerSession(
        RobotDescription.from_yaml(str(_ROOT / "robots/openarm/robot.yaml")),
        cell_m=0.02,
        grounding_wait_s=5.0,
    )
    s.place_zones = {z.target_id: z for z in zones}
    s.grid_stamp_ns = 10 * _S
    zone = next(z for z in zones if z.name == "upper_slot_2")
    region = s._zone_region(zone.target_id, zone.region.frame_id)
    assert region is not None and region.evidence_ref == "declared:upper_slot_2"
    assert region.pose == zone.region.pose


_AISLE = Path("/home/allopart/workspace/openral-pro-amr-aisle/scenes/deploy/amr_restock_aisle.yaml")


@pytest.mark.skipif(not _AISLE.exists(), reason="the private Pro aisle scene is not on this host")
def test_at_each_aisle_dock_the_slot_of_that_shelf_is_chosen(
    reach: tuple[tuple[float, float, float], float],
) -> None:
    """The real aisle scene: the base exactly at each ``nav_goals`` dock picks its own shelf's
    ``s<k>_lower_slot_1`` out of the six that share the alias ``lower_1``."""
    import warnings

    from openral_core import DeployScene, RobotDescription
    from openral_core.geometry import homogeneous_from_quat_xyz
    from openral_world_state.object_tracker import TrackerSession
    from openral_world_state.object_tracker.session import zone_candidates_target_id

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        scene = DeployScene.from_yaml(str(_AISLE))
    s = TrackerSession(
        RobotDescription.from_yaml(str(_ROOT / "robots/openamrobot_v2/robot.yaml")),
        cell_m=0.02,
        grounding_wait_s=5.0,
    )
    s.place_zones = {z.target_id: z for z in scene.place_zones}
    s.zone_reach = reach
    s.grid_stamp_ns = 10 * _S
    target = zone_candidates_target_id([z for z in scene.place_zones if "lower_1" in z.aliases])
    assert len(target.split("|")) == 6
    for goal in scene.nav_goals:
        base_in_map = homogeneous_from_quat_xyz(goal.pose.xyz, goal.pose.quat_xyzw)
        s.zone_transform = lambda frame, stamp_ns, t=base_in_map: np.linalg.inv(t)
        region = s._zone_region(target, s.frame_id)
        assert region is not None, goal.name
        assert region.evidence_ref == f"declared:{goal.name}_lower_slot_1"
