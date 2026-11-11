"""The two MakerMods Metal Isaac example scenes: options, rubrics, generated assets, mounts.

Fixtures: ``scenes/sim/isaac_makermods_metal_pen_holder.yaml``,
``scenes/sim/isaac_makermods_metal_pick_place.yaml``, their assets under
``scenes/sim/assets/makermods_metal_{pen_holder,pick_place}/`` (written by
``tools/make_metal_isaac_usd.py`` from the MuJoCo twins' options) and
``robots/makermods_metal{,_station}``. Host side only: no Isaac, no weights.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from openral_core import RobotDescription, SimScene, load_scene_strict
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.isaac_sim import (
    IsaacLinkBox,
    IsaacSimOptions,
    _build_robot_spec,
    _environment_spec,
    _placement_success_spec,
    default_camera_mounts,
)
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[2]
_PEN = _REPO / "scenes/sim/isaac_makermods_metal_pen_holder.yaml"
_CUBE = _REPO / "scenes/sim/isaac_makermods_metal_pick_place.yaml"
_PEN_PROMPT = "Move the pen holder with the left arm and put the pen into it with the right arm"
_CUBE_PROMPT = "Pick up the cube and place it at the marked location"


@pytest.fixture(scope="module")
def tools() -> Any:
    path = str(_REPO / "tools")
    if path not in sys.path:
        sys.path.insert(0, path)
    import isaac_manifest_scene
    import make_metal_isaac_usd

    return isaac_manifest_scene, make_metal_isaac_usd


def _options(path: Path) -> IsaacSimOptions:
    return IsaacSimOptions.model_validate(
        load_scene_strict(str(path), SimScene).scene.backend_options
    )


def test_scenes_use_their_station_as_environment_and_the_twins_rubric_prompts() -> None:
    for path, robot, prompt, objects in (
        (_PEN, "makermods_metal_station", _PEN_PROMPT, ["pen_holder", "pen"]),
        (_CUBE, "makermods_metal", _CUBE_PROMPT, ["marker", "cube"]),
    ):
        scene = load_scene_strict(str(path), SimScene)
        assert scene.robot_id == robot and scene.task.instruction == prompt
        assert scene.scene.assets_uri is not None and scene.scene.assets_uri.endswith("table.usda")
        opts = _options(path)
        assert [o.name for o in opts.objects] == objects
        assert opts.rtx_settings["/rtx/post/aa/op"] == 2  # FXAA, not DLSS
        assert opts.gripper_followers_track == "measured"
        assert opts.placement_success is not None and opts.placement_success.require_released
        desc = RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml"))
        limits = {j.name: j.position_limits for j in desc.joints}
        for name, q in opts.initial_joint_positions.items():
            assert limits[name] is not None, name
            lo, hi = limits[name]  # type: ignore[misc]  # reason: asserted not None above
            assert lo <= q <= hi, (name, q)  # the sidecar refuses a start past a limit
        assert set(opts.link_box_colliders) <= {
            *(f"{p}link{k}" for p in ("", "left_", "right_") for k in (7, 8))
        }


@pytest.mark.parametrize(
    ("offset", "grade"),
    [
        ((0.0, 0.0, 0.06), "complete"),  # standing in the cup, centre below the rim
        ((0.02, -0.015, 0.05), "complete"),  # leaning inside
        ((0.0, 0.0, 0.14), "partial"),  # on the cup, centre above the rim
        ((0.0, 0.0, 0.25), "fail"),  # held above it
        ((0.06, 0.0, 0.01), "fail"),  # on the desk beside it
    ],
)
def test_pen_grades_against_the_cup(
    tools: Any, offset: tuple[float, float, float], grade: str
) -> None:
    scene_mod, _ = tools
    ps = _options(_PEN).placement_success
    spec = _placement_success_spec(ps, _PEN_PROMPT)
    assert spec is not None
    cup = np.array([0.18, 0.0, 0.09])
    at = scene_mod.placement_boxes_at(spec["placements"][0], {"pen_holder": cup})
    up, level = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])
    got = scene_mod.placement_grade(cup + np.array(offset), level, up, at, 20.0, spec["grades"])
    assert got == grade


@pytest.mark.parametrize(
    ("offset", "grade"),
    [
        ((0.0, 0.0), "complete"),  # centred on the sticker
        ((0.010, -0.008), "complete"),  # about half the footprint on it
        ((0.025, 0.0), "partial"),  # an edge on it
        ((0.045, 0.0), "fail"),  # beside it
    ],
)
def test_cube_grades_against_the_marker(
    tools: Any, offset: tuple[float, float], grade: str
) -> None:
    scene_mod, _ = tools
    spec = _placement_success_spec(_options(_CUBE).placement_success, _CUBE_PROMPT)
    assert spec is not None
    at = scene_mod.placement_boxes_at(spec["placements"][0], {})
    marker = np.array([0.278, 0.076, 0.0155])
    up, level = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])
    pos = marker + np.array([*offset, 0.0])
    assert scene_mod.placement_grade(pos, level, up, at, 20.0, spec["grades"]) == grade


def test_committed_assets_are_the_generators_output(tools: Any) -> None:
    from openral_sim.backends.metal_pen_holder._assets import MetalPenHolderOptions
    from openral_sim.backends.metal_pick_place._assets import MetalPickPlaceOptions

    _, gen = tools
    pen, cube = MetalPenHolderOptions(), MetalPickPlaceOptions()
    for path, text in (
        (gen.PEN_DIR / "table.usda", gen.pen_table_usda(pen)),
        (gen.PEN_DIR / "pen_holder.usda", gen.pen_holder_usda(pen)),
        (gen.PEN_DIR / "pen.usda", gen.pen_usda(pen)),
        (gen.CUBE_DIR / "table.usda", gen.cube_table_usda(cube)),
        (gen.CUBE_DIR / "cube.usda", gen.cube_usda(cube)),
        (gen.CUBE_DIR / "marker.usda", gen.marker_usda(cube)),
    ):
        assert path.read_text(encoding="utf-8") == text, path.name
        for ref in (line.split("@")[1] for line in text.splitlines() if "@" in line):
            assert (path.parent / ref).is_file(), ref  # every texture the USD names is committed
        assert path.stat().st_size < 2_000_000  # the repo's large-file limit
    # The cup's interior region fits inside its walls at the floor.
    region = _options(_PEN).placement_success.regions["in_holder"]  # type: ignore[union-attr]  # reason: asserted by the scene test
    inner = pen.holder_bottom_width_m / 2 - pen.holder_wall_m
    assert region.max_xyz[0] <= inner + 1e-9 and region.max_xyz[2] <= pen.holder_height_m + 1e-9


def test_new_scene_options_refuse_bad_values() -> None:
    with pytest.raises(ValidationError, match="carb setting paths"):
        IsaacSimOptions(rtx_settings={"rtx/post/aa/op": 2})
    with pytest.raises(ValidationError, match="half_extents"):
        IsaacLinkBox(xyz=(0.0, 0.0, 0.0), half_extents=(0.01, 0.0, 0.01))
    spec = _environment_spec(_options(_PEN))
    assert spec["rtx_settings"]["/rtx/post/aa/op"] == 2
    assert spec["link_box_colliders"]["left_link7"]["half_extents"] == [0.001, 0.009, 0.034]


def test_station_wrists_ride_the_grippers_and_the_top_keeps_its_static_mount() -> None:
    robot_dir = _REPO / "robots/makermods_metal_station"
    desc = RobotDescription.from_yaml(str(robot_dir / "robot.yaml"))
    mounts = default_camera_mounts(
        desc, robot_dir, {"station_base", "left_gripper_base", "right_gripper_base"}
    )
    for side in ("left", "right"):
        m = mounts[f"wrist_{side}"]  # sim_placement: the last-resort mount
        assert m.link == f"{side}_gripper_base" and m.axes == "usd"
    assert mounts["top"].link == "station_base" and mounts["top"].axes == "world"


def test_station_fingers_follow_the_driven_joint8() -> None:
    robot_dir = _REPO / "robots/makermods_metal_station"
    desc = RobotDescription.from_yaml(str(robot_dir / "robot.yaml"))
    try:
        spec = _build_robot_spec(desc, "makermods_metal_station", manifest_dir=robot_dir)
    except ROSConfigError as exc:  # the pinned metal_description clone needs network once
        pytest.skip(f"metal_description unavailable: {exc}")
    grippers = {g["name"]: g for g in spec["grippers"]}
    for side in ("left", "right"):
        g = grippers[f"{side}_gripper"]
        assert g["leader"] == f"{side}_joint8"
        # joint8 = -joint7 in the URDF, so joint7 = -joint8 follows the driven joint8.
        assert g["followers"] == [{"dof": f"{side}_joint7", "multiplier": -1.0, "offset": -0.0}]
