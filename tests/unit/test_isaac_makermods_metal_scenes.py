"""The two MakerMods Metal Isaac example scenes: options, rubrics, generated assets, mounts.

Fixtures: ``scenes/sim/isaac_makermods_metal_pen_holder.yaml``,
``scenes/sim/isaac_makermods_metal_pick_place.yaml``, their assets under
``scenes/sim/assets/makermods_metal_{pen_holder,pick_place}/`` (written by
``tools/make_metal_isaac_usd.py`` from the MuJoCo twins' options) and
``robots/makermods_metal{,_station}``. Host side only: no Isaac, no weights.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from openral_core import RobotDescription, SimScene, load_scene_strict
from openral_core.assets import AssetFetchError
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
def tools() -> tuple[ModuleType, ModuleType]:
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
    tools: tuple[ModuleType, ModuleType], offset: tuple[float, float, float], grade: str
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
    ("yaw_deg", "offset", "grade"),
    [
        # 40 mm out along the cup's turned x axis: past its 34.5 mm half width, though a
        # world-axis box (issue #561) holds the point.
        (45.0, (0.04 * math.cos(math.pi / 4), 0.04 * math.sin(math.pi / 4), 0.06), "fail"),
        (45.0, (0.04, 0.0, 0.06), "complete"),  # towards the turned box's corner
        (45.0, (0.0, 0.045, 0.06), "complete"),
        (45.0, (0.033, 0.033, 0.06), "fail"),  # world-box corner = the turned box's side, 47 mm
        (90.0, (0.03, -0.03, 0.06), "complete"),
        (45.0, (0.0, 0.06, 0.14), "partial"),  # on the turned rim box (45 mm half), 60 mm out
    ],
)
def test_pen_grades_against_a_turned_cup(
    tools: tuple[ModuleType, ModuleType],
    yaw_deg: float,
    offset: tuple[float, float, float],
    grade: str,
) -> None:
    scene_mod, _ = tools
    spec = _placement_success_spec(_options(_PEN).placement_success, _PEN_PROMPT)
    assert spec is not None
    cup = np.array([0.18, 0.0, 0.09])
    yaw = math.radians(yaw_deg)
    quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    at = scene_mod.placement_boxes_at(
        spec["placements"][0], {"pen_holder": cup}, {"pen_holder": quat}
    )
    up, level = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])
    got = scene_mod.placement_grade(cup + np.array(offset), level, up, at, 20.0, spec["grades"])
    assert got == grade


def test_pen_grades_against_a_tilted_cup(tools: tuple[ModuleType, ModuleType]) -> None:
    # The cup leans 30 deg about world x; the pen stands 90 mm up its axis: in the cup,
    # though 45 mm out along world y (past the world-axis box's 34.5 mm).
    scene_mod, _ = tools
    spec = _placement_success_spec(_options(_PEN).placement_success, _PEN_PROMPT)
    assert spec is not None
    cup = np.array([0.18, 0.0, 0.09])
    lean = math.radians(30.0)
    quat = np.array([math.cos(lean / 2), math.sin(lean / 2), 0.0, 0.0])
    pen = cup + np.array([0.0, -0.09 * math.sin(lean), 0.09 * math.cos(lean)])
    at = scene_mod.placement_boxes_at(
        spec["placements"][0], {"pen_holder": cup}, {"pen_holder": quat}
    )
    up, level = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])
    assert scene_mod.placement_grade(pen, level, up, at, 20.0, spec["grades"]) == "complete"


def test_only_unset_urdf_velocities_are_lifted(tools: tuple[ModuleType, ModuleType]) -> None:
    # SO-101: every joint sets velocity 10 and has a <transmission> naming it; the
    # transmission's <joint> reference is not a joint of its own.
    scene_mod, _ = tools
    urdf = (_REPO / "robots/so101_follower/so101_follower.urdf").read_text(encoding="utf-8")
    assert scene_mod.urdf_unset_velocity_joints(urdf) == set()


def test_recorded_pen_in_a_turned_tilted_cup_grades_complete(
    tools: tuple[ModuleType, ModuleType],
) -> None:
    # Episode 0's resting pen and cup in Isaac (q5090, 2026-11-12): the left arm left the
    # cup turned -63 deg and leaning 8.8 deg; in the cup's frame the pen is 25 mm out.
    scene_mod, _ = tools
    spec = _placement_success_spec(_options(_PEN).placement_success, _PEN_PROMPT)
    assert spec is not None
    cup = np.array([0.20190029, -0.02100779, 0.08630169])
    cup_q = np.array([0.84964603, -0.04911549, 0.05858206, -0.52178299])
    pen = np.array([0.23456274, -0.00597222, 0.16209197])
    pen_q = np.array([0.82338262, 0.14358030, -0.51826292, 0.18118894])
    at = scene_mod.placement_boxes_at(
        spec["placements"][0], {"pen_holder": cup}, {"pen_holder": cup_q}
    )
    up = scene_mod.object_up_axis(0.0, 0.0)
    assert (
        scene_mod.placement_grade(
            pen, pen_q, up, at, float(spec["upright_tol_deg"]), spec["grades"]
        )
        == "complete"
    )


@pytest.mark.parametrize(
    ("offset", "grade"),
    [
        ((0.0, 0.0), "complete"),  # centred on the sticker
        ((0.010, -0.008), "complete"),  # about half the footprint on it
        ((0.02, 0.0), "partial"),  # an edge on it
        ((0.022, 0.022), "partial"),  # a corner on it
        ((0.03, 0.03), "fail"),  # diagonally beside it: no overlap (centre 42 mm out)
        ((0.045, 0.0), "fail"),  # beside it
    ],
)
def test_cube_grades_against_the_marker(
    tools: tuple[ModuleType, ModuleType], offset: tuple[float, float], grade: str
) -> None:
    scene_mod, _ = tools
    spec = _placement_success_spec(_options(_CUBE).placement_success, _CUBE_PROMPT)
    assert spec is not None
    at = scene_mod.placement_boxes_at(spec["placements"][0], {})
    marker = np.array([0.278, 0.076, 0.0155])
    up, level = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])
    pos = marker + np.array([*offset, 0.0])
    assert scene_mod.placement_grade(pos, level, up, at, 20.0, spec["grades"]) == grade


def test_committed_assets_are_the_generators_output(tools: tuple[ModuleType, ModuleType]) -> None:
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
    for path in ("rtx/post/aa/op", "/physics/timeStepsPerSecond", "/app/runLoops/main/rate"):
        with pytest.raises(ValidationError, match="carb setting paths"):
            IsaacSimOptions(rtx_settings={path: 2})
    with pytest.raises(ValidationError, match="half_extents"):
        IsaacLinkBox(xyz=(0.0, 0.0, 0.0), half_extents=(0.01, 0.0, 0.01))
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValidationError, match="finite"):
            IsaacLinkBox(xyz=(0.0, bad, 0.0), half_extents=(0.01, 0.01, 0.01))
        with pytest.raises(ValidationError, match="finite"):
            IsaacLinkBox(xyz=(0.0, 0.0, 0.0), half_extents=(0.01, bad, 0.01))
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
        if not isinstance(exc.__cause__, AssetFetchError):
            raise  # a broken manifest or gripper convention is a failure, not a skip
        pytest.skip(f"metal_description unavailable: {exc}")
    grippers = {g["name"]: g for g in spec["grippers"]}
    for side in ("left", "right"):
        g = grippers[f"{side}_gripper"]
        assert g["leader"] == f"{side}_joint8"
        # joint8 = -joint7 in the URDF, so joint7 = -joint8 follows the driven joint8.
        assert g["followers"] == [{"dof": f"{side}_joint7", "multiplier": -1.0, "offset": -0.0}]


def test_link_box_colliders_leave_a_nested_child_link_colliding(
    tools: tuple[ModuleType, ModuleType],
) -> None:
    """Isaac 6 nests link bodies: boxing a parent link must not disable its child's colliders."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    scene_mod, _ = tools
    stage = Usd.Stage.CreateInMemory()
    for path in ("/robot/gripper_base", "/robot/gripper_base/link7"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, f"{path}/collision").GetPrim())
    box = {"xyz": [0.0, 0.0, 0.01], "half_extents": [0.01, 0.01, 0.01]}
    sidecar = scene_mod.IsaacManifestScene.__new__(scene_mod.IsaacManifestScene)
    sidecar._spec = {"link_box_colliders": {"gripper_base": box}}
    sidecar._set_link_box_colliders(stage, "/robot")

    def enabled(path: str) -> bool:
        attr = UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr()
        return bool(attr.Get()) if attr.HasAuthoredValue() else True

    assert not enabled("/robot/gripper_base/collision")
    assert enabled("/robot/gripper_base/link7/collision")
    assert enabled("/robot/gripper_base/openral_box_collider")
