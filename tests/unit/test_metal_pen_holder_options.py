# SPDX-License-Identifier: Apache-2.0
"""``metal_pen_holder`` options, grading, suite and the ACT rSkill (host side, no render).

The scene speaks the makermods/pen_holder_bi_20260723_161553 dataset's units (degrees,
gripper stroke) and converts once per arm; these checks pin the 14-D conversion, the
cup-interior grading, the shipped scene and 10-layout suite, and the rSkill manifest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from openral_core import (
    RobotDescription,
    SimScene,
    load_benchmark_suite,
    raise_on_invalid_suite,
)
from openral_rskill.loader import load_rskill_manifest
from openral_sim import SCENES
from openral_sim.backends.metal_pen_holder._assets import (
    MetalPenHolderOptions,
    holder_hex_texture,
    holder_inner_half_width,
)
from openral_sim.backends.metal_pen_holder.env import pen_holder_grade, pen_points_in_holder

_ROOT = Path(__file__).resolve().parents[2]
_PROMPT = "Move the pen holder with the left arm and put the pen into it with the right arm"
_UPRIGHT = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # pen axis (x) up


def test_shipped_scene_validates_against_the_registered_options_model() -> None:
    sc = SimScene.from_yaml(str(_ROOT / "scenes" / "sim" / "makermods_metal_pen_holder.yaml"))
    opts = SCENES.validate_options(sc.scene.id, sc.scene.backend_options)
    assert isinstance(opts, MetalPenHolderOptions)
    assert (opts.joint_units, opts.gripper_scale, opts.top_background) == (
        "degrees",
        100.0,
        "plate",
    )
    assert sc.robot_id == "makermods_metal_station"
    assert sc.scene.cameras == ["top", "wrist_left", "wrist_right"]
    assert sc.task.instruction == _PROMPT
    assert SCENES.meta("metal_pen_holder")["converts_policy_units"] is True


def test_the_cup_widens_toward_its_rim() -> None:
    o = MetalPenHolderOptions()
    lo, hi = holder_inner_half_width(o, 0.0), holder_inner_half_width(o, o.holder_height_m)
    assert lo < hi
    assert hi == o.holder_top_width_m / 2 - o.holder_wall_m


def test_pen_grades_against_the_cup_interior() -> None:
    o, eye, cup = MetalPenHolderOptions(), np.eye(3), np.array([0.2, 0.0, 0.1])
    standing = pen_points_in_holder(cup + np.array([0, 0, 0.07]), _UPRIGHT, cup, eye, o)
    assert standing[0] and 0 < standing[1] < 1  # centre inside, the top sticks out
    beside = pen_points_in_holder(cup + np.array([0.15, 0, -0.09]), eye, cup, eye, o)
    assert beside == (False, 0.0)
    leaning = pen_points_in_holder(cup + np.array([0.0, 0.0, 0.14]), _UPRIGHT, cup, eye, o)
    assert not leaning[0] and leaning[1] > 0  # tip in, centre above the rim
    assert pen_holder_grade(*standing, lifted=True) == "complete"
    assert pen_holder_grade(*leaning, lifted=True) == "partial"
    assert pen_holder_grade(*beside, lifted=True) == "semi"
    assert pen_holder_grade(*beside, lifted=False) == "fail"


def test_the_grading_follows_the_cup_frame() -> None:
    o = MetalPenHolderOptions()
    tipped = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # cup on its side
    cup = np.zeros(3)
    # Upright in the world but across the lying cup's axis: not inside it.
    assert pen_points_in_holder(np.array([0, 0, 0.05]), _UPRIGHT, cup, tipped, o)[0] is False


def test_hex_texture_is_white_with_grey_holes() -> None:
    img = holder_hex_texture(128)
    assert img.shape == (128, 128, 3)
    assert (img[:10] == 255).all()  # the solid band at the rim
    assert img.min() < 200 and (img == 255).mean() > 0.3


def test_suite_is_ten_layouts_inside_the_dataset_spread() -> None:
    scenes = load_benchmark_suite(str(_ROOT / "benchmarks" / "makermods_metal_pen_holder.yaml"))
    raise_on_invalid_suite(scenes, suite_id="makermods_metal_pen_holder")
    assert len(scenes) == 10
    starts = set()
    for s in scenes:
        assert (s.robot_id, s.n_episodes, s.task.max_steps) == ("makermods_metal_station", 3, 900)
        assert s.task.instruction == _PROMPT
        o = SCENES.validate_options(s.scene.id, s.scene.backend_options)
        assert isinstance(o, MetalPenHolderOptions)
        assert 0.44 <= o.holder_start[0] <= 0.48 and 0.28 <= o.holder_start[1] <= 0.32
        assert 0.22 <= o.pen_start[0] <= 0.28 and -0.35 <= o.pen_start[1] <= -0.29
        starts.add((o.holder_start, o.pen_start))
    assert len(starts) == 10


def test_rskill_speaks_the_scene_units() -> None:
    m = load_rskill_manifest(str(_ROOT / "rskills" / "act-pen-holder-bi"))
    robot = RobotDescription.from_yaml(
        str(_ROOT / "robots" / "makermods_metal_station" / "robot.yaml")
    )
    assert m.model_family == "act" and "@a4af0cc4" in (m.weights_uri or "")
    assert m.action_contract is not None and m.action_contract.joint_units is not None
    assert m.action_contract.gripper_scale == 100.0 and m.action_contract.dim == 14
    from openral_sim.sim_runner import _check_policy_units_owned

    sc = SimScene.from_yaml(str(_ROOT / "scenes" / "sim" / "makermods_metal_pen_holder.yaml"))
    _check_policy_units_owned(m, robot, "metal_pen_holder", sc.scene.backend_options)
    keys = {s.vla_feature_key for s in robot.sensors}
    assert {s.vla_feature_key for s in m.sensors_required} <= keys


@settings(max_examples=40, deadline=None)
@given(
    hx=st.floats(0.40, 0.52),
    hy=st.floats(0.24, 0.34),
    yaw=st.floats(-10.0, 10.0),
    jitter=st.floats(0.0, 0.02),
    top=st.floats(0.08, 0.11),
)
def test_options_round_trip_through_json(
    hx: float, hy: float, yaw: float, jitter: float, top: float
) -> None:
    o = MetalPenHolderOptions(
        holder_start=(hx, hy, yaw), spawn_jitter_m=jitter, holder_top_width_m=top
    )
    back = MetalPenHolderOptions.model_validate_json(o.model_dump_json())
    assert back == o
    assert "holder_start" in MetalPenHolderOptions.model_json_schema()["properties"]
