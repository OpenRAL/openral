# SPDX-License-Identifier: Apache-2.0
"""``metal_pick_place`` options model + the scene's single policy-unit conversion.

The scene speaks the HappyEthan/metal_pick_place dataset's units (degrees,
gripper stroke 0..100 with joint8 metres = stroke / 2000) and converts exactly
once; these checks pin that conversion and its inverse, and that the registry
validates the shipped scene YAML's ``backend_options`` with this model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import SimScene, load_benchmark_suite, raise_on_invalid_suite
from openral_core.exceptions import ROSConfigError
from openral_sim import SCENES
from openral_sim.backends.metal_pick_place._assets import (
    MetalPickPlaceOptions,
    mujoco_principal_pixel,
)
from openral_sim.backends.metal_pick_place.env import (
    PLACEMENT_GRADE_SCORES,
    footprint_overlap,
    placement_grade,
    policy_action_to_ctrl,
    qpos_to_policy_state,
)

_ROOT = Path(__file__).resolve().parents[2]
_EP0 = np.load(_ROOT / "tests" / "unit" / "fixtures" / "metal_pick_place_ep00.npz")


def test_shipped_yaml_validates_against_the_registered_options_model() -> None:
    scene = SimScene.from_yaml(str(_ROOT / "scenes" / "sim" / "makermods_metal_pick_place.yaml"))
    opts = SCENES.validate_options(scene.scene.id, scene.scene.backend_options)
    assert isinstance(opts, MetalPickPlaceOptions)
    assert (opts.joint_units, opts.gripper_scale, opts.layout) == ("degrees", 100.0, "A")
    assert SCENES.meta("metal_pick_place")["converts_policy_units"] is True
    assert SCENES.resolve_robot("metal_pick_place", None) == "makermods_metal"


@pytest.mark.parametrize(
    "bad",
    [{"layout": "Z"}, {"initial_joint_positions": [0.0] * 6}, {"leader_arm": "ghost"}, {"nope": 1}],
)
def test_bad_options_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ROSConfigError):
        SCENES.validate_options("metal_pick_place", bad)


def test_degree_dataset_round_trip_matches_the_dataset_gripper_rule() -> None:
    """Every real ep0 state: state -> ctrl -> qpos -> state is the identity (stroke <= 100)."""
    for state in _EP0["state"].astype(np.float64):
        ctrl = policy_action_to_ctrl(state, joint_units="degrees", gripper_scale=100.0)
        np.testing.assert_allclose(ctrl[:6], np.radians(state[:6]))
        stroke = min(max(state[6], 0.0), 100.0)
        assert ctrl[6] == pytest.approx(stroke / 2000.0)  # joint8 m = stroke / 2000
        qpos = np.r_[ctrl[:6], -ctrl[6], ctrl[6]]
        back = qpos_to_policy_state(qpos, joint_units="degrees", gripper_scale=100.0)
        np.testing.assert_allclose(back[:6], state[:6], atol=1e-9)
        assert back[6] == pytest.approx(stroke)


def test_gripper_offset_round_trip() -> None:
    """The closed-gripper reading (stroke 2.5) maps to fingers touching and back."""
    off = MetalPickPlaceOptions().gripper_offset_m
    for stroke in (2.5, 27.5, 100.0):
        a = np.r_[np.zeros(6), stroke]
        ctrl = policy_action_to_ctrl(
            a, joint_units="degrees", gripper_scale=100.0, gripper_offset_m=off
        )
        assert ctrl[6] == pytest.approx(max(0.0, stroke / 2000 - off))
        back = qpos_to_policy_state(
            np.r_[ctrl[:6], -ctrl[6], ctrl[6]],
            joint_units="degrees",
            gripper_scale=100.0,
            gripper_offset_m=off,
        )
        assert back[6] == pytest.approx(stroke)


def test_radian_unit_gripper_surface_is_the_manifest_one() -> None:
    a = np.array([0.1, -0.5, 0.7, 0.2, -0.3, 0.4, 0.5])
    ctrl = policy_action_to_ctrl(a, joint_units="radians", gripper_scale=1.0)
    np.testing.assert_allclose(ctrl, [*a[:6], 0.025])
    with pytest.raises(ROSConfigError):
        policy_action_to_ctrl(a[:6], joint_units="radians", gripper_scale=1.0)


def test_principal_pixel_maps_to_mujoco_offset() -> None:
    assert mujoco_principal_pixel(320.0, 240.0, 640, 480) == (0.0, 0.0)
    assert mujoco_principal_pixel(287.8, 195.0, 640, 480) == pytest.approx((32.2, 45.0))


def _yaw(deg: float) -> NDArray[np.float64]:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _grade(offset_m: float, *, lifted: bool, yaw_deg: float = 0.0) -> tuple[float, str]:
    """Overlap + grade of the scene's 25 mm cube ``offset_m`` along x from the 3.9 cm marker."""
    o = MetalPickPlaceOptions()
    marker = np.array([0.278, 0.076])
    overlap = footprint_overlap(
        marker + np.array([offset_m, 0.0]),
        _yaw(yaw_deg),
        o.cube_edge_m,
        marker,
        o.marker_diameter_m,
    )
    return overlap, placement_grade(overlap, lifted)


def test_centred_cube_is_complete_at_any_yaw() -> None:
    for yaw in (0.0, 30.0, 45.0):
        overlap, grade = _grade(0.0, lifted=True, yaw_deg=yaw)
        assert overlap > 0.9 and grade == "complete"  # 25 mm square in a 39 mm disc: corners out
    assert PLACEMENT_GRADE_SCORES[grade] == 1.0


def test_half_off_the_marker_is_the_partial_boundary() -> None:
    # Cube centre on the disc edge: under half of the footprint is over it.
    overlap, grade = _grade(0.0195, lifted=True)
    assert 0.3 < overlap < 0.5 and grade == "partial"
    # Find the offset where the footprint is half over the disc: complete just inside it.
    offsets = np.linspace(0.0, 0.0195, 200)
    half = next(d for d in offsets if _grade(d, lifted=True)[0] < 0.5)
    assert _grade(half - 0.001, lifted=False)[1] == "complete"  # a push onto it counts
    assert _grade(half + 0.001, lifted=False)[1] == "partial"
    assert placement_grade(0.5, lifted=False) == "complete"
    assert placement_grade(0.4999, lifted=False) == "partial"


def test_far_cube_is_semi_if_lifted_else_fail() -> None:
    assert _grade(0.05, lifted=True) == (0.0, "semi")
    assert _grade(0.05, lifted=False) == (0.0, "fail")
    assert list(PLACEMENT_GRADE_SCORES.values()) == [1.0, 0.5, 0.25, 0.0]


def test_benchmark_suite_is_three_layout_a_scenes_inside_the_dataset_layout() -> None:
    scenes = load_benchmark_suite(str(_ROOT / "benchmarks" / "makermods_metal_pick_place.yaml"))
    raise_on_invalid_suite(scenes, suite_id="makermods_metal_pick_place")
    assert len(scenes) == 3
    defaults = MetalPickPlaceOptions().layouts
    per_layout: dict[str, int] = {}
    for sc in scenes:
        opts = SCENES.validate_options(sc.scene.id, sc.scene.backend_options)
        assert isinstance(opts, MetalPickPlaceOptions)
        assert sc.task.id == f"metal_pick_place/{opts.layout}"
        assert (sc.n_episodes, sc.task.max_steps, opts.spawn_jitter_m) == (3, 900, 0.0)
        name = opts.layout[0]  # A1 -> dataset layout A
        per_layout[name] = per_layout.get(name, 0) + 1
        cube, marker = opts.layouts[opts.layout]
        (dc, dm) = defaults[name]
        assert np.abs(np.subtract(cube, dc)).max() <= 0.015 + 1e-9
        assert np.abs(np.subtract(marker, dm)).max() <= 0.015 + 1e-9
    assert per_layout == {"A": 3}
