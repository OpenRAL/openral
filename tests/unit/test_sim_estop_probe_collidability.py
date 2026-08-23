# SPDX-License-Identifier: Apache-2.0
"""The near-miss probe measures between SOLID geoms, on every side.

A geom with neither ``contype`` nor ``conaffinity`` cannot collide with
anything: MuJoCo never generates a contact for it and the safety kernel never
checks it, so a signed distance against one is not a penetration. The world side
of the probe has excluded them since rounds 5/6 reported the payload "134 mm
inside ``cab_1_left_group_reg_main``", a RoboCasa region marker.

The **robot and payload sides were not filtered**, and scoping a probe side by
*body* does not scope it to solid geometry — a robosuite link body carries its
visual meshes alongside its collision geom, and an attached payload carries its
own region bounding box. The 2026-08-23 validation round produced two verdicts
off exactly that:

* fridge — ``robot0_g42_vis ~ fridge_main_group_g43 @ 0.000 m``, adjudicated
  ``real-contact``, while the same link's ``robot0_link7_collision`` was
  2.5 mm clear;
* baguette / sink_cup — ``obj_reg_bbox``, the payload's own region marker,
  ranked ahead of every solid payload geom.

Real compiled MuJoCo models throughout, no mocks (CLAUDE.md §1.11). Each
configuration puts a visual shell exactly at 0.000 m from a fixture while the
solid geometry it wraps is provably clear, so a probe that still ranks the
shell first is unambiguous.
"""

from __future__ import annotations

from typing import Any

import pytest
from openral_hal.depth_cloud import robot_self_body_ids
from openral_hal.sim_sensor_bridge import estop_ground_truth_snapshot

mujoco = pytest.importorskip("mujoco")

# Robosuite/RoboCasa naming, so the real `robot_self_body_ids` prefix derivation
# resolves the robot as it does on a composed kitchen scene.
#
# `robot0_link7` carries BOTH a solid sphere (r=0.04) and a visual shell
# (r=0.06) about the same centre, which is the panda's real arrangement. The
# counter's near face sits at `counter_x - 0.2`:
#   counter_x = 0.26 → shell surface touches it exactly (0.000 m) while the
#                      collision sphere is 20 mm clear.
# `obj_main` is the carried payload and carries the same pairing: a 20 mm solid
# box inside a 60 mm region marker.
_MJCF = """
<mujoco model="estop_probe_collidability">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="robot0_base">
      <geom name="base_col" type="cylinder" size="0.06 0.05"/>
      <body name="robot0_link7" pos="0 0 0.5">
        <joint name="robot0_joint7" type="hinge" axis="0 1 0"/>
        <geom name="robot0_link7_collision" type="sphere" size="0.04"/>
        <geom name="robot0_g42_vis" type="sphere" size="0.06"
              contype="0" conaffinity="0"/>
      </body>
    </body>
    <body name="counter_main" pos="{counter_x} 0 0.5">
      <geom name="counter_top" type="box" size="0.2 0.4 0.1"/>
    </body>
    <body name="obj_main" pos="{obj_pos}">
      <freejoint name="obj_joint"/>
      <geom name="obj_g0" type="box" size="0.02 0.02 0.02"/>
      <geom name="obj_reg_bbox" type="box" size="0.06 0.06 0.06"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

# Shell touching the counter, solid sphere 20 mm clear; payload parked away.
_ROBOT_SIDE = {"counter_x": 0.26, "obj_pos": "3 0 0.5"}
# Counter parked away; the payload's region marker (half 0.06) touches the
# link's visual shell (r=0.06) at x=0.12, while `obj_g0` (half 0.02) is 60 mm
# from `robot0_link7_collision` (r=0.04).
_PAYLOAD_SIDE = {"counter_x": 3.0, "obj_pos": "0.12 0 0.5"}


def _model_data(**placement: Any) -> tuple[Any, Any]:
    """Compile the MJCF at one placement and settle kinematics."""
    model = mujoco.MjModel.from_xml_string(_MJCF.format(**placement))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _robot_bodies(model: Any) -> frozenset[int]:
    return robot_self_body_ids(model, ["robot0_joint7"])


def _body_ids(model: Any, *names: str) -> frozenset[int]:
    return frozenset(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) for name in names
    )


def _geoms(pairs: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(p["geom_a"]), str(p["geom_b"])) for p in pairs}


def test_a_visual_shell_on_the_robot_is_never_probed() -> None:
    """The fridge verdict, reproduced and then refused.

    ``robot0_g42_vis`` wraps the link's collision sphere and touches the counter
    at exactly 0.000 m. It must not appear at all, and the nearest pair must be
    the collision sphere's real 20 mm clearance.
    """
    model, data = _model_data(**_ROBOT_SIDE)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        base_frame_body="robot0_base",
    )
    pairs = snapshot["nearest_robot_world_pairs"]
    assert isinstance(pairs, list)
    assert pairs, "the counter is 20 mm away; the probe must report it"
    assert not [p for p in pairs if str(p["geom_a"]).endswith("_vis")]
    assert pairs[0]["geom_a"] == "robot0_link7_collision"
    assert pairs[0]["distance_m"] == pytest.approx(0.02, abs=1e-6)


def test_the_excluded_robot_geoms_are_counted_not_silently_dropped() -> None:
    """An omission the record does not disclose is indistinguishable from a bug."""
    model, data = _model_data(**_ROBOT_SIDE)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        base_frame_body="robot0_base",
    )
    coverage = snapshot["nearest_probe_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["noncollidable_side_geoms_excluded"] == 1  # robot0_g42_vis
    assert coverage["side_geoms"] == 2  # base_col + robot0_link7_collision
    # Nothing is carried here, so the parked payload IS world — and its region
    # marker is the world side's own excluded geom.
    assert coverage["noncollidable_world_geoms_excluded"] == 1  # obj_reg_bbox
    assert coverage["noncollidable_other_geoms_excluded"] == 1


def test_the_payloads_own_region_marker_is_never_probed() -> None:
    """``obj_reg_bbox`` is the payload's bounding box, not the payload.

    It touches the link's visual shell at 0.000 m here. Neither geom is solid,
    so the pair must not exist on either side of the payload↔robot probe — the
    one whose "other" side is an explicit body set and was therefore left
    unfiltered on the argument that the set was already the kernel's links.
    """
    model, data = _model_data(**_PAYLOAD_SIDE)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=_body_ids(model, "obj_main"),
        base_frame_body="robot0_base",
    )
    pairs = snapshot["nearest_payload_robot_pairs"]
    assert isinstance(pairs, list)
    assert pairs
    assert ("obj_reg_bbox", "robot0_g42_vis") not in _geoms(pairs)
    assert not [p for p in pairs if "reg_bbox" in str(p["geom_a"])]
    assert not [p for p in pairs if str(p["geom_b"]).endswith("_vis")]
    assert pairs[0]["geom_a"] == "obj_g0"
    assert pairs[0]["geom_b"] == "robot0_link7_collision"
    assert pairs[0]["distance_m"] == pytest.approx(0.06, abs=1e-6)


def test_both_sides_of_the_payload_probe_report_their_exclusions() -> None:
    """The attestation a downstream adjudicator reads before trusting a 0 m pair."""
    model, data = _model_data(**_PAYLOAD_SIDE)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=_body_ids(model, "obj_main"),
        base_frame_body="robot0_base",
    )
    for key in (
        "nearest_probe_coverage",
        "nearest_payload_world_coverage",
        "nearest_payload_robot_coverage",
    ):
        coverage = snapshot[key]
        assert isinstance(coverage, dict), key
        assert "noncollidable_side_geoms_excluded" in coverage, key
        assert "noncollidable_other_geoms_excluded" in coverage, key
    payload_robot = snapshot["nearest_payload_robot_coverage"]
    assert isinstance(payload_robot, dict)
    assert payload_robot["noncollidable_side_geoms_excluded"] == 1  # obj_reg_bbox
    assert payload_robot["noncollidable_other_geoms_excluded"] == 1  # robot0_g42_vis


def test_the_harness_trusts_a_zero_metre_pair_only_once_it_is_attested() -> None:
    """The producer fix and its consumer, checked against one another.

    ``tools/validation_matrix.py`` promotes a ``<= 0 m`` pair to
    ``real-contact`` only when the snapshot says both probe sides were
    collidability-filtered. A snapshot this HAL writes says so; the recorded
    2026-08-22 / 2026-08-23 ones do not, and the harness refuses them.
    """
    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "validation_matrix_collidability", repo_root / "tools" / "validation_matrix.py"
    )
    assert spec is not None and spec.loader is not None
    validation_matrix = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validation_matrix
    spec.loader.exec_module(validation_matrix)

    model, data = _model_data(**_PAYLOAD_SIDE)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=_body_ids(model, "obj_main"),
        base_frame_body="robot0_base",
    )
    assert validation_matrix.probe_is_collidability_filtered(snapshot) is True
