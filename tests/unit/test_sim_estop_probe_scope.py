# SPDX-License-Identifier: Apache-2.0
"""Near-miss probe scope + payload probes — the four gaps the field round exposed.

The first adjudication round on the Spark used the stop record for real
verdicts and found four ways it could mislead:

1. On a mobile manipulator every reported pair saturated on
   ``mobilebase``↔floor contact at 0-2 mm — the robot merely standing on the
   ground, geometry the manifest deliberately keeps OUT of
   ``collision_geometry`` — which hid an arm that was 17-30 mm inside a
   freezer door and would have produced a WRONG verdict.
2. A carried payload logged only *realized* contacts, so a payload↔world
   margin stop had nothing to adjudicate with.
3. Payload↔robot-link self-pairs were never probed at all, leaving a
   ``sink_cup``↔``panda_link5`` stop at −0.62 mm unadjudicable.
4. ``robot_world_contacts == []`` reads as "nothing was touching", which is
   false: MuJoCo contype/conaffinity exclusions suppress contacts entirely.

Round 5/6 added a fifth: the probe measured against **non-collidable marker
geometry** and reported "payload −134 mm inside ``cab_1_left_group_reg_main``"
— a RoboCasa region marker, physically meaningless. Solidity and suppression
are different things, and this fixture now models both separately:

* ``freezer_door`` is SOLID but its bitmasks are disjoint from the robot's
  (``contype=2 conaffinity=2`` vs the default ``1``/``1``), so MuJoCo generates
  no contact however deep the arm goes — the real shape of the field's "arm
  30 mm inside a door with ``ncon == 0``", and the exact case the near-miss
  probe exists to adjudicate. It must keep being measured.
* ``cab_1_left_group_reg_main`` carries neither ``contype`` nor
  ``conaffinity``: it is not solid at all, cannot collide with anything, and is
  never checked by the kernel. It must never be measured.

These drive the real ``robots/panda_mobile/robot.yaml`` manifest (whose
``collision_geometry`` covers ``panda_link1..7`` and deliberately omits
``base_link`` and ``panda_finger_pair``) against a real compiled MuJoCo model
of a wheeled base on a floor plane — no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

from typing import Any

import pytest
from openral_core import RobotDescription
from openral_hal.depth_cloud import robot_self_body_ids
from openral_hal.sim_sensor_bridge import estop_ground_truth_snapshot, kernel_checked_body_ids

mujoco = pytest.importorskip("mujoco")

# A robocasa-shaped mobile manipulator: `mobilebase0_*` chassis + four wheels
# resting on a floor plane, `robot0_link1..7` arm, `gripper0_*` fingers, and
# the two world fixtures the field stops involved. Joint names are the exact
# `sim_joint_name`s panda_mobile's manifest declares, so both the self-filter
# (prefix-derived) and the kernel scope (child_link-derived) resolve for real.
#
# `freezer_door` is solid with bitmasks disjoint from the robot's
# (2/2 vs the default 1/1) — the field pathology: MuJoCo generates NO contact
# for it however deep the arm goes, yet it is a real obstacle and must be
# measured. `cab_1_left_group_reg_main` is the opposite: a RoboCasa region
# marker with neither contype nor conaffinity, overlapping both the arm and the
# payload's parking spot. It is not solid, so measuring against it invents the
# "-134 mm inside" phantom of rounds 5/6.
_MJCF = """
<mujoco model="estop_probe_scope">
  <option gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="mobilebase0_base" pos="0 0 0.12">
      <joint name="mobilebase0_joint_mobile_forward" type="slide" axis="1 0 0"/>
      <joint name="mobilebase0_joint_mobile_side" type="slide" axis="0 1 0"/>
      <joint name="mobilebase0_joint_mobile_yaw" type="hinge" axis="0 0 1"/>
      <geom name="chassis_col" type="box" size="0.25 0.25 0.06"/>
      <geom name="skirt_front" type="box" pos="0.2 0 -0.055" size="0.05 0.2 0.005"/>
      <geom name="skirt_rear" type="box" pos="-0.2 0 -0.055" size="0.05 0.2 0.005"/>
      <body name="mobilebase0_wheel_fl" pos="0.2 0.2 -0.06">
        <geom name="wheel_fl_col" type="sphere" size="0.06"/>
      </body>
      <body name="mobilebase0_wheel_fr" pos="0.2 -0.2 -0.06">
        <geom name="wheel_fr_col" type="sphere" size="0.06"/>
      </body>
      <body name="mobilebase0_wheel_rl" pos="-0.2 0.2 -0.06">
        <geom name="wheel_rl_col" type="sphere" size="0.06"/>
      </body>
      <body name="mobilebase0_wheel_rr" pos="-0.2 -0.2 -0.06">
        <geom name="wheel_rr_col" type="sphere" size="0.06"/>
      </body>
      <body name="mobilebase0_caster_fl" pos="0.24 0.0 -0.09">
        <geom name="caster_fl_col" type="sphere" size="0.03"/>
      </body>
      <body name="mobilebase0_caster_fr" pos="-0.24 0.0 -0.09">
        <geom name="caster_fr_col" type="sphere" size="0.03"/>
      </body>
      <body name="mobilebase0_caster_rl" pos="0.0 0.24 -0.09">
        <geom name="caster_rl_col" type="sphere" size="0.03"/>
      </body>
      <body name="mobilebase0_caster_rr" pos="0.0 -0.24 -0.09">
        <geom name="caster_rr_col" type="sphere" size="0.03"/>
      </body>
      <body name="mobilebase0_support" pos="0 0 0.06">
        <geom name="support_col" type="cylinder" size="0.08 0.15"/>
        <body name="robot0_link1" pos="0 0 0.3">
          <joint name="robot0_joint1" type="hinge" axis="0 0 1"/>
          <geom name="link1_col" type="capsule" fromto="0 0 0 0 0 0.1" size="0.05"/>
          <body name="robot0_link2" pos="0 0 0.1">
            <joint name="robot0_joint2" type="hinge" axis="0 1 0"/>
            <geom name="link2_col" type="capsule" fromto="0 0 0 0 0 0.1" size="0.05"/>
            <body name="robot0_link3" pos="0 0 0.1">
              <joint name="robot0_joint3" type="hinge" axis="0 0 1"/>
              <geom name="link3_col" type="capsule" fromto="0 0 0 0 0 0.1" size="0.045"/>
              <body name="robot0_link4" pos="0 0 0.1">
                <joint name="robot0_joint4" type="hinge" axis="0 1 0"/>
                <geom name="link4_col" type="capsule" fromto="0 0 0 0 0 0.1" size="0.045"/>
                <body name="robot0_link5" pos="0 0 0.1">
                  <joint name="robot0_joint5" type="hinge" axis="0 0 1"/>
                  <geom name="link5_col" type="capsule" fromto="0 0 0 0 0 0.1" size="0.04"/>
                  <body name="robot0_link6" pos="0 0 0.1">
                    <joint name="robot0_joint6" type="hinge" axis="0 1 0"/>
                    <geom name="link6_col" type="sphere" size="0.04"/>
                    <body name="robot0_link7" pos="0.1 0 0">
                      <joint name="robot0_joint7" type="hinge" axis="0 0 1"/>
                      <geom name="link7_col" type="sphere" size="0.04"/>
                      <body name="gripper0_right_finger" pos="0 -0.06 0">
                        <joint name="gripper0_right_finger_joint1" type="slide" axis="0 1 0"/>
                        <geom name="finger_col" type="box" size="0.01 0.02 0.03"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="freezer_door" pos="{door_x} 0 0.98">
      <geom name="door_panel" type="box" size="0.02 0.4 0.3" contype="2" conaffinity="2"/>
    </body>
    <body name="cab_1_left_group_reg_main" pos="0 0.25 0.8">
      <geom name="cab_1_left_group_reg_main_g0" type="box" size="0.5 0.5 0.4"
            contype="0" conaffinity="0"/>
    </body>
    <body name="counter_main" pos="0 {counter_y} 0.6">
      <geom name="counter_top" type="box" size="0.4 0.02 0.3"/>
    </body>
    <body name="sink_cup" pos="{cup_pos}">
      <freejoint name="sink_cup_joint"/>
      <geom name="cup_body" type="sphere" size="0.03"/>
    </body>
  </worldbody>
</mujoco>
"""

# link7's centre sits at (0.10, 0, 0.98) with r=0.04, so its surface reaches
# x=0.14; the door's near face is at ``door_x - 0.02``. 0.143 puts the arm
# 17 mm inside the door (the field number), 0.165 leaves a 5 mm margin — the
# gap class the probe exists for, and the one base-floor noise can outrank.
_ARM_IN_DOOR_X = 0.143
_ARM_PENETRATION_M = -0.017
_ARM_NEAR_DOOR_X = 0.165
_ARM_NEAR_DOOR_GAP_M = 0.005
_DOOR_CLEAR_X = 1.5
# Payload parked clear, 3 mm off the counter's near face (y = 0.58), or
# 0.62 mm inside link5's capsule (surface at y = 0.04, z = 0.93).
_CUP_CLEAR = "2.0 2.0 0.5"
_CUP_NEAR_COUNTER = "0 0.547 0.6"
_COUNTER_NEAR_Y = 0.6
_COUNTER_FAR_Y = 3.0
_CUP_PAYLOAD_WORLD_GAP_M = 0.003
_CUP_IN_LINK5 = "0 0.06938 0.93"
_CUP_LINK5_PENETRATION_M = -0.00062

_ROBOT_SIM_JOINTS = [
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
    "robot0_joint1",
    "robot0_joint2",
    "robot0_joint3",
    "robot0_joint4",
    "robot0_joint5",
    "robot0_joint6",
    "robot0_joint7",
    "gripper0_right_finger_joint1",
]


def _model_data(
    *,
    door_x: float = _DOOR_CLEAR_X,
    counter_y: float = _COUNTER_FAR_Y,
    cup_pos: str = _CUP_CLEAR,
) -> tuple[Any, Any]:
    """Compile the scene and settle kinematics (base wheels resting on the floor)."""
    model = mujoco.MjModel.from_xml_string(
        _MJCF.format(door_x=door_x, counter_y=counter_y, cup_pos=cup_pos)
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _panda_mobile() -> RobotDescription:
    """The real mobile-manipulator manifest whose collision_geometry omits the base."""
    return RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")


def _robot_bodies(model: Any) -> frozenset[int]:
    return robot_self_body_ids(model, _ROBOT_SIM_JOINTS)


def _cup_ids(model: Any) -> frozenset[int]:
    return frozenset({int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sink_cup"))})


def _bodies_in(pairs: Any) -> set[str]:
    return {str(p["body_a"]) for p in pairs} | {str(p["body_b"]) for p in pairs}


def test_kernel_scope_is_the_manifests_collision_geometry() -> None:
    """The probe scope is exactly the links the kernel checks — no base, no fingers."""
    model, _data = _model_data()
    scope = kernel_checked_body_ids(model, _panda_mobile())

    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in scope}
    assert names == {f"robot0_link{i}" for i in range(1, 8)}
    # The bodies the manifest deliberately exempts must not be in scope.
    assert not names & {
        "mobilebase0_base",
        "mobilebase0_support",
        "mobilebase0_wheel_fl",
        "mobilebase0_wheel_fr",
        "mobilebase0_wheel_rl",
        "mobilebase0_wheel_rr",
        "gripper0_right_finger",
    }


def test_unscoped_probe_saturates_on_base_floor_pairs() -> None:
    """The field failure, reproduced: whole-robot scope buries the arm in floor noise.

    The base rests on eight floor contact points at 0.000 m. A margin stop —
    the arm 5 mm off the freezer door, exactly the class the probe exists to
    adjudicate — sorts behind all eight, so every reported slot is the robot
    standing on the ground and the arm is invisible.
    """
    model, data = _model_data(door_x=_ARM_NEAR_DOOR_X)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        max_pairs=8,
    )

    pairs = snapshot["nearest_robot_world_pairs"]
    assert snapshot["probe_robot_scope"] == "all_robot_bodies"
    assert isinstance(pairs, list) and pairs
    # Every reported slot is the robot standing on the ground.
    assert {str(p["body_b"]) for p in pairs} == {"world"}  # the floor plane's body
    assert all(str(p["body_a"]).startswith("mobilebase0_") for p in pairs)
    assert "robot0_link7" not in _bodies_in(pairs), "the arm is buried — the field bug"


def test_kernel_scoped_probe_surfaces_the_arm_at_the_door() -> None:
    """Scoped to kernel-checked links, the same stop reports the arm, not the floor."""
    model, data = _model_data(door_x=_ARM_NEAR_DOOR_X)
    scope = kernel_checked_body_ids(model, _panda_mobile())
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        probe_body_ids=scope,
        max_pairs=8,
    )

    assert snapshot["probe_robot_scope"] == "kernel_checked_links"
    pairs = snapshot["nearest_robot_world_pairs"]
    assert isinstance(pairs, list) and pairs
    closest = pairs[0]
    assert closest["body_a"] == "robot0_link7"
    assert closest["body_b"] == "freezer_door"
    assert closest["distance_m"] == pytest.approx(_ARM_NEAR_DOOR_GAP_M, abs=5e-4)
    # No base/wheel/floor noise survives the scope.
    assert not _bodies_in(pairs) & {"world", "mobilebase0_base", "mobilebase0_wheel_fl"}
    excluded = snapshot["probe_excluded_robot_bodies"]
    assert isinstance(excluded, list)
    assert "mobilebase0_wheel_fl" in excluded
    assert "gripper0_right_finger" in excluded


def test_penetration_survives_a_starved_call_budget() -> None:
    """A tiny budget truncates the far end: real interpenetration still gets probed.

    The floor plane is unbounded (``geom_rbound == 0``), so it formally ranks
    ahead of every other candidate — that is how a kitchen floor consumed the
    whole call budget in the field. Planes now rank behind genuine
    bounding-sphere overlap, so even at ``max_calls=2`` with the whole robot
    in scope the arm inside the door is found.
    """
    model, data = _model_data(door_x=_ARM_IN_DOOR_X)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        max_pairs=2,
        max_calls=2,
    )

    pairs = snapshot["nearest_robot_world_pairs"]
    assert isinstance(pairs, list) and pairs
    assert pairs[0]["body_a"] == "robot0_link7"
    assert pairs[0]["body_b"] == "freezer_door"


def test_contact_list_is_empty_while_the_arm_is_deep_inside_the_door() -> None:
    """`ncon == 0` is not "nothing touching": contype/conaffinity suppress contacts."""
    model, data = _model_data(door_x=_ARM_IN_DOOR_X)
    scope = kernel_checked_body_ids(model, _panda_mobile())
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        probe_body_ids=scope,
    )

    # The door's bitmasks don't meet the robot's — MuJoCo reports no contact at
    # all, yet the arm is 17 mm inside a solid door.
    assert not [
        c
        for c in snapshot["robot_world_contacts"]  # type: ignore[union-attr]
        if c["body_b"] == "freezer_door"
    ]
    pairs = snapshot["nearest_robot_world_pairs"]
    assert isinstance(pairs, list)
    assert pairs[0]["distance_m"] == pytest.approx(_ARM_PENETRATION_M, abs=1e-3)
    caveat = snapshot["contacts_caveat"]
    assert isinstance(caveat, str)
    assert "does NOT mean no interpenetration" in caveat


def test_payload_world_near_miss_is_probed() -> None:
    """A carried payload 3 mm off a counter: the margin stop is now adjudicable."""
    model, data = _model_data(counter_y=_COUNTER_NEAR_Y, cup_pos=_CUP_NEAR_COUNTER)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=_cup_ids(model),
        probe_body_ids=kernel_checked_body_ids(model, _panda_mobile()),
    )

    assert snapshot["stop_class"] == "attached_payload"
    assert snapshot["payload_contacts"] == []  # a near miss realizes no contact
    pairs = snapshot["nearest_payload_world_pairs"]
    assert isinstance(pairs, list) and pairs
    closest = pairs[0]
    assert closest["body_a"] == "sink_cup"
    assert closest["body_b"] == "counter_main"
    assert closest["distance_m"] == pytest.approx(_CUP_PAYLOAD_WORLD_GAP_M, abs=5e-4)
    # Every probe attests its own coverage: a payload pair is only readable as
    # absent when the probe actually reached the whole candidate set.
    coverage = snapshot["nearest_payload_world_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["side_geoms"] == coverage["side_geoms_probed"]
    assert coverage["truncated"] is False


def test_payload_robot_self_pair_is_probed() -> None:
    """The sink_cup↔panda_link5 self-pair stop (−0.62 mm) is now reported."""
    model, data = _model_data(cup_pos=_CUP_IN_LINK5)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=_cup_ids(model),
        probe_body_ids=kernel_checked_body_ids(model, _panda_mobile()),
    )

    pairs = snapshot["nearest_payload_robot_pairs"]
    assert isinstance(pairs, list) and pairs
    closest = pairs[0]
    assert closest["body_a"] == "sink_cup"
    assert closest["body_b"] == "robot0_link5"
    assert closest["distance_m"] == pytest.approx(_CUP_LINK5_PENETRATION_M, abs=2e-4)
    # The payload never appears as its own world obstacle.
    assert "sink_cup" not in _bodies_in(snapshot["nearest_robot_world_pairs"])
    coverage = snapshot["nearest_payload_robot_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["side_geoms"] == coverage["side_geoms_probed"]


def test_region_marker_never_reports_a_phantom_robot_world_penetration() -> None:
    """A non-collidable marker swallowing the arm is not an obstacle it is inside.

    ``cab_1_left_group_reg_main`` overlaps ``robot0_link7`` by ~0.3 m here.
    With neither ``contype`` nor ``conaffinity`` it cannot collide with
    anything and the kernel never checks it, so a signed distance against it is
    not a penetration — it is the round 5/6 phantom that read "payload −134 mm
    inside cab_1_left_group_reg_main". The real door 5 mm away must still rank.
    """
    model, data = _model_data(door_x=_ARM_NEAR_DOOR_X)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        probe_body_ids=kernel_checked_body_ids(model, _panda_mobile()),
    )

    pairs = snapshot["nearest_robot_world_pairs"]
    assert isinstance(pairs, list) and pairs
    assert "cab_1_left_group_reg_main" not in _bodies_in(pairs), (
        "the probe measured against a non-collidable marker — a phantom "
        f"penetration; got {[(p['body_a'], p['body_b'], p['distance_m']) for p in pairs]}"
    )
    assert pairs[0]["body_b"] == "freezer_door", (
        "a solid obstacle whose bitmasks merely fail to meet the robot's must "
        "still be measured — that is the case the probe exists for"
    )
    # The omission is reported, not silent (CLAUDE.md §1.4).
    coverage = snapshot["nearest_probe_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["noncollidable_world_geoms_excluded"] == 1


def test_region_marker_never_reports_a_phantom_payload_penetration() -> None:
    """Same rule on the payload↔world probe: the counter ranks, the marker is gone."""
    model, data = _model_data(counter_y=_COUNTER_NEAR_Y, cup_pos=_CUP_NEAR_COUNTER)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        attached_body_ids=_cup_ids(model),
        probe_body_ids=kernel_checked_body_ids(model, _panda_mobile()),
    )

    pairs = snapshot["nearest_payload_world_pairs"]
    assert isinstance(pairs, list) and pairs
    assert "cab_1_left_group_reg_main" not in _bodies_in(pairs), (
        "the carried payload was reported deep inside a marker it cannot touch"
    )
    assert pairs[0]["body_b"] == "counter_main"
    coverage = snapshot["nearest_payload_world_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["noncollidable_world_geoms_excluded"] == 1


def test_unattached_stop_emits_no_payload_probes() -> None:
    """Payload probes stay empty with nothing carried — no phantom sections."""
    model, data = _model_data(door_x=_ARM_IN_DOOR_X)
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        probe_body_ids=kernel_checked_body_ids(model, _panda_mobile()),
    )

    assert snapshot["stop_class"] == "robot_world"
    assert snapshot["nearest_payload_world_pairs"] == []
    assert snapshot["nearest_payload_robot_pairs"] == []
    # …and no phantom coverage either: an empty block, not a zeroed probe that
    # reads as "we looked and found nothing".
    assert snapshot["nearest_payload_world_coverage"] == {}
    assert snapshot["nearest_payload_robot_coverage"] == {}
    # The robot↔world probe did run, so it attests coverage for real.
    robot_world = snapshot["nearest_probe_coverage"]
    assert isinstance(robot_world, dict)
    assert robot_world["side_geoms"] == robot_world["side_geoms_probed"]


def test_link_link_self_pairs_are_probed(caplog: pytest.LogCaptureFixture) -> None:
    """#216: the pair a `kind=self` stop between two bare links names.

    Before this the probe could not produce one at all — every other probe
    excludes the whole robot from its far side — so the 2026-09-04 battery's
    `panda_link5`/`panda_link7` stop had no ground truth, and the harness scored
    it against a kitchen island instead (#208).
    """
    del caplog
    model, data = _model_data()
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        probe_body_ids=kernel_checked_body_ids(model, _panda_mobile()),
    )

    pairs = snapshot["nearest_link_link_pairs"]
    assert isinstance(pairs, list) and pairs, "the arm's own links must be measurable"
    for pair in pairs:
        # Both sides are robot links. Nothing from the world may leak in.
        assert str(pair["body_a"]).startswith(("robot0_", "mobilebase0_"))
        assert str(pair["body_b"]).startswith(("robot0_", "mobilebase0_"))
        # A link's own geoms touch by construction; the kernel skips the pair
        # (`lb == lb2`) and a 0 m self-pair would sort straight to the top and
        # bury whichever pair actually stopped the run.
        assert pair["body_a"] != pair["body_b"]

    # The mirror is gone too: with one body set on both sides every pair would
    # otherwise appear as (a, b) and (b, a), doubling the records and halving
    # the effective `max_pairs`.
    seen = [frozenset({str(p["body_a"]), str(p["body_b"])}) for p in pairs]
    assert len(seen) == len({frozenset(s) for s in seen})

    coverage = snapshot["nearest_link_link_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["certified_pairs"] >= 1


def test_the_link_link_budget_is_published_and_carries_no_voxel_term() -> None:
    """A link-vs-link stop has an OBB on both sides and no voxel on either.

    Charging it the world block's `corner_slop + voxel_half_diagonal` would
    budget a comparison the stop never made. The BOX case's budget is a
    snapshot-wide upper bound (`2 * max_corner_slop`); the HULL case has no
    snapshot-wide equivalent, because a hull's overhang past its source mesh
    is a per-link quantity a consumer sums from `collision_model_slop.links`
    for the two specific links the kernel named (openral#221) — never maxed
    across links the way the box term is.
    """
    model, data = _model_data()
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        probe_body_ids=kernel_checked_body_ids(model, _panda_mobile()),
        description=_panda_mobile(),
    )

    budget = snapshot["adjudication_budget"]
    assert isinstance(budget, dict)
    block = budget["link_link"]
    assert isinstance(block, dict)
    slop = float(budget["max_corner_slop_m"])
    assert block["admissible_gap_box_m"] == pytest.approx(2.0 * slop)
    assert "admissible_gap_hull_m" not in block
    # Both sides are links, so the world block's voxel term must not appear here.
    assert "voxel_half_diagonal_m" not in block


def test_hull_overhang_is_published_per_link_for_every_stage_two_hull() -> None:
    """The #221 term: how far each link's declared hull reaches past its mesh.

    `panda_link1` runs the 26-DOP only (its hull is over the vertex budget),
    so it carries no `hull_overhang_m` at all -- never a silent `0`.
    """
    model, data = _model_data()
    snapshot = estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=_robot_bodies(model),
        probe_body_ids=kernel_checked_body_ids(model, _panda_mobile()),
        description=_panda_mobile(),
    )

    links = snapshot["adjudication_budget"]["collision_model_slop"]["links"]  # type: ignore[index]
    for name in ("panda_link2", "panda_link5", "panda_link7"):
        overhang = links[name]["hull_overhang_m"]
        assert isinstance(overhang, float)
        assert overhang > 0.0
    assert links["panda_link1"]["hull_overhang_m"] is None
