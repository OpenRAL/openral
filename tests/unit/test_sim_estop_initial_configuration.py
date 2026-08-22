# SPDX-License-Identifier: Apache-2.0
"""A kernel stop that fires before the robot is ever commanded is a SCENE defect.

The 2026-08-22 ``robocasa/PickPlaceFridgeShelfToDrawer`` round (seed 1) E-stopped
at sim ``t=4.85 s`` with ``panda_link7`` 24.7 mm inside ``voxel_169769``, having
applied exactly zero action chunks: the MuJoCo ground truth put ``robot0_link6``
at 0.000 m and ``robot0_link7`` at 2.5 mm from ``fridge_main_group_freezer_door``
while every joint velocity was ~0. The robot had not moved — the scene reset
*spawned* it interpenetrating the open freezer door. The kernel was right to
refuse it, but the only artifact naming the stop read exactly like a mid-task
one, and the round was spent debugging a policy that never got to act.

These tests drive
:func:`openral_hal.sim_sensor_bridge.initial_configuration_stop_record` against a
snapshot produced by the real :func:`~openral_hal.sim_sensor_bridge.estop_ground_truth_snapshot`
over a real compiled ``MjModel``/``MjData`` (no mocks, CLAUDE.md §1.11), in the
two shapes that matter: a stop with nothing ever applied (the fridge case) and a
stop after the HAL has actuated (an ordinary mid-task stop, which must stay
unclassified).
"""

from __future__ import annotations

import pytest
from openral_core import JointState
from openral_hal.depth_cloud import robot_self_body_ids
from openral_hal.sim_sensor_bridge import (
    estop_ground_truth_snapshot,
    initial_configuration_stop_record,
)

mujoco = pytest.importorskip("mujoco")

# Robosuite/RoboCasa body + joint naming (``robot0_*``), so the real
# ``robot_self_body_ids`` prefix derivation resolves the robot exactly as it does
# on a composed kitchen scene. ``fridge_main_group_freezer_door`` carries the
# upstream RoboCasa fixture name the field stop actually reported.
_MJCF = """
<mujoco model="estop_initial_configuration">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="robot0_base">
      <geom name="base_col" type="cylinder" size="0.06 0.05"/>
      <body name="robot0_link7" pos="0 0 0.5">
        <joint name="robot0_joint7" type="hinge" axis="0 1 0"/>
        <geom name="link7_col" type="sphere" size="0.04"/>
      </body>
    </body>
    <body name="fridge_main_group_freezer_door" pos="0.2225 0 0.5">
      <geom name="fridge_main_group_freezer_door_main" type="box" size="0.2 0.4 0.2"/>
    </body>
  </worldbody>
</mujoco>
"""
# Sphere centre (0, 0, 0.5) r=0.04; the door's near face sits at 0.2225-0.2 =
# 0.0225, so the link overlaps it by 17.5 mm — the same class of interpenetration
# the field run recorded, and deep enough that MuJoCo does report the contact.
_PENETRATION_M = -0.0175
# A monotonic ns stamp standing in for ``SimAttachedHAL.last_action_ns`` after a
# real ``send_action``. Any non-zero value means "the robot has been commanded".
_ACTUATED_NS = 1_755_100_000_000_000_000


def _snapshot() -> dict[str, object]:
    """Real ground truth for an arm spawned inside the freezer door."""
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joint_state = JointState(
        name=["robot0_joint7"],
        position=[float(data.qpos[0])],
        velocity=[float(data.qvel[0])],
        effort=[0.0],
        stamp_ns=_ACTUATED_NS,
    )
    return estop_ground_truth_snapshot(
        model,
        data,
        robot_body_ids=robot_self_body_ids(model, ["robot0_joint7"]),
        base_frame_body="robot0_base",
        joint_state=joint_state,
    )


def test_stop_before_any_applied_chunk_is_named_an_initial_configuration_violation() -> None:
    """``last_action_ns == 0`` → the refused pose came from the scene reset."""
    snapshot = _snapshot()
    # Precondition: the fixture really does reproduce an interpenetrating pose.
    contacts = snapshot["robot_world_contacts"]
    assert isinstance(contacts, list) and contacts, "the link is inside the freezer door"
    assert contacts[0]["body_b"] == "fridge_main_group_freezer_door"
    assert contacts[0]["distance_m"] == pytest.approx(_PENETRATION_M, abs=1e-3)

    record = initial_configuration_stop_record(
        snapshot, stop_seq=1, last_action_ns=0, candidate_chunks_seen=0
    )

    assert record is not None
    assert record["violation"] == "initial_configuration"
    assert record["stop_seq"] == 1
    assert record["candidate_chunks_seen"] == 0
    # The line must carry the actionable geometry, not just a verdict.
    pair = record["nearest_robot_world_pair"]
    assert isinstance(pair, dict)
    assert pair["body_a"] == "robot0_link7"
    assert pair["body_b"] == "fridge_main_group_freezer_door"
    assert pair["distance_m"] == pytest.approx(_PENETRATION_M, abs=1e-3)
    assert record["sim_time_s"] == snapshot["sim_time_s"]
    # It must point at the scene config, so nobody re-debugs the policy.
    detail = record["detail"]
    assert isinstance(detail, str)
    assert "scene" in detail and "layout_ids" in detail


def test_stop_after_an_applied_action_is_not_classified() -> None:
    """A mid-task stop stays the plain ground-truth snapshot — no false alarm."""
    record = initial_configuration_stop_record(
        _snapshot(), stop_seq=2, last_action_ns=_ACTUATED_NS, candidate_chunks_seen=3
    )

    assert record is None


def test_a_rejected_chunk_still_counts_as_never_applied() -> None:
    """Candidates the kernel refused were never applied — the pose is still the spawn one.

    ``candidate_chunks_seen`` is reported for context but must not gate: a chunk
    the kernel rejected never reached ``send_action``, so the configuration is
    still exactly the one the scene reset produced.
    """
    record = initial_configuration_stop_record(
        _snapshot(), stop_seq=3, last_action_ns=0, candidate_chunks_seen=2
    )

    assert record is not None
    assert record["violation"] == "initial_configuration"
    assert record["candidate_chunks_seen"] == 2


def test_record_survives_a_snapshot_without_probe_fields() -> None:
    """A HAL read fault must not cost the classification — the verdict still lands."""
    record = initial_configuration_stop_record(
        {}, stop_seq=4, last_action_ns=0, candidate_chunks_seen=0
    )

    assert record is not None
    assert record["violation"] == "initial_configuration"
    assert "nearest_robot_world_pair" not in record
    assert "sim_time_s" not in record
