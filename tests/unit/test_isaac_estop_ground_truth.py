"""Isaac e-stop ground truth: the PhysX contact fold and the stop record it feeds.

GPU-free halves of the Isaac ``sim.estop_ground_truth_snapshot`` path: the
sidecar's ``aggregate_robot_contacts`` (imported from ``tools/``, which needs no
live Kit app) and the HAL's ``isaac_estop_ground_truth_record``. The live PhysX
half runs in ``tests/sim/test_franka_isaac_contact_ground_truth.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from openral_core import JointState
from openral_hal.sim_sensor_bridge import isaac_estop_ground_truth_record

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from isaac_manifest_scene import CONTACTS_CAVEAT, aggregate_robot_contacts

ROBOT = "/panda"


def _pt(c0: str, c1: str, sep: float, impulse: float = 0.0) -> dict[str, object]:
    return {
        "actor0": c0.rsplit("/", 1)[0] if c0.startswith(ROBOT) else c0,
        "actor1": c1.rsplit("/", 1)[0] if c1.startswith(ROBOT) else c1,
        "collider0": c0,
        "collider1": c1,
        "separation": sep,
        "impulse": (impulse, 0.0, 0.0),
        "position": (0.0, 0.0, sep),
    }


def test_fold_keeps_only_robot_vs_world_pairs_deepest_first() -> None:
    link0 = "/panda/panda_link0/collisions"
    link4 = "/panda/panda_link4/collisions"
    points = [
        _pt("/World/defaultGroundPlane/collisionPlane", link0, 0.0005),
        _pt(link4, "/World/objects/box", -0.003, impulse=0.2),
        _pt("/World/objects/box", link4, -0.011, impulse=0.5),
        _pt(link4, "/panda/panda_link5/collisions", -0.001),  # self contact
        _pt("/World/objects/box", "/World/defaultGroundPlane/collisionPlane", 0.0),
    ]
    pairs, self_points = aggregate_robot_contacts(points, ROBOT + "/")
    assert self_points == 1
    assert [(p["link"], p["other_prim"]) for p in pairs] == [
        ("panda_link4", "/World/objects/box"),
        ("panda_link0", "/World/defaultGroundPlane/collisionPlane"),
    ]
    box = pairs[0]
    assert box["robot_collider"] == link4
    assert box["min_separation_m"] == -0.011
    assert box["contact_points"] == 2
    assert box["impulse_ns"] == 0.7
    assert box["deepest_point_xyz"] == [0.0, 0.0, -0.011]


def test_a_prim_sharing_the_robot_prefix_is_not_the_robot() -> None:
    pairs, _ = aggregate_robot_contacts(
        [_pt("/panda_table/top", "/panda/panda_hand/collisions", -0.002)], ROBOT
    )
    assert [p["other_prim"] for p in pairs] == ["/panda_table/top"]


def test_record_keeps_the_mujoco_keys_and_is_json_safe() -> None:
    pairs, self_points = aggregate_robot_contacts(
        [_pt("/World/objects/box", "/panda/panda_link4/collisions", -0.011)], ROBOT
    )
    reply = {
        "sim_time_ns": 2_500_000_000,
        "robot_prim": ROBOT,
        "robot_world_contacts": pairs,
        "self_contact_points": self_points,
        "nearest_robot_world_pairs": None,
        "nearest_unavailable_reason": "no shape-to-shape distance",
        "contacts_caveat": CONTACTS_CAVEAT,
    }
    joints = JointState(name=["panda_joint1"], position=[0.1], velocity=[0.0], stamp_ns=7)
    record = isaac_estop_ground_truth_record(
        reply,
        joint_state=joints,
        evidence_voxel={"index": 12, "resolution": 0.05},
        attached_object_ids=["cracker_box"],
    )
    json.dumps(record)
    assert record["backend"] == "isaacsim"
    assert record["sim_time_s"] == 2.5
    assert record["stop_class"] == "attached_payload"
    assert record["attached_object_ids"] == ["cracker_box"]
    assert record["max_penetration_m"] == 0.011
    assert record["robot_world_contacts"] == pairs
    assert record["nearest_robot_world_pairs"] is None
    assert record["contacts_caveat"] == CONTACTS_CAVEAT
    assert record["robot_joint_state"] == {
        "name": ["panda_joint1"],
        "position": [0.1],
        "velocity": [0.0],
        "stamp_ns": 7,
    }
    assert record["evidence_voxel"] == {"index": 12, "resolution": 0.05}


def test_an_empty_reply_records_no_penetration_not_a_crash() -> None:
    record = isaac_estop_ground_truth_record({})
    assert record["robot_world_contacts"] == []
    assert record["max_penetration_m"] == 0.0
    assert record["sim_time_s"] is None
    assert record["stop_class"] == "robot_world"
