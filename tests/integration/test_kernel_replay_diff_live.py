"""Live: ``tools/kernel_replay_diff.py`` replays a real bag into two real safety kernels.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` (a sourced overlay with ``openral_safety_kernel`` built). A
synthetic MCAP bag holds the grasp-rig scene (one finger capsule on j0, one occupied 25 mm cell
it sweeps into): 60 candidates alternating clear and on-cell. Two kernels with the same
parameters must agree on every verdict; a kernel B with world collision off must diverge at the
first on-cell candidate, so the tool is not a rubber stamp. The tool must leave no kernel behind.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENRAL_TEST_ROS_LIVE"),
    reason="live kernels: set OPENRAL_TEST_ROS_LIVE=1 and source install/",
)

_PARAMS: dict[str, Any] = {
    "n_dof": 1,
    "joint_position_min": [-3.14],
    "joint_position_max": [3.14],
    "joint_velocity_max": [3.15],
    "joint_torque_max": [5.0],
    "self_collision_enabled": False,
    "world_voxel_enabled": True,
    "world_voxel_margin_m": 0.0,
    "world_voxel_deadline_ms": 2000.0,
    "world_voxel_max_cells": 64,
    "attached_collision_enabled": False,
    "collision_n_links": 1,
    "collision_parent": [-1],
    "collision_joint_kind": [1],
    "collision_dof_index": [0],
    "collision_origin_xyzrpy": [0.0] * 6,
    "collision_axis": [0.0, 0.0, 1.0],
    "collision_capsule_link": [0],
    "collision_capsule_radius": [0.01],
    "collision_capsule_half_length": [0.0],
    "collision_capsule_origin_xyzrpy": [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
    "collision_allowed_pairs": [],
    "collision_link_names": ["link0"],
    "collision_joint_names": ["j0"],
    "collision_state_deadline_ms": 5000.0,
    "hal_mode": "sim",
}


def _tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "kernel_replay_diff_live", REPO_ROOT / "tools" / "kernel_replay_diff.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_bag(path: Path) -> None:
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from openral_msgs.msg import ActionChunk, OccupancyVoxels
    from rclpy.serialization import serialize_message
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for i, (name, typ) in enumerate(
        (
            ("/clock", "rosgraph_msgs/msg/Clock"),
            ("/joint_states", "sensor_msgs/msg/JointState"),
            ("/openral/world_voxels", "openral_msgs/msg/OccupancyVoxels"),
            ("/openral/candidate_action", "openral_msgs/msg/ActionChunk"),
        )
    ):
        writer.create_topic(
            rosbag2_py.TopicMetadata(id=i, name=name, type=typ, serialization_format="cdr")
        )
    for k in range(60):
        ns = 1_000_000_000 + k * 50_000_000
        clock = Clock()
        clock.clock.sec, clock.clock.nanosec = ns // 10**9, ns % 10**9
        writer.write("/clock", serialize_message(clock), ns)
        q = 1.5707963 if (k // 10) % 2 == 0 else 0.0  # ten clear, ten on the cell, ...
        js = JointState()
        js.name, js.position, js.header.stamp = ["j0"], [q], clock.clock
        writer.write("/joint_states", serialize_message(js), ns)
        vox = OccupancyVoxels()
        vox.orientation.w = 1.0
        vox.header.frame_id = "base_link"
        vox.header.stamp = vox.source_stamp = clock.clock
        vox.resolution = 0.025
        vox.size_x = vox.size_y = vox.size_z = 1
        vox.origin.x, vox.origin.y, vox.origin.z = 0.0375, -0.0125, -0.0125
        vox.occupancy = [1]
        writer.write("/openral/world_voxels", serialize_message(vox), ns)
        chunk = ActionChunk()
        chunk.control_mode, chunk.horizon, chunk.n_dof, chunk.flat = 0, 1, 1, [q]
        chunk.rskill_id = "pick"
        writer.write("/openral/candidate_action", serialize_message(chunk), ns + 1)
    del writer


def _kernels_left() -> int:
    out = subprocess.run(
        ["pgrep", "-fc", "safety_kernel_nod[e]"], capture_output=True, text=True, check=False
    )
    return int(out.stdout.strip() or 0)


def test_identical_kernels_agree_and_a_different_one_is_caught(tmp_path: Path) -> None:
    pytest.importorskip("rclpy")
    krd = _tool()
    bag = tmp_path / "bag"
    _write_bag(bag)
    before = _kernels_left()

    same = krd.replay(bag, _PARAMS, _PARAMS, tmp_path / "same")
    assert same["candidates"] == 60
    assert same["different"] == 0, same["diffs"][:3]

    off = dict(_PARAMS, world_voxel_enabled=False)
    diff = krd.replay(bag, _PARAMS, off, tmp_path / "diff")
    assert diff["different"] > 0
    first = diff["diffs"][0]
    assert first["index"] == 10, "the first on-cell candidate"
    assert not first["a"]["passed"] and first["a"]["latched"]
    assert first["b"]["passed"] and not first["b"]["latched"]
    assert _kernels_left() == before, "every replay kernel is stopped"
