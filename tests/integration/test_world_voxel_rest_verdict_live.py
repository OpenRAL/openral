"""``tools/world_voxel_rest_verdict.py`` judges a recorded map at rest with the real kernel.

The tool exists because the kernel's world-voxel check only runs on a chunk it judges:
an idle cell never exercises it. Issue #356 put the OpenArm torso in the kernel model,
and its foot-plate slab reaches 1 mm below the surface the robot stands on. Whether the
real map then refuses a hold at rest is the open question; this test pins the tool that
answers it, end to end on real parts (CLAUDE.md §1.11):

- the kernel parameters come from composing the real ``openarm_real_world_voxels.yaml``
  graph (``hal_mode=real``, unit ``thor``), as ``openral deploy run`` would;
- the bags are real rosbag2 (mcap) files, written by ``rosbag2_py``'s writer, holding an
  ``OccupancyVoxels`` grid and a ``/joint_states`` sample named by the vendor
  (``sim_joint_name``) spelling, as a twin-pass recording carries them;
- the verdict is the real ``safety_kernel_node``'s.

The grid is a flat support surface at the torso's foot level (``z = -0.698`` in
``openarm_base``), on the octree's own lattice (2 cm cells, boundaries on multiples of the
resolution), with the robot self-filter applied the way it applies it: a surface point is
dropped when it lies within the filter's padding of a torso hull, and a cell is occupied
when any surface point in it survives. That is the cell's map if its camera sees the
surface next to the foot plate. The control grid is the same surface 0.10 m lower, clear
of every link.

Gates: ``OPENRAL_TEST_ROS_LIVE=1`` + ROS_DISTRO + rclpy + openral_msgs + colcon-built kernel.
``scripts/ros_live_tests.sh`` runs it (``tests/unit/test_ros_live_targets.py`` keeps it listed).
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

_LIVE = os.environ.get("OPENRAL_TEST_ROS_LIVE") == "1" and bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="live-ROS test — set OPENRAL_TEST_ROS_LIVE=1 with ROS 2 sourced "
    "(scripts/ros_live_tests.sh does both).",
)
if _LIVE:
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    pytest.importorskip("rosbag2_py")
    pytest.importorskip("trimesh")

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SCENE = _REPO / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"
_TORSO = "openarm_body_link0"
_RES = 0.02
_SURFACE_Z = -0.698  # the torso's foot in openarm_base (manifest torso mount)


def _tool() -> ModuleType:
    path = _REPO / "tools" / "world_voxel_rest_verdict.py"
    spec = importlib.util.spec_from_file_location("world_voxel_rest_verdict", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _tool()


@pytest.fixture(scope="module")
def node_params(tool: ModuleType) -> dict[str, dict[str, Any]]:
    return tool.deploy_node_params(_SCENE, "thor")


def _torso_hulls() -> list[Any]:
    """Each torso slab's exact hull as a mesh in openarm_base (the torso is static)."""
    import trimesh
    from openral_core import RobotDescription

    from tests.unit.test_collision_geometry_zero_pose import _link_poses, _tf

    robot = RobotDescription.from_yaml(str(_REPO / "robots" / "openarm" / "robot.yaml"))
    link = _link_poses(robot, {j.name: 0.0 for j in robot.joints})[_TORSO]
    hulls = []
    for g in robot.collision_geometry:
        if g.link_name != _TORSO:
            continue
        assert g.tight_geometry is not None
        box = link @ _tf(g.origin_xyz_rpy[:3], g.origin_xyz_rpy[3:])
        v = np.asarray(g.tight_geometry.hull_vertices_m, dtype=float)
        hulls.append(trimesh.convex.convex_hull(v @ box[:3, :3].T + box[:3, 3]))
    assert len(hulls) == 3
    return hulls


def _surface_grid(z_surface: float, padding_m: float, frame: str) -> Any:
    """A support surface at ``z_surface`` on the octree lattice, self-filtered at ``padding_m``."""
    import trimesh
    from openral_msgs.msg import OccupancyVoxels

    half = 0.40
    lo = np.array([-half, -half, np.floor(z_surface / _RES) * _RES - _RES])
    cells = round(2 * half / _RES)
    size = np.array([cells, cells, 3])
    occ = np.zeros(size[::-1], dtype=np.uint8)  # [z, y, x]
    k = int(np.floor((z_surface - lo[2]) / _RES))
    hulls = _torso_hulls()
    near = np.array([[-0.156 - 0.06, -0.096 - 0.06], [0.096 + 0.06, 0.096 + 0.06]])
    sub = (np.arange(10) + 0.5) / 10 * _RES  # 2 mm surface samples per cell side
    for iy in range(size[1]):
        for ix in range(size[0]):
            x0, y0 = lo[0] + ix * _RES, lo[1] + iy * _RES
            if not (
                near[0, 0] < x0 + _RES
                and x0 < near[1, 0]
                and near[0, 1] < y0 + _RES
                and y0 < near[1, 1]
            ):
                occ[k, iy, ix] = 1  # far from the torso: every surface point survives
                continue
            gx, gy = np.meshgrid(x0 + sub, y0 + sub)
            pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z_surface)])
            dist = np.full(len(pts), np.inf)
            for hull in hulls:
                d = -trimesh.proximity.signed_distance(hull, pts)  # > 0 outside
                dist = np.minimum(dist, d)
            occ[k, iy, ix] = 1 if bool((dist > padding_m).any()) else 0
    grid = OccupancyVoxels()
    grid.header.frame_id = frame
    grid.origin.x, grid.origin.y, grid.origin.z = (float(v) for v in lo)
    grid.orientation.w = 1.0
    grid.resolution = _RES
    grid.size_x, grid.size_y, grid.size_z = (int(v) for v in size)
    grid.occupancy = occ.ravel().tolist()
    return grid


def _write_bag(path: pathlib.Path, grid: Any, joint_state: Any) -> pathlib.Path:
    import rosbag2_py
    from rclpy.serialization import serialize_message

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for i, (topic, msg_type, msg) in enumerate(
        (
            ("/openral/world_voxels", "openral_msgs/msg/OccupancyVoxels", grid),
            ("/joint_states", "sensor_msgs/msg/JointState", joint_state),
        )
    ):
        writer.create_topic(
            rosbag2_py.TopicMetadata(id=i, name=topic, type=msg_type, serialization_format="cdr")
        )
        writer.write(topic, serialize_message(msg), 1_000_000_000 + i)
    del writer
    return path


def _vendor_joint_state(node_params: dict[str, dict[str, Any]]) -> Any:
    from sensor_msgs.msg import JointState

    names = list(node_params["kernel"]["collision_joint_names"])
    aliases = list(node_params["self_filter"]["collision_joint_aliases"])
    js = JointState()
    js.name = [a or n for n, a in zip(names, aliases, strict=True)]
    js.position = [0.0] * len(names)
    return js


def test_the_deploy_graph_hands_the_kernel_the_real_cell_values(
    node_params: dict[str, dict[str, Any]],
) -> None:
    kernel = node_params["kernel"]
    assert kernel["world_voxel_enabled"] is True
    assert kernel["world_voxel_margin_m"] == pytest.approx(0.02)
    assert _TORSO in kernel["collision_link_names"]
    assert node_params["self_filter"]["padding_m"] == pytest.approx(0.02)


def test_a_surface_the_torso_stands_on_refuses_a_hold_at_rest(
    tool: ModuleType, node_params: dict[str, dict[str, Any]], tmp_path: pathlib.Path
) -> None:
    grid = _surface_grid(_SURFACE_Z, node_params["self_filter"]["padding_m"], "openarm_base")
    js = _vendor_joint_state(node_params)
    bag = _write_bag(tmp_path / "surface", grid, js)
    assert tool.main(["--bag", str(bag), "--scene", str(_SCENE), "--unit", "thor"]) == 1
    messages = tool.read_last_messages(bag, ["/openral/world_voxels", "/joint_states"])
    row = tool.rest_row(
        list(node_params["kernel"]["collision_joint_names"]),
        list(node_params["self_filter"]["collision_joint_aliases"]),
        messages["/joint_states"],
        "measured",
    )
    verdict = tool.judge_hold_row(
        node_params["kernel"],
        messages["/openral/world_voxels"],
        row,
        log_path=tmp_path / "kernel.log",
    )
    assert verdict.outcome == "refused", verdict
    assert verdict.evidence["collision_kind"] == "world"
    assert verdict.evidence["link_a"] == _TORSO, verdict.evidence
    assert verdict.cell_centre_m is not None
    assert verdict.cell_centre_m[2] == pytest.approx(-0.69, abs=1e-9)  # the surface's cell layer


def test_the_same_surface_lower_down_is_accepted(
    tool: ModuleType, node_params: dict[str, dict[str, Any]], tmp_path: pathlib.Path
) -> None:
    grid = _surface_grid(_SURFACE_Z - 0.10, node_params["self_filter"]["padding_m"], "openarm_base")
    js = _vendor_joint_state(node_params)
    bag = _write_bag(tmp_path / "lower", grid, js)
    assert tool.main(["--bag", str(bag), "--scene", str(_SCENE), "--unit", "thor"]) == 0
