# SPDX-License-Identifier: Apache-2.0
"""The real safety kernel stops on an obstacle anywhere in the A1 fingers' stroke.

The Galaxea A1's fingers have no joint state: the kernel poses them closed, and
``FixedAttachment.sweep_xyz`` makes their lowered geometry cover the whole 30 mm
travel. This drives the real ``safety_kernel_node`` with the parameters the real
manifest lowers to (``collision_params_from_description``, world voxels on), at the
rest pose, against a single occupied voxel:

* inside the closed finger: the chunk is dropped, a world collision on the finger;
* reached only by the fully open finger (a point the closed finger's own mesh is
  more than 15 mm from): dropped the same way;
* an all-free grid first, in both runs: the chunk passes, so the drop is the voxel.

Gates: ROS_DISTRO + rclpy + openral_msgs on a colcon-built workspace, and the pinned
vendor URDF (skips when unfetchable).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.environ.get("ROS_DISTRO"), reason="ROS_DISTRO not set — needs a sourced ROS 2."
    ),
]
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")
pytest.importorskip("yourdfpy", reason="URDF lowering extras not installed")

from openral_core import LinkCollisionGeometry, RobotDescription  # noqa: E402
from openral_core.exceptions import ROSConfigError  # noqa: E402
from openral_core.kinematics import transform_from_xyz_rpy  # noqa: E402
from openral_safety.envelope_loader import (  # noqa: E402
    collision_params_from_description,
    compute_intersection,
    kernel_params_from_envelope,
)
from openral_safety.urdf_lowering import _link_clouds  # noqa: E402

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

_MANIFEST = Path(__file__).resolve().parents[3] / "robots" / "galaxea_a1" / "robot.yaml"
_RES = 0.005  # one 5 mm voxel


def _kernel_params(desc: RobotDescription) -> dict[str, object]:
    params: dict[str, object] = dict(kernel_params_from_envelope(compute_intersection(desc, None)))
    params.update(collision_params_from_description(desc))
    params.update(
        {
            "world_voxel_enabled": True,
            "world_voxel_margin_m": 0.0,
            "world_voxel_deadline_ms": 2000.0,
            "world_voxel_max_cells": 27,
            "collision_joint_names": [j.name for j in desc.joints],
            "collision_seed_dt_s": 0.0,
            "collision_state_deadline_ms": 5000.0,
        }
    )
    return params


def _finger_in_base(params: dict[str, object], link: str) -> NDArray[np.float64]:
    """The finger link's pose in the base frame at the rest pose (every joint 0)."""
    names = list(params["collision_link_names"])  # type: ignore[call-overload]  # reason: param dict values are untyped
    parent = list(params["collision_parent"])  # type: ignore[call-overload]  # reason: as above
    origins = np.asarray(params["collision_origin_xyzrpy"], dtype=np.float64).reshape(-1, 6)
    pose = np.eye(4)
    i = names.index(link)
    while i >= 0:
        o = origins[i]
        pose = np.asarray(transform_from_xyz_rpy(o[:3], o[3:]), dtype=np.float64) @ pose
        i = parent[i]
    return pose


def _obstacles(desc: RobotDescription, params: dict[str, object]) -> dict[str, NDArray[np.float64]]:
    """Base-frame points: inside the closed finger, and reached only by the open one."""
    from openral_hal._galaxea_a1_assets import ensure_a1_urdf

    try:
        urdf = ensure_a1_urdf()
    except ROSConfigError as exc:
        pytest.skip(f"Galaxea A1 vendor files not fetchable on this host: {exc}")
    (closed,) = [
        np.asarray(c.vertices, dtype=np.float64)
        for ln, c in _link_clouds(str(urdf), {}, None)
        if ln == "gripper1" and not isinstance(c, LinkCollisionGeometry)
    ]
    (sweep,) = [a.sweep_xyz for a in desc.fixed_attachments if a.child_link == "gripper1"]
    assert sweep is not None
    travel = np.asarray(sweep)
    tip = closed[np.argmax(closed @ travel)] + travel - 0.5 * _RES * travel / np.linalg.norm(travel)
    assert np.min(np.linalg.norm(closed - tip, axis=1)) > 0.015, "the tip is not open-only"
    place = _finger_in_base(params, "gripper1")

    def to_base(p: NDArray[np.float64]) -> NDArray[np.float64]:
        placed: NDArray[np.float64] = (place @ np.r_[p, 1.0])[:3]
        return placed

    return {"closed": to_base(closed.mean(axis=0)), "open_only": to_base(tip)}


@pytest.mark.parametrize("where", ["closed", "open_only"])
def test_kernel_stops_on_an_obstacle_in_the_finger_stroke(where: str) -> None:
    import rclpy
    from geometry_msgs.msg import Point, Quaternion
    from openral_msgs.msg import ActionChunk, FailureTrigger, OccupancyVoxels
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty

    desc = RobotDescription.from_yaml(str(_MANIFEST))
    params = _kernel_params(desc)
    point = _obstacles(desc, params)[where]
    node_name = f"safety_kernel_a1_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(params, node_name, isolated_domain_id())
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("a1_finger_sweep_helper")
            assert activate_kernel_node(node_name, helper), "kernel activation failed"
            safe: dict[str, ActionChunk] = {}
            failures: list[FailureTrigger] = []
            estops: list[Empty] = []
            helper.create_subscription(
                ActionChunk, "/openral/safe_action", lambda m: safe.__setitem__(m.trace_id, m), 10
            )
            helper.create_subscription(
                FailureTrigger, "/openral/failure/safety", failures.append, 50
            )
            helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
            pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
            voxel_pub = helper.create_publisher(
                OccupancyVoxels,
                "/openral/world_voxels",
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )
            js_pub = helper.create_publisher(
                JointState,
                "/joint_states",
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            )
            executor = SingleThreadedExecutor()
            executor.add_node(helper)
            deadline = time.time() + 5.0
            while time.time() < deadline and pub.get_subscription_count() < 1:
                executor.spin_once(timeout_sec=0.05)

            js = JointState()
            js.name = [j.name for j in desc.joints]
            js.position = [0.0] * len(desc.joints)
            grid = OccupancyVoxels()
            grid.header.frame_id = desc.base_frame
            corner = point - 1.5 * _RES  # 3x3x3 lattice, the occupied cell centred on the point
            grid.origin = Point(x=float(corner[0]), y=float(corner[1]), z=float(corner[2]))
            grid.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            grid.resolution = _RES
            grid.size_x = grid.size_y = grid.size_z = 3

            def pump(seconds: float, occupied: bool) -> None:
                grid.occupancy = [0] * 27
                if occupied:
                    grid.occupancy[13] = 1  # (1, 1, 1)
                end = time.time() + seconds
                while time.time() < end:
                    js.header.stamp = grid.header.stamp = helper.get_clock().now().to_msg()
                    grid.source_stamp = grid.header.stamp
                    js_pub.publish(js)
                    voxel_pub.publish(grid)
                    executor.spin_once(timeout_sec=0.02)

            def send(trace: str) -> None:
                chunk = ActionChunk()
                chunk.control_mode = 0  # JOINT_POSITION
                chunk.horizon, chunk.n_dof = 1, len(desc.joints)
                chunk.flat = [0.0] * len(desc.joints)
                chunk.rskill_id, chunk.trace_id = "openral/a1-finger-sweep", trace
                pub.publish(chunk)

            pump(1.0, occupied=False)
            send("free")
            pump(1.2, occupied=False)
            assert "free" in safe, "the rest pose must pass an all-free grid"
            assert not estops

            pump(0.6, occupied=True)
            send("obstacle")
            pump(1.2, occupied=True)
            assert "obstacle" not in safe, f"the {where} obstacle reached safe_action"
            assert estops, f"the {where} obstacle must fire /openral/estop"
            evidence = json.loads(failures[-1].evidence_json)
            assert failures[-1].kind == FailureTrigger.KIND_COLLISION
            assert evidence["collision_kind"] == "world"
            assert evidence["link_a"] in {"gripper1", "gripper2", "arm_seg6"}
            if where == "open_only":
                assert evidence["link_a"] == "gripper1", evidence
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(proc)
