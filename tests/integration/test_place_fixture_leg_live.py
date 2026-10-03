# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: the real place producer leg verifies a unit fixture against the map.

Real pick-and-place design §2.3 (drafted ADR-0097 / ADR-0092 D6 amendments). Real rclpy,
a real ``VisionAttachmentBridge`` on the bimanual OpenArm manifest with
``place_fixture_enabled`` and the test unit's surveyed ``cell:shelf_top``
(``tests/unit/fixtures/robot_units/openarm_shelf_cell.yaml`` through the real
``load_robot_unit``), real tf2 (the Thor cell's measured ZED mount and the left hand),
real ``openral_msgs`` on the wire: a ``PlaceDeclaration`` on
``/openral/place_declaration``, an ``OccupancyVoxels`` grid on ``/openral/world_voxels``,
and a position stall on the left gripper so the bridge attaches its fallback box (no depth frame:
the conservative jaw-span box). Every ``/openral/attachment_state`` goes through a real
``WorldStateAggregator`` and the World State node's own ``build_world_state_stamped_msg``
into a real ``safety_kernel_node`` (the real cell's parameters, attached check on).

Asserts: the declaration rides region-less while the grid has no face (and the kernel's
``place_region`` diagnostic stays ``-``); with the face mapped the envelope carries the
fixture box as the region with the specified ``evidence_ref`` and the kernel arms it
(``live:cell:shelf_top:geom=0``); the hand lowered until the box rests on the face →
the object carries the ``declared_fixture`` support witness the kernel accepts; the grid
going silent retracts the region (``grid_stale``) and the witness; the aggregator never
raises.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``. Locally::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    PYTHONPATH=$PWD/packages/world_state:$PYTHONPATH OPENRAL_TEST_ROS_LIVE=1 \\
        pytest tests/integration/test_place_fixture_leg_live.py
"""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from typing import Any

import numpy as np
import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy node + colcon openral_msgs / safety kernel overlay — set "
    "OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_ROBOT_YAML = _REPO / "robots" / "openarm" / "robot.yaml"
_SHELF_UNIT = _REPO / "tests" / "unit" / "fixtures" / "robot_units" / "openarm_shelf_cell.yaml"
_BASE = "openarm_base"
_LINK7 = "openarm_left_link7"
_RES = 0.02
_FACE_Z = 0.31  # cell:shelf_top's top face in openarm_base
_ORIGIN = (0.20, -0.46, 0.21)  # cell centres at z = 0.22 + k * 0.02: k=4 is the shelf layer
_SIZE = (26, 46, 12)


def _wait_until(predicate: Any, *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _grid(stamp: Any, *, face: bool) -> Any:
    """The shelf's one cell layer (centres 1 cm under the face) when ``face``, else empty."""
    from openral_msgs.msg import OccupancyVoxels

    sx, sy, sz = _SIZE
    k, j, i = np.meshgrid(np.arange(sz), np.arange(sy), np.arange(sx), indexing="ij")
    centres = (np.stack((i, j, k), axis=-1).reshape(-1, 3) + 0.5) * _RES + np.asarray(_ORIGIN)
    occupied = (
        face
        & (np.abs(centres[:, 0] - 0.45) <= 0.20 + 1e-9)
        & (np.abs(centres[:, 1]) <= 0.40 + 1e-9)
        & (np.abs(centres[:, 2] - (_FACE_Z - _RES / 2)) < 1e-6)
    )
    msg = OccupancyVoxels()
    msg.header.frame_id = _BASE
    msg.header.stamp = stamp
    msg.source_stamp = stamp
    msg.origin.x, msg.origin.y, msg.origin.z = _ORIGIN
    msg.orientation.w = 1.0
    msg.resolution = _RES
    msg.size_x, msg.size_y, msg.size_z = _SIZE
    msg.occupancy = occupied.astype(np.uint8).tolist()
    return msg


def _declaration_msg(stamp_ns: int) -> Any:
    from openral_core import PlaceDeclaration
    from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg

    msg = PlaceDeclarationMsg()
    PlaceDeclaration(
        target_id="cell:shelf_top",
        rskill_id="openral/itest-place-fixture",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        timeout_s=120.0,
        stamp_ns=stamp_ns,
    ).fill_idl(msg)
    return msg


def test_place_fixture_leg_verifies_attests_and_retracts(
    tmp_path: pathlib.Path,
) -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    world_state_ros = pytest.importorskip("openral_world_state_ros.lifecycle_node")

    from geometry_msgs.msg import TransformStamped
    from openral_core import (
        Action,
        AttachedCollisionObject,
        AttachmentEvidenceKind,
        ControlMode,
        JointState,
        PlaceDeclaration,
        PlaceRegion,
        RobotDescription,
        load_robot_unit,
    )
    from openral_hal.vision_attachment_bridge import (
        VisionAttachmentBridge,
        VisionAttachmentConfig,
    )
    from openral_msgs.msg import AttachmentState, OccupancyVoxels, WorldStateStamped
    from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg
    from openral_world_state import WorldStateAggregator
    from rcl_interfaces.msg import Log
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

    from tests.integration.test_grasp_target_leg_live import _thor_mount
    from tests.integration.test_safety_kernel_grasp_target_band import _kernel_params
    from tests.integration.test_vision_attachment_optical_frame_live import _tf_msg
    from tests.sim.safety._kernel_subprocess import (
        activate_kernel_node,
        isolated_domain_id,
        start_kernel,
        terminate_kernel,
    )

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))
    robot_dir = tmp_path / "openarm"
    (robot_dir / "units").mkdir(parents=True)
    shutil.copy(_ROBOT_YAML, robot_dir / "robot.yaml")
    shutil.copy(_SHELF_UNIT, robot_dir / "units" / "shelf_cell.yaml")
    unit = load_robot_unit(robot_dir / "robot.yaml", "shelf_cell")
    span = 0.05  # VisionGateConfig.jaw_span_m: the fallback box's half-extent

    kernel_name = f"safety_kernel_place_fixture_{uuid.uuid4().hex[:8]}"
    params = {
        **_kernel_params(grasp_allowance_enabled=False),
        "attached_collision_enabled": True,
        "world_voxel_max_cells": int(np.prod(_SIZE)),
    }
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        proc = start_kernel(
            params,
            kernel_name,
            isolated_domain_id(),
            log_path=log_path,
            params_file=pathlib.Path(td) / "kernel_params.yaml",
        )
        try:
            time.sleep(1.5)
            rclpy.init()
            node = Node("test_place_fixture_hal")
            peer = Node("test_place_fixture_peer")
            assert activate_kernel_node(kernel_name, peer), "kernel activation failed"

            logs: list[str] = []
            peer.create_subscription(Log, "/rosout", lambda m: logs.append(str(m.msg)), 500)
            diagnostics: dict[str, str] = {}

            def on_diagnostics(msg: Any) -> None:
                for status in msg.status:
                    for kv in status.values:
                        if kv.key == "place_region":
                            diagnostics["place_region"] = kv.value

            from diagnostic_msgs.msg import DiagnosticArray

            peer.create_subscription(DiagnosticArray, "/diagnostics", on_diagnostics, 10)

            aggregator = WorldStateAggregator(description)
            aggregator_errors: list[str] = []
            envelopes: list[Any] = []
            reliable_kl1 = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=1)
            state_pub = peer.create_publisher(
                WorldStateStamped, "/openral/world_state_fast", reliable_kl1
            )

            def on_state(msg: Any) -> None:
                envelopes.append(msg)
                stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
                try:
                    aggregator.update_attached_objects(
                        [AttachedCollisionObject.from_idl(item) for item in msg.objects],
                        revision=int(msg.revision),
                        stamp_ns=stamp_ns,
                        place_declaration=(
                            PlaceDeclaration.from_idl(msg.place_declaration)
                            if msg.place_declaration_valid
                            else None
                        ),
                    )
                except ValueError as exc:
                    aggregator_errors.append(str(exc))
                    return
                state_pub.publish(
                    world_state_ros.build_world_state_stamped_msg(peer, aggregator.snapshot())
                )

            latched = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            )
            peer.create_subscription(
                AttachmentState, "/openral/attachment_state", on_state, latched
            )
            declaration_pub = peer.create_publisher(
                PlaceDeclarationMsg, "/openral/place_declaration", latched
            )
            voxel_pub = peer.create_publisher(
                OccupancyVoxels, "/openral/world_voxels", reliable_kl1
            )
            StaticTransformBroadcaster(peer).sendTransform(
                [_tf_msg(_BASE, "zed_camera_link", _thor_mount())]
            )
            hand_broadcaster = TransformBroadcaster(peer)
            grid_stamps: list[int] = []
            # The fallback box's bottom starts well above the free volume (face + 0.10 m).
            state: dict[str, Any] = {
                "grid": True,
                "face": False,
                "loaded": False,
                "bottom": 0.50,
                "box_in_link": np.zeros(3),  # the attachment's pose_in_link, once attached
            }

            def publish_hand() -> None:
                """link7 posed so the fallback box's bottom sits at ``state["bottom"]``."""
                xyz = np.array([0.45, 0.0, state["bottom"] + span]) - state["box_in_link"]
                hand = TransformStamped()
                hand.header.frame_id = _BASE
                hand.header.stamp = peer.get_clock().now().to_msg()
                hand.child_frame_id = _LINK7
                t = hand.transform.translation
                t.x, t.y, t.z = (float(v) for v in xyz)
                hand.transform.rotation.w = 1.0
                hand_broadcaster.sendTransform(hand)

            bridge = VisionAttachmentBridge(
                node,
                description,
                config=VisionAttachmentConfig(
                    camera="head_zed",
                    depth_topic="/test_place_fixture/depth",  # never published
                    service_name="/openral/perception/segment_in_view_place_fixture_itest",
                    place_fixture_enabled=True,
                    unit_fixtures=unit.fixtures,
                    robot_unit=unit.unit,
                ),
            )

            def feed() -> None:
                # Loaded: commanded closed, the left jaw stalls 0.2 rad short (position
                # stall). Released: commanded open, the jaw opens past its hold.
                bridge.observe_command(
                    Action(
                        control_mode=ControlMode.JOINT_POSITION,
                        horizon=1,
                        joint_names=["left_gripper", "right_gripper"],
                        joint_targets=[[0.0 if state["loaded"] else 0.7, 0.0]],
                    )
                )
                bridge.observe_joint_state(
                    JointState(
                        name=["left_gripper", "right_gripper"],
                        position=[0.2 if state["loaded"] else 0.5, -0.0116],
                        effort=[0.0, 0.0],  # the real driver's hard-coded zeros
                        stamp_ns=time.monotonic_ns(),
                    )
                )

            def publish_grid() -> None:
                if state["grid"]:
                    stamp = peer.get_clock().now()
                    grid_stamps.append(int(stamp.nanoseconds))
                    voxel_pub.publish(_grid(stamp.to_msg(), face=state["face"]))

            bridge.setup()
            node.create_timer(1.0 / 30.0, feed)
            peer.create_timer(0.2, publish_grid)
            peer.create_timer(0.05, publish_hand)  # as robot_state_publisher does
            executor = MultiThreadedExecutor()
            executor.add_node(node)
            executor.add_node(peer)
            spin = threading.Thread(target=executor.spin, daemon=True)
            spin.start()

            def latest() -> Any:
                return envelopes[-1] if envelopes else None

            def region_valid() -> bool:
                msg = latest()
                return bool(
                    msg is not None
                    and msg.place_declaration_valid
                    and msg.place_declaration.region_valid
                )

            def place_lines() -> list[str]:
                return [line for line in logs if "place_fixture" in line]

            try:
                # ── 1. No face in the map: the declaration rides region-less. ─────
                declaration_pub.publish(_declaration_msg(int(node.get_clock().now().nanoseconds)))
                state["loaded"] = True  # ATTACH → the fallback box (no depth frame)
                assert _wait_until(
                    lambda: (
                        latest() is not None
                        and latest().place_declaration_valid
                        and len(latest().objects) == 1
                    ),
                    timeout_s=10.0,
                ), f"no declaration + payload on the envelope; {place_lines()}"
                assert _wait_until(
                    lambda: any("reason=face_missing" in line for line in place_lines()),
                    timeout_s=5.0,
                ), place_lines()
                assert not region_valid()
                assert latest().place_declaration.target_id == "cell:shelf_top"
                assert _wait_until(lambda: diagnostics.get("place_region") == "-", timeout_s=5.0)

                # ── 2. The face mapped: the fixture box is the region; the kernel arms it.
                state["face"] = True
                assert _wait_until(region_valid, timeout_s=5.0), place_lines()
                declared = latest().place_declaration
                region = PlaceRegion.from_idl(declared.region)
                fixture = unit.fixture("cell:shelf_top")
                assert region.frame_id == _BASE
                assert region.pose.xyz == pytest.approx(fixture.pose.xyz)
                assert region.half_extents == pytest.approx(fixture.half_extents)
                assert region.geometry == ()
                assert region.stamp_ns in grid_stamps, "region stamp is not a grid stamp"
                assert region.evidence_ref == (
                    f"unit_fixture:shelf_cell/cell:shelf_top@2026-10-02;"
                    f"map_verified@{region.stamp_ns}"
                )
                assert declared.rskill_id == "openral/itest-place-fixture"
                assert latest().objects[0].support_contact_valid is False
                assert _wait_until(
                    lambda: diagnostics.get("place_region") == "live:cell:shelf_top:geom=0",
                    timeout_s=5.0,
                ), diagnostics

                # ── 3. The box rests on the face: the declared_fixture witness. ───
                # Read the box's offset in link7 off the wire (no depth frame: the bridge
                # puts it wherever its fallback does), then lower the hand onto the face.
                payload = AttachedCollisionObject.from_idl(latest().objects[0])
                state["box_in_link"] = np.asarray(payload.pose_in_link.xyz)
                state["bottom"] = _FACE_Z + 0.004
                assert _wait_until(
                    lambda: latest().objects and latest().objects[0].support_contact_valid,
                    timeout_s=5.0,
                ), place_lines()
                witness = AttachedCollisionObject.from_idl(latest().objects[0]).support_contact
                assert witness is not None
                assert witness.support_id == "cell:shelf_top"
                assert witness.evidence_kind is AttachmentEvidenceKind.DECLARED_FIXTURE
                assert "not sensed contact" in (witness.evidence_ref or "")
                assert witness.max_penetration_m == pytest.approx(0.01)
                assert 0.0 < witness.patch_radius_m <= 0.5
                assert witness.contact_normal_in_object == pytest.approx((0.0, 0.0, 1.0))
                assert any("not sensed contact" in line for line in place_lines())
                assert _wait_until(
                    lambda: (
                        "safety.support_witness_armed object=grasped_payload:left_gripper "
                        "support=cell:shelf_top" in log_path.read_text(errors="replace")
                    ),
                    timeout_s=5.0,
                ), "the kernel did not accept the witness"
                assert region_valid(), f"the resting payload failed its own check: {place_lines()}"

                # ── 4. The grid stops: region and witness retracted. ──────────────
                state["grid"] = False
                assert _wait_until(lambda: not region_valid(), timeout_s=3.0), (
                    "region outlived the grid"
                )
                assert latest().place_declaration_valid, "the declaration itself stays"
                assert any("reason=grid_stale" in line for line in place_lines())
                assert _wait_until(
                    lambda: not latest().objects[0].support_contact_valid, timeout_s=2.0
                ), "witness outlived the verification"
                assert _wait_until(lambda: diagnostics.get("place_region") == "-", timeout_s=5.0), (
                    diagnostics
                )
                assert aggregator_errors == []
            finally:
                with suppress(Exception):
                    bridge.teardown()
                executor.shutdown()
                spin.join(timeout=5.0)
                node.destroy_node()
                peer.destroy_node()
                rclpy.shutdown()
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')[-6000:]}"
            ) from exc
        finally:
            terminate_kernel(proc)
