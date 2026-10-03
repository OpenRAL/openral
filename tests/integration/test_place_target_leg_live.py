# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: the real place producer leg measures the surface under the carried payload.

Real pick-and-place design §2.3 (drafted ADR-0097 / ADR-0092 D6 amendments). Nothing is
surveyed and nobody names a place target: real rclpy, a real ``VisionAttachmentBridge``
on the bimanual OpenArm manifest with ``place_target_enabled``, real tf2 (the Thor cell's
measured ZED mount and the left hand), real ``openral_msgs`` on the wire: an
``OccupancyVoxels`` grid with a table top on ``/openral/world_voxels`` and a position
stall on the left gripper so the bridge attaches its fallback box (no depth frame: the
conservative jaw-span box). The only ``/openral/place_declaration`` is the runner's
goal-scope one (``place_approach_enabled``: ``target_id = surface``, no object, no box, no
region — exactly ``rskill_runner_node._goal_scope_place_declaration``'s). Every
``/openral/attachment_state`` goes through a real ``WorldStateAggregator`` and the World
State node's own ``build_world_state_stamped_msg`` into a real ``safety_kernel_node``
(the real cell's parameters, attached check on).

Asserts: with no goal, the payload held over the table arms nothing (no declaration on
the envelope, the kernel's ``place_region`` diagnostic ``-``); under the goal, held beyond
the search depth it still arms nothing; lowered to within it, the leg attaches the
surface it measured under the payload to the goal's declaration (scoped to that payload)
as a one-voxel slab on the measured plane and the kernel arms it; lowered until the box
rests on the table → the object carries the ``map_support_proximity`` witness the kernel
accepts; the goal's retraction ends the region and the witness at once and nothing
re-arms while the payload stays held over the table; a new goal re-measures and re-arms
both; a stalled octomap (grids still arriving with a fresh ``header.stamp`` but the same
old ``source_stamp``) ages the region and the witness out on the data's age — never past
the producer's bound, and the kernel (set here to half of it) drops the aging region on
its own as ``region_stale``; the aggregator never raises.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``. Locally::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    PYTHONPATH=$PWD/packages/world_state:$PYTHONPATH OPENRAL_TEST_ROS_LIVE=1 \\
        pytest tests/integration/test_place_target_leg_live.py
"""

from __future__ import annotations

import os
import pathlib
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
_BASE = "openarm_base"
_LINK7 = "openarm_left_link7"
_RES = 0.02
_FACE_Z = 0.31  # the table top's measured face in openarm_base
_ORIGIN = (0.20, -0.46, 0.21)  # cell centres at z = 0.22 + k * 0.02: k=4 is the shelf layer
_SIZE = (26, 46, 12)


def _wait_until(predicate: Any, *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _grid(stamp: Any, *, face: bool, source: Any = None) -> Any:
    """The table's one cell layer (centres 1 cm under the face) when ``face``, else empty."""
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
    msg.source_stamp = stamp if source is None else source
    msg.origin.x, msg.origin.y, msg.origin.z = _ORIGIN
    msg.orientation.w = 1.0
    msg.resolution = _RES
    msg.size_x, msg.size_y, msg.size_z = _SIZE
    msg.occupancy = occupied.astype(np.uint8).tolist()
    return msg


def test_place_target_leg_measures_attests_ages_out_and_retracts_under_the_goal() -> None:
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
    span = 0.05  # VisionGateConfig.jaw_span_m: the fallback box's half-extent

    kernel_name = f"safety_kernel_place_target_{uuid.uuid4().hex[:8]}"
    params = {
        **_kernel_params(grasp_allowance_enabled=False),
        "attached_collision_enabled": True,
        "world_voxel_max_cells": int(np.prod(_SIZE)),
        # Half the producer's 2 s freeze, so the kernel's OWN stale-region drop is seen
        # (the deploy sets both to 2 x the voxel deadline; the kernel is the backstop).
        "place_region_max_age_s": 1.0,
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
            node = Node("test_place_target_hal")
            peer = Node("test_place_target_peer")
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
            voxel_pub = peer.create_publisher(
                OccupancyVoxels, "/openral/world_voxels", reliable_kl1
            )
            StaticTransformBroadcaster(peer).sendTransform(
                [_tf_msg(_BASE, "zed_camera_link", _thor_mount())]
            )
            hand_broadcaster = TransformBroadcaster(peer)
            grid_stamps: list[int] = []
            # The fallback box's bottom starts beyond the search depth (0.20 m) of the face.
            state: dict[str, Any] = {
                "grid": True,
                "face": True,
                "loaded": False,
                "bottom": _FACE_Z + 0.30,
                "box_in_link": np.zeros(3),  # the attachment's pose_in_link, once attached
                "source": None,  # a frozen source_stamp (stalled octree), else fresh
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
                    depth_topic="/test_place_target/depth",  # never published
                    service_name="/openral/perception/segment_in_view_place_target_itest",
                    place_target_enabled=True,
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
                    source = state["source"]  # a stalled octree: the data stamp is frozen
                    if source is None:
                        grid_stamps.append(int(stamp.nanoseconds))
                    voxel_pub.publish(_grid(stamp.to_msg(), face=state["face"], source=source))

            declaration_pub = peer.create_publisher(
                PlaceDeclarationMsg, "/openral/place_declaration", latched
            )
            goal = PlaceDeclaration(  # rskill_runner_node._goal_scope_place_declaration
                target_id="surface",
                rskill_id="openral/itest-place",
                trace_id="itest",
                timeout_s=60.0,
                stamp_ns=0,
            )

            goal_stamp = [0]

            def declare(*, active: bool) -> None:
                """Arm (stamped now, as at goal start) or retract the goal's declaration."""
                if active:
                    goal_stamp[0] = int(peer.get_clock().now().nanoseconds)
                msg = PlaceDeclarationMsg()
                goal.model_copy(update={"stamp_ns": goal_stamp[0], "active": active}).fill_idl(msg)
                declaration_pub.publish(msg)

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
                return [line for line in logs if "place_target" in line]

            try:
                # ── 0. No goal: over the table, nothing arms. ───────────────────
                state["loaded"] = True  # ATTACH → the fallback box (no depth frame)
                assert _wait_until(
                    lambda: latest() is not None and len(latest().objects) == 1, timeout_s=10.0
                ), f"no payload on the envelope; {place_lines()}"
                payload = AttachedCollisionObject.from_idl(latest().objects[0])
                state["box_in_link"] = np.asarray(payload.pose_in_link.xyz)
                state["bottom"] = _FACE_Z + 0.10
                assert _wait_until(
                    lambda: any("reason=no_declaration" in line for line in place_lines()),
                    timeout_s=5.0,
                ), place_lines()
                time.sleep(1.0)  # two measurement periods over a measurable table
                assert not latest().place_declaration_valid, "no goal, no allowance"

                # ── 1. Goal declared, held high above the table: no region. ──────
                state["bottom"] = _FACE_Z + 0.30
                declare(active=True)
                assert _wait_until(
                    lambda: any("reason=no_surface" in line for line in place_lines()),
                    timeout_s=5.0,
                ), place_lines()
                assert latest().place_declaration_valid and not region_valid()
                assert _wait_until(lambda: diagnostics.get("place_region") == "-", timeout_s=5.0)

                # ── 2. Lowered over the table: the region rides the goal's declaration.
                state["bottom"] = _FACE_Z + 0.10
                assert _wait_until(region_valid, timeout_s=5.0), place_lines()
                declared = latest().place_declaration
                target = "surface"
                assert declared.target_id == target, "the goal's own target"
                assert declared.stamp_ns == goal_stamp[0], "the goal's own stamp"
                assert declared.object_id == payload.object_id, "scoped to the carried payload"
                region = PlaceRegion.from_idl(declared.region)
                assert region.frame_id == _BASE
                assert region.geometry == ()
                assert region.half_extents[2] == pytest.approx(_RES / 2)
                assert region.pose.xyz[2] == pytest.approx(_FACE_Z - _RES / 2)
                assert region.pose.xyz[0] == pytest.approx(0.45, abs=_RES)
                assert region.pose.xyz[1] == pytest.approx(0.0, abs=_RES)
                assert region.stamp_ns in grid_stamps, "region stamp is not a grid data stamp"
                assert "map_measured_support:plane_z=0.310" in region.evidence_ref
                assert latest().objects[0].support_contact_valid is False
                assert _wait_until(
                    lambda: diagnostics.get("place_region") == f"live:{target}:geom=0",
                    timeout_s=5.0,
                ), diagnostics

                # ── 3. The box rests on the table: the map-support witness. ───────
                state["bottom"] = _FACE_Z + 0.004
                assert _wait_until(
                    lambda: latest().objects and latest().objects[0].support_contact_valid,
                    timeout_s=5.0,
                ), place_lines()
                witness = AttachedCollisionObject.from_idl(latest().objects[0]).support_contact
                assert witness is not None
                assert witness.support_id == target
                assert witness.evidence_kind is AttachmentEvidenceKind.MAP_SUPPORT_PROXIMITY
                assert "not sensed contact" in (witness.evidence_ref or "")
                assert witness.max_penetration_m == pytest.approx(0.01)
                assert 0.0 < witness.patch_radius_m <= 0.5
                assert witness.contact_normal_in_object == pytest.approx((0.0, 0.0, 1.0))
                assert any("not sensed contact" in line for line in place_lines())
                assert _wait_until(
                    lambda: (
                        f"safety.support_witness_armed object={payload.object_id} "
                        f"support={target}" in log_path.read_text(errors="replace")
                    ),
                    timeout_s=5.0,
                ), "the kernel did not accept the witness"

                # ── 4. The goal ends (retraction): region and witness die at once, and
                # nothing re-arms while the payload stays held on the table. ───────
                declare(active=False)
                assert _wait_until(
                    lambda: (
                        not latest().place_declaration_valid
                        and not latest().objects[0].support_contact_valid
                    ),
                    timeout_s=2.0,
                ), place_lines()
                state["bottom"] = _FACE_Z + 0.10  # measurable again, over the same table
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    msg = latest()
                    assert not msg.place_declaration_valid, "the retraction re-armed"
                    assert not msg.objects[0].support_contact_valid, "the witness re-armed"
                    time.sleep(0.05)
                assert _wait_until(lambda: diagnostics.get("place_region") == "-", timeout_s=5.0), (
                    diagnostics
                )
                # ── 5. A new goal: re-measured, re-armed, and a new witness. ─────
                declare(active=True)
                assert _wait_until(region_valid, timeout_s=5.0), place_lines()
                assert latest().place_declaration.stamp_ns == goal_stamp[0]
                state["bottom"] = _FACE_Z + 0.004
                assert _wait_until(
                    lambda: latest().objects and latest().objects[0].support_contact_valid,
                    timeout_s=5.0,
                ), place_lines()

                # ── 6. The octree stalls: grids keep arriving with a fresh header but
                # the same old source_stamp. The payload is lifted back over the patch
                # first, so every support cell is in view and only the data's age (never
                # the republished header) can retire the region. ─────────────────────
                state["bottom"] = _FACE_Z + 0.10
                time.sleep(1.2)  # two measurement ticks re-verify the patch in view
                assert region_valid(), place_lines()
                state["source"] = peer.get_clock().now().to_msg()
                bound_ns = int(2 * VisionAttachmentConfig().grid_max_age_s * 1e9)

                def aged_within_bound() -> bool:
                    msg = latest()
                    if msg.place_declaration_valid and msg.place_declaration.region_valid:
                        now = int(peer.get_clock().now().nanoseconds)
                        assert now - int(msg.place_declaration.region.stamp_ns) <= bound_ns + int(
                            2e8
                        ), "a published region outlived the kernel's age bound"
                        return False
                    return True

                assert _wait_until(aged_within_bound, timeout_s=5.0), (
                    "region outlived the stalled data"
                )
                assert any("grid_stale:" in line for line in place_lines()), place_lines()
                assert any("reason=freeze_ttl" in line for line in place_lines()), place_lines()
                assert "reason=region_stale" in log_path.read_text(errors="replace"), (
                    "the kernel never dropped the aging region on its own bound"
                )
                assert latest().place_declaration_valid, "the goal's declaration stays"
                assert _wait_until(
                    lambda: not latest().objects[0].support_contact_valid, timeout_s=2.0
                ), "witness outlived the region"
                assert _wait_until(lambda: diagnostics.get("place_region") == "-", timeout_s=5.0), (
                    diagnostics
                )

                assert aggregator_errors == []
            finally:
                # Stop the executor first: a timer callback waiting on the bridge lock must
                # not run on entities the teardown destroyed.
                executor.shutdown()
                spin.join(timeout=5.0)
                with suppress(Exception):
                    bridge.teardown()
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
