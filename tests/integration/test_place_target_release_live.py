# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: a payload set down on a measured surface keeps its witness through the release.

Real pick-and-place design §2.3 "Release": the witness, if one is live, keeps the table
partition correct while the frozen record is published. Nothing is surveyed and nothing
names a place target: real rclpy, a real ``VisionAttachmentBridge`` on the bimanual OpenArm
manifest with the place-target leg measuring the table under the carried payload (the
only declaration is the runner's goal-scope one, ``target_id = surface``), real
tf2, every
``/openral/attachment_state`` through a real ``WorldStateAggregator`` and the World State
node's ``build_world_state_stamped_msg`` into a real ``safety_kernel_node`` (the real cell's
parameters, attached check on at margin 0, as the vision leg turns it on).

The payload (the bridge's fallback box) is held 10 cm over the table until the leg has
measured and latched the surface under it, then lowered with its bottom 5 mm below the
measured face, the grip opens (DETACH), and the frozen record stays on the face. There is no
octomap bridge
here, so the grid is published the way its payload clearing leaves it with the witness live —
an explicit assumption: the shelf layer, plus the one co-planar cell layer under the payload
that ``support_patch_withholds`` keeps (centres 1 cm above the face). The cells of that layer
the payload's edge overlaps by less than half a voxel sit above the measured slab (the place
region's approach allowance does not reach them) and are not embedded residue — only the
witness's support-contact exemption covers them. One retreat candidate at the release
configuration, per row:

=====================================================  ========================================
row                                                    verdict
=====================================================  ========================================
place leg on: witness armed while held, carried on    ACCEPTED (support-contact exemption
the frozen record                                     covers the patch)
place leg off: frozen record with no witness          REFUSED, ``attached:<id>`` vs a voxel
=====================================================  ========================================

The kernel refreshes its witness latch only when it checks a candidate, against the measured
configuration; no candidate is sent while the hand holds the payload (the synthetic hand TF is
not the manifest FK at ``q = 0``), and the frozen record is in ``openarm_base``, so its contact
is the same at every configuration.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` (and a colcon-built ``openral_safety_kernel``)::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    PYTHONPATH=$PWD/packages/world_state:$PYTHONPATH OPENRAL_TEST_ROS_LIVE=1 \\
        pytest tests/integration/test_place_target_release_live.py
"""

from __future__ import annotations

import json
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

from tests.integration.test_place_target_leg_live import (
    _BASE,
    _FACE_Z,
    _LINK7,
    _ROBOT_YAML,
    _SIZE,
    _grid,
    _wait_until,
)

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy node + colcon openral_msgs / safety kernel overlay — set "
    "OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)

_OBJECT_ID = "grasped_payload:left_gripper"
_ARMED = f"safety.support_witness_armed object={_OBJECT_ID} support=surface"
_SPAN = 0.05  # VisionGateConfig.jaw_span_m: the fallback box's half-extent
#: The payload's centre x: its +x face at 0.506, 6 mm into the i = 15 cell column.
_PAYLOAD_X = 0.456
#: The co-planar layer (k = 5, centres z = 0.32) under the payload (x 0.406..0.506,
#: y -0.05..0.05): cell centres x = 0.21 + 0.02 i, y = -0.45 + 0.02 j. Columns i = 10..14
#: lie inside the payload (the kernel's embedded-residue rule exempts them on their own);
#: column i = 15 (x 0.50..0.52) it overlaps by 6 mm — less than the half voxel that makes a
#: cell embedded, above the measured slab the region's allowance covers, inside the witness's
#: patch and height bound. Only the witness can exempt those cells.
_PATCH_CELLS = frozenset(
    i + _SIZE[0] * (j + _SIZE[1] * 5) for i in range(10, 16) for j in range(21, 25)
)


def _patch_grid(stamp: Any) -> Any:
    """The table face plus the withheld co-planar patch under the payload."""
    msg = _grid(stamp, face=True)
    occupancy = list(msg.occupancy)
    for idx in _PATCH_CELLS:
        occupancy[idx] = 1
    msg.occupancy = occupancy
    return msg


@pytest.mark.parametrize("place_leg", [True, False], ids=["witness", "no_witness"])
def test_the_frozen_record_keeps_its_witness_and_the_kernel_accepts_the_retreat(
    place_leg: bool,
) -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    world_state_ros = pytest.importorskip("openral_world_state_ros.lifecycle_node")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS 2 not sourced")

    from geometry_msgs.msg import TransformStamped
    from openral_core import (
        Action,
        AttachedCollisionObject,
        ControlMode,
        JointState,
        PlaceDeclaration,
        RobotDescription,
    )
    from openral_hal.vision_attachment_bridge import (
        VisionAttachmentBridge,
        VisionAttachmentConfig,
    )
    from openral_msgs.msg import (
        ActionChunk,
        AttachmentState,
        FailureTrigger,
        OccupancyVoxels,
        WorldStateStamped,
    )
    from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg
    from openral_world_state import WorldStateAggregator
    from rcl_interfaces.msg import Log
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState as JointStateMsg
    from std_msgs.msg import Empty
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
    joint_names = [j.name for j in description.joints]

    kernel_name = f"safety_kernel_place_release_{uuid.uuid4().hex[:8]}"
    params = {
        **_kernel_params(grasp_allowance_enabled=False),
        "attached_collision_enabled": True,
        "attached_collision_margin_m": 0.0,
        "attached_collision_deadline_ms": 1000.0,
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
            node = Node("test_place_release_hal")
            peer = Node("test_place_release_peer")
            assert activate_kernel_node(kernel_name, peer), "kernel activation failed"

            logs: list[str] = []
            peer.create_subscription(Log, "/rosout", lambda m: logs.append(str(m.msg)), 500)
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
            joint_pub = peer.create_publisher(JointStateMsg, "/joint_states", 10)
            chunk_pub = peer.create_publisher(ActionChunk, "/openral/candidate_action", 10)
            safe: dict[str, Any] = {}
            failures: list[Any] = []
            estops: list[Any] = []
            peer.create_subscription(
                ActionChunk, "/openral/safe_action", lambda m: safe.__setitem__(m.trace_id, m), 10
            )
            peer.create_subscription(FailureTrigger, "/openral/failure/safety", failures.append, 50)
            peer.create_subscription(Empty, "/openral/estop", estops.append, 10)
            StaticTransformBroadcaster(peer).sendTransform(
                [_tf_msg(_BASE, "zed_camera_link", _thor_mount())]
            )
            hand_broadcaster = TransformBroadcaster(peer)
            state: dict[str, Any] = {
                "loaded": False,
                "bottom": _FACE_Z + 0.10,
                "box_in_link": np.zeros(3),
                "clearing": False,  # the grid the way payload clearing leaves it, once set down
            }

            def publish_hand() -> None:
                """link7 posed so the fallback box's bottom sits at ``state["bottom"]``."""
                xyz = np.array([_PAYLOAD_X, 0.0, state["bottom"] + _SPAN]) - state["box_in_link"]
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
                    depth_topic="/test_place_release/depth",  # never published
                    service_name="/openral/perception/segment_in_view_place_release_itest",
                    place_target_enabled=place_leg,
                    # Only the jaws clearing could end the window; they never do here.
                    release_timeout_s=60.0,
                ),
            )

            def feed() -> None:
                # Loaded: commanded closed, the left jaw stalls 0.2 rad short (position
                # stall). Released: commanded open, the jaw opens past its hold.
                left = 0.0 if state["loaded"] else 0.7
                bridge.observe_command(
                    Action(
                        control_mode=ControlMode.JOINT_POSITION,
                        horizon=1,
                        joint_names=["left_gripper", "right_gripper"],
                        joint_targets=[[left, 0.0]],
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

            def publish_joint_state() -> None:
                js = JointStateMsg()
                js.header.stamp = peer.get_clock().now().to_msg()
                js.name = joint_names
                js.position = [0.0] * len(joint_names)
                joint_pub.publish(js)

            bridge.setup()
            node.create_timer(1.0 / 30.0, feed)
            # The goal in progress: the runner's goal-scope declaration
            # (``place_approach_enabled``) — no target, no object, no region. The leg
            # attaches its measured region to it and never declares on its own.
            goal = PlaceDeclarationMsg()
            PlaceDeclaration(
                target_id="surface",
                rskill_id="openral/itest-place-target",
                trace_id="itest",
                timeout_s=60.0,
                stamp_ns=int(peer.get_clock().now().nanoseconds),
            ).fill_idl(goal)
            peer.create_publisher(
                PlaceDeclarationMsg, "/openral/place_declaration", latched
            ).publish(goal)

            def publish_grid() -> None:
                stamp = peer.get_clock().now().to_msg()
                voxel_pub.publish(
                    _patch_grid(stamp) if state["clearing"] else _grid(stamp, face=True)
                )

            peer.create_timer(0.2, publish_grid)
            peer.create_timer(0.05, publish_hand)
            peer.create_timer(0.1, publish_joint_state)
            executor = MultiThreadedExecutor()
            executor.add_node(node)
            executor.add_node(peer)
            spin_thread = threading.Thread(target=executor.spin, daemon=True)
            spin_thread.start()

            def latest_object() -> AttachedCollisionObject | None:
                msg = envelopes[-1] if envelopes else None
                if msg is None or len(msg.objects) != 1:
                    return None
                return AttachedCollisionObject.from_idl(msg.objects[0])

            def armed_count() -> int:
                return log_path.read_text(errors="replace").count(_ARMED)

            def place_lines() -> list[str]:
                return [line for line in logs if "place_target" in line or "release" in line]

            try:
                # ── Hold over the table until the surface under it is measured, then
                # lower the fallback box 5 mm into the table layer. ────────────────────
                state["loaded"] = True
                assert _wait_until(
                    lambda: (o := latest_object()) is not None and o.attach_link == _LINK7,
                    timeout_s=10.0,
                ), f"no held payload; {place_lines()}"
                held = latest_object()
                assert held is not None and held.object_id == _OBJECT_ID
                state["box_in_link"] = np.asarray(held.pose_in_link.xyz)
                if place_leg:
                    assert _wait_until(
                        lambda: (
                            bool(envelopes)
                            and envelopes[-1].place_declaration_valid
                            and envelopes[-1].place_declaration.region_valid
                        ),
                        timeout_s=5.0,
                    ), f"no surface measured under the payload; {place_lines()}"
                state["bottom"] = _FACE_Z - 0.005
                state["clearing"] = True
                if place_leg:
                    assert _wait_until(
                        lambda: getattr(latest_object(), "support_contact", None) is not None,
                        timeout_s=5.0,
                    ), place_lines()
                    assert _wait_until(lambda: armed_count() == 1, timeout_s=5.0), (
                        "the kernel did not arm the held witness"
                    )
                    held = latest_object()
                    assert held is not None
                    witness = held.support_contact
                    assert witness is not None
                else:
                    time.sleep(1.0)  # the hand settles on the face; nothing can arm

                # ── DETACH: the frozen record, in the base frame, on the face. ─────────
                state["loaded"] = False
                assert _wait_until(
                    lambda: (o := latest_object()) is not None and o.attach_link == _BASE,
                    timeout_s=5.0,
                ), f"DETACH did not freeze the payload; {place_lines()}"
                # > the witness re-evaluation period and a heartbeat, but well inside the
                # region's 2 s freeze: the witness dies with the region (kernel age bound).
                time.sleep(0.3)
                frozen = latest_object()
                assert frozen is not None and frozen.attach_link == _BASE
                assert (frozen.object_id, frozen.stamp_ns) == (held.object_id, held.stamp_ns)
                if place_leg:
                    assert frozen.support_contact == witness, (
                        f"the frozen record lost its witness; {place_lines()}"
                    )
                    assert armed_count() == 1, "same key: the kernel must not re-arm"
                else:
                    assert frozen.support_contact is None

                # ── One retreat candidate at the release configuration. ────────────────
                trace = f"release-{'witness' if place_leg else 'none'}"
                chunk = ActionChunk()
                chunk.control_mode = 0  # JOINT_POSITION
                chunk.horizon = 1
                chunk.n_dof = len(joint_names)
                chunk.flat = [0.0] * len(joint_names)
                chunk.rskill_id = "openral/itest-place-target"
                chunk.trace_id = trace
                chunk_pub.publish(chunk)
                if place_leg:
                    assert _wait_until(lambda: trace in safe, timeout_s=10.0), (
                        f"the retreat was refused: {[f.evidence_json for f in failures]}"
                    )
                    assert not estops and not failures
                    assert "safety.support_witness_separated" not in log_path.read_text(
                        errors="replace"
                    )
                else:
                    assert _wait_until(lambda: bool(failures) and bool(estops), timeout_s=10.0)
                    time.sleep(0.4)
                    assert trace not in safe
                    trigger = failures[-1]
                    assert trigger.kind == FailureTrigger.KIND_COLLISION
                    evidence = json.loads(trigger.evidence_json)
                    assert evidence["collision_kind"] == "world", evidence
                    assert evidence["link_a"] == f"attached:{_OBJECT_ID}", evidence
                    assert str(evidence["link_b_or_object"]).startswith("voxel_"), evidence
                assert aggregator_errors == []
            finally:
                # Stop the executor first: a timer callback waiting on the bridge lock must
                # not run on entities the teardown destroyed.
                executor.shutdown()
                spin_thread.join(timeout=5.0)
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
