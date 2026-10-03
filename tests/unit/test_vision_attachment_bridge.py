"""HAL-side vision attachment client: fallback decisions and the mask/depth wire.

The bridge's ROS half needs a live graph and is covered by
``tests/integration/test_segment_in_view_service.py``. What is covered here is
everything the barrier's *safety* argument rests on and that can be decided
without one:

* ``resolve_segment_outcome`` — the
  full truth table of "what does this round trip mean", including that **no**
  input combination produces "skip the attachment".
* the ``mono8`` mask wire, encoded by the perception node and decoded by the
  HAL, round-tripped through a **real** SAM 2.1 mask fixture rather than a
  hand-drawn square (CLAUDE.md §1.11).
* ``depth_grid_from_image``, the depth decoder that
  turns a driver's ``32FC1`` / ``16UC1`` frame into the metric raster the
  producer gates on.
* the per-gripper leg wiring ``VisionAttachmentBridge.__init__`` resolves from
  the real SO-101 and bimanual OpenArm manifests (it creates no ROS entities, so
  ``node=None`` is the real constructor, not a double), the commanded-target feed
  (``observe_command``) and the grasp-target region payload (``region_attachment``).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from openral_core import (
    Action,
    ControlMode,
    GripperClosureCalibration,
    JointState,
    RobotDescription,
)
from openral_core.exceptions import ROSConfigError
from openral_hal.vision_attachment_bridge import (
    DEFAULT_SEGMENT_SERVICE,
    VisionAttachmentBridge,
    VisionAttachmentConfig,
    decode_mono8_mask,
    mask_depth_skew_reason,
    resolve_segment_outcome,
)
from PIL import Image

_ERASER_MASK = Path("tests/unit/fixtures/sam2_wrist_eraser_mask.png")


def _real_mask() -> np.ndarray:
    """The committed SAM 2.1 mask of the eraser in the SO-101's jaws."""
    return np.asarray(Image.open(_ERASER_MASK).convert("L")) > 0


def test_a_clean_reply_uses_its_masks() -> None:
    """ok=True with candidates is the only path that reaches the gates."""
    outcome = resolve_segment_outcome(timed_out=False, ok=True, failure_reason="", mask_count=3)
    assert outcome.use_masks is True
    assert outcome.reason == ""


def test_a_deadline_miss_falls_back_and_says_so() -> None:
    """The bounded wait expiring is a named, typed fallback — never a longer wait."""
    outcome = resolve_segment_outcome(timed_out=True, ok=False, failure_reason="", mask_count=0)
    assert outcome.use_masks is False
    assert outcome.reason.startswith("ROSDeadlineMissed:")


def test_a_deadline_miss_wins_over_a_late_ok() -> None:
    """Once the deadline fired, a reply that would have been fine is still late."""
    outcome = resolve_segment_outcome(timed_out=True, ok=True, failure_reason="", mask_count=3)
    assert outcome.use_masks is False
    assert outcome.reason.startswith("ROSDeadlineMissed:")


def test_a_typed_server_failure_is_carried_verbatim() -> None:
    """The node's own typed reason reaches the trace unrewritten."""
    outcome = resolve_segment_outcome(
        timed_out=False,
        ok=False,
        failure_reason="ROSPerceptionStale: no frame cached for camera 'wrist'",
        mask_count=0,
    )
    assert outcome.use_masks is False
    assert outcome.reason == "ROSPerceptionStale: no frame cached for camera 'wrist'"


def test_an_untyped_server_failure_still_gets_a_reason() -> None:
    """ok=False with an empty reason is still named, never silent."""
    outcome = resolve_segment_outcome(timed_out=False, ok=False, failure_reason="", mask_count=0)
    assert outcome.use_masks is False
    assert outcome.reason.startswith("ROSPerceptionStale:")


def test_ok_with_no_candidates_falls_back() -> None:
    """A success flag with an empty mask array cannot be gated, so it degrades."""
    outcome = resolve_segment_outcome(timed_out=False, ok=True, failure_reason="", mask_count=0)
    assert outcome.use_masks is False
    assert outcome.reason.startswith("ROSPerceptionStale:")


@pytest.mark.parametrize("timed_out", [True, False])
@pytest.mark.parametrize("ok", [True, False])
@pytest.mark.parametrize("mask_count", [0, 1, 3])
def test_every_outcome_is_named_or_clean(timed_out: bool, ok: bool, mask_count: int) -> None:
    """No input combination yields an unexplained fallback.

    The producer turns ``use_masks=False`` into the conservative GRIPPER_CLOSURE
    box, so an unnamed one would be an attachment nobody could explain from the
    trace (CLAUDE.md §1.4).
    """
    outcome = resolve_segment_outcome(
        timed_out=timed_out, ok=ok, failure_reason="", mask_count=mask_count
    )
    assert outcome.use_masks is (bool(outcome.reason) is False)
    assert outcome.use_masks or outcome.reason, "a fallback must carry its reason"


def test_mono8_round_trip_preserves_a_real_sam2_mask() -> None:
    """The perception encoder and the HAL decoder agree, pixel for pixel."""
    from openral_perception_ros.segmenter_node import mono8_bytes_from_mask

    mask = _real_mask()
    height, width = mask.shape
    payload = mono8_bytes_from_mask(mask)
    assert len(payload) == height * width
    decoded = decode_mono8_mask(payload, height=height, width=width)
    assert decoded.dtype == np.bool_
    assert np.array_equal(decoded, mask)
    assert int(decoded.sum()) == int(mask.sum()) > 0


def test_mono8_decode_accepts_any_nonzero_pixel() -> None:
    """A mask that arrived as 254 is still a mask, not an empty one."""
    decoded = decode_mono8_mask(bytes([0, 1, 254, 255]), height=2, width=2)
    assert decoded.tolist() == [[False, True], [True, True]]


def test_mono8_decode_rejects_a_size_mismatch() -> None:
    """A truncated payload is a typed error, not a reshaped guess."""
    with pytest.raises(ROSConfigError, match="mono8"):
        decode_mono8_mask(bytes([255, 0, 0]), height=2, width=2)


def test_a_mask_is_only_paired_with_depth_from_the_same_instant() -> None:
    """A mask captured one frame off its depth passes; one several frames off is refused."""
    depth_ns = 5_000_000_000
    skew = VisionAttachmentConfig().mask_depth_max_skew_s
    assert mask_depth_skew_reason([depth_ns], depth_ns, max_skew_s=skew) == ""
    assert mask_depth_skew_reason([depth_ns - 66_000_000], depth_ns, max_skew_s=skew) == ""
    reason = mask_depth_skew_reason(
        [depth_ns, depth_ns + 300_000_000], depth_ns, max_skew_s=skew
    )  # one stale candidate refuses the reply
    assert reason.startswith("ROSPerceptionStale:") and "0.300 s" in reason


def test_a_non_positive_mask_depth_skew_is_refused() -> None:
    with pytest.raises(ROSConfigError, match="mask_depth_max_skew_s"):
        VisionAttachmentBridge(
            None,
            _openarm(),
            config=VisionAttachmentConfig(camera="head_zed", mask_depth_max_skew_s=0.0),
        )


def test_default_config_points_at_the_perception_node_service() -> None:
    """The two halves agree on the service name without a literal in each."""
    from openral_perception_ros.segmenter_node import (
        DEFAULT_SEGMENT_SERVICE as PERCEPTION_DEFAULT,
    )

    assert VisionAttachmentConfig().service_name == DEFAULT_SEGMENT_SERVICE
    assert DEFAULT_SEGMENT_SERVICE == PERCEPTION_DEFAULT


def test_default_deadline_is_bounded_and_barrier_sized() -> None:
    """The wait is finite and stays the order of the ~100 ms barrier it rides in."""
    deadline = VisionAttachmentConfig().deadline_s
    assert 0.0 < deadline <= 1.0


def test_depth_decode_32fc1_is_metres() -> None:
    """A 32FC1 driver frame decodes straight through, zeros preserved."""
    pytest.importorskip("sensor_msgs")
    from openral_hal.depth_cloud import depth_grid_from_image
    from sensor_msgs.msg import Image as RosImage

    grid = np.array([[0.25, 0.0], [1.5, 2.0]], dtype="<f4")
    msg = RosImage()
    msg.height, msg.width = 2, 2
    msg.encoding = "32FC1"
    msg.step = 8
    msg.data = grid.tobytes()
    decoded = depth_grid_from_image(msg)
    assert decoded.dtype == np.float64
    assert decoded.tolist() == [[0.25, 0.0], [1.5, 2.0]]


def test_depth_decode_16uc1_is_millimetres() -> None:
    """A RealSense/OAK-D 16UC1 frame is rescaled to metres."""
    pytest.importorskip("sensor_msgs")
    from openral_hal.depth_cloud import depth_grid_from_image
    from sensor_msgs.msg import Image as RosImage

    grid = np.array([[250, 0], [1500, 2000]], dtype="<u2")
    msg = RosImage()
    msg.height, msg.width = 2, 2
    msg.encoding = "16UC1"
    msg.step = 4
    msg.data = grid.tobytes()
    decoded = depth_grid_from_image(msg)
    assert np.allclose(decoded, [[0.25, 0.0], [1.5, 2.0]])


def test_depth_decode_rejects_an_unsupported_encoding() -> None:
    """An RGB frame on the depth topic is a typed error, not garbage geometry."""
    pytest.importorskip("sensor_msgs")
    from openral_hal.depth_cloud import depth_grid_from_image
    from sensor_msgs.msg import Image as RosImage

    msg = RosImage()
    msg.height, msg.width = 1, 1
    msg.encoding = "rgb8"
    msg.step = 3
    msg.data = bytes(3)
    with pytest.raises(ROSConfigError, match="unsupported depth encoding"):
        depth_grid_from_image(msg)


def test_depth_decode_rejects_a_truncated_payload() -> None:
    """A short buffer never becomes a silently smaller depth frame."""
    pytest.importorskip("sensor_msgs")
    from openral_hal.depth_cloud import depth_grid_from_image
    from sensor_msgs.msg import Image as RosImage

    msg = RosImage()
    msg.height, msg.width = 4, 4
    msg.encoding = "32FC1"
    msg.step = 16
    msg.data = np.zeros(4, dtype="<f4").tobytes()
    with pytest.raises(ROSConfigError, match="payload has 4 samples"):
        depth_grid_from_image(msg)


# ── One leg per gripper (real manifests; ``__init__`` creates no ROS entities) ──


def _openarm() -> RobotDescription:
    return RobotDescription.from_yaml("robots/openarm/robot.yaml")


def test_an_uncalibrated_gripper_refuses_the_bridge() -> None:
    """SO-101 declares no ``closure_calibration``: a typed error, never guessed thresholds."""
    with pytest.raises(ROSConfigError, match="closure_calibration"):
        VisionAttachmentBridge(
            None,
            RobotDescription.from_yaml("robots/so101_follower/robot.yaml"),
            config=VisionAttachmentConfig(camera="wrist"),
        )


def _so101_calibrated() -> RobotDescription:
    """SO-101 with an ILLUSTRATIVE calibration (none is measured) so its TCP wiring loads."""
    description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    calibration = GripperClosureCalibration(
        closed_position=0.0, closed_rest_offset=0.0, stall_gap=0.1, settle_tolerance=0.01
    )
    joints = [
        j.model_copy(update={"closure_calibration": calibration}) if j.role == "gripper" else j
        for j in description.joints
    ]
    return description.model_copy(update={"joints": joints})


def _action(rows: list[list[float]], names: list[str] | None = None) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=len(rows),
        joint_targets=rows,
        joint_names=names,
    )


def test_a_whole_vector_action_commands_both_jaws_from_its_last_row() -> None:
    """Rows in manifest order; the trajectory goal (last row) is the command."""
    description = _openarm()
    bridge = VisionAttachmentBridge(
        None, description, config=VisionAttachmentConfig(camera="head_zed")
    )
    names = [j.name for j in description.joints]
    first, last = [0.0] * len(names), [0.0] * len(names)
    first[names.index("left_gripper")] = 0.7
    last[names.index("left_gripper")] = 0.05
    last[names.index("right_gripper")] = -0.6
    bridge.observe_command(_action([first, last]))
    left, right = bridge._legs
    assert left.trigger.last_command == 0.05
    assert right.trigger.last_command == -0.6


def test_a_slot_action_commands_only_the_jaw_it_names() -> None:
    """ADR-0102: a zero-padded slot owns only ``joint_names``; the other hand keeps its command."""
    description = _openarm()
    bridge = VisionAttachmentBridge(
        None, description, config=VisionAttachmentConfig(camera="head_zed")
    )
    left, right = bridge._legs
    right.trigger.command(-0.5)
    slot = [f"left_joint{i}" for i in range(1, 8)] + ["left_gripper"]
    bridge.observe_command(_action([[0.1] * 7 + [0.0] + [0.0] * 8], names=slot))
    assert left.trigger.last_command == 0.0
    assert right.trigger.last_command == -0.5, "a zero pad is not a right-hand close command"


def _openarm_slot_group(left_jaw: float, right_jaw: float, *, tick: int = 1) -> list[Action]:
    """The four actions the real runner dispatches for one OpenArm v2 bimanual tick.

    Built by the runner's own ``_dispatch_slots`` against the real manifest — arm slots
    zero-padded to 16 dof at their manifest indices (``_pad_joint_payload``), gripper
    slots ``GRIPPER_POSITION`` with ``gripper=[value]`` and ``ee_name`` the gripper
    joint — then stamped with the tick the runner adds.
    """
    import importlib.util

    from openral_core import ActionSlot

    src = Path("packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py")
    spec = importlib.util.spec_from_file_location("_bridge_test_rskill_runner_node", src)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    slots = [
        ActionSlot(
            range=(0, 6),
            control_mode=ControlMode.JOINT_POSITION,
            joint_names=[f"left_joint{i}" for i in range(1, 8)],
        ),
        ActionSlot(range=(7, 7), control_mode=ControlMode.GRIPPER_POSITION, ee="left_gripper"),
        ActionSlot(
            range=(8, 14),
            control_mode=ControlMode.JOINT_POSITION,
            joint_names=[f"right_joint{i}" for i in range(1, 8)],
        ),
        ActionSlot(range=(15, 15), control_mode=ControlMode.GRIPPER_POSITION, ee="right_gripper"),
    ]
    vector = np.array([0.1] * 7 + [left_jaw] + [-0.1] * 7 + [right_jaw], dtype=np.float32)
    group = runner._dispatch_slots(slots, vector, description=_openarm())
    for action in group:
        action.tick_index = tick
        action.runner_session_id = 0xA11CE
    return list(group)


def test_a_real_openarm_slot_group_commands_both_jaws_as_the_hal_composes_it() -> None:
    """ADR-0102: the jaws are GRIPPER_POSITION slots; folded once the tick is whole.

    The targets are the very values ``compose_slot_group`` hands the OpenArm
    controllers, so the trigger's "short of the command" is measured against what the
    jaw was actually told.
    """
    from openral_hal._slot_group import compose_slot_group

    description = _openarm()
    bridge = VisionAttachmentBridge(
        None, description, config=VisionAttachmentConfig(camera="head_zed")
    )
    left, right = bridge._legs
    group = _openarm_slot_group(0.0, -0.75)
    assert [a.control_mode for a in group].count(ControlMode.GRIPPER_POSITION) == 2
    for action in group[:-1]:
        bridge.observe_command(action)
        assert left.trigger.last_command is None, "an incomplete tick commanded nothing yet"
    bridge.observe_command(group[-1])
    names = [j.name for j in description.joints]
    hal_row = compose_slot_group(group, names)
    assert left.trigger.last_command == pytest.approx(hal_row[names.index("left_gripper")])
    assert right.trigger.last_command == pytest.approx(hal_row[names.index("right_gripper")])
    assert (left.trigger.last_command, right.trigger.last_command) == pytest.approx((0.0, -0.75))


def test_an_incomplete_slot_group_never_commands_a_jaw() -> None:
    """A tick the kernel cut short never reached the robot, so its gripper slot is not a command."""
    bridge = VisionAttachmentBridge(
        None, _openarm(), config=VisionAttachmentConfig(camera="head_zed")
    )
    left, _ = bridge._legs
    for action in _openarm_slot_group(0.0, 0.0, tick=1)[:3]:  # right gripper slot rejected
        bridge.observe_command(action)
    for action in _openarm_slot_group(0.7, -0.7, tick=2):
        bridge.observe_command(action)
    assert left.trigger.last_command == pytest.approx(0.7), "tick 1's close was never applied"


def test_a_right_arm_slot_reads_the_right_jaw_column_not_the_left_pad() -> None:
    """A padded right slot owns ``right_gripper`` at manifest index 15, not at its own index 7.

    Index 7 of a right-hand padded row is ``left_gripper``'s zero pad: read there, a
    right jaw held open reads "commanded closed" and its stall test turns an open hand
    into an ATTACH.
    """
    description = _openarm()
    bridge = VisionAttachmentBridge(
        None, description, config=VisionAttachmentConfig(camera="head_zed")
    )
    left, right = bridge._legs
    left.trigger.command(0.7)
    names = [j.name for j in description.joints]
    row = [0.0] * len(names)
    owned = [f"right_joint{i}" for i in range(1, 8)] + ["right_gripper"]
    for name in owned:
        row[names.index(name)] = -0.1
    row[names.index("right_gripper")] = -0.6  # held open
    bridge.observe_command(_action([row], names=owned))
    assert right.trigger.last_command == pytest.approx(-0.6)
    assert left.trigger.last_command == pytest.approx(0.7), "the left pad commands nothing"
    # And the open right hand never attaches, however long it sits still.
    for tick in range(60):
        bridge.observe_joint_state(_gripper_sample(left=0.7, right=-0.6, stamp_ns=_at(tick)))
    assert not right.trigger.attached


def test_a_compact_row_is_read_positionally() -> None:
    """``len(row) == len(joint_names)``: ``joint_names[i]`` owns ``row[i]``."""
    bridge = VisionAttachmentBridge(
        None, _openarm(), config=VisionAttachmentConfig(camera="head_zed")
    )
    left, right = bridge._legs
    bridge.observe_command(_action([[0.05, -0.6]], names=["left_gripper", "right_gripper"]))
    assert (left.trigger.last_command, right.trigger.last_command) == (0.05, -0.6)
    bridge.observe_command(_action([[-0.3]], names=["right_gripper"]))
    assert (left.trigger.last_command, right.trigger.last_command) == (0.05, -0.3)


def test_a_row_of_any_other_length_commands_nothing() -> None:
    """Neither full dof nor as long as its names: unplaceable, so no jaw command."""
    bridge = VisionAttachmentBridge(
        None, _openarm(), config=VisionAttachmentConfig(camera="head_zed")
    )
    bridge.observe_command(_action([[0.0, 0.0, 0.0]], names=["left_gripper", "right_gripper"]))
    bridge.observe_command(_action([[0.0] * 8]))
    assert all(leg.trigger.last_command is None for leg in bridge._legs)


def test_a_non_joint_position_action_commands_no_jaw() -> None:
    bridge = VisionAttachmentBridge(
        None, _openarm(), config=VisionAttachmentConfig(camera="head_zed")
    )
    bridge.observe_command(
        Action(control_mode=ControlMode.JOINT_VELOCITY, horizon=1, joint_velocities=[[0.0] * 16])
    )
    assert all(leg.trigger.last_command is None for leg in bridge._legs)


def test_the_region_payload_is_the_declared_object_at_the_region() -> None:
    """Design §2.2 handover: the latched region, posed in the attach link, named for the kernel."""
    from openral_core import AttachmentEvidenceKind, DeployScene, PlaceRegion, Pose6D
    from openral_core.geometry import homogeneous_from_quat_xyz
    from openral_hal.vision_attachment_bridge import region_attachment

    scene = DeployScene.from_yaml(
        "tests/unit/fixtures/scenes/openarm_direct_dispatch_grasp.yaml"
    ).grasp_declaration
    assert scene is not None
    region = PlaceRegion(
        frame_id="openarm_base",
        pose=Pose6D(xyz=(0.45, 0.0, 0.09), quat_xyzw=(0, 0, 0, 1), frame_id="openarm_base"),
        half_extents=(0.03, 0.04, 0.05),
        evidence_ref="segment_in_view:test@0",
        stamp_ns=5,
    )
    declaration = scene.model_copy(update={"region": region})
    # attach link <- base: the hand 0.4 m out along x, rotated 90 deg about z.
    half = np.sqrt(0.5)
    t_base_from_link = homogeneous_from_quat_xyz((0.4, 0.0, 0.1), (0.0, 0.0, half, half))
    held = region_attachment(
        declaration,
        attach_link="openarm_left_link7",
        touch_links=("openarm_left_finger_pair",),
        t_link_from_region=np.linalg.inv(t_base_from_link),
        stamp_ns=7,
    )
    assert held.object_id == (declaration.object_id or declaration.target_id)
    assert held.evidence_kind is AttachmentEvidenceKind.GRASP_TARGET_REGION
    assert held.attach_link == "openarm_left_link7"
    (prim,) = held.primitives
    assert prim.shape.half_extents_m == (0.03, 0.04, 0.05)
    back = t_base_from_link @ homogeneous_from_quat_xyz(
        held.pose_in_link.xyz, held.pose_in_link.quat_xyzw
    )
    np.testing.assert_allclose(back[:3, 3], (0.45, 0.0, 0.09), atol=1e-9)
    np.testing.assert_allclose(back[:3, :3], np.eye(3), atol=1e-9)


def test_a_bimanual_bridge_builds_one_leg_per_gripper() -> None:
    """OpenArm's two hands each get a trigger, a producer and a unique object id."""
    bridge = VisionAttachmentBridge(
        None, _openarm(), config=VisionAttachmentConfig(camera="head_zed")
    )
    assert bridge.gripper_joint_names == ("left_gripper", "right_gripper")
    left, right = bridge._legs
    assert left.producer.attach_link == "openarm_left_link7"
    assert right.producer.attach_link == "openarm_right_link7"
    assert left.object_id == "grasped_payload:left_gripper"
    assert left.object_id != right.object_id
    assert bridge.attachment_action_ack_ready()


def test_the_bimanual_tcp_is_the_gripper_joint_origin_without_tf() -> None:
    """The TCP comes from the manifest, so a link missing from TF cannot break it."""
    description = _openarm()
    bridge = VisionAttachmentBridge(
        None, description, config=VisionAttachmentConfig(camera="head_zed")
    )
    left_joint = next(j for j in description.joints if j.name == "left_gripper")
    left = bridge._legs[0]
    assert left.tcp_in_link == left_joint.origin_xyz == (-0.00143, -0.018, -0.068)
    assert left.tcp_frame == ""


def test_a_single_tcp_frame_override_is_refused_on_two_grippers() -> None:
    """``tcp_frame`` cannot say which hand it means, so it is a typed error."""
    with pytest.raises(ROSConfigError, match="single-gripper"):
        VisionAttachmentBridge(
            None,
            _openarm(),
            config=VisionAttachmentConfig(camera="head_zed", tcp_frame="openarm_left_hand_tcp"),
        )


def test_the_single_gripper_bridge_keeps_its_tf_tcp() -> None:
    """SO-101 still looks its TCP up as the moving jaw through tf2, as before."""
    bridge = VisionAttachmentBridge(
        None, _so101_calibrated(), config=VisionAttachmentConfig(camera="wrist")
    )
    (leg,) = bridge._legs
    assert leg.producer.attach_link == "gripper_base"
    assert leg.tcp_frame == "moving_jaw"


def test_tf_frames_maps_an_attach_link_to_its_published_name() -> None:
    """OpenArm's ``link7`` is published as ``ee_base_link``; unmapped links pass through."""
    mapped = VisionAttachmentBridge(
        None,
        _openarm(),
        config=VisionAttachmentConfig(
            camera="head_zed",
            tf_frames={"openarm_left_link7": "openarm_left_ee_base_link"},
        ),
    )
    assert mapped.tf_frame("openarm_left_link7") == "openarm_left_ee_base_link"
    assert mapped.tf_frame("openarm_right_link7") == "openarm_right_link7"
    plain = VisionAttachmentBridge(
        None, _openarm(), config=VisionAttachmentConfig(camera="head_zed")
    )
    assert plain.tf_frame("openarm_left_link7") == "openarm_left_link7"


def test_tf_frames_rejects_a_link_the_manifest_does_not_have() -> None:
    """A mapping for a non-existent link is a typo, not a silent no-op."""
    with pytest.raises(ROSConfigError, match="not links of"):
        VisionAttachmentBridge(
            None,
            _openarm(),
            config=VisionAttachmentConfig(
                camera="head_zed", tf_frames={"openarm_left_ee_base_link": "x"}
            ),
        )


def _at(tick: int) -> int:
    """Stamp of the ``tick``-th 30 Hz sample."""
    return 1_000_000_000 + tick * 33_333_333


def _gripper_sample(*, left: float | None, right: float | None, stamp_ns: int) -> JointState:
    """One OpenArm read carrying a jaw position for whichever grippers have a value.

    Effort is the real driver's: a zero for every joint, every tick.
    """
    named = [
        (name, value)
        for name, value in (("left_gripper", left), ("right_gripper", right))
        if value is not None
    ]
    return JointState(
        name=[name for name, _ in named] or ["left_joint1"],
        position=[value for _, value in named] or [0.0],
        effort=[0.0] * max(len(named), 1),
        stamp_ns=stamp_ns,
    )


def test_the_heartbeat_evidence_needs_every_gripper_for_n_samples_in_a_row() -> None:
    """Evidence is live only after N complete samples; one hand going quiet restarts it.

    Real bimanual bridge, no ROS: ``observe_joint_state`` with unloaded jaws
    fires no event, so it never touches the (absent) node.
    """
    bridge = VisionAttachmentBridge(
        None, _openarm(), config=VisionAttachmentConfig(camera="head_zed")
    )
    unloaded = 0.01  # an open-ish jaw, no command yet: present, never an event
    now = time.monotonic
    for tick in range(2):
        bridge.observe_joint_state(
            _gripper_sample(left=unloaded, right=unloaded, stamp_ns=_at(tick))
        )
    assert not bridge._evidence.live(now_s=now()), "two samples are under the debounce"
    bridge.observe_joint_state(_gripper_sample(left=unloaded, right=None, stamp_ns=_at(2)))
    bridge.observe_joint_state(_gripper_sample(left=unloaded, right=unloaded, stamp_ns=_at(3)))
    assert not bridge._evidence.live(now_s=now()), "a missing hand must restart the count"
    for tick in range(4, 6):
        bridge.observe_joint_state(
            _gripper_sample(left=unloaded, right=unloaded, stamp_ns=_at(tick))
        )
    assert bridge._evidence.live(now_s=now())
    assert not bridge._evidence.live(now_s=now() + 0.6), "evidence older than 0.5 s is dead"
    bridge.observe_joint_state(_gripper_sample(left=float("nan"), right=unloaded, stamp_ns=_at(6)))
    assert not bridge._evidence.live(now_s=now()), "a non-finite jaw position is a dead channel"
    assert bridge.missing_position_ticks == 2
    later = now() + 1.0
    bridge._evidence.observe(complete=True, stamp_ns=_at(7), now_s=later)
    assert not bridge._evidence.live(now_s=later), "a channel back from a gap must re-earn N"


def test_a_repeated_cached_joint_state_is_no_heartbeat_evidence() -> None:
    """One cached sample re-read every tick (stamp jittering by us) never makes evidence live.

    Nor does it keep evidence alive once real samples stop: liveness is refreshed only by
    a new sample.
    """
    bridge = VisionAttachmentBridge(
        None, _openarm(), config=VisionAttachmentConfig(camera="head_zed")
    )
    for i in range(30):
        jitter = (i % 5 - 2) * 1_000
        bridge.observe_joint_state(_gripper_sample(left=0.5, right=-0.5, stamp_ns=_at(0) + jitter))
    assert not bridge._evidence.live(now_s=time.monotonic()), "one sample, re-read, is one sample"
    for tick in range(1, 4):
        bridge.observe_joint_state(_gripper_sample(left=0.5, right=-0.5, stamp_ns=_at(tick)))
    assert bridge._evidence.live(now_s=time.monotonic())
    last_s = time.monotonic()
    bridge._evidence.observe(complete=True, stamp_ns=_at(3) + 2_000, now_s=last_s + 0.4)
    assert not bridge._evidence.live(now_s=last_s + 0.6), "a re-read refreshed liveness"
    assert all(leg.trigger.repeated_samples == 29 for leg in bridge._legs)


def test_the_segment_request_is_sent_in_the_cameras_optical_frame() -> None:
    """Empty ``frame_id`` and the prompts carried by ``T_cam_from_link``.

    The segmenter reads an empty frame as "already in my optical frame", so the
    points must be the attach-link prompts moved through the inverse of the
    tf2 ``link <- camera`` transform the bridge already holds.
    """
    pytest.importorskip("openral_msgs")
    from openral_core.geometry import homogeneous_from_quat_xyz
    from openral_hal.vision_attachment_bridge import build_segment_request

    # A generic link <- optical transform (non-trivial rotation and offset).
    half = np.sqrt(0.5)
    t_link_from_cam = homogeneous_from_quat_xyz((0.1, -0.2, 0.35), (half, 0.0, half, 0.0))
    tcp = (-0.00143, -0.018, -0.068)  # OpenArm left_gripper origin_xyz
    tip = (0.0, 0.03, -0.1)
    request = build_segment_request(
        stamp_ns=1_500_000_000,
        camera="head_zed",
        t_link_from_cam=t_link_from_cam,
        tcp_in_link=tcp,
        negatives_in_link=[tip],
    )
    t_cam_from_link = np.linalg.inv(t_link_from_cam)
    assert request.frame_id == ""
    assert request.camera == "head_zed"
    assert (request.stamp.sec, request.stamp.nanosec) == (1, 500_000_000)
    for sent, point in ((request.tcp_point, tcp), (request.negative_points[0], tip)):
        np.testing.assert_allclose(
            (sent.x, sent.y, sent.z), (t_cam_from_link @ np.array([*point, 1.0]))[:3]
        )
