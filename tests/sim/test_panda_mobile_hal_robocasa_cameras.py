"""Only cameras the RoboCasa scene can render get a ``/openral/cameras/*`` topic.

``robots/panda_mobile/robot.yaml`` declares four RGB sensors, but one of them
— ``head`` — is a *synthetic* forward navigation camera that the RoboCasa
backend renders only when ``OPENRAL_ROBOCASA_HEAD_CAM=1`` is exported and the
scene carries a mobile base. Nothing in the repo exports it, so on every
manipulation scene the manifest declares a camera the backend never produces.

``SimSensorBridge`` used to advertise a publisher for every declared RGB
sensor at configure time, so ``/openral/cameras/head/image`` appeared on the
graph and then stayed silent forever. A subscriber cannot tell that apart from
a slow stream, so it waits indefinitely (observed live on
``scenes/deploy/robocasa_baguette.yaml``: the topic was listed by ``ros2 topic
list`` and a recorder captured 0 frames). Cameras are now advertised lazily,
on their first real frame.

No mocks (CLAUDE.md §1.11): real ``panda_mobile`` manifest, real RoboCasa
``PickPlaceCounterToCabinet`` env via the deploy-sim HAL path, real rclpy node,
real ``SimSensorBridge``.

Gates: RoboCasa assets + ``mujoco`` for the backend, ``ROS_DISTRO`` + ``rclpy``
for the bridge. Skips cleanly when either half is missing.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("openral_sim")
pytest.importorskip("mujoco")
pytest.importorskip("robocasa")  # robocasa (robosuite >=1.5) ⊥ libero (robosuite 1.4)

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)
pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — the bridge half of this test needs a sourced ROS 2 install.",
)

rclpy = pytest.importorskip("rclpy")

from openral_core import (  # noqa: E402  # reason: after importorskip gates
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    JointState,
    Pose6D,
    RobotDescription,
    WorldState,
)
from openral_hal import build_hal  # noqa: E402
from openral_hal.sim_sensor_bridge import SimSensorBridge  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PANDA_MOBILE = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
_BAGUETTE = _REPO_ROOT / "scenes" / "deploy" / "robocasa_baguette.yaml"

# The three arm-mounted cameras robosuite really owns for a PandaMobile scene
# (robot0_agentview_left / _right / eye_in_hand). The HAL hands them up keyed by
# the VLA camera slot; ``SimSensorBridge`` maps each manifest sensor name onto
# its slot via ``vla_feature_key`` at publish time.
_RENDERED = {"shoulder_left", "shoulder_right", "wrist"}
_RENDERED_OBS_KEYS = {"camera1", "camera2", "camera3"}
# ``head`` is the odd one out: its manifest ``vla_feature_key`` is
# ``observation.images.head``, so sensor name and obs key coincide — neither
# appears in the backend's obs dict unless the synth renderer is switched on.
_SYNTHETIC_OPT_IN = "head"


def _attachment(object_id: str, body_name: str) -> AttachedCollisionObject:
    return AttachedCollisionObject(
        object_id=object_id,
        attach_link="panda_link7",
        touch_links=["panda_finger_pair"],
        primitives=[
            AttachedCollisionPrimitive(
                shape=BoxShape(half_extents_m=(0.1, 0.03, 0.03)),
                pose_in_object=Pose6D(
                    xyz=(0.0, 0.0, 0.0),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id=object_id,
                ),
            )
        ],
        pose_in_link=Pose6D(
            xyz=(0.0, 0.0, 0.16),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="panda_link7",
        ),
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.SIM_CONTACT,
        evidence_ref=f"mujoco_body:{body_name}",
        stamp_ns=1,
    )


@pytest.fixture
def hal() -> Iterator[Any]:
    """A connected panda_mobile HAL attached to the real baguette scene."""
    desc = RobotDescription.from_yaml(str(_PANDA_MOBILE))
    built = build_hal(desc, mode="sim", sim_env_yaml=str(_BAGUETTE))
    built.connect()
    try:
        yield built
    finally:
        built.disconnect()


def test_head_camera_is_not_rendered_without_the_opt_in(
    hal: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ground truth: the backend supplies no ``head`` frame by default.

    This is the fact the whole bug rests on — robosuite/RoboCasa own no
    nav-facing camera, and OpenRAL's synthetic replacement is env-gated off.
    """
    monkeypatch.delenv("OPENRAL_ROBOCASA_HEAD_CAM", raising=False)
    images = hal.read_images()
    assert set(images) >= _RENDERED_OBS_KEYS, f"expected the arm cameras, got {sorted(images)}"
    assert _SYNTHETIC_OPT_IN not in images, (
        "robocasa rendered a 'head' frame with OPENRAL_ROBOCASA_HEAD_CAM unset — "
        "the opt-in gate regressed"
    )


def test_bridge_advertises_only_renderable_cameras(
    hal: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bridge puts a topic on the graph only for cameras that produce frames."""
    monkeypatch.delenv("OPENRAL_ROBOCASA_HEAD_CAM", raising=False)
    desc = RobotDescription.from_yaml(str(_PANDA_MOBILE))
    assert _SYNTHETIC_OPT_IN in {s.name for s in desc.sensors if s.modality == "rgb"}, (
        "fixture drift: panda_mobile no longer declares a 'head' RGB sensor, so "
        "this regression guard would pass vacuously"
    )

    rclpy.init()
    node = rclpy.create_node("test_sim_bridge_camera_advertise")
    bridge = SimSensorBridge(node=node, hal=hal, description=desc, viewer_enabled=False)
    try:
        bridge.setup()
        # Nothing is advertised until a frame lands, so drive the publish tick
        # directly rather than racing the timer.
        for _ in range(3):
            hal.read_state()
            bridge._publish_images()

        advertised = {
            name
            for name, _types in node.get_topic_names_and_types()
            if name.startswith("/openral/cameras/") and name.endswith("/image")
        }
        for cam in _RENDERED:
            assert f"/openral/cameras/{cam}/image" in advertised, (
                f"'{cam}' renders frames but was never advertised; got {sorted(advertised)}"
            )
        assert f"/openral/cameras/{_SYNTHETIC_OPT_IN}/image" not in advertised, (
            "a camera the backend never renders was advertised — a subscriber "
            "would block on it forever"
        )
    finally:
        bridge.teardown()
        node.destroy_node()
        rclpy.shutdown()


def test_bridge_masks_multiple_attached_objects_without_reset(hal: Any) -> None:
    """Two carried RoboCasa bodies leave depth atomically and detach independently."""
    from openral_hal.depth_cloud import depth_synth_kwargs
    from openral_msgs.msg import AttachmentState
    from openral_sim.backends.depth_camera import synthesize_depth_pointcloud
    from openral_world_state_ros.lifecycle_node import build_world_state_stamped_msg
    from std_msgs.msg import UInt64

    desc = RobotDescription.from_yaml(str(_PANDA_MOBILE))
    handles = hal.mujoco_handles()
    assert handles is not None
    model, data = handles
    depth_spec = next(sensor for sensor in desc.sensors if sensor.name == "front_depth")
    kwargs = depth_synth_kwargs(
        depth_spec,
        max_range_default=5.0,
        render_size=(512, 512),
    )

    rclpy.init()
    node = rclpy.create_node("test_sim_bridge_attached_object_mask")
    bridge = SimSensorBridge(node=node, hal=hal, description=desc, viewer_enabled=False)
    try:
        bridge._resolve_depth_base_body(model)

        def point_count() -> int:
            return int(
                synthesize_depth_pointcloud(
                    model=model,
                    data=data,
                    stride=4,
                    exclude_body_ids=bridge._depth_excluded_body_ids(),
                    **kwargs,
                ).shape[0]
            )

        baseline = point_count()
        baguette = _attachment("baguette_seed1", "obj_main")
        distractor = _attachment("counter_distractor_seed1", "distr_counter_main")

        revision = 0

        def publish_attachment_state(
            attachments: list[AttachedCollisionObject],
        ) -> int:
            nonlocal revision
            revision += 1
            wire = build_world_state_stamped_msg(
                None,
                WorldState(
                    stamp_ns=1,
                    joint_state=JointState(
                        name=["panda_joint1"],
                        position=[0.0],
                        stamp_ns=1,
                    ),
                    attached_objects=attachments,
                ),
            )
            state = AttachmentState()
            state.revision = revision
            state.objects = list(wire.attached_objects)
            bridge._on_attachment_state(state)
            return revision

        attached_revision = publish_attachment_state([baguette, distractor])
        assert point_count() == baseline  # additions wait for kernel acknowledgement
        bridge._on_attachment_state_applied(UInt64(data=attached_revision))
        both_masked = point_count()
        assert both_masked < baseline

        detached_revision = publish_attachment_state([baguette])
        baguette_only = point_count()
        assert both_masked < baguette_only < baseline
        bridge._on_attachment_state_applied(UInt64(data=detached_revision))
        assert point_count() == baguette_only

        empty_revision = publish_attachment_state([])
        assert point_count() == baseline
        bridge._on_attachment_state_applied(UInt64(data=empty_revision))
        assert point_count() == baseline
    finally:
        node.destroy_node()
        rclpy.shutdown()
