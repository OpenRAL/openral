# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: #418 tracker -> World State aggregator -> the real C++ safety kernel.

The rig (``_tracker_rig``) runs the real ``object_tracker_node`` and the real World State
lifecycle node on the SO-101 PickOrange ``top_depth`` camera. The kernel is the real
``safety_kernel_node`` binary, started with the exact parameters ``deploy_e2e.launch.py``
composes for ``scenes/deploy/isaac_leisaac_pick_orange.yaml`` at ``object_tracking: tracking``
(``use_sim_time`` off: the rig runs on wall time).

Asserts, issue #418 test matrix A ("Integration"):

* every hop's QoS is its data class (CLAUDE.md §2): sensor inputs BEST_EFFORT/VOLATILE,
  ``tracked_objects`` and ``world_state_fast`` RELIABLE/VOLATILE (the kernel's own
  subscription included), ``attachment_state`` RELIABLE/TRANSIENT_LOCAL, grasp events
  RELIABLE/VOLATILE;
* the kernel ingests what the tracker publishes: it acknowledges the tracker's
  ``attachment_state`` revision, and with the set accepted it never reports the tracked
  binding as having ``no_accepted_set``;
* the kernel arms its ADR-0115 grasp allowance over the tracker's box of the grounded,
  unheld orange (``evidence=tracked:trk:1``), and the ATTACH snapshot still carries it, so
  the kernel latches that box at handover (ADR-0115) instead of dropping the region;
* in ``tracking`` the tracker is the only ``attachment_state`` author, with the HAL's vision
  attachment bridge wired the way the launch wires it in ``tracking``;
* a frame mismatch is refused end to end: once the map arrives in another frame than the
  tracker's set, the kernel logs ``safety.tracked_objects_rejected reason=frame_mismatch``
  and the binding loses its accepted set.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` plus a built ``openral_safety_kernel``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.integration._tracker_rig import tracker_rig, wait

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy nodes + colcon openral_msgs/openral_safety_kernel overlay — set "
    "OPENRAL_TEST_ROS_LIVE=1 and source install/setup.bash first.",
)

_REPO = Path(__file__).resolve().parents[2]
_SCENE = _REPO / "scenes" / "deploy" / "isaac_leisaac_pick_orange.yaml"
_LAUNCH = _REPO / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"


def _kernel_params(mode: str) -> dict[str, object]:
    """The kernel parameters ``deploy_e2e.launch.py`` composes for PickOrange in ``mode``."""
    for dep in ("launch", "launch_ros", "openral_cli", "mujoco"):
        pytest.importorskip(dep, reason=f"{dep} composes the deploy launch")
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters
    from openral_cli.deploy_sim import resolve_launch_invocation

    scene = yaml.safe_load(_SCENE.read_text())
    scene["runtime"] = {**(scene.get("runtime") or {}), "object_tracking": mode}
    scene["runtime"]["grounding_wait_s"] = 5.0
    path = Path(tempfile.mkdtemp()) / "scene.yaml"
    path.write_text(yaml.safe_dump(scene))
    invocation = resolve_launch_invocation(
        config=path,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        deploy_config=path,
        hal_param_overrides={"viewer_enabled": False},
        hal_mode="sim",
        enable_dashboard=False,
    )
    args = dict(t.split(":=", 1) for t in invocation.argv_template if ":=" in t)
    args["hal_params_file"] = "/tmp/openral-test-hal-params.yaml"
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_kernel_live", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_kernel_live"] = module
    spec.loader.exec_module(module)
    ctx = LaunchContext()
    ctx.launch_configurations.update(args)
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    ctx.launch_configurations.update(args)
    ctx.launch_configurations["enable_dashboard"] = "false"
    (kernel,) = [
        e
        for e in module.compose_runtime_graph(ctx)
        if isinstance(e, Node)
        and getattr(e, "_Node__node_executable", None) == "safety_kernel_node"
    ]
    params: dict[str, object] = {}
    for p in evaluate_parameters(ctx, kernel._Node__parameters):
        if isinstance(p, dict):
            params.update(p)
    params["use_sim_time"] = False
    return params


def _activate(helper: Any, node_name: str) -> bool:
    """CONFIGURE then ACTIVATE the kernel; the rig's executor spins ``helper``."""
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState

    client = helper.create_client(ChangeState, f"/{node_name}/change_state")
    if not client.wait_for_service(timeout_sec=15.0):
        return False
    for t in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
        req = ChangeState.Request()
        req.transition.id = t
        fut = client.call_async(req)
        if not wait(fut.done, 15.0) or not fut.result().success:
            return False
    return True


def _qos(helper: Any, topic: str, node: str, *, publisher: bool) -> tuple[str, str]:
    """(reliability, durability) of ``node``'s endpoint on ``topic``."""
    infos = (
        helper.get_publishers_info_by_topic(topic)
        if publisher
        else helper.get_subscriptions_info_by_topic(topic)
    )
    (info,) = [i for i in infos if i.node_name == node]
    return info.qos_profile.reliability.name, info.qos_profile.durability.name


def test_tracker_to_kernel_ingest_qos_and_frame_mismatch_refusal() -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    if shutil.which("ros2") is None:
        pytest.skip("ros2 not on PATH; source install/setup.bash first")
    from openral_core import CameraTopicKind, camera_topic
    from rcl_interfaces.msg import Log
    from std_msgs.msg import UInt64

    from tests.sim.safety._kernel_subprocess import start_kernel, terminate_kernel

    params = _kernel_params("tracking")
    assert params["attached_collision_enabled"] is True
    # ADR-0121 step 3: the K5 release decision is a provider rule; no provider, not emitted.
    assert "tracked_release_jaw_joints" not in params
    name = f"openral_safety_kernel_{uuid.uuid4().hex[:8]}"
    log_path = Path(tempfile.mkdtemp()) / "kernel.log"
    domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    kernel = start_kernel(
        params,
        name,
        domain,
        log_path=log_path,
        params_file=log_path.with_suffix(".yaml"),
    )
    try:
        with tracker_rig() as rig:
            from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

            logs: list[str] = []
            acks: list[int] = []
            rig.helper.create_subscription(
                Log,
                "/rosout",
                lambda m: logs.append(m.msg) if m.name == name else None,
                QoSProfile(depth=1000, reliability=QoSReliabilityPolicy.RELIABLE),
            )
            rig.helper.create_subscription(
                UInt64,
                "/openral/attachment_state_applied",
                lambda m: acks.append(int(m.data)),
                QoSProfile(
                    depth=1,
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
            assert _activate(rig.helper, name), log_path.read_text(errors="replace")[-4000:]
            tracker = rig.tracker("tracking")
            rig.declare()
            assert rig.ground()
            assert wait(lambda: bool(rig.fast) and rig.fast[-1].tracked_objects_valid, 5.0)
            assert params["grasp_allowance_enabled"] is True

            # QoS per data class, read off the live graph (the kernel's endpoints included).
            h, tn, ws = rig.helper, tracker.get_name(), rig.world_state.get_name()
            reliable, latched = ("RELIABLE", "VOLATILE"), ("RELIABLE", "TRANSIENT_LOCAL")
            assert _qos(h, "/openral/tracked_objects", tn, publisher=True) == reliable
            assert _qos(h, "/openral/world_state_fast", ws, publisher=True) == reliable
            assert _qos(h, "/openral/world_state_fast", name, publisher=False) == reliable
            assert _qos(h, "/openral/attachment_state", tn, publisher=True) == latched
            assert _qos(h, "/openral/grasp_events", tn, publisher=False) == reliable
            for kind in (
                CameraTopicKind.DEPTH_IMAGE,
                CameraTopicKind.DEPTH_CAMERA_INFO,
                CameraTopicKind.ROBOT_MASK,
            ):
                topic = camera_topic("top", kind)
                assert _qos(h, topic, tn, publisher=False)[0] == "BEST_EFFORT", topic
            assert _qos(h, "/joint_states", tn, publisher=False)[0] == "BEST_EFFORT"

            def hold_frames(grid_frame: str | None = None) -> None:
                rig.publish_hold()
                rig.frame(grid_frame)
                time.sleep(0.1)

            # The grounded, unheld orange's box arms the kernel's grasp allowance.
            def armed() -> bool:
                hold_frames()
                return any(
                    "safety.grasp_region_armed target=orange" in m and "evidence=tracked:trk:1" in m
                    for m in logs
                )

            assert wait(armed, 10.0), [m for m in logs if "grasp_region" in m]

            # ATTACH at the hold. The kernel acknowledges the tracker's revision.
            rig.attach_event()
            assert wait(lambda: bool(rig.attach[-1].objects), 5.0)
            revision = int(rig.attach[-1].revision)

            def acked() -> bool:
                hold_frames()
                return revision in acks

            assert wait(acked, 10.0), (acks, revision)
            for _ in range(20):  # 2 s with the set accepted and the arm at the hold
                hold_frames()
            assert not [m for m in logs if "safety.tracked_objects_rejected" in m], logs
            assert not [m for m in logs if "reason=no_accepted_set" in m], logs
            # Held: the attachment snapshot carries the box, and the kernel latches it.
            assert any(
                "safety.grasp_region_latched target=orange at handover" in m for m in logs
            ), [m for m in logs if "grasp_region" in m]
            assert not [m for m in logs if "grasp_region_dropped reason=no_region" in m], logs

            # The map arrives in another frame than the tracker's set: refused whole.
            n = len(logs)

            def refused() -> bool:
                hold_frames("odom")
                return any(
                    "safety.tracked_objects_rejected reason=frame_mismatch" in m for m in logs[n:]
                )

            assert wait(refused, 10.0), logs[n:]
            assert rig.fast[-1].tracked_objects_valid  # still on the wire; the kernel refuses

            def off_hand() -> bool:
                hold_frames("odom")
                return any(
                    "safety.attachment_binding_off_hand object=trk:1" in m
                    and "reason=no_accepted_set" in m
                    for m in logs[n:]
                )

            assert wait(off_hand, 5.0), logs[n:]
            tracker.destroy_node()
    finally:
        terminate_kernel(kernel)


def test_in_tracking_the_tracker_is_the_only_attachment_state_author() -> None:
    """The HAL's vision attachment bridge, as ``tracking`` wires it, opens no publisher."""
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    from openral_hal.vision_attachment_bridge import (
        VisionAttachmentBridge,
        VisionAttachmentConfig,
    )

    with tracker_rig() as rig:
        tracker = rig.tracker("tracking")
        hal = VisionAttachmentBridge(
            rig.helper,
            rig.description,
            config=VisionAttachmentConfig(camera="wrist", publish_attachment_state=False),
        )
        hal.setup()
        rig.declare()
        assert rig.ground()

        def authors() -> list[str]:
            return sorted(
                i.node_name
                for i in rig.helper.get_publishers_info_by_topic("/openral/attachment_state")
            )

        assert wait(lambda: authors() == [tracker.get_name()], 5.0), authors()
        rig.publish_hold()
        rig.attach_event()
        assert wait(lambda: bool(rig.attach) and bool(rig.attach[-1].objects), 5.0)
        assert authors() == [tracker.get_name()]
        # The control: the same bridge as `off` wires it is a second author.
        VisionAttachmentBridge(
            rig.helper,
            rig.description,
            config=VisionAttachmentConfig(camera="wrist"),
        ).setup()
        assert wait(lambda: len(authors()) == 2, 5.0), authors()
        tracker.destroy_node()
