"""A dual-Rizon one-arm slot row passes the kernel's position check only with in-limit padding.

The kernel checks EVERY joint of a ``JOINT_POSITION`` row against its position limits,
uncommanded ones included (``validator.cpp``). The dual Rizon's ``left_joint7`` is limited to
``[-2.9671, -0.2231]`` (home ±90° on the reference rig), so a right-arm slot that the runner
zero-padded was refused at that index. ``_pad_joint_payload`` now pads with
``_position_fill`` (each joint's in-limit value nearest 0.0). This proves both on the real
``safety_kernel_node`` binary with the real ``robots/rizon_dual/robot.yaml`` envelope and
collision model, both arms measured at the SRDF ``home`` pose:

=====================================================  ==============================
right-arm slot at home, left joints padded with         verdict
=====================================================  ==============================
``_position_fill`` (left_joint7 = -0.2231)              ACCEPTED
``0.0`` (the old padding)                               REFUSED at ``left_joint7``
=====================================================  ==============================

Real throughout (CLAUDE.md §1.11): the real manifest, the runner's own padding helpers, the
kernel parameters ``deploy_e2e.launch.py`` derives.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from types import ModuleType
from typing import Any

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

from openral_core import RobotDescription  # noqa: E402
from openral_safety.envelope_loader import (  # noqa: E402
    collision_params_from_description,
    compute_intersection,
    kernel_params_from_envelope,
)

from tests.sim.safety._kernel_subprocess import (  # noqa: E402
    activate_kernel_node,
    isolated_domain_id,
    start_kernel,
    terminate_kernel,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_ROBOT_DIR = _REPO / "robots" / "rizon_dual"
_RUNNER = _REPO / "packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py"
_RSKILL_ID = "openral/rizon-slot-padding"


def _description() -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROBOT_DIR / "robot.yaml"))


def _home() -> dict[str, float]:
    root = ET.parse(_ROBOT_DIR / "rizon_dual.srdf").getroot()
    state = root.find("group_state[@name='home']")
    assert state is not None
    return {str(j.get("name")): float(str(j.get("value"))) for j in state.findall("joint")}


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rskill_runner_node_padding", _RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kernel_params(description: RobotDescription) -> dict[str, object]:
    params: dict[str, object] = {
        **kernel_params_from_envelope(compute_intersection(description, skill=None, deploy=None)),
        **collision_params_from_description(description),
    }
    params["collision_joint_names"] = [j.name for j in description.joints]
    params["collision_seed_dt_s"] = 0.0
    params["collision_state_deadline_ms"] = 1000.0
    return params


def _right_slot(row: list[float], right: list[str], trace: str) -> Any:
    from openral_msgs.msg import ActionChunk

    chunk = ActionChunk()
    chunk.control_mode = 0  # JOINT_POSITION
    chunk.horizon = 1
    chunk.n_dof = len(row)
    chunk.flat = row
    chunk.joint_names = right
    chunk.rskill_id = _RSKILL_ID
    chunk.trace_id = trace
    return chunk


def test_a_one_arm_slot_passes_only_with_in_limit_padding() -> None:
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty

    description = _description()
    joint_names = [j.name for j in description.joints]
    home = _home()
    measured = [home[n] for n in joint_names]
    right = [f"right_joint{i}" for i in range(1, 8)]
    index = {n: i for i, n in enumerate(joint_names)}
    runner = _runner()
    target = [home[n] for n in right]
    filled = runner._pad_joint_payload(
        target, right, index, len(joint_names), runner._position_fill(description)
    )
    zeroed = runner._pad_joint_payload(target, right, index, len(joint_names))
    assert filled[index["left_joint7"]] == -0.2231
    assert zeroed[index["left_joint7"]] == 0.0

    node_name = f"safety_kernel_rizon_pad_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        proc = start_kernel(
            _kernel_params(description),
            node_name,
            isolated_domain_id(),
            log_path=log_path,
            params_file=pathlib.Path(td) / "kernel_params.yaml",
        )
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("rizon_pad_helper")
                assert activate_kernel_node(node_name, helper), "kernel activation failed"
                executor = SingleThreadedExecutor()
                executor.add_node(helper)
                safe: dict[str, Any] = {}
                failures: list[Any] = []
                estops: list[Any] = []
                chunk_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
                safe_sub = helper.create_subscription(
                    ActionChunk,
                    "/openral/safe_action",
                    lambda m: safe.__setitem__(m.trace_id, m),
                    10,
                )
                helper.create_subscription(
                    FailureTrigger, "/openral/failure/safety", failures.append, 10
                )
                helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
                joint_pub = helper.create_publisher(JointState, "/joint_states", 10)

                def publish_joint_state() -> None:
                    js = JointState()
                    js.header.stamp = helper.get_clock().now().to_msg()
                    js.name = joint_names
                    js.position = list(measured)
                    joint_pub.publish(js)

                helper.create_timer(0.05, publish_joint_state)

                def spin(seconds: float) -> None:
                    end = time.time() + seconds
                    while time.time() < end:
                        executor.spin_once(timeout_sec=0.02)

                deadline = time.time() + 5.0
                while time.time() < deadline and not (
                    chunk_pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1
                ):
                    executor.spin_once(timeout_sec=0.05)

                def send(chunk: Any) -> None:
                    chunk_pub.publish(chunk)
                    end = time.time() + 10.0
                    while time.time() < end and chunk.trace_id not in safe and not estops:
                        executor.spin_once(timeout_sec=0.02)
                    spin(0.4)

                spin(0.6)
                send(_right_slot(filled, right, "filled"))
                assert "filled" in safe, "in-limit padding must pass the position check"
                assert not estops

                send(_right_slot(zeroed, right, "zeroed"))
                assert "zeroed" not in safe, "0.0 at left_joint7 is outside its limits"
                trigger = failures[-1]
                assert trigger.kind == FailureTrigger.KIND_WORKSPACE
                evidence = json.loads(trigger.evidence_json)
                assert evidence["kind"] == "workspace", evidence
                assert evidence["ee_name"] == "left_joint7", evidence  # the joint, by name
            finally:
                rclpy.shutdown()
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            ) from exc
        finally:
            terminate_kernel(proc)
