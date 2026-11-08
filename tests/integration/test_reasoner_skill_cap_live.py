"""A capped reasoner node offers the LLM only the top-k skills of its seeded palette (#494).

``ReasonerCore`` narrowing is unit-tested in ``tests/unit/test_skill_subset.py``;
this drives the path a deployment takes: ``OPENRAL_REASONER_MAX_SKILL_TOOLS`` is
read by ``resolve_max_skill_tools`` in ``ReasonerNode.on_configure``, the palette
is seeded from the real ``rskills/`` manifests for ``franka_panda``, and an
operator prompt on ``/openral/prompt`` preempts a real tick. The LLM endpoint is
the ``FakeToolUseClient`` process-boundary fake, which records the palette it was
offered.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` (listed in ``scripts/ros_live_tests.sh``)::

    just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest tests/integration/test_reasoner_skill_cap_live.py -v
"""

from __future__ import annotations

import os
import pathlib
import time

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy node construction — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "and source install/setup.bash first."
)

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_RSKILLS = _REPO_ROOT / "rskills"
_ROBOT_YAML = _REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"
_CAP = 2
_ASK = "move the end-effector to the pre-grasp pose"
_EEF_POSE = "OpenRAL/rskill-moveit-multi-eef_pose"


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_capped_node_offers_top_k_skills_on_a_real_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client sees ``k`` skills chosen for the operator's ask; the palette is intact."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import EmitPromptTool
    from openral_msgs.msg import PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    monkeypatch.setenv("OPENRAL_REASONER_MAX_SKILL_TOOLS", str(_CAP))
    offered: list[ToolPalette] = []

    def select(_context: str, palette: ToolPalette) -> EmitPromptTool:
        offered.append(palette)
        return EmitPromptTool(target_topic="/openral/prompt", text="ack", rationale="test")

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"robot_yaml:={_ROBOT_YAML}",
            "-p",
            "grasp_target_voxel_m:=0.05",
            "-p",
            f"rskill_search_paths:=[{_RSKILLS}]",
        ],
    )
    try:
        reasoner = ReasonerNode(client=FakeToolUseClient(selector=select))
        reasoner.trigger_configure()
        reasoner.trigger_activate()
        assert reasoner._core is not None
        assert reasoner._core.max_skill_tools == _CAP
        full = reasoner._palette
        assert len(full.skills) > _CAP, (
            f"fixture drift: the seeded franka palette has {len(full.skills)} skills, "
            f"not more than the cap of {_CAP}, so nothing would be narrowed"
        )

        pub_node = rclpy.create_node("openral_test_skill_cap_publisher")
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", qos)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(pub_node)

        msg = PromptStamped()
        msg.header.stamp = pub_node.get_clock().now().to_msg()
        msg.header.frame_id = "openral_test_skill_cap_publisher"
        msg.text = _ASK
        msg.metadata_json = '{"priority": 100}'
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not offered:
            pub.publish(msg)
            executor.spin_once(timeout_sec=0.2)

        executor.remove_node(reasoner)
        executor.remove_node(pub_node)
        pub_node.destroy_node()
        reasoner.trigger_deactivate()
        reasoner.trigger_cleanup()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert offered, "the reasoner never called the LLM within 10 s of the operator prompt"
    first = offered[0]
    assert len(first.skills) == _CAP
    assert first.execute_rskill_ids <= full.execute_rskill_ids
    assert len(full.skills) > _CAP  # the node's own palette was not narrowed in place
    if _EEF_POSE in full.execute_rskill_ids:
        assert _EEF_POSE in first.execute_rskill_ids  # the operator's ask keeps its slot
    # Everything but the skills is offered unchanged.
    assert first.model_dump(exclude={"skills", "execute_rskill_ids"}) == full.model_dump(
        exclude={"skills", "execute_rskill_ids"}
    )
