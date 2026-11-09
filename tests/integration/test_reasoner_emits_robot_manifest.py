"""The deploy's reasoner node emits the robot manifest (ADR-0129 Interface A).

At configure, and again on ``/openral/skill_registry_changed``, the reasoner builds the
manifest from the robot it was given, the rSkills on its search paths and the launch's
resolved ``DeployRuntime`` — and sends it as an ``openral.robot.manifest`` OTLP log
record. Driven against the real ``robots/so101_follower`` + ``rskills/`` tree, a real
rclpy graph and the real OTel SDK (an in-memory exporter in place of the network).

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the rest of the live reasoner suite::

    just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest \\
        tests/integration/test_reasoner_emits_robot_manifest.py -v \\
        -p no:launch_testing -p no:launch_ros
"""

from __future__ import annotations

import os
import pathlib
import time

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_ROBOT_YAML = _REPO_ROOT / "robots" / "so101_follower" / "robot.yaml"
_SKILL = "skill:OpenRAL/rskill-smolvla-so101-eraser_place-bf16"


@pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy node construction — set OPENRAL_TEST_ROS_LIVE=1, source install/setup.bash",
)
def test_reasoner_emits_the_manifest_at_configure_and_on_registry_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import DeployRuntime, RobotManifest
    from openral_observability import semconv
    from openral_reasoner_ros import ReasonerNode
    from opentelemetry._logs import get_logger_provider, set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
    from std_msgs.msg import Empty

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]  # reason: unannotated SDK __init__
    provider = get_logger_provider()
    if not isinstance(provider, LoggerProvider):
        provider = LoggerProvider()
        set_logger_provider(provider)
    processor = SimpleLogRecordProcessor(exporter)
    provider.add_log_record_processor(processor)
    monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", "1")
    monkeypatch.delenv("OPENRAL_DASHBOARD_REMOTE_CONTROL", raising=False)

    runtime = DeployRuntime(enable_reasoner=True, enable_slam=False).model_dump_json(
        exclude_none=True
    )
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"robot_yaml:={_ROBOT_YAML}",
            "-p",
            "grasp_target_voxel_m:=0.05",
            "-p",
            f"rskill_search_paths:=[{_REPO_ROOT / 'rskills'}]",
            "-p",
            f"manifest_runtime_json:='{runtime}'",
            "-p",
            "manifest_bridge_on:=true",
            "-p",
            "manifest_unit_serial:=SN-123",
        ],
    )
    try:
        reasoner = ReasonerNode(tick_hz=2.0, client=FakeToolUseClient(responses=[]))
        reasoner.trigger_configure()

        def manifests() -> list[RobotManifest]:
            return [
                RobotManifest.model_validate_json(str(r.log_record.body))
                for r in exporter.get_finished_logs()
                if r.log_record.event_name == semconv.EVENT_ROBOT_MANIFEST
            ]

        (first,) = manifests()
        assert (first.robot_id, first.unit_serial) == ("so101_follower", "SN-123")
        assert {"cam:top", "cam:wrist", "reasoner", "viz3d"} <= {p.id for p in first.panels}
        assert "map" not in {p.id for p in first.panels}
        actions = {a.id: a for a in first.actions}
        assert {"estop", "estop_reset", "prompt", _SKILL} <= set(actions)
        assert actions["estop_reset"].remote is False

        # `ral skill install|remove` fires the registry topic (latched: TRANSIENT_LOCAL).
        from openral_reasoner_ros.reasoner_node import _QOS_REGISTRY_CHANGED

        helper = rclpy.create_node("manifest_registry_poke")
        pub = helper.create_publisher(
            Empty, "/openral/skill_registry_changed", _QOS_REGISTRY_CHANGED
        )
        pub.publish(Empty())
        deadline = time.monotonic() + 5.0
        while len(manifests()) < 2 and time.monotonic() < deadline:
            rclpy.spin_once(reasoner, timeout_sec=0.2)
        assert len(manifests()) >= 2, "a registry change did not re-emit the manifest"
        assert manifests()[-1] == first
        helper.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()
        # The provider is process-global: later tests must not keep writing to this one.
        processor.shutdown()  # type: ignore[no-untyped-call]  # reason: unannotated SDK method
