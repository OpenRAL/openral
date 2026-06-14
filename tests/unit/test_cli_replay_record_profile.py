"""ADR-0018 F7 + F9 — ``openral replay`` / ``openral record`` / ``openral profile session``.

These tests exercise the typer surface with real options against real
mcap fixtures (F7) and the real LTTng module's gate-off path (F9). No
ROS 2 install, no ``lttng`` binary required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openral_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _write_one_message_bag(bag_path: Path, traceparent: str, topic: str) -> None:
    """Tiny mcap fixture — one jsonschema-encoded message carrying ``traceparent``."""
    from mcap.writer import Writer

    with bag_path.open("wb") as f:
        w = Writer(f)
        w.start()
        schema_id = w.register_schema(
            name="openral_msgs/FailureTrigger",
            encoding="jsonschema",
            data=b"{}",
        )
        channel_id = w.register_channel(
            topic=topic, message_encoding="jsonschema", schema_id=schema_id
        )
        w.add_message(
            channel_id=channel_id,
            log_time=1_000_000_000,
            publish_time=1_000_000_000,
            data=json.dumps({"trace_id": traceparent, "rskill_id": "smolvla-libero"}).encode(
                "utf-8"
            ),
        )
        w.finish()


def test_ral_replay_writes_timeline_to_out(tmp_path: Path) -> None:
    bag = tmp_path / "rollout.mcap"
    traceparent = "00-" + ("ab" * 16) + "-" + ("01" * 8) + "-01"
    _write_one_message_bag(bag, traceparent, "/openral/failure/rskill")

    out = tmp_path / "timeline.json"
    result = runner.invoke(
        app,
        ["replay", str(bag), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    decoded = json.loads(out.read_text(encoding="utf-8"))
    assert decoded["trace_id"] == "ab" * 16
    assert decoded["bag_trace_ids"][0]["bag_message_count"] == 1
    assert len(decoded["timeline"]) == 1
    assert decoded["timeline"][0]["kind"] == "bag"
    assert decoded["timeline"][0]["topic"] == "/openral/failure/rskill"


def test_ral_replay_prints_json_to_stdout_when_no_out(tmp_path: Path) -> None:
    bag = tmp_path / "rollout.mcap"
    traceparent = "00-" + ("cd" * 16) + "-" + ("02" * 8) + "-01"
    _write_one_message_bag(bag, traceparent, "/openral/safe_action")
    result = runner.invoke(app, ["replay", str(bag)])
    assert result.exit_code == 0, result.output
    # Find the JSON block — typer prints it after the cli.command span.
    json_start = result.output.index("{")
    decoded = json.loads(result.output[json_start:])
    assert decoded["trace_id"] == "cd" * 16


def test_ral_replay_frame_pivots_into_dataset_trace(tmp_path: Path) -> None:
    """ISSUE-109: ``--frame <repo>/<ep>/<frame>`` resolves the join trace from the dataset.

    A single recorder fans one traced tick out to BOTH a LeRobotDatasetSink
    and a Rosbag2Sink, so the on-disk frame and the bag tick share a
    trace_id. ``--frame`` reads the frame's trace_id and uses it as the
    join key — the emitted timeline must key on exactly that trace.
    """
    import numpy as np

    pytest.importorskip(
        "lerobot",
        reason="lerobot>=0.5.1 not installed; install via `uv pip install lerobot>=0.5.1`",
    )
    from openral_core import RobotDescription
    from openral_dataset import LeRobotDatasetSink, RolloutRecorder, Rosbag2Sink
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    repo_root = Path(__file__).resolve().parents[2]
    robot = RobotDescription.from_yaml(str(repo_root / "robots" / "so100_follower" / "robot.yaml"))

    ds_root = tmp_path / "ds"
    bag = tmp_path / "rollout.mcap"
    lerobot_sink = LeRobotDatasetSink(
        root=ds_root, robot=robot, fps=30.0, repo_id="openral/dataset-test"
    )
    bag_sink = Rosbag2Sink(bag_path=bag)
    rec = RolloutRecorder(robot=robot, task_string="t", fps=30.0, sinks=[lerobot_sink, bag_sink])

    state = np.zeros(robot.observation_spec.state_shape, dtype=np.float32)
    action = np.zeros(robot.action_spec.dim, dtype=np.float32)
    images = {
        "camera1": np.zeros((256, 256, 3), dtype=np.uint8),
        "camera2": np.zeros((256, 256, 3), dtype=np.uint8),
    }

    tracer = TracerProvider().get_tracer("test")
    rec.episode_start()
    with tracer.start_as_current_span("rskill.tick"):
        ctx = trace.get_current_span().get_span_context()
        exp_trace = f"{ctx.trace_id:032x}"
        rec.record_frame(observation_state=state, images=images, action=action)
    rec.episode_end(success=True)
    rec.finalize()

    out = tmp_path / "timeline.json"
    result = runner.invoke(
        app,
        [
            "replay",
            str(bag),
            "--frame",
            "openral/dataset-test/0/0",
            "--dataset-root",
            str(ds_root),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    decoded = json.loads(out.read_text(encoding="utf-8"))
    assert decoded["trace_id"] == exp_trace
    # The bag's /openral/tick carries the same trace_id → timeline has it.
    assert any(e["trace_id"] == exp_trace for e in decoded["timeline"])


def test_ral_replay_frame_requires_dataset_root(tmp_path: Path) -> None:
    """``--frame`` without ``--dataset-root`` is a clean usage error."""
    bag = tmp_path / "rollout.mcap"
    traceparent = "00-" + ("ab" * 16) + "-" + ("01" * 8) + "-01"
    _write_one_message_bag(bag, traceparent, "/openral/tick")
    result = runner.invoke(app, ["replay", str(bag), "--frame", "openral/dataset-test/0/0"])
    assert result.exit_code != 0
    assert "dataset-root" in result.output


def test_ral_replay_frame_and_trace_mutually_exclusive(tmp_path: Path) -> None:
    """Passing both ``--frame`` and ``--trace`` is rejected."""
    bag = tmp_path / "rollout.mcap"
    traceparent = "00-" + ("ab" * 16) + "-" + ("01" * 8) + "-01"
    _write_one_message_bag(bag, traceparent, "/openral/tick")
    result = runner.invoke(
        app,
        [
            "replay",
            str(bag),
            "--frame",
            "openral/dataset-test/0/0",
            "--dataset-root",
            str(tmp_path),
            "--trace",
            "ab" * 16,
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_ral_record_dry_run_prints_ros2_bag_record_argv(tmp_path: Path) -> None:
    out_dir = tmp_path / "bag_out"
    result = runner.invoke(
        app,
        ["record", "--out", str(out_dir), "--profile", "slim", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "ros2 bag record" in result.output
    assert "--regex" in result.output
    assert "/openral/safe_action" in result.output


def test_ral_record_rejects_unknown_profile(tmp_path: Path) -> None:
    out_dir = tmp_path / "bag_out"
    result = runner.invoke(
        app,
        ["record", "--out", str(out_dir), "--profile", "huge", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "unknown profile" in result.output


def test_ral_profile_session_unknown_action_exits_nonzero() -> None:
    result = runner.invoke(app, ["profile", "session", "ramble"])
    assert result.exit_code != 0
    assert "unknown action" in result.output


def test_ral_profile_session_start_errors_when_lttng_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """start_session surfaces the lttng-tools-missing message through the CLI."""
    import openral_observability.tracing_lttng as ttng

    def _no_lttng(name: str) -> str | None:
        if name == "lttng":
            return None
        return f"/usr/bin/{name}"

    monkeypatch.setattr(ttng.shutil, "which", _no_lttng)
    result = runner.invoke(
        app,
        ["profile", "session", "start", "--output", str(tmp_path), "--name", "ut"],
    )
    assert result.exit_code != 0
    assert "lttng-tools not found" in result.output
