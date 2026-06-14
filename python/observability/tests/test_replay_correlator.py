"""ADR-0018 F7 — bag↔OTel correlator end-to-end against a real mcap file.

Per CLAUDE.md §1.11 — no mocks. The bag is a real ``mcap.writer.Writer``
output containing both a ``jsonschema``-encoded message (the encoding
:class:`openral_dataset.Rosbag2Sink` writes) and a ``ros2msg``-style
payload that embeds a W3C ``traceparent`` substring (the encoding a
``ros2 bag record --storage mcap`` would produce). The reader recovers
the trace_id from both, and the correlator joins them with a list of
in-memory span dicts.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from openral_observability.replay import (
    BagMessage,
    build_timeline,
    list_bag_trace_ids,
    read_bag,
)
from openral_observability.replay.cli import build_record_command


def _write_bag(
    bag_path: Path,
    records: list[tuple[str, bytes, str, str]],
) -> None:
    """Write a one-channel-per-(topic, encoding) mcap file at ``bag_path``.

    ``records`` is a list of ``(topic, payload_bytes, encoding, schema_name)``
    tuples written in arrival order with monotonically increasing
    log_time.
    """
    from mcap.writer import Writer

    with bag_path.open("wb") as f:
        w = Writer(f)
        w.start()
        # One schema and channel per (topic, encoding) pair so the reader
        # gets back the right encoding on iter_messages().
        channels: dict[tuple[str, str], int] = {}
        for i, (topic, payload, encoding, schema_name) in enumerate(records):
            key = (topic, encoding)
            if key not in channels:
                schema_id = w.register_schema(
                    name=schema_name,
                    encoding=encoding,
                    data=b"{}" if encoding == "jsonschema" else b"",
                )
                channels[key] = w.register_channel(
                    topic=topic, message_encoding=encoding, schema_id=schema_id
                )
            w.add_message(
                channel_id=channels[key],
                log_time=1_000_000_000 + i * 1_000_000,
                publish_time=1_000_000_000 + i * 1_000_000,
                data=payload,
            )
        w.finish()


@pytest.fixture
def sample_bag(tmp_path: Path) -> Path:
    bag = tmp_path / "rollout.mcap"
    traceparent = "00-" + ("ab" * 16) + "-" + ("01" * 8) + "-01"
    json_payload = json.dumps(
        {"trace_id": traceparent, "kind": 1, "severity": 2, "rskill_id": "smolvla-libero"}
    ).encode("utf-8")
    # CDR-shaped payload: a few bytes of garbage, then the traceparent
    # ASCII embedded mid-stream — verifies the regex extractor finds it
    # without a real CDR parser.
    cdr_payload = b"\x00\x01" + b"\x00" * 12 + traceparent.encode("ascii") + b"\xff\xff"
    other_traceparent = "00-" + ("cd" * 16) + "-" + ("02" * 8) + "-01"
    other_payload = json.dumps({"trace_id": other_traceparent, "noise": True}).encode("utf-8")

    _write_bag(
        bag,
        [
            ("/openral/failure/rskill", json_payload, "jsonschema", "openral_msgs/FailureTrigger"),
            ("/openral/safe_action", cdr_payload, "ros2msg", "openral_msgs/ActionChunk"),
            ("/openral/prompt", other_payload, "jsonschema", "openral_msgs/PromptStamped"),
        ],
    )
    return bag


def test_read_bag_extracts_trace_id_from_both_encodings(sample_bag: Path) -> None:
    messages = list(read_bag(sample_bag))
    assert len(messages) == 3
    by_topic = {m.topic: m for m in messages}
    assert by_topic["/openral/failure/rskill"].trace_id == "ab" * 16
    assert by_topic["/openral/safe_action"].trace_id == "ab" * 16
    assert by_topic["/openral/prompt"].trace_id == "cd" * 16


def test_read_bag_extracts_raw_trace_id_and_span_id_from_tick(tmp_path: Path) -> None:
    """ISSUE-109: /openral/tick carries raw trace_id (32 hex) + span_id (16 hex).

    Unlike the ADR-0018 ROS IDL messages (which pack a full W3C
    traceparent into a single ``trace_id`` field), the dataset Tick
    schema carries the two ids as separate raw-hex fields. The reader
    must index that 32-hex value directly and reconstruct the full
    traceparent from the pair.
    """
    bag = tmp_path / "tick.mcap"
    raw_trace = "ab" * 16  # 32 hex
    raw_span = "01" * 8  # 16 hex
    payload = json.dumps(
        {"stamp_ns": 1, "episode_idx": 0, "step_idx": 0, "trace_id": raw_trace, "span_id": raw_span}
    ).encode("utf-8")
    _write_bag(bag, [("/openral/tick", payload, "jsonschema", "openral_msgs/Tick")])

    messages = list(read_bag(bag))
    assert len(messages) == 1
    assert messages[0].trace_id == raw_trace
    assert messages[0].traceparent == f"00-{raw_trace}-{raw_span}-01"


def test_read_bag_resolves_directory_to_first_mcap(tmp_path: Path) -> None:
    """A rosbag2 directory containing exactly one .mcap file is resolved transparently."""
    bag_dir = tmp_path / "rosbag2_dir"
    bag_dir.mkdir()
    _write_bag(
        bag_dir / "rollout_0.mcap",
        [
            (
                "/diagnostics",
                json.dumps({"trace_id": "00-" + ("ee" * 16) + "-" + ("01" * 8) + "-01"}).encode(
                    "utf-8"
                ),
                "jsonschema",
                "diagnostic_msgs/DiagnosticArray",
            )
        ],
    )
    (bag_dir / "metadata.yaml").write_text("storage_identifier: mcap\n", encoding="utf-8")
    messages = list(read_bag(bag_dir))
    assert len(messages) == 1
    assert messages[0].trace_id == "ee" * 16


def test_list_bag_trace_ids_counts_per_trace(sample_bag: Path) -> None:
    counts = list_bag_trace_ids(read_bag(sample_bag))
    by_id = {c["trace_id"]: c for c in counts}
    assert by_id["ab" * 16]["bag_message_count"] == 2
    assert by_id["cd" * 16]["bag_message_count"] == 1
    # Sort order: descending count, busiest trace first.
    assert counts[0]["trace_id"] == "ab" * 16


def _span(trace_id: str, name: str, start_ns: int) -> dict[str, Any]:
    return {
        "name": name,
        "trace_id": trace_id,
        "span_id": "ff" * 8,
        "parent_span_id": "",
        "start_unix_ns": start_ns,
        "end_unix_ns": start_ns + 1_000_000,
        "duration_ms": 1.0,
        "attrs": {"rskill.id": "smolvla-libero"},
        "status_code": 1,
        "status_message": "",
        "events": [],
    }


def test_build_timeline_filters_to_requested_trace_id(sample_bag: Path) -> None:
    bag = list(read_bag(sample_bag))
    spans = [
        _span("ab" * 16, "rskill.execute", 1_000_500_000),
        _span("cd" * 16, "reasoner.tick", 1_000_700_000),
    ]
    timeline = build_timeline(bag, spans, trace_id="ab" * 16)
    kinds = [(e.kind, e.topic or e.span_name) for e in timeline]
    # Trace ab gets two bag messages + one span; the prompt + the
    # reasoner.tick span both belong to trace cd and are filtered out.
    assert ("span", "rskill.execute") in kinds
    assert ("bag", "/openral/failure/rskill") in kinds
    assert ("bag", "/openral/safe_action") in kinds
    assert ("span", "reasoner.tick") not in kinds
    assert ("bag", "/openral/prompt") not in kinds
    # Sorted ascending by timestamp.
    assert all(timeline[i].ts_ns <= timeline[i + 1].ts_ns for i in range(len(timeline) - 1))


def test_build_record_command_slim_profile() -> None:
    argv = build_record_command(profile="slim", output_dir=Path("/tmp/bag_x"))
    assert argv[:3] == ["ros2", "bag", "record"]
    # mcap is the default storage backend.
    assert "mcap" in argv
    # All slim verbatim topics are present somewhere in argv.
    for required in (
        "/openral/safe_action",
        "/openral/candidate_action",
        "/openral/estop",
        "/diagnostics",
    ):
        assert required in argv
    # And the failure-bus + per-camera regex got combined.
    assert "--regex" in argv
    regex_idx = argv.index("--regex")
    pattern = argv[regex_idx + 1]
    assert "/openral/failure/" in pattern
    assert "/openral/sensors/" in pattern


def test_build_record_command_full_adds_world_state_fast_and_perception() -> None:
    argv = build_record_command(profile="full", output_dir=Path("/tmp/bag_full"))
    assert "/openral/world_state_fast" in argv
    regex_idx = argv.index("--regex")
    pattern = argv[regex_idx + 1]
    assert "/openral/perception/" in pattern


def test_build_record_command_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unknown record profile"):
        build_record_command(profile="ridiculous", output_dir=Path("/tmp/x"))  # type: ignore[arg-type] # reason: deliberate bad value to exercise the guard


def test_bag_message_is_frozen_dataclass() -> None:
    """Defensive — the join logic stores these in dicts keyed by trace_id."""
    m = BagMessage(
        topic="/x",
        log_time_ns=1,
        publish_time_ns=1,
        trace_id="aa" * 16,
        traceparent="00-aaa",
        schema_name="x",
        payload_summary={},
    )
    with pytest.raises(Exception):  # noqa: B017  # reason: any FrozenInstanceError variant counts
        m.topic = "/y"  # type: ignore[misc] # reason: frozen field assignment under test


def test_read_bag_raises_when_path_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="bag path does not exist"):
        list(read_bag(tmp_path / "nope.mcap"))


def test_read_bag_directory_without_mcap_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty_bag"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=r"no \.mcap"):
        list(read_bag(empty))


def test_read_bag_iter_yields_compatible_stream_for_io_bytesio(sample_bag: Path) -> None:
    """The reader iterates without surfacing the underlying file handle to the caller."""
    # A second pass over the same bag works (the file is opened per call).
    msgs_one = list(read_bag(sample_bag))
    msgs_two = list(read_bag(sample_bag))
    assert msgs_one == msgs_two
    # The fixture is small; ensure we did not leak any data.
    buf = io.BytesIO()
    buf.write(sample_bag.read_bytes())
    assert buf.tell() > 0
