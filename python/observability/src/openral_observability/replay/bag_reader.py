"""Iterate rosbag2 / mcap files and surface the ADR-0018 ``trace_id`` field.

The reader uses the bare ``mcap`` PyPI library (already a dependency of
``openral_dataset`` per ADR-0019). It does **not** import ``rosbag2_py``
so it works on developer laptops and CI runners that have not sourced
ROS 2 — `ros2 bag record --storage mcap` writes the same on-disk format
this reader consumes.

Two payload encodings are accepted:

* ``jsonschema`` (the encoding used by :class:`openral_dataset.Rosbag2Sink`)
  — the payload is UTF-8 JSON and we read ``trace_id`` directly.
* ``ros2msg`` (the encoding used by a native ``ros2 bag record``) — the
  CDR-serialised payload is not decoded here; we scan the bytes for a
  W3C ``traceparent`` substring (``00-<32 hex>-<16 hex>-<2 hex>``) and
  surface that as the trace_id. ROS message strings are length-prefixed
  ASCII inside the CDR envelope so the regex match is robust against
  field reordering and message-type evolution.

Per CLAUDE.md §1.11 — no mocks. Tests round-trip a real mcap file with
both encodings.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = ["BagMessage", "read_bag"]

# W3C traceparent v0: ``00-<32 hex trace>-<16 hex span>-<2 hex flags>``.
# Anchored to the version byte so partial hits in other fields never
# match. Two compiled patterns — one for the bytes payload of a CDR
# (``ros2msg`` encoding) blob, one for the JSON string payload of a
# ``jsonschema``-encoded message.
_TRACEPARENT_RE_BYTES: Final[re.Pattern[bytes]] = re.compile(
    rb"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})"
)
_TRACEPARENT_RE_STR: Final[re.Pattern[str]] = re.compile(
    r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})"
)
# Raw 32-hex trace id (the `openral_msgs/Tick` schema carries trace_id +
# span_id as separate fields, not a packed traceparent — ISSUE-109).
_RAW_TRACE_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{32}$")
_RAW_SPAN_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{16}$")


def _trace_id_from_json_payload(payload: dict[str, Any]) -> tuple[str, str]:
    """Return ``(trace_id_hex, traceparent)`` from a ``jsonschema`` payload.

    Two conventions coexist on the bus:

    * ADR-0018 ROS IDL messages (ActionChunk / FailureTrigger / …) pack a
      full W3C ``traceparent`` into a single ``trace_id`` field.
    * The ADR-0019 ``openral_msgs/Tick`` schema carries ``trace_id``
      (32 hex) and ``span_id`` (16 hex) as separate raw fields (ISSUE-109).

    Both resolve to the 32-hex trace component used for the join; the
    full traceparent is reconstructed from the pair when only raw fields
    are present.
    """
    raw_tid = payload.get("trace_id")
    if not isinstance(raw_tid, str) or not raw_tid:
        return "", ""
    # Packed traceparent (legacy ROS IDL convention).
    m_str = _TRACEPARENT_RE_STR.match(raw_tid)
    if m_str:
        return m_str.group(1), raw_tid
    # Raw 32-hex trace id (Tick convention) — reconstruct the traceparent
    # from the sibling span_id when it is a valid 16-hex value.
    if _RAW_TRACE_ID_RE.match(raw_tid):
        raw_span = payload.get("span_id")
        if isinstance(raw_span, str) and _RAW_SPAN_ID_RE.match(raw_span):
            return raw_tid, f"00-{raw_tid}-{raw_span}-01"
        return raw_tid, ""
    return "", ""


@dataclass(frozen=True)
class BagMessage:
    """One rosbag2 record surfaced to the F7 correlator.

    Attributes:
        topic: ROS topic name (e.g. ``/openral/safe_action``).
        log_time_ns: mcap log_time — the recorder's wall clock on write.
        publish_time_ns: publisher-side stamp; ``log_time_ns`` if the
            recorder did not preserve it.
        trace_id: 32-hex-char trace_id extracted from the message body,
            or empty if absent. ROS messages defined under ADR-0018
            carry it as a top-level ``string trace_id`` field; older
            messages may not.
        traceparent: Full W3C ``traceparent`` when extracted from a
            ``ros2msg`` payload (carries the span_id too); empty
            otherwise. The correlator joins on ``trace_id`` but exposes
            ``traceparent`` so callers that want full
            :func:`openral_observability.propagation.extract_traceparent`
            semantics have it.
        schema_name: mcap schema name, useful for filtering downstream.
        payload_summary: Decoded JSON (when encoding is ``jsonschema``)
            or ``{"_encoding": "ros2msg", "byte_len": int}`` otherwise.
            Kept small; the full bytes stay in the bag.
    """

    topic: str
    log_time_ns: int
    publish_time_ns: int
    trace_id: str
    traceparent: str
    schema_name: str
    payload_summary: dict[str, Any]


def _resolve_mcap_path(path: Path) -> Path:
    """Find the actual ``.mcap`` file inside a rosbag2 directory.

    `ros2 bag record -o foo` produces ``foo/foo_0.mcap`` plus
    ``metadata.yaml``. A bare ``.mcap`` path is also accepted.
    """
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.mcap"))
        if not candidates:
            msg = f"no .mcap files in rosbag2 directory: {path}"
            raise FileNotFoundError(msg)
        return candidates[0]
    msg = f"bag path does not exist: {path}"
    raise FileNotFoundError(msg)


def _extract_traceparent_from_ros2msg(payload: bytes) -> tuple[str, str]:
    """Return ``(trace_id_hex, traceparent)`` from a CDR payload, or empties."""
    m = _TRACEPARENT_RE_BYTES.search(payload)
    if m is None:
        return "", ""
    trace_id_hex = m.group(1).decode("ascii")
    return trace_id_hex, m.group(0).decode("ascii")


def read_bag(bag_path: str | Path) -> Iterator[BagMessage]:
    """Yield every message in ``bag_path`` as a :class:`BagMessage`.

    Args:
        bag_path: Path to an mcap file or to a rosbag2 directory
            containing one. The reader picks the first ``*.mcap`` in
            sorted order when given a directory.

    Yields:
        One :class:`BagMessage` per record, in mcap order (write
        order). The correlator re-sorts by ``log_time_ns`` so callers
        may stream this lazily.

    Raises:
        FileNotFoundError: If ``bag_path`` does not exist or contains
            no ``.mcap`` file.
    """
    from mcap.reader import make_reader

    resolved = _resolve_mcap_path(Path(bag_path))
    with resolved.open("rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            encoding = schema.encoding if schema is not None else ""
            schema_name = schema.name if schema is not None else ""
            trace_id = ""
            traceparent = ""
            payload_summary: dict[str, Any]
            if encoding == "jsonschema":
                try:
                    payload = json.loads(message.data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload = {}
                if isinstance(payload, dict):
                    trace_id, traceparent = _trace_id_from_json_payload(payload)
                payload_summary = payload if isinstance(payload, dict) else {}
            else:
                trace_id, traceparent = _extract_traceparent_from_ros2msg(message.data)
                payload_summary = {
                    "_encoding": encoding or "unknown",
                    "byte_len": len(message.data),
                }
            yield BagMessage(
                topic=channel.topic,
                log_time_ns=int(message.log_time),
                publish_time_ns=int(message.publish_time),
                trace_id=trace_id,
                traceparent=traceparent,
                schema_name=schema_name,
                payload_summary=payload_summary,
            )
