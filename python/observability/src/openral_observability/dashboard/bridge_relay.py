r"""Latest-wins relay for the ``/ws`` proxy to ``foxglove_bridge``.

Through a robot's Cloudflare Tunnel the link to the browser is often slower than
the bridge. Relaying every frame in order then builds a queue, and the 3D view
falls seconds behind. This relay reads the Foxglove WebSocket protocol
(``foxglove.websocket.v1``) it already carries and, **for snapshot-type topics
only** (images, point clouds, occupancy grids, laser scans), keeps just the
newest message per subscription while the browser side is busy.

Everything else is relayed unchanged and in order: every text (JSON) frame,
every binary frame that is not message data, and message data for incremental
topics (``/tf``, marker arrays, the URDF string), where dropping a message would
lose state, and for sampled streams (joint states, IMU, odometry), which a
plot needs whole. The relay never edits a frame.

Ordered frames are bounded: once ``max_ordered_bytes`` of them wait unsent,
``full()`` turns true and the caller stops reading the bridge until they are
sent, so the bridge's own ``send_buffer_limit`` decides what to shed, as it did
before this relay existed.

Example:
    >>> import struct
    >>> relay = BridgeRelay()
    >>> relay.from_bridge(
    ...     '{"op":"advertise","channels":[{"id":7,"topic":"/map",'
    ...     '"encoding":"cdr","schemaName":"nav_msgs/msg/OccupancyGrid"}]}'
    ... )
    >>> relay.from_browser('{"op":"subscribe","subscriptions":[{"id":1,"channelId":7}]}')
    >>> old = b"\x01" + struct.pack("<IQ", 1, 1) + b"old"
    >>> new = b"\x01" + struct.pack("<IQ", 1, 2) + b"new"
    >>> relay.from_bridge(old)
    >>> relay.from_bridge(new)
    >>> sent = relay.drain()  # the browser was busy: the advertise, then only the newest map
    >>> [f[:14] if isinstance(f, str) else f[-3:] for f in sent]
    ['{"op":"adverti', b'new']
"""

from __future__ import annotations

import json
import struct
from collections import deque

__all__ = ["LATEST_WINS_SCHEMAS", "MAX_ORDERED_BYTES", "BridgeRelay", "Frame"]

Frame = str | bytes

#: Message types where only the newest message matters: each one is a full,
#: heavy snapshot of its stream. Sampled streams (``JointState``, ``Imu``,
#: ``Odometry``) are deliberately absent: a plot needs every sample, and two
#: publishers may share one topic with partial messages. ``/msg/`` is dropped
#: before the lookup, so both the ROS 2 (``sensor_msgs/msg/Image``) and ROS 1
#: (``sensor_msgs/Image``) spellings match.
LATEST_WINS_SCHEMAS: frozenset[str] = frozenset(
    {
        "sensor_msgs/Image",
        "sensor_msgs/CompressedImage",
        "sensor_msgs/CameraInfo",
        "sensor_msgs/PointCloud2",
        "sensor_msgs/LaserScan",
        "nav_msgs/OccupancyGrid",
    }
)

#: Unsent ordered frames the relay holds before ``full()`` asks the caller to
#: stop reading the bridge (foxglove_bridge's ``send_buffer_limit`` is 10 MB).
MAX_ORDERED_BYTES = 8 * 1024 * 1024

_MESSAGE_DATA = 0x01
_HEADER = struct.Struct("<BI")  # opcode, subscription id (then u64 timestamp, payload)


def _schema_key(name: object) -> str:
    return str(name).replace("/msg/", "/")


class BridgeRelay:
    """Ordered relay of bridge→browser frames with per-subscription coalescing.

    Feed every bridge→browser frame to ``from_bridge`` and every browser→bridge
    frame to ``from_browser`` (it is only read, to learn the subscriptions), then
    send what ``drain`` returns. Single-threaded: call it from one event loop.
    """

    def __init__(self, max_ordered_bytes: int = MAX_ORDERED_BYTES) -> None:
        """Start with no channels or subscriptions known."""
        self._latest_channels: set[int] = set()
        self._latest_subs: set[int] = set()
        # A frame, or a subscription id standing for "the newest of that subscription".
        self._queue: deque[Frame | int] = deque()
        self._newest: dict[int, bytes] = {}
        self._ordered_bytes = 0
        self._max_ordered_bytes = max_ordered_bytes
        self.dropped = 0
        """Snapshot messages replaced by a newer one before they were sent."""

    def from_browser(self, frame: Frame) -> None:
        """Learn subscriptions from a browser→bridge frame (the frame is not changed)."""
        msg = _json(frame)
        if msg is None:
            return
        if msg.get("op") == "subscribe":
            for sub in _dicts(msg.get("subscriptions")):
                sid, cid = sub.get("id"), sub.get("channelId")
                if not isinstance(sid, int):
                    continue
                if isinstance(cid, int) and cid in self._latest_channels:
                    self._latest_subs.add(sid)
                else:
                    self._latest_subs.discard(sid)
        elif msg.get("op") == "unsubscribe":
            for sid in _list(msg.get("subscriptionIds")):
                if isinstance(sid, int):
                    self._latest_subs.discard(sid)

    def from_bridge(self, frame: Frame) -> None:
        """Queue a bridge→browser frame, replacing an older unsent snapshot of its subscription."""
        if isinstance(frame, str):
            self._learn_channels(frame)
            self._queue.append(frame)
            self._ordered_bytes += len(frame)
            return
        if len(frame) >= _HEADER.size and frame[0] == _MESSAGE_DATA:
            sid = _HEADER.unpack_from(frame)[1]
            if sid in self._latest_subs:
                if sid in self._newest:
                    self.dropped += 1
                else:
                    self._queue.append(sid)
                self._newest[sid] = frame
                return
        self._queue.append(frame)
        self._ordered_bytes += len(frame)

    def full(self) -> bool:
        """Whether the unsent ordered frames reached the bound: stop reading the bridge."""
        return self._ordered_bytes >= self._max_ordered_bytes

    def pending(self) -> bool:
        """Whether ``drain`` has anything to return."""
        return bool(self._queue)

    def drain(self) -> list[Frame]:
        """Every frame to send now, in order."""
        out: list[Frame] = []
        while self._queue:
            item = self._queue.popleft()
            if isinstance(item, int):
                frame = self._newest.pop(item, None)
                if frame is not None:
                    out.append(frame)
            else:
                out.append(item)
        self._ordered_bytes = 0
        return out

    def _learn_channels(self, text: str) -> None:
        msg = _json(text)
        if msg is None:
            return
        if msg.get("op") == "advertise":
            for ch in _dicts(msg.get("channels")):
                cid = ch.get("id")
                if (
                    isinstance(cid, int)
                    and _schema_key(ch.get("schemaName")) in LATEST_WINS_SCHEMAS
                ):
                    self._latest_channels.add(cid)
                elif isinstance(cid, int):
                    self._latest_channels.discard(cid)
        elif msg.get("op") == "unadvertise":
            for cid in _list(msg.get("channelIds")):
                if isinstance(cid, int):
                    self._latest_channels.discard(cid)


def _json(frame: Frame) -> dict[str, object] | None:
    if not isinstance(frame, str):
        return None
    try:
        msg = json.loads(frame)
    except ValueError:
        return None
    return msg if isinstance(msg, dict) else None


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _dicts(value: object) -> list[dict[str, object]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []
