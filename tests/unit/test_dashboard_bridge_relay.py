"""The ``/ws`` proxy's latest-wins relay (``openral_observability.dashboard.bridge_relay``).

Frames are real ``foxglove.websocket.v1`` frames, built byte for byte as
``foxglove_bridge`` sends them: JSON ``advertise``/``subscribe`` and binary
message data (opcode 0x01, u32 subscription id, u64 timestamp, payload).
"""

from __future__ import annotations

import json
import struct

from openral_observability.dashboard import TelemetryStore, create_app
from openral_observability.dashboard.bridge_relay import LATEST_WINS_SCHEMAS, BridgeRelay
from starlette.testclient import TestClient

from tests.unit.fakes.fake_openral_issuer import FakeBridge

_CHANNELS = [
    {"id": 1, "topic": "/openral/cameras/top/image", "schemaName": "sensor_msgs/msg/Image"},
    {"id": 2, "topic": "/tf", "schemaName": "tf2_msgs/msg/TFMessage"},
    {"id": 3, "topic": "/map", "schemaName": "nav_msgs/msg/OccupancyGrid"},
    {
        "id": 4,
        "topic": "/openral/viz/collisions",
        "schemaName": "visualization_msgs/msg/MarkerArray",
    },
]


def _advertise() -> str:
    channels = [{**c, "encoding": "cdr", "schema": ""} for c in _CHANNELS]
    return json.dumps({"op": "advertise", "channels": channels})


def _subscribe(*pairs: tuple[int, int]) -> str:
    subs = [{"id": sid, "channelId": cid} for sid, cid in pairs]
    return json.dumps({"op": "subscribe", "subscriptions": subs})


def _data(sid: int, stamp: int, payload: bytes = b"") -> bytes:
    return b"\x01" + struct.pack("<IQ", sid, stamp) + payload


def _stamp(frame: str | bytes) -> int:
    assert isinstance(frame, bytes)
    return int(struct.unpack_from("<Q", frame, 5)[0])


def _relay() -> BridgeRelay:
    relay = BridgeRelay()
    relay.from_bridge(_advertise())
    relay.from_browser(_subscribe((10, 1), (20, 2), (30, 3), (40, 4)))
    assert relay.drain() == [_advertise()]
    return relay


def test_snapshot_topics_keep_only_the_newest_unsent_message() -> None:
    relay = _relay()
    for stamp in range(5):
        relay.from_bridge(_data(10, stamp, b"img"))
        relay.from_bridge(_data(30, stamp, b"map"))
    out = relay.drain()
    assert [(f[1], _stamp(f)) for f in out if isinstance(f, bytes)] == [(10, 4), (30, 4)]
    assert relay.dropped == 8
    # Once sent, the next message queues again.
    relay.from_bridge(_data(10, 5))
    assert [_stamp(f) for f in relay.drain()] == [5]


def test_an_unsubscribe_drops_the_unsent_snapshot_of_that_subscription() -> None:
    relay = _relay()
    relay.from_bridge(_data(10, 1, b"img"))
    relay.from_bridge(_data(30, 1, b"map"))
    relay.from_browser(json.dumps({"op": "unsubscribe", "subscriptionIds": [10]}))
    # The browser reuses id 10 for another channel: the old image is never sent.
    relay.from_browser(_subscribe((10, 3)))
    out = [(f[1], _stamp(f)) for f in relay.drain() if isinstance(f, bytes)]
    assert out == [(30, 1)]
    relay.from_bridge(_data(10, 2, b"map-2"))
    assert [_stamp(f) for f in relay.drain()] == [2]


def test_incremental_topics_and_control_frames_are_relayed_in_order_unchanged() -> None:
    relay = _relay()
    sent: list[str | bytes] = [
        _data(20, 1, b"tf-a"),
        _data(40, 1, b"markers-a"),
        json.dumps({"op": "status", "level": 0, "message": "hi"}),
        _data(20, 2, b"tf-b"),
        b"\x02" + struct.pack("<Q", 99),  # Time frame
        _data(40, 2, b"markers-b"),
    ]
    for frame in sent:
        relay.from_bridge(frame)
    assert relay.drain() == sent
    assert relay.dropped == 0


def test_unsubscribing_and_resubscribing_to_another_channel_stops_coalescing() -> None:
    relay = _relay()
    relay.from_browser(json.dumps({"op": "unsubscribe", "subscriptionIds": [10]}))
    relay.from_browser(_subscribe((10, 2)))  # id reused for /tf
    frames = [_data(10, 1), _data(10, 2)]
    for frame in frames:
        relay.from_bridge(frame)
    assert relay.drain() == frames


def test_sampled_streams_are_never_coalesced() -> None:
    # A plot needs every joint-state sample, and two publishers may share the topic.
    relay = BridgeRelay()
    channel = {"id": 9, "topic": "/joint_states", "schemaName": "sensor_msgs/msg/JointState"}
    relay.from_bridge(json.dumps({"op": "advertise", "channels": [channel]}))
    relay.from_browser(_subscribe((90, 9)))
    relay.drain()
    frames = [_data(90, stamp, b"js") for stamp in range(4)]
    for frame in frames:
        relay.from_bridge(frame)
    assert relay.drain() == frames
    assert relay.dropped == 0


def test_ordered_frames_are_bounded_and_snapshots_do_not_count() -> None:
    relay = BridgeRelay(max_ordered_bytes=1000)
    relay.from_bridge(_advertise())
    relay.from_browser(_subscribe((10, 1), (20, 2)))
    relay.drain()
    for stamp in range(50):  # snapshot frames replace each other: never "full"
        relay.from_bridge(_data(10, stamp, b"x" * 500))
    assert not relay.full()
    while not relay.full():
        relay.from_bridge(_data(20, 0, b"t" * 100))
    assert len(relay.drain()) < 12  # ~1000 bytes of /tf plus the one image
    assert relay.full()  # a drained batch still counts while it is being sent
    relay.sent()
    assert not relay.full()  # sending releases the bound


def test_a_malformed_subscribe_is_ignored_not_fatal() -> None:
    relay = _relay()
    for bad in (
        '{"op":"subscribe","subscriptions":[{"id":1,"channelId":[]}]}',
        '{"op":"subscribe","subscriptions":[{"id":{},"channelId":1}, 7]}',
        '{"op":"unsubscribe","subscriptionIds":[[1]]}',
        "not json",
        "[1, 2]",
    ):
        relay.from_browser(bad)  # the frame still goes to the bridge, which rejects it
    relay.from_bridge(_data(1, 1))
    relay.from_bridge(_data(1, 2))
    assert len(relay.drain()) == 2  # sub 1 was never classed as a snapshot subscription


def test_every_ros2_snapshot_schema_is_in_ros1_spelling() -> None:
    # The lookup drops "/msg/", so the set must hold the short spelling only.
    assert all("/msg/" not in name for name in LATEST_WINS_SCHEMAS)


def test_ws_proxy_relays_through_the_latest_wins_relay() -> None:
    # The fake bridge echoes, so what the browser sends comes back as bridge output:
    # its advertise teaches the relay the channels, and the burst of map frames must
    # arrive in order and end with the newest, however many were coalesced.
    with FakeBridge().running() as bridge:
        app = create_app(TelemetryStore(), bridge_url=f"ws://127.0.0.1:{bridge.port}")
        here = {"host": "127.0.0.1:4318"}  # TestClient sends "testserver" on websockets
        with (
            TestClient(app, base_url="http://127.0.0.1:4318", client=("127.0.0.1", 5)) as local,
            local.websocket_connect(
                "/ws", headers=here, subprotocols=["foxglove.websocket.v1"]
            ) as ws,
        ):
            ws.send_text(_advertise())
            assert ws.receive_text() == _advertise()
            ws.send_text(_subscribe((30, 3)))
            assert ws.receive_text() == _subscribe((30, 3))
            for stamp in range(50):
                ws.send_bytes(_data(30, stamp, b"map"))
            ws.send_text('{"op": "status", "level": 0, "message": "end"}')
            stamps: list[int] = []
            while True:
                msg = ws.receive()
                if msg.get("text") is not None:
                    break
                stamps.append(_stamp(msg["bytes"]))
    assert stamps == sorted(stamps)
    assert stamps[-1] == 49
