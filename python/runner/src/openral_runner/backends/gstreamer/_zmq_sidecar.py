"""Shared ZMQ REQ/REP transport for gstreamer sidecar-client backends.

``LocateAnythingDetector`` and ``QwenSceneVlm`` are both ZMQ REQ/REP clients
for an auto-spawned sidecar process; their ``_connect``, ``_rpc`` and
``_try_ping`` methods are byte-identical. This mixin holds exactly those
three — nothing else. ``_spawn_and_wait``, ``_ensure_ready`` and ``close``
genuinely differ per backend (different CLI args, error messages, ports) and
stay on each class; forcing them together would just parameterize the
difference back in.
"""

from __future__ import annotations

from typing import Any


class ZmqSidecarMixin:
    """REQ/REP transport for a ZMQ sidecar: connect, RPC, ping.

    A host class must set ``_host``, ``_port``, ``_request_timeout_ms``,
    ``_zmq`` and ``_ctx`` before calling ``_connect``, and provide ``_sock``
    (starting at ``None``).
    """

    _host: str
    _port: int
    _request_timeout_ms: int
    _zmq: Any
    _ctx: Any
    _sock: Any

    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        sock = self._ctx.socket(self._zmq.REQ)
        sock.setsockopt(self._zmq.LINGER, 0)
        sock.setsockopt(self._zmq.RCVTIMEO, self._request_timeout_ms)
        sock.setsockopt(self._zmq.SNDTIMEO, 5000)
        sock.connect(f"tcp://{self._host}:{self._port}")
        self._sock = sock

    def _rpc(self, req: dict[str, object], *, recv_timeout_ms: int | None = None) -> dict[str, Any]:
        """Send one request and return the decoded reply.

        Recreates the (strict REQ/REP) socket on timeout so a missed reply
        can't wedge it.
        """
        import msgpack  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: PLC0415 — lazy: only needed when the sidecar is used

        assert self._sock is not None
        if recv_timeout_ms is not None:
            self._sock.setsockopt(self._zmq.RCVTIMEO, recv_timeout_ms)
        try:
            self._sock.send(msgpack.packb(req, use_bin_type=True))
            reply: dict[str, Any] = msgpack.unpackb(self._sock.recv(), raw=False)
        except self._zmq.error.Again:
            self._connect()  # REQ can't recover from a missed reply; reset it
            raise
        finally:
            if recv_timeout_ms is not None:
                self._sock.setsockopt(self._zmq.RCVTIMEO, self._request_timeout_ms)
        return reply

    def _try_ping(self, *, recv_timeout_ms: int = 1000) -> bool:
        try:
            reply = self._rpc({"op": "ping"}, recv_timeout_ms=recv_timeout_ms)
        except self._zmq.error.Again:
            return False
        return bool(reply.get("ok"))
