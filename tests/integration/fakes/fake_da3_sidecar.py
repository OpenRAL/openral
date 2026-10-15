"""A DA3 depth sidecar stand-in at its network boundary (ZMQ REP + msgpack).

Speaks ``tools/_da3_depth_server.py``'s wire protocol and answers every ``depth``
request with what DA3METRIC would predict for a fronto-parallel plane at
``plane_z_m``: canonical depth ``z * 300 / focal`` at its ``(h, w)`` raster, ``focal``
being the camera focal the test declares scaled to that raster. ``model`` is reported
like the real server, so the provider's metric-model check can be exercised. Under
``tests/integration/fakes/`` per CLAUDE.md §1.11; production code never imports it.
"""

from __future__ import annotations

import threading
from typing import Any

import msgpack
import numpy as np
import zmq

__all__ = ["FakeDa3Sidecar"]


class FakeDa3Sidecar:
    """Serve plane depth on ``tcp://127.0.0.1:<port>`` from a daemon thread."""

    def __init__(
        self, *, model: str, plane_z_m: float, raster_hw: tuple[int, int], focal_px: float
    ) -> None:
        self.model = model
        self.requests = 0
        h, w = raster_hw
        self._canonical = np.full((h, w), plane_z_m * 300.0 / focal_px, dtype=np.float32)
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.setsockopt(zmq.LINGER, 0)
        self.port = self._sock.bind_to_random_port("tcp://127.0.0.1")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        h, w = self._canonical.shape
        while not self._stop.is_set():
            try:
                req: dict[str, Any] = msgpack.unpackb(self._sock.recv(), raw=False)
            except zmq.Again:
                continue
            self.requests += 1
            if req.get("op") != "depth":
                self._sock.send(msgpack.packb({"ok": True, "model": self.model}))
                continue
            self._sock.send(
                msgpack.packb(
                    {
                        "ok": True,
                        "depth": self._canonical.tobytes(),
                        "h": h,
                        "w": w,
                        # DA3's own (deliberately wrong) K: the metric path must ignore it.
                        "fx": 1.0,
                        "fy": 1.0,
                        "cx": 0.0,
                        "cy": 0.0,
                        "latency_ms": 1.0,
                        "model": self.model,
                    },
                    use_bin_type=True,
                )
            )

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sock.close(linger=0)
        self._ctx.term()
