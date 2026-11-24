"""A depth_estimator sidecar stand-in at its network boundary (ZMQ REP + msgpack).

Speaks the wire protocol of ``tools/_da3_depth_server.py`` / ``tools/_depth_pro2_server.py``
and answers every ``depth`` request with a fronto-parallel plane at ``plane_z_m``: as
DA3METRIC's canonical depth ``z * 300 / focal`` at its ``(h, w)`` raster (``focal`` being
the camera focal the test declares scaled to that raster), or, ``metric``, as metres like
Depth Pro 2. ``model`` is reported like the real servers, so the provider's model check
can be exercised, and every ``f_px`` it is sent is recorded. Under
``tests/integration/fakes/`` per CLAUDE.md §1.11; production code never imports it.
"""

from __future__ import annotations

import threading
from typing import Any

import msgpack
import numpy as np
import zmq

__all__ = ["FakeDepthSidecar"]


class FakeDepthSidecar:
    """Serve plane depth on ``tcp://127.0.0.1:<port>`` from a daemon thread."""

    def __init__(
        self,
        *,
        model: str,
        plane_z_m: float,
        raster_hw: tuple[int, int],
        focal_px: float,
        metric: bool = False,
    ) -> None:
        """``metric``: answer metres (Depth Pro 2), else DA3METRIC's canonical focal-300."""
        self.model = model
        self.requests = 0
        self.f_px: list[float] = []  # every ``f_px`` a depth request carried
        h, w = raster_hw
        raw = plane_z_m if metric else plane_z_m * 300.0 / focal_px
        self._canonical = np.full((h, w), raw, dtype=np.float32)
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
            if "f_px" in req:
                self.f_px.append(float(req["f_px"]))
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
