"""Thin client for the DA3 monocular metric-depth sidecar.

Reuses the SAME `depth-anything/DA3-SMALL` sidecar the SLAM / nvblox depth
provider uses (`tools/_da3_depth_server.py`, booted by
`tools/da3_depth_sidecar.py`): an RGB frame in, a metric-depth map (float32
metres) + estimated pinhole intrinsics out, over ZMQ REQ/REP + msgpack. DA3 is
**monocular**, so this works identically on a sim-rendered RGB frame and a real
camera — the whole point of leveraging it for a policy that wants depth.

Unlike `openral_sim.sidecar.SidecarClient` (whose wire framing is
`{"endpoint","data"}`), the DA3 server speaks the perception bus's
`{"op": ...}` protocol, so this client talks it directly (mirroring
`openral_perception_ros.depth_provider_node`). It pings the default port first
and only auto-spawns a sidecar when none answers — so when SLAM already runs a
DA3 sidecar, the policy shares it instead of loading a second copy.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

_log = structlog.get_logger(__name__)

# The DA3 sidecar's default bind port (tools/_da3_depth_server.py:--port and the
# depth_provider_node's `sidecar_port` default) — share SLAM's sidecar on it.
DEFAULT_DA3_PORT = 5771
_PING_TIMEOUT_MS = 2_000
_INFER_TIMEOUT_MS = 30_000
_BOOT_TIMEOUT_S = 900.0  # first boot provisions the DA3 venv + downloads weights
_RGB_CHANNELS = 3


def _locate_da3_boot_script() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "da3_depth_sidecar.py"
        if candidate.is_file():
            return candidate
    raise ROSConfigError(f"Could not locate tools/da3_depth_sidecar.py upwards from {here}.")


@dataclass
class Da3DepthClient:
    """ZMQ client for the DA3 metric-depth sidecar (ping-reuse, else auto-spawn)."""

    host: str = "127.0.0.1"
    port: int = DEFAULT_DA3_PORT
    process_res: int = 504
    auto_spawn: bool = True
    _sock: Any = None
    _ctx: Any = None
    _child: subprocess.Popen[bytes] | None = field(default=None)

    def connect(self) -> None:
        """Bind to an existing DA3 sidecar (ping-reuse), else auto-spawn one."""
        import zmq

        self._ctx = zmq.Context.instance()
        if self._try_ping():
            _log.info("da3.reuse_existing", port=self.port)
            return
        if not self.auto_spawn:
            raise ROSConfigError(
                f"No DA3 depth sidecar on {self.host}:{self.port} and auto-spawn is off. "
                "Start it: `python tools/da3_depth_sidecar.py --port "
                f"{self.port}` (or set OPENRAL_INTERNVLA_N1_AUTO_SPAWN=1)."
            )
        self._spawn()
        deadline = _BOOT_TIMEOUT_S
        waited = 0.0
        while waited < deadline:
            if self._try_ping():
                _log.info("da3.spawned", port=self.port)
                return
            time.sleep(2.0)
            waited += 2.0
            if self._child is not None and self._child.poll() is not None:
                raise ROSConfigError(
                    f"DA3 sidecar exited during boot (code {self._child.returncode}); "
                    "inspect its stdout above."
                )
        raise ROSConfigError(
            f"DA3 sidecar did not answer ping within {deadline:.0f}s on {self.host}:{self.port}."
        )

    def infer(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Return a metric-depth map (float32 metres) resized to ``rgb``'s HxW.

        ``rgb`` is HxWx3 uint8. The DA3 model estimates depth at its own
        ``process_res``; the returned map is resized back to the input frame so
        it is pixel-aligned with the RGB the policy consumes.
        """
        import msgpack
        import zmq
        from PIL import Image

        if rgb.ndim != _RGB_CHANNELS or rgb.shape[2] != _RGB_CHANNELS:
            raise ValueError(f"Da3DepthClient.infer expects HxWx3 uint8, got {rgb.shape}")
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb), "RGB").save(buf, format="PNG")
        try:
            self._sock.send(
                msgpack.packb(
                    {"op": "depth", "image": buf.getvalue(), "process_res": self.process_res},
                    use_bin_type=True,
                )
            )
            rep = msgpack.unpackb(self._sock.recv(), raw=False)
        except zmq.ZMQError as exc:
            raise ROSRuntimeError(f"DA3 depth request failed: {exc}") from exc
        if not rep.get("ok"):
            raise ROSRuntimeError(f"DA3 sidecar error: {rep.get('error')!r}")
        dh, dw = int(rep["h"]), int(rep["w"])
        depth: NDArray[np.float32] = np.frombuffer(rep["depth"], dtype=np.float32).reshape(dh, dw)
        if (dh, dw) != (h, w):
            depth = np.asarray(Image.fromarray(depth, mode="F").resize((w, h)), dtype=np.float32)
        return np.ascontiguousarray(depth, dtype=np.float32)

    def close(self) -> None:
        """Close the socket and reap an auto-spawned sidecar (idempotent)."""
        if self._sock is not None:
            with contextlib.suppress(Exception):  # teardown must not raise
                self._sock.close(linger=0)
            self._sock = None
        if self._child is not None and self._child.poll() is None:
            self._child.terminate()
            try:
                self._child.wait(timeout=5.0)
            except Exception:  # reason: best-effort reap
                self._child.kill()

    # ── internals ────────────────────────────────────────────────────────────

    def _try_ping(self) -> bool:
        import msgpack
        import zmq

        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, _PING_TIMEOUT_MS)
        sock.setsockopt(zmq.SNDTIMEO, _PING_TIMEOUT_MS)
        sock.connect(f"tcp://{self.host}:{self.port}")
        try:
            sock.send(msgpack.packb({"op": "ping"}, use_bin_type=True))
            rep = msgpack.unpackb(sock.recv(), raw=False)
        except zmq.ZMQError:
            sock.close(linger=0)
            return False
        if not (isinstance(rep, dict) and rep.get("ok")):
            sock.close(linger=0)
            return False
        # Promote the ping socket to the live inference socket (raise the timeout).
        sock.setsockopt(zmq.RCVTIMEO, _INFER_TIMEOUT_MS)
        sock.setsockopt(zmq.SNDTIMEO, _INFER_TIMEOUT_MS)
        self._sock = sock
        return True

    def _spawn(self) -> None:
        script = _locate_da3_boot_script()
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        argv = [sys.executable, str(script), "--port", str(self.port)]
        _log.info("da3.spawning", argv=argv)
        self._child = subprocess.Popen(argv, env=env, start_new_session=True)
