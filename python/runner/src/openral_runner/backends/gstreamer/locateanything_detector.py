"""Open-vocabulary detector backend backed by the LocateAnything-3B sidecar.

``nvidia/LocateAnything-3B`` is a visual-grounding VLM (MoonViT vision tower +
Qwen2.5-3B) that, given an RGB image and a free-text query, emits structured
``<ref>label</ref><box><x1><y1><x2><y2></box>`` tokens (coordinates normalized
to ``[0, 1000]``). It cannot share the runtime's ``transformers>=5`` env, so it
runs in an isolated ``transformers==4.57.1`` sidecar process
(:mod:`tools.locateanything_sidecar`); this backend is the ZMQ client.

It implements the same ``detect(frame_bgr, width, height, sensor_id) ->
ObjectsMetadata | None`` interface as
:class:`~openral_runner.backends.gstreamer.objects_detector.ObjectsDetector`,
so :class:`~openral_runner.backends.gstreamer.detector_runner.DetectorRunner`
can drive it from a camera tee (wired in a later PR).

**Open-vocabulary query (static default + dynamic override).** The query
defaults to the manifest's ``detector.labels`` (joined with ``</c>`` for the
multi-category prompt). :meth:`LocateAnythingDetector.set_query` overrides it at
runtime — the hook the S2 reasoner will drive once the goal-params path is wired.

**Confidence.** LocateAnything is a grounding model: it emits boxes but no
per-box scores. Every detection is therefore reported with ``confidence=1.0``;
``score_threshold`` does not apply. This is recorded honestly rather than
fabricating a score (CLAUDE.md §1.2).
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from openral_core import ObjectDetection2D, ObjectsMetadata
from openral_core.exceptions import ROSConfigError

# ``<ref>label</ref>`` or a 4-coord ``<box>`` (point boxes have 2 coords and are
# ignored for object detection). Matched together so each box binds to the most
# recent preceding ref label, in document order.
_TOKEN_RE = re.compile(r"<ref>(.*?)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

# Degenerate-box guards: the model can loop on a repeated box token when it
# fails to emit an end token, producing a tail of identical near-full-image
# slivers (observed: ``<box><981><0><1000><1000></box>`` x ~140).
_MIN_SIDE_FRAC = 0.02  # drop boxes thinner than 2% of the image in either axis
_MAX_AREA_FRAC = 0.85  # drop boxes covering >85% of the image


def parse_grounding_answer(
    answer: str, *, fallback_label: str = "object", norm: int = 1000
) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Parse LocateAnything's raw text into ``(label, (x1, y1, x2, y2))`` boxes.

    Coordinates stay in the model's normalized ``[0, norm]`` space. Each box
    takes the most recent ``<ref>`` label (or ``fallback_label`` if none seen
    yet). Exact duplicates and degenerate boxes (slivers / near-full-image) are
    dropped.

    Args:
        answer: Raw generated text from the model.
        fallback_label: Label for boxes emitted before any ``<ref>`` token.
        norm: Coordinate normalization range (LocateAnything uses 1000).

    Returns:
        Boxes in document order, coordinates normalized and corner-ordered.
    """
    current = fallback_label
    seen: set[tuple[str, int, int, int, int]] = set()
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for m in _TOKEN_RE.finditer(answer):
        ref = m.group(1)
        if ref is not None:
            current = ref.strip() or fallback_label
            continue
        x1, y1, x2, y2 = (int(m.group(i)) for i in range(2, 6))
        key = (current, x1, y1, x2, y2)
        if key in seen:
            continue
        seen.add(key)
        lo_x, hi_x = sorted((x1, x2))
        lo_y, hi_y = sorted((y1, y2))
        bw, bh = (hi_x - lo_x) / norm, (hi_y - lo_y) / norm
        if bw <= _MIN_SIDE_FRAC or bh <= _MIN_SIDE_FRAC:
            continue
        if bw * bh >= _MAX_AREA_FRAC:
            continue
        out.append((current, (lo_x, lo_y, hi_x, hi_y)))
    return out


def build_objects_metadata(
    answer: str,
    *,
    width: int,
    height: int,
    model_id: str,
    sensor_id: str,
    fallback_label: str = "object",
    norm: int = 1000,
) -> ObjectsMetadata | None:
    """Build :class:`ObjectsMetadata` from a raw grounding answer.

    Normalized boxes are scaled into the ``width`` x ``height`` pixel space and
    clipped to frame bounds. Returns ``None`` if no valid detections remain.
    """
    dets: list[ObjectDetection2D] = []
    for label, (x1, y1, x2, y2) in parse_grounding_answer(
        answer, fallback_label=fallback_label, norm=norm
    ):
        px = (
            max(0, min(round(x1 / norm * width), width)),
            max(0, min(round(y1 / norm * height), height)),
            max(0, min(round(x2 / norm * width), width)),
            max(0, min(round(y2 / norm * height), height)),
        )
        if px[2] <= px[0] or px[3] <= px[1]:
            continue
        dets.append(ObjectDetection2D(label=label, confidence=1.0, bbox_xyxy=px))
    if not dets:
        return None
    return ObjectsMetadata(
        sensor_id=sensor_id,
        detections=dets,
        model_id=model_id,
        frame_width=width,
        frame_height=height,
    )


def _find_sidecar_script() -> Path:
    """Locate ``tools/locateanything_sidecar.py`` (env override or repo walk)."""
    override = os.environ.get("OPENRAL_LOCATEANYTHING_SIDECAR")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tools" / "locateanything_sidecar.py"
        if cand.exists():
            return cand
    raise ROSConfigError(
        "could not locate tools/locateanything_sidecar.py; set "
        "OPENRAL_LOCATEANYTHING_SIDECAR to its path"
    )


class LocateAnythingDetector:
    """ZMQ client + auto-managed lifecycle for the LocateAnything sidecar.

    Mirrors the RLDX-1 adapter pattern: ping the server, auto-spawn the sidecar
    if it isn't already up, and tear down only the child we started.
    """

    def __init__(
        self,
        *,
        labels: list[str],
        model_id: str,
        weights_source: str = "nvidia/LocateAnything-3B",
        host: str = "127.0.0.1",
        port: int = 5757,
        query: str | None = None,
        auto_spawn: bool = True,
        boot_timeout_s: float = 1200.0,
        request_timeout_s: float = 180.0,
        max_side: int = 1024,
        max_new_tokens: int = 1024,
        mode: str = "hybrid",
    ) -> None:
        """Store config; connection to the sidecar is deferred to first detect()."""
        if not labels:
            raise ROSConfigError("LocateAnythingDetector requires at least one label")

        self.kind = "objects"
        self._labels = list(labels)
        self._model_id = model_id
        self._weights_source = weights_source
        self._query = query or "</c>".join(labels)
        self._fallback_label = labels[0]
        self._host = host
        self._port = port
        self._auto_spawn = auto_spawn
        self._boot_timeout_s = boot_timeout_s
        self._max_side = max_side
        self._max_new_tokens = max_new_tokens
        self._mode = mode
        self._request_timeout_ms = int(request_timeout_s * 1000)
        # Connection is established lazily on first detect() so construction is
        # cheap and side-effect-free — the dispatch path (build_manifest_detector)
        # and tests can build the backend without a running sidecar or a GPU.
        # `Any` mirrors the rldx adapter: pyzmq attrs aren't typed under strict.
        self._zmq: Any = None
        self._ctx: Any = None
        self._sock: Any = None
        self._child: subprocess.Popen[bytes] | None = None

    # -- wire ---------------------------------------------------------------

    def _ensure_ready(self) -> None:
        """Connect to the sidecar (spawning it if needed) on first use."""
        if self._sock is not None:
            return
        try:
            import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: PLC0415 — lazy: keep zmq off the module import path
        except ImportError as exc:  # pragma: no cover — env-provisioning guard
            # The detector-node-side ZMQ + msgpack client (shared sidecar
            # transport with rldx) lives in the ``locateanything`` extra. Without
            # it the deploy-sim detector leg fails per-request with a bare
            # "No module named 'zmq'"; surface the actionable fix instead.
            raise ROSConfigError(
                "LocateAnything detector needs the ZMQ + msgpack sidecar client; "
                "install it with `uv sync --group locateanything` "
                "(provides pyzmq + msgpack)."
            ) from exc

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._connect()
        if not self._try_ping():
            if not self._auto_spawn:
                raise ROSConfigError(
                    f"no LocateAnything sidecar at tcp://{self._host}:{self._port} "
                    "and auto_spawn=False"
                )
            self._spawn_and_wait(self._boot_timeout_s)

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

    def _spawn_and_wait(self, boot_timeout_s: float) -> None:
        import sys  # noqa: PLC0415 — lazy: only needed on the auto-spawn path

        script = _find_sidecar_script()
        cmd = [
            sys.executable,
            str(script),
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--model",
            self._weights_source,
            "--max-side",
            str(self._max_side),
        ]
        print(f"[la-detector] spawning sidecar: {' '.join(cmd)}", flush=True)
        self._child = subprocess.Popen(cmd)
        deadline = time.monotonic() + boot_timeout_s
        while time.monotonic() < deadline:
            if self._child.poll() is not None:
                raise ROSConfigError(
                    f"LocateAnything sidecar exited early (code {self._child.returncode})"
                )
            if self._try_ping():
                print("[la-detector] sidecar ready", flush=True)
                return
            time.sleep(2.0)
        raise ROSConfigError(f"LocateAnything sidecar not ready within {boot_timeout_s}s")

    # -- public api ---------------------------------------------------------

    def set_query(self, text: str) -> None:
        """Override the detection query at runtime (dynamic open-vocab hook)."""
        if not text.strip():
            raise ROSConfigError("detection query must be non-empty")
        self._query = text.strip()

    def detect(
        self, frame_bgr: bytes, width: int, height: int, sensor_id: str
    ) -> ObjectsMetadata | None:
        """Detect the current (persistent) query in a raw BGR frame."""
        return self.detect_with_query(frame_bgr, width, height, sensor_id, self._query)

    def detect_with_query(
        self, frame_bgr: bytes, width: int, height: int, sensor_id: str, query: str
    ) -> ObjectsMetadata | None:
        """One-shot detect for ``query`` without mutating the persistent query.

        Used by the on-demand ``locate_in_view`` service so a reasoner query
        doesn't change what the continuous detection leg grounds.
        """
        import numpy as np  # noqa: PLC0415 — lazy: keep numpy/PIL off the import path
        from PIL import Image  # noqa: PLC0415

        self._ensure_ready()

        arr = np.frombuffer(frame_bgr, dtype=np.uint8).reshape(height, width, 3)
        rgb = arr[:, :, ::-1]  # BGR -> RGB
        buf = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(buf, format="PNG")

        reply = self._rpc(
            {
                "op": "detect",
                "image": buf.getvalue(),
                "query": query or self._query,
                "max_side": self._max_side,
                "mode": self._mode,
                "max_new_tokens": self._max_new_tokens,
            }
        )
        if not reply.get("ok"):
            raise ROSConfigError(f"LocateAnything sidecar error: {reply.get('error')}")
        return build_objects_metadata(
            reply["answer"],
            width=width,
            height=height,
            model_id=self._model_id,
            sensor_id=sensor_id,
            fallback_label=query or self._fallback_label,
            norm=int(reply.get("norm", 1000)),
        )

    def close(self) -> None:
        """Close the socket and terminate the sidecar if we spawned it."""
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        if self._child is not None and self._child.poll() is None:
            with contextlib.suppress(Exception):
                self._rpc({"op": "shutdown"}, recv_timeout_ms=2000)
            try:
                self._child.terminate()
                self._child.wait(timeout=10)
            except Exception:  # best-effort teardown; escalate to kill
                self._child.kill()
            self._child = None
