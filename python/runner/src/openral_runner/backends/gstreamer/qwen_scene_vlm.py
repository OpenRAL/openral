"""Scene-VLM backend backed by the Qwen3.5-4B sidecar (ADR-0047).

``Qwen/Qwen3.5-4B`` is a natively-multimodal vision-language model that, given
an RGB image and a natural-language question, returns a free-text answer. It
powers the reasoner's read-only ``query_scene`` tool — open-ended scene-state
verification ("has the robot grasped the mug?", "is the task complete?",
"did we drop the object?").

The model runs NF4 in an isolated sidecar process
(:mod:`tools.qwen_vlm_sidecar`) for dependency / VRAM isolation; this backend is
the ZMQ client. It mirrors the lifecycle of
:class:`~openral_runner.backends.gstreamer.locateanything_detector.LocateAnythingDetector`
(lazy connect, auto-spawn, teardown only the child we started) but its result is
*text*, not :class:`~openral_core.ObjectsMetadata` — a scene VLM is a reasoning
aid, not a localizer (use the detector for boxes).
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from openral_core import RSkillManifest
from openral_core.exceptions import ROSConfigError


def _find_sidecar_script() -> Path:
    """Locate ``tools/qwen_vlm_sidecar.py`` (env override or repo walk)."""
    override = os.environ.get("OPENRAL_QWEN_VLM_SIDECAR")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tools" / "qwen_vlm_sidecar.py"
        if cand.exists():
            return cand
    raise ROSConfigError(
        "could not locate tools/qwen_vlm_sidecar.py; set OPENRAL_QWEN_VLM_SIDECAR to its path"
    )


class QwenSceneVlm:
    """ZMQ client + auto-managed lifecycle for the Qwen3.5-4B scene-VLM sidecar.

    Ping the server, auto-spawn the sidecar if it isn't already up, and tear
    down only the child we started.
    """

    def __init__(
        self,
        *,
        model_id: str,
        weights_source: str = "Qwen/Qwen3.5-4B",
        host: str = "127.0.0.1",
        port: int = 5759,
        auto_spawn: bool = True,
        boot_timeout_s: float = 1200.0,
        request_timeout_s: float = 180.0,
        max_side: int = 1024,
        max_new_tokens: int = 256,
    ) -> None:
        """Store config; connection to the sidecar is deferred to first query()."""
        self._model_id = model_id
        self._weights_source = weights_source
        self._host = host
        self._port = port
        self._auto_spawn = auto_spawn
        self._boot_timeout_s = boot_timeout_s
        self._max_side = max_side
        self._max_new_tokens = max_new_tokens
        self._request_timeout_ms = int(request_timeout_s * 1000)
        # Connection is established lazily on first query() so construction is
        # cheap and side-effect-free — the dispatch path and tests can build the
        # backend without a running sidecar or a GPU. `Any` because pyzmq attrs
        # aren't typed under strict (mirrors the LocateAnything backend).
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
            raise ROSConfigError(
                "Qwen scene VLM needs the ZMQ + msgpack sidecar client; "
                "install it with `uv sync --group qwen-vlm` (provides pyzmq + msgpack)."
            ) from exc

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._connect()
        if not self._try_ping():
            if not self._auto_spawn:
                raise ROSConfigError(
                    f"no Qwen scene-VLM sidecar at tcp://{self._host}:{self._port} "
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
        print(f"[qwen-vlm] spawning sidecar: {' '.join(cmd)}", flush=True)
        self._child = subprocess.Popen(cmd)
        deadline = time.monotonic() + boot_timeout_s
        while time.monotonic() < deadline:
            if self._child.poll() is not None:
                raise ROSConfigError(
                    f"Qwen scene-VLM sidecar exited early (code {self._child.returncode})"
                )
            if self._try_ping():
                print("[qwen-vlm] sidecar ready", flush=True)
                return
            time.sleep(2.0)
        raise ROSConfigError(f"Qwen scene-VLM sidecar not ready within {boot_timeout_s}s")

    # -- public api ---------------------------------------------------------

    def query(self, frame_bgr: bytes, width: int, height: int, question: str) -> str:
        """Ask ``question`` about a raw BGR frame; return the VLM's text answer.

        Args:
            frame_bgr: Raw BGR888 bytes (``height * width * 3``).
            width: Frame width in pixels.
            height: Frame height in pixels.
            question: Natural-language question about the scene.

        Returns:
            The model's free-text answer (whitespace-stripped).

        Raises:
            ROSConfigError: If the question is empty or the sidecar errors.
        """
        if not question.strip():
            raise ROSConfigError("scene query question must be non-empty")

        import numpy as np  # noqa: PLC0415 — lazy: keep numpy/PIL off the import path
        from PIL import Image  # noqa: PLC0415

        self._ensure_ready()

        arr = np.frombuffer(frame_bgr, dtype=np.uint8).reshape(height, width, 3)
        rgb = arr[:, :, ::-1]  # BGR -> RGB
        buf = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(buf, format="PNG")

        reply = self._rpc(
            {
                "op": "query",
                "image": buf.getvalue(),
                "question": question.strip(),
                "max_side": self._max_side,
                "max_new_tokens": self._max_new_tokens,
            }
        )
        if not reply.get("ok"):
            raise ROSConfigError(f"Qwen scene-VLM sidecar error: {reply.get('error')}")
        return str(reply["answer"]).strip()

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


def build_scene_vlm(
    manifest: RSkillManifest,
    *,
    host: str = "127.0.0.1",
    port: int = 5759,
) -> QwenSceneVlm:
    """Build a :class:`QwenSceneVlm` from a ``kind: "vlm"`` rSkill manifest.

    Args:
        manifest: A validated rSkill manifest with ``kind == "vlm"``.
        host: Sidecar host to connect to.
        port: Sidecar port to connect to.

    Returns:
        A lazily-connecting :class:`QwenSceneVlm` (no sidecar spawned until the
        first :meth:`QwenSceneVlm.query`).

    Raises:
        ROSConfigError: If the manifest is not ``kind == "vlm"``.
    """
    if manifest.kind != "vlm":
        raise ROSConfigError(
            f"build_scene_vlm requires kind='vlm', got {manifest.kind!r} for {manifest.name!r}"
        )
    # Load ``weights_uri`` — the deployable checkpoint (the pre-quantized NF4
    # repo, e.g. OpenRAL/rskill-qwen35-4b-nf4) — NOT ``source_repo``, which is
    # provenance (the raw upstream model it was quantized from). The sidecar
    # server auto-detects a pre-quantized config and loads it directly.
    raw = manifest.weights_uri or manifest.source_repo or "Qwen/Qwen3.5-4B"
    weights_source = raw.removeprefix("hf://").split("@", 1)[0]
    return QwenSceneVlm(
        model_id=manifest.name,
        weights_source=weights_source,
        host=host,
        port=port,
    )
