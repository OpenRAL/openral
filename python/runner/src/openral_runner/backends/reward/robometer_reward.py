"""Node-side ZMQ client for the Robometer reward-monitor sidecar (ADR-0057).

Mirrors :class:`openral_runner.backends.gstreamer.qwen_scene_vlm.QwenSceneVlm`:
ping → auto-spawn the sidecar if absent → talk msgpack over a strict REQ/REP
socket → tear down only the child we started. The sidecar is **stateless** — it
scores a clip of frames + a task instruction and returns per-frame progress +
success arrays. The rolling buffer / windowing lives node-side
(:class:`~openral_runner.backends.reward.frame_source.RollingFrameBuffer`).

Nothing here imports torch / transformers / numpy at module load; the heavy
model runs in the sidecar venv.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openral_core import RSkillManifest
from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from openral_runner.backends.reward.frame_source import Frame

# Default sidecar port — distinct from the scene-VLM (5759) and detector ports.
_DEFAULT_PORT = 5769
# |progress trend per sample| below this reads as "stalled" (no meaningful change).
_STALL_TREND_EPS = 0.002


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """``k`` evenly-spaced indices into ``range(n)``, always including the last.

    Used to subsample a frame window to a fixed budget so the reward model's
    vision-transformer activation stays bounded on an 8 GB GPU (ADR-0058). The
    newest frame (index ``n-1``) is always kept — the reasoner reads
    ``progress_now`` from it. Returns ``list(range(n))`` when ``n <= k``.
    """
    if n <= k:
        return list(range(n))
    step = (n - 1) / (k - 1) if k > 1 else 0.0
    idx = sorted({min(n - 1, round(i * step)) for i in range(k)})
    if idx[-1] != n - 1:
        idx[-1] = n - 1
    return idx


def _find_sidecar_script() -> Path:
    """Locate ``tools/robometer_sidecar.py`` (env override or repo walk)."""
    override = os.environ.get("OPENRAL_ROBOMETER_SIDECAR")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tools" / "robometer_sidecar.py"
        if cand.exists():
            return cand
    raise ROSConfigError(
        "could not locate tools/robometer_sidecar.py; set OPENRAL_ROBOMETER_SIDECAR to its path"
    )


class RobometerReward:
    """ZMQ client + auto-managed lifecycle for the Robometer reward sidecar."""

    def __init__(
        self,
        *,
        model_id: str,
        weights_source: str = "robometer/Robometer-4B",
        host: str = "127.0.0.1",
        port: int = _DEFAULT_PORT,
        auto_spawn: bool = True,
        boot_timeout_s: float = 1200.0,
        request_timeout_s: float = 180.0,
        num_bins: int = 100,
        success_threshold: float = 0.5,
        max_frames: int = 8,
    ) -> None:
        """Store config; connection to the sidecar is deferred to first use."""
        self._model_id = model_id
        self._weights_source = weights_source
        self._host = host
        self._port = port
        self._auto_spawn = auto_spawn
        self._boot_timeout_s = boot_timeout_s
        self._num_bins = num_bins
        self._success_threshold = success_threshold
        # Activation memory for the vision-transformer forward scales with the
        # number of frames (x resolution); a full 8 s x 3 fps window of 640x480
        # frames OOMs a 3.3 GB-resident model on an 8 GB GPU (ADR-0058, observed
        # in deploy-sim). Evenly subsample the window to at most this many frames
        # so the reward forward stays co-resident with the sim (and a small VLA).
        self._max_frames = max(1, max_frames)
        self._request_timeout_ms = int(request_timeout_s * 1000)
        # Lazy connection (mirrors QwenSceneVlm). `Any` because pyzmq attrs
        # aren't typed under strict.
        self._zmq: Any = None
        self._ctx: Any = None
        self._sock: Any = None
        self._child: subprocess.Popen[bytes] | None = None

    # -- wire ---------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if self._sock is not None:
            return
        try:
            import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: PLC0415 — lazy: keep zmq off the import path
        except ImportError as exc:  # pragma: no cover — env-provisioning guard
            raise ROSConfigError(
                "Robometer reward monitor needs the ZMQ + msgpack sidecar client; "
                "install it with `uv sync --group robometer` (provides pyzmq + msgpack)."
            ) from exc

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._connect()
        if not self._try_ping():
            if not self._auto_spawn:
                raise ROSConfigError(
                    f"no Robometer reward sidecar at tcp://{self._host}:{self._port} "
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
        import msgpack  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: PLC0415 — lazy

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
        import sys  # noqa: PLC0415 — lazy

        script = _find_sidecar_script()
        cmd = [
            sys.executable,
            str(script),
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--weights",
            self._weights_source,
        ]
        print(f"[robometer] spawning sidecar: {' '.join(cmd)}", flush=True)
        self._child = subprocess.Popen(cmd)
        deadline = time.monotonic() + boot_timeout_s
        while time.monotonic() < deadline:
            if self._child.poll() is not None:
                raise ROSConfigError(
                    f"Robometer reward sidecar exited early (code {self._child.returncode})"
                )
            if self._try_ping():
                print("[robometer] sidecar ready", flush=True)
                return
            time.sleep(2.0)
        raise ROSConfigError(f"Robometer reward sidecar not ready within {boot_timeout_s}s")

    # -- public api ---------------------------------------------------------

    def score(self, frames: list[Frame], task: str) -> tuple[list[float], list[float]]:
        """Score a clip → ``(progress_series, success_series)``, per frame.

        Args:
            frames: Buffered frames (all the same ``width``/``height``), oldest
                first. Each carries raw BGR888 bytes.
            task: The natural-language task instruction.

        Returns:
            ``(progress, success)`` lists of equal length to ``frames``,
            progress normalized to the model's discrete-mode ``[0, 1]`` range.

        Raises:
            ROSConfigError: empty clip / empty task / sidecar error.
        """
        if not frames:
            raise ROSConfigError("reward score requires at least one frame")
        if not task.strip():
            raise ROSConfigError("reward score requires a non-empty task instruction")
        # Bound activation memory: evenly subsample to <= max_frames.
        if len(frames) > self._max_frames:
            idx = _evenly_spaced_indices(len(frames), self._max_frames)
            print(
                f"[robometer] subsampling {len(frames)} -> {len(idx)} frames "
                f"(max_frames={self._max_frames}) to bound activation memory",
                flush=True,
            )
            frames = [frames[i] for i in idx]
        w, h = frames[0].width, frames[0].height
        if any(f.width != w or f.height != h for f in frames):
            raise ROSConfigError("all frames in a clip must share width/height")

        self._ensure_ready()
        reply = self._rpc(
            {
                "op": "score",
                "frames": b"".join(f.bgr for f in frames),
                "n": len(frames),
                "width": w,
                "height": h,
                "task": task.strip(),
                "num_bins": self._num_bins,
            }
        )
        if not reply.get("ok"):
            raise ROSConfigError(f"Robometer reward sidecar error: {reply.get('error')}")
        progress = [float(x) for x in reply["progress"]]
        success = [float(x) for x in reply["success"]]
        return progress, success

    def assess(self, frames: list[Frame], task: str) -> dict[str, Any]:
        """Score ``frames`` and summarize the window for the Reasoner.

        Returns a dict with ``progress_now``, ``success_now``,
        ``progress_trend``, ``success_trend``, ``stalled``, ``succeeded``
        (success_now ≥ threshold), and ``frames_seen``.
        """
        from openral_runner.backends.reward.frame_source import trend  # noqa: PLC0415

        progress, success = self.score(frames, task)
        p_trend = trend(progress)
        return {
            "progress_now": progress[-1],
            "success_now": success[-1],
            "progress_trend": p_trend,
            "success_trend": trend(success),
            "stalled": abs(p_trend) < _STALL_TREND_EPS,
            "succeeded": success[-1] >= self._success_threshold,
            "frames_seen": len(frames),
        }

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


def build_reward_monitor(
    manifest: RSkillManifest,
    *,
    host: str = "127.0.0.1",
    port: int = _DEFAULT_PORT,
) -> RobometerReward:
    """Build a :class:`RobometerReward` from a ``kind: "reward"`` rSkill manifest.

    Args:
        manifest: A validated rSkill manifest with ``kind == "reward"``.
        host: Sidecar host to connect to.
        port: Sidecar port to connect to.

    Returns:
        A lazily-connecting :class:`RobometerReward` (no sidecar spawned until
        the first :meth:`RobometerReward.score`).

    Raises:
        ROSConfigError: If the manifest is not ``kind == "reward"`` or lacks a
            ``reward`` block.
    """
    if manifest.kind != "reward":
        raise ROSConfigError(
            f"build_reward_monitor requires kind='reward', got {manifest.kind!r} "
            f"for {manifest.name!r}"
        )
    if manifest.reward is None:  # pragma: no cover — validator guarantees this
        raise ROSConfigError(f"reward manifest {manifest.name!r} has no `reward` block")
    raw = manifest.weights_uri or manifest.source_repo or "robometer/Robometer-4B"
    # hf://org/repo[@rev] -> "org/repo[@rev]" (sidecar resolves rev); local:///path
    # -> "/path" (a pre-quantized checkpoint dir loaded directly as 4-bit).
    if raw.startswith("local://"):
        weights_source = raw.removeprefix("local://")
    else:
        weights_source = raw.removeprefix("hf://").split("@", 1)[0]
    return RobometerReward(
        model_id=manifest.name,
        weights_source=weights_source,
        host=host,
        port=port,
        num_bins=manifest.reward.num_bins,
        success_threshold=manifest.reward.success_threshold,
    )
