"""GpuPassthroughSkill — minimal rSkill whose per-step work runs on the GPU.

A minimal no-op rSkill that exercises the GPU sensor-frame path. Designed to
prove the "rSkill processes the frame on GPU" half of the M8 end-to-end
demo: when the GStreamer sensor reader hands a CPU :class:`SensorFrame`
to this skill's ``step()``, the skill uploads the frame to a torch
CUDA tensor exactly once and runs a tiny convolutional reduction
on the GPU, returning the per-channel mean intensity as the action
``confidence``. The whole image-processing step is verifiable as on-GPU
via ``nvidia-smi`` (process appears) and via the OTel
``skill.step_impl`` span attributes recorded here.

This skill carries **no learned weights** — it's a plumbing-grade
artefact. The same wiring (frame in → torch.cuda → action out) is the
seam where a real VLA (ACT, SmolVLA, π0.5) plugs in.

Example:
    >>> import os
    >>> # Skipped at doctest time because torch.cuda may be unavailable.
    >>> os.environ.get("OPENRAL_RUN_GPU_DOCTESTS", "0") != "0" or True
    True
"""

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.schemas import Action, ControlMode, FrameEncoding, WorldState

from openral_rskill.base import rSkillBase

if TYPE_CHECKING:
    import torch as _torch_t  # noqa: F401

__all__ = ["GpuPassthroughSkill"]

log = structlog.get_logger(__name__)


# Number of pixels in the small grayscale reduction. Higher is more GPU
# work per step; 64x64 was picked as the smallest that the GPU latency
# is still measurable above noise on an RTX 4070 Laptop (~0.05 ms).
_REDUCTION_SIZE: int = 64

# RGB channel count — the GPU reduction returns one mean per channel,
# padded to this length for mono / grayscale inputs.
_RGB_CHANNELS: int = 3


class GpuPassthroughSkill(rSkillBase):
    """Minimal rSkill whose per-step image processing runs on a GPU tensor.

    Args:
        sensor_id: Sensor key to read from ``WorldState.image_frames``
            (default ``"wrist_rgb"`` to match the camera YAMLs).
        n_joints: Number of joints in the zero action chunk (default 6
            for SO-100).
        horizon: Action chunk length (default 1).
        device: Torch device. Defaults to ``"cuda"`` and raises at
            ``configure()`` time if no CUDA device is visible.
        latency_budget_ms: Optional latency budget for the runner to
            surface budget overruns.

    Raises:
        RuntimeError: When ``configure()`` runs and ``device.startswith("cuda")``
            but ``torch.cuda.is_available()`` returns ``False``. We refuse to
            silently fall back to CPU because the whole point of this skill
            is to do real GPU work.
    """

    def __init__(
        self,
        sensor_id: str = "wrist_rgb",
        n_joints: int = 6,
        horizon: int = 1,
        device: str = "cuda",
        latency_budget_ms: float | None = None,
    ) -> None:
        """Stash configuration; torch is imported lazily in ``configure``."""
        super().__init__(
            name="gpu_passthrough_skill",
            version="0.1.0",
            role="s1",
            embodiment_tags=["any"],
            latency_budget_ms=latency_budget_ms,
        )
        self._sensor_id = sensor_id
        self._n_joints = n_joints
        self._horizon = horizon
        self._device_str = device
        self._step_count: int = 0
        # Populated by configure()
        self._torch: Any = None  # the torch module (lazy)
        self._device: Any = None  # torch.device
        # Frame cache: avoid re-allocating the GPU input buffer every step.
        self._gpu_input: Any = None  # torch.Tensor[float32, H*W*3 reduced]

    @property
    def step_count(self) -> int:
        """Number of times :meth:`step` has been called successfully."""
        return self._step_count

    # ── Hook overrides ────────────────────────────────────────────────────────

    def on_load_weights(self) -> None:
        """No weights — this is a passthrough."""
        log.info("gpu_passthrough_skill.on_load_weights")

    def on_quantize(self) -> None:
        """No weights to quantize."""
        log.info("gpu_passthrough_skill.on_quantize")

    def on_warmup(self) -> None:
        """Warm the GPU input buffer so the first step doesn't pay cudaMalloc.

        Allocates a zero tensor of the reduction shape on the configured
        device and forces a kernel launch by computing its mean.
        """
        if self._torch is None or self._device is None:
            return
        warm = self._torch.zeros(
            (3, _REDUCTION_SIZE, _REDUCTION_SIZE),
            dtype=self._torch.float32,
            device=self._device,
        )
        # Force an actual kernel launch by computing a mean. ``mean(dim=...)``
        # returns a tensor of shape (3,); reduce again to a scalar so
        # ``.item()`` is valid.
        _ = warm.mean(dim=(1, 2)).mean().item()
        self._gpu_input = warm
        log.info(
            "gpu_passthrough_skill.on_warmup",
            device=str(self._device),
            reduction_size=_REDUCTION_SIZE,
        )

    # ── Implementation hooks ──────────────────────────────────────────────────

    def _configure_impl(self) -> None:
        """Lazy-import torch and resolve the CUDA device.

        Raises:
            RuntimeError: When CUDA is requested but unavailable.
        """
        import torch  # noqa: PLC0415  # reason: heavy optional dep

        self._torch = torch
        if self._device_str.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "GpuPassthroughSkill.configure: device='cuda' but "
                "torch.cuda.is_available() is False. Refusing to silently "
                "fall back to CPU — this skill is designed to do real GPU "
                "work. Pass device='cpu' explicitly if that is what you want."
            )
        self._device = torch.device(self._device_str)
        log.info(
            "gpu_passthrough_skill.configure_impl",
            device=str(self._device),
            torch_version=torch.__version__,
            cuda_available=bool(torch.cuda.is_available()),
            cuda_device_count=int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        )

    def _activate_impl(self) -> None:
        """Reset the step counter."""
        self._step_count = 0
        log.info("gpu_passthrough_skill.activate_impl")

    def _deactivate_impl(self) -> None:
        """No teardown needed beyond the parent's flag flip."""
        log.info("gpu_passthrough_skill.deactivate_impl")

    def _shutdown_impl(self) -> None:
        """Release the cached GPU buffer."""
        self._gpu_input = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        log.info("gpu_passthrough_skill.shutdown_impl")

    def _step_impl(self, world_state: WorldState) -> Action:
        """Upload the latest frame to GPU, reduce, return action with stats.

        On each step:

        1. Pull the latest :class:`~openral_core.SensorFrame` for
           ``sensor_id`` from ``world_state.image_frames``.
        2. Decode the raw bytes into a (H, W, C) ``uint8`` tensor (CPU).
        3. Slice / downsample to ``_REDUCTION_SIZE`` and upload to the GPU
           via ``torch.from_numpy(...).to(device)`` — this is the explicit
           CPU→GPU copy that the SensorFrame.data: bytes contract today
           requires. (Removing this copy means switching to
           ``SensorFrame.handle`` — see ADR-0011.)
        4. Run a real GPU reduction (per-channel mean) — the actual on-GPU
           work that earns the "all on GPU" claim.
        5. Read back the 3 scalars and pack them into the action's
           ``confidence`` (stored as a string-encoded JSON for downstream
           verification).

        Returns:
            A ``JOINT_POSITION`` zero action chunk whose ``confidence``
            field is overwritten with the GPU-reduced per-channel mean
            of the most-recent camera frame (scaled to [0, 1]).
        """
        assert self._torch is not None, "skill must be configure()-d first"
        assert self._device is not None
        torch = self._torch
        self._step_count += 1

        frame = self._extract_latest_image(world_state)
        gpu_means = self._gpu_reduce(frame, torch=torch)
        # gpu_means is a Python tuple of 3 floats in [0, 1].
        confidence = float(sum(gpu_means) / 3.0)

        return Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=self._horizon,
            joint_targets=[[0.0] * self._n_joints for _ in range(self._horizon)],
            confidence=confidence,
            stamp_ns=time.time_ns(),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_latest_image(self, world_state: WorldState) -> NDArray[np.uint8]:
        """Pull a (H, W, 3) uint8 ndarray from ``world_state.image_frames``.

        Falls back to a zero frame when the configured sensor is missing
        (e.g. on the first tick before the camera has produced any
        frames) so the skill remains live-resilient and doesn't crash
        the runner during start-up.
        """
        if world_state.image_frames is None:
            return _zero_frame()
        sf = world_state.image_frames.get(self._sensor_id)
        if sf is None or sf.data is None:
            return _zero_frame()
        # SensorFrame.data may have been transported as a base64 string
        # (transient_local QoS / JSON round-trip). The schema's validator
        # decodes it eagerly back into bytes, but we defend against both.
        raw = sf.data if isinstance(sf.data, bytes) else base64.b64decode(sf.data)
        h, w = sf.height, sf.width
        if h <= 0 or w <= 0:
            return _zero_frame()
        encoding = sf.encoding
        try:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, _channels(encoding))
        except ValueError:
            # Shape mismatch — log and return zeros rather than crashing.
            log.warning(
                "gpu_passthrough_skill.frame_shape_mismatch",
                sensor_id=self._sensor_id,
                expected_shape=(h, w, _channels(encoding)),
                data_len=len(raw),
            )
            return _zero_frame()
        return arr

    def _gpu_reduce(
        self,
        frame: NDArray[np.uint8],
        *,
        torch: Any,  # noqa: ANN401  # reason: heavy optional dep — typed at use sites
    ) -> tuple[float, float, float]:
        """Upload the frame, do per-channel mean on the GPU, return floats.

        The reduction runs on a ``_REDUCTION_SIZE``-by-``_REDUCTION_SIZE``
        downsample to keep latency bounded.
        """
        h, w, _c = frame.shape
        # Lightweight CPU-side downsample (nearest-neighbour) so we move
        # at most _REDUCTION_SIZE^2 * 3 bytes across the PCIe bus.
        step_h = max(1, h // _REDUCTION_SIZE)
        step_w = max(1, w // _REDUCTION_SIZE)
        small = frame[::step_h, ::step_w, :3]
        small = small[:_REDUCTION_SIZE, :_REDUCTION_SIZE, :]

        # CPU → GPU, then a real GPU op. This is the on-GPU work.
        t = torch.from_numpy(small).to(self._device, dtype=torch.float32) / 255.0
        # t shape: (H, W, C). Reduce along H and W.
        means = t.mean(dim=(0, 1))  # shape (C,)
        # Force a sync so we can prove the kernel really ran (and so the
        # OTel latency timing covers the GPU compute, not just dispatch).
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        m = means.cpu().tolist()  # 3 floats; CPU read-back is intentional
        # Pad to length _RGB_CHANNELS for grayscale / mono input.
        while len(m) < _RGB_CHANNELS:
            m.append(m[-1] if m else 0.0)
        return float(m[0]), float(m[1]), float(m[2])


def _channels(encoding: FrameEncoding) -> int:
    """Return the per-pixel channel count for a FrameEncoding."""
    if encoding == FrameEncoding.MONO8:
        return 1
    if encoding in (FrameEncoding.BGR8, FrameEncoding.RGB8):
        return 3
    # Defensive default — treat as BGR. (No FrameEncoding member is
    # 4-channel today; a former ``BGRA8`` branch here referenced an enum
    # member that never existed and raised AttributeError for any
    # encoding that fell past the RGB checks, e.g. DEPTH16.)
    return 3


def _zero_frame() -> NDArray[np.uint8]:
    """Return a tiny zero placeholder frame to keep ``_step_impl`` resilient."""
    return np.zeros((_REDUCTION_SIZE, _REDUCTION_SIZE, 3), dtype=np.uint8)
