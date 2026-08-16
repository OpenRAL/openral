"""In-process SAM 2.1 promptable segmenter for ``kind: "segmenter"`` rSkills.

The segmenter sibling of
:mod:`~openral_runner.backends.gstreamer.omdet_turbo_detector`, and it lives
here for the same reason: SAM 2.1 is a first-class ``transformers``
architecture (:class:`Sam2Model` + :class:`Sam2Processor`), so it loads under
the runtime's own ``transformers`` and runs **in process** — no sidecar venv,
no ZMQ, no quantization beyond the bf16 compute dtype. It is selected for
manifests whose ``segmenter.engine`` is
:attr:`~openral_core.schemas.SegmenterEngine.SAM2_HF`.

**What it answers.** A detector is asked a *semantic* question ("where are the
cups?") and replies with labelled, scored boxes. This backend is asked a
*geometric* one — "which pixels belong to the thing under this point?" — and
replies with binary masks. It has no label vocabulary and reports no
thresholdable confidence, because its consumer (the HAL's vision
attachment-evidence producer) is forbidden from gating on model confidence.
See :func:`segment` for the measurement that makes that a hard rule.

**Placement.** It runs beside the detector rSkills on the GStreamer graph side
because the HAL is deliberately kept torch-free; the HAL asks for masks over
the ``openral_msgs/srv/SegmentInView`` service rather than importing this
module.

**Lifecycle.** Resident and warmed at node activate, not loaded on demand. At a
measured ~297 MiB peak (RTX 4070 Laptop, bf16) it co-resides trivially with a
~3.5 GiB VLA on an 8 GB card, and the one-shot call must fit inside the HAL's
existing ~100 ms deferred-ack barrier. A warmed call is ~53 ms; the **first**
call is ~742 ms (kernel autotune), which would not fit — hence
:meth:`Sam2Segmenter.warm_up`, which must be called at activate.

**Resolution is not a lever.** ``Sam2Processor`` resizes every input to 1024²
internally, so latency was measured flat (46-52 ms) from 224² to 1024². Do not
spend budget shrinking the wrist camera for this model's sake.

``torch`` / ``transformers`` are imported lazily at first use so this module
imports cleanly on hosts without the wheels (mirroring
:mod:`~openral_runner.backends.gstreamer.omdet_turbo_detector`). Install with
``just sync --group sam2``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openral_core.exceptions import ROSConfigError, ROSRuntimeError

if TYPE_CHECKING:  # pragma: no cover — typing only, keeps numpy off the import path
    import numpy as np
    from numpy.typing import NDArray
    from openral_core.schemas import IntrinsicsPinhole

__all__ = [
    "MaskCandidate",
    "Sam2Segmenter",
    "build_mask_candidates",
    "project_point_to_pixel",
]

# A point closer than this to the camera's optical plane has no meaningful
# projection (the pinhole division blows up). Matches the epsilon
# `openral_world_state.object_lift` uses for the same guard.
_DEPTH_EPS = 1e-6


@dataclass(frozen=True)
class MaskCandidate:
    """One mask hypothesis returned for a prompt.

    SAM 2's multimask head emits three *nested* hypotheses per point prompt
    (roughly subpart / part / whole). Which one is right is a question only
    geometry can answer, so all surviving candidates are handed to the consumer
    rather than resolved here by score.

    Attributes:
        mask: ``(height, width)`` boolean array, ``True`` where the pixel
            belongs to the prompted object, at the source frame's own
            resolution.
        score_advisory: The model's own IoU estimate for this candidate.
            **Advisory only** — recorded in the trace, never thresholded.
        area_px: Number of set pixels, precomputed because every consumer wants
            it and it is what :attr:`min_mask_area_px` filters on.
    """

    mask: NDArray[np.bool_]
    score_advisory: float
    area_px: int


def project_point_to_pixel(
    point_xyz: tuple[float, float, float],
    intrinsics: IntrinsicsPinhole,
) -> tuple[float, float] | None:
    """Project a point in the camera **optical** frame to continuous pixels.

    The prompt crosses the ``SegmentInView`` boundary as a 3-D point precisely
    so that intrinsics stay on this side of it; this is where they are applied.
    REP-103 optical convention: ``+x`` right, ``+y`` down, ``+z`` forward.

    Args:
        point_xyz: Point in the camera optical frame, in metres.
        intrinsics: Pinhole intrinsics whose ``width``/``height`` bound the
            image rectangle. Scale them with
            :func:`openral_core.scale_intrinsics_to` first if the frame is not
            rendered at the calibrated resolution.

    Returns:
        ``(u, v)`` continuous pixel coordinates, or ``None`` when the point is
        behind (or on) the optical plane or falls outside the image — both of
        which mean the caller has no usable prompt, not that it should guess
        one.

    Example:
        >>> from openral_core import IntrinsicsPinhole
        >>> k = IntrinsicsPinhole(width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0)
        >>> project_point_to_pixel((0.0, 0.0, 1.0), k)
        (320.0, 240.0)
        >>> project_point_to_pixel((0.0, 0.0, -1.0), k) is None
        True
    """
    x, y, z = point_xyz
    if z <= _DEPTH_EPS:
        return None
    u = intrinsics.fx * x / z + intrinsics.cx
    v = intrinsics.fy * y / z + intrinsics.cy
    if not (0.0 <= u <= float(intrinsics.width) and 0.0 <= v <= float(intrinsics.height)):
        return None
    return (float(u), float(v))


class Sam2Segmenter:
    """In-process Transformers SAM 2.1 promptable segmenter.

    Loads ``Sam2Processor`` + ``Sam2Model`` from ``weights_source`` on first
    use, moves the model to CUDA when available (else CPU), and answers point
    prompts with binary masks at the source frame's resolution. The load is
    deferred so construction is cheap and side-effect-free — the dispatch path
    and unit tests can build the backend without the wheels or a GPU.

    Args:
        model_id: Identifier recorded alongside every emitted mask.
        weights_source: HF repo id to load (``facebook/sam2.1-hiera-small``).
        max_prompt_points: Upper bound on prompt points per call, from the
            manifest's :attr:`~openral_core.SegmenterContract.max_prompt_points`.
        multimask: Ask for the three-hypothesis head instead of a single mask.
        min_mask_area_px: Candidates with fewer set pixels are dropped as
            degenerate.
        device: ``"auto"`` picks ``cuda`` when available else ``cpu``; or pass
            an explicit ``"cpu"`` / ``"cuda"`` / ``"cuda:N"``.

    Raises:
        ROSConfigError: If ``weights_source`` is empty or a bound is not
            positive.
    """

    def __init__(
        self,
        *,
        model_id: str,
        weights_source: str,
        max_prompt_points: int = 8,
        multimask: bool = True,
        min_mask_area_px: int = 64,
        device: str = "auto",
    ) -> None:
        """Store config; the model/processor load is deferred to first use."""
        if not weights_source:
            raise ROSConfigError("Sam2Segmenter requires a non-empty weights_source")
        if max_prompt_points < 1 or min_mask_area_px < 1:
            raise ROSConfigError(
                "Sam2Segmenter requires max_prompt_points >= 1 and min_mask_area_px >= 1 "
                f"(got {max_prompt_points} / {min_mask_area_px})."
            )
        self.kind = "segmenter"
        self._model_id = model_id
        self._weights_source = weights_source
        self._max_prompt_points = max_prompt_points
        self._multimask = multimask
        self._min_mask_area_px = min_mask_area_px
        self._device_pref = device
        # Lazily populated on first use. ``Any`` because transformers/torch ship
        # partial / heavy stubs not worth importing under strict here.
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device: str = "cpu"

    @property
    def model_id(self) -> str:
        """Identifier recorded alongside every emitted mask."""
        return self._model_id

    def _ensure_ready(self) -> None:
        """Load torch + transformers and the model/processor on first use."""
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415 — lazy: keep torch off the module import path
            from transformers import Sam2Model, Sam2Processor  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover — env-provisioning guard
            raise ROSRuntimeError(
                "Sam2Segmenter needs 'torch' + 'transformers'. Install with: "
                "just sync --group sam2 (provides torch + transformers + pillow)."
            ) from exc

        self._torch = torch
        if self._device_pref == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._device_pref
        self._processor = Sam2Processor.from_pretrained(self._weights_source)
        # bf16 on CUDA (the measured 74 MiB / 297 MiB peak configuration); fp32
        # on CPU, where bf16 matmuls are slower rather than faster.
        dtype = torch.bfloat16 if self._device.startswith("cuda") else torch.float32
        # ``Any``: transformers' ``.to()`` overloads are typed for
        # ``PreTrainedModel`` and do not accept the concrete subclass here.
        model: Any = Sam2Model.from_pretrained(self._weights_source, dtype=dtype)
        self._model = model.to(self._device).eval()

    def warm_up(self, *, width: int = 320, height: int = 240) -> None:
        """Load the model and burn one forward pass, at node activate.

        The HAL's deferred-ack barrier already waits ~100 ms for a depth frame,
        which a warmed ~53 ms segmentation fits inside with room to spare. The
        **first** call does not: it was measured at ~742 ms on the reference
        GPU (weights load plus CUDA kernel autotune). Calling this at activate
        moves that cost off the first real attach event.

        Args:
            width: Warm-up frame width in pixels.
            height: Warm-up frame height in pixels.
        """
        import numpy as np  # noqa: PLC0415 — lazy: keep numpy off the import path

        self._ensure_ready()
        blank = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
        self.segment(
            blank,
            width,
            height,
            positive_points=[(width / 2.0, height / 2.0)],
        )

    def segment(
        self,
        frame_bgr: bytes,
        width: int,
        height: int,
        *,
        positive_points: Sequence[tuple[float, float]],
        negative_points: Sequence[tuple[float, float]] = (),
    ) -> list[MaskCandidate]:
        """Segment the object under the prompt points in one BGR frame.

        Candidates are returned ordered by **area ascending** — a deterministic,
        purely geometric ordering over SAM 2's nested subpart/part/whole
        hypotheses. They are deliberately *not* ordered or filtered by
        :attr:`MaskCandidate.score_advisory`: a mis-aimed point prompt on a real
        third-person frame was measured returning a mask covering 59.8% of the
        image — essentially the whole tablecloth — at this model's **top** score
        of 0.977. Model confidence cannot distinguish a correct mask from a
        confidently wrong one, so the consumer resolves the ambiguity on
        geometry (containment between the jaws, payload extent, depth validity)
        and this backend hands it everything it needs to do so.

        Args:
            frame_bgr: Raw BGR bytes from the GStreamer appsink (``w*h*3``).
            width: Frame width in pixels.
            height: Frame height in pixels.
            positive_points: ``(u, v)`` pixel prompts on the target object —
                normally the single TF-projected TCP.
            negative_points: ``(u, v)`` pixel prompts to exclude, normally the
                jaw tips, which keep the mask off the gripper fingers.

        Returns:
            Surviving :class:`MaskCandidate` s, area ascending. Empty when the
            frame bytes do not match ``width * height * 3`` or when no candidate
            clears ``min_mask_area_px``.

        Raises:
            ROSConfigError: If no positive point is given, or the total prompt
                exceeds ``max_prompt_points``.
            ROSRuntimeError: If torch / transformers are not installed.
        """
        import numpy as np  # noqa: PLC0415 — lazy: keep numpy off the import path
        from PIL import Image  # noqa: PLC0415

        if not positive_points:
            raise ROSConfigError("Sam2Segmenter.segment requires at least one positive point")
        total = len(positive_points) + len(negative_points)
        if total > self._max_prompt_points:
            raise ROSConfigError(
                f"Sam2Segmenter.segment got {total} prompt points, "
                f"max_prompt_points is {self._max_prompt_points}."
            )

        self._ensure_ready()
        try:
            arr = np.frombuffer(frame_bgr, dtype=np.uint8).reshape(height, width, 3)
        except ValueError:
            # Byte length doesn't match width*height*3 — caps mismatch upstream; drop.
            return []
        image = Image.fromarray(arr[:, :, ::-1], "RGB")  # BGR -> RGB

        points = [[float(u), float(v)] for u, v in positive_points]
        labels = [1] * len(positive_points)
        points += [[float(u), float(v)] for u, v in negative_points]
        labels += [0] * len(negative_points)

        inputs = self._processor(
            images=image,
            input_points=[[points]],
            input_labels=[[labels]],
            return_tensors="pt",
        ).to(self._device)
        with self._torch.no_grad():
            outputs = self._model(**inputs, multimask_output=self._multimask)
        masks = self._processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0]
        # post_process_masks returns (num_prompts, num_candidates, H, W); this
        # backend always sends exactly one prompt group.
        candidates = masks[0] if masks.ndim == 4 else masks  # noqa: PLR2004
        scores = [float(s) for s in outputs.iou_scores.reshape(-1).tolist()]
        return build_mask_candidates(
            [np.asarray(m.cpu().numpy(), dtype=bool) for m in candidates],
            scores=scores,
            min_mask_area_px=self._min_mask_area_px,
        )

    def close(self) -> None:
        """Release the model and free CUDA memory if we loaded onto the GPU."""
        had_cuda = self._model is not None and self._device.startswith("cuda")
        self._model = None
        self._processor = None
        if had_cuda and self._torch is not None:  # best-effort VRAM reclaim
            self._torch.cuda.empty_cache()


def build_mask_candidates(
    masks: Sequence[NDArray[np.bool_]],
    *,
    scores: Sequence[float],
    min_mask_area_px: int,
) -> list[MaskCandidate]:
    """Filter and order decoded masks into :class:`MaskCandidate` s.

    Pure (no torch / transformers) so the conversion is unit-testable without a
    GPU or a model load, mirroring
    :func:`~openral_runner.backends.gstreamer.omdet_turbo_detector.build_objects_metadata_from_results`.
    Masks with fewer than ``min_mask_area_px`` set pixels are dropped; survivors
    are sorted by area ascending (SAM 2's nested subpart → part → whole), never
    by score.

    Args:
        masks: Per-candidate boolean arrays, parallel to ``scores``.
        scores: Per-candidate advisory model IoU.
        min_mask_area_px: Minimum set-pixel count to keep a candidate.

    Returns:
        Surviving candidates, area ascending.

    Raises:
        ROSConfigError: If ``masks`` and ``scores`` have different lengths.

    Example:
        >>> import numpy as np
        >>> a = np.zeros((4, 4), dtype=bool)
        >>> a[0, 0] = True
        >>> b = np.ones((4, 4), dtype=bool)
        >>> cands = build_mask_candidates([b, a], scores=[0.9, 0.1], min_mask_area_px=1)
        >>> [c.area_px for c in cands]
        [1, 16]
    """
    if len(masks) != len(scores):
        raise ROSConfigError(
            f"build_mask_candidates: masks/scores length mismatch ({len(masks)}/{len(scores)})."
        )
    out = [
        MaskCandidate(mask=mask, score_advisory=float(score), area_px=int(mask.sum()))
        for mask, score in zip(masks, scores, strict=True)
        if int(mask.sum()) >= min_mask_area_px
    ]
    out.sort(key=lambda c: c.area_px)
    return out
