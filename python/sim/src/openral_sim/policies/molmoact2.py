"""MolmoAct2 (Ai2) policy adapter.

Wraps Ai2's `MolmoAct2 <https://huggingface.co/allenai/MolmoAct2-LIBERO>`_
action reasoning model — a Molmo2-ER embodied-reasoning VLM backbone with a
flow-matching continuous-action expert grafted on via per-layer KV-cache
conditioning (arXiv:2605.02881). The LIBERO finetune
(``allenai/MolmoAct2-LIBERO``) scores 97.2 % on the LIBERO suite (98.1 % for
the depth-reasoning ``-Think`` variant), edging out π0.5.

Unlike the other in-tree VLA adapters, MolmoAct2 is **not** a lerobot policy.
It ships as a transformers *custom-code* model (``trust_remote_code``,
``auto_map`` → ``MolmoAct2ForConditionalGeneration``) and is driven through its
own :meth:`predict_action` API rather than lerobot's ``select_action`` queue:

- Bare rSkill reference required as weights URI (the manifest is the
  robot/sensor/IO contract; the eval layer never loads weights without one).
- The model + processor + ``norm_stats.json`` come from the manifest's
  ``source_repo`` (``hf://allenai/MolmoAct2-LIBERO``) via transformers'
  ``AutoModelForImageTextToText`` / ``AutoProcessor``.
- The eval-layer ``Observation`` (flat ``state`` + ``images`` dict) is turned
  into the ``predict_action(images=[agentview, wrist], task=, state=, ...)``
  call. ``predict_action`` returns a *chunk* of denormalized actions in robot
  scale; the adapter replays it one step at a time (``n_action_steps`` cadence)
  and re-infers when the queue empties — the same closed-loop replay π0.5 gets
  for free from lerobot's ``select_action``.

NF4 quantization reuses the adapter-agnostic helpers in
:mod:`openral_sim._quantization` (``quantize_nf4_in_place`` +
``load_prequantized_state_for_rskill``), the same ones π0.5 uses; they operate
on any ``torch.nn.Module`` tree and make no π0.5-specific assumption. The
bf16 MolmoAct2 backbone is ~11 GiB and OOMs an 8 GiB consumer GPU; NF4 brings
the working set to ~4 GiB so the rollout fits. ``tools/quantize_rskill.py
--loader transformers`` produces the matching prequant pack at the rSkill's
``weights_uri`` so the on-line bf16→nf4 conversion is skipped at load time.

This module imports torch / transformers lazily so installing ``openral-sim``
never pulls them transitively.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_observability import inference_span
from openral_rskill._diagnostics import phase_timer
from openral_rskill._vla_core import (
    resolve_camera_keys,
    resolve_device,
    resolve_image_preprocessing,
    resolve_rskill_repo_id,
    resolve_state_dim,
)

# NF4 quantization + prequantized-state load live in the adapter-agnostic
# helper module so π0.5 / MolmoAct2 / future families share one bnb rewrite
# path. The dtype-resolution helpers (manifest_dtype, torch_dtype_for,
# default_dtype_for_device) live there too so every adapter speaks the same
# QuantizationDtype enum vocabulary.
from openral_sim._quantization import (
    default_dtype_for_device,
    detect_prequantized_nf4,
    load_prequantized_state_for_rskill,
    manifest_dtype,
    peek_safetensors_keys,
    quantize_nf4_in_place,
    targeted_reset_parameters,
    tie_transformers_weights,
    torch_dtype_for,
)
from openral_sim.policies._policy_loading import load_manifest_for_spec
from openral_sim.registry import POLICIES

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation

_log = structlog.get_logger(__name__)

# MolmoAct2-LIBERO is trained on the LIBERO mixture; the matching
# normalization tag selects the right state/action norm_stats entry. The
# checkpoint's norm_stats.json carries the "libero" tag (README §Intended Use).
_DEFAULT_NORM_TAG = "libero"
# Flow-matching solver steps. README default is the checkpoint config value
# (10); fewer steps trade a little accuracy for latency.
_DEFAULT_NUM_STEPS = 10
# predict_action returns a batched (B, n_action_steps, action_dim) chunk.
_BATCHED_CHUNK_NDIM = 3
# Operator-facing override for the image processor's multi-crop count
# (MolmoAct2ImageProcessor.max_crops, checkpoint default 8). Mirrors the existing
# OPENRAL_SIM_SEQUENTIAL_INIT env knob convention. Each extra crop adds a 378 px
# tile (≈729 patches → ≈182 pooled image tokens) with quadratic attention cost,
# so it is a *secondary* activation lever. NOTE (measured on an 8 GiB RTX 4070,
# transformers 5.x): on the SO-101/LIBERO checkpoints the inference peak is set
# by the LM token-embedding step (~6 GiB resident + a ~1.5 GiB embedding `cat`),
# NOT the vision crops — so capping crops does not by itself change the peak, and
# transformers 5.x's *fast* MolmoAct2ImageProcessor does not honour ``max_crops``
# the way the slow one did. The actual 8 GiB enabler is the CUDA expandable-
# segments allocator (see :func:`_enable_expandable_segments`). This knob is kept
# for the slow-processor path and much larger frames. Precedence:
# ``vla.extra["image_max_crops"]`` → ``OPENRAL_MOLMOACT2_MAX_CROPS`` env →
# ``manifest.image_preprocessing.image_max_crops`` → ``None`` (checkpoint default 8).
_MAX_CROPS_ENV = "OPENRAL_MOLMOACT2_MAX_CROPS"

# MolmoAct2 NF4 is ~6 GiB resident (the bf16 vocab embeddings + vision tower
# dominate; the nf4 Linears are ~3.5 GiB) and peaks ~7.63 GiB during a chunk —
# right at the edge of an 8 GiB consumer card (a "8 GB" laptop GPU exposes only
# ~7.6 GiB usable). Without the CUDA caching allocator's expandable-segments mode
# the first forward's ~1.5 GiB embedding `cat` cannot be placed contiguously and
# OOMs even with several hundred MiB nominally free. expandable_segments fixes the
# fragmentation and the rollout fits reproducibly. bitsandbytes 4-bit + a tight
# card is the textbook case for this setting.
_CUDA_ALLOC_ENV = "PYTORCH_CUDA_ALLOC_CONF"
_EXPANDABLE_SEGMENTS = "expandable_segments:True"


def _enable_expandable_segments() -> None:
    """Enable the CUDA expandable-segments allocator for the MolmoAct2 load.

    Sets ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` via
    :meth:`os.environ.setdefault` (an operator export wins) **before the first
    CUDA allocation** in this process. The caching allocator reads the variable
    lazily on its first allocation rather than at ``import torch``, so setting it
    here — at the top of the molmoact2 build, ahead of the model's first
    ``.to(cuda)`` — takes effect even though torch is already imported. Verified
    on an 8 GiB RTX 4070: without it the SO-101 NF4 chunk OOMs at the embedding
    `cat`; with it the rollout peaks ~7.63 GiB and fits. No-op when the variable
    is already set or already contains ``expandable_segments``.
    """
    current = os.environ.get(_CUDA_ALLOC_ENV)
    if current is not None:
        if "expandable_segments" not in current:
            _log.info(
                "molmoact2_alloc_conf_preset",
                value=current,
                note="PYTORCH_CUDA_ALLOC_CONF already set; not adding expandable_segments.",
            )
        return
    os.environ[_CUDA_ALLOC_ENV] = _EXPANDABLE_SEGMENTS
    _log.info("molmoact2_expandable_segments_enabled", value=_EXPANDABLE_SEGMENTS)


def _molmoact2_phase(name: str, **fields: Any) -> Any:
    """Shortcut for ``phase_timer(name, prefix="molmoact2", gpu_mb=True, ...)``.

    Mirrors π0.5's ``_pi05_phase`` so a MolmoAct2 load emits the same
    ``molmoact2_<name>_{start,heartbeat,done}`` event shape with the GPU
    footprint reading that distinguishes a slow CPU allocation from a slow
    CUDA call. ``gpu_mb=True`` because every measurable phase
    (``from_pretrained`` → ``quantize_nf4`` → ``prequant_state_load`` →
    ``to_device``) sits adjacent to a GPU transfer.
    """
    return phase_timer(name, prefix="molmoact2", gpu_mb=True, log=_log, **fields)


def _import_transformers() -> tuple[Any, Any]:
    """Import transformers' ``AutoModelForImageTextToText`` + ``AutoProcessor``.

    MolmoAct2 is a transformers custom-code model (``auto_map`` →
    ``modeling_molmoact2.MolmoAct2ForConditionalGeneration``), loaded with
    ``trust_remote_code=True`` — it is NOT a lerobot policy, so this adapter
    does not go through :func:`lazy_import_lerobot`.

    Returns ``(AutoModelForImageTextToText, AutoProcessor)``. Both are returned
    untyped (``Any``) because transformers has no strict stubs in this
    workspace — same convention the lerobot adapters use for their inline
    policy-class imports.

    Raises:
        ROSConfigError: If transformers / torch are not installed.
    """
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - opt-in dependency
        raise ROSConfigError(
            "MolmoAct2 adapter requires transformers (custom-code model loaded "
            "via AutoModelForImageTextToText + trust_remote_code). Install with: "
            f"just sync --all-packages --group libero (underlying: {exc!r})"
        ) from exc
    return AutoModelForImageTextToText, AutoProcessor


@contextlib.contextmanager
def _hf_offline_if_cached(repo_id: str, probe_file: str = "config.json") -> Any:
    """Flip ``HF_HUB_OFFLINE`` on for the inner block when ``probe_file`` is cached.

    MolmoAct2 is a ``trust_remote_code`` custom-code model: every
    ``from_pretrained`` re-validates ``modeling_molmoact2.py`` (and the other
    auto-mapped sources) against the Hub with a HEAD round-trip, even on a
    fully warm cache — a stream of ``httpx`` INFO lines on each load. When the
    file the inner block is about to read is already in the local cache we flip
    ``huggingface_hub.constants.HF_HUB_OFFLINE`` (which transformers'
    ``is_offline_mode`` reads) so the cached read does zero HEADs. Cold caches
    stay online so the first read downloads exactly once. Mirrors the
    ``call_make_processors_cached_first`` offline-probe the lerobot adapters
    use for their tokenizer reloads.

    ``probe_file`` must name the file the inner block actually fetches: the
    ``from_pretrained`` load gate probes ``config.json`` (pulled by the load),
    while the ``predict_action`` wrap probes ``norm_stats.json`` (fetched lazily
    on the first inference). Gating on ``config.json`` there was a bug — on a
    first run the model load warms ``config.json`` but never ``norm_stats.json``,
    so the offline flag blocked the lazy norm-stats download with a
    ``LocalEntryNotFoundError`` that surfaced as "normalization stats file is
    missing".
    """
    import huggingface_hub.constants as _hc
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id, probe_file)
    if not isinstance(cached, str):
        # Cold (or sentinel _CACHED_NO_EXIST) — let the load talk to the Hub.
        yield
        return
    saved = _hc.HF_HUB_OFFLINE
    _hc.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        _hc.HF_HUB_OFFLINE = saved


def _strip_hf_uri(uri: str | None, *, field_name: str) -> str:
    """Strip the ``hf://`` prefix from a manifest URI, validating it is present."""
    value = (uri or "").strip()
    if not value.startswith("hf://"):
        raise ROSConfigError(
            f"MolmoAct2 adapter needs the rSkill manifest's {field_name} to be an "
            f"hf:// repo (e.g. 'hf://allenai/MolmoAct2-LIBERO'), got {value!r}."
        )
    return value[len("hf://") :]


def _split_repo_revision(repo: str) -> tuple[str, str | None]:
    """Split a ``owner/repo@revision`` string into ``(repo_id, revision)``.

    HuggingFace ``from_pretrained`` treats its first positional argument as a
    bare repo id; a ``@<sha>`` pin must be passed via the ``revision`` kwarg or
    it is silently ignored (security audit 2026-06, finding H5 — the pin was
    being concatenated onto the repo id and dropped). Returns ``revision=None``
    when no pin is present.

    Args:
        repo: A repo id, optionally suffixed with ``@<branch-or-sha>``.

    Returns:
        ``(repo_id, revision_or_None)``.
    """
    repo_id, _, revision = repo.partition("@")
    return repo_id, (revision or None)


# MolmoAct2 is a transformers *custom-code* model: ``from_pretrained`` executes
# ``modeling_*.py`` shipped in the repo (``trust_remote_code=True``). The repo id
# is manifest/operator-supplied and rSkill signature verification is not yet
# implemented (ADR-0006), so this is a remote-code-execution sink. Require an
# explicit operator acknowledgement, mirroring ``OPENRAL_ALLOW_UNSAFE_PICKLE``
# and ``OPENRAL_ALLOW_NONCOMMERCIAL`` (security audit 2026-06, finding C3).
_ALLOW_REMOTE_CODE_ENV = "OPENRAL_ALLOW_REMOTE_CODE"


def _require_remote_code_ack(source_repo: str, revision: str | None) -> None:
    """Refuse to load a ``trust_remote_code`` model unless the operator opts in.

    Raises unless ``OPENRAL_ALLOW_REMOTE_CODE=1`` acknowledges that the repo's
    custom code is trusted; logs a structured warning when enabled. Pinning a
    ``@<sha>`` revision is strongly recommended (and surfaced in the warning).

    Args:
        source_repo: The HF repo id whose custom code will be executed.
        revision: The pinned revision, or ``None`` if unpinned.

    Raises:
        ROSConfigError: If the acknowledgement env var is not set to ``"1"``.
    """
    if os.environ.get(_ALLOW_REMOTE_CODE_ENV, "0") != "1":
        raise ROSConfigError(
            f"MolmoAct2 loads custom code from '{source_repo}' via "
            "trust_remote_code=True, which executes arbitrary Python from the repo "
            "(remote-code-execution risk for untrusted or unverified weights). rSkill "
            "signature verification is not yet implemented (ADR-0006), so this is "
            f"blocked by default. To load a TRUSTED repo, set: export {_ALLOW_REMOTE_CODE_ENV}=1 "
            "(pin a revision SHA in the manifest's source_repo for reproducibility)."
        )
    if revision is None:
        _log.warning(
            "molmoact2.remote_code_unpinned",
            repo=source_repo,
            env=_ALLOW_REMOTE_CODE_ENV,
            note="Executing custom code from an UNPINNED repo; pin @<sha> in source_repo.",
        )
    else:
        _log.warning(
            "molmoact2.remote_code_trusted",
            repo=source_repo,
            revision=revision,
            env=_ALLOW_REMOTE_CODE_ENV,
            note="Executing custom code from the repo; ensure it is trusted.",
        )


@dataclass
class _MolmoAct2Adapter:
    """MolmoAct2 policy adapter — drives ``predict_action`` with chunk replay."""

    spec: VLASpec
    device: str
    _model: Any
    _processor: Any
    _torch: Any
    # source_repo for the lazy norm_stats.json fetch predict_action does on its
    # first call (outside the load path); used to keep that fetch offline on a
    # warm cache so inference emits no httpx HEAD stream either. None disables.
    _source_repo: str | None = None
    _norm_tag: str = _DEFAULT_NORM_TAG
    _num_steps: int = _DEFAULT_NUM_STEPS
    _n_action_steps: int | None = None
    _enable_cuda_graph: bool = False
    _flip_images_180: bool = True
    _flip_vertical: bool = False
    _state_dim: int | None = None
    # predict_action emits actions at the checkpoint's padded ``max_action_dim``
    # (32); the real embodiment action is the leading slice. Set from the
    # manifest's ``action_contract.dim`` (LIBERO Franka = 7: 6-DoF eef delta +
    # gripper) so the chunk is trimmed to what the env's actuators accept.
    _action_dim: int | None = None
    _camera_keys: tuple[str, ...] = field(default_factory=lambda: ("camera1", "camera2"))
    _last_input_frame: NDArray[np.uint8] | None = None
    # Wrap forward in ``torch.amp.autocast(device_type, dtype)`` when set — the
    # flow-matching expert materialises some intermediates in fp32 even with
    # bf16 / nf4 params, so autocast keeps mixed-precision matmuls valid.
    _autocast_dtype: Any = None
    # Closed-loop replay queue: predict_action returns a chunk of actions;
    # step() pops them one at a time and re-infers when empty.
    _action_queue: list[NDArray[np.float32]] = field(default_factory=list)

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        self._action_queue = []

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        if not self._action_queue:
            self._action_queue = self._predict_chunk(observation, instruction)
        return self._action_queue.pop(0)

    def close(self) -> None:
        if self.device.startswith("cuda"):
            with contextlib.suppress(Exception):
                self._torch.cuda.empty_cache()

    def _predict_chunk(
        self, observation: Observation, instruction: str
    ) -> list[NDArray[np.float32]]:
        """Run ``predict_action`` once and return the replayable action chunk."""
        torch = self._torch
        images = self._collect_images(observation)
        state = self._collect_state(observation)
        task = instruction or observation.get("task", "")

        device_type = self.device.split(":", 1)[0]
        if self._autocast_dtype is not None and device_type in {"cuda", "cpu"}:
            autocast_ctx: Any = torch.amp.autocast(
                device_type=device_type, dtype=self._autocast_dtype
            )
        else:
            autocast_ctx = contextlib.nullcontext()

        predict_kwargs: dict[str, Any] = {
            "processor": self._processor,
            "images": images,
            "task": task,
            "state": state,
            "norm_tag": self._norm_tag,
            "inference_action_mode": "continuous",
            "enable_depth_reasoning": False,
            "num_steps": self._num_steps,
            "normalize_language": True,
            "enable_cuda_graph": self._enable_cuda_graph,
        }
        if self._n_action_steps is not None:
            predict_kwargs["n_action_steps"] = self._n_action_steps

        # predict_action lazily fetches norm_stats.json from source_repo on its
        # first call; gate offline on norm_stats.json itself (not config.json) so
        # a cold-norm_stats first run stays online to download it, and only a
        # warm cache flips offline (no httpx HEAD stream).
        offline_ctx = (
            _hf_offline_if_cached(self._source_repo, probe_file="norm_stats.json")
            if self._source_repo is not None
            else contextlib.nullcontext()
        )
        with inference_span(kind="chunk"), torch.no_grad(), autocast_ctx, offline_ctx:
            out = self._model.predict_action(**predict_kwargs)

        actions = out.actions if hasattr(out, "actions") else out
        arr = np.asarray(actions.detach().to(torch.float32).cpu().numpy(), dtype=np.float32)
        # predict_action returns (B, n_action_steps, action_dim); drop the batch
        # dim and normalise a lone (action_dim,) to a 1-step chunk.
        if arr.ndim == _BATCHED_CHUNK_NDIM:
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr[None, :]
        # Trim the padded action width (max_action_dim, e.g. 32) to the
        # embodiment's real action dim so the env's actuators get a 7-vector.
        if self._action_dim is not None and arr.shape[-1] > self._action_dim:
            arr = arr[:, : self._action_dim]
        return [np.ascontiguousarray(row, dtype=np.float32) for row in arr]

    def _collect_images(self, observation: Observation) -> list[NDArray[np.uint8]]:
        """Build the ordered ``[agentview, wrist]`` image list predict_action wants.

        Camera order is significant (README: ``images`` must preserve
        ``[agentview_rgb, wrist_rgb]``); ``_camera_keys`` already encodes that
        order (LIBERO ``camera1`` = agentview/front, ``camera2`` = wrist).
        """
        from openral_sim.policies._video_capture import to_input_frame

        raw = observation.get("images", {})
        images: list[NDArray[np.uint8]] = []
        wrist_idx = len(self._camera_keys) - 1
        for i, cam_key in enumerate(self._camera_keys):
            img = raw.get(cam_key)
            if img is None:
                continue
            frame = self._orient(np.asarray(img))
            if i == wrist_idx:
                self._last_input_frame = to_input_frame(img, flip_180=self._flip_images_180)
            images.append(np.ascontiguousarray(frame, dtype=np.uint8))
        if not images:
            raise ROSConfigError(
                "MolmoAct2 adapter got no camera frames; expected observation "
                f"images for {self._camera_keys!r}, saw {list(raw)!r}."
            )
        return images

    def _orient(self, frame: NDArray[Any]) -> NDArray[Any]:
        """Apply the manifest's 180° / vertical flips to a HWC uint8 frame."""
        if self._flip_images_180:
            frame = frame[::-1, ::-1]
        if self._flip_vertical:
            frame = frame[::-1]
        return frame

    def _collect_state(self, observation: Observation) -> NDArray[np.float32]:
        """Build the raw robot-state vector (predict_action normalizes it itself)."""
        state = observation.get("state")
        if state is None:
            raise ROSConfigError(
                "MolmoAct2 `predict_action` requires a proprio `state`; the "
                "observation carried none."
            )
        state_np = np.asarray(state, dtype=np.float32)
        if self._state_dim is not None and state_np.shape[0] != self._state_dim:
            if state_np.shape[0] < self._state_dim:
                pad = np.zeros(self._state_dim - state_np.shape[0], dtype=np.float32)
                state_np = np.concatenate([state_np, pad])
            else:
                state_np = state_np[: self._state_dim]
        return state_np


def _resolve_max_crops(spec: VLASpec, manifest: Any | None) -> int | None:
    """Resolve the image-processor ``max_crops`` override, or ``None``.

    Precedence: ``vla.extra["image_max_crops"]`` → ``OPENRAL_MOLMOACT2_MAX_CROPS``
    env → ``manifest.image_preprocessing.image_max_crops`` (the per-checkpoint
    default the rSkill ships, e.g. the SO-101 skill pins 4 for an out-of-the-box
    8 GiB fit) → ``None`` (keep the checkpoint default of 8). See
    :data:`_MAX_CROPS_ENV`.
    """
    extra = spec.extra if hasattr(spec, "extra") else {}
    raw: Any = extra.get("image_max_crops")
    if raw is None:
        raw = os.environ.get(_MAX_CROPS_ENV)
    if raw is None:
        manifest_ip = getattr(manifest, "image_preprocessing", None) if manifest else None
        raw = getattr(manifest_ip, "image_max_crops", None) if manifest_ip else None
    if raw is None:
        return None
    crops = int(raw)
    if crops < 1:
        raise ROSConfigError(
            f"image_max_crops must be >= 1, got {crops!r} "
            f"(via vla.extra, {_MAX_CROPS_ENV}, or manifest.image_preprocessing)."
        )
    return crops


def _load_molmoact2_model(  # noqa: PLR0915  # reason: load-phase orchestration (fast meta-init vs slow from_pretrained / quantize / prequant / to_device) is naturally long
    *,
    torch: Any,
    auto_model_cls: Any,
    auto_processor_cls: Any,
    source_repo: str,
    revision: str | None,
    spec: VLASpec,
    device: str,
    dtype_str: str,
    max_crops: int | None,
) -> tuple[Any, Any, bool, Any]:
    """Load the processor + (optionally NF4-quantized) model onto ``device``.

    Returns ``(model, processor, use_nf4, torch_dtype)``. Split out of
    :func:`_build_molmoact2` so the build function stays under the statement
    cap; all phases stay wrapped in :func:`_molmoact2_phase`.
    """
    # Gate the custom-code execution before any from_pretrained touches the repo.
    _require_remote_code_ack(source_repo, revision)

    use_nf4 = dtype_str.lower() in {"nf4", "4bit", "int4"}
    torch_dtype = torch_dtype_for(torch, None if use_nf4 else dtype_str, device)

    # The processor + config resolve the trust_remote_code sources against the
    # Hub; on a warm cache that is pure HEAD-request noise. Load offline-if-cached
    # so a re-run emits no httpx HEAD stream.
    with _hf_offline_if_cached(source_repo), _molmoact2_phase("processor", repo=source_repo):
        processor = auto_processor_cls.from_pretrained(
            source_repo, revision=revision, trust_remote_code=True
        )
        if max_crops is not None:
            image_processor = getattr(processor, "image_processor", None)
            if image_processor is not None and hasattr(image_processor, "max_crops"):
                _log.info(
                    "molmoact2_max_crops_override",
                    max_crops=max_crops,
                    checkpoint_default=getattr(image_processor, "max_crops", None),
                )
                image_processor.max_crops = max_crops

    # Fast meta-init path: when the rSkill ships a prequantized nf4 pack, skip
    # the ~200 s ``from_pretrained`` that materialises the full ~5.5 B-param
    # bf16 backbone on CPU (per-tensor ``torch.empty`` allocation + init is the
    # bottleneck, not the safetensors read — same finding as π0.5). Build the
    # graph on the meta device (shape, no storage), rewrite Linears to
    # ``Linear4bit`` shells (also meta), materialise real storage exactly once
    # via ``to_empty``, then let the prequant state load supply every weight.
    # Mirrors the π0.5 nf4 fast path (see :mod:`openral_sim.policies.pi05`).
    # Unlike π0.5 we need no manual buffer reconstruction: MolmoAct2's
    # ``MolmoAct2RotaryEmbedding`` self-heals a meta/garbage ``inv_freq`` (it is
    # ``persistent=True`` → restored by the pack; the non-persistent cos/sin
    # caches are rebuilt lazily inside ``build_cache``).
    prequant_repo = detect_prequantized_nf4(spec) if use_nf4 and device.startswith("cuda") else None
    if prequant_repo is not None:
        from accelerate import init_empty_weights  # type: ignore[import-untyped]
        from transformers import AutoConfig

        fast_state_keys = peek_safetensors_keys(prequant_repo)
        with (
            _hf_offline_if_cached(source_repo),
            _molmoact2_phase("load_config", repo=source_repo),
        ):
            config = AutoConfig.from_pretrained(
                source_repo, revision=revision, trust_remote_code=True
            )
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with (
                _molmoact2_phase(
                    "init_empty_weights", repo=source_repo, prequant_repo=prequant_repo
                ),
                init_empty_weights(),
            ):
                model = auto_model_cls.from_config(config, trust_remote_code=True)
        finally:
            torch.set_default_dtype(prev_dtype)
        with _molmoact2_phase("quantize_nf4"):
            quantize_nf4_in_place(
                model, torch=torch, compute_dtype=torch.bfloat16, new_modules_on_meta=True
            )
        # Materialise meta params on CPU first, then a single move to the
        # target device — going straight to ``to_empty(device=cuda)`` would
        # allocate each Linear's full bf16 footprint on GPU before the nf4
        # pack shrinks it, OOMing an 8 GiB card (see π0.5's to_empty comment).
        with _molmoact2_phase("to_empty"):
            model.to_empty(device="cpu")
        with _molmoact2_phase("tie_weights"):
            tie_transformers_weights(model)
        with _molmoact2_phase("reset_parameters"):
            targeted_reset_parameters(model, covered_keys=fast_state_keys)
        with _molmoact2_phase("prequant_state_load"):
            load_prequantized_state_for_rskill(
                model, spec, torch=torch, log_event_prefix="molmoact2"
            )
        with _molmoact2_phase("to_device", device=device):
            model = model.to(device=device)
        model.eval()
        return model, processor, use_nf4, torch_dtype

    # Slow path: no prequant pack (or bf16 / non-CUDA). Materialise the full
    # bf16 backbone on CPU, then on-the-fly quantize before the device move —
    # instantiating the ~5.5 B backbone on CUDA in fp32 would OOM an 8 GiB card
    # before the nf4 rewrite ever runs.
    with (
        _hf_offline_if_cached(source_repo),
        _molmoact2_phase("from_pretrained", repo=source_repo, dtype=dtype_str),
    ):
        model = auto_model_cls.from_pretrained(
            source_repo,
            revision=revision,
            trust_remote_code=True,
            dtype=torch_dtype if not use_nf4 else torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    if not use_nf4:
        with _molmoact2_phase("cast_and_to_device", device=device, dtype=dtype_str):
            model = model.to(dtype=torch_dtype).to(device=device)
        model.eval()
        return model, processor, use_nf4, torch_dtype

    if not device.startswith("cuda"):
        raise ROSConfigError(
            "nf4 quantization for MolmoAct2 requires a CUDA device; got "
            f"device={device!r}. Set vla.extra.dtype='bf16' to load on CPU/MPS "
            "(needs >=12 GiB)."
        )
    # Pre-cast every fp32 leaf to bf16 before quantization so the surviving
    # non-Linear bits (embeddings, norms, biases, small heads kept in compute
    # dtype) are uniformly bf16 — otherwise the forward trips "expected mat1
    # and mat2 to have the same dtype: float != BFloat16".
    with _molmoact2_phase("precast_bf16"):
        for p in model.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
        for b in model.buffers():
            if b.dtype == torch.float32:
                b.data = b.data.to(torch.bfloat16)
    with _molmoact2_phase("quantize_nf4"):
        quantize_nf4_in_place(model, torch=torch, compute_dtype=torch.bfloat16)
    # Overlay the prequantized nf4 pack shipped at the rSkill's weights_uri so
    # the ~25 s on-line bf16→nf4 conversion on ``.to(cuda)`` is replaced by a
    # fast state-dict load. Silent no-op + fallback if the pack is absent or its
    # keys don't line up (then ``.to(cuda)`` re-packs).
    with _molmoact2_phase("prequant_state_load"):
        load_prequantized_state_for_rskill(model, spec, torch=torch, log_event_prefix="molmoact2")
    with _molmoact2_phase("to_device", device=device):
        model = model.to(device=device)
    model.eval()
    return model, processor, use_nf4, torch_dtype


@POLICIES.register("molmoact2")
def _build_molmoact2(env_cfg: Any) -> _MolmoAct2Adapter:
    """Load a MolmoAct2 LIBERO finetune as a transformers custom-code model."""
    spec = env_cfg.vla
    device = resolve_device(spec)

    # Enable the CUDA expandable-segments allocator before any CUDA work so the
    # ~7.6 GiB inference peak fits an 8 GiB card (see _enable_expandable_segments).
    if device.startswith("cuda"):
        _enable_expandable_segments()

    with _molmoact2_phase("imports"):
        import torch

        auto_model_cls, auto_processor_cls = _import_transformers()

    # The rSkill manifest is the source of truth. ``weights_uri`` points at the
    # NF4 prequant repo (used for the fast prequant overlay); ``source_repo``
    # points at the upstream bf16 checkpoint that carries the model graph,
    # processor, custom code and norm_stats.
    resolve_rskill_repo_id(spec.weights_uri, adapter_name="MolmoAct2")  # validates rSkill reference
    manifest = load_manifest_for_spec(spec)
    if manifest is None:
        raise ROSConfigError(
            "MolmoAct2 adapter requires a bare rSkill reference as weights_uri "
            f"resolving to a manifest; got {spec.weights_uri!r}."
        )
    source_repo, source_revision = _split_repo_revision(
        _strip_hf_uri(manifest.source_repo, field_name="source_repo")
    )

    dtype_str = manifest_dtype(spec, manifest=manifest) or default_dtype_for_device(device)
    model, processor, use_nf4, torch_dtype = _load_molmoact2_model(
        torch=torch,
        auto_model_cls=auto_model_cls,
        auto_processor_cls=auto_processor_cls,
        source_repo=source_repo,
        revision=source_revision,
        spec=spec,
        device=device,
        dtype_str=dtype_str,
        max_crops=_resolve_max_crops(spec, manifest),
    )

    ip = resolve_image_preprocessing(manifest, spec.extra)
    state_dim = resolve_state_dim(manifest, spec.extra)
    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(manifest, spec.extra, scene_cameras=scene_cameras)

    # Replay cadence + solver knobs: vla.extra overrides win, then the manifest,
    # then the checkpoint/README defaults.
    extra = spec.extra if hasattr(spec, "extra") else {}
    n_action_steps = extra.get("n_action_steps", manifest.n_action_steps)
    # The checkpoint hard-caps the chunk it will emit at config.max_action_horizon
    # (the LIBERO checkpoint = 10); predict_action raises if asked for more. Clamp
    # so a stale manifest / CLI value degrades to the full chunk instead of erroring.
    max_horizon = int(getattr(model.config, "max_action_horizon", 0) or 0)
    if n_action_steps and max_horizon and int(n_action_steps) > max_horizon:
        _log.warning(
            "molmoact2_n_action_steps_clamped",
            requested=int(n_action_steps),
            max_action_horizon=max_horizon,
        )
        n_action_steps = max_horizon
    num_steps = int(extra.get("num_steps", _DEFAULT_NUM_STEPS))
    # Precedence: vla.extra → manifest.image_preprocessing.norm_tag → default
    manifest_norm_tag = None
    if ip.norm_tag is not None:
        manifest_norm_tag = ip.norm_tag
    norm_tag = str(extra.get("norm_tag", manifest_norm_tag or _DEFAULT_NORM_TAG))
    # CUDA graphs default OFF: they pin static buffers (extra VRAM) and don't
    # compose cleanly with bnb's 4-bit dequant on an 8 GiB card.
    enable_cuda_graph = bool(extra.get("enable_cuda_graph", False))

    reduced_precision = use_nf4 or torch_dtype == torch.bfloat16
    autocast_dtype = (
        torch.bfloat16
        if reduced_precision
        else (torch.float16 if torch_dtype == torch.float16 else None)
    )

    return _MolmoAct2Adapter(
        spec=spec,
        device=device,
        _model=model,
        _processor=processor,
        _torch=torch,
        _source_repo=source_repo,
        _norm_tag=norm_tag,
        _num_steps=num_steps,
        _n_action_steps=int(n_action_steps) if n_action_steps else None,
        _enable_cuda_graph=enable_cuda_graph,
        _flip_images_180=ip.flip_180,
        _flip_vertical=ip.flip_vertical,
        _state_dim=state_dim,
        _action_dim=(
            manifest.action_contract.dim if manifest.action_contract is not None else None
        ),
        _camera_keys=cam_keys,
        _autocast_dtype=autocast_dtype,
    )
