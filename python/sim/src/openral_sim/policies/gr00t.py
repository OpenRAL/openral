"""NVIDIA Isaac GR00T N1.7 policy adapter — in-process lerobot backend.

Historically GR00T ran out-of-process in a Python-3.10 ZMQ sidecar (flash-attn +
Isaac-GR00T pin 3.10). lerobot 0.6.0 ships a native, in-process ``GrootPolicy``
(``lerobot.policies.groot``) that loads GR00T **N1.7** under the workspace's
Python 3.12 — the Cosmos-Reason2 / Qwen3-VL backbone is a stock
``Qwen3VLForConditionalGeneration`` available in ``transformers>=5.4``, so the
whole py3.10 rationale disappears for N1.7. This adapter therefore mirrors the
in-process :mod:`openral_sim.policies.smolvla` adapter instead of forking a
sidecar.

RLDX-1 (a GR00T-**N1.5** finetune) still runs in its ZMQ sidecar via the
``rldx`` adapter — native lerobot rejects N1.5 — so
:class:`openral_sim.policies.rldx._Gr00tFamilySidecarAdapter` is untouched.

In-process NF4
--------------
Native ``GrootPolicy`` has no quantization knob and loads the ~3 B model in
fp32 params / bf16 compute (~6 GB), which will not co-fit an 8 GB card. The
transformers ``BitsAndBytesConfig`` / ``device_map`` path is unavailable here —
it requires ``accelerate``, which the workspace venv does not carry (it only
lived in the retired py3.10 sidecar). So we reuse OpenRAL's accelerate-free
:func:`openral_sim._quantization.quantize_nf4_in_place` (the same NF4 mechanism
pi05 / molmoact2 use): after a normal CPU load we rewrite the Qwen3-VL
backbone's large ``nn.Linear`` layers into ``bnb.nn.Linear4bit`` shells, then
``policy.to(cuda)`` packs them to NF4. Only the backbone is quantized; the small
diffusion action head stays in bf16 — this preserves action quality and
side-steps the GR00T DiT ``TimestepEncoder`` uint8 bug (bug (b) below).

The official 2026 BEHAVIOR-1K checkpoint is a deliberate exception to this
native path: its organizer runtime is pinned to ``wensi-ai/Isaac-GR00T`` under
Python 3.10, so a manifest with
``policy_extras.implementation=behavior_b1k_sidecar`` dispatches to
:mod:`openral_sim.policies.behavior_groot` before the lerobot loader runs.

Embodiment mapping
------------------
The rSkill manifest declares its OpenRAL robot tag (e.g. ``franka_panda`` /
``so101_follower``). GR00T's ``embodiment_tag`` is its OWN namespace, carried in
``policy_extras.embodiment_tag`` — ``libero_sim`` for the LIBERO checkpoint,
``new_embodiment`` for the SO-101 fruit checkpoint. Mapping it explicitly is
load-bearing: it selects that embodiment's modality config (image views,
state/action layout) AND the action-decode transform. Feed the wrong tag and
eval scores 0 %.

The per-embodiment I/O contract — state width, action width, and GR00T video
modality keys — is read from the rSkill rather than hard-coded: ``state_dim`` /
``action_dim`` from ``state_contract.dim`` / ``action_contract.dim`` and the
image keys from ``policy_extras.image_modality_keys`` (``image``/``wrist_image``
for LIBERO, ``front``/``wrist`` for SO-101). The LIBERO values remain the
fallback when a manifest omits them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_rskill._diagnostics import phase_timer
from openral_rskill._vla_core import (
    build_chunk_executor,
    release_torch_modules,
    resolve_camera_keys,
    resolve_device,
    resolve_image_preprocessing,
    resolve_rskill_repo_revision,
)

from openral_sim.policies._policy_loading import lazy_import_lerobot, load_manifest_for_spec
from openral_sim.registry import POLICIES

_log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.policy import PolicyAdapter
    from openral_sim.rollout import Observation

# The gr00t-n17-libero rSkill targets nvidia/GR00T-N1.7-LIBERO. GR00T's own
# embodiment namespace calls the LIBERO-Panda embodiment ``libero_sim`` (id 2 in
# the checkpoint's embodiment_id.json); its modality config declares the
# ``image`` + ``wrist_image`` views and a 7-D ``x,y,z,roll,pitch,yaw,gripper``
# state/action. Overridable via ``OPENRAL_GR00T_EMBODIMENT_TAG`` /
# ``vla.extra.embodiment_tag`` for other GR00T checkpoints.
_GR00T_DEFAULT_EMBODIMENT_TAG = "libero_sim"

# GR00T libero_sim consumes an 8-D proprio state — the checkpoint's modality
# stats are x,y,z,roll,pitch,yaw (1-D each) + gripper (2-D), 8-D total, which
# matches the OpenRAL LIBERO backend's ``eef_pos(3) ‖ axisangle(3) ‖
# gripper_qpos(2)`` vector one-for-one. The action is 7-D (gripper collapses to
# a single scalar in the checkpoint's action modality).
_GR00T_LIBERO_STATE_DIM = 8
_GR00T_LIBERO_ACTION_DIM = 7

# GR00T's libero_sim video modality keys (from the checkpoint processor_config).
# The lerobot GR00T pack step reads ``observation.images.<modality_key>``.
_GR00T_LIBERO_IMAGE_KEYS = ("image", "wrist_image")

# predict_action_chunk returns a rank-3 ``(B, T, action_dim)`` tensor.
_CHUNK_RANK_BTD = 3


def _groot_phase(name: str, **fields: Any) -> Any:
    """``phase_timer`` shortcut mirroring the smolvla adapter's phase logging."""
    return phase_timer(name, prefix="gr00t", log=_log, **fields)


def _env_bool(name: str, default: bool) -> bool:
    """Parse a permissive bool env var (``1`` / ``true`` / ``yes`` / ``on``)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _import_real_groot_policy() -> Any:
    """Return the real lerobot ``GrootPolicy``, evicting the 0.5.x compat stub.

    ``openral_rskill._lerobot_compat`` installs an empty stub for
    ``lerobot.policies.groot.modeling_groot`` (a workaround for lerobot
    0.5.0/0.5.1's GR00T-N1.5 config crashing to import under transformers 5.x).
    lerobot 0.6.0 ships a real, importable ``GrootPolicy`` (GR00T N1.7), so the
    stub is now stale and would shadow it with a class that has no
    ``from_pretrained``. Evict the stub and import the real module.

    ``_lerobot_compat`` is now self-disabling on lerobot >= 0.6.0 (it only stubs
    when the real ``modeling_groot`` genuinely fails to import), so on a healthy
    install there is nothing to evict — this stays as belt-and-suspenders for a
    stale interpreter that imported an older lerobot first.
    """
    import importlib
    import sys

    name = "lerobot.policies.groot.modeling_groot"
    stubbed = sys.modules.get(name)
    stub_policy = getattr(stubbed, "GrootPolicy", None) if stubbed is not None else None
    if stubbed is not None and not hasattr(stub_policy, "from_pretrained"):
        del sys.modules[name]
    module = importlib.import_module(name)
    return module.GrootPolicy


def _quantize_groot_nf4(policy: Any, torch: Any, *, scope: str) -> None:
    """NF4-quantize the GR00T model in place (accelerate-free).

    ``scope`` selects how much of the model is packed to 4-bit:

    - ``"backbone"`` (default): only ``_groot_model.backbone`` (the ~2 B Qwen3-VL
      VLM). The diffusion action head stays bf16. Enough for the LIBERO N1.7
      checkpoint (16-layer DiT head) to fit 8 GB.
    - ``"model"``: the whole ``_groot_model`` — backbone **and** the diffusion
      action head's large ``Linear`` blocks. Needed for checkpoints with a
      heavier head (e.g. the SO-101 fruit checkpoint's 32-layer DiT), which
      overshoot 8 GB by ~300 MiB when the head stays bf16.

    Two GR00T-specific bnb hazards are avoided structurally in both scopes:

    (a) ``.to(dtype=...)`` on a bnb model raises ``"cannot cast a bitsandbytes
        model in a new dtype"``. We never dtype-cast the quantized model — the
        rewrite keeps bf16 placeholder weights until a pure device-move
        ``policy.to(cuda)`` packs them, and ``model_params_fp32=False`` stops
        ``GrootPolicy`` from fp32-casting the params before we quantize.

    (b) The GR00T DiT ``TimestepEncoder.forward`` infers its compute dtype from
        ``next(self.parameters()).dtype``; a 4bit-packed encoder makes that
        ``uint8`` and the following SiLU crashes (``silu not implemented for
        'Byte'``). ``quantize_nf4_in_place`` only rewrites ``Linear`` modules with
        ``>=4M`` weight elements, so the small ``TimestepEncoder`` MLP (its
        Linears are ~0.26–1 M params) stays bf16 even under ``scope="model"`` —
        the encoder reads bf16, not uint8, and the bug cannot recur.
    """
    from openral_sim._quantization import quantize_nf4_in_place

    target = policy._groot_model if scope == "model" else policy._groot_model.backbone
    quantize_nf4_in_place(
        target,
        torch=torch,
        compute_dtype=torch.bfloat16,
    )


def _patch_groot_dtype_property(torch: Any) -> None:
    """Make ``GR00TN17.dtype`` report the first *floating* param dtype.

    ``GR00TN17`` overrides ``dtype`` as ``next(iter(self.parameters())).dtype``.
    Once the Qwen3-VL backbone is NF4-quantized its first parameter is a
    ``Params4bit`` whose ``.dtype`` is ``torch.uint8``; ``GR00TN17.prepare_input``
    then casts the float image/state inputs to ``uint8`` and inference produces
    garbage. Returning the first *floating-point* parameter's dtype (bf16 for the
    unquantized action head / skipped embeddings) restores the intended compute
    dtype. This is a strict no-op for a non-quantized load (the first parameter
    is already floating), so we apply it unconditionally and leave it in place —
    it only ever changes behaviour for the quantized GR00T model, which no other
    adapter loads in-process.
    """
    from lerobot.policies.groot import groot_n1_7 as _gm

    def _first_float_dtype(self: Any) -> Any:
        for param in self.parameters():
            if param.is_floating_point():
                return param.dtype
        return torch.bfloat16

    _gm.GR00TN17.dtype = property(_first_float_dtype)


# GR00T N1.7's VLM backbone is a stock Qwen3-VL-2B-Instruct (lerobot's own
# hard-coded backbone config is literally "a Qwen3-VL-2B-Instruct layout").
# lerobot builds the model from that hard-coded config (no fetch), but its
# *processor* factory hard-codes ``AutoTokenizer/ImageProcessor/VideoProcessor
# .from_pretrained("nvidia/Cosmos-Reason2-2B")`` — a **gated** NVIDIA repo — so
# loading a GR00T-N1.7 checkpoint 401s at processor build even though every
# weight is local. The tokenizer/processors are byte-identical to the public,
# ungated ``Qwen/Qwen3-VL-2B-Instruct`` (same vocab/merges; matching vision +
# eos special-token ids), so we point the processor there. This keeps the
# ``nvidia_open_model`` rSkill loadable with no NVIDIA gate and no HF login.
# Override with OPENRAL_GR00T_BACKBONE_MODEL=nvidia/Cosmos-Reason2-2B to use the
# exact upstream repo (requires accepting its license + an authenticated token).
_GR00T_PUBLIC_BACKBONE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def _redirect_groot_backbone_processor() -> None:
    """Point lerobot's GR00T-N1.7 processor at the ungated Qwen3-VL backbone.

    lerobot resolves ``GROOT_N1_7_BACKBONE_MODEL`` as a module global at
    processor-build time (``processor_groot.py`` passes it explicitly into the
    VLM encode step), so rebinding the name on that module redirects the
    tokenizer/image/video loads without touching the model-construction path
    (which uses the checkpoint's own hard-coded Cosmos config, no fetch).
    """
    backbone = os.environ.get("OPENRAL_GR00T_BACKBONE_MODEL") or _GR00T_PUBLIC_BACKBONE_MODEL
    from lerobot.policies.groot import processor_groot as _pg

    _pg.GROOT_N1_7_BACKBONE_MODEL = backbone


def _to_groot_state(state: Any, expected_dim: int) -> NDArray[np.float32]:
    """Return the GR00T proprio vector as flat float32, width-checked.

    ``expected_dim`` comes from the rSkill's ``state_contract.dim``: 8 for the
    LIBERO eef-pose vector (``eef_pos(3) ‖ axisangle(3) ‖ gripper_qpos(2)``),
    6 for the SO-101 ``single_arm(5) ‖ gripper(1)`` joint vector. GR00T pads to
    ``max_state_dim`` internally, so the adapter only enforces the embodiment's
    declared width — no re-slicing.
    """
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ROSConfigError(
            f"gr00t adapter expects a {expected_dim}-D `state` (from the rSkill's "
            f"state_contract.dim), got {arr.shape[0]}-D. The scene adapter must "
            "populate the checkpoint's proprio vector for this embodiment."
        )
    return arr


@dataclass
class _GrootAdapter:
    """In-process lerobot ``GrootPolicy`` adapter (returns per-step actions).

    GR00T N1.7's ``libero_sim`` embodiment uses **native relative actions**: the
    ``GrootN17ActionDecodeStep`` postprocessor decodes a chunk against the raw
    state cached by the pack step at prediction time, and refuses to decode a
    single popped action one step at a time. So — unlike the smolvla adapter's
    per-step ``select_action`` — we predict the **full chunk**, decode it while
    the pack-step state is fresh, queue the decoded absolute actions, and pop one
    per ``step`` (replanning every ``_replan_steps``, GR00T's 8-of-16 execution
    horizon for ``libero_sim``). This mirrors the rldx adapter's chunk-replay.
    """

    spec: VLASpec
    device: str
    _policy: Any
    _preprocessor: Any
    _postprocessor: Any
    _torch: Any
    _replan_steps: int = 8
    _flip_images_180: bool = False
    _flip_vertical: bool = False
    _camera_keys: tuple[str, ...] = field(default_factory=lambda: ("camera1", "camera2"))
    # GR00T modality video keys aligned positionally with ``_camera_keys``.
    _image_input_keys: tuple[str, ...] = _GR00T_LIBERO_IMAGE_KEYS
    # Declared proprio width (state_contract.dim): 8 = LIBERO, 6 = SO-101.
    _state_dim: int = _GR00T_LIBERO_STATE_DIM
    _chunk_executor: Any = None
    _last_input_frame: NDArray[np.uint8] | None = None

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        if self._chunk_executor is not None:
            self._chunk_executor.reset()
        elif hasattr(self._policy, "reset"):
            self._policy.reset()

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        if self._chunk_executor is not None:
            action = self._chunk_executor.select_action(lambda: (observation, instruction))
            return np.asarray(action, dtype=np.float32)
        return np.asarray(self._chunk_forward((observation, instruction))[0], dtype=np.float32)

    def _chunk_forward(self, payload: tuple[Observation, str]) -> list[NDArray[np.float32]]:
        """Predict and decode one execution chunk."""
        observation, instruction = payload
        batch = self._build_batch(observation, instruction)
        batch = self._preprocessor(batch)
        chunk = self._policy.predict_action_chunk(batch)
        chunk = self._postprocessor(chunk)
        chunk_np = chunk.detach().to("cpu").float().numpy()
        # predict_action_chunk returns (B=1, T, action_dim); squeeze the batch.
        if chunk_np.ndim == _CHUNK_RANK_BTD and chunk_np.shape[0] == 1:
            chunk_np = chunk_np[0]
        elif chunk_np.ndim == 1:
            chunk_np = chunk_np[None, :]
        return [np.asarray(action, dtype=np.float32) for action in chunk_np[: self._replan_steps]]

    def close(self) -> None:
        """Drop the loaded modules, then reclaim their VRAM.

        Order matters: ``empty_cache()`` only returns already-free blocks,
        so flushing while this adapter still holds the policy frees nothing.
        See :func:`openral_rskill._vla_core.release_torch_modules`.
        """
        if self._chunk_executor is not None:
            self._chunk_executor.stop()
            self._chunk_executor = None
        release_torch_modules(
            self,
            "_policy",
            "_preprocessor",
            "_postprocessor",
            device=self.device,
            torch=self._torch,
        )

    def _build_batch(self, observation: Observation, instruction: str) -> dict[str, Any]:
        """Assemble the lerobot GR00T batch (``observation.images.*`` + state + task).

        Images are passed as float ``CHW`` in ``[0, 1]`` (the lerobot
        convention the GR00T pack/VLM steps consume); the 180° flip that GR00T's
        own LIBERO eval env applies to both cameras is honoured here via the
        rSkill ``image_preprocessing.flip_180`` flag.
        """
        torch = self._torch
        batch: dict[str, Any] = {"task": instruction or observation.get("task", "")}

        images = observation.get("images", {})
        from openral_sim.policies._video_capture import tile_input_frames, to_input_frame

        preview_frames: list[NDArray[np.uint8]] = []
        for cam_key, modality_key in zip(self._camera_keys, self._image_input_keys, strict=False):
            img = images.get(cam_key)
            if img is None:
                continue
            preview = to_input_frame(img, flip_180=self._flip_images_180)
            if preview is not None:
                preview_frames.append(preview)
            t = torch.tensor(np.asarray(img), dtype=torch.float32).div(255.0).permute(2, 0, 1)
            if self._flip_images_180:
                t = torch.flip(t, dims=[1, 2])
            if self._flip_vertical:
                t = torch.flip(t, dims=[1])
            batch[f"observation.images.{modality_key}"] = t.to(self.device)
        self._last_input_frame = tile_input_frames(preview_frames)

        state = observation.get("state")
        if state is None:
            raise ROSConfigError(
                "gr00t adapter requires a non-empty LIBERO proprio `state` on the "
                "observation; the scene adapter must populate it."
            )
        state_np = _to_groot_state(state, self._state_dim)
        batch["observation.state"] = torch.from_numpy(state_np).to(self.device)
        return batch


def _build_groot_config(
    *,
    local_path: str,
    embodiment_tag: str,
    quantize: bool,
    image_keys: tuple[str, ...],
    state_dim: int,
    action_dim: int,
) -> Any:
    """Construct a ``GrootConfig`` pinned to the LIBERO embodiment + I/O dims.

    ``embodiment_tag`` MUST be set at construction (not mutated afterwards): the
    ``libero_sim`` gripper-flip / action-decode transform is resolved in
    ``GrootConfig.__post_init__``. Explicit ``input_features`` / ``output_features``
    pin the action head to the real 7-D LIBERO action instead of the 132-D
    padded default.
    """
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.utils.constants import ACTION, OBS_STATE

    input_features: dict[str, Any] = {
        f"observation.images.{key}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        for key in image_keys
    }
    input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}

    return GrootConfig(
        base_model_path=local_path,
        embodiment_tag=embodiment_tag,
        # Quantized params cannot be fp32-cast (see _groot_nf4_config bug (a)).
        model_params_fp32=not quantize,
        input_features=input_features,
        output_features=output_features,
    )


@POLICIES.register("gr00t")
def _build_gr00t(env_cfg: Any) -> PolicyAdapter:  # noqa: PLR0915  # reason: staged loader
    """Load the in-process lerobot ``GrootPolicy`` (GR00T N1.7) backend.

    YAML knobs (via ``vla.extra``): ``device``, ``embodiment_tag``,
    ``quantization`` (``nf4`` default / ``none``), ``camera_keys``.
    Environment overrides: ``OPENRAL_GR00T_EMBODIMENT_TAG``,
    ``OPENRAL_GR00T_QUANTIZATION``.
    """
    spec = env_cfg.vla
    device = resolve_device(spec)
    extra = dict(spec.extra or {})

    manifest = load_manifest_for_spec(spec)
    if manifest is None:
        raise ROSConfigError(
            "gr00t adapter requires a bare rSkill reference as weights_uri so the "
            "loader can resolve the GR00T checkpoint + its embodiment contract."
        )
    # Merge the manifest's policy_extras under any scene/CLI spec.extra overrides
    # (spec.extra wins) so a checkpoint's embodiment_tag / image_modality_keys
    # travel in its rSkill without every scene having to repeat them.
    policy_extras = dict(getattr(manifest, "policy_extras", {}) or {})
    extra = {**policy_extras, **extra}
    if extra.get("implementation") == "behavior_b1k_sidecar":
        from openral_sim.policies.behavior_groot import (
            build_behavior_groot_policy,
        )

        return build_behavior_groot_policy(env_cfg, manifest, extra)

    repo_id, revision = resolve_rskill_repo_revision(spec.weights_uri, adapter_name="GR00T")
    embodiment_tag = str(
        os.environ.get("OPENRAL_GR00T_EMBODIMENT_TAG")
        or extra.get("embodiment_tag", _GR00T_DEFAULT_EMBODIMENT_TAG)
    )

    # I/O contract per embodiment. LIBERO (8-D state / 7-D action /
    # image,wrist_image) is the fallback; SO-101 (6-D / 6-D / front,wrist)
    # and any other GR00T checkpoint come from the manifest contracts +
    # policy_extras.image_modality_keys.
    img_keys_raw = extra.get("image_modality_keys")
    image_input_keys = (
        tuple(str(k) for k in img_keys_raw)
        if isinstance(img_keys_raw, (list, tuple)) and img_keys_raw
        else _GR00T_LIBERO_IMAGE_KEYS
    )
    state_dim = int(getattr(manifest.state_contract, "dim", 0) or _GR00T_LIBERO_STATE_DIM)
    action_dim = int(getattr(manifest.action_contract, "dim", 0) or _GR00T_LIBERO_ACTION_DIM)
    quantization = str(
        os.environ.get("OPENRAL_GR00T_QUANTIZATION") or extra.get("quantization", "nf4")
    ).lower()
    quantize = quantization not in {"none", "", "fp16", "bf16", "fp32"}
    # How much of the model to pack to NF4 — "backbone" (default; LIBERO fits
    # 8 GB) or "model" (backbone + DiT head; needed for heavier-head checkpoints
    # like SO-101 fruit that overshoot 8 GB when the head stays bf16).
    quantize_scope = str(
        os.environ.get("OPENRAL_GR00T_QUANTIZE_SCOPE") or extra.get("quantize_scope", "backbone")
    ).lower()

    ip = resolve_image_preprocessing(manifest, spec.extra)
    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    cam_keys = resolve_camera_keys(manifest, spec.extra, scene_cameras=scene_cameras)

    with _groot_phase("imports"):
        torch, make_processors = lazy_import_lerobot("GR00T")
        from huggingface_hub import snapshot_download

        GrootPolicy = _import_real_groot_policy()  # noqa: N806 — class object, not a value

    # The GR00T processor factory only recognises a *local* raw-checkpoint dir
    # (is_raw_groot_n1_7_checkpoint reads config.json off disk); an HF id would
    # be treated as a serialized-pipeline path and fail. Snapshot once and drive
    # both the model + processor loads off the local path.
    with _groot_phase("snapshot", repo=repo_id):
        # Skip training-only artifacts: a raw GR00T training checkpoint (e.g.
        # aaronsu11/GR00T-N1.7-3B-SO101-FruitPicking) ships an ~13 GB optimizer.pt
        # plus rng/scheduler/trainer state that inference never touches — pulling
        # them ~doubles the download and can stall the load. The inference set
        # (model shards + index + config + embodiment_id + experiment_cfg +
        # processor_config + statistics) is untouched.
        local_path = snapshot_download(
            repo_id,
            revision=revision,
            ignore_patterns=[
                "optimizer.pt",
                "rng_state*.pth",
                "scheduler.pt",
                "trainer_state.json",
                "training_args.bin",
                "wandb_config.json",
            ],
        )

    _patch_groot_dtype_property(torch)
    _redirect_groot_backbone_processor()

    config = _build_groot_config(
        local_path=local_path,
        embodiment_tag=embodiment_tag,
        quantize=quantize,
        image_keys=image_input_keys,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Load on CPU (no device_map / quantization_config → no accelerate needed),
    # NF4-rewrite the backbone in place, then a pure device-move packs the
    # Linear4bit shells and lands the whole policy on the GPU.
    prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with _groot_phase("from_pretrained", repo=repo_id, quantization=quantization):
            policy = GrootPolicy.from_pretrained(local_path, config=config)
        if quantize:
            with _groot_phase("quantize_nf4", scope=quantize_scope):
                _quantize_groot_nf4(policy, torch, scope=quantize_scope)
        with _groot_phase("to_device", device=device):
            policy = policy.to(device)
    finally:
        torch.set_default_dtype(prev_default_dtype)
    policy.eval()

    with _groot_phase("make_processors"):
        preprocessor, postprocessor = make_processors(policy.config, pretrained_path=local_path)

    # Execution horizon: GR00T resolves this to min(n_action_steps, checkpoint
    # action_horizon, embodiment exec horizon) = 8 for libero_sim. Replay that
    # many decoded actions per inference before replanning.
    replan_steps = int(getattr(policy, "_action_queue_steps", 8)) or 8

    adapter = _GrootAdapter(
        spec=spec,
        device=device,
        _policy=policy,
        _preprocessor=preprocessor,
        _postprocessor=postprocessor,
        _torch=torch,
        _replan_steps=replan_steps,
        _flip_images_180=ip.flip_180,
        _flip_vertical=ip.flip_vertical,
        _camera_keys=tuple(cam_keys),
        _image_input_keys=image_input_keys,
        _state_dim=state_dim,
    )
    adapter._chunk_executor = build_chunk_executor(
        spec.extra,
        policy=policy,
        chunk_fn=adapter._chunk_forward,
        chunk_size=replan_steps,
        adapter_name="gr00t",
    )
    return adapter
