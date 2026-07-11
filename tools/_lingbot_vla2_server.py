"""LingBot-VLA 2.0 inference server — runs inside the sidecar venv (no openral import).

This is the **server side** of the LingBot-VLA 2.0 sidecar: it loads the upstream
``LingbotVLAv2Server`` (NF4 Qwen3-VL backbone + bf16 sparse-MoE action expert) from
Robbyant's ``lingbotvla`` package (https://github.com/robbyant/lingbot-vla-v2,
Apache-2.0 code + weights) and answers ``ping`` / ``reset`` / ``get_action`` /
``close`` over ZMQ REQ/REP framed by msgpack — the same ndarray wire the
:class:`openral_sim.sidecar.SidecarClient` speaks. It is ``os.execvpe``-d by the
boot helper :mod:`tools.lingbot_vla2_sidecar` *after* the repo checkout + torch-2.8
venv are provisioned, so it only ever runs under the sidecar interpreter and never
imports ``openral_*`` (that stack pins torch>=2.9 / transformers>=5, incompatible
with the upstream torch==2.8.0 / transformers==4.57.3 pins — CLAUDE.md §3).

Wire protocol (msgpack ``{"endpoint","data"}`` in, dict out; ndarrays via the
``__ndarray__`` / ``np.save`` sentinel that mirrors
``openral_sim.sidecar.encode_ndarray``)::

    ping        -> {"ok": True, "model": <id>, "robo_name": <robo>}
    reset       -> {"ok": True}                       data: {"robo_name": <robo>}
    get_action  -> {"action": (chunk, 14) float32,    data: {"observation": {
                    "action_keys": [...]}                "images": {cam_high,cam_left_wrist,cam_right_wrist},
                                                          "state": (14,), "task": <str>}}
    close       -> {"ok": True}  (stops the server)

The 6.38 B model does not fit an 8 GB card at bf16 (12.8 GB), so the Qwen3-VL
backbone is NF4-quantized in place (``--quantization nf4``) while the MoE action
expert stays bf16; ``none`` loads bf16 for ≥16 GB GPUs.

Attention backend: flash-attn is intentionally NOT installed in the sidecar venv
(it is not in the upstream ``requirements.txt`` and needs a source build). The
upstream ``QwenvlWithExpertV2Model.__init__`` hardcodes ``_attn_implementation =
"flash_attention_2"`` on the VLM / text / expert configs, so we coerce those
configs to a flash-free backend (``--attn sdpa`` by default; ``eager`` as the
universally-correct fallback) before and after the model is built (see
:func:`_install_attn_fallback`).

CLAUDE.md compliance: real upstream code in a real subprocess (no mocks, §1.11);
py-version/dep isolation is the only safe bridge (§3); the model's Apache-2.0
weights carry no license guard.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

_NDARRAY_SENTINEL = "__ndarray__"


def _encode_ndarray(obj: Any) -> Any:
    """Mirror ``openral_sim.sidecar.encode_ndarray`` (msgpack-only ndarray wire)."""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {_NDARRAY_SENTINEL: True, "npy": buf.getvalue()}
    return obj


def _decode_ndarray(obj: dict[str, Any]) -> Any:
    if _NDARRAY_SENTINEL in obj and "npy" in obj:
        return np.load(io.BytesIO(obj["npy"]), allow_pickle=False)
    return obj


def _resolve_repo() -> Path:
    override = os.environ.get("OPENRAL_LINGBOT_VLA2_REPO")
    default = Path.home() / ".cache" / "openral" / "lingbot-vla2-sidecar" / "source"
    repo = Path(override).expanduser() if override else default
    if not (repo / "lingbotvla").is_dir():
        raise SystemExit(
            f"lingbotvla repo checkout not found at {repo}. The boot helper "
            "(tools/lingbot_vla2_sidecar.py) clones it; set "
            "OPENRAL_LINGBOT_VLA2_REPO to reuse an existing checkout."
        )
    return repo


def _resolve_qwen() -> str:
    for env in ("OPENRAL_QWEN3VL_PATH", "QWEN3VL_PATH"):
        val = os.environ.get(env)
        if val:
            return val
    # Fall back to the HF id; the server will fetch it if not cached.
    return "Qwen/Qwen3-VL-4B-Instruct"


def _resolve_checkpoint(model: str) -> str:
    """Return a local checkpoint dir (dir of ``*.safetensors``) for ``model``.

    A local path is used verbatim; an HF id is snapshot-downloaded. The id may be
    a manifest-style ref: an optional ``hf://`` scheme is stripped and an optional
    ``@<branch-or-sha>`` revision pin is split off and passed as ``revision=``
    (``huggingface_hub`` ignores an ``@sha`` glued onto the repo id — the same
    ``resolve_rskill_repo_revision`` convention the lerobot adapters use). Only the
    quantized pack / configs / tokenizer are fetched, never a stray fp32 shard set.
    """
    # A local checkpoint override wins over the passed id — lets an operator serve
    # a local fp32 checkout (skipping the HF download) regardless of the rskill's
    # weights_uri, mirroring OPENRAL_LINGBOT_VLA2_REPO for the code checkout.
    override = os.environ.get("OPENRAL_LINGBOT_VLA2_CKPT")
    if override and Path(override).expanduser().is_dir():
        return str(Path(override).expanduser())
    p = Path(model).expanduser()
    if p.is_dir():
        return str(p)
    from huggingface_hub import snapshot_download

    ref = model[len("hf://") :] if model.startswith("hf://") else model
    repo_id, _, revision = ref.partition("@")
    return snapshot_download(
        repo_id,
        revision=revision or None,
        # ``*.yaml`` fetches the V1 checkpoint's ``lingbotvla_cli.yaml`` (the V1
        # loader reads it from the ckpt dir directly); harmless for v2.
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "tokenizer*"],
    )


def _write_cli_yaml(repo: Path, ckpt_dir: str, qwen: str) -> Path:
    """Reconstruct the training config the upstream loader reads.

    ``LingbotVLAv2Server.load_vla`` reads ``lingbotvla_cli.yaml`` at
    ``ckpt.parent.parent.parent`` — the HF release does not ship it, but the
    repo's ``configs/vla/robotwin/robotwin.yaml`` carries every architecture dim.
    We copy it with the checkpoint / backbone / norm-stats paths patched and
    place it at the location the loader expects.
    """
    import yaml

    tmpl = yaml.safe_load((repo / "configs/vla/robotwin/robotwin.yaml").read_text())
    tmpl["model"]["model_path"] = ckpt_dir
    tmpl["model"]["tokenizer_path"] = qwen
    tmpl["data"]["norm_stats_file"] = "assets/norm_stats/robotwin.json"
    tmpl["data"].setdefault("img_size", 256)
    # FeatureInfo.update_info / get_normalizer ast.literal_eval each joints +
    # norm_type entry, so they must be string-reprs of the {name: val} dicts,
    # not YAML-parsed dicts (verified live: FeatureTransform crashes otherwise).
    for _key in ("joints", "norm_type"):
        if _key in tmpl["data"]:
            tmpl["data"][_key] = [
                str(entry) if isinstance(entry, dict) else entry for entry in tmpl["data"][_key]
            ]
    # loader path: Path(model_path).parent.parent.parent / "lingbotvla_cli.yaml"
    cli = Path(ckpt_dir).parent.parent.parent / "lingbotvla_cli.yaml"
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text(yaml.safe_dump(tmpl))
    return cli


def _install_lerobot_stub() -> None:
    """Stub any ``lerobot.*`` import so the model class imports.

    The upstream model import chain pulls in ``lingbotvla.data`` (training-only
    ``lerobot`` imports scattered across submodules) — not needed for inference,
    and ``lerobot`` cannot be installed here anyway (it wants transformers 5.x,
    conflicting with the pinned 4.57.3). A meta-path finder returns a permissive
    stub module for every ``lerobot`` submodule. Verified live: without this the
    ``from deploy.lingbot_vla_v2_policy import ...`` chain ModuleNotFound's.
    """
    import importlib.abc
    import importlib.machinery
    import types

    class _Dummy:
        def __init__(self, *a: Any, **k: Any) -> None: ...

        def __call__(self, *a: Any, **k: Any) -> None:
            return None

    class _AnyStub(types.ModuleType):
        __path__: ClassVar[list[str]] = []

        def __getattr__(self, name: str) -> Any:
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _Dummy

    class _LerobotFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
            if fullname == "lerobot" or fullname.startswith("lerobot."):
                return importlib.machinery.ModuleSpec(fullname, self)
            return None

        def create_module(self, spec: Any) -> Any:
            return _AnyStub(spec.name)

        def exec_module(self, module: Any) -> None: ...

    if not any(type(f).__name__ == "_LerobotFinder" for f in sys.meta_path):
        sys.meta_path.insert(0, _LerobotFinder())


def _coerce_attn_config(config: Any, target: str) -> None:
    """Recursively rewrite ``_attn_implementation`` on ``config`` + sub-configs.

    The upstream ``QwenvlWithExpertV2Model.__init__`` hardcodes
    ``flash_attention_2`` on the VLM / ``text_config`` / ``qwen_expert_config``
    (and ``vision_config`` inherits ``vit_attn_implementation`` which also
    defaults to flash). We walk every attribute that is itself a config
    (``*_config``) so the whole tree lands on a flash-free backend.
    """
    if config is None:
        return
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = target
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = target
    for name in dir(config):
        if not name.endswith("_config"):
            continue
        child = getattr(config, name, None)
        if child is not None and hasattr(child, "_attn_implementation"):
            _coerce_attn_config(child, target)


def _install_attn_fallback(*, target: str) -> None:
    """Patch the two Qwen ``_from_config`` classmethods to drop flash attention.

    ``QwenvlWithExpertV2Model.__init__`` calls
    ``Qwen3VLForConditionalGeneration._from_config(vlm_config)`` and
    ``Qwen2ForCausalLM._from_config(qwen_expert_config)`` with configs it has
    just pinned to ``flash_attention_2``. flash-attn is not installed in the
    sidecar venv, so we intercept both classmethods and coerce the incoming
    config (and its sub-configs) to ``target`` before delegating — the attention
    modules bind their backend at construction from ``config._attn_implementation``.
    Idempotent; a no-op for anyone who already passes an explicit backend.
    """
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2ForCausalLM
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import Qwen3VLForConditionalGeneration

    for cls in (Qwen3VLForConditionalGeneration, Qwen2ForCausalLM):
        orig = cls._from_config.__func__  # type: ignore[attr-defined]  # reason: classmethod unwrap

        def _patched(
            inner_cls: Any, config: Any, *args: Any, _orig: Any = orig, **kwargs: Any
        ) -> Any:
            _coerce_attn_config(config, target)
            kwargs.setdefault("attn_implementation", target)
            try:
                return _orig(inner_cls, config, *args, **kwargs)
            except TypeError:
                # Older signatures do not accept the attn_implementation kwarg.
                kwargs.pop("attn_implementation", None)
                return _orig(inner_cls, config, *args, **kwargs)

        cls._from_config = classmethod(_patched)  # type: ignore[assignment]  # reason: monkeypatch


def _nf4_backbone_in_place(
    backbone: Any,
    *,
    torch: Any,
    min_params: int = 2_000_000,
    skip_names: frozenset[str] = frozenset(),
) -> None:
    """Self-contained NF4 rewrite of ``nn.Linear`` -> ``Linear4bit`` (bnb packs on ``.cuda()``).

    ``skip_names`` leaves matching child modules unquantized. The V1 Qwen2.5-VL
    backbone needs ``{"o_proj"}`` skipped: its interleaved-attention forward reads
    ``o_proj.weight.dtype`` (uint8 once packed) to cast the attention output, which
    would corrupt the bf16 activations to Byte — keeping o_proj bf16 also improves
    accuracy at ~0.3 GB cost.
    """
    import bitsandbytes as bnb

    def _replace(module: Any) -> None:
        for name, child in list(module.named_children()):
            if name in skip_names:
                continue
            if isinstance(child, torch.nn.Linear) and child.weight.numel() >= min_params:
                new = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data.clone(), requires_grad=False, quant_type="nf4"
                )
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(
                        child.bias.data.clone().to(torch.bfloat16), requires_grad=False
                    )
                setattr(module, name, new)
            else:
                _replace(child)

    _replace(backbone)


# bnb serialises a packed Params4bit into ``<prefix>.weight`` (packed uint8) plus
# these sibling metadata tensors (double/nested quant is on by default in the
# bnb our sidecar pins, so ``.absmax`` is itself quantized with ``.nested_*``).
# The suffixes mirror ``openral_sim._quantization._BNB_META_SUFFIXES`` so the
# on-disk pack format is identical to every other OpenRAL nf4 rSkill.
_BNB_META_SUFFIXES = (
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
    ".quant_state.bitsandbytes__nf4",
    ".quant_state.bitsandbytes__fp4",
)


def _detect_prequantized(ckpt_dir: str) -> bool:
    """True when ``ckpt_dir`` carries a pre-quantized NF4 pack.

    The house sentinel is ``quantization_metadata.json`` with
    ``quantization.scheme == "nf4"`` (produced by
    ``tools/quantize_lingbot_vla2.py``, same shape as
    ``tools/quantize_rskill.py`` writes for the lerobot nf4 rSkills). When present
    the server can skip the 25.5 GB fp32 read + the on-line bf16->nf4 pack and
    overlay the packed weights straight into ``Linear4bit`` shells.
    """
    import json

    meta_path = Path(ckpt_dir) / "quantization_metadata.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(meta.get("quantization", {}).get("scheme") == "nf4")


def _install_prequantized_backbone(
    vla: Any, state: dict[str, Any], *, device: str, torch: Any
) -> tuple[int, set[str]]:
    """Overlay packed nf4 weights onto the ``Linear4bit`` shells in ``vla``.

    Self-contained twin of ``openral_sim._quantization.install_prequantized_linears``
    (the sidecar never imports ``openral_*``). Walks the policy, and for every
    ``bnb.nn.Linear4bit`` rebuilds ``weight`` via ``Params4bit.from_prequantized``
    so the packed uint8 tensor lands directly on ``device`` with no intermediate
    bf16 materialisation (the ~30 s ``.to(cuda)`` re-pack the standard path runs).

    Returns ``(n_modules_rebuilt, consumed_state_keys)`` so the caller can drop the
    consumed keys from the residual bf16 ``load_state_dict`` overlay.
    """
    del torch  # kept for parity with the sibling helpers
    import bitsandbytes as bnb

    consumed: set[str] = set()
    count = 0
    for prefix, module in vla.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        weight_key = f"{prefix}.weight"
        if weight_key not in state:
            continue
        quantized_stats: dict[str, Any] = {}
        for suffix in _BNB_META_SUFFIXES:
            full = f"{weight_key}{suffix}"
            if full in state:
                quantized_stats[suffix.lstrip(".")] = state[full]
                consumed.add(full)
        consumed.add(weight_key)
        module.weight = bnb.nn.Params4bit.from_prequantized(
            data=state[weight_key],
            quantized_stats=quantized_stats,
            requires_grad=False,
            device=device,
        )
        bias_key = f"{prefix}.bias"
        if module.bias is not None and bias_key in state:
            module.bias.data = state[bias_key].to(device).to(module.bias.dtype)
            consumed.add(bias_key)
        count += 1
    return count, consumed


def _overlay_prequantized(vla: Any, ckpt_dir: str, *, torch: Any) -> None:
    """Load ``model.safetensors`` from ``ckpt_dir`` into the nf4 shells + bf16 rest.

    The backbone ``Linear4bit`` modules get their packed weights via
    :func:`_install_prequantized_backbone`; every remaining bf16 tensor (MoE
    expert, align heads, action MLPs, embeddings, vision tower) is applied with a
    non-strict ``load_state_dict`` (the consumed nf4 keys are dropped so PyTorch
    does not flag them as missing on the already-rebuilt modules). Prequant NF4 is
    CUDA-only, so the packed weights are placed on ``cuda`` directly.
    """
    from safetensors.torch import load_file

    weights_path = Path(ckpt_dir) / "model.safetensors"
    if not weights_path.is_file():
        raise SystemExit(
            f"[lingbot_vla2_server] pre-quantized ckpt {ckpt_dir} has no "
            "model.safetensors; the quantization_metadata.json sentinel is present "
            "but the packed weights are missing."
        )
    state = load_file(str(weights_path), device="cpu")
    loaded, consumed = _install_prequantized_backbone(vla, state, device="cuda", torch=torch)
    leftover = {k: v for k, v in state.items() if k not in consumed}
    missing, unexpected = vla.load_state_dict(leftover, strict=False)
    print(
        f"[lingbot_vla2_server] prequant overlay: {loaded} nf4 modules, "
        f"{len(leftover)} bf16 residual keys, {len(unexpected)} unexpected "
        f"(nf4-shell missing slots are expected, not an error).",
        flush=True,
    )


def _follow_model_device(srv: Any, *, torch: Any) -> None:
    """Make ``srv.sample_actions_fn`` move its tensor inputs to the model's device.

    The upstream ``sample_actions_batch`` moves inputs to a hardcoded
    ``device="cuda"`` before calling the sampler; on a CPU (or otherwise
    non-cuda) deployment that would land inputs on the GPU while the weights sit
    on CPU. We wrap the sampler so every tensor arg/kwarg follows the model's
    real device, resolved per call (the model's ``sample_actions`` is otherwise
    device-agnostic — ``device = state.device``).
    """
    model = srv.vla.model
    orig = srv.sample_actions_fn

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        dev = next(model.parameters()).device
        moved_args = [a.to(dev) if isinstance(a, torch.Tensor) else a for a in args]
        moved_kwargs = {
            k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()
        }
        return orig(*moved_args, **moved_kwargs)

    srv.sample_actions_fn = _wrapped


def _install_cpu_moe_fallback() -> None:
    """Replace the CUDA-only fused MoE kernel with a pure-torch SwiGLU expert loop.

    The action expert routes tokens through ``Qwen2FusedExperts.forward`` ->
    ``lingbotvla.ops.fused_moe.fused_moe_forward``, whose ``group_gemm`` Triton
    kernel hard-asserts ``input.is_cuda`` (``ops/group_gemm/kernel/moe.py``), so
    the expert cannot run on CPU. The math is a standard top-k SwiGLU mixture over
    the packed per-expert weights (``gate_proj``/``up_proj``/``down_proj`` of shape
    ``[E, I, H]``/``[E, I, H]``/``[E, H, I]``): ``down(silu(gate·x) * (up·x))``,
    scaled by the token's routing weight and summed over its selected experts —
    exactly what the fused kernel computes (``fused_moe.py`` steps 4-10). We monkey-
    patch ``Qwen2FusedExperts.forward`` with that per-expert loop; it matches the
    method the outer ``Qwen2MoeSparseMoeBlock.forward`` already calls on the fused
    path, so only the inner kernel changes. Idempotent.
    """
    import torch
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2FusedExperts
    from torch.nn.functional import linear, silu

    if getattr(Qwen2FusedExperts, "_openral_cpu_moe", False):
        return

    def _torch_forward(
        self: Any,
        module: Any,
        num_experts: int,
        routing_weights: Any,
        selected_experts: Any,
        hidden_states: Any,
    ) -> Any:
        # hidden_states (N, H); routing_weights / selected_experts (N, top_k).
        out = torch.zeros_like(hidden_states)
        for e in range(num_experts):
            sel = selected_experts == e  # (N, top_k)
            tok = sel.any(dim=1)  # (N,) tokens routed to expert e
            if not bool(tok.any()):
                continue
            x = hidden_states[tok]
            hidden = silu(linear(x, self.gate_proj[e])) * linear(x, self.up_proj[e])
            y = linear(hidden, self.down_proj[e])
            weight = (routing_weights[tok] * sel[tok].to(routing_weights.dtype)).sum(
                dim=1, keepdim=True
            )
            out[tok] += (y * weight).to(out.dtype)
        return out

    Qwen2FusedExperts.forward = _torch_forward  # type: ignore[method-assign]  # reason: CPU MoE patch
    Qwen2FusedExperts._openral_cpu_moe = True  # type: ignore[attr-defined]  # reason: idempotence guard


class _LingBotPolicy:
    """Loads ``LingbotVLAv2Server`` (NF4 backbone / bf16 expert) and serves actions."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        repo = _resolve_repo()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        # The loader resolves configs/ + assets/ relative to CWD.
        os.chdir(repo)
        qwen = _resolve_qwen()
        os.environ["QWEN3VL_PATH"] = qwen
        ckpt_dir = _resolve_checkpoint(args.model)
        _write_cli_yaml(repo, ckpt_dir, qwen)

        # The model import chain pulls training-only lerobot imports; stub them
        # before importing the deploy module (see _install_lerobot_stub).
        _install_lerobot_stub()

        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch

        self._torch = torch
        self._robo_name = args.robo_name
        quant = args.quantization
        device = args.device
        # bitsandbytes NF4 packs its weights on ``.cuda()`` and has no CPU kernel,
        # so a CPU deployment can only run bf16. Skip the quant rewrite (and say
        # so) rather than crash — the model then needs ~12.8 GB RAM at bf16.
        if device == "cpu" and quant == "nf4":
            print(
                "[lingbot_vla2_server] device=cpu: NF4 needs CUDA (bitsandbytes); "
                "loading bf16 on CPU instead (~12.8 GB RAM).",
                flush=True,
            )
            quant = "none"

        # A pre-quantized NF4 checkpoint (quantization_metadata.json scheme=nf4)
        # ships packed 4-bit weights that only bitsandbytes' CUDA kernels can
        # dequantize, so a CPU deployment cannot load it. Refuse early with a clear
        # pointer to the fp32 upstream ckpt rather than crashing deep in the overlay.
        is_prequant = _detect_prequantized(ckpt_dir)
        if is_prequant and device == "cpu":
            raise SystemExit(
                f"[lingbot_vla2_server] {ckpt_dir} is a pre-quantized NF4 pack, which "
                "requires CUDA (bitsandbytes has no CPU 4-bit kernel). Re-run with "
                "--device cuda, or point --model at the fp32 upstream checkpoint "
                "(robbyant/lingbot-vla-v2-6b) for a CPU/bf16 load."
            )

        # flash-attn is not installed in the sidecar venv; coerce the upstream
        # hardcoded flash_attention_2 configs to a flash-free backend BEFORE the
        # model is built (the attention modules bind at construction).
        _install_attn_fallback(target=args.attn)

        # Replicate LingbotVLAv2Server.__init__ but inject quantization before
        # the device move (the stock server only offers fp32 / bf16).
        srv = LingbotVLAv2Server.__new__(LingbotVLAv2Server)
        srv.adaptive_ensemble_alpha = 0.1
        srv.action_ensemble_horizon = 8
        # Return the FULL predicted chunk (chunk_ret) untruncated: upstream infer
        # slices the chunk to use_length when use_length > 0 (modeling infer path),
        # so use_length=1 would emit a single 14-D step per inference and defeat
        # the adapter's chunk replay. use_length=-1 keeps all predicted steps; the
        # openral adapter slices its own replan window (_replan_steps).
        srv.use_length = -1
        srv.chunk_ret = True
        srv.robot_norm_path = None
        srv.task_description = None
        srv.use_compile = False
        apply_lingbot_qwen3_vl_patch()
        if is_prequant:
            # Fast path: skip the 25.5 GB fp32 read + the ~30 s on-line bf16->nf4
            # pack. Build the graph structure only (weight load stubbed to a no-op),
            # rewrite the backbone Linears to Linear4bit shells, then overlay the
            # packed nf4 weights + bf16 remainder straight from the prequant
            # safetensors. Building under a bf16 default dtype halves the transient
            # CPU footprint of the throwaway init (every param is overwritten by the
            # overlay a moment later).
            srv.load_model_weights = lambda *a, **k: None
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                srv.vla = srv.load_vla(ckpt_dir)
            finally:
                torch.set_default_dtype(prev_dtype)
            _coerce_attn_config(getattr(srv.vla, "config", None), args.attn)
            srv.vla.model = srv.vla.model.to(torch.bfloat16)
            _nf4_backbone_in_place(srv.vla.model.qwenvl_with_expert.qwenvl, torch=torch)
            _overlay_prequantized(srv.vla, ckpt_dir, torch=torch)
            srv.vla = srv.vla.to(device).eval()
        else:
            srv.vla = srv.load_vla(ckpt_dir)  # CPU build + strict load_state_dict
            # Belt-and-suspenders: coerce any attn config the build left on flash.
            _coerce_attn_config(getattr(srv.vla, "config", None), args.attn)
            srv.vla.model = srv.vla.model.to(torch.bfloat16)
            if quant == "nf4":
                _nf4_backbone_in_place(srv.vla.model.qwenvl_with_expert.qwenvl, torch=torch)
            srv.vla = srv.vla.to(device).eval()
        # ``sample_actions_batch`` hardcodes ``device="cuda"`` before the model
        # forward (upstream deploy path), which would move inputs to the GPU while
        # our weights sit on CPU. Wrap the sampler so inputs always follow the
        # model's real device; the model's own ``sample_actions`` is otherwise
        # device-agnostic (``device = state.device``). No-op cost on the CUDA path,
        # so only install it off-GPU.
        if device != "cuda":
            _follow_model_device(srv, torch=torch)
            # The sparse-MoE action expert dispatches through a custom Triton
            # group_gemm kernel that hard-asserts CUDA (lingbotvla.ops.group_gemm);
            # swap it for a pure-torch SwiGLU MoE so the expert runs off-GPU.
            _install_cpu_moe_fallback()
        srv.global_step = 0
        srv.last_action_chunk = None
        srv.last_normalized_action_chunk = None
        srv.use_bf16 = True
        srv.use_fp32 = False
        srv.action_key = "action"
        srv.reset(self._robo_name)
        self._srv = srv
        self._action_keys = list(srv.vla.feature_transform.org_features["actions"])
        mode = "nf4-prequant" if is_prequant else quant
        if device == "cuda" and torch.cuda.is_available():
            print(
                f"[lingbot_vla2_server] loaded ({mode}, attn={args.attn}); "
                f"VRAM={torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
                flush=True,
            )
        else:
            print(
                f"[lingbot_vla2_server] loaded ({mode}, attn={args.attn}, device={device})",
                flush=True,
            )

    def reset(self, robo_name: str | None = None) -> None:
        self._srv.reset(robo_name or self._robo_name)

    def get_action(self, obs: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
        images = obs["images"]
        raw = {
            "observation.images.cam_high": np.asarray(images["cam_high"], dtype=np.uint8),
            "observation.images.cam_left_wrist": np.asarray(
                images["cam_left_wrist"], dtype=np.uint8
            ),
            "observation.images.cam_right_wrist": np.asarray(
                images["cam_right_wrist"], dtype=np.uint8
            ),
            "observation.state": np.asarray(obs["state"], dtype=np.float32),
            "task": str(obs.get("task", "")),
        }
        out = self._srv.infer(raw)  # chunk_ret -> {action_key: (chunk, dim)}
        chunk = np.concatenate(
            [np.asarray(out[k], dtype=np.float32) for k in self._action_keys], axis=-1
        )
        return chunk, self._action_keys


# ── LingBot-VLA 1.0 (4B) variant ────────────────────────────────────────────
# The V1 family (paper "A Pragmatic VLA Foundation Model", arXiv 2601.18692) is a
# Qwen2.5-VL-3B backbone + a *dense* Qwen2 flow-matching action expert — a
# different upstream repo (github.com/robbyant/lingbot-vla), deploy class
# (LingbotVLAServer), and stack (transformers==4.51.3 + lerobot==0.4.2, flat
# layout, no lerobot stub) than the v2 6B MoE model. Its checkpoint ships real
# config.json + lingbotvla_cli.yaml, so no _write_cli_yaml reconstruction is
# needed. It shares the transport, obs contract, and NF4 recipe, so the two run
# through the same server behind a --variant switch.


def _resolve_repo_v1() -> Path:
    override = os.environ.get("OPENRAL_LINGBOT_VLA_REPO")
    default = Path.home() / ".cache" / "openral" / "lingbot-vla-v1-sidecar" / "source"
    repo = Path(override).expanduser() if override else default
    if not (repo / "lingbotvla").is_dir():
        raise SystemExit(
            f"lingbotvla (V1) repo checkout not found at {repo}. The boot helper "
            "(tools/lingbot_vla2_sidecar.py --variant v1) clones it; set "
            "OPENRAL_LINGBOT_VLA_REPO to reuse an existing checkout."
        )
    return repo


def _resolve_qwen25() -> str:
    for env in ("OPENRAL_QWEN25VL_PATH", "QWEN25_PATH"):
        val = os.environ.get(env)
        if val:
            return val
    return "Qwen/Qwen2.5-VL-3B-Instruct"


def _install_attn_fallback_v1(*, target: str) -> None:
    """Coerce every ``PreTrainedModel._from_config`` off flash for the V1 model.

    The V1 builders call ``<Model>._from_config(cfg, use_flash_attention_2=True)``
    at several sites (Qwen2.5-VL, its vision tower, the Qwen2 expert); the custom
    vision/text attention only registers ``eager`` + ``flash_attention_2`` (no
    sdpa), and flash-attn is not installed. Patch the shared base classmethod once:
    strip the flash flag, coerce the config tree, force ``attn_implementation``.
    """
    from transformers.modeling_utils import PreTrainedModel

    orig = PreTrainedModel._from_config.__func__  # type: ignore[attr-defined]  # reason: classmethod unwrap

    def _patched(inner_cls: Any, config: Any, *a: Any, _orig: Any = orig, **k: Any) -> Any:
        _coerce_attn_config(config, target)
        k.pop("use_flash_attention_2", None)
        k["attn_implementation"] = target
        try:
            return _orig(inner_cls, config, *a, **k)
        except (TypeError, ValueError):
            k.pop("attn_implementation", None)
            return _orig(inner_cls, config, *a, **k)

    PreTrainedModel._from_config = classmethod(_patched)  # type: ignore[assignment]  # reason: monkeypatch


def _patch_eager_vision_rotary_v1() -> None:
    """Inject the missing ``rotate_half`` into the V1 eager vision-attention path.

    Upstream's eager vision attention references ``rotate_half`` without importing
    it (only the flash path was exercised); pull the canonical impl from
    ``modeling_lingbot_vla`` into the ``qwenvl_in_vla`` module namespace.
    """
    import lingbotvla.models.vla.pi0.qwenvl_in_vla as _qvl
    from lingbotvla.models.vla.pi0.modeling_lingbot_vla import rotate_half as _rotate_half

    _qvl.rotate_half = _rotate_half  # type: ignore[attr-defined]  # reason: fill upstream gap


class _LingBotV1Policy:
    """Loads the V1 ``LingbotVLAServer`` (NF4 backbone / bf16 expert) and serves actions."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        repo = _resolve_repo_v1()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        os.chdir(repo)  # configs/ + assets/ resolved relative to CWD
        os.environ["QWEN25_PATH"] = _resolve_qwen25()
        ckpt_dir = _resolve_checkpoint(args.model)

        # V1 attention only registers eager/flash; flash is absent, so eager is the
        # only flash-free backend (sdpa raises KeyError in the custom vision/text
        # attention). The --attn choice is coerced to eager for v1.
        target = "eager"
        _install_attn_fallback_v1(target=target)
        _patch_eager_vision_rotary_v1()

        from deploy.lingbot_vla_policy import LingbotVLAServer

        self._torch = torch
        self._robo_name = args.robo_name
        quant = args.quantization
        device = args.device
        if device == "cpu" and quant == "nf4":
            print(
                "[lingbot_vla2_server:v1] device=cpu: NF4 needs CUDA (bitsandbytes); "
                "loading bf16 on CPU instead (~8.4 GB RAM).",
                flush=True,
            )
            quant = "none"
        is_prequant = _detect_prequantized(ckpt_dir)
        if is_prequant and device == "cpu":
            raise SystemExit(
                f"[lingbot_vla2_server:v1] {ckpt_dir} is a pre-quantized NF4 pack, which "
                "requires CUDA (bitsandbytes has no CPU 4-bit kernel). Re-run with "
                "--device cuda, or point --model at the fp32 upstream checkpoint "
                "(robbyant/lingbot-vla-4b-posttrain-robotwin) for a CPU/bf16 load."
            )

        srv = LingbotVLAServer.__new__(LingbotVLAServer)
        srv.adaptive_ensemble_alpha = 0.1
        srv.action_ensemble_horizon = 8
        # Return the FULL predicted chunk untruncated (use_length=-1); the adapter
        # slices its own replan window.
        srv.use_length = -1
        srv.use_compile = False
        srv.num_denoising_step = 10
        srv.robot_norm_path = None

        # load_vla builds fp32 on CPU then calls policy.cuda(); no-op cuda during
        # the build so the 16.8 GB fp32 graph never hits an 8 GB card before NF4.
        _orig_cuda = torch.nn.Module.cuda
        torch.nn.Module.cuda = lambda self, *a, **k: self  # type: ignore[method-assign]  # reason: build-time guard
        try:
            if is_prequant:
                srv.load_model_weights = lambda *a, **k: None  # type: ignore[attr-defined]  # reason: prequant overlay replaces the load
                prev_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.bfloat16)
                try:
                    srv.vla = srv.load_vla(ckpt_dir)
                finally:
                    torch.set_default_dtype(prev_dtype)
            else:
                srv.vla = srv.load_vla(ckpt_dir)  # CPU build + strict load_state_dict
        finally:
            torch.nn.Module.cuda = _orig_cuda  # type: ignore[method-assign]  # reason: restore guard

        _coerce_attn_config(getattr(srv.vla, "config", None), target)
        srv.vla.model = srv.vla.model.to(torch.bfloat16)
        if is_prequant:
            _nf4_backbone_in_place(
                srv.vla.model.qwenvl_with_expert.qwenvl,
                torch=torch,
                skip_names=frozenset({"o_proj"}),
            )
            _overlay_prequantized(srv.vla, ckpt_dir, torch=torch)
            srv.vla = srv.vla.to(device).eval()
        else:
            if quant == "nf4":
                _nf4_backbone_in_place(
                    srv.vla.model.qwenvl_with_expert.qwenvl,
                    torch=torch,
                    skip_names=frozenset({"o_proj"}),
                )
            srv.vla = srv.vla.to(device).eval()

        srv.global_step = 0
        srv.last_action_chunk = None
        srv.use_bf16 = True
        srv.reset(self._robo_name)
        self._srv = srv
        self._action_keys = list(srv.vla.feature_transform.org_features["actions"])
        mode = "nf4-prequant" if is_prequant else quant
        if device == "cuda" and torch.cuda.is_available():
            print(
                f"[lingbot_vla2_server:v1] loaded ({mode}, attn={target}); "
                f"VRAM={torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
                flush=True,
            )
        else:
            print(
                f"[lingbot_vla2_server:v1] loaded ({mode}, attn={target}, device={device})",
                flush=True,
            )

    def reset(self, robo_name: str | None = None) -> None:
        self._srv.reset(robo_name or self._robo_name)

    def get_action(self, obs: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
        images = obs["images"]
        raw = {
            "observation.images.cam_high": np.asarray(images["cam_high"], dtype=np.uint8),
            "observation.images.cam_left_wrist": np.asarray(images["cam_left_wrist"], dtype=np.uint8),
            "observation.images.cam_right_wrist": np.asarray(
                images["cam_right_wrist"], dtype=np.uint8
            ),
            "observation.state": np.asarray(obs["state"], dtype=np.float32),
            "task": str(obs.get("task", "")),
        }
        out = self._srv.infer(raw)  # {action_key: (chunk, dim)}
        chunk = np.concatenate(
            [np.asarray(out[k], dtype=np.float32) for k in self._action_keys], axis=-1
        )
        return chunk, self._action_keys


def _serve(
    policy: _LingBotPolicy | _LingBotV1Policy,
    *,
    host: str,
    port: int,
    model: str,
    robo_name: str,
) -> int:
    import msgpack
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[lingbot_vla2_server] serving on tcp://{host}:{port}", flush=True)
    running = True
    while running:
        raw = sock.recv()
        req = msgpack.unpackb(raw, object_hook=_decode_ndarray, raw=False)
        endpoint = req.get("endpoint")
        data = req.get("data", {}) or {}
        try:
            if endpoint == "ping":
                reply: dict[str, Any] = {"ok": True, "model": model, "robo_name": robo_name}
            elif endpoint == "reset":
                policy.reset(data.get("robo_name"))
                reply = {"ok": True}
            elif endpoint == "get_action":
                action, keys = policy.get_action(data["observation"])
                reply = {"action": action, "action_keys": keys}
            elif endpoint == "close":
                reply = {"ok": True}
                running = False
            else:
                reply = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as exc:  # surface any sidecar-side fault to the client
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        sock.send(msgpack.packb(reply, default=_encode_ndarray, use_bin_type=True))
    sock.close(linger=0)
    ctx.term()
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenRAL LingBot-VLA 2.0 policy server (sidecar venv)")
    p.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    p.add_argument("--robo-name", default="robotwin", help="upstream robot config stem")
    p.add_argument("--quantization", choices=("none", "nf4"), default="nf4")
    p.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Inference device. cpu forces bf16 (NF4 needs CUDA) — frees the GPU "
        "for a co-resident SAPIEN sim on an 8 GB card, at a large latency cost.",
    )
    p.add_argument(
        "--attn",
        choices=("sdpa", "eager"),
        default="sdpa",
        help="flash-free attention backend to replace the upstream flash_attention_2 hardcode.",
    )
    p.add_argument(
        "--variant",
        choices=("v2", "v1"),
        default="v2",
        help="Model family: v2 = 6B Qwen3-VL MoE (LingBot-VLA 2.0); v1 = 4B "
        "Qwen2.5-VL dense expert (LingBot-VLA 1.0 / posttrain-robotwin). v1 forces "
        "attn=eager (its custom attention has no sdpa kernel).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = _parse_args(argv)
    policy: _LingBotPolicy | _LingBotV1Policy = (
        _LingBotV1Policy(args) if args.variant == "v1" else _LingBotPolicy(args)
    )
    return _serve(
        policy, host=args.host, port=args.port, model=args.model, robo_name=args.robo_name
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
