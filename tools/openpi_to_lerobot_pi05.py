"""Convert an OpenPI π0.5 Orbax checkpoint into a LeRobot PI05 checkpoint.

The RoboCasa365 release stores OpenPI/JAX params as stacked Orbax arrays. LeRobot
PI05 expects an unstacked PyTorch ``model.safetensors`` plus the usual policy
processor sidecars. This script performs only that deterministic format bridge;
quantization stays in ``tools/quantize_rskill.py``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

_OPENPI_RESTORE_DEPS = """
OpenPI restore dependencies are not installed in this environment.
Use a throwaway uv invocation, for example:

  uv run --group robocasa \\
    --with 'jax[cpu]==0.5.3' --with 'flax==0.10.2' \\
    --with 'orbax-checkpoint==0.11.13' --with 'jaxtyping==0.2.36' \\
    --with 'beartype==0.19.0' --with etils --with augmax \\
    --with openpi-client --with einops --with sentencepiece --with tyro \\
    --with msgpack --with safetensors --with huggingface_hub \\
    --with torch --with transformers --with 'numpy<2' --with ml_collections \\
    --with chex==0.1.90 --with treescope --with dm-tree --with equinox \\
    python tools/openpi_to_lerobot_pi05.py ...
""".strip()


def _maybe_add_openpi_src(path: str | None) -> None:
    if path:
        sys.path.insert(0, str(Path(path).resolve() / "src"))


def _restore_openpi_params(params_dir: Path) -> dict[str, Any]:
    try:
        import numpy as np

        traverse_util = cast(Any, importlib.import_module("flax.traverse_util"))
        openpi_model = cast(Any, importlib.import_module("openpi.models.model"))
    except Exception as exc:  # pragma: no cover - dependency gate
        raise RuntimeError(_OPENPI_RESTORE_DEPS) from exc

    params = openpi_model.restore_params(params_dir, restore_type=np.ndarray)
    return cast(dict[str, Any], traverse_util.flatten_dict(params, sep="."))


def _download_params(source: str, revision: str | None, checkpoint_subdir: str) -> Path:
    local = Path(source)
    if local.exists():
        return (
            (local / checkpoint_subdir / "params").resolve()
            if checkpoint_subdir
            else local.resolve()
        )
    root = Path(
        snapshot_download(
            repo_id=source,
            repo_type="model",
            revision=revision,
            allow_patterns=[f"{checkpoint_subdir.rstrip('/')}/params/**"],
        )
    )
    return root / checkpoint_subdir / "params"


def _tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    out = torch.from_numpy(value)
    return out.to(dtype=dtype) if dtype is not None else out


def _linear(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    return _tensor(value.T.copy(), dtype=dtype)


def _vision_attention(value: Any, layer: int, *, out: bool = False) -> torch.Tensor:
    arr = value[layer]
    if out:
        return _tensor(arr.reshape(-1, arr.shape[-1]).T.copy())
    return _tensor(arr.reshape(-1, arr.shape[0]).T.copy())


def _gemma_q(value: Any, layer: int, dtype: torch.dtype) -> torch.Tensor:
    arr = value[layer]
    return _tensor(arr.transpose(0, 2, 1).reshape(-1, arr.shape[1]).copy(), dtype)


def _gemma_kv(value: Any, layer: int, kv: int, dtype: torch.dtype) -> torch.Tensor:
    return _linear(value[layer, kv, 0], dtype)


def _gemma_o_main(value: Any, layer: int, dtype: torch.dtype) -> torch.Tensor:
    arr = value[layer]
    return _tensor(arr.transpose(2, 0, 1).reshape(-1, arr.shape[-1]).copy(), dtype)


def _gemma_o_expert(value: Any, layer: int, dtype: torch.dtype) -> torch.Tensor:
    arr = value[layer]
    return _tensor(arr.reshape(-1, arr.shape[-1]).T.copy(), dtype)


def _map_state(params: dict[str, Any]) -> dict[str, torch.Tensor]:
    bf16 = torch.bfloat16
    state: dict[str, torch.Tensor] = {}

    vision = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    state[f"{vision}.embeddings.patch_embedding.weight"] = _tensor(
        params["PaliGemma.img.embedding.kernel"].transpose(3, 2, 0, 1).copy()
    )
    state[f"{vision}.embeddings.patch_embedding.bias"] = _tensor(
        params["PaliGemma.img.embedding.bias"]
    )
    state[f"{vision}.embeddings.position_embedding.weight"] = _tensor(
        params["PaliGemma.img.pos_embedding"][0]
    )
    for i in range(27):
        dst = f"{vision}.encoder.layers.{i}"
        state[f"{dst}.layer_norm1.weight"] = _tensor(
            params["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.scale"][i]
        )
        state[f"{dst}.layer_norm1.bias"] = _tensor(
            params["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.bias"][i]
        )
        state[f"{dst}.layer_norm2.weight"] = _tensor(
            params["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.scale"][i]
        )
        state[f"{dst}.layer_norm2.bias"] = _tensor(
            params["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.bias"][i]
        )
        attn = "PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0"
        for src, dst_name in {"key": "k_proj", "value": "v_proj", "query": "q_proj"}.items():
            state[f"{dst}.self_attn.{dst_name}.weight"] = _vision_attention(
                params[f"{attn}.{src}.kernel"], i
            )
            state[f"{dst}.self_attn.{dst_name}.bias"] = _tensor(
                params[f"{attn}.{src}.bias"][i].reshape(-1)
            )
        state[f"{dst}.self_attn.out_proj.weight"] = _vision_attention(
            params[f"{attn}.out.kernel"], i, out=True
        )
        state[f"{dst}.self_attn.out_proj.bias"] = _tensor(params[f"{attn}.out.bias"][i])
        state[f"{dst}.mlp.fc1.weight"] = _linear(
            params["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.kernel"][i]
        )
        state[f"{dst}.mlp.fc1.bias"] = _tensor(
            params["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.bias"][i]
        )
        state[f"{dst}.mlp.fc2.weight"] = _linear(
            params["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.kernel"][i]
        )
        state[f"{dst}.mlp.fc2.bias"] = _tensor(
            params["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.bias"][i]
        )
    state[f"{vision}.post_layernorm.weight"] = _tensor(
        params["PaliGemma.img.Transformer.encoder_norm.scale"]
    )
    state[f"{vision}.post_layernorm.bias"] = _tensor(
        params["PaliGemma.img.Transformer.encoder_norm.bias"]
    )
    state["model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"] = (
        _linear(params["PaliGemma.img.head.kernel"])
    )
    state["model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"] = (
        _tensor(params["PaliGemma.img.head.bias"])
    )

    lang = "model.paligemma_with_expert.paligemma.model.language_model"
    embed = _tensor(params["PaliGemma.llm.embedder.input_embedding"], bf16)
    state[f"{lang}.embed_tokens.weight"] = embed
    state["model.paligemma_with_expert.paligemma.lm_head.weight"] = embed.clone()
    for i in range(18):
        dst = f"{lang}.layers.{i}"
        state[f"{dst}.self_attn.q_proj.weight"] = _gemma_q(
            params["PaliGemma.llm.layers.attn.q_einsum.w"], i, bf16
        )
        state[f"{dst}.self_attn.k_proj.weight"] = _gemma_kv(
            params["PaliGemma.llm.layers.attn.kv_einsum.w"], i, 0, bf16
        )
        state[f"{dst}.self_attn.v_proj.weight"] = _gemma_kv(
            params["PaliGemma.llm.layers.attn.kv_einsum.w"], i, 1, bf16
        )
        state[f"{dst}.self_attn.o_proj.weight"] = _gemma_o_main(
            params["PaliGemma.llm.layers.attn.attn_vec_einsum.w"], i, bf16
        )
        state[f"{dst}.mlp.gate_proj.weight"] = _linear(
            params["PaliGemma.llm.layers.mlp.gating_einsum"][i, 0], bf16
        )
        state[f"{dst}.mlp.up_proj.weight"] = _linear(
            params["PaliGemma.llm.layers.mlp.gating_einsum"][i, 1], bf16
        )
        state[f"{dst}.mlp.down_proj.weight"] = _linear(
            params["PaliGemma.llm.layers.mlp.linear"][i], bf16
        )
        state[f"{dst}.input_layernorm.weight"] = _tensor(
            params["PaliGemma.llm.layers.pre_attention_norm.scale"][i]
        )
        state[f"{dst}.post_attention_layernorm.weight"] = _tensor(
            params["PaliGemma.llm.layers.pre_ffw_norm.scale"][i]
        )
    state[f"{lang}.norm.weight"] = _tensor(params["PaliGemma.llm.final_norm.scale"])

    expert = "model.paligemma_with_expert.gemma_expert"
    # Unused by PI05 sampling, but LeRobot's strict loader expects it.
    state[f"{expert}.lm_head.weight"] = embed[:, :1024].clone()
    for i in range(18):
        dst = f"{expert}.model.layers.{i}"
        state[f"{dst}.self_attn.q_proj.weight"] = _gemma_q(
            params["PaliGemma.llm.layers.attn.q_einsum_1.w"], i, bf16
        )
        state[f"{dst}.self_attn.k_proj.weight"] = _gemma_kv(
            params["PaliGemma.llm.layers.attn.kv_einsum_1.w"], i, 0, bf16
        )
        state[f"{dst}.self_attn.v_proj.weight"] = _gemma_kv(
            params["PaliGemma.llm.layers.attn.kv_einsum_1.w"], i, 1, bf16
        )
        state[f"{dst}.self_attn.o_proj.weight"] = _gemma_o_expert(
            params["PaliGemma.llm.layers.attn.attn_vec_einsum_1.w"], i, bf16
        )
        state[f"{dst}.mlp.gate_proj.weight"] = _linear(
            params["PaliGemma.llm.layers.mlp_1.gating_einsum"][i, 0], bf16
        )
        state[f"{dst}.mlp.up_proj.weight"] = _linear(
            params["PaliGemma.llm.layers.mlp_1.gating_einsum"][i, 1], bf16
        )
        state[f"{dst}.mlp.down_proj.weight"] = _linear(
            params["PaliGemma.llm.layers.mlp_1.linear"][i], bf16
        )
        state[f"{dst}.input_layernorm.dense.weight"] = _linear(
            params["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.kernel"][i]
        )
        state[f"{dst}.input_layernorm.dense.bias"] = _tensor(
            params["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.bias"][i]
        )
        state[f"{dst}.post_attention_layernorm.dense.weight"] = _linear(
            params["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.kernel"][i]
        )
        state[f"{dst}.post_attention_layernorm.dense.bias"] = _tensor(
            params["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.bias"][i]
        )
    state[f"{expert}.model.norm.dense.weight"] = _linear(
        params["PaliGemma.llm.final_norm_1.Dense_0.kernel"]
    )
    state[f"{expert}.model.norm.dense.bias"] = _tensor(
        params["PaliGemma.llm.final_norm_1.Dense_0.bias"]
    )

    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        state[f"model.{name}.weight"] = _linear(params[f"{name}.kernel"])
        state[f"model.{name}.bias"] = _tensor(params[f"{name}.bias"])
    return state


def _copy_sidecars(template_repo: str, out_dir: Path) -> None:
    snapshot = Path(
        snapshot_download(
            repo_id=template_repo,
            repo_type="model",
            allow_patterns=["*.json", "*.md", "*.safetensors", ".gitattributes"],
            ignore_patterns=["model.safetensors", "quantization_metadata.json"],
        )
    )
    for src in snapshot.iterdir():
        if not src.is_file() or src.name == "model.safetensors":
            continue
        shutil.copy(src, out_dir / src.name)


def _patch_config(out_dir: Path) -> None:
    config = json.loads((out_dir / "config.json").read_text())
    config["device"] = "cpu"
    config["dtype"] = "bfloat16"
    config["compile_model"] = False
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def _validate_shapes(out_dir: Path, state: dict[str, torch.Tensor]) -> None:
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    except Exception as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("LeRobot PI05 is required to validate converted state shapes.") from exc

    cfg = PreTrainedConfig.from_pretrained(out_dir)
    cfg.device = "meta"
    cfg.compile_model = False
    target = PI05Policy(cfg).state_dict()
    missing = sorted(set(target) - set(state))
    extra = sorted(set(state) - set(target))
    bad_shapes = sorted(
        (k, tuple(state[k].shape), tuple(target[k].shape))
        for k in set(target) & set(state)
        if tuple(state[k].shape) != tuple(target[k].shape)
    )
    if missing or extra or bad_shapes:
        raise RuntimeError(
            "converted state does not match LeRobot PI05: "
            f"missing={missing[:5]} extra={extra[:5]} bad_shapes={bad_shapes[:5]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="robocasa/robocasa365_checkpoints")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--checkpoint-subdir",
        default="pi05_pretrain_human300/multitask_learning/75000",
    )
    parser.add_argument("--template-rskill", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--openpi-src", default=os.environ.get("OPENPI_SRC"))
    args = parser.parse_args()

    _maybe_add_openpi_src(args.openpi_src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_sidecars(args.template_rskill, out_dir)
    _patch_config(out_dir)

    params_dir = _download_params(args.source, args.revision, args.checkpoint_subdir)
    print(f"[openpi->lerobot] restoring {params_dir}", flush=True)
    state = _map_state(_restore_openpi_params(params_dir))
    _validate_shapes(out_dir, state)
    out_path = out_dir / "model.safetensors"
    print(f"[openpi->lerobot] writing {out_path} ({len(state)} tensors)", flush=True)
    save_file(state, str(out_path))
    print("[openpi->lerobot] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
