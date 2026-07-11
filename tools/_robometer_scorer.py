"""Robometer reward scorer, loaded in ``reward_monitor_node``.

Native LeRobot backend (lerobot >= 0.6.0). The reward model is
``lerobot.rewards.robometer.RobometerRewardModel`` — a vanilla
``AutoModelForImageTextToText`` (Qwen3-VL-4B) with three prediction heads, loaded
with plain ``transformers`` (no ``robometer`` git package, no ``auto_map``, no
pinned ``transformers==4.57.1``). To fit an 8 GB GPU we keep OpenRAL's NF4
pre-quantized checkpoint (``OpenRAL/rskill-robometer_4b-any-general-nf4``, ~3.3 GB resident)
and load its packed 4-bit weights DIRECTLY into the native ``nn.Module``:

  * build the native skeleton on the ``meta`` device (no Qwen weight download —
    the empty Qwen3-VL architecture comes from ``lerobot/Robometer-4B``'s baked
    ``vlm_config``);
  * drop the LM vocab projection (``lm_head``) — Robometer reads only hidden
    states, so the 388M-param tied head is dead weight;
  * install empty ``bitsandbytes`` ``Linear4bit`` shells on the large Linears;
  * remap the upstream RBM backbone keys (``model.language_model.*`` /
    ``model.visual.*``) onto the native ``AutoModelForImageTextToText`` nesting
    (``model.model.*``) and ``Params4bit.from_prequantized`` the packed weights;
  * assign the folded non-persistent rotary buffers.

Per-frame decode: the native ``RobometerRewardModel.compute_reward`` returns only
the LAST-frame scalar. OpenRAL needs the full per-frame progress[]/success[]
series (for the reasoner's trend / plateau / stall logic), so we call the
lower-level ``_compute_rbm_logits`` + module-level ``decode_progress_outputs``
directly. The progress head is 10 bins wide (verified against the published
checkpoint); ``decode_progress_outputs`` derives the bin count from the logit
width, so the ``num_bins`` wire field is advisory only (kept for wire compat).

Imported lazily by ``openral_runner.backends.reward.robometer_reward`` so normal
package imports still avoid torch / transformers.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.rewards.robometer.configuration_robometer import RobometerConfig

import _robometer_quant as q

q.set_cublas_workspace_env()  # MUST precede CUDA init (torch import below)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

q.apply_determinism()

# The published checkpoint's native config (Qwen3-VL-4B, 10-bin discrete progress
# head). Only its small config.json is fetched — never the 9 GB bf16 weights; the
# baked ``vlm_config`` gives us the empty architecture for the meta build.
_NATIVE_CONFIG_REPO = "lerobot/Robometer-4B"


def _split_repo_rev(weights: str) -> tuple[str, str | None]:
    """``repo@rev`` -> (repo, rev); a bare repo/path -> (repo, None)."""
    if "@" in weights and not os.path.isdir(weights):
        repo, rev = weights.rsplit("@", 1)
        return repo, rev
    return weights, None


def _resolve_local_dir(weights: str) -> str:
    """Return a local directory for ``weights`` (snapshot_download if it's an HF id)."""
    weights = weights.removeprefix("local://")
    if os.path.isdir(weights):
        return weights
    from huggingface_hub import snapshot_download

    repo, rev = _split_repo_rev(weights)
    return snapshot_download(repo, revision=rev)


def _native_config() -> RobometerConfig:
    """Build the native ``RobometerConfig`` from the published checkpoint's config.

    ``RobometerConfig.from_pretrained`` can't parse the saved ``config.json``
    (draccus rejects its ``type`` field), so we filter the raw JSON down to the
    dataclass fields. The baked ``vlm_config`` (Qwen3-VL, vocab 151,674) means no
    Qwen weight/config download is needed for the meta build.
    """
    from huggingface_hub import hf_hub_download
    from lerobot.rewards.robometer.configuration_robometer import RobometerConfig

    raw = json.load(open(hf_hub_download(_NATIVE_CONFIG_REPO, "config.json")))  # noqa: SIM115 — one-shot
    valid = {f.name for f in fields(RobometerConfig)}
    skip = {"input_features", "output_features", "normalization_mapping"}
    kw = {k: v for k, v in raw.items() if k in valid and k not in skip}
    return RobometerConfig(**kw)


def _remap_backbone_key(key: str) -> str:
    """Upstream RBM key -> native ``AutoModelForImageTextToText`` nesting.

    The pre-quantized checkpoint stores the backbone as ``model.language_model.*``
    / ``model.visual.*`` (upstream ``RBM.model`` == the bare ``Qwen3VLModel``).
    The native module wraps it in a ``…ForConditionalGeneration``, so those live
    one level deeper at ``model.model.*``. Heads / ``frame_pool_attn`` are shared.
    """
    if key.startswith(("model.language_model", "model.visual")):
        return "model." + key
    return key


class _Scorer:
    """Loads the NF4 Robometer model once and scores clips (discrete -> [0,1])."""

    def __init__(self, weights: str, device: str = "cuda") -> None:
        self.device = device
        local = _resolve_local_dir(weights)
        if not q.is_prequantized_checkpoint(local):
            raise RuntimeError(
                f"{local} is not an NF4 pre-quantized checkpoint (no bnb nf4 keys). "
                "The native scorer loads OpenRAL/rskill-robometer_4b-any-general-nf4; the bf16 "
                "lerobot/Robometer-4B is too large for an 8 GB GPU."
            )
        self.cfg, self.model = self._load_prequantized(local)
        self.model.eval()

        from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep

        # Processor from the SAME checkpoint dir: the exact Qwen3-VL tokenizer +
        # image/video preprocessor the weights shipped with. ``max_frames=None``
        # keeps every frame the node-side client sends (it already subsampled to
        # the activation budget), so progress[] length == frames sent.
        self.step = RobometerEncoderProcessorStep(
            base_model_id=local,
            max_frames=None,
            use_multi_image=self.cfg.use_multi_image,
            use_per_frame_progress_token=self.cfg.use_per_frame_progress_token,
        )
        if device == "cuda":
            torch.cuda.synchronize()
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"[robometer] ready (native nf4): {vram:.2f} GB on {device}", flush=True)

    def _load_prequantized(self, local: str) -> tuple[RobometerConfig, Any]:
        """Meta-build the native module, then drop in the packed NF4 weights."""
        from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel
        from safetensors.torch import load_file

        cfg = _native_config()
        cfg.device = self.device
        print(f"[robometer] native meta-build ({cfg.progress_discrete_bins} bins) ...", flush=True)
        with torch.device("meta"):
            model = RobometerRewardModel(cfg)
        # Direct construction defaults to "eager"; production + determinism use "sdpa".
        for c in (
            model.model.config,
            getattr(model.model.config, "text_config", None),
            getattr(model.model.config, "vision_config", None),
        ):
            if c is not None:
                c._attn_implementation = "sdpa"
        # Robometer reads only hidden_states; the LM vocab projection is dead
        # weight (388M tied params) and would otherwise quantize to an empty shell.
        model.model.lm_head = torch.nn.Identity()

        q.install_linear4bit_shells(model, torch.bfloat16)
        raw = load_file(os.path.join(local, "model.safetensors"), device=self.device)
        state = {_remap_backbone_key(k): v for k, v in raw.items()}
        consumed = q.install_prequantized(model, state, self.device)
        leftover = {k: v for k, v in state.items() if k not in consumed}
        model.load_state_dict(leftover, strict=False, assign=True)
        # ``original_inv_freq`` is a non-persistent rope buffer equal to
        # ``inv_freq`` at init (default rope scaling), so it's absent from the
        # checkpoint — seed it so ``assign_meta_buffers`` can fill it.
        for bname, _buf in list(model.named_buffers()):
            if bname.endswith("original_inv_freq") and bname not in state:
                src = bname.replace("original_inv_freq", "inv_freq")
                if src in state:
                    state[bname] = state[src]
        n_buf = q.assign_meta_buffers(model, state, self.device)
        still_meta = [n for n, p in model.named_parameters() if p.is_meta]
        if still_meta:
            raise RuntimeError(f"meta params left after prequant load: {still_meta[:5]}")
        print(f"[robometer] installed NF4 + {n_buf} rotary buffers", flush=True)
        return cfg, model

    @torch.no_grad()
    def score(
        self, frames_rgb: NDArray[np.uint8], task: str, num_bins: int
    ) -> tuple[list[float], list[float]]:
        del num_bins  # advisory only: bin count is derived from the 10-wide head
        from lerobot.rewards.robometer.modeling_robometer import decode_progress_outputs

        encoded = self.step.encode_samples([(frames_rgb, task)])
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in encoded.items()}
        prog_logits, succ_logits = self.model._compute_rbm_logits(inputs)
        decoded = decode_progress_outputs(
            prog_logits, succ_logits, is_discrete_mode=self.cfg.use_discrete_progress
        )
        prog = np.asarray(decoded["progress_pred"][0], dtype=np.float32)
        succ = (
            np.asarray(decoded["success_probs"][0], dtype=np.float32)
            if decoded["success_probs"]
            else np.zeros_like(prog)
        )
        return prog.tolist(), succ.tolist()
