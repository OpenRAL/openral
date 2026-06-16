# Robometer-4B — Phase 0 findings ledger

Gating spike for ADR-0057 (`kind: reward` rSkill). See
`docs/superpowers/specs/2026-06-15-robometer-reward-rskill-design.md` and
`docs/superpowers/plans/2026-06-15-robometer-reward-rskill.md`.

## Source

- **Paper:** *Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons*, arXiv 2603.02115 (pub 2 Mar 2026). Authors: Anthony Liang et al. (USC/UW).
- **Project page:** https://robometer.github.io/
- **Code:** https://github.com/robometer/robometer (README also references `github.com/aliang8/robometer`). Package manager: `uv` (`pyproject.toml`, `setup.py`).
- **Weights:** https://huggingface.co/robometer/Robometer-4B — Apache-2.0, bf16, 8.91 GB (`model-0000{1,2}-of-00002.safetensors`), `model_type: qwen3_vl`, `architectures: ["RFM"]`, no `auto_map`, no Hub-side modeling `.py`.

## `RFM_IMPORT` / class resolution (RESOLVED — confirms vanilla AutoModel will NOT load)

- The actual model class is **`RBM`**, not `RFM`: `class RBM(PredictionHeadsMixin, PreTrainedModel)` in `robometer/models/rbm.py`. `config.json` advertises `architectures: ["RFM"]`; the package's loader maps that name. **Therefore `AutoModel.from_pretrained` cannot instantiate it — the `robometer` package is required.**
- Loader (from `scripts/example_inference_local.py`):
  ```python
  from robometer.utils.save import load_model_from_hf
  from robometer.evals.eval_server import compute_batch_outputs
  from robometer.utils.setup_utils import setup_batch_collator
  from robometer.data.dataset_types import ProgressSample, Trajectory
  exp_config, tokenizer, processor, reward_model = load_model_from_hf(model_path, device=device)
  ```

## `RFM_VENDOR_FILES` (decision: pinned pip install, NOT per-run clone)

- Inference touches a non-trivial subset of the package, so a "single tiny module" vendor is not accurate:
  `robometer/models/{rbm,heads,rewind_transformer,utils}.py`, `robometer/data/dataset_types.py`,
  `robometer/utils/{save,setup_utils,timer,logger}.py`, `robometer/evals/eval_server.py`.
- **Footprint decision:** install the package **once** into the isolated, cached sidecar venv via a pinned
  `pip install "robometer @ git+https://github.com/robometer/robometer@<SHA>"` (or `uv pip install`), not a
  per-run `git clone`. Weights resolve from the HF cache (one-time). This satisfies the "no env bloat / no
  massive repo every run" constraint because the sidecar env is isolated and built once.
- **Pin SHA:** `a669dffc241d7d76bec12f36efd4084d914d017c` (robometer/robometer HEAD as of 2026-06-15). Install: `uv pip install "robometer[robometer,quantization] @ git+https://github.com/robometer/robometer@a669dffc…"`. See `_vendor/PROVENANCE.md`.
- **Deps (from pyproject):** `torch==2.8.0`, `transformers>=4.57`, `xformers==0.0.32.post2`, `decord>=0.6.0`, `qwen-vl-utils[decord]==0.0.14`, `opencv-*-headless`, `bitsandbytes`, `hydra-core`, `trl==0.20.0` (robometer extra). `robometer`/`vlac` extras conflict.

## `RFM_LOAD`

- `load_model_from_hf(model_path="robometer/Robometer-4B", device=...)` → `(exp_config, tokenizer, processor, reward_model)`. `rbm.py` imports `Qwen3VLModel` (needs transformers ≥4.57), `Qwen2_5_VLModel`, `SmolVLMModel`.
- **Empirically confirmed:** loads `class RBM` (`ExperimentConfig`), 4,447,004,940 params (4.03 B trainable), heads `progress_head` / `success_head` / `preference_head` + `frame_pool_attn` all present. Unsloth patches in on import (Unsloth-processed checkpoint). Processor/tokenizer pulled from the base model (none in the HF repo).
- **⚠️ transformers version pin (HARD constraint):** the resolver installs transformers **5.5.0**, which breaks robometer — the processor `__call__` kwargs API changed in 5.x ("Kwargs … have to be in `processor_kwargs` dict"), so the collator drops `input_ids` → `KeyError: 'input_ids'` in `forward_model`. **Must pin `transformers==4.57.1`** (downgrades huggingface-hub to 0.36.2). The production sidecar env must pin this.

## `RFM_INPUT`

- Video frames as `np.ndarray (T, H, W, C)` uint8 + task instruction string → wrapped in `Trajectory` → `ProgressSample` → `setup_batch_collator(...)` → `progress_inputs`.
- Default sampling **fps = 3** (`example_inference_local.py --fps 3`).

## `RFM_INPUT` (empirically confirmed)

- `progress_inputs` (nested under `batch["progress_inputs"]` from the collator) has keys:
  `['attention_mask', 'image_grid_thw', 'input_ids', 'pixel_values', 'resample_attempts']`.
- Tokenizer/processor resolved from base `Qwen/Qwen3-VL-4B-Instruct` (vocab 151643, vision/video pad tokens).
- bf16; xformers attention path (FA2 off). Default fps=3 in the upstream example.

## `RFM_OUTPUT` (RESOLVED + empirically confirmed)

- `compute_batch_outputs(reward_model, tokenizer, progress_inputs, sample_type="progress", is_discrete_mode, num_bins)` returns a dict:
  - `progress_pred` — per-frame progress sequence.
  - `outputs_success` → `{"success_probs": ...}` — **per-frame success probability, 0–1** (nested!). Observed shape `(8,)` for 8 frames, values 0.001–0.013 on random-noise frames (correctly "not succeeding").
- `forward()` (raw) returns `(ModelOutput, timing_raw)` with `progress_logits`/`success_logits` (dicts "A"/"B") + optional `pref_logits`; heads `progress_head`/`success_head`/`preference_head` + `frame_pool_attn`.
- Preference path via `sample_type="preference"` (not probed; trajectory-pair comparison).

## `RFM_PROGRESS_RANGE` (RESOLVED — use DISCRETE mode)

- **Continuous mode** (`is_discrete_mode=False`): `progress_pred` is **raw regression**, shape `(80,)` for 8 frames (finer-than-frame granularity), range observed ≈ **−1.58…3.95** — NOT normalized; would need our own normalization.
- **Discrete mode** (`is_discrete_mode=True, num_bins=100`): `progress_pred` is **per-frame**, shape `(8,)`, values in **0–1** (binned expected value). This is the per-frame normalized progress OpenRAL wants.
- **Decision:** `RewardContract` defaults to **discrete mode** (`num_bins` configurable, e.g. 100) → per-frame progress ∈ [0,1] + per-frame success ∈ [0,1]. `progress_range = [0.0, 1.0]`.

## GATE RESULT: ✅ PASSED (empirically)

- Model loads locally via the pinned `robometer` package (class `RBM`, 4.447 B params, all 3 heads present); a full forward runs on CPU and emits per-frame progress + success. Vanilla `AutoModel` confirmed insufficient. transformers **must** be pinned to `4.57.1`. Discrete mode yields the desired normalized 0–1 per-frame progress + success. Proceed to Phase 1.

## Phase 2 — NF4 quantization (✅ empirically confirmed)

Probe: `_vendor/quant_probe.py` (replicates `openral_sim._quantization.quantize_nf4_in_place`:
`nn.Linear` with `weight.numel() >= 4M` → `bnb.nn.Linear4bit(quant_type="nf4", compute_dtype=bf16)`;
pack on `.to(cuda)`). Run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

- **236** Linear modules rewritten to NF4.
- **NF4 resident VRAM: 3.33 GB** (down from 8.91 GB bf16).
- **Peak VRAM incl. 8-frame @224 forward: 3.56 GB** → **4.44 GB headroom** on the 8 GB GPU.
- Forward still correct post-quant (discrete mode): `progress_pred (8,)` ∈ [0.43, 0.54], `success_probs (8,)` ∈ [0.06, 0.29].
- **Parallel-to-VLA verdict (revised): FEASIBLE on 8 GB.** Robometer NF4 (~3.56 GB peak, 8 frames) + a small NF4 VLA (SmolVLA ~1.5–2 GB) ≈ 5.5 GB. Caveat: activation peak scales with frame-window size / resolution / num_bins — keep the window bounded. The `quantize_rskill.py` `--loader transformers` path will NOT work as-is (no `auto_map`); production packaging must quantize via the `robometer` loader (this probe's path) and teach the sidecar's load-back to use it.

## Environment / host facts

- Reference host: RTX 4070 Laptop, 8 GB (7.8 GB free). Disk: 69 GB free (93% used). HF cache already 162 GB.
- Sidecar env will be isolated at a dedicated venv; do NOT modify the main repo venv.

## GATE STATUS

- **Class-resolution risk (R0): RESOLVED at design level** — load path is known and uses the `robometer` package (pinned pip install). Vanilla AutoModel is confirmed insufficient.
- **Remaining empirical confirmation (Task 0.2/0.3):** install pinned `robometer` into isolated venv → download weights (one-time) → run a CPU/GPU probe to confirm `load_model_from_hf` + `compute_batch_outputs` run with inference-only deps and to capture the exact `progress_pred`/`success_probs` shapes + default progress range. Proceed to Phase 1 only if the probe succeeds.
