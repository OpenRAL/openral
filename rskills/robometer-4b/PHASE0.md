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
- **Pin SHA:** _TBD in Task 0.1 step 1 (record the commit used)._

## `RFM_LOAD`

- `load_model_from_hf(model_path="robometer/Robometer-4B", device=...)` → `(exp_config, tokenizer, processor, reward_model)`. dtype/trust_remote_code args not surfaced in the example script; confirm empirically in the probe. `rbm.py` imports `Qwen3VLModel` (needs recent transformers; falls back if absent), `Qwen2_5_VLModel`, `SmolVLMModel`.

## `RFM_INPUT`

- Video frames as `np.ndarray (T, H, W, C)` uint8 + task instruction string → wrapped in `Trajectory` → `ProgressSample` → `setup_batch_collator(...)` → `progress_inputs`.
- Default sampling **fps = 3** (`example_inference_local.py --fps 3`).

## `RFM_OUTPUT` (RESOLVED)

- `forward()` returns `(ModelOutput, timing_raw)`; `ModelOutput` carries `progress_logits` (dict "A"/"B"), `success_logits` (dict "A"/"B"), optional `pref_logits`. Heads: `progress_head`, `success_head`, `preference_head`.
- Post-processed via `compute_batch_outputs(reward_model, tokenizer, inputs, sample_type="progress", is_discrete_mode=..., num_bins=...)` → dict with:
  - `progress_pred` — per-trajectory list of per-frame progress (the normalized 0–1 signal we feed the Reasoner).
  - `success_probs` — per-frame success probability.
  - (preference path via `sample_type="preference"`.)

## `RFM_PROGRESS_RANGE`

- Continuous 0–1 progress; a discrete/binned mode exists (`is_discrete_mode`, `num_bins`). _Confirm the default mode + exact range empirically in the probe, then set `RewardContract.progress_range`._

## Environment / host facts

- Reference host: RTX 4070 Laptop, 8 GB (7.8 GB free). Disk: 69 GB free (93% used). HF cache already 162 GB.
- Sidecar env will be isolated at a dedicated venv; do NOT modify the main repo venv.

## GATE STATUS

- **Class-resolution risk (R0): RESOLVED at design level** — load path is known and uses the `robometer` package (pinned pip install). Vanilla AutoModel is confirmed insufficient.
- **Remaining empirical confirmation (Task 0.2/0.3):** install pinned `robometer` into isolated venv → download weights (one-time) → run a CPU/GPU probe to confirm `load_model_from_hf` + `compute_batch_outputs` run with inference-only deps and to capture the exact `progress_pred`/`success_probs` shapes + default progress range. Proceed to Phase 1 only if the probe succeeds.
