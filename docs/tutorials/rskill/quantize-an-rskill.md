# Quantize an rSkill to fit your GPU

**Goal:** pick the weight precision an rSkill runs at, declare it in the
manifest, and check that your host can run it.

Precision is never silent: it is declared in the manifest, reported in logs and
traces, and gated by the host's capabilities.

| | Open OpenRAL | OpenRAL Pro |
| --- | --- | --- |
| Formats | `fp32`, `fp16`, `bf16` | adds `int8`, `fp8`, `int4`/NF4, `fp4_nvfp4` |
| `tools/quantize_rskill.py` | refuses to run | runs |

Formats of 8 bits or fewer (quantizing at load, loading a pre-quantized pack, or
loading a checkpoint quantized upstream) are part of OpenRAL Pro (ADR-0123).
Without Pro the load raises `ROSConfigError` naming OpenRAL Pro; it never falls
back to bf16. Sections 1-3 below work on the open install; section 4 needs Pro.

## Prerequisites

```bash
just bootstrap && just sync
uv run openral doctor        # shows the host's supported dtypes
```

## 1. Declare the dtype in the manifest

```yaml
runtime: "pytorch"
quantization:
  dtype: "bf16"       # fp32 | fp16 | bf16 (open); int8 | fp8 | int4 | fp4_nvfp4 (Pro)
  backend: "pytorch"  # which backend executes the model
min_vram_gb:
  fp32: 14.0
  bf16: 7.0
```

`quantization.dtype` always declares what **runs**, for every family. A
checkpoint stored at a different precision records that in
`quantization.extra.stored_dtype`.

`min_vram_gb` is a per-dtype map and **informational only**: nothing refuses to
load on it. `openral doctor` and the deploy-sim co-residency preflight read it
to warn before a run, so fill it in honestly.

Two naming traps (Pro formats):

- **`int4` and `nf4` are the same weights.** The schema enum value is `int4`;
  the naming token in a Hub repo id is `nf4`. A manifest may say either.
- **`int8` means LLM.int8, not torchao dynamic int8.** There is no `nf8`.

## 2. Check what your host can run

`rSkill.check_quantization_dtype` compares the manifest's `dtype` with the
host's `gpu_supported_dtypes` and raises `ROSCapabilityMismatch` if it is not in
the set. The set comes from your GPU's compute capability: Blackwell adds
`fp4_nvfp4`, Hopper through Ampere stop at `int4`, Turing drops `bf16`, and
Volta drops both. `fp8` appears in no row: it reaches hardware as Q/DQ nodes in
a TensorRT engine, not as a runtime cast, so an `fp8` manifest fails this check
on every probed host. An empty set skips the check.

```bash
uv run openral doctor                                   # the `ComputeSpec (local) / dtypes` row
uv run openral rskill check <id> --robot robot.yaml     # per-section verdict for one skill
```

`rskill check` prints embodiment, capability flags, GPU runtime, GPU dtype,
sensors and actuators, and whether the skill runs on this host. It catches a
dtype your card cannot execute before any weights move.

## 3. Override the dtype at run time

One variable overrides the dtype for every family:

```bash
OPENRAL_QUANTIZATION_DTYPE=bf16 uv run openral sim run --config … --rskill …
```

Resolution order: `$OPENRAL_QUANTIZATION_DTYPE` → `VLASpec.quantization`
(typed) → `spec.extra["dtype"]` → `quantization.dtype` → the adapter's default.

Each family then checks the result against what its load path supports and
raises `ROSConfigError` (naming the family, the token, its source and the
supported set) instead of loading something else. When the resolved dtype
differs from the declared one, a WARNING names both: the package's `benchmarks:`
numbers describe the declared dtype, not what you ran.

| Family | Accepts |
| --- | --- |
| pi05 | nf4 / int8 / bf16 / fp16 / fp32 / none |
| molmoact2 | nf4 / bf16 / fp16 / fp32 / none |
| openvla | nf4 (CUDA only) / bf16 / none |
| gr00t (in-process) | nf4 / fp32 / none (an unpacked load runs fp32 params) |
| smolvla | bf16 (checkpoint-native split) |
| act / diffusion / xvla / diffuser_actor | fp32 |
| lingbot_va_a1 | bf16 (the external server's precision) |
| rldx / gr00t_b1k / internvla_n1 | nf4 / int8 / none (bf16 → none) |
| xr1 / lingbot_vla / lingbot_vla2 | nf4 / none (bf16 → none) |

`nf4` and `int8` entries need Pro. Sidecars load their `none` mode at bf16, so
`bf16` maps onto it.

## 4. With OpenRAL Pro: 4-bit and 8-bit weights

This section needs the OpenRAL Pro quantization package. NF4 comes from Pro
(`openral-pro-quant` brings `bitsandbytes`); no open dependency group carries it.

### Pick a dtype

| Dtype | Footprint vs bf16 | Use it when |
| --- | --- | --- |
| `bf16` | baseline | The checkpoint fits and you want the reference number |
| `int8` | ~50% | 8-bit is enough |
| `nf4` / `int4` | ~25% | The model will not otherwise fit |

Quantization is not free: `rskills/pi05-libero-int8` notes that LLM.int8 fits
π0.5 in 8 GB but costs task success versus bf16. Measure on your task before
publishing a number ([Run a benchmark](../benchmark/run-a-benchmark.md)).

Only large Linears are quantized: those with at least `--min-params` weight
elements (default 4,000,000; see `tools/quantize_rskill.py`). Biases stay in the
compute dtype.

### Partial scopes

Some adapters quantize only part of the tree. GR00T reads `quantize_scope` from
`quantization.extra` (default `backbone`):

```yaml
quantization:
  dtype: "bf16"              # what the checkpoint ships as
  backend: "pytorch"
  extra:
    quantize_scope: "model"  # whole model, not just the VLM backbone
```

Scope has no runtime override. A GR00T skill that is stored (and named) `…-bf16`
but NF4-packed on load declares `dtype: "int4"` with
`extra: {stored_dtype: "bf16"}`.

### On-line vs pre-quantized

Both paths give 4-bit weights; they differ in load time, not accuracy.

- **On-line (default).** The adapter downloads bf16 weights and rewrites the
  Linears at every launch. This is slow (about 90 s per checkpoint on a
  mobile-class GPU).
- **Pre-quantized.** The packed weights live in the Hub repo next to a
  `quantization_metadata.json` sentinel and install directly, skipping the
  conversion. Pre-quantize any 4-bit rSkill you load more than once.

`int8` has no pre-quantized path; it always runs the on-line rewrite.

### Publish a pre-quantized checkpoint

`tools/quantize_rskill.py` loads a lerobot policy, quantizes it, writes
`model.safetensors` plus the sentinel, and uploads the result. It fails without
Pro.

```bash
HF_TOKEN=<token-with-repo-write> uv run python tools/quantize_rskill.py \
    --source <local-or-hf-lerobot-policy> \
    --target <hf-org>/<rskill-id>-nf4
```

For a policy whose modeling class is not the default, add
`--policy-class lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy` (or
your class). Other flags: `--scheme {nf4,int8}` (default `nf4`; `int8` is
upload-only, with no runtime fast path), `--min-params`, `--skip-upload` (run
the quantization with no token or upload), `--loader`, `--trust-remote-code`.
The tool is Hub-mutating and is not part of CI.

Then point the manifest's `weights_uri` at the new repo and set
`quantization.dtype` to match. The repo id's last segment carries the dtype
(`…-nf4`, `…-int8`); see the
[naming convention](write-and-publish-an-rskill.md#naming-convention).

Some checkpoints ship as a transformers-native pre-quantized layout with an
embedded `quantization_config` and load directly as 4-bit.
`tools/build_qwen_vlm_nf4_checkpoint.py` and
`tools/build_robometer_nf4_checkpoint.py` build that shape, and
`tools/quantize_lingbot_vla2.py` builds the LingBot pack; all need Pro. The
sidecar-venv builders take Pro's config as `--quantization-config`, and the
LingBot and BEHAVIOR GR00T sidecars also take `--pro-quant-module` (the path of
Pro's sidecar rewrites file). Sidecar loaders validate that configuration
before quantizing, and LingBot refuses a packed checkpoint whose launch
settings differ from the recorded ones.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ROSConfigError` naming OpenRAL Pro | The dtype is ≤8-bit. Install Pro, or set `OPENRAL_QUANTIZATION_DTYPE=bf16` (or `fp16`/`fp32`) if the family accepts it. |
| `ROSCapabilityMismatch` on dtype | The host's GPU does not support it; see section 2. |
| `ROSConfigError` naming a family and token | The family's load path does not accept that dtype; see the table in section 3. |
| `quantize_rskill.py` refuses to start | It needs OpenRAL Pro. |

## Next

- [Write & publish an rSkill](write-and-publish-an-rskill.md)
- [Run a benchmark](../benchmark/run-a-benchmark.md) — measure what a dtype costs in task success.
- [rSkill catalogue](../../reference/rskills.md) — the quantization posture of each in-tree entry.
- [VLA compatibility](../../reference/vla_compatibility.md) — observed dims, normalization, per-checkpoint notes.
