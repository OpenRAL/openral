# Quantize an rSkill to fit your GPU

Most VLA checkpoints ship at bf16 or fp32 and assume a datacentre card;
quantization runs them on 8 GB-class hardware. It is never silent: declared in
the manifest, reported in logs and traces, and gated by the host's capabilities
([CLAUDE.md §1.4](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md)).

> **Sub-16-bit formats need OpenRAL Pro (ADR-0123).** Open OpenRAL loads and
> runs `fp32`, `fp16` and `bf16`. `int8`, `fp8`, `int4`/NF4 and `fp4_nvfp4`
> (quantizing at load, loading a pre-quantized pack, or a checkpoint quantized
> upstream) and `tools/quantize_rskill.py` need the OpenRAL Pro quantization
> package. Without it, the load raises `ROSConfigError` naming OpenRAL Pro; it
> never falls back to bf16. The steps below describe a Pro install.

---

## 1. Declare it in the manifest

```yaml
runtime: "pytorch"
quantization:
  dtype: "nf4"        # fp32 | fp16 | bf16 | int8 | fp8 | int4 | fp4_nvfp4
  backend: "pytorch"  # which backend executes the quantized model
min_vram_gb:
  fp32: 14.0
  bf16: 7.0
```

Two `dtype` traps:

- **`int4` and `nf4` are the same weights, different names.** The schema enum
  value is `int4`; 4-bit checkpoints ship as bitsandbytes NF4, so the *naming*
  token in a Hub repo id is `nf4`. A manifest may say either.
- **`int8` means LLM.int8, not torchao dynamic int8.** bitsandbytes offers only
  4-bit and 8-bit Linear classes; there is no `nf8`.

`min_vram_gb` is a per-dtype map and **informational only**: nothing refuses to
load on it. `openral doctor` and the deploy-sim co-residency preflight read it
to warn before a run, so fill it in honestly.

### What actually gates a load

`rSkill.check_quantization_dtype` compares the manifest's `dtype` against the
host's `gpu_supported_dtypes` and raises `ROSCapabilityMismatch` if it is not
in the set. That set is derived from your GPU's compute capability: Blackwell
adds `fp4_nvfp4`, Hopper through Ampere stop at `int4`, Turing drops `bf16`,
and Volta drops both. `fp8` appears in no row: it reaches hardware as Q/DQ
nodes baked into a TensorRT engine, not as a runtime cast, so an `fp8` manifest
fails this check on every probed host. Check yours:

```bash
uv run openral doctor        # the `ComputeSpec (local) / dtypes` row
```

An empty set skips the check, so a host whose probe returned nothing is
permissive, not blocked.

An `nf4` manifest also needs `bitsandbytes`, which ships in the `sim`, `libero`
and `robocasa` groups (not `metaworld` or `maniskill3`). Without it the load
raises rather than silently running at full precision.

---

## 2. Pick a dtype

| Dtype | Footprint vs bf16 | Use it when |
| --- | --- | --- |
| `bf16` | baseline | The checkpoint fits and you want the reference number |
| `int8` | ~50% | 8-bit is enough; lossless on most attention workloads |
| `nf4` / `int4` | ~25% | The model will not otherwise fit at all |

Quantization is not free: `rskills/pi05-libero-int8` notes that LLM.int8 fits
π0.5 in 8 GB **but costs task success versus bf16**. Measure on your task
before publishing a number; a published one is a benchmark result like any
other ([Run a benchmark](../benchmark/run-a-benchmark.md)).

Only large Linears are touched: every `torch.nn.Linear` whose weight has at
least 4M elements (`DEFAULT_MIN_PARAMS_TO_QUANTIZE` in OpenRAL Pro's quantization module). Bias terms stay in the
compute dtype for numerical safety.

### Partial scopes

Some adapters quantize only part of the tree. GR00T reads `quantize_scope` from
the manifest's `quantization.extra`, defaulting to `backbone`:

```yaml
quantization:
  dtype: "bf16"              # what the checkpoint ships as
  backend: "pytorch"
  extra:
    quantize_scope: "model"  # whole model, not just the VLM backbone
```

`rskills/gr00t-n17-b1k-turning-on-radio` uses `model` as one of two measures
that fit it on an 8 GB host (the other is dropping the Qwen3-VL `lm_head`,
which its wrapper never reads). Scope has no runtime override; set it in
`quantization.extra`.

### Declared dtype is the runtime dtype

`quantization.dtype` always declares what **runs**, for every family;
`rSkill.check_quantization_dtype` and the VRAM preflight (`active_min_vram_gb`)
key on it. A checkpoint stored at a different precision
records that in `quantization.extra.stored_dtype`: the GR00T skills are stored
(and *named*) `…-bf16` but NF4-packed on load, so they declare
`dtype: "int4"` with `extra: {stored_dtype: "bf16"}`.

### One override for every family

```bash
OPENRAL_QUANTIZATION_DTYPE=bf16 openral sim run --config … --rskill …
```

Resolution order is `$OPENRAL_QUANTIZATION_DTYPE` → `VLASpec.quantization`
(typed) → `spec.extra["dtype"]` → `quantization.dtype` → the adapter's default.

Each family then checks the result against what its load path supports and
raises `ROSConfigError` (naming the family, the token, its source, and the
supported set) instead of silently loading something else:

| Family | Accepts |
|---|---|
| pi05 | nf4 / int8 / bf16 / fp16 / fp32 / none |
| molmoact2 | nf4 / bf16 / fp16 / fp32 / none |
| openvla | nf4 (CUDA only) / bf16 / none |
| gr00t (in-process) | nf4 / fp32 / none (an unpacked load runs fp32 params) |
| smolvla | bf16 (checkpoint-native split) |
| act / diffusion / xvla / diffuser_actor | fp32 |
| lingbot_va_a1 | bf16 (the external server's precision) |
| rldx / gr00t_b1k / internvla_n1 | nf4 / int8 / none (bf16 → none) |
| xr1 / lingbot_vla / lingbot_vla2 | nf4 / none (bf16 → none) |

Sidecars load their `none` mode at bf16, so `bf16` maps onto it.

Every load logs which source won. When the resolved dtype differs from the
declared one, a **WARNING** names both: the package's `benchmarks:` numbers
describe the declared dtype, not what you just ran.

> This replaced the per-family `OPENRAL_GR00T_QUANTIZATION` /
> `OPENRAL_RLDX_QUANTIZATION` / `OPENRAL_LINGBOT_VLA{,2}_QUANTIZATION`, which
> only ever covered their own sidecar adapters.

---

## 3. On-line vs pre-quantized

Two paths to 4-bit weights; they differ in load time, not accuracy.

**On-line (default).** The adapter downloads bf16 weights and calls OpenRAL
Pro's `quantize_nf4_in_place` (through `openral_rskill.quantization_gate.pro_quantization`), which rewrites the Linears and defers the actual pack
to the next `.to(cuda)`. Costs roughly 90 s per checkpoint on a 4070-mobile,
**every launch**.

**Pre-quantized (fast path).** The packed weights live in the Hub repo
alongside a `quantization_metadata.json` sentinel. Pro's
`load_prequantized_state_for_rskill` spots the sentinel, downloads them, and
installs them directly via `Params4bit.from_prequantized`, skipping the
conversion. It is a no-op on a bf16 repo, so adapters call it unconditionally.

On π0.5 with a warm cache this takes nf4 load from 95 s to 10 s. Pre-quantize
any 4-bit rSkill that will be loaded more than once.

> `int8` has **no** prequant fast path. The SCB sub-state lives inside
> `Int8Params`, which makes a separate Hub artefact brittle, so int8 always
> runs the on-line rewrite.

---

## 4. Publish a pre-quantized checkpoint

`tools/quantize_rskill.py` loads any lerobot policy, quantizes it, writes
`model.safetensors` plus the `quantization_metadata.json` sentinel, and uploads
the result:

```bash
HF_TOKEN=<token-with-repo-write> uv run python tools/quantize_rskill.py \
    --source <local-or-hf-lerobot-policy> \
    --target <hf-org>/<rskill-id>-nf4
```

For a policy whose modeling class is not the default, point at it:

```bash
HF_TOKEN=<token> uv run python tools/quantize_rskill.py \
    --source <hf-org>/<smolvla-finetune> \
    --target <hf-org>/<smolvla-finetune>-nf4 \
    --policy-class lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy
```

The upload is bandwidth-bound (roughly 15–30 minutes for a ~2 GiB nf4 bundle on
home broadband). This one-shot Hub-mutating tool is deliberately **not** part of
CI. `--skip-upload` exercises the quantization without a token or upload;
`--scheme` defaults to `nf4`; `--min-params` is the on-line path's 4M-element
threshold.

Then point your manifest's `weights_uri` at the new repo and set
`quantization.dtype` to match. The repo id's last segment carries the dtype
(`…-nf4`, `…-int8`), so the posture is visible before any download
([Naming convention](write-and-publish-an-rskill.md#naming-convention)).

Some checkpoints instead ship as a transformers-native pre-quantized
`save_pretrained` layout with an embedded `quantization_config`; those load
directly as 4-bit with no bf16 spike. `tools/build_qwen_vlm_nf4_checkpoint.py`
and `tools/build_robometer_nf4_checkpoint.py` build that shape. Both, and
`tools/quantize_lingbot_vla2.py`, need OpenRAL Pro: the sidecar-venv builders
take its config as `--quantization-config` (from
`openral_rskill.quantization_gate.sidecar_quantization_args`).
The LingBot and BEHAVIOR GR00T sidecars rewrite their own Linears, so they also
take `--pro-quant-module`: the path of Pro's sidecar rewrites file
(`sidecar_quantization_args(..., rewrites=True)`), which they load by path
because their venvs cannot install Pro. bf16 / fp16 / fp32 sidecar runs need
neither.

---

## 5. Check it before you run it

```bash
openral rskill check <id> --robot robot.yaml
```

Prints a per-section breakdown (embodiment, capability flags, GPU runtime, GPU
dtype, sensors, actuators) and whether the skill will run on this host. It
catches a dtype your card cannot execute before any weights move.

---

## See also

- [Write & publish an rSkill](write-and-publish-an-rskill.md) — the full
  packaging walkthrough.
- [Run a benchmark](../benchmark/run-a-benchmark.md) — measuring what a dtype
  costs you in task success.
- [`docs/reference/rskills.md`](../../reference/rskills.md) — the in-tree
  catalogue, with the quantization posture of each entry.
- [`docs/reference/vla_compatibility.md`](../../reference/vla_compatibility.md)
  — observed dims, normalization and per-checkpoint notes.

Sidecar loaders and the LingBot pack builder validate Pro’s JSON configuration
before quantizing and apply its bitsandbytes settings (compute dtype, quantization
type, double quantization, storage dtype, and int8 threshold/weight retention).
Skip-module lists accept qualified module paths. LingBot refuses a packed checkpoint
when launch precision settings or skip lists differ from the recorded configuration;
legacy packs use NF4, bf16 compute, double quantization, uint8 storage, and an empty skip list.
