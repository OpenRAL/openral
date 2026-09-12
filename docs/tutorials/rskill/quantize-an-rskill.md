# Quantize an rSkill to fit your GPU

Most VLA checkpoints are published at bf16 or fp32 and assume a datacentre
card. Quantization is how OpenRAL runs them on 8 GB-class hardware. This page
covers what the manifest's `quantization` block means, the two ways weights get
quantized, and how to publish a pre-quantized checkpoint of your own.

Quantization is never silent: it is declared in the manifest, reported in logs
and traces, and gated by the host's capabilities
([CLAUDE.md §1.4](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md)).

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

Two things about `dtype` that trip people up:

- **`int4` and `nf4` are the same weights, different names.** The schema enum
  value is `int4`; 4-bit checkpoints ship as bitsandbytes NF4, so the *naming*
  token in a Hub repo id is `nf4`. A manifest may say either.
- **`int8` means LLM.int8, not torchao dynamic int8.** bitsandbytes offers only
  4-bit and 8-bit Linear classes — there is no `nf8`.

`min_vram_gb` is a per-dtype map and is **informational only** — nothing
refuses to load on it. It is what `openral doctor` and the deploy-sim
co-residency preflight read to warn you before a run, so fill it in honestly.

### What actually gates a load

`rSkill.check_quantization_dtype` compares the manifest's `dtype` against the
host's `gpu_supported_dtypes` and raises `ROSCapabilityMismatch` if it is not
in the set. That set is derived from your GPU's compute capability — Ada and
Ampere give you `int4`, Turing drops `bf16`, Volta drops both. Check yours:

```bash
uv run openral doctor        # the `ComputeSpec (local) / dtypes` row
```

An empty set skips the check, so a host whose probe returned nothing is
permissive rather than blocked.

An `nf4` manifest also needs `bitsandbytes` present — it ships in the `sim`,
`libero` and `robocasa` groups (not `metaworld` or `maniskill3`). Without it the
load raises rather than silently running at full precision.

---

## 2. Pick a dtype

| Dtype | Footprint vs bf16 | Use it when |
| --- | --- | --- |
| `bf16` | baseline | The checkpoint fits and you want the reference number |
| `int8` | ~50% | 8-bit is enough; lossless on most attention workloads |
| `nf4` / `int4` | ~25% | The model will not otherwise fit at all |

Quantization is not free, and the tree carries a documented example of the
cost: `rskills/pi05-libero-int8` notes that LLM.int8 fits π0.5 in 8 GB **but
costs task success versus bf16**. Measure on your task before publishing a
number — and if you publish one, it is a benchmark result like any other
([Run a benchmark](../benchmark/run-a-benchmark.md)).

Only the large Linears are touched. The walk rewrites every `torch.nn.Linear`
whose weight has at least 4M elements (`DEFAULT_MIN_PARAMS_TO_QUANTIZE`), and
bias terms stay in the compute dtype for numerical safety.

### Partial scopes

Some adapters quantize only part of the tree. GR00T takes a
`policy_extras.quantize_scope`, defaulting to `backbone`:

```yaml
policy_extras:
  quantize_scope: "model"     # whole model, not just the VLM backbone
```

`rskills/gr00t-n17-so101-fruit` uses `model` to reach 5.8 GiB peak on an 8 GB
card. Override at runtime with `OPENRAL_GR00T_QUANTIZE_SCOPE`.

---

## 3. On-line vs pre-quantized

There are two paths to 4-bit weights, and the difference is load time, not
accuracy.

**On-line (default).** The adapter downloads bf16 weights and calls
`quantize_nf4_in_place`, which rewrites the Linears and defers the actual pack
to the next `.to(cuda)`. Costs roughly 90 s per checkpoint on a 4070-mobile,
**every launch**.

**Pre-quantized (fast path).** The packed weights live in the Hub repo
alongside a `quantization_metadata.json` sentinel.
`load_prequantized_state_for_rskill` spots the sentinel, downloads them, and
installs them directly via `Params4bit.from_prequantized` — skipping the
conversion entirely. It is a silent no-op on a bf16 repo, so adapters call it
unconditionally.

The published effect on π0.5 with a warm cache is 95 s → 10 s for nf4. If you
are shipping a 4-bit rSkill anyone will load more than once, pre-quantize it.

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

Two things to expect: the upload is bandwidth-bound (roughly 15–30 minutes for
a ~2 GiB nf4 bundle on home broadband), and this is a one-shot Hub-mutating
tool that is deliberately **not** part of CI. To exercise the quantization
without a token or an upload, pass `--skip-upload`; `--scheme` defaults to
`nf4`, and `--min-params` is the same 4M-element threshold the on-line path
uses.

Then point your manifest's `weights_uri` at the new repo and set
`quantization.dtype` to match. The naming convention puts the dtype in the
repo id's last segment — `…-nf4`, `…-int8` — so the posture is visible before
anything is downloaded
([Naming convention](write-and-publish-an-rskill.md#naming-convention)).

Some checkpoints instead ship as a transformers-native pre-quantized
`save_pretrained` layout with an embedded `quantization_config`; those load
directly as 4-bit with no bf16 spike. `tools/build_qwen_vlm_nf4_checkpoint.py`
and `tools/build_robometer_nf4_checkpoint.py` build that shape.

---

## 5. Check it before you run it

```bash
openral rskill check <id> --robot robot.yaml
```

Prints a per-section breakdown — embodiment, capability flags, GPU runtime,
GPU dtype, sensors, actuators — and tells you whether the skill will run on
this host. It is the cheap check that catches a dtype your card cannot execute,
before any weights move.

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
