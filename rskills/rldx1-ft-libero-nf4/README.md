---
tags:
  - openral
  - rskill
  - rldx
  - vla
  - franka
  - libero
  - manipulation
  - non-commercial
license: other
license_name: rlwrld-model-license-v1.0
license_link: https://huggingface.co/RLWRLD/RLDX-1-PT
language:
  - en
---

# rskill-rldx1-ft-libero-nf4

> **openral rSkill** — RLWRLD's [RLDX-1-FT-LIBERO](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) (Qwen3-VL-8B backbone + Multi-Stream Action Transformer, ~6.9 B params), packaged for the [openral](https://github.com/OpenRAL/openral) robot agent framework.

This package wraps the upstream RLDX-1 LIBERO finetune with a `rskill.yaml` manifest that adds capability checking, license surfacing, latency budgets, and `openral` registry integration. **It does not copy model weights.** Weights are downloaded directly from the upstream HF repo at first use.

---

## License — read this first

RLDX-1 weights ship under the **RLWRLD Model License v1.0** — a non-commercial, attribution-required license that explicitly forbids military / weapons / non-consensual-surveillance use.

* Loader posture: `RSkillManifest.is_commercial_use_allowed → False`.
* Activation: set `OPENRAL_ALLOW_NONCOMMERCIAL=1` for research use.
* Commercial deployment: requires a separate vendor agreement with RLWRLD.

The wrapping openral code is Apache-2.0; the wire-protocol client is a re-implementation against the upstream Apache-2.0 server source (`rldx/policy/server_client.py`).

---

## Architecture (one paragraph)

RLDX-1 is a vision-language-action model that pairs a **Qwen3-VL-8B-Instruct** visual backbone (frozen tokenizer; 64 "cognition tokens" condensed via a perceiver) with a **Multi-Stream Action Transformer (MSAT)** — an MM-DiT extension running three coupled streams (cognition / physics / action) and a flow-matching action head producing a 16-step action chunk per forward. Real-Time Chunking (RTC) is supported but disabled in this manifest's defaults. See the upstream repo at https://github.com/RLWRLD/RLDX-1 for the model class.

## Supported robots

| Robot | Embodiment tag | Status | Notes |
|---|---|---|---|
| Franka Panda (LIBERO sim) | `franka_panda` | ✓ scaffolded (LIBERO-10 reproduction deferred) | Native eval target for this FT-LIBERO checkpoint. |
| Other 7-DoF arms | — | requires obs-format adapter | State dim is 8-D LIBERO; cameras are `observation.images.camera{1,2}` at 320×180. |

The RLDX-1 family ships sibling rSkills for SIMPLER (Google / WidowX), RoboCasa Kitchen / RC365, GR-1, DROID, and the PT foundation checkpoint — all delegate their architecture / license / quantization sections to this README via the `openral:rskill-readme-delegates-to:` marker.

## Sensors required

| Key | Modality | Min resolution | Format |
|---|---|---|---|
| `observation.images.camera1` | RGB | 320 × 180 | uint8 / `float32 [0,1]` (preprocessor handles either) |
| `observation.images.camera2` | RGB | 320 × 180 | same |
| `observation.state`          | proprioception | (8,) | float32, LIBERO Franka layout (`pos3 + axisangle3 + grip2`) |

Two RGB streams + an 8-D proprio state. The sidecar's `observation` adapter (`openral_sim.policies.rldx`) stacks the last four frames per camera to satisfy the GENERAL_EMBODIMENT 4-step video horizon, so callers only need to publish the current frame per tick.

## Manifest summary

| Field | Value |
|---|---|
| `name` | `OpenRAL/rskill-rldx1-ft-libero-nf4` |
| `version` | `0.1.0` |
| `license` | `rlwrld_non_commercial` (non-commercial, attribution-required; see [License](#license--read-this-first) above) |
| `role` | `s1` |
| `embodiment_tags` | `franka_panda` |
| `runtime` / `quantization.dtype` | `pytorch` / `int4` (NF4 backbone, bf16 head) |
| `weights_uri` | `hf://RLWRLD/RLDX-1-FT-LIBERO` |
| `chunk_size` | 16 (n_action_steps = 8; half-chunk replay) |
| `latency_budget.per_chunk_ms` | 1500 ms (sidecar RTT-bound; backbone is 8 B params) |
| `commercial_use_allowed` | **`false`** |

Full schema: [`openral_core.schemas.RSkillManifest`](../../python/core/src/openral_core/schemas.py).

---

## Auto-managed sidecar (why this rskill is unusual)

RLDX-1 cannot run in-process with openral. We evaluated the in-process path and found it impractical:

1. The `rldx` Python package pins `requires-python = "==3.10.*"` plus strict majors on `numpy==1.26.4` / `torch==2.7.0` / `transformers==4.57.0` / `flash-attn==2.7.4`. Our workspace is **Python 3.12** with `numpy>=2` / `torch>=2.10` / `transformers>=5` (CLAUDE.md §3); downgrading would break smolvla, pi05, xVLA, ACT, DP.
2. `config.json` declares `architectures = ["RLDX"]` — a custom class that exists only in the `rldx` package. The HF checkpoint does not ship `modeling_rldx.py`, so there is no `trust_remote_code` escape from `AutoModel.from_pretrained` requiring the local rldx package.
3. Force-installing `rldx` with `--no-deps` cascades through 15+ packages with major-version-incompatible APIs (albumentations 2.x vs 1.4, lmdb, av, dm-tree, …) — and even past the imports the model load runs through transformers 5.x against rldx code written for 4.57.
4. Reimplementing the policy in openral would mean porting ~25 kLOC of custom Triton kernels + MSAT flow-matching code — out of scope.

So we run RLDX-1 as an **out-of-process sidecar**, communicating over its native ZMQ + msgpack wire protocol (ADR-0010-aligned). The `openral_sim.policies.rldx` adapter **auto-manages the sidecar lifecycle** so end users never invoke the boot helper by hand: it pings the port, forks `tools/rldx_sidecar.py` if no server is up, polls until the server answers, and tears the child down on `close()`. The single 3.10 venv at `~/.cache/openral/rldx-sidecar/source/.venv` is reused across **every** RLDX-1 rSkill — one cached env on disk, not one per checkpoint.

```
                     ┌──────────────────────────────────────┐
 openral sim run …       │  openral main venv (py 3.12)         │
        │            │   ↳ openral_sim.policies.rldx        │
        ▼            │       (ZMQ REQ + msgpack client +    │
 SimRunner ─────────►│        auto-spawn lifecycle)         │
        ▲            └──────────────┬───────────────────────┘
        │                           │  forks once, then
        │                           │  tcp://127.0.0.1:5555
        │                           ▼
        │            ┌──────────────────────────────────────┐
        │            │  rldx sidecar venv (py 3.10,         │
        │            │  ~/.cache/openral/rldx-sidecar/...)  │
        └────────────│   ↳ python -m rldx.eval.run_rldx_server
                     │       --use-sim-policy-wrapper       │
                     │       (Qwen3-VL backbone + MSAT)     │
                     └──────────────────────────────────────┘
```

**End-user workflow — one command:**

```bash
openral sim run --config scenes/benchmark/libero_spatial.yaml --rskill rskills/rldx1-ft-libero-nf4 \
            --rskill rskills/rldx1-ft-libero-nf4
```

The first run on a fresh host clones https://github.com/RLWRLD/RLDX-1, builds the 3.10 venv with `uv sync`, downloads the ~14 GiB bf16 checkpoint, and NF4-quantizes the backbone. Subsequent runs reuse all of that.

**Manual / debugging boot:** if you want to keep a long-running sidecar (e.g. on a shared GPU host) or debug a server crash, set `OPENRAL_RLDX_AUTO_SPAWN=0` and run the helper yourself:

```bash
python tools/rldx_sidecar.py \
    --model RLWRLD/RLDX-1-FT-LIBERO \
    --port 5555 \
    --quantization nf4
OPENRAL_RLDX_AUTO_SPAWN=0 OPENRAL_RLDX_PORT=5555 \
    openral sim run --config scenes/benchmark/libero_spatial.yaml --rskill rskills/rldx1-ft-libero-nf4 \
                --rskill rskills/rldx1-ft-libero-nf4
```

---

## Quantization — 8 GiB GPU?

Stock weights are **bf16, ~14 GiB on disk**. The sidecar `--quantization nf4` flag NF4-quantizes the Qwen3-VL backbone via bitsandbytes (weight-only, compute_dtype=bf16) and **leaves the MSAT diffusion head at bf16** — quantizing the flow-matching head silently corrupts action quality. Approximate VRAM:

| Path | Weights | Activations + KV | Total peak | Fits 8 GiB? |
|---|---|---|---|---|
| `bf16` (stock) | ~14 GiB | ~3–5 GiB | ~18–20 GiB | ❌ |
| `int4` backbone + bf16 head (this manifest) | ~4 GiB | ~3 GiB | **~7 GiB** | ⚠ tight |
| `int4` everywhere (NOT recommended) | ~3.5 GiB | ~3 GiB | ~6.5 GiB | ✅ but actions degrade |

**Measured on the bundled RTX 4070-mobile 8 GiB:**

* Stock bf16 path OOMs during state-dict load (~7.45 GiB allocated of 7.62 GiB usable, before the MSAT head even allocates).
* **`--quantization nf4` works and fits** — VRAM levels off at **6.3 GiB after weight load** on the 8 GiB card. The launcher applies NF4 via a `transformers.AutoModel.from_pretrained` monkey-patch (see `tools/rldx_sidecar.py`). The upstream `policy_loader.py:172` loads the whole stack through a single `AutoModel.from_pretrained` call, so the monkey-patch quantizes the MSAT diffusion head together with the Qwen3-VL backbone — a known concern for action quality that we accept as the price of fitting on 8 GiB (pass `--quantization none` on a ≥12 GiB card for the clean reference path).
* The sidecar ships, listens on `tcp://127.0.0.1:5555`, answers `ping` over ZMQ+msgpack, and the openral `rldx` adapter connects end-to-end.

**Open: end-to-end inference round-trip on FT-LIBERO.** The shipped FT-LIBERO checkpoint's bundled processor_config.json registers `general_embodiment` (4-frame video horizon `[-6,-4,-2,0]`, OXE-style state keys: `eef_pos_absolute` / `eef_rot_absolute` / `gripper_close`), not `libero_panda`. The openral adapter has been updated to stack 4 historical frames per camera, but the upstream `policy_loader.py` still rejects `EmbodimentTag.LIBERO_PANDA` against this checkpoint's processor (`KeyError: 'libero_panda'`) — so we boot with `--embodiment-tag GENERAL_EMBODIMENT` and rely on `RLDXSimPolicyWrapper`'s LIBERO↔OXE key alias path (rldx/policy/rldx_policy.py:300-317 + 525-538). Closing the loop on a `openral sim run` rollout requires one more pass to confirm the action-key reverse-map works and the chunk replays cleanly; see `eval/libero_10.json` `source.status: deferred` for the reproduction CLI.

**Gripper convention (GH-133).** The policy emits `action.gripper` in the RLDS dataset convention (`[0, 1]`, `0=close`, `1=open`); LIBERO/robosuite's OSC controller consumes `[-1, +1]` with `-1=open` / `+1=close`. The openral `rldx` adapter mirrors the upstream `rldx/eval/sim/LIBERO/libero_env.py::{normalize,invert}_gripper_action` pair inside `_assemble_libero_chunk` (helper `_rldx_gripper_to_libero`) so the gripper actuates correctly. Before that fix, every command landed at ~0 and the Franka gripper never opened or closed.

Recommended GPU: **≥12 GiB VRAM** for production runs; 8 GiB is the lower bound the scaffold was validated against.

---

## Quick start

```python
import os
os.environ["OPENRAL_ALLOW_NONCOMMERCIAL"] = "1"

from openral_rskill.loader import rSkill
pkg = rSkill.from_pretrained("OpenRAL/rskill-rldx1-ft-libero-nf4")

# Or offline:
pkg = rSkill.from_yaml("rskills/rldx1-ft-libero-nf4/rskill.yaml")
print(pkg.manifest.weights_uri)  # → hf://RLWRLD/RLDX-1-FT-LIBERO
print(pkg.manifest.is_commercial_use_allowed)  # → False
```

---

## Eval reproduction

The `eval/*.json` blocks in this package are scaffolded with `reproduced_locally: false` and `status: deferred`. To close the loop on a `≥ 12 GiB` GPU — the adapter auto-spawns the sidecar, so it's a single command:

```bash
openral benchmark run \
    --suite libero_10 \
    --rskill rskills/rldx1-ft-libero-nf4
```

`openral benchmark run` writes a fresh `eval/libero_10.json` with `reproduced_locally: true` and the measured per-task success rates.

---

## Provenance

* Upstream weights: [RLWRLD/RLDX-1-FT-LIBERO](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO)
* Base model: [RLWRLD/RLDX-1-PT](https://huggingface.co/RLWRLD/RLDX-1-PT)
* Visual backbone: Qwen3-VL-8B-Instruct
* Source code: https://github.com/RLWRLD/RLDX-1 (Apache-2.0)
* Collection: https://huggingface.co/collections/RLWRLD/rldx-1
