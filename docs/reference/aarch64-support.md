# aarch64 CUDA Hosts (GB10 / DGX Spark, Jetson Thor)

OpenRAL's out-of-process policy sidecars each provision their own venv from
their own pin set, so "does OpenRAL run on ARM?" is answered per sidecar, not
once for the repo. This page is that per-sidecar answer, plus the wheel-index
facts behind it.

Scope: **Linux aarch64 with an NVIDIA CUDA GPU** — NVIDIA GB10 (DGX Spark),
GB200/GH200 SBSA, Jetson Thor. It does not cover Apple Silicon (no CUDA) or
aarch64 hosts without a GPU.

## The torch 2.8.0 gap

PyTorch's `cu128` index publishes an aarch64 wheel for every recent release
**except 2.8.0**, and the `triton` version that torch 2.8.0 requires has no
aarch64 wheel on any index either. Both gaps close at torch 2.9:

| package | aarch64 Linux wheel | notes |
|---|---|---|
| `torch==2.7.1+cu128` | ✅ `manylinux_2_28_aarch64` | but requires `triton==3.3.1` ❌ |
| `torch==2.8.0+cu128` | ❌ x86_64 + win_amd64 only | requires `triton==3.4.0` ❌ |
| `torch==2.9.0/2.9.1+cu128` | ✅ `manylinux_2_28_aarch64` | requires `triton==3.5.x` ✅ |
| `torch==2.9.0/2.9.1+cu130` | ✅ `manylinux_2_28_aarch64` | requires `triton==3.5.x` ✅ |
| `triton` ≤ `3.4.0` | ❌ x86_64 only | |
| `triton` ≥ `3.5.0` | ✅ `manylinux_2_28_aarch64` | |
| `torchvision` / `torchaudio` `cu128` | ❌ *no* aarch64 wheel on the index | resolve to the PyPI build, which is CUDA-enabled on aarch64 — `torchvision.ops.nms` on CUDA tensors verified working on GB10 |
| `torchcodec` < `0.11.0` | ❌ x86_64 + darwin-arm64 only | first aarch64 wheel is 0.11.0 |

So torch 2.8.0 is a single-release hole in an otherwise continuous series, and
it is the version several upstream VLA projects happen to pin. Every sidecar
this repo controls the pins for now targets **`torch==2.9.1` on the `cu128`
index** for exactly this reason. Reproduce the gap with:

```bash
uv venv --python 3.12 /tmp/probe && \
uv pip install --dry-run --python /tmp/probe/bin/python \
  --torch-backend=cu128 torch==2.8.0     # fails: no matching platform tag
uv pip install --dry-run --python /tmp/probe/bin/python \
  --torch-backend=cu128 torch==2.9.1     # resolves
```

## GB10 compute capability

GB10 is **`sm_121`**. No PyTorch build ships `sm_121` SASS today, so torch
warns at first CUDA call:

```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

The warning is not fatal — `sm_120` code runs on `sm_121`. Verified live on a
GB10 (driver 580.126.09, CUDA 13.0) with `torch==2.9.1`: bf16 matmul,
`torchvision.ops.nms`, and a `bitsandbytes` NF4 `Linear4bit` forward all
succeed. Both indexes work and both emit the same warning, so there is no
reason to move off `cu128`:

| index | `torch.cuda.get_arch_list()` |
|---|---|
| `cu128` | `sm_80 sm_90 sm_100 sm_120` |
| `cu130` | `sm_80 sm_90 sm_100 sm_110 sm_120 compute_120` |

## Per-sidecar status

| sidecar | aarch64 | notes |
|---|---|---|
| `tools/qwen_vlm_sidecar.py` | ✅ | `torch==2.9.1` in `sidecar_requirements/qwen_vlm.lock`; whole lock resolves on aarch64 (63 packages) |
| `tools/locateanything_sidecar.py` | ✅ | `torch==2.9.1` in `sidecar_requirements/locateanything.lock`. `decord` — an unused video reader inherited from the upstream NVIDIA Space requirements, x86_64/win-only — is marker-scoped off aarch64; it was the second blocker after torch. |
| `tools/da3_depth_sidecar.py` | ❌ | not a torch problem: `depth-anything-3` requires `open3d`, whose only cp312 release (0.19.0) is `manylinux_2_31_x86_64` / macOS / win. No aarch64 build exists. |
| `tools/lingbot_vla2_sidecar.py` (v2) | ✅ | upstream `requirements.txt` pins torch 2.8.0 / triton 3.4.0 / torchcodec 0.6.0; the boot helper feeds `uv pip install --overrides` a 2.9.1 / 3.5.1 torch stack (`_V2_OVERRIDES`), with torchcodec marker-scoped off aarch64. Full upstream requirement set verified to resolve on **both** platforms (aarch64: 129 packages, no torchcodec; x86_64: 130 with `torchcodec==0.9.1+cu128`). |
| `tools/xr1_sidecar.py` | ✅ | all three install passes verified live on GB10 — see below |
| `tools/lingbot_vla2_sidecar.py --variant v1` | ❌ | `lerobot==0.4.2` caps `torch<2.8.0`; the versions with aarch64 wheels are all outside that cap (2.9.x above it, 2.7.x below it but needs x86-only `triton==3.3.1`). Also pins `torchcodec==0.6.0`, x86-only. Lifting this means moving V1 off lerobot 0.4.2. |
| `tools/rldx_sidecar.py` | ❌ | upstream RLDX-1 packaging, not a torch-version issue — `uv sync`s a `pyproject.toml` needing `torchcodec==0.4.0` (x86-only) and a required `flash-attn` with no wheel anywhere. Upstream's Blackwell path (`pixi.toml`) is hard-pinned `platforms = ["linux-64"]`. Not fixable from OpenRAL's side. |
| `tools/internvla_n1_sidecar.py` | ⚠️ | `torch==2.6.0` installs from PyPI (no `--torch-backend`), which on aarch64 is the **CPU** build — the sidecar runs, on CPU. Raising it is bounded by the upstream `transformers==4.51.0` pin. |

### XR-1 verified live on GB10

XR-1 is the only sidecar needing `flash-attn`, which publishes no wheels on
PyPI (only `flash_attn-<v>.tar.gz`), so its `--no-build-isolation` pass is the
one most likely to break on a new platform. It does not: all three passes
complete on aarch64 against torch 2.9.1, and the resulting stack runs on the
GPU.

```
torch 2.9.1+cu128   transformers 4.57.1   bitsandbytes 0.49.2   flash_attn 2.8.3
bitsandbytes Linear4bit (NF4, bf16 compute) forward   → ok
flash_attn.flash_attn_func  bf16 (1,8,4,64)           → ok
```

That is the dependency stack, not an end-to-end XR-1 rollout — the checkpoint
itself has not been run on this host.

## The workspace venv itself

`just sync` resolves torch from **PyPI**, not from a `cu128` index. On x86_64
the PyPI wheel is CUDA-enabled; on aarch64 it is **CPU-only** — a freshly
synced workspace on a DGX Spark reports:

```
>>> import torch; torch.__version__
'2.9.1+cpu'
```

In-process policies (SmolVLA, ACT, diffusion, GR00T N1.7, …) therefore run on
CPU there. The sidecars are unaffected: they pass `--torch-backend=cu128`
explicitly and get real CUDA builds. Pointing the workspace itself at the
aarch64 CUDA index is a separate change from this page's scope.

## Related

- [Sim Environments](sim-environments.md) — how a failed provision surfaces
  (`ROSConfigError` with the quoted `uv` tail).
- [Toolchain](../contributing/toolchain.md) — `just sync` and dependency groups.
