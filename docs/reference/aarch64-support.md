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
| `torchvision` / `torchaudio` on `cu128` | ✅ `manylinux_2_28_aarch64` | **but the filename carries no `+cu128` local tag**, unlike the x86_64 wheel — grepping the index for `+cu128` misses it entirely |
| `torchvision` on **PyPI**, aarch64 | ⚠️ CPU-only | different file from the index build (`7fb7590c…`, 2.39 MB vs `bd33a7cc…`, 8.49 MB). `readelf -d` shows the index build links `libtorch_cuda.so` / `libc10_cuda.so` / `libcudart.so.12` and the PyPI one does not; on the PyPI build `torchvision.ops.nms` on CUDA tensors raises `NotImplementedError: Could not run 'torchvision::nms' with arguments from the 'CUDA' backend` |
| `torchcodec` < `0.11.0` | ❌ x86_64 + darwin-arm64 only | first aarch64 wheel is 0.11.0 |
| `decord` (all versions) | ❌ x86_64 + win only | use `decord2`, which ships aarch64 and installs the same `decord` module |
| `open3d` | ❌ no `cp312` aarch64 build exists | 0.19.0 is x86_64/macOS/win; ≤0.18.0 stops at `cp311` |

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

| index | `torch.cuda.get_arch_list()` |
|---|---|
| `cu128` | `sm_80 sm_90 sm_100 sm_120` |
| `cu130` | `sm_80 sm_90 sm_100 sm_110 sm_120 compute_120` |

**That warning is only half the story, and the benign half.** It is about
*precompiled* SASS, and there it really is harmless — `sm_120` cubins run on
`sm_121`. Verified live on a GB10 (driver 580.126.09, CUDA 13.0) with
`torch==2.9.1+cu128`: bf16 matmul, `torchvision.ops.nms`, a `bitsandbytes` NF4
`Linear4bit` forward and a `flash_attn_func` bf16 forward all succeed.

The dangerous half is silent: **anything compiled at *runtime* for the live
device fails**, because the `cu128` wheels bundle CUDA 12.8 compilers that do
not know `sm_121`. Two separate compilers, two separate failures:

| compiler | shipped by | ceiling | symptom on GB10 |
|---|---|---|---|
| `nvrtc` | `nvidia-cuda-nvrtc-cu12==12.8.93` (torch's exact pin) | `sm_120` | `nvrtc: error: invalid value for --gpu-architecture (-arch)` |
| `ptxas` | bundled inside `triton==3.5.1` | `sm_120a` | `ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'` |

This is not theoretical, and it does not fail cleanly:

- **Every** Qwen scene-VLM query died — `modeling_qwen3_5.py` reduces the vision
  grid with `image_grid_thw.prod(-1)`, and `prod` is a jiterator op. `torch.sum`
  is precompiled and works; `torch.prod` is not and does not.
- **DA3 depth failed on request #2, not #1** — its `@torch.jit.script
  affine_inverse` only gets NVRTC-fused once the profiling executor warms up.
  The worst possible shape for a provider streaming frames.
- **LingBot-VLA v2** hit nvrtc first, then ptxas: *every* Triton kernel failed,
  down to a three-line `tl.store`.

### The fix

Both compilers are swappable without touching the torch build:

- `tools/sidecar_requirements/aarch64-nvrtc-override.txt` — a uv `--overrides`
  file raising nvrtc to **12.9.86**, whose arch list is
  `[50 … 100, 101, 103, 120, **121**]`. It must carry *both* marker branches
  (`== "aarch64"` and `!= "aarch64"`): a uv override replaces every requirement
  for the package it names, so a lone aarch64 line leaves x86_64 with no nvrtc
  at all instead of falling back to torch's pin. Passed at **install** time as
  well as compile time — torch pins `nvidia-cuda-nvrtc-cu12==12.8.93` exactly,
  so uv otherwise rejects the lock's aarch64 line as a conflict.
- `nvidia-cuda-nvcc-cu12==12.9.86` — a pip-installable `ptxas` that does list
  `sm_121 sm_121a`. `openral_sim._sidecar_common.make_isolated_env` points
  `TRITON_PTXAS_PATH` at the venv-local copy (`venv_ptxas`), so no host CUDA
  toolkit is required. Installed only by sidecars that run Triton kernels.

Moving aarch64 to the `cu130` index fixes both too (nvrtc 13.0 and its ptxas
know `sm_121`) and was verified working end-to-end. It was not chosen: it swaps
the whole stack onto a parallel `nvidia-*-cu13` package set and, because
`--torch-backend` is one global choice per compile, would force a
per-architecture lockfile. Replacing two compiler shims keeps the torch build,
the CUDA runtime and all of x86_64 byte-identical.

## Per-sidecar status

| sidecar | aarch64 | notes |
|---|---|---|
| `tools/qwen_vlm_sidecar.py` | ✅ | `torch==2.9.1` + the nvrtc override in `sidecar_requirements/qwen_vlm.lock`. **Live-verified end to end on GB10**: a real image question answered correctly over the ZMQ wire, 3.29 GB VRAM, 90.7 s load. Was the sidecar that exposed the nvrtc ceiling — before the override every single query died, because `modeling_qwen3_5.py` reduces the vision grid with `image_grid_thw.prod(-1)`. |
| `tools/locateanything_sidecar.py` | ✅ | `torch==2.9.1` in `sidecar_requirements/locateanything.lock`. `decord` (x86_64/win-only) was the second blocker after torch, and it is **not** droppable — the checkpoint's `trust_remote_code` `processing_locateanything.py` imports it at top level and transformers' `check_imports` rejects the module without it. Resolved by marker-swapping in `decord2` on aarch64: a maintained Apache-2.0 fork that publishes `manylinux_2_28_aarch64` and installs the same top-level `decord` package. |
| `tools/da3_depth_sidecar.py` | ✅ | not a torch problem — two `depth-anything-3` dependencies have no aarch64 wheel (`open3d`, `pycolmap`), and its scripted `affine_inverse` is where the sm_121 nvrtc ceiling was first found. Fixed by an aarch64-only `--no-deps` install recipe + the shared nvrtc override; verified live on GB10 — see below. |
| `tools/lingbot_vla2_sidecar.py` (v2) | ✅ | upstream `requirements.txt` pins torch 2.8.0 / triton 3.4.0 / torchcodec 0.6.0; the boot helper feeds `uv pip install --overrides` a 2.9.1 / 3.5.1 torch stack (`_V2_OVERRIDES`), with torchcodec marker-scoped off aarch64. Full upstream requirement set verified to resolve on **both** platforms (aarch64: 129 packages, no torchcodec; x86_64: 130 with `torchcodec==0.9.1+cu128`). Also installs `nvidia-cuda-nvcc-cu12` for a `sm_121`-capable `ptxas` — it is the only sidecar running Triton kernels of its own. **Live-verified end to end on GB10** from a fresh home with no manual intervention: real `(50, 14)` action chunk, finite and input-responsive, `min`/`max` bit-identical to the pre-fix baseline; the upstream MoE kernels compile at `arch: sm121` and the primary `robby_moe` path runs (the fallback-only kernels never appear in the Triton cache). 6.97 GB VRAM, 6.2 s warmed chunk. |
| `tools/xr1_sidecar.py` | ✅ | all three install passes verified live on GB10 — see below |
| `tools/lingbot_vla2_sidecar.py --variant v1` | ❌ | `lerobot==0.4.2` caps `torch<2.8.0`; the versions with aarch64 wheels are all outside that cap (2.9.x above it, 2.7.x below it but needs x86-only `triton==3.3.1`). Also pins `torchcodec==0.6.0`, x86-only. Lifting this means moving V1 off lerobot 0.4.2. |
| `tools/rldx_sidecar.py` | ✅ | upstream RLDX-1 packaging, not a torch-version issue — but fixable, contrary to the first read of issue #88. `uv sync` really does die on `torchcodec==0.4.0`, so aarch64 takes an override-driven `uv pip install -e <source>` instead of `uv sync`; `torchcodec` and `flash-attn` are marker-dropped and torch moves to 2.9.1. **Live-verified end to end on GB10** with real `RLWRLD/RLDX-1-FT-LIBERO` weights — see below. |
| `tools/internvla_n1_sidecar.py` | ✅ | was `torch==2.6.0` from plain PyPI (no `--torch-backend`), which on aarch64 is the **CPU** build. The `transformers==4.51.0` bound that was thought to hold it there is not real — 4.51.0 declares `torch>=2.0`, diffusers 0.32.2 `torch>=1.4`, neither has an upper bound — so `_PINNED_DEPS` moved to `torch==2.9.1` / `torchvision==0.24.1` on `cu128` with the shared nvrtc override. **Live-verified end to end on GB10** — see below. |

### RLDX-1 verified live on GB10

RLDX-1 was written off as unfixable in the first pass at issue #88, on three
claims. Re-tested on this host, one held, one was half-true, and one was
backwards:

| claim | verdict |
|---|---|
| `uv sync` fails on `torchcodec==0.4.0` | **True, still.** `uv sync --dry-run` in a fresh clone resolves 168 packages and then dies: *"Distribution `torchcodec==0.4.0` can't be installed because it doesn't have a source distribution or wheel for the current platform … only has wheels for `manylinux_2_28_x86_64`, `macosx_11_0_arm64`."* But torchcodec is a video **dataset** decoder — `rldx/utils/video_utils.py` imports it inside `try/except (ImportError, RuntimeError)` and only reaches it via `video_backend="torchcodec"` on the training / replay / open-loop-eval path. `run_rldx_server` → `RLDXPolicy` never decodes a video; the sidecar is handed decoded uint8 frames over ZMQ. Marker-dropping it costs nothing at inference. |
| `flash-attn` has "no wheel anywhere" | **Half-true, and not the blocker.** PyPI carries only `flash_attn-2.8.3.tar.gz` — correct. But the real wheels live on the GitHub release, which flash-attn's own `setup.py` fetches by `linux_<machine>` + torch minor + cpython tag; v2.8.3 ships 53 of them, including two `linux_aarch64` — and both are **cp312**, while `rldx` pins `requires-python = "==3.10.*"`. (That cp312 aarch64 prebuilt is also what XR-1's "18 s flash-attn build" actually was.) So aarch64+cp310 would need a real hour-plus source build. It never has to: **nothing in `rldx` imports `flash_attn`** — the backbone reaches it through transformers' `ALL_ATTENTION_FUNCTIONS` — and upstream ships the opt-out itself in `rldx/model/modules/backbone/adapter.py`: `_DEFAULT_ATTN_IMPL = os.environ.get("RLDX_ATTN_IMPL", "flash_attention_2")`, documented for "environments that cannot build flash-attn". |
| upstream's Blackwell path (`pixi.toml`) is `platforms = ["linux-64"]`, so it's irrelevant | **Backwards.** pixi is indeed x86_64-only and is not the install mechanism here, but its *contents* are the strongest evidence the fix is safe: for Blackwell upstream themselves bump **torch to 2.8.0+cu128 or 2.10.0+cu130**, **torchcodec to 0.7.0**, and **flash-attn to 2.8.3 source-built** — while leaving `transformers==4.57.0` and every other pin identical. Moving torch off the `pyproject.toml` 2.7.0 pin is upstream's own supported posture on new hardware, not an OpenRAL invention. |

So the aarch64 branch replaces `uv sync` (which cannot succeed) with
`uv pip install -e <source> --torch-backend=cu128` under two `--overrides`
files — `sidecar_requirements/rldx-aarch64-override.txt` (torch 2.9.1 /
torchvision 0.24.1; `torchcodec` and `flash-attn` marker-dropped) and the shared
`aarch64-nvrtc-override.txt`. Same `<source>/.venv`, same Python 3.10, same
wrapper. x86_64 still runs upstream's `uv sync` against upstream's `uv.lock`,
byte-identical. The boot wrapper sets `RLDX_ATTN_IMPL=sdpa` only when
`flash_attn` is genuinely not importable, so the x86_64 venv keeps
FlashAttention-2.

Upstream's full `[project] dependencies` set resolves and installs on aarch64
cp310 — 126 packages, including `deepspeed==0.17.6` from sdist:

```
python 3.10.20   torch 2.9.1+cu128   torchvision 0.24.1   transformers 4.57.0
numpy 1.26.4     triton 3.5.1        bitsandbytes 0.50.0  rldx 1.0.1
nvidia-cuda-nvrtc-cu12 12.9.86       flash_attn: absent   torchcodec: absent
```

Verified end-to-end on GB10 with real `RLWRLD/RLDX-1-FT-LIBERO` weights (13 GiB
on disk, NF4 backbone), booted through `tools/rldx_sidecar.py` and driven over
the real ZMQ REQ/REP + msgpack ndarray wire the `rldx` adapter uses —
`ping` → `reset` → four `get_action` calls with the LIBERO-flat obs contract
(two `(1, 4, 256, 256, 3)` uint8 camera stacks + seven `state.*` scalars +
`annotation.human.action.task_description`):

```
ping   {'status': 'ok', 'message': 'Server is running'}
reset  {'cleared_sessions': []}
action keys: action.{x,y,z,roll,pitch,yaw,gripper}  →  (16, 7) chunk
all four chunks finite; values change with the observation
first request 9.1 s, warmed 8.4-8.6 s;  VRAM 6335 MiB process
```

Run on the exact shipped configuration — `nvidia-cuda-nvcc-cu12` uninstalled,
`flash_attn` and `torchcodec` absent — after `ensure_pip_venv` repaired the
existing venv off the changed sentinel (`repairing stale (dependency spec
changed) venv`), which also exercises the self-repair path.

That is the sidecar's contract satisfied, not a rollout: LIBERO success rate
under sdpa attention has not been measured here. Two caveats worth carrying:

- **Latency.** ~8 s per 16-action chunk is far above the manifest's
  `latency_budget.per_chunk_ms: 1500`. Some of that is the host — three other
  CUDA processes were resident throughout — and some is sdpa instead of
  FlashAttention-2 on an 8 B backbone with a 4-frame × 2-camera video stack.
  Treat the number as "finite and sane", not as a budget measurement.
- **No `ptxas` shim.** Unlike LingBot, this sidecar does *not* install
  `nvidia-cuda-nvcc-cu12`: `~/.triton/cache` gained no entries across the four
  real inferences, so no Triton kernel is compiled on the serving path and the
  `sm_121` assembler ceiling is never reached. The nvrtc override is still
  carried — that one is torch's own jiterator and is not opt-in.

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

### DA3 depth verified live on GB10

The DA3 depth sidecar hit three separate aarch64 walls, none of them the torch
version. The first two are packaging:

| dependency | aarch64 wheel | used by | resolution |
|---|---|---|---|
| `open3d` | ❌ the only cp312 release (0.19.0) is `manylinux_2_31_x86_64` / macOS / win; aarch64 builds stop at 0.18.0 / cp311 | `depth_anything_3.bench.*` only (DTU / ETH3D / ScanNet++ / 7-Scenes evaluators) | dropped — never on the inference path |
| `pycolmap` | ❌ **every** release ever published is x86_64-linux / macOS-arm64 / win_amd64, and there is no sdist to build from | `depth_anything_3/utils/export/colmap.py`, which `api.py` pulls in eagerly via `utils/export/__init__` | import deferred (see below) |

Open3D *does* publish `manylinux_2_35_aarch64` cp312 wheels on its rolling
`main-devel` GitHub release, so that one was solvable two ways; dropping it is
the cheaper of the two. `pycolmap` has a conda-forge `linux-aarch64` build but
no pip route to it, and building it means building COLMAP itself from source.

So `tools/da3_depth_sidecar.py` installs `depth-anything-3` with `--no-deps` on
aarch64 against an explicit pin set (upstream's dependency list minus `open3d`,
`pycolmap`, `xformers` — x86-only and already optional upstream — and
`pre-commit`, plus `addict`, an undeclared upstream dependency that `--no-deps`
stops arriving transitively), then rewrites the one module-level `import
pycolmap` into a deferred proxy. That is a deferred import, not a stub:
`export_to_colmap` still raises the real `ModuleNotFoundError` if anyone calls
it. The rewrite is anchored to `depth-anything-3==0.1.1` and raises rather than
patching blind if upstream moves.

The third wall was the sm_121 nvrtc ceiling described above, and DA3 is where it
first surfaced — in its nastiest form, because the failure starts at request #2:

```
[0] OK    shape=(378,504) min=0.6887 max=1.3081
[1] FAIL  RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)
[2] FAIL  ... and every request after it
```

`@torch.jit.script affine_inverse` (`utils/geometry.py`) is a 4×4 affine inverse
whose two `torch.cat`s the TensorExpr fuser hands to NVRTC — but only once the
profiling executor has warmed up, which is why the first frame is clean. So the
aarch64 install pass also carries `--overrides
sidecar_requirements/aarch64-nvrtc-override.txt`, raising nvrtc to 12.9.86.
Nothing about DA3 is disabled: TorchScript and the fuser stay on and the kernel
is compiled, correctly, for the live device.

Confirmed inside the provisioned sidecar venv — the fuser is not merely
tolerated, it runs:

```
nvrtc supported archs : [50 … 90, 100, 101, 103, 120, 121]   # 121 present
device capability     : (12, 1)  NVIDIA GB10
texpr_fuser_enabled   : True
affine_inverse        : ScriptFunction
TensorExprGroup in last optimized graph : True   # the fused kernel really built
```

Re-enabling the fuser does **not** perturb the output. Same image, same
checkpoint, fuser on vs `PYTORCH_JIT=0`, comparing raw bytes:

```
sha256 fuser ON  = d65b2af66c10a787f6272bd178dbd6bd36ca399131db8e48f68a2396ad09e9d0
sha256 fuser OFF = d65b2af66c10a787f6272bd178dbd6bd36ca399131db8e48f68a2396ad09e9d0
max |Δ| = 0.0        intrinsics byte-identical
```

Verified end-to-end on GB10 — real `depth-anything/DA3-SMALL` weights, real
image, real ZMQ REQ/REP + msgpack wire, **10 consecutive requests, all ok**
(the old failure mode would have killed #2):

```
depth-anything/DA3-SMALL   torch 2.9.1+cu128   nvidia-cuda-nvrtc-cu12 12.9.86
VRAM 0.14 GB   582 MiB process
input 640×480 RGB → depth (378, 504) float32, all finite
min 0.6886822 m   max 1.3080868 m   mean 0.9340287 m   median 0.9099 m
intrinsics fx 466.7730  fy 467.9905  cx 252.0  cy 189.0
load 5.3 s;  first request 1227 ms, warmed 95–134 ms (~9 Hz over the wire)
in-process, no PNG/ZMQ overhead: warmed median 89.6 ms (fuser on) vs
94.6 ms (fuser off) — within run-to-run noise on a contended GPU
```

Latency is *not* a reason to prefer either setting: this host was running four
other CUDA processes throughout, and the fused kernel is 12 elements. The reason
to keep the fuser on is that leaving it off is a permanent behavioural change to
dodge a stale assembler.

Do not treat the `cuda capability 12.1` warning quoted above as a signal for
any of this. It *is* emitted on this host under torch 2.9.1+cu128 (re-checked
directly: `python -c "import torch; torch.zeros(1, device='cuda')"` prints it),
but it fires on the first CUDA call regardless, it is about SASS, and it is
benign. The nvrtc/ptxas ceiling is a separate and completely silent limit —
nothing is logged until a runtime-compiled op actually throws. That is why it
went unnoticed here, and why "the warning is harmless" was a trap rather than a
reassurance.

### InternVLA-N1 verified live on GB10

This sidecar was the last ⚠️ row. `_PINNED_DEPS` carried `torch==2.6.0` with a
plain `uv pip install` (no `--torch-backend`), so on aarch64 uv resolved PyPI's
CPU-only wheel. The ⚠️ ("runs, on CPU") was in fact generous: the server's
default `--device cuda:0` becomes `device_map={"": "cuda:0"}`, and that venv
cannot honour it —

```
$ <old-venv>/bin/python -c "import torch; torch.zeros(1).to('cuda:0')"
torch 2.6.0+cpu   cuda_avail False
AssertionError: Torch not compiled with CUDA enabled
```

— so on this host the pre-fix sidecar could not serve the checkpoint at all
without also being forced onto `--device cpu`. Note also that `torch==2.6.0`
predates the `cu128` index entirely (`--torch-backend=cu128 torch==2.6.0` →
"no version of torch==2.6.0"), so the flag alone would not have been a fix.

The pin's own comment claimed torch was "the newest release transformers 4.51.0
supports". That is not a real constraint — checked against the published
metadata, `transformers==4.51.0` declares `torch>=2.0`, `diffusers==0.32.2`
declares `torch>=1.4`, and `accelerate==1.4.0` declares `torch>=2.0.0`; none of
the three has an upper bound. So the pin moved to the repo-standard
`torch==2.9.1` / `torchvision==0.24.1` on `cu128`, with the shared nvrtc
override on **every** uv pass (this sidecar has four). The `diffusers==0.32.2`
pin is untouched — it is a checkpoint-compatibility pin, not a torch one.

Verified from a **fresh** `--home` (clone, submodule, venv, four install passes,
no manual intervention) with the real `InternRobotics/InternVLA-N1-DualVLN`
weights, real head-camera frames from this rSkill's own recorded run, real DA3
metric depth, over the real ZMQ REQ/REP + msgpack wire, driven by the real
`_InternVLAN1Adapter`:

```
py3.11   torch 2.9.1+cu128   torchvision 0.24.1   transformers 4.51.0
diffusers 0.32.2   bitsandbytes 0.50.0   triton 3.5.1   numpy 1.26.4
nvidia-cuda-nvrtc-cu12 12.9.86   nvrtc archs [50 … 120, 121]

torch.cuda.is_available() True   NVIDIA GB10 (12, 1)
load 106.5 s   VRAM 6.06 GB allocator / 8499 MiB process (nvidia-smi)
21 real steps, every reply finite:
  BODY_TWIST [0, 0, 0, 0, 0, ±0.261799]  = 15.00 deg/s yaw, vx 0.0
  1.0–2.7 s per System-2 replan (budget 2500 ms)
```

The `±0.261799` rad/s turn is the same output the pre-existing x86_64 8 GB
RTX 4070 validation recorded on this frame, and 6.06 GB matches its 6.02 GB —
the bump changed the device, not the model.

The System-1 NextDiT was checked separately, because a torch/diffusers bump is
exactly how a DiT silently loads at the wrong width (the `LuminaFeedForward`
SwiGLU trap this sidecar already pins `diffusers==0.32.2` for). Built standalone
under the new stack and compared against the checkpoint's real tensors:
**330 / 330 parameters present, zero missing, zero extra, zero shape
mismatches**, with `layers.0.feed_forward.linear_{1,3}` at the checkpoint's
`(1024, 384)` — the reduced SwiGLU width, not 1536. It also *runs*: a real S2
latent `(1, 4, 3584)` from a real frame through `generate_traj` gives a finite
`(22, 3)` trajectory in 0.36 s.

> **Unrelated pre-existing gap found while doing that.** Under `--quantization
> nf4` the System-1 branch cannot run at all: `llm_int8_skip_modules` skips
> `traj_dit` / `navdp` / `action_{encoder,decoder}` but not `memory_encoder`,
> whose `nn.TransformerEncoder` self-attention gets bitsandbytes-quantized.
> `F.multi_head_attention_forward` then calls plain `linear()` on the packed
> weight and raises `RuntimeError: self and mat2 must have the same dtype, but
> got BFloat16 and Byte`. The same call succeeds at `--quantization none`, and
> nothing in the failing path is torch-version-dependent, so this is not a
> regression from the bump — but the NF4 rSkill would hit it the first time
> System-2 emits a pixel goal instead of a discrete action.

## The workspace venv itself

The sidecars each pass `--torch-backend=cu128` to their own `uv pip install`,
so they always got real CUDA builds. The **workspace** venv did not: `uv sync`
resolved torch from PyPI, whose aarch64 wheel is CPU-only, and a freshly synced
DGX Spark reported `torch.__version__ == '2.9.1+cpu'`. Every *in-process*
policy — SmolVLA, ACT, diffusion, pi05, GR00T N1.7, MolmoAct2, the Robometer
reward model — ran on CPU.

`--torch-backend` is a `uv pip`-only flag; it does not exist on `uv sync` or
`uv lock`. So the root `pyproject.toml` expresses the same intent two ways:

- an `explicit = true` `[[tool.uv.index]]` for `download.pytorch.org/whl/cu128`,
  with `[tool.uv.sources]` pinning **torch and torchvision** to it under
  `platform_machine == 'aarch64' and sys_platform == 'linux'`. torchvision must
  be pinned too, and must additionally be named as a direct dependency — it is
  otherwise purely transitive (lerobot / timm / open-clip), and
  `[tool.uv.sources]` only binds direct requirements, so the pin would be a
  silent no-op and you would get a CPU vision stack under a CUDA torch;
- an `[tool.uv] override-dependencies` pair raising `nvidia-cuda-nvrtc-cu12` to
  12.9.86 on aarch64 — the workspace hits the same sm_121 jiterator ceiling as
  the sidecars, so without it an in-process policy dies the moment it touches a
  jitted op.

Verified on the GB10 after `just sync`:

```
torch       2.9.1+cu128
torchvision 0.24.1
cuda avail  True
device      NVIDIA GB10 (12, 1)
tv nms cuda [0]        # torchvision CUDA op
jit prod    24         # the nvrtc jiterator path
```

x86_64 resolution is unchanged — still PyPI `torch==2.9.1`, `torchcodec==0.9.1`,
`torchvision==0.24.1`, `nvidia-cuda-nvrtc-cu12==12.8.93`.

> **Upgrading an existing aarch64 venv needs one extra step.** The CPU and CUDA
> torchvision wheels share the version string `0.24.1` (the index build carries
> no `+cu128` local tag), so `uv sync` sees the installed version as satisfying
> the lock and does **not** swap it — you end up with a CUDA torch and a
> CPU-only torchvision, and `torchvision.ops.nms` on a CUDA tensor raises
> `NotImplementedError`. Force it once:
>
> ```bash
> uv sync --frozen --reinstall-package torchvision
> ```
>
> Fresh venvs are unaffected.

## GStreamer colour conversion on plain L4T

Jetson Thor images ship the L4T multimedia stack (`nvvidconv`) but **not**
DeepStream (`nvvideoconvert`), and the two elements do not advertise the same
system-memory src caps. On a Jetson AGX Thor:

```console
$ gst-inspect-1.0 --exists nvvideoconvert; echo $?
1
$ gst-inspect-1.0 --exists nvvidconv; echo $?
0
$ gst-inspect-1.0 nvvidconv          # SRC template, video/x-raw
format: { I420, UYVY, YUY2, YVYU, NV12, NV16, NV24, GRAY8, BGRx, RGBA, Y42B, Y444 }
```

`nvvideoconvert` lists packed 3-byte `BGR`; `nvvidconv` lists `BGRx` and no
`BGR`. So a leg that pins `format=BGR` directly on `nvvidconv` never links:

```console
$ gst-launch-1.0 videotestsrc ! nvvidconv ! video/x-raw,format=BGR ! fakesink
WARNING: erroneous pipeline: could not link nvvconv0 to fakesink0,
  nvvconv0 can't handle caps video/x-raw, format=(string)BGR
```

`openral_runner.backends.gstreamer.pipeline.bgr_convert_chain` owns this
distinction for every leg the builder emits: `nvvidconv` is bridged as
`nvvidconv ! video/x-raw,format=BGRx ! videoconvert`, while `nvvideoconvert`
(and stock `videoconvert`) stay direct-to-`BGR` so DeepStream hosts pay no
extra CPU colour conversion. The choice is made on the resolved **element
name**, not on a platform guess.

Note that the NVMM policy leg itself (`video/x-raw(memory:NVMM)` caps, the
default for `PipelineSpec.enable_nvmm`) still needs the private
`openral-pro-trt` package to consume the `NvBufSurface`; without it the reader
reports a bus error by design (no silent fallback — CLAUDE.md §1.4). Pass
`enable_nvmm: false` in `backend_params` for a system-memory-only Jetson host.

## Related

- [Sim Environments](sim-environments.md) — how a failed provision surfaces
  (`ROSConfigError` with the quoted `uv` tail).
- [Toolchain](../contributing/toolchain.md) — `just sync` and dependency groups.
