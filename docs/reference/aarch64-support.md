# aarch64 CUDA Hosts (GB10 / DGX Spark, Jetson Thor)

Each out-of-process policy sidecar provisions its own venv from its own pin set,
so "does OpenRAL run on ARM?" is answered per sidecar. This page gives that
answer, plus the wheel-index facts behind it.

Scope: **Linux aarch64 with an NVIDIA CUDA GPU** — NVIDIA GB10 (DGX Spark),
GB200/GH200 SBSA, Jetson Thor. Apple Silicon (no CUDA) and GPU-less aarch64
hosts are out of scope.

## Current routing

| layer | aarch64 | x86_64 |
|---|---|---|
| CUDA index (`openral_sim._sidecar_common.torch_backend()`) | `cu130` | `cu128` |
| workspace venv (`pyproject.toml` overrides) | `torch>=2.13,<2.14`, `torchvision>=0.28,<0.29`; lock resolves `2.13.0+cu130` / `0.28.0+cu130`, `torchcodec 0.11.1` | `torch<2.10`; lock resolves `2.9.1` / `0.24.1`, `torchcodec 0.9.1` |
| policy sidecars | `torch==2.9.1+cu130` (XR-1: `2.10.0`) | `torch==2.9.1+cu128` |
| Cosmos 3 reasoner sidecar | vLLM's own PyPI torch stack | same |

`torchvision` is ABI-locked to a torch minor, so the root pins both.
`uv sync` resolves PyPI torch, whose aarch64 wheel is CPU-only; the root
`pyproject.toml` therefore binds torch and torchvision to the `cu130` index
under an `aarch64 and linux` marker (`--torch-backend` exists only on
`uv pip`). A venv synced under an older routing can hold a CPU-only
`torchvision`; if `torchvision.ops.nms` on a CUDA tensor raises
`NotImplementedError`, run `just sync --frozen --reinstall-package torchvision`.

## Wheel-index facts

PyTorch's `cu128` index has an aarch64 wheel for every recent release
**except 2.8.0**, and the `triton` that 2.8.0 needs has no aarch64 wheel
anywhere. Both gaps close at torch 2.9, which is why every sidecar whose pins
we control targets `2.9.1`.

| package | aarch64 Linux status |
|---|---|
| `torch` 2.7.1 | wheel exists, but needs `triton==3.3.1` (x86_64 only) |
| `torch` 2.8.0 | no wheel (needs `triton==3.4.0`, x86_64 only) |
| `torch` 2.9.x on `cu128` / `cu130` | wheel exists; `triton>=3.5` has aarch64 wheels |
| `torchvision` / `torchaudio` on `cu128` | wheel exists, but the filename has no `+cu128` tag, so grepping the index for it misses it |
| `torchvision` on PyPI | CPU-only build; CUDA `torchvision::nms` raises `NotImplementedError` |
| `torchcodec` < 0.11.0 | x86_64 and macOS only |
| `decord` | x86_64 and Windows only; use `decord2` (same `decord` module) |
| `open3d` | no `cp312` aarch64 build |

## Compute capability: GB10 `sm_121`, Thor `sm_110`

| index | `torch.cuda.get_arch_list()` |
|---|---|
| `cu128` | `sm_80 sm_90 sm_100 sm_120` |
| `cu130` | `sm_80 sm_90 sm_100 sm_110 sm_120 compute_120` |

The two parts need opposite readings of the same torch warning
("capability 12.1 / 11.0 is not in the supported range"):

- **GB10 (`sm_121`):** the warning is about precompiled SASS and is harmless.
  `sm_120` cubins run on `sm_121`. The risk is *runtime compilation*: the
  `cu128` wheels bundle CUDA 12.8 compilers that do not know `sm_121`.
- **Thor (`sm_110`):** a different Blackwell family. The `cu128` arch list has
  no `sm_110` SASS and no `compute_*` entry to JIT from, so every CUDA op
  fails with `no kernel image is available for execution on the device`. Only
  `cu130` works, which is why the workspace and sidecars route aarch64 there.

Check `torch.cuda.get_arch_list()` on any new aarch64 part instead of treating
the warning as advisory.

### Runtime compilers (GB10 on `cu128`)

| compiler | shipped by | ceiling | symptom |
|---|---|---|---|
| `nvrtc` | `nvidia-cuda-nvrtc-cu12==12.8.93` (torch's pin) | `sm_120` | `nvrtc: error: invalid value for --gpu-architecture (-arch)` |
| `ptxas` | bundled in `triton==3.5.1` | `sm_120a` | `ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'` |

This fails late and unevenly: `torch.sum` is precompiled and works, while the
jiterator op `torch.prod` (used by Qwen's vision grid) dies; a
`@torch.jit.script` function only fuses once the profiler warms up, so DA3
depth failed on request #2, not #1; every Triton kernel failed on LingBot-VLA 2.

Two shims fix it without changing the torch build:

- `tools/sidecar_requirements/aarch64-nvrtc-override.txt` — a uv `--overrides`
  file raising nvrtc to 12.9.86 (knows `sm_121`). It carries both marker
  branches (`== "aarch64"` and `!= "aarch64"`), because an override replaces
  every requirement for the package it names. It is passed at install time as
  well, since torch pins nvrtc exactly. On `cu130` it is a no-op: torch then
  requires `nvidia-cuda-nvrtc==13.0.48`, which knows `sm_110` and `sm_121`.
- `nvidia-cuda-nvcc==13.0.88` — a pip-installable `ptxas` that knows `sm_110`
  and `sm_121`. `make_isolated_env` points `TRITON_PTXAS_PATH` at the venv copy
  (`venv_ptxas`), so no host CUDA toolkit is needed. Only sidecars that run
  Triton kernels install it.

## Per-sidecar status

GB10 columns were verified live on a GB10, on the `cu128` routing, then
spot-checked on `cu130` (bf16 matmul, the nvrtc `prod` op and a CUDA `nms` at
capability `(12, 1)`; no model forward re-run). "resolves" under Thor means a
`uv pip install --dry-run --python-platform aarch64-manylinux_2_28
--torch-backend=cu130` resolution, not a run on Thor hardware.

| sidecar | GB10 | Thor | aarch64 handling |
|---|---|---|---|
| `tools/qwen_vlm_sidecar.py` | run | resolves | `qwen_vlm-aarch64.lock` (hash-locked, `cu130`) + nvrtc override. |
| `tools/locateanything_sidecar.py` | run | resolves | `locateanything-aarch64.lock`; `decord2` replaces `decord` (needed by the checkpoint's `trust_remote_code` processor). |
| `tools/da3_depth_sidecar.py` | run | resolves | `open3d` / `pycolmap` have no aarch64 wheel, so a `--no-deps` install recipe plus the nvrtc override. DA3METRIC-LARGE: finite `(378, 504)` depth, 64-66 ms warm on GB10. |
| `tools/depth_estimator_sidecar.py` | not run | not run | no aarch64 handling. |
| `tools/depth_pro2_sidecar.py` | not run | not run | x86_64 only; installs `depth_pro_2` from a pinned GitHub commit with the plain torch backend. |
| `tools/ovb_sidecar.py` | not run | not run | selects `ovb-aarch64.lock` when `torch_backend()` is `cu130`; plus the nvrtc override. |
| `tools/lingbot_vla2_sidecar.py` (v2) | run | run | Overrides upstream's torch 2.8.0 / triton 3.4.0 to 2.9.1 / 3.5.1, marker-drops `torchcodec`, installs with `uv pip install --no-config` (the root's `torch>=2.13` override would otherwise break the resolve) and adds `nvidia-cuda-nvcc==13.0.88` as `ptxas`. Thor: NF4 checkpoint boots in 74 s, `robby_moe` Triton kernels compile for `arch: 110`. |
| `tools/lingbot_vla2_sidecar.py --variant v1` | no | no | `lerobot==0.4.2` caps `torch<2.8.0`; no such torch has aarch64 wheels, and `torchcodec==0.5` is x86-only. |
| `tools/xr1_sidecar.py` | run | run | `torch 2.10.0` / `torchvision 0.25.0` / `torchaudio 2.10.0` on `cu130` plus flash-attn 2.8.3's `cu13torch2.10` aarch64 wheel, named by URL. That wheel has no `sm_110` SASS or PTX, so on Thor `_xr1_server.py` probes flash-attn and loads with `attn_implementation="eager"`. |
| `tools/rldx_sidecar.py` | run | not resolved | `uv sync` fails on `torchcodec==0.4.0`, so aarch64 runs `uv pip install -e <source>` under `rldx-aarch64-override.txt` (torch 2.9.1; `torchcodec` and `flash-attn` dropped) plus the nvrtc override. The wrapper sets `RLDX_ATTN_IMPL=sdpa` only when `flash_attn` is not importable. x86_64 keeps upstream's `uv sync`. |
| `tools/internvla_n1_sidecar.py` | run | resolves | PyPI `torch==2.6.0` is CPU-only on aarch64, so `_PINNED_DEPS` uses `torch==2.9.1` / `torchvision==0.24.1` on the `torch_backend()` index plus the nvrtc override. |
| `tools/cosmos3_reasoner_sidecar.py` | not run | not run | vLLM's own PyPI torch stack (`cosmos3_reasoner.lock`). |
| `tools/robotwin_sidecar.py`, `rlbench_sidecar.py`, `rlbench_3dda_sidecar.py`, `behavior_groot_sidecar.py`, `behavior_scene_sidecar.py` | not run | not run | no aarch64 handling in the sidecar; treat as x86_64 only. |

`isaac_sidecar.py` and `galaxea_a1_ros1_sidecar.py` are not torch-policy
sidecars and are not covered here.

## The workspace venv

The root `pyproject.toml` expresses the CUDA routing two ways (see
[Current routing](#current-routing)):

- an `explicit = true` `[[tool.uv.index]]` for `download.pytorch.org/whl/cu130`,
  with `[tool.uv.sources]` binding **torch and torchvision** to it on
  `aarch64 and linux`. torchvision must also be a direct dependency; otherwise
  it is purely transitive, `[tool.uv.sources]` skips it, and a CPU vision stack
  lands under a CUDA torch;
- `[tool.uv] override-dependencies` raising `nvidia-cuda-nvrtc-cu12` to 12.9.86
  on aarch64 (the `sm_121` jiterator ceiling above).

## No `expandable_segments` on Tegra

`openral deploy sim` / `deploy run` default the CUDA caching allocator to
`expandable_segments:True` for every node, because on an 8 GiB discrete card
the default allocator fragments and OOMs a VLA that otherwise fits. On a
Jetson the setting is left unset (`openral_core.is_tegra_host()`): the iGPU
shares system RAM, so there is no fragmentation headroom to recover, and
torch's expandable path queries NVML GPU-fabric info, which the iGPU cannot
answer. On an AGX Orin that raised `Expected NVML_SUCCESS ==
DriverAPI::get()->nvmlDeviceGetGpuFabricInfoV_(...)` mid-load of π0.5.

An explicit `PYTORCH_ALLOC_CONF` in the operator's environment still passes
through. Every place that enables the setting goes through
`openral_sim._sidecar_common.default_expandable_segments()`; scripts inside an
isolated sidecar venv, and `tools/_demo_env.sh`, repeat the
`/etc/nv_tegra_release` check inline. GB10 keeps the default: its OS is DGX OS,
not L4T. Whether DGX OS ships `/etc/nv_tegra_release` is unverified.

## GStreamer colour conversion on plain L4T

Jetson Thor images ship the L4T multimedia stack (`nvvidconv`) but **not**
DeepStream (`nvvideoconvert`). `nvvideoconvert` lists packed 3-byte `BGR` on
its system-memory src pad; `nvvidconv` lists `BGRx` and no `BGR`, so a leg that
pins `format=BGR` directly on `nvvidconv` never links.

`openral_runner.backends.gstreamer.pipeline.bgr_convert_chain` handles this for
every leg the builder emits: `nvvidconv` is bridged as
`nvvidconv ! video/x-raw,format=BGRx ! videoconvert`, while `nvvideoconvert`
and stock `videoconvert` stay direct-to-`BGR`. The choice is made on the
resolved element name, not on a platform guess.

The NVMM policy leg (`video/x-raw(memory:NVMM)` caps, the default for
`PipelineSpec.enable_nvmm`) needs the private OpenRAL Pro TensorRT/NVMM runtime
to consume the `NvBufSurface`; without it the reader reports a bus error by
design (CLAUDE.md §1.4, no silent fallback). Pass `enable_nvmm: false` in
`backend_params` for a system-memory-only Jetson host.

## Related

- [Sim Environments](sim-environments.md) — how a failed provision surfaces
  (`ROSConfigError` with the quoted `uv` tail).
- [Toolchain](../contributing/toolchain.md) — `just sync` and dependency groups.
