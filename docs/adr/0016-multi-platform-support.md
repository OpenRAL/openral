# ADR-0016: Multi-platform support — x86 (CUDA + CPU) and L4T (Orin + older Jetson)

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)
- Related: CLAUDE.md §1.11 (no mocks), §1.13 (METHODS reuse), §1.14
  (docs travel with code), §6.1 (layer discipline), §7.10 (repo state
  map); [ADR-0010](0010-inference-runner.md) (Inference runner);
  [ADR-0011](0011-nvmm-handoff.md) (NVMM handoff). Closes
  [issue #89](https://github.com/OpenRAL/openral/issues/89).

> **Numbering note.** Issue #89's research write-up labelled this work
> "ADR-0012" before [ADR-0012 (Open-core licensing)](0012-open-core-licensing.md)
> landed in PR #87. The next free ADR slot is **0013**, which this
> document takes; nothing else changes.

## Context

PR #86 (`feat(runner): GStreamerSensorReader (M8, ADR-0010 PR I)`)
is the first PR in the repo that explicitly recognises **multiple
compute platforms** at runtime. It ships:

- a `Platform` enum (`TEGRA / NVIDIA_DESKTOP / CPU_ONLY`) at
  `python/runner/src/openral_runner/backends/gstreamer/pipeline.py:60`,
- a `detect_platform()` probe at `pipeline.py:172` that reads
  `/etc/nv_tegra_release`, falls back to `gst-inspect-1.0` for
  `nvh264dec`, then to CPU-only,
- two Dockerfiles (`docker/inference/Dockerfile.x86`,
  `docker/inference/Dockerfile.l4t`) plus an `x86-ros` variant for the
  ROS-tee smoke test,
- `Justfile` targets `docker-build-x86`, `docker-build-l4t`,
  `docker-smoke-x86`, `docker-smoke-x86-ros`.

This is the right *shape* for cross-platform support, but it stops at
the sensor-ingest layer. Other parts of the stack already have
platform-aware probes (`_probe_jetson` /
`DTYPES_BY_COMPUTE_CAPABILITY` / `JETSON_BOARD_TOPS` in
`python/detect/src/openral_detect/probes/gpu.py`,
`auto_select_quant` in
`python/rskill/src/openral_rskill/quantization.py:72`)
that were written in isolation, with no shared contract on **which
targets the runtime supports** or **what "supported" means**.

This ADR establishes that contract. It is the deliverable for PR 1/3
of issue #89; PR 2/3 (detection upgrades) and PR 3/3 (CI / Docker
matrix) cite this ADR as their authority.

### In scope

- **x86** with NVIDIA dGPU (Turing / Ampere / Ada / Hopper / Blackwell).
- **x86** CPU-only (no NVIDIA GPU).
- **Jetson Orin** (AGX / Orin NX / Orin Nano) on JetPack r36 / L4T 36.4.
- **Generic Jetson** — Xavier / Xavier NX — on the same `l4t` image
  with degraded-compute capability (no BF16, no INT4 tensor-core path).

### Out of scope (deferred to a future ADR if/when they land)

- **Thor** and **Spark** (Tegra Blackwell / NVFP4) — covered by the
  existing `SensorReaderBackend.HOLOSCAN` reserved enum value
  ([ADR-0010 Amendment 2026-05-12](0010-inference-runner.md#amendments)).
- **NVIDIA Holoscan SDK** as a sensor-ingest backend (same amendment).
- **DeepStream SDK** — proprietary; disqualified by CLAUDE.md §1.9 / §12.
- **NVFP4** / Blackwell-Tegra-specific quantization dtypes.
- **macOS / Apple Silicon** as a deploy target — the
  `_probe_apple_silicon` probe at `gpu.py:355` stays as a
  *development-host* affordance only.

### What "supported" means in this ADR

A *supported* target is one where, for every milestone delivered on
`main`:

1. The `python/` workspace imports cleanly (`uv sync`).
2. `openral doctor`, `openral detect`, and `openral sim run`'s pure-CPU path exit 0.
3. Where a GPU is present, `openral deploy` against a representative
   `RobotEnvironment` (e.g. `deployments/so100_hello_gstreamer_*`)
   completes with `budget_violations == 0` for ≥30 ticks.
4. There is a CI signal — hosted runner for x86 + `ubuntu-24.04-arm`,
   self-hosted `[self-hosted, l4t]` for L4T — that catches regression
   per category (import-only on hosted aarch64; GPU end-to-end on
   self-hosted).

"Best-effort" means the import path is unbroken and the heuristics are
defensible, but no CI signal is asserted. Maxwell Nano falls under
best-effort (see *Open questions*).

## Decision

### 1. Canonical platform matrix

The three named user targets collapse to **two Dockerfile families**:

| Target name (user-facing) | Arch    | Silicon              | OS / SDK                  | CUDA CC      | Image                |
|---------------------------|---------|----------------------|---------------------------|--------------|----------------------|
| `x86-cuda`                | x86_64  | Ampere+ dGPU         | Ubuntu 24.04 + CUDA 12.x  | 7.5 – 10.0   | `x86` (default)      |
| `x86-cpu`                 | x86_64  | none                 | Ubuntu 24.04              | n/a          | `x86` (`WITH_CUDA=0`)|
| `l4t-orin`                | aarch64 | Tegra Ampere (CC 8.7)| JetPack r36 (L4T 36.4)    | 8.7          | `l4t`                |
| `l4t-xavier`              | aarch64 | Tegra Volta (CC 7.2) | JetPack r35–r36 (degraded)| 7.2          | `l4t` (shared)       |
| `l4t-nano-maxwell`        | aarch64 | Tegra Maxwell        | Legacy JetPack 4.x        | 5.3          | `l4t` (best-effort)  |

**Image consolidation rules:**

1. **Orin and Xavier share `l4t`.** JetPack r36.4 supports AGX Orin /
   Orin NX / Orin Nano natively, and Xavier-class boards in
   degraded-compute mode (CC 7.2 → FP16/INT8 only). The image is one
   Dockerfile; runtime behaviour differs because `_probe_jetson` and
   `auto_select_quant` read the actual compute capability.
2. **x86 stays one Dockerfile** with `WITH_CUDA` build arg controlling
   the `FROM` base
   (`nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04` vs `ubuntu:24.04`)
   and the GStreamer-nvcodec apt set. Avoids a third top-level
   Dockerfile.
3. **No Holoscan branch.** The
   [ADR-0010 Amendment 2026-05-12](0010-inference-runner.md#amendments)
   evaluated Holoscan against lean GStreamer + custom NvBufSurface
   glue and deferred Holoscan. This ADR endorses that deferral.
4. **Maxwell Nano (CC 5.3) is best-effort.** `JETSON_BOARD_TOPS`
   lists it at 0.5 TOPS, but the NVMM zero-copy path through
   `libnvbufsurface.so` is not guaranteed on stripped L4T 32.x images
   and recent JetPack drops dropped its support. The `l4t` image will
   install; `openral doctor` will emit an `info` warning that the NVMM
   path is unavailable; no `openral deploy` GPU end-to-end guarantee is
   made. See *Open questions*.

The matrix is endorsed-final. Two Dockerfiles (`x86`, `l4t`); two
runtime `Platform` values for the NVIDIA branches
(`NVIDIA_DESKTOP`, `TEGRA`) plus `CPU_ONLY`; one canonical mapping
from `(arch, compute_capability)` → image family.

### 2. `Platform` enum is final for this scope

The current three-way `Platform` enum (`pipeline.py:60`) is
**sufficient and final** for the in-scope targets:

- **Orin AGX / Orin NX / Orin Nano** → `TEGRA` (via
  `/etc/nv_tegra_release`).
- **Xavier / Xavier NX** → `TEGRA` (same).
- **Maxwell Nano** → `TEGRA` (same, best-effort).
- **x86 + NVIDIA dGPU** → `NVIDIA_DESKTOP` (via `nvh264dec` probe).
- **x86 CPU-only / WSL2 without GPU passthrough / Apple Silicon dev
  host** → `CPU_ONLY`.

A future contributor must **not** split `TEGRA` into `ORIN` /
`XAVIER` / `NANO`: those distinctions are SoC-level and belong in
`JetsonInfo.cuda_compute_capability` (already populated by
`_probe_jetson`). The `Platform` enum names the GStreamer
*element-family* selector, not the SoC.

### 3. Detection gaps to close (PR 2/3 scope)

PR 2/3 of issue #89 closes the following gaps in
`python/detect/src/openral_detect/probes/gpu.py`:

1. **Replace the `"Orin" in board` heuristic** at `gpu.py:307` and
   `:340` with explicit board-name branches:
   - `"Orin"` → `(8, 7)`.
   - `"Xavier"` → `(7, 2)`.
   - `"Nano"` & not `"Orin"` (Maxwell) → `(5, 3)`.
   - unknown → `None` + structured warning (don't guess `(7, 2)`).
2. **Add `RobotCapabilities.nvmm_available: bool`** populated by
   probing the same paths that
   `python/runner/src/openral_runner/backends/gstreamer/nvbufsurface.py`
   already walks (`/usr/lib/aarch64-linux-gnu/tegra/libnvbufsurface.so`).
   The field surfaces what the runtime needs to decide whether to
   take the zero-copy path. The on-disk `schema_version` stays at
   `"0.1"` while we are pre-publish, so no migrator is required for
   this additive field (CLAUDE.md §1.6).
3. **Recorded fixtures** under `tests/unit/fixtures/jetson/` for
   `_probe_jetson` — one per supported board (Orin AGX / Orin NX /
   Orin Nano / Xavier NX). Real device output, no mocks (CLAUDE.md
   §1.11).
4. **Pin-tests for `auto_select_quant`** at `quantization.py:72`
   asserting the expected dtype on a recorded `DeviceInfo` for each
   target board:
   - x86 CPU-only → `int8_dynamic`.
   - Orin AGX (32 / 64 GB shared) → `bf16`.
   - Orin Nano / Xavier NX (8 GB shared) → `int4`.
   - Xavier (CC 7.2, > 8 GB) → `fp16`.
   - x86 dGPU > 8 GB, CC ≥ 8.0 → `bf16`; CC < 8.0 → `fp16`.

`DTYPES_BY_COMPUTE_CAPABILITY` at `gpu.py:79` already advertises the
correct dtype set for CC 7.2 and CC 8.0 / 8.7; **no code change**
required there.

### 4. Optional-dependency story (PR 3/3 scope, partial)

The root `pyproject.toml` currently exposes a `gstreamer` extra. PR 3/3 makes the extras matrix canonical:

| Extra            | Installs                                  | Used by         |
|------------------|-------------------------------------------|-----------------|
| `gstreamer`      | `PyGObject`, GStreamer bindings           | `x86`, `l4t`    |
| `gstreamer-nvmm` | `pycuda`, ctypes glue                     | `l4t`           |

Every optional import in the codebase must `except ImportError` with a
typed `ROSConfigError` pointing at the right `uv sync --extra` command
— the existing pattern in `python/sim/src/openral_sim/adapters/`.

### 5. CI matrix (PR 3/3 scope)

`.github/workflows/test-python.yml` today runs `ubuntu-22.04` /
`ubuntu-24.04` / `macos-14` (all x86) under Python 3.12. There is no
aarch64 or Jetson signal. PR 3/3 adds:

1. **Hosted x86 Ubuntu** unchanged — unit + ruff + mypy + schema export.
2. **`ubuntu-24.04-arm`** (GitHub's free aarch64 hosted runner, GA 2025).
   Pure-Python lint / mypy / schema parity on arm. No GPU; catches
   aarch64-only import bugs (e.g. the `tegra` library paths in
   `nvbufsurface.py`).
3. **Self-hosted `[self-hosted, l4t]` runner pool**. The HIL
   convention documented in `tests/README.md` already uses
   `[self-hosted, lab-<robot>]`; the same pattern extends here. A
   single `l4t` label covers Orin vs Xavier — runtime behaviour
   differs but CI scheduling does not.
4. **Docker image build matrix** (`docker-build.yml`, new). On every
   push to `main`, build and push `x86` (`--platform linux/amd64`)
   and `l4t` (`--platform linux/arm64`) to GHCR. Buildx + QEMU is for
   image *assembly only*; the smoke tests (`docker-smoke-*`) must run
   on native hardware. CUDA and NVMM do not work under QEMU.

### 6. CLI exposure

`openral doctor` (in `python/cli/src/openral_cli/main.py`) gains a
`Platform` row that prints the consolidated image family the host
matches — one of `x86-cuda` / `x86-cpu` / `l4t-orin` / `l4t-xavier` /
`l4t-nano-maxwell` / `unsupported` — and a one-line reason. The
underlying data already comes through `_check_gpu`; the new row is a
one-line summary derived from it. No new public symbol on the
detection side beyond `RobotCapabilities.nvmm_available` (Decision 3).

## Consequences

### Good

- A future contributor opening `openral deploy` on an Orin Nano can read
  one table and know which image to pull, which dtype the runtime
  will pick, and what CI will catch a regression in.
- The `Platform` enum + the detection probes are pinned to a single
  contract; no more drive-by additions of "is this a Tegra?" probes
  in random files.
- The "no mocks" stance (CLAUDE.md §1.11) is preserved: every
  platform-conditional code path will have a recorded real-device
  fixture under `tests/unit/fixtures/jetson/` *or* a `pytest.skip`
  with a typed reason — never a `MagicMock`.
- The licensing layer added by [ADR-0012](0012-open-core-licensing.md)
  is unaffected: all the modules touched by PR 2/3 and PR 3/3
  (`python/detect/`, `python/rskill/`, `python/runner/`, `python/cli/`,
  `python/core/`) sit firmly in Tier 1 (Apache-2.0).

### Bad / costly

- Maintaining a self-hosted `l4t` runner pool is operational cost; the
  team has to commit to keeping a real Orin board (or two) in CI
  rotation. PR 3/3 documents the runner-onboarding playbook.
- The `RobotCapabilities` additive field (Decision 3) lands on the
  pre-publish baseline (`schema_version: "0.1"`) — every committed
  `robots/*/robot.yaml` must be re-validated, but no migrator is
  required while we are pre-publish (CLAUDE.md §1.6).
- `ubuntu-24.04-arm` hosted runners are still GitHub-Actions-beta-class;
  flakes there may slow merges. PR 3/3 lands the runner with `continue-on-error: true`
  for the first two weeks, then promotes it to required.

### Things this ADR explicitly does NOT decide

- Whether to support Thor / Spark beyond the reserved enum value.
- Whether to ever ship NVFP4 quantization. `DTYPES_BY_COMPUTE_CAPABILITY`
  already lists `FP4_NVFP4` at CC 10.0 but the loader path for it is
  out of scope.
- Whether to add a Mac-arm64 deploy image. Apple Silicon stays a
  development affordance via the existing `_probe_apple_silicon`.
- Whether `gstreamer-nvmm` should be an extra of `openral_runner`
  or a top-level workspace extra. PR 3/3 will resolve this in the
  `pyproject.toml` change.

## Open questions

1. **Is Maxwell Nano (CC 5.3) officially supported, or best-effort?**
   `JETSON_BOARD_TOPS` lists it; the plan treats it as best-effort
   because `libnvbufsurface.so` paths are not guaranteed on the L4T
   32.x family. This ADR provisionally calls it best-effort. PR 2/3
   may promote it to supported if a recorded fixture and a passing
   `openral deploy` end-to-end can be demonstrated on a real Maxwell Nano.

2. **What is the contract for an Xavier in JetPack r35 vs r36?**
   JetPack r36.4 explicitly supports Xavier in degraded-compute mode;
   r35 supported it fully. The image pins r36 — older r35 stacks
   are not tested. PR 2/3 should add a `openral doctor` warning when the
   detected `jetpack_version` is older than r36.

3. **Should `Platform.detect_platform()` accept an injectable path
   for `/etc/nv_tegra_release`?** Today it reads a module-level
   `Final[Path]` constant at `pipeline.py:52`.
   Unit testing it on a non-Tegra dev host requires either monkey-
   patching the constant or running inside a container that doesn't
   exist on PR-build runners. PR 2/3 should refactor the constant
   into an optional parameter so the unit test does not need to
   touch the filesystem.

## Verification

This ADR is research-only. The reviewer checks:

- **§1 image table covers all three user-named targets** — x86 (CUDA
  or CPU-only), Orin (AGX / NX / Nano), generic Jetson (Xavier /
  Xavier NX / Maxwell Nano). 3 / 3 covered, no unmapped device.
- **Every gap in §3 references a concrete file** with line range, so
  PR 2/3 / PR 3/3 implementers can start without re-exploring.
- **No mocks, no smoke tests** in the proposed test plan
  (CLAUDE.md §1.11 / §5.4) — fixtures are recorded real-device
  output, sim tier uses real LIBERO / MetaWorld, HIL tier uses real
  boards.
- **Layer discipline.** No new layer is introduced; the gaps are all
  inside layers that already exist (Detection in Layer 0/1, Skill
  quantization in Layer 3, Runner in the cross-cutting Inference
  Runner spanning HAL / Sensors / Skill). No CLAUDE.md §6.1 layer
  boundary is crossed.

End-to-end smoke on each image family is the exit criterion of PR 3/3,
not of this ADR:

```bash
# x86-cuda (RTX 3090+)
docker run --rm --gpus all \
  -v "$(pwd)/examples:/workspace/examples:ro" \
  openral:x86-latest \
  --config deployments/so100_hello_gstreamer_videotestsrc.yaml \
  --max-ticks 30
# expect: exit 0, budget_violations=0, Platform=NVIDIA_DESKTOP in trace

# x86-cpu (no GPU, WITH_CUDA=0 build)
docker run --rm \
  -v "$(pwd)/examples:/workspace/examples:ro" \
  openral:x86-cpu-latest \
  --config deployments/so100_hello_gstreamer_videotestsrc.yaml \
  --max-ticks 30
# expect: exit 0, budget_violations=0, Platform=CPU_ONLY, no NVMM frames

# l4t on a real Orin
docker run --rm --runtime nvidia \
  -v "$(pwd)/examples:/workspace/examples:ro" \
  openral:l4t-latest \
  --config deployments/so100_hello_gstreamer_videotestsrc.yaml \
  --max-ticks 30
# expect: exit 0, budget_violations=0, Platform=TEGRA,
#         libnvbufsurface.so loaded, CUDA context acquired
```

## Alternatives considered (and rejected)

- **Three Dockerfiles (x86-cuda, x86-cpu, l4t).** Rejected: the
  CUDA/CPU split is purely an apt-package and `FROM` line difference;
  a build arg keeps maintenance to one file.
- **Split `Platform.TEGRA` into `ORIN` / `XAVIER` / `NANO`.** Rejected:
  the enum names a *GStreamer element-family selector*. SoC
  distinctions belong in `JetsonInfo.cuda_compute_capability`, which
  is already populated.
- **Add a `Platform.HOLOSCAN` value now.** Rejected: ADR-0010 Amendment
  2026-05-12 already reserves `SensorReaderBackend.HOLOSCAN`. Adding
  a `Platform.HOLOSCAN` value would conflate the
  *backend selection* (Holoscan vs GStreamer) with the
  *element-family selection* (which CUDA stack to drive).
- **Treat Apple Silicon as a deploy target.** Rejected: no `mps`
  inference adapter; the `_probe_apple_silicon` path exists only so
  `openral detect` and `openral doctor` give useful output on a dev laptop.
- **Pin a single CUDA minor version (12.6) across both images.**
  Considered. The `l4t` image is constrained by what JetPack ships;
  matching x86 to that constraint would lock x86 to an older CUDA
  than the workspace's `torch 2.10` ABI requires. The two images
  ship different CUDA minors (12.x on x86, JetPack-bundled on l4t)
  and the runtime detection code does not care.

## Amendments

### 2026-05-18 — Status flipped Proposed → Accepted

The single-Dockerfile + `PLATFORM` build-arg pattern declared in the
Decision section landed via PR #112 (merge commit `93d28af`) — the
final piece of the consolidation work the 2026-05-14 single-Dockerfile
amendment described. The x86-cuda / x86-cpu / l4t flavors all build
off one `Dockerfile` driven by `--build-arg PLATFORM=...`; CI matrices
in `.github/workflows/` exercise each. The `Platform` enum, runtime
detection helpers, and `JetsonInfo` schema referenced in the Decision
are live in `openral_detect` / `openral_core`.

No behavioural change against the Decision text — only the status field
flips.
