# ADR-0069 — Compute deployment targets: edge / local / cloud

- **Status:** Proposed 2026-06-23.
- **Date:** 2026-06-23
- **ADR number:** `0069` (0068 is the highest accepted).
- **Related:**
  - ADR-0008 — `openral_detect` auto-provisioning package: the origin of `ComputeSpec` population.
  - ADR-0010 — Inference runner (S1 fast policy); the "hot path" deployment target that defaults to edge.
  - ADR-0016 — Multi-platform support (x86 CUDA + L4T/Jetson); informs what `compute_edge` looks like on Jetson.
  - ADR-0018 — ROS 2 reasoner (S2 slow reasoning); the deployment target for `compute_cloud`.
  - ADR-0046 — NVIDIA GR00T out-of-process backend; precedent for out-of-process compute with ZMQ/SSH.
  - ADR-0065 — cuMotion GPU gate (`supports_cumotion()`); the immediate consumer that must be target-aware.
  - ComputeSpec extraction PR (June 2026): split GPU/runtime fields out of `RobotCapabilities` into `ComputeSpec`. This ADR extends that work.

---

## Context

### The single-slot problem

The June 2026 `ComputeSpec` extraction moved GPU/runtime fields from `RobotCapabilities` into a new `ComputeSpec` model attached to `RobotDescription.compute`. That was a correct separation of concerns (physical robot capabilities vs. runtime host compute). It left one open problem: **`RobotDescription.compute` is a single, undifferentiated slot**. It cannot express *where* the compute lives.

In practice, OpenRAL deployments run compute across three distinct tiers:

| Tier | Description | Example hardware |
|---|---|---|
| **Edge** | Compute physically on the robot | Jetson AGX Orin, NVIDIA Thor, Isaac ARC |
| **Local** | Laptop / workstation tethered to the robot or running the simulation | RTX 4090 dev machine, MacBook with Apple Silicon |
| **Cloud** | Remote compute reachable via SSH or HTTP | NVIDIA DGX, AWS p4d.24xlarge, OpenRAL fleet inference server |

The current schema collapses these into one blob. This causes three concrete problems:

**Problem 1 — Simulation host is invisible.**
When `openral detect` runs on a laptop to produce a `so100_follower/robot.yaml`, the SO-100 itself has no GPU (no edge compute). The laptop has an RTX 4090. With the single-slot design, `openral detect` is forced to choose: populate `compute` with the laptop's GPU (wrong — it's not on the robot) or leave it `None` (wrong — skills then get the "unknown, accept all" skip). Neither is correct.

**Problem 2 — rSkill routing has no compute target.**
`RSkillManifest.role: s1 | s2 | s0` declares *when* a skill runs in the dual-system architecture, but not *where* its inference should execute. An S1 fast policy that weights 3 GB should run on the edge Jetson (low latency, no network hop). An S2 reasoning step that needs a 70B LLM should run on cloud. There is no schema field binding role to target.

**Problem 3 — Compatibility checks use the wrong compute.**
`rSkill.check_capabilities(manifest, caps, compute=robot.compute)` passes a single `ComputeSpec` to the runtime/dtype gate. If `robot.compute` is the edge Jetson (INT8, 275 TOPS) but the skill is cloud-dispatched (the cloud host has FP16/FP8 and 320 GB VRAM), the check will wrongly reject skills the cloud can run and wrongly accept skills the Jetson cannot.

### Dispatch topology (from CLAUDE.md §3)

> The dispatcher (edge/cloud/split), like every OpenRAL package, is Apache-2.0. Deadline fallback mandatory.

The dual-system pattern already implies:
- **S0** (cerebellar, 500–1000 Hz) → C++ only, must be on the robot (edge).
- **S1** (fast policy, 30–200 Hz action chunks) → inference runner; default edge, fallback local.
- **S2** (slow reasoning, ~0.2 Hz) → reasoner core; can be cloud.

This ADR makes that implicit topology **explicit and enforceable in the schema**.

---

## Decision

### D1 — Replace `RobotDescription.compute` with three named slots

```python
class RobotDescription(BaseModel):
    # ... existing fields ...

    # Replaces compute: ComputeSpec | None = None
    compute_edge:  ComputeSpec | None = None
    """On-robot SoC compute (Jetson, Thor, Isaac ARC, …).
    Populated by ``openral detect`` when an embedded accelerator is detected on
    the local host (jtop backend) or declared statically in robot.yaml for
    robots with known fixed hardware. None for arm-only robots without onboard GPU."""

    compute_local: ComputeSpec | None = None
    """Workstation / laptop compute on the machine that runs ``openral detect``.
    Always populated by ``openral detect`` (a laptop with no GPU gets a
    CPU-only ComputeSpec with pytorch/onnx/gguf runtimes). None only in a
    static robot.yaml that has never been through ``openral detect``."""

    compute_cloud: ComputeSpec | None = None
    """Remote compute reachable via SSH or HTTP.
    Populated by ``openral detect --cloud <target>`` when a remote endpoint is
    specified. None when no cloud backend is configured."""

    compute_cloud_endpoint: str | None = None
    """Address of the cloud compute endpoint.
    Format: ``ssh://<user>@<host>[:<port>]`` for SSH-probed endpoints, or
    ``http[s]://<host>:<port>`` for HTTP inference servers.
    Required when compute_cloud is set; None otherwise."""
```

**Migration:** `compute` is removed. All existing robot YAMLs have `compute: null` (or absent) — no data loss. A one-time migration note in the schema docstring covers in-tree YAMLs; `from_yaml` validation catches any YAML that uses the old field name via Pydantic's `extra="forbid"`.

### D2 — `openral detect` populates edge and local from the local probe

The detect-and-assemble flow (`_enrich_compute` / `build_compute_spec`) is split into two paths:

```
detect_hardware() → GpuProbeResult
  ├─ gpu.jetson is not None  → compute_edge  (Jetson: TOPS, CC, RAM, nvmm)
  ├─ gpu.apple_silicon       → compute_local (M-series: unified mem, MLX)
  ├─ gpu.nvidia (discrete)   → compute_local (RTX/Tesla: VRAM, CC, dtypes)
  └─ CPU only                → compute_local (pytorch/onnx/gguf, no GPU fields)
```

**Rationale for the split:** A Jetson is always an *onboard* SoC → edge. A discrete NVIDIA GPU on the machine running `openral detect` is *not* on the robot → local. Apple Silicon is the detection host → local.

**Special case — detection on the robot itself (e.g. `openral detect` run directly on a Jetson):** Jetson SoC → `compute_edge`. If the operator is running detect *from* the Jetson, that is the edge host; `compute_local` and `compute_edge` will point to the same hardware. The assembler must not de-duplicate them; the operator can clear `compute_local` in the manifest if they want to express "edge-only robot".

### D3 — `openral detect --cloud <endpoint>` populates `compute_cloud`

Cloud endpoints are declared as:
- **`ssh://user@host[:port]`** — the assembler opens an SSH connection, runs `openral detect --no-write --json` on the remote host, receives the `DetectionReport` JSON, and calls `build_compute_spec` on it. The resulting `ComputeSpec` is stored in `compute_cloud`; the endpoint string is stored in `compute_cloud_endpoint`.
- **`https://host:port`** — a future HTTP inference server endpoint. The `ComputeSpec` is declared manually in `robot.yaml` (no live probe yet). `compute_cloud_endpoint` stores the URL.

SSH probing is guarded by a `--cloud` flag; it is **never run automatically** without operator intent. The `DetectionReport` received over SSH contains sensitive hardware data and must not be logged outside of structlog at DEBUG level.

### D4 — `RSkillManifest` gains a `deployment_target` field

```python
class RSkillManifest(BaseModel):
    # ... existing fields ...
    deployment_target: Literal["edge", "local", "cloud"] = "edge"
    """Where inference for this skill executes.
    - ``edge``:  on-robot SoC (S0/S1 default). Falls back to ``local`` when
                 ``RobotDescription.compute_edge`` is None.
    - ``local``: laptop/workstation tethered to the robot or running sim.
    - ``cloud``: remote endpoint declared via ``compute_cloud_endpoint``.
    S0 and S1 skills must be ``edge`` or ``local`` (the loader rejects ``cloud``
    for S0; S1 ``cloud`` raises a warning and requires explicit opt-in via
    ``OPENRAL_ALLOW_CLOUD_S1=1``).
    S2 skills may be any target; ``cloud`` is the default for large LLM-backed
    reasoner calls.
    """
```

**Constraint table:**

| `role` | Allowed `deployment_target` | Default |
|---|---|---|
| `s0` | `edge`, `local` | `edge` |
| `s1` | `edge`, `local`, `cloud` (with env var opt-in) | `edge` |
| `s2` | `edge`, `local`, `cloud` | `cloud` |

The loader (`rSkill.check_compatibility`) enforces this table at load-time:
- S0 + `cloud` → `ROSCapabilityMismatch` (hard reject, no env var override).
- S1 + `cloud` without `OPENRAL_ALLOW_CLOUD_S1=1` → `ROSCapabilityMismatch` with a clear message.

### D5 — Compatibility checks select the correct `ComputeSpec` slot

`rSkill.check_runtime`, `check_quantization_dtype`, `check_capabilities`, and `check_compatibility` accept a new `deployment_target` argument and select the right slot:

```python
def check_runtime(manifest, caps, *, compute: ComputeSpec | None = None,
                  deployment_target: Literal["edge","local","cloud"] = "edge") -> None: ...
```

`check_compatibility(manifest, robot)` resolves the slot automatically:

```python
def _resolve_compute(robot: RobotDescription,
                     target: Literal["edge","local","cloud"]) -> ComputeSpec | None:
    match target:
        case "edge":
            # Fall back to local when no edge compute is declared.
            return robot.compute_edge or robot.compute_local
        case "local":
            return robot.compute_local
        case "cloud":
            return robot.compute_cloud
```

The "unknown — skip" semantics (empty `gpu_supported_runtimes` / `gpu_supported_dtypes`) are preserved: if the resolved slot is `None`, runtime/dtype checks are skipped.

### D6 — `maybe_inject_cumotion_pipeline` uses `compute_edge` with local fallback

`ROSActionRskill` calls `maybe_inject_cumotion_pipeline` with the resolved edge-or-local compute:

```python
compute = description.compute_edge or description.compute_local
maybe_inject_cumotion_pipeline(goal, interface_type=..., compute=compute)
```

cuMotion requires Ampere+ *on the machine running `move_group`*. In a tethered deployment that is the laptop (local); in a full robot deployment it is the Jetson (edge). The fallback chain `compute_edge or compute_local` correctly handles both.

### D7 — `openral doctor` and `openral detect` output shows all three slots

`openral doctor` emits three `ComputeSpec` blocks (edge / local / cloud) in place of the single block added in the June 2026 doctor patch. Empty slots display as `absent`. `openral detect` writes all three to the assembled `robot.yaml`.

---

## Consequences

**Positive:**
- The simulation host's GPU is now correctly attributed (local, not edge) — runtime/dtype checks work in sim.
- Skill routing is explicit and enforceable — no more silent mismatch between where a skill runs and what hardware is checked.
- Cloud inference becomes a first-class deployment path with a real schema contract.
- The existing "unknown → skip" fallback is preserved for static manifests without any compute slots filled.

**Negative / risks:**
- **Schema migration:** `compute` → `compute_edge` + `compute_local` + `compute_cloud` is a backward-incompatible field rename. Any YAML hand-authored with `compute:` will fail `extra="forbid"` validation. In-tree YAMLs are all `compute: null` (no data), so migration is a schema-version bump with a one-line migrator.
- **SSH probe surface:** D3 introduces a new network operation. It must be gated behind `--cloud`, never automatic, and must never log the remote `DetectionReport` at INFO level (sensitive hardware info).
- **S1 cloud opt-in adds an env var:** `OPENRAL_ALLOW_CLOUD_S1=1` is a deployment-specific override. It must be documented in the runner README and not set by default anywhere in the OpenRAL codebase.
- **`_resolve_compute` edge→local fallback** means a tethered laptop deployment continues to work after the migration with no manifest changes — this is the correct graceful degradation. But it means a Jetson-equipped robot that is also tethered will use `compute_edge` for edge skills and silently ignore `compute_local` for them, which is the desired behavior.

---

## Implementation plan

Each phase is an independently mergeable PR. Phase 1 is the only mandatory unlock for the others.

**Phase 1 — Schema + migration (mandatory, ADR ratification gate)**
- Rename `RobotDescription.compute` → `compute_edge` + `compute_local` + `compute_cloud` + `compute_cloud_endpoint`.
- Bump `schema_version` → `"0.2"` (backward-incompatible field rename; first bump since publish).
- Ship a migrator: `python tools/migrate_robot_yaml.py <path>` reads a v0.1 YAML with `compute:` and writes a v0.2 YAML with `compute_edge:` or `compute_local:` based on the detected hardware type in `onboard_compute["gpu_probe"]`.
- Update `rSkill.check_runtime` / `check_quantization_dtype` / `check_capabilities` / `check_compatibility` to accept `deployment_target` and call `_resolve_compute`.
- Update `_enrich_compute` / `build_compute_spec` in `assemble.py` to populate `compute_edge` vs `compute_local`.
- Update `maybe_inject_cumotion_pipeline` per D6.
- Update all tests (test_capabilities_gpu_fields, test_detect_assemble, test_cumotion_pipeline_select, test_doctor, test_schemas_fuzz).
- Update `docs/methods/00-core-schemas.md` + `docs/methods/04-rskill.md`.
- Update repo-state-map.

**Phase 2 — `RSkillManifest.deployment_target` + loader enforcement**
- Add `deployment_target` field to `RSkillManifest` with default `"edge"`.
- Add constraint table enforcement in `rSkill.check_compatibility` (loader).
- Update fuzz tests + any rSkill manifests in `rskills/` that need explicit targets.

**Phase 3 — `openral detect --cloud <endpoint>` (SSH probe)**
- Add `--cloud` flag to `openral detect`.
- Implement SSH-based remote `openral detect --no-write --json` invocation.
- Populate `compute_cloud` + `compute_cloud_endpoint` in assembled description.
- Gate behind `OPENRAL_ALLOW_CLOUD_PROBE=1` env var for safety.
- Add integration test with a fake SSH server (process boundary fake, per CLAUDE.md §1.11).

**Phase 4 — `openral doctor` three-slot display (D7)**
- Extend `_check_compute_spec` to emit three groups: edge / local / cloud.
- Update `test_doctor.py`.

---

## Non-goals / out of scope

- **Cloud inference server implementation.** This ADR defines the `compute_cloud` schema and SSH probe; it does not define the HTTP inference server that would be the other endpoint type. That is a future ADR in the deploy-path cluster.
- **Fleet-wide compute registry.** `compute_cloud_endpoint` is per-robot; a multi-robot fleet sharing a cloud backend is a scheduler concern outside this ADR.
- **VRAM budget arbitration across skills.** ADR-0050 (single-resident-skill VRAM eviction) handles VRAM contention; this ADR does not change that. The three slots are independent — no cross-slot VRAM accounting.
- **`schema_version` for scene or rSkill manifests.** Only `RobotDescription` is affected.
