# ADR-0077 — A VLA names its reward model, and the pair must fit GPU VRAM together

- **Status:** Accepted 2026-06-29.
- **Date:** 2026-06-29
- **ADR number:** `0077`. The integer is not load-bearing — cross-refs use it
  as a stable handle.
- **Supersedes / amends:** extends ADR-0057 (Robometer reward rSkill) and
  ADR-0074 (reward-driven completion). Complements ADR-0050 (VRAM peer eviction).

## Summary

A VLA emits **no success signal of its own**. The reasoner only knows whether a
running policy is making progress, has plateaued, or has finished from a **reward
model scoring it live** (ADR-0074: the reward-watcher cancels an in-flight VLA and
the three-tier verdict gates completion). Without a reward signal the reasoner is
blind: the VLA runs until the patience backstop with no notion of success, and the
plateau / completion logic has nothing to read.

Therefore: **whenever a VLA runs, its reward model is resident on the GPU next to
it.** Two changes make that a contract rather than a deploy-time accident:

1. A VLA manifest names its reward model — `reward_rskill_name` (Layer 0 schema).
2. Before a VLA is loaded, the deploy/runner verifies the VLA + reward pair fits in
   GPU VRAM (both sizes are declared in their manifests). If it does not fit, we
   **fail fast and notify** — we do **not** silently run a VLA with no reward
   signal, and we do **not** fall back to a different progress estimator.

## Context

ADR-0050 lets the reasoner evict detectors/locators before a co-resident grab
policy. The reward monitor is deliberately **not** evictable — it must stay
resident to score the VLA live (ADR-0074 §2). That is correct, but the pairing was
implicit: the reward model was chosen by a separate deploy flag
(`--enable-reward-monitor` + `reward_manifest_path`), wholly decoupled from the VLA
the reasoner picks at runtime. Nothing tied "this VLA" to "that reward model", and
nothing checked that both fit before the VLA tried to load.

Measured footprints for the reference pair (RTX 4070, 8 GB):

| Model | dtype | VRAM |
|-------|-------|------|
| `smolvla-libero` | bf16 | **0.93 GB** (measured at load) |
| `robometer-4b-nf4` | int4 (nf4) | **3.6 GB** |
| **pair** | | **~4.5 GB ≪ 7.62 GB → fits** |

So the common pair fits with wide headroom — the model footprint is small. (A live
deploy OOM we saw was the *whole* stack — sim + ROS + un-evicted detectors +
fragmentation — not the model pair; that is a separate budgeting concern from this
ADR's pair check.) The point of the pre-check is the *general* case: a larger
reward model (a future personalized SARM) or a larger VLA could exceed the GPU, and
we want that caught **before** launch with a clear message, not as a mid-run CUDA
OOM.

## Decision

### 1. `reward_rskill_name` on the VLA manifest (Layer 0 → schema)

`RSkillManifest` gains an optional field:

```python
reward_rskill_name: str | None = None   # ADR-0077
```

- Names the reward/progress-monitor rSkill this VLA pairs with — an rSkill `name`
  (e.g. `"OpenRAL/rskill-robometer-4b-nf4"`). Robometer is the default today; the
  field exists so a VLA can pin a **specialized / personalized SARM** later.
- **Allowed only for `kind == "vla"`** (a top-level validator guard, mirroring the
  `playbook` guard); forbidden on every other kind.
- `None` means "use the deployment default reward model" — backward-compatible for
  every existing VLA manifest. It does **not** mean "run without reward": the deploy
  still pairs a reward model (its default) with the VLA.

This is an **additive, backward-compatible** field, so `schema_version` evolves in
place (no migrator — CLAUDE.md §1.6).

### 2. Declared VRAM, queryable per active dtype

`RSkillManifest.active_min_vram_gb()` returns `min_vram_gb[quantization.dtype]` (or
`None` when undeclared). VLA manifests that pair with a reward model **must** declare
`min_vram_gb` for their active dtype so the pair can be checked — `smolvla-libero`
gains `min_vram_gb: {bf16: 1.2}` (0.93 GB weights + activation headroom).

### 3. Pre-load pair fit check (Layer 0 helper → enforced by deploy/runner)

`assert_vla_reward_fits(vla, reward, gpu_total_gb, *, margin_gb=0.5)`:

- Raises `ROSConfigError` if either manifest does not declare `min_vram_gb` for its
  active dtype (we cannot verify a co-residency we are about to *require*).
- Raises `ROSGPUMemoryError` if `vla + reward + margin > gpu_total_gb`, with the
  per-model breakdown in the message.
- Returns the combined GB on success.

The check is a **necessary** condition (the model pair's footprint), not a
sufficient one (it does not account for the sim/ROS overhead, which ADR-0050
eviction and the deploy budget handle separately). It runs at the point a specific
VLA is about to load — the reasoner picks the VLA at runtime (`deploy_sim` does not
preselect it), so the guard lives where the pairing is known:

- **Palette time (reasoner):** a VLA whose declared pair cannot fit the GPU is
  dropped from the palette with a logged reason, so the reasoner never dispatches an
  unrunnable VLA.
- **Load time (runner):** defense-in-depth before `from_pretrained`; on failure it
  emits a typed `FailureTrigger` so the reasoner sees "VLA X needs reward Y
  co-resident; combined Z GB > GPU W GB" and hands off, instead of a bare OOM crash.

### What stays the same

- The reward monitor stays **non-evictable** (ADR-0050) — it must score the VLA
  live. This ADR makes that residency a *checked* requirement, not a hope.
- The reward verdict / plateau logic (ADR-0074) is unchanged; it simply now always
  has a reward signal to read when a VLA runs (or the run never started).
- `reward: RewardContract` (the reward-kind manifest's own contract) is untouched —
  `reward_rskill_name` is a *reference* from the VLA side, a different field.

## Consequences

- A VLA always runs with a reward signal, or does not run at all — the reasoner is
  never blind to an executing policy.
- Oversized pairs fail **before** launch with an actionable message, not as a
  mid-run CUDA OOM.
- VLA manifests that pair with reward must declare `min_vram_gb` — a small, healthy
  forcing function (the sizes were always knowable; now they are recorded).

## Alternatives considered

- **Time-share / post-hoc reward** (evict the VLA, load reward, score after): gives
  no live progress and no in-flight cancel — the reasoner cannot tell a stuck policy
  from a working one until the patience backstop. Rejected: the live signal is the
  whole point.
- **VLM fallback when reward does not fit** (ask a scene VLM "is it done?"): a VLM
  cannot tell *when* the VLA finishes or whether it is progressing frame-to-frame;
  it is not a substitute for a reward model. Rejected.
- **Keep the pairing implicit (deploy flag only):** leaves "this VLA ↔ that reward"
  unrecorded and unchecked — the status quo this ADR replaces.

## Rollout

1. `reward_rskill_name` field + validator guard + `active_min_vram_gb` +
   `assert_vla_reward_fits` (Layer 0) — this PR.
2. `smolvla-libero` declares `min_vram_gb` + `reward_rskill_name` — this PR.
3. Deploy/runner enforcement (palette drop + load-time guard + reward resolution
   from the field) — wired and live-tested next.
