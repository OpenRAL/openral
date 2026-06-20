# ADR-0060 — Benchmark task-data compatibility gate (`evaluated_tasks`)

- **Status:** Accepted 2026-06-18. Implemented + verified end-to-end on an 8 GB GPU host
  (gate blocks a LiftCube policy on `maniskill3/PickCube-v1` and a pick-place policy on
  `so101_box/tube_insertion`, before any rollout).
- **Date:** 2026-06-18
- **ADR number:** `0060`. `0058` (standardized-description-assets) and `0059`
  (foxglove-live-scene-visualization) are claimed; the integer is not load-bearing —
  cross-refs use filenames.
- **Related:**
  - The embodiment/sensor gate (`rSkill.check_compatibility` → `check_capabilities` +
    `check_sensors`) — this ADR adds the missing *task-data* axis next to it.
  - ADR-0009 — `openral benchmark scene` / `run` producers this gate sits inside.
  - ADR-0019 — `state_contract` / `action_contract` dims (a different, shape-level contract).

## Context

The sim runner verifies that an rSkill's **embodiment** and **sensors** match the robot
(`ROSCapabilityMismatch` otherwise). Nothing verified that the rSkill was trained for the
**task the benchmark evaluates**. So a checkpoint trained on one task ran happily on a
different benchmark task and produced a *plausible-looking but unsolvable* rollout:

- `smolvla-maniskill-franka` wraps `Calvert0921/smolvla_franka_**liftcube**_1000` (trained to
  *lift* a cube) but was paired with `maniskill3/**PickCube-v1**` (grasp **and place at a
  goal** + stay static). Frames show a clean grasp/lift; success is never satisfied → 0/40.
- `pi05-so101-**pickplace**-nf4` was paired with `so101_box/**tube_insertion**` and
  `tabletop_push/**push_to_goal**` — neither is pick-place. 0/40 each.

These wasted GPU time and, worse, would have reported meaningless success numbers if run as
paper-comparison evals. The embodiment gate could not catch them (the robot matches fine).

## Decision

Add an explicit, optional task-allowlist to the rSkill manifest and gate the benchmark
runner on it.

- **Schema:** `RSkillManifest.evaluated_tasks: list[str] = []` — the benchmark task ids /
  families the checkpoint was trained or validated for. Additive + optional, so it is a
  backward-compatible change (no `schema_version` bump; CLAUDE.md §1.6). Legacy manifests
  omit it and stay runnable.
- **Matching** (`openral_sim.benchmark._task_matches`): an entry covers a scene when it equals
  the scene's `task.id`, is a `"<scene>/<…>"` prefix family (so `"libero_spatial"` covers
  `libero_spatial/0..9`), or equals the bare `scene.id`.
- **Gate** (`openral_sim.benchmark.check_benchmark_task_compatibility`), called from
  `run_benchmark_scene` **before** the rollout loop:
  - `evaluated_tasks` non-empty and the scene's task **not** covered → raise
    `ROSCapabilityMismatch` (fail fast, no GPU work).
  - `evaluated_tasks` empty → **permissive**: log `rskill_task_compat_undeclared` and proceed
    (legacy rSkills are not broken; the gate tightens only as manifests declare their tasks).
  - Skipped for the built-in mock policies and `hf://` URIs (no local manifest).

**Scope:** the gate fires on the **benchmark** path (`openral benchmark scene` / `run`) — the
protocol/paper-comparison + website-showcase path where a task mismatch is never acceptable.
`openral sim run` (free experimentation / debugging) is intentionally **not** gated.

## Consequences

- Mismatched benchmark pairings now fail fast with an actionable typed error naming the
  declared tasks vs the scene's task — instead of silently scoring 0.
- A benchmark scene whose only candidate rSkill is task-mismatched is surfaced as having **no
  compatible rSkill** (e.g. ManiSkill PickCube, SO-101 insertion/push), which is the honest
  signal to source or finetune a task-matched checkpoint.
- Populating `evaluated_tasks` across the rSkill catalogue is incremental: the two known
  offenders are declared now (blocking their bad pairings); the rest tighten over time.
- Matching is intentionally simple (exact / prefix / scene-id). It is a *declared-intent*
  allowlist, not an automatic inference from `dataset_uri` — explicit beats implicit
  (CLAUDE.md §1.4).
