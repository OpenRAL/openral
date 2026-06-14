# ADR-0050: Single-resident-skill VRAM eviction (unload-on-switch)

- Status: **Proposed**
- Date: 2026-06-12
- Related: [ADR-0018](0018-ros2-reasoner-supervisor.md) (skill runtime + reasoner);
  [ADR-0025](0025-reasoner-managed-background-services.md) (LifecycleTransitionTool + lifecycle peers);
  [ADR-0037](0037-gstreamer-perception-bus-object-detection.md) (`kind: detector` perception producers);
  [ADR-0043](0043-locate-in-view-reasoner-tool.md) (live `locate_in_view` open-vocab detector);
  [ADR-0046](0046-nvidia-gr00t-backend.md) (out-of-process VLA sidecars).

## Context

On an 8 GB GPU (the RTX 4070 Laptop reference dev host) the autonomous
find→navigate→grab loop cannot complete: the open-vocab detector
(`LocateAnything-3B`, NF4 sidecar, ~5.3 GB peak) and the grab policy
(`pi05-robocasa365`, ~4.3 GB) **do not co-reside in 8 GB**. They never run
*simultaneously* in the reasoner cascade (`locate_in_view` → then
`execute_rskill`), but both stay **resident**, and the overlap OOMs.

Today nothing evicts GPU models:

- The skill runner (`rskill_runner_node._execute_cb`) overwrites
  `self._active_skill = skill` on each dispatch. The prior model is never
  explicitly released (no `empty_cache`, no weight drop); it lingers until GC.
- `rSkillBase` has a symmetric lifecycle (`configure`/`on_load_weights` →
  `activate` → `deactivate` → `shutdown`) but **no `on_unload_weights` hook** —
  there is no contract for releasing weights.
- The detector runs as an always-on producer. `deploy_sim.py` already carries a
  comment lamenting "GR00T/RLDX weights resident … starves the GPU (~6.5 GiB)".

A general policy is wanted: **at most one heavy model GPU-resident at a time;
the previous one unloads when the reasoner switches to another** — generalized
to all rSkills and the detector, not a one-off.

### Constraint discovered during design

`RosImageObjectDetectorNode` is a **plain `rclpy.node.Node`, not a
`LifecycleNode`** (and is not in the reasoner's `lifecycle_peer_node_ids`, which
today lists only `openral_slam_toolbox`). So "the reasoner deactivates the
detector via the existing `LifecycleTransitionTool`" is **not** wireable without
first converting the detector to a lifecycle node. `LocateAnythingDetector.close()`
already terminates the sidecar subprocess (frees its VRAM) — the release
primitive exists; the lifecycle host does not.

## Decision

A **single-resident-skill eviction** policy built from four parts:

1. **`rSkillBase.on_unload_weights()` hook (new, default no-op)** — symmetric
   with `on_load_weights`. `shutdown()` calls it before transitioning to
   `FINALIZED` and clears `weights_loaded`. Subclasses override to drop model
   references + `torch.cuda.empty_cache()` (or terminate their sidecar). This is
   the **generalized contract**: any GPU-backed skill releases here.

2. **Skill-runner eviction** — the runner keeps the resolved `_active_skill`
   keyed by `(rskill_id, revision)`. On an `execute_rskill` whose key **differs**
   from the resident skill, it calls `old.shutdown()` (→ `on_unload_weights`,
   freeing VRAM) **before** resolving/loading the new skill. Re-dispatching the
   **same** key reuses the resident skill (no reload). Node `on_cleanup` /
   `on_shutdown` evict the resident skill.

3. **Detector → `LifecycleNode`** — convert `RosImageObjectDetectorNode` to a
   managed lifecycle node. `on_activate` builds/starts the detector (sidecar);
   `on_deactivate` calls `self._detector.close()` (releases sidecar VRAM);
   `on_cleanup` drops it. Launch wires it under the existing
   `lifecycle_manager` and adds `openral_ros_image_detector` to the reasoner's
   `lifecycle_peer_node_ids` when `enable_object_detector`.

4. **Reasoner sequencing** — uses the **existing** `LifecycleTransitionTool`
   (ADR-0025): before dispatching a GPU-heavy actuation skill it can
   `deactivate` the detector, and `activate` it again afterward. The detector
   being a lifecycle peer (part 3) is what makes this expressible.

   **Amendment (2026-06-12) — automatic pre-dispatch eviction.** Relying on the
   LLM to emit the `deactivate` was unreliable: in the live autonomous robocasa
   run the reasoner dispatched a VLA without freeing the detector, so the
   detector (~1.3 GB) co-resident with the policy (~4.5 GB) CUDA-OOM'd the 8 GB
   card at load (`rldx_sidecar_died_during_boot`). The deactivation is now an
   **automatic, deterministic pre-dispatch policy** in `reasoner_node`, not an
   LLM choice: a `vram_lifecycle_peers` parameter (the deploy launch sets it to
   `openral_ros_image_detector` when `--enable-object-detector`) lists the GPU
   peers the reasoner **deactivates before every `execute_rskill`** and
   **reactivates on its result** (`_free_vram_peers_then_send` /
   `_reactivate_vram_peers`). The send is *sequenced* behind the `change_state`
   responses so the VRAM is released before the goal reaches the runner.
   Reactivation fires on the terminal result and on goal-reject/error (never on
   `deadline`, where the policy may still be resident). This is distinct from
   `lifecycle_peer_node_ids`, which only surfaces peers to the LLM tool palette.

   The reasoner change alone was **not sufficient**: the launch autostart
   (`_autostart_lifecycle`) re-activated the detector ~15 ms after each
   deactivate, because its activate handler matched a bare
   `goal_state="inactive"` — which also fires on a *runtime* deactivate
   (`active → deactivating → inactive`), not just the boot configure. Scoping it
   to `start_state="configuring"` makes the autostart one-shot, so a
   reasoner-driven deactivate sticks. Verified live (2026-06-12): detector frees
   ~1.3 GB, the `rldx1-ft-rc365-nf4` VLA then loads and runs policy steps on the
   8 GB card instead of OOMing at load.

## Alternatives considered

- **Central VRAM arbiter service.** A node all GPU consumers register with;
  acquiring exclusive GPU auto-evicts the holder. Cleanest fully-automatic
  policy, but new always-on infrastructure + protocol. Rejected as heavier than
  needed; the lifecycle primitives already exist for (1)–(4).
- **Idle-timer sidecar unload.** Detector sidecar self-releases after N seconds
  idle, reloads on next detect. No reasoner/lifecycle changes, but timer-based
  (not switch-driven), and pays full reload latency on every wake. Doesn't match
  the "unload when the reasoner switches" requirement. Kept as a possible
  detector-local optimization, not the mechanism.

## Consequences

- **Positive:** the autonomous detect→navigate→grab loop fits in 8 GB; VRAM
  policy is explicit and generalized; reuses ADR-0025 lifecycle primitives.
- **Negative / costs:** switching skills now pays a **reload** (weights + warmup)
  on each change — acceptable for the S2-paced autonomous loop, not for tight
  S1 skill alternation. Converting the detector to a lifecycle node touches
  launch + lifecycle-manager wiring. Spans three layers (perception 1/3, skill
  runtime 3, reasoner 4) — phase the PR (see below).
- **Eviction must never weaken safety:** unload happens only between goals, never
  mid-`step`; `ROSSafetyViolation` handling is unchanged.

## Testing

- Unit: `rSkillBase.shutdown()` invokes `on_unload_weights` + clears
  `weights_loaded`; runner evicts on key-change and reuses on same-key (fake
  skills counting load/unload calls).
- Integration (`launch_testing`): detector lifecycle `activate→deactivate`
  releases the sidecar (process gone); reasoner `LifecycleTransition(detector,
  deactivate)` succeeds with the detector as a peer.
- Sim/HIL: the 8 GB co-residency case — `locate_in_view` then `execute_rskill`
  on `panda_mobile`/robocasa completes without CUDA OOM (the reproduction this
  ADR exists to fix).

## Phasing (PR boundaries — avoid the >800-line / multi-layer single PR)

- **P1** — `rSkillBase.on_unload_weights` + runner eviction/caching (layer 3) + unit tests.
- **P2** — detector → `LifecycleNode` + sidecar release on deactivate + launch wiring (layer 1/3) + integration test.
- **P3** — reasoner detector lifecycle peer + sequencing + the 8 GB sim repro.
