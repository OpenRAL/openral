# Sim + Dashboard Demo — Feasibility Report

> Date: 2026-06-12 · Host: RTX 4070 Laptop, 8 GB, X11. All findings live-verified unless noted.

## Headline

The **MuJoCo / RoboCasa panda_mobile** autonomous "perceive → navigate → grab" demo has its
**full perception chain working live on 8 GB**: the **2D→3D object-lift** (previously reported as
the blocker) now closes end-to-end — `world_state.detected_objects` is populated with the baguette
at a correct map-frame pose. The recording infrastructure, the VRAM-management feature, the
costmap, and open-vocab perception are all verified. The reasoner's autonomous
**`recall_object → dispatch`** loop is now verified too: it correctly *gates* the grab on a
resolvable 3D pose and dispatches a VLA once recall succeeds (see below). Two gaps remain for a
fully hands-off baguette grab — a goal-word↔detector-label vocabulary mismatch, and 8 GB
detector↔VLA co-residency (ADR-0050 eviction).

> **2026-06-12 (update — supersedes the "remaining blocker" claim below):** the object-lift was
> live-verified working on current `master` (PR #317 merged). The earlier "lift is broken /
> `sensor_id=front_depth` mismatch / octree stays empty" diagnosis **no longer reproduces** — it
> was the pre-#317 state. See **"Object-lift — RESOLVED"** below for the live evidence.

## What was built + verified this session

| Item | Status | Evidence |
|------|--------|----------|
| **ADR-0050 single-resident-skill VRAM eviction** (P1 runner eviction + P2 detector LifecycleNode + P3 reasoner peer) | ✅ live-verified | deactivate detector → GPU **4385→1138 MiB** (-3.2 GB); reactivate rebuilds; 15 unit/structural tests, ruff clean |
| **Costmap z-extent fix** (`nav2_panda_mobile.yaml` voxel 0.8→1.2 m) | ✅ live-verified | `out of map bounds … cannot raytrace` warnings: spamming → **0**; `/local_costmap/voxel_grid` populated |
| **Open-vocab perception** (LocateAnything-3B @ 512²) | ✅ live-verified | `locate_in_view('baguette')` → **found=True, conf 1.0, bbox [180,307,231,348]**; continuous leg publishes `1 objects` |
| **Scene render 128²→512²** (`robocasa_baguette.yaml`) | ✅ fix | at 128² every query returned found=False (baguette ~20 px, upscale-to-1024 too blurry); 512² resolves it |
| **Reasoner LLM** (ollama `gemma4:31b-cloud`, the launch default) | ✅ works | given a goal it emits real tool calls (`dispatch: recall_object`); idle "tick error: no tool_calls block" is benign (no goal) |
| **pi05 grab** (direct dispatch) | ✅ earlier | real base nav + arm reach + safety-gated grasp attempt |
| **Recording** (`tools/record_demo.sh`, ffmpeg x11grab on the DP-4-3 monitor) | ✅ | valid MP4; x11grab works (this is a real Xorg session) |

## Object-lift — RESOLVED (live-verified 2026-06-12, PR #317)

The 2D→3D object-lift **closes end-to-end** on current `master`. Verified by booting deploy-sim
(`robocasa_baguette.yaml`) with the octomap leg + an open-vocab continuous detector and probing the
live ROS graph:

```
openral deploy sim --config scenes/deploy/robocasa_baguette.yaml \
  --enable-octomap --no-enable-octomap-kernel-check \
  --enable-slam --object-detector-manifest rskills/omdet-turbo-indoor/rskill.yaml
```

| Link in the chain | Live evidence |
|---|---|
| depth cloud → octree | octomap_server builds **3489 nodes**, **0** message-filter drops |
| octree → `/openral/world_voxels` | bridge publishes **737 occupied cells** (40³ grid, base_link frame) |
| detector stamping | `sensor_id=agentview_left` (a real RGB camera — **not** `front_depth`) |
| detector → detections | omdet-turbo-indoor publishes the baguette as `bread` continuously |
| lift → `world_state.detected_objects` | **31 objects in the `map` frame**, incl. `bread` at **(4.83, −1.00, 0.94)** — robot base at (4.37, −1.28); a correct baguette-on-counter pose |

The earlier root-cause ("detector on `agentview_left` RGB has no paired depth; stamped
`sensor_id=front_depth`") was the **pre-#317** state. PR #317's cross-frame lift (RGB optical TF +
octomap/kernel decoupling, ADR-0052/0051) projects the agentview bbox against the depth-built voxel
map across frames via TF — no same-camera depth needed. The lift integration is regression-covered
by `tests/integration/test_object_lift_world_state.py` (happy-path / no-voxels / eviction).

**Hard requirement:** the lift needs the `map` frame (poses are emitted in `detected_object_frame`,
default `map`), so **SLAM must be running** (`t_map_from_base` is `None` otherwise → every detection
skipped). The `robocasa_baguette.yaml` scene is the full SLAM+Nav2 stack, so this holds in the demo;
a minimal `--no-enable-slam` graph is the one config where the lift correctly produces nothing.

### Autonomous reasoner loop — VERIFIED (live, 2026-06-12, ollama `gemma4:31b-cloud`)

Full stack booted (SLAM+Nav2+octomap+omdet detector+reasoner) and a goal injected on
`/openral/prompt`. The reasoner ingests `world_state_slow.detected_objects` into spatial memory
(`spatial_memory_ingest=true`) and runs the ADR-0044 `recall_object → (grid-refined) approach →
grasp` ladder. Two goals, two outcomes — both confirm the gating works:

- **`"pick up the baguette"`** → `recall_object` returns **no match** (the lift labels the object
  `bread`; CLIP cosine "baguette"~"bread" < 0.85 and no substring hit) → 5-query budget exhausted →
  **`handing off`**. **The VLA is never dispatched** — the reasoner correctly *gates* the grab on a
  resolvable 3D pose.
- **`"pick up the bread"`** → `recall_object` **matches** (exact label) → reasoner
  **dispatches `execute_rskill rskill_id=rldx1-ft-rc365-nf4 prompt='pick up the bread'`**. The
  recall→dispatch loop closes end-to-end.

So the **two real gaps** for a hands-off baguette grab were (both now addressed):

1. **Vocabulary mismatch** (goal word vs detector label) — **mitigated (#14)**. omdet labels it
   `bread`; the goal says `baguette`, and `recall_object` matching is literal (exact/substring +
   CLIP ≥ 0.85), so it couldn't resolve and gated. The LLM understands baguette≈bread but was never
   shown the label — the context only carried `[objects] 30 objects` (a count). Fix: the reasoner's
   `## WORLD_STATE` now renders `scene_objects[map]: bread@(...), …` so the LLM sees the lifted
   labels and maps the goal noun onto them itself. (Alternatives still valid: drive the detector
   open-vocab with the goal noun — `--object-detector-manifest rskills/locateanything-3b-nf4 …
   --object-detector-query "baguette"`; or let the reasoner dispatch pi05/rldx directly on the goal,
   the mobile-manip shortcut — the VLA drives base AND arm, no 3D pose needed.)
2. **8 GB co-residency** — **FIXED (#15, live-verified)**. The detector (~1.3 GB) stayed resident
   when the VLA was dispatched, so rldx1 (~4.57 GB) + MuJoCo CUDA-OOM'd at load. The reasoner now
   auto-deactivates GPU lifecycle peers (`vram_lifecycle_peers`) before each `execute_rskill` and
   reactivates after; the launch autostart was made one-shot (`start_state="configuring"`) so the
   deactivate sticks. Live 2026-06-12: detector frees its VRAM, `rldx1-ft-rc365-nf4` loads and runs
   policy steps driving toward the bread — no OOM (GPU 7.7/8.2 GB).

`locate_in_view` (live 2D detector, on_demand) also works and is the natural fresh-scene rung.

## Per-clip feasibility (recap)
- **Clip 1 — robocasa nav-pick:** semi-auto (direct pi05 dispatch) works today; the perception
  blocker is resolved (object-lift closes), so full autonomy now hinges only on the downstream
  `recall_object → navigate → grab` reasoner loop (above), not on perception.
- **Clip 2 — OpenArm pick:** deploy-sim renders viewer+cameras; not re-run this session.
- **Clip 3 — Isaac Franka pick:** `sim run` path verified in ADR-0045; deploy-sim variant unverified.
- **Clip 4 — Isaac panda_mobile nav:** deploy-sim verified live in ADR-0045.

## Uncommitted work (as of this report)
ADR-0050 (P1/P2/P3), costmap fix, 512² scene change, `tools/record_demo.sh`, `tools/_demo_env.sh`.
Pending PR bookkeeping: `docs/METHODS` entry for `rSkillBase.on_unload_weights`, repo-state-map,
and a commit. Pre-existing unrelated failure noted: `test_rskill_runner_node ...passthrough`
(diagnostics-heartbeat timing) fails on clean source too.
