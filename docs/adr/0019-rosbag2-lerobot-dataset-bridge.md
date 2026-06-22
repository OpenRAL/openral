# ADR-0019: rosbag2 ↔ LeRobotDataset v3 bridge

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)
- Related: CLAUDE.md §6.1 (Layer 7 — Observability), §1.11 (no mocks),
  §1.14 (docs travel with code), §7.2 (smallest viable PR / pre-approval),
  §12 (new top-level packages need an ADR); pairs with
  [ADR-0017](0017-dashboard-otlp-receiver.md) (dashboard OTLP receiver);
  closes the rosbag2 placeholder in
  [ADR-0010 §6](0010-inference-runner.md).

## Context

The Week-4 roadmap (`docs/roadmap/index.md:131`) calls for a "rosbag2 ↔
LeRobotDataset v3 recorder — every successful skill execution becomes a
sync-video + state + action-chunk row; `openral dataset push
hf://openral/dataset-<name>` with consent prompt." Today the repo
references rosbag2 in two places (this ADR set and runner backend
comments) but contains **no rosbag2 writer, no
LeRobotDataset writer, and no camera-topic publisher on the hardware
runner**. The OTel semconv module already reserves
`openral.dataset.repo_id / episode_idx / frame_idx` placeholders
(`python/observability/src/openral_observability/semconv.py:143–145`) —
they are meaningless until this bridge wires them.

The bridge is the durable counterpart of ADR-0017's transient
observability fan-out. ADR-0017 lets a developer *see* a skill
execution live; this bridge lets a developer *replay it, fine-tune on
it, and publish it* later. The two ship together as the Week-4
observability deliverable.

Five orthogonal design questions need answers up front:

1. **Where does the code live?** New top-level package, submodule under
   `openral_observability`, or scattered across `openral_sim` +
   `openral_runner` + `tools/`?
2. **Which dataset format?** LeRobotDataset v2.1 (file-per-episode,
   stable for ~12 months) or v3.0 (chunked, codebase_version="3.0",
   released April 2026 in lerobot v0.5.1).
3. **What happens to failed episodes?** Discard (datasets = only
   demonstrations of success) or persist with a `next.success=False`
   tag (datasets = full distribution).
4. **License posture on produced datasets.** The OpenRAL code is
   Apache-2.0 across this layer, but a *dataset* carries an
   independent license string.
5. **PR sizing.** The full bidirectional scope is ~1500–2000 LOC,
   exceeding CLAUDE.md §7.2's 800-LOC pre-approval threshold.

## Decision

### 1. Package layout — new top-level `python/dataset/` (`openral_dataset`)

The bridge ships as a new workspace package adjacent to
`python/observability/`, not nested inside it. Justifications:

- **Lifecycle separation.** Observability is *transient* telemetry
  exported on the wire (OTLP → collector → backend). Datasets are
  *durable* artifacts written to disk and Hugging Face Hub (parquet,
  mp4, mcap). Coupling them mixes a "configure SDK and forget" library
  with a "manage files, codecs, and licenses" library.
- **Dependency surface.** `openral_observability` today depends on
  OTel SDK + structlog. The bridge needs `lerobot>=0.5.1`, `pyarrow`,
  `rosbag2_py`, `mcap`, `mcap-ros2-support`, ffmpeg-via-lerobot —
  hundreds of MB of transitive deps. Pulling them into observability
  would slow `uv sync` for every consumer of observability.
- **License posture.** Telemetry is uniformly Apache-2.0. Datasets
  carry per-dataset license strings (default `CC-BY-4.0`, but
  consumers may legitimately ship `CC-BY-NC`, `CC0`, or a custom
  license). Keeping the producing code in its own package keeps the
  open-core boundary §1.9 cares about clean.
- **Roadmap framing.** `docs/roadmap/index.md:131` lists "Dataset
  bridge" as a *peer* to "OpenTelemetry" in the Week-4 deliverables,
  not as a sub-item.
- **§12 is not a barrier.** The "STOP, propose in an ADR first"
  instruction is satisfied by this ADR; we're writing it either way.

`openral_dataset` imports `openral_observability` for the `semconv`
constants — the trace-correlation handle (span IDs in `Tick.msg`)
crosses the package boundary without difficulty.

This does **not** add a layer. It is a sub-responsibility of Layer 7
(Observability) per CLAUDE.md §6.1, factored out for separation of
concerns. The 8-layer model stays.

### 2. Dataset format — LeRobotDataset v3.0

v3.0 is the current `lerobot` codebase_version (`"3.0"`), released
April 2026 with `lerobot==0.5.1`. Adopt it directly; do not ship a v2.1
writer.

Reasons:

- **File-count scalability.** v2.1 wrote one Parquet + one MP4 per
  episode. v3.0 batches multiple episodes per file (0–5 MB chunks).
  A 10 000-episode dataset goes from ~20 000 files to ~100 files —
  hundreds of times fewer inode lookups during training I/O.
- **Random-access reads.** v3.0's `meta/episodes/chunk-*/file_*.parquet`
  carries per-episode offsets into both Parquet data and MP4 video
  streams. `lerobot.datasets.LeRobotDataset.delta_timestamps` no
  longer has to load whole-episode payloads.
- **Codec stability.** v3.0 locks codec parameters in `info.json`
  after the first episode, removing the v2.1 mid-dataset codec-skew
  bug.
- **Workspace already ships lerobot.** `python/hal/pyproject.toml`
  and the `[dependency-groups] libero` / `metaworld` groups already
  pin lerobot; bumping the floor to `>=0.5.1` in the new
  `openral_dataset` package is non-invasive.

`lerobot.scripts.convert_dataset_v21_to_v30.py` upstream covers
back-conversion if a v2.1 dataset shows up; we do not re-implement
it.

### 3. Failure policy — persist all episodes with `next.success` flag

Every episode is written. Successful and failed rollouts both produce
rows; `next.success` is the boolean flag. Top-level
`meta/info.json["metadata"]["dataset_success_rate"]` reports the
ratio so downstream consumers can filter.

Reasons:

- **Imitation literature.** Policies trained on success + failure
  consistently beat policies trained on success-only when the failure
  distribution is unbiased (ALOHA, RT-2 ablations). Failures are
  signal, not noise.
- **Reasoner training.** The replanning ladder (§6.6) needs labelled
  failures to learn substitute / goal-replan decisions.
- **Consent decoupling.** The decision to *persist* is independent of
  the decision to *publish*. The consent gate lives at
  `openral dataset push` (PR5), not at recorder time. Local
  `--dataset-out` writes stay on disk under the user's control.

### 4. License posture — per-dataset string, default `CC-BY-4.0`

Each produced dataset carries a `license` field in
`meta/info.json["metadata"]["license"]`, defaulting to `CC-BY-4.0`
(the LeRobot convention) and overridable via `--dataset-license <spdx>`
on `openral sim run` and `openral dataset push`. The package code stays
Apache-2.0; the data license is independent.

Datasets containing PII (human faces in camera frames, audio,
biometric joint trajectories) MUST set a more restrictive license.
The PR5 consent prompt enforces this disclosure.

### 5. PR sizing — pre-approved exception to §7.2

Total LOC for the bridge series is estimated at 1500–2000 across
production code + tests + docs, over §7.2's 800-LOC informal
threshold. The series is split into six discrete PRs so each one is
reviewable as the *smallest viable change* in dependency order:

| PR  | Scope                                                                   | LOC est. |
| --- | ----------------------------------------------------------------------- | -------- |
| PR0 | This ADR + repo-state-map block + roadmap flip                          | ~150     |
| PR1 | `openral_dataset` package: `RolloutRecorder`, `LeRobotDatasetSink`, `schema_map`; sim wiring | ~400     |
| PR2 | `SensorRosPublisher` + new `openral_sensors_ros` lifecycle package      | ~300     |
| PR3 | `Rosbag2Sink` + `Tick.msg` / `Episode.msg` IDLs + hardware episode API  | ~500     |
| PR4 | `Rosbag2ToLeRobotConverter` + `openral dataset from-bag`                    | ~300     |
| PR5 | `openral dataset push` + consent prompt + `_hf_publish` de-dup              | ~200     |

Each PR includes its own tests, docs, and `docs/METHODS.md` updates
per §1.14. Sim-side (PR1) ships first because it has no
ROS / GStreamer / hardware dependencies and exercises the full
`RolloutRecorder` → `DatasetSink` → LeRobot v3.0 path end-to-end.

## Consequences

- New workspace package `openral_dataset` at `python/dataset/`. Added
  to `[tool.uv.sources]` in the root `pyproject.toml`.
- `lerobot>=0.5.1` is now a first-class dep of `openral_dataset`
  (lazy-imported at sink instantiation so the package stays
  importable on hosts without lerobot, with a typed
  `ROSConfigError` raised on construction without it).
- Two new OTel semconv constants in
  `openral_observability/semconv.py`:
  `EVENT_EPISODE_CLOSED`, `DATASET_EPISODE_SUCCESS`. The existing
  `DATASET_REPO_ID` / `DATASET_EPISODE_IDX` / `DATASET_FRAME_IDX`
  placeholders are now live.
- `SimRunner` accepts an optional `recorder: RolloutRecorder | None`
  kwarg. When set, the recorder is fed in parallel with the existing
  `_EpisodeBuffer` (additive — the buffer is not replaced; the
  per-episode video pipeline and `RSkillEvalResult` writer stay
  unchanged).
- `HardwareRunner` (PR3) gets explicit `episode_start(task_string)` /
  `episode_end(*, success)` methods. These also land as
  `NotImplementedError` defaults on `InferenceRunnerBase` so future
  runners must opt in.
- New ROS 2 package `openral_sensors_ros` (PR2) lifts the camera-topic
  publisher out of `python/runner/.../backends/gstreamer/ros_tee.py`
  and generalises it to non-GStreamer sources. The GStreamer
  zero-copy path is preserved; the new path is a parallel consumer
  for OpenCV / RealSense readers.
- New IDLs `openral_msgs/Tick` and `openral_msgs/Episode` (PR3)
  extend the existing `packages/msgs/` package.
- `openral dataset` CLI subgroup (PR4 / PR5) with `from-bag` and `push`
  subcommands. `tools/rskill_publisher.py` is refactored to share
  `_hf_publish` helpers with `openral dataset push` (de-dup per §1.13).

## Amendments — 2026-05-18 (post-merge revert)

After landing the PR series, a follow-up review concluded that the
sink's first-frame state/action/camera shape derivation was the
wrong contract: a buggy policy that emits wrong-shape actions would
silently produce a malformed dataset. The bridge now requires every
shape to be **declared up-front**:

* **Hardware path**: `RobotDescription.observation_spec` /
  `action_spec` and `SensorSpec.intrinsics` are authoritative.
* **Sim path**: per ADR-0007, the sim-specific contract lives on the
  rSkill manifest (`state_contract.dim` and the newly-added
  `action_contract.dim`); the camera shape comes from the scene
  config (`SceneSpec.observation_height/width` — sim renders all
  cameras at one resolution, often different from the physical
  sensor's intrinsics).

Concrete changes:

1. **New schema**: `openral_core.schemas.ActionContract` mirrors
   `StateContract`. `RSkillManifest.action_contract` is the new
   optional field. Both contracts are required for any rSkill that
   wants bridge support.
2. **Sink reverted**: `LeRobotDatasetSink._create_dataset` no
   longer takes a `first_frame` argument. The features dict is
   resolved at `__init__` from the robot's specs + caller-provided
   overrides. Per-frame `write_frame` validates every shape
   strictly and raises `ValueError` on mismatch.
3. **CLI wires manifest contracts**: `openral sim run --dataset-out`
   loads the rSkill manifest, reads `state_contract.dim` +
   `action_contract.dim`, and passes them as `state_shape` /
   `action_dim` overrides to `LeRobotDatasetSink`. The scene's
   `observation_height/width` flow through as `camera_shape`.
4. **All 19 rSkill manifests** under `rskills/` now declare
   `state_contract` + `action_contract` (act-aloha, ACT-LIBERO,
   diffusion-pusht, pi05-*, smolvla-*, xvla-libero, every RLDX
   variant, template).
5. **All 11 robot manifests** under `robots/` already had
   intrinsics on every camera-bearing sensor (audit confirmed). No
   manifest changes needed there.

Smoke verification (real GPU + real VLA weights + real sim envs):

| Config | rSkill | State | Action | Result |
| --- | --- | --- | --- | --- |
| PushT + Diffusion | diffusion-pusht | 2 | 2 | ✅ |
| Franka + LIBERO + pi05 | pi05-libero-nf4 | 8 | 7 | ✅ |
| Franka + LIBERO + SmolVLA | smolvla-libero | 8 | 7 | ✅ |
| Franka + LIBERO + xVLA | xvla-libero | 8 | 7 | ✅ |
| Franka + LIBERO + ACT | act-libero | 8 | 7 | ✅ |
| Sawyer + MetaWorld + SmolVLA | smolvla-metaworld | 4 | 4 | ✅ |
| Aloha + ACT (cube) | act-aloha | 14 | 14 | blocked: upstream dm_control × mujoco 3.8.0 |
| Aloha + ACT (insertion) | act-aloha-insertion | 14 | 14 | same |
| RoboCasa / GR1 / rldx1 sidecar | (skipped — separate venv / sidecar) | | | not in this verification scope |

The two Aloha failures are upstream env issues (`'MjModel' object
has no attribute 'flex_bendingadr'` from dm_control 1.0.41 reading a
mujoco 3.8.0 model). The bridge code path is correct — the same path
that ACT-LIBERO uses for action emission and the multi-robot bridge
test exercises against Aloha at the schema-binding layer (passes 4/4).

The MetaWorld config (`scenes/benchmark/metaworld_push.yaml`)
was updated from declaring `observation_height/width: 256` to
`480` because the MetaWorld backend adapter does not honour the
scene-level resize (the docstring claims it does; the code does
not). The bridge's strict shape validation caught the mismatch — a
real bug that previously would have silently produced a malformed
dataset.

## Verification

Each PR ships its own verification commands per the bridge plan; the
ADR itself is verified by:

- `mkdocs build --strict` — markdown link integrity.
- `docs/architecture/repo-state-map.html` carries the new
  `python/dataset/` block adjacent to `python/observability/`.
- `docs/roadmap/index.md:131` flips from 🔵 planned to 🟡 in flight
  on PR0 acceptance, and to ✅ on PR5 merge.

PR-1 verification (canonical, sim-side):
```
uv run pytest python/dataset/tests -v
uv run openral sim run --config scenes/sim/libero_spatial.yaml \
                   --rskill rskills/mock-1 \
                   --n-episodes 2 \
                   --dataset-out /tmp/ds
uv run python -c "
from lerobot.datasets import LeRobotDataset
d = LeRobotDataset('/tmp/ds')
assert len(d) > 0
print(d.meta.info['metadata']['dataset_success_rate'])
"
```

Per CLAUDE.md §1.11: every test loads real
`RobotDescription.from_yaml("robots/so100_follower/robot.yaml")` and
exercises a real `lerobot.datasets.LeRobotDatasetWriter`. lerobot is
behind the `libero` / `metaworld` dependency groups today; tests
`pytest.skip` with a typed reason on hosts without it, never with a
mock.

## Amendments — 2026-06-08 (three-tier scene paths)

ADR-0041 split `scenes/` into deploy/sim/benchmark tiers and stripped
rSkill names from filenames. Two updates in this ADR:

- The MetaWorld config reference moved from
  `scenes/benchmarks/smolvla_metaworld_push.yaml` to
  `scenes/benchmark/metaworld_push.yaml` (singular `benchmark/`,
  rSkill name dropped). MetaWorld also has no SimScene-tier sibling
  post-refactor — `metaworld_push.yaml` exists only at the
  BenchmarkScene tier.
- The regression-reproduction example above (`uv run openral sim run
  --config … --dataset-out …`) switched simulator from MetaWorld to
  LIBERO, pointing at `scenes/sim/libero_spatial.yaml`. Reason:
  `--dataset-out` is exclusive to `openral sim run` (a SimScene-tier
  command), and MetaWorld lacks a SimScene sibling. The MetaWorld-
  specific bug coverage referenced by this amendment is preserved
  in the test suite — the demo command just needs a SimScene to
  drive end-to-end. See ADR-0041 and
  [`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md) for the per-tier strict-
  CLI matrix.

## Amendments — 2026-06-09 (per-frame OTel correlation — issue #109)

Closes the last deferred OTel piece from the [2026-05-17 amendment on
ADR-0010](0010-inference-runner.md): per-frame `(trace_id, span_id)` on
written dataset rows. The reverse link (the `openral.dataset.repo_id` /
`episode_idx` / `frame_idx` span attributes) already shipped; this adds
the **forward link** so a row pivots back into its trace.

- **Capture point.** `RolloutRecorder.record_frame` reads the active
  `rskill.tick` span's context (`get_current_span().get_span_context()`)
  and stamps the 32-hex `trace_id` + 16-hex `span_id` onto the
  `DatasetFrame`. Capture happens here — not inside a sink — because the
  `Rosbag2Sink` defers its mcap write to a worker thread where the OTel
  context is no longer in scope; the ids must ride on the frame. Absent a
  valid span the fields degrade to `""`.
- **Persistence.** `LeRobotDatasetSink` declares `trace_id` / `span_id`
  as v3 `string` features (plain `datasets.Value("string")` parquet
  columns, readable without decoding episode videos). `Rosbag2Sink`
  writes the same ids into the `/openral/tick` record (the schema already
  declared the fields).
- **Offline fidelity.** `record_frame` takes optional `trace_id` /
  `span_id` overrides; `Rosbag2ToLeRobotConverter` passes each bag tick's
  original ids so an offline bag→LeRobot conversion preserves the source
  rollout's trace rather than stamping the convert run's own (empty)
  context.
- **Pivot.** `openral_dataset.read_frame_trace(root, episode_idx,
  frame_idx)` reads a row's `(trace_id, span_id)` straight from parquet,
  and `openral replay --frame <repo_id>/<ep>/<frame> --dataset-root <dir>`
  resolves that trace_id as the bag↔span join key. The
  `openral_observability` bag reader learned the raw-`trace_id`+`span_id`
  Tick convention (it previously assumed every `jsonschema` payload packed
  a full W3C `traceparent` in one field).
- **Dataset- and episode-level pointers.** Because `trace_id` is
  run-constant (every `rskill.tick` shares the one `cli.command` root
  trace) but `span_id` is per-tick, the sink also writes coarser
  pointers so a consumer need not scan the data parquet: the distinct
  `trace_ids` + `n_traces` land in `meta/info.json["metadata"]`
  (dataset-level), and a `meta/openral_traces.json` sidecar maps every
  `episode_index → trace_id` (episode-level — kept out of
  `meta/episodes/*.parquet` because v3 drops string features from its
  per-episode stats). The episode map is the granularity that matters
  for datasets accumulated across multiple runs (resume-append), where
  episodes carry different traces.
- **Not done (separate PR).** The optional SemVer-major
  `trace_id` → `traceparent` rename + `tracestate` on `openral_msgs`
  (with a `tools/schema_migrator/` entry per CLAUDE.md §1.6) is out of
  scope and deliberately deferred.

## Amendment (2026-06-22) — deploy-graph recording (bus-centric)

The original bridge recorded **sim rollouts** (`SimRunner` → `RolloutRecorder`).
This amendment closes the deploy side: `openral deploy sim` / `deploy run` can
record the **live ROS graph** to a rosbag2 mcap that `openral dataset from-bag`
converts to a LeRobotDataset v3 — the same offline path, fed from the bus.

**Why not the in-process recorder.** The deploy graph does not run
`SimRunner`/`HardwareRunner`; `rskill_runner_node` drives its own
`skill.step(snapshot)` loop (the full `HardwareRunner` integration is still
deferred pending the F2 `WorldStateStamped` work). Rather than couple recording
into that actuation loop, recording attaches as a **bus observer**.

**Design.**
- **`openral_runner.DatasetRecorderBridge`** — mirrors `WorldCloudBridge`:
  constructed against the shared runtime node so its subscriptions ride the
  same executor. It joins three already-on-the-graph signals per frame:
  proprio + camera frames from the shared `WorldStateAggregator` snapshot,
  the action from `openral_msgs/ActionChunk.flat` on
  `/openral/candidate_action`, and episode boundaries from
  `openral_msgs/Episode`. Writes through the (enriched) `Rosbag2Sink`. It owns
  no actuation logic and is embodiment-agnostic.
- **Episode markers on the bus.** `rskill_runner_node` publishes
  `Episode(PHASE_START)` on goal accept and `Episode(PHASE_END, success)` in a
  `finally` (every exit path closes the episode).
- **Enriched bag.** `Rosbag2Sink` now writes the inline `observation_state` +
  `action` arrays into `/openral/tick` and one base64 raw-u8 frame per camera
  on `/openral/dataset/image`; the converter derives the LeRobot feature shapes
  from the recorded bag itself, so robots whose proprio/action layout lives only
  on the rSkill contract (e.g. `franka_panda`, `observation_spec=None`) convert.
- **Wiring.** `openral deploy sim/run --dataset-out PATH` →
  `sim_e2e.launch.py` `dataset_out` arg → `runtime_node` param →
  `compose_runtime(dataset_out=…)` attaches the bridge; teardown finalizes the
  bag.

**Verified live (2026-06-22):** `openral deploy sim --config
scenes/deploy/libero_pnp.yaml --dataset-out …` (franka/LIBERO, MuJoCo) recorded
a 178-frame episode (real 8-D proprio + 2× 256×256 video + PHASE_START/END);
`openral dataset from-bag` produced a reloadable LeRobotDataset v3. The
`act-libero` `ExecuteRskill` goal drove the loop.

**Known follow-ups.**
- **Multi-slot action fidelity.** For slot-dispatched skills (ADR-0028b — e.g.
  LIBERO cartesian_delta = 6-D cartesian + 1-D gripper as *separate*
  `ActionChunk`s), the bridge currently records the last-delivered chunk, so the
  dataset's `action` reflects one slot (the gripper) rather than the full env
  action. Single-`ActionChunk` skills (joint-position robots: so100, openarm,
  …) record the full action correctly. The fix is to reassemble the per-tick
  slot chunks into one action vector (or record the pre-split action the node
  computes); deferred.
- **`HardwareRunner` rename.** `HardwareRunner` already drives digital twins via
  the sim HAL, so the name misleads (it is a HAL-driven runner, not
  hardware-specific). A rename to `DeployRunner`/`HalRunner` + unifying the
  `rskill_runner_node` loop onto it is the longer-term path to a single
  recorder-capable inference loop across sim-run, deploy-sim, and deploy-run.
