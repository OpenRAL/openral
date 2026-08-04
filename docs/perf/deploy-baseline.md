# Deploy bringup baseline

Measured on the reference host, `refactor/deployment_optimization` @ 2026-08-04.

Re-run end to end after a **clean rebuild** — worktree venv via `just sync`, full
`colcon build --cmake-clean-cache` (12 packages incl. the C++ safety kernel),
and a fresh `openral:x86` image — on **both host and Docker**.

**Host:** RTX 4070 Laptop (8 GB), SO-101 follower on `/dev/ttyACM0`, two USB
bench cameras, ROS 2 Jazzy, warm HF cache.

Both paths were run end to end as a full ROS graph — not a component harness.
Numbers come from the `deploy.bringup` spans and the dashboard's own
`/api/state`, i.e. the same surface an operator sees.

---

## `openral deploy sim` — `scenes/deploy/so101_box.yaml` (MuJoCo twin)

| node | transition | duration |
|---|---|---|
| `openral_hal_so100` | `on_activate` | **733.8 ms** |
| `openral_hal_so100` | `on_configure` | **267.5 ms** |
| `openral_world_state` | `on_configure` | 71.3 ms |
| `openral_world_state` | `on_activate` | 0.4 ms |
| `openral_skill_runner` | `on_configure` | 17.8 ms |
| `openral_skill_runner` | `on_activate` | 0.0 ms |
| `openral_reasoner` | `on_configure` | 0.6 ms → **FAILURE** (no LLM key on this host) |

## `openral deploy run` — `scenes/deploy/so101_bench.yaml` (real arm)

Run with `--no-enable-reward-monitor` to isolate these numbers from the known
6.5× reward-monitor load starvation, which is tracked separately.

| node | transition | duration |
|---|---|---|
| `openral_hal_so100` | `on_configure` | **111.7 ms** |
| `openral_hal_so100` | `on_activate` | 43.2 ms |
| `openral_world_state` | `on_configure` | 32.3 ms |
| `openral_world_state` | `on_activate` | 0.6 ms |
| `openral_skill_runner` | `on_configure` | 18.1 ms |
| `openral_reasoner` | `on_configure` | 0.8 ms → **FAILURE** (no LLM key) |

Skill dispatch (`rskill-smolvla-so101-eraser_place-bf16`, real arm actuating):
**34.4 s wall** for a 20 s execution budget, i.e. **~14 s** of load. Resident
GPU after load: **1676 MiB** (SmolVLA only). Teardown returned the card to
**16 MiB**.

### The headline: the 300 s timeouts are ~1000× the real cost

`sim_e2e.launch.py` gives the HAL and reasoner a **300 s** transition timeout,
justified in a comment by "HAL `on_configure` takes ~6 s, or ~27 s on a cold
robocasa kitchen". For an SO-101 the measured cost is **267 ms in sim and
112 ms on the real arm**. The comment is not wrong — it describes robocasa —
but until now there was no way to know it did not describe *your* robot. That
is the entire point of the `deploy.bringup` span.

Bringup is not where deploy latency lives. **Model load is**: ~14 s against
~0.9 s of total lifecycle transitions.

---

## Event-log noise

One second of a 30 Hz two-camera deploy previously produced **121 `info`
rows/s**, cycling the 200-slot ring every 1.65 s. Measured live after the
span-band inversion, on both paths:

| | debug | info | warn | error |
|---|---|---|---|---|
| `deploy sim` | 200 | 6 | 0 | 1 |
| `deploy run` | 200 | 6 | 0 | 1 |

Every `info` row is a `deploy.bringup` line; the single `error` is the
reasoner's failed configure. Routine 30 Hz traffic contributes **zero** rows to
the operator's INFO view.

---

## Three bugs only the live run could find

Each of these passed unit tests and would have shipped.

1. **The warm-up never ran.** `warm_up_lerobot_policy` called
   `policy.select_action` on a raw batch, but SmolVLA reads
   `observation.language.tokens`, which the *preprocessor* produces by
   tokenising `task`. Live result: `KeyError: 'observation.language.tokens'`.
   The non-fatal guard worked exactly as designed — a warning, deploy
   continued — which is precisely why it went unnoticed until the log was
   read. Fixed by running the batch through the adapter's own preprocessor,
   as `step()` does. Confirmed on the real arm: zero `warmup_failed`.

2. **`info` rows could not survive the flood.** Demoting the per-tick spans
   took info *generation* to zero, but the main ring is one FIFO shared with
   the debug stream: 193 of 201 rows were `hal.read_state`, about 7 s of
   history. Every `deploy.bringup` row was evicted within seconds of being
   emitted; only the reasoner's ERROR row survived, because errors were the
   only thing mirrored into the protected lane. An info row that cannot
   outlive the flood is no more useful than one that was never emitted. The
   lane now mirrors everything above debug.

3. **`rskill.execute` is emitted at two rates.** `rskill_runner_node` opens
   one per dispatched goal; `rSkillBase.step()` opens one per *tick*, under
   the same name. Promoting it to a headline put **60 rows into a 20 s
   window** — reproducing the exact flood the allow-list exists to prevent,
   and evicting the real headlines from the protected lane. Demoted. It
   cannot be a headline until the per-step emitter is renamed.

---

## Not captured

`phase_timer` load phases (`smolvla_imports` / `from_pretrained` /
`to_device`) are emitted at INFO through the structlog→OTel bridge, so they
reach the collector but not the console, and they had cycled out of the
dashboard ring by the time it was queried. The ~14 s load total above is
wall-clock around the dispatch, not a phase breakdown. For a per-phase
breakdown use the offline profiler, which renders the same phases directly:

```bash
just profile-load rskills/rskill-smolvla-so101-eraser_place-bf16
```

Also not covered here: the reasoner tick path (this host has no LLM key, so
`on_configure` fails by design) and the reward-monitor leg.


---

# Round 2 — clean rebuild, host **and** Docker

The numbers above came from a worktree that borrowed the main repo's venv via a
`PYTHONPATH` shadow. Redone properly: own venv (8.5 GB), own colcon overlay
(12 packages, `--cmake-clean-cache`, C++ safety kernel included), and a fresh
image built from this branch.

## Bringup — host vs container (real SO-101, reward monitor off)

| node · transition | host | docker |
|---|---|---|
| `openral_hal_so100` · on_configure | 111.7 ms | **91.7 ms** |
| `openral_hal_so100` · on_activate | 43.2 ms | 45.0 ms |
| `openral_world_state` · on_configure | 32.3 ms | 39.0 ms |
| `openral_skill_runner` · on_configure | 18.1 ms | 9.4 ms |
| `openral_reasoner` · on_configure | 0.8 ms (FAILURE) | 0.6 ms (FAILURE) |

Event log on both: ~200 debug / 6-9 info / 1 error, every info row a bringup
line. Cards populated identically — 6 joints, 2 camera thumbnails (8-10 KB),
system card (CPU / RAM / GPU 2058 MiB of 8188).

## rSkill inference — SmolVLA on the real arm

167 distinct `rskill.chunk_inference` chunks sampled on the host, `pytorch` /
`cuda:0`, `chunk_size=1`:

| | ms |
|---|---|
| min | 6.05 |
| **p50** | **8.80** |
| p95 | 15.63 |
| max | 426.62 |

Sustained ~114 Hz against the 33.3 ms budget at 30 Hz — the policy is roughly
3.4x faster than the control loop needs. The 426 ms max is the cold first
chunk. Docker's card sampled 7.30 ms, consistent with the host p50.

## Skill load: the image is faster than the host

| | wall | load |
|---|---|---|
| host | 51.4 s (30 s budget) | **~21 s** |
| docker | 32 s (20 s budget) | **~12 s** |

The image wins because `UV_COMPILE_BYTECODE=1` ships precompiled `.pyc`; the
host venv byte-compiles the torch/lerobot tree on first import. That is direct
confirmation the Dockerfile setting is doing its job.

---

## Three more defects, found only by building and running properly

1. **`HF_HOME` never reached the runtime image.** The pin was added next to
   `UV_COMPILE_BYTECODE` in the **builder** stage, and `ENV` does not cross
   stages — `docker run … env` showed `HF_HOME=` empty. So the fix shipped in
   the previous commit did nothing at all. Moved to the `final` stage;
   verified `HF_HOME=/opt/openral/hf-cache` at runtime, and the mounted cache
   produced **zero re-downloads**. (`UV_COMPILE_BYTECODE` is correctly
   builder-only — it only affects install-time `.pyc` generation.)

2. **Host tooling cannot drive the container graph.** The image sets
   `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`; a host shell with `RMW` unset gets
   Fast-DDS, so `ros2 action send_goal` from the host hangs to its timeout
   against a container action server even with `--network host`. Dispatch from
   inside the container, or export the matching RMW on the host.

3. **A stale HAL node silently poisons the serial bus.** Two `deploy run`
   attempts failed with `Failed to write 'Lock' on id_=2 … id_=4 … There is no
   status packet!`, a different servo each time. All six servos answered a
   direct probe; the cause was a leftover `lifecycle_node.py` from a killed
   run still holding the Feetech bus. Worth knowing that the symptom points at
   the hardware and the cause is a process.

---

## Latency metrics — unified with their spans (was a gap)

All four were recorded only inside `openral_runner.InferenceRunnerBase` /
`DeployRunner`, which the ROS deploy graph never instantiates:
`rskill_runner_node` runs its own tick loop ("mirrors `DeployRunner._tick_impl`
but trims", per its own comment). A live `deploy run` therefore produced
per-chunk inference *spans* and **no latency histograms at all** — no p95, no
threshold line, nothing in the Metrics panel for the number an operator most
wants.

Three of the four now come from the same seam as their span, which is what
makes them impossible to diverge:

| metric | now emitted by |
|---|---|
| `openral.inference.duration` | `inference_span` — covers eval *and* deploy, since both open that span |
| `openral.hal.read_state.duration` | `_hal_duration_metric`, paired with the span in the shared HAL base |
| `openral.hal.send_action.duration` | same |

Verified live on the real SO-101 with the ollama reasoner up. Before a skill
dispatch the panel carried only `hal.read_state.duration` (50 samples — the HAL
ticks without a skill). After one dispatch:

    openral.inference.duration        14 samples
    openral.hal.read_state.duration   63 samples
    openral.hal.send_action.duration   6 samples

`openral.tick.duration` is deliberately **not** unified. Its unit is the
library runner's whole tick — sensors + HAL + skill + safety — and no such unit
exists on the ROS graph, where those are four separate processes. Recording the
skill step under that name would give one metric two meanings, the same trap as
`rskill.execute`.

---

## Reasoner verified end to end with ollama

The earlier rounds ran with no LLM key, so `openral_reasoner.on_configure`
failed by design and the reasoner path was never exercised. Re-run against a
local ollama (`qwen3:4b`, pulled because the two `*:cloud` models on this host
return `requires a subscription`):

* `deploy.bringup · node=openral_reasoner · transition=on_configure · 336.9 ms`
  — **info**, i.e. a successful configure, with "6 skills in palette". That
  336.9 ms includes the palette seed, which now uses the fast dependency probe.
* `reasoner.tick` fired at **52.1 s** then **8.9 s** — the first tick pays
  ollama's cold model load.
* The mission state populated correctly (`t1: "place the erase on the blue
  square"`, active).
* `qwen3:4b` then returned `error_kind: ROSReasonerInvalidPlan` with
  `tool=null`. That is a model-capability limit, not a wiring fault — a 4 B
  model against the full tool palette, consistent with the earlier note that
  glm-4.6v thrashes where glm-5.2 drives. The infrastructure works; the model
  is too small to plan.

Configured via the deprecated-but-supported legacy shim
(`OPENRAL_REASONER_LLM_PROVIDER=ollama`, `OPENRAL_REASONER_LLM_MODEL=qwen3:4b`).
Note the model-first path cannot express this today: the launch forwards
`OPENRAL_REASONER_MODEL` and `OPENRAL_REASONER_ENDPOINT` into the reasoner's
`additional_env` but **not** `OPENRAL_REASONER_DIALECT`, which an uncurated
model id requires.

## A promotion that measurement did *not* condemn

`world.scene_objects` looked like it was flooding the info band — ~20 rows in
one snapshot, the same shape as the `rskill.execute` mistake. Measured over a
20 s window it fires at **0.10/s**; the rows had simply accumulated over
several minutes. Left as a headline span. Worth recording that the check was
run, because the two preceding rounds each found a real one.
