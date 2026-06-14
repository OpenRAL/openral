# ADR-0048: A sim `/clock` publisher for the deploy-sim graph

- Status: **Proposed**
- Date: 2026-06-11
- Related: [ADR-0025](0025-reasoner-managed-background-services.md) (Nav2 / slam_toolbox
  bringup); [ADR-0030](0030-geometric-safety-collision-checking.md) (kernel world-collision
  check via octomap); [ADR-0034](0034-deploy-sim-scene-attach-for-arms.md) (deploy-sim
  scene-attach + the 2026-06-04 *idle-stepper* amendment that makes a continuous sim clock
  feasible); [ADR-0045](0045-isaac-sim-backend-integration.md) (Isaac Sim sidecar);
  [ADR-0046](0046-nvidia-gr00t-backend.md) (GR00T sidecar).
- Supersedes nothing. **Depends on a safety-WG review (see §Safety review) before it may move
  to Accepted** — it changes the clock the kernel's world-collision check runs on.

## Context

### The bug this came from

`openral deploy sim --config scenes/deploy/robocasa_pnp.yaml` drove the `panda_mobile` base
**into** a kitchen object during a Nav2 `navigate_to_pose`, instead of routing around it. The
controller logged `Control loop missed its desired rate of 20.0000 Hz. Current loop rate is
inf Hz.` repeatedly, then the goal aborted (`STATUS_ABORTED … unpopulated/too-small costmap`).

Root cause: **a split clock domain.** deploy-sim publishes **no** `/clock` topic, yet the Nav2
stack, `slam_toolbox`, and `robot_state_publisher` were launched with `use_sim_time:=true`. In
ROS 2, `use_sim_time=true` with no `/clock` publisher does **not** fall back to wall time — the
node's clock pins at `t=0` forever. Meanwhile the HAL (the authoritative source of
`odom→base_link` TF and `/scan`) stamps on **wall-clock** (`time.time_ns()`, ~1.78×10⁹ s). So:

- the local/global costmap (clock=0) saw every `/scan` as ~1.78×10⁹ s "in the future",
  rejected it on the observation-buffer / TF-tolerance check, and stayed **empty** → the
  controller planned a straight line through the obstacle → **collision**;
- the controller loop `dt = now − last = 0` every iteration → `1/0 = inf Hz`;
- `slam_toolbox` (clock=0) never advanced its pose-graph → `map→odom` stayed degenerate.

This is the **same failure mode** ADR-0030's octomap leg already works around: `octomap_server`,
`octomap_bridge`, and `ros_image_detector` were each pinned to `use_sim_time:=false` precisely
because "deploy-sim has no free-running /clock publisher … the cloud looks 'in the future',
every insert is dropped, and the octree stays empty so the arm crashes into the table
uncaught." The nav leg simply never got that memo.

### The stopgap already landed (`fix/nav-collision`)

The immediate fix replaced the scattered, disagreeing `use_sim_time` literals with **one**
computed source of truth in `sim_e2e.launch.py`: a launch arg `enable_sim_clock`
(default `false`) resolved once via `_resolve_sim_clock()` and threaded through **every** node
in `compose_runtime_graph` (HAL, Nav2, slam_toolbox, robot_state_publisher, octomap,
octomap_bridge, detector). With no `/clock` publisher the default keeps the **entire** graph on
wall-clock, coherent with the HAL — and a single node can no longer silently disagree.

That stops the collision. But it also means the graph is **permanently on wall-clock**, which
forfeits the benefits of sim-time: deterministic replay (CLAUDE.md §1.8), correct time-warp
when the sim runs slower/faster than realtime, and clean `bt_navigator` deadlines measured in
sim-seconds. `enable_sim_clock` is deliberately wired as the *switch* for this ADR: the durable
fix is to make a sim `/clock` exist so the switch can be flipped to `true`.

### Why it is feasible now (it wasn't before)

Before ADR-0034's 2026-06-04 *idle-stepper* amendment, the sim only advanced while a skill sent
actions (step-on-action). A `/clock` derived from sim-time would have frozen between actions,
and Nav2 — which needs `/clock` to advance to emit `cmd_vel` — could never bootstrap (the sim
needed `cmd_vel` to advance; `cmd_vel` needed the clock to advance). The idle stepper breaks
that deadlock: it steps the env with a HOLD action at `camera_rate_hz` (~10 Hz wall) whenever
idle, so **sim time now advances continuously** (~0.5× realtime idle at `control_freq=20` →
`control_dt=0.05 s`; ~1× while Nav2 drives the base). A `/clock` read from the simulator's own
time is therefore now publishable on a continuously-advancing basis.

## Decision

Add an **optional sim `/clock` publisher** to the deploy-sim graph, gated by the existing
`enable_sim_clock` flag, and flip the whole graph to sim-time when it is on. Concretely:

### 1. Expose simulation time through the `SimRollout` seam

Add an **optional** `sim_time_ns() -> int | None` to the `SimRollout` protocol
(`python/sim/src/openral_sim/rollout.py`). It returns the backend's authoritative elapsed
simulation time in nanoseconds, or `None` when the backend has no sim clock.

- **MuJoCo / robocasa / robosuite backends**: return `round(data.time * 1e9)` (the MuJoCo
  `MjData.time`, advanced by `model.opt.timestep × n_substeps` per `env.step`).
- **Sidecar backends (Isaac Sim ADR-0045, GR00T ADR-0046)**: add a `sim_time` field to the ZMQ
  step/reset reply and surface it here. Until the sidecar protocol carries it, return `None`.
- **Pure-gym backends without a clock (pusht, …)**: return `None`.

`SimAttachedHAL` re-exposes this as `sim_time_ns()`, reading it after each
`_step_and_cache` (both the `send_action` and `idle_step` paths) so the value is monotonic and
fresh.

### 2. Publish `/clock` from the HAL's sensor bridge

The HAL — already the clock authority for `/scan` + TF — publishes `rosgraph_msgs/msg/Clock`
from `hal.sim_time_ns()` on **every** env step (action **and** idle), at the idle-stepper /
sensor-bridge cadence. Requirements:

- **monotonic** — never republish a stale or decreasing stamp;
- **first** — `/clock` must be on the bus before the sim-time consumers start their first
  control tick (startup ordering, §3);
- **gated** — only created when `enable_sim_clock=true` **and** `hal.sim_time_ns()` is non-`None`.

### 3. Flip the graph to sim-time via the existing flag

`enable_sim_clock=true` sets `use_sim_time=true` for the whole `compose_runtime_graph`
(already wired by the stopgap), now including the HAL itself, so `/scan`, TF, and joint_states
carry **sim-time** stamps consistent with `/clock`. Startup ordering: bring the HAL (clock
publisher) to ACTIVE before Nav2's controller and the costmaps begin ticking — the existing
`OnStateTransition(hal → active)` gate that already defers the Nav2 include is the natural hook.

### 4. Backend-capability gate (the CLI must refuse silent breakage)

`openral deploy sim` resolves `enable_sim_clock` against the scene's backend:

- backend exposes sim-time (`sim_time_ns() is not None`) → may enable;
- backend returns `None` (sidecars before the protocol bump; clock-less gym envs) → **force
  `false` and log it** (CLAUDE.md §1.4 — no hidden fallback). Never set `use_sim_time=true`
  for a backend that cannot feed `/clock`; that reintroduces the exact `t=0` bug.

### Backend capability matrix

| Backend | Sim-time source | `/clock` feasible | Notes |
|---|---|---|---|
| robocasa / robosuite (MuJoCo) | `MjData.time` | ✅ | Primary target; idle stepper already advances it |
| LIBERO / MetaWorld / aloha (MuJoCo) | `MjData.time` | ✅ | Same accessor |
| Isaac Sim (ADR-0045 sidecar) | PhysX sim-time, **inside the sidecar** | ⚠️ needs ZMQ reply field | Hardest case; no ROS topics today |
| GR00T (ADR-0046 sidecar) | n/a (policy, not sim) | n/a | Not a sim backend |
| pusht / clock-less gym | none | ❌ | Forced wall-clock |

## Safety review

**This section gates the ADR.** Per CLAUDE.md §3, changes touching the safety kernel's
behavior require (a) a safety-WG reviewer, (b) a hazard-log update, and (c) tests proving the
new behavior is **at least as conservative**. The kernel's **world-collision** check
(ADR-0030) rasterizes `octomap_server`'s octree into `/openral/world_voxels`; ADR-0034's note
shows the octree is pinned to wall-clock *for safety* — a sim-time octree with no `/clock`
goes empty and the arm "crashes into the table uncaught."

Implications to be reviewed **before Accepted**:

1. **The world-collision octree moves onto sim-time when the flag is on.** This is *arguably
   more* correct — the octree would then share one clock with the depth cloud and the TF it is
   inserted against, removing the wall-vs-sim skew that ADR-0030/0034 fought. But it is a
   change to the clock the kernel's `validate()` consumes, so it is **in scope for safety-WG
   sign-off**, not a launch-config detail.
2. **Empty-octree-on-misconfig must fail safe, not open.** The capability gate (§4) must make it
   impossible to run `use_sim_time=true` without a live, advancing `/clock`. The required test:
   with `enable_sim_clock=true` but a stalled/absent clock, the kernel must **not** report a
   clear workspace (no silent "octree empty → no obstacle → motion allowed"). Prefer an explicit
   `ROSPerceptionStale` (sensor older than deadline) over an empty-and-permissive octree.
3. **Staleness deadlines are in seconds.** `ROSPerceptionStale` thresholds and `bt_navigator`
   timeouts are wall-second-tuned today; under sim-time warp (~0.5× idle) they must be
   re-derived in sim-seconds so a slow sim is not mistaken for a stale sensor (false E-stop) and
   a fast sim does not mask a genuinely stale frame (missed hazard).
4. **Monotonicity is a safety property.** A `/clock` that jumps backward (e.g. an episode reset
   that rewinds `MjData.time`) could make a fresh obstacle frame look old. The publisher must
   guarantee a monotonically non-decreasing stamp across `env.reset` (offset the published
   clock so it never rewinds), and the reset path's interaction with the world-collision check
   must be in the hazard log.
5. **No safety check may be disabled by this flag.** `enable_sim_clock` only selects the *clock
   source*; it must not gate, weaken, or bypass any kernel check (CLAUDE.md §3 "never add a flag
   that disables safety").

**Conservatism argument to be validated:** with a correctly-advancing sim `/clock`, every
consumer (costmap, octree, kernel) reads one coherent clock, eliminating the wall-vs-sim skew
that currently *drops* obstacle observations. The new path should therefore mark **more**
obstacles, not fewer. The safety-WG must confirm this with a test that the world-collision
check flags a known obstacle under `enable_sim_clock=true` that the wall-clock path also flags
(at least as conservative), plus the fail-safe test in (2).

## Consequences

**Positive**
- Deterministic, replayable skill executions (CLAUDE.md §1.8): sim-time replays bit-for-bit;
  wall-clock does not.
- One coherent clock across the graph — no more wall-vs-sim TF skew dropping scans/clouds.
- `bt_navigator` / controller deadlines become sim-second-accurate regardless of host speed.

**Negative / cost**
- Touches four layers (sim backend, HAL, launch graph, safety kernel review) → larger blast
  radius than the stopgap; must ship behind the default-`false` flag until proven.
- Sidecar backends (Isaac) need a protocol bump before they can participate.
- `/clock` advances at sim-rate, not wall-rate (bursty: ~0.5× idle, ~1× under nav) — operators
  watching wall-time will see actions take longer/shorter than wall-seconds.

## Alternatives considered

1. **Stay on wall-clock forever (ship only the stopgap).** Simplest; loses determinism and
   sim-time correctness; leaves `enable_sim_clock` as a dead switch. Acceptable as the interim
   state, which is exactly what the stopgap delivers.
2. **A standalone `/clock` node that integrates wall-time at a fixed rate.** Decouples from the
   sim, but publishes *fake* sim-time unrelated to `MjData.time` — it would desync from the
   physics the moment the sim runs off-realtime, reintroducing skew. Rejected (CLAUDE.md §1.2
   truth-over-plausibility).
3. **Make Nav2 alone wall-clock and leave the arm leg as-is.** This is the stopgap; it is
   correct but forfeits sim-time graph-wide. This ADR is the follow-on.

## Rollout (phased, flag-gated)

1. **Phase 0 (done — `fix/nav-collision`).** Single `enable_sim_clock` flag, default `false`;
   whole graph wall-clock; collision fixed; regression test pins the default.
2. **Phase 1 (done — `feat/sim-time-ns-rollout-seam`).** `sim_time_ns()` on `SimRollout` +
   MuJoCo backends + `SimAttachedHAL`, monotonic across `env.step` / `env.reset`.
3. **Phase 2 (done — `feat/deploy-sim-clock-publisher`).** The HAL publishes `/clock` from the
   captured `sim_time_ns` (via the ADR-0049 publisher thread; `ProprioFrame.sim_time_ns`), gated
   on `use_sim_time` + a live sim clock with a loud refuse-and-warn when the backend has none. A
   `--enable-sim-clock` CLI flag (default off) flips the graph. **Key subtlety:** the sim-driving
   idle stepper must run on a `SYSTEM_TIME` clock, else its node-clock timer deadlocks (no step →
   no `/clock` → no fire). **Verified live** (robocasa, RTX 4070): `/clock` advances ~0.46×
   realtime, Nav2/slam/octomap run on sim-time, costmap populated (1382 cells), no "inf Hz".
4. **Phase 3 (safety-gated — NOT signed off).** `enable_sim_clock=true` *already* moves the
   world-collision octree onto sim-time (octomap inherits the flag). It is **opt-in, default
   off**, and the CLI help flags it as safety-WG-gated — but the safety-WG review, hazard-log
   update, and the conservatism + fail-safe tests in §Safety review are **still required** before
   it is a documented, supported (default-on) configuration.
5. **Phase 4 (done, code-only — `feat/deploy-sim-clock-publisher`).** The Isaac sidecar reply
   carries `sim_time_ns` (`tools/_isaac_scene_base.sim_time_ns` from `SimulationContext.current_time`)
   and `_IsaacSimSidecar.sim_time_ns()` surfaces it. **Untested** — Isaac Sim is a ~50 GB,
   RTX-only, license-gated Omniverse install absent on the dev host (and the disk has <13 GB
   free), so the Isaac e2e is deferred to an Omniverse-provisioned machine. GR00T is a policy,
   not a sim backend, so it is out of scope.

## Open questions

- What is the right published-clock rate — every `env.step`, or decimated to the control rate?
  (Too sparse starves Nav2's 20 Hz loop; too dense floods the bus.)
- Does robocasa's `env.reset` rewind `MjData.time`, and if so, what monotonic offset does the
  publisher apply?
- Should `enable_sim_clock` default flip to `true` for MuJoCo scenes once Phase 3 lands, or stay
  opt-in pending field confidence?
