# openral_nav2_bringup

Bringup wrapper for upstream **Nav2** as a Reasoner-managed
background service (the second such service, after
[`openral_slam_bringup`](../openral_slam_bringup/)). We do **not** reimplement
Nav2 — this package only ships the per-deployment launch + parameter glue so the
OpenRAL Reasoner can start/stop navigation on demand and route a
`NavigateToPose` goal.

```
lidar profile:  /scan ─▶ payload scan filter ─▶ /openral/nav2/scan ─▶ Nav2 (obstacle_layer)
visual profile: /map ───────────────────────────────────────────────▶ Nav2 (static_layer)
                                                          └──/navigate_to_pose──▶ /cmd_vel
                /openral/world_state_fast ─▶ payload footprint ─▶ */footprint
```

## Costmap profiles (backend-agnostic)

Nav2 is selected to match the SLAM backend via the `slam_backend` launch arg, so
navigation works **regardless of how the 2D map is built**:

| `slam_backend` | Config | Costmap obstacle source |
|---|---|---|
| `lidar` (default) | `nav2_panda_mobile.yaml` | `/scan` via `obstacle_layer`/`voxel_layer` |
| `visual` | `nav2_visual.yaml` | **`/map`** `OccupancyGrid` via `static_layer` |

The **visual** profile lets a lidar-less robot (cuVSLAM + nvblox)
navigate: the global+local costmaps consume the backend-agnostic `/map` (which
nvblox publishes, remapped from its `static_occupancy_grid`) via `static_layer`
with `map_subscribe_transient_local: False` (nvblox's `/map` is RELIABLE+VOLATILE,
not latched), and the collision_monitor's `/scan` source is disabled. Everything
else mirrors the lidar base — `nav2_visual.yaml` is **generated** from it by
`tools/gen_nav2_visual.py` (re-run after editing the base). Verified live: the
visual profile activated and `ComputePathToPose` returned a path consuming only
`/map` (no `/scan`).

> 3D-lifted detected objects are already backend-agnostic — they
> use the `map` **TF frame** (which cuVSLAM publishes like slam_toolbox), not the
> `/map` topic, so they map into the world identically on both backends.

## Run

Normally started by the Reasoner as a background service when the active goal
needs navigation; the `openral deploy sim` / `deploy run` graph wires
the leg when the robot declares a lidar (`has_lidar`) **or** vision SLAM
(`has_vision_slam`) and forwards the resolved `slam_backend`.
Standalone:

```bash
ros2 launch openral_nav2_bringup nav2.launch.py slam_backend:=lidar    # /scan
ros2 launch openral_nav2_bringup nav2.launch.py slam_backend:=visual   # /map
```

Requires upstream `ros-${ROS_DISTRO}-navigation2` / `nav2_bringup` (in the deploy
images) and a map source — `openral_slam_bringup` (slam_toolbox `/map` for lidar,
or cuVSLAM+nvblox `/map` for visual).

## The Nav2 ↔ safety-kernel boundary

Two independent world models run side by side here, and conflating them is the
mistake this section exists to prevent. Neither one covers for the other.

| | **Nav2** | **C++ safety kernel** |
|---|---|---|
| Checks | the **base footprint** — one 2-D polygon | the **arm's link capsules + attached payload primitives** — 3-D |
| Against | the 2-D costmap | `/openral/world_voxels`, a 3-D occupancy grid |
| Fed by | `/openral/nav2/scan` (lidar) or `/map` (visual) | `openral_octomap_bridge`, from `octomap_server`'s octree |
| Acts by | refusing a path / trajectory, then `/cmd_vel` | vetoing `/openral/candidate_action` → `/openral/safe_action`, or E-stopping |
| Blind to | anything not in the 2-D grid: overhangs, table tops, the arm's own reach | anything outside the local voxel box, and the base's path |

**`/cmd_vel` does not pass through the kernel, and that is a recorded decision**
(ADR-0040: base collision avoidance relies entirely on Nav2's costmap). Nav2
publishes `geometry_msgs/Twist`; `openral_hal.mobile_base_bridge.MobileBaseBridge`
maps each message to a `BODY_TWIST` `Action` and applies it through the HAL
node's `_send_action_traced` — it never reaches `/openral/candidate_action`, so
the kernel never sees it, never bounds it, and never vetoes it. Velocity caps on
this path are Nav2's own `velocity_smoother`. This is stated in the bridge's
module docstring too; it is a boundary, not an oversight, and it is written
down in both places on purpose.

**The one thing that does cross.** `_on_cmd_vel` returns early while the HAL
node is latched E-stopped, so an E-stop — the kernel's included — does stop the
Nav2 path, at the HAL rather than in the kernel. Stopping is the only kernel
authority over base motion. There is no per-command bound.

**What this costs, and why the dynamic footprint below matters.** A carried
object is checked in 3-D by the kernel and is *not* checked at all by Nav2
unless Nav2's footprint says it is there. Without the footprint publisher, the
robot's arm and payload are protected while the chassis drives them into a
counter. So the payload has to appear in **exactly one** place on each side of
the boundary, and the two sides are deliberately opposite:

* **Kernel side** — the payload is *robot*: collision-active attached geometry
  the kernel keeps checking, and `openral_octomap_bridge` clears its cells out
  of the voxel grid so the robot is not stopped by itself.
* **Nav2 side** — the payload is *robot* here too: it joins the footprint
  polygon, and `payload_scan_filter_node` drops its returns from the scan the
  costmaps and the collision monitor read, for the same reason.

Both sides read the same attachment set on `/openral/world_state_fast`, so they
cannot disagree about what is attached.

## Nav2 is base-only

**The costmaps' footprint is the manifest's bare chassis, and nothing grows
it.** The 2-D costmap owns base geometry; the 3-D safety kernel owns the arm
and anything carried. This replaces the dynamic footprint publisher that
PR #143 shipped, which is now removed.

### Why the growth was wrong

It projected 3-D geometry onto a 2-D costmap, and the two do not describe the
same world.

**It forbade the poses the tasks require.** Every RoboCasa place target is a
fixture the payload must *enter* — a cabinet, a sink, a fridge. Grow the
footprint over the payload and its ground projection lands on the fixture the
base has to approach, so Nav2 refuses the one approach that succeeds. Placing
into a fridge is not an edge case for this robot; it is the task.

**And it protected against nothing here.** Measured on
`scenes/deploy/robocasa_deliver_straw.yaml`:

| | measured |
| --- | ---: |
| `odom → base_link` (TF) | **0.700 m** |
| local costmap voxel column (`origin_z` + `z_resolution × z_voxels`) | 0.00 – 1.28 m |
| where scan returns land | **≈ 0.70 m** |
| carried object (`glass_cup`) | **0.981 m** |

The costmap is one horizontal slice, and a carried object rides ~0.28 m above
it. A payload cannot collide with an obstacle the costmap knows about unless
that obstacle is *also* tall — which the costmap has no way to represent. So
the growth traded a real, frequent false block for protection against a case it
could not distinguish anyway.

**What is genuinely given up.** A payload sticking forward could clip a *tall,
thin* obstacle that the base itself clears. The kernel catches that in 3-D —
its octomap bridge covers a ball of r = 1.05 m centred at z = 0.5 in
`base_frame`, i.e. z ∈ [−0.55, 1.55], which contains the payload — but as an
**E-stop, not an avoidance**, because `/cmd_vel` never passes through it
(ADR-0040, above). That is the accepted cost: a rare stop instead of a routine
refusal to do the task.

### What replaces it

Nothing new. `RobotDescription.nav2_param_overrides()` already substitutes
`footprint_polygon` into both costmaps' `footprint` at launch, so the polygon is
static and correct without a publisher. What did NOT go away is the scan filter
(next section) and `CostCritic.consider_footprint`, now **`true`** — with a
fixed chassis polygon, scoring the real outline instead of the centre cell is
strictly more accurate and was measured at +0.53 ms on the live loop.

### A defect this surfaced, not yet fixed

While measuring the above: `openral_sim.backends.robocasa.synthesize_laser_scan_2d`
casts its rays at **world z = 0.30 m** (`origin[2] = laser_height_m`, absolute —
the base body is at z = 0.000), but publishes the result in `base_link`, which
TF puts at **0.700 m**. The sim therefore samples the world at one height and
tells Nav2 the returns came from another, 0.40 m higher. This config's
`voxel_layer` comment (`z_resolution` raised to reach 1.28 m for a lidar
"at z≈1.05 m") is downstream of the same confusion. Neither the base-only
decision nor the measurements above turn on it — the payload is above every
candidate height — but it should be reconciled before anyone reasons about
obstacle heights again.

## The robot's own returns

A 2-D lidar on a real mobile base sees the base: chassis, mast, arm. Unfiltered,
those returns mark the costmap and never clear, and the robot concludes it is
surrounded by itself. **Sim hides this completely** —
`openral_sim.backends.robocasa.synthesize_laser_scan_2d` compares each
`mujoco.mj_ray` hit's `body_rootid` against the base body's and re-casts past
its own tree, so a self-return never enters `/scan` in the first place. That
mechanism is a MuJoCo body-id comparison; it has no real-hardware counterpart,
and this repo has no lidar driver, no lidar launch file, and no `SensorSpec`
field that could express a mount pose, a blind sector or an angle mask. Until #194 the only
real knob was `panda_mobile`'s `range_min_m: 0.55`, a blunt radial cutoff
that deleted every real obstacle inside 0.55 m in every direction to hide a
chassis whose circumscribed radius is 0.43 m. #194 lowered that field to the
sensor minimum (0.05 m) on the strength of this node, so the shaped filter
below is now the only self-exclusion the hardware path has.

`payload_scan_filter_node` therefore filters the robot too, with a shaped test
instead of a radial one: a beam is dropped when its endpoint, transformed into
`base_frame`, lies inside the manifest's **bare chassis** `footprint_polygon`.

What the swap bought, measured on the live graph (`robocasa_deliver_straw`,
pinned seed 3, whole graph relaunched per arm; raw output in
`docs/reference/data/base-scan-range-min-2026-09-02.jsonl`):

| | `/scan` usable | inside 0.55 m | reaching `/openral/nav2/scan` | nearest |
|---|---|---|---|---|
| `range_min_m: 0.55` | 192 | **0** | 192 | 0.555 m |
| `range_min_m: 0.05` | 344 | 152 | **252** (92 dropped as chassis) | 0.344 m |

So 60 real near-field returns now reach Nav2 where none could before, and the
shaped filter still removes the 92 whose endpoints it can prove are the robot.
Every one of the 152 near returns resolves to real kitchen geometry by body
name — cabinet doors, the fridge housing, the freezer — and none to the robot:
`robot0_link0`, `robot0_link7` and `mobilebase0_wheeled_base` all share the base
body's `body_rootid`, so the sim fan's identity self-exclusion already covers
the whole tree including the arm. The MPPI loop is unmoved: 10.03 → 10.25
ms/cycle against the 50 ms budget, 600 → 601 cycles in 30 s, none dropped.

The conservative direction here is the **opposite** of the payload's, and that
is the whole design:

* For the payload, the dangerous mistake is failing to remove — a payload left
  in the costmap only makes Nav2 more cautious. Bad input keeps it.
* For a self-return, the dangerous mistake is removing a real obstacle we
  mistook for the robot. So the self half removes only what it can *prove*,
  and on a missing manifest, an unresolvable `base_frame ← scan_frame` TF or a
  degenerate polygon it removes **nothing**.

Why the chassis polygon is a proof:

* A return inside the chassis outline is the chassis, or an object standing
  where the chassis already is — not a place an object can be.
* It is the *same* polygon this package publishes to Nav2 as the robot. Nav2's
  `footprint_clearing_enabled` already frees those cells every update and
  `collision_monitor` reads the same outline, so removing those returns takes
  away nothing Nav2 could have acted on. What it does remove is the
  collision-monitor false positive — that node reads the raw scan with no
  costmap clearing in between, and it is what brakes for the robot's own body.
* It is the **bare** chassis, never the payload-grown hull: the hull spans free
  air between chassis and payload, and the payload's own primitives already
  cover the payload exactly.
* The kernel's per-link OBBs in `link_collision` are deliberately *conservative*
  over-approximations. Over-bounding is right for a collision check and wrong
  for deleting sensor returns — the air between a link and its box is air a real
  obstacle can occupy — so this node does not use them, and the arm above the
  scan plane is out of scope for it.

`self_margin_m` defaults to `0.0` and should stay there; every millimetre past
the chassis deletes returns Nav2 *would* have acted on. Without `robot_yaml`
the self half simply does not run and the node warns once.

## Measured: what `consider_footprint: true` costs, and why it is `true`

`benchmark/cost_critic_footprint_bench.cpp` times upstream's own
`FootprintCollisionChecker::footprintCostAtPose` against the real Jazzy
`libnav2_costmap_2d_core`, at the real polygons and the shipped 3 m / 0.05 m
local costmap. It is out of the CMake build on purpose (a measurement, not an
artifact); the build line is in its header comment.

On an i5-8600K, at `batch_size 2000 × time_steps 56 / trajectory_point_step 2`
= **56 000 calls per controller iteration** (four runs; the ms/iteration spread
is in brackets):

| | circumscribed radius | ns/call | ms/iteration | Δ vs `false` | of the 50 ms cycle |
|---|---|---|---|---|---|
| base only | 0.444 m | 149 | 8.3 [8.28–8.32] | **+8.1** | **17 %** |
| carrying (0.860 m reach) | 0.863 m | 178 | 9.9 [9.87–9.94] | **+9.7** | **20 %** |

The point-only path (`consider_footprint: false`) is 0.15 ms — the flag is
essentially the whole cost.

**It is every sampled point, not a fraction of them.**
`CostCritic::findCircumscribedCost` returns `0.0` whenever `inflation_radius`
is below the footprint's *circumscribed* radius, and `inCollision`'s guard is
`cost >= possible_collision_cost_ || possible_collision_cost_ < 1.0f` — so a
`0.0` makes the full-footprint check unconditional. Both polygons are in that
regime against the shipped `inflation_radius: 0.40`, so the worst case is the
normal case and the number above is not data-dependent. This also corrects a
comment this PR shipped: the ~0.364 m circumscribed radius quoted there came
from `robot_radius`, but the costmaps are configured with the *polygon*, whose
padded farthest vertex is 0.444 m. **That correction is independent of the
flag** — it is a property of the polygon, and it stands whether
`consider_footprint` is on or off.

Raising `inflation_radius` above 0.444 m would restore the cheap gate for the
bare chassis. Nothing restores it while carrying — the payload's circumscribed
radius is most of the 3 m local costmap — so it is left at 0.40 m rather than
changed blind.

### MEASURED 2026-08-28: the full cycle fits with the flag on

The precondition below has been met. `scenes/deploy/robocasa_deliver_straw.yaml`
was driven with the full stack (SLAM + Nav2 + octomap + kernel gate) on
`q-laptop`, and `controller_server`'s own CPU per published control cycle was
measured in all four arms, 2 runs each, each run validating which polygon the
costmap actually adopted:

| footprint | `consider_footprint` | CPU / cycle | of 50 ms |
| --- | --- | ---: | ---: |
| bare (0.72 m) | `false` *(ships today)* | 9.58 ms | 19 % |
| bare (0.72 m) | `true` | 10.11 ms | 20 % |
| grown (1.23 m) | `false` | 9.79 ms | 20 % |
| **grown (1.23 m)** | **`true`** | **9.97 ms** | **20 %** |

**The loop fits in every arm**, with ~40 ms of headroom: 500-501 cycles per 25 s
window (exactly 20 Hz, none dropped) and **one** `Control loop missed its
desired rate` warning in the whole session. Nav2 logged
`inflation radius (0.400000) is smaller than the circumscribed radius
(0.908020)` for the grown polygon, confirming the cheap gate was defeated and
this is the unconditional-check regime, i.e. the worst case.

**The measured delta of the flag was +0.53 ms (bare) / +0.18 ms (grown), not
the +8.1 / +9.7 ms the isolated benchmark below predicted.** That order of
magnitude is *not* explained here; the likeliest cause is how often
`CostCritic::inCollision` is actually reached per iteration versus the 56 000
calls the benchmark assumes, which is a property of `CostCritic::score` whose
source is not in the Jazzy binary install. Reported as a discrepancy, not
resolved.

**Caveats, both real.** The payload was *injected* — the probe published the
grown polygon, because no policy ran (`attached_objects count=0` throughout) —
so this is the controller carrying a grown footprint, not a policy-driven carry.
And it is one host, one route, one kitchen; a busier local costmap raises the
call count that the paragraph above turns on.

**Nothing here has been flipped.** The config, its comment and
`test_mppi_does_not_yet_consider_the_footprint` change together or not at all,
and that remains a maintainer decision. Method, full numbers and the validity
check: [`docs/reference/robocasa-carry-survey.md`](../../docs/reference/robocasa-carry-survey.md);
raw output in `docs/reference/data/nav2-mppi-loop-2026-08-28.jsonl`.

### The flag is now `true` — decided 2026-08-29

The deferral asked for one thing: *"run the composite scene with the whole MPPI
loop timed against the 50 ms budget, and show the full cycle still fits with the
flag on."* That is the measurement above, and it does.

Flipped together, as the deferral required:

* `config/nav2_panda_mobile.yaml` → `consider_footprint: true` (and
  `config/nav2_visual.yaml`, regenerated from it by `tools/gen_nav2_visual.py`).
* Its CostCritic comment, rewritten around the live numbers.
* `test/test_nav2_launch.py::test_mppi_considers_the_full_footprint` — renamed
  from `test_mppi_does_not_yet_consider_the_footprint`, still pinning the value
  so it cannot drift back silently.

**The polygon it scores is the bare chassis**, which is what makes this
uncontroversial: for a 0.70 × 0.50 m rectangle a centre-cell test is simply
wrong about the poses this base actually uses, since it routinely parks with
less clearance than its own 0.444 m circumscribed radius against counters the
rectangle clears.

**`inflation_radius` is deliberately left at 0.40 m.** Raising it to ≥ 0.444 m
would restore CostCritic's cheap gate and make the flag nearly free — and with
no payload growth, the circumscribed radius is now a constant of the manifest
rather than a function of what is being held, so that change is well defined for
the first time. It is still a navigation-behaviour change that moves path cost
everywhere, so it is a separate decision and has not been made.

### What is still open (issue #108)

* **A scene now drives the base while carrying** — `DeliverStraw`, pinned at
  seed 3 by [`scenes/deploy/robocasa_deliver_straw.yaml`](../../scenes/deploy/robocasa_deliver_straw.yaml).
  It is upstream RoboCasa, not a custom task, and it is in `composite_seen` —
  inside the `target50` set XR-1 RoboCasa365 reports against — so the policy
  stays in distribution. Measured at reset: the straw starts **0.50 m** away in
  the drawer the base is parked at (inside the Panda's 0.855 m reach, so it is
  grasped before any base motion) and the glass cup it must end up inside sits
  **3.795 m** away on the dining counter. `GetToastedBread` also qualifies, at
  up to 3.48 m.

  This was measured, not read off the source, and that distinction is load
  bearing: classifying the task source statically gives the *opposite*, wrong
  answer, because `Kitchen.get_fixture`'s docstring ("will search for fixture
  close to ref (within 0.10m)") does not describe its code — which keeps
  candidates within 0.10 m *of the nearest one*, a tie-break rather than a
  bound. Method and full measurements:
  [`docs/reference/robocasa-carry-survey.md`](../../docs/reference/robocasa-carry-survey.md).

  **Criteria 1 and 4 are met. Criteria 2 and 3 are obsolete** — both existed
  only to exercise a payload-grown footprint, and Nav2 is now base-only:

  * **Criterion 2** ("an aperture the bare chassis clears but the payload-grown
    polygon does not") has nothing to decide: there is no grown polygon. The
    property it was groping for — that a *rectangle* fits where its
    circumscribed *circle* does not — is real (measured free-corridor
    bottlenecks run 0.19–0.24 m against a 0.444 m circumscribed radius) and is
    exactly what `consider_footprint: true` now reads.
  * **Criterion 3** ("lidar-visible obstacles at the payload's height") was
    unsatisfiable and is now moot. The payload rides at ~0.98 m while scan
    returns land at ~0.70 m, so it never enters the slice. Under base-only that
    is the *desired* state, not a gap: the payload belongs to the kernel's 3-D
    check, and the scan filter's job is to keep it out of Nav2's world rather
    than into it.

  `loading_fridge`, the candidate this file used to name, is disqualified on the
  measurement: none of its eight classes is in `target50`, so it would put XR-1
  out of distribution.

  **What remains before #108 closes.** The loop measurement is done and the
  flag is flipped (above). What has *not* been run is the scene end to end under
  a policy: every measurement here was taken with the base driven by a direct
  `NavigateToPose` and the reasoner off, so `attached_objects` stayed 0
  throughout. An XR-1 run that opens the drawer, grasps the straw and carries it
  is the remaining acceptance, and the scene's closed-drawer precondition means
  a failure there needs reading carefully before it is called a Nav2 failure.
