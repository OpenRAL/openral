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
navigation works regardless of how the 2D map is built:

| `slam_backend` | Config | Costmap obstacle source |
|---|---|---|
| `lidar` (default) | `nav2_panda_mobile.yaml` | `/scan` via `obstacle_layer`/`voxel_layer` |
| `visual` | `nav2_visual.yaml` | **`/map`** `OccupancyGrid` via `static_layer` |

The **visual** profile lets a lidar-less robot (cuVSLAM + nvblox) navigate:
costmaps consume `/map` (nvblox's `static_occupancy_grid`, remapped) via
`static_layer` with `map_subscribe_transient_local: False` (nvblox's `/map` is
RELIABLE+VOLATILE, not latched), and the collision_monitor's `/scan` source is
disabled. Everything else mirrors the lidar base — `nav2_visual.yaml` is
**generated** from it by `tools/gen_nav2_visual.py` (re-run after editing the
base).

> 3D-lifted detected objects use the `map` **TF frame** (published by both
> backends), not the `/map` topic, so they map into the world identically on
> both.

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

## The bond timeout, and why it is 30 s not 4 s (issue #256)

`lifecycle_manager_navigation` holds a *bond* with every server it manages
(`controller_server`, `planner_server`, `smoother_server`, `behavior_server`,
`collision_monitor`). When one misses its heartbeat for `bond_timeout`, the
manager declares it dead and **deactivates the entire Nav2 stack**. Nav2's
default is **4 s**, which on a loaded host is short enough to be missed by a
scheduling hiccup rather than by a real fault.

The failure is silent and total, which is what makes it dangerous to measure
against: no traceback, nothing exits non-zero, and the graph stays *up*. It
simply stops navigating. A validation run then burns its whole deadline doing
nothing and gets scored as the policy failing — `deadline-no-grasp`. In the
2026-09-06 ceiling battery this killed **31 of 89 valid runs**, 25 of them
naming `controller_server`, and every one was counted against the policy
(`docs/reference/collision-validation-evidence.md`).

`BOND_TIMEOUT_S = 30.0` in `launch/nav2.launch.py` raises it. Two things about
how, both load-bearing:

- **It cannot go in the params file.** Upstream `navigation_launch.py` passes
  `lifecycle_manager_navigation` only `{autostart, node_names}` and never the
  params file, so a `lifecycle_manager_navigation:` block in
  `config/nav2_*.yaml` is silently ignored. It is applied as a `SetParameter`
  inside a scoped `GroupAction` around the include instead. Confirm it on a
  live graph, not by reading the yaml:

  ```bash
  ros2 param get /lifecycle_manager_navigation bond_timeout   # -> 30.0
  ```

- **It is a liveness timeout, not a safety check.** The E-stop path is
  `openral_safety_kernel` (see the boundary section below) and is untouched by
  this. Raising it trades a slower reaction to a genuinely hung server against
  not mistaking a descheduled one for a dead one. It is **raised, not disabled**
  (`0.0` would turn bond monitoring off entirely), so a server that really dies
  is still caught.

A related trigger was fixed earlier and is documented on `collision_monitor` in
`config/nav2_panda_mobile.yaml`: pointing it at a `base_footprint` the
`panda_mobile` HAL never publishes wedged it in a TF-lookup loop, stopping its
heartbeat and cascading the same way.

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

**`/cmd_vel` does not pass through the kernel** (ADR-0040: base collision
avoidance relies entirely on Nav2's costmap; ADR-0099 re-affirms it, correcting
ADR-0040's stated reason — no in-tree HAL advertised `body_twist`; `panda_mobile`
now does). Nav2 publishes `geometry_msgs/Twist`;
`openral_hal.mobile_base_bridge.MobileBaseBridge` maps it to a `BODY_TWIST`
`Action` via the HAL node's `_send_action_traced` — never reaching
`/openral/candidate_action`, so the kernel never sees, bounds, or vetoes it.
Velocity caps on this path are Nav2's own `velocity_smoother`. Also stated in
the bridge's module docstring.

**The one thing that does cross.** `_on_cmd_vel` returns early while the HAL
node is latched E-stopped, so an E-stop — the kernel's included — does stop the
Nav2 path, at the HAL rather than in the kernel. Stopping is the only kernel
authority over base motion. There is no per-command bound.

**What this costs.** A carried object is checked in 3-D by the kernel and is
*not* checked at all by Nav2 — nothing grows the footprint over it, by decision
(ADR-0099, and "Nav2 is base-only" below). The arm and the payload are protected
against a 3-D obstacle map while the chassis is free to drive them into a
counter, and the kernel's only authority over that is to stop. So the payload
has to appear in **exactly one** place on each side of the boundary, and the two
sides are deliberately opposite:

* **Kernel side** — the payload is *robot*: collision-active attached geometry
  the kernel keeps checking, and `openral_octomap_bridge` clears its cells out
  of the voxel grid so the robot is not stopped by itself.
* **Nav2 side** — the payload is *robot* here too, but only in the sense that it
  is removed: `payload_scan_filter_node` drops its returns from the scan the
  costmaps and the collision monitor read, so a carried object never becomes an
  obstacle that travels with the robot. It does **not** join the footprint.

Both sides read the same attachment set on `/openral/world_state_fast`, so they
cannot disagree about what is attached.

**How that is proved, and where.**
`tests/integration/test_nav2_scan_filter_live.py` sweeps a real
`nav2_costmap_2d`'s published grid and asserts zero cells are marked inside the
chassis ∪ payload silhouette, using this package's own `base_footprint_polygon`
/ `points_in_convex_polygon` / `points_in_primitive` against the real
`robots/panda_mobile/robot.yaml`. The rig runs with `footprint_clearing_enabled:
False` so Nav2's own clearing cannot be what keeps the silhouette clean, and a
control test with the self-filter unconfigured shows the sweep failing (#183).

The same assertion has also been run on the real scenes:
`tools/_nav2_costmap_silhouette_probe.py` attaches to a live `openral deploy sim`
graph and sweeps every published costmap the same way. Raw output in
`docs/reference/data/nav2-costmap-silhouette-2026-09-04.jsonl`; one graph launch
per scene, driven by a direct `NavigateToPose` 2.0 m ahead:

| | `robocasa_baguette` | `robocasa_deliver_straw` |
| --- | ---: | ---: |
| base travelled | 1.682 m | 1.668 m |
| **local** costmap samples | 95 | 107 |
| silhouette cells per sample | 138 | 141 |
| peak `LETHAL` cells elsewhere on the map | 208 | 66 |
| **`LETHAL` cells inside the silhouette** | **0** | **0** |
| **global** costmap samples | 50 | 52 |
| peak cost anywhere on the global map | **0** | **0** |

The **global** row was vacuous here because of issue #211 (below); re-run after
the fix it reads **254** on both scenes, silhouette still clean against 164 and
254 marked cells elsewhere.

A second independent run reproduced the conclusion (0 `LETHAL` cells inside a
140/141-cell silhouette on baguette/deliver_straw) though the incidental counts
differ, as expected for a live scene run.

Two coverage limits on this table: sim's `synthesize_laser_scan_2d` re-casts
through the robot's own MuJoCo kinematic tree, so a self-return never enters
`/scan` in sim — a clean silhouette there is the end state being right, not
proof of what this node does. And `attached_objects` stayed 0 on both runs
(nothing grasped), so the payload half was unmeasured here; both gaps are
covered by the deterministic sweep above, which has a control.

#### The payload half, measured on a scene (2026-09-05)

`robocasa_baguette`, seed 1, `runtime.enable_reasoner: false`, XR-1
`robocasa365` policy dispatched directly at `/openral/execute_rskill`
(`tools/validation_matrix.py`'s direct-dispatch stack).

| run | base travel | payload measured | `LETHAL` inside silhouette | verdict |
| --- | ---: | --- | ---: | --- |
| 13 | 0.523 m | both costmaps | **0** | clean |
| 15 | 0.601 m | both costmaps | **0** | clean |
| 17 | 0.748 m | both costmaps | **0** | clean |

This required a probe fix: `tools/_nav2_costmap_silhouette_probe.py` compared a
count of placed *primitives* against a count of *objects*; sim payloads come
from `extract_body_primitives` over a MuJoCo body subtree (16 primitives on
this baguette), so `placed == declared` could never hold and every run was
filed as a partial placement. It now counts per object (placed only when every
primitive of that object projected).
`tests/unit/test_nav2_costmap_probe_payload_count.py` pins it; reverting the
count fails 2 of its 4 tests.

Run 14 flagged 2 global / 4 local cells at x = 0.39–0.44 m in `base_link`,
4–9 cm past the 0.35 m chassis edge — a probe false positive: both costmaps'
only source is the planar scan, and the payload rides at z = 1.43 m in `odom`
against a 0.30 m scan plane (0.59–0.75 m clearance, per the probe's
`payload_z_span_in_base_m`). These are the counter the robot was parked at,
seen at scan height, beneath a carried object clearing the lidar by >0.5 m. The
probe now reports that span in every verdict.

Not yet measured: a completed task. Every run E-stopped or dropped the payload
before the place phase (grasp held 5–15 s, goal ended `safety_estop`) — this is
the costmap silhouette during a real carry (#108's payload-half ask), not a
task-success claim. `docs/reference/collision-validation-evidence.md` records
2/5 completions on this scene; the E-stops are known open collision-stack work.

### A third defect this surfaced — fixed here (issue #212)

A self-return that reached the cost grid once was permanent. The filter fails
open by design (no `base_frame <- scan_frame` TF yet, unreadable manifest,
degenerate polygon → republish untouched) — correct for the payload half, but
not the self half: the working filter removes exactly the beam whose ray would
have cleared that cell, so a mark let through once can never be retracted.

Measured on a real `nav2_costmap_2d` with this package's topic wiring,
`footprint_clearing_enabled: False`:

| phase | scan on `/openral/nav2/scan` | cells marked inside the chassis |
| --- | --- | ---: |
| 1 | unfiltered ring at 0.20 m (the fail-open window) | **32** |
| 2 | filtered ring — every self beam `inf` — for 20 s | **32** (unchanged) |
| 3 | real returns at 3.0 m on the same bearings | **0** |

Phase 3 is the control: Nav2's clearing works fine, there simply has to be a
ray. `collision_monitor` reads the same topic with no costmap-side clearing at
all.

Fix: make phase 1 impossible rather than undo it. While a self-polygon is
configured and its TF has never resolved, the node publishes nothing, bounded
by `self_tf_grace_s` (default 5 s); after that it reverts to pass-through and
logs an error. The gate arms once — a TF gap after the first successful
resolve still fails open (the one-scan-at-a-time case the payload half already
covers).

Rejected alternative: write `range_max` instead of `inf` for a dropped beam so
Nav2 raytraces the bearing clear. Wrong segment — Nav2 clears the ray out to
`raytrace_max_range` (3.0 m here), an order of magnitude past the chassis, and
the bearing is permanently occluded, so cells the ray erases were marked from
other robot poses and nothing will ever re-mark them. Pinned as
`test_raytrace_clearing_a_dropped_beam_would_erase_a_real_obstacle`: a real
obstacle 0.25 m past the chassis edge, already lethal, is deleted by one such
beam.

Residual: a mark that does land inside the chassis is freed by Nav2's own
`footprint_clearing_enabled` (default `True`, not overridden in
`config/nav2_panda_mobile.yaml`) — it frees cells whose *centre* falls inside
the published polygon, so a cell straddling the boundary is not reached; that
sub-cell-wide band is covered only by the bounded grace fallback above.

`tests/integration/test_nav2_scan_filter_live.py` still gates its costmap on
the filter's own output: the gate makes the node's startup safe, the sweep
asserts the steady state.

### A second defect this surfaced — fixed here (issue #211)

The global costmap was empty: max cost `0` over 50/52 published samples across
both scenes, no non-zero cell anywhere, while the local costmap on the same
graph peaked at `254` with ~2000 non-zero cells. Both read the same filtered
`/openral/nav2/scan`; `static_layer` is deliberately out of the global chain
(rolling window, SLAM-from-scratch); `planner_server` logged no warning, so
`NavfnPlanner` planned against a blank 20 x 20 m grid.

Root cause: the height filter, not the topic or TF (97 of 102 scans
transformed into `map` at their own stamps). `ObservationBuffer` applies
`min/max_obstacle_height` in the costmap's *own* global frame, and the two
costmaps don't share one:

| TF | z |
| --- | ---: |
| `odom → base_link` | +0.700 m |
| `odom → base_scan` (local costmap frame) | **+0.300 m** |
| `map → odom` (slam_toolbox flattens `base_link` to z=0 in `map`) | −0.700 m |
| `map → base_scan` (global costmap frame) | **−0.400 m** |

Both costmaps carry `min_obstacle_height: 0.0`. In `odom` returns sit at
+0.30 m and are kept; in `map` they sit at −0.40 m and every one is discarded.
Same scan, same parameters, opposite outcome. The probe now reports `VACUOUS -
the costmap marked nothing anywhere` for this state and exits non-zero rather
than printing `clean`.

Fix: turn the height gate off on the global costmap rather than retune it.
`/openral/nav2/scan` is planar and `obstacle_layer` is a 2-D layer, so every
point shares one z — the gate can only admit all or none. Set to
`min_obstacle_height: -10.0` / `max_obstacle_height: 10.0`, past any offset
this stack can produce (largest: the 0.700 m pedestal) — inert rather than
tuned, so it cannot break again if a frame's floor moves.

Nav2 applies the cut twice, from two differently-scoped parameters of the same
name — both had to move:

| where | parameter | upstream default |
| --- | --- | ---: |
| `ObservationBuffer::bufferCloud` — at buffer time, after the transform | `obstacle_layer.scan.min/max_obstacle_height` | `0.0` / **`0.0`** |
| `ObstacleLayer::updateBounds` — again, per point, while marking | `obstacle_layer.min/max_obstacle_height` | `0.0` / `2.0` |

Rejected alternatives: stopping slam_toolbox injecting z into `map → odom`
would rewrite a frame contract the octomap bridge and the kernel's voxel grid
(ADR-0095) also read; moving `base_link` to ground level contradicts ADR-0095
(the arm mount, deliberately); copying this window to the local costmap would
be wrong — that layer is a `VoxelLayer` with a real z column (`origin_z` 0.0,
16 x 0.08 m) where height bounds carry meaning.

Evidence: `tests/integration/test_nav2_global_costmap_height_live.py` — a real
`nav2_costmap_2d` reading the shipped config, under the measured
`map → odom → base_link → base_scan` chain, marks a 1 m obstacle at cost 254;
the paired control with the pre-fix gate restored marks nothing, and fails
with "the global costmap marked nothing anywhere".

On the scenes, same method/host, only the four height parameters changed:

| global costmap | `robocasa_baguette` | `robocasa_deliver_straw` |
| --- | ---: | ---: |
| `max_cost_seen` | 0 → **254** | 0 → **254** |
| `nonzero_cells_max` | 0 → **2818** | 0 → **4239** |
| `lethal_cells_anywhere_max` | 0 → **164** | 0 → **254** |
| chassis cells per sample | 138 | 136 |
| **`LETHAL` cells inside the silhouette** | **0** | **0** |
| probe verdict | `VACUOUS` → **`clean`** | `VACUOUS` → **`clean`** |

The global half of the silhouette table above was clean only because nothing
was marked anywhere; it now reports `non_vacuous: true` and stays clean against
164/254 marked cells elsewhere in the same samples. Raw output appended to
`docs/reference/data/nav2-costmap-silhouette-2026-09-04.jsonl`.

The payload half is still unmeasured on scenes — `attached_objects` stayed `0`
on every run (nothing grasped); that is #108's own gate, untouched by #211.

## Nav2 is base-only

The costmaps' footprint is the manifest's bare chassis, and nothing grows it:
the 2-D costmap owns base geometry, the 3-D safety kernel owns the arm and
anything carried. This replaces the dynamic footprint publisher PR #143
shipped, now removed.

Layer-boundary decision, recorded as **ADR-0099** in the private
`OpenRAL/management` log (hazard analysis: that repo's `safety/hazard-log.md`
Entry 023). The ADR was written after the code (PR #186); the six days in
which this boundary existed only in this README are part of what it records.

### Why the growth was wrong

It projected 3-D geometry onto a 2-D costmap — the two don't describe the same
world.

It forbade the poses the tasks require: every RoboCasa place target is a
fixture the payload must enter (cabinet, sink, fridge). Growing the footprint
over the payload lands its ground projection on the fixture the base must
approach, so Nav2 refuses the one approach that succeeds.

And it protected against nothing here. Measured on
`scenes/deploy/robocasa_deliver_straw.yaml`:

| | measured |
| --- | ---: |
| `odom → base_link` (TF) | **0.700 m** |
| local costmap voxel column (`origin_z` + `z_resolution × z_voxels`) | 0.00 – 1.28 m |
| where scan returns land | **≈ 0.70 m** |
| carried object (`glass_cup`) | **0.981 m** |

The costmap is one horizontal slice; a carried object rides ~0.28 m above it.
A payload can't collide with an obstacle the costmap knows about unless that
obstacle is also tall, which the costmap can't represent — so the growth
traded a real, frequent false block for protection against a case it couldn't
distinguish anyway.

What's genuinely given up: a payload sticking forward could clip a tall, thin
obstacle the base itself clears. The kernel catches that in 3-D — its octomap
bridge covers a ball of r = 1.05 m centred at z = 0.5 in `base_frame`
(z ∈ [−0.55, 1.55], which contains the payload) — but as an E-stop, not an
avoidance, since `/cmd_vel` never passes through it (ADR-0040/ADR-0099, above).

### What replaces it

Nothing new: `RobotDescription.nav2_param_overrides()` already substitutes
`footprint_polygon` into both costmaps' `footprint` at launch, so the polygon
is static and correct without a publisher. The scan filter (next section) and
`CostCritic.consider_footprint` (now `true`) remain — with a fixed chassis
polygon, scoring the real outline instead of the centre cell is strictly more
accurate, measured at +0.53 ms on the live loop.

### A defect this surfaced — since fixed

`openral_sim.backends.robocasa.synthesize_laser_scan_2d` cast rays at world
z = 0.30 m (`origin[2] = laser_height_m`, absolute; base body at z = 0.000) but
published the result in `base_link`, which TF puts at 0.700 m — the sim
sampled the world at one height and told Nav2 the returns came from another,
0.40 m higher.

Fixed by giving both the same number: `robots/panda_mobile/robot.yaml` gives
the lidar its own `base_scan` frame with the mount in
`static_transform_xyz_rpy` (−0.40 m from `base_link`); `deploy_e2e.launch.py`
publishes that as the `base_link → base_scan` static TF;
`SimSensorBridge._scan_world_height_m` adds the same offset to the base's own
world z when casting — so the ray and the frame can no longer disagree,
including when the base changes height (a ramp, a lift column).

## The robot's own returns

A 2-D lidar on a real mobile base sees the base itself: chassis, mast, arm.
Unfiltered, those returns mark the costmap and never clear. Sim hides this
completely — `openral_sim.backends.robocasa.synthesize_laser_scan_2d` compares
each `mujoco.mj_ray` hit's `body_rootid` against the base body's and re-casts
past its own tree, so a self-return never enters `/scan` there. That mechanism
has no real-hardware counterpart, and this repo has no lidar driver, launch
file, or `SensorSpec` field for a mount pose, blind sector, or angle mask.
Until #194 the only real knob was `panda_mobile`'s `range_min_m: 0.55`, a
blunt radial cutoff deleting every real obstacle inside 0.55 m to hide a
chassis whose circumscribed radius is 0.43 m. #194 lowered it to the sensor
minimum (0.05 m) on the strength of this node.

`payload_scan_filter_node` filters the robot too, with a shaped test instead
of a radial one: a beam is dropped when its endpoint, transformed into
`base_frame`, lies inside the manifest's bare chassis `footprint_polygon`.

Measured on the live graph (`robocasa_deliver_straw`, pinned seed 3, whole
graph relaunched per arm; raw output in
`docs/reference/data/base-scan-range-min-2026-09-02.jsonl`):

| | `/scan` usable | inside 0.55 m | reaching `/openral/nav2/scan` | nearest |
|---|---|---|---|---|
| `range_min_m: 0.55` | 192 | **0** | 192 | 0.555 m |
| `range_min_m: 0.05` | 344 | 152 | **252** (92 dropped as chassis) | 0.344 m |

60 real near-field returns now reach Nav2 where none could before, and the
shaped filter still removes the 92 whose endpoints it can prove are the robot.
All 152 near returns resolve to real kitchen geometry by body name (cabinet
doors, fridge housing, freezer), none to the robot (`robot0_link0`,
`robot0_link7`, `mobilebase0_wheeled_base` share the base body's
`body_rootid`). MPPI loop unmoved: 10.03 → 10.25 ms/cycle against the 50 ms
budget, 600 → 601 cycles in 30 s, none dropped.

The conservative direction is opposite the payload's: for the payload, not
removing is the safe failure (Nav2 gets more cautious); for a self-return,
removing a real obstacle mistaken for the robot is the dangerous one, so the
self half removes only what it can prove and, on a missing manifest, an
unresolvable `base_frame ← scan_frame` TF, or a degenerate polygon, removes
nothing.

Why the chassis polygon is a proof:

* A return inside the chassis outline is the chassis, or an object standing
  where the chassis already is — not a place an object can be.
* It's the same polygon this package publishes to Nav2 as the robot:
  `footprint_clearing_enabled` already frees those cells every update and
  `collision_monitor` reads the same outline — removing these returns takes
  away nothing Nav2 could have acted on, but does remove the collision-monitor
  false positive (that node has no costmap clearing in between).
* It's the bare chassis, never the payload-grown hull: the hull spans free air
  between chassis and payload, and the payload's own primitives already cover
  the payload exactly.
* The kernel's per-link OBBs in `link_collision` are deliberately conservative
  over-approximations — right for a collision check, wrong for deleting
  sensor returns (the air between a link and its box is air a real obstacle
  can occupy) — so this node does not use them; the arm above the scan plane
  is out of scope for it.

`self_margin_m` defaults to `0.0` and should stay there — every millimetre
past the chassis deletes returns Nav2 would have acted on. Without
`robot_yaml` the self half does not run and the node warns once.

## Measured: what `consider_footprint: true` costs, and why it is `true`

`benchmark/cost_critic_footprint_bench.cpp` times upstream's
`FootprintCollisionChecker::footprintCostAtPose` against the real Jazzy
`libnav2_costmap_2d_core`, at the real polygons and the shipped 3 m / 0.05 m
local costmap. It is out of the CMake build on purpose (a measurement, not an
artifact); the build line is in its header comment.

On an i5-8600K, at `batch_size 2000 × time_steps 56 / trajectory_point_step 2`
= 56 000 calls per controller iteration (four runs; ms/iteration spread in
brackets):

| | circumscribed radius | ns/call | ms/iteration | Δ vs `false` | of the 50 ms cycle |
|---|---|---|---|---|---|
| base only | 0.444 m | 149 | 8.3 [8.28–8.32] | **+8.1** | **17 %** |
| carrying (0.860 m reach) | 0.863 m | 178 | 9.9 [9.87–9.94] | **+9.7** | **20 %** |

The point-only path (`consider_footprint: false`) is 0.15 ms — the flag is
essentially the whole cost, because `CostCritic::findCircumscribedCost`
returns `0.0` whenever `inflation_radius` is below the footprint's
circumscribed radius, and `inCollision`'s guard (`cost >=
possible_collision_cost_ || possible_collision_cost_ < 1.0f`) then makes the
full-footprint check unconditional. Both polygons are in that regime against
the shipped `inflation_radius: 0.40`, so the worst case is the normal case.
(This also corrects an earlier comment: the ~0.364 m circumscribed radius came
from `robot_radius`, but the costmaps use the *polygon*, whose padded farthest
vertex is 0.444 m — independent of the flag.)

Raising `inflation_radius` above 0.444 m would restore the cheap gate for the
bare chassis; nothing restores it while carrying (the payload's circumscribed
radius is most of the 3 m local costmap), so it is left at 0.40 m.

### MEASURED 2026-08-28: the full cycle fits with the flag on

`scenes/deploy/robocasa_deliver_straw.yaml` driven with the full stack (SLAM +
Nav2 + octomap + kernel gate) on `q-laptop`; `controller_server`'s own CPU per
published control cycle, all four arms, 2 runs each:

| footprint | `consider_footprint` | CPU / cycle | of 50 ms |
| --- | --- | ---: | ---: |
| bare (0.72 m) | `false` *(shipped then)* | 9.58 ms | 19 % |
| bare (0.72 m) | `true` | 10.11 ms | 20 % |
| grown (1.23 m) | `false` | 9.79 ms | 20 % |
| **grown (1.23 m)** | **`true`** | **9.97 ms** | **20 %** |

The loop fits in every arm, ~40 ms of headroom: 500-501 cycles per 25 s window
(20 Hz, none dropped), one `Control loop missed its desired rate` warning in
the whole session. Nav2 logged `inflation radius (0.400000) is smaller than
the circumscribed radius (0.908020)` for the grown polygon, confirming the
unconditional-check (worst-case) regime.

The measured delta was +0.53 ms (bare) / +0.18 ms (grown), not the +8.1 / +9.7
ms the isolated benchmark predicted — likely `CostCritic::inCollision` is
reached less often per iteration than the benchmark's assumed 56 000 calls
(`CostCritic::score`'s source is not in the Jazzy binary install). Reported as
a discrepancy, not resolved.

Caveats: the payload was injected by the probe (no policy ran,
`attached_objects count=0`), so this is the controller carrying a grown
footprint, not a policy-driven carry; and it is one host, one route, one
kitchen. Method, full numbers and the validity check:
[`docs/reference/robocasa-carry-survey.md`](../../docs/reference/robocasa-carry-survey.md);
raw output in `docs/reference/data/nav2-mppi-loop-2026-08-28.jsonl`.

### The flag is now `true` — decided 2026-08-29

Flipped together:

* `config/nav2_panda_mobile.yaml` → `consider_footprint: true` (and
  `config/nav2_visual.yaml`, regenerated by `tools/gen_nav2_visual.py`).
* Its CostCritic comment, rewritten around the live numbers.
* `test/test_nav2_launch.py::test_mppi_considers_the_full_footprint` — renamed
  from `test_mppi_does_not_yet_consider_the_footprint`, still pinning the
  value.

The polygon it scores is the bare chassis, for a 0.70 × 0.50 m rectangle that
routinely parks with less clearance than its own 0.444 m circumscribed radius.

`inflation_radius` is deliberately left at 0.40 m: raising it to ≥ 0.444 m
would restore CostCritic's cheap gate (now well-defined with no payload
growth), but it is a separate navigation-behaviour decision, not made here.

### What is still open (issue #108)

* A scene now drives the base while carrying — `DeliverStraw`, pinned at seed 3
  by [`scenes/deploy/robocasa_deliver_straw.yaml`](../../scenes/deploy/robocasa_deliver_straw.yaml),
  upstream RoboCasa, in `composite_seen` (inside the `target50` set XR-1
  RoboCasa365 reports against). Measured at reset: the straw starts 0.50 m
  away in the drawer the base is parked at (inside the Panda's 0.855 m reach,
  grasped before any base motion); the glass cup it must end up inside sits
  3.795 m away. `GetToastedBread` also qualifies, at up to 3.48 m. (Measured,
  not read off `Kitchen.get_fixture`'s docstring, which describes a tie-break
  within 0.10 m of the nearest candidate, not a bound — classifying statically
  gives the wrong answer. Full measurements:
  [`docs/reference/robocasa-carry-survey.md`](../../docs/reference/robocasa-carry-survey.md).)

  Criteria 1 and 4 are met; criteria 2 and 3 are obsolete — both existed only
  to exercise a payload-grown footprint, and Nav2 is now base-only:
  * Criterion 2 ("an aperture the bare chassis clears but the payload-grown
    polygon does not") has nothing to decide: there is no grown polygon. The
    underlying property — a rectangle fits where its circumscribed circle
    doesn't (measured free-corridor bottlenecks 0.19–0.24 m against 0.444 m) —
    is exactly what `consider_footprint: true` now reads.
  * Criterion 3 ("lidar-visible obstacles at the payload's height") is moot:
    the payload rides at ~0.98 m, scan returns land at ~0.70 m, so it never
    enters the slice — the desired state under base-only, not a gap.

  `loading_fridge` is disqualified: none of its eight classes is in
  `target50`, which would put XR-1 out of distribution.

  Remaining before #108 closes: the loop measurement is done and the flag is
  flipped; the payload half has been measured on a scene (`robocasa_baguette`
  under a real XR-1 grasp with the base driving, three clean runs — see "The
  payload half, measured on a scene" above). Still missing: a completed
  episode — every run so far E-stopped or dropped the payload before the
  place phase.
