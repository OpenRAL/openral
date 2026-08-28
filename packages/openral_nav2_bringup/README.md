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

## Dynamic footprint while carrying

Two nodes ride with Nav2 (launch arg `payload_footprint`, default on):

| Node | In | Out |
|---|---|---|
| `openral_nav2_payload_footprint` | `/openral/world_state_fast` + `base_frame ← attach_link` TF | `geometry_msgs/Polygon` on `/local_costmap/footprint` + `/global_costmap/footprint` |
| `openral_nav2_payload_scan_filter` | `/scan` + the same attachment set + the manifest's chassis outline | `sensor_msgs/LaserScan` on `/openral/nav2/scan` |

The published polygon is the convex hull of the manifest's
`footprint_polygon` and the ground projection of every attached collision
primitive, each placed by `TF(base_frame ← attach_link) · pose_in_link ·
pose_in_object` — the kernel's own composition. Boxes project exactly; spheres
and capsules project onto *circumscribed* N-gons, so the polygon contains the
true shape rather than inscribing it.

**Verified against the Jazzy binaries, not from memory.** `nav2_costmap_2d`'s
`Costmap2DROS` subscribes an **unstamped `geometry_msgs/Polygon`** on its own
*relative* `footprint` topic (`/local_costmap/footprint` — not `~/footprint` on
the parent server), calls `setRobotFootprintPolygon` on every message, and
republishes the oriented result as a `PolygonStamped` on `published_footprint`,
padded by `footprint_padding` (default 0.01 m). `nav2_collision_monitor`,
`nav2_behaviors`, `opennav_docking` and `IsPathValid` all read that republished
polygon, so writing the two costmaps reaches every consumer — with one
deliberate exception today, MPPI's own scoring, which needs
`CostCritic.consider_footprint` and is left off pending measurement (below).

Config consequences, both in `config/nav2_panda_mobile.yaml`:

* The costmaps declare a **`footprint` polygon**, not just `robot_radius`. A
  circle cannot express an asymmetric payload. Nav2 reads `""` and `"[]"` as
  "use `robot_radius`", so `RobotDescription.nav2_param_overrides()` emits
  `"[]"` for a robot whose manifest has no `footprint_polygon` rather than
  letting it inherit panda's outline.
* `CostCritic.consider_footprint` is left at **`false`**, deliberately. With
  `false`, MPPI scores the centre cell only and the payload-inclusive polygon
  changes nothing *for the controller* — a forward-reaching payload grows the
  circumscribed radius, but the inflation layer keys its cost cache off the
  *inscribed* one, which the payload does not move. Turning it on is a real
  navigation-behaviour change costing a measured **+8.1 ms** bare / **+9.7 ms**
  carrying per 20 Hz iteration, 17–20 % of the 50 ms budget, while the rest of
  the MPPI loop around CostCritic is still unmeasured. See
  [below](#measured-what-consider_footprint-true-would-cost-and-why-it-is-false).
  The value is pinned by
  `test/test_nav2_launch.py::test_mppi_does_not_yet_consider_the_footprint`.

**Failure directions are opposite on purpose.** The footprint publisher refuses
to narrow: a missing attach-link TF, a primitive the kernel would itself reject,
or a stale world state makes it hold the last polygon, because shrinking the
footprint is the only direction that can drive a payload into something. The
scan filter's two halves both fail toward *more* obstacles: an unplaceable
payload stays in the scan (keeping Nav2 more cautious, and keeping the topic
alive so the costmap never goes blind), and an unplaceable chassis means no
self-return is removed at all.

Nav2 also clears its own footprint (`obstacle_layer`'s
`footprint_clearing_enabled`, default `True`, verified live). That is a second
line, not a substitute: it reaches only cells inside the current polygon, and
the collision monitor reads the scan with no costmap in between.

## The robot's own returns

A 2-D lidar on a real mobile base sees the base: chassis, mast, arm. Unfiltered,
those returns mark the costmap and never clear, and the robot concludes it is
surrounded by itself. **Sim hides this completely** —
`openral_sim.backends.robocasa.synthesize_laser_scan_2d` compares each
`mujoco.mj_ray` hit's `body_rootid` against the base body's and re-casts past
its own tree, so a self-return never enters `/scan` in the first place. That
mechanism is a MuJoCo body-id comparison; it has no real-hardware counterpart,
and this repo has no lidar driver, no lidar launch file, and no `SensorSpec`
field that could express a mount pose, a blind sector or an angle mask. The only
real knob today is `panda_mobile`'s `range_min_m: 0.55`, a blunt radial cutoff
that deletes every real obstacle inside 0.55 m in every direction to hide a
chassis whose circumscribed radius is 0.43 m.

`payload_scan_filter_node` therefore filters the robot too, with a shaped test
instead of a radial one: a beam is dropped when its endpoint, transformed into
`base_frame`, lies inside the manifest's **bare chassis** `footprint_polygon`.

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

## Measured: what `consider_footprint: true` would cost, and why it is `false`

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

### Why the flag is deferred, not forgotten

Knowing the flag's own price is not the same as knowing the loop fits. What
this measurement does **not** settle is the rest of the MPPI loop *around*
CostCritic, and that genuinely needs the live scene below: no RoboCasa atomic
task drives the base at 20 Hz while an object is attached, so the surrounding
cycle time under a grown footprint has never been observed. Committing 17–20 %
of a 50 ms budget against an unmeasured remainder is a navigation-behaviour
change made blind, so the flag stays `false` for now.

The precondition to flip it is explicit: run the composite scene specified
below with the **whole** MPPI loop timed against the 50 ms budget, and show the
full cycle still fits with the flag on. Then `config/nav2_panda_mobile.yaml`,
its CostCritic comment, and
`test/test_nav2_launch.py::test_mppi_does_not_yet_consider_the_footprint` flip
together. If the remainder turns out too tight, `trajectory_point_step` is the
knob that buys the flag room — not the flag itself.

Until then the dynamic footprint is still load-bearing everywhere else: the
published polygon reaches the behaviour server's collision checker,
`opennav_docking`, the collision monitor's approach polygon and `IsPathValid`,
and the payload is out of the scan both costmaps read. MPPI's own scoring is
the single consumer waiting on the measurement.

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

  **Criteria 1 and 4 are met. Criteria 2 and 3 are not — and the same
  measurements show neither is reachable as written:**

  * **Criterion 3 cannot be met by any counter-height carry.** The payload rides
    at z ~ 0.97-1.03 m; `synthesize_laser_scan_2d` casts at 0.30 m. The payload
    never enters the scan plane, so the *payload* half of
    `payload_scan_filter_node` is inert here regardless of scene. The footprint
    publisher is unaffected (it ignores height by design), and the *chassis*
    half of the filter is still live.
  * **Criterion 2 needs the polygon check it is describing.** Free-corridor
    bottlenecks between the carry endpoints measure 0.19-0.24 m — under the bare
    chassis's own 0.444 m circumscribed radius, which would mean the robot fits
    nowhere, including its start pose. It does fit: the chassis is a 0.70 x 0.50 m
    rectangle tucked against a counter that lies inside its circumscribed circle
    and outside the rectangle. So "an aperture the bare chassis clears but the
    grown polygon does not" is only decidable with an oriented-polygon test —
    i.e. with `consider_footprint` on, the very flag it was meant to justify.

  The original four criteria, kept for reference:
  1. a **carry path long enough to require base translation** — source and
     destination on fixtures far enough apart that the arm alone cannot bridge
     them, i.e. beyond the manipulator's reach from one base pose (> ~1.0 m of
     required base displacement for panda_mobile), so `NavigateToPose` actually
     runs while an object is attached;
  2. a **gap the payload's footprint decides** — at least one aperture the bare
     chassis clears but the payload-grown polygon does not, or a turn where the
     0.86 m circumscribed radius sweeps furniture the 0.44 m one misses.
     Without this the dynamic footprint is exercised but never *load-bearing*,
     and the run cannot distinguish a working footprint from a decorative one;
  3. **lidar-visible obstacles at the payload's height**, so the scan filter's
     two halves are both live rather than trivially satisfied;
  4. **determinism** — a fixed `seed` and pinned `init_robot_base_ref`, so the
     acceptance is a pass/fail rather than a distribution.

  `loading_fridge`, the candidate this file used to name, is disqualified on the
  measurement: none of its eight classes is in `target50`, so it would put XR-1
  out of distribution.

  **What remains before #108 closes** is the run itself, not the scene: drive
  `scenes/deploy/robocasa_deliver_straw.yaml` end to end with Nav2 and the
  OctoMap kernel gate enabled, and time the *whole* MPPI loop against the 50 ms
  budget with `consider_footprint` on. That is the measurement the flip was
  deferred pending — it is the first thing that drives the controller at 20 Hz
  with a grown footprint — and so it is also what unblocks
  `CostCritic.consider_footprint`.
