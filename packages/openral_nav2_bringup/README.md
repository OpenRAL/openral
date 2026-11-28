# openral_nav2_bringup

Upstream **Nav2** as a Reasoner-managed background service (after
[`openral_slam_bringup`](../openral_slam_bringup/)). OpenRAL does not
reimplement Nav2: this package ships only the launch and parameter glue so the
Reasoner can route a `NavigateToPose` goal, plus two small nodes (a scan filter
and an optional `cmd_vel` relay).

```
lidar profile:  /scan ─▶ payload scan filter ─▶ /openral/nav2/scan ─▶ Nav2 (obstacle_layer)
                /openral/world_state_fast ─┘
visual profile: /map ───────────────────────────────────────────────▶ Nav2 (static_layer)
                                                          └──/navigate_to_pose──▶ /cmd_vel
```

## Run

Normally started by the `openral deploy sim` / `deploy run` graph when the robot
declares a lidar (`has_lidar`) **or** vision SLAM (`has_vision_slam`); the deploy
forwards the resolved `slam_backend`. Standalone:

```bash
ros2 launch openral_nav2_bringup nav2.launch.py slam_backend:=lidar    # /scan
ros2 launch openral_nav2_bringup nav2.launch.py slam_backend:=visual   # /map
```

Requires `ros-${ROS_DISTRO}-navigation2` / `nav2_bringup` (in the deploy images)
and a map source from `openral_slam_bringup`: slam_toolbox `/map` for lidar, or
cuVSLAM + nvblox `/map` for visual.

| Launch arg | Default | Meaning |
|---|---|---|
| `slam_backend` | `lidar` | `lidar` or `none` selects `nav2_panda_mobile.yaml`; `visual` selects `nav2_visual.yaml` |
| `params_file` | `""` | Explicit Nav2 params file; empty selects by `slam_backend` |
| `robot_yaml` | `""` | Rewrites the params with `RobotDescription.nav2_param_overrides()` (footprint, radius, inflation, motion model), clamps Nav2's velocity limits to the manifest's `max_base_*` bounds, docks a differential base through the rotation shim (below), and enables the scan filter's self half |
| `payload_scan_filter` | `true` | Run `payload_scan_filter_node` (lidar backend only) |
| `cmd_vel_relay` | `false` | Run `twist_to_action_relay_node.py` (see below) |
| `use_sim_time` / `autostart` / `use_composition` | `true` / `true` / `False` | Passed to Nav2 |

## Costmap profiles

| `slam_backend` | Config | Costmap obstacle source |
|---|---|---|
| `lidar` (default) | `config/nav2_panda_mobile.yaml` | `/openral/nav2/scan` via `obstacle_layer` / `voxel_layer` |
| `visual` | `config/nav2_visual.yaml` | the `/map` `OccupancyGrid` via `static_layer` |

The visual profile lets a lidar-less robot (cuVSLAM + nvblox) navigate. Its
`static_layer` sets `map_subscribe_transient_local: False` because nvblox's
`/map` is RELIABLE + VOLATILE, not latched, and the collision monitor's `/scan`
source is disabled. `nav2_visual.yaml` is **generated** from the lidar file by
`tools/gen_nav2_visual.py` (`just gen-nav2-visual`); re-run it after editing the
base file. 3D-lifted detected objects use the `map` TF frame (published by both
backends), not the `/map` topic.

## Nav2 is base-only

The costmaps' footprint is the manifest's bare chassis (a static polygon that
`nav2_param_overrides()` substitutes at launch). Nothing grows it for a carried
object: the 2-D costmap owns base geometry, the 3-D safety kernel owns the arm
and anything carried. Growing the footprint over a payload would forbid the
approaches the tasks need (every place target is a fixture the payload must
enter) and the costmap, a single horizontal slice, could not tell a tall
obstacle from a low one anyway. The decision is ADR-0099.

MPPI's `CostCritic` runs with `consider_footprint: true`, so it scores the real
chassis outline rather than the centre cell. `inflation_radius` is derived from
the circumscribed radius of the footprint polygon plus 0.05 m (0.48 m for panda),
which keeps the critic's cheap collision gate active.

## The Nav2 and safety-kernel boundary

Two independent world models run side by side. Neither covers for the other.

| | **Nav2** | **C++ safety kernel** |
|---|---|---|
| Checks | the **base footprint**, one 2-D polygon | the **arm's link capsules and attached payload primitives**, in 3-D |
| Against | the 2-D costmap | `/openral/world_voxels`, a 3-D occupancy grid |
| Fed by | `/openral/nav2/scan` (lidar) or `/map` (visual) | `openral_octomap_bridge`, from `octomap_server`'s octree |
| Acts by | refusing a path or trajectory, then `/cmd_vel` | vetoing `/openral/candidate_action` into `/openral/safe_action`, or E-stopping |
| Blind to | anything not in the 2-D grid: overhangs, table tops, the arm's reach | anything outside the local voxel box, and the base's path |

**`/cmd_vel` does not pass through the kernel** (ADR-0040, re-affirmed by
ADR-0099). Nav2 publishes `geometry_msgs/Twist`;
`openral_hal.mobile_base_bridge.MobileBaseBridge` maps it to a `BODY_TWIST`
`Action` inside the HAL node, never touching `/openral/candidate_action`. Base
collision avoidance relies entirely on Nav2's costmap, and speed caps on this
path are Nav2's own `velocity_smoother`. The one thing that does cross is the
E-stop: the HAL node's `_on_cmd_vel` returns early while it is latched, so an
E-stop (the kernel's included) stops the Nav2 path at the HAL.

**Nav2's speed limits never exceed the manifest's.** With `robot_yaml`, the
launch clamps `velocity_smoother.max_velocity` / `min_velocity` and MPPI
`vx_max` / `wz_max` to `safety.max_base_linear_speed_m_s` /
`max_base_angular_speed_rad_s` (`_cap_speeds`; it only lowers a limit). The
smoother clamps each axis separately, so its `(vx, vy)` corner is scaled onto the
kernel's `‖(vx, vy)‖` disk. Every `/cmd_vel` producer passes through the
smoother, so on the relay path Nav2 slows to the bound instead of tripping the
kernel's E-stop. `g1` runs at 0.5 m/s and 1.0 rad/s; `r1pro` at 0.5 m/s (the
base file's `vx`) and 0.3 rad/s.

**A differential base docks on the goal pose.** With `base_kinematics:
differential` (`openamrobot_v2`) the launch wraps MPPI in
`RotationShimController` (`_dock_differential`): MPPI drives to the goal
position with `GoalAngleCritic` off, then the shim turns in place to the goal
heading; the goal checker tightens to 0.03 m / 0.02 rad, and a
`PoseProgressChecker` counts that turn as progress. `GoalAngleCritic` stays on
only inside the 3 cm tolerance, so MPPI still steers the heading if the shim's
in-place turn is blocked by a nearby obstacle. MPPI alone (it can't

strafe, and the goal-angle critic turns it 0.5 m out) stopped 7-13 cm and
2.5 deg off on the AMR restock aisle; with the shim, 2-10 cm and <= 1.3 deg,
the rest being SLAM map error.

**Opt-in supervised relay, for bases without `base_joints`.** A robot that
advertises `body_twist` but declares no `base_joints` (today `g1`, `r1pro`) gets
no `MobileBaseBridge`, so nothing in-tree consumes its `/cmd_vel`.
`nav2.launch.py cmd_vel_relay:=true robot_yaml:=<manifest>` starts
`twist_to_action_relay_node.py`, which republishes each `Twist` as a horizon-1
`BODY_TWIST` chunk on `/openral/candidate_action`. The kernel bounds `‖v‖` and
`‖ω‖` by the manifest's `max_base_*_speed` (veto and E-stop) and forwards on
`/openral/safe_action`. After `cmd_vel_timeout_s` of silence following motion
the relay publishes one zero twist. It refuses a manifest with `base_joints`
(that path is the bridge above) and a manifest missing either `max_base_*_speed`
bound (the kernel skips an unset bound). It adds a speed bound and the E-stop
latch, not path checking: Nav2's costmap is still the base's only collision
avoidance. It is off by default and `deploy_e2e.launch.py` does not enable it.

| Relay parameter | Default |
|---|---|
| `robot_yaml` | `""` (required) |
| `cmd_vel_topic` | `/cmd_vel` |
| `candidate_action_topic` | `/openral/candidate_action` |
| `cmd_vel_timeout_s` | `0.5` (wall clock; must be positive) |

**What this costs.** The kernel checks a carried object in 3-D; Nav2 does not
check it at all. The arm and payload are protected against a 3-D obstacle map
while the chassis is free to drive them into a counter, and the kernel can only
stop that, not steer around it. The payload therefore appears once on each side,
deliberately opposite:

* **Kernel side:** the payload is *robot*, collision-active attached geometry.
  `openral_octomap_bridge` clears its cells out of the voxel grid only for a
  tracker-bound attachment; a geometry-only payload stays mapped and the kernel
  may stop the arm on it (fail closed; see the octomap bridge README).
* **Nav2 side:** `payload_scan_filter_node` drops the payload's returns from the
  scan the costmaps and the collision monitor read, so a carried object never
  becomes an obstacle that travels with the robot. It does **not** join the
  footprint.

Both sides read the same attachment set on `/openral/world_state_fast`, so they
cannot disagree about what is attached.

## `payload_scan_filter_node`

Republishes `/scan` (BEST_EFFORT) on `/openral/nav2/scan` with two kinds of
return removed. Nav2's costmaps and `collision_monitor` read the filtered topic.

* **Payload half.** Drops returns whose endpoints fall inside a carried object's
  collision primitives, taken from the attachment set on
  `/openral/world_state_fast`. Not removing is the safe failure here (Nav2 gets
  more cautious), so this half fails open.
* **Self half.** A 2-D lidar on a real base sees the chassis, mast and arm.
  Unfiltered, those returns mark the costmap and never clear, because a working
  filter removes exactly the beam whose ray would have cleared the cell. A beam
  is dropped when its endpoint, transformed into `base_frame`, lies inside the
  manifest's bare chassis `footprint_polygon`. Removing a real obstacle mistaken
  for the robot is the dangerous direction, so this half removes only what it
  can prove: with no manifest, an unresolvable `base_frame <- scan_frame` TF or a
  degenerate polygon it removes nothing. It never uses the kernel's link boxes
  (conservative over-approximations: right for collision checks, wrong for
  deleting sensor returns), and never the payload-grown hull. Sim hides the
  problem because `synthesize_laser_scan_2d` re-casts past the robot's own body
  tree; there is no real-hardware counterpart.

A self-return that reaches the cost grid once is permanent, so while a self
polygon is configured and its TF has never resolved, the node publishes nothing,
for at most `self_tf_grace_s`; after that it reverts to pass-through and logs an
error. The gate arms once: a TF gap after the first successful resolve fails open.
Writing `range_max` instead of `inf` for a dropped beam was rejected: Nav2 would
ray-clear out to `raytrace_max_range`, erasing real obstacles behind the chassis.

| Parameter | Default | Meaning |
|---|---|---|
| `input_topic` / `output_topic` | `/scan` / `/openral/nav2/scan` | Scan in and out |
| `world_state_topic` | `/openral/world_state_fast` | Attachment source |
| `attached_state_timeout_s` | `0.5` | Age past which the attachment set is ignored |
| `payload_margin_m` | `0.0` | Padding on payload primitives |
| `robot_yaml` | `""` | Enables the self half; without it the node warns once |
| `base_frame` | `""` | Frame of the footprint polygon (from the manifest when empty) |
| `self_margin_m` | `0.0` | Padding on the chassis polygon; leave at 0, every millimetre deletes returns Nav2 would act on |
| `circle_samples` | `12` | Samples per circular primitive |
| `self_tf_grace_s` | `5.0` | Startup gate described above |
| `tf_timeout_ms` | `50` | TF lookup timeout |

Because the lidar sees the chassis only through this filter, a real robot's lidar
should keep its sensor-minimum `range_min_m`; a blunt radial cutoff would delete
real obstacles near the base.

**Proof.** `tests/integration/test_nav2_scan_filter_live.py` sweeps a real
`nav2_costmap_2d`'s published grid and asserts zero cells are marked inside the
chassis and payload silhouette, against `robots/panda_mobile/robot.yaml`, with
`footprint_clearing_enabled: False` so Nav2's own clearing cannot hide a miss;
a control test with the self-filter unconfigured shows the sweep failing.
`tools/_nav2_costmap_silhouette_probe.py` runs the same sweep against a live
`openral deploy sim` graph.

## Costmap height gate

`obstacle_layer` applies `min/max_obstacle_height` in the costmap's own global
frame, and the local (`odom`) and global (`map`) costmaps do not share one: with
the panda mobile base, scan returns sit at +0.30 m in `odom` but -0.40 m in `map`
(slam_toolbox flattens `base_link` to z = 0), so a `0.0` lower bound discarded
every return on the global costmap and `NavfnPlanner` planned against a blank
grid. `/openral/nav2/scan` is planar, so the gate can only admit all or none; the
shipped config sets the global costmap's gate inert (`-10.0` / `10.0`) in both
places Nav2 applies it (`obstacle_layer.scan.*` at buffer time and
`obstacle_layer.*` per point). The local costmap keeps its real limits, since it
is a `VoxelLayer` with a true z column.
`tests/integration/test_nav2_global_costmap_height_live.py` pins it, with a paired
control that fails when the old gate is restored.

## Bring-up robustness

**Bond timeout.** `lifecycle_manager_navigation` holds a bond with each server
(`controller_server`, `planner_server`, `smoother_server`, `behavior_server`,
`collision_monitor`). When one misses its heartbeat for `bond_timeout`, the
manager deactivates the **whole** Nav2 stack, silently: nothing exits non-zero,
the graph stays up and stops navigating, and a validation run is scored as a
policy failure. Nav2's default of 4 s is missed under host load, so
`BOND_TIMEOUT_S = 30.0` in `launch/nav2.launch.py` raises it. It is applied as a
`SetParameter` inside a scoped `GroupAction` around the upstream include, because
upstream `navigation_launch.py` never passes the params file to the manager, so a
`lifecycle_manager_navigation:` block in `config/nav2_*.yaml` is silently ignored.
Confirm it on a live graph, not in the yaml:

```bash
ros2 param get /lifecycle_manager_navigation bond_timeout   # -> 30.0
```

It is a liveness timeout, not a safety check (the E-stop path is
`openral_safety_kernel`). It is raised, not disabled: `0.0` would turn bond
monitoring off. `collision_monitor` must name a `base_frame` the HAL publishes;
a missing one wedged it in a TF-lookup loop and cascaded the same way.

**Aborted bring-up.** The manager brings the servers up once. If one fails to
activate it logs `Failed to bring up all requested nodes. Aborting bringup.` and
stops, with no retry: servers already activated stay active, the rest stay
inactive, and `bt_navigator` rejects every goal as "action server inactive".
The usual cause is DDS discovery on a large graph (Nav2 starts last), which the
deploy Fast DDS profile addresses with `maxInitialPeersRange` 120
([`docs/reference/dds-large-messages.md`](../../docs/reference/dds-large-messages.md),
"Localhost discovery reach"); `wait_for_service_timeout` is 30 s in `config/*.yaml`
so slow but working discovery does not abort it. Nothing restarts an aborted
bring-up. A goal the inactive stack rejects fails with the lifecycle state of the
serving node in the reason (`ROSActionRskill`), so the reasoner reads "the stack
is not up", not "bad goal".

## Benchmark

`benchmark/cost_critic_footprint_bench.cpp` times upstream's
`FootprintCollisionChecker::footprintCostAtPose` against the Jazzy
`libnav2_costmap_2d_core` at the real polygons. It is outside the CMake build on
purpose (the build line is in its header). With `inflation_radius` below the
footprint's circumscribed radius, `CostCritic` makes the full-footprint check
unconditional, which is the worst case; deriving `inflation_radius` from the
polygon (above) restores the cheap gate for the bare chassis. Live
`controller_server` CPU with the flag on stays near 20 % of the 50 ms cycle
([`docs/reference/robocasa-carry-survey.md`](../../docs/reference/robocasa-carry-survey.md)).

## See also

- [`docs/reference/collision-validation-evidence.md`](../../docs/reference/collision-validation-evidence.md): measured outcomes of the collision stack.
- [`openral_slam_bringup`](../openral_slam_bringup/): the `/map` source.
