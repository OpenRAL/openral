# ADR-0025: Reasoner-managed background services (SLAM, perception trees, …)

- Status: **Accepted** — core (slam_toolbox peer + dashboard occupancy card + MoveIt integration test + panda_mobile HAL stub + robocasa base velocity / LaserScan obs-dict surface + reasoner.tick dashboard family + grasping rSkill manifest + sample sim config) is live on the same branch. The mobile-base ROS lifecycle node + Nav2/slam_toolbox peers landed in the 2026-05-27 amendment (below); a full headless Nav2/SLAM sim test remains deferred.
- Date: 2026-05-25 (proposed) / 2026-05-26 (accepted)
- Related: [ADR-0024](0024-ros-wrapped-rskills.md) (the rSkill kind
  discriminator this ADR explicitly does NOT extend); [ADR-0018](0018-ros2-reasoner-supervisor.md)
  §4 (`LifecycleTransitionTool`, the primitive this ADR formalises);
  [ADR-0020](0020-cpp-safety-kernel.md) (the safety_kernel pattern this
  ADR generalises); CLAUDE.md §3 (architecture discipline — eight
  layers, dual-system pattern).

## Context

ADR-0024 added `kind: ros_action` / `ros_service` so the Reasoner can
invoke an upstream ROS 2 action server (MoveIt's MoveGroup, Nav2's
NavigateToPose) as an rSkill, dispatched via `ExecuteRskillTool`. That
covers **goal-bounded** wrapped servers: send a goal, wait for the
result, close the goal with `success=True`.

It does not cover **continuously-running background services** —
`slam_toolbox` (publishes `/map` while alive), RTAB-Map (3D mapping),
perception pipelines (object detection, scene-change), behaviour-tree
controllers. These need to:

1. Start once, run for a long time (minutes to hours).
2. Be paused / resumed by the Reasoner as the active task changes
   (e.g. start SLAM on entry to a new floor, pause it during dense
   manipulation, resume it on exit).
3. Run **concurrently** with foreground rSkills, not block them.
4. Produce a live data stream that downstream layers (Nav2,
   `world_state`, the dashboard) consume.

Trying to dress these up as rSkills fights the abstraction:

- `RskillRunnerNode._active_skill` is hard one-at-a-time
  (`packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py:147`).
  A "background" rSkill would either need to immediately return and
  spawn something external (an anti-pattern — the rSkill is no longer
  the unit of actuation), or require concurrent rSkill execution (a
  significant abstraction change with safety implications, since the
  supervisor's per-row check is keyed to one active skill).
- rSkill manifests carry shape that doesn't apply (`actuators_required`,
  `chunk_size`, `latency_budget.per_chunk_ms`) and gate license posture
  on the upstream weights URI (`weights_uri`) — a wrapped
  long-lived service has none of these.
- The reasoner-LLM tool palette per ADR-0022 surfaces one
  `execute_rskill__<slug>` tool per skill. A background service is not
  what the LLM picks to "do" something; it's infrastructure the LLM
  ensures is running while it does other things.

The right primitive already exists. ADR-0018 §4 defines four
`ReasonerToolCall` variants; one of them is `LifecycleTransitionTool`
(`python/core/src/openral_core/schemas.py:3965`), which lets the
Reasoner drive `configure / activate / deactivate / cleanup` on any
ROS 2 lifecycle peer by node name. The C++ safety_kernel
(`cpp/openral_safety_kernel/src/safety_kernel_node.cpp`) is already
managed this way — spawned by `sim_e2e.launch.py:159` as a
`LifecycleNode`, auto-driven `UNCONFIGURED → INACTIVE` by the launcher's
event handler, and (in a future PR) `INACTIVE → ACTIVE` by the
Reasoner.

## Decision

Long-lived continuous ROS 2 services that the Reasoner needs to start
/ stop / pause are **`LifecycleNode` peers managed by
`LifecycleTransitionTool`** — never rSkills.

The contract has four parts:

### 1. Each service ships its own bringup package

Per-service ament-python package, e.g. `packages/openral_slam_bringup/`,
containing:
- A `launch/<service>.launch.py` that declares the upstream node as a
  `LifecycleNode` under the `/openral/<service>` namespace, with
  parameters loaded from a deployment-scoped YAML.
- A `config/<service>.yaml` carrying sensible defaults (topic remaps,
  frame names, rate caps).
- A `package.xml` declaring an apt dependency on the upstream ROS 2
  package (`ros-${ROS_DISTRO}-slam-toolbox`, `ros-${ROS_DISTRO}-rtabmap`,
  …). The bringup package itself is Apache-2.0; the upstream
  binaries' licenses are surfaced by the host distribution.

This isolation keeps the upstream dependency optional — deployments
that don't need SLAM never pay for `slam_toolbox` being installed.

### 2. The deploy launch opts in via a flag

`packages/openral_rskill_ros/launch/sim_e2e.launch.py` declares one
`LaunchConfiguration` per managed service (e.g. `enable_slam`,
default `false`). When enabled, the launch includes the service's
bringup `launch.py` and registers an `_autostart_lifecycle` event
handler so the node auto-transitions `UNCONFIGURED → INACTIVE` at
launch (mirroring how `safety_kernel` is brought up today at
`sim_e2e.launch.py:215`).

The `INACTIVE → ACTIVE` transition stays under the Reasoner's
control — the launcher must not auto-activate background services
because that would defeat the "Reasoner decides when SLAM runs" goal.

### 3. CLI surfaces the flag + registers the node in the palette

`openral deploy sim` grows a per-service `--enable-<service>` flag (e.g.
`--enable-slam`). When set, the CLI:

- Forwards `enable_<service>:=true` to the launch.
- Adds `/openral/<service>` to the Reasoner's `node_ids` set so
  `LifecycleTransitionTool` client-side validation accepts it
  (`python/reasoner/src/openral_reasoner/palette.py:148`).

Deployments that compose their own runtime out of tree do the same
two steps by hand.

### 4. Observability bridge: rclpy → OTLP

Background services often publish topics that operators want to see
on the dashboard (`/map` for slam_toolbox, `/global_costmap` for Nav2,
…). The dashboard is OTLP-only — it never subscribes to ROS topics
directly. The bridge pattern: a small in-process node-helper composed
into the existing `RskillRunnerNode` (via
`packages/openral_rskill_ros/openral_rskill_ros/compose.py`) subscribes
to the topic, throttles to a reasonable rate (1 Hz for maps), and
emits a domain-specific OTLP span family carrying the payload as
attributes.

For slam_toolbox: `SlamMapBridge(Node)` (new, in `openral_runner`)
subscribes `/map` and emits a `slam.occupancy_grid` span with
`width`, `height`, `resolution`, `origin_x`, `origin_y`, `frame_id`,
`png_b64` (the occupancy grid rasterised to PNG and base64-inlined,
mirroring the camera-card pattern at
`python/observability/src/openral_observability/dashboard/store.py:556`).
The dashboard adds a per-service card mirroring `.camera`.

This pattern generalises: any future service adds (i) a bridge class,
(ii) a span family, (iii) a dashboard card.

## What this ADR does NOT change

- The four `ReasonerToolCall` variants (ADR-0018 §4) — palette stays
  closed.
- The `RSkillKind` discriminator (ADR-0024) — no new values; `wam`
  stays reserved.
- The safety pipeline. Background services do NOT command actuators
  through `/openral/candidate_action`; they publish their own
  output topics that downstream skills (Nav2 reading `/map`) consume.
  Actuation is still gated by the supervisor on the rSkill side.
- The reasoner LLM tool palette schema. Background services appear
  only as `node_ids` (already a `ToolPalette` field today); the LLM
  doesn't see them as `execute_rskill__` tools because they aren't
  rSkills.

## Out-of-scope (deferred follow-ups)

- **Concurrent rSkill execution.** A real "background rSkill" that
  ran alongside a foreground one would need a per-track
  `RskillRunnerNode` or an in-process concurrency primitive in the
  runner. ADR-0025 explicitly sidesteps this by saying background
  things aren't rSkills.
- **Auto-restart on crash.** A service that crashes today stays down
  until the Reasoner notices via a `LifecycleTransition` failure or
  a topic stops publishing. A "supervise + restart" wrapper is its
  own concern.
- **Lifecycle transitions other than the four `LifecycleTransitionTool`
  states.** `error_processing` recovery is left to the upstream
  service's own conventions.
- **Tracking issue: `safety: supervisor consumption of SLAM map`** —
  bringing the live occupancy grid into the supervisor's check is the
  follow-up that this PR's collision-check discussion (ADR-0024
  §Out-of-scope item 1) is blocked on.
- **Mobile-base HAL + LaserScan source.** SLAM is useless without a
  scan source; the in-tree mobile-base wiring is the parallel piece
  landing on this branch. Documented here for context; covered in
  this PR's piece C (the `panda_mobile` HAL stub).

## Concrete first instance: `slam_toolbox`

Lands in this PR as the canonical example:

- New `packages/openral_slam_bringup/` (piece A of the plan).
- `sim_e2e.launch.py` opt-in `enable_slam` flag.
- `openral deploy sim --enable-slam` flag + `node_ids` registration.
- `SlamMapBridge` + dashboard `.slam-map` card (piece B).
- `panda_mobile` HAL stub + synthetic LaserScan source (piece C) so
  there is a real laser to map from end-to-end.

End-to-end demo: `openral deploy sim --config
scenes/deploy/robocasa_pnp.yaml --enable-slam`.
slam_toolbox boots in `INACTIVE`; the Reasoner emits
`LifecycleTransitionTool(node="/openral/slam_toolbox",
transition="configure")` then `"activate"`; `/map` starts publishing;
the dashboard card updates; `OpenRAL/rskill-nav2-navigate-to-pose`
becomes useful.

## Verification

- Schema: no schema changes — `LifecycleTransitionTool` and
  `ToolPalette.node_ids` already carry the contract.
- Unit (piece A): launch-file unit test asserts `enable_slam:=true`
  spawns the LifecycleNode under the expected node name with the
  expected parameters.
- Unit (piece B): dashboard store handler test asserts that emitting
  a `slam.occupancy_grid` span populates the `slam` topic slot with
  the expected fields.
- Integration (piece A + B + C): `ros2 launch openral_rskill_ros
  sim_e2e.launch.py enable_slam:=true` boots; with a synthetic
  `/scan` (or the piece-C LaserScan), `/map` appears; the dashboard
  shows the live card.
- Test gating: hosts without `ros-${ROS_DISTRO}-slam-toolbox`
  apt-installed `pytest.skip(reason=...)` the integration test
  per CLAUDE.md §1.11.

## Amendment 2026-05-27: Nav2 as the second peer

slam_toolbox alone is unusable for autonomous behaviour — `/map` is
read-only. The natural pair, `nav2_bringup`, ships in this amendment
under the same ADR-0025 contract with **two structural deviations**
from the slam_toolbox template, both intentional:

1. **Always-on lifecycle.** Nav2's in-stack
   `lifecycle_manager_navigation` brings the planner / controller /
   behavior / smoother / velocity_smoother sub-nodes to ACTIVE
   automatically. We don't expose a `LifecycleTransitionTool` for
   each sub-node — the Reasoner triggers Nav2 by dispatching the
   `OpenRAL/rskill-nav2-navigate-to-pose` wrapped-action rSkill,
   which sends a `NavigateToPose` goal to `/navigate_to_pose`. That
   is the right primitive: Nav2's lifecycle is a planner-startup
   concern, not a per-tick decision.

2. **`/cmd_vel` is out-of-scope for the safety supervisor.** Nav2's
   behaviour tree publishes `geometry_msgs/Twist` on `/cmd_vel`,
   bypassing OpenRAL's per-row `/openral/candidate_action` →
   `/openral/safe_action` envelope check. The panda_mobile HAL
   accepts `/cmd_vel` directly (translated to a 6-vec BODY_TWIST
   Action, bypassing the supervisor) — velocity caps are enforced
   by Nav2's own `velocity_smoother`, not by us. Documented at
   the package boundary; ADR-0024 §"Out-of-scope" pre-records this
   trade-off for wrapped-action rSkills. Hosts that need a
   supervised cmd_vel path run an external `twist_to_action` relay
   that publishes on `/openral/candidate_action` instead.

### Concrete second instance: `openral_nav2_bringup`

- New `packages/openral_nav2_bringup/` — wraps upstream
  `nav2_bringup/navigation_launch.py` via `IncludeLaunchDescription`,
  parameterised by a panda_mobile-tuned
  `config/nav2_panda_mobile.yaml` (MPPI `motion_model: "Omni"` for
  the holonomic base, `robot_radius: 0.35` for the chassis envelope,
  symmetric `vy_min: -0.5`).
- `sim_e2e.launch.py` opt-in `enable_nav2` flag — defaults to
  `enable_slam` (lidar-equipped robots auto-bring both).
- `openral deploy sim --enable-nav2 / --no-enable-nav2` flag; auto-on
  tracks `--enable-slam`.
- `packages/openral_hal_panda_mobile` subscribes `/cmd_vel`
  (`cmd_vel_topic` ROS param; empty string disables) and routes
  each Twist into a `Action(control_mode=BODY_TWIST)` applied
  directly via the HAL's integrator. Empty topic name disables the
  bridge.

End-to-end demo: `openral deploy sim --config
scenes/deploy/robocasa_pnp.yaml`. slam_toolbox +
Nav2 + the cmd_vel bridge all boot; the Reasoner dispatches
`OpenRAL/rskill-nav2-navigate-to-pose` against a `/navigate_to_pose`
goal; Nav2 plans a path against the slam_toolbox map and drives
the base.

### Future work surfaced by this amendment

The 16-D `human300_16d` / `rc365` state contracts in the
in-tree pi05 / rldx rSkills target a wrapped task-space observation
(`[eef_pos(3), eef_quat(4), base_pos(3), base_quat(4),
gripper_qpos(2)]`) that the in-tree deploy_sim path doesn't
synthesise (the HAL exposes raw JointState). The reasoner palette
filter now emits these drops at INFO (not WARN) for known wrapped-
task-space layouts — see
`packages/openral_reasoner_ros/openral_reasoner_ros/reasoner_node.py`.
The architectural fix is a state-contract adapter registry:
robot manifests declare which wrapped layouts they can produce
(via a HAL-side FK + gripper-state reader); the reasoner admits
adapter-bridged rSkills. That work is deferred — for the navigate-
kitchen use case Nav2 alone suffices.

## Amendment 2026-05-29: robot pose on the `slam.occupancy_grid` span

The `slam.occupancy_grid` span (§4) now also carries the robot's
map-frame pose — `robot_x` / `robot_y` / `robot_yaw`, plus `base_frame`
and `footprint_radius_m` — obtained via a tf2 `map→base_frame` lookup
performed by `SlamMapBridge` on each `/map` emit (`base_frame` and
`footprint_radius_m` are wired from `robot.yaml` through `compose.py`).
The dashboard's SLAM Map card consumes these attributes to draw the
robot over the occupancy map as a footprint circle plus a heading
wedge. The lookup degrades gracefully: when TF is unavailable the span
omits the pose attributes and the card simply renders no robot marker.

## Amendment 2026-05-30: footprint polygon on the `slam.occupancy_grid` span

The `slam.occupancy_grid` span (§4) now also carries the robot's base
footprint polygon as `openral.slam.footprint_polygon_xy` — a flat
base-frame XY float array `[x0,y0,x1,y1,…]` sourced from
`RobotDescription.footprint_polygon` and wired through `compose.py` into
`SlamMapBridge`. When a pose lookup succeeds and a polygon is declared,
the dashboard's SLAM Map card draws it as an oriented polygon plus a
heading marker at the robot's map-frame pose, falling back to the
`footprint_radius` circle when no polygon is declared.

## Amendment 2026-05-31: synthetic `/scan` heading uses the base body's world orientation

The `panda_mobile` synthetic `/scan`
(`openral_sim.backends.robocasa.synthesize_laser_scan_2d`) cast its beam
fan from the base body's **world** position (`data.xpos`) but took the
fan **heading** from the yaw *joint* (`qpos[mobile_yaw]`). Under a
composed RoboCasa kitchen the robot is placed by rotating the
`mobilebase0` body to its spawn facing while the `mobile_yaw` joint
stays at 0, so the joint value omits the spawn rotation — the same
reason the origin had already been switched off `qpos` to `data.xpos`.
The result: every scan was rotated by the (constant) spawn facing
relative to the `odom → base_link` TF (which carries the true world
orientation via `robot0_base_quat`), so slam_toolbox built a clean but
**rotated** occupancy grid and the dashboard SLAM card showed the map
turned relative to the simulated kitchen.

Fix: derive the heading from the base body's world rotation matrix
(`atan2(xmat[1,0], xmat[0,0])`), consistent with the world-frame
origin. The synthetic-MJCF fallback (no resolvable base body) keeps the
`qpos` yaw, which is correct when the joints sit directly on an
identity-spawn body. Regression test:
`tests/unit/test_robocasa_adapter_helpers.py::test_synthesize_laser_scan_2d_uses_world_orientation_under_spawn_rotation`.

Note — `read_panda_mobile_base_velocity` is **not** affected by the same
issue, despite also de-rotating with `qpos[mobile_yaw]`. There the spawn
rotation cancels: the slide-joint axes are defined in the base body's
static (spawn-rotated) frame, so `qvel[forward/side]` are already along
the spawn-rotated axes (`v_world = Rz(spawn) @ (qf, qs)`), and the body's
world yaw is `spawn + joint_yaw`, giving `v_body = Rz(-(spawn+joint_yaw))
@ Rz(spawn) @ (qf, qs) = Rz(-joint_yaw) @ (qf, qs)` — exactly what the
qpos-yaw de-rotation computes. Verified against MuJoCo's
`mj_objectVelocity` ground truth and locked by
`tests/unit/test_robocasa_adapter_helpers.py::test_read_panda_mobile_base_velocity_correct_under_spawn_rotation`.
The scan heading had no such cancelling partner (its origin is a world
*position*, not a joint-axis velocity), which is why only the heading
needed the `xmat` fix.

## Amendment 2026-06-08: three-tier scene paths (ADR-0041)

ADR-0041 split `scenes/` into deploy/sim/benchmark tiers. Both
end-to-end demos in this ADR (`openral deploy sim --config … --enable-slam`
and the Nav2 amendment's `openral deploy sim --config …`) now point at
`scenes/deploy/robocasa_pnp.yaml` because there is no DeployScene
sibling for the old `scenes/benchmarks/panda_mobile_navigate_kitchen.yaml`
and `openral deploy sim` rejects non-DeployScene tiers strictly. The
substrate (`panda_mobile` HAL in a robocasa kitchen) is unchanged; the
robocasa scene id is `PickPlaceCounterToCabinet` rather than
`NavigateKitchen`, which does not affect the SLAM/Nav2 lifecycle
plumbing exercised by the demos (the rSkill and `--enable-slam` flag
do the work; the scene block is just the substrate). See ADR-0041 and
[`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md) for the per-tier strict-
CLI matrix.
