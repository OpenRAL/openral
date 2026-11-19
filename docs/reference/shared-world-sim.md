# Several robots in one MuJoCo world (`openral deploy world`)

`openral deploy sim` runs one robot in its own world. `openral deploy world`
runs several robots in **one** MuJoCo world, so they can see and touch the same
table and objects, and each other (issue #666). Sim only.

```bash
openral deploy world --config scenes/worlds/so101_handover_table.yaml
```

That fixture is two SO-101 follower arms facing each other across a table, with
a 3 cm cube between them, within reach of both. The command prints each robot's
ROS domain, dashboard URL and log file. Ctrl-C stops everything.

Arguments after `--` go to every robot's `openral deploy sim`. On a headless
host, or to keep the graphs small:

```bash
openral deploy world --config scenes/worlds/so101_handover_table.yaml -- \
    --hal viewer_enabled=false --no-foxglove --no-object-detector
```

To talk to one robot, use its domain: `ROS_DOMAIN_ID=36 openral prompt "…"`,
or open its dashboard.

## How it works

```
                 openral deploy world
                 └─ SharedWorldServer: ONE MjModel/MjData (both arms + arena)
                        ▲ Unix socket            ▲ Unix socket
  ROS_DOMAIN_ID=36      │                        │     ROS_DOMAIN_ID=40
  openral deploy sim ───┘                        └─── openral deploy sim
  HAL (SharedWorldArmHAL, mirror of the world)        HAL (mirror of the world)
  safety kernel, runner, reasoner, dashboard …        safety kernel, runner, …
```

- **One physics process.** `deploy world` composes every robot's own MJCF and
  the world's `arena_mjcf` into one model and steps it.
- **One unchanged graph per robot.** Each robot is an ordinary
  `openral deploy sim` on its own `ROS_DOMAIN_ID` and ports. Topic names, the
  launch file, the safety kernel and the runner are untouched; the two graphs
  never see each other on DDS.
- **The HAL is the only thing that knows.** Each graph's HAL is a
  `SharedWorldArmHAL`: a `MujocoArmHAL` that keeps a *mirror* of the world for
  its cameras, joint reads and `/clock`, and asks the server to advance instead
  of stepping the mirror.
- **Each robot sees the world in its own base frame.** A robot's mirror has its
  own base at the origin and its own MJCF elements unprefixed, exactly like a
  single-robot twin; the other robots appear as `<name>_…` bodies. So sensor
  extrinsics, the safety workspace and place zones mean what they mean in a
  single-robot deploy. Base poses are planar (`base_xyz`, `base_yaw_deg`), so
  gravity is the same in every view.

## Robots see each other: peers in the safety check

Robots in one workcell must know about each other and not collide. Each
robot's safety kernel already refuses a command that drives one of its links
into an occupied cell of its world grid (an `OccupancyVoxels` in its base frame,
refused when stale). In a shared world that grid holds **the other robots**:

- `openral deploy world` binds each robot's HAL to the world; `deploy sim` then
  passes the launch `shared_world:=` / `shared_world_robot:=`. The launch starts
  `openral_hal.peer_voxels` in that robot's graph, turns its kernel's world
  check on over the robot's coverage ball, and remaps the kernel's
  `/openral/world_voxels` to `/openral/peer_voxels`. Only the kernel reads the
  peers: the other world-grid readers (object lift, tracker, attachment
  barrier) expect a perceived map and keep their no-octomap behaviour.
- `peer_voxels` reads the world as that robot sees it, takes every collision
  geom of the other robots (by kinematic root, not by name), and rasterizes
  their bounding boxes, in the frame of the body `base_frame` names, into the
  cells of the coverage ball (the same ball, resolution and cell budget the
  launch gives the kernel). A cell is occupied whenever a peer box can touch it.
- **Where a peer is going, too.** Each peer is rasterized where it is and at
  poses up to the joint positions its position servos settle at (gain, gear and
  `ctrlrange` included), sampled at most 0.05 rad apart and 25% past the target
  (a P servo overshoots), so a peer commanded into the space between the arms is
  occupied there before it moves.
- **Tight freshness.** The kernel accepts a peer grid for 0.2 s of world time
  (receipt deadline and data-age budget), four of `peer_voxels`' 20 Hz periods,
  instead of the octomap's 1 s / 1.5 s. What remains is the race between two
  robots commanded in the same instant, each checked against the other's last
  grid: bounded by the peer's motion over those 0.2 s.
- **Fail-closed.** Every grid is built from a fresh read of the world and is
  stamped with that world time (`source_stamp`). A non-finite pose withholds the
  grid. When the world stops answering, `peer_voxels` exits, the kernel's
  receipt deadline (`world_voxel_deadline_s`) and data-age budget refuse every
  command, and the launch shuts that robot's graph down (which ends the world).
- **Refused combinations.** The launch refuses `shared_world` with the octomap
  leg (two world grids), with the kernel's world check off, off the simulation
  clock (the peers carry world time), and with the rules that tolerate hand
  contact with grid cells (`grasp_allowance_enabled`,
  `workspace_contact_enabled`, an `exemption_provider`): those would tolerate
  contact with the other robot. `deploy sim` refuses the octomap case up front
  and names the remedy (`-- --no-enable-octomap`).
- The kernel is unchanged: this is an input it already consumes.

Measured on the fixture on q5090 (each robot driven by scripted joint targets
through its kernel):

| Command | Without peers (`3fb80f14`) | With peers |
|---|---|---|
| `a` pan -0.4 rad (away from `b`) | executes | executes (210 of 211 actions passed) |
| `a` pan +0.9 rad into `b` at rest | executes: `a` reaches 0.894 rad and pushes `b`'s joints off their pose | refused on its first row (`safety.collision kind=world a=gripper b=voxel_…`, E-stop latched); `a` never moves, `b` untouched |
| `b` pan +0.9 rad into `a` at rest | — | refused the same way on `b`'s kernel |
| `b` pan +0.9 rad with `a` at -0.4 rad (4 cm clearance) | — | executes |
| `a` pan -0.2 rad (clear) after its `peer_voxels` is killed | — | refused, `voxel_unavailable` |

**Limits.** A robot of a shared world runs with its octomap leg off. The peers
are their collision geoms' bounding boxes, a few millimetres larger than the
meshes. A held object is not part of its robot's box set.

**Real robots in one workcell (follow-up).** The rasterizer takes oriented boxes
in the robot's base frame (`PeerBox`), whatever their source. For real robots
the source would be each peer's manifest `collision_geometry`, posed by its
`/joint_states` (bridged from its domain) and its surveyed base pose in this
robot's base frame, plus its commanded targets, with the same freshness rule and
the octomap merge. That source needs a decision on how peers' states cross
domains and a hazard-log entry before it is built.

## Time

The single-robot rule holds for every robot: **one action is one control
period**, and each graph's `/clock` is the world's simulation time.

- A robot is *commanding* from an action until its next idle step (the sim
  bridge idle-steps a robot that has had no action for 2 s).
- The world advances one control period once **every commanding robot** has
  sent its action. Two robots running skills are in lockstep; a robot between
  inference chunks makes the other wait for it.
- A commanding robot that sends nothing for `barrier_timeout_s` (default 5 s,
  wall time, at most 7 s) is dropped from the barrier: the world goes on, its
  last targets are held, and `shared_world.barrier_timeout` is logged.
- With no robot commanding, the world runs on wall time.
- A robot whose HAL stops (shutdown, or its `estop`) leaves the world. Its arm
  is first retargeted to the pose it has at that moment, so it holds there while
  the other robots keep running. When its HAL comes back it takes those held
  targets, so the arm does not move on re-joining. A graph-level E-stop works as
  in a single-robot deploy: the safety kernel stops passing actions, the robot
  goes idle and its arm holds its last accepted target.
- While a robot waits for the other at the barrier, its HAL node is blocked in
  `send_action` (as it is in any slow simulator step): its camera and joint-state
  publishing pause for that wait. Simulation time does not advance meanwhile.

## Writing a world

```yaml
id: so101_handover_table
arena_mjcf: assets/handover_table.xml      # table, objects, lights; its frame is the world frame
robots:
  - name: a                                # lowercase letters and digits
    scene: ../deploy/so101_shared_table.yaml
    base_xyz: [0.1, 0.27, 0.0]             # the robot's MJCF origin in the world
    base_yaw_deg: 0.0
    ros_domain_id: 36
    dashboard_port: 4336
    foxglove_port: 8836
  - name: b
    scene: ../deploy/so101_shared_table.yaml
    base_xyz: [-0.1, -0.27, 0.0]
    base_yaw_deg: 180.0
    ros_domain_id: 40
    dashboard_port: 4340
    foxglove_port: 8840
```

- A robot's `scene` is a `DeployScene` that builds a bare MuJoCo twin: it names
  `robot_id`, declares no `composition`, and its `scene.id` is not a registered
  sim scene. Several robots may share one file.
- `ros_domain_id` values must be at least 2 apart: under the Fast-DDS
  large-data profile adjacent domains see each other. Prefer domains below 102
  (above, a domain's RTPS ports fall in Linux's ephemeral port range) and avoid
  0, 42, 50-101 and 77, which OpenRAL uses elsewhere.
- Arena element names must not clash with a robot's (the model then does not
  compile, and the error names the clash).

## Limits of this first version

- Robots whose manifest derives its sim HAL (`hal.sim: null` and a `sim:`
  block with no `keyframe_index`): fixed base, hinge and slide joints, no
  actuator activation states.
  Robots with their own sim HAL class, scene-attached scenes, mobile bases and
  the Isaac backend are refused.
- Every robot must have the same control period. `deploy world` checks this,
  and that every robot's view of the world compiles, before it starts a graph.
- The world steps and renders with the first robot's MJCF `<option>` and
  `<visual>`; mixing robots whose MJCFs need different solver options is not
  supported yet.
- Robots with a depth camera and the octomap leg (the peers would need to
  be merged into that grid) are refused; see above.
- The peer check is a simulation input. The shared-workspace hazard (two arms
  in one space) needs a hazard-log entry before any real-robot source is built.
- Nothing coordinates the robots: no claims on objects, no hand-over skill.
  That is the fleet layer's job (OpenRAL Pro).
- One dashboard per robot; there is no merged view.

## Decisions taken here

Recorded for review; the matching entry in the private decision log is owed.

1. **One DDS domain per robot, one shared physics process** rather than one
   namespaced graph. About 70 absolute topic names across ten packages,
   including the C++ safety kernel, stay as they are.
2. **A mirror per robot in its own base frame** rather than name prefixes and
   index offsets in the HAL: everything a graph reads from MuJoCo keeps its
   single-robot meaning.
3. **Lockstep with a bounded wait** rather than free-running physics: a slow
   policy slows the world instead of losing control periods, as in a
   single-robot deploy; a hung graph cannot stall the other for more than
   `barrier_timeout_s`.
4. **`SharedWorldScene` is a new file type** (`scenes/worlds/`); `DeployScene`
   is unchanged.
5. **Peers reach the kernel as occupied cells** on its existing
   `/openral/world_voxels` input, not as a new kernel input: no Safety-WG-gated
   code changes, and the existing freshness rules make a stale peer fail
   closed.
