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
- Each graph's safety kernel knows only its own robot. The other robot is
  visible in its cameras, but is not a collision object for the kernel unless
  the robot has a depth sensor feeding the voxel map. Two arms can therefore
  collide in the shared world. Before this is used for anything but simulation
  experiments, a shared-workspace hazard needs a hazard-log entry.
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
