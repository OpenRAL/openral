# Several real robots in one workcell (`openral deploy run`)

Status: design only, no code (OpenRAL/openral-pro#99 item 3). Updated 2026-11-25
(OpenRAL/openral#765, OpenRAL/openral-pro#106): several robots start with
`openral deploy fleet`, part of OpenRAL Pro. `deploy sim` and `deploy run` start one robot. A
scene that lists `robots` refuses there and names `deploy fleet`.

## 1. What sim already does, and what carries over

`openral deploy fleet` (OpenRAL Pro) runs a scene with several robots in sim. History: until
OpenRAL/openral#765 this was open `openral deploy sim` (`openral_cli.deploy_robots`,
`geometry.scene_for_robot`, OpenRAL/openral#708). It runs:

- one ROS graph per robot, each on its own exported `ROS_DOMAIN_ID`, dashboard port and
  Foxglove port (domains 30, 34, 38, 46; dashboards 4330 + i);
- each graph runs the unchanged one-robot `deploy sim` on its own slice of the scene, a file
  `deploy fleet` writes (`$OPENRAL_SCENE_ROBOT` names the robot): its `base_pose` in the
  shared `world` frame, its nav goals and place zones re-expressed in its own `map`, which
  starts at that `base_pose`;
- a static `world -> map` TF per graph and `world_pose` / `latch_detail` in each
  robot's `GET /api/mission` (this PR), so the platform's fleet console and super reasoner
  place every robot in one frame and see why one stopped;
- one shared simulator (the Isaac sidecar), the only thing that joins the graphs.

A real workcell keeps the first three and replaces the last with the physical cell. Each
robot's graph stays the unchanged one-robot `deploy run` graph: its HAL, its C++ safety
kernel, its reasoner, on its own host or its own domain on a shared host.

## 2. Per-robot graphs and domains

- **One graph per robot, one domain per graph**, as `deploy fleet` starts them. A robot on
  its own host (the usual case: each robot carries its compute) still needs a distinct domain,
  since its peers are read from their own domains. Robots on one host get distinct domains too.
- **The launcher.** `deploy fleet` writes each robot's slice and starts the unchanged
  one-robot graph on it, with `$OPENRAL_SCENE_ROBOT`, `$OPENRAL_SCENE_PEERS` and the robot's
  exported `ROS_DOMAIN_ID`. On real hardware it runs `deploy run --config <slice>` on each
  robot's host (OpenRAL/openral-pro#106 step 3). There is no `--robot` flag: the slice is
  already one robot's scene.
- **The kernel reads no other domain.** `peer_voxels`, in this robot's graph, reads each
  peer's domain (section 4.1) and hands the kernel one grid. The kernel checks its staleness
  like any other input.

## 3. Each robot's `world` pose from a calibrated workcell frame

In sim, a robot's `base_pose` in the fleet scene is exact: the sidecar spawns the robot there. On real hardware
it is a claim that must be measured.

- **The workcell frame.** One fixed, surveyed frame per cell (`world`): fiducials on the
  floor or the cell structure, or the building map's origin. It is the frame of every
  scene `nav_goals` / `place_zones` pose, as in sim.
- **Each robot's start.** Calibrated, not typed: either a docking station whose pose in
  `world` is surveyed once (the robot starts docked; its map origin is the dock), or a
  localisation against a prior map of the cell expressed in `world` (AMCL/slam_toolbox
  localisation mode, initial pose from the dock). Either gives `world -> map` per robot;
  the launch publishes it as in sim, and `world_pose` serves it.
- **Where it lives.** A per-unit, per-cell calibration file beside the unit overlay
  (`robots/<id>/units/<unit>.yaml` pattern): `cell`, `world_from_dock` and the measurement's
  date and residual. `deploy run` refuses a robot whose unit has no calibration
  for this cell, the way it refuses an uncalibrated depth mount today
  (`depth_extrinsic_problems`).
- **Drift.** `world -> map` is static only while the robot's map is the one it started.
  A robot that relocalises (map reset, SLAM loop closure moving `map -> odom` far) must
  republish it from the localisation, not the start. Until that exists, `world_pose` is
  "where the map started", documented as such; the console must not draw a robot whose
  localisation confidence is lost.

## 4. Each robot knows where its peers are, and its kernel refuses a move into one

Decision (2026-11-22): a robot must know where every other robot of its workcell is (its
`world` pose, its URDF collision geometry, its current `/joint_states`) and its safety kernel
must refuse a motion that would hit one. A contact exemption never covers a peer robot. There
are no reservations, exclusion zones or pairwise target checks: robots work in parallel, real
or sim, and a several-robot scene otherwise runs exactly like a one-robot scene (grasp and
place exemptions included).

The peer grid is built (OpenRAL/openral#725): besides seeing a peer as an obstacle in its
octomap, each robot's kernel checks the peers' own geometry. It is refusal-only: it adds
occupied cells, never removes one, so it stays open (CLAUDE.md §1.9).

### 4.1 Peer state across ROS domains

Each robot's graph is on its own `ROS_DOMAIN_ID`. `$OPENRAL_SCENE_PEERS` names each peer,
its `robot_id` and its domain (`ScenePeer`). There is no bridge: `peer_voxels`, in this
robot's graph, joins each peer's domain directly and reads:

- the peer's `/joint_states` (measured, the peer's own stamps kept);
- the peer's `world -> map -> odom -> base` from its `/tf` and `/tf_static`; each graph
  publishes its own static `world -> map`.

The peer's URDF comes from its manifest (`ScenePeer.robot_id` -> `robots/<id>/robot.yaml`),
which every graph of the scene can resolve; no URDF crosses the wire. In sim the stamps are
sim-clock stamps. On real hardware each host's clock must be synchronised: `deploy run` with
peers, and `deploy_e2e.launch.py` under `hal_mode:=real` (so a direct `ros2 launch` too),
refuses a leap status that is not `Normal` or a worst-case clock error (chrony's offset plus
root dispersion plus half the root delay) over 10 ms, provisional until the Safety-WG sign-off.

### 4.2 Posing a peer into this robot's frame

`this.base <- this.map <- world <- peer.map <- peer.odom <- peer.base`: both `world -> map`
edges are the scene's `base_pose`s (static), the `map -> odom -> base` edges are each robot's
live localisation. A peer producer node in this robot's graph (`peer_voxels`, revived from
the one removed with `openral deploy world`, now reading `/joint_states` instead of a shared
MuJoCo world) runs FK of the peer's URDF collision geometry at its measured joints, poses it
into this robot's `base_frame`, rasterizes it into the cells of the kernel's coverage ball,
swept along the peer's measured joint velocity over the freshness window (as the removed
producer did), and publishes it stamped with the oldest input it used.

### 4.3 What the kernel consumes

A **second world grid**, `/openral/peer_voxels` (`OccupancyVoxels`, the type it already
reads), next to the octomap's `/openral/world_voxels`. The removed design remapped the
kernel's one grid input to the peers and refused the octomap leg; both are needed now, so the
kernel gains a second grid subscription (same checks, its own receipt deadline and data-age
budget, 300 ms: hazard-log Entry 103). A stale or missing peer grid
while the scene lists peers is a refusal (`voxel_unavailable`), never free space.

### 4.4 Exemptions never cover a peer

Every contact rule (grasp-target exemption, workspace contact, the place allowance, the
ADR-0121 exemption provider, payload own-cells) reads only the octomap grid. The kernel checks
the peer grid with no exemption at all: a hand link inside a live grasp region still trips on
a peer cell. Because the peer grid is a separate input, this needs no per-cell ownership in
the octomap and no change to the exemption provider's API. (The octomap still sees a peer as
ordinary occupancy, which an exemption may tolerate; the peer grid is what refuses it.)

### 4.5 Simultaneous moves

Each kernel checks its command against the peers' measured pose swept over the freshness
window, so two robots commanded in the same instant are each checked against the other's
state at most that old. This is the bound; there is no barrier comparing their targets. The
hazard-log entry states it with the measured window.

### 4.6 Safety-WG review

- the kernel's second grid input and its freshness bound (a kernel change: TDD, hazard log);
- the peer producer's FK and sweep (is the swept volume conservative for the peer's speed?);
- clock synchronisation and peer latency on real hardware, and what happens on a partition
  (every robot's peer grid goes stale: all stop);
- that no exemption path reads the peer grid (a test per rule);
- ADR-0137 and hazard-log Entry 102 rewritten for this design (OpenRAL/management#131).

## 5. Open/Pro split

Per ADR-0083 fleet coordination is OpenRAL Pro; per CLAUDE.md §1.9 every refusal-only safety
check stays open.

| Piece | Where | Why |
| --- | --- | --- |
| One-robot `DeployScene`; `$OPENRAL_SCENE_ROBOT` and `$OPENRAL_SCENE_PEERS` (`ScenePeer`) | open | Describes one robot's graph |
| Refusals: peers without a `world` pose, a shared or unexported domain, an unsynchronised clock, a scene that lists `robots` | open | Refusal-only checks |
| `world -> map` TF, `world_pose`, `latch_detail` in `/api/mission` | open | Each robot reporting about itself |
| Per-unit workcell calibration (`world_from_dock`) and its refusal | open | Per-robot calibration, like depth mounts |
| `peer_voxels`, the kernel's peer grid | open | Refusal-only check; relaxes nothing |
| The `openral.cli` entry-point seam | open | Lets an installed package add a command |
| `openral deploy fleet`: the fleet scene, its slices, per-robot domains/ports, starting every robot | Pro | Starts several robots (ADR-0083) |
| Fleet coordinator: goal allocation, the super reasoner, the fleet console | Pro (platform) | Fleet coordination (ADR-0083) |
| Contact exemptions (provider) | Pro, as today | Never reads the peer grid |

## 6. Not decided here

- Whether real robots of one cell may share one host (domains on one machine) or must each
  run on their own; this design allows both.
- The relocalisation path that keeps `world -> map` true after a map reset.
- The measured peer latency on a real cell network, against the 300 ms budget.
