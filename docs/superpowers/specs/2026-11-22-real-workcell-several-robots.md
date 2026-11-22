# Several real robots in one workcell (`openral deploy run`)

Status: design only, no code (OpenRAL/openral-pro#99 item 3). `openral deploy run` keeps
refusing a `DeployScene` that lists `robots` (`deploy_sim.py`, "`deploy run` drives one robot
per graph on its own host: not yet").

## 1. What sim already does, and what carries over

`openral deploy sim` of a scene with several robots (OpenRAL/openral#708) runs:

- one ROS graph per robot, each on its own `ROS_DOMAIN_ID`, dashboard port and Foxglove port
  (`deploy_robots.robot_slots`: domains 30, 34, 38, 46; dashboards 4330 + i);
- each graph reads its own one-robot view of the scene (`$OPENRAL_SCENE_ROBOT`,
  `geometry.scene_for_robot`): its `base_pose` in the shared `world` frame, its nav goals and
  place zones re-expressed in its own `map`, which starts at that `base_pose`;
- a static `world -> map` TF per graph and `world_pose` / `latch_detail` in each
  robot's `GET /api/mission` (this PR), so the platform's fleet console and super reasoner
  place every robot in one frame and see why one stopped;
- one shared simulator (the Isaac sidecar), the only thing that joins the graphs.

A real workcell keeps the first three and replaces the last with the physical cell. Each
robot's graph stays the unchanged one-robot `deploy run` graph: its HAL, its C++ safety
kernel, its reasoner, on its own host or its own domain on a shared host.

## 2. Per-robot graphs and domains

- **One graph per robot, one domain per graph**, as in `deploy_robots`. A robot on its own
  host (the usual case: each robot carries its compute) needs only a distinct domain if the
  hosts share a network segment; robots on one host use `robot_slots` unchanged.
- **The launcher.** `deploy run` on a scene with `robots` would not start every robot's
  graph from one process the way `deploy sim` does: a real robot's graph runs on that
  robot's own computer. Proposed: `openral deploy run --config <scene> --robot <name>` on each
  robot's host, which sets `$OPENRAL_SCENE_ROBOT=<name>` and the slot's domain/ports, then
  runs today's one-robot `deploy run`. No new process supervisor; the operator (or a fleet
  agent) starts each robot. `--robot` is required whenever the scene lists `robots`.
- **No cross-domain DDS in the safety path.** A robot's kernel never subscribes to another
  domain. Anything a peer contributes (section 4) arrives through an explicit, typed bridge
  whose staleness the kernel checks like any other input.

## 3. Each robot's `world` pose from a calibrated workcell frame

In sim, `SceneRobot.base_pose` is exact: the sidecar spawns the robot there. On real hardware
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
  date and residual. `deploy run --robot` refuses a scene robot whose unit has no calibration
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

Today (OpenRAL/openral#708) a robot sees a peer only through its own sensors, as an obstacle
in its octomap. The design below adds the peers' own geometry. It is refusal-only: it adds
occupied cells, never removes one, so it stays open (CLAUDE.md §1.9).

### 4.1 Peer state across ROS domains

Each robot's graph is on its own `ROS_DOMAIN_ID` (`deploy_robots.robot_slots`). Per robot, a
`domain_bridge` (or one Zenoh router for the cell) brings into this robot's domain, under a
per-peer namespace `/openral/peers/<name>/`:

- the peer's `/joint_states` (measured, the peer's own stamps kept);
- the peer's `map -> odom -> base` (its localisation: `/tf` filtered to those two edges);
- its `world_pose` (the static `world -> map`), already published per graph (this PR).

The peer's URDF comes from its manifest (`SceneRobot.robot_id` -> `robots/<id>/robot.yaml`),
which every graph of the scene can resolve; no URDF crosses the wire. In sim the bridge carries
the sim-clock stamps; on real hardware each host's clock must be synchronised (chrony/PTP)
and the bridge's measured latency sets the freshness bound.

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
budget, about 0.2 s, set from the bridge's measured latency). A stale or missing peer grid
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
- clock synchronisation and bridge latency on real hardware, and what happens on a partition
  (every robot's peer grid goes stale: all stop);
- that no exemption path reads the peer grid (a test per rule);
- ADR-0137 and hazard-log Entry 102 rewritten for this design (OpenRAL/management#131).

## 5. Open/Pro split

Per ADR-0083 fleet coordination is OpenRAL Pro; per CLAUDE.md §1.9 every refusal-only safety
check stays open.

| Piece | Where | Why |
| --- | --- | --- |
| `DeployScene.robots`, `scene_for_robot`, per-robot domains/ports, `--robot` on `deploy run` | open | Describes one robot's graph |
| `world -> map` TF, `world_pose`, `latch_detail` in `/api/mission` | open | Each robot reporting about itself |
| Per-unit workcell calibration (`world_from_dock`) and its refusal | open | Per-robot calibration, like depth mounts |
| Peer state bridge, peer producer, the kernel's peer grid | open | Refusal-only check; relaxes nothing |
| Fleet coordinator: goal allocation, the super reasoner, the fleet console | Pro (platform) | Fleet coordination (ADR-0083) |
| Contact exemptions (provider) | Pro, as today | Never reads the peer grid |

## 6. Not decided here

- Whether real robots of one cell may share one host (domains on one machine) or must each
  run on their own; this design allows both.
- The relocalisation path that keeps `world -> map` true after a map reset.
- The bridge (domain_bridge vs Zenoh) and its measured latency budget on the cell network.
