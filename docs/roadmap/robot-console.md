# Robot console — implementation plan

The robot UI (`python/observability/ui/`, ADR-0129) becomes an **operator console**: one
screen built around the robot, with the 3D scene drawn in the page instead of an
embedded Lichtblick. The same build runs on the robot (`openral dashboard`) and on
openral.com (`/robot-ui/`, through the robot's Cloudflare Tunnel).

Users: operators who supervise what the robot does. Light and dark themes.

## Layout

| Zone | Contents | Can the operator hide it? |
|---|---|---|
| Header | robot + unit, mode/role, connection, safety pill, Layout, **E-STOP** | no |
| Prompt bar | one line to the reasoner (the `prompt` action); disabled with the reason when the reasoner is off, its last lifecycle transition failed or stopped it (`topics.lifecycle`, from `deploy.bringup` spans), or the viewer may not prompt | no |
| Cameras | fixed-height row: one tile per camera that streams viewable frames, ordered **left → center → right**, then the 2D occupancy map when the robot has one; missing slots are dropped and the rest centered | per camera |
| 3D scene | URDF posed from TF, occupancy grid, point cloud, world-state objects, collision boxes, payload, markers; a layer list. Fills the rest of the screen | layers, per layer |
| Right pane | one tab at a time: Reasoner (default), Events, World, Safety, Skill, Robot, System. Scrolls on its own; drag its edge to resize | per tab, reorderable |
| Timeline | replay only (phase 3); not shown live | — |

The page itself does not scroll (below 1000 px wide it stacks and scrolls normally).

Layout choices (visible tabs, their order, the default tab, the pane width, the 3D layers, the theme)
are saved in the browser, per robot. Nothing from a robot is rendered as HTML.

## Phases

1. **This PR — console + native 3D.**
   - Manifest: `ManifestPanel.position` (`left` | `center` | `right`) on camera panels,
     derived from the robot description (see below). Older UIs ignore it.
   - UI: header, centered camera strip, native 3D scene, tabbed right pane, layout
     settings, light/dark theme. The old `viz3d` (Lichtblick) panel is replaced.
   - `/ws` proxy: **latest-wins** for snapshot-type topics, so a slow link drops old
     frames instead of queuing them.
2. **Trajectory + payload detail.** The action-chunk path needs a topic: the runner
   publishes the planned end-effector poses (new `MarkerArray`
   `/openral/viz/trajectory`). Payload uses `/openral/viz/attachments` (already there).
3. **Replay in the same view.** The recorder's MCAP feeds the same stores through a
   player; the timeline (episode, reasoner, rSkill, safety lanes; joint plots on
   demand) appears only in replay, with "Jump to live".
4. **Server-side layout settings** per operator (openral.com account), replacing
   browser storage.

## Camera position

`build_robot_manifest` sets each camera panel's `position` from what the sensor's `name`,
`frame_id` and `parent_frame` say (`openral_core.camera_position`):

1. `left` / `right` when one names a side (`left_wrist`, `cam_right`).
2. `left` when one says wrist, hand, gripper or eef (the SO-101 `wrist` camera).
3. `center` otherwise: overhead, head, torso, base and world cameras (the SO-101 `top`
   OAK-D Pro, the G1 `head`).

Camera tiles keep manifest order inside a slot. A missing slot leaves no gap.

## Topics the 3D scene reads

All already pass the bridge allowlist (`openral_foxglove_bringup.topics`). The scene
subscribes only to the layers that are switched on.

| Layer | Topic | Type | Latest-wins |
|---|---|---|---|
| Robot model | `/openral/robot_description`, `/robot_description` | `std_msgs/String` (URDF) | no |
| Frames and link poses | `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | no |
| Joint fallback (no link TF) | `/joint_states` | `sensor_msgs/JointState` | no |
| Occupancy grid | `/map` | `nav_msgs/OccupancyGrid` | yes |
| Point cloud | `/openral/viz/world_cloud/self_filtered_rgb` (the self filter's colour cloud thinned to 40k points by `bucket2_markers`), `/octomap_point_cloud_centers`, `/openral/world_voxels_cloud`, in that priority: every candidate until one delivers, then the ones below it are dropped (an advertised topic can be silent) | `sensor_msgs/PointCloud2` | yes |
| Camera depth | `/openral/viz/cameras/<name>/points`: each depth camera's cloud, and the deploy's own cloud (an estimated-depth camera's, or a pinned driver topic, named after its path), thinned to 40k points by `bucket2_markers` (the raw cloud, 33 MB a frame from a 1080p ZED, is never subscribed; `bucket2_markers` subscribes to a source only while its thinned copy has a viewer) | `sensor_msgs/PointCloud2` | yes |
| Lidar | `/scan`, else `/openral/nav2/scan` | `sensor_msgs/LaserScan` | yes |
| Collision boxes | `/openral/viz/static_geometry`, `/openral/viz/collisions` | `visualization_msgs/MarkerArray` | no |
| Payload | `/openral/viz/attachments` | `visualization_msgs/MarkerArray` | no |
| Objects | world state from the dashboard stream (`world_state` objects) | SSE snapshot | n/a |

Meshes referenced as `package://…` load through the bridge's `assets` capability
(allowlist `ASSET_URI_ALLOWLIST`), three at a time, each mesh once; a failed download is
retried on the next connection. Other mesh URIs fall back to simple link geometry.
Texture URLs inside a robot-supplied URDF or Collada file are never fetched.

The robot model is posed from `/tf` (URDF link frames), not from `/joint_states`: a HAL
may publish manifest joint names and units that differ from the URDF's.

The fixed frame is the world (`map`, else `odom`) when TF connects one to the robot's
base, so a mobile robot drives through a still map; **Follow robot** (top right, saved per
robot) pins the view to the base instead. A fixed-base arm has no world frame and no
button.

## Keeping it fast through the tunnel

- **Subscribe on demand.** A layer that is off is unsubscribed; the bridge sends nothing for it.
- **Latest-wins proxy.** The robot's `/ws` proxy parses the Foxglove protocol it already
  relays. For snapshot types (images, point clouds, occupancy grids, laser scans) it keeps only the newest frame per subscription while the socket is
  busy. Incremental types (TF, markers, URDF), sampled streams (joint states, IMU,
  odometry) and every control frame are relayed in order, unchanged; their unsent
  backlog is bounded (8 MiB), after which the proxy stops reading the bridge.
- **Decode in the page, at display rate.** Messages are decoded (CDR) only when drawn,
  at most once per animation frame per topic; point clouds are decimated to a point budget.
- **Cameras stay on JPEG snapshots** (`/api/camera/<id>/snapshot.jpg`, 1 Hz remote, 4 Hz
  local), as today.

## Libraries (all permissive)

`three` (MIT), `urdf-loader` (Apache-2.0), `@foxglove/ws-protocol` (MIT),
`@foxglove/rosmsg` + `@foxglove/rosmsg2-serialization` (MIT).
