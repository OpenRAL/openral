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
| Cameras | one tile per image sensor, ordered **left → center → right**; missing slots are dropped and the rest centered | per camera |
| 3D scene | URDF + TF, occupancy grid, point cloud, world-state objects, collision boxes, payload, markers; a layer list | layers, per layer |
| Right pane | one tab at a time: Reasoner (default), Events, World, Safety, Skill, Robot, System | per tab, reorderable |
| Timeline | replay only (phase 3); not shown live | — |

Layout choices (visible tabs, their order, the default tab, the 3D layers, the theme)
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

`build_robot_manifest` sets each camera panel's `position`:

1. `left` / `right` when the sensor name or `frame_id` says so (`left_wrist`, `cam_right`).
2. `left` for a camera that moves with the arm: `parent_frame` (else `frame_id`) is a robot
   link other than the base (the SO-101 `wrist` camera on `gripper`), or the name or frame
   says wrist, hand, gripper or eef.
3. `center` otherwise (world- or base-fixed, e.g. the SO-101 `top` OAK-D Pro).

Camera tiles keep manifest order inside a slot. A missing slot leaves no gap.

## Topics the 3D scene reads

All already pass the bridge allowlist (`openral_foxglove_bringup.topics`). The scene
subscribes only to the layers that are switched on.

| Layer | Topic | Type | Latest-wins |
|---|---|---|---|
| Robot model | `/openral/robot_description`, `/robot_description` | `std_msgs/String` (URDF) | no |
| Frames | `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | no |
| Occupancy grid | `/map` | `nav_msgs/OccupancyGrid` | yes |
| Point cloud | `/openral/world_cloud/self_filtered_rgb`, `/octomap_point_cloud_centers`, `/openral/world_voxels_cloud` | `sensor_msgs/PointCloud2` | yes |
| Collision boxes | `/openral/viz/static_geometry`, `/openral/viz/collisions` | `visualization_msgs/MarkerArray` | no |
| Payload | `/openral/viz/attachments` | `visualization_msgs/MarkerArray` | no |
| Objects | world state from the dashboard stream (`world_state` objects) | SSE snapshot | n/a |

Meshes referenced as `package://…` load through the bridge's `assets` capability
(allowlist `ASSET_URI_ALLOWLIST`). Other mesh URIs fall back to simple link geometry.

## Keeping it fast through the tunnel

- **Subscribe on demand.** A layer that is off is unsubscribed; the bridge sends nothing for it.
- **Latest-wins proxy.** The robot's `/ws` proxy parses the Foxglove protocol it already
  relays. For snapshot types (images, point clouds, occupancy grids, joint states, laser
  scans, odometry) it keeps only the newest frame per subscription while the socket is
  busy. Incremental types (TF, markers, URDF) and every control frame are relayed in
  order, unchanged.
- **Decode in the page, at display rate.** Messages are decoded (CDR) only when drawn,
  at most once per animation frame per topic; point clouds are decimated to a point budget.
- **Cameras stay on JPEG snapshots** (`/api/camera/<id>/snapshot.jpg`, 1 Hz remote, 4 Hz
  local), as today.

## Libraries (all permissive)

`three` (MIT), `urdf-loader` (Apache-2.0), `@foxglove/ws-protocol` (MIT),
`@foxglove/rosmsg` + `@foxglove/rosmsg2-serialization` (MIT).
