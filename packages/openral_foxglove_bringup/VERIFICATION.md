# Prototype verification record

Spike from `docs/investigations/foxglove-dashboard-port-feasibility.md`.
Verified 2026-06-16 on ROS 2 Jazzy, `foxglove_bridge` 3.2.6.

## What was proven (real, end-to-end)

| Claim | Evidence | Status |
|---|---|---|
| Package generates a valid launch graph | `generate_launch_description()` → 6 entities (4 args + 2 gated nodes); imports clean under ROS Python 3.12 | ✅ |
| Safety invariants hold | 15/15 unit tests pass (`test/test_foxglove_launch.py`): read-only caps, safety/e-stop/action topics never matched by the allowlist, layout only references whitelisted topics | ✅ |
| Bridge starts with our posture | Log: `Server listening on port 8765`; `ss` shows bind on **`127.0.0.1`** (loopback default, not upstream `0.0.0.0`) | ✅ |
| Read-only over the wire | A live Foxglove-protocol client read `serverInfo.capabilities = ["connectionGraph","assets"]` — **no `clientPublish`, no `services`**. A connected viewer cannot publish or call services. | ✅ |
| Bucket-1 topics exposed natively | Bridge advertised `/tf_static -> tf2_msgs/msg/TFMessage` (real TF from two `static_transform_publisher`s: `map→odom→base_link`) | ✅ |
| Non-whitelisted topics excluded | `/rosout`, `/parameter_events` were on the wire but **not** advertised by the bridge | ✅ |
| **End-to-end message delivery** | A Foxglove-protocol client subscribed and received **`/tf` 40 msgs/2s + `/joint_states` 40 msgs/2s** (20 Hz) through the bridge; bridge log: `created ROS subscription on /tf … successfully`. Native schemas `tf2_msgs/msg/TFMessage`, `sensor_msgs/msg/JointState`. | ✅ |
| Pixel render in Foxglove Studio | **Not captured here** — the headless env has no installable browser (Chrome needs sudo; bundled Chromium was removed). Data delivery is proven at the protocol layer; connect any Foxglove client to `ws://localhost:8765` and set the 3D panel's Fixed frame to `map`. | ⚠️ env-blocked |

## Stale-bridge gotcha (foxglove-sdk-cpp v0.18.0)

The bridge is `foxglove-sdk-cpp/v0.18.0`. Observed: if a topic appears, its
publisher dies, and the topic later reappears from a new publisher, the
**already-running bridge advertises the channel but never forwards data**
(client subscribes, gets zero messages, no error). Restarting the bridge after
the publishers are steady fixes it immediately (verified: 0 → 20 Hz delivery).

**Implication for the port:** launch `foxglove_bridge` *after* the OpenRAL
topic producers are up, or restart it if the topic graph churns (e.g. a sim
relaunch). Worth tracking as an upstream bridge robustness issue.

## Reproduce

```bash
# Terminal 1 — bridge
source /opt/ros/jazzy/setup.bash
PYTHONPATH=packages/openral_foxglove_bringup:$PYTHONPATH \
  ros2 launch packages/openral_foxglove_bringup/launch/foxglove.launch.py

# Terminal 2 — a real topic to look at (until the OpenRAL sim is running)
ros2 run tf2_ros static_transform_publisher --frame-id map --child-frame-id odom
ros2 run tf2_ros static_transform_publisher --x 1 --frame-id odom --child-frame-id base_link

# Then: app.foxglove.dev → Open connection → Foxglove WebSocket → ws://localhost:8765
#       → import config/openral_layout.json
```

## Gotcha found: subprotocol is `foxglove.sdk.v1`

`foxglove_bridge` 3.2.6 is the **new Rust/`tokio-tungstenite` SDK bridge**. It
negotiates the WebSocket subprotocol **`foxglove.sdk.v1`**, not the legacy
`foxglove.websocket.v1`. A client offering only the old string is rejected with
a misleading `400 Bad Request: Missing expected sec-websocket-protocol header`.

- **Browsers / current Foxglove Studio & app.foxglove.dev**: negotiate this
  automatically — no action needed.
- **Custom/CLI clients** (and older self-hosted Studio builds): must offer
  `foxglove.sdk.v1`. This is the one real compatibility caveat for the port.

## Verified against a real OpenRAL deploy-sim graph

Built the ROS workspace (`just ros2-build`, 24 pkgs) and ran the real
`openral deploy sim --config scenes/deploy/openarm_tabletop.yaml
--no-object-detector` graph (HAL + safety kernel + reasoner + dashboard, fully
ACTIVE). Pointed the bridge at it.

| Result | Evidence |
|---|---|
| Bridge exposes the real Bucket-1 topics | advertised `/joint_states`, `/tf`, `/tf_static`, `/map`, `/openral/cameras/{base,left_wrist,right_wrist}/image` |
| **Safety topics excluded, live** | `/openral/estop`, `/openral/safe_action`, `/openral/candidate_action`, `/openral/failure/safety` were on the graph but **not advertised** — the allowlist holds against a real graph |
| **Real cameras + joints deliver end-to-end** | through the bridge in 3 s: `/openral/cameras/base/image` 30 msgs/27 MB, `left_wrist` 30 msgs/27 MB, `/joint_states` 90 msgs (30 Hz) |
| `/tf`, `/map` empty at idle | no `robot_state_publisher` in the deploy graph (see below); SLAM off for the fixed-base arm |

### Idle-stepping is fixed (cameras stream with no skill running)

Earlier I wrongly concluded cameras need a skill to step the sim — that was a
premature reading taken seconds after boot. **Corrected:** master has an
autonomous idle-step timer (`sim_sensor_bridge.py`, ADR-0034 idle-stepper
amendment) gated **only** on the HAL exposing `idle_step`, *not* on
`enable_sim_clock`. Empirically, the idle OpenArm graph (no skill, reasoner
unable to dispatch — no LLM) streamed cameras (base ~6 Hz, wrists ~1-2 Hz) and
`/joint_states` at 30 Hz. So the cameras populate Foxglove at idle.

Note: raw uncompressed images are heavy (~9 MB/s/camera); enable image
compression or fewer cameras on memory-constrained hosts.

## /tf + robot-model rendering (`with_robot_state_publisher`)

deploy-sim publishes `/joint_states` but not dynamic `/tf` (no
`robot_state_publisher` in its graph), so the 3D panel had no frames and could
not draw the robot. The launch now offers opt-in publishers:

```bash
ros2 launch openral_foxglove_bringup foxglove.launch.py \
  with_robot_state_publisher:=true \
  with_joint_state_publisher:=true \   # only WITHOUT a sim — else it fights real /joint_states
  robot_description_urdf:=$(python -c "from robot_descriptions import panda_description; print(panda_description.URDF_PATH)")
```

**Verified (isolated, `ROS_DOMAIN_ID=42`, panda URDF):** rsp turned
`/joint_states` + URDF into real panda link transforms — `/tf` carrying
`panda_link0→panda_link1`, `panda_hand→panda_leftfinger`, … at 10 Hz — plus
`/robot_description` and `/tf_static`. Delivered end-to-end through the bridge:
`/tf` 30 msgs, `/robot_description` 1 msg (15 KB URDF), `/tf_static` 1 msg.

Two caveats:
- **Meshes:** the example-robot-data panda URDF references
  `package://example-robot-data/...` meshes, which aren't an ament package on
  the ROS path, so Foxglove renders link **frames/structure** but not textured
  meshes. A URDF whose meshes resolve via `package://<ament-pkg>` (served by the
  bridge's `assets` capability) would show full geometry.
- **OpenArm has no local URDF** (`robots/openarm/robot.yaml`, ADR-0027:
  `urdf_path` deliberately unset). This feature pairs with robots that resolve a
  URDF — e.g. `franka_panda` / `panda_mobile` (`panda_description`), `ur5e`,
  `so101_follower`. Under a real deploy-sim, set only
  `with_robot_state_publisher:=true` (the sim is the `/joint_states` source).

## Host note

`cam2image` could not be remapped onto a camera topic in this environment
because the host's miniforge Python 3.13 shadows ROS's 3.12 and breaks ros2cli
`--ros-args` remap forwarding (unrelated to this package). The real deploy-sim
above made that stand-in unnecessary.
