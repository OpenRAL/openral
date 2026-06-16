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

## Not provable without the running OpenRAL sim

The camera/`/map`/octomap panels need their real producers (HAL sim sensor
bridge, slam_toolbox, octomap). Those weren't brought up headless. The render
*mechanism* is identical to the verified TF path — a whitelisted topic with a
native schema — so lighting them up is "run the sim", not "more bridge work".
A `cam2image` stand-in could not be remapped onto the camera topic because
this host's miniforge Python 3.13 shadows ROS's 3.12 and breaks ros2cli
`--ros-args` remap forwarding (unrelated to this package).
