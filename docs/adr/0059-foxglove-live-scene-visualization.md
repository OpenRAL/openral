# ADR-0059: Foxglove as the live-scene visualization surface (read-only, hybrid with OTel)

- Status: **Accepted**
- Date: 2026-06-16
- Related: [ADR-0017](0017-dashboard-otlp-receiver.md) (the `openral dashboard`
  OTLP/HTTP receiver this ADR deliberately keeps for traces/metrics/health — the
  two surfaces are complementary, not a replacement); [ADR-0018](0018-ros2-reasoner-supervisor.md)
  §F7 (the replay correlator — must not regress; the OTel plane is untouched here);
  [ADR-0027](0027-rskill-state-contract-bindings.md) / [ADR-0058](0058-standardized-description-assets.md)
  (the `assets.urdf` → `robot_state_publisher` chain that already puts `/tf` +
  `/robot_description` on the deploy-sim bus, so the Foxglove 3D panel draws the
  robot with no extra wiring); [ADR-0034](0034-deploy-sim-scene-attach-for-arms.md) idle-stepper
  (cameras stream at idle, so panels populate without a running skill);
  [ADR-0048](0048-deploy-sim-clock-publisher.md) (`/clock` → `use_sim_time`
  alignment for Foxglove timestamps); issue #44 (loopback-only dashboard posture);
  CLAUDE.md §1.1 (safety beats helpfulness), §3 Layer 7 (Observability boundary),
  §3 Safety (safety-WG review).

> Investigation (panel-by-panel feasibility, the Bucket-1/2/3 effort split, the
> two-data-plane analysis). Prototype + protocol-level verification: [`packages/openral_foxglove_bringup/`](https://github.com/OpenRAL/openral/tree/master/packages/openral_foxglove_bringup/)
> (`README.md`, `VERIFICATION.md`). This ADR records the decision to graduate
> that spike into a supported, gated component and integrate it into the deploy
> path; the investigation carries the working detail.

## Context

OpenRAL has **two parallel data planes** (feasibility report §TL;DR):

1. A **live ROS 2 plane** — HAL, sensors, SLAM, and the runner publish real ROS
   topics (`sensor_msgs/Image`, `sensor_msgs/JointState`,
   `nav_msgs/OccupancyGrid`, `sensor_msgs/PointCloud2`, `/tf`). Foxglove ingests
   these natively.
2. An **OTLP/OpenTelemetry plane** — the `openral dashboard` (ADR-0017) is *not*
   a ROS subscriber. Bridge nodes re-rasterize ROS data into base64 PNG/JPEG
   inside OTel span attributes, ship them over OTLP/HTTP, and fan them out to a
   vanilla-JS SPA over SSE. This plane carries the dashboard's actual
   differentiator: distributed traces, the replay correlator (ADR-0018 F7),
   OTLP metrics, system health, and the OTel-only reasoner/safety cards.

"Port the whole dashboard to Foxglove" is the wrong framing because the two
planes have opposite answers. The live-scene half (images, 2D nav, voxels, joint
states, TF) is **near-free** in Foxglove — and renders *real* 3D where the
current dashboard flattens to fixed-angle PNGs. The observability half (traces,
metrics, telemetry) has **no native home** in Foxglove, which is a visualization
tool, not an observability backend.

A prototype (`openral_foxglove_bringup`) proved the live-scene half end-to-end:
the bridge delivers real cameras + joints + `/tf` through a **read-only**,
loopback, Bucket-1-allowlisted surface against a live `openral deploy sim` graph,
with the safety/e-stop/action topics provably excluded. It is currently labelled
PROTOTYPE/SPIKE and is not wired into the deploy path. Crossing the Layer 7
boundary and changing how operators view a running graph requires this ADR
(CLAUDE.md §3 / §4) before the spike becomes a supported component.

## Decision

**Adopt a hybrid.** Foxglove (via upstream `foxglove_bridge`) becomes the
supported surface for the **live ROS scene**; the `openral dashboard` OTel
receiver (ADR-0017) **stays** as the surface for traces, metrics, system health,
and the reasoner/safety cards. Neither replaces the other. We explicitly reject a
wholesale dashboard replacement (see Alternatives).

Six binding decisions:

1. **Read-only is normative, not a default.** The bridge advertises capabilities
   `[connectionGraph, assets]` only — `clientPublish`, `services`, `parameters`,
   and `parametersSubscribe` are omitted. A connected viewer **cannot publish a
   topic, call a service, or write a parameter**. The surface MUST NOT be able to
   actuate the robot. Binds to `127.0.0.1` by default (issue #44). Topics are an
   explicit **Bucket-1 allowlist** — the safety/e-stop/action topics
   (`/openral/estop`, `/openral/safe_action`, `/openral/candidate_action`,
   `/openral/failure/*`, the prompt bus) are never matched. These three
   invariants are enforced by hermetic unit tests and are the safety contract of
   this component.

2. **Re-enabling any write capability is out of scope and gated.** An E-stop
   *reset* button, a Publish/Teleop panel, or a prompt-input panel — anything that
   re-enables `clientPublish`/`services` — is **not** part of this ADR and
   requires a separate ADR with safety-WG sign-off and a hazard-log entry
   (CLAUDE.md §3 Safety). Python proposes; C++ disposes — and Foxglove proposes
   nothing.

3. **Integrate into deploy-sim, off by default.** `openral deploy sim` gains
   `--foxglove / --no-foxglove` (default **off**) and `--foxglove-port`
   (default 8765), forwarded as `enable_foxglove` / `foxglove_port` to
   `sim_e2e.launch.py`, which spawns the read-only bridge as part of the runtime
   graph. The bridge node is ordered **after** the topic producers (HAL, SLAM,
   octomap, `robot_state_publisher`) to avoid the foxglove-sdk-cpp v0.18.0
   stale-bridge bug (a channel that appears, loses its publisher, then reappears
   is advertised but never forwarded — `VERIFICATION.md`). Default-off keeps the
   headless/CI boot path and the OTel default untouched.

4. **Compress camera images on the wire.** Raw `sensor_msgs/Image` is ~9 MB/s per
   camera (`VERIFICATION.md`); a multi-camera arm saturates a laptop link and
   Foxglove's send buffer. The bringup gains an opt-in
   `image_transport`/`compressed_image_transport` republish path so camera topics
   travel as `sensor_msgs/CompressedImage`, with the `/compressed` topics added to
   the Bucket-1 allowlist. Compression is opt-in so the raw path stays available
   for fidelity-sensitive use.

5. **Bucket-2 completeness via converter nodes, not per-type extensions.** The
   custom `openral_msgs` types worth seeing in 3D (`WorldCollision` →
   `visualization_msgs/MarkerArray`, `OccupancyVoxels` →
   `sensor_msgs/PointCloud2`/voxel markers, detections → markers) are re-published
   into **standard** ROS visualization types by a small read-only converter node,
   rather than a bespoke TypeScript Foxglove extension per type. This keeps
   Foxglove stock and the conversion logic unit-testable as pure Python. MCAP
   recording (`ros2 bag record -s mcap`, scoped to the Bucket-1 allowlist) is
   offered as an opt-in so a session can be replayed in Foxglove offline.

6. **The OTel plane is untouched.** The replay correlator (ADR-0018 F7), OTLP
   metrics, system-health gauges, and the reasoner/safety cards stay on the
   dashboard. No bridge node is removed; no transport contract on the OTel side
   changes. Foxglove taps the live ROS topics directly (the rasterizing bridges
   prove those topics exist), so adding it costs the OTel plane nothing.

## Phasing

| Phase | Deliverable | Surface |
|---|---|---|
| 1 | `--foxglove` deploy-sim integration + lifecycle ordering | `sim_e2e.launch.py`, `deploy_sim.py` |
| 2 | Compressed-image transport for camera topics | `openral_foxglove_bringup` launch + allowlist |
| 3 | Bucket-2 converter node (custom msgs → standard viz types) | `openral_foxglove_bringup` |
| 4 | MCAP recording + expanded Foxglove layout | `openral_foxglove_bringup` |

Phases are independent and ship behind their own opt-in flags; none re-enables a
write capability.

## Consequences

**Positive.** Operators get real 3D of the live scene (better than the PNG
flatten), MCAP recording for free, and a mobile/tablet client, with no bespoke JS
to maintain. The integration is one opt-in flag on a command operators already
run. The read-only posture means the new surface adds **no** actuation attack
surface.

**Negative / costs.** A second viz surface to document and keep current (mitigated
by the clear split: Foxglove = live scene, dashboard = traces/metrics). The
converter node duplicates a little geometry already done in the rasterizing
bridges (accepted — it targets standard types Foxglove renders natively).
`foxglove_bridge` is an added `exec_depend` (Apache-2.0, upstream-maintained).
The stale-bridge bug requires launch ordering discipline (Phase 1 handles it; also
worth an upstream report).

**Neutral.** Default-off means existing deploy-sim behaviour, CI, and the OTel
default are unchanged until an operator opts in.

## Alternatives considered

- **Replace the dashboard wholesale with Foxglove.** Rejected: a third of the
  dashboard is the OTel/OTLP plane (traces, metrics, replay correlator, health),
  which Foxglove has no model for. This would be a rewrite-and-regress, not a port
  (feasibility report §"wrong framing").
- **Per-type TypeScript Foxglove extensions for the custom msgs.** Rejected for
  Bucket-2: a converter node into standard types is cheaper, keeps Foxglove stock,
  and the conversion stays unit-testable in Python (feasibility report §Bucket 2,
  option b).
- **Expose everything read-only (`['.*']`) for convenience.** Rejected: the
  allowlist is a safety control — it is what keeps the e-stop/action topics off the
  wire. The escape hatch (`expose_all_topics`) exists for debugging only and is
  documented as never-on-a-shared-run.

## Safety

This component touches the Observability layer (Layer 7) and is operator-facing,
so the safety posture is explicit and test-enforced:

- **Read-only capabilities** (`[connectionGraph, assets]`) — no client publish, no
  service call, no parameter write. Verified by `test_capabilities_are_read_only`.
- **Bucket-1 allowlist** — safety/e-stop/action topics are never exposed. Verified
  by `test_safety_topics_not_whitelisted` against the real forbidden-topic set.
- **Loopback default** (`127.0.0.1`) — a non-loopback bind is an explicit operator
  opt-in for a trusted LAN; the bridge has no auth/TLS, so a public bind is
  documented as prohibited.

The **read-only** scope recorded here is accepted with safety-WG sign-off: the
surface is view-only and the robot cannot be actuated through it. Any future work
that re-enables a write capability (E-stop reset, Publish/Teleop, prompt input)
is a separate decision requiring its own safety-WG sign-off and a hazard-log
entry — it is not authorized by this ADR.
