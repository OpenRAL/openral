# ROS graph security on a real robot

A ROS 2 graph has no authentication by default: any participant that discovers
it can publish, subscribe, call services and set parameters. On an OpenRAL
deploy that includes commanding the arm.

## Threat

With the ROS defaults (`ROS_DOMAIN_ID` unset = 0, discovery over the subnet),
anyone on the same network as the robot host can:

| Surface | Effect |
| --- | --- |
| publish `ActionChunk` on `/openral/safe_action` | the HAL executes it; the safety kernel is bypassed |
| call `/openral/estop_reset` | clears a latched E-stop once the cooldown has passed |
| `ros2 param set` on the safety kernel | relaxed `joint_velocity_max`, `workspace_box_*`, … would apply on the next configure |
| `ros2 param set /safety_controller blend_ratio` (Unitree G1) | blends the S0 locomotion policy in, up to 1.0 at ≤ 1 /s (`max_blend_ratio_speed` is refused at runtime since #548); see below |
| publish `/openral/prompt` | instructs the reasoner |
| publish `/joint_states`, the deadman heartbeat | the kernel checks against a fake state; a dead operator looks alive |

## What `openral deploy run` does

**Host confinement (default).** A real deploy (`hal_mode=real`) runs with
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and a per-robot `ROS_DOMAIN_ID`,
a stable hash of the `robot_id` into 10–101 (above 101 collides with Linux
ephemeral ports) that never lands on a reserved domain
(`openral_cli._dds_scope.RESERVED_DOMAINS`): 0, 42 (`tools/_demo_env.sh`), 77
(the sim), 50–99 (per-PID test domains, which also cover the lab harnesses'
76/80–92) and 100–101 (package test conftests). That leaves 39 domains, 10–49
minus 42. A hash that lands on a reserved domain is re-hashed into the free
ones; every other `robot_id` keeps the domain it had before the reservations
(`openarm` stays 43; `so101_follower` moved from 52 to 39). The chosen scope is printed
as `ros scope:` before launch, followed by the `export` line another terminal
needs: `openral prompt` and `openral estop reset` run `rclpy` on their own
shell's environment, so they reach the robot only on the same domain and
range. An exported non-empty value of either variable
wins; an exported empty value counts as unset. Simulations are unchanged
(`LOCALHOST`, domain 77). Implementation:
`openral_cli._dds_scope.confine_real_scope`.

**Networked graphs need an explicit opt-in.** If `ROS_STATIC_PEERS` or a Fast DDS
`ROS_DISCOVERY_SERVER` is set, `RMW_IMPLEMENTATION=rmw_zenoh_cpp` (Zenoh reads
neither discovery variable and its router listens on every interface), a DDS
vendor config lists peers (`CYCLONEDDS_URI` with `<Peers>`, a
`FASTRTPS_DEFAULT_PROFILES_FILE` / `FASTDDS_DEFAULT_PROFILES_FILE` with
`initialPeersList` or `discoveryServersList`; a file that cannot be read counts too), or
`ROS_AUTOMATIC_DISCOVERY_RANGE` is anything but `LOCALHOST`, the deploy
refuses with `ROSConfigError` unless `OPENRAL_ALLOW_NETWORK_ROS=1` is set, and
logs `dds_scope.real_robot_graph_networked` as a warning when it is. An unset
range still defaults to `LOCALHOST` when only `ROS_STATIC_PEERS` is exported,
so an opted-in graph reaches the listed peers, not the whole subnet. Opt in
only on a trusted, isolated network (a robot-only VLAN or a direct cable). `OFF` is refused outright: in Jazzy it hides nodes
even from each other on one host, so the robot's lifecycle nodes would never
find one another.

**Kernel limits are fixed after configure.** The safety kernel refuses every
`set_parameters` call once it has left `UNCONFIGURED`, and after a failed configure or any cleanup,
so a peer cannot relax a limit loaded from the manifest. Change a limit in
`robot.yaml` and relaunch.

**A hand-run launch is checked too.** `openral deploy run` sets the defaults
above; a hand-run
`ros2 launch openral_rskill_ros deploy_e2e.launch.py hal_mode:=real` (any
`hal_mode` other than `sim`) gets no
defaults but applies the same decision to the environment it was started with,
and refuses with `ROSConfigError` when the graph would reach other hosts —
including when `ROS_AUTOMATIC_DISCOVERY_RANGE` is unset, which ROS reads as
`SUBNET`. Export `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and a non-zero,
non-reserved `ROS_DOMAIN_ID` first, or set `OPENRAL_ALLOW_NETWORK_ROS=1`.
`ROS_LOCALHOST_ONLY=1` (deprecated) also counts as host-confined, since rcl lets
it override both variables. The shared decision is
`openral_core.ros_graph_scope.require_host_confined_ros_scope`; the launch hook
is `_require_host_confined_real_graph` in `deploy_e2e.launch.py`.

**The deadman watchdog and hardware E-stop: parameters fixed once configured.**
`openral_deadman_watchdog` and `openral_hardware_estop` refuse every
`set_parameters` on an already-declared parameter from the start of their first
`on_configure` on, also after a failed configure and after a cleanup, so a peer
cannot raise `safe_action_deadline_s` / `first_chunk_deadline_s` on a running
cell. A set before the first configure still works (nothing is supervised yet);
see `packages/openral_safety_watchdog/README.md`.

**Unitree G1: the blend parameters.** NVIDIA's `SafetyController`
(`isaac_ros_deploy` f3aa6b89, `deps.repos`) declares `blend_ratio` and
`max_blend_ratio_speed` with `read_only = false`, applies a runtime set on the next tick
(range 0–10 /s), and on activation snaps the applied blend to its target with no ramp.
The G1 bringup runs it as `openral_hal_unitree/GuardedSafetyController` (issue #548),
which refuses to configure with `max_blend_ratio_speed` outside (0, 1.0] /s, refuses
every runtime change to it, and resets `blend_ratio` to 0.0 on activation so the
policy always starts blended out (a refusal there would leave a strict controller swap
half-done, with the joints unclaimed). Its
`out_of_domain_detection.*` parameters already refuse a runtime set, and the rest are
read at configure. `blend_ratio` itself stays writable because it is the operator's
gantry blend-in: a participant that can reach `/safety_controller`'s parameter services
can still blend the policy in, at the operator's ≤ 1 /s ramp. Host confinement is the
control on who that is until SROS2 enclaves land (below); an opted-in network graph
(`OPENRAL_ALLOW_NETWORK_ROS=1`) must be treated as able to start the G1's locomotion
policy.

Host confinement limits *discovery*; it is not a firewall. Combine it with a
host firewall that drops inbound DDS ports (UDP 7400+) on interfaces the robot
does not need.

## Setups that must opt in

Every in-tree real HAL runs on the deploy host (the vendor `ros2_control`
bringup and scene `drivers:` are started inside the same launch), so the
default needs nothing. Opt in when a node of the graph lives on another
computer, for example:

- a camera driver, HAL or `ros2_control` stack on a second machine (a Jetson
  next to the arm, a Thor/Orin cell driven from a workstation);
- `rviz2` / `ros2 topic echo` from a laptop (the Foxglove bridge and the robot
  dashboard are WebSocket/HTTP and need no DDS opt-in).

On **every** host of that graph:

```bash
export ROS_DOMAIN_ID=23                     # same value everywhere; not 0 or a reserved one
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET # or LOCALHOST + ROS_STATIC_PEERS=<ip>
export OPENRAL_ALLOW_NETWORK_ROS=1          # on the host running `openral deploy run`
```

A same-host driver started by hand in another terminal (not from the scene's
`drivers:` block) must use the same scope: read the `ros scope:` line the deploy
printed, or export `ROS_DOMAIN_ID` and `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`
in both terminals.

## Follow-up: SROS2 enclaves

Confinement narrows who can reach the graph; it does not stop a process on the
robot host, or an opted-in peer, from publishing anything. SROS2 (DDS-Security
with per-enclave permissions) is the control for that and is not wired yet.
Recommended permissions profile:

| Enclave | Publish | Subscribe | Services |
| --- | --- | --- | --- |
| `/openral/safety_kernel` | `/openral/safe_action`, `/openral/estop`, `/openral/safety_status`, `/openral/failure/safety` | `/openral/candidate_action`, `/joint_states`, deadman heartbeat, world voxels | serves `/openral/estop_reset` |
| `/openral/hal` | `/joint_states`, HAL diagnostics | `/openral/safe_action` (only the kernel may publish it), `/openral/estop` | — |
| `/openral/runtime` (rSkill) | `/openral/candidate_action` | sensors, `/joint_states` | — |
| `/openral/supervisor`, `/openral/dashboard` | `/openral/prompt`, deadman heartbeat (dashboard) | status topics | **only** callers of `/openral/estop_reset` |
| everything else | its own topics | read-only | none of the above; no parameter services on the kernel or on `/safety_controller` |

Default-deny for every other enclave, `ROS_SECURITY_STRATEGY=Enforce`, keys
provisioned per robot. Until it lands, treat the robot network as part of the
safety boundary.
