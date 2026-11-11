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
| publish `/openral/prompt` | instructs the reasoner |
| publish `/joint_states`, the deadman heartbeat | the kernel checks against a fake state; a dead operator looks alive |

## What `openral deploy run` does

**Host confinement (default).** A real deploy (`hal_mode=real`) runs with
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and a per-robot `ROS_DOMAIN_ID`,
a stable hash of the `robot_id` into 10–101 (skipping 77, the sim domain;
above 101 collides with Linux ephemeral ports). The chosen scope is printed
as `ros scope:` before launch. An exported non-empty value of either variable
wins; an exported empty value counts as unset. Simulations are unchanged
(`LOCALHOST`, domain 77). Implementation:
`openral_cli._dds_scope.confine_real_scope`.

**Networked graphs need an explicit opt-in.** If `ROS_STATIC_PEERS` is set, or
`ROS_AUTOMATIC_DISCOVERY_RANGE` is anything but `LOCALHOST` / `OFF`, the deploy
refuses with `ROSConfigError` unless `OPENRAL_ALLOW_NETWORK_ROS=1` is set, and
logs `dds_scope.real_robot_graph_networked` as a warning when it is. Opt in
only on a trusted, isolated network (a robot-only VLAN or a direct cable).

**Kernel limits are fixed after configure.** The safety kernel refuses every
`set_parameters` call once it has left `UNCONFIGURED`, and after any cleanup,
so a peer cannot relax a limit loaded from the manifest. Change a limit in
`robot.yaml` and relaunch.

Only `openral deploy run` applies this. A hand-run
`ros2 launch openral_rskill_ros deploy_e2e.launch.py hal_mode:=real` inherits
your shell's scope as-is: export `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and a
non-zero `ROS_DOMAIN_ID` first. `ROS_LOCALHOST_ONLY=1` (deprecated) also counts
as host-confined, since rcl lets it override both variables.

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
export ROS_DOMAIN_ID=42                     # same value everywhere; not 0
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
| everything else | its own topics | read-only | none of the above; no parameter services on the kernel |

Default-deny for every other enclave, `ROS_SECURITY_STRATEGY=Enforce`, keys
provisioned per robot. Until it lands, treat the robot network as part of the
safety boundary.
