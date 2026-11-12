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
range. Pass them `--robot <robot_id>` instead and they derive that scope
themselves with the same `confine_real_scope` (`openral_cli._dds_scope.join_robot_scope`,
#557), printing the domain they used and where it came from; an exported
`ROS_DOMAIN_ID` still wins there too, and a mismatch with the robot's domain is
named in that line. They then act only if `openral_hal_<robot_id>` is on the
joined graph (`require_robot_on_graph`, exit 3 otherwise), because 39 free
domains are shared by every robot id: `openarm`, `makermods_metal` and
`openamrobot_v2` all derive 43. The deploy prints the `--robot` hint only when
its graph is on the derived domain with LOCALHOST discovery. An exported non-empty `ROS_DOMAIN_ID` wins; an exported empty value of
either variable counts as unset. Simulations get the same range rule, on domain 77
(`openral_cli._dds_scope.confine_sim_scope`). Implementation:
`openral_cli._dds_scope.confine_real_scope`.

**An exported `SUBNET` is ROS's default, not a choice.** `source
/opt/ros/jazzy/setup.bash` exports `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`
whenever it is unset (`ros_environment`'s `1.ros_discovery.dsv` is
`set-if-unset;ROS_AUTOMATIC_DISCOVERY_RANGE;SUBNET`), so every freshly sourced
shell carries it. Until 2026-11-12 the deploy read it as an operator's choice:
`deploy sim` ran on the LAN and the host confinement of #227 never applied, and
`deploy run` refused a real robot as networked unless `LOCALHOST` was exported
by hand. Both now treat `SUBNET` like an unset range — replaced by `LOCALHOST` —
unless `OPENRAL_ALLOW_NETWORK_ROS=1` says the LAN is wanted. Any other exported
range (`LOCALHOST`, `OFF`, `SYSTEM_DEFAULT`) is one ROS never sets, so it is the
operator's and wins as before. The decision is
`openral_core.ros_graph_scope.discovery_range_is_ros_default`.

**Networked graphs need an explicit opt-in.** If `ROS_STATIC_PEERS` or a Fast DDS
`ROS_DISCOVERY_SERVER` is set, `RMW_IMPLEMENTATION=rmw_zenoh_cpp` (Zenoh reads
neither discovery variable and its router listens on every interface), a DDS
vendor config lists peers (`CYCLONEDDS_URI` with `<Peers>`, a
`FASTRTPS_DEFAULT_PROFILES_FILE` / `FASTDDS_DEFAULT_PROFILES_FILE` with
`initialPeersList` or `discoveryServersList`; a file that cannot be read counts too), or
`ROS_AUTOMATIC_DISCOVERY_RANGE` is anything but `LOCALHOST`, the deploy
refuses with `ROSConfigError` unless `OPENRAL_ALLOW_NETWORK_ROS=1` is set **and** the
graph runs under SROS2 in Enforce mode ([below](#sros2-enclaves-adr-0130)); when both
hold, `deploy_e2e.launch.py` logs one `deploy_e2e.real_robot_graph_networked` warning
per deploy (hand-run or `deploy run`). An unset
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
`SUBNET`, and when it is the `SUBNET` that `setup.bash` exports. The launch
checks rather than defaults: its own ROS node and every node it starts inherit
the environment `ros2 launch` was started with. Export `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and a non-zero,
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
can still blend the policy in, at the operator's ≤ 1 /s ramp. Under SROS2
([below](#sros2-enclaves-adr-0130)) only the operator enclave may set it; without
SROS2, host confinement is the only control on who that is, and a networked graph
cannot start without SROS2.

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

Run `openral security init <robot_id>` on the robot host and copy the keystore
(see [what each host needs](#set-it-up)). Then, on **every** host of that graph:

```bash
export ROS_DOMAIN_ID=23                     # same value everywhere; not 0 or a reserved one
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET # or LOCALHOST + ROS_STATIC_PEERS=<ip>
export ROS_SECURITY_ENABLE=true ROS_SECURITY_STRATEGY=Enforce ROS_SECURITY_KEYSTORE=<keystore>
export OPENRAL_ALLOW_NETWORK_ROS=1          # on the host running `openral deploy run`
                                            # (without it, deploy run/sim replace SUBNET
                                            # with LOCALHOST: setup.bash exports SUBNET)
```

A same-host driver started by hand in another terminal (not from the scene's
`drivers:` block) must use the same scope: read the `ros scope:` line the deploy
printed, or export `ROS_DOMAIN_ID` and `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`
in both terminals. Under SROS2 it must also run in the HAL enclave
(`export ROS_SECURITY_ENCLAVE_OVERRIDE=/openral/hal`): a driver in the default
enclave is refused publishing joint states and controller commands.

## SROS2 enclaves (ADR-0130)

Confinement limits who can reach the graph. SROS2 (DDS-Security) controls what a
participant that reaches it may do: every participant authenticates with an enclave's certificate and
is held to that enclave's signed permissions. **A networked real cell must run
under it**: with `OPENRAL_ALLOW_NETWORK_ROS=1`, `openral deploy run` and a hand-run
real launch also require `ROS_SECURITY_ENABLE=true`, `ROS_SECURITY_STRATEGY=Enforce`
and a keystore that holds the five OpenRAL enclaves for the graph's
`ROS_DOMAIN_ID`. There is no flag around it. A host-confined cell and a sim may
run secured too; whenever `ROS_SECURITY_ENABLE=true` is exported the keystore is
checked before launch (`openral_core.ros_security.require_secure_ros_graph`): every
enclave present, the signed governance admits the domain, and the signed permissions
really deny the protected surfaces (an allow-everything keystore made with plain
`ros2 security create_enclave` is refused).

### Set it up

```bash
openral security init <robot_id>   # prints the exports below
export ROS_DOMAIN_ID=<id> ROS_SECURITY_KEYSTORE=~/.local/share/openral/sros2/<robot_id> \
       ROS_SECURITY_ENABLE=true ROS_SECURITY_STRATEGY=Enforce
openral deploy run --config <scene>.yaml
```

`openral security init` renders the robot's policy
(`openral_core.ros_security.render_sros2_policy`, kept as `policy.xml` in the
keystore), runs `ros2 security generate_artifacts` for the deploy's domain (the
exported `ROS_DOMAIN_ID`, else the robot's hashed one; `--domain-id` overrides,
77 for a sim), and makes the private keys owner-only. `--force` regenerates.
Needs a sourced ROS 2 with `sros2` and `openssl`.

A second host joins with the same keystore and exports. Give it only what it
needs: a laptop running `rviz2` needs the default enclave (`enclaves/` files plus
`public/`), never `private/` or another enclave's `key.pem`.

### Enclaves

| Enclave | Who | Denied |
| --- | --- | --- |
| `/openral/safety_kernel` | the C++ kernel | nothing |
| `/openral/hal` | the HAL, the vendor `ros2_control` bringup, scene `drivers:` | publishing `safe_action` / `estop_cleared`; calling `estop_reset`; lifecycle and parameter writes on protected nodes |
| `/openral/launch` | `ros2 launch` and `tools/lifecycle_autostart.py` | publishing command and state topics; the trajectory action |
| `/openral/operator` | the dashboard, `openral estop reset`, `openral prompt` | publishing `safe_action`, controller commands and state topics; the trajectory action; `/controller_manager/*`; lifecycle and parameter writes on protected nodes except `/safety_controller` parameters (the G1 blend-in) |
| `/` (default) | everything else, including any peer given the default key | all of the above |

- **Command topics:** `/openral/safe_action`, `/openral/estop_cleared`, `/*/commands`,
  `/*/joint_trajectory`, the action `/*/follow_joint_trajectory`, and the manifest's
  `onboard_compute.command_topic` / `gripper_command_topic`.
- **State topics:** `/joint_states`, `/openral_hal_*/joint_states`, and
  `onboard_compute.joint_state_topic`.
- **Protected nodes:** `openral_safety_kernel`, `openral_hal_*`,
  `openral_deadman_watchdog`, `openral_hardware_estop`,
  `openral_human_estop_forwarder`, `openral_compliance_recorder` and `safety_controller`
  (`change_state`, `set_parameters`, `set_parameters_atomically`).

Every enclave may subscribe to every topic and publish `/openral/estop`.
**Serving** is denied where a fake server could answer the real one's callers
first: only the kernel may serve `/openral/estop_reset` (a fake "success" would make
`openral estop reset` broadcast `estop_cleared` while the kernel stays latched), and
only the HAL and kernel enclaves may serve `/controller_manager/*` and the
trajectory action. A protected node's own `change_state` / `set_parameters` stay
servable wherever it runs.

The reasoner runs in `/`, so under SROS2 it cannot cycle the kernel, the HAL or the
E-stop producers; such a `LifecycleTransitionTool` is logged as refused.

**How a process gets its enclave.** `ROS_SECURITY_ENCLAVE_OVERRIDE`, per process.
`openral deploy` starts `ros2 launch` in `/openral/launch`. The launch creates its own
ROS node, then hands its children `/` and names the kernel, HAL and dashboard
explicitly. The vendor bringup and scene drivers are wrapped in a scope that sets
`/openral/hal`. That scope covers what starts while the include is visited; a
bringup that defers a process (the Unitree `switch_controllers` step) pins the
enclave on it itself. A hand-run secured launch must export
`ROS_SECURITY_ENCLAVE_OVERRIDE=/openral/launch` or it is refused.

**Cyclone DDS quirk.** Cyclone DDS 0.10 (Jazzy's default) refuses to create a topic
that any deny rule names, even a publish-only rule: the HAL could not subscribe to
`safe_action` and the watchdog could not serve its own `set_parameters`. Fast DDS
does not have this problem. The keystore therefore lists an allow rule for the
opposite direction ahead of each one-sided deny (never for a surface denied in
both directions, so it grants no serving) and re-signs the permissions
(`openral_cli.security._allow_the_other_side`). The live test
`tests/integration/test_sros2_enclaves_live.py` checks both directions per enclave, on
Cyclone and Fast DDS.

**Measured cost** (dev host, Cyclone): 1280×720 RGB at 30 Hz was received at 24.0 Hz
under encryption vs 26.6 Hz without, with 18 % more subscriber CPU.

### What it does not cover

- **Same-user key access.** Keys are files: a process running as the deploy user can
  read any enclave's key. Keep the keystore on the robot, owner-only, and give
  remote hosts only the default enclave.
- **Unlisted topics.** `/openral/prompt*`, perception topics and `/cmd_vel` (Nav2
  drives a base through it, ADR-0127) stay writable by every enclave.
- **Vendor topics outside the conventions.** Declare a HAL's controller topic under
  `onboard_compute` (`command_topic`, `gripper_command_topic`, `joint_state_topic`)
  so the policy names it, then re-run `openral security init --force`.
- **Rotation.** sros2 certificates last ten years and cannot be revoked; regenerate
  and redistribute to rotate.
- **Zenoh.** `rmw_zenoh_cpp` stays refused on a networked real cell.
