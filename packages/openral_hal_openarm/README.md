# openral_hal_openarm

Real-hardware bringup for the Enactic **OpenArm v2** 16-DoF bimanual arm —
bringup only. The OpenArm HAL lifecycle node is the generic
[`openral_hal_node`](../openral_hal_node/README.md) (sim: `hal.sim: null` →
`MujocoArmHAL.from_description`; real: `openral_hal.openarm_real:OpenArmRealHAL`),
spawned by `openral deploy sim|run` under the node name `openral_hal_openarm`.
Lifecycle coverage in `tests/integration/test_openarm_hal_lifecycle.py` and
`packages/openral_hal_node/test/test_lifecycle_node.py`.

This package ships what that node cannot: `launch/real_bringup.launch.py`,
`deps.repos` and `patches/` for the vendor `ros2_control` stack.
`robots/openarm/robot.yaml` names the launch through
`hal.real_bringup: "openral_hal_openarm:real_bringup.launch.py"`.

The real arm is reached through `openarm_bringup`'s own `ros2_control` stack:
on hardware the `controller_manager` and the C++ `openarm_hardware`
SystemInterface own the 750 Hz loop (`update_rate: 750` in `openarm_bimanual_controllers.yaml` at the pinned SHA), and the HAL node only publishes to their
controllers.

## Real-hardware bringup

`OpenArmRealHAL` only publishes to four already-running `ros2_control`
controllers — it never starts `controller_manager` itself (same pattern as
`FrankaPandaRealHAL` and `franka_ros2`). Something still has to start that
graph on the real CAN bus before the HAL can move anything, and until this
launch file existed nothing in this repo did.

`launch/real_bringup.launch.py` is that something: a thin include of
upstream `openarm_bringup`'s `openarm.bimanual.launch.py` with
`use_fake_hardware:=false` and this HAL's own CAN interface defaults
(`openarm_left` / `openarm_right`, matching `OpenArmRealHAL`'s
`_LEFT_CAN_INTERFACE` / `_RIGHT_CAN_INTERFACE`).

One-time host provisioning — build `openarm_bringup` plus the CAN-fixed
`openarm_description` into a colcon workspace:

```bash
vcs import ros2_ws/src < packages/openral_hal_openarm/deps.repos
git -C ros2_ws/src/openarm_description apply \
  <repo>/packages/openral_hal_openarm/patches/0001-openarm-v20-forward-can-interface-args.patch
# CLI11 is an undeclared dependency of openarm_can, and colcon cannot order it
# (openarm_can declares no dependency on it), so build it first:
git clone --branch v2.4.2 --depth 1 https://github.com/CLIUtils/CLI11.git ros2_ws/src/CLI11
colcon build --packages-select CLI11
colcon build   # openarm_hardware is a runtime pluginlib dep — build everything
```

`deps.repos` pins the `openarm_ros2` / `openarm_description` / `openarm_can`
SHAs validated together on real hardware. The patch fixes an upstream xacro bug
that silently drops the `*_can_interface` launch arguments and falls back to
`can0`/`can1` — on a host where those names exist but are the wrong bus, the
hardware plugin activates cleanly and publishes all-zero `/joint_states` with no
error anywhere in the ROS graph. The patch header carries the upstream
permalinks and the exact pinned base; drop it once the fix lands upstream.

Also run the [OpenArm CAN setup](https://docs.openarm.dev/setup/openarm-setup/can-setup),
naming the two interfaces `openarm_left` / `openarm_right` via a udev `dev_id`
rule to match the defaults above — or pass `left_can_interface:=` /
`right_can_interface:=` overrides.

Then, with people clear of the arms (activation enables the motors and returns
them toward zero), `openral deploy run` starts this graph itself:

```bash
openral deploy run --config scenes/deploy/<your-cell>.yaml
```

The committed real-hardware scene is `scenes/deploy/openarm_bench.yaml`, written
for one bench cell (CAN interface names, camera udev paths); another cell adapts
it. The cameras are robot cameras, so their real-hardware `deploy_binding`s live
in `robots/openarm/robot.yaml`, not in the scene — a deploy scene never touches a
robot camera, and only a workcell camera (its own name) goes under `sensors:`.
A vendor camera driver like `zed_wrapper` is started by the scene's `drivers:`
block, covered by `tests/unit/test_scene_drivers.py`.

`deploy_e2e.launch.py` includes `launch/real_bringup.launch.py` whenever
`hal_mode:=real`, because the manifest declares it as `hal.real_bringup`. Do not
also launch it by hand:
a second copy puts a second `/joint_states` publisher on the graph, and
`openral deploy run` then refuses to start at all (the shared-graph guard in
`openral_cli._dds_scope`, #227).

To bring the controllers up on their own — verifying the CAN wiring before
involving the rest of the stack — the launch file is still directly runnable:

```bash
ros2 launch openral_hal_openarm real_bringup.launch.py
```

Verify before trusting `/joint_states`. CAN traffic and controller "active"
state are each necessary but not sufficient: `OpenArmHW::read()` returns `OK`
even when no motor replied.

```bash
ros2 control list_controllers   # joint_state_broadcaster + all 4 controllers active
ros2 topic hz /joint_states     # ~750 Hz
ip -statistics link show openarm_left    # RX/TX climbing
ip -statistics link show openarm_right
```

A per-second `controller_manager` overrun warning at 750 Hz is expected on a USB
CAN-FD adapter (its ~1.1 ms read round-trip eats the 1.33 ms budget; effective
rate ~600 Hz).

The generic HAL node runs with `hal_mode:=real` in the same graph. It
attaches its `RosControlTransport` automatically and leaves the global
`/joint_states` to the controller's own `joint_state_broadcaster`. It starts
concurrently with the bringup rather than after it, so while the controllers
spawn it logs `read_state failed: Joint state is N s old` for a few seconds and
then recovers on its own — warn-only, and not a fault.

### Checking the transport without the arm

`tests/sim/test_openarm_hal_ros2_control.py` stands the same stack up over
`mock_components/GenericSystem` — real `controller_manager`, real
`joint_trajectory_controller`, real `joint_state_broadcaster`, fake motors — and
drives it through the production `RosControlHAL` + `RosControlTransport`. Run it
before touching the rig: it catches a mistyped command topic, a wrong
`ControllerKind` or a joint-name mismatch in seconds, with no power on the arm.
It needs only `ros-$ROS_DISTRO-ros2-control` and
`ros-$ROS_DISTRO-ros2-controllers`, not the vendor stack, and skips where those
are absent. What it cannot check is anything physical — following error, CAN
timing, bus health — which is why the checks above still matter.
