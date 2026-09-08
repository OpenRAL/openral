# openral_hal_openarm

ROS 2 lifecycle-node wrapper around `openral_hal.OpenArmMujocoHAL` so the
Enactic **OpenArm v2** 16-DoF bimanual arm can participate in the
`openral deploy sim` graph (`sim_e2e.launch.py` → C++ safety kernel → HAL).

Spawned by `openral deploy sim --robot openarm` via
`_ROBOT_HAL_REGISTRY["openarm"]` (see
`python/cli/src/openral_cli/deploy_sim.py`). Subscribes `/openral/safe_action`
+ `/openral/estop`, publishes `/joint_states`, and — under sim
scene-attach — `/openral/cameras/*` + the MuJoCo viewer.

This node wraps the MuJoCo twin; `HAL.connect()` resolves the MJCF on first
use. Lifecycle coverage in `tests/integration/test_openarm_hal_lifecycle.py`.

The real arm is reached a different way — `openral_hal.openarm_real:OpenArmRealHAL`
(the manifest's `hal.real`) commands `openarm_bringup`'s own `ros2_control`
stack, so on hardware the `controller_manager` and the C++ `openarm_hardware`
SystemInterface own the 400 Hz loop and this node is not in the path.

## Real-hardware bringup

`OpenArmRealHAL` only publishes to four already-running `ros2_control`
controllers — it never starts `controller_manager` itself (same pattern as
`FrankaPandaRealHAL` and `franka_ros2`). Something still has to start that
graph on the real CAN bus before the HAL can move anything, and until this
launch file existed nothing in this repo did.

`launch/openarm_real_bringup.launch.py` is that something: a thin include of
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
them toward zero):

```bash
ros2 launch openral_hal_openarm openarm_real_bringup.launch.py
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

Only then start this package's lifecycle node with `hal_mode:=real`. It attaches
its `RosControlTransport` automatically and leaves the global `/joint_states` to
the controller's own `joint_state_broadcaster`.

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
