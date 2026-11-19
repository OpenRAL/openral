# openral_hal_rizon

Real-hardware bringup for two Flexiv Rizon 4s driven as one robot
([`robots/rizon_dual`](../../robots/rizon_dual/README.md)). Bringup-only: the HAL node is
the generic `openral_hal_node`, the Python adapter is the generic
`openral_hal.ros_control.GroupedRosControlHAL` (its `groups` are in the manifest), and the
manifest reaches this package
through `hal.real_bringup: "openral_hal_rizon:real_bringup.launch.py"`.

Everything below the HAL is upstream Flexiv code at the `deps.repos` pins:

```
GroupedRosControlHAL → /left_arm_controller, /right_arm_controller (JointTrajectory)
  → controller_manager (1 kHz)
  → 2 × flexiv_hardware/FlexivHardwareInterface (flexiv_ros2 jazzy-v1.9)
  → flexiv_rdk 1.9 (NRT joint position) → Ethernet → each arm's controller
```

| File | What |
| --- | --- |
| `urdf/rizon_dual.urdf.xacro` | Upstream `flexiv_description`'s `rizon_robot` macro twice on one `world`, plus one `FlexivHardwareInterface` block per arm. Not upstream's `rizon_dual.urdf.xacro`: its `FlexivDualHardwareInterface` needs `flexiv_drdk`, every tag of which requires an RDK newer than the 1.9.0 the arms accept. |
| `config/rizon_dual_controllers.yaml` | Shaped on upstream `flexiv_bringup/config/rizon_dual_controllers.yaml`; controller names are the manifest's `hal.parameters.defaults.groups`. |
| `config/rizon_dual_mounting.yaml` | The reference rig's mounts (`flexiv_description`'s mounting schema). |
| `launch/real_bringup.launch.py` | `robot_state_publisher` + `ros2_control_node` + spawners, behind the host-confined-graph check. |
| `deps.repos`, `provision_flexiv.sh` | Pins and the one-shot build of the upstream stack. |
| `tools/render_rizon_dual_urdf.py` (repo) | Renders `robots/rizon_dual/rizon_dual.urdf` from the xacro (`--check` to verify); the only way to regenerate it. |

## Provisioning (once per host)

Needs ROS 2 Jazzy with `ros-jazzy-ros2-control`, `ros-jazzy-ros2-controllers` and
`python3-vcstool`.

```bash
bash packages/openral_hal_rizon/provision_flexiv.sh ~/flexiv_ws
```

It imports the three upstream repos, runs `flexiv_rdk`'s own
`build_and_install_dependencies_not_in_ros2.sh` (Boost 1.83, SpaceVecAlg, RBDyn), builds
a **private static Fast-DDS 2.14.5 / Fast-CDR 2.2.5** into `~/flexiv_ws/dds_install`,
then `flexiv_rdk` and the three ROS packages. The private DDS is not optional: the vendor
RDK `.a` is compiled against Fast-CDR 2.2.5, and Jazzy's current Fast-CDR (≥ 2.2.6, whose
`Cdr` grew from 216 to 296 bytes) makes the RDK's first message deserialize smash the
stack (`*** stack smashing detected ***` in `RealtimeStatesPubSubType::deserialize`,
reproduced on q5090 with Fast-CDR 2.2.8). The copy is static and hidden, so it stays
inside `libflexiv_hardware.so`.

Each arm must be in **Remote mode (Ethernet)** on its controller. The pinned
`FlexivHardwareInterface` constructs its RDK `Robot` with no network-interface whitelist,
so the RDK discovers the arms on every interface; there is no launch argument to narrow it.

## Run

```bash
source ~/flexiv_ws/install/setup.bash
openral deploy run --config scenes/deploy/rizon_dual_bench.yaml
```

**Activation enables both arms.** `FlexivHardwareInterface::on_activate` clears a fault
and calls `Robot::Enable()` (brakes released, servo on); the trajectory controllers then
hold the measured pose, so nothing moves until a trajectory arrives. Keep each arm's
E-stop in reach. An arm whose E-stop is pressed never becomes operational, and
`on_activate` waits for it.

**Stopping the deploy does not disable the arms.** On teardown the upstream interface
calls `Robot::Stop()`, which halts motion but leaves both arms enabled ("operational /
Ready", brakes released; verified on q5090). Disable them on the controller or press the
E-stops before entering the cell.

The launch defaults are the reference rig's serials; `deploy run` forwards no launch
arguments, so another rig overrides them in the launch or runs it by hand:

```bash
ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST .venv/bin/python "$(command -v ros2)" launch \
  openral_hal_rizon real_bringup.launch.py robot_sn_left:=<sn> robot_sn_right:=<sn>
```
