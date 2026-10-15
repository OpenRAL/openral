# Upstream patches — `openarm_description`

Patches applied to **upstream** `enactic/openarm_description` in the colcon
workspace that runs the real OpenArm, so OpenRAL tracks upstream instead of a
fork. Each is a reviewable diff with a stated upstream status, dropped once
upstream merges it.

They only affect the `robot_description` that `openarm_bringup` generates **at
launch** for real hardware, not `robots/openarm/openarm.urdf` (the flat URDF
vendored for planning and collision checking, expanded with
`mock_components/GenericSystem` and carrying no CAN parameters).

## `0001-v20-forward-can-interface-args.patch`

**Status:** not yet submitted upstream. Drop this file once it merges.

**What it fixes.** On OpenArm v2.0 the `left_can_interface` /
`right_can_interface` launch arguments are silently ignored. The bimanual
ros2_control macro and `openarm_bringup`'s launch file support them end to end,
but the v2.0 *top-level* xacro fails to declare and forward them, so the macro
falls back to its `can1` / `can0` defaults. The v1.0 top-level xacro declares
all three args correctly; this is a v2.0-only omission.

**Why it matters here.** A CAN interface name is a property of the host, not
the robot: this rig's buses are udev-pinned to `openarm_left` / `openarm_right`
and `can1` / `can0` do not exist on it. Unpatched, the failure is silent: the
URDF builds, controllers spawn, `send_action` succeeds, and the arms do not
move. The HAL's CAN preflight misses it too, because `openarm_left` and
`openarm_right` *are* up; the description points elsewhere.

**Reproduction** (pristine upstream, real hardware plugin, requesting this
rig's bus names):

```
plugins      : ['openarm_hardware/OpenArmHW']
can_interface: ['can1', 'can0']          <- requested openarm_left/openarm_right
```

With the patch applied, same invocation:

```
can_interface: ['openarm_left', 'openarm_right']
```

**Blast radius.** Strictly additive — two `<xacro:arg>` declarations and two
forwarding attributes. The defaults stay `can1` / `can0`, so a caller that
passes nothing gets byte-identical output to upstream (verified).

## Applying

```sh
./apply.sh /path/to/colcon_ws/src/openarm_description
```

Idempotent: re-running on an already-patched tree reports it and exits 0.
