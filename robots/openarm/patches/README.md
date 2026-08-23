# Upstream patches — `openarm_description`

Patches applied to **upstream** `enactic/openarm_description` in the colcon
workspace that runs the real OpenArm. They exist so OpenRAL can track upstream
directly instead of maintaining a fork: each one is a reviewable diff with a
stated upstream status, and each disappears the moment upstream merges it.

Nothing here affects `robots/openarm/openarm.urdf`, the flat URDF this repo
vendors for planning and collision checking. That file is expanded with
`mock_components/GenericSystem` and carries no CAN parameters at all. These
patches only affect the `robot_description` that `openarm_bringup` generates
**at launch** for real hardware.

## `0001-v20-forward-can-interface-args.patch`

**Status:** not yet submitted upstream. Drop this file once it merges.

**What it fixes.** On OpenArm v2.0, the `left_can_interface` /
`right_can_interface` launch arguments are silently ignored. Both the
bimanual ros2_control macro and `openarm_bringup`'s launch file already
support them end to end — only the v2.0 *top-level* xacro fails to declare and
forward them, so the values land on nothing and the macro falls back to its
`can1` / `can0` defaults. The v1.0 top-level xacro declares all three args
correctly; this is a v2.0-only omission.

**Why it matters here.** A CAN interface name is a property of the host, not
the robot: this rig's buses are udev-pinned to `openarm_left` / `openarm_right`
and `can1` / `can0` do not exist on it. Unpatched, the failure is silent in the
worst way — the URDF builds, controllers spawn, `send_action` succeeds, and the
arms do not move. The HAL's CAN preflight does not catch it either, because
`openarm_left` and `openarm_right` genuinely *are* up; it is the description
pointing somewhere else.

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
