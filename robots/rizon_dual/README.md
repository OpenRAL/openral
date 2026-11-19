# `rizon_dual` — two Flexiv Rizon 4s as one robot

Two Flexiv Rizon 4s arms on one column, driven as a single 14-DoF embodiment (left
joints 0–6, right 7–13). Reference rig: the q5090 cell (left arm `Rizon4s-063642` on
the 180°-rolled mount, right arm `Rizon4s-063618`, robot software v3.11).

| Field | Value |
| --- | --- |
| `hal.real` | `openral_hal.ros_control:GroupedRosControlHAL`, config-only: `hal.parameters.defaults.groups` names one `JointTrajectoryController` per arm |
| `hal.real_bringup` | `openral_hal_rizon:real_bringup.launch.py` |
| `hal.sim` | none — no `sim:` block (real-only for now) |
| Deploy scene | `scenes/deploy/rizon_dual_bench.yaml` |
| `sdk_kind` | `closed_with_api` (the RDK's core is a vendor-precompiled `.a`) |

## Where the numbers come from

- **Kinematics, meshes, limits:** upstream
  [`flexiv_description`](https://github.com/flexivrobotics/flexiv_description) `jazzy-v1.9`
  (`Rizon4s`), composed by `packages/openral_hal_rizon/urdf/rizon_dual.urdf.xacro`.
  `rizon_dual.urdf` is that xacro rendered with `ros2_control:=false` by
  `python3 tools/render_rizon_dual_urdf.py` (with the provisioned
  `~/flexiv_ws` sourced): meshes become `rd:rizon4_description` refs (`robot_descriptions`'
  Flexiv clone, byte-identical for Rizon4s), and each link's bare `origin`/`mass`/`inertia`
  is wrapped in `<inertial>` (flexiv_description jazzy-v1.9's `get_inertias` emits them
  unwrapped, which URDF parsers ignore). Re-render after changing the xacro or the
  mounting (`--check` verifies), then re-lower the collision model; never hand-edit.
- **Position limits:** Flexiv's *soft* limits (the RDK refuses targets past them), and
  joint 7 held to home ±90° on the reference rig, where the wrist cabling winds past that.
- **Mounting:** `packages/openral_hal_rizon/config/rizon_dual_mounting.yaml` (measured on
  the rig); `fixed_attachments` mirror it for the safety kernel.
- **Collision model:** `openral collision lower --robot robots/rizon_dual/robot.yaml
  --write` (capsules from the URDF collision meshes; adjacent-link ACM only).
  Safety-WG review owed.
- **Rest pose:** `rizon_dual.srdf`'s `home` group_state, the rig's measured home (joint 7's
  limits exclude `q = 0`); the zero-pose collision test checks it.
- **Safety thresholds:** provisional (none measured on the rig yet), under the 1 rad/s
  the reference rig's teleop sessions ran at.

## Not covered

- **Jaws.** The rig's Damiao servo jaws (CAN, `openarm_left`/`openarm_right`) have no
  upstream ros2_control driver, so they are not in this embodiment.
- **Cameras.** None declared.
- **In-place e-stop re-arm.** `estop_recovery` stays `RESTART_REQUIRED`.
- **Disable on teardown.** Stopping the deploy leaves both arms enabled (upstream
  `Robot::Stop()` does not disable them): disable them or press the E-stops before
  entering the cell.

## Bringup

See [`packages/openral_hal_rizon/README.md`](../../packages/openral_hal_rizon/README.md).
