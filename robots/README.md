# Robot manifests (`robots/`)

One `RobotDescription` manifest (`robot.yaml`) per embodiment: the normative
joint / sensor / capability / safety / HAL contract every other layer (rSkills,
scenes, the eval registry, the HAL itself) validates against. The
`openral_sim.policies.robots` adapter scans `robots/*/robot.yaml` at import
time and auto-registers each one; `$OPENRAL_ROBOTS_DIR/<id>/robot.yaml` is
looked up first, for out-of-tree manifests.

There are 27 robot directories, each with a `robot.yaml` and most with a
`README.md` of provenance notes (kinematic source, MJCF wiring, sim-vs-real
caveats). The inventory (real HAL, sim HAL, deploy scenes, status per robot)
lives in one place: [`docs/reference/robots.md`](../docs/reference/robots.md).

`hal.real` is the source of truth for real-hardware support: `null` or absent
means no HAL and no physical rig. `hal.sim` being `null` does not mean no sim
support: when the manifest has a `sim:` block, `openral_hal.build_hal` derives
`MujocoArmHAL.from_description` from it. Some embodiments (`gr1`, `pusht_2d`,
`sawyer`, `google_robot`) run only through a scene adapter's `fixed_robot=`
binding, some in an out-of-process sidecar.

## Robot manifests name code — load them from trusted sources only

`hal.sim` and `hal.real` are `"module.path:Attribute"` strings that
`openral_hal.build_hal` imports and calls with the manifest's HAL parameters, so a
manifest is code, not just data. Only OpenRAL's own packages are admitted
(top-level module `openral_*`, which includes OpenRAL Pro's `openral_pro_*`),
checked when the manifest loads and again before the import
(`openral_core.code_refs`); anything else is a `ROSConfigError` naming the field.
`OPENRAL_ALLOW_EXTERNAL_PLUGINS=1` admits another module with a warning on every
use. Load a manifest (including one from `$OPENRAL_ROBOTS_DIR` or the Hub) only from
a source you trust.

## Add a new robot

```bash
openral detect         # probe the host and write robots/<name>/robot.yaml directly
```

or author `robots/<name>/robot.yaml` by hand against the `RobotDescription`
schema (`python/core/src/openral_core/schemas.py`), using any manifest above for
the shape (joints, `capabilities.embodiment_tags`, `safety`, `hal.{sim,real}`,
optional `sim:` MuJoCo wiring). A new manifest is auto-registered on next
import, visible to `openral sim list` / `openral rskill check` / the eval
registry with no code change.

A manifest with `hal.real` set must declare every value the real path reads —
the schema refuses it otherwise, naming each gap: `action_spec.control_freq_hz`,
the kernel's `safety` thresholds, `safety.joint_state_staleness_limit_s` (the
HAL's and the runner's joint-state window, measured on the raw stream; measure
it with `tools/joint_state_staleness_probe.py --robot`. A runner that reads the
HAL's rate-limited `~/joint_states` republish gets two republish periods on top,
capped at 0.5 s), and the starting-pose approach
per joint type — `safety.starting_pose_max_joint_speed_rad_s` /
`starting_pose_tolerance_rad` for revolute joints, `..._m_s` / `..._m` when the
robot has a prismatic joint (the runner also caps each joint at its
`velocity_limit`). A value copied from another robot rather than measured on
this one is marked `provisional` in the manifest comment.

Robot manifests are at `schema_version: "0.2"`, and only `"0.2"` loads. There is
no migration path: a `"0.1"` manifest is refused with a `ROSConfigError`. To update
one, move `hal.parameters.defaults.staleness_limit_s` to
`safety.joint_state_staleness_limit_s`, measure and declare every real-hardware
safety field above on the rig, and set `schema_version: "0.2"`.
