# widowx — Trossen WidowX-250s (sim-only)

The sim-pseudo manifest the eval layer loads for the ManiSkill3 / SimplerEnv
WidowX envs only: `python/sim/src/openral_sim/backends/simpler_env.py`
registers the `simpler_env` scene with `fixed_robot="widowx"`; its
`simpler_env/widowx_*` tasks are the BridgeData V2 / SimplerEnv WidowX setup.

The WidowX-250s is a real 6-DoF + gripper desktop arm (Trossen Robotics), but
OpenRAL ships no real HAL for it (`hal.real: null`). Kinematic and safety numbers
come from the public Trossen data sheet.

## MuJoCo twin

`hal.sim: null` plus the manifest's `sim:` block make `build_hal` derive a
`MujocoArmHAL` on the Menagerie MJCF (`rd:widow_mj_description`):

```bash
openral deploy sim --config scenes/deploy/widowx_twin.yaml
```

- **Joints:** the six arm joints map 1:1 to qpos / actuators 0-5; `gripper` is the
  `gripper` actuator on `left_finger` (slide 0.015-0.037 m), mirrored onto
  `right_finger` by the MJCF equality, reported and commanded as 0 = closed, 1 = open
  (`normalized_open_unit`). SimplerEnv scenes override that with their own
  `normalized_open_symmetric`.
- **Start pose:** the MJCF's `home` keyframe (`sim.keyframe_index: 0`). At qpos 0 the
  fingers sit outside their slide ranges and jam.
- **Rate:** `action_spec.control_freq_hz: 5.0`, the BridgeData V2 / ManiSkill3
  bridge env rate. One action is one 0.2 s period.
- **Camera:** `top` is world-fixed in front of the arm (`sim_placement`), a sim
  choice rather than the Bridge camera pose.
- **Actions:** joint-space only. The Bridge / SimplerEnv rSkills emit EEF deltas
  and run on `scenes/sim/widowx_*.yaml`, not on the twin.

Test: `tests/sim/test_widowx_hal_mujoco.py`.

Used by VLA papers: Octo, OpenVLA, π0 (Bridge fragment via SimplerEnv).
See the ManiSkill3 / SimplerEnv scene-adapter integration rationale in `docs/methods/07-eval-sim.md`.
