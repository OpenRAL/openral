# widowx — Trossen WidowX-250s (sim-only)

`RobotDescription` manifest used by:

- `python/sim/src/openral_sim/adapters/simpler_env.py` — registers
  `simpler_env/widowx_*` scenes for the BridgeData V2 / SimplerEnv
  WidowX setup.

The WidowX-250s is a real 5-DoF + gripper desktop arm (Trossen
Robotics), but OpenRAL ships no HAL adapter for it today.
This manifest is the sim-pseudo manifest the eval layer loads for
ManiSkill3 / SimplerEnv WidowX envs only.

Kinematic + safety numbers come from the public Trossen data sheet;
when a real-HW HAL lands the manifest will declare `hal.real` (today both
`hal.sim` and `hal.real` are null — scene-only).

Used by VLA papers: Octo, OpenVLA, π0 (Bridge fragment via SimplerEnv).
See `docs/adr/0010-maniskill3-simpler-env-backends.md` for the
integration rationale.
