# widowx — Trossen WidowX-250s (sim-only)

The sim-pseudo manifest the eval layer loads for the ManiSkill3 / SimplerEnv
WidowX envs only: `python/sim/src/openral_sim/backends/simpler_env.py`
registers the `simpler_env` scene with `fixed_robot="widowx"`; its
`simpler_env/widowx_*` tasks are the BridgeData V2 / SimplerEnv WidowX setup.

The WidowX-250s is a real 5-DoF + gripper desktop arm (Trossen Robotics), but
OpenRAL ships no HAL adapter for it: `hal.sim` and `hal.real` are both null
(scene-only) until a real-HW HAL lands. Kinematic and safety numbers come from
the public Trossen data sheet.

Used by VLA papers: Octo, OpenVLA, π0 (Bridge fragment via SimplerEnv).
See the ManiSkill3 / SimplerEnv scene-adapter integration rationale in `docs/methods/07-eval-sim.md`.
