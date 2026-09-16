# google_robot — Google Everyday Robot (sim-only)

`RobotDescription` manifest for this embodiment. **No working sim path exists
today**: `python/sim/src/openral_sim/backends/maniskill3.py` is a generic
`maniskill3/<env_id>` pass-through and has no GoogleRobot-specific task
registration, and no `scenes/` YAML binds one. `simpler_env.py`'s
`google_robot_*` friendly names exist in `ENVIRONMENT_MAP`, but their MS3
envs (`GraspSingleOpenedCokeCanInScene`, `MoveNearGoogleBakedTexInScene`, …)
are not registered upstream, so they raise `NameNotFound` — the
`simpler_env_google_robot` benchmark suite was removed for that reason (see
`docs/methods/07-eval-sim.md`). Only the WidowX bridge tasks are wired
end-to-end through `simpler_env.py` today.

Per the robot/sim split convention, this is a **sim-only pseudo-embodiment** (like `pusht_2d`).
The Everyday Robot platform was Alphabet-internal and discontinued in
2023; the kinematic and safety numbers are coarse approximations, not
from a vendor data sheet. The detailed IO contracts (state shape,
action shape, camera resolution) live in the matching scene adapters
under `python/sim/src/openral_sim/backends/`.

Used by VLA papers: RT-1, RT-2, Octo, OpenVLA, π0 (SimplerEnv eval
slice). See the ManiSkill3 / SimplerEnv scene-adapter integration rationale in `docs/methods/07-eval-sim.md`.
