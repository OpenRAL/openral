# google_robot — Google Everyday Robot (sim-only)

`RobotDescription` manifest used by:

- `python/sim/src/openral_sim/adapters/maniskill3.py` — registers
  `maniskill3/<env_id>` scenes for ManiSkill3 GoogleRobot tasks.
- `python/sim/src/openral_sim/adapters/simpler_env.py` — registers
  `simpler_env/google_robot_*` scenes for the real-to-sim correlator.

Per ADR-0007 this is a **sim-only pseudo-embodiment** (like `pusht_2d`).
The Everyday Robot platform was Alphabet-internal and discontinued in
2023; the kinematic and safety numbers are coarse approximations, not
from a vendor data sheet. The detailed IO contracts (state shape,
action shape, camera resolution) live in the matching scene adapters
under `python/sim/src/openral_sim/adapters/`.

Used by VLA papers: RT-1, RT-2, Octo, OpenVLA, π0 (SimplerEnv eval
slice). See `docs/adr/0010-maniskill3-simpler-env-backends.md` for
the integration rationale.
