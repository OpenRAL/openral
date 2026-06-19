# ADR-0014: ManiSkill3 and SimplerEnv as opt-in sim backends

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)
- Related: ADR-0002 (eval/sim environments), ADR-0007 (robot/sim split),
  ADR-0009 (separate sim from benchmarking)

## Context

After ADR-0009 PR D landed, `openral benchmark run` could drive any
`BenchmarkSpec` end-to-end against any registered `(scene, robot, vla)`
triple. The catalogue under `benchmarks/` then grew to cover LIBERO
(spatial/object/goal/10), MetaWorld MT50, gym-aloha (cube + insertion),
and gym-pusht — every sim backend we already had wired.

The next question is: **which sim backend do we add next, and why?**
The VLA literature has converged on a small number of standard evals:

| Backend | License | Install footprint | API style | GPU parallel? | Citation share |
|---|---|---|---|---|---|
| **ManiSkill3** (Tao et al., RSS '25) | Apache-2.0 | `pip install mani-skill` (PyPI) | gym/gymnasium | Yes (SAPIEN, up to 30k FPS) | Rapidly growing |
| **SimplerEnv** (Li et al., CoRL '24) | MIT | source install on top of ManiSkill | gym-style | Inherits | The metric VLA papers actually report (MMRV / real-sim Pearson) |
| RoboCasa | MIT code + CC-BY assets | source, ~10 GB assets, robosuite-master pin | robosuite-native | No | Strong for kitchen tasks |
| CALVIN | MIT | py3.8 only | gym | No | Declining |
| RLBench | MIT code + closed CoppeliaSim | hard | gym | No | Declining |
| VLABench | Apache-2.0 | source, large asset set | MuJoCo | No | Emerging |

The two clear "next" choices are **ManiSkill3** and **SimplerEnv**:

- ManiSkill3 ships as `pip install mani-skill` on PyPI, Apache-2.0,
  Gymnasium-native, GPU-parallel via SAPIEN. No version conflict with
  the existing LIBERO/MetaWorld stack — SAPIEN and MuJoCo live in
  separate process trees. Ships VLA baselines (Octo, RDT-1B, RT-X) so
  policy adapters land cheap.
- SimplerEnv is the only widely-cited benchmark explicitly designed as
  a **real-to-sim correlator** (MMRV, Pearson) — it answers "does my
  sim number predict the real-robot number?" rather than "does it
  match the leaderboard?". That is the metric the OpenVLA / π0 / Octo
  papers report. It depends on ManiSkill (originally MS2, increasingly
  MS3 via the `maniskill3` branch), so it rides on top of the same
  install we add for ManiSkill3.

RoboCasa is deferred. The robosuite-1.5 pin conflicts with LIBERO's
robosuite-1.4 in the same Python environment, and the 10 GB asset
download needs an opt-in fetcher. Worth doing later as a separate
extras-group + ADR; not a fit for "one extras group, two new backends".

## Decision

Add ManiSkill3 and SimplerEnv as two new sim backends, both **opt-in**
via uv extras groups so the default `uv sync` does not pull SAPIEN.

1. **Two extras groups in `pyproject.toml`** under `[project.optional-dependencies]`:
   - `maniskill3 = ["mani-skill>=3.0.0b9"]`
   - `simpler-env = ["mani-skill>=3.0.0b9", "simpler-env"]` (note:
     `simpler-env` is not on PyPI today; the dep entry uses the git
     URL of the upstream `maniskill3` branch).
2. **Two scene adapters under `python/sim/src/openral_sim/{policies,backends}/`**:
   - `maniskill3.py` registers scene IDs of the form
     `"maniskill3/<task_id>"` (e.g. `maniskill3/PickCube-v1`).
     Lazily imports `mani_skill.envs` so the absence of the extra
     never breaks import of `openral_sim`.
   - `simpler_env.py` registers scene IDs of the form
     `"simpler_env/<task_id>"` (e.g.
     `simpler_env/google_robot_pick_coke_can`). Same lazy-import
     discipline.
3. **Two new `RobotDescription` manifests** under `robots/`:
   - `robots/google_robot/robot.yaml` — RT-1 / RT-2 / Octo target.
   - `robots/widowx/robot.yaml` — Bridge-Data target (used by both
     ManiSkill3 and SimplerEnv).
4. **Two new benchmark YAMLs** under `benchmarks/`:
   - `benchmarks/maniskill3_pick_place.yaml` (small PickCube + StackCube
     sample — 2 tasks × 5 seeds = 10 rollouts; deliberately tiny so it
     completes in a CI gate once the extras land).
   - `benchmarks/simpler_env_google_robot.yaml` (the four canonical
     Google-robot tasks).
5. **Schema tests** under `tests/unit/test_benchmark_schemas.py` follow
   the existing parametrised pattern so a typo in either new YAML fails
   loud at test time without requiring the heavy extras.
6. **Sim tests** under `tests/sim/test_<robot>_<vla>_<sim>.py` follow the
   `pytest.importorskip` pattern so they run on hosts that have the
   extras installed and skip cleanly elsewhere.
7. **Docs**: `docs/METHODS.md` gets the two new adapters; the repo
   state map (`docs/architecture/repo-state-map.html`) gets two new
   blocks in the SCENES layer (yellow — source present, no full
   reproduction yet).

## Consequences

- The default `uv sync` footprint does not change; both backends are
  opt-in.
- The `openral benchmark run --suite maniskill3_pick_place --vla …`
  invocation works for users who run `uv sync --extra maniskill3` (or
  `--extra simpler-env`); without the extra it raises a typed
  `ROSConfigError` from the lazy import inside the scene factory,
  pointing the user at the install command.
- ManiSkill3 ships in pre-release versions today (`>=3.0.0b9` at time
  of writing). The extras-group pin uses a lower bound rather than an
  exact pin so SemVer-compatible patches flow in. When MS3 hits
  1.0.0 we tighten the pin in a one-line update.
- `simpler-env` has no PyPI release. The extras-group entry uses the
  upstream git URL; that is acceptable because the extra is opt-in.
  When the upstream cuts a release we move to a PyPI entry.
- RoboCasa, VLABench, CALVIN, RLBench all remain deferred — see the
  context table above. A future ADR can introduce each as another
  opt-in extras group on the same pattern this ADR establishes.

## Alternatives considered

- **RoboCasa first**: rejected. The robosuite version pin conflicts
  with LIBERO and the 10 GB asset download adds non-trivial UX work
  (`openral sim assets fetch robocasa`). Better tackled in a dedicated ADR.
- **Single combined `sim-extras` group**: rejected. ManiSkill3 and
  SimplerEnv have different upstream cadence (one PyPI, one git) and
  different value props (raw sim throughput vs real-to-sim
  correlation). Keeping them separable lets users pick.
- **Pull SAPIEN into the default install**: rejected. SAPIEN is a
  ~200 MB binary wheel with platform-specific quirks (Vulkan on
  Linux); pulling it into the default workspace breaks the "import
  `openral_sim` works everywhere" property the lazy-import
  discipline gives us today.

## Verification

- `uv sync --extra maniskill3` succeeds on a Linux host with Vulkan.
- `openral benchmark run --suite maniskill3_pick_place --rskill placeholder --dry-run`
  prints the planned (task × seed) matrix without importing SAPIEN
  (the dry-run never hits the lazy factory).
- `openral benchmark run --suite maniskill3_pick_place --rskill placeholder`
  with the extra installed actually runs the rollouts and writes the
  JSON.
- `tests/unit/test_benchmark_schemas.py` parametrised cases pass on
  every host, without the extras installed.

## Amendments

### 2026-05-18 — Status flipped Proposed → Accepted

Both backends are on disk and exercised by benchmark YAMLs:

- `python/sim/src/openral_sim/backends/maniskill3.py` — ManiSkill3 / SAPIEN
  backend, lazy-imported via the sim factory so `import openral_sim` does
  not pull in SAPIEN.
- `python/sim/src/openral_sim/backends/simpler_env.py` — SimplerEnv
  backend layered on top of ManiSkill3.
- `benchmarks/maniskill3_pick_place.yaml` — ManiSkill3 PnP suite.
- `benchmarks/simpler_env_google_robot.yaml` — SimplerEnv google-robot
  real-to-sim correlation suite.

The opt-in extras-group pattern declared in the Decision is in place via
the `[dependency-groups]` block at root `pyproject.toml`. No behavioural
change against the Decision text — only the status field flips.

### 2026-05-22 — `openral sim run` compatibility against MS3 v3.0.x

Two adapter fixes landed alongside the first `scenes/` configs
(`maniskill3_pick_cube.yaml`, `simpler_env_widowx_carrot.yaml`) and the
matching `tests/sim/test_franka_panda_maniskill3_adapter.py` /
`tests/sim/test_widowx_simpler_env_adapter.py` integration tests:

- **ManiSkill3** — the default `obs_mode` was bumped from `rgb+state`
  (which collapses agent kinematics into a single flat tensor) to
  `state_dict+rgb` so the adapter's `_extract_state` actually surfaces
  `agent.qpos` + `agent.qvel`. Both `obs_mode` and `control_mode` stay
  overridable via `scene.backend_options`.
- **SimplerEnv** — upstream `simpler_env.make()` injects
  `prepackaged_config=True` and `obs_mode='rgbd'`, both of which MS3
  v3.0.x rejects. The adapter now calls `gym.make` directly with the
  `ENVIRONMENT_MAP[friendly_name]` kwargs and the only obs mode the
  Bridge digital-twin envs advertise (`rgb+segmentation`), and bumps
  the upstream `-v0` env-id suffix to whichever `-v*` is actually
  registered (MS3 v3.0.x carries `-v1`). The obs / state / action
  extraction helpers are shared with the ManiSkill3 adapter since
  SimplerEnv now sits on top of MS3.

Today only the four WidowX bridge tasks
(`widowx_carrot_on_plate`, `widowx_spoon_on_towel`,
`widowx_stack_cube`, `widowx_put_eggplant_in_basket`) are wired
end-to-end through MS3 v3.0.x. The `google_robot_*` friendly names in
`simpler_env.ENVIRONMENT_MAP` resolve to env ids that are not yet
registered upstream; revisit when MS3 ports them.

### 2026-05-22 — RLDX-1 SimplerEnv wire schemas

`rldx1-ft-simpler-{widowx,google}-nf4` now have first-class layout
support in the RLDX adapter:

- New `StateContract.layout` values `simpler_widowx` /
  `simpler_google` (extra="forbid" rejects unknown layouts so manifests
  that drift surface as a `pydantic.ValidationError` at load).
- `python/sim/src/openral_sim/policies/rldx.py` grew
  `_build_simpler_widowx_obs` / `_build_simpler_google_obs` matching
  the upstream `rldx/eval/sim/SimplerEnv/simpler_env.py` wire schema
  exactly: WidowX uses `video.image_0` + bridge-rotated Euler RPY +
  `state.pad=0` sentinel + raw gripper; Google uses `video.image` +
  xyzw-quat state + gripper closedness. The chunk assembler binarizes
  the WidowX gripper column (`2*(g>0.5) - 1`) per upstream
  `_postprocess_gripper`. Google's sticky-gripper state machine is a
  follow-up.
- Layout drives the upstream `EmbodimentTag` automatically
  (`simpler_widowx` → `OXE_BRIDGE_ORIG`, `simpler_google` →
  `OXE_FRACTAL`), so `tools/rldx_sidecar.py` boots with the right
  modality config without the user having to override
  `OPENRAL_RLDX_EMBODIMENT_TAG`. The enum module also defines
  `OXE_WIDOWX` / `OXE_GOOGLE`, but the published FT-SIMPLER-*
  checkpoints' `processor_config.json` only ships `bridge_orig` and
  `fractal20220817_data` modality buckets — passing the unused enum
  names crashes `PolicyLoader.load` with `KeyError: oxe_widowx`.
- `python/sim/src/openral_sim/backends/simpler_env.py` rebuilds the
  legacy MS2 `obs['agent']['eef_pos']` 8-vector (the upstream
  reference still expects it) from the SAPIEN `ee_gripper_link` /
  `link_ee` pose + last qpos channel, since MS3 v3.0.x dropped the
  field from the Bridge digital-twin envs.
- Wire shape pinned by `tests/unit/test_rldx_simpler_env_wire_shape.py`
  (10 tests; runs without the heavy sidecar bootstrap).

No native RLDX-1 finetune targets MS3 PickCube / StackCube /
PushCube, so the ManiSkill3 adapter still has no first-class RLDX
rSkill — use `pi05`, `smolvla`, or `act` checkpoints with a
Panda-compatible embodiment instead.

## Amendment (2026-06-02): `simpler_env_google_robot` suite removed

The `benchmarks/simpler_env_google_robot.yaml` suite (and its
`BenchmarkName` member) have been **removed**. SimplerEnv on MS3 v3.0.x
registers only the WidowX bridge envs; the Google-robot tasks
(`GraspSingleOpenedCokeCanInScene`, `MoveNearGoogleBakedTexInScene`, the
drawer envs) remain in `simpler_env.ENVIRONMENT_MAP` but are **not registered
upstream**, so `openral benchmark run --suite simpler_env_google_robot` raised
`gymnasium.error.NameNotFound`. `simpler_env_widowx` stays (validated
end-to-end). The `rldx1-ft-simpler-google-nf4` rSkill was removed (2026-06-13)
since no Google-robot scene ships to run it on; the `simpler_google` obs-builder
layout and the `robots/google_robot` manifest are kept so a Google-robot rSkill
can be re-added when upstream registers the envs and the suite returns.

## Amendment (2026-06-19): CALVIN evaluated for issue #52 — deferral reaffirmed

GitHub issue [#52](https://github.com/OpenRAL/openral/issues/52) proposed adding
**CALVIN** (Mees et al., 2022, [2112.03227](https://arxiv.org/abs/2112.03227)) —
a long-horizon, language-conditioned Franka benchmark on the canonical ABC→D
generalization split — as a follow-up to the benchmark-tier cleanup in PR #48.
After evaluation the CALVIN backend is **not added now**; the deferral recorded
in the context table above (`CALVIN | MIT | py3.8 only | gym | Declining`) is
**reaffirmed**, with concrete reasoning:

- **Python incompatibility / out-of-process sidecar required.** `calvin_env`
  (the PyBullet environment CALVIN ships) targets Python 3.8 and does not import
  under the repo's pinned 3.12 core (CLAUDE.md §2). Unlike the ManiSkill3 /
  SimplerEnv backends in this ADR — which sit behind a lazy import in the same
  process — CALVIN would have to run **out-of-process in a dedicated sidecar
  venv**, the heavier pattern established for Isaac Sim (ADR-0045) and the GR00T
  / RLDX policy backends. That is materially more infrastructure than the "one
  opt-in extras group, one lazy-imported backend" shape this ADR establishes.

- **No verified policy adapter.** None of the CALVIN policy families the issue
  lists (HULC / HULC++, 3D Diffuser Actor, RoboFlamingo, GR-1 / Seer / MDT) has
  an existing OpenRAL adapter or a checkpoint verified to run in our catalogue.
  A benchmark with no task-matched, runnable rSkill fails the ADR-0060
  compatibility gate and could only ship as unscoreable scaffolding — exactly
  the "plausible-but-unscoreable 0s" that PR #48 set out to remove from the
  benchmark tier.

- **Declining citation share.** As the context table notes, CALVIN's share of
  newly reported VLA results is declining relative to LIBERO / SimplerEnv /
  ManiSkill3, all of which the catalogue already covers end-to-end. The marginal
  credibility gain is low against the Tier-3 effort.

- **No-mocks constraint.** CLAUDE.md §1.11 / §1.2 forbid shipping a backend
  whose numbers we have not actually produced. A faithful CALVIN landing
  therefore requires a real py3.8 sidecar install plus a real policy rollout on
  a GPU host — out of scope for a docs-tier follow-up to PR #48.

**Decision:** issue #52 is closed as *deferred / not-planned for now*. CALVIN
remains a candidate. Reintroducing it would follow the Isaac/GR00T sidecar
pattern (ADR-0045) and warrant its own ADR covering the sidecar boot, the chosen
policy family's adapter, and the ABC→D suite — at which point this deferral is
superseded.
