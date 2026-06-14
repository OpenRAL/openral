# ADR-0007: Separate physical robot manifests from simulator IO contracts

- Status: Accepted
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)

## Context

Five `robots/<id>/robot.yaml` manifests in tree describe what is supposed
to be a *physical robot* (CLAUDE.md §6.1, Layer 0 — HAL): kinematic chain,
end-effector, capabilities, safety envelope. In practice, three of them
have been **co-tuned to a particular simulator's IO conventions** rather
than the underlying hardware:

| YAML | Physical robot it claims to describe | Sim-specific bits leaking in |
|---|---|---|
| `robots/libero_franka/robot.yaml` | Franka Emika Panda (7-DoF) | LIBERO's 8-D `eef_pos+axisangle+gripper_qpos` state, 7-D delta-EEF action, image flip 180°. |
| `robots/metaworld_sawyer/robot.yaml` | Rethink Sawyer (7-DoF) | MetaWorld's 4-D `agent_pos` (XYZ + gripper) state, 4-D delta-XYZ-plus-gripper action. |
| `robots/pusht_2d/robot.yaml` | n/a — PushT is a synthetic 2-D benchmark | The whole manifest is a sim-only pseudo-robot. |
| `robots/aloha_bimanual/robot.yaml` | Trossen ALOHA (2 × 7-DoF) | None — gym-aloha already mirrors the real robot's joints. |
| `robots/so100_follower/robot.yaml` | LeRobot SO-100 follower | None — same description for sim and real hardware. |

Two consequences:

1. **Two `RobotDescription`s for one physical robot.** `libero_franka` and
   the in-code `FRANKA_PANDA_DESCRIPTION` (`python/hal/src/openral_hal/franka_panda.py:168`)
   describe the same physical Panda but with different joint names,
   limits, and `observation_spec`/`action_spec`. The drift was invisible
   until the manifest-vs-HAL regression test (PR 2 / issue #54) exposed
   it.
2. **Robot identity is overloaded.** "`robot_id: libero_franka`" really
   means "Franka Panda *running inside LIBERO*". Same Panda checkpoint,
   different sim → different `robot_id` → different manifest. That maps
   awkwardly onto the embodiment-tag compatibility check, the rSkill
   capability matcher, and the architectural seam between Layer 0 (HAL)
   and the eval scene (which is closer to Layer 5 / the `openral_sim`
   scene adapter).

The **`observation_spec` / `action_spec` fields on `RobotDescription`
are also unused** — `rg "\.observation_spec|\.action_spec"` returns zero
hits across `python/`, `tests/`, and `examples/`. They were declared in
YAML but never consumed at runtime. The sim-imposed IO contract is in
fact already encoded in the eval scene adapters
(`python/sim/src/openral_sim/{policies,backends}/{libero,metaworld,pusht}.py`)
and in each rSkill's `RSkillManifest`. So the cleanup is simpler than
the description above suggests: we delete the misleading specs from
the per-robot YAMLs, document the sim-imposed contract in the matching
scene adapter (where it already operationally lives), and rename the
YAMLs so the `robot_id` axis tracks physical embodiment.

## Decision

1. **`robots/<id>/robot.yaml` describes a physical embodiment only.**
   No sim-specific `observation_spec` / `action_spec`. No "this is the
   Franka *as LIBERO sees it*" tuning. Joint limits, EE, capabilities,
   safety envelope — everything that survives the change of simulator.
2. **`openral_sim.{policies,backends}.<scene>` is the source of truth for
   sim-imposed IO contracts.** The LIBERO scene adapter declares the
   8-D EEF state, 7-D delta-EEF action, and 180° image flip; the
   MetaWorld adapter declares its 4-D `agent_pos` etc. These already
   exist in code; this ADR formalises that they are not duplicated
   into the per-robot YAML.
3. **Rename to physical embodiments.** `robots/libero_franka/` →
   `robots/franka_panda/`. `robots/metaworld_sawyer/` →
   `robots/sawyer/`. The new `robots/franka_panda/robot.yaml` mirrors
   `FRANKA_PANDA_DESCRIPTION` (the existing manifest-vs-HAL drift-guard
   test catches divergence going forward). The new `robots/sawyer/`
   describes the physical Rethink Sawyer; no real-HW HAL today
   (tracked by issue #57).
4. **`pusht_2d` stays.** PushT genuinely has no real robot; the
   manifest stays as a documented "scene-pseudo-robot" — the schema
   needs *some* `robot_id` to round-trip through `SimEnvironment`,
   and `pusht_2d` is the canonical 2-D-tip embodiment of the PushT
   benchmark. Its README is updated to make the sim-only nature
   explicit.
5. **`aloha_bimanual` stays unchanged.** It already describes the
   real Trossen ALOHA; the gym-aloha sim mirrors that one-to-one.
6. **Hard rename, no shim.** Pre-1.0 (`openral-core` is `0.2.0`,
   every other workspace package is `0.1.0`). No deprecation
   re-export under the old `robot_id`s. CLAUDE.md anti-fallback
   principle (§9 / §1).
7. **`mock_robot` stays in this PR**; its removal is a separate PR
   (planned PR 9 — depends on this split for fixture migration) so
   the diff here stays focused.

## Consequences

- `scenes/{smolvla,xvla,pi05}_libero_spatial.yaml` migrate
  `robot_id: libero_franka` → `robot_id: franka_panda`.
- `scenes/benchmark/metaworld_push.yaml` migrates
  `robot_id: metaworld_sawyer` → `robot_id: sawyer`.
- `scenes/{act_aloha_transfer_cube, diffusion_pusht}.yaml`
  unchanged.
- `tests/unit/test_sim_environment_schemas.py` test fixture migrates
  to `franka_panda`.
- `robots/libero_franka/`, `robots/metaworld_sawyer/` deleted.
- New `robots/franka_panda/`, `robots/sawyer/` added.
- Auto-discovery (PR 3) means no `_BUILTIN_ROBOTS` edit is required.
- `docs/reference/vla_compatibility.md` already lists `franka_panda`
  in the Robot tag column for π0.5/xVLA/GR00T-LIBERO checkpoints; no
  churn there. The §3.1 LIBERO header now points at
  `robots/franka_panda/`.
- `openral_sim` runner code is unchanged — it never read
  `RobotDescription.observation_spec` / `action_spec`. The LIBERO and
  MetaWorld scene adapter docstrings are extended to document the
  sim-imposed IO contract that previously lived (uselessly) in YAML.
- The robot manifest-vs-HAL drift-guard (`tests/unit/test_robot_manifests_match_hal_constants.py`,
  PR 2) extends to `robots/franka_panda/robot.yaml` ↔
  `FRANKA_PANDA_DESCRIPTION`.

## Migration

Single PR (PR 6). No multi-step deprecation. Anyone with an out-of-tree
`SimEnvironment` config hard-pinned to `robot_id: libero_franka` /
`metaworld_sawyer` updates the field in the same revision they bump
to this commit.

## Why not other options

- **Keep both YAMLs and document the conflation.** Surfaces the
  problem but doesn't fix it; the "two `RobotDescription`s for one
  physical Panda" trap stays.
- **Move `observation_spec` / `action_spec` to `SceneSpec` / a new
  `EmbodimentProfile` model.** Tempting on principle, but the fields
  have *no runtime consumer* today; adding new typed plumbing for
  unused fields is the opposite of what CLAUDE.md operating principle 4
  says ("explicit beats implicit; no hidden plumbing"). When a real
  consumer shows up — e.g. an action-space adapter inside the runner
  — that's the moment to add the typed surface.
- **Keep `pusht_2d` but move it under a new `scenes/` namespace.**
  The schema currently requires a `robot_id`, and threading a "this is
  a synthetic embodiment, not a real robot" flag through the
  `SimEnvironment` validator is more churn than the benefit warrants.
  A `README.md` note on `robots/pusht_2d/` is the smaller answer.

## Amendment 2026-06-08 — three-tier scene paths

ADR-0041 split `scenes/` into three tiers
(`scenes/deploy/`, `scenes/sim/`, `scenes/benchmark/`) and stripped rSkill
names from filenames. The MetaWorld migration example above moves from
`scenes/benchmarks/smolvla_metaworld_push.yaml` to
`scenes/benchmark/metaworld_push.yaml` (singular `benchmark/`, rSkill name
removed; an rSkill is now passed at the CLI via `--rskill`). The Decision
text and schema are unchanged — only on-disk paths are renamed. See
ADR-0041 for the tier hierarchy and
[`scenes/README.md`](https://github.com/OpenRAL/openral/blob/master/scenes/README.md) for the per-tier authoring
guide.
