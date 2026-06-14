# ADR-0027: rSkill state-contract bindings + layout adapter registry

- Status: **Proposed**
- Date: 2026-05-27
- Related: [ADR-0007](0007-robot-sim-split.md) (per-checkpoint
  action contract — the symmetric pattern on the action side);
  [ADR-0013](0013-rskill-manifest-actuators-and-processors.md) (closed `EmbodimentTag` literal —
  this ADR adds an analogous closed `StateLayout` enum-of-layouts);
  [ADR-0014](0014-maniskill3-simpler-env-backends.md) (the original `StateContract` —
  this ADR extends it with `bindings`);
  [ADR-0019](0019-rosbag2-lerobot-dataset-bridge.md) (state/action contracts as
  dataset-bridge contracts);
  [ADR-0025](0025-reasoner-managed-background-services.md) §"State-contract
  adapters" (the forward reference this ADR closes);
  CLAUDE.md §3 (types are the contract), §1.4 (explicit beats implicit),
  §1.6 (schemas evolve, never silently).

## Context

The reasoner today rejects every VLA rSkill whose
`state_contract.layout ∈ {"human300_16d", "rc365", "smolvla_9d", …}`
when it's dispatched onto a robot whose raw `JointState` width
differs from `state_contract.dim`. Log line, today:

```
palette: skipping rSkill 'pi05-robocasa365-human300-nf4'
(model_family='pi05'): targets wrapped task-space layout
'human300_16d' (dim=16); deploy_sim feeds raw JointState (dim=10).
Run via ``openral sim run --vla ...`` for the sim-adapter path.
```

The `panda_mobile` robot's raw `JointState` is 10-D (3 planar base + 7
Franka arm). The pi05 / rldx checkpoints were trained on a 16-D
*task-space composite* (EE pose in base frame + base pose in world +
gripper fingers — see `python/sim/src/openral_sim/backends/robocasa.py:508`).
The two are not interchangeable: same robot, same DoF, different
encoding.

Today the only path that produces the 16-D vector is the *sim adapter*
inside `openral_sim.backends.robocasa` — bound to robosuite's
`robot0_base_to_eef_pos` / `robot0_base_quat` / etc. observation keys.
That works for `openral sim run --vla ...` because robosuite mints those
keys directly; it does **not** work for `deploy sim` (which goes through
the ROS skill runner consuming `/joint_states`) and never works on
real hardware. The result: every wrapped-task-space VLA is filtered out
at palette seed, panda_mobile gets 0 dispatchable VLA skills, and the
Nav2-plus-VLA pattern the user wants is structurally blocked.

This is a state-contract gap, not a robot gap. **The same Franka with
the same URDF can absolutely produce the 16-D `human300_16d` vector**:
joint angles → URDF forward kinematics → `/tf` → assemble. The missing
piece is the assembler — what ADR-0025 §"State-contract adapters" calls
out as a follow-up.

## Decision

Introduce a **layout-adapter registry** at layer 2 (World State) that
assembles the per-checkpoint state vector at runtime, and extend
`StateContract` with a `bindings` block so the manifest can declare
*which* TF frames + joint names this robot uses to populate the layout.

### Schema (extension to `openral_core.StateContract`)

```python
class StateContractBindings(BaseModel):
    """Per-robot source bindings for an rSkill's `state_contract.layout`.

    Symmetric to ``ControlModeSemantics`` on the action side: the
    rSkill manifest names the *shape* (closed-enum ``layout``); these
    bindings name the *sources* on the deploying robot. The
    assembler joins shape × bindings + live JointState + live TF.
    """

    model_config = ConfigDict(extra="forbid")

    eef_frame: str | None = None              # tf2 link of the end effector
    base_frame: str | None = None             # tf2 link of the mobile base
    world_frame: str | None = "map"           # tf2 root frame
    gripper_qpos_joints: list[str] = []       # JointState names for gripper width
    quaternion_convention: Literal["xyzw", "wxyz"] = "xyzw"


class StateContract(BaseModel):
    ...
    layout: StateLayout | None = None         # existing, closed Literal
    dim: int | None = None                    # existing
    bindings: StateContractBindings | None = None   # NEW
```

`bindings` is optional for layouts whose assembler is a pure joint-space
slice (`"libero"`, `"aloha"`) and required for task-space composites
(`"human300_16d"`, `"rc365"`, `"gr1"`). The validator
enforces the per-layout requirement at manifest load.

### Adapter package layout

New ament-Python package `python/state_adapter/`:

```
python/state_adapter/
├── pyproject.toml
└── src/openral_state_adapter/
    ├── __init__.py        # re-exports LAYOUT_ASSEMBLERS, assemble_state
    ├── _registry.py       # dict[StateLayout, Assembler] + register decorator
    ├── _protocol.py       # Assembler Protocol (joint_state, tf_buffer, bindings → np.ndarray)
    └── layouts/
        ├── human300_16d.py   # 3 + 4 + 3 + 4 + 2 = 16
        └── rc365.py          # human300_16d wrapper
```

Each layout file is ~30–60 LoC: one pure function reading
`(bindings, joint_state, tf_buffer) → np.ndarray`. **No robot
knowledge.** The registry maps closed `StateLayout` literal → assembler.

### Integration points

1. **Skill runner (`rskill_runner_node.py:_step_impl`, line ~1284):**
   When `manifest.state_contract.layout in LAYOUT_ASSEMBLERS`, the runner
   calls the registered assembler with the manifest's `bindings` + a
   `tf2_ros.Buffer` + the current `JointState`, and stuffs the result
   into `obs["state"]` instead of the raw joint slice. When the layout
   is absent or unregistered, the existing joint-permutation path stays
   verbatim — backward compatible.

2. **Reasoner filter (`reasoner_node.py:_maybe_seed_palette_from_search_paths`,
   line ~795):** flip the "wrapped task-space layout" drop:
   ```python
   from openral_state_adapter import LAYOUT_ASSEMBLERS
   if sc.layout in _WRAPPED_TASK_SPACE_LAYOUTS and sc.layout in LAYOUT_ASSEMBLERS:
       state_compatible.append(m)
       continue
   ```
   With an assembler registered, the skill enters the palette; without one,
   the existing drop behaviour holds.

3. **Launch (`sim_e2e.launch.py`):** ensure `robot_state_publisher` is
   already in the launch tree (`robots/<id>/robot.yaml` carries the URDF
   path; the launch include feeds it). This is the live TF source the
   assemblers read.

### Manifest update (fixture, ADR-0019 dataset bridge)

`rskills/pi05-robocasa365-human300-nf4/rskill.yaml`:
```yaml
state_contract:
  layout: "human300_16d"
  dim: 16
  bindings:
    eef_frame: "panda_hand_tcp"
    base_frame: "base_link"
    world_frame: "odom"
    gripper_qpos_joints: ["panda_gripper"]
    quaternion_convention: "xyzw"
```

The same manifest works for every robot whose URDF declares
`panda_hand_tcp` / `base_link` and a parallel-gripper joint. Per-robot
overrides (e.g. a Stretch with an `eef` frame name) go in the robot's
own per-skill binding override (out of scope for this ADR — note as
follow-up).

Note on `eef_frame: "panda_hand_tcp"`: robosuite's
`robot0_base_to_eef_pos` reads the eef SITE (`gripper0_right_grip_site`,
~0.097 m along the hand's local z-axis from the wrist body), not the
hand body itself. The URDF's `panda_hand_tcp` fixed link sits at
xyz=`0 0 0.1034` along panda_hand's z and lands within ~6 mm of the
MJCF site. Pointing the binding at `panda_hand` alone leaves a ~9 cm
gap in `base_to_eef.z` that destabilises NF4-quantized VLAs.

## Consequences

**Positive**

- `panda_mobile` + Nav2 + pi05/rldx VLAs work concurrently in
  `deploy sim` once the assembler for the matching layout is registered.
- Adding a new VLA family (e.g. `gr1_39d` for the humanoid) is one new
  file under `layouts/`, no HAL changes, no robot changes.
- The state-contract is now a fully-typed contract end-to-end:
  manifest declares shape + source bindings; assembler is the *only*
  code path; the closed `StateLayout` enum gates the registry.
- Symmetric to the existing action-side pattern (`ControlModeSemantics`
  with `joint_order` + `reference_frame` + `gripper_convention`),
  closing a known gap.

**Negative / cost**

- The skill_runner adds a `tf2_ros.Buffer` + a `/tf` subscription. ~3 ms
  per step for the TF lookups on a 5 Hz tick — acceptable; rejected
  for >100 Hz hot paths (those are C++ ros2_control today and don't
  go through this runner).
- Manifest authors of new wrapped-task-space VLAs must populate
  `bindings`. The validator surfaces the requirement at install time
  (`ral skill install` fails with the missing-fields list).
- Closed `StateLayout` enum bounds the supported set — a brand-new
  layout requires (a) adding to the literal, (b) writing the assembler,
  (c) declaring bindings on the manifest. By design — silent layout
  drift is the failure mode this ADR exists to prevent.

**Out of scope**

- A *generic* declarative layout (per-field `{kind: tf|joint, ...}`
  list) that obviates the closed enum + per-layout assemblers. Real
  risk: subtle drift between training-time layout and a manifest
  author's free-form spec. Revisit if the closed enum grows beyond
  ~10 layouts.
- Action-side adapter symmetry — `ActionContract.representation`
  already has `joint_order` + `reference_frame` + `gripper_convention`
  doing the equivalent job. No action-side ADR needed here.
- Real-hardware integration. The assembler reads `/tf` and `/joint_states`
  — both standard ROS topics — so any HAL emitting them works without
  change. HAL HIL tests will exercise this path in a follow-up.

## Known follow-up — robot_state_publisher in the panda_mobile launch

The `human300_16d` assembler reads `base_link → panda_hand` from `/tf`.
The `panda_mobile` HAL today publishes `odom → base_link` (`packages/openral_hal_panda_mobile/lifecycle_node.py:509`)
and slam_toolbox supplies `map → odom`, but **the per-link arm TF
(`panda_link0 → … → panda_hand`) is not on the bus**: there is no
`robot_state_publisher` instance running, and no URDF file lives under
`robots/panda_mobile/` (the simulation drives a robosuite-side MJCF).

Until that's wired, the reasoner will admit a pi05 / rldx rSkill that
declares `human300_16d`, but the first `_step_impl` will raise
`tf2.LookupException` and fail the goal with a clean
`ROSRskillGoalSatisfied=False` message. The operator sees the failure
immediately — no silent zero-fill.

Concrete unblock (next PR):

1. Add a `franka_description` URDF (or compose one from `robot_descriptions`)
   under `robots/panda_mobile/urdf/panda_arm.urdf.xacro`, parented to
   `base_link` at the robosuite mount offset.
2. Include `robot_state_publisher` in `packages/openral_rskill_ros/launch/sim_e2e.launch.py`
   (the launch already includes slam_toolbox + Nav2; add one more node).
3. The `panda_mobile` HAL publishes joint angles on `/joint_states` already
   (`lifecycle_node.py:570`); `robot_state_publisher` consumes that
   + the URDF and emits the per-link TF. No HAL changes needed.

That makes panda_mobile + Nav2 (base) + pi05 (arm) work concurrently
in `deploy sim` — the end-to-end use case ADR-0027 exists to enable.

## Implementation sequence

Per CLAUDE.md §4.2 (smallest viable PR; each commit independently
reviewable):

1. **`docs(adr): ADR-0027`** — this file only.
2. **`feat(schemas): StateContractBindings + state_contract.bindings`**
   — Pydantic model + cross-validator that enforces `bindings` is set
   for task-space layouts. Hypothesis fuzz + a real-fixture
   round-trip test.
3. **`feat(state_adapter): openral_state_adapter package + human300_16d`**
   — new package, registry, one assembler, unit tests with synthetic
   TF + JointState frames (no mocks of TF / JointState themselves — real
   ROS message types).
4. **`feat(runner+reasoner): wire LAYOUT_ASSEMBLERS into skill_runner + flip filter`**
   — the runner consults the registry; the reasoner admits-with-adapter;
   `rskills/pi05-robocasa365-human300-nf4/rskill.yaml` gains `bindings`;
   live `deploy sim` test sequences Nav2 → pi05 on panda_mobile.
