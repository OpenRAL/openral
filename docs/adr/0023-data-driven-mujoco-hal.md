# ADR-0023: Data-driven MuJoCo HAL — move per-robot constants into `RobotDescription.sim`

- Status: Proposed
- Date: 2026-05-24
- Amended: 2026-05-24 (see Amendments below)

## Context

Every MuJoCo-backed HAL adapter in the open core today (SO-100, Franka,
UR5e, UR10e, Flexiv Rizon-4, Unitree G1, Unitree H1) inherits from
`openral_hal._mujoco_arm.MujocoArmHAL` and passes a fixed set of
constants to its `__init__`:

* `mjcf_path` — resolved by a module-private `_<robot>_mjcf_path()`
  function that imports a specific `robot_descriptions.<X>_mj_description`
  module.
* `joint_qpos_addr: dict[str, int]` — usually a 1-to-1 mapping of the
  robot's `description.joints` list to MJCF `qpos` indices, sometimes
  with a floating-base offset (G1, H1).
* `joint_qvel_addr: dict[str, int] | None` — non-trivial only for
  floating-base robots (G1, H1).
* `actuator_index: dict[str, int]` — usually 1-to-1 with the joint list.
* `gripper_joint`, `gripper_ctrl_range`, `gripper_qpos_addrs`,
  `gripper_qpos_scale` — gripper wiring.
* `_read_gripper_normalised` — overridden by `SO100MujocoHAL` because
  its menagerie `Jaw` joint is revolute with a non-zero closed position
  (`-0.174` rad), so the base "sum-over-scale" math reports ≈0 over the
  first 9 % of the open range.

The result: **every robot ships ~80 lines of Python whose only purpose is
to hand a handful of integers and floats to the same base class**. The
constants are not derivable from `description.joints` alone (the MJCF
asset is external; the floating-base offset depends on the URDF; gripper
qpos indices depend on finger count), so they cannot be dropped — but
they also do not belong inside the HAL implementation. Per
CLAUDE.md §3 the `RobotDescription` is the normative
contract; today's split forces a contributor adding a new robot to write
a Python file with hardcoded indices instead of editing one `robot.yaml`.

The user-facing entry points (`openral sim run`, `openral deploy run`) already
dispatch by `robot_id` via `robots/<id>/robot.yaml`, so the
*scope mismatch* is purely inside the HAL layer:

```
openral sim run --robot ur5e          # already manifest-driven
  → SCENES factory → backend (libero/metaworld/...)
  → (only some scenes use MujocoArmHAL directly)

tests/sim/test_*_hal_mujoco.py    # NOT manifest-driven
  → SO100MujocoHAL() / UR5eHAL() / ... — each hardcodes its constants

openral deploy run --transport.digital_twin=true
  → HAL_REGISTRY (factory.py) — only "so100_follower" registered today
```

## Decision

Move the MuJoCo wiring constants into `RobotDescription` and collapse
the per-robot adapters into a single data-driven base.

### 1. Schema: add `RobotDescription.sim: SimDescription | None`

New Pydantic model in `openral_core.schemas`:

```python
class GripperReadMode(str, Enum):
    SUM_OVER_SCALE = "sum_over_scale"      # default (Franka parallel)
    AFFINE_LOW_HIGH = "affine_low_high"    # (qpos - low) / (high - low) (SO-100 Jaw)

class SimGripperDescription(BaseModel):
    joint: str                              # name from description.joints
    ctrl_range: tuple[float, float]
    qpos_addrs: tuple[int, ...]
    qpos_scale: float
    read_mode: GripperReadMode = GripperReadMode.SUM_OVER_SCALE

class SimDescription(BaseModel):
    """MuJoCo wiring for a robot.  Consumed by openral_hal._mujoco_arm.

    All fields are optional with safe defaults derived from
    description.joints when the robot has no floating base and a 1:1
    joint→qpos/actuator mapping.
    """
    mjcf_uri: str
    """One of:
      * 'file:/abs/path/to.xml'           — explicit filesystem path
      * 'robot_descriptions:<module>'     — defer to robot_descriptions package;
                                            value is the module name whose
                                            ``MJCF_PATH`` attribute is read at
                                            connect() time.
    """
    floating_base: bool = False
    """If True, joint_qpos starts at index 7 (free joint) and
    joint_qvel at index 6, matching MuJoCo's free-joint convention."""
    joint_qpos_addr: dict[str, int] | None = None
    """Override the default 1:1 mapping (in description.joints order,
    offset by 7 if floating_base)."""
    joint_qvel_addr: dict[str, int] | None = None
    """Override the default 1:1 mapping (in description.joints order,
    offset by 6 if floating_base)."""
    actuator_index: dict[str, int] | None = None
    """Override the default 1:1 mapping."""
    gripper: SimGripperDescription | None = None
    settle_steps_default: int = 1
```

### 2. `MujocoArmHAL.from_description(description)` factory

```python
@classmethod
def from_description(
    cls,
    description: RobotDescription,
    *,
    settle_steps: int | None = None,
    gravity_enabled: bool = True,
    staleness_limit_s: float = 0.5,
    mjcf_path_override: str | None = None,
) -> MujocoArmHAL: ...
```

Reads `description.sim`, computes any defaults, resolves `mjcf_uri` to a
filesystem path (importing `robot_descriptions.<name>` lazily), then
delegates to `__init__`. The existing `__init__` signature stays the
same so existing call sites (and tests) keep working; the new factory is
the recommended path.

The base class grows a `gripper_read_mode: GripperReadMode` parameter and
implements both modes in `_read_gripper_normalised`. The SO-100 override
is deleted.

### 3. Subclasses become thin wrappers

`SO100MujocoHAL`, `FrankaPandaHAL`, `UR5eHAL`, `UR10eHAL`,
`Rizon4MujocoHAL`, `G1MujocoHAL` each shrink to:

```python
class UR5eHAL(MujocoArmHAL):
    def __init__(self, **kwargs) -> None:
        from openral_hal._real_description import load_description
        super().__init__(
            **MujocoArmHAL._from_description_kwargs(
                load_description("ur5e"), **kwargs,
            )
        )
```

— or are eliminated entirely if `robots/<id>/robot.yaml` carries
`sdk_entry: openral_hal._mujoco_arm:MujocoArmHAL.from_description`
(once the runner factory learns to call class methods).

`H1MujocoHAL` keeps its `_per_step_update` override (torque control via
PD law on every `mj_step`) but loses its constants block.

### 4. Out of scope (separate ADR)

* `SawyerRealHAL` — real-hw-only.
* Sim *scene* backends (`libero`, `metaworld`, `robocasa`, etc.) — they
  do not use `MujocoArmHAL`. Each backend's robot selection already
  goes through `robot_id`.

## Consequences

### Migration

* `openral_core` schema stays at the pre-publish baseline
  (`schema_version: "0.1"`); the `sim:` field is additive and
  optional, so no migrator is required (CLAUDE.md §1.6).
* Existing `robots/*/robot.yaml` files for SO-100, Franka, UR5e, UR10e,
  Rizon-4, G1, H1 gain a `sim:` block populated with the constants
  extracted from the deleted Python files. Hand-edited manifests stay
  in sync with the JSON Schema export (`just schema-export`).
* `docs/METHODS.md` gains entries for `SimDescription`,
  `SimGripperDescription`, `GripperReadMode`,
  `MujocoArmHAL.from_description`. Removes the per-robot constant
  blocks.

### Risk

* Schema breaks for downstream consumers if they construct
  `RobotDescription` from kwargs without `sim`. Mitigated by the
  field being optional with a `None` default; only consumers that read
  `description.sim` care about it.
* Subclass identity matters in two existing call sites:
  `tests/sim/test_all_hals_via_runner.py` and the safety-kernel sim
  tests parametrize over `isinstance(hal, FrankaPandaHAL)` etc. Those
  identity checks need to switch to `description.name` checks.

### Reversibility

* If the new factory turns out to be the wrong abstraction (e.g. the
  field set proves insufficient for a future robot), the per-robot
  subclasses can be re-introduced without touching `RobotDescription` —
  the `sim:` block stays declarative.
* If `SimDescription` proves redundant (e.g. we adopt a full URDF→MJCF
  resolver), it can be deprecated by making every field optional and
  having `from_description` synthesize them from the URDF. The schema
  field would survive as a typed override.

## Alternatives considered

1. **Full URDF→MJCF auto-resolver.** Parse the URDF in
   `description.urdf_path`, build the MJCF dynamically, infer joint
   indices from the parser's output. **Rejected**: every supported robot
   already has a hand-tuned MJCF in `robot_descriptions` or
   `mujoco_menagerie`; reimplementing the URDF→MJCF pipeline is a
   multi-quarter project that doesn't materially improve the experience
   for the user, who just wants to point at a robot.

2. **Leave HALs as-is; only refactor the registry.** Keep the
   hardcoded constants inside each subclass but route every robot's
   `sdk_entry` through `MujocoArmHAL` directly. **Rejected**: the
   hardcoded constants are the actual pain point; routing alone leaves
   the per-robot files in place.

3. **Move constants into `RobotCapabilities` instead of a new
   `SimDescription`.** **Rejected**: the constants are MuJoCo-specific,
   `RobotCapabilities` is a sim-agnostic surface used by skill
   compatibility checks. Mixing concerns there would force every
   non-sim consumer to know about MuJoCo indices.

## Open questions

* Should `mjcf_uri` support `package://` URIs (ROS convention) too? Out
  of scope; the schemes covered (`file:`, `robot_descriptions:`,
  `gym_aloha:`, `openarm_v2:`) cover every robot in tree.
* Should hand-edits of `robots/*/robot.yaml` block on
  `MujocoArmHAL.from_description` success for each manifest? Yes — the
  test added in this PR exercises every robot.yaml end-to-end through
  the factory and runs in CI against the same fixtures.

## Amendments

### 2026-05-22 — Bimanual scope pulled in

Aloha and OpenArm v2 were originally deferred ("Out of scope" §4 above)
because their bimanual layouts didn't fit the single-arm + single-gripper
shape of the first cut. After landing the single-arm path, the gap
between them and the rest of the fleet became the new pain point: every
new bimanual robot would either bypass the manifest contract or
duplicate ~200 lines of constants. We re-scoped to cover them in this
same ADR.

**Schema extensions** (all additive on top of the original `SimDescription`):

* `SimDescription.gripper: SimGripperDescription | None` →
  `SimDescription.grippers: list[SimGripperDescription]`. Single-arm
  robots ship one entry; bimanual robots ship two.  The `model_validator`
  on `RobotDescription` checks every entry's joint against
  `description.joints` and rejects duplicates.
* `SimDescription.keyframe_index: int | None` — when set,
  `MujocoArmHAL.connect()` runs
  `mj_resetDataKeyframe(model, data, idx)` before `mj_forward`. Required
  by the gym-aloha bimanual MJCF (its parallel-jaw `ctrlrange=[0.021,
  0.057]` puts the actuator outside the default `qpos=0` and the
  position controller can't recover).
* `SimDescription.seed_ctrl_from_qpos: bool = False` — when True,
  `MujocoArmHAL.connect()` seeds `data.ctrl[actuator] =
  data.qpos[joint_qpos_addr]` for every controllable joint. Required by
  OpenArm v2's `<position>` actuators with per-class PD gains; without
  it every joint would be driven to `ctrl == 0` on first `mj_step`.
* `GripperReadMode.PASSTHROUGH` — report `qpos[addrs[0]]` verbatim
  (metres for Aloha prismatic fingers; radians for OpenArm revolute
  jaws).  Public surface is **not** normalised to `[0, 1]`; Skills must
  accept the raw range.
* `GripperWriteMode` enum (NORMALISED | PASSTHROUGH) — replaces the
  implicit "always normalised" assumption.  `PASSTHROUGH` writes the
  action's gripper value directly to `ctrl`; MuJoCo's `ctrlrange` does
  the clipping.
* `SimGripperDescription.actuator_index: int | None` — explicit override
  for the actuator that receives the (mapped) gripper command.  Defaults
  to the joint-wide `sim.actuator_index` map.
* `SimGripperDescription.mirror_actuator_index: int | None` — optional
  second actuator that receives the **negation** of the command.  Models
  Aloha's parallel jaws (positive finger + negative finger driven
  antisymmetrically) without forcing the second finger into
  `description.joints`.

**URI resolver extensions**: `resolve_mjcf_uri` learned two new schemes
alongside `robot_descriptions:` and `file:` — `gym_aloha:<scene>` (loads
`gym_aloha/assets/<scene>.xml`) and `openarm_v2:bimanual` (calls
`ensure_openarm_v2_mjcf` to fetch the upstream v2 MJCF that
`robot_descriptions` doesn't pin yet).

**HAL collapse**: `AlohaMujocoHAL` (was ~250 lines on `HALBase`) and
`OpenArmMujocoHAL` (was ~200 lines on `HALBase`) are now thin
`MujocoArmHAL` subclasses that delegate to `_sim_kwargs_for`. The bespoke
`connect()` logic (keyframe reset for Aloha, ctrl seeding for OpenArm)
moved into the shared base, driven by the manifest flags.

**Documentation**: the sim tutorial
(`docs/tutorials/sim/create-a-sim-environment.md` §3) gained a "The
`sim:` block" subsection covering every field including the bimanual
hooks.  The HAL README's "Adding a new MuJoCo-backed robot" recipe was
updated to mention `mirror_actuator_index` and `keyframe_index`.

**Tests**: `tests/sim/test_data_driven_mujoco_hal.py` parametrises across
all nine robots — the seven single-arm + Aloha + OpenArm — and the
Python-vs-YAML drift guard now compares every field of every
`SimGripperDescription` entry pairwise.  The 63 existing bimanual sim
tests pass against the shrunken subclasses unchanged.

The original "Out of scope" entry for the bimanual robots is left as a
strikethrough above so the historical decision is preserved.

### 2026-05-24 — ADR renumbered 0021 → 0023

This document was originally filed as ADR-0021 in draft. A numbering
collision with the curl-installer ADR was resolved by reassigning the
next free slots: ADR-0022 for rSkill action vocabulary, ADR-0023 (this
document) for data-driven MuJoCo HAL. All internal cross-references
updated in the same renumbering commit.
