# ADR-0024: ROS-wrapped rSkills (`kind: ros_action` / `ros_service`)

- Status: **Proposed**
- Date: 2026-05-25
- Related: [ADR-0018](0018-ros2-reasoner-supervisor.md) §4 (the closed
  tool palette this ADR keeps closed); [ADR-0020](0020-cpp-safety-kernel.md)
  (the safety supervisor / kernel that gates wrapped trajectories);
  [ADR-0022](0022-rskill-action-vocabulary.md) (the action / verb
  vocabulary surfaced to the LLM); [ADR-0013](0013-rskill-manifest-actuators-and-processors.md)
  (V1 in-place extension precedent); CLAUDE.md §1.1 (safety beats
  helpfulness), §3 (architecture discipline), §6.4 (rSkill packaging).

## Context

Every rSkill in OpenRAL today is a learnable **Vision-Language-Action
policy**: `RSkillManifest.model_family` is a closed `Literal` over
`{smolvla, pi05, xvla, act, diffusion, rldx}` and the only loader path
is `openral_sim.factory.make_policy` → a torch policy adapter wrapped
in `_PolicyAdapterSkill` (the shim in
`packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py`).
The hot-path contract is
`_step_impl(world_state: WorldState) -> Action` — a streaming policy
that emits action chunks at ~30 Hz.

This shape is the right one for VLAs but wrong for the rest of the
ROS 2 ecosystem we want the Reasoner to be able to invoke:

* **MoveIt** motion planning — a one-shot planner that returns a
  `trajectory_msgs/JointTrajectory` after ~1–2 s of FCL-based
  collision-checked planning.
* **Nav2** navigation — a long-horizon behaviour-tree controller that
  drives `cmd_vel` directly to the base.
* SLAM lifecycle (start mapping, save map), trajectory
  optimisation (TOPP-RA, …), perception primitives
  (FoundationPose, …) — all already-shipped ROS 2 packages.

The Reasoner's existing `ExecuteRskillTool` (the only LLM-visible
dispatcher per ADR-0018 §4) is the natural surface to invoke these:
no new tool variant means no LLM-contract change, no palette explosion,
no second dispatch path to maintain. What we need is a way for an
`rskill.yaml` to declare "this skill wraps a ROS 2 action / service
instead of carrying torch weights".

## Decision

Introduce a new discriminator on `RSkillManifest`:

```python
RSkillKind = Literal["vla", "wam", "ros_action", "ros_service"]
kind: RSkillKind  # required, no default
```

* **`"vla"`** — today's learnable policy. `model_family` and
  `weights_uri` required; `ros_integration` forbidden. Every
  pre-existing in-tree rSkill is migrated to `kind: vla` in this
  ADR's PR (mechanical insertion after the `role:` line of every
  `rskill.yaml` under `rskills/`).
* **`"ros_action"` / `"ros_service"`** — wraps a running ROS 2 action
  or service. `model_family`, `weights_uri`,
  `processors`, `state_contract`, `action_contract`,
  `n_action_steps`, `image_preprocessing`, `starting_pose` all
  forbidden (none of them have meaning). A new `RosIntegration`
  block is required:

  ```python
  class RosIntegration(BaseModel):
      package: str                              # "moveit_msgs", "nav2_msgs"
      interface_type: str                       # "MoveGroup", "NavigateToPose"
      interface_name: str                       # "/move_action", "/navigate_to_pose"
      result_trajectory_field: str | None       # dotted accessor into the result
      default_goal_json: str                    # validated JSON dict literal
      ros_dependencies: list[str]               # apt / colcon packages
  ```

  `chunk_size` is pinned to `1` for these kinds (validator-enforced).
* **`"wam"`** — reserved for the future World Action Model resolver
  branch (CLAUDE.md §3). Schema accepts it; the loader rejects it at
  resolve time with `ROSConfigError` so the discriminator is
  forward-compatible without exposing a half-built path.

A new adapter, `ROSActionRskill(rSkillBase)` in
`python/rskill/src/openral_rskill/ros_action_rskill.py`, is selected
by the local-resolver branch when `manifest.kind in {"ros_action",
"ros_service"}`. The adapter has two operating modes:

* **Trajectory mode** (`result_trajectory_field` set, e.g. MoveIt) —
  on first `step()` the adapter sends the goal, awaits the result,
  extracts a `trajectory_msgs/JointTrajectory`, reorders its
  `joint_names` into the host `RobotDescription.joints` order via
  `build_joint_permutation_from_names`, caches the waypoint list,
  and returns waypoint 0 as a 1-row `Action`. Subsequent `step()`
  calls return successive waypoints. After the last waypoint, the
  adapter raises a new typed completion signal,
  `ROSRskillGoalSatisfied`.
* **Result-only mode** (`result_trajectory_field is None`, e.g.
  Nav2) — the wrapped server drives actuators on its own; the
  adapter just awaits the result and raises
  `ROSRskillGoalSatisfied` on success. No `Action` chunk is
  emitted to `/openral/candidate_action`.

The `RskillRunnerNode`'s execute-callback loop is extended with a
single specific `except ROSRskillGoalSatisfied:` clause that breaks
the loop cleanly so the `ExecuteSkill` goal closes with
`success=True`. The resolver factory `make_default_skill_resolver(node,
search_paths=[...])` captures the host lifecycle node so wrapped
adapters can build `ActionClient` / service client handles on the same
node — futures share the runner's existing rclpy spin, exactly the
pattern `_maybe_reset_hal_to_starting_pose` already uses.

### What does NOT change

* **`ExecuteRskillTool` stays the only LLM-visible dispatcher.**
  ADR-0018 §4's closed palette is preserved. Wrapped skills appear
  as additional `execute_rskill__<slug>` entries alongside VLAs; the
  per-skill `description` / `actions` / `objects` / `scenes` fields
  (ADR-0022) drive the LLM tool surface.
* **`build_tool_palette` filter is unchanged.** It only consults
  `role`, `embodiment_tags`, `capabilities_required`, and license;
  it never reads `kind` or `model_family`. Wrapped skills with
  `role: s1` + matching embodiment surface naturally. A regression
  test pins this so a future change can't accidentally start
  filtering on `kind`.
* **Safety pipeline routing.** Trajectory-mode wrapped skills route
  every waypoint through `ROSPublishingHAL` → `/openral/candidate_action`
  → safety supervisor → `/openral/safe_action` → HAL. CLAUDE.md §3
  "Python proposes; C++ disposes" still holds.
* **`schema_version`** stays at `"0.1"` — additive change with safe
  defaults, per the manifest docstring's "extended in place"
  precedent (CLAUDE.md §1.6).

## Two reference skills land in the same PR

* [`rskills/rskill-moveit-joints/`](../../rskills/rskill-moveit-joints/)
  — trajectory mode against `moveit_msgs/action/MoveGroup` (introduced here as
  `openral-moveit-plan-arm`; renamed `rskill-moveit-joints` under ADR-0054).
* [`rskills/rskill-nav2-navigate-to-pose/`](../../rskills/rskill-nav2-navigate-to-pose/)
  — result-only mode against `nav2_msgs/action/NavigateToPose`.

Both ship a real `rskill.yaml` + `README.md` that pass the
`openral_cli._rskill_doc_validator` publish gate.

## Safety

The OpenRAL safety supervisor (`packages/openral_safety/openral_safety/
supervisor_node.py`) checks only **row 0** of every `ActionChunk`
today. Packing a multi-waypoint trajectory as one chunk with
`horizon=N` would let rows 1..N actuate unchecked — unacceptable for
a planner whose output is N waypoints. This ADR therefore pins
**`chunk_size = 1`** for wrapped trajectory-mode skills via the
manifest validator, so each waypoint becomes its own `ActionChunk`
and gets its own per-joint envelope check.

We deliberately accept several scope limits in this PR and track each
as a separate follow-up issue:

1. **`safety: supervisor collision check via planning scene`** —
   OpenRAL's supervisor does NOT do collision checking. We trust
   MoveIt's internal FCL pass. A real collision check inside the
   supervisor would need a planning-scene representation, an FCL /
   Bullet dependency, and a sync mechanism with perception — a
   multi-PR effort with its own ADR.
2. **`safety: supervisor velocity / jerk envelope check`** — no
   bound exists today. A planner emitting a rough trajectory
   actuates because the supervisor only checks per-joint position.
3. **`safety: bring Nav2-style cmd_vel under the supervisor`** —
   result-only wrapped skills (Nav2) publish `/cmd_vel` directly,
   bypassing `/openral/candidate_action`. Blocked on (1) a
   mobile-base HAL declaring `body_twist` in
   `supported_control_modes` (none exist in-tree today) and (2) a
   velocity envelope landing in the supervisor.
4. **`hal: add mobile-base HAL (turtlebot4 / jackal) for Nav2
   skill`** — every in-tree HAL is an arm.
5. **`rskill: structured-prompt goals for ros_action skills`** —
   v1 hard-codes the target into `ros_integration.default_goal_json`.
   The follow-up ADR extends `RSkillToolEntry` with a
   JSON-Schema-fragment field and `ExecuteRskillTool` with
   `prompt_metadata_json` so the LLM can specify per-call targets.
6. **`rskill: WAM resolver branch`** — implement the
   `kind: "wam"` dispatch path.

## Out-of-scope (deferred)

* New `ReasonerToolCall` variants. ADR-0018 §4 stays closed.
* `Action.is_terminal` boolean field — rejected in favour of the
  typed `ROSRskillGoalSatisfied` exception, because
  `ROSPublishingHAL._action_to_chunk` builds the wire `ActionChunk`
  field-by-field and would silently drop an extra `Action` field;
  a control-flow exception cannot be ignored.
* Bumping `schema_version` past `"0.1"`. Pre-release iteration
  evolves the surface in place (CLAUDE.md §1.6).
* Collision check inside the OpenRAL supervisor (item 1 above).

## License

Each wrapped-ROS rSkill carries its own `license:` posture
independently of the upstream ROS package's license. MoveIt is
BSD-3-Clause, Nav2 is Apache-2.0; both reference skills in this PR
ship under Apache-2.0 (the wrapper manifest + README). The wrapped
binaries / IDLs are installed via `ros-${ROS_DISTRO}-*` and their
licenses are surfaced by the host distribution, not bundled.

## Tests

Shipped in this PR (see `tests/unit/test_rskill_manifest_kinds.py`,
`tests/unit/test_ros_action_rskill.py`,
`tests/unit/test_palette_kind_filter.py`):

* Schema unit — every kind / forbidden-field combination explicitly
  asserted; the `chunk_size==1` constraint pinned.
* Migration audit — every `rskills/*/rskill.yaml` must declare
  `kind:` explicitly (no default).
* Adapter unit — real `rclpy` action server / service server
  fixtures (no mocks per CLAUDE.md §1.11); trajectory mode
  exercises waypoint replay and `ROSRskillGoalSatisfied`
  termination; result-only mode exercises the Nav2 shape.
* Palette regression — pin that `build_tool_palette` does NOT
  filter on `kind` or `model_family`.

Integration / sim tests (gated on a real MoveIt / Nav2 launch) are
listed in the plan and tracked as follow-ups; this PR's gate is the
unit tier.

## Migration

Mechanical: every existing `rskills/*/rskill.yaml` and
`rskills/template/rskill.yaml` gets one new line `kind: "vla"`
inserted right after the `role:` line. The edit lands in-place — no
migrator stays in tree per CLAUDE.md §1.6. No semantic change to
existing skills.
