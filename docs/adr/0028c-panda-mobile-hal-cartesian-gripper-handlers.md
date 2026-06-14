# ADR-0028c: panda_mobile HAL CARTESIAN_DELTA + GRIPPER_POSITION handlers

- Status: **Proposed**
- Date: 2026-05-27
- Related: [ADR-0028](0028-rskill-action-contract-slots.md) (parent —
  sub-ADR split rationale); [ADR-0028b](0028b-rskill-action-contract-slots-dispatch.md)
  (slot dispatch — emits the typed chunks this ADR consumes);
  [ADR-0025](0025-reasoner-managed-background-services.md) (panda_mobile
  HAL already accepts ``JOINT_POSITION`` + ``BODY_TWIST``); CLAUDE.md
  §3 (HAL is layer 0 — the boundary that turns typed bytes into
  motor commands).

## Context

ADR-0028b's slot dispatch emits typed ``ActionChunk``s with
``control_mode ∈ {CARTESIAN_DELTA, GRIPPER_POSITION, BODY_TWIST,
JOINT_POSITION}`` per the RoboCasa pi0.5 / rldx1 manifests'
``action_contract.slots`` blocks. The supervisor's per-mode envelope
(ADR-0028b step 5) validates each chunk; ``/openral/safe_action``
relays them to ``openral_hal_panda_mobile``. But today's HAL whitelist
only accepts ``JOINT_POSITION`` + ``BODY_TWIST`` (lifecycle_node.py:752),
so the cartesian + gripper chunks were dropped on arrival with a
``warn`` log — the demo couldn't actuate.

This ADR opens the HAL's whitelist to the remaining two surfaces and
adds the matching handlers on both the in-memory digital-twin
(``PandaMobileHAL``) and the sim-attached (``SimAttachedHAL`` →
robosuite) paths.

## Decision

### Lifecycle node decoder (`packages/openral_hal_panda_mobile/.../lifecycle_node.py`)

`_on_safe_action` was building every Action with
``joint_targets=[flat[:n_dof]]`` regardless of ``control_mode`` — the
legacy lie that conflated cartesian / gripper / body-twist payloads
onto one field. After this ADR the decoder builds a typed Action per
mode:

- ``JOINT_POSITION``  → ``Action(joint_targets=[row])``
- ``BODY_TWIST``      → ``Action(body_twist=[tuple(row)], frame_id=…)``
- ``CARTESIAN_DELTA`` → ``Action(cartesian_delta=[tuple(row)], ee_name=…, frame_id=…)``
- ``GRIPPER_POSITION``→ ``Action(gripper=row, ee_name=…)``

The whitelist grows accordingly:

```python
accepted = {
    ControlMode.JOINT_POSITION,
    ControlMode.BODY_TWIST,
    ControlMode.CARTESIAN_DELTA,
    ControlMode.GRIPPER_POSITION,
}
```

Each non-joint mode checks ``n_dof`` matches its expected width (6
for cartesian / twist, 1 for gripper) and drops the chunk with a
typed warn log on mismatch.

### Nav2 ``/cmd_vel`` bridge migration

``_on_cmd_vel`` previously also packed its body twist into
``joint_targets=[row]``. Updated to use the typed ``body_twist`` field
so the entire HAL pipeline stops carrying cartesian / twist data in
the joint slot.

### Digital-twin `PandaMobileHAL` (`python/hal/src/openral_hal/panda_mobile.py`)

Two new ``_apply_*`` methods:

- ``_apply_cartesian_delta(row: list[float])``: stamps the latest
  commanded 6-vec OSC delta onto ``self._last_cartesian_delta`` for
  dashboard observability. The digital twin has no Jacobian / forward
  kinematics, so ``_qpos`` is unchanged — physical motion lives in
  the sim-attached path. This lets the lifecycle node continue
  publishing ``JointState`` while the dashboard can overlay
  command-vs-reality on the OSC channel.
- ``_apply_gripper_position(width: float)``: writes the trailing
  ``_qpos`` slot (index ``len(PANDA_MOBILE_JOINT_NAMES) - 1`` — 10
  today, the ``panda_gripper`` joint added in ADR-0028a). No clamping
  — the safety supervisor's ``gripper_min`` / ``gripper_max`` envelope
  (ADR-0028b step 5) already validates the input.

``_apply_joint_position`` gains a third accepted width (11 = base +
arm + gripper) alongside the existing 7 (arm-only) and 10 (base +
arm) shapes. The 10-wide form is preserved for MoveIt trajectory
replay; new policies emitting the full 11-wide vector get the
gripper slot honoured.

``PANDA_MOBILE_JOINT_NAMES`` grows by one (``panda_gripper``) so
``_qpos`` is sized 11 and the published ``JointState`` matches
``robots/panda_mobile/robot.yaml`` after ADR-0028a. This is the
ADR-0028a drift fix that should travel with the gripper-joint
addition.

### Sim-attached `SimAttachedHAL` (`python/hal/src/openral_hal/sim_attached.py`)

``send_action``:

- ``BODY_TWIST``: read from ``action.body_twist[0]`` (was
  ``action.joint_targets[0]`` — pre-0028c legacy lie). Continues to
  bypass ``env.step`` and write the three base qpos slots directly.
- ``CARTESIAN_DELTA`` + ``GRIPPER_POSITION``: route through
  ``pack_action_for_env`` and ``env.step`` so robosuite's composite
  controller (OSC arm + gripper actuator) does the physics.

``pack_action_for_env`` grows two new modes:

- ``CARTESIAN_DELTA``: read ``action.cartesian_delta[0]`` (6-vec OSC),
  fill env slots ``[3:9]`` (the robosuite PandaMobile composite's arm
  OSC slots). Base + gripper slots stay zero on this chunk; the SimAttachedHAL
  must merge with prior chunks if a tick wants combined surfaces.
- ``GRIPPER_POSITION``: read ``action.gripper[0]`` (1-vec), fill the
  trailing env slot (gripper actuator). Arm + base stay zero.

### Multi-chunk-per-tick semantics

The slot dispatcher (ADR-0028b step 3) emits 3 typed ``ActionChunk``s
per RoboCasa tick (one each for arm cartesian / gripper / base twist).
Each chunk drives one ``send_action`` call on the HAL, which means
robosuite's ``env.step`` runs up to 3 times per OpenRAL tick (the
base-twist path bypasses ``env.step`` and writes qpos directly).

This is acceptable for the demo — physics advances at 3× the OpenRAL
tick rate, but the policy still observes at the configured rate so
the closed loop remains stable. A later refactor will introduce a
per-tick action accumulator (collect chunks by ``trace_id``, flush
to a single ``env.step`` on trace boundary) so the physics step rate
matches the policy rate exactly. Tracked as ADR-0028d follow-up; not
in this ADR's scope.

## Consequences

**Positive**

- The trace at the top of ADR-0028 runs to completion: the 12-D
  RoboCasa policy vector splits → 3 typed chunks → 3 ``env.step`` calls
  per tick → arm moves via OSC, gripper opens/closes, base translates.
  The kettle (eventually) gets picked up.
- ``Action.body_twist`` / ``Action.cartesian_delta`` / ``Action.gripper``
  fields are now load-bearing across the entire stack — no more
  joint_targets-everywhere lie.
- ``PandaMobileHAL`` joint inventory matches its robot.yaml after the
  ADR-0028a drift fix; published ``JointState`` is correctly sized for
  the 11-DoF chain.

**Negative / cost**

- ``env.step`` rate ≠ policy rate during multi-chunk ticks. Demo runs
  but physics is "ahead" of policy timestamps by a few sim cycles per
  OpenRAL tick. Tracked for ADR-0028d.
- The digital-twin HAL's CARTESIAN_DELTA path is Jacobian-free; it
  observes the command but doesn't actuate. Operators running deploy
  paths against the digital twin (no robosuite) will see arm pose
  unchanged. This is the correct behaviour for a pure twin — real
  motion needs a kinematic chain we don't carry in-process — but it
  warrants a one-line operator-doc clarification.
- The sim-attached path assumes the env's action vector layout matches
  the robosuite ``PandaMobile`` BASIC composite + OSC arm + gripper
  composition (slots 0-2 = base, 3-8 = arm OSC, last = gripper). Other
  robosuite configurations (e.g. with a torso slot) need a
  per-env-action-dim packer plug-in. Not blocking RoboCasa.

**Out of scope**

- Real-hardware HALs. ``openral_hal_panda_mobile_real`` doesn't exist
  yet — when it does (libfranka FCI + Nav2 stack), it gets its own
  per-mode dispatch.
- The ``env.step``-rate fix. ADR-0028d (action-accumulator).
- BODY_TWIST through ``env.step`` (today bypasses for valid reasons —
  see the existing comment in ``sim_attached.py``). The slot-dispatched
  BODY_TWIST chunk continues to write qpos directly.

## Implementation sequence

Per CLAUDE.md §4.2:

1. **`docs(adr): ADR-0028c`** — this file.
2. **`feat(hal): panda_mobile lifecycle decoder + digital-twin + sim-attached typed dispatch`** —
   the substantive change. Lifecycle node + digital twin + sim-attached
   all in one commit because they share the typed-Action contract and
   would break each other partially. Tests cover all three layers.
   ``PANDA_MOBILE_JOINT_NAMES`` migration from 10 → 11 lives here
   (ADR-0028a drift fix arriving with the gripper handler).
