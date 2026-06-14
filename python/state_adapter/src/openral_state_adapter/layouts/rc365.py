"""``rc365`` layout assembler (ADR-0027).

``rc365`` and :mod:`openral_state_adapter.layouts.human300_16d` describe
the **same 16-D physical state** — ``[base_to_eef.pos(3),
base_to_eef.quat(4), world_to_base.pos(3), world_to_base.quat(4),
gripper_qpos(2)] = 16``. The difference is purely how downstream policy
adapters slice that vector into modality keys:

* ``human300_16d`` consumers (e.g. lerobot pi0.5 ``observation.state``)
  read the flat 16-D vector as-is.
* ``rc365`` consumers (RLDX-1 GENERAL_EMBODIMENT) re-slice the same 16
  bytes into 5 separate state keys at the policy adapter boundary
  (``_RC365_STATE_SLICES_FROM_HUMAN300`` in
  ``openral_sim.policies.rldx``):

    state.end_effector_position_relative ← bytes[0:3]
    state.end_effector_rotation_relative ← bytes[3:7]
    state.base_position                  ← bytes[7:10]
    state.base_rotation                  ← bytes[10:14]
    state.gripper_qpos                   ← bytes[14:16]

The slicing is the policy's concern, not the state assembler's. The
assembler's job is just to produce the 16-D physical state — identical
work for both layouts — so register the existing
:func:`openral_state_adapter.layouts.human300_16d.assemble_human300_16d`
under the ``rc365`` key as well. New layouts that genuinely differ in
shape (field order, frame convention, gripper encoding) still get their
own assembler file.
"""

from __future__ import annotations

from openral_state_adapter._registry import register
from openral_state_adapter.layouts.human300_16d import assemble_human300_16d

register("rc365", assemble_human300_16d)
