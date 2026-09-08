"""``rc365`` layout assembler.

``rc365`` and ``openral_state_adapter.layouts.human300_16d`` describe the
same 16-D physical state — ``[base_to_eef.pos(3), base_to_eef.quat(4),
world_to_base.pos(3), world_to_base.quat(4), gripper_qpos(2)] = 16``. They
differ only in how downstream policy adapters slice it:

* ``human300_16d`` consumers (e.g. lerobot pi0.5 ``observation.state``) read
  the flat vector as-is.
* ``rc365`` consumers (RLDX-1 GENERAL_EMBODIMENT) re-slice the same bytes
  into 5 state keys at the policy adapter boundary
  (``_RC365_STATE_SLICES_FROM_HUMAN300`` in ``openral_sim.policies.rldx``):
  ``end_effector_position_relative``[0:3],
  ``end_effector_rotation_relative``[3:7], ``base_position``[7:10],
  ``base_rotation``[10:14], ``gripper_qpos``[14:16].

Slicing is the policy's concern, not the assembler's, so this module just
registers the existing
``assemble_human300_16d``
under the ``rc365`` key. New layouts that differ in shape (field order,
frame convention, gripper encoding) still get their own assembler file.
"""

from __future__ import annotations

from openral_state_adapter._registry import register
from openral_state_adapter.layouts.human300_16d import assemble_human300_16d

register("rc365", assemble_human300_16d)
