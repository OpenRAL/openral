"""``libero_eef8d`` layout assembler.

Mirrors LIBERO's 8-D task-space proprio state verbatim — verified against
``python/sim/src/openral_sim/backends/libero.py:219``
(``LiberoBackend._wrap_obs``)::

    eef_pos        (3) +   # world-frame EE position
    eef_axisangle  (3) +   # world-frame EE orientation, axis-angle
    gripper_qpos   (2) = 8

Consumed by the LIBERO-finetuned checkpoints ``hf://lerobot/smolvla_libero``,
``hf://OpenRAL/rskill-pi05-franka_panda-libero_spatial-int8``, and
``hf://OpenRAL/rskill-xvla-franka_panda-libero_spatial-bf16``.

The benchmark path (``openral sim run``) reads this state from the LIBERO
env directly; the deploy path (``openral deploy sim``) assembles it here
from live TF + ``JointState``. Without this layout the skill_runner falls
back to the raw joint-space vector, driving an EE-pose-trained policy
incoherently. Task-space sibling of
:mod:`~openral_state_adapter.layouts.human300_16d` (base-relative, 16-D,
mobile base); LIBERO is fixed-base, so EE pose is absolute in the world
frame, matching robosuite's ``robot0_eef_pos``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from openral_core import ROSConfigError

from openral_state_adapter._registry import register

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from openral_core import StateContractBindings

    from openral_state_adapter._protocol import TfLookup


_DIM = 8
_N_GRIPPER_JOINTS = 2
_EPS = 1e-10


def _quat_xyzw_to_axisangle(
    quat_xyzw: tuple[float, float, float, float],
) -> NDArray[np.float32]:
    """``(x, y, z, w)`` quaternion → ``(3,)`` axis-angle vector.

    Byte-for-byte the same formula as the benchmark's ``_quat_to_axisangle``
    (``openral_sim.backends.libero``) so the deploy state matches the training
    distribution: ``angle = 2·acos(w)``, ``axis = (x, y, z) / sqrt(1 - w**2)``,
    output ``axis · angle``; the near-identity rotation (``den ≤ eps``) maps to
    the zero vector.
    """
    x, y, z, w = quat_xyzw
    w = float(np.clip(w, -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den > _EPS:
        angle = 2.0 * float(np.arccos(w))
        axis = np.asarray((x, y, z), dtype=np.float32) / den
        return (axis * angle).astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def assemble_libero_eef8d(
    bindings: StateContractBindings,
    joint_positions: dict[str, float],
    tf_lookup: TfLookup,
) -> NDArray[np.float32]:
    """Assemble the 8-D LIBERO task-space state from live TF + JointState.

    Args:
        bindings: Manifest-declared per-robot bindings. MUST carry
            ``eef_frame`` and 1 or 2 ``gripper_qpos_joints``. ``world_frame``
            defaults to ``"world"`` (robosuite's root) when unset.
        joint_positions: ``JointState.name`` → position for every joint in the
            latest ``/joint_states`` message.
        tf_lookup: Callable returning a ``TransformView`` (position +
            ``quaternion_xyzw``) for a ``(target, source)`` frame pair.

    Returns:
        8-D ``np.float32`` vector ordered ``[eef_pos(3), eef_axisangle(3),
        gripper_qpos(2)]``.

    Raises:
        ROSConfigError: ``eef_frame`` missing, or the gripper-joint count is
            neither 1 (parallel-gripper abstraction, mirrored to ``[v, -v]``)
            nor 2 (per-finger). The manifest validator normally catches this
            at install time.
        KeyError: A gripper joint named in ``bindings`` isn't in
            ``joint_positions`` — surfaces a sim/HAL/topic-name skew instead of
            silently zero-filling the gripper slot.
    """
    if bindings.eef_frame is None:
        raise ROSConfigError(
            "assemble_libero_eef8d: bindings.eef_frame is required (the layout "
            "reads the world-frame EE pose). The manifest validator normally "
            "catches this at install time."
        )
    # Validate the gripper binding BEFORE any TF I/O — a config error must not
    # depend on TF availability.
    n_joints = len(bindings.gripper_qpos_joints)
    if n_joints == _N_GRIPPER_JOINTS:
        gripper = [joint_positions[name] for name in bindings.gripper_qpos_joints]
    elif n_joints == 1:
        v = joint_positions[bindings.gripper_qpos_joints[0]]
        gripper = [v, -v]
    else:
        raise ROSConfigError(
            "assemble_libero_eef8d: libero_eef8d expects 1 (parallel gripper "
            f"abstraction) or {_N_GRIPPER_JOINTS} (per-finger) gripper joints; "
            f"manifest declared {n_joints}: {bindings.gripper_qpos_joints!r}.",
        )

    # ``world_to_eef``: EE origin in world frame, matching robosuite's
    # ``robot0_eef_pos``. ``world_frame`` binding defaults to ``"map"``
    # (SLAM root) but LIBERO is fixed-base with no SLAM — manifest MUST
    # set ``world_frame`` to the HAL-published sim root (e.g. ``"world"``);
    # ``"map"`` would be unavailable or meters-off on a fresh boot.
    world_to_eef = tf_lookup(bindings.world_frame, bindings.eef_frame)

    out = np.empty(_DIM, dtype=np.float32)
    out[0:3] = np.asarray(world_to_eef.position, dtype=np.float32)
    out[3:6] = _quat_xyzw_to_axisangle(world_to_eef.quaternion_xyzw)
    out[6:8] = np.asarray(gripper, dtype=np.float32)
    return out


register("libero_eef8d", assemble_libero_eef8d)
