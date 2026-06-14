"""``human300_16d`` layout assembler (ADR-0027).

Mirrors the RoboCasa365 ``pi05_pretrain_human300`` training-time layout
verbatim — verified against
``python/sim/src/openral_sim/backends/robocasa.py:508``:

    base_to_eef_pos  (3) +
    base_to_eef_quat (4) +
    base_pos         (3) +
    base_quat        (4) +
    gripper_qpos     (2) = 16

The pi05 checkpoint distributed at:

* ``hf://OpenRAL/rskill-pi05-robocasa365-human300-nf4``

consumes this layout. Picking the wrong field order silently feeds a
quaternion component into a gripper slot — the gripper finger angles
occupy the LAST two dims (dims 12-13 are the base quaternion), so the
concatenation order above is load-bearing.
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


_DIM = 16
_N_GRIPPER_JOINTS = 2


def _canonicalize_quat_xyzw(
    quat_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Pick the positive-w hemisphere for a unit quaternion.

    ``q`` and ``-q`` represent the same rotation but encode different
    bytes, so two equally-correct sources (RoboCasa proprio vs ROS TF)
    can disagree on sign and produce a state vector that differs by
    `max_abs_diff = 2` on the quaternion slots even though the
    physical pose is identical. Canonicalising to ``w >= 0`` makes
    both sources land on the same hemisphere; for the ``w == 0``
    edge case (180 deg rotations) the first non-zero component of
    ``(x, y, z)`` is forced positive instead, which keeps the rule
    deterministic and side-agnostic.

    Apply at every site that materialises a quaternion into the
    state vector the policy consumes (deploy_sim's assembler AND
    sim_run's ``_wrap_obs`` — the policy was trained against raw
    RoboCasa output and is robust to either hemisphere, but a
    consistent convention makes the dump-diff regression test
    actually pin a sign drift instead of treating it as expected
    noise).
    """
    x, y, z, w = quat_xyzw
    # Pick the hemisphere where the first non-zero component of
    # ``(w, x, y, z)`` is positive. For the typical ``|w| > 0`` case
    # this reduces to "force w >= 0"; the ``w == 0`` edge falls
    # through to (x, y, z) in order.
    for comp in (w, x, y, z):
        if comp > 0.0:
            return quat_xyzw
        if comp < 0.0:
            return (-x, -y, -z, -w)
    return quat_xyzw  # identity quaternion (0, 0, 0, 0) — degenerate but harmless


def _quat_to_layout(
    quat_xyzw: tuple[float, float, float, float], convention: str
) -> tuple[float, ...]:
    """Permute the canonical ``xyzw`` quaternion to the layout's convention.

    ROS / TF2 emit ``xyzw``. Checkpoints trained on a ``wxyz`` convention
    (e.g. some upstream openpi configs) need a single permutation at the
    boundary. Done here in the assembler so downstream code can assume
    "whatever the bindings declared".

    Sign canonicalisation runs BEFORE the convention permutation so
    both ``xyzw`` and ``wxyz`` consumers see the same hemisphere.
    """
    canonical = _canonicalize_quat_xyzw(quat_xyzw)
    if convention == "wxyz":
        x, y, z, w = canonical
        return (w, x, y, z)
    return canonical  # "xyzw" — pass through


def assemble_human300_16d(
    bindings: StateContractBindings,
    joint_positions: dict[str, float],
    tf_lookup: TfLookup,
) -> NDArray[np.float32]:
    """Assemble the 16-D human300 state vector from live TF + JointState.

    Args:
        bindings: Manifest-declared per-robot bindings. MUST carry
            ``eef_frame``, ``base_frame``, ``world_frame`` (defaults
            to ``"map"``) and 2 entries in ``gripper_qpos_joints``.
        joint_positions: ``JointState.name`` → position, all joints
            observed in the latest ``/joint_states`` message.
        tf_lookup: Callable returning ``TransformView`` (position +
            quaternion_xyzw) for a ``(target, source)`` frame pair.

    Returns:
        16-D ``np.float32`` vector ordered:
        ``[base_to_eef.pos(3), base_to_eef.quat(4), world_to_base.pos(3),
        world_to_base.quat(4), gripper_qpos(2)]``.

    Raises:
        ROSConfigError: When ``bindings`` is missing required frames
            (the manifest validator catches this at install time; this
            guard surfaces a corrupt registry).
        KeyError: When a gripper joint named in ``bindings`` isn't in
            ``joint_positions`` — surfaces a sim/HAL/topic-name skew
            instead of silently zero-filling the gripper slot.
    """
    if bindings.eef_frame is None or bindings.base_frame is None:
        raise ROSConfigError(
            "assemble_human300_16d: bindings.eef_frame and bindings.base_frame "
            f"are required (got eef_frame={bindings.eef_frame!r}, "
            f"base_frame={bindings.base_frame!r}). The manifest validator "
            "normally catches this at install time."
        )
    world_frame = bindings.world_frame or "map"

    # `base_to_eef`: ``target=base_frame``, ``source=eef_frame``. The
    # ``geometry_msgs/TransformStamped`` semantics are "transform such
    # that a point in source_frame is mapped to a point in target_frame"
    # — so the translation field IS the position of the source frame's
    # origin expressed in the target frame, which is what the checkpoint
    # expects in the ``base_to_eef`` slot.
    base_to_eef = tf_lookup(bindings.base_frame, bindings.eef_frame)
    world_to_base = tf_lookup(world_frame, bindings.base_frame)

    convention = bindings.quaternion_convention
    eef_q = _quat_to_layout(base_to_eef.quaternion_xyzw, convention)
    base_q = _quat_to_layout(world_to_base.quaternion_xyzw, convention)

    # human300_16d's gripper slot is 2-D (finger1, finger2). Two sources
    # are supported:
    #   * Two named joints — the manifest binds both finger qpos joints
    #     explicitly. Used when the underlying RobotDescription exposes
    #     each MJCF finger as a separate joint.
    #   * One named joint — the manifest binds a single parallel-gripper
    #     joint (openral's canonical 1-DoF abstraction over a robosuite
    #     ``two_finger`` mimic). Mirror it to ``[v, -v]`` to reconstruct
    #     the training-distribution shape. Matches robosuite's franka
    #     ``gripper0_finger_joint1`` / ``gripper0_finger_joint2`` parity
    #     (opposite-sign open/close on the parallel mechanism).
    n_joints = len(bindings.gripper_qpos_joints)
    if n_joints == _N_GRIPPER_JOINTS:
        gripper = [joint_positions[name] for name in bindings.gripper_qpos_joints]
    elif n_joints == 1:
        v = joint_positions[bindings.gripper_qpos_joints[0]]
        gripper = [v, -v]
    else:
        raise ROSConfigError(
            "assemble_human300_16d: human300_16d expects 1 (parallel "
            f"gripper abstraction) or {_N_GRIPPER_JOINTS} (per-finger) "
            "gripper joints; manifest declared "
            f"{n_joints}: {bindings.gripper_qpos_joints!r}.",
        )

    out = np.empty(_DIM, dtype=np.float32)
    out[0:3] = base_to_eef.position
    out[3:7] = eef_q
    out[7:10] = world_to_base.position
    out[10:14] = base_q
    out[14:16] = gripper
    return out


register("human300_16d", assemble_human300_16d)
