"""MuJoCo HAL adapter for the SO-100 follower arm (digital twin via Menagerie).

This module wraps the upstream DeepMind ``mujoco_menagerie`` SO-100 MJCF
(``trs_so_arm100/so_arm100.xml``, vendored via ``robot_descriptions``) as a
:class:`openral_hal.HAL` Protocol implementation, mirroring the
:class:`openral_hal.UR5eHAL` / :class:`openral_hal.FrankaPandaHAL` pattern.

It complements two existing SO-100 paths in the repo:

* :class:`openral_hal.SO100FollowerHAL` — talks to the **real** lerobot
  driver over USB serial.  Production path.
* :class:`openral_hal.SO100DigitalTwin` — kinematic-only in-process state
  holder (no physics).  Used as a drop-in for ``SO100FollowerHAL`` in
  unit tests.

This module adds the **third leg**: a real-physics MuJoCo twin reusable
under ``tests/sim/`` to validate the 6-DoF action contract end-to-end
before the physical arm is connected.  See CLAUDE.md §1.11 ("real
component or ``pytest.skip`` — nothing in between").

Joint inventory
---------------
The menagerie XML and the canonical :data:`openral_hal.SO100_DESCRIPTION`
use different names — the menagerie follows the SO-ARM-100 mechanical
naming (``Rotation``, ``Pitch``, ``Elbow``, …), while lerobot and the
description use functional names (``shoulder_pan``, ``shoulder_lift``,
``elbow_flex``, …).  This module maps between the two; the description
joint order is preserved on the public ``read_state`` / ``send_action``
surface.

============== ===============  ==============================
description    menagerie joint  qpos / actuator idx (menagerie)
============== ===============  ==============================
shoulder_pan   Rotation         0
shoulder_lift  Pitch            1
elbow_flex     Elbow            2
wrist_flex     Wrist_Pitch      3
wrist_roll     Wrist_Roll       4
gripper        Jaw              5 (revolute jaw rotation)
============== ===============  ==============================

The MJCF has 6 ``position`` actuators in the same order; the gripper is a
revolute joint with range ``[-0.174, 1.75]`` rad which is normalised to
``[0, 1]`` on the public surface (0 = closed, 1 = fully open) so the
``SO100_DESCRIPTION`` gripper contract is honoured.

Example:
    >>> from openral_hal import SO100MujocoHAL, SO100_DESCRIPTION
    >>> hal = SO100MujocoHAL(gravity_enabled=False)  # doctest: +SKIP
    >>> hal.connect()  # doctest: +SKIP
    >>> state = hal.read_state()  # doctest: +SKIP
    >>> len(state.position) == len(SO100_DESCRIPTION.joints)  # doctest: +SKIP
    True
    >>> hal.disconnect()  # doctest: +SKIP
"""

from __future__ import annotations

from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal.so100_follower import SO100_DESCRIPTION

__all__ = ["SO100MujocoHAL"]


# ── HAL ──────────────────────────────────────────────────────────────────────


class SO100MujocoHAL(MujocoArmHAL):
    """HAL adapter for the SO-100 follower arm (MuJoCo-backed simulation).

    The HAL exposes the canonical 6 SO-100 joints (5 arm + 1 gripper) on
    its public surface, using the lerobot-style names from
    :data:`openral_hal.SO100_DESCRIPTION`.  Internally it drives the
    Menagerie MJCF's 6 position actuators (``Rotation``, ``Pitch``,
    ``Elbow``, ``Wrist_Pitch``, ``Wrist_Roll``, ``Jaw``) and maps the
    revolute Jaw range ``[-0.174, 1.75]`` rad to a normalised ``[0, 1]``
    gripper channel — so the same 6-DoF action chunk drives the sim twin
    and the real hardware HAL (:class:`openral_hal.SO100FollowerHAL`)
    identically.

    The MJCF position limits are tighter than the conservative
    ``[-π, π]`` declared on :data:`openral_hal.SO100_DESCRIPTION`; commands
    outside the MJCF range are clipped by MuJoCo's own position
    controllers.  Tests should command targets inside the menagerie
    range (e.g. ``Rotation`` is ``[-1.92, 1.92]``).

    Args:
        mjcf_path: Optional override for the MJCF file path.  When
            ``None``, the file is fetched lazily from
            ``robot_descriptions`` (``mujoco_menagerie/trs_so_arm100/so_arm100.xml``).
        settle_steps: Number of MuJoCo physics steps performed in
            :meth:`send_action`.  Defaults to ``1``; raise it in tests
            that assert the arm has settled at the commanded pose.
        gravity_enabled: When ``False``, gravity is zeroed at
            ``connect()`` time for deterministic closed-loop tests.
        staleness_limit_s: Maximum age of a cached state.

    Example:
        >>> from openral_hal import SO100MujocoHAL  # doctest: +SKIP
        >>> hal = SO100MujocoHAL(gravity_enabled=False)  # doctest: +SKIP
        >>> hal.connect()  # doctest: +SKIP
        >>> state = hal.read_state()  # doctest: +SKIP
        >>> len(state.position)  # 5 arm + 1 gripper  # doctest: +SKIP
        6
        >>> hal.disconnect()  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        mjcf_path: str | None = None,
        settle_steps: int = 1,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Initialise the SO-100 MuJoCo HAL; no MuJoCo state is created until ``connect()``.

        All wiring (MJCF URI, joint indices, Jaw ``affine_low_high`` gripper
        read mode) lives in :data:`SO100_DESCRIPTION.sim` (ADR-0023).
        """
        self._init_from_description(
            SO100_DESCRIPTION,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )
