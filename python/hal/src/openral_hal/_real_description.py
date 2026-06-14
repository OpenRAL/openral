"""Internal helper to derive a real-hardware ``RobotDescription`` from a sim baseline.

Both the UR (PR #60, ``ur_real.py``) and the Franka / Sawyer / ALOHA (issues
#56, #57, #58) real-HW HALs need to publish a :class:`RobotDescription`
that shares kinematics + safety envelope + capabilities + HAL entrypoints
with a "sim baseline" but flips the ``sdk_kind`` to a closed-with-api license
posture. Since ADR-0031 the sim and real HAL import strings both live in the
shared ``hal: HalEntrypoints`` block (``hal.sim`` / ``hal.real``), so the real
description inherits the same ``hal`` from *base* and only ``sdk_kind`` differs.

This module exposes one helper, :func:`make_real_description`, so every
real-HW adapter spells the derivation the same way and a future contributor
adding a sixth real-HW arm doesn't reinvent the pattern.

The module is **internal** (leading underscore) — it is not re-exported
from :mod:`openral_hal`.

Example:
    >>> from openral_hal._real_description import make_real_description
    >>> from openral_hal import FRANKA_PANDA_DESCRIPTION
    >>> real = make_real_description(
    ...     FRANKA_PANDA_DESCRIPTION,
    ...     sdk_kind="closed_with_api",
    ... )
    >>> real.sdk_kind
    'closed_with_api'
    >>> # HAL entrypoints, kinematics + safety envelope are shared with the baseline:
    >>> real.hal == FRANKA_PANDA_DESCRIPTION.hal
    True
    >>> real.joints == FRANKA_PANDA_DESCRIPTION.joints
    True
    >>> real.safety == FRANKA_PANDA_DESCRIPTION.safety
    True
"""

from __future__ import annotations

from typing import Literal

from openral_core.schemas import RobotDescription

__all__ = ["make_real_description"]


_SdkKind = Literal["open", "closed_with_api", "closed"]


def make_real_description(
    base: RobotDescription,
    *,
    sdk_kind: _SdkKind,
) -> RobotDescription:
    """Return a copy of *base* with the real-HW ``sdk_kind`` license posture.

    The returned :class:`RobotDescription` shares every other field with
    *base* — joint specs, end-effectors, safety envelope, capabilities,
    sensors, ``onboard_compute``, ``observation_spec`` / ``action_spec``, and
    the ``hal`` entrypoints (``hal.sim`` / ``hal.real``, ADR-0031). Only
    ``sdk_kind`` is overridden.

    Implementation is :meth:`pydantic.BaseModel.model_copy` with an
    ``update`` dict; the result is a shallow copy so consumers that mutate
    nested fields (they shouldn't) would mutate *base* too.

    Args:
        base: The sim baseline manifest (already carrying the shared
            ``hal: HalEntrypoints`` block with both sim + real entrypoints).
        sdk_kind: New ``sdk_kind``; one of ``"open"``, ``"closed_with_api"``,
            or ``"closed"`` per the :class:`RobotDescription` schema.

    Returns:
        A new :class:`RobotDescription` with the overridden ``sdk_kind``.
    """
    return base.model_copy(update={"sdk_kind": sdk_kind})
