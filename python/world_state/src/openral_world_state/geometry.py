"""Shared gaze geometry — re-export shim (ADR-0044; relocated to core in ADR-0065).

The look-at math now lives in :mod:`openral_core.geometry` so every layer —
including the layer-0 HAL camera rig — can compute camera orientations from one
source without a backward dependency on world-state (layer 2). This module
re-exports it verbatim so existing ``from openral_world_state.geometry import …``
call sites (sim composers, the look-at rSkill, tests) keep working unchanged.
"""

from __future__ import annotations

from openral_core.geometry import (
    ViewAxis,
    compute_gaze_pose,
    look_at_quat_wxyz,
    rotation_to_quat_wxyz,
)

__all__ = ["ViewAxis", "compute_gaze_pose", "look_at_quat_wxyz", "rotation_to_quat_wxyz"]
