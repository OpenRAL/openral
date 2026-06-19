"""Built-in VLA / policy adapters for sim eval — registered at import time.

Mirrors :mod:`openral_sim.backends` but for the policy half of the
``(robot × scene × task × VLA)`` quad. To add a new policy, drop a module
here and register a factory in :data:`openral_sim.POLICIES`.

The factories themselves are responsible for lazily importing heavy backends
(torch, lerobot, openpi-client, …) so installing ``openral-sim`` never
pulls those transitively.
"""

from __future__ import annotations


def _register_policies() -> None:
    """Import side-effect modules that register policy factories."""
    # Order matters only for error-message friendliness.
    from openral_sim.policies import (
        act,
        diffusion,
        gr00t,
        mock,
        molmoact2,
        pi05,
        rlbench_3dda,
        rldx,
        robots,
        smolvla,
        xvla,
    )


_register_policies()
