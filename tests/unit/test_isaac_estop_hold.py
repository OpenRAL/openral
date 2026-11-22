"""A latched robot of a shared Isaac world steps a stop, not a freeze (ADR-0137)."""

from __future__ import annotations

import numpy as np


def test_a_latched_robot_of_a_shared_sidecar_stops_where_it_was_measured() -> None:
    """ADR-0137: arm joints at their measured positions, grippers held, base stopped."""
    from types import SimpleNamespace

    from openral_sim.backends.isaac_sim import IsaacActionLayout, _IsaacSimSidecar

    layout = IsaacActionLayout(
        joints=("x", "j1", "j2", "grip", "mount"),
        roles=("base", "arm", "arm", "gripper", "fixed"),
    )
    fake = SimpleNamespace(_client=SimpleNamespace(robot="a"), layout=layout, action_dim=6)
    fake.idle_action = lambda: _IsaacSimSidecar.idle_action(fake)  # type: ignore[arg-type]
    obs = {"joint_positions": [9.0, 0.1, -0.2, 0.03, 0.0]}
    hold = _IsaacSimSidecar.estop_hold_action(fake, obs)  # type: ignore[arg-type]
    assert hold is not None
    np.testing.assert_allclose(hold[:2], [0.1, -0.2], rtol=0, atol=1e-6)
    assert np.isnan(hold[2]), "the gripper keeps its target: a held carton stays held"
    np.testing.assert_array_equal(hold[3:], [0.0, 0.0, 0.0])
    # A sidecar of its own freezes; an observation that is not every joint is no stop.
    alone = SimpleNamespace(_client=SimpleNamespace(robot=""), layout=layout, action_dim=6)
    assert _IsaacSimSidecar.estop_hold_action(alone, obs) is None  # type: ignore[arg-type]
    assert _IsaacSimSidecar.estop_hold_action(fake, {"joint_positions": [0.1]}) is None  # type: ignore[arg-type]
    assert _IsaacSimSidecar.estop_hold_action(fake, None) is None  # type: ignore[arg-type]
