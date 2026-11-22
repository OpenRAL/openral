"""A latched robot of a shared Isaac world steps a hold, not a freeze (ADR-0137)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from openral_sim.backends.isaac_sim import IsaacActionLayout, _IsaacSimSidecar


def _fake(robot: str) -> SimpleNamespace:
    layout = IsaacActionLayout(
        joints=("x", "j1", "j2", "grip", "mount"),
        roles=("base", "arm", "arm", "gripper", "fixed"),
    )
    fake = SimpleNamespace(_client=SimpleNamespace(robot=robot), layout=layout, action_dim=6)
    fake.idle_action = lambda: _IsaacSimSidecar.idle_action(fake)  # type: ignore[arg-type]
    return fake


def test_a_latched_robot_of_a_shared_sidecar_holds_and_stops_its_base() -> None:
    hold = _IsaacSimSidecar.estop_hold_action(_fake("a"), None)  # type: ignore[arg-type]
    assert hold is not None
    assert np.all(np.isnan(hold[:3])), "every joint target kept: nothing released or re-aimed"
    np.testing.assert_array_equal(hold[3:], [0.0, 0.0, 0.0])


def test_a_latched_robot_alone_in_its_sidecar_freezes() -> None:
    assert _IsaacSimSidecar.estop_hold_action(_fake(""), None) is None  # type: ignore[arg-type]
