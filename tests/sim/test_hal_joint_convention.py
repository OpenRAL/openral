"""Joint-convention guard: a HAL's published joints must obey its own kinematic limits.

The safety kernel's collision FK is built from each robot's ``robot.yaml`` joint
chain (axes + ``position_limits``, ADR-0030). If the HAL publishes a joint value
in a DIFFERENT sign/axis convention than that model, the kernel forward-kinematics
a configuration the real robot is not in — a mirror-image arm — which both
false-positives self-collision E-stops AND can MISS real collisions (GH #13).

Such a mismatch is detectable on any joint whose declared ``position_limits`` are
entirely one sign (e.g. the Franka ``panda_joint4`` range ``[-3.0718, -0.0698]``):
a sign flip lands the published value OUTSIDE the declared range. This module
asserts the cheap, robust invariant — **every HAL-published joint position lies
within that joint's declared ``position_limits``** — so a convention regression
trips a test instead of silently feeding the safety kernel a wrong pose.

Coverage today:
* ``franka_panda`` (native mujoco_menagerie panda) — runs here; the raw-qpos read
  matches the URDF-derived model, so it PASSES (the baseline / methodology check).
* ``panda_mobile`` (robocasa-backed) — confirmed to VIOLATE this on ``panda_joint4``
  (publishes ``+2.6`` rad, out of ``[-3.0718, -0.0698]``): the robosuite/robocasa
  MJCF joint4 convention disagrees with the kernel's URDF FK model. Tracked in
  GH #13 (safety-WG). Asserting it here needs the robocasa env; see that issue for
  the full FK-vs-MuJoCo-xpos verification.
"""

from __future__ import annotations

import pytest

try:
    import mujoco  # noqa: F401
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

try:
    from robot_descriptions import panda_mj_description as _panda_desc

    _ = _panda_desc.MJCF_PATH  # triggers lazy clone / cache lookup
    _MJCF_ERROR: str | None = None
except Exception as exc:
    _MJCF_ERROR = str(exc)

from openral_core import Action, ControlMode, RobotDescription
from openral_hal import FrankaPandaHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
    pytest.mark.skipif(_MJCF_ERROR is not None, reason=f"Panda MJCF unavailable: {_MJCF_ERROR}"),
]

# Tolerance on the declared limit boundary (rad). MuJoCo clamps limited joints to
# their range, so a faithful read sits inside; this only forgives float epsilon.
_LIMIT_TOL_RAD = 1e-3


def assert_joints_within_declared_limits(state: object, description: RobotDescription) -> None:
    """Assert each non-gripper joint position is within its declared limits.

    The invariant the kernel FK depends on: the HAL publishes joints in the same
    convention the ``robot.yaml`` model declares. A sign/axis flip on an
    asymmetric-range joint lands the value outside ``position_limits``.
    """
    arm_joints = [j for j in description.joints if j.role != "gripper" and j.position_limits]
    names = list(state.name)  # type: ignore[attr-defined]
    positions = list(state.position)  # type: ignore[attr-defined]
    by_name = dict(zip(names, positions, strict=False))
    violations: list[str] = []
    for joint in arm_joints:
        if joint.name not in by_name:
            continue
        pos = by_name[joint.name]
        lo, hi = joint.position_limits  # type: ignore[misc]
        if not (lo - _LIMIT_TOL_RAD <= pos <= hi + _LIMIT_TOL_RAD):
            violations.append(f"{joint.name}={pos:+.4f} outside [{lo:+.4f}, {hi:+.4f}]")
    assert not violations, (
        "HAL published joint(s) outside the robot.yaml position_limits — the "
        "kernel collision FK would run on a wrong (mirror-image) pose:\n  "
        + "\n  ".join(violations)
    )


def test_franka_panda_published_joints_obey_declared_limits() -> None:
    """Native Franka (menagerie) read is convention-consistent — the baseline.

    Commands a valid in-limits pose (joint4 negative, per its ``[-3.0718,
    -0.0698]`` range) and reads it back: a faithful raw-qpos read lands inside
    the declared limits AND round-trips the commanded sign. A flipped convention
    (the panda_mobile/robocasa bug, GH #13) would read joint4 back positive,
    out of range.
    """
    # In-limits target: joint4 = -2.0 ∈ [-3.0718, -0.0698]; others within range.
    target = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7, 0.0]
    hal = FrankaPandaHAL(gravity_enabled=False, settle_steps=1500)
    hal.connect()
    try:
        hal.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[target])
        )
        state = hal.read_state()
        # Round-trips the commanded sign (would flip under a convention bug)...
        assert state.position[3] == pytest.approx(target[3], abs=1e-2)
        # ...and stays within the robot.yaml declared limits.
        assert_joints_within_declared_limits(state, hal.description)
    finally:
        hal.disconnect()
