"""Joint-convention guard: a HAL's published joints must obey its own kinematic limits.

The safety kernel's collision FK is built from each robot's ``robot.yaml`` joint
chain (axes + ``position_limits``, ADR-0030). If a HAL ever published a joint
value in a DIFFERENT sign/axis convention than that model, the kernel would
forward-kinematics a configuration the real robot is not in — a mirror-image arm
— which both false-positives self-collision E-stops AND can MISS real collisions.

Such a mismatch would be detectable on any joint whose declared ``position_limits``
are entirely one sign (e.g. the Franka ``panda_joint4`` range ``[-3.0718,
-0.0698]``): a sign flip lands the published value OUTSIDE the declared range.
This module asserts the cheap, robust invariant — **every HAL-published joint
position lies within that joint's declared ``position_limits``** — so a future
convention regression trips a test instead of silently feeding the safety kernel
a wrong pose.

Status: no such mismatch exists today. A live robocasa pin (2026-06-16) read the
``panda_mobile`` ``panda_joint4`` three ways at the same configuration — raw
MuJoCo ``robot0_joint4`` qpos, robosuite ``robot0_joint_pos[3]``, and
``SimAttachedHAL.read_state`` — and all three agreed at ``-2.2384`` rad (negative,
inside ``[-3.0718, -0.0698]``). An earlier "joint4 publishes +2.6" report was a
parsing artifact in a throwaway analysis script (``str.strip('- \\n')`` ate the
leading minus sign), not a defect in OpenRAL. This guard is kept as a forward
invariant, not a bug repro. See GH #13 for the full pin.

Cross-HAL audit (2026-06-16): NO read path in ANY HAL applies a sign/axis
transform — ``SimAttachedHAL`` and ``MujocoArmHAL`` read raw ``qpos`` keyed by
joint name, and the ros2_control bridge republishes ``/joint_states`` verbatim.
So a flip cannot be *introduced* in code; the only residual risk is a per-robot
``robot.yaml``-vs-MJCF data discrepancy. Of every shipped ``robot.yaml``, only
``franka_panda`` / ``panda_mobile`` have an asymmetric (single-sign) arm joint
(``panda_joint4``) — the lone case where the range invariant can even catch a
flip; both verified consistent. The other arms have symmetric ranges (a flip
would be range-invisible) but are covered structurally by the no-transform read.

Coverage today (each commands a mid-range in-limits pose, reads it back, asserts
within limits; skips if its MJCF asset is unavailable):
* ``franka_panda`` (menagerie) — also round-trips the commanded joint4 sign.
* ``openarm`` (openarm_v2 bimanual), ``so100`` (menagerie), ``aloha`` (gym_aloha).
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
from openral_hal.aloha import AlohaMujocoHAL
from openral_hal.openarm import OpenArmMujocoHAL
from openral_hal.so100_mujoco import SO100MujocoHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
]

_needs_panda_mjcf = pytest.mark.skipif(
    _MJCF_ERROR is not None, reason=f"Panda MJCF unavailable: {_MJCF_ERROR}"
)

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


def _midrange_target(description: RobotDescription) -> list[float]:
    """A per-joint target at the midpoint of each arm joint's declared limits.

    Gripper / limit-less joints get ``0.0``. Commanding the midpoint guarantees an
    in-limits goal for any arm, so a faithful raw-qpos read back must land inside
    the declared range — and a flipped convention on an asymmetric joint would not.
    """
    target: list[float] = []
    for joint in description.joints:
        if joint.role == "gripper" or not joint.position_limits:
            target.append(0.0)
            continue
        lo, hi = joint.position_limits
        target.append((lo + hi) / 2.0)
    return target


@_needs_panda_mjcf
def test_franka_panda_published_joints_obey_declared_limits() -> None:
    """Native Franka (menagerie) read is convention-consistent — the invariant.

    Commands a valid in-limits pose (joint4 negative, per its ``[-3.0718,
    -0.0698]`` range) and reads it back: a faithful raw-qpos read lands inside
    the declared limits AND round-trips the commanded sign. A flipped convention
    would read joint4 back positive, out of range, and fail this guard.
    """
    # In-limits target: joint4 = -2.0 ∈ [-3.0718, -0.0698]; others within range.
    target = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7, 0.0]
    hal = FrankaPandaHAL(gravity_enabled=False, settle_steps=1500)
    hal.connect()
    try:
        hal.send_action(Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[target]))
        state = hal.read_state()
        # Round-trips the commanded sign (would flip under a convention bug)...
        assert state.position[3] == pytest.approx(target[3], abs=1e-2)
        # ...and stays within the robot.yaml declared limits.
        assert_joints_within_declared_limits(state, hal.description)
    finally:
        hal.disconnect()


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: OpenArmMujocoHAL(gravity_enabled=False, settle_steps=1500), id="openarm"
        ),
        pytest.param(lambda: SO100MujocoHAL(gravity_enabled=False, settle_steps=1500), id="so100"),
        pytest.param(lambda: AlohaMujocoHAL(gravity_enabled=False, settle_steps=1500), id="aloha"),
    ],
)
def test_native_mujoco_arm_published_joints_obey_declared_limits(factory) -> None:  # type: ignore[no-untyped-def]  # reason: pytest param is a zero-arg HAL factory; precise Callable type adds no safety here
    """Every native MuJoCo arm HAL reads back a commanded mid-range pose in-limits.

    Generalizes the franka invariant to the other instantiable native arms. Each
    resolves its own MJCF (menagerie / openarm_v2 / gym_aloha); an unavailable
    asset ``pytest.skip``s rather than failing (per CLAUDE.md §1.11 — no fakes).
    """
    try:
        hal = factory()
        hal.connect()
    except Exception as exc:  # asset clone / MJCF parse may raise non-ImportError types
        pytest.skip(f"HAL MJCF asset unavailable: {type(exc).__name__}: {exc}")
    try:
        hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                joint_targets=[_midrange_target(hal.description)],
            )
        )
        assert_joints_within_declared_limits(hal.read_state(), hal.description)
    finally:
        hal.disconnect()
