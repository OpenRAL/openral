"""Regression guard — a normalised gripper channel declares normalised limits.

Issue #62. ``JointSpec.position_limits`` is the unit of the value that travels
on that channel: the safety kernel's envelope
(:func:`openral_safety.envelope_loader._extract_joint_limits`), the runner's
pre-clamp (``rskill_runner_node._make_policy_adapter_skill``) and the
supervisor's ``gripper_min`` / ``gripper_max`` all compare a **commanded
Action value** against it.

For a gripper whose HAL contract is a normalised ``[0, 1]`` jaw fraction —
which is exactly what ``sim.grippers[].write_mode: normalised`` declares, since
:meth:`openral_hal._mujoco_arm.MujocoArmHAL._gripper_command_to_raw` clips its
input to ``[0, 1]`` before mapping it onto ``ctrl_range`` — declaring the jaw's
*mechanical* radian range instead makes the envelope stop constraining that
channel: a malformed ``1.5`` fraction sits inside ``[-0.1745, 1.7453]``, clears
both the pre-clamp and the kernel, and reaches the HAL, which clips it
silently. ``robots/so101_follower`` and ``robots/so100_follower`` both shipped
that way.

``PASSTHROUGH`` grippers are deliberately out of scope: their channel carries
the actuator's native unit (metres for Aloha, radians for OpenArm), so their
limits are robot-specific and there is no cross-manifest invariant to assert.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.schemas import GripperWriteMode

_MANIFESTS = sorted(p for p in Path("robots").glob("*/robot.yaml"))


@pytest.mark.parametrize("manifest_path", _MANIFESTS, ids=lambda p: p.parent.name)
def test_normalised_gripper_declares_normalised_position_limits(manifest_path: Path) -> None:
    """A ``normalised`` sim gripper's joint must declare ``[0.0, 1.0]``."""
    description = RobotDescription.from_yaml(str(manifest_path))
    sim = description.sim
    if sim is None or not sim.grippers:
        pytest.skip(f"{manifest_path.parent.name} declares no sim grippers")

    by_name = {j.name: j for j in description.joints}
    offenders: list[str] = []
    checked = 0
    for gripper in sim.grippers:
        if gripper.write_mode is not GripperWriteMode.NORMALISED:
            continue
        joint = by_name.get(gripper.joint)
        assert joint is not None, (
            f"sim.grippers[].joint {gripper.joint!r} is not in joints[] — "
            f"{manifest_path} declares a gripper for a joint that does not exist."
        )
        checked += 1
        if joint.position_limits != (0.0, 1.0):
            offenders.append(f"{joint.name}={joint.position_limits}")

    if checked == 0:
        pytest.skip(f"{manifest_path.parent.name} declares no normalised sim gripper")
    assert not offenders, (
        f"{manifest_path}: a `normalised` gripper's channel carries a [0, 1] jaw "
        f"fraction, so its `position_limits` must be [0.0, 1.0] — the safety "
        f"envelope and the runner pre-clamp compare the commanded fraction "
        f"against it. Put the mechanical range in `sim.grippers[].ctrl_range` "
        f"instead. Offending joint(s): {', '.join(offenders)}"
    )


def test_so_arm_gripper_channel_is_constrained() -> None:
    """The #62 repro: a 1.5 jaw fraction must now be outside the envelope.

    Pins the *behavioural* consequence rather than the literal numbers — with
    the old radian limits, ``1.5`` sat comfortably inside ``[-0.1745, 1.7453]``
    and reached the HAL untouched.
    """
    for robot_id in ("so100_follower", "so101_follower"):
        description = RobotDescription.from_yaml(f"robots/{robot_id}/robot.yaml")
        gripper = next(j for j in description.joints if j.name == "gripper")
        assert gripper.position_limits is not None
        low, high = gripper.position_limits
        assert not (low <= 1.5 <= high), (
            f"{robot_id}: a malformed 1.5 jaw fraction is still inside the "
            f"gripper envelope [{low}, {high}]"
        )
        assert (low, high) == (0.0, 1.0)
