# SPDX-License-Identifier: Apache-2.0
"""OpenAMRobot 2.0's wrist cameras ride the hand where OpenArm's twin carries them.

The hands are OpenArm v2.0 hands, so the wrist cameras take the OpenArm twin's mount
(``camera_wrist_left/right`` in ``openarm_v20_bimanual.xml``) instead of upstream's
placeholder frames, which looked sideways along the hand's +x and never saw the
fingers. The URDF optical frame (TF) and the MuJoCo camera (which Isaac also mounts
from) must both sit there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from openral_core import RobotDescription, ROSConfigError

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"


def _camera_in_hand(
    hal: Any,
    camera: str,
    hand: str,  # reason: two unrelated HAL classes, no shared type
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The camera's position and rotation in its hand body, from the twin's FK."""
    mujoco = pytest.importorskip("mujoco")
    handles = hal.mujoco_handles()
    assert handles is not None
    model, data = handles
    mujoco.mj_forward(model, data)
    cam, body = model.camera(camera).id, model.body(hand).id
    rb = data.xmat[body].reshape(3, 3)
    rc = data.cam_xmat[cam].reshape(3, 3)
    return rb.T @ (data.cam_xpos[cam] - data.xpos[body]), rb.T @ rc


@pytest.mark.parametrize("side", ["left", "right"])
def test_wrist_camera_matches_the_openarm_twin(side: str) -> None:
    pytest.importorskip("mujoco")
    from openral_hal.openamrobot_v2 import OpenAMRobotV2MujocoHAL
    from openral_hal.openarm import OpenArmMujocoHAL

    hand = f"openarm_{side}_ee_base_link"
    amr = OpenAMRobotV2MujocoHAL(
        description=RobotDescription.from_yaml(str(_ROBOTS / "openamrobot_v2" / "robot.yaml"))
    )
    arm = OpenArmMujocoHAL()
    try:
        amr.connect()
        arm.connect()
    except ROSConfigError as exc:  # pinned mesh clone unreachable (offline host)
        pytest.skip(f"twin unavailable: {exc}")
    try:
        p_amr, r_amr = _camera_in_hand(amr, f"wrist_{side}", hand)
        p_arm, r_arm = _camera_in_hand(arm, f"camera_wrist_{side}", hand)
    finally:
        amr.disconnect()
        arm.disconnect()
    assert np.allclose(p_amr, p_arm, atol=1e-4)
    assert np.allclose(r_amr, r_arm, atol=1e-4)
    # A MuJoCo camera looks down its -z: toward the fingers (hand -z), not sideways.
    assert (r_amr @ [0.0, 0.0, -1.0])[2] < -0.8
