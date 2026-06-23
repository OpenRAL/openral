"""SimSensorBridge broadcasts an optical-frame TF only for liftable RGB cameras (ADR-0051)."""

from __future__ import annotations

from openral_core import RobotDescription
from openral_hal.sim_sensor_bridge import _optical_frame_rgb_cameras


def test_panda_mobile_shoulder_cameras_are_liftable_wrist_is_not() -> None:
    """RGB cameras with a dedicated *_optical_frame get a TF; a link-framed one does not.

    panda_mobile's ``shoulder_left`` / ``shoulder_right`` (per ADR-0070) declare
    ``*_optical_frame`` frames (the SimSensorBridge broadcasts
    ``base_link -> <camera>_optical_frame`` from the live MuJoCo pose so the
    object-lift can project the world voxel map into them). The eye-in-hand
    ``wrist`` rides a robot link (``panda_hand``) already in TF from
    robot_state_publisher and must be excluded.
    """
    desc = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    liftable = {s.name for s in _optical_frame_rgb_cameras(desc.sensors)}
    assert "shoulder_left" in liftable
    assert "shoulder_right" in liftable
    assert "wrist" not in liftable  # frame_id is the panda_hand link, not an optical frame
    # Depth sensors are never in the RGB optical-frame set (they publish their own TF).
    assert all(s.modality == "rgb" for s in _optical_frame_rgb_cameras(desc.sensors))
