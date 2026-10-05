# SPDX-License-Identifier: Apache-2.0
"""The OpenArm twin's head camera sits exactly at ``head_zed``'s manifest mount.

``head_zed`` carries two descriptions of one pose: the TF mount
(``parent_frame: openarm_base`` + ``static_transform_xyz_rpy``, what every consumer
locates the depth by) and the ``sim_placement`` the camera rig splices into the MJCF
(what MuJoCo actually casts and renders from). The sim bridge stamps the raster in the
mount's optical child, so the two must agree or every sim cloud, mask and lift lands
off the geometry it was cast from. Expressed from ``openarm_left_base_link``, which is
``openarm_base`` shifted by the manifest's ``openarm_base -> openarm_left_link0``
attachment with the same orientation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from openral_core import RobotDescription
from openral_core.geometry import look_at_quat_wxyz
from openral_hal.depth_cloud import BODY_TO_OPTICAL_QUAT_XYZW, depth_optical_frame_id

_ROBOT_YAML = Path(__file__).resolve().parents[2] / "robots" / "openarm" / "robot.yaml"


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ROS fixed-axis roll/pitch/yaw: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _quat_xyzw_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))


def test_body_to_optical_is_rep103() -> None:
    # Optical x right / y down / z forward, in body x forward / y left / z up.
    expected = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    assert np.allclose(_quat_xyzw_to_matrix(*BODY_TO_OPTICAL_QUAT_XYZW), expected)


def test_head_zed_sim_camera_matches_its_tf_mount() -> None:
    description = RobotDescription.from_yaml(str(_ROBOT_YAML))
    spec = next(s for s in description.sensors if s.name == "head_zed")
    placement = spec.sim_placement
    assert placement is not None
    assert placement.parent_body == "openarm_left_base_link"
    assert spec.parent_frame == description.base_frame == "openarm_base"
    assert depth_optical_frame_id(spec) == "zed_camera_link_optical_frame"
    left_mount = next(
        a for a in description.fixed_attachments if a.child_link == "openarm_left_link0"
    )
    assert left_mount.parent_link == "openarm_base"
    assert np.allclose(left_mount.origin_rpy, 0.0)

    x, y, z, roll, pitch, yaw = spec.static_transform_xyz_rpy
    mount = _rpy(roll, pitch, yaw)
    # Position: the mount, re-expressed from the left base link.
    assert np.allclose(np.array(placement.pos) + np.array(left_mount.origin_xyz), [x, y, z])
    # Orientation: the MJCF camera (-z view, +y up) as an optical frame equals the
    # mount's REP-103 optical child.
    w, qx, qy, qz = look_at_quat_wxyz(
        placement.pos, placement.target, up=placement.up, view_axis="-z"
    )
    camera_optical = _quat_xyzw_to_matrix(qx, qy, qz, w) @ np.diag([1.0, -1.0, -1.0])
    mount_optical = mount @ _quat_xyzw_to_matrix(*BODY_TO_OPTICAL_QUAT_XYZW)
    assert _angle_deg(camera_optical, mount_optical) < 0.05
