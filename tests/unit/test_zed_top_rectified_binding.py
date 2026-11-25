# SPDX-License-Identifier: Apache-2.0
"""On a real cell, a ZED-M robot's colour ``top`` and depth ``head_zed`` are pixel-aligned.

The ZED driver registers its depth to the RECTIFIED left image: same frame, same K, no
distortion (measured on the Orin cell at VGA, 2026-11-25). The unrectified
``rgb/color/raw`` image has its own K and Brown distortion, so binding ``top`` to it puts
colour and depth on different pixel grids and every pinhole back-projection of a ``top``
pixel through the depth drifts toward the image edges.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"


@pytest.mark.parametrize("robot", ["openarm", "openamrobot_v2"])
def test_top_binds_the_image_the_depth_is_registered_to(robot: str) -> None:
    desc = RobotDescription.from_yaml(str(_ROBOTS / robot / "robot.yaml"))
    sensors = {s.name: s for s in desc.sensors}
    top, depth = sensors["top"].deploy_binding, sensors["head_zed"].deploy_binding
    assert top is not None and depth is not None
    assert depth.backend_params["topic"] == "/zed/zed_node/depth/depth_registered"
    assert top.backend_params["topic"] == "/zed/zed_node/rgb/color/rect/image"
