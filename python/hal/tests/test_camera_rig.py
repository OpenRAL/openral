"""Generic camera rig — splice manifest cameras into a bare MJCF (ADR-0065)."""

from __future__ import annotations

import re

import pytest
from openral_core import CameraSimPlacement, SensorSpec
from openral_core.exceptions import ROSConfigError
from openral_hal._camera_rig import rig_cameras_into_mjcf

# A minimal MJCF with one named body and no cameras / floor / lights.
_BARE = """<mujoco model="t">
  <worldbody>
    <body name="gripper">
      <geom type="box" size="0.01 0.01 0.01"/>
    </body>
  </worldbody>
</mujoco>"""


def _rgb(name: str, parent_body: str | None) -> SensorSpec:
    return SensorSpec(
        name=name,
        modality="rgb",
        frame_id="f",
        rate_hz=30.0,
        sim_placement=CameraSimPlacement(
            parent_body=parent_body, pos=(0.0, 0.0, 0.5), target=(0.0, 0.0, 0.0)
        ),
    )


def test_splices_world_and_body_cameras() -> None:
    xml, changed = rig_cameras_into_mjcf(_BARE, [_rgb("front", None), _rgb("wrist", "gripper")])
    assert changed
    cams = re.findall(r'<camera[^>]*name="([^"]+)"', xml)
    assert set(cams) == {"front", "wrist"}
    # world camera lands in worldbody; wrist camera lands inside the gripper body.
    body = re.search(r'<body name="gripper">(.*?)</body>', xml, re.DOTALL).group(1)
    assert 'name="wrist"' in body
    assert 'name="front"' not in body
    # staging added: visual-only floor + fill light.
    assert "camrig_floor" in xml and "<headlight" in xml
    assert 'contype="0"' in xml  # floor must not collide


def test_idempotent_when_cameras_present() -> None:
    # A model that already declares the cameras is left unchanged.
    once, _ = rig_cameras_into_mjcf(_BARE, [_rgb("front", None), _rgb("wrist", "gripper")])
    twice, changed = rig_cameras_into_mjcf(once, [_rgb("front", None), _rgb("wrist", "gripper")])
    assert changed is False
    assert twice == once


def test_noop_without_sim_placement() -> None:
    plain = SensorSpec(name="front", modality="rgb", frame_id="f", rate_hz=30.0)
    xml, changed = rig_cameras_into_mjcf(_BARE, [plain])
    assert changed is False
    assert xml == _BARE


def test_unknown_parent_body_raises() -> None:
    with pytest.raises(ROSConfigError, match="nonexistent"):
        rig_cameras_into_mjcf(_BARE, [_rgb("wrist", "nonexistent")])


def test_fovy_derived_from_intrinsics() -> None:
    from openral_core import IntrinsicsPinhole

    s = SensorSpec(
        name="front",
        modality="rgb",
        frame_id="f",
        rate_hz=30.0,
        intrinsics=IntrinsicsPinhole(width=640, height=480, fx=480.0, fy=480.0, cx=320.0, cy=240.0),
        sim_placement=CameraSimPlacement(pos=(0.0, 0.0, 0.5), target=(0.0, 0.0, 0.0)),
    )
    xml, _ = rig_cameras_into_mjcf(_BARE, [s])
    fovy = float(re.search(r'name="front"[^>]*fovy="([0-9.]+)"', xml).group(1))
    # 2*atan(480/(2*480)) = 53.13 deg
    assert abs(fovy - 53.13) < 0.1
