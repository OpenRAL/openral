# SPDX-License-Identifier: Apache-2.0
"""The depth synth must see what the world can TOUCH, not what it can show.

#149 taught the ground-truth probe that a geom with neither ``contype`` nor
``conaffinity`` cannot collide with anything, and must never be promoted to a
real contact. Nothing taught the perception path the same thing: ``mj_ray``
struck whatever was *visible*, OctoMap integrated it, and the safety kernel
E-stopped on cells no body can ever occupy — while the probe adjudicating that
stop was required to ignore the only geometry backing them (#174).

On RoboCasa this is not a rounding error. 703 of the 1675 geoms in the
``baguette`` scene are non-collidable decoration, and 18.8 % of one live map's
occupied cells were backed by nothing else.

The direction of the fix is what makes it safe, and it is the property these
tests hold: filtering can only move a return **farther** along its ray or
remove it entirely. A collidable surface stays hittable, so no touchable
geometry can leave the map. On hardware the term does not exist — a visible
object is solid — so this is the sim matching the world it stands in for.

Real compiled ``MjModel``s throughout, no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import numpy as np
import pytest
from openral_sim.backends.depth_camera import noncollidable_geom_ids, synthesize_depth_image

mujoco = pytest.importorskip("mujoco")

# A camera at z = 1.0 with MuJoCo's default attitude looks straight down -z,
# so a slab's depth is simply the drop from the camera to its top face.
_CAMERA_Z = 1.0
_SOLID_TOP_Z = 0.52
_DECOR_TOP_Z = 0.71

# The decoration is the RoboCasa case verbatim: a wide thin panel hanging in
# front of the surface a policy actually has to reach.
_SCENE = """
<mujoco model="noncollidable_depth">
  <worldbody>
    <camera name="depth_cam" pos="0 0 {camera_z}"/>
    {solid}
    <body name="decor" pos="0 0 0.7">
      <geom name="decor_panel" type="box" size="0.4 0.4 0.01"
            contype="0" conaffinity="0" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""
_SOLID = """
    <body name="counter" pos="0 0 0.5">
      <geom name="counter_top" type="box" size="0.4 0.4 0.02"/>
    </body>
"""

_SYNTH = {
    "camera_name": "depth_cam",
    "width": 16,
    "height": 16,
    "fx": 16.0,
    "fy": 16.0,
    "cx": 8.0,
    "cy": 8.0,
    "max_range_m": 5.0,
}


def _depth(*, solid: bool) -> np.ndarray:
    """The dense depth raster of the scene, with or without a solid behind."""
    xml = _SCENE.format(camera_z=_CAMERA_Z, solid=_SOLID if solid else "")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return synthesize_depth_image(model=model, data=data, **_SYNTH)


def test_noncollidable_geom_ids_names_only_the_untouchable() -> None:
    """The predicate is MuJoCo's own, not a rendering convention."""
    model = mujoco.MjModel.from_xml_string(_SCENE.format(camera_z=_CAMERA_Z, solid=_SOLID))
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
        for gid in noncollidable_geom_ids(model)
    }

    assert names == {"decor_panel"}


def test_the_solid_behind_the_decoration_is_what_the_depth_reports() -> None:
    """A panel no body can touch must not stand in for the surface behind it."""
    depth = _depth(solid=True)
    centre = float(depth[8, 8])

    assert centre == pytest.approx(_CAMERA_Z - _SOLID_TOP_Z, abs=1e-4)
    # And emphatically NOT the decoration, 190 mm nearer — the reading that
    # put a phantom obstacle in the map between the arm and the counter.
    assert centre - (_CAMERA_Z - _DECOR_TOP_Z) == pytest.approx(0.19, abs=1e-4)


def test_decoration_over_nothing_returns_nothing() -> None:
    """With no solid behind it, an untouchable panel leaves the ray empty.

    This is the half that removes map cells rather than moving them, so it is
    the half a reviewer should look hardest at: the cell is emptied precisely
    because nothing that can be collided with was ever in it.
    """
    depth = _depth(solid=False)

    assert float(depth[8, 8]) == 0.0, "0.0 is the depth image's no-return sentinel"
    assert not np.any(depth > 0.0)


def test_filtering_never_pulls_a_return_nearer() -> None:
    """The safety property, over every pixel: depth is non-decreasing.

    The unfiltered cast is what master did. Filtering removes candidates from
    the nearest-hit search and can therefore only push the answer away from
    the camera or lose it — never bring it closer, which is the only direction
    that could put the kernel's obstacle set at risk.
    """
    xml = _SCENE.format(camera_z=_CAMERA_Z, solid=_SOLID)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    filtered = synthesize_depth_image(model=model, data=data, **_SYNTH)
    # Cast the same rays with nothing hidden, by making the decoration
    # collidable — the identical geometry, minus the reason to skip it.
    model.geom_contype[int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "decor_panel"))] = 1
    unfiltered = synthesize_depth_image(model=model, data=data, **_SYNTH)

    returned = (filtered > 0.0) & (unfiltered > 0.0)
    assert returned.any()
    assert np.all(filtered[returned] >= unfiltered[returned] - 1e-9)
