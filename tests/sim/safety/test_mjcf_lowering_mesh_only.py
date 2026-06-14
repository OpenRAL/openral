"""Regression: a mesh-only MJCF lowers to the disabled-self-collision sentinel.

``openral_safety.mjcf_lowering.lower_collision_params`` only approximates *convex
analytic primitives* (sphere / capsule / cylinder / box) as kernel capsules;
mesh and plane collision geoms are skipped (``_first_collidable_geom``). A robot
whose MJCF carries **only** mesh collision geometry — e.g. the SO-101
``so_arm101`` ``new_calib`` model used by ``openral deploy sim --config
scenes/deploy/so101_box.yaml`` — therefore lowers to zero
capsules.

Before the fix, the function still returned the full param dict with
``self_collision_enabled: True`` and **empty** ``collision_capsule_*`` lists.
``sim_e2e.launch.py`` forwards those as ROS parameters on the safety_kernel
node, and ``launch_ros`` normalises an empty Python list to ``()`` —
``ensure_argument_type`` then rejects the whole launch with
``Expected 'value' to be one of [...], but got '()' of type tuple`` *before any
node starts*. The contract (mirrored by the manifest-geometry path
``collision_params_from_description``) is to return the
``{self_collision_enabled: False}`` sentinel when there is no lowerable
geometry, so the kernel runs its scalar envelope check exactly as before.

Gate (CLAUDE.md §1.11 / §1.12): mujoco + openral_safety (the ROS overlay
package). Otherwise pytest.skip — never faked.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("openral_safety")

from openral_safety.mjcf_lowering import lower_collision_params  # noqa: E402

# Two hinge-jointed bodies whose ONLY collision geometry is a mesh (the exact
# shape of the SO-101 ``new_calib`` model: mesh visuals + mesh collision, no
# primitive capsule/box geoms). A mesh geom is collidable by default but is not
# a lowerable primitive, so each body contributes zero capsules.
_MESH_ONLY_MJCF = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/>
  </asset>
  <worldbody>
    <body name="link1" pos="0 0 0.1">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="mesh" mesh="tet"/>
      <body name="link2" pos="0 0 0.1">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="mesh" mesh="tet"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# Same chain but with a primitive capsule on each body — the positive control:
# lowering must still produce the enabled, fully-populated param dict.
_PRIMITIVE_MJCF = """
<mujoco>
  <worldbody>
    <body name="link1" pos="0 0 0.1">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
      <body name="link2" pos="0 0 0.1">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_mesh_only_mjcf_returns_disabled_sentinel() -> None:
    """A mesh-only model yields exactly ``{self_collision_enabled: False}``.

    No ``collision_capsule_*`` (or any other) keys — so the launch never
    forwards an empty list as a ROS parameter.
    """
    model = mujoco.MjModel.from_xml_string(_MESH_ONLY_MJCF)
    params = lower_collision_params(model, ["j1", "j2"])
    assert params == {"self_collision_enabled": False}


def test_primitive_mjcf_still_lowers_to_enabled_params() -> None:
    """Positive control: primitive geoms still lower to non-empty capsule arrays."""
    model = mujoco.MjModel.from_xml_string(_PRIMITIVE_MJCF)
    params = lower_collision_params(model, ["j1", "j2"])
    assert params["self_collision_enabled"] is True
    # One capsule per body, all parallel arrays non-empty.
    assert params["collision_capsule_link"], "primitive model must lower to ≥1 capsule"
    n_caps = len(params["collision_capsule_link"])
    assert len(params["collision_capsule_radius"]) == n_caps
    assert len(params["collision_capsule_half_length"]) == n_caps
    assert len(params["collision_capsule_origin_xyzrpy"]) == 6 * n_caps
