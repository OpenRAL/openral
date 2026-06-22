"""Regression: MJCF collision lowering must assign ``dof_index`` by joint ORDER,
not by joint-name equality with the manifest.

``openral_safety.mjcf_lowering.lower_collision_params`` lowers a compiled
``mujoco.MjModel`` to the kernel's collision parameters. Each movable
(hinge/slide) joint needs a ``dof_index`` — the column of the manifest
``ActionChunk.flat`` joint vector (and ``MjData.qpos`` actuated order) that
drives that link — so the kernel's allocation-free forward kinematics can place
the link at the *commanded* configuration. A ``dof_index`` of ``-1`` marks an
immovable joint: the FK never reads its angle and freezes the link at its rest
transform.

**The bug (pre-fix):** the lowering keyed its dof lookup by the MANIFEST joint
names but read the MJCF's own joint names. Real robots name their MJCF joints
differently from the manifest (``panda_joint1`` vs ``joint1``; ``shoulder_pan``
vs ``Rotation``), so every lookup missed and *every* ``dof_index`` collapsed to
``-1``. The kernel then FK'd the whole arm at its rest pose regardless of the
commanded chunk — "self-collision check enabled" became a silent no-op for every
MJCF robot whose joint names differ from its manifest (franka, so100, so101).
``openral deploy sim`` *prefers* the MJCF-lowered model, so the kernel's
geometric-collision differentiator was inert in the primary deploy path.

The contract: ``description.joints`` enumerates joints in the same order as the
robot's MuJoCo actuators (``python/hal/.../_mujoco_arm.py`` docstring), so the
i-th movable MJCF joint maps to manifest index ``i`` — independent of naming.

Gate (CLAUDE.md §1.11 / §1.12): mujoco + openral_safety. Otherwise pytest.skip —
never faked. Fully hermetic: inline MJCF strings, no asset download.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("openral_safety")

from openral_safety.mjcf_lowering import lower_collision_params  # noqa: E402

# Two capsule-jointed links whose MJCF joint names ("Rotation", "Pitch") do NOT
# match the manifest joint order passed in — exactly the so100 situation
# (manifest ``shoulder_pan, shoulder_lift`` vs MJCF ``Rotation, Pitch``).
_MISMATCHED_NAMES_MJCF = """
<mujoco>
  <worldbody>
    <body name="shoulder" pos="0 0 0.1">
      <joint name="Rotation" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
      <body name="upper_arm" pos="0 0 0.1">
        <joint name="Pitch" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# Three movable links but only two manifest joints — the extra (e.g. franka's
# second, mimic, gripper finger) must lower to ``-1`` rather than index past the
# end of the commanded joint vector.
_EXTRA_JOINT_MJCF = """
<mujoco>
  <worldbody>
    <body name="link1" pos="0 0 0.1">
      <joint name="a" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
      <body name="link2" pos="0 0 0.1">
        <joint name="b" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
        <body name="link3" pos="0 0 0.1">
          <joint name="c" type="slide" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# A fixed (welded) link between two movable ones: the fixed link consumes no
# joint column, so the movable joints keep contiguous indices 0, 1.
_FIXED_LINK_MJCF = """
<mujoco>
  <worldbody>
    <body name="link1" pos="0 0 0.1">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
      <body name="welded" pos="0 0 0.1">
        <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
        <body name="link3" pos="0 0 0.1">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_mismatched_joint_names_still_map_dof_index_by_order() -> None:
    """MJCF joint names differing from the manifest must NOT zero every dof_index.

    The two movable links map to manifest columns 0 and 1 by order, even though
    neither MJCF joint name ("Rotation"/"Pitch") appears in the manifest list.
    """
    model = mujoco.MjModel.from_xml_string(_MISMATCHED_NAMES_MJCF)
    params = lower_collision_params(model, ["shoulder_pan", "shoulder_lift"])
    assert params["self_collision_enabled"] is True
    assert params["collision_dof_index"] == [0, 1], (
        "movable joints must map to manifest columns by order; all -1 means the "
        "kernel FK is frozen at the rest pose (self-collision no-op)"
    )
    # Both are hinge joints -> joint_kind 1.
    assert params["collision_joint_kind"] == [1, 1]


def test_movable_joints_beyond_manifest_length_map_to_minus_one() -> None:
    """A movable joint past the commanded joint count lowers to -1 (no OOB index)."""
    model = mujoco.MjModel.from_xml_string(_EXTRA_JOINT_MJCF)
    params = lower_collision_params(model, ["j_a", "j_b"])  # only two columns
    assert params["collision_dof_index"] == [0, 1, -1]


def test_fixed_link_does_not_consume_a_joint_column() -> None:
    """A welded link gets dof_index -1; movable joints stay contiguous 0,1."""
    model = mujoco.MjModel.from_xml_string(_FIXED_LINK_MJCF)
    params = lower_collision_params(model, ["j1", "j2"])
    assert params["collision_dof_index"] == [0, -1, 1]
    assert params["collision_joint_kind"] == [1, 0, 1]
