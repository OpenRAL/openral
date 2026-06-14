"""robosuite environments parameterised for the SO-100.

robosuite's stock manipulation envs (``Lift``, ``PickPlace``, ``Stack``,
…) are robot-agnostic — they only need ``robots=[…]`` plus a composite
controller config and a few task-specific dimensions. The SO-100 has a
~25 cm reach (vs the Panda's ~85 cm), but instead of moving the table
we **bolt the SO-100 onto the standard ``TableArena``'s top** (the
same way real users deploy it on a desk). That keeps robosuite's
shipped cameras (``agentview``, ``frontview``) framing the correct
workspace and lets us reuse :class:`Lift` verbatim.

For control we use robosuite's stock **OSC_POSITION** controller —
3-DOF operational-space position control that does Cartesian-to-joint
IK internally. Three reasons:

* the SO-100 is 5-DOF, so 6-DOF OSC_POSE is over-determined for it
  (orientation drifts to whatever the IK chooses); OSC_POSITION
  targets just the eef position and lets the wrist orientation
  fall out naturally;
* it removes ~120 lines of grid-search + Jacobian glue we'd
  otherwise need in the scripted policy — the policy just commands
  Cartesian deltas;
* robosuite ships per-task tuning for OSC (kp / output_max ranges)
  that's been validated across the Panda / Sawyer / UR5e family.

Reuses :class:`robosuite.environments.manipulation.lift.Lift` verbatim
— no copy-paste of the env body.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import robosuite
from robosuite.environments.manipulation.lift import Lift
from robosuite.utils.placement_samplers import UniformRandomSampler

from openral_sim.backends.so100_robosuite.model import SO100  # noqa: F401  registers SO100Robot

__all__ = ["make_so100_lift_env", "so100_osc_controller_config"]


# Lift's stock 4 cm lift threshold is sized for the Panda's reach
# envelope (~85 cm); the SO-100 with its ~25 cm reach + heavy gravity
# droop tops out around 3 cm of clear lift from its mid pose. 2 cm
# keeps the test honest while staying inside the SO-100's authority.
_DEFAULT_LIFT_HEIGHT_M = 0.02


class _So100Lift(Lift):  # type: ignore[misc]  # reason: robosuite has no type stubs
    """:class:`Lift` re-parameterised for the SO-100's smaller workspace.

    Overrides only what the SO-100's geometry actually forces:

    * mounts the arm on the table top (the Panda default mount
      already provides the 80 cm of vertical clearance robosuite
      bakes into ``agentview``);
    * sizes the block to fit the SO-100 jaw aperture (~3 cm at full
      open) and stands it upright so the grip site lands at block
      mid-height while the fixed jaw clears the table;
    * relaxes the lift-success threshold from the Panda-default 4 cm
      to 2 cm of clear lift above the block's resting bottom face.

    Everything else (reward shaping, observables, placement sampler
    hook, reset logic, …) is inherited verbatim from :class:`Lift`.
    """

    def __init__(
        self,
        *args: Any,
        lift_height_m: float = _DEFAULT_LIFT_HEIGHT_M,
        cube_half_extent_m: float = 0.012,
        cube_block_height_m: float = 0.04,
        **kwargs: Any,
    ) -> None:
        self._lift_height_m = float(lift_height_m)
        self._cube_half_extent_m = float(cube_half_extent_m)
        self._cube_block_height_m = float(cube_block_height_m)
        super().__init__(*args, **kwargs)

    def _check_success(self) -> bool:
        """Success = block's BOTTOM face lifted clear of the table top.

        Lift's stock check (``cube_z > table_top + lift_height``) is
        oblivious to the block's own size, which works when the
        ``BoxObject`` is the Lift-default small cube. For the SO-100
        block (which is several cm tall to clear the gripper geometry)
        we require the block's BOTTOM to clear the table by
        :attr:`_lift_height_m` — that way the threshold scales with
        the block height instead of producing instant "success" at
        spawn.
        """
        cube_z = self.sim.data.body_xpos[self.cube_body_id][2]
        table_top_z = float(self.model.mujoco_arena.table_offset[2])  # type: ignore[has-type]  # reason: robosuite has no type stubs
        block_bottom_z = cube_z - self._cube_block_height_m / 2.0
        return bool(block_bottom_z > table_top_z + self._lift_height_m)

    def _load_model(self) -> None:
        # Reuse Lift._load_model verbatim and then resize the block +
        # mount the SO-100 on the table top. Lift instantiates a
        # ``BoxObject`` at 2.0-2.2 cm half-extent (too big for the SO-100
        # jaw, too short to grip cleanly above the table) and
        # ``set_base_xpos`` parks the arm at table_offset.z (which the
        # Panda's RethinkMount lifts further; the SO-100's NullMount
        # doesn't, so we patch the z directly here).
        super()._load_model()
        from robosuite.models.objects import BoxObject  # local: avoid eager import
        from robosuite.models.tasks import ManipulationTask
        from robosuite.utils.mjcf_utils import CustomMaterial

        # Bolt the SO-100 onto the table top (just like a real-world
        # desk deployment). The Panda's table_offset is (0, 0, 0.8);
        # we sit the SO-100 ~10 cm behind the table centre so the cube
        # placement region (which Lift centres on table_offset.x) falls
        # inside the SO-100's 25 cm reach. We also rotate the arm 90°
        # around z so the menagerie's natural -y workspace direction
        # maps to world +x (where Lift puts the cube).
        robot = self.robots[0].robot_model
        table_top_z = float(self.table_offset[2]) + float(self.table_full_size[2]) / 2.0
        base_pos = np.array([self.table_offset[0] - 0.12, self.table_offset[1], table_top_z])
        robot.set_base_xpos(base_pos)
        robot.set_base_ori(np.array([0.0, 0.0, np.pi / 2.0]))

        half = self._cube_half_extent_m
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )
        # The block is an upright box rather than a true cube: thin x/y
        # footprint (so the SO-100 jaw aperture clears it) but a taller
        # z so the grasp height clears the gripper geometry. When the
        # grip_site lands at the block's z-centre, the wrist body sits
        # ~2 cm above the table and the fixed jaw (which extends 4-5 cm
        # down from the wrist) wraps the block instead of colliding
        # with the table top.
        self.cube = BoxObject(
            name="cube",
            size_min=[half, half, self._cube_block_height_m / 2.0],
            size_max=[half, half, self._cube_block_height_m / 2.0],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        # Rebuild the ManipulationTask with the resized block + relocated
        # robot. We keep the original arena instance so the cameras and
        # textures don't get rebuilt from scratch on every reset.
        self.model = ManipulationTask(
            mujoco_arena=self.model.mujoco_arena,  # type: ignore[has-type]  # reason: robosuite has no type stubs
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=self.cube,
        )


def so100_osc_controller_config() -> dict[str, Any]:
    """Build a robosuite composite controller config for the SO-100.

    * Arm: stock **OSC_POSITION** (3-DOF operational-space position).
      The SO-100 is 5-DOF so 6-DOF OSC_POSE would over-constrain it;
      OSC_POSITION targets just the eef Cartesian position and lets
      the wrist orientation fall out of the IK. ``output_max`` is
      shrunk from the Panda default (5 cm/step) to 1 cm/step because
      the SO-100's small mass matrix amplifies any commanded delta —
      Panda-default steps make the arm overshoot wildly.
    * Gripper: ``GRIP`` (registered alias for
      :class:`SimpleGripController`) — maps a [-1, 1] command to
      symmetric torque around zero on the single Jaw actuator.

    Loaded from robosuite's shipped ``parts/osc_position.json`` and
    only the SO-100-specific tuning is overridden; we don't duplicate
    the rest of the schema.
    """
    parts_dir = Path(robosuite.__file__).parent / "controllers" / "config" / "default" / "parts"
    base = json.loads((parts_dir / "osc_position.json").read_text())
    arm_cfg = copy.deepcopy(base)
    # Slow the per-step Cartesian delta down (Panda default 5 cm is
    # too large for the SO-100's reach + small mass matrix).
    arm_cfg["output_max"] = [0.01, 0.01, 0.01]
    arm_cfg["output_min"] = [-0.01, -0.01, -0.01]
    # The Panda kp=150 is calibrated against ~2 N·m gravity torques;
    # the SO-100's mass matrix is 30-50× smaller, so OSC's
    # ``mass_matrix @ desired_acc`` output gets attenuated to nearly
    # zero unless kp scales up to match.
    arm_cfg["kp"] = 1500
    arm_cfg["kp_limits"] = [0, 5000]
    arm_cfg["damping_ratio"] = 1.0
    # Interpret Cartesian deltas in WORLD frame. The default ``base``
    # frame would re-rotate our world-frame commands by the SO-100's
    # 90°-z base orientation (see :meth:`_So100Lift._load_model`),
    # which swaps the x and y axes silently — the eef ends up
    # tracking the wrong direction.
    arm_cfg["input_ref_frame"] = "world"
    arm_cfg["gripper"] = {"type": "GRIP"}
    return {
        "type": "BASIC",
        "body_parts": {"right": arm_cfg},
    }


def make_so100_lift_env(
    *,
    has_renderer: bool = False,
    has_offscreen_renderer: bool = True,
    use_camera_obs: bool = True,
    camera_names: tuple[str, ...] | str = ("agentview", "robot0_eye_in_hand"),
    camera_heights: int = 256,
    camera_widths: int = 256,
    horizon: int = 300,
    control_freq: int = 20,
    table_full_size: tuple[float, float, float] = (0.4, 0.4, 0.05),
    cube_half_extent_m: float = 0.012,
    cube_block_height_m: float = 0.04,
    x_range: tuple[float, float] = (-0.02, 0.05),
    y_range: tuple[float, float] = (-0.04, 0.04),
    seed: int | None = None,
    lift_height_m: float = _DEFAULT_LIFT_HEIGHT_M,
    reward_shaping: bool = True,
) -> _So100Lift:
    """Build a SO-100-flavoured :class:`Lift` env.

    Args:
        has_renderer: Show the interactive MuJoCo viewer (single-threaded).
        has_offscreen_renderer: Render to offscreen buffers (camera obs).
        use_camera_obs: Include rendered camera frames in the observation.
        camera_names: robosuite camera names. The default pair is the
            scene's third-person ``agentview`` plus the SO-100's
            ``eye_in_hand`` camera (prefixed by robosuite naming).
        camera_heights: Height of camera frame.
        camera_widths: Width of camera frame.
        horizon: Episode budget in env steps.
        control_freq: Policy update rate in Hz.
        table_full_size: ``(x, y, z)`` table size. The default 40×40×5 cm
            keeps the cube well inside the SO-100's reach.
        cube_half_extent_m: Half x/y edge length of the block.
        cube_block_height_m: Full z-height of the block. The block is
            taller than the half-extent so the SO-100 can grasp the
            block's mid-height while the fixed jaw clears the table.
        x_range: Cube placement bounds along table x in the table's
            local frame. Negative ``x`` puts the cube closer to the
            robot.
        y_range: Cube placement bounds along table y.
        seed: RNG seed for placement sampling.
        lift_height_m: Block bottom-face clearance above the table top
            that counts as a successful lift.
        reward_shaping: Enable the dense reaching+grasping reward
            component (useful for scripted-policy debug curves).

    Returns:
        A configured :class:`_So100Lift` ready for ``env.reset()`` /
        ``env.step()``.
    """
    # Reuse Lift's hard-coded table_offset (0, 0, 0.8) so the standard
    # robosuite cameras (``agentview``, ``frontview``) frame the scene
    # correctly. We only narrow the placement sampler's x/y range to
    # match the SO-100's reach.
    panda_default_table_offset = np.array((0.0, 0.0, 0.8), dtype=np.float64)
    placement_initializer = UniformRandomSampler(
        name="ObjectSampler",
        mujoco_objects=None,  # filled by Lift._load_model via add_objects
        x_range=list(x_range),
        y_range=list(y_range),
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=panda_default_table_offset,
        z_offset=0.01,
    )

    if seed is not None:
        np.random.seed(int(seed))
    return _So100Lift(
        robots=["SO100"],
        gripper_types="default",
        controller_configs=so100_osc_controller_config(),
        table_full_size=table_full_size,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        use_camera_obs=use_camera_obs,
        camera_names=list(camera_names) if isinstance(camera_names, tuple) else camera_names,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        horizon=horizon,
        control_freq=control_freq,
        ignore_done=False,
        hard_reset=True,
        reward_shaping=reward_shaping,
        placement_initializer=placement_initializer,
        lift_height_m=lift_height_m,
        cube_half_extent_m=cube_half_extent_m,
        cube_block_height_m=cube_block_height_m,
    )
