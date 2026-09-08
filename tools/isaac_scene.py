"""Minimal Isaac Sim lift-cube scene for the sidecar.

Runs under the Isaac Sim py3.11 venv only; imported by ``isaac_sidecar.py`` AFTER
``SimulationApp`` is live (every import here needs a running Kit app).

Isaac Sim **core**, not Isaac Lab: the PyPI ``isaaclab`` (2.3.x) wheel omits
``isaaclab.sim``/``isaaclab.envs`` (need the git-source install plus
``isaaclab_assets``/``isaaclab_tasks``, not on PyPI). Core's
``isaacsim.core.api.World`` + ``Franka`` + ``Camera`` are on PyPI and enough for
a real PhysX+RTX scene; the full Isaac Lab manager-based env (OSC terms, task
MDP) is a follow-up once the source install is provisioned.

Franka on a ground plane, red cube in front, two RTX cameras (``camera1`` front
agent-view, ``camera2`` eye-in-hand on ``panda_hand``, for the LIBERO two-camera
contract). Action: 8-vector ``[dq0..dq6, gripper]`` (>0 open, <=0 close). Reward
is cube height; succeeds above ``_LIFT_SUCCESS_Z``. Lifecycle/obs skeleton:
``_isaac_scene_base.IsaacSceneBase``.

``camera1``/``camera2`` are legacy ordinal keys the HAL bridge maps via
``vla_feature_key`` fallback. TODO: pass canonical scene-side camera names
(``front``/``wrist``) through the isaac-sidecar protocol directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _isaac_scene_base import (
    IsaacSceneBase,
    franka_joint_positions,
    franka_joint_velocities,
)
from numpy.typing import NDArray

_ARM_DOF = 7
_ACTION_DIM = 8  # 7 arm joint deltas + 1 gripper command
_ARM_DELTA_SCALE = 0.05  # rad per unit action — keeps a unit action sane
_GRIPPER_OPEN = 0.04  # Franka finger joint upper (m)
_GRIPPER_CLOSED = 0.0
_LIFT_SUCCESS_Z = 0.10  # cube CoM height (m) counted as "lifted"
_CUBE_HALF = 0.025  # 5 cm cube → 2.5 cm half-extent
_AGENT_CAMERA_POS = np.array([2.5, 0.0, 1.25], dtype=np.float64)


class IsaacLiftScene(IsaacSceneBase):
    """A real Isaac Sim PhysX + RTX lift-cube scene, driven step-by-step."""

    warmup_steps = 3
    physics_substeps = 1

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_dim = _ACTION_DIM
        self._franka: Any = None
        self._cube: Any = None
        self._camera: Any = None
        self._cam_wrist: Any = None
        self._ArticulationAction: Any = None
        self._last_cube_z = 0.0

    def build(self) -> None:
        """Construct the stage: ground + Franka + cube + camera."""
        import isaacsim.core.utils.numpy.rotations as rot_utils
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators.examples.franka import Franka
        from isaacsim.sensors.camera import Camera

        self._ArticulationAction = ArticulationAction

        # NOTE: no device="cuda:0" — forcing GPU PhysX hangs the first
        # world.reset()/step warmup for minutes on an 8 GB-class laptop GPU;
        # default device renders the same scene in ~15 s.
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()
        self._cube = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0.45, 0.0, _CUBE_HALF]),
                scale=np.array([2 * _CUBE_HALF, 2 * _CUBE_HALF, 2 * _CUBE_HALF]),
                color=np.array([1.0, 0.1, 0.1]),
            )
        )
        self._franka = self._world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
        # camera1: front agent-view of the workspace.
        self._camera = Camera(
            prim_path="/World/Camera",
            resolution=(self.obs_width, self.obs_height),
        )
        # camera2: eye-in-hand wrist camera, parented to the gripper so it
        # follows the arm (mirrors the LIBERO agentview + eye_in_hand pair the
        # bowl_plate scene exposes; lets a two-camera rSkill clear the gate).
        self._cam_wrist = Camera(
            prim_path="/World/Franka/panda_hand/WristCam",
            resolution=(self.obs_width, self.obs_height),
        )
        self._world.reset()
        self._camera.initialize()
        self._cam_wrist.initialize()
        # Look down at the workspace from front-right.
        self._camera.set_world_pose(
            _AGENT_CAMERA_POS,
            rot_utils.euler_angles_to_quats(np.array([0, 35, 180]), degrees=True),
        )
        # Wrist camera: small forward+down offset from the hand frame, looking
        # along the gripper approach axis (same local pose as bowl_plate).
        self._cam_wrist.set_local_pose(
            np.array([0.05, 0.0, 0.02]),
            rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
            camera_axes="usd",
        )

    # ── IsaacSceneBase template methods ──────────────────────────────────────

    def _on_reset(self, rng: np.random.Generator) -> None:
        # Jitter the cube within a small reachable patch in front of the arm.
        cx = 0.45 + float(rng.uniform(-0.05, 0.05))
        cy = 0.0 + float(rng.uniform(-0.10, 0.10))
        self._cube.set_world_pose(np.array([cx, cy, _CUBE_HALF]), np.array([1.0, 0.0, 0.0, 0.0]))

    def _apply_action(self, action: NDArray[np.float32]) -> None:
        joints = np.asarray(self._franka.get_joint_positions(), dtype=np.float32)
        target = joints.copy()
        target[:_ARM_DOF] = joints[:_ARM_DOF] + action[:_ARM_DOF] * _ARM_DELTA_SCALE
        finger = _GRIPPER_OPEN if float(action[7]) > 0.0 else _GRIPPER_CLOSED
        if target.shape[0] >= _ARM_DOF + 2:
            target[_ARM_DOF] = finger
            target[_ARM_DOF + 1] = finger
        self._franka.get_articulation_controller().apply_action(
            self._ArticulationAction(joint_positions=target)
        )

    def _images(self) -> dict[str, NDArray[np.uint8]]:
        return {
            "camera1": self._grab(self._camera),
            "camera2": self._grab(self._cam_wrist),
        }

    def _state(self) -> NDArray[np.float32]:
        joints = np.asarray(self._franka.get_joint_positions(), dtype=np.float32).reshape(-1)
        cube_pos = np.asarray(self._cube.get_world_pose()[0], dtype=np.float32).reshape(-1)
        return np.concatenate([joints, cube_pos]).astype(np.float32)

    def _reward_terminated(self) -> tuple[float, bool]:
        self._last_cube_z = float(np.asarray(self._cube.get_world_pose()[0], dtype=np.float32)[2])
        return self._last_cube_z, self._last_cube_z > _LIFT_SUCCESS_Z

    def _extra_info(self) -> dict[str, Any]:
        return {"cube_z": self._last_cube_z}

    def _joint_positions(self) -> NDArray[np.float32] | None:
        return franka_joint_positions(self._franka)

    def _joint_velocities(self) -> NDArray[np.float32] | None:
        return franka_joint_velocities(self._franka)
