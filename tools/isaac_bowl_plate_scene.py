"""Isaac Sim table + bowl + plate Franka scene with the LIBERO obs/action contract.

Runs under the Isaac Sim py3.11 venv only (imported by ``isaac_sidecar.py`` after
``SimulationApp`` is live). Built on Isaac Sim **core** — no Isaac Lab — proving
that moving a robot and doing end-effector control needs neither Isaac Lab nor its
OSC term:

* arm motion: ``ArticulationController.apply_action`` on the core ``Franka``;
* end-effector control: the core ``isaacsim.robot_motion.motion_generation`` Lula
  kinematics solver (``LulaKinematicsSolver`` + ``ArticulationKinematicsSolver``)
  does position-delta inverse kinematics on the ``right_gripper`` frame.

The scene mirrors the LIBERO contract so a LIBERO-finetuned rSkill (act-libero /
smolvla-libero) can drive it through ``openral sim run`` unchanged:

* obs ``images``: ``camera1`` = front agent-view, ``camera2`` = secondary prop view;
* obs ``state``: 8-D ``[eef_pos(3) ‖ eef_axisangle(3) ‖ gripper_qpos(2)]``
  (matches ``openral_sim.backends.libero._wrap_obs``);
* action: 7-D OSC-pose delta ``[dx, dy, dz, drx, dry, drz, gripper]`` — the
  position delta drives a Lula IK target (orientation held); ``gripper>0`` closes.

The lifecycle/obs skeleton lives in :class:`_isaac_scene_base.IsaacSceneBase`;
this class supplies only the build + control + obs specifics. Props are
out-of-distribution for a LIBERO policy, so this validates that the pipeline RUNS
and the arm MOVES — not task success.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
from _isaac_scene_base import (
    IsaacSceneBase,
    franka_joint_positions,
    franka_joint_velocities,
)
from numpy.typing import NDArray

_ARM_DOF = 7
_ACTION_DIM = 7  # LIBERO OSC-pose delta: dpos(3) + drot(3) + gripper(1)
_POS_SCALE = 0.03  # metres per unit position-delta action — visible but stable
_GRIPPER_OPEN = 0.04
_GRIPPER_CLOSED = 0.0
_TABLE_TOP_Z = 0.0  # table surface at the robot base height
_OBJ_Z = 0.03
_TASK_CENTER = np.array([0.50, 0.0, _OBJ_Z], dtype=np.float64)
_WORKSPACE_LOW = np.array([0.32, -0.32, 0.07], dtype=np.float64)
_WORKSPACE_HIGH = np.array([0.72, 0.32, 0.46], dtype=np.float64)


def _quat_wxyz_to_axisangle(quat: NDArray[np.float32]) -> NDArray[np.float32]:
    """(w,x,y,z) quaternion → (3,) axis-angle. Mirrors libero._quat_to_axisangle."""
    eps = 1e-10
    w = float(np.clip(quat[0], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den > eps:
        angle = 2.0 * np.arccos(w)
        axis = np.asarray(quat[1:], dtype=np.float32) / den
        return (axis * angle).astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def _look_at_quat_wxyz(
    cam_pos: NDArray[np.float64],
    target: NDArray[np.float64],
    world_up: list[float],
    rot_utils: Any,
) -> NDArray[np.float64]:
    """World (w,x,y,z) quat aiming a USD camera (view = local −Z, up = +Y) at ``target``."""
    d = np.asarray(target, float) - np.asarray(cam_pos, float)
    d = d / (np.linalg.norm(d) + 1e-9)  # forward = −Z_cam
    up = np.asarray(world_up, float)
    right = np.cross(d, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(d, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    true_up = np.cross(right, d)
    rot = np.column_stack([right, true_up, -d])  # cols: X=right, Y=up, Z=−forward
    return np.asarray(rot_utils.rot_matrices_to_quats(rot.astype(np.float64)))


class IsaacBowlPlateScene(IsaacSceneBase):
    """A real Isaac Sim core table/bowl/plate Franka scene, LIBERO-shaped."""

    warmup_steps = 4
    physics_substeps = 4  # sim steps per action so the arm tracks the IK target

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_dim = _ACTION_DIM
        self._franka: Any = None
        self._bowl: Any = None
        self._cam_agent: Any = None
        self._cam_wrist: Any = None
        self._art_ik: Any = None
        self._rot_utils: Any = None
        self._last_ik_ok = False

    def build(self) -> None:
        import isaacsim.core.utils.numpy.rotations as rot_utils
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCylinder, FixedCuboid
        from isaacsim.robot.manipulators.examples.franka import Franka
        from isaacsim.robot_motion.motion_generation import (
            ArticulationKinematicsSolver,
            LulaKinematicsSolver,
            interface_config_loader,
        )
        from isaacsim.sensors.camera import Camera

        self._rot_utils = rot_utils
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()

        # Flat table surface in front of the arm.
        self._world.scene.add(
            FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=np.array([0.55, 0.0, _TABLE_TOP_Z - 0.02]),
                scale=np.array([0.7, 1.0, 0.04]),
                color=np.array([0.45, 0.30, 0.18]),
            )
        )
        # Plate: a thin cylinder primitive (no YCB plate USD ships in 5.1).
        self._world.scene.add(
            DynamicCylinder(
                prim_path="/World/Plate",
                name="plate",
                position=np.array([0.50, -0.18, _OBJ_Z]),
                radius=0.10,
                height=0.012,
                color=np.array([0.9, 0.9, 0.92]),
            )
        )
        # Bowl: prefer the real YCB mesh; fall back to a primitive if offline.
        self._add_bowl(position=np.array([0.50, 0.18, _OBJ_Z]))

        self._franka = self._world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

        # Front agent-view camera looking across the full bowl/plate workspace.
        self._cam_agent = Camera(
            prim_path="/World/AgentView", resolution=(self.obs_width, self.obs_height)
        )
        # Secondary prop view. A true child wrist camera clips into the Franka hand
        # in headless RTX, yielding black/hand-only frames; this stable view keeps
        # the LIBERO two-camera contract while making both task props visible.
        self._cam_wrist = Camera(
            prim_path="/World/WristView",
            resolution=(self.obs_width, self.obs_height),
        )

        self._world.reset()
        self._cam_agent.initialize()
        self._cam_wrist.initialize()
        agent_pos = np.array([1.25, 0.0, 0.90], dtype=np.float64)
        self._cam_agent.set_world_pose(
            agent_pos,
            _look_at_quat_wxyz(agent_pos, _TASK_CENTER, [0.0, 0.0, 1.0], rot_utils),
            camera_axes="usd",
        )

        # Lula IK solver (core motion_generation, NOT Isaac Lab).
        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        kin = LulaKinematicsSolver(**cfg)
        self._art_ik = ArticulationKinematicsSolver(self._franka, kin, "right_gripper")
        base_pos, base_quat = self._franka.get_world_pose()
        kin.set_robot_base_pose(np.asarray(base_pos), np.asarray(base_quat))

        self._cam_wrist.set_world_pose(
            np.array([1.50, 0.0, 1.00], dtype=np.float64),
            rot_utils.euler_angles_to_quats(np.array([0, 45, 180]), degrees=True),
        )

    def _add_bowl(self, position: NDArray[np.float32]) -> None:
        """Reference the YCB bowl USD; fall back to a primitive bowl if offline."""
        try:
            from isaacsim.core.prims import SingleRigidPrim
            from isaacsim.core.utils.stage import add_reference_to_stage
            from isaacsim.storage.native import get_assets_root_path

            root = get_assets_root_path()
            if root is None:
                raise RuntimeError("no assets root")
            usd = root + "/Isaac/Props/YCB/Axis_Aligned/024_bowl.usd"
            add_reference_to_stage(usd, "/World/Bowl")
            self._bowl = SingleRigidPrim(
                "/World/Bowl", name="bowl", position=position, scale=np.array([1.0, 1.0, 1.0])
            )
            self._world.scene.add(self._bowl)
        except Exception:
            from isaacsim.core.api.objects import DynamicCylinder

            self._bowl = self._world.scene.add(
                DynamicCylinder(
                    prim_path="/World/Bowl",
                    name="bowl",
                    position=position,
                    radius=0.07,
                    height=0.05,
                    color=np.array([0.2, 0.4, 0.85]),
                )
            )

    # ── IsaacSceneBase template methods ──────────────────────────────────────

    def _apply_action(self, action: NDArray[np.float32]) -> None:
        pos, rot_mat = self._art_ik.compute_end_effector_pose()
        quat = self._rot_utils.rot_matrices_to_quats(np.asarray(rot_mat))
        target = np.clip(
            np.asarray(pos, dtype=np.float64) + action[:3].astype(np.float64) * _POS_SCALE,
            _WORKSPACE_LOW,
            _WORKSPACE_HIGH,
        )
        ik_action, ok = self._art_ik.compute_inverse_kinematics(
            target_position=target, target_orientation=np.asarray(quat)
        )
        self._last_ik_ok = bool(ok)
        if ok and ik_action.joint_positions is not None:
            self._franka.get_articulation_controller().apply_action(ik_action)
        # Gripper: LIBERO convention action[6] > 0 → close.
        close = float(action[6]) > 0.0
        with contextlib.suppress(Exception):
            self._franka.gripper.apply_action(
                self._franka.gripper.forward(action="close" if close else "open")
            )

    def _images(self) -> dict[str, NDArray[np.uint8]]:
        return {"camera1": self._grab(self._cam_agent), "camera2": self._grab(self._cam_wrist)}

    def _state(self) -> NDArray[np.float32]:
        pos, rot_mat = self._art_ik.compute_end_effector_pose()
        quat = self._rot_utils.rot_matrices_to_quats(np.asarray(rot_mat))
        axisangle = _quat_wxyz_to_axisangle(np.asarray(quat, dtype=np.float32))
        gripper = np.asarray(self._franka.get_joint_positions(), dtype=np.float32)[_ARM_DOF:]
        if gripper.shape[0] < 2:
            gripper = np.zeros(2, dtype=np.float32)
        return np.concatenate([np.asarray(pos, dtype=np.float32), axisangle, gripper[:2]]).astype(
            np.float32
        )

    def _reward_terminated(self) -> tuple[float, bool]:
        return 0.0, False  # OOD task — pipeline/motion check, not task success

    def _extra_info(self) -> dict[str, Any]:
        return {"ik_ok": self._last_ik_ok}

    def _joint_positions(self) -> NDArray[np.float32] | None:
        return franka_joint_positions(self._franka)

    def _joint_velocities(self) -> NDArray[np.float32] | None:
        return franka_joint_velocities(self._franka)
