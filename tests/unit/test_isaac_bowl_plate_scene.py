"""GPU-free unit coverage for the Isaac bowl/plate scene helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "tools").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("repo root not found")


def _add_tools_to_path() -> None:
    tools = str(_repo_root() / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)


def test_before_render_hook_runs_before_rendered_steps() -> None:
    _add_tools_to_path()
    from _isaac_scene_base import IsaacSceneBase

    class _World:
        def __init__(self) -> None:
            self.events: list[str] = []

        def reset(self) -> None:
            self.events.append("reset")

        def step(self, *, render: bool) -> None:
            self.events.append(f"step:{render}")

    class _Scene(IsaacSceneBase):
        warmup_steps = 2
        physics_substeps = 3

        def __init__(self) -> None:
            super().__init__(
                obs_height=2, obs_width=2, instruction="x", success_key="ok", max_steps=5
            )
            self.action_dim = 1
            self._world = _World()

        def _before_render(self) -> None:
            self._world.events.append("before_render")

        def _apply_action(self, action: NDArray[np.float32]) -> None:
            self._world.events.append(f"action:{float(action[0])}")

        def _images(self) -> dict[str, NDArray[np.uint8]]:
            return {"camera1": np.zeros((2, 2, 3), dtype=np.uint8)}

        def _state(self) -> NDArray[np.float32]:
            return np.zeros(1, dtype=np.float32)

        def _reward_terminated(self) -> tuple[float, bool]:
            return 0.0, False

    scene = _Scene()
    scene.reset(seed=0)
    assert scene._world.events == [
        "reset",
        "before_render",
        "step:True",
        "before_render",
        "step:True",
    ]

    scene._world.events.clear()
    scene.step(np.asarray([1.0], dtype=np.float32))
    assert scene._world.events == [
        "action:1.0",
        "step:False",
        "step:False",
        "before_render",
        "step:True",
    ]


def test_bowl_plate_action_target_is_clamped_to_table_workspace() -> None:
    _add_tools_to_path()
    import isaac_bowl_plate_scene as scene_mod

    class _RotUtils:
        def rot_matrices_to_quats(self, _rot: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    class _IK:
        def __init__(self) -> None:
            self.target_position: NDArray[np.float64] | None = None

        def compute_end_effector_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            return np.asarray([0.50, 0.0, 0.20], dtype=np.float64), np.eye(3, dtype=np.float64)

        def compute_inverse_kinematics(
            self,
            *,
            target_position: NDArray[np.float64],
            target_orientation: NDArray[np.float64],
        ) -> tuple[SimpleNamespace, bool]:
            del target_orientation
            self.target_position = target_position
            return SimpleNamespace(joint_positions=np.zeros(7, dtype=np.float32)), True

    class _Controller:
        def __init__(self) -> None:
            self.applied: object | None = None

        def apply_action(self, action: object) -> None:
            self.applied = action

    class _Gripper:
        def __init__(self) -> None:
            self.action: str | None = None

        def forward(self, *, action: str) -> str:
            return action

        def apply_action(self, action: str) -> None:
            self.action = action

    class _Franka:
        def __init__(self) -> None:
            self.controller = _Controller()
            self.gripper = _Gripper()

        def get_articulation_controller(self) -> _Controller:
            return self.controller

    rollout = scene_mod.IsaacBowlPlateScene(
        obs_height=256,
        obs_width=256,
        instruction="put the bowl on the plate",
        success_key="is_success",
        max_steps=200,
    )
    ik = _IK()
    franka = _Franka()
    rollout._art_ik = ik
    rollout._rot_utils = _RotUtils()
    rollout._franka = franka

    rollout._apply_action(np.asarray([100.0, -100.0, 100.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    assert ik.target_position is not None
    np.testing.assert_allclose(
        ik.target_position,
        np.asarray([0.72, -0.32, 0.46], dtype=np.float64),
        rtol=0,
        atol=1e-9,
    )
    assert franka.controller.applied is not None
    assert franka.gripper.action == "close"
