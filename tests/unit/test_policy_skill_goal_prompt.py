"""The resident VLA shim takes each goal's prompt per step, without a reload.

The skill runner keeps one policy resident per ``(rskill_id, revision)`` and
calls ``set_goal_prompt`` when a new goal reuses it. These tests drive the real
``_make_policy_adapter_skill`` shim over a real manifest + robot description;
the only double is the adapter at the ``PolicyAdapter`` protocol boundary,
which records the instruction each inference was conditioned on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from openral_core import JointState, RobotDescription, VLASpec, WorldState
from openral_rskill.loader import load_rskill_manifest

_ROOT = Path(__file__).resolve().parents[2]
_SKILL_DIR = _ROOT / "rskills" / "gr00t-n17-b1k-turning-on-radio"


class _RecordingAdapter:
    """``PolicyAdapter`` that records every instruction and reset."""

    spec = VLASpec(id="gr00t_b1k", weights_uri=str(_SKILL_DIR))
    device = "cpu"

    def __init__(self) -> None:
        self.instructions: list[str] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def step(self, observation: dict[str, object], instruction: str) -> NDArray[np.float32]:
        del observation
        self.instructions.append(instruction)
        return np.zeros(23, dtype=np.float32)

    def close(self) -> None:
        return None


def _world_state(robot: RobotDescription) -> WorldState:
    return WorldState(
        stamp_ns=1,
        joint_state=JointState(
            name=[joint.name for joint in robot.joints],
            position=[0.0] * len(robot.joints),
            velocity=[0.0] * len(robot.joints),
            stamp_ns=1,
        ),
        policy_state=np.zeros(61, dtype=np.float32).tolist(),
        diagnostics={"policy_state": "ok"},
    )


def _skill(
    adapter: _RecordingAdapter, *, default_prompt: str | None = None
) -> tuple[Any, RobotDescription]:
    from openral_rskill_ros.rskill_runner_node import _make_policy_adapter_skill

    robot = RobotDescription.from_yaml(str(_ROOT / "robots" / "r1pro" / "robot.yaml"))
    manifest = load_rskill_manifest(str(_SKILL_DIR))
    if default_prompt is not None:
        manifest = manifest.model_copy(update={"default_prompt": default_prompt})
    skill = _make_policy_adapter_skill(
        manifest=manifest, adapter=adapter, prompt="first subtask", description=robot
    )
    adapter.instructions.clear()  # drop the activate-time warm-up step
    return skill, robot


def test_set_goal_prompt_conditions_the_next_steps_and_drops_the_old_chunk() -> None:
    adapter = _RecordingAdapter()
    skill, robot = _skill(adapter)
    skill.step(_world_state(robot))
    resets_before = adapter.resets

    skill.set_goal_prompt("second subtask")
    skill.step(_world_state(robot))

    assert adapter.instructions == ["first subtask", "second subtask"]
    # The previous goal's buffered chunk / prefetch was inferred from the old
    # prompt; it must be dropped before the new goal's first step.
    assert adapter.resets == resets_before + 1


def test_set_goal_prompt_empty_falls_back_to_the_checkpoint_prompt() -> None:
    adapter = _RecordingAdapter()
    skill, robot = _skill(adapter, default_prompt="turn on the radio")
    skill.set_goal_prompt("")
    skill.step(_world_state(robot))
    assert adapter.instructions == ["turn on the radio"]
