"""Sim test: deploy sim grades the dispatched goal's prompt (``placement_success``).

Boots the shipped ``scenes/deploy/isaac_openarm_warehouse.yaml`` with a
``placement_success`` block through the real deploy loader (task
``isaac_sim/_hal_deploy_noop``, empty instruction) on a real Isaac Sim, wrapped
in the real ``SimAttachedHAL``:

* the env builds and the HAL connects -- the regression: the empty instruction
  raised ``placement_success has no entry for the scene's instruction ''``;
* a prompt the scene does not grade logs ``sim.placement_grade_unavailable``;
* the graded prompt's rubric goes live on ``set_active_task``: the cracker box,
  resting in its region, reaches ``complete`` once settled, without ending the
  deploy episode, and the goal's end logs ``sim.placement_grade_final``.

Skip policy: as ``test_openarm_hal_isaac_warehouse``.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import yaml

from tests.sim.conftest import _repo_root, _sidecar_python_available

_WIRE_MISSING = [m for m in ("zmq", "msgpack") if importlib.util.find_spec(m) is None]

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        bool(_WIRE_MISSING),
        reason="isaac_sim wire needs " + ", ".join(_WIRE_MISSING) + " (just sync --group isaacsim)",
    ),
    pytest.mark.skipif(
        not _sidecar_python_available(),
        reason="Isaac Sim sidecar venv not provisioned (set OPENRAL_ISAAC_SIDECAR_PYTHON)",
    ),
]

_PROMPT = "leave the cracker box on the pallet"
_SUGAR = "put the sugar box in the aisle"


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    import openral_sim.backends  # noqa: F401 — registers the isaac_sim scene factory
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    scene = yaml.safe_load(
        (_repo_root() / "scenes/deploy/isaac_openarm_warehouse.yaml").read_text()
    )
    scene["scene"]["backend_options"]["placement_success"] = {
        "regions": {
            # Around the cracker box's resting place on the pallet (spawned at x -0.55,
            # y 4.85; the deck top is z 0.21), and an empty patch of aisle floor.
            "pallet_spot": {"min_xyz": [-0.75, 4.65, 0.15], "max_xyz": [-0.35, 5.0, 0.5]},
            "aisle": {"min_xyz": [-2.0, 6.0, 0.0], "max_xyz": [-1.5, 6.5, 0.3]},
        },
        "tasks": {
            _PROMPT: [{"object": "cracker_box", "target": "pallet_spot"}],
            _SUGAR: [{"object": "sugar_box", "target": "aisle"}],
        },
    }
    path = tmp_path_factory.mktemp("scene") / "isaac_openarm_warehouse_graded.yaml"
    path.write_text(yaml.safe_dump(scene))
    sim_env, _seed = build_sim_env_from_yaml(str(path))
    assert sim_env.task.instruction == ""
    yield sim_env
    sim_env.close()


def _lines(out: str, event: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len(event) + 1 :])
        for line in out.splitlines()
        if line.startswith(f"{event} ")
    ]


def test_deploy_sim_grades_the_dispatched_prompt(
    env: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL

    desc = RobotDescription.from_yaml(str(_repo_root() / "robots" / "openarm" / "robot.yaml"))
    hal = SimAttachedHAL(env, desc)
    hal.connect()  # the HAL connects with placement_success and an empty task

    # No goal yet: nothing graded, and the sidecar keeps no rubric active.
    assert "placement_grade" not in env.step(np.full(env.action_dim, np.nan, np.float32)).info

    hal.set_active_task("wave hello")
    (unavailable,) = _lines(capsys.readouterr().out, "sim.placement_grade_unavailable")
    assert unavailable["prompt"] == "wave hello"
    assert unavailable["graded_prompts"] == sorted([_PROMPT, _SUGAR])

    hal.set_active_task(_PROMPT)
    for _ in range(60):  # settle_s 1 s at 30 Hz, plus margin
        hal.idle_step()
    grades = _lines(capsys.readouterr().out, "sim.placement_grade")
    assert grades and grades[-1]["grade"] == "complete", grades
    assert grades[-1]["prompt"] == _PROMPT and grades[-1]["goal_active"] is True

    # Complete never ends a deploy episode: the scene only reports it.
    result = env.step(np.full(env.action_dim, np.nan, np.float32))
    assert result.info["placement_grade"] == "complete"
    assert result.terminated is False

    hal.set_active_task("")
    (final,) = _lines(capsys.readouterr().out, "sim.placement_grade_final")
    assert (final["prompt"], final["grade"], final["best_grade"]) == (
        _PROMPT,
        "complete",
        "complete",
    )
    assert final["best_score"] == 1.0

    # The next goal's rubric replaces it: the sugar box is not in the aisle.
    assert env.set_task(_SUGAR)["graded"] is True
    assert env.step(np.full(env.action_dim, np.nan, np.float32)).info["placement_grade"] == "fail"
    hal.disconnect()
