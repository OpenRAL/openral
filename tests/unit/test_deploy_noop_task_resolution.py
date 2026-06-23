"""Deploy-scene taskless adapters resolve to concrete simulator tasks.

``openral deploy sim`` converts taskless ``DeployScene`` YAML into a synthetic
``<scene>/_hal_deploy_noop`` task. Backends that pass task ids into an external
gym factory must map that synthetic id to a real default task before building.
"""

from __future__ import annotations

import os

import pytest
from openral_core import PhysicsBackend, SceneSpec, SimEnvironment, TaskSpec, VLASpec
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.maniskill3 import _action_dim_from_space, _task_id_for_env
from openral_sim.backends.robotwin import _task_name_for_env as _robotwin_task_name_for_env
from openral_sim.backends.simpler_env import _headless_display_guard, _task_name_for_env


def _env(scene_id: str, task_id: str, *, deploy_task_id: str | None = None) -> SimEnvironment:
    options: dict[str, object] = {}
    if deploy_task_id is not None:
        options["deploy_task_id"] = deploy_task_id
    return SimEnvironment(
        robot_id="franka_panda" if scene_id == "maniskill3" else "widowx",
        scene=SceneSpec(id=scene_id, backend=PhysicsBackend.SAPIEN, backend_options=options),
        task=TaskSpec(id=task_id, scene_id=scene_id, instruction="hold"),
        vla=VLASpec(id="noop", weights_uri="stub"),
        seed=0,
    )


def test_maniskill_deploy_noop_uses_default_task() -> None:
    env_cfg = _env("maniskill3", "maniskill3/_hal_deploy_noop")

    assert _task_id_for_env(env_cfg) == "PickCube-v1"


def test_maniskill_deploy_noop_allows_backend_override() -> None:
    env_cfg = _env(
        "maniskill3",
        "maniskill3/_hal_deploy_noop",
        deploy_task_id="LiftCube-v1",
    )

    assert _task_id_for_env(env_cfg) == "LiftCube-v1"


def test_maniskill_regular_task_still_parses_task_id() -> None:
    env_cfg = _env("maniskill3", "maniskill3/StackCube-v1")

    assert _task_id_for_env(env_cfg) == "StackCube-v1"


def test_simpler_env_deploy_noop_uses_default_task() -> None:
    env_cfg = _env("simpler_env", "simpler_env/_hal_deploy_noop")

    assert _task_name_for_env(env_cfg) == "widowx_carrot_on_plate"


def test_simpler_env_deploy_noop_allows_backend_override() -> None:
    env_cfg = _env(
        "simpler_env",
        "simpler_env/_hal_deploy_noop",
        deploy_task_id="widowx_spoon_on_towel",
    )

    assert _task_name_for_env(env_cfg) == "widowx_spoon_on_towel"


def test_robotwin_deploy_noop_uses_default_task() -> None:
    env_cfg = _env("robotwin", "robotwin/_hal_deploy_noop")

    assert _robotwin_task_name_for_env(env_cfg) == "lift_pot"


def test_robotwin_deploy_noop_allows_backend_override() -> None:
    env_cfg = _env(
        "robotwin",
        "robotwin/_hal_deploy_noop",
        deploy_task_id="pick_apple_messy",
    )

    assert _robotwin_task_name_for_env(env_cfg) == "pick_apple_messy"


def test_simpler_env_regular_task_still_parses_task_id() -> None:
    env_cfg = _env("simpler_env", "simpler_env/widowx_carrot_on_plate")

    assert _task_name_for_env(env_cfg) == "widowx_carrot_on_plate"


def test_simpler_env_headless_display_guard_restores_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISPLAY", ":1")

    with _headless_display_guard(view_requested=False):
        assert "DISPLAY" not in os.environ

    assert os.environ["DISPLAY"] == ":1"


def test_sapien_action_dim_uses_single_env_action_width() -> None:
    class Space:
        shape = (1, 8)

    assert _action_dim_from_space(Space()) == 8


def test_sapien_action_dim_requires_shaped_space() -> None:
    class Space:
        shape = ()

    try:
        _action_dim_from_space(Space())
    except ROSConfigError as exc:
        assert "action_space has no shape" in str(exc)
    else:
        raise AssertionError("expected ROSConfigError")
