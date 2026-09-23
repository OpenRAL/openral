"""Deploy-sim bring-up stays scene-id agnostic: backends own their deploy semantics."""

from __future__ import annotations

import pytest


def test_native_scene_builds_without_raising() -> None:
    # tabletop_push strictly rejects unknown backend_options keys; bring-up must
    # not inject any (the old robocasa-only `ignore_done` injection is gone —
    # robocasa's own `enable_continuous` sets it on the live env).
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, _seed = build_sim_env_from_yaml("scenes/sim/tabletop_cube_push.yaml")
    assert env is not None


def test_deploy_task_id_is_the_same_inert_suffix_for_every_scene() -> None:
    from openral_hal.sim_bringup import _synthesise_deploy_task_id

    assert _synthesise_deploy_task_id("libero_spatial") == "libero_spatial/_hal_deploy_noop"
    assert _synthesise_deploy_task_id("tabletop_cube_push") == "tabletop_cube_push/_hal_deploy_noop"


def test_libero_deploy_scene_task_id_parses_to_task_zero() -> None:
    # End-to-end through _load_scene_for_hal: the libero deploy scene (no task:)
    # upcasts to a SimScene whose inert task id the LIBERO parser maps to task 0.
    from openral_hal.sim_bringup import _load_scene_for_hal

    scene = _load_scene_for_hal("scenes/deploy/libero_pnp.yaml")
    assert scene.task.id == "libero_spatial/_hal_deploy_noop"
    libero = pytest.importorskip("openral_sim.backends.libero")
    assert libero._parse_task_id(scene.task.id, scene.scene.id) == 0
