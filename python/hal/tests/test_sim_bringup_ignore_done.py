"""ignore_done is injected only for robocasa scenes, never strict native backends (§1.15 fix)."""

from __future__ import annotations

import pytest
from openral_core import SimScene, load_scene_strict
from openral_hal.sim_bringup import _maybe_force_ignore_done


def test_robocasa_scene_gets_ignore_done_injected() -> None:
    # robocasa is the only backend that reads opts.ignore_done; deploy-sim
    # forces it True so continuous stepping never trips the episode guard.
    scene_env = load_scene_strict("scenes/sim/robocasa_panda_mobile_kitchen.yaml", SimScene)
    out = _maybe_force_ignore_done(scene_env)
    assert (out.scene.backend_options or {}).get("ignore_done") is True


def test_native_scene_not_injected() -> None:
    # tabletop_cube_push strictly rejects unknown backend_options keys; the gate
    # must leave native scenes untouched.
    scene_env = load_scene_strict("scenes/sim/tabletop_cube_push.yaml", SimScene)
    out = _maybe_force_ignore_done(scene_env)
    assert "ignore_done" not in (out.scene.backend_options or {})


def test_native_scene_builds_without_raising() -> None:
    # End-to-end: the old blind injection raised ROSConfigError on tabletop_push.
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, _seed = build_sim_env_from_yaml("scenes/sim/tabletop_cube_push.yaml")
    assert env is not None


def test_deploy_task_id_index_parsing_suite_gets_integer_index() -> None:
    # LIBERO suites parse task.id as "<suite>/<int>"; the deploy-promoted noop
    # task must carry a valid integer index (0) or the franka HAL never
    # configures ("libero task index '_hal_deploy_noop' is not an integer").
    from openral_hal.sim_bringup import _synthesise_deploy_task_id

    assert _synthesise_deploy_task_id("libero_spatial") == "libero_spatial/0"
    assert _synthesise_deploy_task_id("libero_10") == "libero_10/0"


def test_deploy_task_id_non_index_backend_stays_noop() -> None:
    # Backends that ignore task.id (so101, robocasa, native MjSpec) keep the
    # inert noop suffix they never read.
    from openral_hal.sim_bringup import _synthesise_deploy_task_id

    assert _synthesise_deploy_task_id("robocasa_kitchen") == "robocasa_kitchen/_hal_deploy_noop"
    assert _synthesise_deploy_task_id("tabletop_cube_push") == "tabletop_cube_push/_hal_deploy_noop"


def test_libero_deploy_scene_synthesises_parseable_task_id() -> None:
    # End-to-end through _load_scene_for_hal: the libero deploy scene (no task:)
    # must upcast to a SimScene whose task.id the LIBERO backend accepts.
    from openral_hal.sim_bringup import _load_scene_for_hal

    scene = _load_scene_for_hal("scenes/deploy/libero_pnp.yaml")
    assert scene.task.id == "libero_spatial/0"
    # The LIBERO backend's own parser must accept it (no integer-parse error).
    libero = pytest.importorskip("openral_sim.backends.libero")
    assert libero._parse_task_id(scene.task.id, scene.scene.id) == 0
