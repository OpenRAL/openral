"""Smoke test for the OpenArm v2 tabletop scene + robosuite OSC wiring.

Loads the scene end-to-end (real MJCF compile, real MjSim, real OSC),
resets it, runs a handful of zero-action steps, and asserts the
observation dict matches the rSkill manifest's IO contract (16-D state,
3 RGB cameras keyed top/wrist_left/wrist_right, task instruction
present). No mocks, no stubs — CLAUDE.md §1.11 / §5.4.

Skipped automatically when ``robosuite`` is not installed (CI without
the ``robocasa`` extras group).
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from openral_sim.rollout import SimRollout

# Use `importlib.util.find_spec` + `pytestmark` rather than module-level
# `pytest.importorskip`: with `tests/sim/__init__.py` making this
# directory a Package, a Skipped raised at module-import time poisons the
# whole `tests/sim` Package collection ("found no collectors for ..." on
# every sibling). Deferring the decision to `pytestmark` keeps this
# module importable when optional deps are missing.
_ROBOSUITE_MISSING = importlib.util.find_spec("robosuite") is None

pytestmark = [
    pytest.mark.skipif(
        _ROBOSUITE_MISSING,
        reason=(
            "OpenArm tabletop scene needs robosuite>=1.5 "
            "(just sync --all-packages --group robocasa)."
        ),
    ),
]


def _build_env_cfg() -> object:
    """Construct a real SimEnvironment that selects the openarm scene."""
    from openral_core import Pose6D, SceneSpec, SimEnvironment, TaskSpec, VLASpec

    scene = SceneSpec(
        id="openarm_tabletop_pnp",
        backend="mujoco",
        observation_height=256,
        observation_width=256,
        backend_options={"render_width": 128, "render_height": 128, "max_steps": 32},
    )
    task = TaskSpec(
        id="openarm/pnp_cube_to_drawer",
        scene_id="openarm_tabletop_pnp",
        instruction="pick the red cube and place it in the drawer",
        max_steps=32,
    )
    vla = VLASpec(id="mock-noop", weights_uri="mock-noop", device="cpu")
    # The openarm scene requires an explicit base mount (ADR-0002 Amendment 3).
    base_pose = Pose6D(xyz=(0.20, 0.0, 0.55), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="world")
    return SimEnvironment(
        robot_id="openarm",
        scene=scene,
        task=task,
        vla=vla,
        base_pose=base_pose,
        seed=0,
        n_episodes=1,
    )


def test_openarm_tabletop_scene_loads_and_steps() -> None:
    """Scene compiles, exposes the IO contract, and steps under zero action."""
    from openral_sim import SCENES
    from openral_sim.backends.openarm_robosuite import env as _env  # noqa: F401

    env_cfg = _build_env_cfg()
    factory = SCENES.get("openarm_tabletop_pnp")
    rollout = factory(env_cfg)

    try:
        _assert_loads_and_steps(rollout)
    finally:
        rollout.close()


def test_openarm_tabletop_scene_builds_through_loader() -> None:
    """Same scene builds + reset/steps via the deploy-sim ``build_sim_env_from_yaml`` loader.

    Regression guard for the loader dropping ``SimScene.base_pose``:
    the ``openarm_tabletop_pnp`` scene mandates ``base_pose`` at compose time
    (ADR-0002 Amendment 3), so before the fix this raised at compose time —
    BEFORE the HAL/action-dim probe was even reached. The loader now threads
    the YAML's ``base_pose`` into the composed ``SimEnvironment``, so the scene
    builds through the loader exactly as it does through the direct factory.
    """
    pytest.importorskip("mujoco")
    from openral_hal.sim_bringup import build_sim_env_from_yaml
    from openral_sim.backends.openarm_robosuite import env as _env  # noqa: F401

    rollout, seed = build_sim_env_from_yaml(
        "scenes/sim/openarm_tabletop.yaml", robot_id_fallback="openarm"
    )
    assert seed == 0
    try:
        _assert_loads_and_steps(rollout)
    finally:
        rollout.close()


def test_openarm_tabletop_deploy_scene_builds_through_loader() -> None:
    """DeployScene YAMLs (env-only, no task) compose through the HAL loader.

    Regression guard for ``_load_scene_for_hal``'s DeployScene-to-SimScene
    upcast: ``DeployScene`` has no ``metadata`` field (only ``SimScene`` /
    ``BenchmarkScene`` carry one). Earlier code mistakenly read
    ``deploy.metadata`` when promoting, which would raise ``AttributeError``
    at runtime on any DeployScene YAML. mypy --strict caught it in CI on
    PR #274; this runtime test (the only one that loads a
    ``scenes/deploy/*.yaml`` through ``build_sim_env_from_yaml``) closes
    the coverage gap that let the bug ship.

    Mirrors :func:`test_openarm_tabletop_scene_builds_through_loader` but
    points at the env-only ``scenes/deploy/openarm_tabletop.yaml`` sibling
    (no ``task:`` block — exercises the DeployScene branch of
    ``_load_scene_for_hal``, which synthesises a noop :class:`TaskSpec`).
    The HAL's ``SimAttachedHAL`` never reads ``task.id`` / ``.instruction``
    / ``.max_steps`` / ``.success_key`` (it drives ``env.step`` directly),
    so the noop task is invisible at runtime — the openarm env itself
    bakes the task description into the obs (it's not derived from the
    SimEnvironment.task spec).
    """
    pytest.importorskip("mujoco")
    from openral_hal.sim_bringup import build_sim_env_from_yaml
    from openral_sim.backends.openarm_robosuite import env as _env  # noqa: F401

    # The loader succeeding at all is the substance of this test: before
    # the fix, ``_load_scene_for_hal`` raised ``AttributeError`` on
    # ``deploy.metadata`` before reaching SimEnvironment composition.
    rollout, seed = build_sim_env_from_yaml(
        "scenes/deploy/openarm_tabletop.yaml", robot_id_fallback="openarm"
    )
    # DeployScene has no ``seed:`` field; the upcast SimScene falls back to
    # the schema default of 0 (see ``SimScene.seed: int = 0``).
    assert seed == 0
    try:
        obs = rollout.reset(seed=0)
        # IO contract matches the openarm_tabletop_pnp scene (same scene
        # block as the SimScene sibling YAML — only the task block differs).
        assert set(obs).issuperset({"images", "state", "task"})
        assert obs["state"].shape == (16,), f"state dim={obs['state'].shape} expected (16,)"
        assert obs["state"].dtype == np.float32
        assert set(obs["images"]) == {"top", "wrist_left", "wrist_right"}

        # Step under zero action — proves the noop task plumbing doesn't
        # trip the underlying robosuite OSC + sim.step path.
        zero_action = np.zeros(16, dtype=np.float32)
        for _ in range(3):
            step = rollout.step(zero_action)
            assert step.observation["state"].shape == (16,)
    finally:
        rollout.close()


def _assert_loads_and_steps(rollout: SimRollout) -> None:
    """Assert the openarm rollout exposes its IO contract and steps cleanly.

    Shared body for both the direct-factory and loader-path tests; the caller
    owns ``rollout.close()``.
    """
    obs = rollout.reset(seed=0)
    assert set(obs).issuperset({"images", "state", "task"})
    assert obs["task"] == "pick the red cube and place it in the drawer"

    # IO contract from rskills/pi05-openarm-bimanual-pick-pipe-nf4/rskill.yaml.
    assert obs["state"].shape == (16,), f"state dim={obs['state'].shape} expected (16,)"
    assert obs["state"].dtype == np.float32
    assert set(obs["images"]) == {"top", "wrist_left", "wrist_right"}
    for cam, frame in obs["images"].items():
        assert frame.ndim == 3 and frame.shape[2] == 3, (
            f"camera {cam!r} returned shape {frame.shape}, expected HxWx3"
        )
        assert frame.dtype == np.uint8

    # Step under zero action — confirms the OSC + sim.step path doesn't
    # raise. Action is 16-D (7+1 per arm: arms + grippers), matching the
    # scene's joint count / the rSkill action_contract.dim.
    zero_action = np.zeros(16, dtype=np.float32)
    for _ in range(5):
        step = rollout.step(zero_action)
        assert step.observation["state"].shape == (16,)
        assert "drawer_pos" in step.info

    # Render path: top camera frame matches what reset() returned.
    rendered = rollout.render()
    assert rendered is not None and rendered.shape == obs["images"]["top"].shape

    # mujoco_handles() must expose the live (model, data) so
    # `openral sim run --view` can attach a passive viewer. It is a duck-typed
    # extension (not on the SimRollout Protocol), so resolve it via getattr.
    mujoco_handles = getattr(rollout, "mujoco_handles", None)
    assert mujoco_handles is not None, "openarm rollout must expose mujoco_handles()"
    model, _data = mujoco_handles()
    assert model.nu == 16, f"expected 16 motor actuators, got {model.nu}"
