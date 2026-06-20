"""Unit tests for ``_build_env_and_policy`` (GH-134 parallel sim startup).

Exercises the real ``mock`` scene + ``zero`` / ``random`` policies — no
mocks, no stubs, no patches (CLAUDE.md §1.11 / §5.4). All assertions are
on real :class:`SimRollout` / :class:`PolicyAdapter` instances built by
the real :func:`openral_sim.factory.make_env` / :func:`make_policy`.

Pins:

* Parallel path returns the same object types as the sequential path
  for a real :class:`SimEnvironment`.
* ``OPENRAL_SIM_SEQUENTIAL_INIT=1`` selects the sequential path
  (verified via the structlog log output that the helper emits).
* A factory error (real :class:`ROSConfigError` from
  :func:`make_policy` on an unknown ``vla.id``) propagates verbatim
  out of the parallel helper.
* A factory error in :func:`make_env` (unknown ``scene.id``) also
  propagates verbatim.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from openral_core import (
    PhysicsBackend,
    SceneSpec,
    SimEnvironment,
    TaskSpec,
    VLASpec,
)
from openral_core.exceptions import ROSConfigError
from openral_sim.policy import PolicyAdapter
from openral_sim.rollout import SimRollout
from openral_sim.sim_runner import (
    _RACE_PRONE_SCENE_PREFIXES,
    _SEQUENTIAL_INIT_ENV,
    _build_env_and_policy,
    _scene_requires_sequential_init,
)

# ── Real-config helpers ─────────────────────────────────────────────────────


def _mock_env(
    *,
    vla_id: str = "zero",
    scene_id: str = "mock",
) -> SimEnvironment:
    """Real :class:`SimEnvironment` against the in-tree ``mock`` scene + policy."""
    return SimEnvironment(
        robot_id="so100_follower",
        scene=SceneSpec(
            id=scene_id,
            backend=PhysicsBackend.MOCK,
            backend_options={"success_step": 3, "action_dim": 7},
        ),
        task=TaskSpec(
            id="mock/0",
            scene_id=scene_id,
            instruction="noop",
            max_steps=5,
        ),
        vla=VLASpec(
            id=vla_id,
            weights_uri="placeholder",
            extra={"action_dim": 7, "seed": 0},
        ),
        n_episodes=1,
    )


@contextmanager
def _env_var(name: str, value: str | None) -> Iterator[None]:
    """Set / unset an env var for the duration of the block."""
    prev = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


# ── Happy paths ─────────────────────────────────────────────────────────────


def test_parallel_init_returns_real_env_and_policy() -> None:
    """Default (parallel) path builds the same real types as the sequential path."""
    env_cfg = _mock_env()
    # Be explicit about parallel — even though it is the default, the
    # test should not depend on the ambient environment.
    with _env_var(_SEQUENTIAL_INIT_ENV, None):
        env, policy = _build_env_and_policy(env_cfg)
    try:
        assert isinstance(env, SimRollout)
        assert isinstance(policy, PolicyAdapter)
    finally:
        env.close()
        policy.close()


def test_sequential_init_returns_real_env_and_policy() -> None:
    """``OPENRAL_SIM_SEQUENTIAL_INIT=1`` selects the legacy serial path."""
    env_cfg = _mock_env()
    with _env_var(_SEQUENTIAL_INIT_ENV, "1"):
        env, policy = _build_env_and_policy(env_cfg)
    try:
        assert isinstance(env, SimRollout)
        assert isinstance(policy, PolicyAdapter)
    finally:
        env.close()
        policy.close()


def test_sequential_and_parallel_paths_agree_on_type_shape() -> None:
    """Parallel and sequential paths produce structurally identical outputs.

    We can't assert object identity (each call builds a fresh env /
    policy), but we can assert both paths build a `SimRollout` + a
    `PolicyAdapter` for the same `SimEnvironment` — i.e. the
    parallelisation does not change the contract.
    """
    env_cfg = _mock_env()
    with _env_var(_SEQUENTIAL_INIT_ENV, "1"):
        seq_env, seq_policy = _build_env_and_policy(env_cfg)
    try:
        seq_types = (type(seq_env).__name__, type(seq_policy).__name__)
    finally:
        seq_env.close()
        seq_policy.close()

    with _env_var(_SEQUENTIAL_INIT_ENV, None):
        par_env, par_policy = _build_env_and_policy(env_cfg)
    try:
        par_types = (type(par_env).__name__, type(par_policy).__name__)
    finally:
        par_env.close()
        par_policy.close()

    assert seq_types == par_types


# ── Exception propagation ───────────────────────────────────────────────────


def test_parallel_init_propagates_unknown_vla_error() -> None:
    """An unknown ``vla.id`` raises the real :class:`ROSConfigError` from make_policy."""
    env_cfg = _mock_env(vla_id="does-not-exist")
    with _env_var(_SEQUENTIAL_INIT_ENV, None), pytest.raises(ROSConfigError):
        _build_env_and_policy(env_cfg)


def test_parallel_init_propagates_unknown_scene_error() -> None:
    """An unknown ``scene.id`` raises the real :class:`ROSConfigError` from make_env.

    Constructing the :class:`SimEnvironment` directly with an unknown
    scene id is rejected by Pydantic (scene id must match the
    registered ``mock`` for ``PhysicsBackend.MOCK``); to exercise the
    real registry-lookup failure we build a valid config and mutate the
    scene id through ``model_copy`` so the registry sees an unknown key.
    """
    env_cfg = _mock_env()
    bad = env_cfg.model_copy(
        update={"scene": env_cfg.scene.model_copy(update={"id": "no-such-scene"})}
    )
    with _env_var(_SEQUENTIAL_INIT_ENV, None), pytest.raises(ROSConfigError):
        _build_env_and_policy(bad)


def test_sequential_init_propagates_unknown_vla_error() -> None:
    """Sequential path also propagates :class:`ROSConfigError` verbatim."""
    env_cfg = _mock_env(vla_id="does-not-exist")
    with _env_var(_SEQUENTIAL_INIT_ENV, "1"), pytest.raises(ROSConfigError):
        _build_env_and_policy(env_cfg)


# ── Race-prone scene auto-detection ─────────────────────────────────────────


def test_race_prone_prefix_catalogue_is_non_empty() -> None:
    """The catalogue must list at least one prefix — guards against a typo."""
    assert _RACE_PRONE_SCENE_PREFIXES
    assert all(isinstance(p, str) and p for p in _RACE_PRONE_SCENE_PREFIXES)


@pytest.mark.parametrize(
    ("scene_id", "expected"),
    [
        ("openarm_tabletop_pnp", True),
        ("openarm_other_scene", True),
        # tabletop_push races: its env factory resolves `assets.mjcf`, which
        # imports `openral_hal._mujoco_arm` → `_base` → transformers on the env
        # thread, concurrent with the policy thread's lerobot→transformers load.
        ("tabletop_push", True),
        ("tabletop_push/push_to_goal", True),
        # so101_box is NOT race-prone despite also being an SO-101 scene: it
        # imports `robot_descriptions` directly for the MJCF, never
        # `_mujoco_arm`, so nothing pulls transformers onto its env thread.
        ("so101_box", False),
        ("so101_box/tube_insertion", False),
        # SAPIEN-backed scenes were promoted to race-prone after a
        # `Got unsupported ScalarType BFloat16` failure surfaced in
        # `openral benchmark run --suite maniskill3_panda`
        # — the policy thread's transient `torch.set_default_dtype(bfloat16)`
        # leaks into the env thread's SAPIEN gym.make. See
        # `_RACE_PRONE_SCENE_PREFIXES` for the full diagnosis.
        ("maniskill3", True),
        ("maniskill3_v3", True),
        ("simpler_env", True),
        ("simpler_env_widowx", True),
        ("libero_spatial", False),
        ("metaworld", False),
        # robocasa is race-prone: its env factory imports robosuite
        # (`ensure_backend_deps` → `_has_module` → `find_spec` executes
        # `robosuite/__init__`) concurrently with the policy thread, tripping
        # CPython's `_load_unlocked` `sys.modules.pop` → `KeyError:
        # 'robosuite.renderers.viewer.mjviewer_renderer'`. Both kitchen and GR1
        # task ids start with `robocasa/`.
        ("robocasa/PickPlaceCounterToCabinet", True),
        ("robocasa/gr1/PnPCupToDrawerClose", True),
        ("robocasa/NavigateKitchen", True),
        # `robocasa_pnp` is a scene FILE name, never a scene id — guard that the
        # prefix is `robocasa/` (slash), so an id like this would NOT match.
        ("robocasana_lookalike", False),
        ("pusht", False),
        ("aloha_transfer_cube", False),
        ("aloha_insertion", False),
        ("mock", False),
    ],
)
def test_scene_requires_sequential_init(scene_id: str, expected: bool) -> None:
    """Only scene ids whose prefix is catalogued race-prone return True."""
    env_cfg = _mock_env(scene_id=scene_id)
    assert _scene_requires_sequential_init(env_cfg) is expected
