"""SimAttachedHAL probes the OpenArm v2 tabletop env's true action width.

Split out of ``test_sim_attached_action_dim.py``, which is a **LIBERO** file:
LIBERO pins ``robosuite==1.4`` while the ``openarm_tabletop_pnp`` scene needs
``robosuite>=1.5`` (the ``robocasa`` dependency group — see
``openral_sim._deps._openarm_robosuite_plan``). The two pins are mutually
exclusive in one venv (CLAUDE.md — "LIBERO↔RoboCasa groups are mutually
exclusive"), and ``tools/test_selection.toml`` assigns a dependency lane
**per file**, so a file holding both halves is always wrong for one of them.

Concretely: run inside the ``libero`` lane, the LIBERO tests above this one
imported robosuite 1.4, then this test's ``ensure_backend_deps`` probe found
the wrong robosuite and ``_assert_no_live_dependency_swap`` correctly refused
to swap the package under live objects — so the test failed instead of
running, on every PR whose diff selected ``python/hal/tests/``. Keeping it in
its own file lets ``requirement_globs.robocasa`` give it the robosuite>=1.5
env it actually needs.
"""

from __future__ import annotations

import pytest
from _renderer_probe import requires_renderer


@requires_renderer
def test_openarm_tabletop_action_dim_matches_state_dim() -> None:
    """Native openarm_tabletop_pnp reports its bimanual state_dim; the HAL probe resolves it.

    Built through the deploy-sim ``build_sim_env_from_yaml`` loader (robosuite
    MJCF wrapper). The ``openarm_tabletop_pnp`` scene mandates a ``base_pose``
    at compose time; the loader now propagates the
    SimScene YAML's ``base_pose`` into the composed ``SimEnvironment``,
    so the scene builds through the loader exactly as it does through the direct
    factory path.
    """
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/openarm_tabletop.yaml", robot_id_fallback="openarm"
    )
    # state_dim is derived from the manifest joint count (bimanual OpenArm v2).
    assert env.action_dim == env._state_dim
    assert env.action_dim > 0

    desc = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed)
    hal.connect()
    assert hal._env_action_dim == env.action_dim
