"""Sim regression: the RoboCasa GR1 tabletop env must *build* (issue #44).

The GR1 fork (``robocasa-gr1-tabletop-tasks`` 0.2.0, NVIDIA's GR00T-N1
release) and the kitchen fork share an editable clone of
``ARISE-Initiative/robosuite`` *master*. The GR1 fork only supports
robosuite 1.5.0/1.5.1; when the install plan rode floating master a
drifting master commit that refactored the robot base-class API broke
the GR1 env build with ``ValueError: Invalid base type to add to robot!``
at ``robosuite/models/robots/robot_model.py:add_base`` — while the
kitchen fork kept working and master still reported version ``"1.5.2"``.
``_deps._ROBOSUITE_PIN`` now pins both forks to one verified commit.

This test catches a future re-break: it composes the GR1 scene exactly as
``openral sim run`` does and builds the env (``_build_robocasa_sim`` →
``gym.make("gr1_unified/...")`` → ``reset()`` → ``_load_model`` →
``add_base``). No VLA / sidecar / CUDA — the issue's failure is at env
construction, before the policy matters. The full VLA-driven episode is
exercised manually on the GPU host via ``openral sim run --config
scenes/sim/robocasa_gr1_pnp_cup_to_drawer.yaml --rskill
rskills/rldx1-ft-gr1-nf4 --view``.

Skips unless the GR1 fork is the *active* robocasa install: the kitchen
and GR1 forks share the ``robocasa`` package name and only one can be
installed at a time, so this test never swaps the venv — it runs only
where ``openral sim run`` (or a prior GR1 scene) already provisioned the
fork.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tests.sim.conftest import mujoco_renderer_probe_error

# Defer all skip decisions to `pytestmark` (not module-level
# `pytest.skip`/`importorskip`): with `tests/sim/__init__.py` making this a
# Package, a Skipped raised during collection of THIS file would mark the
# whole `tests/sim` Package skipped and drop every sibling. See the same
# note in `test_panda_mobile_pi05_robocasa.py`.
_ROBOSUITE_MISSING = importlib.util.find_spec("robosuite") is None


def _gr1_fork_inactive() -> str:
    """Reason string if the GR1 robocasa fork is not the active install, else ""."""
    if _ROBOSUITE_MISSING:
        return "robosuite not installed"
    from openral_sim._deps import _has_robocasa_gr1

    if not _has_robocasa_gr1():
        return (
            "robocasa GR1 fork not the active install — run a GR1 scene first "
            "(`openral sim run --config scenes/sim/robocasa_gr1_pnp_cup_to_drawer.yaml ...`)"
        )
    return ""


_GR1_INACTIVE = _gr1_fork_inactive()
_RENDERER_ERROR = mujoco_renderer_probe_error() if not _GR1_INACTIVE else "gr1 fork inactive"

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(bool(_GR1_INACTIVE), reason=_GR1_INACTIVE or "gr1 fork inactive"),
    pytest.mark.skipif(
        bool(_RENDERER_ERROR), reason=_RENDERER_ERROR or "no MuJoCo offscreen renderer"
    ),
]


_REPO_ROOT = Path(__file__).parent.parent.parent
_CONFIG = _REPO_ROOT / "scenes" / "sim" / "robocasa_gr1_pnp_cup_to_drawer.yaml"
_RSKILL = "rskills/rldx1-ft-gr1-nf4"


@pytest.fixture(scope="module")
def env_cfg():
    """Compose the GR1 SimEnvironment the way the CLI does (rSkill → VLASpec only)."""
    from tests.sim.conftest import compose_sim_env

    if not _CONFIG.exists():
        pytest.skip(f"sim config not found at {_CONFIG}")
    if not (_REPO_ROOT / _RSKILL / "rskill.yaml").exists():
        pytest.skip(f"rSkill manifest not found at {_RSKILL}")
    # The rSkill only provides the VLASpec; the env build never loads weights.
    return compose_sim_env(_CONFIG, rskill_uri=_RSKILL, n_episodes=1, max_steps=2)


class TestGr1RoboCasaEnvBuild:
    def test_scene_is_the_gr1_tabletop(self, env_cfg) -> None:
        """Sanity: the composed scene is the GR1 PnP tabletop on a GR1 robot."""
        assert env_cfg.scene.id == "robocasa/gr1/PnPCupToDrawerClose"
        assert env_cfg.robot_id == "gr1"

    def test_env_builds_and_resets_without_invalid_base_type(self, env_cfg) -> None:
        """Build the GR1 env and reset it — the issue #44 crash point.

        A robosuite drift regression surfaces here as
        ``ValueError: Invalid base type to add to robot!`` raised from
        ``add_base`` during ``_load_model``. A clean reset returning a
        29-D GR1 observation proves the pinned robosuite still composes
        the GR1 robot.
        """
        from openral_sim.backends.robocasa import _build_robocasa_sim

        sim = _build_robocasa_sim(env_cfg, scene_id=env_cfg.scene.id)
        try:
            obs = sim.reset(seed=0)
            assert isinstance(obs, dict)
            state = obs.get("state")
            assert state is not None, obs.keys()
            # GR1 ArmsAndWaist proprioception is the 29-D layout the
            # scene declares (state_layout: gr1) — [waist(3) | right_arm(7)
            # | left_arm(7) | right_hand(6) | left_hand(6)].
            assert np.asarray(state).shape[-1] == 29, np.asarray(state).shape
        finally:
            sim.close()
