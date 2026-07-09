r"""openral sim runner — swappable (robot x scene x task x VLA) for rSkill validation.

Typical usage::

    from openral_sim import SimRunner

    # ``SimRunner`` consumes the runtime-composed ``SimEnvironment``
    # (scene + task + VLA). The on-disk YAML is a ``SimScene`` (scene +
    # task, no VLA); the CLI composes the ``SimEnvironment`` from a
    # ``SimScene`` + an rSkill manifest. See ``openral sim run`` and
    # :func:`openral_sim.cli._load_or_build_env` for the canonical
    # compose path.
    env_cfg: SimEnvironment = ...  # composed by the CLI
    runner = SimRunner(env_cfg)
    runner.activate()
    runner.run(max_ticks=env_cfg.n_episodes * ((env_cfg.task.max_steps or 1000) + 1))
    for episode in runner.episode_results:
        print(episode.success, episode.steps, episode.mean_step_latency_ms)
    runner.deactivate()

CLI::

    openral sim run --config scenes/benchmark/libero_spatial.yaml \
            --rskill smolvla-libero
    openral sim run --robot franka_panda --scene libero_spatial \
            --task libero_spatial/0 \
            --rskill smolvla-libero
    openral benchmark run --suite libero_spatial \
            --rskill smolvla-libero

The sim package itself depends only on ``openral-core``,
``openral-runner``, and ``openral-rskill``. Physics backends
(LIBERO, MetaWorld, ...) are imported lazily by the registered adapters
so installing this package never pulls heavyweight ML deps.
"""

from __future__ import annotations

# Trigger built-in policy + backend registration (LIBERO, MetaWorld, mock,
# smolvla, …). Importing the two subpackages runs their `_register_*`
# side effects.
from openral_sim._video import save_episode_mp4
from openral_sim.backends import _register_backends as _register_backends
from openral_sim.benchmark import default_output_path, run_benchmark
from openral_sim.factory import make_env, make_policy
from openral_sim.policies import _register_policies as _register_policies
from openral_sim.policy import PolicyAdapter
from openral_sim.registry import POLICIES, ROBOTS, SCENES
from openral_sim.rollout import EpisodeResult, SimRollout
from openral_sim.sim_runner import SimRunner

__all__ = [
    "POLICIES",
    "ROBOTS",
    "SCENES",
    "EpisodeResult",
    "PolicyAdapter",
    "SimRollout",
    "SimRunner",
    "default_output_path",
    "make_env",
    "make_policy",
    "run_benchmark",
    "save_episode_mp4",
]

__version__ = "0.1.0"
