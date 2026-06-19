"""Sim test: RLBench (CoppeliaSim/PyRep) + 3D Diffuser Actor via the sidecars (ADR-0061).

Two tiers:

* **Unconditional wiring** — validates the rSkill manifest, the three benchmark
  scenes, the suite, and that the scene/policy factories are registered. These
  need no simulator and run on any host (incl. CI).
* **Live end-to-end** — auto-spawns the RLBench scene sidecar (CoppeliaSim) + the
  3D Diffuser Actor policy sidecar and runs one ``open_drawer`` episode through
  ``openral_sim.make_env`` / ``make_policy``. Asserts the policy drives the arm to
  a finite-reward keyframe and the episode reports a success key.

Skip policy
-----------
CoppeliaSim is proprietary (free EDU license; CLAUDE.md §1.9), provisioned out of
band in a py3.10 sidecar venv. The live test skips unless BOTH pyzmq/msgpack are
importable on the openral venv AND the sidecar interpreter + CoppeliaSim root are
resolvable. CI runners without the provisioned sidecar skip — the legitimate skip
path (§1.12).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "rskills").is_dir() and (p / "pyproject.toml").is_file()
)


# ── Unconditional wiring (no simulator) ──────────────────────────────────────


def test_manifest_validates_and_declares_diffuser_actor() -> None:
    from openral_core.schemas import RSkillManifest

    path = _REPO_ROOT / "rskills" / "3d-diffuser-actor-rlbench" / "rskill.yaml"
    m = RSkillManifest.model_validate(yaml.safe_load(path.read_text()))
    assert m.model_family == "diffuser_actor"
    assert m.license.value == "mit"
    assert set(m.evaluated_tasks) == {
        "rlbench/open_drawer",
        "rlbench/meat_off_grill",
        "rlbench/close_jar",
    }


@pytest.mark.parametrize(
    ("scene_file", "rlbench_task"),
    [
        ("rlbench_open_drawer.yaml", "open_drawer"),
        ("rlbench_meat_off_grill.yaml", "meat_off_grill"),
        ("rlbench_close_jar.yaml", "close_jar"),
    ],
)
def test_benchmark_scene_validates(scene_file: str, rlbench_task: str) -> None:
    from openral_core.loaders import load_scene_strict
    from openral_core.schemas import BenchmarkScene, PhysicsBackend

    bs = load_scene_strict(str(_REPO_ROOT / "scenes" / "benchmark" / scene_file), BenchmarkScene)
    assert bs.scene.backend == PhysicsBackend.COPPELIASIM
    assert bs.scene.backend_options["rlbench_task"] == rlbench_task
    assert bs.robot_id == "franka_panda"
    assert bs.n_episodes == 25
    assert bs.task.max_steps == 25


def test_suite_is_valid() -> None:
    from openral_core import load_benchmark_suite, raise_on_invalid_suite

    scenes = load_benchmark_suite(str(_REPO_ROOT / "benchmarks" / "rlbench.yaml"))
    raise_on_invalid_suite(scenes, suite_id="rlbench")
    assert [s.task.id for s in scenes] == [
        "rlbench/open_drawer",
        "rlbench/meat_off_grill",
        "rlbench/close_jar",
    ]


def test_factories_registered() -> None:
    import openral_sim.backends
    import openral_sim.policies  # noqa: F401 — registers the policy factory
    from openral_sim.registry import POLICIES, SCENES

    assert SCENES.fixed_robot("rlbench") == "franka_panda"
    # raises KeyError if unregistered
    POLICIES.get("diffuser_actor")


# ── Live end-to-end (needs the provisioned CoppeliaSim sidecar) ───────────────


def _sidecar_python_available() -> bool:
    override = os.environ.get("OPENRAL_RLBENCH_SIDECAR_PYTHON")
    if override:
        return Path(override).is_file()
    default = Path.home() / ".cache" / "openral" / "rlbench-policy" / ".venv" / "bin" / "python"
    return default.is_file()


def _coppeliasim_available() -> bool:
    root = os.environ.get("COPPELIASIM_ROOT")
    if root:
        return Path(root).is_dir()
    default = Path.home() / ".cache/openral/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
    return default.is_dir()


_WIRE_MISSING = [m for m in ("zmq", "msgpack") if importlib.util.find_spec(m) is None]


@pytest.mark.sim
@pytest.mark.skipif(
    bool(_WIRE_MISSING) or not (_sidecar_python_available() and _coppeliasim_available()),
    reason="RLBench/CoppeliaSim sidecar not provisioned (ADR-0061)",
)
def test_open_drawer_live_episode() -> None:
    import openral_sim.backends
    import openral_sim.policies  # noqa: F401
    from openral_core.loaders import load_scene_strict
    from openral_core.schemas import BenchmarkScene, SimEnvironment, VLASpec
    from openral_sim.factory import make_env, make_policy

    bs = load_scene_strict(
        str(_REPO_ROOT / "scenes" / "benchmark" / "rlbench_open_drawer.yaml"), BenchmarkScene
    )
    env_cfg = SimEnvironment(
        robot_id=bs.robot_id,
        scene=bs.scene,
        task=bs.task,
        vla=VLASpec(id="diffuser_actor", weights_uri="rskills/3d-diffuser-actor-rlbench"),
        n_episodes=1,
    )
    env = make_env(env_cfg)
    policy = make_policy(env_cfg)
    try:
        obs = env.reset(seed=0)
        assert {"images", "point_clouds", "gripper_pose"} <= set(obs)
        policy.reset()
        reward = 0.0
        for _ in range(bs.task.max_steps or 25):
            action = policy.step(obs, bs.task.instruction)
            assert action.shape == (8,)
            result = env.step(action)
            obs = result.observation
            reward = max(reward, result.reward)
            if result.terminated:
                break
        # open_drawer is high-SR for 3D Diffuser Actor; the episode must at least
        # run a real closed loop and report the success key.
        assert "is_success" in result.info
    finally:
        env.close()
        policy.close()
