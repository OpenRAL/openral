"""Sim test: VLABench (MuJoCo + dm_control) + SmolVLA via the native backend.

Two tiers:

* **Unconditional wiring** — validates the ``smolvla-vlabench`` rSkill manifest, the
  ``vlabench_select_fruit`` benchmark scene, and that the scene/policy factories are
  registered. These need no simulator and run on any host (incl. CI).
* **Live end-to-end** — builds the native VLABench env + the SmolVLA policy through
  ``openral_sim.make_env`` / ``make_policy`` and runs a few real steps of
  ``vlabench/select_fruit``. Asserts the 7-D state / 7-D action / 3-camera contract and
  that the episode reports the ``is_success`` key. It does NOT assert success:
  ``smolvla_vlabench`` is a 0% integration baseline — the gate is that the
  wiring drives a real closed loop.

Skip policy
-----------
VLABench has no PyPI release and rides a ~12 GB CC-BY asset bundle; both are externally
provisioned (the ``vlabench`` install plan handles the Python side, the assets are a manual
Google-Drive fetch — CLAUDE.md §1.9). The live test skips unless VLABench is
importable, the asset bundle is unpacked, and CUDA is present. CI runners without the
provisioned backend skip — the legitimate skip path (§1.12).
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


def test_manifest_validates_and_declares_smolvla_vlabench() -> None:
    from openral_core.schemas import RSkillManifest

    path = _REPO_ROOT / "rskills" / "smolvla-vlabench" / "rskill.yaml"
    m = RSkillManifest.model_validate(yaml.safe_load(path.read_text()))
    assert m.model_family == "smolvla"
    assert m.license.value == "apache-2.0"
    assert m.evaluated_tasks == ["vlabench"]
    # Native VLABench contract: 3 RGB cameras, 7-D state, 7-D action.
    assert {s.vla_feature_key for s in m.sensors_required} == {
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    }
    assert m.state_contract.dim == 7
    assert m.action_contract.dim == 7


def test_benchmark_scene_validates() -> None:
    from openral_core.loaders import load_scene_strict
    from openral_core.schemas import BenchmarkScene, PhysicsBackend

    bs = load_scene_strict(
        str(_REPO_ROOT / "scenes" / "benchmark" / "vlabench_select_fruit.yaml"), BenchmarkScene
    )
    assert bs.scene.backend == PhysicsBackend.MUJOCO
    assert bs.scene.id == "vlabench"
    assert bs.task.id == "vlabench/select_fruit"
    assert bs.task.success_key == "is_success"
    assert bs.task.max_steps == 500
    assert bs.robot_id == "franka_panda"


def test_factories_registered() -> None:
    import openral_sim.backends
    import openral_sim.policies  # noqa: F401 — registers the policy factory
    from openral_sim.registry import POLICIES, SCENES

    assert SCENES.fixed_robot("vlabench") == "franka_panda"
    # raises KeyError if unregistered
    POLICIES.get("smolvla")


def test_install_plan_registered() -> None:
    """The VLABench ensure_backend_deps plan resolves and probes without a simulator."""
    from openral_sim._deps import get_plan

    plan = get_plan("vlabench")
    assert plan.backend_id == "vlabench"
    # Probe is a pure import check — safe to call on any host (returns False when absent).
    assert plan.probe() in (True, False)
    # The plan installs editable + stubs rrt_algorithms (no external repo pull for it).
    assert any("rrt_algorithms stub" in s.description for s in plan.steps)


# ── Live end-to-end (needs the provisioned VLABench backend + assets) ─────────


def _vlabench_ready() -> bool:
    """VLABench is importable AND its asset bundle is unpacked."""
    if importlib.util.find_spec("VLABench") is None:
        return False
    root = os.environ.get("VLABENCH_ROOT") or os.environ.get("OPENRAL_VLABENCH_ROOT")
    if not root:
        import VLABench  # type: ignore[import-not-found,import-untyped,unused-ignore]

        assert VLABench.__file__ is not None
        root = os.path.dirname(VLABench.__file__)
    obj_dir = Path(root) / "assets" / "obj"
    return obj_dir.is_dir() and any(obj_dir.iterdir())


_CUDA_AVAILABLE = False
if importlib.util.find_spec("torch") is not None:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.skipif(
    not (_vlabench_ready() and _CUDA_AVAILABLE),
    reason="VLABench backend + assets not provisioned, or no CUDA",
)
def test_select_fruit_live_episode() -> None:
    import openral_sim.backends
    import openral_sim.policies  # noqa: F401
    from openral_core.loaders import load_scene_strict
    from openral_core.schemas import BenchmarkScene, SimEnvironment, VLASpec
    from openral_sim.factory import make_env, make_policy

    os.environ.setdefault("MUJOCO_GL", "egl")
    bs = load_scene_strict(
        str(_REPO_ROOT / "scenes" / "benchmark" / "vlabench_select_fruit.yaml"), BenchmarkScene
    )
    env_cfg = SimEnvironment(
        robot_id=bs.robot_id,
        scene=bs.scene,
        task=bs.task,
        vla=VLASpec(id="smolvla", weights_uri="rskills/smolvla-vlabench"),
        n_episodes=1,
    )
    env = make_env(env_cfg)
    policy = make_policy(env_cfg)
    try:
        obs = env.reset(seed=0)
        assert set(obs["images"]) == {"camera1", "camera2", "camera3"}
        assert obs["state"].shape == (7,)
        policy.reset()
        result = None
        # A few steps is enough for the contract; smolvla_vlabench scores 0% so we
        # assert the closed loop runs + reports the success key, not that it wins.
        for _ in range(3):
            action = policy.step(obs, bs.task.instruction)
            assert action.shape == (7,)
            result = env.step(action)
            obs = result.observation
            if result.terminated:
                break
        assert result is not None
        assert "is_success" in result.info
    finally:
        env.close()
        policy.close()
