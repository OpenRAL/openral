"""Every YAML under ``scenes/`` must load as its tier's typed schema.

This is the cheap, per-PR guard that catches stale example configs
(missing ``vla:`` block removal, schema drift, etc.) without requiring
the GPU rollout that ``tools/audit_sim_configs.py`` performs. CLAUDE.md
§1.11 — real schemas, no mocks; we load the real YAMLs that ship in the
tree.

Per ADR-0041, scenes live in three tiers — ``scenes/deploy/`` loads as
:class:`DeployScene`, ``scenes/sim/`` as :class:`SimScene`,
``scenes/benchmark/`` as :class:`BenchmarkScene`. A failure here means
the YAML carries a key the schema no longer recognises, or a required
key was removed, or a legacy ``vla:`` block was reintroduced.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import BenchmarkScene, DeployScene, RoboCasaBackendOptions, SimScene

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENES_DIR = REPO_ROOT / "scenes"

_TIER_LOADERS = {"sim": SimScene, "benchmark": BenchmarkScene, "deploy": DeployScene}


def _yamls(subdir: str) -> list[Path]:
    tier_dir = SCENES_DIR / subdir
    if not tier_dir.exists():
        return []
    return sorted(tier_dir.rglob("*.yaml"))


@pytest.mark.parametrize(
    "yaml_path",
    _yamls("deploy"),
    ids=lambda p: p.relative_to(REPO_ROOT).as_posix(),
)
def test_deploy_yaml_loads_as_deploy_scene(yaml_path: Path) -> None:
    """Each ``scenes/deploy/`` YAML validates as :class:`DeployScene`.

    Deploy scenes are env-only (no task, no eval config) — the reasoner
    picks the rSkill at runtime.
    """
    DeployScene.from_yaml(str(yaml_path))


@pytest.mark.parametrize(
    "yaml_path",
    _yamls("sim"),
    ids=lambda p: p.relative_to(REPO_ROOT).as_posix(),
)
def test_sim_yaml_loads_as_sim_scene(yaml_path: Path) -> None:
    """Each ``scenes/sim/`` YAML validates as :class:`SimScene`.

    Sim scenes carry an optional task for ``openral sim run`` smoke
    tests and tutorials; eval-specific fields are optional.
    """
    SimScene.from_yaml(str(yaml_path))


@pytest.mark.parametrize(
    "yaml_path",
    _yamls("benchmark"),
    ids=lambda p: p.relative_to(REPO_ROOT).as_posix(),
)
def test_benchmark_yaml_loads_as_benchmark_scene(yaml_path: Path) -> None:
    """Each ``scenes/benchmark/`` YAML validates as :class:`BenchmarkScene`.

    Benchmark scenes require ``n_episodes``, ``seed``, structured
    :class:`BenchmarkMetadata` (paper + honest_scope), and a task with
    both ``max_steps`` and ``success_key`` so a leaderboard number can
    be reproduced from the YAML alone.
    """
    BenchmarkScene.from_yaml(str(yaml_path))


def _tier_of(p: Path) -> str:
    return p.relative_to(SCENES_DIR).parts[0]


def _robocasa_scene_yamls() -> list[Path]:
    """Every shipped scene whose ``scene.id`` routes to the RoboCasa backend."""
    out: list[Path] = []
    for p in _yamls("sim") + _yamls("benchmark") + _yamls("deploy"):
        scene = _TIER_LOADERS[_tier_of(p)].from_yaml(str(p))
        if str(scene.scene.id).startswith("robocasa"):
            out.append(p)
    return out


@pytest.mark.parametrize(
    "yaml_path",
    _robocasa_scene_yamls(),
    ids=lambda p: p.relative_to(REPO_ROOT).as_posix(),
)
def test_robocasa_scene_backend_options_validate(yaml_path: Path) -> None:
    """Each RoboCasa scene's ``backend_options`` must pass the adapter's validator.

    ``_build_robocasa_sim`` calls
    ``RoboCasaBackendOptions.model_validate(scene.backend_options)`` at
    scene-factory time, so a YAML that omits a required key (e.g. a
    ``mode='prebuilt'`` scene with no ``prebuilt_task``) loads as a
    :class:`SimScene` but blows up the instant ``openral sim run`` /
    ``deploy sim`` builds the env. This guard catches that at unit speed.
    """
    scene = _TIER_LOADERS[_tier_of(yaml_path)].from_yaml(str(yaml_path))
    RoboCasaBackendOptions.model_validate(scene.scene.backend_options or {})


@pytest.mark.parametrize(
    "yaml_path",
    _yamls("sim"),
    ids=lambda p: p.relative_to(REPO_ROOT).as_posix(),
)
def test_sim_scene_omits_robot_id_when_robot_is_fixed(yaml_path: Path) -> None:
    """A fixed-robot sim scene must NOT carry ``robot_id:``.

    ``openral sim run`` raises ``ROSConfigError`` ("hard-fixes") when a
    scene whose backend hard-wires the physics robot (LIBERO, MetaWorld,
    PushT, ALOHA, RoboCasa) also carries a ``robot_id:`` — so such a YAML
    is unrunnable via ``sim run`` even though it loads as a SimScene.
    ``robot_id:`` is reserved for free-axis scenes (``SCENES.fixed_robot``
    is ``None``). See tests/unit/test_sim_run_fixed_robot_guard.py.
    """
    from openral_sim.registry import SCENES

    scene = SimScene.from_yaml(str(yaml_path))
    if SCENES.fixed_robot(scene.scene.id) is not None:
        assert scene.robot_id is None, (
            f"{yaml_path.name}: scene {scene.scene.id!r} hard-fixes its robot; "
            f"drop `robot_id: {scene.robot_id}` (sim run rejects it)."
        )
