"""Tests for ``openral_core.load_scene_strict``.

The helper enforces strict per-tier YAML acceptance for the three scene-driven
CLIs (``openral deploy sim``, ``openral sim run``, ``openral benchmark scene``)
so a YAML one tier off raises ``ROSConfigError`` with a redirect message
pointing at the right command.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml as _yaml
from openral_core import BenchmarkScene, DeployScene, SimScene
from openral_core.exceptions import ROSConfigError


def _write(tmp_path: Path, name: str, data: dict[str, object]) -> Path:
    p = tmp_path / name
    p.write_text(_yaml.safe_dump(data), encoding="utf-8")
    return p


# Reusable fixture payloads.

_DEPLOY_YAML: dict[str, object] = {
    "scene": {"id": "libero_spatial", "backend": "mujoco"},
}

_SIM_YAML: dict[str, object] = {
    "scene": {"id": "libero_spatial", "backend": "mujoco"},
    "task": {"id": "libero_spatial/0", "scene_id": "libero_spatial"},
}

_SIM_YAML_WITH_BUDGET: dict[str, object] = {
    "scene": {"id": "libero_spatial", "backend": "mujoco"},
    "task": {
        "id": "libero_spatial/0",
        "scene_id": "libero_spatial",
        "success_key": "is_success",
        "max_steps": 100,
    },
}

_BENCHMARK_YAML: dict[str, object] = {
    "scene": {"id": "libero_spatial", "backend": "mujoco"},
    "task": {
        "id": "libero_spatial/0",
        "scene_id": "libero_spatial",
        "success_key": "is_success",
        "max_steps": 100,
    },
    "n_episodes": 500,
    "seed": 42,
    "metadata": {
        "paper": "https://arxiv.org/abs/2309.11500",
        "honest_scope": "Task 0 of LIBERO-Spatial.",
    },
}


# ── DeployScene strict ─────────────────────────────────────────────────────


def test_deploy_strict_accepts_deploy_yaml(tmp_path: Path) -> None:
    from openral_core import load_scene_strict

    p = _write(tmp_path, "deploy.yaml", _DEPLOY_YAML)
    out = load_scene_strict(str(p), DeployScene)
    assert isinstance(out, DeployScene)
    assert out.scene.id == "libero_spatial"


def test_deploy_strict_rejects_sim_yaml_with_redirect(tmp_path: Path) -> None:
    from openral_core import load_scene_strict

    p = _write(tmp_path, "sim.yaml", _SIM_YAML)
    with pytest.raises(ROSConfigError) as exc:
        load_scene_strict(str(p), DeployScene)
    msg = str(exc.value)
    assert "task" in msg
    assert "openral sim run" in msg


def test_deploy_strict_rejects_benchmark_yaml(tmp_path: Path) -> None:
    from openral_core import load_scene_strict

    p = _write(tmp_path, "bench.yaml", _BENCHMARK_YAML)
    with pytest.raises(ROSConfigError):
        load_scene_strict(str(p), DeployScene)


# ── SimScene strict ────────────────────────────────────────────────────────


def test_sim_strict_accepts_sim_yaml(tmp_path: Path) -> None:
    from openral_core import load_scene_strict

    p = _write(tmp_path, "sim.yaml", _SIM_YAML)
    out = load_scene_strict(str(p), SimScene)
    assert isinstance(out, SimScene)
    assert out.task.id == "libero_spatial/0"
    assert out.task.max_steps is None  # SimScene allows optional


def test_sim_strict_accepts_sim_yaml_with_budget(tmp_path: Path) -> None:
    """SimScene accepts max_steps/success_key if the user wants them."""
    from openral_core import load_scene_strict

    p = _write(tmp_path, "sim.yaml", _SIM_YAML_WITH_BUDGET)
    out = load_scene_strict(str(p), SimScene)
    assert out.task.max_steps == 100


def test_sim_strict_rejects_deploy_yaml_with_redirect(tmp_path: Path) -> None:
    from openral_core import load_scene_strict

    p = _write(tmp_path, "deploy.yaml", _DEPLOY_YAML)
    with pytest.raises(ROSConfigError) as exc:
        load_scene_strict(str(p), SimScene)
    msg = str(exc.value)
    assert "task" in msg
    assert "openral deploy sim" in msg


def test_sim_strict_rejects_benchmark_yaml_with_redirect(tmp_path: Path) -> None:
    """A YAML that fully validates as BenchmarkScene must be rejected
    (otherwise SimScene would silently widen and lose the eval contract)."""
    from openral_core import load_scene_strict

    p = _write(tmp_path, "bench.yaml", _BENCHMARK_YAML)
    with pytest.raises(ROSConfigError) as exc:
        load_scene_strict(str(p), SimScene)
    msg = str(exc.value)
    assert "BenchmarkScene" in msg
    assert "openral benchmark scene" in msg


# ── BenchmarkScene strict ──────────────────────────────────────────────────


def test_benchmark_strict_accepts_benchmark_yaml(tmp_path: Path) -> None:
    from openral_core import load_scene_strict

    p = _write(tmp_path, "bench.yaml", _BENCHMARK_YAML)
    out = load_scene_strict(str(p), BenchmarkScene)
    assert isinstance(out, BenchmarkScene)
    assert out.n_episodes == 500
    assert out.metadata.paper == "https://arxiv.org/abs/2309.11500"


def test_benchmark_strict_rejects_sim_yaml(tmp_path: Path) -> None:
    from openral_core import load_scene_strict

    p = _write(tmp_path, "sim.yaml", _SIM_YAML)
    with pytest.raises(ROSConfigError):
        load_scene_strict(str(p), BenchmarkScene)


def test_benchmark_strict_rejects_deploy_yaml(tmp_path: Path) -> None:
    from openral_core import load_scene_strict

    p = _write(tmp_path, "deploy.yaml", _DEPLOY_YAML)
    with pytest.raises(ROSConfigError):
        load_scene_strict(str(p), BenchmarkScene)


# ── File-level errors ──────────────────────────────────────────────────────


def test_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    from openral_core import load_scene_strict

    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ROSConfigError, match="must be a mapping"):
        load_scene_strict(str(p), SimScene)


def test_rejects_missing_file(tmp_path: Path) -> None:
    from openral_core import load_scene_strict

    p = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_scene_strict(str(p), SimScene)


# ── Real fixtures from scenes/ ─────────────────────────────────────────────


def test_real_benchmark_yaml_loads_strict() -> None:
    """Smoke-test against a real BenchmarkScene fixture."""
    from openral_core import load_scene_strict

    p = Path("scenes/benchmark/libero_spatial.yaml")
    if not p.exists():
        pytest.skip(f"{p} not present")
    out = load_scene_strict(str(p), BenchmarkScene)
    assert isinstance(out, BenchmarkScene)


def test_real_sim_yaml_loads_strict() -> None:
    """Smoke-test against a real SimScene fixture."""
    from openral_core import load_scene_strict

    p = Path("scenes/sim/tabletop_cube_push.yaml")
    if not p.exists():
        pytest.skip(f"{p} not present")
    out = load_scene_strict(str(p), SimScene)
    assert isinstance(out, SimScene)


def test_real_deploy_yaml_loads_strict() -> None:
    """Smoke-test against a real DeployScene fixture."""
    from openral_core import load_scene_strict

    p = Path("scenes/deploy/libero_pnp.yaml")
    if not p.exists():
        pytest.skip(f"{p} not present")
    out = load_scene_strict(str(p), DeployScene)
    assert isinstance(out, DeployScene)
