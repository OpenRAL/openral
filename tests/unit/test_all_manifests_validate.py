"""Parametrized schema-compliance sweep for every in-tree manifest.

Validates all ``robots/<id>/robot.yaml``, ``scenes/benchmark/*.yaml``,
``scenes/sim/*.yaml``, and ``scenes/deploy/*.yaml`` fixtures against their
canonical Pydantic schemas so a newly-added or modified manifest that
silently violates the contract is caught before it reaches CI.

Coverage
--------
- Every ``robots/<id>/robot.yaml``                 → :class:`RobotDescription`
- Every ``scenes/benchmark/*.yaml``                → :class:`BenchmarkScene`
- Every ``scenes/sim/*.yaml``                      → :class:`SimScene`
- Every ``scenes/deploy/*.yaml``                   → :class:`DeployScene`
- Per-robot: each declared :class:`SensorSpec` is also individually
  validated (modality-specific invariants — intrinsics for cameras,
  range/channels for lidar, etc.) so that a sensor added with the wrong
  field combination is caught independently of the full-robot round-trip.

Notes
-----
- Discovery is intentionally dynamic: new files are automatically picked up
  without any change to this file.
- Scene files must be placed in the correct subdirectory (``benchmark/``,
  ``sim/``, or ``deploy/``) — the directory name determines which schema
  tier is applied.
- The fixture search paths are relative to the repository root (CWD when
  ``pytest`` is invoked via ``just test`` / ``just lint``).  Tests are
  skipped when the search root is absent so ``pip install openral-cli``
  wheel installs do not break.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


# ─── Repository-root helpers ──────────────────────────────────────────────────

def _repo_root() -> Path | None:
    """Return the repository root by walking up from this file, or None."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "robots").is_dir() and (parent / "scenes").is_dir():
            return parent
    return None


_ROOT = _repo_root()


def _collect(rel: str) -> list[Path]:
    """Return all YAML files under *rel* (relative to repo root) or []."""
    if _ROOT is None:
        return []
    base = _ROOT / rel
    if not base.is_dir():
        return []
    return sorted(base.glob("*.yaml"))


def _collect_robots() -> list[Path]:
    """Return every ``robots/<id>/robot.yaml`` present in-tree."""
    if _ROOT is None:
        return []
    robots_dir = _ROOT / "robots"
    if not robots_dir.is_dir():
        return []
    return sorted(
        d / "robot.yaml"
        for d in robots_dir.iterdir()
        if d.is_dir() and (d / "robot.yaml").is_file()
    )


# ─── Robot manifest validation ────────────────────────────────────────────────

_ROBOT_YAMLS = _collect_robots()


@pytest.mark.parametrize(
    "robot_yaml",
    _ROBOT_YAMLS,
    ids=[p.parent.name for p in _ROBOT_YAMLS],
)
def test_robot_manifest_validates(robot_yaml: Path) -> None:
    """Every ``robots/<id>/robot.yaml`` round-trips through ``RobotDescription``."""
    if not _ROBOT_YAMLS:
        pytest.skip("No robots/ directory found (wheel install?)")
    from openral_core.schemas import RobotDescription

    desc = RobotDescription.from_yaml(str(robot_yaml))
    assert desc.name, f"{robot_yaml}: RobotDescription.name must be non-empty"
    assert desc.joints, f"{robot_yaml}: RobotDescription must declare at least one joint"


# ─── Per-robot sensor spec validation ────────────────────────────────────────


def _sensor_params() -> list[tuple[Path, Any]]:
    """Build (robot_yaml, raw_sensor_dict) pairs for every declared sensor."""
    if _ROOT is None:
        return []
    params = []
    for robot_yaml in _collect_robots():
        with open(robot_yaml, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        for sensor in data.get("sensors", []) or []:
            params.append((robot_yaml, sensor))
    return params


_SENSOR_PARAMS = _sensor_params()


@pytest.mark.parametrize(
    "robot_yaml,sensor_raw",
    _SENSOR_PARAMS,
    ids=[
        f"{p.parent.name}/{s.get('name', '?')}"
        for p, s in _SENSOR_PARAMS
    ],
)
def test_sensor_spec_validates(robot_yaml: Path, sensor_raw: dict[str, Any]) -> None:
    """Each sensor entry inside a robot manifest validates as :class:`SensorSpec`.

    Also checks modality-specific invariants:
    - RGB / depth / stereo: ``intrinsics`` block present.
    - LiDAR: ``n_channels``, ``range_min_m``, ``range_max_m`` all present.
    """
    if not _SENSOR_PARAMS:
        pytest.skip("No robots/ directory found (wheel install?)")
    from openral_core.schemas import SensorModality, SensorSpec

    spec = SensorSpec.model_validate(sensor_raw)

    if spec.modality in (
        SensorModality.RGB,
        SensorModality.DEPTH,
        SensorModality.STEREO,
    ):
        assert spec.intrinsics is not None, (
            f"{robot_yaml}: sensor '{spec.name}' has modality={spec.modality!r} "
            "but is missing 'intrinsics' (width, height, fx, fy, cx, cy)."
        )

    if spec.modality in (SensorModality.LIDAR_2D, SensorModality.POINT_CLOUD):
        missing = [
            f for f in ("n_channels", "range_min_m", "range_max_m")
            if getattr(spec, f) is None
        ]
        assert not missing, (
            f"{robot_yaml}: lidar/point-cloud sensor '{spec.name}' is missing: {missing}."
        )


# ─── BenchmarkScene validation ────────────────────────────────────────────────

_BENCHMARK_YAMLS = _collect("scenes/benchmark")


@pytest.mark.parametrize(
    "scene_yaml",
    _BENCHMARK_YAMLS,
    ids=[p.stem for p in _BENCHMARK_YAMLS],
)
def test_benchmark_scene_validates(scene_yaml: Path) -> None:
    """Every ``scenes/benchmark/*.yaml`` round-trips through ``BenchmarkScene``."""
    if not _BENCHMARK_YAMLS:
        pytest.skip("No scenes/benchmark/ directory found (wheel install?)")
    from openral_core.schemas import BenchmarkScene

    scene = BenchmarkScene.from_yaml(str(scene_yaml))
    assert scene.scene.id, f"{scene_yaml}: BenchmarkScene.scene.id must be non-empty"
    assert scene.task.success_key, (
        f"{scene_yaml}: BenchmarkScene.task.success_key is required"
    )
    assert scene.task.max_steps is not None, (
        f"{scene_yaml}: BenchmarkScene.task.max_steps is required"
    )
    assert scene.n_episodes > 0, (
        f"{scene_yaml}: BenchmarkScene.n_episodes must be > 0"
    )


# ─── SimScene validation ──────────────────────────────────────────────────────

_SIM_YAMLS = _collect("scenes/sim")


@pytest.mark.parametrize(
    "scene_yaml",
    _SIM_YAMLS,
    ids=[p.stem for p in _SIM_YAMLS],
)
def test_sim_scene_validates(scene_yaml: Path) -> None:
    """Every ``scenes/sim/*.yaml`` round-trips through ``SimScene``."""
    if not _SIM_YAMLS:
        pytest.skip("No scenes/sim/ directory found (wheel install?)")
    from openral_core.schemas import SimScene

    scene = SimScene.from_yaml(str(scene_yaml))
    assert scene.scene.id, f"{scene_yaml}: SimScene.scene.id must be non-empty"
    assert scene.task.id, f"{scene_yaml}: SimScene.task.id must be non-empty"


# ─── DeployScene validation ───────────────────────────────────────────────────

_DEPLOY_YAMLS = _collect("scenes/deploy")


@pytest.mark.parametrize(
    "scene_yaml",
    _DEPLOY_YAMLS,
    ids=[p.stem for p in _DEPLOY_YAMLS],
)
def test_deploy_scene_validates(scene_yaml: Path) -> None:
    """Every ``scenes/deploy/*.yaml`` round-trips through ``DeployScene``."""
    if not _DEPLOY_YAMLS:
        pytest.skip("No scenes/deploy/ directory found (wheel install?)")
    from openral_core.schemas import DeployScene

    scene = DeployScene.from_yaml(str(scene_yaml))
    assert scene.scene.id, f"{scene_yaml}: DeployScene.scene.id must be non-empty"
