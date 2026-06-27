"""Schema-compliance tests for every in-tree robot, scene, and sensor manifest.

Policy
------
Every manifest YAML in ``robots/``, ``scenes/benchmark/``, ``scenes/sim/``,
and ``scenes/deploy/`` **must** be explicitly listed in the registry below.
If a new file is added to the repository without being added here the
*completeness* tests will fail, forcing the author to both list the file and
verify it passes schema validation.

This is intentionally NOT a dynamic glob-all approach:
- New files must be consciously registered, not silently picked up.
- The registries act as the single-place audit trail of what manifests
  exist and which schema tier each belongs to.

Coverage
--------
- :data:`_ROBOT_IDS`       → ``robots/<id>/robot.yaml``       via :class:`RobotDescription`
- :data:`_BENCHMARK_STEMS` → ``scenes/benchmark/<stem>.yaml`` via :class:`BenchmarkScene`
- :data:`_SIM_STEMS`       → ``scenes/sim/<stem>.yaml``       via :class:`SimScene`
- :data:`_DEPLOY_STEMS`    → ``scenes/deploy/<stem>.yaml``    via :class:`DeployScene`
- Per-robot sensor entries → :class:`SensorSpec` with modality-specific
  invariants (intrinsics for RGB/depth/stereo; n_channels + range for
  lidar/point-cloud).

Adding a new manifest
---------------------
1. Place the YAML in the correct subdirectory (determines schema tier).
2. Add its identifier to the matching registry constant below.
3. Run ``just test -k test_all_manifests`` — all tests must pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

# ─── Explicit registries — update when adding a manifest ──────────────────────

_ROBOT_IDS: list[str] = [
    "aloha_agilex",
    "aloha_bimanual",
    "franka_panda",
    "g1",
    "google_robot",
    "gr1",
    "h1",
    "openarm",
    "panda_mobile",
    "pusht_2d",
    "rizon4",
    "sawyer",
    "so100_follower",
    "so101_follower",
    "ur10e",
    "ur5e",
    "widowx",
]

_BENCHMARK_STEMS: list[str] = [
    "aloha_insertion",
    "aloha_transfer_cube",
    "libero_10",
    "libero_goal",
    "libero_object",
    "libero_spatial",
    "maniskill_pick_cube",
    "metaworld_button_press",
    "metaworld_door_open",
    "metaworld_drawer_open",
    "metaworld_pick_place",
    "metaworld_push",
    "pusht",
    "rlbench_close_jar",
    "rlbench_meat_off_grill",
    "rlbench_open_drawer",
    "robocasa_pnp",
    "robotwin_beat_block_hammer",
    "robotwin_handover_block",
    "robotwin_lift_pot",
    "robotwin_place_empty_cup",
    "robotwin_stack_blocks_two",
    "widowx_carrot_on_plate",
]

_SIM_STEMS: list[str] = [
    "aloha_transfer_cube",
    "isaac_franka_bowl_plate",
    "libero_spatial",
    "openarm_tabletop",
    "robocasa_gr1_pnp_cup_to_drawer",
    "robocasa_panda_mobile_kitchen",
    "robocasa_pnp",
    "so101_tube_insertion",
    "tabletop_cube_push",
    "widowx_carrot_on_plate",
]

_DEPLOY_STEMS: list[str] = [
    "isaac_franka",
    "isaac_franka_bowl",
    "isaac_franka_urdf",
    "isaac_panda_mobile_urdf",
    "libero_object",
    "libero_pnp",
    "openarm_tabletop",
    "robocasa_baguette",
    "robocasa_navigate",
    "robocasa_pnp",
    "so101_box",
]


# ─── Repository-root helper ───────────────────────────────────────────────────


def _repo_root() -> Path | None:
    """Return the repository root by walking up from this file, or None."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "robots").is_dir() and (parent / "scenes").is_dir():
            return parent
    return None


_ROOT = _repo_root()


def _require_root() -> Path:
    if _ROOT is None:
        pytest.skip("No repo root found (wheel install?)")
    return _ROOT  # type: ignore[return-value]


# ─── Completeness guards — fail when the on-disk set ≠ registry ───────────────


def test_robot_registry_is_complete() -> None:
    """Every ``robots/<id>/robot.yaml`` on disk is listed in ``_ROBOT_IDS``."""
    root = _require_root()
    on_disk = sorted(
        d.name for d in (root / "robots").iterdir() if d.is_dir() and (d / "robot.yaml").is_file()
    )
    registered = sorted(_ROBOT_IDS)
    unlisted = [r for r in on_disk if r not in registered]
    assert not unlisted, (
        f"robots/ contains manifests not listed in _ROBOT_IDS: {unlisted}. "
        "Add them to the registry in this file and verify they pass validation."
    )
    missing = [r for r in registered if r not in on_disk]
    assert not missing, (
        f"_ROBOT_IDS references robot IDs with no robot.yaml on disk: {missing}. "
        "Remove stale entries from the registry."
    )


def test_benchmark_scene_registry_is_complete() -> None:
    """Every ``scenes/benchmark/*.yaml`` on disk is listed in ``_BENCHMARK_STEMS``."""
    root = _require_root()
    on_disk = sorted(p.stem for p in (root / "scenes" / "benchmark").glob("*.yaml"))
    registered = sorted(_BENCHMARK_STEMS)
    unlisted = [s for s in on_disk if s not in registered]
    assert not unlisted, (
        f"scenes/benchmark/ contains files not listed in _BENCHMARK_STEMS: {unlisted}. "
        "Add them and verify they pass BenchmarkScene validation."
    )
    missing = [s for s in registered if s not in on_disk]
    assert not missing, f"_BENCHMARK_STEMS has stale entries (no file on disk): {missing}."


def test_sim_scene_registry_is_complete() -> None:
    """Every ``scenes/sim/*.yaml`` on disk is listed in ``_SIM_STEMS``."""
    root = _require_root()
    on_disk = sorted(p.stem for p in (root / "scenes" / "sim").glob("*.yaml"))
    registered = sorted(_SIM_STEMS)
    unlisted = [s for s in on_disk if s not in registered]
    assert not unlisted, (
        f"scenes/sim/ contains files not listed in _SIM_STEMS: {unlisted}. "
        "Add them and verify they pass SimScene validation."
    )
    missing = [s for s in registered if s not in on_disk]
    assert not missing, f"_SIM_STEMS has stale entries (no file on disk): {missing}."


def test_deploy_scene_registry_is_complete() -> None:
    """Every ``scenes/deploy/*.yaml`` on disk is listed in ``_DEPLOY_STEMS``."""
    root = _require_root()
    on_disk = sorted(p.stem for p in (root / "scenes" / "deploy").glob("*.yaml"))
    registered = sorted(_DEPLOY_STEMS)
    unlisted = [s for s in on_disk if s not in registered]
    assert not unlisted, (
        f"scenes/deploy/ contains files not listed in _DEPLOY_STEMS: {unlisted}. "
        "Add them and verify they pass DeployScene validation."
    )
    missing = [s for s in registered if s not in on_disk]
    assert not missing, f"_DEPLOY_STEMS has stale entries (no file on disk): {missing}."


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _robot_yaml(robot_id: str) -> Path:
    return _require_root() / "robots" / robot_id / "robot.yaml"


def _scene_yaml(tier: str, stem: str) -> Path:
    return _require_root() / "scenes" / tier / f"{stem}.yaml"


def _collect_sensors() -> list[tuple[str, Any]]:
    """Return (``robot_id/sensor_name``, raw dict) for every declared sensor."""
    if _ROOT is None:
        return []
    params: list[tuple[str, Any]] = []
    for robot_id in _ROBOT_IDS:
        path = _ROOT / "robots" / robot_id / "robot.yaml"
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        for sensor in data.get("sensors", []) or []:
            params.append((f"{robot_id}/{sensor.get('name', '?')}", sensor))
    return params


_SENSOR_PARAMS = _collect_sensors()


# ─── Robot manifest validation ────────────────────────────────────────────────


@pytest.mark.parametrize("robot_id", _ROBOT_IDS)
def test_robot_manifest_validates(robot_id: str) -> None:
    """``robots/<id>/robot.yaml`` round-trips through :class:`RobotDescription`."""
    from openral_core.schemas import RobotDescription

    desc = RobotDescription.from_yaml(str(_robot_yaml(robot_id)))
    assert desc.name, f"{robot_id}: RobotDescription.name must be non-empty"
    assert desc.joints, f"{robot_id}: RobotDescription must declare at least one joint"


# ─── Per-robot sensor spec validation ────────────────────────────────────────


@pytest.mark.parametrize(
    "sensor_id,sensor_raw",
    _SENSOR_PARAMS,
    ids=[sid for sid, _ in _SENSOR_PARAMS],
)
def test_sensor_spec_validates(sensor_id: str, sensor_raw: dict[str, Any]) -> None:
    """Each sensor entry in a robot manifest validates as :class:`SensorSpec`.

    Modality-specific invariants enforced here (beyond Pydantic field typing):

    - ``rgb`` / ``depth`` / ``stereo`` → ``intrinsics`` block is required.
    - ``lidar_2d`` / ``point_cloud``   → ``n_channels``, ``range_min_m``,
      and ``range_max_m`` are all required.
    """
    from openral_core.schemas import SensorModality, SensorSpec

    spec = SensorSpec.model_validate(sensor_raw)

    if spec.modality in (SensorModality.RGB, SensorModality.DEPTH, SensorModality.STEREO):
        assert spec.intrinsics is not None, (
            f"{sensor_id}: modality={spec.modality!r} but missing 'intrinsics' "
            "(width, height, fx, fy, cx, cy)."
        )

    if spec.modality in (SensorModality.LIDAR_2D, SensorModality.POINT_CLOUD):
        missing = [
            f for f in ("n_channels", "range_min_m", "range_max_m") if getattr(spec, f) is None
        ]
        assert not missing, f"{sensor_id}: lidar/point-cloud sensor missing fields: {missing}."


# ─── BenchmarkScene validation ────────────────────────────────────────────────


@pytest.mark.parametrize("stem", _BENCHMARK_STEMS)
def test_benchmark_scene_validates(stem: str) -> None:
    """``scenes/benchmark/<stem>.yaml`` round-trips through :class:`BenchmarkScene`."""
    from openral_core.schemas import BenchmarkScene

    scene = BenchmarkScene.from_yaml(str(_scene_yaml("benchmark", stem)))
    assert scene.scene.id
    assert scene.task.success_key
    assert scene.task.max_steps is not None
    assert scene.n_episodes > 0


# ─── SimScene validation ──────────────────────────────────────────────────────


@pytest.mark.parametrize("stem", _SIM_STEMS)
def test_sim_scene_validates(stem: str) -> None:
    """``scenes/sim/<stem>.yaml`` round-trips through :class:`SimScene`."""
    from openral_core.schemas import SimScene

    scene = SimScene.from_yaml(str(_scene_yaml("sim", stem)))
    assert scene.scene.id
    assert scene.task.id


# ─── DeployScene validation ───────────────────────────────────────────────────


@pytest.mark.parametrize("stem", _DEPLOY_STEMS)
def test_deploy_scene_validates(stem: str) -> None:
    """``scenes/deploy/<stem>.yaml`` round-trips through :class:`DeployScene`."""
    from openral_core.schemas import DeployScene

    scene = DeployScene.from_yaml(str(_scene_yaml("deploy", stem)))
    assert scene.scene.id
