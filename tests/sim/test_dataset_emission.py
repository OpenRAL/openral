"""ADR-0019 end-to-end sim test: openral sim run --dataset-out with a real VLA.

Drives ``openral_sim.SimRunner`` against
``scenes/sim/aloha_transfer_cube.yaml`` with a real
``openral_dataset.RolloutRecorder`` + ``LeRobotDatasetSink`` attached,
then re-opens the produced LeRobotDataset v3 via the real lerobot
reader and asserts the bridge round-trip works against a wildly
different state shape than SO-100 (14-DoF Aloha bimanual at 50 Hz vs
6-DoF SO-100 at 30 Hz).

Bridges the gap between
``python/dataset/tests/test_sink_lerobot.py`` (sink tested with
fabricated zero frames) and a real sim rollout (real ACT weights, real
gym-aloha physics, real SVT-AV1 video encoding of real 640x480 frames).

Per CLAUDE.md §1.11: no mocks. Skips cleanly when CUDA, lerobot,
torch, or gym_aloha are unavailable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# Use `importlib.util.find_spec` + `pytestmark` rather than module-level
# `pytest.importorskip` / `pytest.skip(allow_module_level=True)`: with
# `tests/sim/__init__.py` making this directory a Package, a Skipped raised
# at module-import time poisons the whole `tests/sim` Package collection
# ("found no collectors for ..." on every sibling). Deferring the decision
# to `pytestmark` keeps this module importable when optional deps are
# missing, so sibling files remain reachable.
_REQUIRED_MODULES = ("torch", "gymnasium", "gym_aloha", "lerobot")
_MISSING_MODULES = tuple(m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None)
_CUDA_AVAILABLE = False
if not _MISSING_MODULES:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason="dataset emission sim test requires " + ", ".join(_MISSING_MODULES),
    ),
    pytest.mark.skipif(
        not _CUDA_AVAILABLE,
        reason="dataset emission sim test requires CUDA",
    ),
]

_REPO_ROOT = Path(__file__).parent.parent.parent
# A SimScene (not a BenchmarkScene): `compose_sim_env` loads via
# `load_scene_strict(..., SimScene)`, which rejects BenchmarkScene YAMLs
# (n_episodes/seed/metadata) — so point at the single-rollout sim sibling.
_CONFIG = _REPO_ROOT / "scenes" / "sim" / "aloha_transfer_cube.yaml"


@pytest.fixture(scope="module")
def env_cfg():
    """Compose the same SimEnvironment the CLI builds for `openral sim run`."""
    from tests.sim.conftest import compose_sim_env

    if not _CONFIG.exists():
        pytest.skip(f"sim config not found at {_CONFIG}")
    return compose_sim_env(
        _CONFIG,
        rskill_uri="rskills/act-aloha",
        n_episodes=1,
        max_steps=5,
    )


def test_sim_run_with_dataset_out_produces_reloadable_v3(env_cfg, tmp_path: Path) -> None:
    """Run a 5-step Aloha rollout with --dataset-out and reload the dataset.

    Mirrors the real CLI invocation in `python/sim/src/openral_sim/cli.py`
    so any drift in how the CLI builds the recorder lands as a test
    failure. The test is short (5 steps × 1 episode) so it fits in the
    sim CI tier's <10 min budget.
    """
    from openral_dataset import LeRobotDatasetSink, RolloutRecorder
    from openral_sim.registry import ROBOTS
    from openral_sim.sim_runner import SimRunner

    # Resolve robot via the same registry path the CLI uses.
    robot = ROBOTS.get(env_cfg.robot_id)()
    assert robot.observation_spec is not None  # type narrowing
    assert robot.action_spec is not None
    fps = float(robot.action_spec.control_freq_hz) if robot.action_spec.control_freq_hz else 30.0

    ds_root = tmp_path / "ds"
    # ADR-0019: pass the camera shape from the scene config (sim renders
    # all cameras at one resolution) and the action dim from the rSkill
    # manifest's action_contract.
    from openral_rskill.loader import load_rskill_manifest

    manifest = load_rskill_manifest("rskills/act-aloha")
    action_dim_override = (
        manifest.action_contract.dim if manifest.action_contract is not None else None
    )
    state_shape_override = (
        (manifest.state_contract.dim,)
        if manifest.state_contract is not None and manifest.state_contract.dim
        else None
    )
    sink = LeRobotDatasetSink(
        root=ds_root,
        robot=robot,
        fps=fps,
        repo_id="openral/dataset-test-aloha-emission",
        state_shape=state_shape_override,
        action_dim=action_dim_override,
        camera_shape=(
            int(env_cfg.scene.observation_height),
            int(env_cfg.scene.observation_width),
        ),
    )
    recorder = RolloutRecorder(
        robot=robot,
        task_string=env_cfg.task.instruction,
        fps=fps,
        sinks=[sink],
        repo_id="openral/dataset-test-aloha-emission",
    )

    runner = SimRunner(env_cfg, recorder=recorder)
    runner.activate()
    try:
        # max_ticks accounts for the leading reset-tick per episode.
        runner.run(max_ticks=env_cfg.n_episodes * (env_cfg.task.max_steps + 1))
    finally:
        runner.deactivate()

    # Reload via the real lerobot reader and prove the bridge round-tripped.
    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("openral/dataset-test-aloha-emission", root=ds_root)
    assert ds.num_episodes == 1
    assert ds.num_frames >= 1, "should have at least one frame in the bag"

    info = json.loads((ds_root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["metadata"]["repo_id"] == "openral/dataset-test-aloha-emission"
    assert info["metadata"]["license"] == "CC-BY-4.0"
    assert "dataset_success_rate" in info["metadata"]

    # Aloha-specific shape assertion — the dataset must round-trip the
    # actual 14-DoF state / action, not silently truncate to a fixed
    # dim. This is the test that catches "the bridge works for SO-100
    # but secretly assumes 6-DoF state" regressions.
    row = ds[0]
    assert row["observation.state"].numel() == 14, (
        f"Aloha is 14-DoF bimanual; got state dim {row['observation.state'].numel()}"
    )
    assert row["action"].numel() == 14, (
        f"Aloha is 14-DoF bimanual; got action dim {row['action'].numel()}"
    )

    # Per-camera video must be present — Aloha declares one `top` camera.
    cam_features = sorted(k for k in info["features"] if k.startswith("observation.images."))
    assert cam_features == ["observation.images.top"], (
        f"Aloha declares one `top` camera; got {cam_features!r}"
    )
