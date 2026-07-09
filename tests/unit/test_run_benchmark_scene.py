"""Unit tests for ``run_benchmark_scene`` — single-scene benchmark loop.

Mirrors :mod:`tests.unit.test_benchmark_runner` but for the
``BenchmarkScene``-based entrypoint that backs ``openral benchmark scene``.
End-to-end coverage of the runtime aggregation against the mock scene +
zero policy — no GPU, no HF Hub, no physics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import (
    BenchmarkMetadata,
    BenchmarkScene,
    PhysicsBackend,
    RSkillEvalResult,
    SceneSpec,
    TaskSpec,
    VLASpec,
)
from openral_sim.benchmark import run_benchmark_scene


def _mini_scene(n_episodes: int = 3, seed: int = 0) -> BenchmarkScene:
    """Build a minimal BenchmarkScene against the in-tree ``mock`` adapter.

    The mock scene reports success at ``success_step`` deterministically,
    so the rolled-up ``avg_success_rate`` is 1.0 for every episode.
    """
    return BenchmarkScene(
        scene=SceneSpec(
            id="mock",
            backend=PhysicsBackend.MOCK,
            backend_options={"success_step": 2, "action_dim": 7},
        ),
        robot_id="so100_follower",
        task=TaskSpec(
            id="mock/0",
            scene_id="mock",
            instruction="noop",
            max_steps=10,
            success_key="is_success",
        ),
        n_episodes=n_episodes,
        seed=seed,
        metadata=BenchmarkMetadata(
            paper="https://example.invalid/mock",
            honest_scope="mock scene — deterministic success at success_step=2",
        ),
    )


def _vla_zero() -> VLASpec:
    """A VLASpec for the built-in mock ``zero`` policy.

    The strict runner bypasses manifest validation only when the URI is the
    exact ``"placeholder"`` sentinel — same contract as the suite runner.
    """
    return VLASpec(id="zero", weights_uri="placeholder", extra={"action_dim": 7})


def test_run_benchmark_scene_end_to_end_mock_zero_policy() -> None:
    """Full ``run_benchmark_scene`` against the mock scene + zero policy.

    Exercises the real SimRunner loop with one task × N seeds derived from
    ``BenchmarkScene.seed`` (paired with ``seed + ep``). The mock scene
    succeeds at step 2, so every episode is a hit.
    """
    scene = _mini_scene(n_episodes=3, seed=0)
    result, episodes = run_benchmark_scene(scene, _vla_zero())

    assert isinstance(result, RSkillEvalResult)
    assert len(episodes) == 3
    assert all(e.success for e in episodes)
    assert result.results["avg_success_rate"] == 1.0
    assert "mock/0_success_rate" in result.results
    # eval_config faithfully echoes the BenchmarkScene-derived protocol.
    assert result.eval_config["n_episodes"] == 3
    assert result.eval_config["seeds"] == [0, 1, 2]
    assert result.eval_config["success_key"] == "is_success"
    # The reproduction CLI points at the per-scene entrypoint, not the suite one.
    assert "openral benchmark scene" in (result.source.reproduction_cli or "")


def test_run_benchmark_scene_seed_offset_propagates() -> None:
    """``BenchmarkScene.seed`` is the start of an ``[seed, seed+n)`` range.

    The benchmark runner must derive its per-episode seed list from the
    scalar ``seed`` field; the SuiteSpec equivalent ships an explicit
    ``protocol.seeds`` list.
    """
    scene = _mini_scene(n_episodes=2, seed=100)
    result, episodes = run_benchmark_scene(scene, _vla_zero())
    assert len(episodes) == 2
    assert result.eval_config["seeds"] == [100, 101]


def test_run_benchmark_scene_writes_validated_skill_eval_result(tmp_path: Path) -> None:
    """The emitted JSON round-trips through :meth:`RSkillEvalResult.from_json`."""
    scene = _mini_scene(n_episodes=2, seed=0)
    result, _ = run_benchmark_scene(scene, _vla_zero())

    out = tmp_path / "mock.json"
    out.write_text(result.model_dump_json(indent=2))

    rehydrated = RSkillEvalResult.from_json(str(out))
    assert rehydrated.benchmark.robot == scene.robot_id
    assert rehydrated.source.reproduced_locally is True
    assert rehydrated.results["avg_success_rate"] == 1.0


def test_run_benchmark_scene_device_override_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``device`` arg threads through to every SimEnvironment.vla.device."""
    from openral_core import SimEnvironment
    from openral_sim.sim_runner import SimRunner

    seen_devices: list[str] = []
    real_init = SimRunner.__init__

    def _spy_init(self: SimRunner, env_cfg: SimEnvironment, **kw: object) -> None:
        seen_devices.append(env_cfg.vla.device)
        real_init(self, env_cfg, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(SimRunner, "__init__", _spy_init)

    scene = _mini_scene(n_episodes=2, seed=0)
    run_benchmark_scene(scene, _vla_zero(), device="cpu")
    assert seen_devices == ["cpu", "cpu"]


def test_run_benchmark_scene_base_pose_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BenchmarkScene.base_pose`` threads into every ``SimEnvironment``.

    Free-axis scene adapters (e.g. ``openarm_tabletop_pnp``) require a
    mounting pose set via ``base_pose:`` in the YAML. The benchmark scene
    runner must carry it through to the composed ``SimEnvironment`` —
    otherwise those scenes raise ``ROSConfigError`` at env construction.
    """
    from openral_core import Pose6D, SimEnvironment
    from openral_sim.sim_runner import SimRunner

    seen_poses: list[Pose6D | None] = []
    real_init = SimRunner.__init__

    def _spy_init(self: SimRunner, env_cfg: SimEnvironment, **kw: object) -> None:
        seen_poses.append(env_cfg.base_pose)
        real_init(self, env_cfg, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(SimRunner, "__init__", _spy_init)

    pose = Pose6D(
        xyz=(0.20, 0.0, 0.55),
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
    )
    scene = _mini_scene(n_episodes=2, seed=0).model_copy(update={"base_pose": pose})
    run_benchmark_scene(scene, _vla_zero())
    assert seen_poses == [pose, pose]


def _spy_view_kwargs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, object]]:
    """Patch ``SimRunner.__init__`` to record ``(view, strict_view)`` per episode."""
    from openral_core import SimEnvironment
    from openral_sim.sim_runner import SimRunner

    seen: list[tuple[object, object]] = []
    real_init = SimRunner.__init__

    def _spy_init(self: SimRunner, env_cfg: SimEnvironment, **kw: object) -> None:
        seen.append((kw.get("view"), kw.get("strict_view")))
        real_init(self, env_cfg, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(SimRunner, "__init__", _spy_init)
    return seen


def test_run_benchmark_scene_view_default_is_headless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default (``view`` unset) keeps the historical headless rollout.

    Benchmark eval artefacts and CI/deploy runs must be unaffected by the
    new ``--view`` plumbing: every ``SimRunner`` is still built with
    ``view=False, strict_view=False`` when no flag is passed.
    """
    seen = _spy_view_kwargs(monkeypatch)
    scene = _mini_scene(n_episodes=2, seed=0)
    run_benchmark_scene(scene, _vla_zero())
    assert seen == [(False, False), (False, False)]


def test_run_benchmark_scene_view_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``view`` is resolved through ``_resolve_view`` and reaches SimRunner.

    Mirrors :func:`test_run_benchmark_scene_base_pose_propagates` — confirms
    the opt-in viewer flag threads from ``run_benchmark_scene`` into every
    per-episode ``SimRunner``. ``view=False`` resolves to a strict-off,
    offscreen rollout on any host (no display required), so the assertion is
    deterministic in headless CI.
    """
    seen = _spy_view_kwargs(monkeypatch)
    scene = _mini_scene(n_episodes=2, seed=0)
    run_benchmark_scene(scene, _vla_zero(), view=False)
    assert seen == [(False, False), (False, False)]
