"""Unit tests for the benchmark runner (ADR-0009 PR D + ADR-0042).

End-to-end coverage of ``openral_sim.run_benchmark`` against the mock
scene + zero policy — no GPU, no HF Hub, no physics. The mock adapter
terminates each episode at step ``success_step`` so a full
``tasks × n_episodes`` matrix completes in well under a second.

ADR-0042 (June 2026) deleted ``BenchmarkSpec``; ``run_benchmark`` and
``_aggregate_results`` now take a bare ``list[BenchmarkScene]`` + a
keyword-only ``suite_id``. Tests build the list directly via
``_mini_suite``.

Also covers the public helper ``default_output_path``: the
canonical mapping from a bare rSkill ref to
``rskills/<dir>/eval/<benchmark>.json``.
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
    SimEnvironment,
    TaskSpec,
    VLASpec,
)
from openral_sim.benchmark import (
    _aggregate_results,
    default_output_path,
    run_benchmark,
)

# ── default_output_path ───────────────────────────────────────────────────────


def test_default_output_path_canonical() -> None:
    assert (
        default_output_path("rskills/smolvla-libero", "libero_spatial")
        == "rskills/smolvla-libero/eval/libero_spatial.json"
    )


def test_default_output_path_rejects_hf_uri() -> None:
    with pytest.raises(ValueError, match="hf://"):
        default_output_path("hf://some/repo", "libero_spatial")


# ── _aggregate_results (synthetic, no rollouts) ───────────────────────────────


_MINI_SUITE_ID = "tiny_mock"


def _mini_suite(
    n_tasks: int = 2,
    n_episodes: int = 3,
) -> tuple[list[BenchmarkScene], str]:
    """Build a tiny mock benchmark suite in the post-ADR-0042 shape.

    Every BenchmarkScene shares the same scene / robot / protocol /
    metadata — the suite-level invariants in
    :func:`openral_core.raise_on_invalid_suite` require it. Tasks differ
    only in ``task.id``.

    Returns:
        ``(scenes, suite_id)`` ready to feed into ``run_benchmark`` /
        ``_aggregate_results`` (both keyword-only on ``suite_id``).
    """
    scene = SceneSpec(
        id="mock",
        backend=PhysicsBackend.MOCK,
        backend_options={"success_step": 2, "action_dim": 7},
    )
    meta = BenchmarkMetadata(
        paper="https://example.invalid/mock-paper",
        honest_scope=f"{n_episodes} episodes per task on the mock scene.",
        display_name="mock-mt2",
        simulator="mock",
    )
    scenes = [
        BenchmarkScene(
            scene=scene,
            task=TaskSpec(
                id=f"mock/{i}",
                scene_id="mock",
                instruction="noop",
                max_steps=20,
                success_key="is_success",
            ),
            robot_id="so100_follower",
            n_episodes=n_episodes,
            seed=0,
            metadata=meta,
        )
        for i in range(n_tasks)
    ]
    return scenes, _MINI_SUITE_ID


def _vla_zero() -> VLASpec:
    """A VLASpec for the built-in mock `zero` policy.

    The strict runner bypasses manifest validation only when the URI is the
    exact ``"placeholder"`` sentinel — that's the contract that lets
    the mock zero policy smoke-test scene adapters without an rSkill.
    """
    return VLASpec(
        id="zero",
        weights_uri="placeholder",
        extra={"action_dim": 7},
    )


def test_aggregate_results_all_success() -> None:
    scenes, suite_id = _mini_suite(n_tasks=2, n_episodes=3)
    per_task = {"mock/0": [True, True, True], "mock/1": [True, True, True]}
    result = _aggregate_results(
        scenes, suite_id=suite_id, vla=_vla_zero(), per_task=per_task, episodes=[]
    )
    assert isinstance(result, RSkillEvalResult)
    assert result.results["avg_success_rate"] == 1.0
    assert result.results["mock/0_success_rate"] == 1.0
    assert result.results["mock/1_success_rate"] == 1.0
    assert result.results["n_episodes_total"] == 6
    assert result.source.reproduced_locally is True


def test_aggregate_results_mixed() -> None:
    scenes, suite_id = _mini_suite(n_tasks=2, n_episodes=2)
    per_task = {"mock/0": [True, False], "mock/1": [True, True]}
    result = _aggregate_results(
        scenes, suite_id=suite_id, vla=_vla_zero(), per_task=per_task, episodes=[]
    )
    assert result.results["mock/0_success_rate"] == pytest.approx(0.5)
    assert result.results["mock/1_success_rate"] == 1.0
    assert result.results["avg_success_rate"] == pytest.approx(0.75)


def test_aggregate_results_records_reproduction_cli() -> None:
    scenes, suite_id = _mini_suite()
    vla = _vla_zero()
    result = _aggregate_results(
        scenes, suite_id=suite_id, vla=vla, per_task={"mock/0": [True]}, episodes=[]
    )
    cli = result.source.reproduction_cli
    assert isinstance(cli, str)
    assert "openral benchmark run" in cli
    assert suite_id in cli
    assert vla.weights_uri in cli
    assert "--rskill" in cli


def test_aggregate_results_uses_display_name_and_simulator_from_metadata() -> None:
    """ADR-0042: ``benchmark.name`` / ``simulator`` flow from per-scene metadata."""
    scenes, suite_id = _mini_suite()
    result = _aggregate_results(
        scenes, suite_id=suite_id, vla=_vla_zero(), per_task={"mock/0": [True]}, episodes=[]
    )
    assert result.benchmark.name == "mock-mt2"
    assert result.benchmark.simulator == "mock"


def test_aggregate_results_falls_back_to_suite_id_and_scene_id() -> None:
    """When ``display_name`` / ``simulator`` are unset, fall back to ``suite_id`` / ``scene.id``."""
    scenes, suite_id = _mini_suite()
    # Strip the optional display fields from every scene's metadata.
    bare_meta = scenes[0].metadata.model_copy(update={"display_name": None, "simulator": None})
    scenes = [s.model_copy(update={"metadata": bare_meta}) for s in scenes]
    result = _aggregate_results(
        scenes, suite_id=suite_id, vla=_vla_zero(), per_task={"mock/0": [True]}, episodes=[]
    )
    assert result.benchmark.name == suite_id
    assert result.benchmark.simulator == scenes[0].scene.id


def test_aggregate_results_arxiv_auto_derived_from_paper_url() -> None:
    """``source.arxiv`` is auto-derived when ``metadata.paper`` is an arxiv URL."""
    scenes, suite_id = _mini_suite()
    arxiv_meta = scenes[0].metadata.model_copy(update={"paper": "https://arxiv.org/abs/2306.03310"})
    scenes = [s.model_copy(update={"metadata": arxiv_meta}) for s in scenes]
    result = _aggregate_results(
        scenes, suite_id=suite_id, vla=_vla_zero(), per_task={"mock/0": [True]}, episodes=[]
    )
    assert result.source.arxiv == "https://arxiv.org/abs/2306.03310"


def test_aggregate_results_arxiv_none_for_non_arxiv_paper() -> None:
    """``source.arxiv`` stays ``None`` when the paper URL is not on arxiv."""
    scenes, suite_id = _mini_suite()
    # _mini_suite ships ``paper="https://example.invalid/mock-paper"`` which
    # does NOT contain ``arxiv.org/`` — should give ``arxiv=None``.
    result = _aggregate_results(
        scenes, suite_id=suite_id, vla=_vla_zero(), per_task={"mock/0": [True]}, episodes=[]
    )
    assert result.source.arxiv is None


# ── run_benchmark end-to-end (mock physics) ───────────────────────────────────


def test_run_benchmark_end_to_end_mock_zero_policy() -> None:
    """Full ``run_benchmark`` against the mock scene + zero policy.

    Exercises the real SimRunner loop: rSkill placeholder bypass,
    SimEnvironment per (task, seed), success aggregation, RSkillEvalResult
    emission. The mock scene reports success at step `success_step=2`
    deterministically, so the rolled-up avg_success_rate must be 1.0.
    """
    scenes, suite_id = _mini_suite(n_tasks=2, n_episodes=3)
    result, episodes = run_benchmark(scenes, suite_id=suite_id, vla=_vla_zero())

    assert isinstance(result, RSkillEvalResult)
    # 2 tasks × 3 episodes = 6 episodes.
    assert len(episodes) == 6
    assert all(e.success for e in episodes)
    assert result.results["avg_success_rate"] == 1.0
    # Every per-task success_rate present in the rollup.
    assert "mock/0_success_rate" in result.results
    assert "mock/1_success_rate" in result.results
    # eval_config faithfully echoes the protocol.
    assert result.eval_config["n_episodes"] == 3
    assert result.eval_config["seeds"] == [0, 1, 2]
    assert result.eval_config["success_key"] == "is_success"


def test_run_benchmark_writes_validated_skill_eval_result(tmp_path: Path) -> None:
    """The emitted JSON round-trips through RSkillEvalResult.from_json."""
    scenes, suite_id = _mini_suite(n_tasks=1, n_episodes=2)
    result, _ = run_benchmark(scenes, suite_id=suite_id, vla=_vla_zero())

    out = tmp_path / f"{suite_id}.json"
    out.write_text(result.model_dump_json(indent=2))

    rehydrated = RSkillEvalResult.from_json(str(out))
    # ADR-0042: ``robot_id`` lives on every ``BenchmarkScene`` (suite
    # invariants guarantee uniformity). Read it from the first scene to
    # mirror what the aggregator copies into ``RSkillEvalBenchmark.robot``.
    assert rehydrated.benchmark.robot == scenes[0].robot_id
    assert rehydrated.source.reproduced_locally is True
    assert rehydrated.results["avg_success_rate"] == 1.0


def test_run_benchmark_device_override_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """``device`` arg passes through to every SimEnvironment.vla.device.

    Drives the real mock scene + zero policy end-to-end (CLAUDE.md §1.11
    — no MagicMock); spies the SimRunner ``__init__`` only to record the
    ``env_cfg.vla.device`` the benchmark loop hands in.
    """
    from openral_sim.sim_runner import SimRunner

    seen_devices: list[str] = []
    real_init = SimRunner.__init__

    def _spy_init(self: SimRunner, env_cfg: SimEnvironment, **kw: object) -> None:
        seen_devices.append(env_cfg.vla.device)
        real_init(self, env_cfg, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(SimRunner, "__init__", _spy_init)

    scenes, suite_id = _mini_suite(n_tasks=1, n_episodes=2)
    run_benchmark(scenes, suite_id=suite_id, vla=_vla_zero(), device="cpu")
    assert seen_devices == ["cpu", "cpu"]
