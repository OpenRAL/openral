"""Unit tests for :class:`openral_sim.SimRunner` (ADR-0010 amendment 1).

Exercises the per-step tick model against the real built-in ``mock``
scene and ``zero`` / ``random`` policies — no mocks, no stubs, no
patches. All assertions are on real :class:`TickResult` / :class:`EpisodeResult`
shapes (CLAUDE.md §1.11).

Pins:

* Protocol conformance against the structural :class:`InferenceRunner`.
* Reset-tick vs step-tick semantics (action_applied flag, inference_ms,
  episode_idx, step_idx).
* :meth:`SimRunner._should_terminate` stops :meth:`run` once
  ``n_episodes`` are emitted, regardless of the ``max_ticks`` ceiling.
* Trailing-episode flush on :meth:`deactivate`.
* Per-episode seeding (``seed + episode_idx``).
* Deactivate idempotence.
"""

from __future__ import annotations

from openral_core import (
    DeadlineOverrunPolicy,
    PhysicsBackend,
    SceneSpec,
    SimEnvironment,
    TaskSpec,
    VLASpec,
)
from openral_runner import InferenceRunner
from openral_sim import SimRunner
from openral_sim.sim_runner import _resolve_step_instruction


def _mock_env(
    *,
    n_episodes: int = 1,
    max_steps: int = 10,
    success_step: int = 3,
    vla_id: str = "zero",
    vla_seed: int = 0,
) -> SimEnvironment:
    """Build a real ``mock`` SimEnvironment — same shape as ADR-0009 unit fixtures."""
    return SimEnvironment(
        robot_id="so100_follower",
        scene=SceneSpec(
            id="mock",
            backend=PhysicsBackend.MOCK,
            backend_options={"success_step": success_step, "action_dim": 7},
        ),
        task=TaskSpec(
            id="mock/0",
            scene_id="mock",
            instruction="noop",
            max_steps=max_steps,
            success_key="is_success",
        ),
        vla=VLASpec(
            id=vla_id,
            weights_uri="placeholder",  # short-circuits manifest load
            extra={"action_dim": 7, "seed": vla_seed},
        ),
        n_episodes=n_episodes,
    )


# ── Protocol + lifecycle ─────────────────────────────────────────────────────


def test_sim_runner_satisfies_inference_runner_protocol() -> None:
    """Structural ``isinstance`` check passes (matches HardwareRunner)."""
    runner = SimRunner(_mock_env())
    assert isinstance(runner, InferenceRunner)


def test_sim_runner_run_emits_n_episode_results() -> None:
    """``run(max_ticks=large)`` stops at ``n_episodes`` via _should_terminate."""
    env_cfg = _mock_env(n_episodes=3, success_step=2, max_steps=5)
    runner = SimRunner(env_cfg)
    runner.activate()
    try:
        # max_ticks deliberately oversized; the hook is the real stop.
        runner.run(max_ticks=1000)
    finally:
        runner.deactivate()
    assert len(runner.episode_results) == 3
    # mock scene terminates at success_step; success is True
    for ep in runner.episode_results:
        assert ep.success is True
        assert ep.steps == 2


# ── Tick-flavour semantics ──────────────────────────────────────────────────


def test_first_tick_is_reset_tick() -> None:
    """The first tick after activate() is a reset tick: no inference, no action."""
    runner = SimRunner(_mock_env())
    runner.activate()
    try:
        tr = runner.tick()
    finally:
        runner.deactivate()
    assert tr.action_applied is False
    assert tr.inference_ms == 0.0
    assert tr.step_idx is None
    assert tr.episode_idx == 0
    assert tr.reward is None
    assert tr.terminated is None
    assert tr.truncated is None


def test_step_ticks_populate_sim_fields() -> None:
    """Step ticks expose step_idx, reward, terminated, truncated."""
    env_cfg = _mock_env(success_step=3, max_steps=5)
    runner = SimRunner(env_cfg)
    runner.activate()
    try:
        reset = runner.tick()  # reset
        step0 = runner.tick()  # step 0
        step1 = runner.tick()  # step 1
        step2 = runner.tick()  # step 2 → terminates
    finally:
        runner.deactivate()
    assert reset.action_applied is False
    assert step0.step_idx == 0
    assert step1.step_idx == 1
    assert step2.step_idx == 2
    assert step2.terminated is True
    assert step2.action_applied is True
    # mock scene rewards 1.0 only on the final (success) step
    assert step0.reward == 0.0
    assert step2.reward == 1.0


def test_truncation_when_max_steps_hit_without_env_signal() -> None:
    """If env never terminates, hitting max_steps marks the tick as truncated."""
    # success_step=20 > max_steps=4 → never terminates naturally
    env_cfg = _mock_env(success_step=20, max_steps=4, n_episodes=1)
    runner = SimRunner(env_cfg)
    runner.activate()
    try:
        runner.run(max_ticks=1000)
    finally:
        runner.deactivate()
    ep = runner.episode_results[0]
    assert ep.steps == 4
    assert ep.success is False


# ── Episode boundary + trailing flush ────────────────────────────────────────


def test_episode_boundary_finalises_result() -> None:
    """A single full episode (reset → 2 steps → terminate) emits exactly one result."""
    env_cfg = _mock_env(n_episodes=1, success_step=2)
    runner = SimRunner(env_cfg)
    runner.activate()
    try:
        runner.run(max_ticks=1000)
    finally:
        runner.deactivate()
    assert len(runner.episode_results) == 1


def test_trailing_episode_flushed_on_deactivate() -> None:
    """If max_ticks cuts us off mid-episode, deactivate emits the partial result."""
    # success_step=100 > max_steps=50; run for 5 ticks only → partial episode
    env_cfg = _mock_env(n_episodes=1, success_step=100, max_steps=50)
    runner = SimRunner(env_cfg)
    runner.activate()
    try:
        runner.run(max_ticks=5)  # 1 reset + 4 steps
    finally:
        runner.deactivate()
    assert len(runner.episode_results) == 1
    ep = runner.episode_results[0]
    assert ep.steps == 4  # 4 step-ticks completed
    assert ep.success is False


def test_deactivate_is_idempotent() -> None:
    """Calling deactivate twice does not double-close the env or re-emit episodes."""
    runner = SimRunner(_mock_env(success_step=2, n_episodes=1))
    runner.activate()
    runner.run(max_ticks=1000)
    n_first = len(runner.episode_results)
    runner.deactivate()
    n_after_first = len(runner.episode_results)
    runner.deactivate()
    assert n_after_first == n_first
    assert len(runner.episode_results) == n_first


# ── Determinism ─────────────────────────────────────────────────────────────


def test_per_episode_seeding_is_deterministic_across_runs() -> None:
    """Two SimRunners over the same env_cfg produce identical EpisodeResult dims.

    The random policy pulls from torch / numpy RNGs which SimRunner seeds
    via ``_seed_global_rngs`` on every reset-tick. Two runs must therefore
    produce the same reward / steps / latency-shape (the *latency values*
    will differ since they're wall-clock, but step counts and rewards are
    deterministic).
    """
    env_cfg = _mock_env(
        n_episodes=2,
        success_step=3,
        max_steps=5,
        vla_id="random",
        vla_seed=42,
    )

    def _run() -> list[tuple[int, float, bool]]:
        r = SimRunner(env_cfg)
        r.activate()
        try:
            r.run(max_ticks=1000)
        finally:
            r.deactivate()
        return [(ep.steps, ep.total_reward, ep.success) for ep in r.episode_results]

    a = _run()
    b = _run()
    assert a == b
    assert len(a) == 2


# ── Deadline policy ─────────────────────────────────────────────────────────


def test_default_deadline_policy_warn_does_not_raise() -> None:
    """SimRunner defaults to WARN — long ticks log but don't raise."""
    # Mock scene is fast; this is mostly a smoke test that the default
    # plumbing doesn't blow up.
    runner = SimRunner(_mock_env(success_step=2, n_episodes=1))
    runner.activate()
    try:
        runner.run(max_ticks=1000)
    finally:
        runner.deactivate()
    assert len(runner.episode_results) == 1


def test_deadline_policy_is_propagated_to_base() -> None:
    """The deadline-overrun policy kwarg reaches the base class unchanged."""
    runner = SimRunner(
        _mock_env(success_step=2, n_episodes=1),
        deadline_overrun_policy=DeadlineOverrunPolicy.RAISE,
    )
    assert runner.deadline_overrun_policy == DeadlineOverrunPolicy.RAISE
    # Default is WARN — pinned so a regression here breaks loudly.
    default_runner = SimRunner(_mock_env(success_step=2, n_episodes=1))
    assert default_runner.deadline_overrun_policy == DeadlineOverrunPolicy.WARN


# ── TickResult fields wiring ────────────────────────────────────────────────


def test_tick_results_increment_tick_idx() -> None:
    """``tick_idx`` increments monotonically across reset + step ticks."""
    runner = SimRunner(_mock_env(success_step=3, n_episodes=1, max_steps=5))
    runner.activate()
    try:
        seen = [runner.tick() for _ in range(4)]
    finally:
        runner.deactivate()
    assert [t.tick_idx for t in seen] == [0, 1, 2, 3]


# ── Per-step instruction precedence (--instruction override) ─────────────────
#
# Regression for the silent `--instruction` override loss: a scene whose
# env exposes a per-episode `obs["task"]` language (for example, RoboCasa
# sampled object name) used to unconditionally beat the user's explicit
# `--instruction`. An explicit override MUST win; the env language must still
# win when the user passed nothing.


def test_explicit_override_beats_env_language() -> None:
    """An explicit ``--instruction`` wins over a scene's env language."""
    instr = _resolve_step_instruction(
        instruction_override="Pick up orange juice and place in the basket",
        obs_task="Pick the milk and place it in the basket",
        task_instruction="",
    )
    assert instr == "Pick up orange juice and place in the basket"


def test_env_language_used_when_no_override() -> None:
    """No override → the env's per-episode language (e.g. RoboCasa) wins.

    Preserves the deliberate priority at ``sim_runner`` that lets RoboCasa
    interpolate the sampled object name into ``obs["task"]``.
    """
    instr = _resolve_step_instruction(
        instruction_override=None,
        obs_task="put the akita_black_bowl on the plate",
        task_instruction="put the bowl on the plate",
    )
    assert instr == "put the akita_black_bowl on the plate"


def test_yaml_instruction_is_final_fallback() -> None:
    """No override and no env language → the static YAML instruction is used."""
    instr = _resolve_step_instruction(
        instruction_override=None,
        obs_task="",
        task_instruction="noop",
    )
    assert instr == "noop"


def test_blank_override_is_ignored() -> None:
    """A whitespace-only override is treated as absent, not as a wipe."""
    instr = _resolve_step_instruction(
        instruction_override="   ",
        obs_task="put the bowl on the plate",
        task_instruction="fallback",
    )
    assert instr == "put the bowl on the plate"


def test_non_string_obs_task_falls_through() -> None:
    """A non-``str`` ``obs["task"]`` (e.g. missing key → None) is skipped."""
    instr = _resolve_step_instruction(
        instruction_override=None,
        obs_task=None,
        task_instruction="noop",
    )
    assert instr == "noop"
