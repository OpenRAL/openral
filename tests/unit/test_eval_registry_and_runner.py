"""Unit tests for the eval registry, factory, and SimRunner contract gate.

:class:`SimRunner` is **strict** by construction (see
``_check_rskill_compatibility``): every sim eval must be backed by an
rSkill manifest and a registered :class:`RobotDescription`. To exercise
the factory + episode loop without a real rSkill, these tests:

- drive :class:`SimRunner` against the mock scene via the
  ``"placeholder"`` sentinel that short-circuits manifest load;
- assert :func:`_check_rskill_compatibility` rejects misconfigured
  :class:`SimEnvironment` values.
"""

from __future__ import annotations

import pytest
from openral_core import (
    PhysicsBackend,
    SceneSpec,
    SimEnvironment,
    TaskSpec,
    VLASpec,
)
from openral_core.exceptions import ROSConfigError
from openral_sim import POLICIES, SCENES, SimRunner, make_env, make_policy
from openral_sim.cli import main as cli_main
from openral_sim.cli import sim_app
from openral_sim.registry import _Registry
from openral_sim.sim_runner import _check_rskill_compatibility
from typer.testing import CliRunner


def _mock_env(**overrides: object) -> SimEnvironment:
    base: dict[str, object] = {
        "robot_id": "so100_follower",
        "scene": SceneSpec(
            id="mock",
            backend=PhysicsBackend.MOCK,
            backend_options={"success_step": 3, "action_dim": 7},
        ),
        "task": TaskSpec(
            id="mock/0",
            scene_id="mock",
            instruction="noop",
            max_steps=10,
            success_key="is_success",
        ),
        "vla": VLASpec(id="zero", weights_uri="mock://noop", extra={"action_dim": 7}),
    }
    base.update(overrides)
    return SimEnvironment(**base)  # type: ignore[arg-type]


def _runnable_env(**overrides: object) -> SimEnvironment:
    """Like :func:`_mock_env` but with the placeholder URI SimRunner accepts."""
    overrides.setdefault(
        "vla",
        VLASpec(
            id="zero",
            weights_uri="placeholder",
            extra={"action_dim": 7},
        ),
    )
    return _mock_env(**overrides)


def test_builtin_registries_populated() -> None:
    """Importing openral_sim registers built-in scenes and policies."""
    assert "mock" in SCENES
    assert "libero_spatial" in SCENES
    assert "metaworld" in SCENES
    assert "zero" in POLICIES
    assert "random" in POLICIES
    assert "smolvla" in POLICIES


def test_registry_unknown_id_raises_with_known_list() -> None:
    reg: _Registry[int] = _Registry("widget")

    @reg.register("a")
    def _factory_a() -> int:
        return 1

    with pytest.raises(ROSConfigError) as excinfo:
        reg.get("typo")
    msg = str(excinfo.value)
    assert "widget" in msg and "typo" in msg and "['a']" in msg


def test_registry_duplicate_registration_rejected() -> None:
    reg: _Registry[int] = _Registry("widget")
    reg.register("a")(lambda: 1)
    with pytest.raises(ROSConfigError):
        reg.register("a")(lambda: 2)


def test_make_env_and_make_policy_roundtrip_mock() -> None:
    env_cfg = _mock_env()
    sim = make_env(env_cfg)
    pol = make_policy(env_cfg)

    obs = sim.reset(seed=0)
    assert "images" in obs and "state" in obs
    action = pol.step(obs, "x")
    assert action.shape == (7,)
    assert float(action.sum()) == 0.0  # zero policy

    result = sim.step(action)
    assert result.terminated is False  # success_step=3, this is step 1
    assert "is_success" in result.info


def test_sim_runner_mock_loop() -> None:
    """:class:`SimRunner` rolls out the mock scene end-to-end against zero policy."""
    env_cfg = _runnable_env()
    runner = SimRunner(env_cfg)
    runner.activate()
    try:
        runner.run(max_ticks=env_cfg.task.max_steps + 1)
    finally:
        runner.deactivate()
    assert len(runner.episode_results) == 1
    result = runner.episode_results[0]
    assert result.success is True
    assert result.steps == 3
    assert result.total_reward == 1.0
    assert result.latency_budget_ms is None  # placeholder URI → no manifest
    assert result.budget_violations == 0


def test_sim_runner_random_policy_does_not_crash() -> None:
    env_cfg = _runnable_env(
        vla=VLASpec(
            id="random",
            weights_uri="placeholder",
            extra={"action_dim": 7, "seed": 1},
        ),
    )
    runner = SimRunner(env_cfg)
    runner.activate()
    try:
        runner.run(max_ticks=env_cfg.task.max_steps + 1)
    finally:
        runner.deactivate()
    assert runner.episode_results[0].steps == 3


def test_render_resolution_bumps_to_rskill_camera_minimum() -> None:
    """A scene rendered below an rSkill's camera ``min_width``/``min_height`` is
    bumped to meet it — so act-libero (trained at 256) runs on a 224-rendered
    scene instead of being rejected by the sensor gate."""
    from pathlib import Path

    from openral_rskill.loader import load_rskill_manifest
    from openral_sim.sim_runner import _required_render_resolution, _with_render_resolution

    repo = Path(__file__).resolve().parents[2]
    act = repo / "rskills" / "act-libero"
    if not act.exists():
        pytest.skip("act-libero rskill not present")
    manifest = load_rskill_manifest(str(act))
    env_cfg = _mock_env(
        scene=SceneSpec(
            id="mock",
            backend=PhysicsBackend.MOCK,
            backend_options={"success_step": 3, "action_dim": 7},
            observation_width=224,
            observation_height=224,
        )
    )
    # act-libero: min_width/min_height 256 -> a 224 scene bumps to 256x256.
    assert _required_render_resolution(manifest, env_cfg) == (256, 256)
    bumped = _with_render_resolution(env_cfg, 256, 256)
    assert (bumped.scene.observation_width, bumped.scene.observation_height) == (256, 256)


def test_render_resolution_no_bump_when_scene_meets_minimum() -> None:
    """A scene that already meets the rSkill's minimums is left untouched — no
    out-of-distribution upscale (smolvla-libero trains at 256)."""
    from pathlib import Path

    from openral_rskill.loader import load_rskill_manifest
    from openral_sim.sim_runner import _required_render_resolution

    repo = Path(__file__).resolve().parents[2]
    smolvla = repo / "rskills" / "smolvla-libero"
    if not smolvla.exists():
        pytest.skip("smolvla-libero rskill not present")
    manifest = load_rskill_manifest(str(smolvla))
    env_cfg = _mock_env(
        scene=SceneSpec(
            id="mock",
            backend=PhysicsBackend.MOCK,
            backend_options={"success_step": 3, "action_dim": 7},
            observation_width=256,
            observation_height=256,
        )
    )
    assert _required_render_resolution(manifest, env_cfg) == (256, 256)


def test_check_rskill_compatibility_rejects_hf_uri() -> None:
    env_cfg = _mock_env(vla=VLASpec(id="zero", weights_uri="hf://x"))
    with pytest.raises(ROSConfigError, match="hf://"):
        _check_rskill_compatibility(env_cfg)


def test_check_rskill_compatibility_rejects_unregistered_robot() -> None:
    env_cfg = _mock_env(
        robot_id="nonexistent_robot",
        vla=VLASpec(id="zero", weights_uri="rskills/smolvla-libero"),
    )
    with pytest.raises(ROSConfigError, match="not registered"):
        _check_rskill_compatibility(env_cfg)


def test_sim_list_subcommand_prints_example_configs() -> None:
    """`openral sim list` prints every ``scenes/**/*.yaml`` as paste-able paths."""
    runner = CliRunner()
    result = runner.invoke(sim_app, ["list"])
    assert result.exit_code == 0, result.output
    assert "scenes/" in result.output
    # ADR-0041: scenes/benchmarks/diffusion_pusht.yaml was renamed to
    # scenes/benchmark/pusht.yaml (rSkill name stripped; tier-only directory).
    assert "scenes/benchmark/pusht.yaml" in result.output


def test_cli_rejects_hf_uri(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI rejects ``--rskill`` values that carry an ``hf://`` scheme.

    The free-flag ``--scene/--vla`` form is gone (see
    ``feat(core,sim): SceneEnvironment + openral sim run --rskill, no legacy``).
    Pass bare names (``rskills/<name>``) or bare HF repo IDs instead.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    # ADR-0041: scenes/benchmarks/smolvla_libero_spatial.yaml + pi05_libero_spatial.yaml
    # were merged into scenes/benchmark/libero_spatial.yaml (BenchmarkScene, rejected
    # by `openral sim run` on the tier guard). This test needs a SimScene-tier fixture
    # so the --rskill hf:// guard fires next; scenes/sim/franka_libero_pnp.yaml is
    # the SimScene-tier custom-BDDL LIBERO route.
    cfg = repo_root / "scenes" / "sim" / "franka_libero_pnp.yaml"
    if not cfg.exists():
        pytest.skip(f"{cfg} missing")

    rc = cli_main(
        [
            "--config",
            str(cfg),
            "--rskill",
            "hf://lerobot/smolvla_libero",
            "--max-steps",
            "5",
            "--n-episodes",
            "1",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "rskill" in err.lower()


def test_cli_missing_args_returns_error() -> None:
    """``openral sim run`` with no --config / --rskill fails with a typed error."""
    rc = cli_main([])
    assert rc == 1
