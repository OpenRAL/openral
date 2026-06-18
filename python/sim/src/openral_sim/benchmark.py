r"""Benchmark runner — loop a ``list[BenchmarkScene]`` and emit a :class:`RSkillEvalResult`.

ADR-0009 (PR D) introduced the runner; ADR-0042 (June 2026) deleted the
``BenchmarkSpec`` wrapper class so a benchmark suite is now a bare
``list[BenchmarkScene]`` on disk (``benchmarks/<suite_id>.yaml``) and in
memory. The suite id is the filename stem. The runner is the **only**
way to produce a ``rskills/<vla>/eval/<suite_id>.json`` with
``reproduced_locally=true``; hand-edited JSONs continue to be valid but
they carry ``reproduced_locally=false`` and a ``reproduction_cli`` that
points back at ``openral benchmark run`` so users can close the loop
locally.

The runner is intentionally a thin layer over :class:`SimRunner`:

* For every (``scene``, ``seed``) tuple in
  ``scenes × range(seed, seed + n_episodes)``, build a one-episode
  :class:`SimEnvironment` and drive it with a fresh :class:`SimRunner`
  (ADR-0010 amendment 1: sim and hardware share the same
  :class:`InferenceRunner` Protocol). Each :class:`BenchmarkScene`
  carries its own ``robot_id``, ``task``, ``n_episodes``, ``seed``, and
  paper provenance (suite-level invariants — uniformity of robot_id /
  n_episodes / seed / metadata, unique task ids, non-empty — are
  validated by :func:`openral_core.raise_on_invalid_suite`).
* Aggregate per-task success rates + the overall average into a
  :class:`RSkillEvalResult`. The numbers are computed inside
  :func:`_aggregate_results` so callers (CLI + tests) share the same
  rolled-up shape.

Example::

    from pathlib import Path
    from openral_core import VLASpec, load_benchmark_suite, raise_on_invalid_suite
    from openral_sim.benchmark import run_benchmark

    path = Path("benchmarks/libero_spatial.yaml")
    scenes = load_benchmark_suite(str(path))
    suite_id = path.stem  # "libero_spatial"
    raise_on_invalid_suite(scenes, suite_id=suite_id)
    vla = VLASpec(id="smolvla", weights_uri="rskills/smolvla-libero")
    result, episodes = run_benchmark(scenes, suite_id=suite_id, vla=vla)
    print(result.results["avg_success_rate"])
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from openral_core import (
        BenchmarkScene,
        RSkillEvalResult,
        RSkillManifest,
        VLASpec,
    )

    from openral_sim.rollout import EpisodeResult


_log = structlog.get_logger(__name__)


def _task_matches(task_id: str, scene_id: str, declared: list[str]) -> bool:
    """Whether ``task_id`` / ``scene_id`` is covered by a ``declared`` entry.

    An entry matches as an exact task id, a ``"<scene>/<...>"`` family prefix
    (so ``"libero_spatial"`` covers ``"libero_spatial/0".."/9"``), or the bare
    ``scene.id``.
    """
    for entry in declared:
        e = entry.rstrip("/")
        if task_id == e or task_id.startswith(f"{e}/") or scene_id == e:
            return True
    return False


def check_benchmark_task_compatibility(
    manifest: RSkillManifest,
    *,
    task_id: str,
    scene_id: str,
) -> None:
    """Gate a benchmark scene's task against the rSkill's ``evaluated_tasks``.

    Complements the embodiment/sensor gate (``rSkill.check_compatibility``) with
    the missing *task-data* axis: a checkpoint trained for one task (e.g.
    LiftCube) must not be silently run on a different benchmark task (PickCube),
    where it produces a sensible-looking rollout that can never satisfy success.

    Args:
        manifest: The rSkill manifest being dispatched.
        task_id: The benchmark scene's ``task.id`` (e.g. ``maniskill3/PickCube-v1``).
        scene_id: The benchmark scene's ``scene.id`` (e.g. ``maniskill3``).

    Raises:
        ROSCapabilityMismatch: ``manifest.evaluated_tasks`` is non-empty and
            none of its entries cover ``task_id`` / ``scene_id``.
    """
    declared = manifest.evaluated_tasks
    if not declared:
        _log.warning(
            "rskill_task_compat_undeclared",
            skill=manifest.name,
            task=task_id,
            scene=scene_id,
        )
        return
    if _task_matches(task_id, scene_id, declared):
        _log.info("rskill_task_compat_ok", skill=manifest.name, task=task_id, declared=declared)
        return
    from openral_core.exceptions import ROSCapabilityMismatch

    raise ROSCapabilityMismatch(
        f"rSkill {manifest.name!r} declares evaluated_tasks={declared}, which do "
        f"not cover this benchmark's task {task_id!r} (scene {scene_id!r}). The "
        "checkpoint was trained/validated for a different task; running it here "
        "yields a plausible-looking rollout that cannot succeed. Pair the scene "
        "with a task-matched rSkill, or add the task to the manifest's "
        "evaluated_tasks if the checkpoint genuinely covers it."
    )


def run_benchmark(
    scenes: list[BenchmarkScene],
    *,
    suite_id: str,
    vla: VLASpec,
    device: str | None = None,
    save_dir: str | None = None,
) -> tuple[RSkillEvalResult, list[EpisodeResult]]:
    """Run a benchmark suite end-to-end against one rSkill.

    ADR-0042: a benchmark suite is a bare ``list[BenchmarkScene]`` plus a
    ``suite_id`` (typically ``Path("benchmarks/<id>.yaml").stem``). Callers
    that load from disk should use :func:`openral_core.load_benchmark_suite`
    and :func:`openral_core.raise_on_invalid_suite` before calling this
    function; the runner does not re-validate suite invariants.

    Args:
        scenes: The list of :class:`BenchmarkScene`s to evaluate. Each
            entry carries its own scene, task, robot, episode count, and
            seed offset. Suite-level invariants (uniformity of robot_id /
            n_episodes / seed / metadata, unique task ids, non-empty) are
            assumed pre-validated.
        suite_id: Stable suite identifier (the YAML filename stem). Used
            in log lines and embedded into ``reproduction_cli``.
        vla: The single free axis — which rSkill to evaluate. Its
            ``weights_uri`` must be a bare rSkill reference (name, path, or
            HF repo ID); the strict runner rejects raw ``hf://`` URIs.
        device: Optional torch device override applied to every rollout
            (``"cpu"``, ``"cuda:0"``, ``"mps"``, ``"auto"``). ``None``
            keeps the manifest's preferred device.
        save_dir: Optional directory written to each :class:`SimEnvironment`
            for adapter-side artefacts (videos, traces). Unrelated to where
            the final :class:`RSkillEvalResult` JSON lives — that path is
            chosen by the caller (see :func:`default_output_path`).

    Returns:
        A pair ``(result, episodes)`` where ``result`` is a validated
        :class:`RSkillEvalResult` ready to be written to
        ``rskills/<vla>/eval/<suite_id>.json`` and ``episodes`` is the
        flat per-(task, seed) list of :class:`EpisodeResult` objects for
        callers that want fine-grained data (e.g. unit tests asserting on
        latency).

    Raises:
        openral_core.exceptions.ROSConfigError: Any error propagated
            from :class:`SimRunner` — typically a missing rSkill
            manifest, an incompatible robot, or an unresolvable rSkill
            reference. The runner does not catch them; the whole suite
            fails so partial JSONs never reach disk.
    """
    from openral_core import SimEnvironment

    from openral_sim.sim_runner import SimRunner

    # Suite invariants (see openral_core.raise_on_invalid_suite) guarantee
    # every BenchmarkScene shares robot_id / n_episodes / seed / metadata.
    # Read from scenes[0] for the seed sweep; pull per-scene robot_id /
    # task / max_steps from each scene inside the loop.
    first = scenes[0]
    seeds = list(range(first.seed, first.seed + first.n_episodes))

    per_task: dict[str, list[bool]] = {}
    all_episodes: list[EpisodeResult] = []

    for scene in scenes:
        task_id = scene.task.id
        # raise_on_invalid_suite asserts robot_id is not None for every
        # scene; BenchmarkScene._require_task_eval_fields asserts
        # task.success_key and task.max_steps are set. Re-narrow for mypy.
        robot_id = scene.robot_id
        max_steps = scene.task.max_steps
        assert robot_id is not None
        assert max_steps is not None
        per_task[task_id] = []
        for seed in seeds:
            vla_for_episode = vla.model_copy(update={"device": device}) if device else vla
            env_cfg = SimEnvironment(
                robot_id=robot_id,
                scene=scene.scene,
                task=scene.task,
                vla=vla_for_episode,
                base_pose=scene.base_pose,
                seed=seed,
                n_episodes=1,
                save_dir=save_dir,
            )
            runner = SimRunner(env_cfg, view=False, strict_view=False)
            try:
                runner.activate()
                runner.run(max_ticks=max_steps + 1)
                episodes = runner.episode_results
            finally:
                runner.deactivate()
            episode = episodes[0]
            per_task[task_id].append(episode.success)
            all_episodes.append(episode)
            _log.info(
                "benchmark_episode_done",
                benchmark=suite_id,
                task=task_id,
                seed=seed,
                success=episode.success,
                steps=episode.steps,
            )

    result = _aggregate_results(
        scenes,
        suite_id=suite_id,
        vla=vla,
        per_task=per_task,
        episodes=all_episodes,
    )
    return result, all_episodes


def _aggregate_results(
    scenes: list[BenchmarkScene],
    *,
    suite_id: str,
    vla: VLASpec,
    per_task: dict[str, list[bool]],
    episodes: list[EpisodeResult],
) -> RSkillEvalResult:
    """Roll up per-task booleans into a :class:`RSkillEvalResult`.

    Pulled out of :func:`run_benchmark` so the same aggregation shape is
    exercised by unit tests that build synthetic per-task lists without
    paying for a full rollout.

    ADR-0042: paper-comparison display strings (``display_name``,
    ``simulator``) and the optional arxiv URL all flow from the per-scene
    :class:`BenchmarkMetadata` block (suite-uniform by invariant). The
    ``RSkillEvalBenchmark.name`` falls back to ``suite_id`` and
    ``RSkillEvalBenchmark.simulator`` falls back to ``scenes[0].scene.id``
    when the optional metadata fields are not set; the arxiv URL is
    auto-derived from the paper field when the URL contains
    ``arxiv.org/`` (mirrors :func:`_aggregate_scene_results`).
    """
    from openral_core import (
        RSkillEvalBenchmark,
        RSkillEvalResult,
        RSkillEvalSource,
    )

    per_task_rate = {tid: (sum(r) / len(r) if r else 0.0) for tid, r in per_task.items()}
    avg = (sum(per_task_rate.values()) / len(per_task_rate)) if per_task_rate else 0.0

    # Suite invariants (see openral_core.raise_on_invalid_suite): every
    # scene shares robot_id / n_episodes / seed / metadata. Read the
    # representative scene from scenes[0]; for ``max_steps`` take the
    # per-task maximum so the suite-level protocol summary describes the
    # worst-case bound rather than being biased by which task happens to
    # be ``scenes[0]`` (matches the pre-Task-10 behaviour where
    # ``protocol.max_steps`` carried the suite-wide ceiling).
    first = scenes[0]
    robot_id = first.robot_id
    success_key = first.task.success_key
    assert robot_id is not None  # raise_on_invalid_suite rejects None
    assert success_key is not None  # BenchmarkScene validator rejects None
    # BenchmarkScene._require_task_eval_fields guarantees task.max_steps
    # is non-None on every scene; the generator expression below is safe.
    max_steps = max(scene.task.max_steps for scene in scenes if scene.task.max_steps is not None)
    seeds = list(range(first.seed, first.seed + first.n_episodes))

    results: dict[str, object] = {
        **{f"{tid}_success_rate": rate for tid, rate in per_task_rate.items()},
        "avg_success_rate": avg,
        "n_tasks": len(scenes),
        "n_episodes_per_task": first.n_episodes,
        "n_episodes_total": sum(len(r) for r in per_task.values()),
    }
    if episodes:
        latencies = [e.mean_step_latency_ms for e in episodes if e.mean_step_latency_ms is not None]
        if latencies:
            results["mean_step_latency_ms_avg"] = sum(latencies) / len(latencies)
        # Diffusion Policy on PushT reports per-rollout max coverage (the
        # closest the T-block got to its goal pose) and averages across
        # seeds — ``avg_success_rate`` on this scene is a different
        # metric from the paper. Emit both so the eval JSON is directly
        # comparable. The reward signal on gym_pusht *is* the coverage
        # IoU at each step (see ``EpisodeResult.max_step_reward`` and the
        # diffusion-pusht eval JSON's protocol notes).
        if first.scene.id == "pusht":
            results["mean_coverage_iou"] = sum(e.max_step_reward for e in episodes) / len(episodes)

    # ADR-0042: per-scene metadata is the single source of truth. ``paper``
    # is the canonical per-scene provenance (required by
    # :class:`BenchmarkMetadata`); the arxiv URL is auto-derived from it
    # when the URL contains ``arxiv.org/`` — matches the per-scene
    # :func:`_aggregate_scene_results` and means a suite eval JSON has
    # the same ``source.arxiv`` policy as a single-scene eval JSON.
    paper = first.metadata.paper
    arxiv = paper if "arxiv.org/" in paper else None

    return RSkillEvalResult(
        schema_version="0.1",
        source=RSkillEvalSource(
            paper=paper or "n/a",
            arxiv=arxiv,
            model_variant=vla.id,
            evaluated_by="OpenRAL:openral benchmark run",
            reproduced_locally=True,
            reproduction_cli=(
                f"openral benchmark run --suite {suite_id} --rskill {vla.weights_uri}"
            ),
            status="reproduced",
        ),
        benchmark=RSkillEvalBenchmark(
            name=first.metadata.display_name or suite_id,
            protocol=(
                f"{first.n_episodes} episodes per task, "
                f"success_key={success_key}, "
                f"max_steps={max_steps}"
            ),
            robot=robot_id,
            simulator=first.metadata.simulator or first.scene.id,
        ),
        eval_config={
            "n_episodes": first.n_episodes,
            "seeds": seeds,
            "success_key": success_key,
            "max_steps": max_steps,
            "vla_id": vla.id,
            "weights_uri": vla.weights_uri,
        },
        results=results,
    )


def run_benchmark_scene(
    scene: BenchmarkScene,
    vla: VLASpec,
    *,
    device: str | None = None,
    save_dir: str | None = None,
    config_path: str | None = None,
    view: bool | None = None,
    record_video: bool = False,
) -> tuple[RSkillEvalResult, list[EpisodeResult]]:
    """Run a :class:`BenchmarkScene` end-to-end against one rSkill.

    Single-scene counterpart of :func:`run_benchmark` — the rollout loop
    that backs ``openral benchmark scene`` (sibling of ``openral benchmark
    run --suite``). Iterates ``range(scene.seed, scene.seed + scene.n_episodes)``
    against the one ``(scene, task)`` pair carried by the
    :class:`BenchmarkScene` and emits the same :class:`RSkillEvalResult`
    JSON shape so paper-comparison reports stay uniform.

    Args:
        scene: The published-protocol benchmark scene — exactly one task
            and a scalar ``seed`` (the seed list is derived as
            ``[seed, seed+1, …, seed+n_episodes-1]``).
        vla: The single free axis — which rSkill to evaluate. ``weights_uri``
            must be a bare rSkill reference; the strict runner rejects
            raw ``hf://`` URIs.
        device: Optional torch device override applied to every rollout.
        save_dir: Optional directory written to each :class:`SimEnvironment`
            for adapter-side artefacts (videos, traces).
        config_path: Optional path to the BenchmarkScene YAML this scene
            was loaded from — embedded into ``RSkillEvalSource.reproduction_cli``
            so reviewers can re-run the exact eval from disk.
        record_video: When True, capture per-step world frames into each
            :class:`EpisodeResult.frames` so callers can write clean
            website MP4s (``openral benchmark scene --save-video``). Off by
            default — eval/CI runs stay allocation-light.
        view: Tri-state viewer flag, identical in meaning to ``openral sim
            run --view/--no-view``. ``None`` (default) keeps the headless
            behaviour benchmark rollouts have always had — eval artefacts
            and CI/deploy runs are unaffected. ``True`` opens a passive
            ``mujoco.viewer`` window per episode and streams the rollout
            (strict — raises on a missing display); ``False`` forces
            offscreen. Resolved through :func:`openral_sim.cli._resolve_view`
            so the ``MUJOCO_GL`` / ``DISPLAY`` semantics match ``sim run``.

    Returns:
        A pair ``(result, episodes)`` where ``result`` is a validated
        :class:`RSkillEvalResult` ready to be written to
        ``rskills/<vla>/eval/scene_<scene_id>.json``.

    Raises:
        openral_core.exceptions.ROSConfigError: When ``scene.robot_id`` is
            ``None`` (the runner cannot construct a :class:`SimEnvironment`
            without an embodiment), or for any error propagated from
            :class:`SimRunner`.
    """
    from openral_core import SimEnvironment
    from openral_core.exceptions import ROSConfigError

    from openral_sim.sim_runner import SimRunner

    if scene.robot_id is None:
        raise ROSConfigError(
            f"BenchmarkScene scene.id={scene.scene.id!r} has no robot_id; "
            "add `robot_id: <id>` to the YAML (e.g. franka_panda for LIBERO, "
            "aloha_bimanual for gym-aloha, pusht_2d for gym-pusht). "
            "The benchmark runner cannot construct a SimEnvironment without "
            "an embodiment."
        )
    # BenchmarkScene._require_task_eval_fields guarantees these are set,
    # but mypy --strict still needs the local narrowing.
    success_key = scene.task.success_key
    max_steps = scene.task.max_steps
    assert success_key is not None
    assert max_steps is not None

    # Task-data gate (ADR-0060): refuse a checkpoint trained for a different
    # task than this benchmark evaluates (e.g. a LiftCube policy on PickCube).
    # Fail fast — before the rollout. Skipped for the built-in mock policies
    # (no manifest) and hf:// URIs (rejected by the sim runner anyway).
    # Permissive when the manifest declares no evaluated_tasks (legacy rSkills
    # warn, don't break).
    from openral_sim.sim_runner import _MOCK_PLACEHOLDER_URI, _MOCK_POLICY_IDS

    is_mock = vla.id in _MOCK_POLICY_IDS and vla.weights_uri == _MOCK_PLACEHOLDER_URI
    if not is_mock and not vla.weights_uri.startswith("hf://"):
        from openral_rskill.loader import load_rskill_manifest

        check_benchmark_task_compatibility(
            load_rskill_manifest(vla.weights_uri),
            task_id=scene.task.id,
            scene_id=scene.scene.id,
        )

    # Default (``view is None``) preserves the headless behaviour benchmark
    # rollouts have always had — eval artefacts and CI/deploy runs must be
    # unaffected. Only an explicit ``--view``/``--no-view`` engages the
    # shared ``sim run`` resolver (display/MUJOCO_GL handling, strict mode).
    if view is None:
        view_flag, strict_view = False, False
    else:
        from openral_sim.cli import _resolve_view

        view_flag, strict_view = _resolve_view(view)

    seeds = list(range(scene.seed, scene.seed + scene.n_episodes))
    successes: list[bool] = []
    all_episodes: list[EpisodeResult] = []

    for seed in seeds:
        vla_for_episode = vla.model_copy(update={"device": device}) if device else vla
        env_cfg = SimEnvironment(
            robot_id=scene.robot_id,
            scene=scene.scene,
            task=scene.task,
            vla=vla_for_episode,
            base_pose=scene.base_pose,
            seed=seed,
            n_episodes=1,
            save_dir=save_dir,
            record_video=record_video,
        )
        runner = SimRunner(env_cfg, view=view_flag, strict_view=strict_view)
        try:
            runner.activate()
            runner.run(max_ticks=max_steps + 1)
            episodes = runner.episode_results
        finally:
            runner.deactivate()
        episode = episodes[0]
        successes.append(episode.success)
        all_episodes.append(episode)
        _log.info(
            "benchmark_scene_episode_done",
            scene=scene.scene.id,
            task=scene.task.id,
            seed=seed,
            success=episode.success,
            steps=episode.steps,
        )

    result = _aggregate_scene_results(scene, vla, successes, all_episodes, config_path)
    return result, all_episodes


def _aggregate_scene_results(
    scene: BenchmarkScene,
    vla: VLASpec,
    successes: list[bool],
    episodes: list[EpisodeResult],
    config_path: str | None,
) -> RSkillEvalResult:
    """Roll up a per-episode list into a :class:`RSkillEvalResult`.

    Single-scene counterpart of :func:`_aggregate_results`. Shares the
    output schema so ``openral benchmark report`` does not need to
    distinguish ``run --suite`` JSONs from ``scene --config`` JSONs.
    """
    from openral_core import (
        RSkillEvalBenchmark,
        RSkillEvalResult,
        RSkillEvalSource,
    )

    # BenchmarkScene-level invariants asserted in run_benchmark_scene().
    assert scene.robot_id is not None
    success_key = scene.task.success_key
    max_steps = scene.task.max_steps
    assert success_key is not None
    assert max_steps is not None

    task_id = scene.task.id
    per_task_rate = (sum(successes) / len(successes)) if successes else 0.0
    seeds = list(range(scene.seed, scene.seed + scene.n_episodes))

    results: dict[str, object] = {
        f"{task_id}_success_rate": per_task_rate,
        "avg_success_rate": per_task_rate,
        "n_tasks": 1,
        "n_episodes_per_task": scene.n_episodes,
        "n_episodes_total": len(successes),
    }
    if episodes:
        latencies = [e.mean_step_latency_ms for e in episodes if e.mean_step_latency_ms is not None]
        if latencies:
            results["mean_step_latency_ms_avg"] = sum(latencies) / len(latencies)
        # PushT special-case — same handling as :func:`_aggregate_results`,
        # since the same scene id can ship as either a SuiteSpec or a
        # BenchmarkScene.
        if scene.scene.id == "pusht":
            results["mean_coverage_iou"] = sum(e.max_step_reward for e in episodes) / len(episodes)

    # Surface the arxiv URL separately when the paper field happens to be
    # one — keeps the eval JSON forward-compatible with rSkill catalogue
    # filters that index on arxiv id.
    paper = scene.metadata.paper
    arxiv = paper if "arxiv.org/" in paper else None

    config_ref = (
        config_path if config_path is not None else f"scenes/benchmark/{scene.scene.id}.yaml"
    )

    return RSkillEvalResult(
        schema_version="0.1",
        source=RSkillEvalSource(
            paper=paper,
            arxiv=arxiv,
            model_variant=vla.id,
            evaluated_by="OpenRAL:openral benchmark scene",
            reproduced_locally=True,
            reproduction_cli=(
                f"openral benchmark scene --config {config_ref} --rskill {vla.weights_uri}"
            ),
            status="reproduced",
        ),
        benchmark=RSkillEvalBenchmark(
            name=scene.scene.id,
            protocol=(
                f"{scene.n_episodes} episodes, seed={scene.seed}, "
                f"success_key={success_key}, max_steps={max_steps}"
            ),
            robot=scene.robot_id,
            simulator=scene.scene.id,
        ),
        eval_config={
            "n_episodes": scene.n_episodes,
            "seeds": seeds,
            "success_key": success_key,
            "max_steps": max_steps,
            "vla_id": vla.id,
            "weights_uri": vla.weights_uri,
        },
        results=results,
    )


def default_output_path(weights_uri: str, benchmark_id: str) -> str:
    """Compute the canonical eval-JSON path from a bare rSkill reference.

    A weights ref like ``rskills/smolvla-libero`` resolves to
    ``rskills/smolvla-libero/eval/<benchmark_id>.json``. The caller is
    responsible for ``mkdir -p`` of the parent directory and for actually
    writing the JSON (typically via :meth:`RSkillEvalResult.model_dump_json`).

    Args:
        weights_uri: The bare rSkill reference from :attr:`VLASpec.weights_uri`.
        benchmark_id: The benchmark suite id (the YAML filename stem) — used
            as the JSON filename stem so re-running the same suite
            overwrites the same eval JSON.

    Returns:
        A relative path string in the form ``<skill_dir>/eval/<id>.json``.

    Raises:
        ValueError: If ``weights_uri`` starts with ``hf://`` (cannot write
            to the HF Hub; only locally-resolvable rSkills are writable).
    """
    if weights_uri.startswith("hf://"):
        raise ValueError(
            f"default_output_path expects a bare rSkill ref (e.g. rskills/<name>); "
            f"got {weights_uri!r}. The benchmark runner only writes JSONs for "
            "locally-resolvable rSkills."
        )
    return f"{weights_uri}/eval/{benchmark_id}.json"


# ── rskill.yaml writeback ─────────────────────────────────────────────────────

# Top-level `benchmarks:` key at column 0. We match it on its own line, then
# replace everything up to (but not including) the next top-level key. This
# is intentionally a *surgical* edit so the comments + ordering of every
# other field in rskill.yaml are preserved verbatim — yaml.safe_dump would
# round-trip the data but drop every comment and reorder keys.
_BENCHMARKS_BLOCK_RE = re.compile(
    r"^benchmarks:[^\n]*\n(?:[ \t][^\n]*\n)*",
    re.MULTILINE,
)


def update_rskill_benchmarks(
    skill_dir: str | Path,
    benchmark_id: str,
    score: float,
) -> Path:
    """Persist a benchmark headline rate back into a skill's ``rskill.yaml``.

    The benchmark runner is the only canonical producer of the headline
    rates that land in :attr:`RSkillManifest.benchmarks`; this helper closes
    the loop so ``openral benchmark run`` finalisation actually updates the
    manifest field that downstream tools (``openral benchmark report``,
    skill_catalog, the hosted dashboard) read from.

    The on-disk edit is a *surgical* replacement of the top-level
    ``benchmarks:`` block — every other line in ``rskill.yaml`` (comments,
    ordering, blank lines) is left untouched. The merged manifest is then
    re-validated through :class:`RSkillManifest` so an unknown
    ``benchmark_id`` (one not in the :data:`BenchmarkName` literal) or an
    out-of-range ``score`` fails loud before the bytes hit disk.

    Args:
        skill_dir: Path to the skill directory (the parent of
            ``rskill.yaml``). Accepts either ``Path`` or string forms.
        benchmark_id: The benchmark suite id (the YAML filename stem) whose
            headline rate to record. MUST be a member of the
            :data:`openral_core.BenchmarkName` literal — the manifest
            schema rejects everything else.
        score: Headline success rate in ``[0.0, 1.0]``. Typically the
            ``avg_success_rate`` from :class:`RSkillEvalResult.results`.

    Returns:
        The :class:`pathlib.Path` of the ``rskill.yaml`` that was written.

    Raises:
        FileNotFoundError: If ``<skill_dir>/rskill.yaml`` does not exist.
        openral_core.exceptions.ROSConfigError: If the merged manifest
            fails :class:`RSkillManifest` validation (unknown
            ``benchmark_id``, score out of range, malformed YAML).
    """
    import yaml
    from openral_core import RSkillManifest
    from openral_core.exceptions import ROSConfigError

    skill_path = Path(skill_dir)
    manifest_path = skill_path / "rskill.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"rskill.yaml not found at {manifest_path}; cannot update benchmarks."
        )

    text = manifest_path.read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ROSConfigError(f"{manifest_path} did not parse to a mapping")

    benchmarks = dict(raw.get("benchmarks") or {})
    benchmarks[benchmark_id] = float(score)
    raw["benchmarks"] = benchmarks

    # Re-validate — raises a pydantic ValidationError (wrapped) if the
    # benchmark_id is not a BenchmarkName literal or the score is out of
    # range. We deliberately validate *before* writing so a bad call does
    # not corrupt the manifest.
    try:
        RSkillManifest.model_validate(raw)
    except Exception as exc:  # pragma: no cover — surfaced as ROSConfigError
        raise ROSConfigError(
            f"refusing to write {manifest_path}: merged manifest failed validation: {exc}"
        ) from exc

    # Render the new benchmarks block. Sort keys so re-running with the
    # same benchmark on different days yields a stable diff.
    if benchmarks:
        rendered_lines = ["benchmarks:\n"]
        for key in sorted(benchmarks):
            rendered_lines.append(f"  {key}: {float(benchmarks[key])}\n")
        rendered = "".join(rendered_lines)
    else:  # pragma: no cover — unreachable, we just added a key
        rendered = "benchmarks: {}\n"

    match = _BENCHMARKS_BLOCK_RE.search(text)
    if match is not None:
        new_text = text[: match.start()] + rendered + text[match.end() :]
    else:
        # No `benchmarks:` key in the manifest yet — append a separator + block
        # at the end of the file. This branch is reachable for legacy
        # manifests authored before the field was added.
        sep = "" if text.endswith("\n") else "\n"
        new_text = f"{text}{sep}\n{rendered}"

    manifest_path.write_text(new_text)
    _log.info(
        "rskill_manifest_benchmarks_updated",
        manifest=str(manifest_path),
        benchmark=benchmark_id,
        score=float(score),
    )
    return manifest_path


def update_rskill_benchmarks_from_uri(
    weights_uri: str,
    benchmark_id: str,
    score: float,
) -> Path:
    """Convenience wrapper that delegates to :func:`update_rskill_benchmarks`.

    Callers (the ``openral benchmark run`` finaliser, future
    ``openral benchmark report --update`` paths) can pass the same
    ``VLASpec.weights_uri`` they already hold.

    Raises:
        ValueError: If ``weights_uri`` starts with ``hf://`` (cannot write
            to the HF Hub; only locally-resolvable rSkills have a writable
            manifest).
    """
    if weights_uri.startswith("hf://"):
        raise ValueError(
            f"update_rskill_benchmarks_from_uri expects a bare rSkill ref; "
            f"got {weights_uri!r}. Only locally-resolvable rSkills carry a "
            "writable manifest."
        )
    return update_rskill_benchmarks(weights_uri, benchmark_id, score)
