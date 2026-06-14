"""Strict YAML loaders for the three scene tiers.

Each scene-driven CLI (``openral deploy sim``, ``openral sim run``,
``openral benchmark scene``) must accept exactly one tier of scene YAML:

* ``DeployScene`` — environment + optional robot only, no task.
* ``SimScene`` — scene + task, eval fields optional (defaults filled).
* ``BenchmarkScene`` — scene + task + ``n_episodes`` / ``seed`` / ``metadata``.

Because :class:`SimScene` extends :class:`DeployScene` and :class:`BenchmarkScene`
extends :class:`SimScene`, a YAML one tier *too rich* still validates against the
expected tier silently — that would let a benchmark eval spec slip into
``openral sim run`` and lose its episode count.  This module centralises the
rejection logic so every CLI gets the same redirect message.
"""

from __future__ import annotations

from pathlib import Path
from typing import overload

import yaml as _yaml
from pydantic import ValidationError

from openral_core.exceptions import ROSConfigError
from openral_core.schemas import BenchmarkScene, DeployScene, SimScene

__all__ = [
    "load_benchmark_suite",
    "load_scene_strict",
    "raise_on_invalid_suite",
]


@overload
def load_scene_strict(path: str, expected: type[BenchmarkScene]) -> BenchmarkScene: ...
@overload
def load_scene_strict(path: str, expected: type[SimScene]) -> SimScene: ...
@overload
def load_scene_strict(path: str, expected: type[DeployScene]) -> DeployScene: ...
def load_scene_strict(
    path: str,
    expected: type[DeployScene] | type[SimScene] | type[BenchmarkScene],
) -> DeployScene | SimScene | BenchmarkScene:
    """Load ``path`` as exactly ``expected``; reject other tiers with a clear error.

    Args:
        path: Filesystem path to a YAML file containing a single scene mapping.
        expected: One of ``DeployScene``, ``SimScene``, ``BenchmarkScene`` —
            the exact tier the calling CLI accepts.

    Returns:
        A validated instance of ``expected``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ROSConfigError: If the YAML root is not a mapping, or if it loads as
            a tier other than ``expected`` (with a redirect message naming the
            right CLI command).

    Example:
        >>> from openral_core import SimScene, load_scene_strict
        >>> scene = load_scene_strict("scenes/sim/tabletop_cube_push.yaml", SimScene)
        >>> scene.scene.id  # scene_id is the @SCENES.register("…") key, not the filename
        'tabletop_push'
    """
    raw_obj = _yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_obj, dict):
        raise ROSConfigError(f"{path}: YAML root must be a mapping, got {type(raw_obj).__name__}")
    raw: dict[str, object] = raw_obj

    if expected is DeployScene:
        return _load_as_deploy(path, raw)
    if expected is SimScene:
        return _load_as_sim(path, raw)
    if expected is BenchmarkScene:
        return _load_as_benchmark(path, raw)
    raise ROSConfigError(
        f"load_scene_strict: unsupported expected type {expected!r} — "
        "must be DeployScene, SimScene, or BenchmarkScene."
    )


# ── Per-tier loaders ───────────────────────────────────────────────────────


def _load_as_deploy(path: str, raw: dict[str, object]) -> DeployScene:
    if "task" in raw:
        raise ROSConfigError(
            f"{path}: this YAML has a 'task:' block, so it is a SimScene or "
            "BenchmarkScene, not a DeployScene. `openral deploy sim` accepts "
            "DeployScene only (scene + optional robot, no task). "
            f"Use `openral sim run --config {path}` instead, or move the YAML "
            "to `scenes/deploy/` and drop the 'task:' block."
        )
    try:
        return DeployScene.model_validate(raw)
    except ValidationError as exc:
        raise ROSConfigError(f"{path}: not a valid DeployScene: {exc}") from exc


def _load_as_sim(path: str, raw: dict[str, object]) -> SimScene:
    # A YAML that fully validates as BenchmarkScene would also validate as
    # SimScene (since BenchmarkScene ⊂ SimScene), so we must catch it first
    # and redirect — otherwise the eval contract (n_episodes, metadata) would
    # be silently ignored by the sim runner.
    try:
        BenchmarkScene.model_validate(raw)
    except ValidationError:
        pass  # Not a BenchmarkScene; fall through to SimScene path.
    else:
        raise ROSConfigError(
            f"{path}: this YAML is a BenchmarkScene "
            "(has n_episodes, seed, and metadata). `openral sim run` accepts "
            "SimScene only. Use `openral benchmark scene --config "
            f"{path}` to run the full eval, or override the episode budget "
            f"with `openral benchmark scene --config {path} --n-episodes 1` "
            "for a smoke run."
        )

    if "task" not in raw:
        raise ROSConfigError(
            f"{path}: this YAML has no 'task:' block. `openral sim run` "
            "requires a SimScene (scene + task). "
            f"Use `openral deploy sim --config {path}` for an env-only scene."
        )

    try:
        return SimScene.model_validate(raw)
    except ValidationError as exc:
        raise ROSConfigError(f"{path}: not a valid SimScene: {exc}") from exc


def _load_as_benchmark(path: str, raw: dict[str, object]) -> BenchmarkScene:
    try:
        return BenchmarkScene.model_validate(raw)
    except ValidationError as exc:
        raise ROSConfigError(
            f"{path}: not a valid BenchmarkScene: {exc}\n"
            "BenchmarkScene requires `n_episodes`, `seed`, `metadata.paper`, "
            "`metadata.honest_scope`, `task.max_steps`, and `task.success_key`."
        ) from exc


# ── Benchmark suite loader (ADR-0042) ──────────────────────────────────────


def load_benchmark_suite(path: str) -> list[BenchmarkScene]:
    """Load a bare list of :class:`BenchmarkScene`s from ``benchmarks/<id>.yaml``.

    ADR-0042 (June 2026) deleted the ``BenchmarkSpec`` wrapper class. A
    benchmark suite YAML is now a bare YAML list at the root; the suite id
    is derived from the filename stem (e.g. ``benchmarks/libero_spatial.yaml``
    has suite id ``"libero_spatial"``). Pre-ADR-0042 the YAML root was a
    ``{id, tasks, metadata}`` mapping wrapping the scenes — this loader
    rejects that shape with an explicit redirect message.

    Per-scene Pydantic validation runs here. Suite-level invariants
    (uniformity, uniqueness, non-empty) are NOT enforced; call
    :func:`raise_on_invalid_suite` separately with the suite id of your
    choice (typically ``Path(path).stem``). This split lets tests
    construct invalid in-memory suites without touching the filesystem.

    Args:
        path: Filesystem path to a benchmark YAML file.

    Returns:
        Validated list of :class:`BenchmarkScene`s in YAML order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ROSConfigError: If the YAML root is not a list (legacy dict-shape
            gets an explicit ADR-0042 redirect), or any entry fails
            :class:`BenchmarkScene` validation.

    Example:
        >>> from pathlib import Path
        >>> from openral_core import load_benchmark_suite, raise_on_invalid_suite
        >>> scenes = load_benchmark_suite("benchmarks/libero_spatial.yaml")
        >>> suite_id = Path("benchmarks/libero_spatial.yaml").stem
        >>> raise_on_invalid_suite(scenes, suite_id=suite_id)
        >>> len(scenes)
        10
    """
    raw_obj = _yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    if isinstance(raw_obj, dict):
        raise ROSConfigError(
            f"{path}: YAML root is a mapping, but ADR-0042 (June 2026) "
            "deleted the BenchmarkSpec wrapper. A benchmark suite YAML is "
            "now a bare list of BenchmarkScene mappings at the root; the "
            "suite id is derived from the filename stem. Remove the "
            "top-level `id:` / `metadata:` block and inline the scenes "
            "directly, promoting `suite` / `simulator` from the suite-level "
            "metadata to per-scene `metadata.display_name` / "
            "`metadata.simulator`."
        )

    if not isinstance(raw_obj, list):
        raise ROSConfigError(
            f"{path}: YAML root must be a list of BenchmarkScene mappings "
            f"(ADR-0042), got {type(raw_obj).__name__}."
        )

    scenes: list[BenchmarkScene] = []
    for i, raw_scene in enumerate(raw_obj):
        if not isinstance(raw_scene, dict):
            raise ROSConfigError(
                f"{path}: entry [{i}] must be a mapping, got {type(raw_scene).__name__}."
            )
        try:
            scenes.append(BenchmarkScene.model_validate(raw_scene))
        except ValidationError as exc:
            raise ROSConfigError(
                f"{path}: entry [{i}] is not a valid BenchmarkScene: {exc}\n"
                "BenchmarkScene requires `n_episodes`, `seed`, "
                "`metadata.paper`, `metadata.honest_scope`, `task.max_steps`, "
                "and `task.success_key`."
            ) from exc
    return scenes


def raise_on_invalid_suite(
    scenes: list[BenchmarkScene],
    *,
    suite_id: str,
) -> None:
    """Raise :class:`ROSConfigError` if ``scenes`` violates suite-level invariants.

    Originally enforced inside ``BenchmarkSpec.model_post_init`` (deleted in
    ADR-0042). Extracted as a free function so callers can validate freshly
    loaded suites independently and so tests can exercise the rules without
    touching the filesystem.

    Suite-level invariants:
        * ``scenes`` MUST be non-empty.
        * Every ``scenes[i].task.id`` MUST be unique within the suite.
        * Every ``scenes[i].robot_id`` MUST be non-``None`` — benchmark
          suites always pin an embodiment.
        * All ``scenes[i].robot_id`` / ``n_episodes`` / ``seed`` /
          ``metadata`` MUST be equal across the list — a suite is a single
          fixed protocol applied across (potentially varying) scenes / tasks.

    Per-scene ``task.success_key`` / ``task.max_steps`` MAY differ (e.g.
    ManiSkill3 pick + stack share one suite but each task has its own step
    budget).

    Args:
        scenes: The list of :class:`BenchmarkScene`s to validate.
        suite_id: The suite identifier — embedded in every error message so
            failures point back to the right ``benchmarks/<id>.yaml`` file.

    Raises:
        ROSConfigError: If any invariant is violated. The first violation
            wins (no batched reporting).
    """
    if not scenes:
        raise ROSConfigError(
            f"benchmark suite {suite_id!r}: scene list is empty; a benchmark "
            f"must declare at least one BenchmarkScene."
        )

    first = scenes[0]
    if first.robot_id is None:
        raise ROSConfigError(
            f"benchmark suite {suite_id!r}: scene {first.task.id!r} has "
            f"robot_id=None; benchmark suites must pin an embodiment "
            f"(add `robot_id: <id>` to the BenchmarkScene YAML)."
        )

    seen: set[str] = set()
    for scene in scenes:
        task_id = scene.task.id
        if task_id in seen:
            raise ROSConfigError(
                f"benchmark suite {suite_id!r}: duplicate task id "
                f"{task_id!r}; task ids must be unique within a suite."
            )
        seen.add(task_id)

        if scene.robot_id is None:
            raise ROSConfigError(
                f"benchmark suite {suite_id!r}: scene {task_id!r} has "
                f"robot_id=None; benchmark suites must pin an embodiment "
                f"(add `robot_id: <id>` to the BenchmarkScene YAML)."
            )
        if scene.robot_id != first.robot_id:
            raise ROSConfigError(
                f"benchmark suite {suite_id!r}: scene {task_id!r} has "
                f"robot_id={scene.robot_id!r} but the suite uses "
                f"{first.robot_id!r}; every scene in a suite must run on "
                f"the same robot."
            )
        if scene.n_episodes != first.n_episodes:
            raise ROSConfigError(
                f"benchmark suite {suite_id!r}: scene {task_id!r} has "
                f"n_episodes={scene.n_episodes} but the suite uses "
                f"{first.n_episodes}; every scene in a suite must share "
                f"the same episode count."
            )
        if scene.seed != first.seed:
            raise ROSConfigError(
                f"benchmark suite {suite_id!r}: scene {task_id!r} has "
                f"seed={scene.seed} but the suite uses {first.seed}; "
                f"every scene in a suite must share the same seed offset."
            )
        if scene.metadata != first.metadata:
            raise ROSConfigError(
                f"benchmark suite {suite_id!r}: scene {task_id!r} has "
                f"metadata={scene.metadata!r} but the suite uses "
                f"{first.metadata!r}; every scene in a suite shares the "
                f"same provenance block (paper, honest_scope, optional "
                f"display_name + simulator)."
            )
