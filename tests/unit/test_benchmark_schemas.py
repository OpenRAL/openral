"""Unit tests for ProtocolSpec and the bare-list benchmark suite shape (ADR-0042).

ADR-0042 (June 2026) deleted the ``BenchmarkSpec`` wrapper class. A benchmark
suite is now a bare ``list[BenchmarkScene]`` on disk + a ``suite_id`` derived
from the YAML filename stem. Suite-level invariants moved out of
``BenchmarkSpec.model_post_init`` into the free function
:func:`openral_core.raise_on_invalid_suite`, which raises
:class:`ROSConfigError` rather than ``pydantic.ValidationError`` so the
suite id can be embedded in the error message.

``ProtocolSpec`` is retained as a standalone schema for ADR / report tooling
(it never moved into ``BenchmarkScene``); its own construction / validation
tests still apply unchanged.

The catalogue-fixture parametric test exercises every YAML under
``benchmarks/`` via :func:`openral_core.load_benchmark_suite` (CLAUDE.md §1.11
— real fixtures, no mocks).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from openral_core import (
    BenchmarkMetadata,
    BenchmarkScene,
    PhysicsBackend,
    ProtocolSpec,
    SceneSpec,
    TaskSpec,
    load_benchmark_suite,
    raise_on_invalid_suite,
)
from openral_core.exceptions import ROSConfigError
from pydantic import ValidationError

# ── ProtocolSpec ──────────────────────────────────────────────────────────────


def test_protocol_spec_defaults() -> None:
    p = ProtocolSpec()
    assert p.n_episodes == 10
    assert p.seeds == list(range(10))
    assert p.success_key == "is_success"
    assert p.max_steps == 280
    assert p.min_reps is None


def test_protocol_spec_seeds_must_cover_n_episodes() -> None:
    """seeds list shorter than n_episodes is rejected — reproducibility hinge."""
    with pytest.raises(ValidationError, match="seeds has 3 entries"):
        ProtocolSpec(n_episodes=10, seeds=[0, 1, 2])


def test_protocol_spec_seeds_may_exceed_n_episodes() -> None:
    """Extra seeds are allowed; the runner will slice ``seeds[:n_episodes]``."""
    p = ProtocolSpec(n_episodes=5, seeds=list(range(50)))
    assert p.n_episodes == 5
    assert len(p.seeds) == 50


def test_protocol_spec_min_reps_capped_at_n_episodes() -> None:
    with pytest.raises(ValidationError, match=r"min_reps \(10\) exceeds"):
        ProtocolSpec(n_episodes=5, seeds=list(range(5)), min_reps=10)


def test_protocol_spec_n_episodes_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ProtocolSpec(n_episodes=0, seeds=[])


# ── BenchmarkScene + suite helpers (ADR-0042) ─────────────────────────────────


_LIBERO_META = BenchmarkMetadata(
    paper="https://arxiv.org/abs/2306.03310",
    honest_scope="10 episodes per task on LIBERO-Spatial.",
    display_name="LIBERO-Spatial",
    simulator="LIBERO (MuJoCo)",
)


def _libero_scene(
    i: int,
    *,
    scene_id: str = "libero_spatial",
    robot_id: str | None = "franka_panda",
    n_episodes: int = 10,
    seed: int = 0,
    metadata: BenchmarkMetadata = _LIBERO_META,
) -> BenchmarkScene:
    return BenchmarkScene(
        scene=SceneSpec(id=scene_id, backend=PhysicsBackend.MUJOCO),
        task=TaskSpec(
            id=f"{scene_id}/{i}",
            scene_id=scene_id,
            instruction="",
            max_steps=280,
            success_key="is_success",
        ),
        robot_id=robot_id,
        n_episodes=n_episodes,
        seed=seed,
        metadata=metadata,
    )


def _libero_scenes(n: int = 10, **scene_kwargs: object) -> list[BenchmarkScene]:
    return [_libero_scene(i, **scene_kwargs) for i in range(n)]  # type: ignore[arg-type]


# ── BenchmarkMetadata — ADR-0042 display fields ───────────────────────────────


def test_benchmark_metadata_display_fields_default_none() -> None:
    """``display_name`` / ``simulator`` are optional — old YAMLs omit them."""
    m = BenchmarkMetadata(
        paper="https://arxiv.org/abs/2306.03310",
        honest_scope="any scope",
    )
    assert m.display_name is None
    assert m.simulator is None


def test_benchmark_metadata_display_fields_round_trip() -> None:
    m = BenchmarkMetadata(
        paper="https://arxiv.org/abs/2306.03310",
        honest_scope="any scope",
        display_name="LIBERO-Spatial",
        simulator="LIBERO (MuJoCo)",
    )
    rebuilt = BenchmarkMetadata.model_validate(m.model_dump(mode="json"))
    assert rebuilt == m


# ── raise_on_invalid_suite — suite-level invariants ───────────────────────────


def test_raise_on_invalid_suite_happy_path() -> None:
    scenes = _libero_scenes()
    # No exception → invariants hold.
    raise_on_invalid_suite(scenes, suite_id="libero_spatial")


def test_raise_on_invalid_suite_rejects_empty_list() -> None:
    with pytest.raises(ROSConfigError, match="scene list is empty"):
        raise_on_invalid_suite([], suite_id="libero_spatial")


def test_raise_on_invalid_suite_embeds_suite_id_in_message() -> None:
    """Every error message embeds ``suite_id`` so failures point at the YAML."""
    with pytest.raises(ROSConfigError, match="'libero_spatial'"):
        raise_on_invalid_suite([], suite_id="libero_spatial")


def test_raise_on_invalid_suite_rejects_duplicate_task_ids() -> None:
    dupes = [_libero_scene(0), _libero_scene(0)]
    with pytest.raises(ROSConfigError, match="duplicate task id"):
        raise_on_invalid_suite(dupes, suite_id="libero_spatial")


def test_raise_on_invalid_suite_rejects_first_scene_missing_robot_id() -> None:
    """The leading scene must pin an embodiment — benchmark suites are embodiment-locked."""
    with pytest.raises(ROSConfigError, match="robot_id=None"):
        raise_on_invalid_suite(
            [_libero_scene(0, robot_id=None), _libero_scene(1)],
            suite_id="libero_spatial",
        )


def test_raise_on_invalid_suite_rejects_mixed_robot_ids() -> None:
    """Every scene in a suite must share one robot — the suite IS embodiment-locked."""
    mixed = [_libero_scene(0), _libero_scene(1, robot_id="aloha_bimanual")]
    with pytest.raises(ROSConfigError, match="every scene in a suite must run on"):
        raise_on_invalid_suite(mixed, suite_id="libero_spatial")


def test_raise_on_invalid_suite_rejects_mixed_n_episodes() -> None:
    mixed = [_libero_scene(0, n_episodes=10), _libero_scene(1, n_episodes=5)]
    with pytest.raises(ROSConfigError, match="must share the same episode count"):
        raise_on_invalid_suite(mixed, suite_id="libero_spatial")


def test_raise_on_invalid_suite_rejects_mixed_seeds() -> None:
    mixed = [_libero_scene(0, seed=0), _libero_scene(1, seed=42)]
    with pytest.raises(ROSConfigError, match="must share the same seed offset"):
        raise_on_invalid_suite(mixed, suite_id="libero_spatial")


def test_raise_on_invalid_suite_rejects_mixed_metadata() -> None:
    other_meta = BenchmarkMetadata(
        paper="https://arxiv.org/abs/9999.99999",
        honest_scope="different scope",
    )
    mixed = [_libero_scene(0), _libero_scene(1, metadata=other_meta)]
    with pytest.raises(ROSConfigError, match="same provenance block"):
        raise_on_invalid_suite(mixed, suite_id="libero_spatial")


def test_raise_on_invalid_suite_allows_mixed_max_steps_per_task() -> None:
    """``task.max_steps`` MAY differ — e.g. MS3 pick (100) + stack (200)."""
    a = _libero_scene(0)
    b = _libero_scene(1)
    b = b.model_copy(update={"task": b.task.model_copy(update={"max_steps": 500})})
    # No exception — per-task budgets are allowed to differ.
    raise_on_invalid_suite([a, b], suite_id="libero_spatial")


def test_raise_on_invalid_suite_allows_mixed_success_key_per_task() -> None:
    """``task.success_key`` MAY differ across tasks (no uniformity invariant)."""
    a = _libero_scene(0)
    b = _libero_scene(1)
    b = b.model_copy(update={"task": b.task.model_copy(update={"success_key": "success"})})
    raise_on_invalid_suite([a, b], suite_id="libero_spatial")


# ── load_benchmark_suite — YAML loader (ADR-0042) ─────────────────────────────


_VALID_YAML = textwrap.dedent(
    """\
    - scene:
        id: libero_spatial
        backend: mujoco
      task:
        id: libero_spatial/0
        scene_id: libero_spatial
        instruction: ""
        max_steps: 280
        success_key: is_success
      robot_id: franka_panda
      n_episodes: 2
      seed: 0
      metadata:
        paper: "https://arxiv.org/abs/2306.03310"
        honest_scope: "2 episodes for the round-trip test."
        display_name: "LIBERO-Spatial (tiny)"
        simulator: "LIBERO (MuJoCo)"
    - scene:
        id: libero_spatial
        backend: mujoco
      task:
        id: libero_spatial/1
        scene_id: libero_spatial
        instruction: ""
        max_steps: 280
        success_key: is_success
      robot_id: franka_panda
      n_episodes: 2
      seed: 0
      metadata:
        paper: "https://arxiv.org/abs/2306.03310"
        honest_scope: "2 episodes for the round-trip test."
        display_name: "LIBERO-Spatial (tiny)"
        simulator: "LIBERO (MuJoCo)"
    """
)


def test_load_benchmark_suite_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "tiny.yaml"
    p.write_text(_VALID_YAML)
    scenes = load_benchmark_suite(str(p))
    assert len(scenes) == 2
    assert all(s.robot_id == "franka_panda" for s in scenes)
    assert scenes[0].metadata.display_name == "LIBERO-Spatial (tiny)"
    assert scenes[0].metadata.simulator == "LIBERO (MuJoCo)"
    # ``load_benchmark_suite`` does NOT enforce suite invariants on its own —
    # callers compose it with ``raise_on_invalid_suite`` explicitly.
    raise_on_invalid_suite(scenes, suite_id=p.stem)


def test_load_benchmark_suite_rejects_legacy_dict_root(tmp_path: Path) -> None:
    """Pre-ADR-0042 ``{id, tasks, metadata}`` wrapper now errors with a redirect."""
    legacy = textwrap.dedent(
        """\
        id: tiny
        metadata: {}
        tasks:
          - scene: {id: libero_spatial, backend: mujoco}
            task:
              id: libero_spatial/0
              scene_id: libero_spatial
              instruction: ""
              max_steps: 280
              success_key: is_success
            robot_id: franka_panda
            n_episodes: 2
            seed: 0
            metadata: {paper: "p", honest_scope: "s"}
        """
    )
    p = tmp_path / "legacy.yaml"
    p.write_text(legacy)
    with pytest.raises(ROSConfigError, match="ADR-0042"):
        load_benchmark_suite(str(p))


def test_load_benchmark_suite_rejects_non_list_non_dict_root(tmp_path: Path) -> None:
    p = tmp_path / "string.yaml"
    p.write_text("just a bare string\n")
    with pytest.raises(ROSConfigError, match="YAML root must be a list"):
        load_benchmark_suite(str(p))


def test_load_benchmark_suite_rejects_invalid_scene_entry(tmp_path: Path) -> None:
    """A list whose entry fails BenchmarkScene validation raises with ``[i]`` context."""
    p = tmp_path / "bad.yaml"
    p.write_text(
        textwrap.dedent(
            """\
            - scene:
                id: libero_spatial
                backend: mujoco
              task:
                id: libero_spatial/0
                scene_id: libero_spatial
                instruction: ""
                max_steps: 280
                success_key: is_success
              robot_id: franka_panda
              # n_episodes missing — BenchmarkScene requires it.
              seed: 0
              metadata: {paper: "p", honest_scope: "s"}
            """
        )
    )
    with pytest.raises(ROSConfigError, match=r"entry \[0\] is not a valid BenchmarkScene"):
        load_benchmark_suite(str(p))


def test_load_benchmark_suite_rejects_non_mapping_entry(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just a string\n")
    with pytest.raises(ROSConfigError, match=r"entry \[0\] must be a mapping"):
        load_benchmark_suite(str(p))


# ── Real fixtures under benchmarks/ ───────────────────────────────────────────


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"


# (fixture stem, expected robot_id, expected scene_id, expected n_tasks,
#  expected n_episodes, expected success_key, expected display_name,
#  expected simulator)
_CATALOGUE_FIXTURES: list[tuple[str, str, str, int, int, str, str, str]] = [
    (
        "libero_spatial",
        "franka_panda",
        "libero_spatial",
        10,
        50,
        "is_success",
        "LIBERO-Spatial",
        "LIBERO (MuJoCo)",
    ),
    (
        "libero_object",
        "franka_panda",
        "libero_object",
        10,
        50,
        "is_success",
        "LIBERO-Object",
        "LIBERO (MuJoCo)",
    ),
    (
        "libero_goal",
        "franka_panda",
        "libero_goal",
        10,
        50,
        "is_success",
        "LIBERO-Goal",
        "LIBERO (MuJoCo)",
    ),
    (
        "libero_10",
        "franka_panda",
        "libero_10",
        10,
        50,
        "is_success",
        "LIBERO-Long",
        "LIBERO (MuJoCo)",
    ),
    (
        "metaworld_mt10",
        "sawyer",
        "metaworld",
        10,
        10,
        "success",
        "MetaWorld MT10",
        "MetaWorld v3 (MuJoCo via lerobot)",
    ),
    (
        "metaworld_mt50",
        "sawyer",
        "metaworld",
        50,
        5,
        "success",
        "MetaWorld MT50",
        "MetaWorld v3 (MuJoCo via lerobot)",
    ),
    (
        # Unified suite (was aloha_transfer_cube.yaml + aloha_insertion.yaml);
        # scenes[0] is the transfer-cube task, so scene_id asserts on that.
        "aloha",
        "aloha_bimanual",
        "aloha_transfer_cube",
        2,
        50,
        "is_success",
        "ALOHA bimanual (gym-aloha)",
        "gym-aloha (MuJoCo)",
    ),
    (
        "pusht",
        "pusht_2d",
        "pusht",
        1,
        50,
        "is_success",
        "PushT (gym-pusht)",
        "gym-pusht (pymunk 2-D)",
    ),
    (
        "maniskill3_panda",
        "franka_panda",
        "maniskill3",
        7,
        100,
        "is_success",
        "ManiSkill3 — Panda tabletop",
        "ManiSkill3 (SAPIEN)",
    ),
    (
        "robocasa_pnp",
        "panda_mobile",
        "robocasa/PickPlaceCounterToCabinet",
        1,
        10,
        "is_success",
        "RoboCasa Atomic PickPlace (counter→cabinet)",
        "RoboCasa 1.0.1 (robosuite + MuJoCo)",
    ),
    (
        "gr1_tabletop",
        "gr1",
        "robocasa/gr1/PnPCupToDrawerClose",
        1,
        10,
        "is_success",
        "RoboCasa GR1 Tabletop Tasks (PnPCupToDrawerClose)",
        "robocasa-gr1-tabletop-tasks 0.2.0 (robosuite + MuJoCo)",
    ),
    (
        "simpler_env_widowx",
        "widowx",
        "simpler_env",
        4,
        5,
        "success",
        "SimplerEnv (WidowX / Bridge V2)",
        "SimplerEnv (SAPIEN via ManiSkill)",
    ),
    (
        "robotwin",
        "aloha_agilex",
        "robotwin",
        5,
        100,
        "is_success",
        "RoboTwin 2.0 — dual-arm (aloha-agilex)",
        "RoboTwin 2.0 (SAPIEN), aloha-agilex",
    ),
]


@pytest.mark.parametrize(
    (
        "stem",
        "robot_id",
        "scene_id",
        "n_tasks",
        "n_episodes",
        "success_key",
        "display_name",
        "simulator",
    ),
    _CATALOGUE_FIXTURES,
    ids=[t[0] for t in _CATALOGUE_FIXTURES],
)
def test_benchmarks_catalogue_fixture_loads_and_passes_invariants(
    stem: str,
    robot_id: str,
    scene_id: str,
    n_tasks: int,
    n_episodes: int,
    success_key: str,
    display_name: str,
    simulator: str,
) -> None:
    """Every YAML under benchmarks/ loads, passes suite invariants, and matches its catalogue row.

    CLAUDE.md §1.11 — real fixtures. Each catalogue YAML is exercised by
    a parametrised case so a typo / drift in any one of them fails loud
    and points at the offending file.

    Asserts on the per-scene fields: suite invariants (see
    :func:`raise_on_invalid_suite`) guarantee uniformity of ``robot_id`` /
    ``n_episodes`` / ``metadata`` across the list; ``success_key`` /
    ``max_steps`` MAY differ per-task by validator contract but every
    shipped suite is uniform on ``success_key``.
    """
    path = _BENCHMARKS_DIR / f"{stem}.yaml"
    if not path.exists():
        pytest.skip(f"benchmarks/{stem}.yaml fixture not present in this checkout")
    scenes = load_benchmark_suite(str(path))
    raise_on_invalid_suite(scenes, suite_id=stem)

    assert len(scenes) == n_tasks
    first = scenes[0]
    assert first.robot_id == robot_id
    assert first.scene.id == scene_id
    assert first.n_episodes == n_episodes
    assert first.task.success_key == success_key
    assert first.metadata.display_name == display_name
    assert first.metadata.simulator == simulator
    # Suite invariants guarantee uniformity; cheap set-comprehension cross-check.
    assert {s.robot_id for s in scenes} == {robot_id}
    assert {s.n_episodes for s in scenes} == {n_episodes}
    # ``scene_id`` documents the leading scene; a suite MAY carry heterogeneous
    # per-scene scene_ids (e.g. aloha bundles AlohaTransferCube-v0 +
    # AlohaInsertion-v0), which raise_on_invalid_suite permits — so the
    # leading scene's task must reference its own scene, no suite-wide equality.
    assert first.task.scene_id == scene_id
    assert len({s.task.id for s in scenes}) == n_tasks  # ids unique
