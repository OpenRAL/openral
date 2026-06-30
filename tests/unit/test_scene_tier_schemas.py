"""Tests for TaskSpec optional fields and DeployScene/SimScene/BenchmarkScene tiers.

TaskSpec ``max_steps``/``success_key`` are optional (added in Task 1).
DeployScene/SimScene/BenchmarkScene are the three scene tiers (added in Task 3).
"""

import pytest
from pydantic import ValidationError


# TaskSpec tests (run first — they have no model imports yet)
def test_taskspec_max_steps_optional():
    from openral_core import TaskSpec

    t = TaskSpec(id="libero_spatial/0", scene_id="libero_spatial")
    assert t.max_steps is None
    assert t.success_key is None


def test_taskspec_max_steps_set():
    from openral_core import TaskSpec

    t = TaskSpec(
        id="libero_spatial/0",
        scene_id="libero_spatial",
        max_steps=200,
        success_key="is_success",
    )
    assert t.max_steps == 200
    assert t.success_key == "is_success"


def test_taskspec_rejects_zero_max_steps():
    from openral_core import TaskSpec

    with pytest.raises(ValidationError):
        TaskSpec(id="libero_spatial/0", scene_id="libero_spatial", max_steps=0)


def test_taskspec_rejects_negative_max_steps():
    from openral_core import TaskSpec

    with pytest.raises(ValidationError):
        TaskSpec(id="libero_spatial/0", scene_id="libero_spatial", max_steps=-1)


# ── DeployScene ──────────────────────────────────────────────────────────


def test_deploy_scene_minimal():
    from openral_core import DeployScene, SceneSpec

    s = SceneSpec(id="libero_spatial", backend="mujoco")
    d = DeployScene(scene=s)
    assert d.robot_id is None
    assert d.base_pose is None
    assert d.composition is None


def test_deploy_scene_rejects_tasks_field():
    """ADR-0073 amendment — deploy goals come from the operator prompt, not the scene."""
    import pytest
    from openral_core import DeployScene
    from pydantic import ValidationError

    base = {"scene": {"id": "libero_spatial", "backend": "mujoco"}}
    DeployScene.model_validate(base)  # valid without tasks
    with pytest.raises(ValidationError):
        DeployScene.model_validate({**base, "tasks": ["pick the milk"]})


def test_deploy_scene_composition_round_trips():
    # ADR-0066 — a deploy scene declares its own MJCF composition (its arena),
    # so the robot manifest doesn't have to carry scene config.
    from openral_core import DeployScene, SceneComposition, SceneSpec

    d = DeployScene(
        scene=SceneSpec(id="openarm_tabletop_pnp", backend="mujoco"),
        robot_id="openarm",
        composition=SceneComposition(composer="pkg.mod:compose", params={"top_camera_fovy": 65.0}),
    )
    assert d.composition is not None
    assert d.composition.composer == "pkg.mod:compose"
    # survives a JSON round-trip (the wire format deploy_sim forwards to the node)
    again = DeployScene.model_validate_json(d.model_dump_json())
    assert again.composition == d.composition


def test_openarm_deploy_scene_owns_composition_robot_manifest_does_not():
    # ADR-0066 separation: the openarm tabletop arena lives on the scene; the
    # robot manifest describes only the robot (no scene_defaults).
    from openral_core import DeployScene, RobotDescription

    scene = DeployScene.from_yaml("scenes/deploy/openarm_tabletop.yaml")
    assert scene.composition is not None
    assert scene.composition.composer.endswith("compose_openarm_tabletop_mjcf")
    robot = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    assert robot.scene_defaults is None


def test_deploy_scene_rejects_task():
    from openral_core import DeployScene

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DeployScene.model_validate(
            {
                "scene": {"id": "libero_spatial", "backend": "mujoco"},
                "task": {"id": "libero_spatial/0", "scene_id": "libero_spatial"},
            }
        )


def test_deploy_scene_rejects_vla_block():
    from openral_core import DeployScene

    with pytest.raises(Exception, match="vla:"):
        DeployScene.model_validate(
            {
                "scene": {"id": "libero_spatial", "backend": "mujoco"},
                "vla": {},
            }
        )


# ── SimScene ─────────────────────────────────────────────────────────────


def test_sim_scene_requires_task():
    from openral_core import SceneSpec, SimScene

    with pytest.raises(ValidationError):
        SimScene(scene=SceneSpec(id="libero_spatial", backend="mujoco"))


def test_sim_scene_scene_id_mismatch_rejected():
    from openral_core import SimScene

    with pytest.raises(ValidationError, match="scene_id"):
        SimScene.model_validate(
            {
                "scene": {"id": "libero_spatial", "backend": "mujoco"},
                "task": {"id": "metaworld/push", "scene_id": "metaworld"},
            }
        )


def test_sim_scene_accepts_no_max_steps():
    from openral_core import SimScene

    s = SimScene.model_validate(
        {
            "scene": {"id": "libero_spatial", "backend": "mujoco"},
            "task": {"id": "libero_spatial/0", "scene_id": "libero_spatial"},
        }
    )
    assert s.task.max_steps is None
    assert s.task.success_key is None
    assert s.n_episodes == 1
    assert s.seed == 0


def test_sim_scene_accepts_benchmark_fields():
    """Extra benchmark fields (n_episodes, seed, metadata) are accepted by SimScene."""
    from openral_core import SimScene

    raw = {
        "scene": {"id": "libero_spatial", "backend": "mujoco"},
        "task": {
            "id": "libero_spatial/0",
            "scene_id": "libero_spatial",
            "success_key": "is_success",
            "max_steps": 100,
        },
        "n_episodes": 500,
        "seed": 42,
        "metadata": {"paper": "https://arxiv.org/abs/2309.11500", "honest_scope": "task 0"},
    }
    s = SimScene.model_validate(raw)
    assert s.n_episodes == 500


# ── BenchmarkScene ───────────────────────────────────────────────────────


def test_benchmark_scene_requires_n_episodes():
    from openral_core import BenchmarkScene

    with pytest.raises(ValidationError, match="n_episodes"):
        BenchmarkScene.model_validate(
            {
                "scene": {"id": "libero_spatial", "backend": "mujoco"},
                "task": {
                    "id": "libero_spatial/0",
                    "scene_id": "libero_spatial",
                    "success_key": "is_success",
                    "max_steps": 100,
                },
                "seed": 0,
                "metadata": {"paper": "https://arxiv.org/abs/2309.11500", "honest_scope": "x"},
            }
        )


def test_benchmark_scene_requires_task_success_key():
    from openral_core import BenchmarkScene

    with pytest.raises(ValidationError, match="success_key"):
        BenchmarkScene.model_validate(
            {
                "scene": {"id": "libero_spatial", "backend": "mujoco"},
                "task": {"id": "libero_spatial/0", "scene_id": "libero_spatial", "max_steps": 100},
                "n_episodes": 10,
                "seed": 0,
                "metadata": {"paper": "https://arxiv.org/abs/2309.11500", "honest_scope": "x"},
            }
        )


def test_benchmark_scene_requires_task_max_steps():
    from openral_core import BenchmarkScene

    with pytest.raises(ValidationError, match="max_steps"):
        BenchmarkScene.model_validate(
            {
                "scene": {"id": "libero_spatial", "backend": "mujoco"},
                "task": {
                    "id": "libero_spatial/0",
                    "scene_id": "libero_spatial",
                    "success_key": "is_success",
                },
                "n_episodes": 10,
                "seed": 0,
                "metadata": {"paper": "https://arxiv.org/abs/2309.11500", "honest_scope": "x"},
            }
        )


def test_benchmark_scene_valid():
    from openral_core import BenchmarkScene

    b = BenchmarkScene.model_validate(
        {
            "scene": {
                "id": "libero_spatial",
                "backend": "mujoco",
                "observation_height": 256,
                "observation_width": 256,
            },
            "task": {
                "id": "libero_spatial/0",
                "scene_id": "libero_spatial",
                "instruction": "",
                "success_key": "is_success",
                "max_steps": 100,
            },
            "n_episodes": 500,
            "seed": 42,
            "metadata": {
                "paper": "https://arxiv.org/abs/2309.11500",
                "honest_scope": "Task 0 of LIBERO-Spatial.",
            },
        }
    )
    assert b.n_episodes == 500
    assert b.metadata.paper == "https://arxiv.org/abs/2309.11500"
