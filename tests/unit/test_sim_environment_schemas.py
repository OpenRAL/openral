"""Unit tests for SceneSpec / TaskSpec / VLASpec / SimEnvironment.

These cover the new contracts added for the eval/sim layer (ADR-0002):
construction, defaults, cross-field validation, YAML round-trip.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from openral_core import (
    PhysicsBackend,
    SceneSpec,
    SimEnvironment,
    TaskSpec,
    VLASpec,
)
from pydantic import ValidationError


def _ok_env(**overrides: object) -> SimEnvironment:
    """Helper: build a valid SimEnvironment, optionally with field overrides."""
    base: dict[str, object] = {
        "robot_id": "so100_follower",
        "scene": SceneSpec(id="mock", backend=PhysicsBackend.MOCK),
        "task": TaskSpec(id="mock/0", scene_id="mock", instruction="x"),
        "vla": VLASpec(id="zero", weights_uri="mock://noop"),
    }
    base.update(overrides)
    return SimEnvironment(**base)  # type: ignore[arg-type]


def test_scene_spec_defaults() -> None:
    s = SceneSpec(id="libero_spatial")
    assert s.backend == PhysicsBackend.MUJOCO
    assert s.observation_height == 256
    assert s.observation_width == 256
    assert s.cameras == []
    assert s.assets_uri is None


def test_task_spec_defaults() -> None:
    t = TaskSpec(id="suite/0", scene_id="suite")
    assert t.max_steps is None
    assert t.success_key is None
    assert t.instruction == ""


def test_vla_spec_defaults() -> None:
    v = VLASpec(id="smolvla", weights_uri="hf://lerobot/smolvla_libero")
    assert v.device == "auto"
    assert v.runtime is None
    assert v.deterministic is False


def test_sim_environment_cross_field_validation() -> None:
    with pytest.raises(ValidationError) as excinfo:
        SimEnvironment(
            robot_id="r",
            scene=SceneSpec(id="A"),
            task=TaskSpec(id="B/0", scene_id="B", instruction=""),
            vla=VLASpec(id="zero", weights_uri="mock://noop"),
        )
    assert "scene_id" in str(excinfo.value)


def test_sim_environment_extra_forbid() -> None:
    """Top-level extra keys are rejected — protects against typos in YAML configs."""
    with pytest.raises(ValidationError):
        SimEnvironment.model_validate(
            {
                "robot_id": "r",
                "scene": {"id": "mock", "backend": "mock"},
                "task": {"id": "mock/0", "scene_id": "mock"},
                "vla": {"id": "zero", "weights_uri": "x"},
                "garbage_typo": True,
            }
        )


def test_sim_environment_n_episodes_validation() -> None:
    with pytest.raises(ValidationError):
        _ok_env(n_episodes=0)


def test_scene_environment_yaml_round_trip(tmp_path: Path) -> None:
    """``SimScene.from_yaml`` is the scene+task YAML entrypoint (post-ADR-0041).

    ``SimEnvironment.from_yaml`` was removed in the
    ``feat(core,sim): SceneEnvironment + openral sim run --rskill, no legacy``
    commit; YAMLs now carry scene + task only, and the policy is supplied
    via ``--rskill`` on the CLI which composes the runtime
    :class:`SimEnvironment`. ADR-0041 renamed ``SceneEnvironment`` to the
    three-tier ``DeployScene`` / ``SimScene`` / ``BenchmarkScene`` hierarchy.
    """
    from openral_core import SimScene

    yaml_text = textwrap.dedent(
        """\
        robot_id: franka_panda
        scene:
          id: libero_spatial
          backend: mujoco
          observation_height: 256
          observation_width: 256
        task:
          id: libero_spatial/0
          scene_id: libero_spatial
          instruction: "pick up the cube"
          max_steps: 100
        seed: 7
        n_episodes: 2
        """
    )
    p = tmp_path / "env.yaml"
    p.write_text(yaml_text)

    se = SimScene.from_yaml(str(p))
    assert se.robot_id == "franka_panda"
    assert se.scene.backend == PhysicsBackend.MUJOCO
    assert se.task.max_steps == 100
    assert se.seed == 7
    assert se.n_episodes == 2


def test_scene_environment_rejects_legacy_vla_block(tmp_path: Path) -> None:
    """YAMLs that still carry a ``vla:`` block fail loud with an actionable error."""
    from openral_core import SimScene
    from openral_core.exceptions import ROSConfigError

    yaml_text = textwrap.dedent(
        """\
        scene:
          id: libero_spatial
          backend: mujoco
        task:
          id: libero_spatial/0
          scene_id: libero_spatial
          instruction: ""
        vla:
          id: smolvla
          weights_uri: "rskills/smolvla-libero"
        """
    )
    p = tmp_path / "legacy.yaml"
    p.write_text(yaml_text)

    with pytest.raises(ROSConfigError, match="--rskill"):
        SimScene.from_yaml(str(p))


def test_sim_environment_from_yaml_raises_typed_error() -> None:
    """``SimEnvironment.from_yaml`` is removed; raises a typed error pointing to the new path."""
    from openral_core.exceptions import ROSConfigError

    with pytest.raises(ROSConfigError, match=r"SimScene\.from_yaml"):
        SimEnvironment.from_yaml("/dev/null")


def test_sim_environment_model_dump_serialisable() -> None:
    """SimEnvironment.model_dump(mode='json') round-trips through validation."""
    env = _ok_env(seed=11, n_episodes=4)
    blob = env.model_dump(mode="json")
    rebuilt = SimEnvironment.model_validate(blob)
    assert rebuilt == env


def test_physics_backend_values() -> None:
    """All declared backends must be string-valued and unique."""
    values = [b.value for b in PhysicsBackend]
    assert len(values) == len(set(values))
    assert "mock" in values
    assert "mujoco" in values
