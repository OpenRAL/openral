"""Unit tests for ``SimScene.base_pose`` (ADR-0002 Amendment 3, ADR-0041).

``base_pose`` is the per-rollout robot mounting pose, in the scene's world
frame. It is honoured by **free-axis** scene adapters only (``mock``,
``maniskill3``, ``simpler_env``); setting it on a ``fixed_robot=`` scene
(LIBERO / MetaWorld / RoboCasa / PushT / ALOHA) is rejected by the CLI
guard. Adapters anchor the pose on the robot manifest's ``base_frame``
(``RobotDescription.base_frame``), so no robot-side schema change is
needed — the existing field is sufficient. ``base_pose`` lives on
:class:`DeployScene` and is inherited by :class:`SimScene` and
:class:`BenchmarkScene` (ADR-0041 three-tier hierarchy).

CLAUDE.md §1.11: real schemas, real CLI runner, no mocks. The CLI guard
fires at config-build time, so no physics dependency is needed to
exercise it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from openral_cli.main import app
from openral_core import (
    PhysicsBackend,
    Pose6D,
    SceneSpec,
    SimEnvironment,
    SimScene,
    TaskSpec,
    VLASpec,
)
from pydantic import ValidationError
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
# Use a SimScene-tier fixed_robot YAML for the base_pose CLI guard test.
# `scenes/sim/libero_spatial.yaml` is a SimScene with a scene-fixed
# `franka_panda`, which is exactly what the base_pose guard needs to reject.
LIBERO_CFG = REPO_ROOT / "scenes" / "sim" / "libero_spatial.yaml"
LIBERO_RSKILL = "rskills/smolvla-libero"

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _identity_pose(xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Pose6D:
    return Pose6D(xyz=xyz, quat_xyzw=_IDENTITY_QUAT, frame_id="world")


def test_scene_environment_base_pose_defaults_to_none() -> None:
    se = SimScene(
        scene=SceneSpec(id="mock", backend=PhysicsBackend.MOCK),
        task=TaskSpec(id="mock/0", scene_id="mock", instruction=""),
    )
    assert se.base_pose is None


def test_scene_environment_base_pose_round_trip(tmp_path: Path) -> None:
    """A YAML carrying ``base_pose:`` validates and round-trips through
    ``model_dump`` → ``model_validate`` without loss."""
    yaml_text = textwrap.dedent(
        """\
        robot_id: so100_follower
        scene:
          id: mock
          backend: mock
        task:
          id: mock/0
          scene_id: mock
          instruction: "pick"
        base_pose:
          xyz: [0.5, 0.0, 0.1]
          quat_xyzw: [0.0, 0.0, 0.7071, 0.7071]
          frame_id: "world"
        """
    )
    p = tmp_path / "env.yaml"
    p.write_text(yaml_text)

    se = SimScene.from_yaml(str(p))
    assert se.base_pose is not None
    assert se.base_pose.xyz == (0.5, 0.0, 0.1)
    assert se.base_pose.frame_id == "world"

    rebuilt = SimScene.model_validate(se.model_dump(mode="json"))
    assert rebuilt == se


def test_scene_environment_top_level_unknown_key_rejected() -> None:
    """``SimScene`` has ``extra='forbid'`` — a typo at the top level (e.g.
    ``base_poose:``) surfaces as a validation error rather than being
    silently dropped, so YAML drift fails loud."""
    with pytest.raises(ValidationError):
        SimScene.model_validate(
            {
                "scene": {"id": "mock", "backend": "mock"},
                "task": {"id": "mock/0", "scene_id": "mock"},
                "base_poose": {  # deliberate typo
                    "xyz": [0.0, 0.0, 0.0],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "frame_id": "world",
                },
            }
        )


def test_sim_environment_carries_base_pose() -> None:
    """The runtime composed form also exposes ``base_pose`` — adapter
    factories receive it on ``env_cfg.base_pose``."""
    env = SimEnvironment(
        robot_id="so100_follower",
        scene=SceneSpec(id="mock", backend=PhysicsBackend.MOCK),
        task=TaskSpec(id="mock/0", scene_id="mock", instruction="x"),
        vla=VLASpec(id="zero", weights_uri="mock://noop"),
        base_pose=_identity_pose((0.1, 0.2, 0.3)),
    )
    assert env.base_pose is not None
    assert env.base_pose.xyz == (0.1, 0.2, 0.3)


def test_fixed_robot_scene_rejects_base_pose_at_cli() -> None:
    """A LIBERO YAML with a ``base_pose:`` block is rejected at CLI compose
    time — the scene hard-fixes its robot and a per-rollout mounting pose
    has no physical meaning."""
    if not LIBERO_CFG.exists():
        pytest.skip("LIBERO config not present on this branch")

    # Read the canonical YAML and inject a base_pose block via model_dump.
    se = SimScene.from_yaml(str(LIBERO_CFG))
    augmented = se.model_copy(update={"base_pose": _identity_pose()})

    # Round-trip through YAML so the CLI loads it the same way the user
    # would. This exercises the real ``SimScene.from_yaml`` path,
    # not just the in-memory copy.
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(augmented.model_dump(mode="json"), fh)
        cfg_path = fh.name

    try:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "sim",
                "run",
                "--config",
                cfg_path,
                "--rskill",
                LIBERO_RSKILL,
                "--no-view",
            ],
        )
        assert result.exit_code != 0
        out = result.output + (result.stderr or "")
        assert "base_pose" in out
        assert "free-axis" in out or "hard-fixes" in out
    finally:
        Path(cfg_path).unlink(missing_ok=True)


def test_free_axis_scene_accepts_base_pose() -> None:
    """A mock scene (free-axis) accepts ``base_pose`` and the runtime
    ``SimEnvironment`` carries it through."""
    se = SimScene(
        robot_id="so100_follower",
        scene=SceneSpec(id="mock", backend=PhysicsBackend.MOCK),
        task=TaskSpec(id="mock/0", scene_id="mock", instruction=""),
        base_pose=_identity_pose((0.0, 0.5, 0.0)),
    )
    assert se.base_pose is not None
    assert se.base_pose.xyz == (0.0, 0.5, 0.0)


def test_base_pose_directly_raises_on_invalid_quaternion_count() -> None:
    """``Pose6D.quat_xyzw`` is a fixed-length tuple; a 3-vector
    quaternion is rejected by Pydantic before any adapter sees it."""
    with pytest.raises(ValidationError):
        Pose6D(
            xyz=(0.0, 0.0, 0.0),
            quat_xyzw=(0.0, 0.0, 1.0),  # type: ignore[arg-type] # reason: testing rejection of wrong shape
            frame_id="world",
        )
