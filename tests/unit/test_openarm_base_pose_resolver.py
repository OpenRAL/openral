"""Unit tests for openarm_robosuite's ``_resolve_base_translation`` helper.

ADR-0002 Amendment 3. ``env_cfg.base_pose`` is the only knob — there
is no legacy ``backend_options`` fallback and no hand-tuned default.

CLAUDE.md §1.11: no mocks. Real :class:`SimEnvironment` instances, the
real resolver function, the real :class:`Pose6D` schema.
"""

from __future__ import annotations

import pytest
from openral_core import (
    PhysicsBackend,
    Pose6D,
    SceneSpec,
    SimEnvironment,
    TaskSpec,
    VLASpec,
)
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.openarm_robosuite.env import _resolve_base_translation


def _env_with(base_pose: Pose6D | None = None) -> SimEnvironment:
    return SimEnvironment(
        robot_id="openarm",
        scene=SceneSpec(
            id="openarm_tabletop_pnp",
            backend=PhysicsBackend.MUJOCO,
        ),
        task=TaskSpec(
            id="openarm_tabletop_pnp/0",
            scene_id="openarm_tabletop_pnp",
            instruction="",
        ),
        vla=VLASpec(id="zero", weights_uri="mock://noop"),
        base_pose=base_pose,
    )


def _pose(xyz: tuple[float, float, float]) -> Pose6D:
    return Pose6D(xyz=xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="world")


def test_base_pose_required() -> None:
    """A YAML that omits ``base_pose`` for this scene fails loud — there
    is no implicit default."""
    env = _env_with(base_pose=None)
    with pytest.raises(ROSConfigError, match="base_pose"):
        _resolve_base_translation(env)


def test_base_pose_extracts_translation() -> None:
    env = _env_with(base_pose=_pose((0.30, 0.0, 0.45)))
    lift_z, forward_x = _resolve_base_translation(env)
    assert lift_z == pytest.approx(0.45)
    assert forward_x == pytest.approx(0.30)


def test_canonical_default_pose_round_trips() -> None:
    """The pose used in `scenes/sim/openarm_tabletop.yaml` (the
    table-clearance defaults the old hand-tuned constants encoded) must
    flow through the resolver as (0.55, 0.20)."""
    env = _env_with(base_pose=_pose((0.20, 0.0, 0.55)))
    lift_z, forward_x = _resolve_base_translation(env)
    assert lift_z == pytest.approx(0.55)
    assert forward_x == pytest.approx(0.20)


def test_nonzero_y_rejected() -> None:
    env = _env_with(base_pose=_pose((0.1, 0.05, 0.4)))
    with pytest.raises(ROSConfigError, match="y"):
        _resolve_base_translation(env)


def test_non_identity_quaternion_rejected() -> None:
    env = _env_with(
        base_pose=Pose6D(
            xyz=(0.1, 0.0, 0.4),
            quat_xyzw=(0.0, 0.0, 0.7071, 0.7071),  # 90° about Z
            frame_id="world",
        )
    )
    with pytest.raises(ROSConfigError, match="identity"):
        _resolve_base_translation(env)
