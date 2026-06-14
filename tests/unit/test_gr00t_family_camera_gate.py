"""Unit tests for the Gr00t-family (GR00T / RLDX) scene-camera-count gate.

GR00T and RLDX sidecar checkpoints read a fixed number of *distinct* camera
streams positionally (LIBERO=2, RC365=3, GR1/Simpler=1) and have **no**
single-view fallback — unlike the in-process lerobot adapters (smolvla / pi05 /
act), which resolve their camera list from ``scene.cameras`` and adapt. On a
scene that renders too few cameras the missing stream used to surface only as an
opaque ``observation.images[...]`` error *after* the multi-minute sidecar boot.

:func:`_require_scene_cameras` turns that into an upfront
:class:`ROSCapabilityMismatch`. These tests pin the gate against the real
:class:`SceneSpec` / :class:`SimEnvironment` schemas (no doubles, per
CLAUDE.md §1.11) — they need neither a GPU nor the ``gr00t`` opt-in group.
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
from openral_core.exceptions import ROSCapabilityMismatch
from openral_sim.policies.rldx import _RLDX_LAYOUT_CAMERA_COUNT, _require_scene_cameras


def _env_with_cameras(cameras: list[str]) -> SimEnvironment:
    """A real SimEnvironment whose scene declares ``cameras`` (empty = default)."""
    return SimEnvironment(
        robot_id="franka_panda",
        scene=SceneSpec(id="cam-gate-fixture", backend=PhysicsBackend.MUJOCO, cameras=cameras),
        task=TaskSpec(
            id="cam-gate-fixture/0",
            scene_id="cam-gate-fixture",
            instruction="noop",
            max_steps=1,
            success_key="is_success",
        ),
        vla=VLASpec(id="gr00t", weights_uri="rskills/gr00t-n17-libero"),
    )


def test_libero_layout_rejects_single_camera_scene() -> None:
    """gr00t/rldx LIBERO (2 cams) on a 1-camera scene → clear early mismatch.

    Mirrors gr00t on ``isaac_franka_lift`` (renders only ``camera1``) vs
    ``isaac_franka_bowl_plate`` (renders ``camera1``+``camera2``, which passes).
    """
    env_cfg = _env_with_cameras(["camera1"])
    with pytest.raises(ROSCapabilityMismatch, match="2 distinct camera views"):
        _require_scene_cameras(
            env_cfg, layout="libero", camera_keys=("camera1", "camera2"), family="gr00t"
        )


def test_libero_layout_accepts_two_camera_scene() -> None:
    env_cfg = _env_with_cameras(["camera1", "camera2"])
    _require_scene_cameras(
        env_cfg, layout="libero", camera_keys=("camera1", "camera2"), family="gr00t"
    )  # must not raise


def test_empty_cameras_is_adapter_default_not_a_mismatch() -> None:
    """A scene that omits ``cameras`` is the LIBERO adapter default (it renders
    camera1+camera2 itself) — never a false-reject of the LIBERO sim scenes."""
    env_cfg = _env_with_cameras([])
    _require_scene_cameras(
        env_cfg, layout="libero", camera_keys=("camera1", "camera2"), family="rldx"
    )  # must not raise


def test_single_view_layout_allows_single_camera_scene() -> None:
    """GR1 / Simpler checkpoints consume one camera — a 1-camera scene is fine."""
    env_cfg = _env_with_cameras(["camera1"])
    for layout in ("gr1", "simpler_widowx", "simpler_google"):
        _require_scene_cameras(
            env_cfg, layout=layout, camera_keys=("camera1", "camera2"), family="rldx"
        )  # must not raise


def test_rc365_layout_requires_three_cameras() -> None:
    env_cfg = _env_with_cameras(["camera1", "camera2"])
    with pytest.raises(ROSCapabilityMismatch, match="3 distinct camera views"):
        _require_scene_cameras(
            env_cfg, layout="rc365", camera_keys=("camera1", "camera2"), family="rldx"
        )
    ok = _env_with_cameras(["camera1", "camera2", "camera3"])
    _require_scene_cameras(
        ok, layout="rc365", camera_keys=("camera1", "camera2"), family="rldx"
    )  # must not raise


def test_layout_camera_count_table_matches_obs_builders() -> None:
    """Guard against the table drifting from the per-layout obs assemblers."""
    assert _RLDX_LAYOUT_CAMERA_COUNT == {
        "libero": 2,
        "gr1": 1,
        "rc365": 3,
        "simpler_widowx": 1,
        "simpler_google": 1,
    }
