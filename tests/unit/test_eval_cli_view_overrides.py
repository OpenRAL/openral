"""Unit tests for ``openral sim run``'s flag composition and viewer resolution.

After the ``feat(core,sim): SceneEnvironment + openral sim run --rskill, no
legacy`` commit, the canonical invocation is::

    openral sim run --config FILE.yaml --rskill rskills/<id>

The old ``--rskill / --robot`` form (legacy free-flag composition)
is gone. ``--config`` and ``--rskill`` are both required; the YAML
carries scene + task only. These tests pin the new flag-composition
contract and the tri-state ``--view / --no-view`` resolution.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openral_core.exceptions import ROSConfigError
from openral_sim.cli import _load_or_build_env, _resolve_view

REPO_ROOT = Path(__file__).resolve().parents[2]
# SimScene-tier LIBERO fixture (ADR-0041). The BenchmarkScene sibling at
# scenes/benchmark/libero_spatial.yaml carries metadata + n_episodes + seed and
# is rejected by `openral sim run` on the tier guard, so these CLI-mechanics
# tests use the slim SimScene fixture instead.
LIBERO_CFG = REPO_ROOT / "scenes" / "sim" / "libero_spatial.yaml"
LIBERO_RSKILL = "rskills/smolvla-libero"


# Default values mirror the Typer callback's defaults so fixtures override
# only the fields each test cares about.
_DEFAULT_ARGS: dict[str, Any] = {
    "config": None,
    "rskill": None,
    "robot": None,
    "task": None,
    "instruction": None,
    "max_steps": None,
    "n_episodes": None,
    "n_action_steps": None,
    "seed": None,
    "device": None,
    "save_dir": None,
    "save_video": None,
    "view": None,
    "verbose": False,
}


def _args(**overrides: Any) -> SimpleNamespace:
    """Build a parsed-args namespace matching the Typer callback's shape."""
    merged = {**_DEFAULT_ARGS, **overrides}
    return SimpleNamespace(**merged)


def _require_libero_cfg() -> None:
    if not LIBERO_CFG.exists():
        pytest.skip(f"{LIBERO_CFG} missing — examples/ tree not vendored")


def test_missing_config_rejects() -> None:
    """``--rskill`` alone (no --config) raises a typed error."""
    args = _args(rskill=LIBERO_RSKILL)
    with pytest.raises(ROSConfigError, match="--config"):
        _load_or_build_env(args)


def test_missing_rskill_rejects() -> None:
    """``--config`` alone (no --rskill) raises a typed error."""
    _require_libero_cfg()
    args = _args(config=LIBERO_CFG)
    with pytest.raises(ROSConfigError, match="--rskill"):
        _load_or_build_env(args)


def test_robot_flag_rejected_on_fixed_robot_scene() -> None:
    """``--robot`` on a fixed-robot scene (LIBERO=franka_panda) is rejected."""
    _require_libero_cfg()
    args = _args(config=LIBERO_CFG, rskill=LIBERO_RSKILL, robot="franka_panda")
    with pytest.raises(ROSConfigError, match="hard-fixes"):
        _load_or_build_env(args)


def test_hf_scheme_rejected() -> None:
    """``--rskill`` value must not carry an ``hf://`` scheme."""
    _require_libero_cfg()
    args = _args(config=LIBERO_CFG, rskill="hf://lerobot/smolvla_libero")
    with pytest.raises(ROSConfigError, match="hf://"):
        _load_or_build_env(args)


def test_task_override_swaps_id_only() -> None:
    """``--task`` swaps task.id while preserving the rest of the TaskSpec."""
    _require_libero_cfg()
    args = _args(config=LIBERO_CFG, rskill=LIBERO_RSKILL, task="libero_spatial/7")
    env = _load_or_build_env(args)
    assert env.task.id == "libero_spatial/7"
    assert env.task.scene_id == "libero_spatial"


def test_task_and_max_steps_compose() -> None:
    """``--task`` and ``--max-steps`` together both apply to the same TaskSpec."""
    _require_libero_cfg()
    args = _args(
        config=LIBERO_CFG,
        rskill=LIBERO_RSKILL,
        task="libero_spatial/4",
        max_steps=42,
    )
    env = _load_or_build_env(args)
    assert env.task.id == "libero_spatial/4"
    assert env.task.max_steps == 42


def test_save_video_enables_record_video_on_env() -> None:
    """``--save-video`` flips record_video on the composed SimEnvironment."""
    _require_libero_cfg()
    args = _args(
        config=LIBERO_CFG,
        rskill=LIBERO_RSKILL,
        save_video=Path("example_videos"),
    )
    env = _load_or_build_env(args)
    assert env.record_video is True


def test_omitting_save_video_preserves_yaml_record_video() -> None:
    """Without ``--save-video`` the CLI must not flip record_video on its own."""
    _require_libero_cfg()
    args = _args(config=LIBERO_CFG, rskill=LIBERO_RSKILL)
    env = _load_or_build_env(args)
    assert env.record_video is False


def test_n_action_steps_threaded_to_spec_extra() -> None:
    """``--n-action-steps N`` lands in ``vla.extra`` for apply_chunk_replay."""
    _require_libero_cfg()
    args = _args(config=LIBERO_CFG, rskill=LIBERO_RSKILL, n_action_steps=7)
    env = _load_or_build_env(args)
    assert env.vla.extra.get("n_action_steps") == 7


def test_resolve_view_explicit_no_view() -> None:
    """``--no-view`` always disables, regardless of display state."""
    assert _resolve_view(False) == (False, False)


def test_resolve_view_explicit_view_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--view`` opts into strict mode (errors loud on missing handles)."""
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    view, strict = _resolve_view(True)
    assert view is True
    assert strict is True


def test_resolve_view_auto_disables_under_egl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (``None``) auto-disables the viewer when ``MUJOCO_GL=egl``."""
    monkeypatch.setenv("MUJOCO_GL", "egl")
    assert _resolve_view(None) == (False, False)


def test_resolve_view_auto_disables_without_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default auto-disables on Linux when ``DISPLAY`` is unset."""
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    assert _resolve_view(None) == (False, False)


def test_resolve_view_auto_enables_with_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default auto-enables (non-strict) when a display is present."""
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr("sys.platform", "linux")
    assert _resolve_view(None) == (True, False)


def test_resolve_view_explicit_view_overrides_mujoco_gl_egl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--view`` + ``MUJOCO_GL=egl`` rewrites the env var to ``glfw`` in-process.

    The ``just sim-*`` recipes hard-code ``MUJOCO_GL=egl`` for headless CI.
    When the user explicitly passes ``--view`` from one of those recipes,
    the original code silently degraded to offscreen (mujoco honoured
    egl). Now we override the env var to glfw before mujoco is imported
    by the scene factory, so the viewer actually opens.
    """
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("DISPLAY", ":0")
    view, strict = _resolve_view(True)
    assert view is True
    assert strict is True
    assert os.environ["MUJOCO_GL"] == "glfw"


def test_resolve_view_explicit_view_raises_without_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--view`` + linux + no DISPLAY raises instead of silently degrading."""
    from openral_core.exceptions import ROSConfigError

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    with pytest.raises(ROSConfigError, match="DISPLAY is unset"):
        _resolve_view(True)


def test_resolve_view_explicit_view_leaves_other_mujoco_gl_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--view`` only overrides ``MUJOCO_GL=egl``; other values pass through."""
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("DISPLAY", ":0")
    _resolve_view(True)
    # osmesa was not the magic value, so it should be untouched.
    assert os.environ["MUJOCO_GL"] == "osmesa"
