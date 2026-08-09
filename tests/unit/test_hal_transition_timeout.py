"""HAL autostart budget must outlast what ``on_configure`` actually does.

``tools/lifecycle_autostart.py`` bounds each lifecycle transition, and
``sim_e2e.launch.py`` spawns it for the HAL. The transition it waits on runs
``build_sim_env_from_yaml`` → the scene factory, which for a sidecar backend
*boots the simulator in-process*: ``connect()`` spawns the sidecar and blocks
in ``_wait_for_boot``.

Those backends carry boot budgets far above the old fixed 300 s literal
(``isaac_sim`` 900 s, ``behavior`` 1200 s, ``robotwin`` 600 s), and the shipped
Isaac / BEHAVIOR deploy scenes raise theirs to ``boot_timeout_s: 1200``. Isaac
Sim 5.1 reaches ``app ready`` in ~13 s on an RTX 4070 Laptop, but a sidecar
that wedges after that burns the client's full ``boot_timeout_s`` before
``connect()`` raises — observed live — so the transition's worst case is the
declared budget, not the nominal boot. When the
autostart expired first it exited non-zero *while ``on_configure`` kept
running*, so the HAL could finish configuring with nothing left to drive
ACTIVATE — parked in INACTIVE, no ``/joint_states``, no cameras, no message
naming the cause.

Real in-tree scene YAMLs and the real backend constants throughout — the point
is that the shipped scenes and the launcher agree (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_hal.sim_bringup import (
    HAL_TRANSITION_TIMEOUT_FLOOR_S,
    HAL_TRANSITION_TIMEOUT_MARGIN_S,
    hal_transition_timeout_s,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEPLOY_SCENES = sorted((_REPO_ROOT / "scenes" / "deploy").glob("*.yaml"))


def _declared_boot_timeout_s(path: Path) -> float | None:
    """The scene's own ``backend_options.boot_timeout_s``, if it sets one."""
    opts = yaml.safe_load(path.read_text())["scene"].get("backend_options") or {}
    raw = opts.get("boot_timeout_s")
    return None if raw is None else float(raw)


def test_no_config_falls_back_to_the_floor() -> None:
    """A launch with no DeployScene keeps the historical 300 s budget."""
    assert hal_transition_timeout_s("") == str(HAL_TRANSITION_TIMEOUT_FLOOR_S)
    assert hal_transition_timeout_s(None) == str(HAL_TRANSITION_TIMEOUT_FLOOR_S)


def test_an_unreadable_config_falls_back_rather_than_raising() -> None:
    """Reporting a bad scene path belongs to the resolver, not the budget helper.

    Raising here would abort the whole launch before any node starts, replacing
    the resolver's specific error with a generic one.
    """
    assert hal_transition_timeout_s("/nonexistent/scene.yaml") == str(
        HAL_TRANSITION_TIMEOUT_FLOOR_S
    )


@pytest.mark.parametrize("scene_path", _DEPLOY_SCENES, ids=lambda p: p.name)
def test_every_shipped_deploy_scene_gets_a_budget_it_can_live_within(
    scene_path: Path,
) -> None:
    """The launcher must wait at least as long as the scene says it may need.

    This is the regression that mattered: five shipped scenes asked for 1200 s
    against a launcher hardcoded to 300 s.
    """
    budget = float(hal_transition_timeout_s(str(scene_path)))
    declared = _declared_boot_timeout_s(scene_path)

    assert budget >= HAL_TRANSITION_TIMEOUT_FLOOR_S, "the floor must never be lowered"
    if declared is not None:
        assert budget >= declared, (
            f"{scene_path.name} declares boot_timeout_s={declared} but the HAL "
            f"autostart would only wait {budget}s"
        )


def test_the_sidecar_scenes_are_the_ones_that_exceed_the_floor() -> None:
    """Pins which scenes need the raised budget, so a new one is a visible choice.

    Only backends that boot a simulator inside ``on_configure`` should push past
    the floor; a scene that merely imports MuJoCo must not quietly buy itself a
    20-minute window in which a wedged HAL goes unreported.
    """
    raised = {
        p.name
        for p in _DEPLOY_SCENES
        if float(hal_transition_timeout_s(str(p))) > HAL_TRANSITION_TIMEOUT_FLOOR_S
    }
    assert raised == {
        "behavior_r1pro.yaml",
        "isaac_franka.yaml",
        "isaac_franka_bowl.yaml",
        "isaac_franka_urdf.yaml",
        "isaac_panda_mobile_urdf.yaml",
    }


def test_the_budget_adds_margin_over_the_declared_boot() -> None:
    """``on_configure`` also imports the sim stack and runs the first reset.

    Granting exactly ``boot_timeout_s`` would let the sidecar consume the whole
    budget and still leave the transition to be reported as a timeout.
    """
    scene = _REPO_ROOT / "scenes" / "deploy" / "isaac_franka.yaml"
    declared = _declared_boot_timeout_s(scene)
    assert declared is not None
    assert float(hal_transition_timeout_s(str(scene))) == pytest.approx(
        declared + HAL_TRANSITION_TIMEOUT_MARGIN_S
    )


def test_backend_boot_defaults_still_exceed_the_floor() -> None:
    """Guards the premise: if these ever drop below the floor, this is dead code.

    Reads the real backend constants — a future tightening of an upstream boot
    time should retire the derivation rather than leave it silently inert.
    """
    from openral_sim.backends import behavior, isaac_sim, robotwin

    for module in (isaac_sim, behavior, robotwin):
        assert module._DEFAULT_BOOT_TIMEOUT_S > HAL_TRANSITION_TIMEOUT_FLOOR_S, (
            f"{module.__name__} boots faster than the floor now — retire the "
            "scene-derived budget instead of leaving it inert"
        )
