"""A scene that pins ``layout_ids`` must actually get that kitchen, and say so.

RoboCasa draws ``(layout_id, style_id)`` from ``env.rng`` inside
``Kitchen._load_model`` on every reset, so ``seed: 1`` alone does not identify a
kitchen — layout, style, fixtures, ``rack_index``, door-open amounts and the
robot spawn deviation all come off that one generator, and any upstream change
to the draw order reshuffles the scene at the same seed. The two scenes here
carry an explicit ``scene.backend_options.layout_ids`` for that reason, and this
file is what stops the pin from being decorative:

* the pinned layout is the layout RoboCasa composes, read back off the live env;
* the pin is checked, not assumed — ``_RoboCasaSim.reset`` raises
  ``ROSConfigError`` when the two disagree, which is the case a task's
  ``EXCLUDE_LAYOUTS`` / ``EXCLUDE_STYLES`` or a replayed ``_ep_meta`` produces;
* an out-of-domain pin is refused at the scene boundary with a typed error
  rather than a bare ``KeyError`` from inside RoboCasa's arena builder.

The expected layout is read from the scene YAML itself rather than duplicated
here, so the assertion cannot drift from the file it is protecting.

No mocks (CLAUDE.md §1.11): the real tracked scene YAMLs, the real registered
RoboCasa adapter, a real robosuite/MuJoCo kitchen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("openral_sim")
pytest.importorskip("mujoco")
pytest.importorskip("robocasa")  # robocasa (robosuite >=1.5) ⊥ libero (robosuite 1.4)

from openral_core import DeployScene, SimEnvironment, TaskSpec, VLASpec
from openral_core.exceptions import ROSConfigError
from openral_sim.registry import SCENES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PINNED_SCENES = (
    _REPO_ROOT / "scenes" / "deploy" / "robocasa_fridge_drawer.yaml",
    _REPO_ROOT / "scenes" / "deploy" / "robocasa_drawer_utensil.yaml",
)


def _compose(scene_yaml: Path, *, backend_overrides: dict[str, object]) -> Any:
    """Build the scene the way ``openral deploy sim`` builds it.

    Mirrors ``openral_hal.sim_bringup.build_sim_env_from_yaml``: load the
    DeployScene, force ``ignore_done`` on for continuous stepping, synthesise
    the noop TaskSpec the HAL never reads, and call the registered factory.
    """
    deploy = DeployScene.model_validate(yaml.safe_load(scene_yaml.read_text()))
    options = {**(deploy.scene.backend_options or {}), "ignore_done": True, **backend_overrides}
    scene = deploy.scene.model_copy(update={"backend_options": options})
    sim_env = SimEnvironment(
        robot_id=deploy.robot_id or SCENES.fixed_robot(scene.id),
        scene=scene,
        task=TaskSpec(
            id=f"{scene.id}/_hal_deploy_noop",
            scene_id=scene.id,
            instruction="",
            max_steps=None,
            success_key=None,
        ),
        vla=VLASpec(id="smolvla", weights_uri="hf://lerobot/smolvla_base"),
        base_pose=deploy.base_pose,
        seed=deploy.seed,
        n_episodes=1,
        record_video=False,
    )
    return SCENES.get(scene.id)(sim_env), deploy


@pytest.mark.sim
@pytest.mark.parametrize("scene_yaml", _PINNED_SCENES, ids=lambda p: p.stem)
def test_pinned_scene_composes_the_layout_it_declares(scene_yaml: Path) -> None:
    """The kitchen RoboCasa builds is the one the scene file names."""
    declared = yaml.safe_load(scene_yaml.read_text())["scene"]["backend_options"]["layout_ids"]
    assert isinstance(declared, list) and len(declared) == 1, (
        f"{scene_yaml.name} must pin exactly one layout for the pin to be "
        f"enforceable; got layout_ids={declared!r}"
    )

    sim, deploy = _compose(scene_yaml, backend_overrides={})
    try:
        sim.reset(seed=deploy.seed)
        env = getattr(sim._env, "unwrapped", sim._env)
        assert int(env.layout_id) == declared[0]
    finally:
        sim.close()


@pytest.mark.sim
def test_a_pin_the_env_cannot_honour_is_refused() -> None:
    """A layout the task excludes must raise, not quietly run a different kitchen.

    ``PickPlaceFridgeShelfToDrawer`` declares ``EXCLUDE_STYLES`` for the
    side-by-side fridges with no drawer, and the same filtering machinery
    applies to layouts. Pinning a style the task excludes empties the pool, so
    the pin cannot be honoured — the run must stop rather than compose
    something else. This is the guard against the failure mode the whole change
    exists for: a scene file describing a kitchen the run did not use.
    """
    scene_yaml = _REPO_ROOT / "scenes" / "deploy" / "robocasa_fridge_drawer.yaml"
    # 23 is the first entry of the task's own EXCLUDE_STYLES.
    sim, deploy = _compose(scene_yaml, backend_overrides={"style_ids": [23]})
    try:
        with pytest.raises((ROSConfigError, ValueError)):
            sim.reset(seed=deploy.seed)
    finally:
        sim.close()


@pytest.mark.sim
def test_out_of_domain_layout_is_refused_before_the_arena_builds() -> None:
    """``layout_ids: 99`` is a typed ROSConfigError, not a KeyError in RoboCasa."""
    scene_yaml = _REPO_ROOT / "scenes" / "deploy" / "robocasa_drawer_utensil.yaml"
    with pytest.raises(ROSConfigError) as excinfo:
        _compose(scene_yaml, backend_overrides={"layout_ids": [99]})
    assert "layout_ids" in str(excinfo.value)
