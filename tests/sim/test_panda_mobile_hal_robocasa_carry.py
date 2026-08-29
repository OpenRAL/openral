"""The carry-while-navigating scene #108 needs, measured on the live env.

`openral_nav2_bringup`'s payload-grown footprint and lidar self-filter (PR #143)
are only exercised when the base *translates* while an object is held.
`scenes/deploy/robocasa_deliver_straw.yaml` pins the one target50 task measured
to do that. These tests are what fails if the pin stops meaning what it says —
if a RoboCasa bump reshuffles the layout the seed selects, or if the task's
object placement changes.

The distances asserted here are the ones written into the scene's comment and
into `docs/reference/robocasa-carry-survey.md`.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from openral_core.schemas import DeployScene
from openral_sim._deps import _has_robocasa_kitchen

from tests.sim.conftest import mujoco_renderer_probe_error

_ROBOCASA_INACTIVE = not _has_robocasa_kitchen()
_RENDERER_ERROR = mujoco_renderer_probe_error() if not _ROBOCASA_INACTIVE else ""

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(_ROBOCASA_INACTIVE, reason="RoboCasa kitchen fork is not active"),
    pytest.mark.skipif(
        bool(_RENDERER_ERROR), reason=_RENDERER_ERROR or "no MuJoCo offscreen renderer"
    ),
]

SCENE = "scenes/deploy/robocasa_deliver_straw.yaml"

#: `openral_nav2_bringup` README, criterion 1: below this the arm bridges the
#: gap from one base pose and `NavigateToPose` never runs.
REQUIRED_BASE_TRANSLATION_M = 1.0

#: The panda_mobile arm cannot reach further than this, so an object beyond it
#: cannot be grasped without driving. Franka Panda's specified reach is 0.855 m;
#: the scene needs the straw comfortably inside it.
ARM_REACH_M = 0.855


def _measured_distances() -> tuple[dict[str, float], int]:
    """Build the pinned scene and measure each object's distance from the base."""
    import mujoco
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(SCENE)
    try:
        env.reset(seed=seed)
        model, data = env.mujoco_handles()
        inner = getattr(env, "_env", None) or getattr(env, "env", None)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_base")
        base_xy = np.array(data.xpos[base_id])[:2]
        out: dict[str, float] = {}
        for name, obj in inner.objects.items():
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
            out[name] = float(np.linalg.norm(np.array(data.xpos[body])[:2] - base_xy))
        return out, int(seed)
    finally:
        env.close()


def test_scene_pins_the_seed_the_measurements_were_taken_at() -> None:
    scene = DeployScene.model_validate(yaml.safe_load(open(SCENE)))
    assert scene.scene.id == "robocasa/DeliverStraw"
    assert scene.seed == 3, "the measured distances below are properties of seed 3"


def test_the_payload_starts_within_reach_and_the_destination_does_not() -> None:
    """Criterion 1: grasp without driving, then drive while holding it.

    Both halves matter. If the straw were out of reach the base would move
    before it was carrying anything; if the cup were within reach the arm would
    bridge the gap and `NavigateToPose` would never run.
    """
    dists, seed = _measured_distances()
    assert seed == 3
    assert dists["straw"] < ARM_REACH_M, (
        f"straw at {dists['straw']:.3f} m is beyond the arm's reach — the base would "
        "have to drive before it is carrying anything"
    )
    assert dists["glass_cup"] > REQUIRED_BASE_TRANSLATION_M, (
        f"glass cup at {dists['glass_cup']:.3f} m no longer requires a base translation; "
        "the scene stops exercising the Nav2 payload footprint"
    )


def test_measured_geometry_matches_what_the_scene_and_docs_claim() -> None:
    """The numbers written into the scene comment and the survey page."""
    dists, _ = _measured_distances()
    assert dists["straw"] == pytest.approx(0.500, abs=0.01)
    assert dists["glass_cup"] == pytest.approx(3.795, abs=0.01)
