"""`mj_geomDistance` is wrong on a real RoboCasa pair, and the shipped probe is not.

This is the regression test the start-state census asked for in its own
recommendation 6 ("stop relying on `mj_geomDistance` for fixture-vs-link
adjudication, or gate it behind a self-check ... a regression test over a fixed
RoboCasa state with a sampled reference would catch it").

It pins the exact state the defect was characterised on — ``robocasa_fridge_
drawer`` at ``layout_ids: [9]``, seed 1, ``robot0_link7_collision`` against
``fridge_right_group_freezer_door_main`` — and asserts three things about it:

1. ``mujoco.mj_geomDistance`` still returns ``+0.000000`` there, with a witness
   segment whose endpoints lie outside **both** geoms. If upstream ever fixes
   this, the test says so loudly rather than passing quietly, because the
   standing caveat in the evidence ledger would then need revisiting.
2. ``openral_hal.convex_distance`` returns ``+0.148512 mm`` with a closed
   separating-axis certificate and a witness on both surfaces.
3. The probe that feeds every stop record
   (``openral_hal.sim_sensor_bridge._nearest_pair_records``) reports the
   certified number and attests it, so a downstream adjudicator can tell.

No mocks (CLAUDE.md §1.11): the real tracked scene YAML through the real
registered RoboCasa adapter, a real robosuite/MuJoCo kitchen, and the robot's
own manifest.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
import yaml

pytest.importorskip("openral_sim")
pytest.importorskip("mujoco")
pytest.importorskip("robocasa")  # robocasa (robosuite >=1.5) ⊥ libero (robosuite 1.4)

import mujoco
import numpy as np

from tests.sim.conftest import mujoco_renderer_probe_error

_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
_SCENE = _REPO_ROOT / "scenes" / "deploy" / "robocasa_fridge_drawer.yaml"
_ROBOT = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"

# The state the defect was characterised on. Layout 9 is NOT the scene's
# shipped pin (30) — it is overridden here on purpose, because 30 is a kitchen
# where the two geoms are nowhere near each other and the degenerate
# configuration does not arise. Pinning the layout that exhibits it is the
# point of the test.
_LAYOUT = 9
_LINK_GEOM = "robot0_link7_collision"
_DOOR_GEOM = "fridge_right_group_freezer_door_main"
# Certified with a duality gap of ~2e-17 m; see the module docstring of
# `openral_hal.convex_distance` for the full characterisation.
_TRUE_GAP_M = 0.000148512
_MJ_WITNESS_SEGMENT_M = 0.126264


def _robocasa_unavailable() -> str:
    if importlib.util.find_spec("robocasa") is None:
        return "robocasa not installed"
    from openral_sim._deps import _has_robocasa_kitchen

    return "" if _has_robocasa_kitchen() else "RoboCasa kitchen fork is not active"


_ROBOCASA_ERROR = _robocasa_unavailable()
_RENDERER_ERROR = mujoco_renderer_probe_error() if not _ROBOCASA_ERROR else ""

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(bool(_ROBOCASA_ERROR), reason=_ROBOCASA_ERROR or "RoboCasa unavailable"),
    pytest.mark.skipif(
        bool(_RENDERER_ERROR), reason=_RENDERER_ERROR or "no MuJoCo offscreen renderer"
    ),
]


@pytest.fixture(scope="module")
def kitchen() -> tuple[Any, Any]:
    """The real layout-9 kitchen, composed the way ``openral deploy sim`` composes it."""
    from openral_core import DeployScene, SimEnvironment, TaskSpec, VLASpec
    from openral_sim.registry import SCENES

    deploy = DeployScene.model_validate(yaml.safe_load(_SCENE.read_text()))
    options = {
        **(deploy.scene.backend_options or {}),
        "ignore_done": True,
        "layout_ids": [_LAYOUT],
    }
    scene = deploy.scene.model_copy(update={"backend_options": options})
    sim = SCENES.get(scene.id)(
        SimEnvironment(
            robot_id=deploy.robot_id or SCENES.fixed_robot(scene.id),
            scene=scene,
            task=TaskSpec(
                id=f"{scene.id}/_instrument_probe",
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
    )
    sim.reset(seed=deploy.seed)
    env = getattr(sim._env, "unwrapped", sim._env)
    return env.sim.model._model, env.sim.data._data


def _geom(model: Any, name: str) -> int:
    geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
    assert geom >= 0, f"{name} is absent from this kitchen; the layout pin has drifted"
    return geom


def test_mj_geomdistance_still_reports_a_witness_outside_both_geoms(
    kitchen: tuple[Any, Any],
) -> None:
    """The defect, reproduced, and refuted by its own witness.

    A nearest-pair segment has its endpoints on the two geoms by definition.
    Here both ends are more than 40 cm clear of the bodies they claim to touch,
    while the returned distance is ``0.000000`` — an internal contradiction the
    caller can detect without knowing the right answer, which is exactly why
    `witness_clearance_m` exists.

    A failure here is not necessarily a regression: if upstream mujoco fixes
    ``mj_geomDistance``, this assertion is the notice to revisit standing
    caveat 8 in ``docs/reference/collision-validation-evidence.md``.
    """
    from openral_hal.convex_distance import witness_clearance_m

    model, data = kitchen
    link, door = _geom(model, _LINK_GEOM), _geom(model, _DOOR_GEOM)

    fromto = np.zeros(6)
    reported = float(mujoco.mj_geomDistance(model, data, link, door, 0.1, fromto))
    segment = float(np.linalg.norm(fromto[3:] - fromto[:3]))

    assert reported == pytest.approx(0.0, abs=1e-9)
    assert segment == pytest.approx(_MJ_WITNESS_SEGMENT_M, abs=1e-5)
    # A 0 m distance whose witness is 126 mm long is already inconsistent; that
    # both endpoints are far outside their own geoms settles it.
    assert witness_clearance_m(model, data, link, fromto[:3]) > 0.4
    assert witness_clearance_m(model, data, door, fromto[3:]) > 0.4


def test_the_certified_instrument_measures_the_same_pair_with_a_proof(
    kitchen: tuple[Any, Any],
) -> None:
    """The replacement gets it right and can show its working."""
    from openral_hal.convex_distance import convex_geom_distance, witness_clearance_m

    model, data = kitchen
    link, door = _geom(model, _LINK_GEOM), _geom(model, _DOOR_GEOM)

    result = convex_geom_distance(model, data, link, door)

    assert result.certified, result.uncertified_reason
    assert result.distance_m == pytest.approx(_TRUE_GAP_M, abs=1e-9)
    assert result.duality_gap_m < 1e-9
    assert result.lower_m == result.upper_m  # mesh-vs-box brackets exactly
    assert witness_clearance_m(model, data, link, result.witness_a) < 1e-9
    assert witness_clearance_m(model, data, door, result.witness_b) < 1e-9


def test_the_failure_is_a_degenerate_configuration_not_a_distance_regime(
    kitchen: tuple[Any, Any],
) -> None:
    """One picometre of displacement is the difference between wrong and right.

    This is what rules out every workaround that keeps the call: no choice of
    ``distmax``, no distance threshold and no "only trust it below N mm" rule
    can separate the good answers from the bad, because the bad ones are
    isolated points in configuration space — and a scene's reset pose, built
    from exact axis-aligned fixture placements, is where such points live.
    """
    model, data = kitchen
    link, door = _geom(model, _LINK_GEOM), _geom(model, _DOOR_GEOM)
    original = np.asarray(data.geom_xpos[link]).copy()

    try:
        for offset in (-1e-12, 1e-12):
            data.geom_xpos[link] = original + np.array([offset, 0.0, 0.0])
            nudged = float(mujoco.mj_geomDistance(model, data, link, door, 0.1, None))
            assert nudged == pytest.approx(_TRUE_GAP_M, abs=1e-6)
    finally:
        data.geom_xpos[link] = original

    assert float(mujoco.mj_geomDistance(model, data, link, door, 0.1, None)) == pytest.approx(
        0.0, abs=1e-9
    )


def test_the_shipped_probe_reports_the_certified_number_and_attests_it(
    kitchen: tuple[Any, Any],
) -> None:
    """The stop record a round would carry is the certified one.

    End to end through the real producer: the kernel-checked link scope, the
    real prefilter, the real round-robin budget. The pair the defect lives on
    must appear at its true distance, and the coverage block must let a
    downstream adjudicator tell certified evidence from uncertified.
    """
    from openral_core import RobotDescription
    from openral_hal.sim_sensor_bridge import _nearest_pair_records, kernel_checked_body_ids

    model, data = kitchen
    description = RobotDescription.from_yaml(_ROBOT)
    scope = kernel_checked_body_ids(model, description)
    assert scope, "panda_mobile declares collision geometry; the probe scope cannot be empty"

    robot_bodies = frozenset(
        body
        for body in range(model.nbody)
        if str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "").startswith(
            ("robot0", "mobilebase0", "gripper0")
        )
    )
    records, coverage = _nearest_pair_records(
        model, data, side=scope, other_excluded=robot_bodies, distmax_m=0.1, max_pairs=32
    )

    assert coverage["uncertified_pairs"] == 0
    assert coverage["certified_pairs"] == len(records) or coverage["certified_pairs"] >= len(
        records
    )
    assert coverage["distance_instrument"] == "openral_hal.convex_distance.convex_geom_distance"

    pair = next(r for r in records if r["geom_a"] == _LINK_GEOM and r["geom_b"] == _DOOR_GEOM)
    assert pair["distance_certified"] is True
    assert float(pair["distance_m"]) == pytest.approx(_TRUE_GAP_M, abs=1e-9)
    assert "witness_a_xyz" in pair and "witness_b_xyz" in pair
    # And the record survives a strict JSON round-trip, which is how a round
    # replays it.
    import json

    assert json.loads(json.dumps(records)) == records
