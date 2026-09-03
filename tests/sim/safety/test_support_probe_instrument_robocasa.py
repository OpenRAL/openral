"""The support-contact witness probe, pinned on one of #170's four false zeros.

Issue #190. ``openral_hal._sim_attachment_evidence._probe_support_hits`` — the
producer of every ``support_witness`` an attachment message carries — still
measures with ``mujoco.mj_geomDistance``, the instrument PR #170 withdrew from
the E-stop evidence path. This is the regression test that pins one recorded
pair through it, so the conversion to
``openral_hal.convex_distance.convex_geom_distance`` has something that fails
if the witness path ever starts attesting a contact that is not there.

The pair is the sharpest of #170's four re-measured false zeros — the
``2026-08-23-master-s1`` baguette stop, ``robot0_link1_collision`` against
``counter_1_left_group_top_left_1``, recorded at ``0.000 m`` and certified at
**+107.930 mm** — because it is the only **solid↔solid** one: #139's
collidability filter would not have caught it, and the same round measured
``robot0_g12_vis``, a visual shell coincident with that same collision geom,
against the same world geom at ``0.107931 m``. One pair, two coincident robot
geoms, one right and one ``0.000``.

The state is rebuilt the way PR #170 and the 2026-08-23 census rebuilt it
(``docs/reference/collision-validation-evidence.md``, "Reconstruction method"):
the round's own scene YAML at its own seed, the robot driven to the
``robot_joint_state`` that round's ``sim.estop_ground_truth_snapshot``
published, the torso — an OmronMobileBase lift no manifest joint covers — set
from ``base_frame_tf``, then **verified** against that same TF. Two independent
checks say the reconstruction is the recorded configuration: the base body
lands within a micrometre of the TF, and the coincident ``robot0_g12_vis``
reproduces the round's own ``0.107931 m`` to 1e-7 m.

What is pinned:

1. ``mj_geomDistance`` at the probe's own ``_SUPPORT_PROBE_GAP_M`` window.
   **The recorded ``0.000 m`` does not reproduce here** — at mujoco 3.8.0 on
   x86_64 this pair measures correctly at every window wide enough to reach it
   and saturates to ``distmax`` below that. That is recorded as a fact about
   this host, not as a refutation of the round: the round's own probe, at its
   own 0.124555 m window, returned ``0.000``. If this assertion ever fails
   because the call returns ``0.000``, the degenerate configuration has
   reappeared *inside the support-witness window*, where it would manufacture a
   support witness on a pair 107.9 mm apart.
2. ``convex_geom_distance`` measures the same pair at the certified
   +107.930 mm with a closed separating-axis certificate.
3. ``_probe_support_hits`` attests **nothing** on it. This must hold under both
   instruments; it is the property the conversion must not break.

No mocks (CLAUDE.md §1.11): the round's real checked-in scene YAML and real
recorded snapshot, the real registered RoboCasa adapter, a real
robosuite/MuJoCo kitchen, and the robot's own manifest.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("openral_sim")
pytest.importorskip("mujoco")
pytest.importorskip("robocasa")  # robocasa (robosuite >=1.5) ⊥ libero (robosuite 1.4)

import mujoco
import numpy as np

from tests.sim.conftest import mujoco_renderer_probe_error

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROUND = (
    _REPO_ROOT
    / "tests"
    / "unit"
    / "fixtures"
    / "validation_matrix"
    / "2026-08-23-master-s1"
    / "baguette"
)
_SCENE = _ROUND / "robocasa_baguette_seed1.yaml"
_DEPLOY_LOG = _ROUND / "run_deploy_excerpt.log"
_ROBOT = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"

# The RoboCasa OmronMobileBase's prismatic arm-mount lift. `panda_mobile`'s
# manifest declares no joint for it, so `robot_joint_state` alone does not
# determine the robot's pose — `base_frame_tf` beside it does (census caveat 1).
_TORSO_JOINT = "mobilebase0_joint_torso_height"

_LINK_GEOM = "robot0_link1_collision"
_COUNTER_GEOM = "counter_1_left_group_top_left_1"
# The visual shell coincident with `_LINK_GEOM`. The round recorded it against
# `_COUNTER_GEOM` at 0.107931 m while recording the collision geom at 0.000 m;
# reproducing its number is what identifies the reconstruction as the recorded
# configuration rather than merely a nearby one.
_VIS_GEOM = "robot0_g12_vis"
_VIS_RECORDED_M = 0.107931

# PR #170's certified value for the pair, duality gap ~4e-15 m.
_CERTIFIED_GAP_M = 0.107930
# The window the round's own E-stop near-miss probe used (the snapshot's
# `nearest_probe_coverage.distmax_m`), which is where it recorded 0.000 m.
_ROUND_PROBE_DISTMAX_M = 0.124555
# `base_frame_tf` is published rounded to 6 decimals, so a micrometre is the
# floor on any reconstruction check against it.
_RECONSTRUCTION_TOL_M = 1e-5


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


def _recorded_snapshot() -> dict[str, Any]:
    """The round's own ``sim.estop_ground_truth_snapshot``, via the round reader.

    ``tools/validation_matrix.py`` is not an installed package; it is loaded by
    path the way ``tests/unit/test_validation_matrix.py`` loads it, so this
    reads the artifact with the same parser the harness derives verdicts with.
    """
    spec = importlib.util.spec_from_file_location(
        "validation_matrix", _REPO_ROOT / "tools" / "validation_matrix.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    snapshot = module.parse_json_log_line(
        _DEPLOY_LOG.read_text().splitlines(), "sim.estop_ground_truth_snapshot"
    )
    assert snapshot is not None, f"{_DEPLOY_LOG} carries no stop snapshot"
    assert snapshot["stop_class"] == "attached_payload"
    return dict(snapshot)


@pytest.fixture(scope="module")
def recorded_stop() -> tuple[Any, Any, dict[str, Any]]:
    """The 08-23 baguette stop, reconstructed and verified against its own TF."""
    from openral_core import DeployScene, RobotDescription, SimEnvironment, TaskSpec, VLASpec
    from openral_sim.registry import SCENES

    snapshot = _recorded_snapshot()
    deploy = DeployScene.model_validate(yaml.safe_load(_SCENE.read_text()))
    scene = deploy.scene.model_copy(
        update={"backend_options": {**(deploy.scene.backend_options or {}), "ignore_done": True}}
    )
    sim = SCENES.get(scene.id)(
        SimEnvironment(
            robot_id=deploy.robot_id or SCENES.fixed_robot(scene.id),
            scene=scene,
            task=TaskSpec(
                id=f"{scene.id}/_support_probe_instrument",
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
    model, data = env.sim.model._model, env.sim.data._data

    # The manifest owns the URDF-joint -> MJCF-joint mapping (`sim_joint_name`);
    # nothing here re-derives it.
    description = RobotDescription.from_yaml(str(_ROBOT))
    sim_joint_of = {joint.name: joint.sim_joint_name or joint.name for joint in description.joints}
    state = snapshot["robot_joint_state"]
    for name, position in zip(state["name"], state["position"], strict=True):
        data.qpos[_qpos_adr(model, sim_joint_of[name])] = float(position)

    # The torso lift is prismatic along world +Z, so it is solved in one step:
    # zero it, read where the TF's own body lands, and take the difference.
    torso = _qpos_adr(model, _TORSO_JOINT)
    tf = snapshot["base_frame_tf"]
    base_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tf["body"]))
    assert base_body >= 0, f"{tf['body']} is absent from this kitchen"
    data.qpos[torso] = 0.0
    mujoco.mj_forward(model, data)
    data.qpos[torso] = float(tf["world_xyz"][2]) - float(data.xpos[base_body][2])
    mujoco.mj_forward(model, data)

    # Verification, not decoration: the reconstruction is only usable if the
    # body `base_frame_tf` names lands where the round said it was.
    residual_m = float(
        np.linalg.norm(np.asarray(data.xpos[base_body]) - np.asarray(tf["world_xyz"]))
    )
    assert residual_m < _RECONSTRUCTION_TOL_M, (
        f"reconstruction is {residual_m * 1e3:.3f} mm off the recorded base_frame_tf; "
        "nothing measured on it describes the stop"
    )
    return model, data, snapshot


def _qpos_adr(model: Any, joint_name: str) -> int:
    joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
    assert joint >= 0, f"{joint_name} is absent from this kitchen; the scene pin has drifted"
    return int(model.jnt_qposadr[joint])


def _geom(model: Any, name: str) -> int:
    geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
    assert geom >= 0, f"{name} is absent from this kitchen; the scene pin has drifted"
    return geom


def test_mj_geomdistance_at_the_support_window_never_reports_a_contact(
    recorded_stop: tuple[Any, Any, dict[str, Any]],
) -> None:
    """What the shipped probe's own call returns on the pair the round zeroed.

    A failure here is the alarm, in either direction:

    * ``0.000`` at the 1 mm window means #170's degenerate configuration has
      reached the support-witness path, where it manufactures a witness for a
      pair 107.9 mm apart — the defect #190 exists to make impossible.
    * a value other than ``distmax`` at 1 mm, or other than the certified gap
      at the round's own window, means this reconstruction is no longer the
      recorded state and the numbers below describe something else.

    The recorded ``0.000 m`` itself is **not** reproduced here (mujoco 3.8.0,
    x86_64): at the round's own 0.124555 m window this call now returns the
    certified answer. That is stated as a fact about this host, not as a
    correction to the round — the round recorded what it recorded, and #170
    withdrew it on the certified instrument's evidence, not on a re-run.
    """
    from openral_hal._sim_attachment_evidence import _SUPPORT_PROBE_GAP_M

    model, data, _ = recorded_stop
    link, counter = _geom(model, _LINK_GEOM), _geom(model, _COUNTER_GEOM)

    fromto = np.zeros(6)
    at_support_window = float(
        mujoco.mj_geomDistance(model, data, link, counter, _SUPPORT_PROBE_GAP_M, fromto)
    )
    assert at_support_window != pytest.approx(0.0, abs=1e-9), (
        f"mj_geomDistance returned {at_support_window:+.6f} m for {_LINK_GEOM} vs "
        f"{_COUNTER_GEOM}, which are certified {_CERTIFIED_GAP_M * 1e3:.3f} mm apart. "
        "The false zero PR #170 characterised is now inside the support-witness "
        "window; _probe_support_hits would attest a contact that is not there."
    )
    # Saturation is the correct answer at a window this far inside the gap.
    assert at_support_window == pytest.approx(_SUPPORT_PROBE_GAP_M, abs=1e-9)

    at_round_window = float(
        mujoco.mj_geomDistance(model, data, link, counter, _ROUND_PROBE_DISTMAX_M, None)
    )
    assert at_round_window == pytest.approx(_CERTIFIED_GAP_M, abs=1e-4), (
        f"at the round's own {_ROUND_PROBE_DISTMAX_M} m window this pair now measures "
        f"{at_round_window:+.6f} m; the round recorded 0.000 m and PR #170 certified "
        f"{_CERTIFIED_GAP_M:+.6f} m"
    )


def test_the_certified_instrument_measures_the_pair_the_round_zeroed(
    recorded_stop: tuple[Any, Any, dict[str, Any]],
) -> None:
    """+107.930 mm, with a proof, on the state the round stopped in.

    The coincident visual shell is the cross-check that makes this the recorded
    configuration and not merely a plausible one: the round measured
    ``robot0_g12_vis`` against the same world geom at 0.107931 m while
    recording the collision geom it is coincident with at 0.000 m.
    """
    from openral_hal.convex_distance import convex_geom_distance, witness_clearance_m

    model, data, _ = recorded_stop
    link, counter = _geom(model, _LINK_GEOM), _geom(model, _COUNTER_GEOM)

    result = convex_geom_distance(model, data, link, counter)

    assert result.certified, result.uncertified_reason
    assert result.distance_m == pytest.approx(_CERTIFIED_GAP_M, abs=1e-4)
    assert result.duality_gap_m < 1e-9
    assert witness_clearance_m(model, data, link, result.witness_a) < 1e-9
    assert witness_clearance_m(model, data, counter, result.witness_b) < 1e-9

    coincident = convex_geom_distance(model, data, _geom(model, _VIS_GEOM), counter)
    assert coincident.certified, coincident.uncertified_reason
    assert coincident.distance_m == pytest.approx(_VIS_RECORDED_M, abs=1e-6)


def test_the_support_witness_probe_attests_nothing_on_it(
    recorded_stop: tuple[Any, Any, dict[str, Any]],
) -> None:
    """The witness path must not claim a contact 107.9 mm away — under either ruler.

    ``_probe_support_hits`` takes plain geom lists, so the arm link stands in
    for the payload side here: what is pinned is the *instrument*, on the exact
    pair that produced a ``0.000 m`` reading in a real round, not a claim that
    ``panda_link1`` is ever carried. An attested hit here is a support witness
    the safety kernel would honour as an exemption.
    """
    from openral_hal._sim_attachment_evidence import _probe_support_hits, _root_motion_body

    model, data, _ = recorded_stop
    link, counter = _geom(model, _LINK_GEOM), _geom(model, _COUNTER_GEOM)
    support_root = _root_motion_body(model, int(model.geom_bodyid[counter]))

    hits = _probe_support_hits(
        model,
        data,
        payload_geoms=[link],
        candidate_geoms=[counter],
        support_root_of_geom={counter: support_root},
    )

    assert hits == [], (
        f"the witness probe attested {len(hits)} support contact(s) on a pair certified "
        f"{_CERTIFIED_GAP_M * 1e3:.3f} mm apart: "
        + ", ".join(f"penetration {hit.penetration_m:+.6f} m" for hit in hits)
    )
