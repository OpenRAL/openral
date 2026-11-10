"""The MJCF lowering's mimic check on a NORMALISED gripper channel (MakerMods Metal).

The Metal's ``gripper`` channel is a [0, 1] fraction; ``sim.grippers[]``
(``write_mode: normalised``) maps it onto joint8's [0, 0.05] m, and the MJCF
couples ``joint7 = -joint8``. Both fingers are ``mimic_joints`` rows posed from
the fraction (``joint7 = -0.05 f``, ``joint8 = +0.05 f``), one of them on the
MJCF's DRIVEN joint itself. The check must compare in the fraction's units —
through the actuator the twin really drives — and must still refuse a wrong
multiplier, a wrong offset or an actuator that does not drive the leader.
Real manifests and the real derived MJCF only (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco", reason="needs mujoco (sim group) for the MJCF")
pytest.importorskip("trimesh")

from openral_core import RobotDescription  # noqa: E402
from openral_core.assets import AssetFetchError, resolve_asset  # noqa: E402
from openral_core.exceptions import ROSConfigError  # noqa: E402
from openral_safety.urdf_lowering import (  # noqa: E402
    _declared_mimic_followers,
    fit_collision_geometry_from_mjcf,
    lower_robot_from_mjcf,
    mjcf_coupled_joints,
)

_MANIFEST = Path(__file__).resolve().parents[2] / "robots" / "makermods_metal" / "robot.yaml"


@pytest.fixture(scope="module")
def metal() -> tuple[RobotDescription, object]:
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    try:
        mjcf = resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=_MANIFEST.parent)  # type: ignore[arg-type]  # reason: the Metal declares assets.mjcf
    except AssetFetchError as exc:
        pytest.skip(f"Metal MJCF unavailable on this host: {exc}")
    return robot, mujoco.MjModel.from_xml_path(str(mjcf))


def _with_mimic(robot: RobotDescription, name: str, **update: float) -> RobotDescription:
    rows = [m.model_copy(update=update) if m.name == name else m for m in robot.mimic_joints]
    return robot.model_copy(update={"mimic_joints": rows})


def test_both_fingers_are_accepted_in_the_fraction_units(
    metal: tuple[RobotDescription, object],
) -> None:
    robot, model = metal
    declared = _declared_mimic_followers(robot, model, mjcf_coupled_joints(model))
    ids = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)) for n in ("joint7", "joint8")
    }
    assert declared == ids


@pytest.mark.parametrize(
    ("name", "update"),
    [
        ("joint7", {"multiplier": -1.0}),  # the MJCF's metre relation, read as a fraction
        ("joint7", {"multiplier": -0.04}),  # 10 mm short of the real stroke
        ("joint7", {"multiplier": 0.05}),  # wrong direction
        ("joint7", {"offset": 0.01}),
        ("joint8", {"multiplier": 0.04}),  # the driven joint itself, wrong scale
        ("joint8", {"multiplier": -0.05}),
        ("joint8", {"offset": -0.01}),
    ],
)
def test_a_wrong_relation_is_still_refused(
    metal: tuple[RobotDescription, object], name: str, update: dict[str, float]
) -> None:
    robot, model = metal
    with pytest.raises(ROSConfigError, match="disagree with the MJCF equality"):
        _declared_mimic_followers(
            _with_mimic(robot, name, **update), model, mjcf_coupled_joints(model)
        )


def test_a_ctrl_range_that_disagrees_with_the_mimic_rows_is_refused(
    metal: tuple[RobotDescription, object],
) -> None:
    """The conversion reads ``sim.grippers[].ctrl_range``; a stroke the rows don't match fails."""
    robot, model = metal
    assert robot.sim is not None
    grip = robot.sim.grippers[0].model_copy(update={"ctrl_range": (0.0, 0.04)})
    wrong = robot.model_copy(update={"sim": robot.sim.model_copy(update={"grippers": [grip]})})
    with pytest.raises(ROSConfigError, match="disagree with the MJCF equality"):
        _declared_mimic_followers(wrong, model, mjcf_coupled_joints(model))


def test_an_actuator_that_does_not_drive_the_leader_is_refused(
    metal: tuple[RobotDescription, object],
) -> None:
    """``actuator_index`` 0 drives joint1, not joint8: the fraction cannot be converted."""
    robot, model = metal
    assert robot.sim is not None
    grip = robot.sim.grippers[0].model_copy(update={"actuator_index": 0})
    wrong = robot.model_copy(update={"sim": robot.sim.model_copy(update={"grippers": [grip]})})
    with pytest.raises(ROSConfigError, match="does not drive"):
        _declared_mimic_followers(wrong, model, mjcf_coupled_joints(model))


def test_the_logical_finger_pair_carries_no_primitive(
    metal: tuple[RobotDescription, object],
) -> None:
    """``finger_pair`` maps to joint8's body, which the ``joint8`` mimic row owns: the kernel
    would slide a ``finger_pair`` primitive by the FRACTION in metres, so it gets none."""
    robot, _ = metal
    links = {
        g.link_name for g in fit_collision_geometry_from_mjcf(robot, manifest_dir=_MANIFEST.parent)
    }
    assert "finger_pair" not in links
    assert {"link7", "link8", "gripper_base", "base_link"} <= links


def test_the_two_mimic_jaws_are_exempt_from_each_other_and_their_mount_only(
    metal: tuple[RobotDescription, object],
) -> None:
    """``link8`` rides the MJCF's driven joint, so it is the leader jaw ``finger_pair`` stands
    for: the closed jaws meet by design (as the OpenArm's two fingers do), and that pair alone
    joins the mount rows — every other finger pair stays CHECKED."""
    robot, _ = metal
    if not robot.collision_geometry:
        pytest.skip("the manifest carries no collision_geometry yet")
    lowered = lower_robot_from_mjcf(
        robot, n_samples=50, manifest_dir=_MANIFEST.parent, acm_only=True
    )
    pairs = {frozenset(p) for p in lowered.allowed_collision_pairs}
    for finger, other in (("link7", "link8"), ("link8", "link7")):
        exempt = {o for p in pairs if finger in p for o in p if o != finger}
        assert exempt == {"gripper_base", other}, exempt
