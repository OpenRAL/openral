"""The lowered collision tree must place links where the robot's own model does.

A connected collision graph is necessary but not sufficient: the links also have
to land in the *right place*. This drives the kernel's own forward kinematics
over the lowered ROS parameters and compares every link's world position at the
zero configuration against the same body in the robot's real MuJoCo model.

This is what makes the ``fixed_attachments`` transforms auditable. If someone
mis-transcribes a mount origin — or estimates one instead of reading it out of
the vendored model, which the schema forbids — the arm lands somewhere the real
robot's never does and this test says by how many millimetres.

Each robot is checked against **its own normative source** — the model its
manifest kinematics were lowered from — because a robot's URDF and its MJCF can
describe the same machine with different intermediate frames. ``g1`` is the
worked example: its URDF splits the waist as ``waist_roll +0.035`` then
``waist_pitch +0.019``, its MJCF as ``+0.044`` then ``+0.0``. Both put the
shoulder at ``z = 0.29178``; only the ``torso_link`` frame between them differs,
by 10 mm. Comparing a URDF-lowered manifest against the MJCF would measure that
convention split and call it a defect. UR is the same story with a rotation.

CLAUDE.md §1.11 — real manifests, real vendored models, no mocks.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openral_core import RobotDescription

mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed (sim extra)")

from openral_core import RobotDescription  # noqa: E402
from openral_core.assets import resolve_asset  # noqa: E402
from openral_safety.envelope_loader import collision_params_from_description  # noqa: E402
from openral_safety.mjcf_lowering import _compose, _rpy_to_mat  # noqa: E402

_ROBOTS_DIR = pathlib.Path(__file__).resolve().parents[3] / "robots"

# Tolerance is float-composition noise only. The defects this guards against
# are millimetre-scale at minimum (a mis-typed digit) and metre-scale at worst
# (a subtree dumped on the base), so 1e-5 m separates signal from arithmetic.
_TOL_M = 1e-5

# robot dir -> manifest-link prefix stripped when matching MJCF body names.
# Only robots whose manifest kinematics are in the MJCF's own frames. ``g1`` is
# URDF-lowered and is checked against its URDF below instead.
_MJCF_FRAME_ROBOTS = {
    "franka_panda": "panda_",
    "h1": "",
    "openarm": "",
}


def _link_world_positions(params: dict[str, object]) -> dict[str, tuple[float, float, float]]:
    """Run the kernel's FK over the lowered params at the all-zero configuration."""
    n = int(params["collision_n_links"])  # type: ignore[arg-type]  # reason: lowering always emits an int here
    parent = params["collision_parent"]
    origin = params["collision_origin_xyzrpy"]
    names = params["collision_link_names"]
    assert isinstance(parent, list)
    assert isinstance(origin, list)
    assert isinstance(names, list)

    rot: list[Any] = [None] * n
    trans: list[Any] = [None] * n
    for i in range(n):
        o = origin[6 * i : 6 * i + 6]
        r, t = _rpy_to_mat(o[3], o[4], o[5]), list(o[:3])
        p = int(parent[i])
        if p >= 0:
            r, t = _compose(rot[p], trans[p], r, t)
        rot[i], trans[i] = r, t
    return {str(names[i]): (trans[i][0], trans[i][1], trans[i][2]) for i in range(n)}


def _mjcf_body_positions(robot: RobotDescription, manifest_dir: pathlib.Path) -> dict[str, Any]:
    """World position of every named body in the robot's own MJCF at ``qpos = 0``."""
    path = resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=manifest_dir)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    out: dict[str, Any] = {}
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if name:
            out[name] = tuple(float(v) for v in data.xpos[body])
    return out


@pytest.mark.parametrize("robot_dir,prefix", sorted(_MJCF_FRAME_ROBOTS.items()))
def test_lowered_links_land_where_the_source_model_puts_them(robot_dir: str, prefix: str) -> None:
    """Every lowered link matching an MJCF body sits within 10 um of it."""
    manifest_dir = _ROBOTS_DIR / robot_dir
    robot = RobotDescription.from_yaml(str(manifest_dir / "robot.yaml"))
    params = collision_params_from_description(robot)
    assert params["self_collision_enabled"] is True, "fixture precondition: has geometry"

    try:
        reference = _mjcf_body_positions(robot, manifest_dir)
    except Exception as exc:  # reason: any resolve/fetch failure is a skip, not a defect
        pytest.skip(f"{robot_dir}: MJCF unavailable ({type(exc).__name__}: {exc})")

    ours = _link_world_positions(params)
    compared = 0
    for name, pos in sorted(ours.items()):
        candidates = [name]
        if prefix and name.startswith(prefix):
            candidates.append(name[len(prefix) :])
        ref = next((reference[c] for c in candidates if c in reference), None)
        if ref is None:
            # Manifest-only links (a synthetic finger pair, a mount frame the
            # MJCF expresses as the worldbody) have no body to compare against.
            continue
        compared += 1
        assert pos == pytest.approx(ref, abs=_TOL_M), (
            f"{robot_dir}: link {name!r} lowers to {pos} but its MJCF body is at "
            f"{ref} — the manifest's kinematics disagree with the robot's own model."
        )

    assert compared >= 5, f"{robot_dir}: only {compared} links matched; the check is vacuous"


def test_franka_hand_lands_at_the_flange_height() -> None:
    """The concrete regression: the gripper is ~0.93 m up, not at the base.

    With ``panda_hand`` orphaned it lowered to a second root at the identity
    frame, i.e. z = 0 — inside the robot's own pedestal.
    """
    manifest_dir = _ROBOTS_DIR / "franka_panda"
    robot = RobotDescription.from_yaml(str(manifest_dir / "robot.yaml"))
    positions = _link_world_positions(collision_params_from_description(robot))

    hand = positions["panda_hand"]
    assert hand[2] > 0.9, f"panda_hand lowered to z={hand[2]:.4f}; expected the flange, ~0.93 m"
    try:
        reference = _mjcf_body_positions(robot, manifest_dir)
    except Exception as exc:  # reason: asset fetch failure is a skip
        pytest.skip(f"MJCF unavailable ({type(exc).__name__}: {exc})")
    assert hand == pytest.approx(reference["hand"], abs=_TOL_M)


def test_g1_arms_are_not_stacked_on_the_pelvis() -> None:
    """Both G1 wrists sit well clear of the pelvis, and level with each other.

    With ``torso_link`` orphaned, all 15 upper-body volumes lowered onto the
    pelvis origin — the arms were, to the kernel, inside the robot's hips.
    """
    robot = RobotDescription.from_yaml(str(_ROBOTS_DIR / "g1" / "robot.yaml"))
    positions = _link_world_positions(collision_params_from_description(robot))

    left = positions["left_wrist_yaw_link"]
    right = positions["right_wrist_yaw_link"]
    pelvis = positions["pelvis"]
    assert pelvis == pytest.approx((0.0, 0.0, 0.0), abs=_TOL_M)
    # Laterally separated (one per side of the body), not coincident.
    assert left[1] > 0.1
    assert right[1] < -0.1
    # Forward of and above the pelvis, i.e. actually on the torso.
    assert left[0] > 0.15
    assert left[2] > 0.05


def test_g1_lowers_exactly_onto_its_urdf() -> None:
    """``g1``'s chain reproduces its URDF — the model its manifest is lowered from.

    The URDF is normative here: ``lower_joint_fk`` reads it, and the fleet drift
    guard (``tests/unit/test_collision_lowering_fleet.py``) holds the manifest to
    what it emits. So this is the check that has teeth for ``g1`` — the waist
    values must be the URDF's, not a same-robot MJCF's.
    """
    yourdfpy = pytest.importorskip("yourdfpy", reason="URDF FK needs the [lowering] group")
    import numpy as np

    robot = RobotDescription.from_yaml(str(_ROBOTS_DIR / "g1" / "robot.yaml"))
    ours = _link_world_positions(collision_params_from_description(robot))

    urdf_path = resolve_asset(robot.assets.urdf.ref, "urdf")  # type: ignore[union-attr]  # reason: g1 declares a urdf
    model = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_collision_scene_graph=True)
    model.update_cfg(np.zeros(model.num_actuated_joints))

    compared = 0
    for name, pos in sorted(ours.items()):
        if name not in model.link_map:
            continue
        compared += 1
        ref = np.asarray(model.get_transform(name), dtype=np.float64)[:3, 3]
        assert pos == pytest.approx(tuple(ref), abs=_TOL_M), (
            f"g1 link {name!r} lowers to {pos} but its URDF places it at {tuple(ref)}"
        )
    assert compared >= 25, f"only {compared} g1 links matched; the check is vacuous"


def test_g1_urdf_and_mjcf_differ_only_in_the_torso_frame() -> None:
    """Pins *why* ``g1`` is checked against its URDF and not its MJCF.

    The two vendored models are the same robot with a different intermediate
    frame: the URDF splits the waist ``+0.035`` then ``+0.019``, the MJCF
    ``+0.044`` then ``+0.0``. Everything from the shoulders up therefore agrees
    to float noise, and only ``torso_link`` itself sits 10 mm apart. Pinning
    that keeps a future reader from "correcting" one model's numbers to the
    other's — which is exactly the mistake this test exists to prevent.
    """
    import numpy as np

    robot = RobotDescription.from_yaml(str(_ROBOTS_DIR / "g1" / "robot.yaml"))
    ours = _link_world_positions(collision_params_from_description(robot))
    try:
        reference = _mjcf_body_positions(robot, _ROBOTS_DIR / "g1")
    except Exception as exc:  # reason: asset fetch failure is a skip
        pytest.skip(f"MJCF unavailable ({type(exc).__name__}: {exc})")

    ours_torso = np.array(ours["torso_link"])
    ref_torso = np.array(reference["torso_link"])
    torso_gap = float(np.linalg.norm(ours_torso - ref_torso))
    assert torso_gap == pytest.approx(0.010, abs=1e-6), (
        "the URDF/MJCF torso convention split changed; re-derive before touching g1"
    )
    # Above the torso the two models agree, so the arms are placed identically.
    for link in ("left_shoulder_pitch_link", "left_wrist_yaw_link", "right_wrist_yaw_link"):
        assert ours[link] == pytest.approx(reference[link], abs=_TOL_M)


def test_openarm_pedestals_are_62mm_apart() -> None:
    """The two arm roots straddle the base rather than sharing its origin."""
    robot = RobotDescription.from_yaml(str(_ROBOTS_DIR / "openarm" / "robot.yaml"))
    positions = _link_world_positions(collision_params_from_description(robot))

    left = positions["openarm_left_link0"]
    right = positions["openarm_right_link0"]
    assert positions["openarm_base"] == pytest.approx((0.0, 0.0, 0.0), abs=_TOL_M)
    assert left[1] - right[1] == pytest.approx(0.062, abs=_TOL_M)
    # And the arms themselves stay separated all the way to the wrists.
    assert positions["openarm_left_link7"] != pytest.approx(
        positions["openarm_right_link7"], abs=1e-3
    )
