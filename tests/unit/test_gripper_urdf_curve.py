"""One gripper calibration curve for every consumer that poses the finger (issue #533).

``JointSpec.urdf_curve`` (or a sim scene's ``backend_options.gripper_joint_curve``, which
``sim_scene_description`` writes into it) maps the jaw's reported value onto its URDF
finger. Before #533 only the Isaac sidecar applied the scene's curve: the HAL's
``~/urdf_joint_states`` (so ``/tf``), the safety kernel and the self-filter posed the
OpenArm restock twin's finger at the reported value, up to ~0.1 rad off mid-grip. These
tests drive the real ``robots/`` manifests through each consumer and require one answer.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from openral_core import (
    JointSpec,
    RobotDescription,
    check_urdf_curve,
    curve_slope,
    curve_value,
    sim_scene_description,
    unit_gripper_curves,
)
from openral_core.assets import AssetFetchError, resolve_asset
from openral_core.exceptions import ROSConfigError
from openral_core.kinematics import link_transform
from openral_hal.lifecycle import urdf_joint_curves, urdf_named_joint_state
from openral_hal.resolver import urdf_joint_map
from openral_safety.envelope_loader import (
    collision_params_from_description,
    ee_link_index_from_collision_params,
    with_joint_curves,
)

_ROOT = Path(__file__).resolve().parents[2]
_ROBOTS = _ROOT / "robots"

#: The OpenArm restock deploy twin's curves (openral-pro scenes/deploy/openarm_restock.yaml),
#: the scene issue #533 was measured on.
_RESTOCK_CURVES: dict[str, list[tuple[float, float]]] = {
    "left_gripper": [(0.0, 0.0), (0.316, 0.42), (0.785, 0.785)],
    "right_gripper": [(0.0, 0.0), (-0.28, -0.42), (-0.785, -0.785)],
}
_GRID = np.linspace(-0.1, 0.9, 41)


def _restock_openarm() -> RobotDescription:
    base = RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))
    return sim_scene_description(base, {"gripper_joint_curve": _RESTOCK_CURVES})


def _tf_value(desc: RobotDescription, urdf: str, joint: str, q: float) -> tuple[str, float]:
    """The URDF joint and value the HAL publishes on ``~/urdf_joint_states`` (``/tf``)."""
    m = urdf_joint_map(desc, urdf)
    assert m is not None
    names = [j.name for j in desc.joints]
    msg = SimpleNamespace(
        header=None,
        name=names,
        position=[q if n == joint else 0.0 for n in names],
        velocity=[1.0 if n == joint else 0.0 for n in names],
        effort=[],
    )
    # Through the HAL's own parameter round trip, as the launch hands it over.
    curves = urdf_joint_curves(len(names), m.curves_param())
    out = urdf_named_joint_state(msg, m.names, m.scales, m.offsets, curves)
    i = names.index(joint)
    k = list(out.name).index(m.names[i])
    return m.names[i], float(out.position[k])


def _seq(params: Mapping[str, object], key: str) -> list[Any]:
    """One ``collision_*`` param array (``[]`` when absent)."""
    return list(cast("list[Any]", params.get(key, [])))


def _kernel_link_value(params: dict[str, object], link: str, q: float) -> float:
    """The joint value the kernel's ``link_joint_value`` gives ``link`` from its params."""
    i = _seq(params, "collision_link_names").index(link)
    first = _seq(params, "collision_curve_point_first")
    count = _seq(params, "collision_curve_point_count")
    pts = np.asarray(params.get("collision_curve_points", []), dtype=np.float64).reshape(-1, 2)
    u = q
    if first and first[i] >= 0:
        seg = pts[first[i] : first[i] + count[i]]
        u = float(np.interp(q, seg[:, 0], seg[:, 1]))
    if int(_seq(params, "collision_joint_kind")[i]) in (3, 4):
        mult = _seq(params, "collision_mimic_multiplier")
        off = _seq(params, "collision_mimic_offset")
        u = mult[i] * u + off[i]
    return u


def test_the_issue_533_mid_grip_point_maps_onto_the_measured_finger() -> None:
    joints = {j.name: j for j in _restock_openarm().joints}
    assert joints["left_gripper"].urdf_value(0.222) == pytest.approx(0.295, abs=1e-3)
    assert joints["left_gripper"].urdf_value(0.762) == pytest.approx(0.767, abs=1e-3)
    assert joints["right_gripper"].urdf_value(-0.222) == pytest.approx(-0.333, abs=1e-3)
    # The scene's points arrive in any order; the joint holds them sorted by value.
    assert joints["right_gripper"].urdf_curve == ((-0.785, -0.785), (-0.28, -0.42), (0.0, 0.0))


@pytest.mark.parametrize("gripper", ["left_gripper", "right_gripper"])
def test_tf_kernel_and_core_fk_pose_the_openarm_finger_at_the_sims_angle(gripper: str) -> None:
    desc = _restock_openarm()
    urdf = (_ROBOTS / "openarm" / "openarm.urdf").read_text()
    joint = next(j for j in desc.joints if j.name == gripper)
    follower = next(m for m in desc.mimic_joints if m.mimics == gripper)
    params = collision_params_from_description(desc)
    plain = RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))
    sign = 1.0 if gripper == "left_gripper" else -1.0
    for q in sign * _GRID:
        assert joint.urdf_curve is not None
        xs = [x for x, _ in joint.urdf_curve]
        us = [u for _, u in joint.urdf_curve]
        sim = float(np.interp(q, xs, us))  # the sidecar's map
        name, tf = _tf_value(desc, urdf, gripper, float(q))
        assert name == joint.sim_joint_name
        assert tf == pytest.approx(sim, abs=1e-9), f"/tf at q={q}"
        assert _kernel_link_value(params, joint.child_link, float(q)) == pytest.approx(sim)
        assert _kernel_link_value(params, follower.child_link, float(q)) == pytest.approx(
            follower.value_at(sim)
        )
        # openral_core.kinematics (the vision leg, the release window) agrees too.
        at_sim = link_transform(desc, joint.parent_link, joint.child_link, {gripper: float(q)})
        expected = link_transform(plain, joint.parent_link, joint.child_link, {gripper: sim})
        np.testing.assert_allclose(at_sim, expected, atol=1e-12)


def test_the_so101_bench_unit_poses_the_real_jaw_by_its_servo_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """lerobot's gripper 0..100 spans the unit's calibrated servo range, closed at the
    URDF lower limit: /tf (the dashboard's 3D jaw) and the kernel follow it, not Isaac's
    1.74533 f, which rendered the closed real jaw ~10 deg open."""
    monkeypatch.delenv("OPENRAL_ROBOT_UNIT", raising=False)
    robot_yaml = _ROBOTS / "so101_follower" / "robot.yaml"
    plain = RobotDescription.from_yaml(str(robot_yaml))
    assert unit_gripper_curves(plain, robot_yaml, None) is plain
    desc = unit_gripper_curves(plain, robot_yaml, "bench_laptop")
    cal = json.loads((_ROOT / "scenes/deploy/calibration/so_follower.json").read_text())
    span = (cal["gripper"]["range_max"] - cal["gripper"]["range_min"]) * 2 * math.pi / 4096
    urdf = (_ROBOTS / "so101_follower" / "so101_follower.urdf").read_text()
    params = collision_params_from_description(desc)
    for f in (0.0, 0.2612, 1.0):
        jaw = -0.174533 + span * f
        name, tf = _tf_value(desc, urdf, "gripper", f)
        assert name == "gripper"
        assert tf == pytest.approx(jaw, abs=1e-4)
        assert _kernel_link_value(params, "jaw", f) == pytest.approx(jaw, abs=1e-4)


def test_the_sidecar_drives_the_finger_the_kernel_checks(_manifest_scene_mod: Any) -> None:
    """The Isaac command map lands the finger on the curve the kernel and /tf read."""
    from openral_sim.backends.isaac_sim import _build_robot_spec, _curve_gripper_specs

    desc = _restock_openarm()
    spec = _build_robot_spec(
        RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml")), "openarm"
    )
    _curve_gripper_specs(spec, _RESTOCK_CURVES)
    g = next(g for g in spec["grippers"] if g["name"] == "left_gripper")
    joint = next(j for j in desc.joints if j.name == "left_gripper")
    for q in np.linspace(0.0, 0.785, 9):
        assert _manifest_scene_mod.manifest_to_urdf_gripper(g, float(q)) == pytest.approx(
            joint.urdf_value(float(q))
        )


def test_without_a_scene_curve_the_openarm_is_unchanged() -> None:
    """Real hardware and the MuJoCo twin: no curve, so nothing a consumer sees moves."""
    desc = RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))
    assert not any(j.urdf_curve for j in desc.joints)
    params = collision_params_from_description(desc)
    assert not any(k.startswith("collision_curve_") for k in params)
    urdf = (_ROBOTS / "openarm" / "openarm.urdf").read_text()
    m = urdf_joint_map(desc, urdf)
    assert m is None or m.curves_param() == []


@pytest.mark.parametrize(
    ("robot", "urdf_joint", "closed", "opened"),
    [
        ("droid_franka_robotiq", "finger_joint", 0.7853982, 0.0),
        ("makermods_metal", "joint7", -0.0, -0.05),
    ],
)
def test_manifest_curves_drive_tf_and_the_kernel_alike(
    robot: str, urdf_joint: str, closed: float, opened: float
) -> None:
    """DROID and Metal carry their curves in the manifest; the rows read the URDF value."""
    desc = RobotDescription.from_yaml(str(_ROBOTS / robot / "robot.yaml"))
    assert desc.assets.urdf is not None
    try:
        urdf_file = resolve_asset(desc.assets.urdf.ref, "urdf", manifest_dir=_ROBOTS / robot)
    except (AssetFetchError, ROSConfigError) as exc:  # a clone needs network on first use
        pytest.skip(f"{robot} URDF unavailable: {exc}")
    assert urdf_file is not None
    gripper = next(j for j in desc.joints if j.role == "gripper")
    params = collision_params_from_description(desc)
    for f, expected in ((0.0, closed), (1.0, opened)):
        name, tf = _tf_value(desc, Path(urdf_file).read_text(), gripper.name, f)
        assert name == urdf_joint
        assert tf == pytest.approx(expected, abs=1e-7)
        for row in desc.mimic_joints:
            assert _kernel_link_value(params, row.child_link, f) == pytest.approx(
                row.value_at(gripper.urdf_value(f))
            )


def test_a_scene_curve_replaces_the_manifest_curve_for_its_twin() -> None:
    """The Metal Isaac scenes close below f = 0.025; the MuJoCo twin's manifest curve does not."""
    base = RobotDescription.from_yaml(str(_ROBOTS / "makermods_metal" / "robot.yaml"))
    scene = sim_scene_description(
        base, {"gripper_joint_curve": {"gripper": [[0.025, 0.0], [1.025, 0.05]]}}
    )
    assert next(j for j in base.joints if j.name == "gripper").urdf_value(0.5) == pytest.approx(
        0.025
    )
    assert next(j for j in scene.joints if j.name == "gripper").urdf_value(0.5) == pytest.approx(
        0.02375
    )
    params = collision_params_from_description(scene)
    assert _seq(params, "collision_curve_points")[:4] == [0.025, 0.0, 1.025, 0.05]


def test_the_ee_link_never_sits_below_a_curved_joint() -> None:
    """The kernel's EE Jacobian cannot predict through a curved link; the picker skips it."""
    params = collision_params_from_description(_restock_openarm())
    ee = ee_link_index_from_collision_params(params)
    first = _seq(params, "collision_curve_point_first")
    parent = _seq(params, "collision_parent")
    link = ee
    while link >= 0:
        assert first[link] < 0, _seq(params, "collision_link_names")[link]
        link = parent[link]


def test_with_joint_curves_is_idempotent_and_drops_stale_arrays() -> None:
    desc = _restock_openarm()
    params = collision_params_from_description(desc)
    assert with_joint_curves(params, desc) == params
    plain = RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))
    assert not any(k.startswith("collision_curve_") for k in with_joint_curves(params, plain))


def test_the_hal_rate_follows_the_curve_slope() -> None:
    desc = _restock_openarm()
    urdf = (_ROBOTS / "openarm" / "openarm.urdf").read_text()
    m = urdf_joint_map(desc, urdf)
    assert m is not None
    names = [j.name for j in desc.joints]
    i = names.index("left_gripper")
    msg = SimpleNamespace(
        header=None,
        name=names,
        position=[0.1 if n == "left_gripper" else 0.0 for n in names],
        velocity=[2.0 if n == "left_gripper" else 0.0 for n in names],
        effort=[],
    )
    out = urdf_named_joint_state(msg, m.names, m.scales, m.offsets, m.curves)
    k = list(out.name).index(m.names[i])
    assert out.velocity[k] == pytest.approx(2.0 * 0.42 / 0.316)


@pytest.mark.parametrize(
    ("points", "match"),
    [
        ([(0.0, 0.0)], "at least 2"),
        ([(0.0, 0.0), (0.0, 0.1)], "strictly"),
        ([(0.0, 0.0), (0.5, 0.2), (1.0, 0.1)], "strictly"),
        ([(0.0, 0.0), (1.0, float("nan"))], "finite"),
    ],
)
def test_bad_curves_are_refused_everywhere(points: list[tuple[float, float]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_urdf_curve(points, "curve")
    with pytest.raises(ValueError, match=match):
        JointSpec(
            name="g",
            joint_type="revolute",
            parent_link="a",
            child_link="b",
            role="gripper",
            urdf_curve=tuple(points),
        )
    base = RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))
    with pytest.raises(ROSConfigError, match="gripper_joint_curve"):
        sim_scene_description(base, {"gripper_joint_curve": {"left_gripper": points}})


def test_a_curve_needs_a_gripper_joint() -> None:
    with pytest.raises(ValueError, match="role 'gripper'"):
        JointSpec(
            name="j",
            joint_type="revolute",
            parent_link="a",
            child_link="b",
            urdf_curve=((0.0, 0.0), (1.0, 1.0)),
        )
    base = RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))
    with pytest.raises(ROSConfigError, match="names no gripper joint"):
        sim_scene_description(base, {"gripper_joint_curve": {"left_joint1": [[0, 0], [1, 1]]}})


def test_curve_value_and_slope_clamp_like_numpy_interp() -> None:
    pts = [(0.0, 0.0), (0.316, 0.42), (0.785, 0.785)]
    xs, us = zip(*pts, strict=True)
    for x in np.linspace(-0.5, 1.5, 81):
        assert curve_value(pts, float(x)) == pytest.approx(float(np.interp(x, xs, us)))
    assert curve_slope(pts, -0.1) == 0.0
    assert curve_slope(pts, 1.0) == 0.0
    assert curve_slope(pts, 0.5) == pytest.approx((0.785 - 0.42) / (0.785 - 0.316))


def test_the_hal_curve_parameter_refuses_a_bad_layout() -> None:
    assert urdf_joint_curves(2, [float("nan")]) == []
    with pytest.raises(ROSConfigError, match="covers 1 joints"):
        urdf_joint_curves(2, [2.0, 0.0, 0.0, 1.0, 1.0])
    with pytest.raises(ROSConfigError, match="bad point count"):
        urdf_joint_curves(1, [3.0, 0.0, 0.0])
    with pytest.raises(ROSConfigError, match="strictly"):
        urdf_joint_curves(1, [2.0, 1.0, 0.0, 0.0, 1.0])


@pytest.fixture
def _manifest_scene_mod() -> object:
    tools = str(_ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import isaac_manifest_scene

    return isaac_manifest_scene


def test_a_nan_reading_stays_nan_through_the_curve() -> None:
    """The clamp must not turn an invalid jaw reading into a finite finger pose."""
    pts = [(0.0, 0.0), (0.316, 0.42), (0.785, 0.785)]
    assert np.isnan(curve_value(pts, float("nan")))
    assert np.isnan(curve_slope(pts, float("nan")))
    base = RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))
    with pytest.raises(ROSConfigError, match="pair"):
        sim_scene_description(
            base, {"gripper_joint_curve": {"left_gripper": [[0, 0, 99], [1, 0.05, 99]]}}
        )
