"""Unit tests for the stateless helpers in the RoboCasa scene adapter.

These cover the pieces of `openral_sim.backends.robocasa` that do
NOT require robocasa / robosuite to be installed:

* `_validate_backend_options` -- the typed `RoboCasaBackendOptions`
  wrapper around `SceneSpec.backend_options`.
* `_resolve_env_name` -- the (scene_id, opts) -> robosuite env_name
  resolver (prebuilt vs procedural).
* Scene-id registration -- the curated atomic-task ids land in
  `SCENES` with `fixed_robot="panda_mobile"`.
* `read_panda_mobile_base_velocity` -- de-rotates qvel into the body
  frame using the OmronMobileBase joint names.
* `synthesize_laser_scan_2d` -- ``mj_multiRay`` fan against a synthetic
  MuJoCo XML with one known-distance box.

CLAUDE.md §1.11 -- real schemas, real registry, real MuJoCo bindings.
The mujoco import is gated on ``pytest.importorskip`` so this module
still loads on hosts without the optional dep.
"""

from __future__ import annotations

import numpy as np
import pytest
from openral_core import RoboCasaBackendOptions, SceneSpec
from openral_core.exceptions import ROSConfigError
from openral_sim import SCENES
from openral_sim.backends.robocasa import (
    _CURATED_PREBUILT_TASKS,
    _resolve_env_name,
    _validate_backend_options,
    read_panda_mobile_base_velocity,
    synthesize_laser_scan_2d,
)


def _make_scene(scene_id: str, backend_options: dict[str, object] | None = None) -> SceneSpec:
    """Build a minimal SceneSpec for the helper tests."""
    return SceneSpec.model_validate({"id": scene_id, "backend_options": backend_options or {}})


def test_validate_backend_options_round_trip_prebuilt() -> None:
    """A valid prebuilt block round-trips through the typed validator."""
    scene = _make_scene(
        "robocasa/PickPlaceCounterToCabinet",
        {"mode": "prebuilt", "prebuilt_task": "PickPlaceCounterToCabinet"},
    )
    opts = _validate_backend_options(scene)
    assert opts.mode == "prebuilt"
    assert opts.prebuilt_task == "PickPlaceCounterToCabinet"


def test_validate_backend_options_round_trip_procedural() -> None:
    """A valid procedural block round-trips through the typed validator."""
    scene = _make_scene(
        "robocasa",
        {
            "mode": "procedural",
            "task_verb": "pnp",
            "kitchen_style": 2,
            "layout_id": 5,
        },
    )
    opts = _validate_backend_options(scene)
    assert opts.mode == "procedural"
    assert opts.task_verb == "pnp"
    assert opts.kitchen_style == 2


def test_validate_backend_options_wraps_pydantic_error() -> None:
    """A malformed backend_options block surfaces as a typed ROSConfigError."""
    scene = _make_scene(
        "robocasa/PickPlaceCounterToCabinet",
        {"mode": "prebuilt"},  # missing prebuilt_task
    )
    with pytest.raises(ROSConfigError) as excinfo:
        _validate_backend_options(scene)
    assert "RoboCasaBackendOptions" in str(excinfo.value)


def test_resolve_env_name_prebuilt() -> None:
    """`robocasa/<Task>` resolves to the trailing task name verbatim."""
    opts = RoboCasaBackendOptions(mode="prebuilt", prebuilt_task="PickPlaceCounterToCabinet")
    assert (
        _resolve_env_name(opts, "robocasa/PickPlaceCounterToCabinet") == "PickPlaceCounterToCabinet"
    )


def test_resolve_env_name_prebuilt_mismatch_rejected() -> None:
    """Scene-id task name and `backend_options.prebuilt_task` must agree."""
    opts = RoboCasaBackendOptions(mode="prebuilt", prebuilt_task="OpenDoor")
    with pytest.raises(ROSConfigError) as excinfo:
        _resolve_env_name(opts, "robocasa/PickPlaceCounterToCabinet")
    assert "disagrees with" in str(excinfo.value)


def test_resolve_env_name_procedural_dispatch() -> None:
    """Each `task_verb` resolves to its registered atomic env."""
    cases = [
        ("pnp", "PickPlaceCounterToCabinet"),
        ("open", "OpenDoor"),
        ("close", "CloseDoor"),
        ("press", "TurnOnMicrowave"),
        ("navigate", "NavigateKitchen"),
    ]
    for verb, env in cases:
        opts = RoboCasaBackendOptions(mode="procedural", task_verb=verb)  # type: ignore[arg-type]
        assert _resolve_env_name(opts, "robocasa") == env


def test_resolve_env_name_procedural_requires_procedural_scene_id() -> None:
    """Passing a `robocasa/<Task>` scene id while mode=procedural is rejected."""
    opts = RoboCasaBackendOptions(mode="procedural", task_verb="pnp")
    with pytest.raises(ROSConfigError):
        _resolve_env_name(opts, "robocasa/PickPlaceCounterToCabinet")


def test_resolve_env_name_unknown_scene_id() -> None:
    """A scene id outside the `robocasa` / `robocasa/<Task>` shape is rejected."""
    opts = RoboCasaBackendOptions(mode="prebuilt", prebuilt_task="X")
    with pytest.raises(ROSConfigError):
        _resolve_env_name(opts, "kitchen/foo")


def test_curated_tasks_register_with_panda_mobile() -> None:
    """Every curated atomic task ends up in `SCENES` with `fixed_robot=panda_mobile`."""
    for task in _CURATED_PREBUILT_TASKS:
        scene_id = f"robocasa/{task}"
        assert scene_id in SCENES, f"missing scene id {scene_id!r}"
        assert SCENES.fixed_robot(scene_id) == "panda_mobile"


def test_procedural_scene_id_registered() -> None:
    """The bare `robocasa` procedural scene id is registered with the fixed robot."""
    assert "robocasa" in SCENES
    assert SCENES.fixed_robot("robocasa") == "panda_mobile"


def test_load_or_build_env_robot_guard_for_robocasa(tmp_path: pytest.TempPathFactory) -> None:
    """A SimScene YAML with ``robot_id:`` on a RoboCasa scene fails the guard.

    The RoboCasa adapter declares ``fixed_robot="panda_mobile"``; carrying
    any ``robot_id:`` on the scene-side YAML raises a typed
    :class:`ROSConfigError` at config-build time via
    :func:`openral_sim.cli._load_or_build_env`.
    """
    import tempfile
    from types import SimpleNamespace

    from openral_sim.cli import _load_or_build_env

    # Write a SimScene YAML (no vla: block!) that wrongly carries
    # robot_id on a fixed-robot scene.
    yaml_text = """\
robot_id: franka_panda
scene:
  id: robocasa/PickPlaceCounterToCabinet
  backend: mujoco
  backend_options:
    mode: prebuilt
    prebuilt_task: PickPlaceCounterToCabinet
task:
  id: robocasa/PickPlaceCounterToCabinet/0
  scene_id: robocasa/PickPlaceCounterToCabinet
  instruction: pick the object and place it in the cabinet
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write(yaml_text)
        path = tf.name
    args = SimpleNamespace(
        config=path,
        rskill="rskills/pi05-robocasa365-human300-nf4",
        robot=None,
        task=None,
        instruction=None,
        max_steps=None,
        n_episodes=None,
        n_action_steps=None,
        seed=None,
        device=None,
        save_dir=None,
        save_video=None,
    )
    with pytest.raises(ROSConfigError) as excinfo:
        _load_or_build_env(args)
    msg = str(excinfo.value)
    assert "robocasa/PickPlaceCounterToCabinet" in msg
    assert "panda_mobile" in msg


# ── PandaMobile base-velocity + LaserScan helpers (ADR-0025) ───────────────
#
# Hermetic — uses a tiny synthetic MJCF that declares the
# OmronMobileBase joint names and one obstacle at a known distance so
# we can assert per-beam ranges and de-rotated body-frame velocity
# without a full robosuite / robocasa install. CLAUDE.md §1.11 — the
# MuJoCo Python bindings are the real component under test; mujoco is
# imported via importorskip so the file still loads on hosts without
# the dep.

# Box is at world x=2 m, on the +x axis. The base sits at origin with
# zero yaw, so the beam pointing along the body +x axis is the one
# expected to report ≈ 2 m. With ``_LASER_DEFAULT_N_BEAMS = 360``, the
# +x beam is index 180 (angles run -π → +π exclusive, evenly spaced).
_KNOWN_BOX_DISTANCE_M = 2.0
_KNOWN_BOX_HALF_EXTENT_M = 0.10
_POSITIVE_X_BEAM_INDEX = 180


_PANDA_MOBILE_MJCF = f"""
<mujoco model="omron_mobile_base_stub">
  <option timestep="0.01"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="joint_mobile_forward" type="slide" axis="1 0 0"/>
      <joint name="joint_mobile_side"    type="slide" axis="0 1 0"/>
      <joint name="joint_mobile_yaw"     type="hinge" axis="0 0 1"/>
      <geom  name="base_geom" type="cylinder" size="0.30 0.10"/>
    </body>
    <body name="obstacle" pos="{_KNOWN_BOX_DISTANCE_M} 0 0.30">
      <geom name="obstacle_geom" type="box" size="{_KNOWN_BOX_HALF_EXTENT_M} 1.0 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


def _build_synthetic_panda_mobile_model() -> tuple[object, object]:
    """Compile and return ``(model, data)`` for the synthetic MJCF.

    Skips the calling test cleanly when ``mujoco`` isn't installed (the
    sim group is optional on dev hosts that aren't running the robocasa
    backend).
    """
    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    import mujoco as _mj

    model = _mj.MjModel.from_xml_string(_PANDA_MOBILE_MJCF)
    data = _mj.MjData(model)
    _mj.mj_forward(model, data)
    return model, data


def test_read_panda_mobile_base_velocity_zero_at_rest() -> None:
    """A freshly-built model at qvel == 0 reports a zero body-frame twist."""
    model, data = _build_synthetic_panda_mobile_model()
    twist = read_panda_mobile_base_velocity(model, data)
    assert isinstance(twist, np.ndarray)
    assert twist.shape == (3,)
    assert twist.dtype == np.float32
    assert np.allclose(twist, np.zeros(3, dtype=np.float32))


def test_read_panda_mobile_base_velocity_derotates_into_body_frame() -> None:
    """A 90° yaw turns a world-frame +x velocity into a body-frame -y velocity.

    Construct: yaw = +π/2 (robot pointing along +y); world-frame qvel
    has vx = 1.0 m/s; the body-frame should read (vx_body, vy_body) ≈
    (0, -1) per the rotation R(-yaw) · v_world. Use math:
    R(-π/2) · [1, 0] = [cos(-π/2)*1 - sin(-π/2)*0,
                        sin(-π/2)*1 + cos(-π/2)*0]
                     = [0, -1].
    """
    import math

    import mujoco as _mj

    model, data = _build_synthetic_panda_mobile_model()
    # Yaw joint qpos addr; the model auto-prefixes per the synthetic MJCF.
    yaw_jid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "joint_mobile_yaw")
    fwd_jid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "joint_mobile_forward")
    data.qpos[int(model.jnt_qposadr[yaw_jid])] = math.pi / 2.0
    data.qvel[int(model.jnt_dofadr[fwd_jid])] = 1.0
    _mj.mj_forward(model, data)

    twist = read_panda_mobile_base_velocity(model, data)
    assert twist.shape == (3,)
    np.testing.assert_allclose(twist[:2], np.array([0.0, -1.0], dtype=np.float32), atol=1e-6)
    assert float(twist[2]) == 0.0


def test_synthesize_laser_scan_2d_finds_known_obstacle() -> None:
    """The body-frame +x beam reports the known box distance ± a small slack.

    The synthetic MJCF puts a 0.10 m half-extent box at world x =
    ``_KNOWN_BOX_DISTANCE_M``, so the +x beam should hit the box's near
    face at ``_KNOWN_BOX_DISTANCE_M - _KNOWN_BOX_HALF_EXTENT_M`` (1.90 m).
    The neighbouring beams should be within a few cm of the same value
    (the box is much wider than the angular step).
    """
    model, data = _build_synthetic_panda_mobile_model()
    ranges = synthesize_laser_scan_2d(model=model, data=data, n_beams=360, max_range_m=10.0)
    assert isinstance(ranges, np.ndarray)
    assert ranges.shape == (360,)
    assert ranges.dtype == np.float32
    assert np.all(ranges >= 0.0)
    assert np.all(ranges <= 10.0)

    expected_near_face = _KNOWN_BOX_DISTANCE_M - _KNOWN_BOX_HALF_EXTENT_M
    forward_beam = float(ranges[_POSITIVE_X_BEAM_INDEX])
    assert abs(forward_beam - expected_near_face) < 0.05, (
        f"forward beam should hit box near face at ~{expected_near_face} m; got {forward_beam}"
    )


# Spawn-rotation regression (the SLAM-map-rotated-vs-kitchen bug, ADR-0025).
# Under a composed robocasa scene the robot is placed by rotating the
# ``mobilebase0`` body to its spawn facing while the ``mobile_yaw`` joint
# stays at 0. The scan origin already reads the body's WORLD pose
# (``data.xpos``) for exactly this reason — the heading must likewise come
# from the body's WORLD orientation (``data.xmat``), not the bare yaw-joint
# qpos. Here the base spawns rotated +90° about z with the yaw joint at 0,
# so the body +x axis points along WORLD +y; the obstacle sits on world +y
# with its near face at ``_KNOWN_BOX_DISTANCE_M - _KNOWN_BOX_HALF_EXTENT_M``.
_SPAWN_ROTATED_PANDA_MOBILE_MJCF = f"""
<mujoco model="omron_mobile_base_spawn_rotated">
  <compiler angle="degree"/>
  <worldbody>
    <body name="base" pos="0 0 0" euler="0 0 90">
      <joint name="joint_mobile_forward" type="slide" axis="1 0 0"/>
      <joint name="joint_mobile_side"    type="slide" axis="0 1 0"/>
      <joint name="joint_mobile_yaw"     type="hinge" axis="0 0 1"/>
      <geom  name="base_geom" type="cylinder" size="0.30 0.10"/>
    </body>
    <body name="obstacle" pos="0 {_KNOWN_BOX_DISTANCE_M} 0.30">
      <geom name="obstacle_geom" type="box" size="1.0 {_KNOWN_BOX_HALF_EXTENT_M} 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_synthesize_laser_scan_2d_uses_world_orientation_under_spawn_rotation() -> None:
    """Scan heading follows the base body's WORLD orientation, not the yaw joint.

    Regression for the occupancy map appearing rotated relative to the
    simulated kitchen: under a composed robocasa scene the ``mobilebase0``
    body carries the robot's spawn facing while the ``mobile_yaw`` joint
    stays at 0. The origin already reads the world body pose
    (``data.xpos``); the heading must likewise come from the body's world
    rotation (``data.xmat``). Here the base spawns at +90° about z with the
    yaw joint at 0, so the body +x axis points along WORLD +y. The obstacle
    is on world +y, so the +x-body beam (index 180) must report its near
    face. Reading yaw from the joint qpos (=0) would aim that beam at world
    +x — empty — and report ``max_range``.
    """
    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    import mujoco as _mj

    model = _mj.MjModel.from_xml_string(_SPAWN_ROTATED_PANDA_MOBILE_MJCF)
    data = _mj.MjData(model)
    _mj.mj_forward(model, data)

    ranges = synthesize_laser_scan_2d(model=model, data=data, n_beams=360, max_range_m=10.0)
    expected_near_face = _KNOWN_BOX_DISTANCE_M - _KNOWN_BOX_HALF_EXTENT_M
    forward_beam = float(ranges[_POSITIVE_X_BEAM_INDEX])
    assert abs(forward_beam - expected_near_face) < 0.05, (
        f"+x-body beam should follow the base's world spawn orientation and hit "
        f"the world-+y obstacle near face at ~{expected_near_face} m; got {forward_beam}. "
        "Scan heading is using the yaw-joint qpos instead of the body's world xmat."
    )


def test_read_panda_mobile_base_velocity_correct_under_spawn_rotation() -> None:
    """Body-frame base twist matches MuJoCo ground truth despite a spawn rotation.

    Companion to the scan-heading fix: unlike the scan *origin* (a world
    position with no cancelling partner), the velocity reader de-rotates
    the base ``qvel`` with the yaw-*joint* qpos and is still correct under a
    composed-scene spawn rotation, because the spawn term cancels. The
    slide-joint axes live in the base body's static (spawn-rotated) frame, so
    ``qvel[forward/side]`` are already along the spawn-rotated axes
    (``v_world = Rz(spawn) @ (qf, qs)``); the body's world yaw is
    ``spawn + joint_yaw``, so ``v_body = Rz(-(spawn+joint_yaw)) @ Rz(spawn) @
    (qf, qs) = Rz(-joint_yaw) @ (qf, qs)`` — exactly the qpos-yaw de-rotation.

    This locks that invariant against MuJoCo's own ``mj_objectVelocity``
    ground truth (the body-local linear velocity), so a future "consistency"
    refactor that switches this reader to a world-yaw de-rotation (which would
    be WRONG here) fails loudly.
    """
    import math

    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    import mujoco as _mj

    model = _mj.MjModel.from_xml_string(_SPAWN_ROTATED_PANDA_MOBILE_MJCF)
    data = _mj.MjData(model)

    yaw_jid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "joint_mobile_yaw")
    fwd_jid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "joint_mobile_forward")
    data.qpos[int(model.jnt_qposadr[yaw_jid])] = math.radians(30.0)
    data.qvel[int(model.jnt_dofadr[fwd_jid])] = 1.0
    _mj.mj_forward(model, data)

    # Ground truth: the base body's linear velocity in its OWN local frame —
    # i.e. the base_link-frame twist the /odom + state assemblers expect.
    base_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, "base")
    obj_vel = np.zeros(6, dtype=np.float64)
    _mj.mj_objectVelocity(model, data, _mj.mjtObj.mjOBJ_BODY, base_id, obj_vel, 1)
    gt_linear_xy = obj_vel[3:5]  # [vx, vy] in body-local frame
    gt_angular_z = float(obj_vel[2])

    twist = read_panda_mobile_base_velocity(model, data)
    assert twist.shape == (3,)
    np.testing.assert_allclose(twist[:2], gt_linear_xy, atol=1e-6)
    np.testing.assert_allclose(float(twist[2]), gt_angular_z, atol=1e-6)


_EMPTY_PANDA_MOBILE_MJCF = """
<mujoco model="omron_mobile_base_alone">
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="joint_mobile_forward" type="slide" axis="1 0 0"/>
      <joint name="joint_mobile_side"    type="slide" axis="0 1 0"/>
      <joint name="joint_mobile_yaw"     type="hinge" axis="0 0 1"/>
      <geom  name="base_geom" type="cylinder" size="0.30 0.10"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_synthesize_laser_scan_2d_excludes_self() -> None:
    """The robot's own chassis (0.30 m radius cylinder) does NOT pollute the scan.

    Build a world that contains only the chassis. With ``bodyexclude``
    set to the base body id, every beam should report the cutoff (no
    other geometry to hit). If the self-exclusion broke, beams would
    instead report ≈ 0.30 m (the chassis radius).
    """
    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    import mujoco as _mj

    model = _mj.MjModel.from_xml_string(_EMPTY_PANDA_MOBILE_MJCF)
    data = _mj.MjData(model)
    _mj.mj_forward(model, data)
    ranges = synthesize_laser_scan_2d(model=model, data=data, n_beams=12, max_range_m=1.5)
    assert ranges.shape == (12,)
    np.testing.assert_allclose(ranges, np.full(12, 1.5, dtype=np.float32), atol=1e-6)


_MULTIBODY_PANDA_MOBILE_MJCF = """
<mujoco model="omron_mobile_base_multibody">
  <worldbody>
    <body name="base" pos="0 0 0">
      <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
      <joint name="joint_mobile_forward" type="slide" axis="1 0 0"/>
      <joint name="joint_mobile_side"    type="slide" axis="0 1 0"/>
      <joint name="joint_mobile_yaw"     type="hinge" axis="0 0 1"/>
      <body name="wheeled_base" pos="0 0 0">
        <!-- half-height 0.50 so the chassis spans the 0.30 m laser
             height (mirrors the real OmronMobileBase, whose
             wheeled_base geometry the beams hit at 0.13-0.54 m). -->
        <geom name="chassis_geom" type="cylinder" size="0.30 0.50"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_synthesize_laser_scan_2d_excludes_whole_robot_tree() -> None:
    """Self-exclusion must cover the ENTIRE robot kinematic tree.

    Regression (ADR-0025 live-sim finding): in the robocasa kitchen the
    chassis collision geometry lives on ``mobilebase0_wheeled_base`` — a
    *child* of the excluded ``mobilebase0_base`` root. A single-body
    ``bodyexclude`` only dropped the (geomless) root, so every one of the
    360 beams hit the wheeled base at ~0.13-0.54 m (below ``range_min``),
    starving slam_toolbox and leaving it publishing an empty 0x0 ``/map``.

    Here the chassis cylinder (0.30 m radius) sits on a child
    ``wheeled_base`` body; the world is otherwise empty. Every beam must
    report the cutoff (1.5 m), NOT the 0.30 m chassis radius.
    """
    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    import mujoco as _mj

    model = _mj.MjModel.from_xml_string(_MULTIBODY_PANDA_MOBILE_MJCF)
    data = _mj.MjData(model)
    _mj.mj_forward(model, data)
    ranges = synthesize_laser_scan_2d(model=model, data=data, n_beams=12, max_range_m=1.5)
    assert ranges.shape == (12,)
    np.testing.assert_allclose(ranges, np.full(12, 1.5, dtype=np.float32), atol=1e-6)


# ── Generalization: non-OmronMobileBase naming via `base_joint_names` ────


_NON_OMRON_MOBILE_MJCF = f"""
<mujoco model="custom_mobile_base">
  <worldbody>
    <body name="custom_chassis" pos="0 0 0">
      <joint name="wheels_drive_forward" type="slide" axis="1 0 0"/>
      <joint name="wheels_drive_side"    type="slide" axis="0 1 0"/>
      <joint name="wheels_drive_yaw"     type="hinge" axis="0 0 1"/>
      <geom  name="chassis_geom" type="cylinder" size="0.20 0.10"/>
    </body>
    <body name="obstacle" pos="{_KNOWN_BOX_DISTANCE_M} 0 0.30">
      <geom name="obstacle_geom" type="box" size="{_KNOWN_BOX_HALF_EXTENT_M} 1.0 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_synthesize_laser_scan_2d_with_caller_supplied_names() -> None:
    """`base_joint_names` override generalises the helper to any mobile base.

    ADR-0025 — the helper is designed to work for any planar
    holonomic base whose MJCF declares (forward, side, yaw) slide/
    hinge joints. The default constants target the robosuite
    OmronMobileBase, but a caller passing
    ``base_joint_names=("wheels_drive_forward", "wheels_drive_side",
    "wheels_drive_yaw")`` should resolve a totally different MJCF.
    Hermetic confirmation that the architecture isn't single-instance
    despite no second lasered-mobile-robot existing in-tree yet.
    """
    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    import mujoco as _mj

    model = _mj.MjModel.from_xml_string(_NON_OMRON_MOBILE_MJCF)
    data = _mj.MjData(model)
    _mj.mj_forward(model, data)

    custom_names = (
        "wheels_drive_forward",
        "wheels_drive_side",
        "wheels_drive_yaw",
    )
    ranges = synthesize_laser_scan_2d(
        model=model,
        data=data,
        base_joint_names=custom_names,
        n_beams=360,
        max_range_m=10.0,
    )
    assert ranges.shape == (360,)
    expected_near_face = _KNOWN_BOX_DISTANCE_M - _KNOWN_BOX_HALF_EXTENT_M
    forward_beam = float(ranges[_POSITIVE_X_BEAM_INDEX])
    assert abs(forward_beam - expected_near_face) < 0.05, (
        f"forward beam should hit box near face at ~{expected_near_face} m; "
        f"got {forward_beam}. The override base_joint_names path is broken."
    )


def test_read_panda_mobile_base_velocity_with_caller_supplied_names() -> None:
    """`base_joint_names` override drives the velocity reader too."""
    import math

    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    import mujoco as _mj

    model = _mj.MjModel.from_xml_string(_NON_OMRON_MOBILE_MJCF)
    data = _mj.MjData(model)
    # Drive the yaw=π/2 + forward velocity setup from
    # `test_read_panda_mobile_base_velocity_derotates_into_body_frame`
    # but with the non-OmronMobileBase joint names.
    yaw_jid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "wheels_drive_yaw")
    fwd_jid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "wheels_drive_forward")
    data.qpos[int(model.jnt_qposadr[yaw_jid])] = math.pi / 2.0
    data.qvel[int(model.jnt_dofadr[fwd_jid])] = 1.0
    _mj.mj_forward(model, data)

    custom_names = ("wheels_drive_forward", "wheels_drive_side", "wheels_drive_yaw")
    twist = read_panda_mobile_base_velocity(model, data, base_joint_names=custom_names)
    assert twist.shape == (3,)
    np.testing.assert_allclose(twist[:2], np.array([0.0, -1.0], dtype=np.float32), atol=1e-6)
    assert float(twist[2]) == 0.0
