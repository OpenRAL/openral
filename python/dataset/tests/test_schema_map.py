"""Tests for ``openral_dataset.schema_map.features_from_robot``.

Per CLAUDE.md §1.11: loads real ``RobotDescription`` manifests from
``robots/*/robot.yaml``. No mocks, no synthetic robots.

The SO-100 tests cover the happy path (a robot with full
observation_spec + action_spec + 2 cameras). The parametrized
``test_every_robot_manifest_*`` cases sweep every
``robots/*/robot.yaml`` in the repo to catch schema drift on the other
10 manifests (Aloha bimanual, Franka Panda, UR5e/UR10e, PushT,
Sawyer, GR1, WidowX, Google Robot, Panda Mobile). Robots without
observation_spec / action_spec MUST raise a typed ValueError with an
actionable message — the converter relies on this for clean
diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_dataset.schema_map import FeatureSpec, features_from_robot


def test_so100_features_have_state_and_action(so100_robot: RobotDescription) -> None:
    feats = features_from_robot(so100_robot, fps=30.0)

    state = feats["observation.state"]
    assert state.dtype == "float32"
    assert state.shape == (6,), f"SO-100 state shape should be (6,); got {state.shape!r}"

    action = feats["action"]
    assert action.dtype == "float32"
    assert action.shape == (6,), f"SO-100 action dim should be 6; got {action.shape!r}"


def test_so100_features_include_both_cameras(so100_robot: RobotDescription) -> None:
    feats = features_from_robot(so100_robot, fps=30.0)

    assert "observation.images.camera1" in feats, (
        "SO-100 manifest declares vla_feature_key=observation.images.camera1 on the front camera"
    )
    assert "observation.images.camera2" in feats, (
        "SO-100 manifest declares vla_feature_key=observation.images.camera2 on the wrist camera"
    )
    for cam_key in ("observation.images.camera1", "observation.images.camera2"):
        cam = feats[cam_key]
        assert cam.dtype == "video"
        # ADR-0019: shape comes from SensorSpec.intrinsics. SO-100's
        # manifest declares 256x256 for both cameras.
        assert cam.shape == (256, 256, 3)


def test_so100_features_include_bookkeeping_columns(so100_robot: RobotDescription) -> None:
    feats = features_from_robot(so100_robot, fps=30.0)

    for key in ("next.reward", "next.done", "next.success", "next.terminated", "next.truncated"):
        assert key in feats, f"missing canonical bookkeeping feature {key!r}"
        # v3 rejects shape (); these must be (1,).
        assert feats[key].shape == (1,), (
            f"{key!r} must have shape (1,) for lerobot v3 compatibility; got {feats[key].shape!r}"
        )

    assert feats["next.reward"].dtype == "float32"
    assert feats["next.success"].dtype == "bool"
    assert feats["next.done"].dtype == "bool"


def test_fps_must_be_positive(so100_robot: RobotDescription) -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        features_from_robot(so100_robot, fps=0.0)
    with pytest.raises(ValueError, match="fps must be positive"):
        features_from_robot(so100_robot, fps=-30.0)


def test_features_are_featurespec_dataclasses(so100_robot: RobotDescription) -> None:
    feats = features_from_robot(so100_robot, fps=30.0)
    for key, spec in feats.items():
        assert isinstance(spec, FeatureSpec), f"{key!r} → {type(spec).__name__}"
        assert spec.key == key, f"FeatureSpec.key {spec.key!r} != map key {key!r}"


# ── Override path — unlocks robots without observation_spec (Franka, GR1, …) ──


def test_overrides_unlock_robots_without_observation_spec(repo_root: Path) -> None:
    """``state_shape_override`` + ``action_dim_override`` work for franka_panda.

    Per ADR-0007 most robots leave observation_spec / action_spec unset
    because the sim-imposed contract lives on the rSkill manifest or
    scene adapter. The bridge sink resolves shapes from the first frame;
    here we exercise that path directly via the override kwargs.
    """
    robot = RobotDescription.from_yaml(str(repo_root / "robots" / "franka_panda" / "robot.yaml"))
    # franka_panda has no observation_spec / action_spec on disk.
    assert robot.observation_spec is None or not robot.observation_spec.state_shape

    # pi05-libero-nf4 contract: 8-D state, 7-D action.
    feats = features_from_robot(
        robot,
        fps=20.0,
        state_shape_override=(8,),
        action_dim_override=7,
    )
    assert feats["observation.state"].shape == (8,)
    assert feats["action"].shape == (7,)
    # The franka_panda manifest declares two cameras; they ride through
    # the bridge unchanged when overrides unlock the spec path.
    assert "observation.images.camera1" in feats
    assert "observation.images.camera2" in feats


def test_overrides_reject_non_positive_dims(so100_robot: RobotDescription) -> None:
    """Overrides validate shape/dim positivity (same contract as on-manifest specs)."""
    with pytest.raises(ValueError, match=r"every dimension must be > 0"):
        features_from_robot(so100_robot, fps=30.0, state_shape_override=(0,), action_dim_override=6)
    with pytest.raises(ValueError, match=r"must be > 0"):
        features_from_robot(so100_robot, fps=30.0, state_shape_override=(6,), action_dim_override=0)


# ── Parametrized sweep across every robot manifest in robots/ ────────────────


def _discover_robot_yamls(repo_root: Path) -> list[Path]:
    """Return every robots/<name>/robot.yaml under the repo root, sorted."""
    return sorted((repo_root / "robots").glob("*/robot.yaml"))


def _yaml_id(path: Path) -> str:
    """pytest parametrize id — the robot directory name."""
    return path.parent.name


@pytest.fixture(params=_discover_robot_yamls(Path(__file__).resolve().parents[3]), ids=_yaml_id)
def any_robot_yaml(request: pytest.FixtureRequest) -> Path:
    """Parametrized: yields every robots/<name>/robot.yaml in turn."""
    return request.param  # type: ignore[no-any-return]


def test_every_robot_manifest_loads(any_robot_yaml: Path) -> None:
    """Every committed robot.yaml must parse via ``RobotDescription.from_yaml``.

    Guard against accidental breakage when someone edits a robot
    manifest without exercising the schema validator.
    """
    robot = RobotDescription.from_yaml(str(any_robot_yaml))
    assert robot.name, f"{any_robot_yaml} produced a nameless RobotDescription"


def test_every_robot_manifest_features_or_typed_error(any_robot_yaml: Path) -> None:
    """``features_from_robot`` either succeeds or raises a typed ``ValueError``.

    The bridge contract is "robots without observation_spec / action_spec
    cannot bind to a LeRobot v3 dataset". This test enforces that the
    failure mode is loud (typed exception) rather than silent (returning
    a half-built features dict, or crashing later inside lerobot).
    """
    robot = RobotDescription.from_yaml(str(any_robot_yaml))
    obs_ok = robot.observation_spec is not None and bool(robot.observation_spec.state_shape)
    act_ok = robot.action_spec is not None and (robot.action_spec.dim or 0) > 0

    if obs_ok and act_ok:
        feats = features_from_robot(robot, fps=30.0)
        # Sanity: every robot that binds must have the canonical
        # state + action + bookkeeping features, regardless of camera count.
        assert "observation.state" in feats
        assert "action" in feats
        for canonical in ("next.reward", "next.done", "next.success"):
            assert canonical in feats
        # State / action shape come from the manifest's spec.
        assert feats["observation.state"].shape == tuple(robot.observation_spec.state_shape)
        assert feats["action"].shape == (robot.action_spec.dim,)
    else:
        # The typed-error contract: missing spec → ValueError with a
        # message that names the robot so the user knows which manifest
        # to fix.
        with pytest.raises(ValueError, match=robot.name):
            features_from_robot(robot, fps=30.0)


def test_every_robot_manifest_camera_keys_are_addressable(any_robot_yaml: Path) -> None:
    """Cameras declared on a robot must produce ``observation.images.*`` features.

    A common drift mode is changing ``vla_feature_key`` without the
    ``observation.images.`` prefix, which silently breaks dataset
    feature naming. Sweep every camera-bearing robot and assert each
    declared camera key actually lands in the features dict (when the
    robot also has observation_spec / action_spec to make
    ``features_from_robot`` succeed).
    """
    robot = RobotDescription.from_yaml(str(any_robot_yaml))
    if robot.observation_spec is None or not robot.observation_spec.state_shape:
        pytest.skip(
            f"{robot.name} has no observation_spec; features_from_robot is "
            "intentionally not callable on this robot"
        )
    if robot.action_spec is None or (robot.action_spec.dim or 0) <= 0:
        pytest.skip(f"{robot.name} has no action_spec; not bridge-bindable")

    feats = features_from_robot(robot, fps=30.0)
    image_sensors = [s for s in robot.sensors if s.vla_feature_key]
    for sensor in image_sensors:
        assert sensor.vla_feature_key in feats, (
            f"{robot.name}: sensor {sensor.name!r} declares "
            f"vla_feature_key={sensor.vla_feature_key!r}, but the bridge did not "
            f"surface it as a feature. Check that vla_feature_key starts with "
            f"'observation.images.' and the sensor modality is in the image set."
        )
