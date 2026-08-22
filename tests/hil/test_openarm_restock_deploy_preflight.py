"""HIL preflight: everything the restocking deploy needs, short of moving.

Runs on the OpenArm cell and answers one question — *if the arms were
powered, would `openral deploy run --config scenes/deploy/openarm_restock_shelf.yaml`
have everything it needs?* Each check is the real artifact against the real
host: the committed scene, the committed robot manifest, the committed rSkill,
the physical CAN links and the three physical cameras.

Deliberately stops short of two things:

- **Actuation.** Nothing here commands a joint. The HAL is connected and its
  bus preflight is exercised, then disconnected.
- **Inference.** Loading 6.74 GiB of BF16 weights and running a forward pass
  is a different tier of test (and needs the policy's processors, which are
  gated behind the PaliGemma tokenizer). This checks the *plumbing* around the
  policy, not the policy.

Skips cleanly off-rig: the camera checks need the rig's udev symlinks and the
HAL check needs both CAN links up.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE = REPO_ROOT / "scenes" / "deploy" / "openarm_restock_shelf.yaml"
ROBOT = REPO_ROOT / "robots" / "openarm" / "robot.yaml"
RSKILL = REPO_ROOT / "rskills" / "rskill-pi05-openarm-restock_shelf-bf16" / "rskill.yaml"

_CAN_LINKS = ("openarm_left", "openarm_right")
_CAMERA_NODES = (
    "/dev/camera_head_stereo",
    "/dev/camera_wrist_left",
    "/dev/camera_wrist_right",
)


def _can_links_up() -> bool:
    from openral_cli.autodetect import enumerate_can_interfaces

    up = {i.name for i in enumerate_can_interfaces() if i.is_up}
    return set(_CAN_LINKS) <= up


def _cameras_present() -> bool:
    return all(Path(p).exists() for p in _CAMERA_NODES)


requires_can = pytest.mark.skipif(
    not _can_links_up(), reason="OpenArm CAN links are not both up — not on the cell"
)
requires_cameras = pytest.mark.skipif(
    not _cameras_present(),
    reason="rig camera udev symlinks absent (/dev/camera_*) — not on the cell",
)


# ── Manifest graph ────────────────────────────────────────────────────────────


def test_scene_robot_and_rskill_all_resolve() -> None:
    from openral_core.schemas import DeployScene, RobotDescription, RSkillManifest

    scene = DeployScene.from_yaml(str(SCENE))
    robot = RobotDescription.from_yaml(str(ROBOT))
    skill = RSkillManifest.from_yaml(str(RSKILL))
    assert scene.robot_id == "openarm"
    assert robot.name == "openarm_v2"
    assert skill.embodiment_tags == ["openarm"]


def test_rskill_is_compatible_with_the_robot() -> None:
    # The umbrella check: embodiment tags + capability flags + sensor
    # requirements, against the manifest the deploy actually loads.
    from openral_core.schemas import RobotDescription, RSkillManifest
    from openral_rskill.loader import rSkill

    robot = RobotDescription.from_yaml(str(ROBOT))
    skill = RSkillManifest.from_yaml(str(RSKILL))
    # The scene's cameras are what satisfy the policy's sensor contract on a
    # real cell, so splice them in exactly as `deploy run` does.
    from openral_core.schemas import DeployScene

    scene = DeployScene.from_yaml(str(SCENE))
    by_name = {s.name: s for s in robot.sensors}
    for s in scene.sensors:
        by_name[s.name] = s
    robot = robot.model_copy(update={"sensors": list(by_name.values())}, deep=True)

    rSkill.check_compatibility(skill, robot)  # raises on any mismatch


def test_scene_supplies_every_feature_key_the_policy_requires() -> None:
    from openral_core.schemas import DeployScene, RSkillManifest

    scene = DeployScene.from_yaml(str(SCENE))
    skill = RSkillManifest.from_yaml(str(RSKILL))
    required = {s.vla_feature_key for s in skill.sensors_required}
    supplied = {s.vla_feature_key for s in scene.sensors}
    assert required <= supplied, f"scene is missing {sorted(required - supplied)}"


def test_every_scene_camera_has_a_deploy_binding() -> None:
    # A sensor with no binding is a camera the deploy cannot open.
    from openral_core.schemas import DeployScene

    for s in DeployScene.from_yaml(str(SCENE)).sensors:
        assert s.deploy_binding is not None, f"{s.name} has no deploy_binding"
        assert s.deploy_binding.backend_params.get("device"), f"{s.name} names no device"


def test_bindings_use_stable_device_paths() -> None:
    # A raw /dev/videoN is assigned in USB enumeration order and silently
    # renumbers on replug or reboot — the binding would then point at a
    # different camera, or at nothing.
    from openral_core.schemas import DeployScene

    for s in DeployScene.from_yaml(str(SCENE)).sensors:
        device = str(s.deploy_binding.backend_params["device"])
        assert not device.removeprefix("/dev/video").isdigit(), (
            f"{s.name} binds the unstable node {device}; use a /dev/camera_* "
            "udev symlink or /dev/v4l/by-id/*"
        )


# ── Real hardware ─────────────────────────────────────────────────────────────


@requires_can
def test_hal_builds_from_the_scene_and_passes_its_bus_preflight() -> None:  # pragma: no cover
    import inspect

    from openral_core.schemas import DeployScene, RobotDescription
    from openral_hal.openarm_real import OpenArmRealHAL

    robot = RobotDescription.from_yaml(str(ROBOT))
    scene = DeployScene.from_yaml(str(SCENE))
    params = {**robot.hal.parameters.defaults, **(scene.hal.defaults if scene.hal else {})}
    accepted = set(inspect.signature(OpenArmRealHAL.__init__).parameters)
    hal = OpenArmRealHAL(robot, **{k: v for k, v in params.items() if k in accepted})

    hal.connect()  # raises ROSConfigError if either bus is down
    try:
        health = hal.health().fields
        assert health["left_can"] == "openarm_left"
        assert health["right_can"] == "openarm_right"
        assert len(hal.command_topics()) == 4
    finally:
        hal.disconnect()


@requires_cameras
@pytest.mark.parametrize(
    "sensor_name,expected_wh",
    [("context", (672, 376)), ("wrist_left", (960, 600)), ("wrist_right", (960, 600))],
)
def test_each_camera_delivers_the_shape_the_policy_trained_on(
    sensor_name: str, expected_wh: tuple[int, int]
) -> None:  # pragma: no cover
    """Open the real camera through its scene binding and check the frame shape.

    ``context`` is the one that matters: the ZED Mini streams both lenses in a
    single 1344x376 frame and the policy trained on the left lens alone
    (the dataset's ``observation.images.context`` is ``[376, 672, 3]``). Without
    the binding's crop the reader yields a perfectly valid image containing two
    half-width views — nothing downstream can distinguish that from a good
    frame, so it has to be caught here.
    """
    pytest.importorskip("cv2", reason="opencv-python-headless not installed")
    from openral_core.schemas import DeployScene
    from openral_runner.backends.opencv_thread import OpenCVThreadSensorReader

    scene = DeployScene.from_yaml(str(SCENE))
    spec = next(s for s in scene.sensors if s.name == sensor_name)
    params = dict(spec.deploy_binding.backend_params)

    reader = OpenCVThreadSensorReader(
        sensor_id=sensor_name,
        device=str(params["device"]),
        fps=int(params.get("fps", 30)),
        width=int(params["width"]),
        height=int(params["height"]),
        crop=params.get("crop"),
    )
    reader.open()
    try:
        frame = None
        deadline = time.monotonic() + 5.0
        while frame is None and time.monotonic() < deadline:
            try:
                frame = reader.read_latest(max_age_ms=2000)
            except Exception:  # reason: poll until the device warms up
                time.sleep(0.05)
        assert frame is not None, f"{sensor_name} opened but delivered no frame in 5 s"
        assert (frame.width, frame.height) == expected_wh
        assert frame.channels == 3
        assert len(frame.data) == frame.width * frame.height * frame.channels
    finally:
        reader.close()
