"""HIL preflight: everything the restocking deploy needs, short of moving.

Runs on the OpenArm cell, answering: if the arms were powered, would
``openral deploy run --config scenes/deploy/openarm_restock_shelf.yaml`` have everything it
needs? Each check is the real artifact against the real host: committed scene, robot
manifest, rSkill, physical CAN links, three physical cameras.

Stops short of two things:

- Actuation: nothing commands a joint. HAL connects, bus preflight runs, then disconnects.
  ADR-0102 checks build real command messages but through the adapter's ``publish_fn`` seam,
  so they never reach the wire — see
  ``test_the_policys_flat_vector_reaches_the_four_controllers_intact``.
- Inference: loading 6.74 GiB BF16 weights + forward pass is a different tier of test (needs
  the policy's processors, gated behind the PaliGemma tokenizer). This checks the plumbing
  around the policy, not the policy.

Skips cleanly off-rig: camera checks need the rig's udev symlinks, HAL check needs both CAN
links up.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.hil.conftest import _can_links_up

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


def _cameras_present() -> bool:
    return all(Path(p).exists() for p in _CAMERA_NODES)


requires_can = pytest.mark.skipif(
    not _can_links_up(_CAN_LINKS), reason="OpenArm CAN links are not both up — not on the cell"
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


# ── ADR-0102: the slot-dispatched 16-DoF vector ───────────────────────────────
#
# Why these run here and not only as unit tests: the unit suite hand-builds the slot group
# and hand-builds the HAL. These drive the REAL dispatcher over the REAL rSkill manifest's
# `slots:` block into a REAL `OpenArmRealHAL` that has passed its bus preflight against the
# physically wired cell, so a manifest/robot-manifest/adapter disagreement about joint ORDER
# shows up here and nowhere else.
#
# Why they cannot move the arm, whether or not it is powered: the only path from the four
# command topics to `openarm_can` and the motors is `openarm_bringup`'s ros2_control stack
# (CLAUDE.md §1.5 — the 400 Hz loop is C++, not Python). These tests never publish to ROS at
# all; they collect messages through the adapter's own `publish_fn` seam, in-process.


def _slot_actions_from_the_real_manifest(tick: int = 1) -> list:  # pragma: no cover
    """Run the production dispatcher over the committed rSkill's slot block.

    The policy vector is ``[0.0, 1.0, … 15.0]`` — every value equals the index
    of the joint it must reach, and `robots/openarm/robot.yaml` orders its 16
    joints ``left_joint1..7, left_gripper, right_joint1..7, right_gripper``. So
    any misroute, off-by-one, or side swap shows up as a value that does not
    match its position, rather than as a plausible-looking pose.
    """
    import numpy as np
    from openral_core.schemas import RobotDescription, RSkillManifest
    from openral_rskill_ros.rskill_runner_node import _dispatch_slots

    manifest = RSkillManifest.from_yaml(str(RSKILL))
    robot = RobotDescription.from_yaml(str(ROBOT))
    vector = np.arange(16, dtype=np.float32)
    actions = _dispatch_slots(manifest.action_contract.slots, vector, description=robot)
    for action in actions:
        # `_dispatch_slots` sets `tick_group_size`; `tick_index` is the runner
        # node's (`tick_index_getter`), so supply it the way the wire does.
        action.tick_index = tick
    return actions


@requires_can
def test_the_policys_flat_vector_reaches_the_four_controllers_intact() -> None:  # pragma: no cover
    """The whole ADR-0102 path on the real cell: policy vector → four messages.

    This is the contract `rskill-pi05-openarm-restock_shelf-bf16` needs and the
    one that silently did not hold before ADR-0102: the OpenArm could not
    declare `gripper_position`, so `send_action` returned early on
    `joint_targets is None` and **discarded the gripper command** — arms moving,
    grippers never actuating, nothing on the wire to say so.
    """
    import inspect

    from openral_core.schemas import DeployScene, RobotDescription
    from openral_hal.openarm_real import OpenArmRealHAL

    pytest.importorskip("openral_rskill_ros", reason="needs the built ROS overlay")

    robot = RobotDescription.from_yaml(str(ROBOT))
    scene = DeployScene.from_yaml(str(SCENE))
    params = {**robot.hal.parameters.defaults, **(scene.hal.defaults if scene.hal else {})}
    accepted = set(inspect.signature(OpenArmRealHAL.__init__).parameters)

    sent: list[tuple[str, dict]] = []
    hal = OpenArmRealHAL(
        robot,
        publish_fn=lambda topic, msg: sent.append((topic, msg)),
        **{k: v for k, v in params.items() if k in accepted},
    )
    hal.connect()  # real bus preflight against the wired cell
    try:
        actions = _slot_actions_from_the_real_manifest()
        assert len(actions) == 4, "the committed manifest declares four slots"
        assert {int(a.tick_group_size) for a in actions} == {4}

        for action in actions[:3]:
            hal.send_action(action)
        assert sent == [], "atomicity: three of four slots must publish nothing"

        hal.send_action(actions[3])
        left_arm, left_grip, right_arm, right_grip = hal.command_topics()
        by_topic = {topic: msg for topic, msg in sent}
        assert set(by_topic) == {left_arm, left_grip, right_arm, right_grip}

        # Every value equals its joint index, so this is the misroute check.
        assert by_topic[left_arm]["joint_targets"] == [[0.0, 1, 2, 3, 4, 5, 6]]
        assert by_topic[left_grip]["joint_targets"] == [[7.0]]
        assert by_topic[right_arm]["joint_targets"] == [[8.0, 9, 10, 11, 12, 13, 14]]
        assert by_topic[right_grip]["joint_targets"] == [[15.0]]

        # Each controller must also be told which joints it was handed — in the ros2_control
        # namespace, not the manifest's. Policy/manifest say `left_joint1`; URDF/
        # `openarm_bringup`'s controllers say `openarm_left_joint1`, and the adapter
        # translates (`OpenArmRealHAL` docstring, `ros2_control_joint_names`). Publishing
        # manifest names would leave every controller rejecting the command. Asserted against
        # the HAL's own public accessor, sliced by the four spans, so this pins the
        # translation itself rather than restating a literal.
        control_names = hal.ros2_control_joint_names()
        assert control_names[0] == "openarm_left_joint1", "ros2_control namespace expected"
        assert by_topic[left_arm]["joint_names"] == control_names[0:7]
        assert by_topic[left_grip]["joint_names"] == control_names[7:8]
        assert by_topic[right_arm]["joint_names"] == control_names[8:15]
        assert by_topic[right_grip]["joint_names"] == control_names[15:16]
    finally:
        hal.disconnect()


@requires_can
def test_slot_arrival_order_does_not_change_what_the_controllers_get() -> None:  # pragma: no cover
    """ADR-0102's point: addressing is by name, not by arrival order.

    Before it, `behavior._compose_action_group` unpacked positionally, making a
    manifest's `slots:` ordering a load-bearing but unwritten wire contract.
    """
    import inspect

    from openral_core.schemas import DeployScene, RobotDescription
    from openral_hal.openarm_real import OpenArmRealHAL

    pytest.importorskip("openral_rskill_ros", reason="needs the built ROS overlay")

    robot = RobotDescription.from_yaml(str(ROBOT))
    scene = DeployScene.from_yaml(str(SCENE))
    params = {**robot.hal.parameters.defaults, **(scene.hal.defaults if scene.hal else {})}
    accepted = set(inspect.signature(OpenArmRealHAL.__init__).parameters)

    runs: list[list[tuple[str, dict]]] = []
    for reverse in (False, True):
        sent: list[tuple[str, dict]] = []
        hal = OpenArmRealHAL(
            robot,
            publish_fn=lambda topic, msg, sink=sent: sink.append((topic, msg)),
            **{k: v for k, v in params.items() if k in accepted},
        )
        hal.connect()
        try:
            actions = _slot_actions_from_the_real_manifest()
            for action in reversed(actions) if reverse else actions:
                hal.send_action(action)
        finally:
            hal.disconnect()
        runs.append(sorted(sent, key=lambda pair: pair[0]))
    assert runs[0] == runs[1]


@requires_can
def test_a_standalone_gripper_action_raises_instead_of_vanishing() -> None:  # pragma: no cover
    """The exact silent failure ADR-0102 exists to end, checked on the cell.

    A lone gripper action has nowhere to go — the four bimanual controllers are
    driven from one 16-DoF vector — and the pre-0102 code dropped it without a
    word. On a powered robot that is arms moving while the grippers never
    actuate, which is why it must be loud.
    """
    import inspect

    from openral_core.exceptions import ROSConfigError
    from openral_core.schemas import DeployScene, RobotDescription
    from openral_hal.openarm_real import OpenArmRealHAL

    pytest.importorskip("openral_rskill_ros", reason="needs the built ROS overlay")

    robot = RobotDescription.from_yaml(str(ROBOT))
    scene = DeployScene.from_yaml(str(SCENE))
    params = {**robot.hal.parameters.defaults, **(scene.hal.defaults if scene.hal else {})}
    accepted = set(inspect.signature(OpenArmRealHAL.__init__).parameters)

    sent: list[tuple[str, dict]] = []
    hal = OpenArmRealHAL(
        robot,
        publish_fn=lambda topic, msg: sent.append((topic, msg)),
        **{k: v for k, v in params.items() if k in accepted},
    )
    hal.connect()
    try:
        gripper = _slot_actions_from_the_real_manifest()[1]
        gripper.tick_group_size = 1  # arriving alone, not as part of a tick
        with pytest.raises(ROSConfigError, match="standalone"):
            hal.send_action(gripper)
        assert sent == []
    finally:
        hal.disconnect()


@requires_cameras
@pytest.mark.parametrize(
    "sensor_name,expected_wh",
    [("top", (672, 376)), ("wrist_left", (960, 600)), ("wrist_right", (960, 600))],
)
def test_each_camera_delivers_the_shape_the_policy_trained_on(
    sensor_name: str, expected_wh: tuple[int, int]
) -> None:  # pragma: no cover
    """Open the real camera through its scene binding and check the frame shape.

    ``top`` — the ZED head camera, which overrides the manifest's sim overhead
    slot of the same name — is the one that matters: the ZED Mini streams both
    lenses in a
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
