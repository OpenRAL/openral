# SPDX-License-Identifier: Apache-2.0
"""The twin proves the real grasp signal: a position-stall ATTACH from MuJoCo jaw positions.

The real OpenArm gripper reports no effort (vendor ``openarm_ros2`` hard-codes it to 0.0),
so the HAL's grasp trigger is ``PositionStallTrigger``: a jaw commanded closed that settles
short of the command. This drives the real ``OpenArmMujocoHAL`` on the vendored v2 MJCF —
the MJCF's own finger position actuator (kp 30, force range +-7) and finger meshes — at the
real 30 Hz action rate, reads ``read_state`` exactly as the lifecycle node does, and feeds
the trigger with the manifest's ``closure_calibration`` (``robots/openarm/robot.yaml``):

* **object** — a free 5 cm box in the left jaws: the jaw stalls ~0.25 rad short of a 0.0
  command (inside the 0.18-0.29 rad band measured on the real gripper) and settles to
  ~1e-4 rad → ATTACH; commanding open → DETACH once the jaw opens past its hold.
* **nothing** — the same close with no box reaches the jaw's rest → no event.
* **dropped** — gravity on and nothing holding the box: it falls before the jaws meet it,
  and the close reads as empty → no event (the false-negative direction is the safe one
  only because the payload is no longer there).

Only scene set-up touches ``MjData`` directly: the box is placed between the opened jaws
(it cannot start there — the twin's start pose has the jaws closed through it). The robot
is driven only through ``send_action``. Gravity is off for the held case because the twin
has no table to rest the box on before the grasp.

Finding recorded for the design note: a box FIXED to the world chatters against the finger
meshes (+-0.02 rad) and never settles — a twin artefact (rigid contact vs. a saturated
position servo), not a property of the real jaw, which is why the payload here is free.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator

import pytest

try:
    import mujoco
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

try:
    from openral_hal._openarm_v2_assets import ensure_openarm_v2_mjcf

    _OPENARM_MJCF: str = ensure_openarm_v2_mjcf()
    _MJCF_ERROR: str | None = None if os.path.isfile(_OPENARM_MJCF) else "MJCF missing"
except Exception as exc:
    _OPENARM_MJCF = ""
    _MJCF_ERROR = str(exc)

from openral_core import Action, ControlMode, RobotDescription
from openral_hal._grasp_trigger import GraspEvent, PositionStallTrigger

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
    pytest.mark.skipif(_MJCF_ERROR is not None, reason=f"OpenArm v2 MJCF: {_MJCF_ERROR}"),
]

_ROBOT_YAML = "robots/openarm/robot.yaml"
_LEFT_GRIPPER = 7  # public 16-DoF slot
_STEPS_PER_TICK = 17  # 0.002 s MJCF timestep x 17 ~= one 30 Hz action period
_OPEN = 0.7
# Between the left finger pads with the arm at its zero pose (MJCF FK, base frame).
_BETWEEN_JAWS = (-0.0008, 0.1535, -0.59)
_BOX_HALF = (0.02, 0.025, 0.02)


@pytest.fixture(scope="module")
def payload_mjcf() -> Iterator[str]:
    """The vendored MJCF plus one free box, written beside it so its meshdir resolves."""
    spec = mujoco.MjSpec.from_file(_OPENARM_MJCF)
    body = spec.worldbody.add_body(name="stall_payload", pos=[0.6, 0.6, 0.0])
    body.add_freejoint(name="stall_payload_free")
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(_BOX_HALF), mass=0.05)
    path = os.path.join(os.path.dirname(_OPENARM_MJCF), f"stall_payload_{uuid.uuid4().hex}.xml")
    with open(path, "w") as fh:
        fh.write(spec.to_xml())
    try:
        yield path
    finally:
        os.remove(path)


def _command(hal: object, left: float) -> None:
    target = [0.0] * 16
    target[_LEFT_GRIPPER] = left
    hal.send_action(  # type: ignore[attr-defined]
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[target],
            stamp_ns=time.time_ns(),
        )
    )


def _run(
    hal: object, trigger: PositionStallTrigger, left: float, ticks: int
) -> tuple[list[GraspEvent], list[float]]:
    """Command ``left`` for ``ticks`` 30 Hz ticks; feed every read to the trigger.

    Each read is re-stamped with the twin's simulated time: the HAL stamps wall time,
    and this loop steps 1/30 s of physics per tick far faster than real time, so wall
    stamps would compress the trigger's seconds-based windows (and read as repeats).
    """
    trigger.command(left)
    events: list[GraspEvent] = []
    trace: list[float] = []
    for _ in range(ticks):
        _command(hal, left)
        state = hal.read_state()  # type: ignore[attr-defined]
        sim_ns = round(float(hal._data.time) * 1e9)  # type: ignore[attr-defined]
        state = state.model_copy(update={"stamp_ns": sim_ns})
        trace.append(float(state.position[state.name.index("left_gripper")]))
        event = trigger.update(state)
        if event is not None:
            events.append(event)
    return events, trace


def _hal(mjcf: str | None, *, gravity: bool) -> object:
    from openral_hal import OpenArmMujocoHAL

    hal = OpenArmMujocoHAL(mjcf_path=mjcf, gravity_enabled=gravity, settle_steps=_STEPS_PER_TICK)
    hal.connect()
    return hal


def _place_payload(hal: object) -> None:
    """Scene set-up: drop the box between the (opened) left jaws, at rest."""
    model, data = hal._model, hal._data  # type: ignore[attr-defined]
    joint = model.joint("stall_payload_free")
    qadr, vadr = joint.qposadr[0], joint.dofadr[0]
    data.qpos[qadr : qadr + 7] = [*_BETWEEN_JAWS, 1.0, 0.0, 0.0, 0.0]
    data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


@pytest.fixture(scope="module")
def openarm() -> RobotDescription:
    return RobotDescription.from_yaml(_ROBOT_YAML)


def test_closing_on_a_held_object_stalls_attaches_and_releases(
    openarm: RobotDescription, payload_mjcf: str
) -> None:
    hal = _hal(payload_mjcf, gravity=False)
    try:
        trigger = PositionStallTrigger(openarm, joint_name="left_gripper")
        events, _ = _run(hal, trigger, _OPEN, 45)
        assert events == []
        _place_payload(hal)
        events, trace = _run(hal, trigger, 0.0, 90)
        stall = trace[-10:]
        assert 0.18 <= sum(stall) / len(stall) <= 0.29, f"stalled at {stall[-1]:.3f} rad"
        assert max(stall) - min(stall) <= 0.001, "the twin's stalled jaw is not flat"
        assert events == [GraspEvent.ATTACH]
        events, _ = _run(hal, trigger, _OPEN, 30)
        assert events == [GraspEvent.DETACH]
    finally:
        hal.disconnect()  # type: ignore[attr-defined]


def test_closing_on_nothing_never_attaches(openarm: RobotDescription) -> None:
    hal = _hal(None, gravity=False)
    try:
        trigger = PositionStallTrigger(openarm, joint_name="left_gripper")
        _run(hal, trigger, _OPEN, 45)
        events, trace = _run(hal, trigger, 0.0, 60)
        assert trace[-1] <= 0.02, f"an empty close rests at {trace[-1]:.4f} rad"
        assert events == []
        assert not trigger.attached
    finally:
        hal.disconnect()  # type: ignore[attr-defined]


def test_a_payload_that_falls_away_reads_as_an_empty_close(
    openarm: RobotDescription, payload_mjcf: str
) -> None:
    hal = _hal(payload_mjcf, gravity=True)
    try:
        trigger = PositionStallTrigger(openarm, joint_name="left_gripper")
        _run(hal, trigger, _OPEN, 45)
        _place_payload(hal)
        events, trace = _run(hal, trigger, 0.0, 60)
        assert trace[-1] <= 0.02
        assert events == []
    finally:
        hal.disconnect()  # type: ignore[attr-defined]
