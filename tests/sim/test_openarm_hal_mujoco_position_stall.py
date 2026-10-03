# SPDX-License-Identifier: Apache-2.0
"""The twin proves the real grasp signal: a position-stall ATTACH from MuJoCo jaw positions.

The real OpenArm gripper reports no effort (vendor ``openarm_ros2`` hard-codes it to 0.0),
so the HAL's grasp trigger is ``PositionStallTrigger``: a jaw commanded closed that settles
short of the command. This drives the real ``OpenArmMujocoHAL`` on the vendored v2 MJCF —
the MJCF's own finger position actuator (kp 30, force range +-7) and finger meshes — at the
real 30 Hz action rate and at 10 Hz, reads ``read_state`` exactly as the lifecycle node does,
and feeds the trigger with the manifest's ``closure_calibration`` (``robots/openarm/robot.yaml``)
and the windows the node derives for that rate (``PositionStallConfig.for_rate``):

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
from openral_hal._grasp_trigger import GraspEvent, PositionStallConfig, PositionStallTrigger

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
    pytest.mark.skipif(_MJCF_ERROR is not None, reason=f"OpenArm v2 MJCF: {_MJCF_ERROR}"),
]

_ROBOT_YAML = "robots/openarm/robot.yaml"
_LEFT_GRIPPER = 7  # public 16-DoF slot
_TIMESTEP_S = 0.002  # the vendored MJCF's timestep
#: Action/joint-state rates the twin is driven at: the OpenArm's own 30 Hz, and a
#: 10 Hz feed whose period equalled the old fixed 0.1 s max_gap_s.
_RATES_HZ = (30.0, 10.0)
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
    hal: object, trigger: PositionStallTrigger, left: float, seconds: float
) -> tuple[list[GraspEvent], list[float]]:
    """Command ``left`` for ``seconds`` of ticks; feed every read to the trigger.

    One tick per action period (the HAL's ``settle_steps``). Each read is re-stamped
    with the twin's simulated time: the HAL stamps wall time, and this loop steps a
    period of physics per tick far faster than real time, so wall stamps would compress
    the trigger's seconds-based windows (and read as repeats).
    """
    trigger.command(left)
    events: list[GraspEvent] = []
    trace: list[float] = []
    period_s = hal._settle_steps * _TIMESTEP_S  # type: ignore[attr-defined]
    for _ in range(round(seconds / period_s)):
        _command(hal, left)
        state = hal.read_state()  # type: ignore[attr-defined]
        sim_ns = round(float(hal._data.time) * 1e9)  # type: ignore[attr-defined]
        state = state.model_copy(update={"stamp_ns": sim_ns})
        trace.append(float(state.position[state.name.index("left_gripper")]))
        event = trigger.update(state)
        if event is not None:
            events.append(event)
    return events, trace


def _hal(mjcf: str | None, *, gravity: bool, rate_hz: float) -> object:
    from openral_hal import OpenArmMujocoHAL

    steps = round(1.0 / (rate_hz * _TIMESTEP_S))  # 17 at 30 Hz, 50 at 10 Hz
    hal = OpenArmMujocoHAL(mjcf_path=mjcf, gravity_enabled=gravity, settle_steps=steps)
    hal.connect()
    return hal


def _trigger(openarm: RobotDescription, rate_hz: float) -> PositionStallTrigger:
    """The left trigger with the windows the HAL node derives for ``rate_hz`` joint states."""
    return PositionStallTrigger(
        openarm, joint_name="left_gripper", config=PositionStallConfig.for_rate(rate_hz)
    )


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


@pytest.mark.parametrize("rate_hz", _RATES_HZ)
def test_closing_on_a_held_object_stalls_attaches_and_releases(
    openarm: RobotDescription, payload_mjcf: str, rate_hz: float
) -> None:
    hal = _hal(payload_mjcf, gravity=False, rate_hz=rate_hz)
    try:
        trigger = _trigger(openarm, rate_hz)
        events, _ = _run(hal, trigger, _OPEN, 1.5)
        assert events == []
        _place_payload(hal)
        events, trace = _run(hal, trigger, 0.0, 3.0)
        stall = trace[-5:]
        assert 0.18 <= sum(stall) / len(stall) <= 0.29, f"stalled at {stall[-1]:.3f} rad"
        assert max(stall) - min(stall) <= 0.001, "the twin's stalled jaw is not flat"
        assert events == [GraspEvent.ATTACH]
        events, _ = _run(hal, trigger, _OPEN, 1.0)
        assert events == [GraspEvent.DETACH]
    finally:
        hal.disconnect()  # type: ignore[attr-defined]


@pytest.mark.parametrize("rate_hz", _RATES_HZ)
def test_closing_on_nothing_never_attaches(openarm: RobotDescription, rate_hz: float) -> None:
    hal = _hal(None, gravity=False, rate_hz=rate_hz)
    try:
        trigger = _trigger(openarm, rate_hz)
        _run(hal, trigger, _OPEN, 1.5)
        events, trace = _run(hal, trigger, 0.0, 2.0)
        assert trace[-1] <= 0.02, f"an empty close rests at {trace[-1]:.4f} rad"
        assert events == []
        assert not trigger.attached
    finally:
        hal.disconnect()  # type: ignore[attr-defined]


@pytest.mark.parametrize("rate_hz", _RATES_HZ)
def test_a_payload_that_falls_away_reads_as_an_empty_close(
    openarm: RobotDescription, payload_mjcf: str, rate_hz: float
) -> None:
    hal = _hal(payload_mjcf, gravity=True, rate_hz=rate_hz)
    try:
        trigger = _trigger(openarm, rate_hz)
        _run(hal, trigger, _OPEN, 1.5)
        _place_payload(hal)
        events, trace = _run(hal, trigger, 0.0, 2.0)
        assert trace[-1] <= 0.02
        assert events == []
    finally:
        hal.disconnect()  # type: ignore[attr-defined]
