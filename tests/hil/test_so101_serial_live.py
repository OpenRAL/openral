"""HIL test: a physically wired SO-101 follower arm on this host's USB serial bus.

This is the first HIL coverage the SO-100/SO-101 family has had. Everything
else that touches ``SO100FollowerHAL`` (``tests/unit/test_so100_follower_hal.py``)
drives it through an injected ``SO100DigitalTwin``, which never opens a serial
port — so the real Feetech transport, the deg↔rad conversion against a real
encoder, and the committed calibration have only ever been exercised by hand.

Gating, and therefore the scene, is read from
``scenes/deploy/so101_bench.yaml``: the port, the lerobot calibration ``id``
and the ``calibration_dir`` all come from that scene's ``hal:`` binding, so
this test and ``openral deploy run --config scenes/deploy/so101_bench.yaml``
can never drift apart. Retuning the bench for a new host is a scene edit.

Two tiers, deliberately separated (same shape as ``test_openarm_can_live``):

- **Bench wiring** — the serial device is present and the deploy's committed
  calibration is on disk and agrees with the robot manifest. Runs whether or
  not the servos have 12 V. Touches no motor.
- **Live servos** — the motor bus answers. Additionally requires a broadcast
  ping to find every declared joint id; an arm whose 12 V supply is off (or
  whose USB-serial adapter is plugged in without it) leaves ``/dev/ttyACM0``
  enumerated and the bus completely dark, which is a **skip** and not a
  failure, because it is a fact about the bench and not about the code.

NON-MOTION by construction. The powered tier calls ``connect()``,
``read_state()`` and ``disconnect()`` and nothing else — never ``send_action``,
never ``reset_to_pose``. Commanding the arm from a pytest process, unattended
and with nobody on the E-stop, is not something this tier does.

One caveat worth knowing before you run this on a live arm: lerobot's
``SOFollower.connect()`` ends in ``configure()``, which cycles torque off and
back on. Re-enabling torque makes each STS3215 hold at its own
``Goal_Position`` register, so an arm left with a stale goal from a previous
session can twitch toward it at connect time. That is inherent to *any*
``SO100FollowerHAL.connect()``, including the one ``deploy run`` performs; it
is not introduced here. Keep a hand near the power switch the first time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openral_hal.so100_follower import SO100FollowerHAL

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENE = _REPO_ROOT / "scenes" / "deploy" / "so101_bench.yaml"
_ROBOT = _REPO_ROOT / "robots" / "so101_follower" / "robot.yaml"

# Feetech servo ids on the SO-ARM bus, in kinematic-chain order. lerobot's
# ``SOFollower`` hard-codes these (shoulder_pan=1 … gripper=6); the committed
# calibration repeats them, which is what makes it checkable against the
# manifest joint order.
_EXPECTED_SERVO_IDS = (1, 2, 3, 4, 5, 6)

# Tolerance on the manifest's joint envelope when judging a *resting* pose.
# The manifest limits are the SO-101 ``new_calib`` MJCF's mechanical limits,
# but a physical servo's calibrated range overruns them by a few degrees of
# calibration tail — the eraser-place rSkill manifest quantifies the same tail
# on this embodiment at up to 8.4° (wrist_flex +103.4 vs +95.0). 0.15 rad
# (8.6°) covers it. This is a plausibility bound, NOT a safety bound: the
# safety kernel clamps commands to the manifest limits exactly, with no
# tolerance, and this test asserts nothing about what may be *commanded*.
_CALIB_TAIL_RAD = 0.15


def _scene_hal_defaults() -> dict[str, object]:
    """The bench scene's ``hal.defaults`` block, or ``{}`` if it declares none."""
    from openral_core.schemas import DeployScene

    scene = DeployScene.from_yaml(str(_SCENE))
    return dict(scene.hal.defaults) if scene.hal is not None else {}


def _bench_port() -> str:
    return str(_scene_hal_defaults().get("port", ""))


def _bench_calibration_file() -> Path:
    """Path to the calibration the scene binds (``<calibration_dir>/<id>.json``).

    A relative ``calibration_dir`` resolves against the scene file's own
    directory — the same rule ``resolve_launch_invocation`` applies, so the
    file this test checks is the file ``deploy run`` will load.
    """
    defaults = _scene_hal_defaults()
    raw_dir = str(defaults.get("calibration_dir", ""))
    identity = str(defaults.get("id", ""))
    if not raw_dir or not identity:
        return Path("/nonexistent")
    directory = Path(raw_dir)
    if not directory.is_absolute():
        directory = _SCENE.parent / directory
    return directory / f"{identity}.json"


def _rig_is_present() -> bool:
    return Path(_bench_port()).exists() and _bench_calibration_file().is_file()


def _servo_ids_answering() -> frozenset[int]:
    """Broadcast-ping the Feetech bus and return the ids that replied.

    Pure read: the port is opened without lerobot's handshake, one broadcast
    ping goes out, and the port is closed with ``disable_torque=False`` so not
    a single register is written. An absent bus, an absent lerobot, or a port
    another process holds all yield the empty set — this is a skip gate, and a
    gate that raises would turn "the arm is off" into a red lab run.
    """
    if not _rig_is_present():
        return frozenset()
    try:
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError:  # reason: lerobot optional -> no live servos to gate on
        return frozenset()
    bus = FeetechMotorsBus(_bench_port(), {})
    try:
        bus.connect(handshake=False)
    except Exception:  # reason: port busy/unopenable is a bench fact -> skip
        return frozenset()
    try:
        found = bus.broadcast_ping()
    except Exception:  # reason: a silent bus raises rather than returning {}
        found = None
    finally:
        bus.disconnect(disable_torque=False)
    return frozenset(found or {})


requires_rig = pytest.mark.skipif(
    not _rig_is_present(),
    reason=(
        f"{_SCENE.name}: serial port {_bench_port()!r} or committed calibration "
        f"{_bench_calibration_file().name!r} is absent — no SO-101 bench on this host"
    ),
)
_ANSWERING = _servo_ids_answering()
requires_live_servos = pytest.mark.skipif(
    not set(_EXPECTED_SERVO_IDS) <= _ANSWERING,
    reason=(
        f"SO-101 serial port is present but only servo ids {sorted(_ANSWERING)} answer a "
        f"broadcast ping (expected {list(_EXPECTED_SERVO_IDS)}) — the motor bus is dark, "
        "which on this arm means the 12 V supply is off (USB alone browns the servos out)"
    ),
)


@pytest.fixture(scope="module")
def live_hal() -> Iterator[SO100FollowerHAL]:
    """One connected ``SO100FollowerHAL`` for the powered tier.

    Module-scoped so the tier connects once: every ``connect()`` cycles servo
    torque (see the module docstring), and doing that three times to run three
    read-only assertions would be gratuitous. Teardown always disconnects,
    which is what de-energises the bus — lerobot's ``disconnect`` disables
    torque on every motor before closing the port.
    """
    from openral_hal.so100_follower import SO100FollowerHAL

    defaults = _scene_hal_defaults()
    hal = SO100FollowerHAL(
        port=_bench_port(),
        id=str(defaults["id"]),
        calibration_dir=str(_bench_calibration_file().parent),
        calibrate_on_connect=False,  # never open the interactive wizard in CI
    )
    hal.connect()
    try:
        yield hal
    finally:
        hal.disconnect()


# ── Bench wiring ──────────────────────────────────────────────────────────────


@requires_rig
def test_the_committed_calibration_matches_the_robot_manifest() -> None:  # pragma: no cover
    """The scene's calibration file describes this manifest's joints, in order.

    A calibration recorded against a different arm (or an older joint order)
    loads without complaint and then silently mis-maps every encoder count, so
    the arm reads plausible-looking nonsense. The file is the deploy's own
    committed artefact, so this is checkable without power.
    """
    from openral_core.schemas import RobotDescription

    manifest = RobotDescription.from_yaml(str(_ROBOT))
    calibration = json.loads(_bench_calibration_file().read_text())

    assert list(calibration) == [j.name for j in manifest.joints]
    assert [calibration[name]["id"] for name in calibration] == list(_EXPECTED_SERVO_IDS)
    for name, entry in calibration.items():
        assert entry["range_min"] < entry["range_max"], f"{name}: empty calibrated range"


@requires_rig
def test_the_manifest_binds_the_real_serial_hal() -> None:  # pragma: no cover
    """``hal.real`` resolves to the class this test drives, and there is no sim HAL.

    SO-101 shares the SO-100's lerobot Feetech follower verbatim; if that
    binding is ever re-pointed, the HIL tier must follow it rather than keep
    testing a class the deploy no longer loads.
    """
    from openral_core.schemas import RobotDescription
    from openral_hal.so100_follower import SO100FollowerHAL

    manifest = RobotDescription.from_yaml(str(_ROBOT))
    assert manifest.hal is not None
    assert manifest.hal.real == "openral_hal.so100_follower:SO100FollowerHAL"
    assert manifest.hal.sim is None
    assert SO100FollowerHAL is not None  # the entry point imports


@requires_rig
@pytest.mark.skipif(
    set(_EXPECTED_SERVO_IDS) <= _ANSWERING,
    reason="servos are powered — the dark-bus diagnosis is covered by the powered tier",
)
def test_a_dark_motor_bus_fails_loud_instead_of_reading_zeros() -> None:  # pragma: no cover
    """An enumerated port with unpowered servos must raise, not connect.

    This is the failure mode ``_preflight_servo_ping`` was written for: the
    USB-serial adapter enumerates on 5 V, so ``/dev/ttyACM0`` exists and looks
    healthy while every servo is dark. ``send_action`` is fire-and-forget, so a
    HAL that connected anyway would command big moves into silence and the
    observed state would simply never change.
    """
    from openral_core.exceptions import ROSError
    from openral_hal.so100_follower import SO100FollowerHAL

    defaults = _scene_hal_defaults()
    hal = SO100FollowerHAL(
        port=_bench_port(),
        id=str(defaults["id"]),
        calibration_dir=str(_bench_calibration_file().parent),
        calibrate_on_connect=False,
    )
    with pytest.raises(ROSError) as excinfo:
        hal.connect()
    # Whichever layer catches it, the message has to name the silent motors.
    assert "motor" in str(excinfo.value).lower()


# ── Live servos ───────────────────────────────────────────────────────────────


@requires_rig
@requires_live_servos
def test_every_declared_joint_answers_the_preflight_ping(
    live_hal: SO100FollowerHAL,
) -> None:  # pragma: no cover
    """``connect()`` returning at all means all six servos answered.

    ``SO100FollowerHAL._preflight_servo_ping`` pings each joint by name and
    raises ``ROSConfigError`` naming any that stay dark, so a successful
    connect is itself the assertion; re-ping here to state which ids that was.
    """
    assert live_hal is not None
    assert set(_EXPECTED_SERVO_IDS) <= _servo_ids_answering()


@requires_rig
@requires_live_servos
def test_read_state_matches_the_manifest_and_is_plausible_radians(
    live_hal: SO100FollowerHAL,
) -> None:  # pragma: no cover
    """A real encoder read lands in the manifest's shape, names, units and envelope.

    The unit assertion is the load-bearing one: lerobot exposes this arm in
    **degrees** and the OpenRAL contract is **radians**, so a regression in
    ``_obs_to_positions`` would leave every arm joint ~57× too large — which
    lands far outside the manifest envelope rather than looking merely odd.
    The gripper is the documented exception: a normalised ``[0, 1]`` jaw
    fraction, not an angle.
    """
    from openral_core.schemas import RobotDescription

    manifest = RobotDescription.from_yaml(str(_ROBOT))
    state = live_hal.read_state()

    assert state.name == [j.name for j in manifest.joints]
    assert len(state.position) == len(manifest.joints)
    assert state.stamp_ns > 0

    for joint, value in zip(manifest.joints, state.position, strict=True):
        assert math.isfinite(value), f"{joint.name} read {value!r}"
        low, high = joint.position_limits
        assert low - _CALIB_TAIL_RAD <= value <= high + _CALIB_TAIL_RAD, (
            f"{joint.name} rests at {value:.4f} — outside the manifest envelope "
            f"[{low:.4f}, {high:.4f}] by more than the {_CALIB_TAIL_RAD} rad "
            "servo-calibration tail. Either the arm is parked outside its own "
            "declared limits, or the deg->rad conversion regressed."
        )


@requires_rig
@requires_live_servos
def test_the_committed_calibration_is_the_one_the_servos_are_running(
    live_hal: SO100FollowerHAL,
) -> None:  # pragma: no cover
    """Read calibration back off the motors and compare to the committed file.

    lerobot writes the calibration into the servos at connect, and resolves it
    from ``<calibration_dir>/<id>.json``. With no ``calibration_dir`` that is
    the ambient ``~/.cache/huggingface`` copy — a second ``so_follower.json``
    for one arm is a real footgun — so proving the *committed* file is the one
    in the motors is the whole point of the scene binding.
    """
    # Private reach: the HAL deliberately exposes no lerobot handle, and there
    # is no public surface for "what calibration is in the servos".
    bus = live_hal._robot.bus
    on_motors = bus.read_calibration()
    committed = json.loads(_bench_calibration_file().read_text())

    assert set(on_motors) == set(committed)
    for name, entry in committed.items():
        actual = on_motors[name]
        assert actual.id == entry["id"], name
        assert actual.homing_offset == entry["homing_offset"], name
        assert actual.range_min == entry["range_min"], name
        assert actual.range_max == entry["range_max"], name
