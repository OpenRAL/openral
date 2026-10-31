"""The tracker's own stalled-jaw event, on the real traces (#418, 2026-10-17, tracking/on sim only).

``fixtures/jaw_close_traces.npz`` (made by ``livesim/418/phase1/mkstall.py``):
* DROID ``r7dr_on1``: the Robotiq ``finger_joint`` measured (fox_mcap, 15 Hz) and the three
  closes the HAL trigger logged (start, target, end, ``short_of_command``, settle spread), in sim
  time; it fired no ATTACH for any of them and the banana was carried 23.6-25.2 s unbound;
* SO-101 ``r9po_camlong``: the gripper measured and the approved command (``/openral/safe_action``
  last row, index 5), 60 Hz. The jaw tracks its command everywhere: it closed on nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from openral_core import RobotDescription
from openral_world_state.object_tracker.stall import JawStallDetector

_F = np.load(Path(__file__).parent / "fixtures" / "jaw_close_traces.npz")
_NS = 1_000_000_000


def _detector(robot: str, joint: str) -> JawStallDetector:
    d = RobotDescription.from_yaml(f"robots/{robot}/robot.yaml")
    cal = next(j for j in d.joints if j.name == joint).closure_calibration
    assert cal is not None
    return JawStallDetector(
        cal.closed_position, cal.closed_rest_offset + cal.stall_gap, cal.settle_tolerance
    )


def test_droid_close_two_reads_the_hold_the_hal_trigger_missed() -> None:
    """r7dr_on1 close 2: command 0.188 from 22.48 to 23.48 s, the jaw at 0.49-0.56 from 22.48 s
    (spread 0.06 over the HAL's settle window, short 0.234 at the close's end): a stall, read
    once the gap and the hold have lasted the dwell."""
    det = _detector("droid_franka_robotiq", "finger_joint")
    start, target, end = (float(v) for v in _F["dr_closes"][1][:3])
    fired = []
    for t, p in zip(_F["dr_t"], _F["dr_pos"], strict=True):
        if start - 1.0 <= t <= end:  # with the closing that led into the hold
            det.command(target if t >= start else 1.0, int(t * _NS))
            if det.update(float(p), int(t * _NS)):
                fired.append(float(t))
    assert len(fired) == 1, fired
    assert start + det.dwell_ns / _NS <= fired[0] <= end, (fired, start, end)


def test_a_jaw_that_tracks_its_command_never_fires() -> None:
    """r9po_camlong: 60 Hz gripper, commands 0.03-0.41, measured within a few hundredths of them
    for the whole run -- closed on nothing (the orange was pushed, not held)."""
    det = _detector("so101_follower", "gripper")
    cmd_t, cmd = _F["so_cmd_t"], _F["so_cmd"]
    fired = 0
    for t, p in zip(_F["so_t"], _F["so_pos"], strict=True):
        i = int(np.searchsorted(cmd_t, t, side="right")) - 1
        if i >= 0:
            det.command(float(cmd[i]), int(t * _NS))
        fired += det.update(float(p), int(t * _NS))
    assert fired == 0


def test_an_empty_droid_close_never_fires() -> None:
    """r7dr_on1 close 1 (17.2-17.5 s, command 0.155): 0.25 s long, the jaw still closing."""
    det = _detector("droid_franka_robotiq", "finger_joint")
    start, target, end = (float(v) for v in _F["dr_closes"][0][:3])
    fired = 0
    for t, p in zip(_F["dr_t"], _F["dr_pos"], strict=True):
        if start - 1.0 <= t <= end + 0.5:
            det.command(target if start <= t <= end else 1.0, int(t * _NS))
            fired += det.update(float(p), int(t * _NS))
    assert fired == 0


def test_the_median_ignores_chatter_but_not_a_closing_jaw() -> None:
    det = _detector("droid_franka_robotiq", "finger_joint")
    det.command(0.19, 0)
    det.update(0.9, 0)  # closed into the hold
    rng = np.random.default_rng(0)
    hold = [0.45 + float(x) for x in rng.uniform(-0.03, 0.03, 40)]  # span 0.06, median 0.45
    fired = [det.update(p, int((i + 1) * 0.0667 * _NS)) for i, p in enumerate(hold)]
    assert sum(fired) == 1
    closing = _detector("droid_franka_robotiq", "finger_joint")
    closing.command(0.19, 0)
    ramp = [0.9 - 0.04 * i for i in range(15)]  # 0.04 per sample: far from settled
    assert not any(closing.update(p, int(i * 0.0667 * _NS)) for i, p in enumerate(ramp[:12]))


def test_a_real_attach_or_the_command_opening_ends_the_hold() -> None:
    det = _detector("droid_franka_robotiq", "finger_joint")
    det.command(0.19, 0)
    det.update(0.9, 0)  # closed into the hold
    n = 1
    while not det.update(0.45, int(n * 0.1 * _NS)):
        n += 1
        assert n < 20
    assert det.latched
    assert not det.update(0.45, int((n + 1) * 0.1 * _NS)), "once per hold"
    det.command(0.9, int((n + 2) * 0.1 * _NS))  # opened: the next close is a new hold
    assert not det.latched
    other = _detector("droid_franka_robotiq", "finger_joint")
    other.command(0.19, 0)
    other.suppress()  # the HAL's real ATTACH covers this hold
    assert not any(other.update(0.45, int(i * 0.1 * _NS)) for i in range(20))


def test_an_open_jaw_under_a_closing_chunk_never_fires() -> None:
    """Spark banana R322 (2026-10-23): the approved chunk's last row closed while the jaw, still
    executing earlier rows, sat fully open at 0.9996 -- short of that command and not moving. A
    derived ATTACH bound the banana where it lay, 9 s before the real grasp. Only a jaw that
    closed and stopped is a hold; the same jaw closing to 0.45 afterwards still fires once."""
    det = _detector("droid_franka_robotiq", "finger_joint")
    det.command(0.18, 0)
    t = [int(i * 0.0667 * _NS) for i in range(60)]
    assert not any(det.update(0.9996, s) for s in t[:30]), "open, not yet closing: no hold"
    closing = [0.9996 - 0.06 * i for i in range(10)]  # 0.9996 -> 0.46
    hold = [0.45] * 20
    fired = [det.update(p, s) for p, s in zip(closing + hold, t[30:], strict=True)]
    assert sum(fired) == 1, fired


_R = np.load(Path(__file__).parent / "fixtures" / "restock_jaw_traces.npz")


def _replay_restock(key: str, jaw_lag_s: float = 0.0) -> tuple[JawStallDetector, list[int]]:
    """Feed one restock trace (jaw and approved command) to the OpenArm read.

    ``jaw_lag_s`` processes every jaw sample that much after the commands received with it (a
    node whose executor queued joint states behind newer commands); each command keeps its
    receipt stamp.
    """
    det = _detector("openarm", "left_gripper")
    jt, jp, ct, cv = (_R[f"{key}_{k}"] for k in ("jaw_t", "jaw", "cmd_t", "cmd"))
    lag = int(jaw_lag_s * _NS)
    jaw = [(int(t) + lag, int(t), float(p), False) for t, p in zip(jt, jp, strict=True)]
    cmd = [(int(t), int(t), float(c), True) for t, c in zip(ct, cv, strict=True)]
    fired: list[int] = []
    for _, t, v, is_cmd in sorted(jaw + cmd, key=lambda e: (e[0], not e[3])):
        if is_cmd:
            det.command(v, t)
        elif det.update(v, t):
            fired.append(t)
    return det, fired


def _lift_mm(key: str, stamp_ns: int) -> float:
    t, z = _R[f"{key}_obj_t"], _R[f"{key}_obj_z"]
    return 1000.0 * float(z[int(np.argmin(np.abs(t - stamp_ns)))] - z[0])


def test_a_restock_carton_lifted_while_the_jaw_settles_is_held_from_the_grasp() -> None:
    """Restock bench clean3_08 (2026-10-31), a 6 cm carton: the policy lifts ~0.3 s after the
    close and the simulated jaw, squeezing at its effort cap, wobbles 1-3 mrad under the lift. The
    old 0.6 s window dated the hold 11 cm into the lift and fired at +23 cm, past the late bind's
    travel guard (untracked bind). The hold dates from the grasp; the read fires early in the
    lift."""
    det, fired = _replay_restock("c08")
    assert len(fired) == 1, fired
    since = det.hold_since_ns
    assert since is not None and int(_R["c08_exec_close"]) <= since <= fired[0]
    assert _lift_mm("c08", since) < 5.0  # the carton has not left the tote
    assert _lift_mm("c08", fired[0]) < 80.0  # was 234 mm


def test_a_pre_shaped_jaw_is_not_held_before_the_close() -> None:
    """Restock bench clean2_02: the policy closes the jaw to 0.67, holds it there at its command,
    then closes onto the carton. At the real close the command leads the lagging jaw by more than
    the gap at once while the band still reaches back into the pre-shape; one sample of gap is not a
    hold (a run-only read fired 0.4 s before the close). It fires once, after the close."""
    det, fired = _replay_restock("c02")
    assert len(fired) == 1, fired
    assert fired[0] > int(_R["c02_exec_close"])
    since = det.hold_since_ns
    assert since is not None and since >= int(_R["c02_exec_close"])
    assert _lift_mm("c02", fired[0]) < 20.0


def test_jaw_samples_read_late_are_judged_against_the_command_they_were_under() -> None:
    """A tracker replay of restock clean3_06 (2026-10-31) read a derived ATTACH 0.2 s before the
    close, the jaw half open: the approved chunk has no stamp, and jaw samples queued behind
    newer commands were judged against the newest one. Offline, 0.4 s of lag fired three bench
    grasps before their close. Read against the command in force at each sample's stamp, the
    hold fires where it does with no lag."""
    for key in ("c08", "c02"):
        _, on_time = _replay_restock(key)
        for lag_s in (0.4, 1.0):
            _, late = _replay_restock(key, jaw_lag_s=lag_s)
            assert late == on_time, (key, lag_s, late, on_time)
            assert late[0] > int(_R[f"{key}_exec_close"])
