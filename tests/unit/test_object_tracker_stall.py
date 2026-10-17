"""The tracker's own stalled-jaw event, on the real traces (#418, 2026-10-17, bind/on sim only).

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
    """r7dr_on1 close 2: command 0.188 from 22.48 to 23.48 s, the jaw chattering down to 0.51 and on
    to 0.43 (spread 0.06 over the HAL's settle window, short 0.234 at the close's end): a stall."""
    det = _detector("droid_franka_robotiq", "finger_joint")
    start, target, end = (float(v) for v in _F["dr_closes"][1][:3])
    fired = []
    for t, p in zip(_F["dr_t"], _F["dr_pos"], strict=True):
        if start <= t <= end:
            det.command(target, int(t * _NS))
            if det.update(float(p), int(t * _NS)):
                fired.append(float(t))
    assert len(fired) == 1, fired
    assert start + 0.5 < fired[0] <= end, (fired, start, end)


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
    rng = np.random.default_rng(0)
    hold = [0.45 + float(x) for x in rng.uniform(-0.03, 0.03, 40)]  # span 0.06, median 0.45
    fired = [det.update(p, int(i * 0.0667 * _NS)) for i, p in enumerate(hold)]
    assert sum(fired) == 1
    closing = _detector("droid_franka_robotiq", "finger_joint")
    closing.command(0.19, 0)
    ramp = [0.9 - 0.04 * i for i in range(15)]  # 0.04 per sample: far from settled
    assert not any(closing.update(p, int(i * 0.0667 * _NS)) for i, p in enumerate(ramp[:12]))


def test_a_real_attach_or_the_command_opening_ends_the_hold() -> None:
    det = _detector("droid_franka_robotiq", "finger_joint")
    det.command(0.19, 0)
    n = 0
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
