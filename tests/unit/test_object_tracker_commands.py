"""The tracker reads the jaw command from every kind of approved chunk (issue #418, WP C).

Real chunks and real jaw traces from the camlong runs r12dr_camlong, r12po_camlong and
r12rs02_camlong_b (``jaw_close_chunks.json`` / ``jaw_close_replay.npz``, SOURCE in
``jaw_close_replay.SOURCE.txt``), real manifests, the real detector.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_world_state.object_tracker.commands import jaw_commands
from openral_world_state.object_tracker.stall import JawStallDetector

FX = Path(__file__).parent / "fixtures"
ROBOTS = {
    "dr": ("robots/droid_franka_robotiq/robot.yaml", "finger_joint"),
    "po": ("robots/so101_follower/robot.yaml", "gripper"),
    "rs02": ("robots/openarm/robot.yaml", "left_gripper"),
}


def _setup(kind: str) -> tuple[RobotDescription, str, JawStallDetector]:
    path, joint = ROBOTS[kind]
    d = RobotDescription.from_yaml(path)
    cal = next(j for j in d.joints if j.name == joint).closure_calibration
    assert cal is not None
    det = JawStallDetector(
        cal.closed_position, cal.closed_rest_offset + cal.stall_gap, cal.settle_tolerance
    )
    return d, joint, det


def _commands(kind: str) -> list[tuple[int, float]]:
    d, joint, _ = _setup(kind)
    grippers = {j.name for j in d.joints if j.role == "gripper"}
    order = [j.name for j in d.joints]
    out = []
    for c in json.loads((FX / "jaw_close_chunks.json").read_text())[kind]:
        got = jaw_commands(
            grippers,
            order,
            n_dof=c["n_dof"],
            flat=c["flat"],
            joint_names=c["names"],
            tick_group_size=c["group"],
            ee_name=c["ee"],
        )
        if joint in got:
            out.append((int(c["t"] * 1e9), got[joint]))
    return out


def _replay(kind: str, t0: float = 0.0, t1: float = 1e9) -> list[float]:
    """Source seconds at which the real detector fires over the recorded trace."""
    _, _, det = _setup(kind)
    z = np.load(FX / "jaw_close_replay.npz")
    meas = [(int(t * 1e9), float(v)) for t, v in zip(z[f"{kind}_mt"], z[f"{kind}_mv"], strict=True)]
    cmds, fired = _commands(kind), []
    ci = 0
    for t, v in meas:
        while ci < len(cmds) and cmds[ci][0] <= t:
            det.command(cmds[ci][1], cmds[ci][0])
            ci += 1
        if det.update(v, t) and t0 <= t * 1e-9 <= t1:
            fired.append(t * 1e-9)
    return fired


def test_an_openarm_slot_group_yields_only_its_gripper_slot_as_a_jaw_command() -> None:
    d, _, _ = _setup("rs02")
    grippers = {j.name for j in d.joints if j.role == "gripper"}
    order = [j.name for j in d.joints]
    chunks = [
        c
        for c in json.loads((FX / "jaw_close_chunks.json").read_text())["rs02"]
        if c["tick"] == 129
    ]
    assert [c["n_dof"] for c in chunks] == [
        16,
        1,
        16,
        1,
    ]  # left arm, left gripper, right arm, right gripper
    got = [
        jaw_commands(
            grippers,
            order,
            n_dof=c["n_dof"],
            flat=c["flat"],
            joint_names=c["names"],
            tick_group_size=c["group"],
            ee_name=c["ee"],
        )
        for c in chunks
    ]
    assert got[0] == got[2] == {}
    assert list(got[1]) == ["left_gripper"] and list(got[3]) == ["right_gripper"]


@pytest.mark.parametrize("kind", ["dr", "po"])
def test_a_full_dof_row_is_read_at_the_grippers_manifest_index(kind: str) -> None:
    cmds = _commands(kind)
    assert len(cmds) > 50 and all(0.0 <= v <= 1.0 for _, v in cmds)


def test_the_restock_stalled_jaw_fires_once_on_the_real_hold_after_the_close_command() -> None:
    fired = _replay("rs02")
    assert len(fired) == 1, fired
    close = next(t for t, v in _commands("rs02") if v < 0.1) * 1e-9
    assert close < fired[0] < close + 1.5  # after the close command, within the hold's dwell


def test_the_droid_stalled_jaw_fires_once_on_the_banana_hold() -> None:
    fired = _replay("dr")
    # The first read is the hold (the jaw settles near 0.45 from 21.7 s). The policy's command
    # wanders afterwards (0.12 -> 0.36 -> 0.11), which re-arms the detector once; a second read
    # for a hold already bound is a no-op in the session (the duplicate returns the held bind).
    assert fired and 21.0 < fired[0] < 22.5, fired


def test_the_so101_jaw_that_tracks_its_command_never_fires_but_the_orange_hold_does() -> None:
    assert _replay("po", 6.3, 7.5) == []  # 0.205 -> 0.205: closed on nothing
    fired = _replay("po")
    assert len(fired) == 1 and fired[0] > 9.4, fired  # the orange hold, not the free closes
