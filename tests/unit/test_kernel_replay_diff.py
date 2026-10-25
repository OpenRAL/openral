"""The ADR-0121 parity replay tool (``tools/kernel_replay_diff.py``): its pure half.

The live two-kernel replay needs ROS 2 and a recorded deploy (the Spark parity runs); what is
pinned here is what decides the comparison: the topics a run records, kernel B's parameters (every
moved in-tree rule off, the provider on, the in-tree values mirrored, the open measurement kept),
and the verdict comparison.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "kernel_replay_diff", REPO_ROOT / "tools" / "kernel_replay_diff.py"
)
assert _spec is not None and _spec.loader is not None
krd = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = krd
_spec.loader.exec_module(krd)

# The DROID banana scene's in-tree kernel posture (#418 `on` + #349 + K7 + K5), as the launch
# passes it.
_A = {
    "hal_mode": "sim",
    "world_voxel_enabled": True,
    "grasp_allowance_enabled": True,
    "grasp_contact_links": ["robotiq_left_finger", "robotiq_right_finger"],
    "workspace_contact_enabled": True,
    "workspace_contact_links": ["robotiq_left_finger", "robotiq_right_finger", "panda_link7"],
    "workspace_contact_max_speed_mps": 0.05,
    "workspace_contact_max_lead_m": 0.04,
    "workspace_contact_control_period_s": 0.0667,
    "workspace_contact_jaw_joints": ["finger_joint"],
    "workspace_contact_brake_enabled": True,
    "workspace_contact_hold_max_consecutive": 30,
    "workspace_explained_cells_enabled": True,
    "tracked_grasp_exemption_enabled": True,
    "tracked_release_jaw_joints": ["finger_joint"],
    "tracked_release_jaw_open_sign": [-1.0],
    "tracked_release_timeout_s": 3.0,
}
_P = "openral_pro_safety_exemptions/ContactExemptions"


def test_the_recorder_records_every_kernel_input() -> None:
    cmd = krd.record_command("/tmp/run/bag")
    assert cmd[:7] == ["ros2", "bag", "record", "--storage", "mcap", "-o", "/tmp/run/bag"]
    for topic in (
        "/clock",
        "/joint_states",
        "/openral/candidate_action",
        "/openral/world_voxels",
        "/openral/world_state_fast",
        "/openral/estop",
    ):
        assert topic in cmd


def test_kernel_b_turns_every_moved_rule_off_and_loads_the_provider() -> None:
    b = krd.provider_params(_A, _P)
    for key in (
        "workspace_contact_enabled",
        "workspace_contact_brake_enabled",
        "workspace_explained_cells_enabled",
        "tracked_grasp_exemption_enabled",
        "payload_own_cells_enabled",
        "pinch_own_cells_enabled",
    ):
        assert b[key] is False, key
    assert b["tracked_release_jaw_joints"] == []
    assert b["workspace_contact_hold_max_consecutive"] == 0
    assert b["exemption_provider"] == _P
    assert b["exemption_provider_links"] == _A["workspace_contact_links"]


def test_kernel_b_keeps_the_open_measurement_and_mirrors_the_rules() -> None:
    b = krd.provider_params(_A, _P)
    assert b["workspace_contact_control_period_s"] == 0.0667, "the measurement stays open"
    assert b["workspace_contact_jaw_joints"] == ["finger_joint"]
    assert b["contact_exemptions.workspace_contact_enabled"] is True
    assert b["contact_exemptions.workspace_contact_max_speed_mps"] == 0.05
    assert b["contact_exemptions.tracked_release_jaw_joints"] == ["finger_joint"]
    assert b["tracked_release_timeout_s"] == 3.0, "the kernel's own release window stays"
    assert b["grasp_allowance_enabled"] is True, "ADR-0115 stays open"


def test_kernel_a_without_any_moved_rule_has_nothing_to_compare() -> None:
    with pytest.raises(ValueError, match="nothing to compare"):
        krd.provider_params({"hal_mode": "sim", "workspace_contact_enabled": False}, _P)


def test_the_param_dump_block_is_found_by_node_name() -> None:
    dump = {"/openral_safety_kernel": {"ros__parameters": {"hal_mode": "sim"}}}
    assert krd._kernel_params(dump) == {"hal_mode": "sim"}
    with pytest.raises(ValueError, match="ros__parameters"):
        krd._kernel_params({"/other": {"ros__parameters": {}}})


def test_verdicts_compare_pass_latch_and_rows() -> None:
    v = krd.Verdict
    assert krd.verdicts_equal(v(True, [0.1, 0.2]), v(True, [0.1, 0.2]))
    assert not krd.verdicts_equal(v(True, [0.1]), v(True, [0.2])), "a brake creep differs"
    assert not krd.verdicts_equal(v(False, latched=True), v(False, latched=False))
    report = krd.summarize([krd.Diff(4, 123, v(True, [0.0]), v(False))], 10)
    assert report["equal"] == 9
    assert report["a_only_passes"] == 1
    assert report["b_only_passes"] == 0
    assert report["diffs"][0]["index"] == 4
