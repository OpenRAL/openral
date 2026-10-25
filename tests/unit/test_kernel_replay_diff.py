"""The ADR-0121 parity replay tool (``tools/kernel_replay_diff.py``): its pure half.

The live two-kernel replay needs ROS 2 and a recorded deploy (the Spark parity runs); what is
pinned here is what decides the comparison: the topics a run records, kernel B's parameters (the
#349 switches off, the rules ADR-0121 step 3 removed out of B's in-tree set, the provider on, the
in-tree values mirrored, the open measurement kept), and the verdict comparison.
"""

from __future__ import annotations

import importlib.util
import re
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

# The DROID banana scene's in-tree kernel posture (#418 `on` + #349 + K7 + K5), as a kernel built
# before ADR-0121 step 3 dumped it.
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
    for key in ("workspace_contact_enabled", "workspace_contact_brake_enabled"):
        assert b[key] is False, key
    assert b["workspace_contact_hold_max_consecutive"] == 0
    for key in krd._PROVIDER_ONLY_RULES:
        assert key not in b, f"{key}: removed from the open kernel, provider-only"
    assert b["contact_exemptions.tracked_grasp_exemption_enabled"] is True
    assert b["contact_exemptions.workspace_explained_cells_enabled"] is True
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


def test_every_moved_rule_and_mirrored_prefix_names_a_declared_kernel_parameter() -> None:
    """A misspelt key silently leaves the in-tree rule on in kernel B (pro review of 286ee01af);
    a provider-only rule the open kernel still declared would run in-tree beside the provider."""
    src = (REPO_ROOT / "cpp/openral_safety_kernel/src/lifecycle_kernel.cpp").read_text()
    declared = set(re.findall(r'declare_parameter<.+?>\(\s*"([^"]+)"', src, re.S))
    assert declared, "the kernel's declare_parameter calls were not found"
    for key in krd._MOVED_RULES_OFF:
        assert key in declared, key
    for key in krd._PROVIDER_ONLY_RULES:
        assert key not in declared, f"{key}: ADR-0121 step 3 removed it from the open kernel"
    named = declared | set(krd._PROVIDER_ONLY_RULES)
    for prefix in krd._MIRRORED_PREFIXES:
        assert any(d.startswith(prefix) for d in named), prefix


def test_only_diffs_outside_the_self_check_noise_fail_tier_2() -> None:
    """Pro tier 2 on master_dr ep3: A-vs-A also differed on some runs (replay noise)."""
    report = {"diffs": [{"index": 12}, {"index": 44}, {"index": 50}]}
    checks = [{"diffs": [{"index": 12}]}, {"diffs": [{"index": 44}, {"index": 45}]}]
    assert krd.outside_noise(report, checks) == [50]
    assert krd.outside_noise(report, []) == [12, 44, 50], "no self-check: every diff counts"


def test_a_dump_without_hand_links_refuses_rather_than_an_empty_mask() -> None:
    """An empty mask makes the kernel refuse the provider: B would run with none (pro review)."""
    with pytest.raises(ValueError, match="hand links"):
        krd.provider_params({"hal_mode": "sim", "tracked_release_jaw_joints": ["gripper"]}, _P)


def test_a_bag_missing_a_required_input_is_named() -> None:
    """Without measured state, voxels or a candidate both kernels refuse alike: the replay
    refuses such a bag rather than report a parity that tested nothing (CodeRabbit #432)."""
    full = dict.fromkeys(krd._REQUIRED_INPUTS, 5)
    assert krd.missing_inputs(full) == []
    assert krd.missing_inputs({**full, "/openral/world_voxels": 0}) == ["/openral/world_voxels"]
    assert krd.missing_inputs({}) == list(krd._REQUIRED_INPUTS)
