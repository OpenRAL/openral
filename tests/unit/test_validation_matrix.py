"""Tests for the validation-matrix harness (``tools/validation_matrix.py``).

CLAUDE.md §1.11 — every input here is a **recorded artifact** copied verbatim
from a real round on the project's DGX Spark (see
``tests/unit/fixtures/validation_matrix/SOURCE.txt``), not a synthesized log.
The assertions are pinned to what
``docs/reference/collision-validation-evidence.md`` concluded about those
rounds, so if the extractor ever stops reproducing the published ledger these
go red.

Three sections:

1. **Verdict derivation** — the four 2026-08-22 scenes must bucket exactly as
   the ledger says, with the same tripping pairs and distances.
2. **Diffing** — the 08-16 and 08-22 baguette runs are the same code, same
   scene, same seed with a different outcome; the diff must say so.
3. **Guardrails** — each refusal is exercised against the real repo.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "validation_matrix"
ROUND_0822 = FIXTURES / "2026-08-22-master-1"
ROUND_0816 = FIXTURES / "2026-08-16-master-1"

# tools/ is not an installed package — load the module by path, the same way
# tests/unit/test_select_tests.py and test_audit_tests.py do.
_spec = importlib.util.spec_from_file_location(
    "validation_matrix", REPO_ROOT / "tools" / "validation_matrix.py"
)
assert _spec is not None and _spec.loader is not None
validation_matrix = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = validation_matrix
_spec.loader.exec_module(validation_matrix)


def _derived(round_dir: Path, tmp_path: Path) -> object:
    """Copy a fixture round into ``tmp_path`` and derive its verdicts there.

    The fixture directory stays inputs-only; ``verdicts`` writes ``verdicts.json``
    and ``NOTES.md`` next to the artifacts, which must not land in the repo.
    """
    from openral_core import ValidationRoundVerdicts

    work = tmp_path / round_dir.name
    shutil.copytree(round_dir, work)
    assert validation_matrix.cmd_verdicts(work) == 0
    return ValidationRoundVerdicts.from_json(str(work / "verdicts.json"))


# ── 1. Verdict derivation ─────────────────────────────────────────────────────


def test_round_derives_the_four_ledger_outcomes(tmp_path: Path) -> None:
    """The 2026-08-22 round buckets exactly as the evidence ledger records it."""
    verdicts = _derived(ROUND_0822, tmp_path)
    assert {s.scene: s.outcome for s in verdicts.scenes} == {
        "baguette": "estop-collision-false-positive",
        "sink_cup": "estop-collision-real",
        "fridge": "estop-collision-real",
        "utensil": "estop-collision-false-positive",
    }
    # "sim.task_success_final = False on all four scenes" — ledger, 2026-08-22.
    assert all(s.task_success_final is False for s in verdicts.scenes)


@pytest.mark.parametrize(
    ("scene", "party_a", "party_b", "step", "min_distance_m"),
    [
        ("baguette", "panda_link5", "voxel_170781", 0, -0.0209178),
        ("sink_cup", "attached:sim:obj_main", "voxel_87084", -1, -0.0133754),
        ("fridge", "panda_link7", "voxel_169769", -1, -0.0247489),
        ("utensil", "panda_link1", "voxel_76001", -1, -0.0172764),
    ],
)
def test_kernel_verdicts_are_transcribed_verbatim(
    tmp_path: Path, scene: str, party_a: str, party_b: str, step: int, min_distance_m: float
) -> None:
    """Each tripping pair and depth matches the ledger's table byte for byte."""
    verdicts = _derived(ROUND_0822, tmp_path)
    stop = verdicts.scene(scene).stop
    assert stop is not None
    assert (stop.party_a, stop.party_b, stop.horizon_step) == (party_a, party_b, step)
    assert stop.min_distance_m == min_distance_m
    # "min_distance_m == sweep_min_distance_m in all four -> no exemption active
    # anywhere this round" — ledger, 2026-08-22.
    assert stop.sweep_min_distance_m == min_distance_m
    assert stop.exemption_active is False
    assert stop.place_allowance_active is False


def test_no_place_allowance_anywhere_in_the_round(tmp_path: Path) -> None:
    """`place_allowance_active=1` occurrences: 0 — ledger, 2026-08-22."""
    verdicts = _derived(ROUND_0822, tmp_path)
    assert all(s.witness.place_allowance_active_lines == 0 for s in verdicts.scenes)


def test_baguette_grasped_and_the_witness_armed_then_separated(tmp_path: Path) -> None:
    """Attach at t=176.32, witness armed, separated, detach at t=226.54."""
    baguette = _derived(ROUND_0822, tmp_path).scene("baguette")
    assert baguette.witness.attach_t_s == pytest.approx(176.32, abs=0.01)
    assert baguette.witness.detach_t_s == pytest.approx(226.54, abs=0.01)
    assert baguette.witness.support_id == "sim:counter_1_left_group_main"
    assert baguette.witness.kernel_witness_armed == 1
    assert baguette.witness.kernel_witness_separated == 1
    assert baguette.witness.place_region_armed is True


def test_fridge_and_utensil_never_grasped(tmp_path: Path) -> None:
    """Neither scene reached a grasp — ledger's "grasped: no" column."""
    verdicts = _derived(ROUND_0822, tmp_path)
    for scene in ("fridge", "utensil"):
        assert verdicts.scene(scene).witness.attach_t_s is None


def test_baguette_false_positive_exceeds_the_quantization_budget(tmp_path: Path) -> None:
    """431/431 pairs probed, untruncated, nothing within 100 mm vs a −20.9 mm read."""
    ground_truth = _derived(ROUND_0822, tmp_path).scene("baguette").ground_truth
    assert ground_truth is not None
    assert ground_truth.verdict == "false-positive"
    assert ground_truth.probed_pairs == 431
    assert ground_truth.probe_truncated is False
    assert ground_truth.nearest_any_m is None  # nothing within distmax at all
    assert ground_truth.distmax_m == 0.1
    # >120 mm of discrepancy against a ~21.7 mm budget (25 mm voxel).
    assert ground_truth.grid_resolution_m == 0.025
    assert ground_truth.quantization_budget_m == pytest.approx(0.021651, abs=1e-6)
    assert ground_truth.discrepancy_m == pytest.approx(0.1209178, abs=1e-6)


def test_utensil_false_positive_is_a_60mm_discrepancy(tmp_path: Path) -> None:
    """robot0_link1 clear by +43.3 mm against a kernel −17.3 mm."""
    ground_truth = _derived(ROUND_0822, tmp_path).scene("utensil").ground_truth
    assert ground_truth is not None
    assert ground_truth.verdict == "false-positive"
    assert ground_truth.nearest_tripping_party_m == pytest.approx(0.043256, abs=1e-6)
    assert ground_truth.discrepancy_m == pytest.approx(0.0605324, abs=1e-6)
    assert ground_truth.nearest_pair["body_b"] == "stack_2_left_group_3_door_main"


def test_sink_cup_stop_is_real_payload_contact(tmp_path: Path) -> None:
    """Nearest payload-world pair at −1.8 mm: the payload really is inside."""
    sink = _derived(ROUND_0822, tmp_path).scene("sink_cup")
    assert sink.stop is not None
    assert sink.stop.involves_payload is True
    assert sink.ground_truth is not None
    assert sink.ground_truth.verdict == "real-contact"
    assert sink.ground_truth.stop_class == "attached_payload"
    assert sink.ground_truth.nearest_any_m == pytest.approx(-0.001759, abs=1e-6)
    assert sink.ground_truth.payload_contacts == 6


def test_fridge_stop_is_real_despite_zero_mujoco_contacts(tmp_path: Path) -> None:
    """The probe's own caveat: 0 contacts is not an emptiness test.

    ``robot0_link6`` sits at 0.000 m from the freezer door with
    ``payload_contacts == 0`` and ``robot_world_contacts == 0``. Adjudicating
    from the contact list would call this a false positive; adjudicating from
    the distance probe — which is what the snapshot tells you to do — does not.
    """
    fridge = _derived(ROUND_0822, tmp_path).scene("fridge")
    assert fridge.ground_truth is not None
    assert fridge.ground_truth.verdict == "real-contact"
    assert fridge.ground_truth.payload_contacts == 0
    assert fridge.ground_truth.nearest_any_m == 0.0
    assert fridge.ground_truth.nearest_pair["body_b"] == "fridge_main_group_freezer_door"
    assert fridge.ground_truth.sim_time_s == pytest.approx(4.85, abs=0.01)


def test_scene_configs_are_hashed_so_a_round_pins_its_scene(tmp_path: Path) -> None:
    """Every verdict carries the digest of the YAML that ran."""
    verdicts = _derived(ROUND_0822, tmp_path)
    for scene in verdicts.scenes:
        assert scene.config_path.startswith("scenes/deploy/")
        assert scene.config_sha256 is not None
        expected = (REPO_ROOT / scene.config_path).read_bytes()
        assert len(scene.config_sha256) == 64
        assert expected  # the tracked scene the digest was taken over exists


def test_notes_are_written_alongside_the_verdicts(tmp_path: Path) -> None:
    """A round can no longer end without a written summary."""
    work = tmp_path / "round"
    shutil.copytree(ROUND_0822, work)
    assert validation_matrix.cmd_verdicts(work) == 0
    notes = (work / "NOTES.md").read_text(encoding="utf-8")
    assert "2edcf67c3b087958d475813fe19234c12e90698c" in notes
    assert "panda_link5" in notes and "voxel_170781" in notes
    assert "Safety-knob overrides present: **False**" in notes


def test_verdicts_refuses_a_round_without_metadata(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert validation_matrix.cmd_verdicts(tmp_path / "empty") == 2


# ── 2. Diffing ────────────────────────────────────────────────────────────────


def test_diff_two_real_rounds_at_the_same_sha(tmp_path: Path) -> None:
    """Same code, same scene, same seed — the baguette outcome still moved.

    The 08-16 run tripped during carry on the attached payload with a deeper
    cell exempted; the 08-22 run carried cleanly and tripped later on an arm
    link with nothing exempted. That is a reproducibility finding, and the diff
    has to surface it without anyone re-reading a log.
    """
    current = _derived(ROUND_0822, tmp_path / "cur")
    baseline = _derived(ROUND_0816, tmp_path / "base")
    diff = validation_matrix.diff_rounds(current, baseline)

    assert diff.same_sha is True
    assert diff.baseline_round_id == "2026-08-16-master-1"
    assert "baguette" in diff.changed_scenes

    baguette = next(s for s in diff.scenes if s.scene == "baguette")
    assert baguette.baseline_outcome == "estop-collision-real"
    assert baguette.outcome == "estop-collision-false-positive"
    moves = baguette.changed_fields
    assert moves["stop.party_a"] == {"from": "attached:sim:obj_main", "to": "panda_link5"}
    assert moves["stop.min_distance_m"]["to"] == -0.0209178
    assert moves["stop.sweep_min_distance_m"]["from"] == -0.0355338


def test_prior_round_records_a_live_exemption(tmp_path: Path) -> None:
    """sweep_min (−35.5 mm) deeper than min_distance (−4.2 mm): a cell was exempted."""
    baguette = _derived(ROUND_0816, tmp_path).scene("baguette")
    assert baguette.stop is not None
    assert baguette.stop.min_distance_m == -0.00418456
    assert baguette.stop.sweep_min_distance_m == -0.0355338
    assert baguette.stop.exemption_active is True
    assert baguette.stop.involves_payload is True


def test_scene_missing_from_the_baseline_counts_as_changed(tmp_path: Path) -> None:
    """The 08-16 round ran baguette only; the other three are new, not 'same'."""
    current = _derived(ROUND_0822, tmp_path / "cur")
    baseline = _derived(ROUND_0816, tmp_path / "base")
    diff = validation_matrix.diff_rounds(current, baseline)
    for scene in ("sink_cup", "fridge", "utensil"):
        delta = next(s for s in diff.scenes if s.scene == scene)
        assert delta.baseline_outcome is None
        assert delta.changed is True


def test_diff_round_trips_through_the_schema(tmp_path: Path) -> None:
    from openral_core import ValidationRoundDiff

    current = _derived(ROUND_0822, tmp_path / "cur")
    baseline = _derived(ROUND_0816, tmp_path / "base")
    diff = validation_matrix.diff_rounds(current, baseline)
    restored = ValidationRoundDiff.model_validate_json(diff.model_dump_json())
    assert restored.changed_scenes == diff.changed_scenes


# ── 3. Guardrails ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["--hal", "collision_margin_m=0.0"],
        ["--hal", "attached_contact_tolerance_m=0.05"],
        ["--enable-octomap-kernel-check=false"],
        ["--disable-safety-watchdog"],
        ["--ros-args", "-p", "estop_topic:=/dev/null"],
    ],
)
def test_safety_knob_overrides_are_refused(argv: list[str]) -> None:
    """The harness never becomes the place a margin quietly moves."""
    with pytest.raises(validation_matrix.GuardrailError, match="safety-knob pattern"):
        validation_matrix.assert_no_safety_overrides(argv)


def test_the_pinned_stack_argv_passes_the_safety_guard() -> None:
    """The stack the matrix actually pins must not trip its own guard."""
    validation_matrix.assert_no_safety_overrides(list(validation_matrix.STACK_ARGV))


def test_wrong_checkout_is_refused() -> None:
    """The launcher hazard: a round once ran the wrong checkout silently."""
    with pytest.raises(validation_matrix.GuardrailError, match="wrong checkout"):
        validation_matrix.assert_sha("0" * 40)


def test_expected_sha_accepts_the_real_head() -> None:
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert validation_matrix.assert_sha(head[:12]) == head


def test_missing_overlay_is_refused(tmp_path: Path) -> None:
    """A round against no overlay at all is a rebuild instruction, not a run."""
    with pytest.raises(validation_matrix.GuardrailError, match="no built overlay"):
        validation_matrix.assert_overlay_fresh(tmp_path / "install")


def test_stale_overlay_is_refused(tmp_path: Path) -> None:
    """An `install/` older than a tracked C++ source silently validates old code."""
    install = tmp_path / "install"
    install.mkdir()
    import os

    # Age the overlay a year behind every source in the tree.
    os.utime(install, ns=(1, 1))
    with pytest.raises(validation_matrix.GuardrailError, match="older than"):
        validation_matrix.assert_overlay_fresh(install)


def test_the_sync_group_set_carries_the_sidecar_wire() -> None:
    """`--group robocasa` alone strips pyzmq and broke the XR-1 adapter once."""
    assert "sidecar-wire" in validation_matrix.SYNC_GROUPS
    assert "robocasa" in validation_matrix.SYNC_GROUPS


def test_matrix_scenes_are_tracked_and_load_as_deploy_scenes() -> None:
    """Every scene the matrix names is a real, tracked DeployScene."""
    from openral_core import DeployScene, load_scene_strict

    assert len(validation_matrix.MATRIX) == 4
    for spec in validation_matrix.MATRIX:
        path = REPO_ROOT / spec.config
        assert path.exists(), spec.config
        scene = load_scene_strict(str(path), expected=DeployScene)
        assert scene.scene.id.startswith("robocasa/")
        assert spec.prompt


def test_quantization_budget_is_the_cell_half_diagonal() -> None:
    assert validation_matrix.quantization_budget_m(0.025) == pytest.approx(0.021651, abs=1e-6)


def test_grid_resolution_comes_from_the_recorded_monitor() -> None:
    """The budget is read from the run, never assumed."""
    records = validation_matrix.read_monitor(ROUND_0822 / "baguette" / "run_monitor.jsonl")
    assert validation_matrix.grid_resolution_from_monitor(records) == 0.025
    assert validation_matrix.grid_resolution_from_monitor([]) is None


def test_a_run_with_no_artifacts_is_a_harness_error(tmp_path: Path) -> None:
    """A scene whose graph never came up must not read as a clean deadline."""
    empty = tmp_path / "scene"
    empty.mkdir()
    verdict = validation_matrix.scene_verdict_from_artifacts(
        empty,
        scene="baguette",
        config_path="scenes/deploy/robocasa_baguette.yaml",
        seed=1,
    )
    assert verdict.outcome == "harness-error"


def test_initial_configuration_stop_outranks_ground_truth(tmp_path: Path) -> None:
    """A stop before the first applied chunk is a scene defect, not a stack one.

    The recorded fridge round predates the ``sim.estop_initial_configuration``
    line (it landed in #139), so this appends that one real line — emitted
    verbatim by ``openral_hal.sim_sensor_bridge`` — to the recorded log and
    checks the classifier re-buckets the same run.
    """
    from openral_hal.sim_sensor_bridge import initial_configuration_stop_record

    work = tmp_path / "round"
    shutil.copytree(ROUND_0822, work)
    snapshot_line = next(
        ln
        for ln in (work / "fridge" / "run_deploy_excerpt.log")
        .read_text(encoding="utf-8")
        .splitlines()
        if "sim.estop_ground_truth_snapshot" in ln
    )
    snapshot = json.loads(snapshot_line[snapshot_line.find("{") :])
    record = initial_configuration_stop_record(
        snapshot, stop_seq=1, last_action_ns=0, candidate_chunks_seen=0
    )
    assert record is not None and record["violation"] == "initial_configuration"
    with (work / "fridge" / "run_deploy_excerpt.log").open("a", encoding="utf-8") as sink:
        sink.write(
            "[lifecycle_node.py-3] sim.estop_initial_configuration "
            + json.dumps(record, sort_keys=True)
            + "\n"
        )

    assert validation_matrix.cmd_verdicts(work) == 0
    from openral_core import ValidationRoundVerdicts

    verdicts = ValidationRoundVerdicts.from_json(str(work / "verdicts.json"))
    assert verdicts.scene("fridge").outcome == "estop-initial-configuration"
    # The other three are untouched by the reclassification.
    assert verdicts.scene("sink_cup").outcome == "estop-collision-real"
