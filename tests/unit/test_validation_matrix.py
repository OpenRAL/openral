"""Tests for the validation-matrix harness (``tools/validation_matrix.py``).

CLAUDE.md §1.11 — every input is a **recorded artifact** copied verbatim from a
real round on the project's DGX Spark (``tests/unit/fixtures/validation_matrix/SOURCE.txt``),
not a synthesized log. Assertions are pinned to
``docs/reference/collision-validation-evidence.md``, so a drift from the
published ledger goes red here.

Sections: 1) verdict derivation for the four 2026-08-22 scenes; 2) diffing
(08-16 vs 08-22 baguette, same code/scene/seed); 3) guardrails; 4) the pinned
stack vs the real ``openral deploy sim`` CLI and tracked scenes; 5) launch
failure vs deadline; 6) importing pre-harness (``bag1``/``seed1`` layout)
rounds; 7) the 2026-08-23 defects (deaf monitor, aborted launch, ignored HAL
budget).

Two of the ledger's published 2026-08-22 verdicts are *corrected* here rather
than reproduced: ``sink_cup`` and ``fridge`` were called ``real-contact`` off
pairs the probe should never have measured (the payload's own ``obj_reg_bbox``
region marker, and ``robot0_g25_vis``, a visual shell). See section 1.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from openral_core import ValidationStopEvidence

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "validation_matrix"
ROUND_0822 = FIXTURES / "2026-08-22-master-1"
ROUND_0816 = FIXTURES / "2026-08-16-master-1"
ROUND_HARNESS_1 = FIXTURES / "2026-08-22-harness-1"
ROUND_0823 = FIXTURES / "2026-08-23-master-s1"
ROUND_NAV143 = FIXTURES / "2026-08-23-nav143-s1"
ROUND_POST200 = FIXTURES / "2026-09-04-post200-2"
ROUND_217_WITH204 = FIXTURES / "2026-09-05-217-with204-1"

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
    """All four 2026-08-22 verdicts are withdrawn, not reversed — never established.

    ``sink_cup``/``fridge`` (``estop-collision-real``) rest on a pair the probe
    should never have measured (payload's own ``obj_reg_bbox`` marker at
    −1.8 mm; ``robot0_g25_vis``, a visual shell, at 0.000 m). ``baguette``/
    ``utensil`` (``estop-collision-false-positive``) were judged against the
    21.7 mm voxel term alone, a strict lower bound on the admissible gap; the
    round predates ``adjudication_budget`` (#144, ``ea1b7e8``).
    """
    verdicts = _derived(ROUND_0822, tmp_path)
    assert {s.scene: s.outcome for s in verdicts.scenes} == {
        "baguette": "estop-collision-unadjudicated",
        "sink_cup": "estop-collision-unadjudicated",
        "fridge": "estop-collision-unadjudicated",
        "utensil": "estop-collision-unadjudicated",
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


def test_baguette_discrepancy_survives_the_measurement_but_not_the_budget(
    tmp_path: Path,
) -> None:
    """431/431 pairs probed, untruncated, nothing within 100 mm vs a −20.9 mm read.

    120.9 mm is a strict lower bound on the discrepancy (untruncated probe,
    nothing within ``distmax_m``); the only budget available predates
    ``adjudication_budget`` and is itself a lower bound (21.7 mm voxel term).
    Two lower bounds don't make a verdict — comparing against the 88.2 mm the
    same robot publishes on a later round is a cross-round inference the
    harness deliberately does not make.
    """
    ground_truth = _derived(ROUND_0822, tmp_path).scene("baguette").ground_truth
    assert ground_truth is not None
    assert ground_truth.verdict == "unadjudicated"
    assert ground_truth.budget_source == "grid-quantization"
    assert "lower bound" in ground_truth.unadjudicated_reason.lower()
    assert ground_truth.probed_pairs == 431
    assert ground_truth.probe_truncated is False
    assert ground_truth.nearest_any_m is None  # nothing within distmax at all
    assert ground_truth.distmax_m == 0.1
    assert ground_truth.grid_resolution_m == 0.025
    assert ground_truth.quantization_budget_m == pytest.approx(0.021651, abs=1e-6)
    assert ground_truth.discrepancy_m == pytest.approx(0.1209178, abs=1e-6)
    # It clears 88.2 mm by 32.7 mm, which is why the ledger keeps it open.
    assert ground_truth.discrepancy_m > 0.08822


def test_the_utensil_false_positive_is_withdrawn_by_its_own_later_rerun(
    tmp_path: Path,
) -> None:
    """The same stop, twice, with opposite verdicts — and the budget is the difference.

    ``panda_link1`` vs ``voxel_76001``, step −1, ``min_distance_m`` −0.0172764,
    60.5 mm discrepancy: identical to 7 sig figs on 08-22 and 08-23. 08-22 (no
    budget) called it a false positive against the 21.7 mm voxel term; 08-23
    (88.2 mm budget) reads the same stop as ``within-quantization``. Both are
    now withdrawn on a third ground — neither round attests its own distances —
    though the −17.3 mm stop is confirmed conservative-and-correct by a
    certified re-measurement (not either round's own probe).
    """
    old = _derived(ROUND_0822, tmp_path / "old")
    new = _derived(ROUND_0823, tmp_path / "new")

    old_stop, new_stop = old.scene("utensil").stop, new.scene("utensil").stop
    assert old_stop is not None and new_stop is not None
    assert (old_stop.party_a, old_stop.party_b, old_stop.horizon_step) == (
        new_stop.party_a,
        new_stop.party_b,
        new_stop.horizon_step,
    )
    assert old_stop.min_distance_m == new_stop.min_distance_m == -0.0172764

    old_gt, new_gt = old.scene("utensil").ground_truth, new.scene("utensil").ground_truth
    assert old_gt is not None and new_gt is not None
    assert old_gt.discrepancy_m == new_gt.discrepancy_m == pytest.approx(0.0605324, abs=1e-6)
    assert old_gt.nearest_tripping_party_m == pytest.approx(0.043256, abs=1e-6)
    assert old_gt.nearest_pair["body_b"] == "stack_2_left_group_3_door_main"

    assert old_gt.budget_source == "grid-quantization"
    assert old_gt.verdict == "unadjudicated"
    assert new_gt.budget_source == "hal-adjudication-budget"
    assert new_gt.admissible_gap_m == pytest.approx(0.08822, abs=1e-6)
    # Both rounds are now withdrawn, for a *third* reason on top of the budget:
    # neither attests its distances. The 08-23 record still names the
    # `within-quantization` it withdraws, which is the reading the ledger cites.
    assert new_gt.verdict == "unadjudicated"
    assert new_gt.probe_distance_certified is False
    assert "withdrawn from 'within-quantization'" in new_gt.unadjudicated_reason


def test_a_lower_bound_budget_can_clear_a_stop_but_never_convict_one() -> None:
    """The asymmetry, stated directly: the voxel term is a lower bound.

    Admissible gap is ``corner_slop(link) + voxel_half_diagonal``, so the voxel
    term alone under-states it: within it stays ``within-quantization``, beyond
    it proves nothing. Driven through the real 2026-08-22 utensil snapshot,
    re-attested as distance-certified (the snapshot's ``+43.256 mm`` is a
    certified re-measurement, separating-axis duality gap 2e-17 m — 2026-08-25
    correction, ``docs/reference/collision-validation-evidence.md``) so the
    instrument rule doesn't withdraw both arms before the asymmetry is
    reachable. The unattested form is exercised directly below.
    """
    from openral_core import ValidationStopEvidence

    lines = (
        (ROUND_0822 / "utensil1" / "seed1_deploy_excerpt.log")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    recorded = validation_matrix.parse_json_log_line(lines, "sim.estop_ground_truth_snapshot")
    assert recorded is not None
    snapshot = {
        **recorded,
        "nearest_probe_coverage": {**recorded["nearest_probe_coverage"], "uncertified_pairs": 0},
    }

    def _verdict(min_distance_m: float) -> str:
        stop = ValidationStopEvidence(
            kind="world",
            party_a="panda_link1",
            party_b="voxel_76001",
            horizon_step=-1,
            min_distance_m=min_distance_m,
        )
        adjudication = validation_matrix.adjudicate_ground_truth(snapshot, stop, 0.025)
        assert adjudication is not None
        return adjudication.verdict

    # nearest_tripping_party_m is +43.256 mm, so the discrepancy is
    # 0.043256 - min_distance_m against a 21.651 mm voxel term.
    assert _verdict(0.03) == "within-quantization"  # 13.3 mm, inside the bound
    assert _verdict(-0.0172764) == "unadjudicated"  # 60.5 mm, beyond it: unknown

    # And as the round actually recorded it — no distance attestation — the
    # instrument rule withdraws the cleared arm as well, naming what it takes.
    unattested = validation_matrix.adjudicate_ground_truth(
        recorded,
        ValidationStopEvidence(
            kind="world",
            party_a="panda_link1",
            party_b="voxel_76001",
            horizon_step=-1,
            min_distance_m=0.03,
        ),
        0.025,
    )
    assert unattested is not None
    assert unattested.verdict == "unadjudicated"
    assert unattested.probe_distance_certified is False
    assert "withdrawn from 'within-quantization'" in unattested.unadjudicated_reason


def test_sink_cup_payload_contact_rests_on_a_region_marker(tmp_path: Path) -> None:
    """−1.8 mm, but against ``obj_reg_bbox`` — the payload's own region marker.

    Published as ``real-contact``, but the pair is the payload's bbox marker
    (no ``contype``/``conaffinity``) against a counter top — the probe never
    filtered the payload side, so it measured its own marker. Honest verdict is
    ``unadjudicated``; the numbers stay recorded.
    """
    sink = _derived(ROUND_0822, tmp_path).scene("sink_cup")
    assert sink.stop is not None
    assert sink.stop.involves_payload is True
    assert sink.ground_truth is not None
    assert sink.ground_truth.verdict == "unadjudicated"
    assert sink.ground_truth.probe_collidability_filtered is False
    assert "visual mesh" in sink.ground_truth.unadjudicated_reason
    assert sink.ground_truth.stop_class == "attached_payload"
    assert sink.ground_truth.nearest_any_m == pytest.approx(-0.001759, abs=1e-6)
    assert sink.ground_truth.nearest_pair["geom_a"] == "obj_reg_bbox"
    assert sink.ground_truth.payload_contacts == 6


def test_fridge_zero_metre_pair_is_a_visual_shell(tmp_path: Path) -> None:
    """The probe's own caveat cuts both ways: 0.000 m is not a contact oracle.

    ``payload_contacts == 0`` and ``robot_world_contacts == 0``, so the contact
    list says nothing. The 0.000 m pair read as the answer is ``robot0_g25_vis``,
    a **visual** geom on ``robot0_link6``; the nearest collision geom on that
    link is 16.1 mm clear — a distance to a geom MuJoCo can never contact isn't
    a penetration.
    """
    fridge = _derived(ROUND_0822, tmp_path).scene("fridge")
    assert fridge.ground_truth is not None
    assert fridge.ground_truth.verdict == "unadjudicated"
    assert fridge.ground_truth.payload_contacts == 0
    assert fridge.ground_truth.nearest_any_m == 0.0
    assert fridge.ground_truth.nearest_pair["geom_a"] == "robot0_g25_vis"
    assert fridge.ground_truth.nearest_pair["body_b"] == "fridge_main_group_freezer_door"
    assert fridge.ground_truth.sim_time_s == pytest.approx(4.85, abs=0.01)


def test_scene_configs_are_hashed_so_a_round_pins_its_scene(tmp_path: Path) -> None:
    """Every verdict carries the digest of the YAML that actually ran.

    For these rounds that's the per-round copy beside the artifacts (not a
    tracked scene) — they pinned the stack in the scene's ``runtime:`` block,
    the surface the harness now materialises for itself.
    """
    import hashlib

    work = tmp_path / ROUND_0822.name
    shutil.copytree(ROUND_0822, work)
    assert validation_matrix.cmd_verdicts(work) == 0
    from openral_core import ValidationRoundVerdicts

    verdicts = ValidationRoundVerdicts.from_json(str(work / "verdicts.json"))
    for scene in verdicts.scenes:
        ran = work / scene.config_path
        assert ran.is_file(), scene.config_path
        assert scene.config_sha256 == hashlib.sha256(ran.read_bytes()).hexdigest()
        # The stack these rounds pinned lives in the scene, not in an argv.
        assert "enable_reasoner: false" in ran.read_text(encoding="utf-8")


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
    """Same code, same scene, same seed — two different runs, one bucket.

    08-16 tripped during carry with a deeper cell exempted; 08-22 carried
    cleanly and tripped later on an arm link with nothing exempted. Both bucket
    ``unadjudicated``, so ``changed`` is ``False`` and ``changed_scenes`` omits
    baguette — the flag is outcome-keyed, so evidence that moved completely can
    still read "same"; ``changed_fields`` is what shows the real move.
    """
    current = _derived(ROUND_0822, tmp_path / "cur")
    baseline = _derived(ROUND_0816, tmp_path / "base")
    diff = validation_matrix.diff_rounds(current, baseline)

    assert diff.same_sha is True
    assert diff.baseline_round_id == "2026-08-16-master-1"
    assert diff.is_reproducibility is True  # same sha AND same seed

    baguette = next(s for s in diff.scenes if s.scene == "baguette")
    assert baguette.baseline_outcome == baguette.outcome == "estop-collision-unadjudicated"
    assert baguette.changed is False
    assert "baguette" not in diff.changed_scenes

    # ...while the run underneath it changed almost entirely.
    moves = baguette.changed_fields
    assert moves["stop.party_a"] == {"from": "attached:sim:obj_main", "to": "panda_link5"}
    assert moves["stop.min_distance_m"]["to"] == -0.0209178
    assert moves["stop.sweep_min_distance_m"]["from"] == -0.0355338
    assert moves["witness.kernel_witness_separated"] == {"from": 0, "to": 1}


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
    records = validation_matrix.read_monitor(ROUND_0822 / "bag1" / "seed1_monitor.jsonl")
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
        for ln in (work / "fridge1" / "seed1_deploy_excerpt.log")
        .read_text(encoding="utf-8")
        .splitlines()
        if "sim.estop_ground_truth_snapshot" in ln
    )
    snapshot = json.loads(snapshot_line[snapshot_line.find("{") :])
    record = initial_configuration_stop_record(
        snapshot, stop_seq=1, last_action_ns=0, candidate_chunks_seen=0
    )
    assert record is not None and record["violation"] == "initial_configuration"
    with (work / "fridge1" / "seed1_deploy_excerpt.log").open("a", encoding="utf-8") as sink:
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
    assert verdicts.scene("sink_cup").outcome == "estop-collision-unadjudicated"


# ── 4. The pinned stack ───────────────────────────────────────────────────────


def _deploy_sim_option_strings() -> set[str]:
    """Every option string the real ``openral deploy sim`` command accepts.

    Built from the live Typer app — the same parser click runs, not a hand-maintained list.
    """
    from openral_cli.main import app
    from typer.main import get_command

    sim = get_command(app).commands["deploy"].commands["sim"]  # type: ignore[attr-defined]  # reason: click.Group
    return {opt for param in sim.params for opt in [*param.opts, *param.secondary_opts]}


def test_every_pinned_flag_exists_on_the_real_deploy_cli() -> None:
    """The round that died in every scene in under a second, as a test.

    ``STACK_ARGV`` once pinned ``--no-enable-reasoner``, a flag ``openral
    deploy sim`` never had (click exited 2 before the ROS graph started, four
    times). Every pinned flag is now checked against the real click command's
    parameter list.
    """
    known = _deploy_sim_option_strings()
    pinned = [token for token in validation_matrix.STACK_ARGV if token.startswith("--")]
    assert pinned, "the matrix pins no flags at all"
    assert set(pinned) <= known, sorted(set(pinned) - known)


def test_the_reasoner_has_no_cli_flag_so_it_is_pinned_in_the_scene() -> None:
    """The eighth knob: ``enable_reasoner`` is resolved from the scene, not argv.

    If ``deploy sim`` ever grows a reasoner flag, this goes red and the pin
    can move to the argv.
    """
    assert not [opt for opt in _deploy_sim_option_strings() if "reasoner" in opt]
    assert dict(validation_matrix.SCENE_RUNTIME_PIN) == {"enable_reasoner": False}


def test_the_tracked_scenes_pin_no_reasoner_so_they_default_to_on() -> None:
    """Why materialising a copy is necessary at all, checked against the tree."""
    from openral_core import DeployScene, load_scene_strict

    for spec in validation_matrix.MATRIX:
        scene = load_scene_strict(str(REPO_ROOT / spec.config), expected=DeployScene)
        assert scene.runtime is None or scene.runtime.enable_reasoner is None


def test_materialised_scene_carries_the_pinned_stack_and_the_seed(tmp_path: Path) -> None:
    """The resolved copy is a loadable DeployScene with the reasoner off."""
    from openral_core import DeployScene, load_scene_strict

    for spec in validation_matrix.MATRIX:
        _, resolved = validation_matrix.materialise_scene(spec, 1, tmp_path)
        scene = load_scene_strict(str(resolved), expected=DeployScene)
        assert scene.runtime is not None
        assert scene.runtime.enable_reasoner is False
        assert scene.seed == 1
        # The tracked scene is untouched — that is the whole point of a copy.
        assert "enable_reasoner" not in (REPO_ROOT / spec.config).read_text(encoding="utf-8")


def test_scene_safety_keys_are_refused_but_composition_is_pinnable(tmp_path: Path) -> None:
    """The scene is the second control surface, and it is guarded too."""
    tracked = REPO_ROOT / "scenes/deploy/robocasa_baguette.yaml"
    text = tracked.read_text(encoding="utf-8")

    composition = tmp_path / "composition.yaml"
    composition.write_text(
        validation_matrix.pin_runtime_block(text, [("enable_reasoner", False)]), encoding="utf-8"
    )
    validation_matrix.assert_scene_safety_unmoved(tracked, composition)  # allowed

    gate = tmp_path / "gate.yaml"
    gate.write_text(
        validation_matrix.pin_runtime_block(text, [("enable_octomap_kernel_check", False)]),
        encoding="utf-8",
    )
    with pytest.raises(validation_matrix.GuardrailError, match="safety key"):
        validation_matrix.assert_scene_safety_unmoved(tracked, gate)

    margin = tmp_path / "margin.yaml"
    margin.write_text(text + "\nhal:\n  collision_margin_m: 0.0\n", encoding="utf-8")
    with pytest.raises(validation_matrix.GuardrailError, match="safety key"):
        validation_matrix.assert_scene_safety_unmoved(tracked, margin)


def test_the_place_declaration_cannot_be_moved_by_a_round(tmp_path: Path) -> None:
    """An ADR-0097 declaration grants an exemption, so a round may not retarget it."""
    tracked = REPO_ROOT / "scenes/deploy/robocasa_baguette.yaml"
    retargeted = tmp_path / "retargeted.yaml"
    retargeted.write_text(
        tracked.read_text(encoding="utf-8").replace(
            "sim:cab_1_left_group_main", "sim:counter_1_left_group_main"
        ),
        encoding="utf-8",
    )
    with pytest.raises(validation_matrix.GuardrailError, match="place_declaration"):
        validation_matrix.assert_scene_safety_unmoved(tracked, retargeted)


def test_a_refused_round_leaves_no_directory_behind() -> None:
    """Exit 3, *no partial round*: the directory is created after the guardrails."""
    round_id = "test-guardrail-refusal-leaves-nothing"
    round_dir = validation_matrix.OUTPUT_ROOT / round_id
    assert not round_dir.exists()
    try:
        assert (
            validation_matrix.main(["run", "--round-id", round_id, "--expect-sha", "0" * 40]) == 3
        )
        assert not round_dir.exists()
    finally:
        shutil.rmtree(round_dir, ignore_errors=True)


# ── 5. A launch failure is never a deadline ───────────────────────────────────


def test_the_first_live_round_buckets_as_harness_error(tmp_path: Path) -> None:
    """The recorded round in which no scene launched, verdicted by the fixed code.

    Every scene of ``2026-08-22-harness-1`` produced only a six-line click
    usage error. It used to be reported ``deadline-no-grasp``/exit 0, because
    ``artifacts_complete`` was ``bool(deploy_lines)`` and a usage error is lines.
    """
    verdicts = _derived(ROUND_HARNESS_1, tmp_path)
    assert {s.outcome for s in verdicts.scenes} == {"harness-error"}
    for scene in verdicts.scenes:
        assert "--no-enable-reasoner" in scene.harness_error_reason
        assert scene.stop is None
    assert validation_matrix.round_exit_code(verdicts) == 4


def test_a_round_that_really_ran_exits_zero(tmp_path: Path) -> None:
    """The exit code is a harness-error signal, not a "some scene stopped" signal."""
    assert validation_matrix.round_exit_code(_derived(ROUND_0822, tmp_path)) == 0


def test_the_launch_failure_marker_clears_artifacts_complete(tmp_path: Path) -> None:
    """The runner's own marker for "the action server never appeared"."""
    scene_dir = tmp_path / "baguette"
    scene_dir.mkdir()
    (scene_dir / "run_deploy.log").write_text("[runtime_node-2] up\n", encoding="utf-8")
    assert (
        validation_matrix.scene_verdict_from_artifacts(
            scene_dir, scene="baguette", config_path="", seed=1
        ).outcome
        == "deadline-no-grasp"
    )
    (scene_dir / f"run_{validation_matrix.LAUNCH_FAILED_MARKER}").write_text(
        "/openral/execute_rskill never appeared\n", encoding="utf-8"
    )
    verdict = validation_matrix.scene_verdict_from_artifacts(
        scene_dir, scene="baguette", config_path="", seed=1
    )
    assert verdict.outcome == "harness-error"
    assert "execute_rskill" in verdict.harness_error_reason


def test_the_notes_name_the_scenes_that_did_not_run(tmp_path: Path) -> None:
    """A harness error has to be legible in the round's own summary."""
    work = tmp_path / ROUND_HARNESS_1.name
    shutil.copytree(ROUND_HARNESS_1, work)
    assert validation_matrix.cmd_verdicts(work) == 0
    notes = (work / "NOTES.md").read_text(encoding="utf-8")
    assert "Harness errors" in notes
    assert "harness-error" in notes


# ── 6. Importing the pre-harness rounds ───────────────────────────────────────


def test_import_maps_the_historical_scene_directories(tmp_path: Path) -> None:
    """bag1/sink1/fridge1/utensil1 + the `seed1` stem, without hand-mapping."""
    from openral_core import ValidationRoundVerdicts

    work = tmp_path / "2026-08-22-master-1"
    shutil.copytree(ROUND_0822, work)
    (work / "metadata.json").unlink()
    assert (
        validation_matrix.main(
            [
                "import-round",
                str(work),
                "--round-id",
                "2026-08-22-master-1",
                "--executed-sha",
                "2edcf67c3b087958d475813fe19234c12e90698c",
                "--stem",
                "seed1",
            ]
        )
        == 0
    )
    metadata = json.loads((work / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["scene_dirs"] == {
        "baguette": "bag1",
        "sink_cup": "sink1",
        "fridge": "fridge1",
        "utensil": "utensil1",
    }
    assert metadata["artifact_stem"] == "seed1"
    verdicts = ValidationRoundVerdicts.from_json(str(work / "verdicts.json"))
    assert verdicts.scene("baguette").outcome == "estop-collision-unadjudicated"


def test_import_reads_the_stack_out_of_the_rounds_own_log() -> None:
    """The imported stack is the resolved launch argv, not anyone's recollection.

    These rounds pinned the reasoner in their scene YAML and passed only
    ``--hal viewer_enabled=false --no-dashboard`` on the command line.
    """
    metadata = json.loads((ROUND_0822 / "metadata.json").read_text(encoding="utf-8"))
    assert "enable_reasoner:=false" in metadata["stack_argv"]
    assert "enable_octomap_kernel_check:=true" in metadata["stack_argv"]
    assert not [token for token in metadata["stack_argv"] if token.startswith("--")]
    # Derived from the same line: which checkout and which robot manifest ran.
    assert metadata["robot_id"] == "panda_mobile"
    assert metadata["robot_manifest_path"] == "robots/panda_mobile/robot.yaml"
    assert metadata["repo_root"].endswith("openral-matrix-baseline")
    # Not derivable from these artifacts, so not invented.
    assert metadata["worktree_clean"] is None


def test_import_dates_the_round_from_its_first_ros_timestamp() -> None:
    """A pre-harness round recorded no start time; its log did."""
    stamped = ["[node-1] [INFO] [1787422850.534169223] [x]: up"]
    assert validation_matrix.parse_log_start_time(stamped) == "2026-08-22T18:20:50.534169+00:00"
    assert validation_matrix.parse_log_start_time(["no stamp here"]) is None


def test_import_refuses_a_round_whose_scenes_did_not_share_a_stack(tmp_path: Path) -> None:
    """Two different stacks are two rounds; recording them as one would lie."""
    work = tmp_path / "mixed"
    shutil.copytree(ROUND_0822, work)
    (work / "metadata.json").unlink()
    log = work / "sink1" / "seed1_deploy_excerpt.log"
    log.write_text(
        log.read_text(encoding="utf-8").replace("enable_slam:=true", "enable_slam:=false"),
        encoding="utf-8",
    )
    assert (
        validation_matrix.main(
            [
                "import-round",
                str(work),
                "--round-id",
                "mixed",
                "--executed-sha",
                "2edcf67c3b087958d475813fe19234c12e90698c",
                "--stem",
                "seed1",
            ]
        )
        == 2
    )
    assert not (work / "metadata.json").exists()


# ── 7. The 2026-08-23 defects ─────────────────────────────────────────────────


def test_every_monitor_of_the_0823_round_received_nothing(tmp_path: Path) -> None:
    """The blocking defect, as the recorded artifacts state it.

    ``openral deploy sim`` unlinks every ``/dev/shm/fastrtps_*`` this user owns
    just before spawning ``ros2 launch``; the monitor's participant, created
    ~6 ms earlier, lost its segments and never received anything again. All 24
    runs of that round wrote exactly two lines — a monitor that stopped early
    still has records.
    """
    verdicts = _derived(ROUND_0823, tmp_path)
    assert [s.monitor_records for s in verdicts.scenes] == [0, 0, 0, 0]
    for scene in verdicts.scenes:
        records = validation_matrix.read_monitor(ROUND_0823 / scene.scene / "run_monitor.jsonl")
        assert [r["event"] for r in records] == ["monitor_started", "monitor_stopped"]
        assert scene.ground_truth is not None
        assert scene.ground_truth.grid_resolution_m is None


def test_a_deaf_monitor_reads_differently_from_an_early_stop(tmp_path: Path) -> None:
    """ "Monitor received nothing" and "stopped before a grid" are not one thing.

    Both surface as ``grid_resolution_m: null``: one is a harness fault whose
    evidence is missing, the other a fact about the run. The 2026-08-23 NOTES
    attributed all four scenes to the early stop.
    """
    work = tmp_path / ROUND_0823.name
    shutil.copytree(ROUND_0823, work)
    assert validation_matrix.cmd_verdicts(work) == 0
    notes = (work / "NOTES.md").read_text(encoding="utf-8")
    assert "**Monitor received nothing**" in notes
    assert "baguette, sink_cup, fridge, utensil" in notes
    # ...and the early-stop line is left saying "none", because that is not
    # what happened here.
    early = next(ln for ln in notes.splitlines() if "before the monitor saw a voxel grid" in ln)
    assert early.rstrip().endswith("none.")


def test_a_deaf_monitor_names_itself_in_the_adjudication() -> None:
    """The same distinction inside ``verdicts.json``, not only in the notes.

    On the 2026-08-22 utensil snapshot (predates the HAL's published budget,
    so grid resolution is the only source of one), deaf-monitor and
    early-stop both read ``grid_resolution_m: null`` until the reason says which.
    """
    lines = (
        (ROUND_0822 / "utensil1" / "seed1_deploy_excerpt.log")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    snapshot = validation_matrix.parse_json_log_line(lines, "sim.estop_ground_truth_snapshot")
    stop = validation_matrix.parse_kernel_collision(lines)
    assert snapshot is not None and stop is not None
    deaf = validation_matrix.adjudicate_ground_truth(snapshot, stop, None, monitor_records=0)
    assert deaf is not None
    assert deaf.verdict == "unadjudicated"
    assert "harness reason" in deaf.unadjudicated_reason

    early = validation_matrix.adjudicate_ground_truth(snapshot, stop, None, monitor_records=412)
    assert early is not None
    assert early.verdict == "unadjudicated"
    assert "stopped before the monitor saw a voxel grid" in early.unadjudicated_reason
    assert "harness reason" not in early.unadjudicated_reason


def test_the_monitor_gate_waits_for_the_deploys_own_readiness_line(tmp_path: Path) -> None:
    """The gate is the deploy CLI's marker, read out of the live deploy log.

    Uses the real constant ``openral deploy sim`` prints, so a rename on
    either side goes red.
    """
    import subprocess

    from openral_cli.deploy_sim import DDS_TRANSPORT_READY_MARKER

    log = tmp_path / "run_deploy.log"
    log.write_text(
        "  argv: ros2 launch openral_rskill_ros deploy_e2e.launch.py\n", encoding="utf-8"
    )
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        # Not there yet: the purge has not run, so joining now loses the SHM.
        assert (
            validation_matrix.wait_for_dds_transport_ready(log, proc, timeout_s=0.3, poll_s=0.05)
            == ""
        )
        with log.open("a", encoding="utf-8") as sink:
            sink.write(f"  {DDS_TRANSPORT_READY_MARKER} rmw=default shm_purged=41\n")
        ready = validation_matrix.wait_for_dds_transport_ready(
            log, proc, timeout_s=5.0, poll_s=0.05
        )
        assert DDS_TRANSPORT_READY_MARKER in ready
        assert "shm_purged=41" in ready
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_the_gate_gives_up_when_the_deploy_dies(tmp_path: Path) -> None:
    """A deploy that exits before printing the marker must not hang the round."""
    import subprocess

    log = tmp_path / "run_deploy.log"
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait(timeout=30)
    assert validation_matrix.wait_for_dds_transport_ready(log, proc, timeout_s=30.0) == ""


def test_an_aborted_launch_is_a_harness_error_not_a_deadline(tmp_path: Path) -> None:
    """The 2026-08-23 nav143 round: ``ros2 launch`` threw and left a partial graph.

    ``payload_footprint_node.py`` was missing from the built overlay; launch
    unwound but already-spawned nodes kept running and logging, with no marker
    file and no usage banner — used to bucket ``deadline-no-grasp``.
    """
    verdicts = _derived(ROUND_NAV143, tmp_path)
    assert {s.outcome for s in verdicts.scenes} == {"harness-error"}
    for scene in verdicts.scenes:
        assert "ros2 launch aborted" in scene.harness_error_reason
        assert "payload_footprint_node.py" in scene.harness_error_reason
    assert validation_matrix.round_exit_code(verdicts) == 4


def test_a_crashed_dispatcher_names_itself(tmp_path: Path) -> None:
    """``dispatch_failure_reason`` was empty on a run that never dispatched.

    The dispatcher prints one JSON line, unless it raises — then its log is a
    Python traceback with no ``status`` at all.
    """
    verdicts = _derived(ROUND_NAV143, tmp_path)
    for scene in verdicts.scenes:
        assert scene.dispatch_failure_reason.startswith("the dispatcher raised")
        assert "/openral/execute_rskill unavailable" in scene.dispatch_failure_reason
        assert scene.wall_s is None


def test_a_dispatcher_that_never_wrote_a_status_is_not_a_deadline(tmp_path: Path) -> None:
    """Even with the launch itself clean, no status line means no result."""
    scene_dir = tmp_path / "baguette"
    scene_dir.mkdir()
    (scene_dir / "run_deploy.log").write_text("[runtime_node-2] up\n", encoding="utf-8")
    shutil.copy2(ROUND_NAV143 / "baguette" / "run_goal.log", scene_dir / "run_goal.log")
    verdict = validation_matrix.scene_verdict_from_artifacts(
        scene_dir, scene="baguette", config_path="", seed=1
    )
    assert verdict.outcome == "harness-error"
    assert "the dispatcher raised" in verdict.harness_error_reason


def test_a_real_deadline_still_reads_as_a_deadline(tmp_path: Path) -> None:
    """The dispatcher's own overrun line is a status, so it must survive.

    ``{"status": -1, ...}`` is what ``_validation_matrix_dispatch.py`` prints
    when the action overruns its deadline with no result — a fact about the
    run, not a harness failure.
    """
    scene_dir = tmp_path / "baguette"
    scene_dir.mkdir()
    (scene_dir / "run_deploy.log").write_text("[runtime_node-2] up\n", encoding="utf-8")
    (scene_dir / "run_goal.log").write_text(
        json.dumps({"latest_chunk": 7, "status": -1, "success": False}) + "\n", encoding="utf-8"
    )
    verdict = validation_matrix.scene_verdict_from_artifacts(
        scene_dir, scene="baguette", config_path="", seed=1
    )
    assert verdict.outcome == "deadline-no-grasp"
    assert verdict.harness_error_reason == ""


def test_the_harness_uses_the_hals_budget_not_a_narrower_one(tmp_path: Path) -> None:
    """88.2 mm from the snapshot, not 21.7 mm recomputed from the grid.

    Admissible gap is the collision model's corner slop plus the voxel term
    (OBB-to-voxel kernel vs mesh-to-mesh probe); the HAL publishes it per run.
    The harness used to recompute only the voxel half-diagonal and call
    conservative stops false positives with it. The 2026-08-23 ``utensil``
    stop is that call: ``robot0_link1`` 43.3 mm clear vs a kernel read of
    −17.3 mm (60.5 mm discrepancy) — narrow term says false positive, HAL
    budget says ``within-quantization`` (kernel correct). Since the
    ``mj_geomDistance`` characterisation the verdict is withdrawn on top of
    that (the 43.3 mm was measured with an instrument that can't attest
    itself); the withdrawal names what it withdraws rather than erasing it.
    The 43.3 mm is independently confirmed by a certified re-measurement
    (2026-08-25 correction, ``docs/reference/collision-validation-evidence.md``),
    a cross-round inference the harness itself does not make.
    """
    utensil = _derived(ROUND_0823, tmp_path).scene("utensil")
    ground_truth = utensil.ground_truth
    assert ground_truth is not None
    assert ground_truth.budget_source == "hal-adjudication-budget"
    assert ground_truth.admissible_gap_m == pytest.approx(0.08822, abs=1e-6)
    assert ground_truth.discrepancy_m == pytest.approx(0.0605324, abs=1e-6)
    # ...and it is then WITHDRAWN, because this round's distances were measured
    # with `mj_geomDistance`. The record still names what it withdraws, so the
    # budget arithmetic under test is visible in it.
    assert ground_truth.verdict == "unadjudicated"
    assert ground_truth.probe_distance_certified is False
    assert "withdrawn from 'within-quantization'" in ground_truth.unadjudicated_reason
    # The narrow term the harness used to apply, for the same stop.
    assert validation_matrix.quantization_budget_m(0.025) == pytest.approx(0.021651, abs=1e-6)
    assert ground_truth.discrepancy_m > validation_matrix.quantization_budget_m(0.025)
    # ...and it is still recorded when a grid resolution is known. This round's
    # monitor was deaf, so it is not.
    assert ground_truth.quantization_budget_m is None


def test_the_hal_budget_is_read_out_of_the_recorded_snapshot() -> None:
    """Straight from the artifact, for both the world and the self-stop case."""
    from openral_core import ValidationStopEvidence

    lines = (
        (ROUND_0823 / "baguette" / "run_deploy_excerpt.log")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    snapshot = validation_matrix.parse_json_log_line(lines, "sim.estop_ground_truth_snapshot")
    assert snapshot is not None
    world = ValidationStopEvidence(
        kind="world",
        party_a="panda_link5",
        party_b="voxel_1",
        horizon_step=0,
        min_distance_m=-0.01,
    )
    payload_self = ValidationStopEvidence(
        kind="self",
        party_a="attached:sim:obj_main",
        party_b="panda_link2",
        horizon_step=0,
        min_distance_m=-0.01,
    )
    assert validation_matrix.hal_admissible_gap_m(snapshot, world) == pytest.approx(0.08822)
    # A payload self stop has an OBB on both sides and no voxel: its own block.
    assert validation_matrix.hal_admissible_gap_m(snapshot, payload_self) == pytest.approx(0.124555)
    # A snapshot recorded BEFORE #266's payload-world block still resolves —
    # to the old top-level number. Absence must read as "this round predates
    # the block", never as "this stop has no budget", which would turn every
    # archived payload-world stop `unadjudicated` at a stroke.
    payload_world = ValidationStopEvidence(
        kind="world",
        party_a="attached:sim:obj_main",
        party_b="voxel_1",
        horizon_step=0,
        min_distance_m=-0.01,
    )
    assert validation_matrix.hal_admissible_gap_m(snapshot, payload_world) == pytest.approx(0.08822)


def test_a_payload_world_stop_is_charged_the_payload_not_the_link() -> None:
    """#266. The class that is 97 % of the 15 mm A/B's stops, finally its own budget.

    A payload-vs-world stop is one payload model against one voxel cube — the
    robot's links are not a party to it. Routing it to the top-level block
    charged it `corner_slop(worst LINK) + voxel_half_diagonal`, a composition
    for a pair the stop does not involve, and on the 2026-08-23 rounds that was
    88.22 mm (``panda_link4``) where the payload's own model needs ~31 mm.

    An over-large budget does not fail loudly; it silently excuses. That is the
    direction that hides a real defect, which is why this is a fix and not a
    refinement.
    """
    from openral_core import ValidationStopEvidence

    snapshot = {
        "adjudication_budget": {
            "admissible_gap_m": 0.08822,  # the LINK composition, still right for an arm stop
            "payload_world_voxel": {
                "max_payload_model_overhang_m": 0.01018,
                "voxel_half_diagonal_m": 0.021651,
                "admissible_gap_m": 0.031831,
                "payload_slop": {"max_corner_slop_m": 0.03488},
            },
        }
    }
    payload_world = ValidationStopEvidence(
        kind="world",
        party_a="attached:sim:obj_main",
        party_b="voxel_9",
        horizon_step=0,
        min_distance_m=-0.01,
    )
    arm_world = ValidationStopEvidence(
        kind="world",
        party_a="panda_link4",
        party_b="voxel_9",
        horizon_step=0,
        min_distance_m=-0.01,
    )
    assert validation_matrix.hal_admissible_gap_m(snapshot, payload_world) == pytest.approx(
        0.031831
    )
    # The arm's own stops are untouched: they really are link-vs-voxel.
    assert validation_matrix.hal_admissible_gap_m(snapshot, arm_world) == pytest.approx(0.08822)
    # And the payload's budget is the MODEL's, not the box's — charging the box
    # would have handed this stop 24.7 mm the kernel's model does not use.
    assert 0.031831 < 0.03488 + 0.021651


def test_the_arm_world_budget_rederives_its_voxel_term_too() -> None:
    """The same defect one block over, found by #266 and fixed with it.

    Every snapshot in the 2026-09-11 A/B published ``voxel_half_diagonal_m:
    0.0`` — the monitor handed ``estop_ground_truth_snapshot`` no
    ``evidence_voxel`` — so every ARM-stop budget was the link term alone:
    88.22 mm where it should have been 109.87 mm. Understated by 21.65 mm, which
    is 25-48 % of a Panda link's 45-88 mm slop.

    It hid here precisely because the link term is large; on the payload block
    the 8.9-19.9 mm overhang made the same omission impossible to miss. The
    direction is the one that matters: an under-stated budget convicts a
    conservative, correct stop.
    """
    from openral_core import ValidationStopEvidence

    arm = ValidationStopEvidence(
        kind="world",
        party_a="panda_link4",
        party_b="voxel_7",
        horizon_step=0,
        min_distance_m=-0.01,
    )
    deaf = {
        "adjudication_budget": {
            "max_corner_slop_m": 0.08822,
            "voxel_half_diagonal_m": 0.0,
            "admissible_gap_m": 0.08822,  # the link term alone
        }
    }
    assert validation_matrix.hal_admissible_gap_m(deaf, arm, 0.025) == pytest.approx(
        0.08822 + validation_matrix.quantization_budget_m(0.025)
    )
    # A round whose monitor DID deliver a voxel keeps its published number.
    heard = {
        "adjudication_budget": {
            "max_corner_slop_m": 0.08822,
            "voxel_half_diagonal_m": 0.021651,
            "admissible_gap_m": 0.109871,
        }
    }
    assert validation_matrix.hal_admissible_gap_m(heard, arm, 0.025) == pytest.approx(0.109871)
    # No slop term to compose with, or no resolution: the published number
    # stands rather than being replaced by a guess.
    bare = {"adjudication_budget": {"admissible_gap_m": 0.08822}}
    assert validation_matrix.hal_admissible_gap_m(bare, arm, 0.025) == pytest.approx(0.08822)
    assert validation_matrix.hal_admissible_gap_m(deaf, arm, None) == pytest.approx(0.08822)


def test_a_zero_voxel_term_is_rederived_not_composed_with() -> None:
    """The bug the first 2026-09-11 A/B run exposed, in this budget itself.

    ``estop_ground_truth_snapshot`` fills ``voxel_half_diagonal_m`` only from an
    ``evidence_voxel`` it was handed; a round whose monitor never delivered one
    publishes ``0.0``. On the top-level block that omission hides behind a
    45-88 mm link term. On the payload block it does not: the payload's own
    overhang is 8.9-19.9 mm, the **same order** as the 21.65 mm being dropped,
    so composing with zero roughly halves the budget.

    Measured consequence: the first A/B run flagged **6 of 16** hull-arm stops
    ``false-positive``, every one of them inside budget once the term was
    restored (6 -> 1, which is the base arm's own count). An under-stated budget
    cries wolf, the one direction an adjudicator must not fail in, so the term
    is re-derived from the round's known grid resolution — and with no
    resolution to re-derive from the budget is ``None`` (``unadjudicated``: "I
    cannot judge this") rather than a number that convicts.
    """
    from openral_core import ValidationStopEvidence

    snapshot = {
        "adjudication_budget": {
            "admissible_gap_m": 0.08822,
            "payload_world_voxel": {
                "max_payload_model_overhang_m": 0.008907,
                "voxel_half_diagonal_m": 0.0,  # the deaf-monitor round
                "admissible_gap_m": 0.008907,
            },
        }
    }
    stop = ValidationStopEvidence(
        kind="world",
        party_a="attached:sim:obj_main",
        party_b="voxel_3",
        horizon_step=0,
        min_distance_m=-0.01,
    )
    assert validation_matrix.hal_admissible_gap_m(snapshot, stop, 0.025) == pytest.approx(
        0.008907 + validation_matrix.quantization_budget_m(0.025)
    )
    # A utensil round's 18.30 mm discrepancy sits INSIDE the restored budget and
    # outside the zero-composed 8.907 mm -- the exact flip the bug produced.
    utensil_discrepancy_m = 0.01830
    zero_composed_m = 0.008907
    assert zero_composed_m < utensil_discrepancy_m
    assert utensil_discrepancy_m < zero_composed_m + validation_matrix.quantization_budget_m(0.025)
    # No resolution to re-derive from: no budget, rather than a convicting one.
    assert validation_matrix.hal_admissible_gap_m(snapshot, stop, None) is None


def test_the_0823_probe_still_ranks_a_visual_geom_first(tmp_path: Path) -> None:
    """The producer-side defect, in the recorded evidence that exposed it.

    ``robot0_g42_vis`` — a visual shell on ``robot0_link7`` — is reported at
    0.000 m from the freezer door, while the same link's collision geom is
    2.5 mm clear; a mesh MuJoCo can never contact can't support ``real-contact``.
    """
    fridge = _derived(ROUND_0823, tmp_path).scene("fridge")
    assert fridge.ground_truth is not None
    assert fridge.ground_truth.nearest_any_m == 0.0
    assert fridge.ground_truth.nearest_pair["geom_a"] == "robot0_g42_vis"
    assert fridge.ground_truth.nearest_pair["geom_b"] == "fridge_main_group_g43"
    assert fridge.ground_truth.probe_collidability_filtered is False
    assert fridge.ground_truth.verdict == "unadjudicated"
    assert "visual mesh" in fridge.ground_truth.unadjudicated_reason
    # The bucket is the initial-configuration one because this stop landed
    # before any action reached the HAL, which outranks the adjudication —
    # `baguette` is the same round's plain collision stop.
    assert fridge.outcome == "estop-initial-configuration"
    baguette = _derived(ROUND_0823, tmp_path / "again").scene("baguette")
    assert baguette.outcome == "estop-collision-unadjudicated"


def test_a_seed_change_at_one_sha_is_not_reproducibility(tmp_path: Path) -> None:
    """The mislabelled comparison: equal SHAs, different scene.

    Seed decides initial configuration, so two rounds at one SHA on different
    seeds are a before/after of two different scenes.
    """
    from openral_core import ValidationRoundVerdicts

    current = _derived(ROUND_0823, tmp_path / "cur")
    work = tmp_path / "base" / ROUND_0823.name
    shutil.copytree(ROUND_0823, work)
    metadata = json.loads((work / "metadata.json").read_text(encoding="utf-8"))
    metadata["seed"] = 2
    metadata["round_id"] = "2026-08-23-master-s2"
    (work / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    assert validation_matrix.cmd_verdicts(work) == 0
    baseline = ValidationRoundVerdicts.from_json(str(work / "verdicts.json"))

    diff = validation_matrix.diff_rounds(current, baseline)
    assert diff.same_sha is True
    assert (diff.seed, diff.baseline_seed) == (1, 2)
    assert diff.same_seed is False
    assert diff.is_reproducibility is False


def test_the_matching_seed_is_still_a_reproducibility_diff(tmp_path: Path) -> None:
    """Same code, same seed — the comparison the label is meant for."""
    current = _derived(ROUND_0823, tmp_path / "cur")
    baseline = _derived(ROUND_0823, tmp_path / "base")
    diff = validation_matrix.diff_rounds(current, baseline)
    assert diff.is_reproducibility is True


def test_collision_scale_env_is_empty_when_the_band_is_not_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A baseline round records no band, which is the shipped default."""
    for env in (
        "OPENRAL_COLLISION_SCALE_PROXIMITY_M",
        "OPENRAL_COLLISION_SCALE_K",
        "OPENRAL_COLLISION_SCALE_MIN",
    ):
        monkeypatch.delenv(env, raising=False)
    assert validation_matrix.collision_scale_env() == {}


def test_collision_scale_env_records_an_armed_band(monkeypatch: pytest.MonkeyPatch) -> None:
    """An A/B round must be distinguishable from a baseline one afterwards.

    The #188 band reaches the kernel through env vars, which
    ``assert_no_safety_overrides`` cannot see (it inspects argv only) — without
    this the metadata for an armed and an unarmed round would be identical.
    """
    monkeypatch.setenv("OPENRAL_COLLISION_SCALE_PROXIMITY_M", "0.05")
    monkeypatch.setenv("OPENRAL_COLLISION_SCALE_K", "20")
    monkeypatch.delenv("OPENRAL_COLLISION_SCALE_MIN", raising=False)
    assert validation_matrix.collision_scale_env() == {
        "collision_scale_proximity_m": 0.05,
        "collision_scale_k": 20.0,
    }


def test_collision_scale_env_ignores_a_value_the_launch_would_ignore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unparseable value is not recorded, because it did not take effect.

    ``deploy_e2e.launch.py`` drops an unparseable band rather than guessing, so
    recording it would misdescribe the round.
    """
    monkeypatch.setenv("OPENRAL_COLLISION_SCALE_PROXIMITY_M", "0,05")
    monkeypatch.delenv("OPENRAL_COLLISION_SCALE_K", raising=False)
    monkeypatch.delenv("OPENRAL_COLLISION_SCALE_MIN", raising=False)
    assert validation_matrix.collision_scale_env() == {}


def test_a_link_vs_link_self_stop_is_not_scored_against_world_geometry() -> None:
    """Issue: `panda_link5`/`panda_link7` scored `false-positive` off the island.

    From the 2026-09-04 post-#200 battery (this stop hit twice, independently).
    The kernel named a self-pair; the snapshot's robot-world probe excludes the
    whole robot, so `robot0_link5`'s nearest pair is a kitchen island 212 mm
    away. Comparing that against a -31.97 mm self-collision depth used to give
    a 244 mm discrepancy against an 88 mm budget and `false-positive` — for a
    stop whose exact hulls interpenetrate. The snapshot holds no evidence about
    link5-vs-link7, so the only honest verdict is `unadjudicated`.
    """
    lines = (
        (ROUND_POST200 / "sink_cup" / "run_deploy_excerpt.log")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    stop = validation_matrix.parse_kernel_collision(lines)
    assert stop is not None
    assert (stop.kind, stop.party_a, stop.party_b) == ("self", "panda_link5", "panda_link7")
    assert stop.min_distance_m == pytest.approx(-0.0319657)

    snapshot = validation_matrix.parse_json_log_line(lines, "sim.estop_ground_truth_snapshot")
    assert snapshot is not None
    # The evidence path is otherwise sound: certified, untruncated, budgeted.
    assert validation_matrix.probe_is_distance_certified(snapshot) is True
    robot_pairs = snapshot["nearest_robot_world_pairs"]
    assert not any(str(p["body_b"]).startswith("robot0_") for p in robot_pairs)
    link5_world = min(p["distance_m"] for p in robot_pairs if p["body_a"] == "robot0_link5")
    assert link5_world == pytest.approx(0.212256613)

    adjudication = validation_matrix.adjudicate_ground_truth(snapshot, stop, 0.025)
    assert adjudication is not None
    assert adjudication.verdict == "unadjudicated"
    assert "self-pair" in adjudication.unadjudicated_reason
    # link5's clearance to the world must not be published as the tripping
    # party's, nor turned into a discrepancy against a self-collision depth.
    assert adjudication.nearest_tripping_party_m is None
    assert adjudication.discrepancy_m is None


# ── #216: a link-vs-link self stop, now that the HAL probes the pair ──────────
#
# #208 stopped the harness scoring such a stop against WORLD geometry. It did
# not make it scorable: the pair was structurally unprobeable. #216 adds the
# link<->link probe, and these pin what the harness may now conclude from it —
# and, just as importantly, what it still may not.


def _link_link_snapshot(
    *,
    pair_distance_m: float,
    max_corner_slop_m: float = 0.0862,
    hull_overhang_a_m: float | None = None,
    hull_overhang_b_m: float | None = None,
) -> dict[str, object]:
    """A snapshot carrying a link<->link pair, in the shape the HAL emits.

    ``hull_overhang_a_m``/``hull_overhang_b_m`` land in
    ``collision_model_slop.links`` (#221's shape, keyed by link name, read by
    ``hal_admissible_gap_m`` for a hull-fidelity stop); omitted, reproduces a
    pre-#221 snapshot.
    """
    links_slop: dict[str, object] = {}
    if hull_overhang_a_m is not None:
        links_slop["panda_link5"] = {"hull_overhang_m": hull_overhang_a_m}
    if hull_overhang_b_m is not None:
        links_slop["panda_link7"] = {"hull_overhang_m": hull_overhang_b_m}
    return {
        "nearest_robot_world_pairs": [
            {
                "body_a": "robot0_link5",
                "body_b": "island_island_group_main",
                "distance_m": 0.212256613,
                "distance_certified": True,
                "distance_method": "gjk",
            }
        ],
        "nearest_probe_coverage": {
            "truncated": False,
            "probed_pairs": 12,
            "certified_pairs": 12,
            "uncertified_pairs": 0,
            "noncollidable_side_geoms_excluded": 3,
            "noncollidable_other_geoms_excluded": 7,
            "distmax_m": 0.2,
        },
        "nearest_link_link_pairs": [
            {
                "body_a": "robot0_link5",
                "body_b": "robot0_link7",
                "distance_m": pair_distance_m,
                "distance_certified": True,
                "distance_method": "gjk",
            }
        ],
        "nearest_link_link_coverage": {
            "truncated": False,
            "probed_pairs": 4,
            "certified_pairs": 4,
            "uncertified_pairs": 0,
            "noncollidable_side_geoms_excluded": 1,
            "noncollidable_other_geoms_excluded": 1,
            "distmax_m": 0.2,
        },
        "adjudication_budget": {
            "max_corner_slop_m": max_corner_slop_m,
            "admissible_gap_m": 0.08822,
            "collision_model_slop": {"links": links_slop},
            "link_link": {
                "max_corner_slop_m": max_corner_slop_m,
                "admissible_gap_box_m": round(2.0 * max_corner_slop_m, 6),
            },
        },
    }


def _link_link_stop(
    *, depth_is_box_bound: bool, min_distance_m: float = -0.0319657
) -> ValidationStopEvidence:
    return ValidationStopEvidence(
        kind="self",
        party_a="panda_link5",
        party_b="panda_link7",
        horizon_step=0,
        min_distance_m=min_distance_m,
        sweep_min_distance_m=min_distance_m,
        depth_is_box_bound=depth_is_box_bound,
    )


def test_a_box_bounded_self_stop_is_adjudicated_against_the_link_link_pair() -> None:
    """The 2026-09-04 stop, scorable at last — and it was a real contact.

    Kernel flagged `depth_is_box_bound` (OBB bound, box term applies); the
    probe puts the two links in contact — the answer #208 could only decline
    to give.
    """
    snapshot = _link_link_snapshot(pair_distance_m=-0.0015)
    adjudication = validation_matrix.adjudicate_ground_truth(
        snapshot, _link_link_stop(depth_is_box_bound=True), 0.025
    )
    assert adjudication is not None
    assert adjudication.verdict == "real-contact"
    # And the number quoted is the SELF pair's, never link5's 212 mm to the island.
    assert adjudication.nearest_tripping_party_m == pytest.approx(-0.0015)


def test_a_hull_self_stop_stays_unadjudicated_when_a_link_has_no_measured_overhang() -> None:
    """Probed, but not scorable — and the reason says which of the two it is.

    Without `depth_is_box_bound` the kernel judged hull fidelity; #221's
    `hull_overhang_m(a) + hull_overhang_m(b)` charge needs BOTH links measured,
    but this snapshot has neither. Charging the box budget anyway would forgive
    a real overlap by up to twice the corner slop (172 mm on `panda_mobile`).
    """
    snapshot = _link_link_snapshot(pair_distance_m=0.004)
    adjudication = validation_matrix.adjudicate_ground_truth(
        snapshot, _link_link_stop(depth_is_box_bound=False), 0.025
    )
    assert adjudication is not None
    assert adjudication.verdict == "unadjudicated"
    assert "hull fidelity" in adjudication.unadjudicated_reason
    assert adjudication.admissible_gap_m is None


def test_a_hull_self_stop_is_adjudicated_once_both_links_have_a_measured_overhang() -> None:
    """#221 closes the gap #220 could only disclose: hull fidelity now scores.

    Hull-fidelity budgets are two orders tighter than the OBB corner-slop
    budget (tenths of a millimetre, not tens); asymmetric per-link values
    prove the budget is a genuine sum, not a maxed/doubled term like the box
    budget.
    """
    snapshot = _link_link_snapshot(
        pair_distance_m=0.0001,
        hull_overhang_a_m=0.00013,
        hull_overhang_b_m=0.00007,
    )
    adjudication = validation_matrix.adjudicate_ground_truth(
        snapshot, _link_link_stop(depth_is_box_bound=False, min_distance_m=-0.00005), 0.025
    )
    assert adjudication is not None
    assert adjudication.verdict == "within-quantization"
    assert adjudication.admissible_gap_m == pytest.approx(0.0002)


def test_a_hull_budget_needs_both_links_measured_not_just_one() -> None:
    """One measured overhang is not a half-budget — it is no budget at all."""
    snapshot = _link_link_snapshot(pair_distance_m=0.0001, hull_overhang_a_m=0.00013)
    adjudication = validation_matrix.adjudicate_ground_truth(
        snapshot, _link_link_stop(depth_is_box_bound=False, min_distance_m=-0.00005), 0.025
    )
    assert adjudication is not None
    assert adjudication.verdict == "unadjudicated"
    assert adjudication.admissible_gap_m is None


def test_a_link_vs_link_stop_never_falls_back_to_the_voxel_budget() -> None:
    """The grid-quantization term is a VOXEL budget, and this stop has no voxel.

    Falling back to it would charge a budget from a comparison the stop never
    made. A grid resolution is passed here precisely so the fallback would
    fire if it were still reachable.
    """
    snapshot = _link_link_snapshot(pair_distance_m=0.004)
    del snapshot["adjudication_budget"]
    adjudication = validation_matrix.adjudicate_ground_truth(
        snapshot, _link_link_stop(depth_is_box_bound=True), 0.025
    )
    assert adjudication is not None
    assert adjudication.verdict == "unadjudicated"
    assert adjudication.budget_source != "grid-quantization"
    assert adjudication.admissible_gap_m is None


def test_an_old_snapshot_without_the_link_link_probe_still_says_unprobed() -> None:
    """The #208 behaviour has to survive for every round recorded before #216."""
    snapshot = _link_link_snapshot(pair_distance_m=-0.0015)
    del snapshot["nearest_link_link_pairs"]
    del snapshot["nearest_link_link_coverage"]
    adjudication = validation_matrix.adjudicate_ground_truth(
        snapshot, _link_link_stop(depth_is_box_bound=True), 0.025
    )
    assert adjudication is not None
    assert adjudication.verdict == "unadjudicated"
    assert "never measures one robot link against another" in adjudication.unadjudicated_reason
    assert adjudication.nearest_tripping_party_m is None


def test_payload_vs_link_stop_is_scored_against_the_link_not_the_world() -> None:
    """#228 — #208 one class over, with the right pair set already in the snapshot.

    The 2026-09-05 `with204-1` baguette round stopped on the CARRIED PAYLOAD
    against a robot LINK (`a=attached:sim:obj_main b=panda_link1`, kind=self,
    -1.55 mm). `involves_payload` used to compare that depth to the payload's
    166 mm clearance to a COUNTERTOP — 167 mm discrepancy vs an 88 mm world
    budget, `false-positive`, off a pair the world probe never measured.

    `nearest_payload_robot_pairs` measures the right pair instead: obj_main
    <-> robot0_link1, certified GJK, +65.5 mm. Against the HAL's self budget
    (124.6 mm = 88.2 link corner slop + 36.3 payload corner slop) a 67 mm
    discrepancy is inside budget — envelope conservatism, not a false positive.

    Fixture is the round's own two log lines, verbatim.
    """
    lines = (
        (ROUND_217_WITH204 / "baguette" / "run_deploy_excerpt.log")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    stop = validation_matrix.parse_kernel_collision(lines)
    assert stop is not None
    assert (stop.kind, stop.party_a, stop.party_b) == (
        "self",
        "attached:sim:obj_main",
        "panda_link1",
    )
    assert stop.involves_payload is True
    assert stop.min_distance_m == pytest.approx(-0.00155364)

    snapshot = validation_matrix.parse_json_log_line(lines, "sim.estop_ground_truth_snapshot")
    assert snapshot is not None
    assert validation_matrix.probe_is_distance_certified(snapshot) is True

    # What the two pair sets say about this payload, so the assertion below is
    # visibly about WHICH one was consulted and not about a lucky number.
    world_nearest = min(p["distance_m"] for p in snapshot["nearest_payload_world_pairs"])
    link1_nearest = min(
        p["distance_m"]
        for p in snapshot["nearest_payload_robot_pairs"]
        if p["body_b"] == "robot0_link1"
    )
    assert world_nearest == pytest.approx(0.166032596)
    assert link1_nearest == pytest.approx(0.065511265)
    self_budget = snapshot["adjudication_budget"]["self_collision"]["admissible_gap_m"]
    assert self_budget == pytest.approx(0.124555)

    adjudication = validation_matrix.adjudicate_ground_truth(snapshot, stop, None)
    assert adjudication is not None
    # The link pair, not the countertop.
    assert adjudication.nearest_tripping_party_m == pytest.approx(link1_nearest)
    assert adjudication.nearest_tripping_party_m != pytest.approx(world_nearest)
    # The attached-payload SELF budget, from the HAL, not the world one.
    assert adjudication.admissible_gap_m == pytest.approx(self_budget)
    assert adjudication.budget_source == "hal-adjudication-budget"
    # 65.5 - (-1.55) = 67.1 mm, inside 124.6 mm: conservative and correct.
    assert adjudication.discrepancy_m == pytest.approx(link1_nearest - stop.min_distance_m)
    assert adjudication.verdict == "within-quantization"


def test_payload_vs_link_with_no_payload_robot_pairs_is_unadjudicated() -> None:
    """Absence of the RIGHT pair set is not evidence; it is a missing instrument.

    Strip `nearest_payload_robot_pairs` (a pre-#220-style snapshot): the stop
    must come back `unadjudicated` naming the pair, never falling through to
    the world probe's clearance.
    """
    lines = (
        (ROUND_217_WITH204 / "baguette" / "run_deploy_excerpt.log")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    stop = validation_matrix.parse_kernel_collision(lines)
    snapshot = validation_matrix.parse_json_log_line(lines, "sim.estop_ground_truth_snapshot")
    assert stop is not None and snapshot is not None
    stripped = {
        k: v
        for k, v in snapshot.items()
        if k not in ("nearest_payload_robot_pairs", "nearest_payload_robot_coverage")
    }

    adjudication = validation_matrix.adjudicate_ground_truth(stripped, stop, None)
    assert adjudication is not None
    assert adjudication.verdict == "unadjudicated", adjudication
    assert adjudication.nearest_tripping_party_m is None
    assert adjudication.discrepancy_m is None


def test_permitted_adjacent_link_overlap_is_not_evidence_of_contact() -> None:
    """An ACM-allowed link overlap must not stamp an unrelated stop `real-contact`.

    Regression for a defect #220 introduced and shipped to `master` on
    2026-09-05. That PR gave the HAL a link-vs-link probe so a self stop could
    finally be scored against the pair the kernel named. The pairs were then
    folded into the adjudicator's `nearest_any`, which drives its first and
    most decisive rule: *any probed pair at or below 0 m → `real-contact`*.

    Adjacent robot links overlap permanently — they are in the robot's
    allowed-collision matrix and the kernel never checks them — so from #220
    onward `nearest_any <= 0` was vacuously true and **every** adjudicable stop
    was stamped `real-contact`, whatever the tripping party's real clearance.

    On this round the kernel stopped the carried payload against a voxel while
    the payload sat **+24.86 mm clear** of the counter. The snapshot also
    records `robot0_link3`/`link4` at −36.3 mm, `link5`/`link6` at −23.0 mm and
    `link4`/`link5` at −4.6 mm: all certified, all permitted, none of them what
    the kernel stopped for. The honest verdict is `within-quantization` — a
    stop of a physically clear robot — and reading it as `real-contact` inverts
    the one measurement the collision programme exists to make.
    """
    from openral_core import ValidationStopEvidence

    lines = (
        (FIXTURES / "2026-09-07-adr0101-live-1" / "utensil" / "run_deploy.log")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    snapshot = validation_matrix.parse_json_log_line(lines, "sim.estop_ground_truth_snapshot")
    assert snapshot is not None

    # The permitted overlaps really are in the record, and really are negative.
    link_link = {
        frozenset((p["body_a"], p["body_b"])): p["distance_m"]
        for p in snapshot["nearest_link_link_pairs"]
    }
    assert link_link[frozenset(("robot0_link3", "robot0_link4"))] < -0.03

    stop = ValidationStopEvidence(
        kind="world",
        party_a="attached:sim:obj_main",
        party_b="voxel_228622",
        horizon_step=0,
        min_distance_m=-0.0040461,
    )
    adjudication = validation_matrix.adjudicate_ground_truth(snapshot, stop, 0.025)
    assert adjudication is not None

    assert adjudication.verdict == "within-quantization", adjudication.verdict
    # `nearest_any` must now describe something the kernel could have stopped
    # for — not the permanently-overlapping pair two joints away.
    assert adjudication.nearest_any_m is not None
    assert adjudication.nearest_any_m > 0.0


def test_a_named_self_pair_still_reaches_nearest_any() -> None:
    """Excluding permitted overlaps must not deafen the self-stop path.

    The fix drops `nearest_link_link_pairs` from `nearest_any` wholesale and
    adds back only the pair the kernel named. If that add-back were missing, a
    genuine link-vs-link self stop in real overlap would stop being detectable
    as contact — trading one blind spot for its mirror image.
    """
    from openral_core import ValidationStopEvidence

    lines = (
        (FIXTURES / "2026-09-07-adr0101-live-1" / "utensil" / "run_deploy.log")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    snapshot = validation_matrix.parse_json_log_line(lines, "sim.estop_ground_truth_snapshot")
    assert snapshot is not None

    # The kernel names the overlapping pair itself: that IS the stop under test.
    stop = ValidationStopEvidence(
        kind="self",
        party_a="panda_link3",
        party_b="panda_link4",
        horizon_step=0,
        min_distance_m=-0.02,
    )
    adjudication = validation_matrix.adjudicate_ground_truth(snapshot, stop, 0.025)
    assert adjudication is not None
    assert adjudication.nearest_any_m is not None
    assert adjudication.nearest_any_m < -0.03, "the named self pair must still be seen"


def test_an_octomap_resolution_override_is_recorded_not_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A round on a finer grid must be distinguishable afterwards from one that was not.

    A finer world-voxel grid shrinks the kernel's quantisation term, so it stops
    *later* and *nearer* -- the override is less conservative, not more.
    ``assert_no_safety_overrides`` inspects the launch argv and cannot see an
    environment variable, so the recording in the round metadata is the only
    thing standing between a 15 mm round and a 25 mm one in the ledger.
    """
    monkeypatch.setenv("OPENRAL_OCTOMAP_RESOLUTION_M", "0.015")
    assert validation_matrix.octomap_resolution_env() == {"octomap_resolution_m": 0.015}


def test_an_octomap_resolution_the_launch_ignores_is_not_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording a value the launch refused would misdescribe the round.

    ``_octomap_resolution`` falls back to the shipped default on anything
    unparseable or outside ``[0.001, 0.5]``, so the round ran at 25 mm. Metadata
    claiming otherwise is worse than metadata saying nothing.
    """
    for bad in ("", "   ", "not-a-number", "0", "-0.015", "1.5"):
        monkeypatch.setenv("OPENRAL_OCTOMAP_RESOLUTION_M", bad)
        assert validation_matrix.octomap_resolution_env() == {}, bad
