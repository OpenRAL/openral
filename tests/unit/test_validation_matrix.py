"""Tests for the validation-matrix harness (``tools/validation_matrix.py``).

CLAUDE.md §1.11 — every input here is a **recorded artifact** copied verbatim
from a real round on the project's DGX Spark (see
``tests/unit/fixtures/validation_matrix/SOURCE.txt``), not a synthesized log.
The assertions are pinned to what
``docs/reference/collision-validation-evidence.md`` concluded about those
rounds, so if the extractor ever stops reproducing the published ledger these
go red.

Five sections:

1. **Verdict derivation** — the four 2026-08-22 scenes must bucket exactly as
   the ledger says, with the same tripping pairs and distances.
2. **Diffing** — the 08-16 and 08-22 baguette runs are the same code, same
   scene, same seed with a different outcome; the diff must say so.
3. **Guardrails** — each refusal is exercised against the real repo.
4. **The pinned stack** — the round the harness ran first died in every scene on
   a flag that does not exist, so the stack it pins is checked against the real
   ``openral deploy sim`` and against the real tracked scenes.
5. **Importing pre-harness rounds** — the two ``master-1`` fixtures are kept in
   their original ``bag1``/``seed1`` layout, so reading them at all exercises
   the importer.
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
ROUND_HARNESS_1 = FIXTURES / "2026-08-22-harness-1"

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
    """Every verdict carries the digest of the YAML that actually ran.

    For these rounds that is the per-round copy kept beside the artifacts, not a
    tracked scene: they pinned their whole stack in the scene's ``runtime:``
    block, which is exactly the surface the harness now materialises for itself.
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
    assert verdicts.scene("sink_cup").outcome == "estop-collision-real"


# ── 4. The pinned stack ───────────────────────────────────────────────────────


def _deploy_sim_option_strings() -> set[str]:
    """Every option string the real ``openral deploy sim`` command accepts.

    Built from the live Typer app, so this is the same parser click runs — not a
    list of flags anyone maintains by hand.
    """
    from openral_cli.main import app
    from typer.main import get_command

    sim = get_command(app).commands["deploy"].commands["sim"]  # type: ignore[attr-defined]  # reason: click.Group
    return {opt for param in sim.params for opt in [*param.opts, *param.secondary_opts]}


def test_every_pinned_flag_exists_on_the_real_deploy_cli() -> None:
    """The round that died in every scene in under a second, as a test.

    ``STACK_ARGV`` pinned ``--no-enable-reasoner``, which ``openral deploy sim``
    has never had: click exited 2 before the ROS graph started, four times.
    Every flag the matrix pins is now checked against the real click command's
    own parameter list, so a flag that does not exist cannot be pinned again.
    """
    known = _deploy_sim_option_strings()
    pinned = [token for token in validation_matrix.STACK_ARGV if token.startswith("--")]
    assert pinned, "the matrix pins no flags at all"
    assert set(pinned) <= known, sorted(set(pinned) - known)


def test_the_reasoner_has_no_cli_flag_so_it_is_pinned_in_the_scene() -> None:
    """The eighth knob: ``enable_reasoner`` is resolved from the scene, not argv.

    This is the fact the harness originally got wrong. If ``deploy sim`` ever
    grows a reasoner flag, this goes red and the pin can move to the argv.
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

    Every scene of ``2026-08-22-harness-1`` produced a six-line click usage
    error and nothing else. It was reported as ``deadline-no-grasp`` with exit
    0, because ``artifacts_complete`` was ``bool(deploy_lines)`` and a usage
    error is lines.
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
    assert verdicts.scene("baguette").outcome == "estop-collision-false-positive"


def test_import_reads_the_stack_out_of_the_rounds_own_log() -> None:
    """The imported stack is the resolved launch argv, not anyone's recollection.

    These rounds pinned the reasoner in their scene YAML and passed only
    ``--hal viewer_enabled=false --no-dashboard`` on the command line, so a
    metadata block claiming a ``--no-enable-reasoner`` flag would describe a
    command that never ran.
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
