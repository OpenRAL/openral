"""Run the collision-stack validation matrix and emit a machine-readable verdict.

For ~17 rounds the four-scene RoboCasa validation matrix was driven by shell and
Python that lived only in ``~/openral-runs/<date>-<name>/scripts/`` on the DGX
Spark: ``run_matrix.sh``, ``drive_round.sh``, ``attach_monitor*.py``,
``postprocess.sh``, ``adjudicate.py``, ``verdict_table.py`` and a per-scene YAML
copy each. Nothing was in the repo, so every round re-derived the same tooling
with drift, several rounds produced no written conclusion at all, and one round
silently executed the wrong checkout. The evidence ledger for all of that is
``docs/reference/collision-validation-evidence.md``; this tool is what feeds it
from now on.

Three subcommands, all reading recorded artifacts only:

* ``run`` — guardrails, then one ``openral deploy sim`` per scene with a monitor
  and an action dispatch alongside it, capturing a fixed artifact set per scene,
  then ``verdicts``. Requires a GPU host with RoboCasa installed.
* ``verdicts`` — derive ``verdicts.json`` (a
  :class:`~openral_core.ValidationRoundVerdicts`) from a round directory. Pure
  and offline: this is the half that is unit-tested against recorded artifacts.
* ``diff`` — compare two rounds' ``verdicts.json`` into a
  :class:`~openral_core.ValidationRoundDiff`, so "what changed since the last
  round" is mechanical.

The guardrails are the point as much as the runner is. ``run`` refuses to start
on a dirty worktree, on a built overlay older than the checked-out sources, when
the resolved launcher is not this checkout's, when the dependency set is missing
``pyzmq`` (the XR-1 sidecar's wire), or when the composed argv contains anything
that looks like a safety-knob override. Each of those closes a specific past
incident — see ``docs/contributing/validation-matrix.md``.

Run::

    uv run python tools/validation_matrix.py run --round-id 2026-08-22-master-1 \\
        --expect-sha $(git rev-parse HEAD)
    uv run python tools/validation_matrix.py verdicts outputs/validation-matrix/<round>
    uv run python tools/validation_matrix.py diff <round> --baseline <prev-round>

Example:
    >>> quantization_budget_m(0.025)
    0.021650635094610966
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
OUTPUT_ROOT: Final[Path] = REPO_ROOT / "outputs" / "validation-matrix"

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping, Sequence

    from openral_core import (
        ValidationGroundTruthAdjudication,
        ValidationRoundDiff,
        ValidationRoundVerdicts,
        ValidationSceneVerdict,
        ValidationStopEvidence,
    )


# ── The matrix ────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SceneSpec:
    """One scene of the validation matrix.

    ``config`` is the *tracked* DeployScene YAML, used verbatim. The stack is
    pinned by CLI flags rather than by a per-round YAML copy, because the deploy
    CLI's precedence is explicit-flag > scene ``runtime:`` > default; copying the
    scene per round is exactly how the historical configs drifted from the
    tracked ones.
    """

    key: str
    config: str
    prompt: str
    deadline_s: int = 420


MATRIX: Final[tuple[SceneSpec, ...]] = (
    SceneSpec(
        "baguette",
        "scenes/deploy/robocasa_baguette.yaml",
        "Pick the baguette from the counter and place it in the cabinet.",
    ),
    SceneSpec(
        "sink_cup",
        "scenes/deploy/robocasa_sink_cup.yaml",
        "Pick the cup from the counter and place it in the sink.",
    ),
    SceneSpec(
        "fridge",
        "scenes/deploy/robocasa_fridge_drawer.yaml",
        "Pick the vegetable from the fridge shelf and place it in the fridge drawer.",
    ),
    SceneSpec(
        "utensil",
        "scenes/deploy/robocasa_drawer_utensil.yaml",
        "Pick the utensil from the counter and place it in the drawer.",
    ),
)

DEFAULT_RSKILL_ID: Final[str] = "OpenRAL/rskill-xr1-panda_mobile-robocasa365-nf4"

# `--group robocasa` alone strips pyzmq, which the XR-1 adapter's sidecar wire
# needs; a round was lost to exactly that. Both groups, always.
SYNC_GROUPS: Final[tuple[str, ...]] = ("robocasa", "sidecar-wire")

# The stack every round has pinned since 2026-08-13: reasoner off (direct
# dispatch), SLAM/Nav2/octomap/kernel-check on, detector + scene VLM off.
STACK_ARGV: Final[tuple[str, ...]] = (
    "--no-enable-reasoner",
    "--enable-slam",
    "--enable-nav2",
    "--enable-octomap",
    "--enable-octomap-kernel-check",
    "--no-object-detector",
    "--no-enable-scene-vlm",
    "--no-dashboard",
    "--hal",
    "viewer_enabled=false",
)

# Anything matching these in the composed argv is a safety-knob override and is
# refused outright (CLAUDE.md §3 — "never add a flag that disables safety"). The
# harness must not be the place a margin quietly moves. Tokens are normalised to
# lowercase with `-` folded to `_` first, so the hyphenated CLI spelling and the
# underscored parameter spelling of the same knob are one pattern.
_SAFETY_KNOB_PATTERNS: Final[tuple[str, ...]] = (
    "safety",
    "estop",
    "e_stop",
    "collision_margin",
    "margin_m",
    "attached_contact_tolerance",
    "allowance",
    "watchdog",
    "velocity_limit",
    "force_limit",
    "deadman",
    # Turning the octomap gate off is disabling the collision check, whichever
    # spelling is used. The matrix pins it ON in `STACK_ARGV`.
    "no_enable_octomap_kernel_check",
    "kernel_check=false",
)

# Sources whose change invalidates the built ROS overlay: the C++ kernel, the
# IDL, the octomap bridge and every HAL/ROS package colcon compiles.
_OVERLAY_SOURCE_DIRS: Final[tuple[str, ...]] = ("cpp", "packages")


# ── Pure helpers (unit-tested against recorded artifacts) ─────────────────────


def quantization_budget_m(grid_resolution_m: float) -> float:
    """Largest kernel/ground-truth discrepancy a correct voxel grid can produce.

    A surface anywhere inside a cell makes the whole cell occupied, so the kernel
    can read as much as half the cell's body diagonal more penetration than the
    true geometry has. A discrepancy beyond this is not quantization.

    Args:
        grid_resolution_m: Occupancy-grid cell edge length in metres.

    Returns:
        Half the cell's body diagonal, in metres.

    Example:
        >>> round(quantization_budget_m(0.025), 6)
        0.021651
    """
    return grid_resolution_m * math.sqrt(3.0) / 2.0


def parse_kernel_collision(lines: Iterable[str]) -> ValidationStopEvidence | None:
    """Transcribe the first ``safety.collision`` line into typed evidence.

    Args:
        lines: Lines of the deploy log (or a ``kernel_evidence.txt`` excerpt).

    Returns:
        The kernel's verdict, or ``None`` when the run was never stopped.

    Example:
        >>> line = (
        ...     "[safety_kernel_node-1] [ERROR] [1.0] [openral_safety_kernel]: "
        ...     "safety.collision kind=world a=panda_link5 b=voxel_170781 step=0 "
        ...     "min_distance_m=-0.0209178 sweep_min_distance_m=-0.0209178 "
        ...     "mode=5 place_allowance_active=0 place_target="
        ... )
        >>> ev = parse_kernel_collision([line])
        >>> ev.party_a, ev.horizon_step, ev.exemption_active
        ('panda_link5', 0, False)
    """
    from openral_core import ValidationStopEvidence  # reason: deferred

    for line in lines:
        if "safety.collision" not in line:
            continue
        fields = _key_values(line.split("safety.collision", 1)[1])
        if "a" not in fields or "min_distance_m" not in fields:
            continue
        sweep = fields.get("sweep_min_distance_m")
        return ValidationStopEvidence(
            kind=fields.get("kind", ""),
            party_a=fields.get("a", ""),
            party_b=fields.get("b", ""),
            horizon_step=int(fields.get("step", "-1")),
            min_distance_m=float(fields["min_distance_m"]),
            sweep_min_distance_m=None if sweep is None else float(sweep),
            place_allowance_active=fields.get("place_allowance_active") == "1",
            place_target=fields.get("place_target", ""),
        )
    return None


def _key_values(blob: str) -> dict[str, str]:
    """Split a ``k=v k=v`` kernel log tail into a mapping.

    Trailing empty values (``place_target=``) are preserved as empty strings,
    which is how the kernel writes "no target".
    """
    out: dict[str, str] = {}
    for token in blob.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        out[key] = value
    return out


def parse_json_log_line(lines: Iterable[str], event: str) -> dict[str, Any] | None:
    """Return the JSON payload of the first ``<event> {...}`` log line.

    The sim HAL emits ``sim.task_success_final``,
    ``sim.estop_ground_truth_snapshot`` and ``sim.estop_initial_configuration``
    as a marker followed by a JSON object; ROS launch prefixes each line with a
    node tag, so the payload is taken from the first ``{`` after the marker.

    Args:
        lines: Lines of the deploy log.
        event: The marker, e.g. ``"sim.task_success_final"``.

    Returns:
        The decoded object, or ``None`` when the marker never appears with a
        parseable payload.

    Example:
        >>> parse_json_log_line(['[n-1] sim.x {"a": 1}'], "sim.x")
        {'a': 1}
    """
    for line in lines:
        idx = line.find(event)
        if idx < 0:
            continue
        brace = line.find("{", idx)
        if brace < 0:
            continue
        with contextlib.suppress(json.JSONDecodeError):
            decoded = json.loads(line[brace:])
            if isinstance(decoded, dict):
                return decoded
    return None


def read_monitor(path: Path) -> list[dict[str, Any]]:
    """Load a monitor JSONL, skipping any line that is not a JSON object.

    Args:
        path: Path to ``<run>_monitor.jsonl``.

    Returns:
        The decoded records in file order; empty when the file is absent.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            with contextlib.suppress(json.JSONDecodeError):
                decoded = json.loads(stripped)
                if isinstance(decoded, dict):
                    records.append(decoded)
    return records


def grid_resolution_from_monitor(records: Sequence[Mapping[str, Any]]) -> float | None:
    """Read the occupancy grid's cell size out of the monitor's voxel records.

    Args:
        records: Monitor JSONL records.

    Returns:
        The first reported resolution in metres, or ``None`` when the run
        published no grid (octomap off, or the graph never reached activation).
    """
    for record in records:
        if record.get("event") == "world_voxels" and record.get("resolution") is not None:
            return float(record["resolution"])
    return None


def build_witness_timeline(
    records: Sequence[Mapping[str, Any]], deploy_lines: Sequence[str]
) -> Any:
    """Reconstruct the attach / witness / declaration lifecycle of one run.

    Producer side comes from the monitor JSONL, consumer side from the kernel's
    own log lines; both are recorded because a disagreement between them is a
    finding in its own right.

    Args:
        records: Monitor JSONL records.
        deploy_lines: Lines of the deploy log.

    Returns:
        A :class:`~openral_core.ValidationWitnessTimeline`.
    """
    from openral_core import ValidationWitnessTimeline  # reason: deferred

    attach_t: float | None = None
    detach_t: float | None = None
    support_id: str | None = None
    declaration_seen = False
    seen_ids: set[str] = set()

    for record in records:
        event = record.get("event")
        if event == "attachment_state":
            objects = record.get("objects") or []
            current = {str(obj.get("id")) for obj in objects}
            if current and attach_t is None:
                attach_t = _as_float(record.get("t"))
            if seen_ids and not current and detach_t is None:
                detach_t = _as_float(record.get("t"))
            seen_ids = current
            for obj in objects:
                witness = obj.get("support_contact")
                if isinstance(witness, dict) and witness.get("support_id"):
                    support_id = str(witness["support_id"])
        elif event == "place_declaration":
            declaration = record.get("declaration")
            if isinstance(declaration, dict) and declaration.get("region"):
                declaration_seen = True

    armed = sum(1 for ln in deploy_lines if "safety.support_witness_armed" in ln)
    separated = sum(1 for ln in deploy_lines if "safety.support_witness_separated" in ln)
    region_armed = any("safety.place_region_armed" in ln for ln in deploy_lines)
    allowance_lines = sum(1 for ln in deploy_lines if "place_allowance_active=1" in ln)

    return ValidationWitnessTimeline(
        attach_t_s=attach_t,
        detach_t_s=detach_t,
        support_id=support_id,
        kernel_witness_armed=armed,
        kernel_witness_separated=separated,
        place_declaration_seen=declaration_seen,
        place_region_armed=region_armed,
        place_allowance_active_lines=allowance_lines,
    )


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


_MUJOCO_LINK_PREFIX: Final[str] = "robot0_"


def _kernel_party_to_mujoco(party: str) -> str:
    """Map a kernel link name onto the body name the simulator probe reports.

    The kernel names Panda links ``panda_link<N>`` (from the URDF); RoboCasa's
    MuJoCo model calls the same bodies ``robot0_link<N>``.
    """
    if party.startswith("panda_"):
        return _MUJOCO_LINK_PREFIX + party[len("panda_") :]
    return party


def adjudicate_ground_truth(
    snapshot: Mapping[str, Any] | None,
    stop: ValidationStopEvidence | None,
    grid_resolution_m: float | None,
) -> ValidationGroundTruthAdjudication | None:
    """Adjudicate a kernel stop against the simulator's own distance probe.

    The rule, in order:

    1. Any probed pair at or below 0 m → ``real-contact``. The kernel was right
       that the configuration is unsafe even when it named a different body:
       the probe's own caveat is that ``contype``/``conaffinity`` exclusions
       suppress MuJoCo *contacts* at real interpenetration, so distance — not
       the contact list — is the test.
    2. Otherwise, compare the clearance of the party the kernel named against
       the kernel's reported depth. A discrepancy beyond
       :func:`quantization_budget_m` cannot be explained by the voxel grid →
       ``false-positive``. Within it → ``within-quantization``, which is
       correct conservative behaviour.
    3. A truncated probe, a missing snapshot or an unknown grid resolution →
       ``unadjudicated``. An *untruncated* probe that returned no pair is not
       missing data: it proves the nearest geometry is beyond ``distmax_m``,
       which is used as a strict lower bound.

    Args:
        snapshot: The ``sim.estop_ground_truth_snapshot`` payload, if recorded.
        stop: The kernel's verdict, if the run was stopped.
        grid_resolution_m: Occupancy-grid cell size, from the monitor.

    Returns:
        The adjudication, or ``None`` when there was no stop to adjudicate.
    """
    from openral_core import (  # reason: deferred
        ValidationGroundTruthAdjudication,
    )

    if stop is None:
        return None
    if snapshot is None:
        return ValidationGroundTruthAdjudication(verdict="unadjudicated")

    coverage = snapshot.get("nearest_probe_coverage") or {}
    truncated = bool(coverage.get("truncated", True))
    distmax = _as_float(coverage.get("distmax_m"))
    probed = coverage.get("probed_pairs")

    robot_pairs = list(snapshot.get("nearest_robot_world_pairs") or [])
    payload_world = list(snapshot.get("nearest_payload_world_pairs") or [])
    payload_robot = list(snapshot.get("nearest_payload_robot_pairs") or [])
    all_pairs = robot_pairs + payload_world + payload_robot

    nearest_any = min((p["distance_m"] for p in all_pairs), default=None)
    nearest_pair = min(all_pairs, key=lambda p: p["distance_m"], default=None)

    if stop.involves_payload:
        party_pairs = payload_world
    else:
        subject = _kernel_party_to_mujoco(stop.party_a)
        party_pairs = [p for p in robot_pairs if str(p.get("body_a")) == subject]
    nearest_party = min((p["distance_m"] for p in party_pairs), default=None)

    budget = None if grid_resolution_m is None else quantization_budget_m(grid_resolution_m)
    lower_bound = nearest_party
    if lower_bound is None and not truncated and distmax is not None and probed:
        # No pair within distmax on an untruncated probe: the nearest solid
        # geometry is provably further away than distmax.
        lower_bound = distmax
    discrepancy = None if lower_bound is None else lower_bound - stop.min_distance_m

    verdict: str
    if nearest_any is not None and nearest_any <= 0.0:
        verdict = "real-contact"
    elif truncated or budget is None or discrepancy is None:
        verdict = "unadjudicated"
    elif discrepancy > budget:
        verdict = "false-positive"
    else:
        verdict = "within-quantization"

    return ValidationGroundTruthAdjudication(
        verdict=verdict,
        stop_class=snapshot.get("stop_class"),
        sim_time_s=_as_float(snapshot.get("sim_time_s")),
        grid_resolution_m=grid_resolution_m,
        quantization_budget_m=budget,
        nearest_any_m=nearest_any,
        nearest_tripping_party_m=nearest_party,
        nearest_pair=dict(nearest_pair) if nearest_pair else {},
        discrepancy_m=discrepancy,
        probed_pairs=None if probed is None else int(probed),
        probe_truncated=truncated,
        distmax_m=distmax,
        payload_contacts=len(snapshot.get("payload_contacts") or []),
    )


def classify_outcome(
    *,
    task_success_ever: bool | None,
    stop: ValidationStopEvidence | None,
    ground_truth: ValidationGroundTruthAdjudication | None,
    initial_configuration: bool,
    grasped: bool,
    artifacts_complete: bool,
) -> str:
    """Bucket one scene into exactly one :data:`~openral_core.ValidationOutcome`.

    Ordering is load-bearing. Task success wins outright. An initial-configuration
    stop is classified as such *before* the ground-truth adjudication runs,
    because a stop the scene reset produced says nothing about the collision
    stack — it is a scene-config defect and no margin change can clear it.

    Args:
        task_success_ever: ``ever_succeeded`` from ``sim.task_success_final``.
        stop: The kernel's verdict, if any.
        ground_truth: The adjudication of that verdict, if any.
        initial_configuration: Whether ``sim.estop_initial_configuration`` fired.
        grasped: Whether the payload was ever attached.
        artifacts_complete: Whether the run produced a usable artifact set.

    Returns:
        The outcome value.

    Example:
        >>> classify_outcome(
        ...     task_success_ever=None,
        ...     stop=None,
        ...     ground_truth=None,
        ...     initial_configuration=False,
        ...     grasped=False,
        ...     artifacts_complete=False,
        ... )
        'harness-error'
    """
    if task_success_ever is True:
        return "completed"
    if not artifacts_complete:
        return "harness-error"
    if stop is None:
        return "deadline-after-grasp" if grasped else "deadline-no-grasp"
    if initial_configuration:
        return "estop-initial-configuration"
    verdict = None if ground_truth is None else ground_truth.verdict
    return {
        "real-contact": "estop-collision-real",
        "false-positive": "estop-collision-false-positive",
        "within-quantization": "estop-collision-within-quantization",
    }.get(verdict or "", "estop-collision-unadjudicated")


def scene_verdict_from_artifacts(
    run_dir: Path,
    *,
    scene: str,
    config_path: str,
    seed: int,
    prompt: str = "",
    rskill_id: str = "",
    stem: str = "run",
) -> ValidationSceneVerdict:
    """Derive one scene's verdict from its recorded artifacts.

    Reads ``<stem>_deploy.log`` (or ``<stem>_deploy_excerpt.log``),
    ``<stem>_monitor.jsonl`` and ``<stem>_goal.log``. Nothing is recomputed from
    a live system and nothing is inferred that the artifacts do not state.

    Args:
        run_dir: The scene's directory inside the round.
        scene: Scene key.
        config_path: Repo-relative path of the DeployScene YAML that ran.
        seed: The seed in force.
        prompt: Instruction dispatched to the policy.
        rskill_id: rSkill the run dispatched.
        stem: Artifact filename stem.

    Returns:
        A :class:`~openral_core.ValidationSceneVerdict`.
    """
    from openral_core import ValidationSceneVerdict  # reason: deferred

    deploy_path = run_dir / f"{stem}_deploy.log"
    if not deploy_path.exists():
        deploy_path = run_dir / f"{stem}_deploy_excerpt.log"
    monitor_path = run_dir / f"{stem}_monitor.jsonl"
    goal_path = run_dir / f"{stem}_goal.log"

    deploy_lines = (
        deploy_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if deploy_path.exists()
        else []
    )
    records = read_monitor(monitor_path)
    goal: dict[str, Any] | None = None
    if goal_path.exists():
        for line in goal_path.read_text(encoding="utf-8", errors="replace").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                decoded = json.loads(line.strip() or "{}")
                if isinstance(decoded, dict) and "status" in decoded:
                    goal = decoded

    stop = parse_kernel_collision(deploy_lines)
    success = parse_json_log_line(deploy_lines, "sim.task_success_final") or {}
    snapshot = parse_json_log_line(deploy_lines, "sim.estop_ground_truth_snapshot")
    initial = parse_json_log_line(deploy_lines, "sim.estop_initial_configuration")
    resolution = grid_resolution_from_monitor(records)
    ground_truth = adjudicate_ground_truth(snapshot, stop, resolution)
    witness = build_witness_timeline(records, deploy_lines)

    outcome = classify_outcome(
        task_success_ever=_as_bool(success.get("ever_succeeded")),
        stop=stop,
        ground_truth=ground_truth,
        initial_configuration=initial is not None,
        grasped=witness.attach_t_s is not None,
        artifacts_complete=bool(deploy_lines),
    )

    artifacts = {
        name: str(path.relative_to(run_dir.parent))
        for name, path in (
            ("deploy_log", deploy_path),
            ("monitor", monitor_path),
            ("goal", goal_path),
        )
        if path.exists()
    }

    return ValidationSceneVerdict(
        scene=scene,
        config_path=config_path,
        config_sha256=_sha256_of(REPO_ROOT / config_path),
        seed=seed,
        prompt=prompt,
        rskill_id=rskill_id,
        outcome=outcome,
        task_success_final=_as_bool(success.get("success")),
        task_success_ever=_as_bool(success.get("ever_succeeded")),
        task_success_steps=_as_int(success.get("steps")),
        task_success_transitions=_as_int(success.get("transitions")),
        stop=stop,
        ground_truth=ground_truth,
        witness=witness,
        dispatch_failure_reason=str((goal or {}).get("failure_reason", "")),
        wall_s=_as_float((goal or {}).get("wall_s")),
        artifacts=artifacts,
    )


def _as_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _sha256_of(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Diffing ───────────────────────────────────────────────────────────────────

_DIFF_FIELDS: Final[tuple[str, ...]] = (
    "outcome",
    "task_success_final",
    "task_success_steps",
    "stop.party_a",
    "stop.party_b",
    "stop.horizon_step",
    "stop.min_distance_m",
    "stop.sweep_min_distance_m",
    "stop.place_allowance_active",
    "ground_truth.verdict",
    "ground_truth.nearest_tripping_party_m",
    "witness.kernel_witness_armed",
    "witness.kernel_witness_separated",
    "witness.place_allowance_active_lines",
)


def _dotted(model: object, path: str) -> object:
    current: object = model
    for part in path.split("."):
        if current is None:
            return None
        current = getattr(current, part, None)
    return current


def diff_rounds(
    current: ValidationRoundVerdicts, baseline: ValidationRoundVerdicts
) -> ValidationRoundDiff:
    """Compare two rounds field by field.

    When both rounds carry the same ``executed_sha`` the result is a pure
    reproducibility comparison; when they differ it is a before/after. The
    distinction matters because the policy is stochastic across runs even at a
    pinned scene seed, so only failure *classes* compare between rounds.

    Args:
        current: The newer round.
        baseline: The round to compare against.

    Returns:
        A :class:`~openral_core.ValidationRoundDiff`.
    """
    from openral_core import (  # reason: deferred
        ValidationRoundDiff,
        ValidationSceneDelta,
    )

    deltas: list[ValidationSceneDelta] = []
    keys = [s.scene for s in current.scenes]
    keys += [s.scene for s in baseline.scenes if s.scene not in keys]
    for key in keys:
        now = current.scene(key)
        was = baseline.scene(key)
        changed_fields: dict[str, dict[str, object]] = {}
        if now is not None and was is not None:
            for field in _DIFF_FIELDS:
                new_value = _dotted(now, field)
                old_value = _dotted(was, field)
                if new_value != old_value:
                    changed_fields[field] = {"from": old_value, "to": new_value}
        deltas.append(
            ValidationSceneDelta(
                scene=key,
                baseline_outcome=None if was is None else was.outcome,
                outcome=None if now is None else now.outcome,
                changed=(now is None or was is None or now.outcome != was.outcome),
                changed_fields=changed_fields,
            )
        )

    return ValidationRoundDiff(
        round_id=current.metadata.round_id,
        baseline_round_id=baseline.metadata.round_id,
        executed_sha=current.metadata.executed_sha,
        baseline_executed_sha=baseline.metadata.executed_sha,
        scenes=deltas,
    )


# ── Guardrails ────────────────────────────────────────────────────────────────


class GuardrailError(RuntimeError):
    """A precondition for running the matrix is not met.

    Raised rather than warned: every one of these has already cost a round.
    """


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def assert_worktree_clean() -> None:
    """Refuse to run from a dirty worktree.

    A round executed from uncommitted changes cannot be reproduced by anyone
    else, and its SHA is a lie.
    """
    if _git("status", "--porcelain"):
        raise GuardrailError(
            "worktree is dirty — commit or stash before running a round, "
            "otherwise the recorded executed_sha does not describe what ran."
        )


def assert_sha(expected: str | None) -> str:
    """Return the executed SHA, refusing when it is not the expected one.

    Args:
        expected: SHA (or prefix) the operator intended to run.

    Returns:
        The full SHA of ``HEAD``.

    Raises:
        GuardrailError: When ``HEAD`` does not start with ``expected``.
    """
    head = _git("rev-parse", "HEAD")
    if expected and not head.startswith(expected):
        raise GuardrailError(
            f"checked-out HEAD {head} is not the requested {expected} — "
            "a round has already been lost to running the wrong checkout."
        )
    return head


def assert_overlay_fresh(install_dir: Path) -> int:
    """Refuse to run against a ROS overlay older than the sources it was built from.

    The kernel, the IDL and the octomap bridge are compiled; a round launched
    against a stale ``install/`` silently validates the *previous* commit's C++.

    Args:
        install_dir: The colcon ``install/`` directory.

    Returns:
        Modification time of the overlay, in nanoseconds.

    Raises:
        GuardrailError: When the overlay is missing or older than a tracked
            source under ``cpp/`` or ``packages/``.
    """
    if not install_dir.exists():
        raise GuardrailError(
            f"no built overlay at {install_dir} — run `just ros2-build` "
            "(clean rebuild when the kernel, msgs or bridge changed)."
        )
    overlay_ns = install_dir.stat().st_mtime_ns
    newest_source = 0
    newest_path = ""
    for rel in _OVERLAY_SOURCE_DIRS:
        root = REPO_ROOT / rel
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".cpp", ".hpp", ".h", ".msg", ".idl"}:
                continue
            mtime = path.stat().st_mtime_ns
            if mtime > newest_source:
                newest_source, newest_path = mtime, str(path.relative_to(REPO_ROOT))
    if newest_source > overlay_ns:
        raise GuardrailError(
            f"built overlay at {install_dir} is older than {newest_path} — "
            "rebuild before running (rm -rf build install log && just ros2-build)."
        )
    return overlay_ns


def resolve_launcher() -> Path:
    """Resolve the ``openral`` entry point, refusing the user-wide wrapper.

    ``~/.local/bin/openral`` hardcodes a repo root and execs the *parent*
    checkout's venv and overlay, so a round launched through it executes another
    branch's code and ``robots/`` manifests. The venv's own binary is the only
    acceptable launcher, and it is invoked by absolute path.

    Returns:
        Absolute path of this checkout's ``openral``.

    Raises:
        GuardrailError: When the venv binary is missing.
    """
    candidate = REPO_ROOT / ".venv" / "bin" / "openral"
    if not candidate.exists():
        raise GuardrailError(
            f"no launcher at {candidate} — run `just sync` in this checkout. "
            "The ~/.local/bin/openral wrapper is not an acceptable substitute: "
            "it execs the parent checkout's venv and overlay."
        )
    return candidate


def assert_sidecar_wire() -> None:
    """Refuse to run when the dependency set is missing the sidecar wire.

    ``just sync --group robocasa`` alone uninstalls ``pyzmq``, which the XR-1
    adapter needs to reach its sidecar; a round was lost to exactly that. The
    correct invocation is ``--group robocasa --group sidecar-wire``.
    """
    try:
        import zmq  # noqa: F401  # reason: availability probe
    except ImportError as exc:
        raise GuardrailError(
            "pyzmq is not installed — the XR-1 adapter's sidecar wire is missing. "
            "Sync with `just sync --group robocasa --group sidecar-wire`; "
            "`--group robocasa` alone strips it."
        ) from exc


def assert_no_safety_overrides(argv: Sequence[str]) -> None:
    """Refuse any argv token that looks like a safety-knob override.

    The matrix exists to observe the safety kernel, never to move it. This is a
    hard refusal, not a warning (CLAUDE.md §1.1, §3).

    Args:
        argv: The fully composed launch argv.

    Raises:
        GuardrailError: On the first match.

    Example:
        >>> assert_no_safety_overrides(["--enable-slam"])
        >>> try:
        ...     assert_no_safety_overrides(["--hal", "collision_margin_m=0.0"])
        ... except GuardrailError as exc:
        ...     "safety-knob pattern" in str(exc)
        True
    """
    for token in argv:
        lowered = token.lower().replace("-", "_")
        for pattern in _SAFETY_KNOB_PATTERNS:
            if pattern in lowered:
                raise GuardrailError(
                    f"argv token {token!r} matches the safety-knob pattern {pattern!r}. "
                    "The validation matrix never overrides a safety parameter — "
                    "if this is a false match, rename the flag rather than relaxing "
                    "the guard."
                )


def gpu_status() -> tuple[str | None, list[str]]:
    """Report the GPU name and any processes already using it.

    The validation host is shared, so a round is announced against what is
    already resident rather than started blind.

    Returns:
        ``(gpu_name, process_descriptions)``. Both empty when there is no
        ``nvidia-smi`` on ``PATH``.
    """
    if shutil.which("nvidia-smi") is None:
        return None, []
    name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    procs = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return (name or None, [ln.strip() for ln in procs.splitlines() if ln.strip()])


# ── The runner ────────────────────────────────────────────────────────────────


def _resolve_seed_config(spec: SceneSpec, seed: int, run_dir: Path) -> tuple[str, Path]:
    """Return the config to launch, materialising a seed override only if needed.

    The tracked scene is used verbatim whenever its own ``seed:`` already matches
    — copying the scene per round is how the historical per-round YAMLs drifted.
    When a different seed is requested, only the ``seed:`` line is rewritten and
    every other byte (including the scene's hand-written safety commentary) is
    left intact, mirroring ``openral collision lower``'s splice-only contract.
    """
    tracked = REPO_ROOT / spec.config
    text = tracked.read_text(encoding="utf-8")
    match = re.search(r"^seed:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if match and int(match.group(1)) == seed:
        return spec.config, tracked
    resolved = run_dir / f"{Path(spec.config).stem}_seed{seed}.yaml"
    rewritten = (
        re.sub(r"^seed:\s*\d+\s*$", f"seed: {seed}", text, count=1, flags=re.MULTILINE)
        if match
        else text.rstrip("\n") + f"\n\nseed: {seed}\n"
    )
    resolved.write_text(rewritten, encoding="utf-8")
    return str(resolved.relative_to(REPO_ROOT)) if resolved.is_relative_to(REPO_ROOT) else str(
        resolved
    ), resolved


def _launch_env(run_dir: Path, stem: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENRAL_REPO_ROOT"] = str(REPO_ROOT)
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env.setdefault("OPENRAL_AUTO_INSTALL_DEPS", "1")
    env.setdefault("OPENRAL_ALLOW_REMOTE_CODE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cinecam = run_dir / f"{stem}_cinecam"
    cinecam.mkdir(parents=True, exist_ok=True)
    env["OPENRAL_CINECAM_DIR"] = str(cinecam)
    return env


def _wait_for_action_server(proc: subprocess.Popen[bytes], timeout_s: int) -> bool:
    """Poll ``ros2 action list`` until the rSkill action server appears."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        listing = subprocess.run(
            ["ros2", "action", "list"], capture_output=True, text=True, check=False
        )
        if "/openral/execute_rskill" in listing.stdout:
            return True
        time.sleep(5.0)
    return False


def _run_one_scene(
    spec: SceneSpec,
    *,
    seed: int,
    run_dir: Path,
    launcher: Path,
    robot_id: str | None,
    rskill_id: str,
    stem: str,
    extra_argv: Sequence[str],
) -> str:
    """Launch one scene, monitor it, dispatch the skill, and tear the graph down.

    Returns the config path that was launched (repo-relative when tracked).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path, config_file = _resolve_seed_config(spec, seed, run_dir)
    deploy_log = run_dir / f"{stem}_deploy.log"
    monitor_log = run_dir / f"{stem}_monitor.jsonl"
    goal_log = run_dir / f"{stem}_goal.log"
    snapshots = run_dir / f"{stem}_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    argv = [
        str(launcher),
        "deploy",
        "sim",
        "--config",
        str(config_file),
        *STACK_ARGV,
        *extra_argv,
    ]
    if robot_id:
        argv += ["--robot", robot_id]
    assert_no_safety_overrides(argv)

    env = _launch_env(run_dir, stem)
    sys.stderr.write(f"[matrix] {spec.key}: launching {config_path}\n")
    with deploy_log.open("wb") as sink:
        proc = subprocess.Popen(
            argv, cwd=REPO_ROOT, env=env, stdout=sink, stderr=sink, start_new_session=True
        )
        monitor: subprocess.Popen[bytes] | None = None
        try:
            if not _wait_for_action_server(proc, timeout_s=600):
                sys.stderr.write(f"[matrix] {spec.key}: action server never appeared\n")
                return config_path
            monitor = subprocess.Popen(
                [
                    str(REPO_ROOT / ".venv" / "bin" / "python"),
                    str(REPO_ROOT / "tools" / "_validation_matrix_monitor.py"),
                    str(monitor_log),
                    str(snapshots),
                ],
                cwd=REPO_ROOT,
                env=env,
            )
            time.sleep(5.0)
            with goal_log.open("wb") as goal_sink:
                subprocess.run(
                    [
                        str(REPO_ROOT / ".venv" / "bin" / "python"),
                        str(REPO_ROOT / "tools" / "_validation_matrix_dispatch.py"),
                        "--deadline-s",
                        str(spec.deadline_s),
                        "--rskill-id",
                        rskill_id,
                        "--prompt",
                        spec.prompt,
                    ],
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=goal_sink,
                    stderr=goal_sink,
                    check=False,
                    timeout=spec.deadline_s + 300,
                )
            time.sleep(5.0)
        finally:
            if monitor is not None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    monitor.send_signal(signal.SIGINT)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    monitor.wait(timeout=20)
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=10)
    return config_path


def _write_derived_artifacts(run_dir: Path, stem: str) -> None:
    """Split the deploy log into the human-readable excerpts a reviewer reads.

    Same set ``postprocess.sh`` produced on the validation host, but derived
    in-process. The verdict never depends on these files — they exist so a
    reviewer can read the evidence without opening a 1.7 MB log.
    """
    deploy = run_dir / f"{stem}_deploy.log"
    if not deploy.exists():
        return
    lines = deploy.read_text(encoding="utf-8", errors="replace").splitlines()
    for name, needle in (
        ("kernel_evidence.txt", "safety."),
        ("task_success.txt", "sim.task_success_final"),
        ("allowance_active.txt", "place_allowance_active=1"),
    ):
        (run_dir / f"{stem}_{name}").write_text(
            "\n".join(ln for ln in lines if needle in ln) + "\n", encoding="utf-8"
        )
    for name, marker in (
        ("gt_snapshot.json", "sim.estop_ground_truth_snapshot"),
        ("gt_evidence.json", "sim.estop_ground_truth_evidence"),
        ("initial_configuration.json", "sim.estop_initial_configuration"),
    ):
        payload = parse_json_log_line(lines, marker)
        if payload is not None:
            (run_dir / f"{stem}_{name}").write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )


def render_notes(verdicts: ValidationRoundVerdicts) -> str:
    """Render the round's human-readable notes.

    Several past rounds have no written summary at all; this makes the note a
    by-product of running rather than a thing someone remembers to write.

    Args:
        verdicts: The round's verdicts.

    Returns:
        Markdown.
    """
    meta = verdicts.metadata
    out = [
        f"# Round {meta.round_id} — validation matrix @ {meta.executed_sha[:7]}",
        "",
        f"- Started: {meta.started_at}  ·  host: {meta.host or 'unknown'}"
        f"  ·  GPU: {meta.gpu_name or 'n/a'}",
        f"- Executed SHA: `{meta.executed_sha}` (worktree clean: {meta.worktree_clean})",
        f"- Launcher: `{meta.launcher_path}`  ·  `OPENRAL_REPO_ROOT={meta.repo_root}`",
        f"- Robot: `{meta.robot_id or 'from scene'}`"
        f"  ·  manifest: `{meta.robot_manifest_path or 'resolved at launch'}`",
        f"- Sync groups: `{' '.join(meta.sync_groups)}`",
        f"- Stack argv: `{' '.join(meta.stack_argv)}`",
        f"- Safety-knob overrides present: **{not meta.safety_overrides_absent}**",
        "",
        "| scene | outcome | task_success | E-stop pair | min / sweep (m) | GT |",
        "|---|---|---|---|---|---|",
    ]
    for scene in verdicts.scenes:
        stop = scene.stop
        pair = (
            "—"
            if stop is None
            else f"`{stop.party_a}` vs `{stop.party_b}` (step {stop.horizon_step})"
        )
        dist = (
            "—"
            if stop is None
            else f"{stop.min_distance_m:.7g} / "
            f"{'—' if stop.sweep_min_distance_m is None else f'{stop.sweep_min_distance_m:.7g}'}"
        )
        gt = "—" if scene.ground_truth is None else scene.ground_truth.verdict
        out.append(
            f"| {scene.scene} | **{scene.outcome}** | {scene.task_success_final} "
            f"| {pair} | {dist} | {gt} |"
        )
    exempted = [s.scene for s in verdicts.scenes if s.stop and s.stop.exemption_active]
    allowance = [s.scene for s in verdicts.scenes if s.witness.place_allowance_active_lines]
    out += [
        "",
        f"- Exemption active at the trip: {', '.join(exempted) if exempted else 'none'}.",
        f"- `place_allowance_active=1` disclosed in: {', '.join(allowance) if allowance else 'none'}.",
        "",
        "Verdicts: `verdicts.json`. Diff against the previous round with "
        "`just validation-matrix-diff <this> <previous>`.",
        "",
    ]
    return "\n".join(out)


# ── Subcommands ───────────────────────────────────────────────────────────────


def _load_round(round_dir: Path) -> ValidationRoundVerdicts:
    from openral_core import ValidationRoundVerdicts  # reason: deferred

    return ValidationRoundVerdicts.from_json(str(round_dir / "verdicts.json"))


def cmd_verdicts(round_dir: Path, *, stem: str = "run") -> int:
    """Re-derive ``verdicts.json`` for an existing round directory."""
    from openral_core import ValidationRoundVerdicts  # reason: deferred

    meta_path = round_dir / "metadata.json"
    if not meta_path.exists():
        sys.stderr.write(f"no metadata.json in {round_dir}\n")
        return 2
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    scenes = []
    for spec in MATRIX:
        run_dir = round_dir / spec.key
        if not run_dir.is_dir():
            continue
        scenes.append(
            scene_verdict_from_artifacts(
                run_dir,
                scene=spec.key,
                config_path=metadata.get("scene_configs", {}).get(spec.key, spec.config),
                seed=int(metadata.get("seed", 1)),
                prompt=spec.prompt,
                rskill_id=metadata.get("rskill_id", DEFAULT_RSKILL_ID),
                stem=stem,
            )
        )
    verdicts = ValidationRoundVerdicts.model_validate(
        {"metadata": metadata, "scenes": [s.model_dump(mode="json") for s in scenes]}
    )
    (round_dir / "verdicts.json").write_text(
        verdicts.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (round_dir / "NOTES.md").write_text(render_notes(verdicts), encoding="utf-8")
    for scene in verdicts.scenes:
        print(f"{scene.scene:10s} {scene.outcome}")
    print(f"\nverdicts -> {round_dir / 'verdicts.json'}")
    print(f"notes    -> {round_dir / 'NOTES.md'}")
    return 0


def cmd_diff(round_dir: Path, baseline_dir: Path, out_path: Path | None) -> int:
    """Diff two rounds' verdicts."""
    diff = diff_rounds(_load_round(round_dir), _load_round(baseline_dir))
    payload = diff.model_dump_json(indent=2) + "\n"
    if out_path is not None:
        out_path.write_text(payload, encoding="utf-8")
    kind = "reproducibility" if diff.same_sha else "before/after"
    print(f"{diff.baseline_round_id} -> {diff.round_id}  ({kind})")
    for delta in diff.scenes:
        flag = "CHANGED" if delta.changed else "same   "
        print(f"  {flag} {delta.scene:10s} {delta.baseline_outcome} -> {delta.outcome}")
        for field, move in delta.changed_fields.items():
            if field == "outcome":
                continue
            print(f"           {field}: {move['from']} -> {move['to']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the matrix end to end."""
    round_dir = OUTPUT_ROOT / args.round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    assert_worktree_clean()
    executed_sha = assert_sha(args.expect_sha)
    overlay_ns = assert_overlay_fresh(REPO_ROOT / "install")
    launcher = resolve_launcher()
    assert_sidecar_wire()
    gpu_name, gpu_procs = gpu_status()
    if gpu_procs and not args.force_shared_gpu:
        raise GuardrailError(
            "the GPU already has compute processes resident:\n  "
            + "\n  ".join(gpu_procs)
            + "\nThis host is shared. Coordinate, or pass --force-shared-gpu "
            "if you know the residents are yours."
        )

    scenes = [s for s in MATRIX if not args.scenes or s.key in args.scenes]
    if not scenes:
        sys.stderr.write(f"no scenes matched {args.scenes!r}\n")
        return 2

    scene_configs: dict[str, str] = {}
    for spec in scenes:
        config_path = _run_one_scene(
            spec,
            seed=args.seed,
            run_dir=round_dir / spec.key,
            launcher=launcher,
            robot_id=args.robot,
            rskill_id=args.rskill_id,
            stem=args.stem,
            extra_argv=args.deploy_arg,
        )
        scene_configs[spec.key] = config_path
        _write_derived_artifacts(round_dir / spec.key, args.stem)

    metadata = {
        "round_id": args.round_id,
        "started_at": _dt.datetime.now(tz=_dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "executed_sha": executed_sha,
        "worktree_clean": True,
        "overlay_built_at_ns": overlay_ns,
        "launcher_path": str(launcher),
        "repo_root": str(REPO_ROOT),
        "robot_id": args.robot,
        "robot_manifest_path": args.robot_manifest,
        "sync_groups": list(SYNC_GROUPS),
        "stack_argv": [*STACK_ARGV, *args.deploy_arg],
        "safety_overrides_absent": True,
        "gpu_name": gpu_name,
        "notes_path": "NOTES.md",
        "seed": args.seed,
        "rskill_id": args.rskill_id,
        "scene_configs": scene_configs,
    }
    (round_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return cmd_verdicts(round_dir, stem=args.stem)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Execute the matrix on this host (GPU + RoboCasa).")
    run.add_argument("--round-id", required=True, help="Round directory name, e.g. 2026-08-22-m1.")
    run.add_argument("--expect-sha", default=None, help="Refuse to run unless HEAD matches.")
    run.add_argument("--seed", type=int, default=1, help="Scene seed (default: 1).")
    run.add_argument("--scenes", nargs="*", default=[], help="Subset of scene keys to run.")
    run.add_argument("--robot", default=None, help="Override the robot id.")
    run.add_argument("--robot-manifest", default=None, help="Record the resolved manifest path.")
    run.add_argument("--rskill-id", default=DEFAULT_RSKILL_ID, help="rSkill to dispatch.")
    run.add_argument("--stem", default="run", help="Artifact filename stem (default: run).")
    run.add_argument(
        "--deploy-arg",
        action="append",
        default=[],
        help="Extra argv for `openral deploy sim` (safety-knob overrides are refused).",
    )
    run.add_argument(
        "--force-shared-gpu",
        action="store_true",
        help="Proceed even though other compute processes hold the GPU.",
    )

    verdicts = sub.add_parser("verdicts", help="Derive verdicts.json from a round directory.")
    verdicts.add_argument("round_dir", type=Path)
    verdicts.add_argument("--stem", default="run")

    diff = sub.add_parser("diff", help="Diff two rounds' verdicts.json.")
    diff.add_argument("round_dir", type=Path)
    diff.add_argument("--baseline", type=Path, required=True)
    diff.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "verdicts":
            return cmd_verdicts(args.round_dir, stem=args.stem)
        return cmd_diff(args.round_dir, args.baseline, args.out)
    except GuardrailError as exc:
        sys.stderr.write(f"GUARDRAIL: {exc}\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
