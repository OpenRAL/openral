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

Four subcommands, all reading recorded artifacts only:

* ``run`` — guardrails, then one ``openral deploy sim`` per scene with a monitor
  and an action dispatch alongside it, capturing a fixed artifact set per scene,
  then ``verdicts``. Requires a GPU host with RoboCasa installed.
* ``verdicts`` — derive ``verdicts.json`` (a
  :class:`~openral_core.ValidationRoundVerdicts`) from a round directory. Pure
  and offline: this is the half that is unit-tested against recorded artifacts.
* ``diff`` — compare two rounds' ``verdicts.json`` into a
  :class:`~openral_core.ValidationRoundDiff`, so "what changed since the last
  round" is mechanical.
* ``import-round`` — write the metadata a pre-harness round never recorded, so
  the ~17 rounds that predate this tool become queryable by ``verdicts`` and
  ``diff`` without hand-mapping their scene directories and filename stems.

The guardrails are the point as much as the runner is. ``run`` refuses to start
on a dirty worktree, on a built overlay older than the checked-out sources, when
the resolved launcher is not this checkout's, when the dependency set is missing
``pyzmq`` (the XR-1 sidecar's wire), or when a safety knob is moved on *either*
control surface — the composed argv or the resolved scene YAML. Each of those
closes a specific past incident — see ``docs/contributing/validation-matrix.md``.

Exit codes: ``0`` clean, ``2`` usage, ``3`` a guardrail refused (nothing is
written — not even the round directory), ``4`` the round ran but at least one
scene bucketed ``harness-error``.

Run::

    uv run python tools/validation_matrix.py run --round-id 2026-08-22-master-1 \\
        --expect-sha $(git rev-parse HEAD)
    uv run python tools/validation_matrix.py verdicts outputs/validation-matrix/<round>
    uv run python tools/validation_matrix.py diff <round> --baseline <prev-round>
    uv run python tools/validation_matrix.py import-round ~/openral-runs/<round> \\
        --round-id <id> --executed-sha <sha> --stem seed1

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

    ``config`` is the *tracked* DeployScene YAML. Seven of the eight pinned
    stack knobs have a CLI flag and are pinned there, because the deploy CLI's
    precedence is explicit-flag > scene ``runtime:`` > default and copying the
    scene per round is exactly how the historical configs drifted from the
    tracked ones. The eighth — ``enable_reasoner`` — has **no flag** at all: it
    is resolved from the scene's ``runtime:`` block and defaults to ``True``
    (``openral_cli.deploy_sim.resolve_launch_invocation``). So each round
    materialises a resolved copy of the tracked scene carrying
    :data:`SCENE_RUNTIME_PIN` (and the seed, when it differs), and the tracked
    file is never touched. The first live round died in every scene in under a
    second passing a ``--no-enable-reasoner`` flag that does not exist.
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

# The flag-pinnable part of the stack every round has pinned since 2026-08-13:
# SLAM/Nav2/octomap/kernel-check on, detector + scene VLM off. The reasoner is
# NOT here — it has no CLI flag; see SCENE_RUNTIME_PIN.
STACK_ARGV: Final[tuple[str, ...]] = (
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

# The one pinned knob with no CLI flag. `openral deploy sim` resolves
# `enable_reasoner` from the scene's `runtime:` block and defaults it to True
# when nothing pins it, and the tracked scenes carry no `runtime:` block — so
# the direct-dispatch stack the matrix has run since 2026-08-13 cannot be
# expressed in argv at all. Every key here must be stack *composition*; a
# safety-relevant key would be refused by `assert_scene_safety_unmoved`.
SCENE_RUNTIME_PIN: Final[tuple[tuple[str, bool], ...]] = (("enable_reasoner", False),)

# Top-level DeployScene keys that carry safety meaning: the kernel envelope, the
# collision-pair allowlist, the HAL parameter block (margins, tolerances) and
# the ADR-0097 place declaration (which grants an exemption). The harness may
# rewrite a scene's seed and its stack composition; it may never move any of
# these, on either control surface.
_SCENE_SAFETY_KEYS: Final[tuple[str, ...]] = (
    "safety",
    "extra_allowed_collision_pairs",
    "hal",
    "place_declaration",
)

# Inside `runtime:`, exactly one key is a safety knob rather than stack
# composition: the octomap→kernel collision gate. `enable_reasoner`,
# `enable_slam`, `enable_nav2`, `enable_octomap`, the detector and the scene VLM
# compose the stack and are the whole point of pinning a scene, so they stay
# pinnable.
_SCENE_SAFETY_RUNTIME_KEYS: Final[tuple[str, ...]] = ("enable_octomap_kernel_check",)

# Sources whose change invalidates the built ROS overlay: the C++ kernel, the
# IDL, the octomap bridge and every HAL/ROS package colcon compiles.
_OVERLAY_SOURCE_DIRS: Final[tuple[str, ...]] = ("cpp", "packages")

# Scene-key → directory names the pre-harness rounds used, so `import-round`
# needs no hand-mapping for the ~17 rounds in `spark:~/openral-runs/`.
LEGACY_SCENE_DIRS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("baguette", ("baguette", "bag1", "bag", "run1")),
    ("sink_cup", ("sink_cup", "sink1", "sink", "sinkcup1")),
    ("fridge", ("fridge", "fridge1")),
    ("utensil", ("utensil", "utensil1", "drawer1")),
)


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


LAUNCH_FAILED_MARKER: Final[str] = "launch_failed.txt"

# click prints its own usage error to the same sink the deploy log captures, so
# a stack that never started still produces lines. The first live round was
# reported as `deadline-no-grasp` with exit 0 for exactly that reason.
_CLI_USAGE_PREFIX: Final[str] = "Usage: openral"
_CLI_ERROR_NEEDLES: Final[tuple[str, ...]] = ("No such option", "Error:", "Missing option")


def detect_launch_failure(run_dir: Path, stem: str, deploy_lines: Sequence[str]) -> str:
    """Say why this scene's artifacts are not a run at all, or return ``""``.

    Three ways a scene can fail to launch, all of them recorded rather than
    inferred:

    1. the runner wrote ``<stem>_launch_failed.txt`` because the rSkill action
       server never appeared;
    2. the deploy CLI rejected its own argv, so ``click`` wrote a usage error
       where the graph's output would have been;
    3. there is no deploy log at all.

    Args:
        run_dir: The scene's directory inside the round.
        stem: Artifact filename stem.
        deploy_lines: Lines of the deploy log.

    Returns:
        A human-readable reason, empty when the scene really ran.

    Example:
        >>> import pathlib
        >>> detect_launch_failure(pathlib.Path("."), "run", [])
        'no deploy log: the scene produced no output at all'
    """
    marker = run_dir / f"{stem}_{LAUNCH_FAILED_MARKER}"
    if marker.exists():
        recorded = marker.read_text(encoding="utf-8", errors="replace").strip()
        return recorded or "the rSkill action server never appeared"
    if not deploy_lines:
        return "no deploy log: the scene produced no output at all"
    first = next((ln for ln in deploy_lines if ln.strip()), "")
    if first.startswith(_CLI_USAGE_PREFIX):
        detail = next(
            (
                ln.strip(" │|")
                for ln in deploy_lines
                if any(needle in ln for needle in _CLI_ERROR_NEEDLES)
            ),
            first.strip(),
        )
        return f"the deploy CLI rejected its own argv before the graph started: {detail}"
    return ""


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

    harness_error_reason = detect_launch_failure(run_dir, stem, deploy_lines)
    outcome = classify_outcome(
        task_success_ever=_as_bool(success.get("ever_succeeded")),
        stop=stop,
        ground_truth=ground_truth,
        initial_configuration=initial is not None,
        grasped=witness.attach_t_s is not None,
        artifacts_complete=not harness_error_reason,
    )

    artifacts = {
        name: str(path.relative_to(run_dir.parent))
        for name, path in (
            ("deploy_log", deploy_path),
            ("monitor", monitor_path),
            ("goal", goal_path),
            ("launch_failed", run_dir / f"{stem}_{LAUNCH_FAILED_MARKER}"),
        )
        if path.exists()
    }

    return ValidationSceneVerdict(
        scene=scene,
        config_path=config_path,
        config_sha256=_sha256_of(_resolve_config_file(run_dir.parent, config_path)),
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
        harness_error_reason=harness_error_reason,
    )


def _as_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _sha256_of(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_config_file(round_dir: Path, config_path: str) -> Path | None:
    """Locate the scene YAML a verdict names, in the round or in the tree.

    A round's resolved copy lives beside its artifacts (imported pre-harness
    rounds keep theirs there too); a tracked scene lives in the checkout. The
    digest is taken over whichever one is actually present, and is ``None`` when
    neither is — never over a guess.
    """
    if not config_path:
        return None
    for candidate in (round_dir / config_path, REPO_ROOT / config_path, Path(config_path)):
        if candidate.is_file():
            return candidate
    return None


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


def _flatten_scene(value: object, prefix: str = "") -> Iterable[tuple[str, object]]:
    """Yield ``(dotted_key, leaf)`` for a parsed YAML document.

    Sequences are leaves: ``extra_allowed_collision_pairs`` means nothing
    element by element.
    """
    if isinstance(value, dict):
        for key, sub in value.items():
            yield from _flatten_scene(sub, f"{prefix}{key}.")
    else:
        yield prefix.rstrip("."), value


def _is_scene_safety_key(dotted: str) -> bool:
    """Whether a scene key carries safety meaning rather than stack composition."""
    parts = dotted.split(".")
    if parts[0] in _SCENE_SAFETY_KEYS:
        return True
    if parts[0] == "runtime" and len(parts) > 1 and parts[1] in _SCENE_SAFETY_RUNTIME_KEYS:
        return True
    leaf = parts[-1].lower().replace("-", "_")
    return any(pattern in leaf for pattern in _SAFETY_KNOB_PATTERNS)


def scene_safety_surface(document: Mapping[str, Any]) -> dict[str, object]:
    """The safety-relevant keys of a parsed DeployScene, flattened.

    The kernel envelope, the collision-pair allowlist, the HAL parameter block
    and the place declaration are safety-relevant wholesale; inside
    ``runtime:`` only the octomap→kernel gate is, because SLAM/Nav2/octomap/
    detector/scene-VLM/reasoner enablement is stack *composition* — pinning it
    is what the harness is for. Anything else whose leaf name looks like a
    margin, tolerance, allowance, limit, watchdog or E-stop is caught by name.

    Args:
        document: A parsed DeployScene YAML.

    Returns:
        ``{dotted_key: value}`` for every safety-relevant key.

    Example:
        >>> scene_safety_surface({"runtime": {"enable_slam": True}})
        {}
        >>> scene_safety_surface({"runtime": {"enable_octomap_kernel_check": False}})
        {'runtime.enable_octomap_kernel_check': False}
    """
    return {
        key: value for key, value in _flatten_scene(document) if key and _is_scene_safety_key(key)
    }


def assert_scene_safety_unmoved(tracked: Path, resolved: Path) -> None:
    """Refuse a materialised scene that moves a safety knob the tracked one sets.

    ``assert_no_safety_overrides`` guards the argv; this guards the *other*
    control surface. The matrix materialises a resolved copy of each scene to
    pin ``runtime.enable_reasoner`` (which has no CLI flag), and a copy is
    exactly where a margin could be moved invisibly. Stack composition may
    differ between the two files; a safety key may not.

    Args:
        tracked: The tracked DeployScene YAML.
        resolved: The round's materialised copy.

    Raises:
        GuardrailError: When any safety-relevant key was added, removed or
            changed in the copy.
    """
    import yaml  # reason: deferred, tools/ is not a package

    before = scene_safety_surface(yaml.safe_load(tracked.read_text(encoding="utf-8")) or {})
    after = scene_safety_surface(yaml.safe_load(resolved.read_text(encoding="utf-8")) or {})
    moved = sorted(
        key
        for key in set(before) | set(after)
        if key not in before or key not in after or before[key] != after[key]
    )
    if moved:
        detail = ", ".join(
            f"{key}: {before.get(key, '<absent>')!r} -> {after.get(key, '<absent>')!r}"
            for key in moved
        )
        raise GuardrailError(
            f"the resolved scene {resolved.name} moves a safety key of {tracked.name} "
            f"({detail}). The validation matrix pins stack composition, never a "
            "safety parameter — on the argv or in the scene."
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


def pin_runtime_block(text: str, pins: Sequence[tuple[str, bool]]) -> str:
    """Splice ``runtime:`` pins into a scene YAML, changing nothing else.

    Splice-only, like ``openral collision lower``: every other byte — including
    each scene's hand-written safety commentary — survives verbatim. An existing
    ``runtime:`` block is edited in place (key rewritten if present, appended to
    the block if not); a scene without one gets the block appended.

    Args:
        text: The tracked scene YAML.
        pins: ``(key, value)`` runtime pins to enforce.

    Returns:
        The rewritten YAML.

    Example:
        >>> print(pin_runtime_block("seed: 1\\n", [("enable_reasoner", False)]).strip())
        seed: 1
        <BLANKLINE>
        # Pinned by the validation matrix: `enable_reasoner` has no CLI flag —
        # `openral deploy sim` reads it from the scene and defaults it to true.
        runtime:
          enable_reasoner: false
    """
    lines = text.splitlines()
    header = next((i for i, ln in enumerate(lines) if re.match(r"^runtime:\s*$", ln)), None)
    if header is None:
        block = "\n".join(f"  {key}: {str(value).lower()}" for key, value in pins)
        return (
            text.rstrip("\n")
            + "\n\n# Pinned by the validation matrix: `enable_reasoner` has no CLI flag —\n"
            "# `openral deploy sim` reads it from the scene and defaults it to true.\n"
            "runtime:\n" + block + "\n"
        )
    end = header + 1
    while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "\t", "#"))):
        end += 1
    for key, value in pins:
        rendered = f"  {key}: {str(value).lower()}"
        existing = next(
            (i for i in range(header + 1, end) if re.match(rf"^\s+{re.escape(key)}\s*:", lines[i])),
            None,
        )
        if existing is None:
            lines.insert(header + 1, rendered)
            end += 1
        else:
            lines[existing] = rendered
    return "\n".join(lines) + "\n"


def materialise_scene(spec: SceneSpec, seed: int, run_dir: Path) -> tuple[str, Path]:
    """Write the round's resolved scene copy and return ``(path_for_metadata, file)``.

    The tracked scene is never touched. The copy carries exactly two round-level
    pins: the seed, and :data:`SCENE_RUNTIME_PIN` — the reasoner, which is the
    one stack knob with no CLI flag. The result is re-parsed to prove the splice
    landed, and checked against the tracked scene so no safety key moved.

    Args:
        spec: The scene being run.
        seed: The seed to pin.
        run_dir: The scene's directory inside the round.

    Returns:
        ``(config_path, config_file)`` — the path recorded in the round metadata
        and the file to launch.

    Raises:
        GuardrailError: When the pinned stack did not survive the splice, or the
            copy moves a safety key.
    """
    import yaml  # reason: deferred, tools/ is not a package

    tracked = REPO_ROOT / spec.config
    text = tracked.read_text(encoding="utf-8")
    match = re.search(r"^seed:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if match is None:
        text = text.rstrip("\n") + f"\n\nseed: {seed}\n"
    elif int(match.group(1)) != seed:
        text = re.sub(r"^seed:\s*\d+\s*$", f"seed: {seed}", text, count=1, flags=re.MULTILINE)
    text = pin_runtime_block(text, SCENE_RUNTIME_PIN)

    resolved = run_dir / f"{Path(spec.config).stem}_seed{seed}.yaml"
    resolved.write_text(text, encoding="utf-8")

    document = yaml.safe_load(text) or {}
    runtime = document.get("runtime") or {}
    unpinned = {
        key: runtime.get(key) for key, value in SCENE_RUNTIME_PIN if runtime.get(key) is not value
    }
    if unpinned or int(document.get("seed", -1)) != seed:
        raise GuardrailError(
            f"the resolved scene {resolved} does not carry the pinned stack "
            f"(seed={document.get('seed')!r}, unpinned={unpinned!r}). Refusing to "
            "run a round whose stack is not the one it records."
        )
    assert_scene_safety_unmoved(tracked, resolved)

    relative = resolved.relative_to(REPO_ROOT) if resolved.is_relative_to(REPO_ROOT) else resolved
    return str(relative), resolved


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
    config_path, config_file = materialise_scene(spec, seed, run_dir)
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
            # The monitor attaches BEFORE the wait for the action server, not
            # five seconds before dispatch: the sim clock starts with the graph,
            # and a scene whose initial configuration trips the kernel at sim
            # t≈4.7 s stops before the rSkill server has ever advertised. The
            # 2026-08-22 utensil scene did exactly that and recorded zero
            # snapshots, a null grid resolution and an `unadjudicated` verdict.
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
            if not _wait_for_action_server(proc, timeout_s=600):
                sys.stderr.write(f"[matrix] {spec.key}: action server never appeared\n")
                (run_dir / f"{stem}_{LAUNCH_FAILED_MARKER}").write_text(
                    "/openral/execute_rskill never appeared, so no skill was ever "
                    "dispatched: the graph did not come up. This is a harness-error, "
                    "not a deadline.\n",
                    encoding="utf-8",
                )
                return config_path
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
        f"- Scene-pinned stack (no CLI flag exists): "
        f"`{' '.join(f'{k}={str(v).lower()}' for k, v in meta.scene_pins.items()) or 'none'}`",
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
    # A stop earlier than the monitor's first `world_voxels` record leaves the
    # adjudication with no grid resolution and therefore no quantization budget.
    # Say so, rather than letting `unadjudicated` read as "nothing to see".
    blind = [
        s.scene
        for s in verdicts.scenes
        if s.ground_truth is not None and s.ground_truth.grid_resolution_m is None
    ]
    broken = [s for s in verdicts.scenes if s.outcome == "harness-error"]
    out += [
        "",
        f"- Exemption active at the trip: {', '.join(exempted) if exempted else 'none'}.",
        f"- `place_allowance_active=1` disclosed in: {', '.join(allowance) if allowance else 'none'}.",
        f"- Stopped before the monitor saw a voxel grid (no quantization budget, so"
        f" `unadjudicated` states only that): {', '.join(blind) if blind else 'none'}.",
    ]
    if broken:
        out += [
            "",
            "**Harness errors — these scenes did not run:**",
            *(f"- `{s.scene}`: {s.harness_error_reason}" for s in broken),
        ]
    out += [
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


def round_exit_code(verdicts: ValidationRoundVerdicts) -> int:
    """``4`` when any scene bucketed ``harness-error``, else ``0``.

    A round in which a scene never launched is not a result, and must not exit
    successfully: the first live round reported a total launch failure in all
    four scenes as ``deadline-no-grasp`` with exit 0.

    Args:
        verdicts: The round's verdicts.

    Returns:
        The process exit code for a completed round.
    """
    return 4 if any(scene.outcome == "harness-error" for scene in verdicts.scenes) else 0


def cmd_verdicts(round_dir: Path, *, stem: str | None = None) -> int:
    """Re-derive ``verdicts.json`` for an existing round directory.

    Args:
        round_dir: The round.
        stem: Artifact filename stem; ``None`` reads ``artifact_stem`` from the
            round's metadata (``"run"`` when it records none), so an imported
            pre-harness round re-derives without repeating ``--stem``.

    Returns:
        ``0``, or ``2`` when the round has no ``metadata.json``.
    """
    from openral_core import ValidationRoundVerdicts  # reason: deferred

    meta_path = round_dir / "metadata.json"
    if not meta_path.exists():
        sys.stderr.write(f"no metadata.json in {round_dir}\n")
        return 2
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    resolved_stem = stem if stem is not None else str(metadata.get("artifact_stem") or "run")
    scene_dirs = dict(metadata.get("scene_dirs") or {})
    scenes = []
    for spec in MATRIX:
        run_dir = round_dir / scene_dirs.get(spec.key, spec.key)
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
                stem=resolved_stem,
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


def parse_launch_argv(lines: Iterable[str]) -> list[str]:
    """The resolved ``ros2 launch`` argv the deploy CLI echoed into its log.

    This is the only artifact that states the stack a run *actually* got, after
    the CLI resolved flag > scene ``runtime:`` > default. Reading it is how a
    pre-harness round's stack is recovered without trusting anyone's memory.

    Args:
        lines: Lines of the deploy log.

    Returns:
        The argv tokens, empty when the log carries no ``argv:`` line.

    Example:
        >>> parse_launch_argv(["  argv: ros2 launch pkg x.py enable_reasoner:=false"])
        ['ros2', 'launch', 'pkg', 'x.py', 'enable_reasoner:=false']
    """
    import shlex  # reason: deferred, used only by `import-round`

    for line in lines:
        idx = line.find("argv: ros2 launch")
        if idx < 0:
            continue
        with contextlib.suppress(ValueError):
            return shlex.split(line[idx + len("argv: ") :])
    return []


_STACK_TOKEN_PREFIXES: Final[tuple[str, ...]] = (
    "enable_",
    "hal_mode:=",
    "clock_origin:=",
    "spatial_memory_ingest:=",
)


def stack_tokens(argv: Sequence[str]) -> list[str]:
    """The stack-defining ``key:=value`` tokens of a resolved launch argv.

    Per-scene tokens (robot YAML, HAL params tempfile, the place declaration)
    are dropped: what is left is the stack the whole round shared.

    Args:
        argv: A resolved ``ros2 launch`` argv, from :func:`parse_launch_argv`.

    Returns:
        The stack tokens, sorted.

    Example:
        >>> stack_tokens(["ros2", "launch", "enable_slam:=true", "robot_yaml:=/r.yaml"])
        ['enable_slam:=true']
    """
    return sorted(t for t in argv if t.startswith(_STACK_TOKEN_PREFIXES) and ":=" in t)


def robot_facts_from_launch_argv(argv: Sequence[str]) -> dict[str, str]:
    """Repo root, robot id and manifest path, as the resolved launch argv states them.

    ``robot_yaml:=<repo_root>/robots/<robot_id>/robot.yaml`` is the one token
    that names the checkout a round ran from — the question a lost round was
    lost to. Derived, never assumed: an unrecognised shape yields ``{}``.

    Args:
        argv: A resolved ``ros2 launch`` argv, from :func:`parse_launch_argv`.

    Returns:
        ``{"repo_root": ..., "robot_id": ..., "robot_manifest_path": ...}``, or
        ``{}`` when the argv carries no recognisable ``robot_yaml:=``.

    Example:
        >>> robot_facts_from_launch_argv(["robot_yaml:=/w/openral/robots/panda_mobile/robot.yaml"])
        {'repo_root': '/w/openral', 'robot_id': 'panda_mobile', 'robot_manifest_path': 'robots/panda_mobile/robot.yaml'}
    """
    value = next((t.split(":=", 1)[1] for t in argv if t.startswith("robot_yaml:=")), "")
    manifest = Path(value)
    if not value or manifest.name != "robot.yaml" or manifest.parent.parent.name != "robots":
        return {}
    return {
        "repo_root": str(manifest.parent.parent.parent),
        "robot_id": manifest.parent.name,
        "robot_manifest_path": f"robots/{manifest.parent.name}/robot.yaml",
    }


def parse_log_start_time(lines: Iterable[str]) -> str | None:
    """UTC timestamp of the first ROS log stamp in a deploy log.

    Pre-harness rounds recorded no start time; their logs did, in the
    ``[1787422850.534169223]`` stamp ROS puts on every line.

    Args:
        lines: Lines of the deploy log.

    Returns:
        An ISO-8601 UTC timestamp, or ``None`` when no stamp is present.

    Example:
        >>> parse_log_start_time(["[node-1] [INFO] [1787422850.534169223] [x]: up"])
        '2026-08-22T18:20:50.534169+00:00'
    """
    for line in lines:
        match = re.search(r"\[(1[0-9]{9})\.([0-9]{9})\]", line)
        if match:
            seconds = int(match.group(1)) + int(match.group(2)) / 1e9
            return _dt.datetime.fromtimestamp(seconds, tz=_dt.UTC).isoformat()
    return None


def resolve_scene_dirs(round_dir: Path, aliases: Mapping[str, str]) -> dict[str, str]:
    """Map each matrix scene onto the directory a round actually kept it in.

    The pre-harness rounds named their scene directories ``bag1``, ``sink1``,
    ``fridge1``, ``utensil1``; diffing one against a harness round previously
    meant hand-mapping those by eye. Explicit ``--scene-alias`` wins over the
    recorded historical names.

    Args:
        round_dir: The round directory.
        aliases: Explicit ``{scene_key: directory}`` overrides.

    Returns:
        ``{scene_key: directory_name}`` for every scene present.

    Example:
        >>> resolve_scene_dirs(Path("/nonexistent"), {"fridge": "fridge1"})
        {'fridge': 'fridge1'}
    """
    found: dict[str, str] = {}
    for key, candidates in LEGACY_SCENE_DIRS:
        if key in aliases:
            found[key] = aliases[key]
            continue
        match = next((name for name in candidates if (round_dir / name).is_dir()), None)
        if match is not None:
            found[key] = match
    return found


def cmd_import(args: argparse.Namespace) -> int:
    """Write the metadata a pre-harness round never recorded, then derive it.

    Offline and read-only apart from the three files it writes into the round
    (``metadata.json``, ``verdicts.json``, ``NOTES.md``). Everything derivable
    from the artifacts is derived — the stack from the deploy log's own resolved
    launch argv, the start time from its first ROS stamp, the scene YAML from
    the round directory. Everything else (the SHA that ran, the host) is a
    recorded fact the operator passes in and is stored as given, never guessed.
    """
    round_dir: Path = args.source
    if not round_dir.is_dir():
        sys.stderr.write(f"no such round directory: {round_dir}\n")
        return 2
    aliases = dict(pair.split("=", 1) for pair in args.scene_alias)
    scene_dirs = resolve_scene_dirs(round_dir, aliases)
    if not scene_dirs:
        sys.stderr.write(
            f"no matrix scene directory found in {round_dir} "
            f"(looked for {[c for _, cs in LEGACY_SCENE_DIRS for c in cs]!r}); "
            "pass --scene-alias <scene>=<dir>\n"
        )
        return 2

    scene_configs: dict[str, str] = {}
    started_at: str | None = None
    stacks: dict[str, list[str]] = {}
    robot: dict[str, str] = {}
    for key, name in scene_dirs.items():
        scene_dir = round_dir / name
        deploy = scene_dir / f"{args.stem}_deploy.log"
        if not deploy.exists():
            deploy = scene_dir / f"{args.stem}_deploy_excerpt.log"
        lines = (
            deploy.read_text(encoding="utf-8", errors="replace").splitlines()
            if deploy.exists()
            else []
        )
        launch_argv = parse_launch_argv(lines)
        stacks[key] = stack_tokens(launch_argv)
        robot = robot or robot_facts_from_launch_argv(launch_argv)
        started_at = started_at or parse_log_start_time(lines)
        config = sorted(scene_dir.glob("*.yaml"))
        if config:
            scene_configs[key] = str(config[0].relative_to(round_dir))

    distinct = {tuple(tokens) for tokens in stacks.values() if tokens}
    if len(distinct) > 1:
        sys.stderr.write(
            "the scenes of this round did not share one stack, so it cannot be "
            f"recorded as a round: {stacks!r}\n"
        )
        return 2

    metadata = {
        "round_id": args.round_id,
        "started_at": started_at or args.started_at or "",
        "host": args.host,
        "executed_sha": args.executed_sha,
        "worktree_clean": None,
        "launcher_path": "",
        "repo_root": robot.get("repo_root", ""),
        "robot_id": robot.get("robot_id"),
        "robot_manifest_path": robot.get("robot_manifest_path"),
        "sync_groups": list(args.sync_group),
        "stack_argv": list(next(iter(distinct), ())),
        "safety_overrides_absent": True,
        "gpu_name": args.gpu_name,
        "notes_path": "NOTES.md",
        "seed": args.seed,
        "rskill_id": args.rskill_id,
        "scene_configs": scene_configs,
        "scene_dirs": scene_dirs,
        "artifact_stem": args.stem,
        "imported_from": args.imported_from or str(round_dir.resolve()),
    }
    if not metadata["started_at"]:
        sys.stderr.write(
            "no ROS timestamp in any deploy log and no --started-at given; "
            "the round cannot be dated from its artifacts\n"
        )
        return 2
    (round_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"imported {round_dir} as {args.round_id}: {scene_dirs} stem={args.stem!r}")
    return cmd_verdicts(round_dir, stem=args.stem)


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the matrix end to end.

    Nothing is written until every guardrail has passed — a refused round leaves
    no directory behind, which is what "exit 3, no partial round" means.
    """
    round_dir = OUTPUT_ROOT / args.round_id

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

    round_dir.mkdir(parents=True, exist_ok=True)
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
        "artifact_stem": args.stem,
        "scene_pins": dict(SCENE_RUNTIME_PIN),
    }
    (round_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    code = cmd_verdicts(round_dir, stem=args.stem)
    return code or round_exit_code(_load_round(round_dir))


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
    verdicts.add_argument(
        "--stem", default=None, help="Artifact stem; default: the round's recorded artifact_stem."
    )

    diff = sub.add_parser("diff", help="Diff two rounds' verdicts.json.")
    diff.add_argument("round_dir", type=Path)
    diff.add_argument("--baseline", type=Path, required=True)
    diff.add_argument("--out", type=Path, default=None)

    imp = sub.add_parser(
        "import-round",
        help="Record the metadata a pre-harness round never wrote, then derive its verdicts.",
    )
    imp.add_argument("source", type=Path, help="The round directory, e.g. ~/openral-runs/<round>.")
    imp.add_argument("--round-id", required=True, help="Round id to record.")
    imp.add_argument(
        "--executed-sha", required=True, help="SHA the round ran (from its own NOTES / build log)."
    )
    imp.add_argument("--stem", default="seed1", help="Artifact filename stem (default: seed1).")
    imp.add_argument(
        "--scene-alias",
        action="append",
        default=[],
        metavar="SCENE=DIR",
        help="Map a matrix scene onto the directory this round used, e.g. baguette=bag1.",
    )
    imp.add_argument("--seed", type=int, default=1, help="Scene seed the round pinned.")
    imp.add_argument("--rskill-id", default=DEFAULT_RSKILL_ID, help="rSkill the round dispatched.")
    imp.add_argument("--host", default="", help="Hostname of the runner (a recorded fact).")
    imp.add_argument("--gpu-name", default=None, help="GPU the round ran on (a recorded fact).")
    imp.add_argument(
        "--sync-group", action="append", default=[], help="Dependency group the round synced with."
    )
    imp.add_argument(
        "--started-at",
        default="",
        help="Fallback start time when the logs carry no ROS timestamp.",
    )
    imp.add_argument(
        "--imported-from",
        default="",
        help="Producer path to record, when importing a copy of the round (default: SOURCE).",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "verdicts":
            return cmd_verdicts(args.round_dir, stem=args.stem)
        if args.command == "import-round":
            return cmd_import(args)
        return cmd_diff(args.round_dir, args.baseline, args.out)
    except GuardrailError as exc:
        sys.stderr.write(f"GUARDRAIL: {exc}\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
