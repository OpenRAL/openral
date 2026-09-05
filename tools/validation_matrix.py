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
            depth_is_box_bound=fields.get("depth_is_box_bound") == "1",
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
        published no grid (octomap off, the graph never reached activation, or
        — see :func:`monitor_subscription_records` — the monitor received
        nothing at all).
    """
    for record in records:
        if record.get("event") == "world_voxels" and record.get("resolution") is not None:
            return float(record["resolution"])
    return None


# The monitor's own lifecycle lines. Every other record it writes came from a
# ROS subscription callback, so their count is what the monitor *received*.
_MONITOR_LIFECYCLE_EVENTS: Final[frozenset[str]] = frozenset({"monitor_started", "monitor_stopped"})


def monitor_subscription_records(records: Sequence[Mapping[str, Any]]) -> int:
    """How many of a monitor's records came from a ROS subscription.

    ``0`` on a monitor that ran for the whole scene is the signature of the
    2026-08-23 defect: the monitor's DDS participant was created before
    ``openral deploy sim`` purged ``/dev/shm/fastrtps_*``, so its shared-memory
    segments were unlinked underneath it and it received nothing for the rest
    of the run. All 24 runs of that round wrote exactly ``monitor_started`` and
    ``monitor_stopped``. That is a *harness* fault, and it must not be read as
    "the run stopped before there was anything to record".

    Args:
        records: Monitor JSONL records.

    Returns:
        The record count excluding ``monitor_started`` / ``monitor_stopped``.

    Example:
        >>> monitor_subscription_records(
        ...     [{"event": "monitor_started"}, {"event": "monitor_stopped"}]
        ... )
        0
    """
    return sum(1 for r in records if r.get("event") not in _MONITOR_LIFECYCLE_EVENTS)


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


# Every coverage block a ground-truth snapshot can carry. Their presence is
# per-probe: the payload blocks are `{}` on a run that never grasped.
_PROBE_COVERAGE_KEYS: Final[tuple[str, ...]] = (
    "nearest_probe_coverage",
    "nearest_payload_world_coverage",
    "nearest_payload_robot_coverage",
)

# The coverage key the HAL writes once it filters BOTH sides of a probe to
# solid geoms. A snapshot without it was recorded by a build that filtered only
# the world side, so a robot or payload geom in its pair list may be a purely
# visual mesh (`openral_hal.sim_sensor_bridge._nearest_pair_records`).
_COLLIDABILITY_ATTESTATION: Final[str] = "noncollidable_side_geoms_excluded"

# The coverage key the HAL writes once every probed distance carries a proof
# (`openral_hal.convex_distance`). A snapshot without it was recorded on
# `mujoco.mj_geomDistance`, which is unreliable for RoboCasa-fixture-vs-panda-
# mesh pairs in two silent modes; a non-zero count means this snapshot itself
# contains a distance the producer could not defend.
_CERTIFIED_DISTANCE_ATTESTATION: Final[str] = "uncertified_pairs"


def probe_is_collidability_filtered(snapshot: Mapping[str, Any]) -> bool:
    """Whether a recorded probe attests that *every* side was solid geoms only.

    A distance measured against a geom with neither ``contype`` nor
    ``conaffinity`` is not a penetration: MuJoCo can never contact it and the
    safety kernel never checks it. Until the HAL filtered the robot and payload
    sides as well as the world side, a stop could be adjudicated
    ``real-contact`` off a visual shell — ``robot0_g42_vis`` at 0.000 m from a
    freezer door whose nearest *solid* pair was 2.5 mm clear.

    Args:
        snapshot: The ``sim.estop_ground_truth_snapshot`` payload.

    Returns:
        ``True`` when every non-empty coverage block carries the attestation.

    Example:
        >>> probe_is_collidability_filtered(
        ...     {"nearest_probe_coverage": {"noncollidable_side_geoms_excluded": 12}}
        ... )
        True
        >>> probe_is_collidability_filtered(
        ...     {"nearest_probe_coverage": {"noncollidable_world_geoms_excluded": 917}}
        ... )
        False
    """
    blocks = [snapshot.get(key) for key in _PROBE_COVERAGE_KEYS]
    present = [b for b in blocks if isinstance(b, dict) and b]
    return bool(present) and all(_COLLIDABILITY_ATTESTATION in b for b in present)


def probe_is_distance_certified(snapshot: Mapping[str, Any]) -> bool:
    """Whether every distance in a recorded probe carries its own proof.

    ``mujoco.mj_geomDistance`` — which every probe recorded before this landed
    was built on — returns confidently wrong values on the pair class these
    stops are adjudicated from: a RoboCasa fixture geom against a panda
    collision mesh. Measured cases include ``+0.000000`` where the truth is
    ``+0.148512 mm``, with a 126 mm witness segment lying outside both geoms;
    and, in the checked-in rounds themselves, ``0.000 m`` recorded where the
    certified distance is ``+14.806 mm`` (08-22 fridge), ``+82.185 mm``
    (08-23 fridge) and ``+107.930 mm`` (08-23 baguette, on a *solid* pair).
    Every one of those is a ``0.000 m`` reading, which is exactly what rule 1
    of :func:`adjudicate_ground_truth` promotes to ``real-contact``.

    So the attestation is the same shape as
    :func:`probe_is_collidability_filtered`, for the same reason and with the
    same fail-closed consequence: a snapshot that does not attest its
    distances cannot support a verdict that rests on one. It does **not**
    reverse such a verdict — the stop becomes ``unadjudicated``, and nothing
    here shows the kernel was wrong.

    Args:
        snapshot: The ``sim.estop_ground_truth_snapshot`` payload.

    Returns:
        ``True`` when every non-empty coverage block reports
        ``uncertified_pairs`` and that count is zero.

    Example:
        >>> probe_is_distance_certified(
        ...     {"nearest_probe_coverage": {"uncertified_pairs": 0, "certified_pairs": 24}}
        ... )
        True
        >>> probe_is_distance_certified(
        ...     {"nearest_probe_coverage": {"uncertified_pairs": 2, "certified_pairs": 22}}
        ... )
        False
        >>> probe_is_distance_certified({"nearest_probe_coverage": {"probed_pairs": 431}})
        False
    """
    blocks = [snapshot.get(key) for key in _PROBE_COVERAGE_KEYS]
    present = [b for b in blocks if isinstance(b, dict) and b]
    return bool(present) and all(
        _CERTIFIED_DISTANCE_ATTESTATION in b and not b[_CERTIFIED_DISTANCE_ATTESTATION]
        for b in present
    )


def hal_admissible_gap_m(snapshot: Mapping[str, Any], stop: ValidationStopEvidence) -> float | None:
    """The HAL's own kernel-vs-probe budget for this stop, when it recorded one.

    The kernel measures OBB-to-voxel; the probes measure mesh-to-mesh. The gap
    a *correct* stop can show is therefore the collision model's corner slop
    plus the voxel half-diagonal — which the sim HAL computes per run and
    publishes as ``adjudication_budget.admissible_gap_m``. Recomputing only the
    voxel term here understates it fourfold (21.7 mm against the HAL's 88.2 mm
    on the 2026-08-23 rounds) and turns conservative, correct stops into
    ``false-positive``.

    An attached-payload **self** stop has an OBB on both sides and no voxel at
    all, so it takes the budget's ``self_collision`` block instead.

    Args:
        snapshot: The ``sim.estop_ground_truth_snapshot`` payload.
        stop: The kernel's verdict.

    Returns:
        The admissible gap in metres, or ``None`` for a snapshot recorded
        before the HAL published a budget.
    """
    budget = snapshot.get("adjudication_budget")
    if not isinstance(budget, dict):
        return None
    if stop.kind == "self" and stop.involves_payload:
        self_block = budget.get("self_collision")
        if isinstance(self_block, dict):
            return _as_float(self_block.get("admissible_gap_m"))
    if stop.kind == "self" and not stop.involves_payload:
        # Link vs link (#216). Two OBBs, no voxel and no payload — or, since
        # #202, two exact HULLS. Which one the kernel used is stated by the
        # kernel rather than guessed: `depth_is_box_bound` (#213) means the
        # reported depth is the OBB's bound, so the box term applies.
        link_block = budget.get("link_link")
        if not isinstance(link_block, dict):
            return None
        if stop.depth_is_box_bound:
            return _as_float(link_block.get("admissible_gap_box_m"))
        return _link_link_hull_gap_m(budget, stop)
    return _as_float(budget.get("admissible_gap_m"))


def _link_link_hull_gap_m(budget: Mapping[str, Any], stop: ValidationStopEvidence) -> float | None:
    """The hull-fidelity half of :func:`hal_admissible_gap_m` (#221).

    The box term above is a snapshot-wide upper bound and needs no pair
    identity; the hull term does not have that luxury -- a hull's overhang
    past its source mesh is a PER-LINK quantity (never maxed across links),
    so it is summed here from the two specific links the kernel named rather
    than published as one snapshot-wide number. Either link missing a
    measured overhang (no stage-2 hull, or a hull with no source mesh) leaves
    the pair with no budget to charge.
    """
    slop = budget.get("collision_model_slop")
    links = slop.get("links") if isinstance(slop, dict) else None
    if not isinstance(links, dict):
        return None
    link_a = links.get(stop.party_a)
    link_b = links.get(stop.party_b)
    if not isinstance(link_a, dict) or not isinstance(link_b, dict):
        return None
    overhang_a = _as_float(link_a.get("hull_overhang_m"))
    overhang_b = _as_float(link_b.get("hull_overhang_m"))
    if overhang_a is None or overhang_b is None:
        return None
    return round(overhang_a + overhang_b, 6)


def adjudicate_ground_truth(
    snapshot: Mapping[str, Any] | None,
    stop: ValidationStopEvidence | None,
    grid_resolution_m: float | None,
    *,
    monitor_records: int | None = None,
) -> ValidationGroundTruthAdjudication | None:
    """Adjudicate a kernel stop against the simulator's own distance probe.

    The rule, in order:

    1. Any probed pair at or below 0 m → ``real-contact``, **provided the probe
       attests that both of its sides were collidability-filtered**
       (:func:`probe_is_collidability_filtered`). The kernel was right that the
       configuration is unsafe even when it named a different body: the probe's
       own caveat is that ``contype``/``conaffinity`` exclusions suppress
       MuJoCo *contacts* at real interpenetration, so distance — not the
       contact list — is the test. Without the attestation a 0 m pair may be a
       purely visual mesh, which is no evidence of anything, so the stop is
       ``unadjudicated`` rather than promoted.
    2. Otherwise, compare the clearance of the party the kernel named against
       the kernel's reported depth. A discrepancy beyond the admissible gap
       cannot be explained by representation → ``false-positive``. Within it →
       ``within-quantization``, which is correct conservative behaviour. The
       gap comes from the HAL's own ``adjudication_budget`` when the snapshot
       carries one (:func:`hal_admissible_gap_m`) and falls back to
       :func:`quantization_budget_m` only when it does not.

       **That fallback is asymmetric, and deliberately so.** The voxel term is
       a strict *lower bound* on the admissible gap — the real gap adds the
       collision model's corner slop, which is the larger term on every panda
       link (45-88 mm against 21.7 mm). So on a snapshot with no published
       budget, ``discrepancy <= budget`` still proves ``within-quantization``
       (it is within the smaller bound, hence within the true one), while
       ``discrepancy > budget`` proves nothing at all and yields
       ``unadjudicated``. Judging the 2026-08-22 rounds on the voxel term alone
       is what produced two "false positives", one of which re-derives as
       ``within-quantization`` the moment a real budget exists for the same
       stop.
       **A link-vs-link self stop never reaches this rule.** The kernel names
       two robot links; the probe's robot side excludes the whole robot from
       the far side, so no probed pair measures one against the other and
       ``nearest_robot_world_pairs`` can only offer ``party_a``'s clearance to
       the *world*. That is a different question, and answering it under the
       self-pair's identity is what scored a genuine ``panda_link5``/
       ``panda_link7`` stop at −31.97 mm as ``false-positive`` off link5's
       212 mm clearance to a kitchen island. Such a stop is ``unadjudicated``,
       with no ``nearest_tripping_party_m`` and no ``discrepancy_m``, until the
       snapshot carries a link-vs-link pair set.

    3. A truncated probe, a missing snapshot or an unknown budget →
       ``unadjudicated``, with ``unadjudicated_reason`` saying which. An
       *untruncated* probe that returned no pair is not missing data: it proves
       the nearest geometry is beyond ``distmax_m``, which is used as a strict
       lower bound.

    4. **Finally**, whatever the ladder concluded is withdrawn to
       ``unadjudicated`` unless the probe attests that its distances are
       certified (:func:`probe_is_distance_certified`). Every rule above reads
       a number off that probe, and ``mujoco.mj_geomDistance`` — which every
       round before this one used — is wrong by 15-108 mm on this exact pair
       class, silently. The check runs *last* rather than first so the record
       still names what is being withdrawn (``withdrawn from 'real-contact':
       …``) instead of erasing it. This withdraws verdicts; it reverses none,
       and it never shows the kernel was wrong.

    Args:
        snapshot: The ``sim.estop_ground_truth_snapshot`` payload, if recorded.
        stop: The kernel's verdict, if the run was stopped.
        grid_resolution_m: Occupancy-grid cell size, from the monitor.
        monitor_records: How many records the monitor received
            (:func:`monitor_subscription_records`), so a missing grid
            resolution can name the right cause: ``0`` means the monitor was
            deaf for the whole run, which is a harness fault, not an early stop.

    Returns:
        The adjudication, or ``None`` when there was no stop to adjudicate.
    """
    from openral_core import (  # reason: deferred
        ValidationGroundTruthAdjudication,
    )

    if stop is None:
        return None
    if snapshot is None:
        return ValidationGroundTruthAdjudication(
            verdict="unadjudicated",
            unadjudicated_reason=("no sim.estop_ground_truth_snapshot was recorded for this stop"),
        )

    filtered = probe_is_collidability_filtered(snapshot)
    certified = probe_is_distance_certified(snapshot)

    robot_pairs = list(snapshot.get("nearest_robot_world_pairs") or [])
    payload_world = list(snapshot.get("nearest_payload_world_pairs") or [])
    payload_robot = list(snapshot.get("nearest_payload_robot_pairs") or [])
    link_link = list(snapshot.get("nearest_link_link_pairs") or [])
    all_pairs = robot_pairs + payload_world + payload_robot + link_link

    # The coverage block has to be the one for the probe whose pairs are read
    # below. Reading `nearest_probe_coverage` (the robot-vs-world probe) for a
    # payload stop lets an untruncated robot probe assert "nothing within
    # distmax" about a payload probe that was never consulted — evidence from
    # the wrong instrument, the same error class as scoring against the wrong
    # bodies (#208, #228). Chosen alongside the pair set, not before it.
    coverage_key = "nearest_probe_coverage"

    nearest_any = min((p["distance_m"] for p in all_pairs), default=None)
    nearest_pair = min(all_pairs, key=lambda p: p["distance_m"], default=None)

    # A link-vs-link self stop names two ROBOT links, and this snapshot cannot
    # speak to that pair at all: the HAL's robot probe excludes the whole robot
    # from the far side (`sim_sensor_bridge._nearest_pairs`, `other_excluded`),
    # so `nearest_robot_world_pairs` only ever holds link-vs-WORLD pairs. Taking
    # `party_a`'s world clearance here answers a different question than the
    # kernel asked and quotes it under the self-pair's identity — which is how
    # `panda_link5`/`panda_link7` at -31.97 mm was scored `false-positive` off
    # link5's 212 mm clearance to a kitchen island.
    #
    # #216 gave the HAL a link<->link probe, so a snapshot that carries one can
    # answer the question after all. A stop the kernel judged at HULL fidelity
    # scores only when #221's per-link `hull_overhang_m` is measured for BOTH
    # links `hal_admissible_gap_m` names; otherwise it still cannot be scored,
    # and it is the missing *budget* that stops it, not a missing pair — the
    # two are worth keeping distinct in the reason.
    self_pair = stop.kind == "self" and not stop.involves_payload
    self_pair_unprobed = self_pair and not link_link

    if stop.involves_payload:
        # The payload is one side. WHICH pair set can answer depends entirely on
        # the other side, and `involves_payload` alone does not say. A voxel or
        # a declared place is world geometry: `payload_world`. A robot link
        # (kind="self") is not, and only `payload_robot` ever measures the
        # payload against the robot. Routing every payload stop through
        # `payload_world` scored a payload-vs-`panda_link1` stop at -1.55 mm
        # off the payload's 166 mm clearance to a countertop (#228) — #208 one
        # class over, with the difference that the right pair set was already
        # in the snapshot and simply never consulted.
        partner_name = stop.party_b if stop.party_a.startswith("attached:") else stop.party_a
        if partner_name.startswith(("voxel_", "place:")):
            party_pairs = payload_world
            coverage_key = "nearest_payload_world_coverage"
        else:
            partner = _kernel_party_to_mujoco(partner_name)
            party_pairs = [p for p in payload_robot if str(p.get("body_b")) == partner]
            coverage_key = "nearest_payload_robot_coverage"
    elif self_pair_unprobed:
        party_pairs = []
    elif self_pair:
        # Both parties are links, so either side of the probed pair may be the
        # one the kernel named; match on both.
        subject = _kernel_party_to_mujoco(stop.party_a)
        partner = _kernel_party_to_mujoco(stop.party_b)
        party_pairs = [
            p
            for p in link_link
            if {str(p.get("body_a")), str(p.get("body_b"))} == {subject, partner}
        ]
    else:
        subject = _kernel_party_to_mujoco(stop.party_a)
        party_pairs = [p for p in robot_pairs if str(p.get("body_a")) == subject]
    nearest_party = min((p["distance_m"] for p in party_pairs), default=None)

    coverage = snapshot.get(coverage_key) or {}
    truncated = bool(coverage.get("truncated", True))
    distmax = _as_float(coverage.get("distmax_m"))
    probed = coverage.get("probed_pairs")

    quantization = None if grid_resolution_m is None else quantization_budget_m(grid_resolution_m)
    hal_gap = hal_admissible_gap_m(snapshot, stop)
    # The grid-quantization fallback is a VOXEL term, and a link-vs-link self
    # stop has no voxel on either side. Falling back to it there would charge a
    # budget from a comparison the stop never made — the same class of error as
    # scoring the pair against world geometry (#208). No HAL budget means no
    # budget.
    budget = hal_gap if hal_gap is not None else (None if self_pair else quantization)
    budget_source = ""
    if hal_gap is not None:
        budget_source = "hal-adjudication-budget"
    elif budget is not None:
        budget_source = "grid-quantization"

    lower_bound = nearest_party
    # No pair within distmax on an untruncated probe: the nearest solid geometry
    # is provably further away than distmax. That inference needs the partner to
    # have been a *candidate* — the tripping link's self-pair partner never is,
    # so its absence from the pair list bounds nothing.
    absence_is_evidence = not self_pair_unprobed and not truncated and bool(probed)
    if lower_bound is None and absence_is_evidence and distmax is not None:
        lower_bound = distmax
    discrepancy = None if lower_bound is None else lower_bound - stop.min_distance_m

    verdict: str
    reason = ""
    if nearest_any is not None and nearest_any <= 0.0:
        if filtered:
            verdict = "real-contact"
        else:
            verdict = "unadjudicated"
            reason = (
                f"the nearest probed pair is at {nearest_any:.6g} m, but this snapshot "
                "does not attest that the robot/payload side of the probe was "
                "collidability-filtered, so the pair may be a purely visual mesh"
            )
    elif truncated:
        verdict = "unadjudicated"
        reason = "the near-miss probe hit its call budget, so an absent pair proves nothing"
    elif self_pair_unprobed:
        verdict = "unadjudicated"
        reason = (
            f"the kernel named the self-pair {stop.party_a!r}/{stop.party_b!r}, and this "
            "probe never measures one robot link against another — its robot side "
            "excludes the whole robot from the far side — so the tripping pair's "
            "clearance is unknown. The link's clearance to the WORLD answers a "
            "different question and is not quoted here"
        )
    elif self_pair and budget is None:
        # The pair WAS probed (#216) — what is missing is a term to judge it by.
        # The kernel did not flag `depth_is_box_bound`, so it judged the pair
        # at hull fidelity, and #221 charges that a `hull_overhang_m(a) +
        # hull_overhang_m(b)` budget when both links have one measured. This
        # branch is reached only when at least one does not -- no stage-2 hull
        # on that link, or a hull with no source mesh on disk to measure
        # against (never defaulted to 0). Charging the box budget instead would
        # forgive a real overlap by up to twice the OBB corner slop.
        verdict = "unadjudicated"
        reason = (
            f"the kernel judged the self-pair {stop.party_a!r}/{stop.party_b!r} at hull "
            "fidelity (depth_is_box_bound is not set), and at least one of the two links "
            "has no measured hull_overhang_m, so no admissible gap can be charged for "
            "this hull-to-mesh comparison. See openral#221"
        )
    elif budget is None:
        verdict = "unadjudicated"
        reason = _no_budget_reason(monitor_records)
    elif discrepancy is None:
        verdict = "unadjudicated"
        reason = (
            f"no probed pair involved {stop.party_a!r}, the party the kernel named, "
            "so its clearance is unknown"
        )
    elif discrepancy > budget:
        if budget_source == "hal-adjudication-budget":
            verdict = "false-positive"
        else:
            # The voxel term is a LOWER BOUND on the admissible gap: the true
            # gap is corner_slop(link) + voxel_half_diagonal, and the slop term
            # is the larger of the two on every panda link. A discrepancy
            # beyond a lower bound therefore establishes nothing. Within it
            # still does — see `within-quantization` below.
            verdict = "unadjudicated"
            reason = _lower_bound_only_reason(discrepancy, budget)
    else:
        verdict = "within-quantization"

    if not certified:
        # The measurement itself, applied after the ladder rather than before
        # it, so the record still says WHAT is being withdrawn. A probe built on
        # `mj_geomDistance` can be wrong by 15-108 mm on exactly this pair
        # class, in both directions of the comparison above: too small a
        # distance manufactures `real-contact` and shrinks the discrepancy
        # toward `within-quantization`, while too large a one removes a pair
        # whose absence is then read as "nothing was near". Neither bias is
        # proved to be the only one, so no verdict may rest on an unattested
        # distance. WITHDRAW, never reverse — this says the evidence cannot
        # decide, not that the kernel was wrong.
        withdrawn = "" if verdict == "unadjudicated" else f"withdrawn from {verdict!r}: "
        certification = (
            "this snapshot does not attest that its probed distances are certified "
            "(`nearest_*_coverage.uncertified_pairs`), so they were measured with "
            "`mujoco.mj_geomDistance`, which is unreliable for RoboCasa-fixture-vs-"
            "panda-mesh pairs — see docs/reference/collision-validation-evidence.md "
            "standing caveat 8"
        )
        verdict = "unadjudicated"
        reason = f"{withdrawn}{certification}" + (
            f". The probe's own reading, which this withdraws: {reason}" if reason else ""
        )

    return ValidationGroundTruthAdjudication(
        verdict=verdict,
        stop_class=snapshot.get("stop_class"),
        sim_time_s=_as_float(snapshot.get("sim_time_s")),
        grid_resolution_m=grid_resolution_m,
        quantization_budget_m=quantization,
        admissible_gap_m=budget,
        budget_source=budget_source,
        probe_collidability_filtered=filtered,
        probe_distance_certified=certified,
        unadjudicated_reason=reason,
        nearest_any_m=nearest_any,
        nearest_tripping_party_m=nearest_party,
        nearest_pair=dict(nearest_pair) if nearest_pair else {},
        discrepancy_m=discrepancy,
        probed_pairs=None if probed is None else int(probed),
        probe_truncated=truncated,
        distmax_m=distmax,
        payload_contacts=len(snapshot.get("payload_contacts") or []),
    )


def _lower_bound_only_reason(discrepancy_m: float, budget_m: float) -> str:
    """Why a discrepancy past the voxel term alone is not a false positive.

    The admissible gap is ``corner_slop(link) + voxel_half_diagonal``, and on
    every panda link the slop term is the larger of the two (45-88 mm against
    21.7 mm). A snapshot recorded before the HAL published ``adjudication_budget``
    (#144, ``ea1b7e8``) supplies only the voxel term, which is a strict lower
    bound on the real gap — so ``discrepancy > budget`` proves nothing, while
    ``discrepancy <= budget`` still proves ``within-quantization``.
    """
    return (
        f"the discrepancy of {discrepancy_m:.6g} m exceeds the {budget_m:.6g} m voxel "
        "term, but this snapshot publishes no adjudication_budget, and the voxel term "
        "alone is a LOWER BOUND on the admissible kernel-vs-probe gap — it omits the "
        "collision model's corner slop, which is the larger term. Exceeding a lower "
        "bound does not establish a false positive; the stop cannot be adjudicated "
        "from these artifacts"
    )


def _no_budget_reason(monitor_records: int | None) -> str:
    """Why no admissible gap was available — a harness fault or a run fact.

    The two read identically in a ``verdicts.json`` (``grid_resolution_m:
    null``) and mean opposite things. A monitor that received nothing is a
    harness fault and the round's evidence is missing; a monitor that received
    plenty but no grid is a run that stopped before octomap published one.
    """
    if monitor_records == 0:
        return (
            "the monitor received nothing at all for the whole scene, so no grid "
            "resolution was ever observed and no budget could be derived — the "
            "evidence is missing for a harness reason, not because the run stopped early"
        )
    if monitor_records is None:
        return "no monitor JSONL was recorded, so the grid resolution is unknown"
    return (
        "the run stopped before the monitor saw a voxel grid, so the grid "
        "resolution — and with it the budget — is unknown"
    )


LAUNCH_FAILED_MARKER: Final[str] = "launch_failed.txt"

# click prints its own usage error to the same sink the deploy log captures, so
# a stack that never started still produces lines. The first live round was
# reported as `deadline-no-grasp` with exit 0 for exactly that reason.
_CLI_USAGE_PREFIX: Final[str] = "Usage: openral"
_CLI_ERROR_NEEDLES: Final[tuple[str, ...]] = ("No such option", "Error:", "Missing option")

# `ros2 launch` writes this and then unwinds. Whatever nodes it had already
# spawned keep running, so the log is long, the graph is partly up, and nothing
# else in it says the launch aborted. At 87dcda1 a missing
# `payload_footprint_node.py` produced exactly this: the dispatcher raised, and
# the scene was bucketed `deadline-no-grasp` with both failure fields empty.
_LAUNCH_EXCEPTION_NEEDLE: Final[str] = "[ERROR] [launch]: Caught exception in launch"
_LAUNCH_EXCEPTION_NOISE: Final[str] = "(see debug for traceback):"


def _launch_exception_detail(deploy_lines: Sequence[str]) -> str:
    """The text of the first ``Caught exception in launch`` line, or ``""``."""
    line = next((ln for ln in deploy_lines if _LAUNCH_EXCEPTION_NEEDLE in ln), "")
    if not line:
        return ""
    tail = line.split(_LAUNCH_EXCEPTION_NEEDLE, 1)[1]
    return tail.replace(_LAUNCH_EXCEPTION_NOISE, "").strip()


def detect_launch_failure(run_dir: Path, stem: str, deploy_lines: Sequence[str]) -> str:
    """Say why this scene's artifacts are not a run at all, or return ``""``.

    Four ways a scene can fail to launch, all of them recorded rather than
    inferred:

    1. the runner wrote ``<stem>_launch_failed.txt`` because the rSkill action
       server never appeared;
    2. ``ros2 launch`` threw — a missing executable, an unparseable launch
       file — and unwound, leaving a partial graph and a long log. This is the
       one that has no marker file and no usage banner, so it used to read as a
       clean run;
    3. the deploy CLI rejected its own argv, so ``click`` wrote a usage error
       where the graph's output would have been;
    4. there is no deploy log at all.

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
        >>> detect_launch_failure(
        ...     pathlib.Path("."),
        ...     "run",
        ...     [
        ...         "[ERROR] [launch]: Caught exception in launch"
        ...         " (see debug for traceback): executable 'x.py' not found"
        ...     ],
        ... )
        "ros2 launch aborted: executable 'x.py' not found"
    """
    caught = _launch_exception_detail(deploy_lines)
    marker = run_dir / f"{stem}_{LAUNCH_FAILED_MARKER}"
    if marker.exists():
        recorded = marker.read_text(encoding="utf-8", errors="replace").strip()
        recorded = recorded or "the rSkill action server never appeared"
        return f"{recorded} (ros2 launch aborted: {caught})" if caught else recorded
    if caught:
        return f"ros2 launch aborted: {caught}"
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


_TRACEBACK_HEADER: Final[str] = "Traceback (most recent call last):"


def parse_goal_log(lines: Sequence[str]) -> tuple[dict[str, Any] | None, str]:
    """The dispatcher's JSON status line, or why it never wrote one.

    ``tools/_validation_matrix_dispatch.py`` prints exactly one JSON object and
    nothing else — *unless* it raises, in which case the log is a Python
    traceback and carries no ``status`` at all. Reading only the JSON left
    ``dispatch_failure_reason`` empty on a run that never dispatched anything,
    which is how an aborted launch came to look like a clean deadline.

    Args:
        lines: Lines of ``<stem>_goal.log``.

    Returns:
        ``(status, failure_reason)``. ``status`` is the decoded JSON object, or
        ``None`` when the dispatcher never produced one; ``failure_reason`` is
        then why, taken verbatim from the traceback's own last line.

    Example:
        >>> parse_goal_log(
        ...     ["Traceback (most recent call last):",
        ...      "  File \\"x.py\\", line 1, in <module>",
        ...      "RuntimeError: /openral/execute_rskill unavailable"]
        ... )[1]
        'the dispatcher raised before reporting a status: RuntimeError: /openral/execute_rskill unavailable'
    """
    status: dict[str, Any] | None = None
    for line in lines:
        with contextlib.suppress(json.JSONDecodeError):
            decoded = json.loads(line.strip() or "{}")
            if isinstance(decoded, dict) and "status" in decoded:
                status = decoded
    if status is not None:
        return status, str(status.get("failure_reason", ""))
    body = [ln.rstrip() for ln in lines if ln.strip()]
    if not body:
        return None, ""
    if any(_TRACEBACK_HEADER in ln for ln in body):
        return None, f"the dispatcher raised before reporting a status: {body[-1].strip()}"
    return None, f"the dispatcher wrote no JSON status line; its log ends: {body[-1].strip()}"


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
    monitor_records = monitor_subscription_records(records) if monitor_path.exists() else None
    goal_lines = (
        goal_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if goal_path.exists()
        else []
    )
    goal, dispatch_failure_reason = parse_goal_log(goal_lines)

    stop = parse_kernel_collision(deploy_lines)
    success = parse_json_log_line(deploy_lines, "sim.task_success_final") or {}
    snapshot = parse_json_log_line(deploy_lines, "sim.estop_ground_truth_snapshot")
    initial = parse_json_log_line(deploy_lines, "sim.estop_initial_configuration")
    resolution = grid_resolution_from_monitor(records)
    ground_truth = adjudicate_ground_truth(
        snapshot, stop, resolution, monitor_records=monitor_records
    )
    witness = build_witness_timeline(records, deploy_lines)

    harness_error_reason = detect_launch_failure(run_dir, stem, deploy_lines)
    if not harness_error_reason and goal is None and goal_lines:
        # The dispatcher produced output but never a status line, so no goal
        # ever reached a terminal state: that is a harness failure, not a
        # deadline. Every such path in `_validation_matrix_dispatch.py` raises
        # (server unavailable, goal rejected); a genuine deadline overrun still
        # prints `{"status": -1, ...}`.
        harness_error_reason = dispatch_failure_reason
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
            ("monitor_gate", run_dir / f"{stem}_monitor_gate.txt"),
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
        monitor_records=monitor_records,
        dispatch_failure_reason=dispatch_failure_reason,
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

    The result is a pure reproducibility comparison only when both rounds carry
    the same ``executed_sha`` **and** the same ``seed``; anything else is a
    before/after. The seed is load-bearing because it decides the scene's
    initial configuration — a seed-1-vs-seed-2 pair at one SHA compares two
    different scenes, and was once labelled ``reproducibility`` on the SHA
    alone. The distinction matters because the policy is stochastic across runs
    even at a pinned seed, so only failure *classes* compare between rounds.

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
        seed=current.metadata.seed,
        baseline_seed=baseline.metadata.seed,
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


def collision_scale_env() -> dict[str, float]:
    """The #188 graded-velocity band this round will actually run with.

    The band reaches the kernel through ``OPENRAL_COLLISION_SCALE_*`` env vars
    (``sim_e2e.launch.py`` reads them), which :func:`assert_no_safety_overrides`
    cannot see — it inspects argv. Recorded rather than refused: arming the band
    *is* the point of the A/B battery, and what must never happen is a round
    that armed it and cannot afterwards be told apart from one that did not.

    Returns:
        The parsed values that are set, keyed by kernel parameter name. Empty
        when the round runs the shipped default (band disabled).
    """
    out: dict[str, float] = {}
    for env, param in (
        ("OPENRAL_COLLISION_SCALE_PROXIMITY_M", "collision_scale_proximity_m"),
        ("OPENRAL_COLLISION_SCALE_K", "collision_scale_k"),
        ("OPENRAL_COLLISION_SCALE_MIN", "collision_scale_min"),
    ):
        raw = os.environ.get(env)
        if not raw:
            continue
        try:
            out[param] = float(raw)
        except ValueError:
            # The launch ignores an unparseable value, so the round did not run
            # with it either; recording it would misdescribe the round.
            continue
    return out


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


def wait_for_dds_transport_ready(
    deploy_log: Path,
    proc: subprocess.Popen[bytes],
    *,
    timeout_s: float,
    poll_s: float = 0.05,
) -> str:
    """Block until the deploy CLI announces its DDS transport is safe to join.

    ``openral deploy sim`` unlinks *every* ``/dev/shm/fastrtps_*`` this user
    owns immediately before spawning ``ros2 launch``
    (``openral_cli.deploy_sim._apply_rmw_default``). A participant created
    before that purge loses its shared-memory segments underneath it: it does
    not error, it simply never receives anything again. The 2026-08-23 round
    lost all 24 monitors that way — every ``run_monitor.jsonl`` in it holds
    exactly ``monitor_started`` and ``monitor_stopped``.

    So the monitor is started on the far side of the marker
    (:data:`~openral_cli.deploy_sim.DDS_TRANSPORT_READY_MARKER`), which the
    deploy prints after the purge and before ``ros2 launch`` is spawned. That
    keeps the whole of the early-stop coverage #145 added: the sim clock does
    not start until the HAL node comes up, tens of seconds after this line, so
    a scene whose initial configuration trips the kernel at sim t≈4.7 s is
    still fully recorded.

    Args:
        deploy_log: The scene's deploy log, being written by ``proc``.
        proc: The running ``openral deploy sim``.
        timeout_s: How long to wait before giving up on the marker.
        poll_s: Re-read interval.

    Returns:
        The marker line, or ``""`` when the deploy exited or the wait timed out
        — the caller starts the monitor anyway and records that it did.
    """
    from openral_cli.deploy_sim import DDS_TRANSPORT_READY_MARKER  # reason: deferred

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if deploy_log.exists():
            for line in deploy_log.read_text(encoding="utf-8", errors="replace").splitlines():
                if DDS_TRANSPORT_READY_MARKER in line:
                    return line.strip()
        if proc.poll() is not None:
            return ""
        time.sleep(poll_s)
    return ""


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
            #
            # But not before the deploy's `/dev/shm/fastrtps_*` purge, which
            # silently unlinks the segments of any participant that already
            # exists — the 2026-08-23 round attached ~6 ms in and every one of
            # its 24 monitors recorded nothing for the whole scene. The gate is
            # the deploy's own readiness marker, so the monitor is up long
            # before the sim clock starts and still on the safe side of the
            # purge.
            ready = wait_for_dds_transport_ready(deploy_log, proc, timeout_s=300.0)
            gate = run_dir / f"{stem}_monitor_gate.txt"
            gate.write_text(
                (
                    f"monitor started after the deploy's readiness line:\n{ready}\n"
                    if ready
                    else "the deploy never printed its DDS readiness line within 300 s; "
                    "the monitor was started anyway and may have lost its shared-memory "
                    "segments to the /dev/shm/fastrtps_* purge — treat an empty "
                    "monitor JSONL from this scene as missing evidence.\n"
                ),
                encoding="utf-8",
            )
            sys.stderr.write(
                f"[matrix] {spec.key}: {'DDS ready, ' if ready else 'DDS readiness TIMED OUT, '}"
                "starting monitor\n"
            )
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
        "- Graded velocity band (#188): "
        + (
            ", ".join(f"`{k}={v}`" for k, v in sorted(meta.collision_scale.items()))
            if meta.collision_scale
            else "**disabled** (shipped default)"
        ),
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
    # Two very different causes of a null `grid_resolution_m`, which the
    # 2026-08-23 round reported as one. A monitor that received nothing has no
    # evidence about the run at all and is a HARNESS fault; a monitor that
    # received plenty but no `world_voxels` describes a run that stopped before
    # octomap published a grid. Naming them apart is the whole point.
    deaf = [s.scene for s in verdicts.scenes if s.monitor_records == 0]
    blind = [
        s.scene
        for s in verdicts.scenes
        if s.monitor_records
        and s.ground_truth is not None
        and s.ground_truth.grid_resolution_m is None
    ]
    broken = [s for s in verdicts.scenes if s.outcome == "harness-error"]
    out += [
        "",
        f"- Exemption active at the trip: {', '.join(exempted) if exempted else 'none'}.",
        f"- `place_allowance_active=1` disclosed in: {', '.join(allowance) if allowance else 'none'}.",
        f"- **Monitor received nothing** (zero records on every subscription — the"
        f" scene's evidence is missing for a harness reason, not absent because the"
        f" run stopped early): {', '.join(deaf) if deaf else 'none'}.",
        f"- Stopped before the monitor saw a voxel grid (no grid resolution, so no"
        f" budget and `unadjudicated` states only that):"
        f" {', '.join(blind) if blind else 'none'}.",
    ]
    unfiltered = [
        s.scene
        for s in verdicts.scenes
        if s.ground_truth is not None and s.ground_truth.probe_collidability_filtered is False
    ]
    if unfiltered:
        out.append(
            "- Probe not collidability-filtered on the robot/payload side, so a 0 m"
            " pair may be a purely visual mesh and cannot support `real-contact`:"
            f" {', '.join(unfiltered)}."
        )
    # The other half of the same epistemics: a budget that is only the voxel
    # term is a lower bound, so it can clear a stop but never convict one.
    lower_bound_only = [
        s.scene
        for s in verdicts.scenes
        if s.ground_truth is not None and s.ground_truth.budget_source == "grid-quantization"
    ]
    if lower_bound_only:
        out.append(
            "- No `adjudication_budget` published (pre-#144 artifacts), so only the voxel"
            " term was available. It is a lower bound on the admissible gap: it can"
            " establish `within-quantization`, never `false-positive`:"
            f" {', '.join(lower_bound_only)}."
        )
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
    if diff.is_reproducibility:
        kind = "reproducibility"
    elif diff.same_sha:
        # Same code, different scene: the seed decides the initial
        # configuration, so this is a before/after however equal the SHAs are.
        kind = f"before/after — same sha, seed {diff.baseline_seed} -> {diff.seed}"
    else:
        kind = "before/after"
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
        "collision_scale": collision_scale_env(),
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
