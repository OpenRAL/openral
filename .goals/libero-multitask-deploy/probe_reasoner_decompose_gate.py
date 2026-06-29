#!/usr/bin/env python3
"""Probe the locate -> decompose_mission transition on a COLLECTIVE goal.

Reproduces — without the sim / VLA / ROS stack — the deploy tick that misbehaves:
the operator goal is a COLLECTIVE target ("Put all the objects on the table into
the basket"), the open-vocab ``locate_in_view`` detector has ALREADY grounded the
concrete goal nouns into the ``located[<cam>]`` context line, yet glm-5.2 keeps
calling ``locate_in_view`` / ``recall_object`` for minutes instead of calling
``decompose_mission`` to split the goal into one grounded subtask per object.

Unlike the live node, nothing forces the transition: the collective-decompose
self-prompt only fires when the LLM picks ``execute_rskill`` (the execute gate,
:meth:`reasoner_node._dispatch_execute_rskill`). While the LLM loops on the
read-only locate tools it never trips that gate — so the run stalls.

We feed the post-locate context directly to the real LLM (no sim) and classify
what it picks, across edge cases that localise WHEN it stalls, then re-run the
same cases under candidate fixes (a strengthened system prompt and/or a
deterministic context nudge) to measure which reliably flips BAD -> GOOD.

Cases:
  A located-full   — every goal noun + basket grounded in ``located``; in_view is
                     the continuous detector's mislabelled clutter. THE bug:
                     GOOD = decompose_mission, BAD = locate/recall loop.
  C nothing-yet    — only the mislabelled in_view clutter, ``located`` EMPTY, the
                     goal nouns NOT grounded. The control: locating IS correct
                     here. GOOD = locate/recall, BAD = premature decompose/execute.
  D looped-already — like A, plus a PERCEPTION history of repeated locate returns
                     (evidence the LLM already looped). GOOD = decompose_mission.

Run (mirrors run_deploy_glm.sh env):
    export OPENRAL_REASONER_LLM_PROVIDER=openrouter
    export OPENRAL_REASONER_LLM_MODEL=z-ai/glm-5.2
    export OPENRAL_REASONER_LLM_API_KEY=$(grep OPENROUTER_OPENRAL_TOKEN ~/workspace/.env.local \
        | cut -d= -f2- | tr -d '"' | xargs)
    PYTHONPATH=python/core/src:python/reasoner/src .venv/bin/python \
        .goals/libero-multitask-deploy/probe_reasoner_decompose_gate.py \
        [--variant V] [--cases A,C,D] [--runs N] [--dry]

  --variant  baseline | prompt | nudge | both   (default baseline)
  --dry      render + print one context per case and exit (no LLM calls / no spend)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openral_core import (
    JointState,
    ObjectDetection2D,
    ObjectsMetadata,
    RobotCapabilities,
    RSkillManifest,
    WorldState,
    is_collective_target,
)
from openral_reasoner import (
    DEFAULT_SYSTEM_PROMPT,
    ContextRenderer,
    MissionState,
    build_tool_palette,
    build_tool_use_client_from_env,
)
from openral_reasoner.context import PerceptionEventRecord, PromptRecord

REPO_ROOT = Path(__file__).resolve().parents[2]

COLLECTIVE_PROMPT = "Put all the objects on the table into the basket."
# The concrete goal nouns the open-vocab locate_in_view detector grounds.
GOAL_NOUNS = ["teapot", "ketchup", "milk", "basket"]
# The continuous (fixed-vocab indoor) detector's mislabelled clutter — what
# `in_view[top]` shows when omdet-turbo-indoor cannot name the goal objects.
CLUTTER = ["bottle", "cup", "bowl", "box"]
RSKILL = "OpenRAL/rskill-smolvla-libero"

# ── candidate fix (i): an explicit transition rule appended to the brief ──────
# Targets the real defect the probe found: glm decomposes on the continuous
# detector's RAW in_view labels even when they are generic/mislabelled clutter and
# the goal objects are not yet CONFIRMED. The `located[]` line (open-vocab
# locate_in_view confirmations) is authoritative; in_view is a fixed-vocabulary
# stream that often mislabels. So: confirm goal objects into `located` first, then
# decompose immediately and stop locating.
PROMPT_FIX = (
    " ## GROUND BEFORE YOU DECOMPOSE (collective goals). The `in_view[<camera>]` "
    "line comes from a FIXED-VOCABULARY continuous detector that frequently "
    "MISLABELS the goal objects (a teapot read as 'bottle', a basket as 'box'); do "
    "NOT build a mission directly from raw `in_view` labels. The authoritative "
    "grounding is the `located[<camera>]` line — open-vocab locate_in_view "
    "confirmations of the actual goal nouns. Procedure for a collective/quantified "
    "goal (all / every / the objects / things): (1) if `located` does NOT yet name "
    "the goal objects, call locate_in_view to CONFIRM which visible objects are the "
    "ones to act on (this populates `located`); (2) the MOMENT `located` names one "
    "or more concrete goal objects, call decompose_mission (one grounded subtask per "
    "located object) and do NOT locate_in_view / recall_object again for an object "
    "already in `located` — re-locating a confirmed object makes no progress."
)


def _in_view(labels: list[str]) -> ObjectsMetadata:
    return ObjectsMetadata(
        sensor_id="top",
        model_id="omdet-turbo-indoor",
        frame_width=640,
        frame_height=480,
        detections=[
            ObjectDetection2D(
                label=lbl,
                confidence=0.85,
                bbox_xyxy=(100 + 40 * i, 200, 140 + 40 * i, 250),
                det_id=i,
            )
            for i, lbl in enumerate(labels)
        ],
    )


def _world_state() -> WorldState:
    """RGB-only realistic state: joints only, NO 3D lift (no scene_objects line)."""
    return WorldState(
        stamp_ns=1,
        joint_state=JointState(
            name=[f"joint_{i}" for i in range(7)], position=[0.0] * 7, stamp_ns=1
        ),
    )


def _renderer_for(case: str, *, nudge: bool) -> ContextRenderer:
    r = ContextRenderer()
    r.set_mission(MissionState.from_prompt(COLLECTIVE_PROMPT))
    # The continuous detector's mislabelled clutter is always streaming.
    r.set_in_view(_in_view(CLUTTER))
    if case in ("A located-full", "D looped-already"):
        # Open-vocab locate already grounded every goal noun + the basket.
        r.note_located(_in_view(GOAL_NOUNS))
    if case == "B basket-only":
        # Only the DESTINATION is grounded; the goal objects are NOT yet found.
        r.note_located(_in_view(["basket"]))
    if case == "E empty-scene":
        # The continuous detector sees nothing either — a true search situation.
        r.set_in_view(_in_view([]))
    if case == "D looped-already":
        # Evidence the LLM already looped: a history of locate returns.
        for noun in GOAL_NOUNS:
            r.append_perception(
                PerceptionEventRecord(
                    kind="locate_in_view",
                    text=f"locate_in_view: {noun!r} IS in view (top) @px",
                    metadata_json="",
                    stamp_ns=2,
                )
            )
    # case C: located stays EMPTY — nothing grounded yet.
    if nudge:
        _maybe_append_nudge(r)
    return r


# ── candidate fix (ii): a deterministic nudge appended to PROMPTS when the ─────
# located/in_view lines already name concrete objects on a collective goal. This
# mirrors the node's _emit_enumeration_invite, but fires on CONTEXT STATE (objects
# grounded) instead of only on a refused execute_rskill — closing the locate-loop
# hole. In the live node this is published as a frame_id="mission" self-prompt.
def _maybe_append_nudge(r: ContextRenderer) -> None:
    mission = r.mission
    active = mission.active() if mission is not None else None
    if active is None or not is_collective_target(active.text):
        return
    grounded = _grounded_labels(r)
    if not grounded:
        return  # nothing to act on yet -> let the LLM locate (case C)
    listed = ", ".join(sorted(grounded))
    r.append_prompt(
        PromptRecord(
            text=(
                f"You have already grounded concrete objects ({listed}) for the collective "
                f"task {active.task_id} ({active.text!r}). STOP locating — call "
                f"decompose_mission(target_task_id={active.task_id!r}, subtasks=[one grounded "
                "subtask per object]) NOW. Do NOT call locate_in_view / recall_object again "
                "for an object already listed above."
            ),
            metadata_json="",
            stamp_ns=3,
            priority=100,
        )
    )


def _grounded_labels(r: ContextRenderer) -> set[str]:
    """Concrete object labels currently named in located/in_view (minus pure clutter).

    The nudge keys off ``located`` (open-vocab confirmed goal nouns); the
    continuous ``in_view`` clutter alone does not count as a grounded goal object.
    """
    rendered = r.render(world_state=_world_state())
    grounded: set[str] = set()
    for line in rendered.splitlines():
        if line.startswith("located["):
            for noun in GOAL_NOUNS:
                if noun in line.lower():
                    grounded.add(noun)
    return grounded


CASES = [
    "A located-full",
    "B basket-only",
    "C nothing-yet",
    "D looped-already",
    "E empty-scene",
]
_GOOD = {
    "A located-full": {"DECOMPOSE"},
    "D looped-already": {"DECOMPOSE"},
    "B basket-only": {"LOCATE", "RECALL"},
    "C nothing-yet": {"LOCATE", "RECALL"},
    "E empty-scene": {"LOCATE", "RECALL"},
}


def _classify(call) -> tuple[str, str]:  # noqa: ANN001
    tool = call.tool
    if tool == "decompose_mission":
        subs = getattr(call, "subtasks", [])
        return "DECOMPOSE", f"{len(subs)} subs: {[s.object_ref for s in subs]}"
    if tool == "locate_in_view":
        return "LOCATE", getattr(call, "object_ref", "") or getattr(call, "query", "")
    if tool == "recall_object":
        return "RECALL", getattr(call, "object_ref", "")
    if tool == "execute_rskill":
        return "EXECUTE", (getattr(call, "prompt", "") or "")[:50]
    if tool == "emit_prompt":
        return "ESCALATE", "emit_prompt"
    return tool.upper(), ""


def main() -> int:
    """Parse args, build the palette + per-case context, and tally tool choices."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant", choices=["baseline", "prompt", "nudge", "both"], default="baseline"
    )
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    # Allow short case keys ("A", "C", "D") or full names.
    want = []
    for raw in args.cases.split(","):
        tok = raw.strip()
        match = next(
            (c for c in CASES if c == tok or c.startswith(tok + " ") or c[0] == tok), None
        )
        if match:
            want.append(match)
    if not want:
        want = CASES

    use_prompt_fix = args.variant in ("prompt", "both")
    use_nudge = args.variant in ("nudge", "both")
    system_prompt = DEFAULT_SYSTEM_PROMPT + (PROMPT_FIX if use_prompt_fix else "")

    manifest = RSkillManifest.from_yaml(REPO_ROOT / "rskills" / "smolvla-libero" / "rskill.yaml")
    palette = build_tool_palette(
        installed_skills=[manifest],
        robot_capabilities=RobotCapabilities(embodiment_tags=list(manifest.embodiment_tags)),
        detector_available=True,
        spatial_memory_available=True,
        task_progress_available=True,
    )

    if args.dry:
        for case in want:
            print("=" * 72)
            print(f"CASE {case}  (nudge={use_nudge})")
            print("-" * 72)
            print(_renderer_for(case, nudge=use_nudge).render(world_state=_world_state()))
        return 0

    client = build_tool_use_client_from_env()
    model = getattr(client, "model_id", "?")
    print(f"LLM: {model}   variant={args.variant}   runs/case={args.runs}")
    print(f"prompt_fix={use_prompt_fix}  nudge={use_nudge}")
    print("GOOD per case: A/D=DECOMPOSE (objects grounded); B/C/E=LOCATE/RECALL (confirm first).\n")

    for case in want:
        ctx = _renderer_for(case, nudge=use_nudge).render(world_state=_world_state())
        outcomes: list[str] = []
        last_detail = ""
        for _ in range(args.runs):
            try:
                call = client.select_tool(
                    context_text=ctx, palette=palette, system_prompt=system_prompt
                )
                cat, last_detail = _classify(call)
            except Exception as exc:  # probe harness: record the failure, don't abort
                cat, last_detail = "ERROR", str(exc)[:80]
            outcomes.append(cat)
        good = _GOOD[case]
        n_good = sum(1 for o in outcomes if o in good)
        verdict = "PASS" if n_good == len(outcomes) else ("MIX " if n_good else "FAIL")
        print(f"[{verdict}] {case:<16} {n_good}/{len(outcomes)}  {'  '.join(outcomes)}")
        print(f"            last: {last_detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
