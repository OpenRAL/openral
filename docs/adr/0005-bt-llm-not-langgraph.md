# ADR-0005: BehaviorTree.CPP v4 XML + typed LLM tool palette for S2 planning, not LangGraph

- Status: Accepted
- Date: 2026-05-24 (retroactive — documents a Week-1 planning-layer decision; the reasoner code itself lands later)
- Amended: 2026-05-24 (see Amendments below)

## Context

CLAUDE.md §6.2 mandates a **dual-system pattern** for every robot agent:
S1 (fast policy, 30–200 Hz, action-chunked VLA) and S2 (slow reasoner,
5–10 Hz, planner). The reasoner has to produce something that:

1. The S1 skill executor can consume **without** a Python-only runtime
   (BT leaves call into ROS 2 actions; the BT executor itself is C++ for
   determinism and hot-reload).
2. Is **inspectable** — a tree in the trace, not a closure graph in
   memory.
3. Is **hot-reloadable** — when the failure detector fires, the
   replanning ladder needs to swap a subtree without restarting the
   process.
4. Has a **bounded vocabulary** — every leaf is a registered Skill, and
   the LLM cannot hallucinate a leaf that isn't installed and capable.

The candidate planning substrates in 2026:

| Substrate | XML/IR format | C++ executor | Hot-reload | LLM-native | Replanning ladder |
|---|---|---|---|---|---|
| **BehaviorTree.CPP v4** | XML (BT v4 grammar) | Yes (`BT::Tree`) | Yes (`registerNodeXML`) | No — needs glue | Subtree-level halt + reload |
| LangGraph | Python `StateGraph` | No | No (Python-resident) | Yes (built for LLM agents) | Needs custom routing |
| LangChain "Agents" | Python | No | No | Yes | None — recursion control via the LLM |
| AutoGen / CrewAI | Python | No | No | Yes | None |
| pddl-stream / TAMP | PDDL+ | C++ available | No | No — symbolic | Native (replanning is the loop) |

The literature (Helix, π0.6, Gemini Robotics, GR00T N2) converges on
BTs for the **executor**, with the LLM as the **author** of the BT
XML. Two patterns are common: (a) prompt the LLM to emit BT XML
directly; (b) prompt the LLM to emit a typed `Plan` object that the
reasoner converts to BT XML deterministically.

The C++ BT executor is mandatory regardless of the choice in this ADR
— ros2_control + lifecycle-node ergonomics + real-time guarantees rule
out a Python executor for S2. The question is what the LLM emits.

## Decision

The S2 planning layer is **typed-LLM-output → deterministic Plan
construction → BT.CPP v4 XML emission**:

1. **The LLM emits a Pydantic `Plan` object**, not raw BT XML.
   - `LLMClient` is a typed Protocol (OpenAI, Anthropic providers
     behind one interface).
   - The structured-output schema is `Plan` from
     `openral_reasoner.plan` (a Pydantic v2 model — per ADR-0003).
   - The provider's structured-output mode enforces the schema at the
     wire level (OpenAI `response_format=Plan`, Anthropic tools).
2. **The reasoner converts `Plan` → BT XML deterministically.** No
   LLM hallucination of the executable artifact; the XML is generated
   by code from a validated `Plan`.
3. **The leaf vocabulary is the local skill registry.**
   - `~/.config/openral/skills.toml` lists installed rSkills + their
     `RSkillManifest.capabilities`.
   - The LLM tool palette is generated from this list at planning
     time. A leaf the LLM tries to emit that isn't installed-and-
     capable raises `ROSPlanningError.ROSReasonerInvalidPlan` (see
     `openral_core.exceptions`).
4. **The C++ executor is BT.CPP v4** (`packages/openral_skill/bt_runner.cpp`,
   planned). Leaves are `RosActionNode`s calling
   `ExecuteRskill.action` on the corresponding skill lifecycle node.
   Hot-reload on `FailureTrigger`.
5. **Replanning ladder is BT-native.** Retry decorators handle local
   retry; subtree halt + LLM re-invocation handles param-tweak and
   substitute-skill. Goal-replan re-prompts the LLM with the failure
   context. Human-handoff is a leaf that publishes a UI event and
   blocks.
6. **No LangGraph, no LangChain Agent, no AutoGen.** These tools live
   one layer above where we need; their value is *orchestrating LLM
   calls*, not orchestrating *robot skills under real-time and safety
   constraints*. They can be used **inside** an `LLMClient`
   implementation (e.g., as a routing layer in front of multiple
   providers) but not as the S2 substrate itself.

## Consequences

- **Pros**
  - The executable artifact (BT XML) is **deterministic given the
    `Plan`** — replays are bit-identical from the trace, satisfying
    CLAUDE.md §8 (reproducibility over speed).
  - The C++ executor gives us real-time guarantees and matches the
    ros2_control / BT.CPP idiom the rest of the robotics ecosystem
    already uses.
  - The typed `Plan` is **testable in isolation** without any LLM —
    `NullReasoner` returns a hand-built `Plan` so plumbing tests
    don't need API credentials.
  - The leaf-vocabulary closure (LLM can only emit installed skills)
    eliminates a class of hallucination failures up-front.
  - Switching LLM providers is a one-line `LLMClient` swap, not a
    framework migration.

- **Cons**
  - We own the `Plan` schema + the deterministic XML emitter. Both
    are small (~200 LoC each) but they're not bought.
  - We do not get LangGraph's free `StateGraph` visualisation; the BT
    XML *is* the visualisation, but tooling for "render this tree as
    a diagram" is on us.
  - When a multi-step LLM dialogue is genuinely needed (e.g., the
    LLM asks the user a clarifying question), we implement it inside
    `LLMClient` — we don't get LangChain's chat-loop helpers for
    free.
  - The C++ BT executor is more work than a Python `StateGraph`
    runner. Mitigated by the existence of BT.CPP v4 as a mature
    upstream library.

## Alternatives considered

- **Pure LangGraph, Python S2.** Rejected — Python S2 can't hit the
  determinism and real-time bar for replanning under failure. Also
  collapses the S1/S2 separation: a Python S2 calling Python S1 has
  no enforcement of the cadence boundary CLAUDE.md §6.2 declares.
- **LLM emits BT XML directly.** Rejected — string-mode XML generation
  is the kind of hallucination surface the structured-output discipline
  exists to avoid. The typed `Plan` lets the provider's
  structured-output mode catch errors at the wire.
- **PDDL + a symbolic planner.** Considered. Strengths: provable
  optimality, no hallucination. Weaknesses: writing PDDL domains by
  hand for every skill set is more work than the LLM tool-palette
  approach; LLMs are demonstrably good at picking the next skill from
  a typed menu but currently bad at writing well-formed PDDL.
- **No S2 at all** (S1-only, like SmolVLA stand-alone). Adequate for
  single-skill demos; insufficient for multi-step manipulation, which
  is the v0.2+ target. Documented as out-of-scope in CLAUDE.md §6.2.

## Why this ADR is retroactive

The decision is encoded in CLAUDE.md §6.2, §7.6, in the explicit
non-goals (CANCELLED block in `docs/architecture/repo-state-map.html`),
and in the roadmap's Week-4 "Reasoner stub (LLM → BT)" entry. The
reasoner code itself is the next layer scaffold to land
(`python/reasoner/` skeleton + Protocols). This ADR records the
substrate choice so that scaffold doesn't have to re-litigate it.

## References

- CLAUDE.md §6.2 (dual-system pattern), §7.6 (working with the planner),
  §10 (exception hierarchy — `ROSPlanningError`).
- `docs/roadmap/index.md` — Week-4 Reasoner stub line.
- `docs/architecture/repo-state-map.html` — Layer 4 (Reasoning), with
  the BT-not-LangGraph stance encoded in CANCELLED.
- BehaviorTree.CPP v4 — <https://www.behaviortree.dev/>.
- ADR-0003 — Pydantic v2 substrate the `Plan` schema rides on.
