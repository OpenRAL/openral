"""Top-k skill subset per mission task, for reasoner models that break on large palettes.

A small on-device reasoner (Cosmos3-Edge, 4B) emits runaway tool calls once it sees
many near-identical skill tools: on Jetson AGX Thor ``franka_panda`` with every query
tool gets 0/20 valid ticks with its 14 skills offered and 19/20 with 4 of them (#440,
#494). The failure tracks the number of skill tools, not prompt tokens, so the fix is
to offer fewer skills, not shorter ones.

``select_skill_subset`` ranks the palette's ``RSkillToolEntry`` records against the
mission text by keyword overlap and returns a narrowed ``ToolPalette`` holding at most
``k`` of them. Everything else in the palette (query tools, ``node_ids``,
``sensor_ids``, detectors) is unchanged. Decode validates against the narrowed
palette, so a call to a skill outside the subset is refused like any off-palette id.

Keyword overlap rather than embeddings: no new dependency, deterministic, and
replayable from the trace (CLAUDE.md §1.8). Equal scores break, in order, on:
skills not yet tried for the active task first (so the substitute-skill rung of
the replanning ladder reaches skills outside the top-k), then higher weight
precision (``PRECISION_RANK``), then ``rskill_id``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from openral_core import QuantizationDtype, RSkillAction
from pydantic import BaseModel, ConfigDict, Field

from openral_reasoner.palette import RSkillToolEntry, ToolPalette

__all__ = [
    "PRECISION_RANK",
    "SUPPORT_ACTIONS",
    "SkillSubset",
    "mission_keywords",
    "query_keywords",
    "score_skill",
    "select_skill_subset",
]

# Field weights: an object or action-verb match says more about whether a skill
# fits the task than a word that happens to appear in its prose description.
_OBJECT_WEIGHT = 3.0
_ACTION_WEIGHT = 3.0
_SCENE_WEIGHT = 1.0
_TEXT_WEIGHT = 1.0

SUPPORT_ACTIONS: frozenset[RSkillAction] = frozenset({RSkillAction.LOOK, RSkillAction.NAVIGATE})
"""Actions of the skills the system prompt's own workflow calls for around any object
task (APPROACH with navigation, FRAME with a look-at). A task text rarely names them
("pick up the bowl" says nothing about aiming the camera), so the best skill of each
of these actions keeps a slot whatever it scores, up to ``k - 1`` slots in total."""

PRECISION_RANK: dict[QuantizationDtype, int] = {
    QuantizationDtype.FP32: 0,
    QuantizationDtype.BF16: 0,
    QuantizationDtype.FP16: 0,
    QuantizationDtype.FP8: 1,
    QuantizationDtype.INT8: 2,
    QuantizationDtype.INT4: 3,
    QuantizationDtype.FP4_NVFP4: 3,
}
"""Tie-break rank of a skill's run dtype, lower first: full / half precision, then
8-bit float, int8, 4-bit. A coarse prior, not a measurement: quantization can cost a
policy real success (pi05's own manifest: int8 62% vs bf16 95% on LIBERO-Spatial)."""

# Words that carry no task meaning. Kept short on purpose: an over-eager list
# drops real object names.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "onto",
        "or",
        "that",
        "the",
        "their",
        "them",
        "then",
        "they",
        "this",
        "to",
        "up",
        "use",
        "via",
        "when",
        "with",
    }
)
# Letters and digits in any script; ``_`` and ``-`` split, so ``eef_pose`` is two words.
_TOKEN_RE = re.compile(r"[^\W_]+")
# Below these lengths a suffix is part of the word, not an inflection ("bus", "ring").
_MIN_PLURAL_LEN = 4
_MIN_IES_LEN = 5
_MIN_ING_LEN = 6
_MIN_ED_LEN = 5


def _forms(word: str) -> frozenset[str]:
    """``word`` and the base forms its English inflection may stand for.

    Over-generates on purpose (``boxes`` → ``boxe``, ``box``): a match needs only one
    shared form, so a spurious form costs nothing and a missing one loses a match.

    Example:
        >>> sorted(_forms("boxes"))
        ['box', 'boxe', 'boxes']
        >>> "pick" in _forms("picking") and "place" in _forms("placing")
        True
    """
    forms = {word}
    if len(word) >= _MIN_PLURAL_LEN and word.endswith("s") and not word.endswith("ss"):
        forms.add(word[:-1])
        if word.endswith("es"):
            forms.add(word[:-2])
        if len(word) >= _MIN_IES_LEN and word.endswith("ies"):
            forms.add(word[:-3] + "y")
        if len(word) >= _MIN_IES_LEN and word.endswith("ves"):
            forms.update({word[:-3] + "f", word[:-3] + "fe"})
    if len(word) >= _MIN_ING_LEN and word.endswith("ing"):
        stem = word[:-3]
        forms.update({stem, stem + "e"})
        if stem[-1] == stem[-2]:
            forms.add(stem[:-1])
    if len(word) >= _MIN_ED_LEN and word.endswith("ed"):
        stem = word[:-2]
        forms.update({stem, word[:-1]})
        if stem[-1] == stem[-2]:  # dropped -> drop, grabbed -> grab
            forms.add(stem[:-1])
    return frozenset(forms)


def mission_keywords(text: str) -> frozenset[str]:
    """Case-folded content words of ``text`` (any script), stopwords removed.

    Example:
        >>> sorted(mission_keywords("Pick up the black bowls and place them on the plate"))
        ['black', 'bowls', 'pick', 'place', 'plate']
        >>> sorted(mission_keywords("Nimm die Schüssel"))
        ['die', 'nimm', 'schüssel']
    """
    return frozenset(t for t in _TOKEN_RE.findall(text.casefold()) if t not in _STOPWORDS)


# Everyday verbs an operator uses for an ``RSkillAction`` value, so "put the
# apple in the basket" matches a ``place`` skill. Query side only.
_VERB_SYNONYMS: dict[str, str] = {
    "put": "place",
    "set": "place",
    "drop": "place",
    "stack": "place",
    "grab": "pick",
    "take": "pick",
    "lift": "pick",
    "fetch": "pick",
    "aim": "look",
    "point": "look",
    "inspect": "look",
    "move": "reach",
    "go": "navigate",
    "drive": "navigate",
    "turn": "rotate",
    "twist": "rotate",
    "shut": "close",
    "press": "push",
    "clean": "wipe",
}


def query_keywords(text: str) -> frozenset[str]:
    """``mission_keywords`` plus the ``RSkillAction`` verb each everyday verb names.

    Inflected verbs map too (``putting`` / ``dropped`` -> ``place``).

    Example:
        >>> sorted(query_keywords("put the apple in the basket"))
        ['apple', 'basket', 'place', 'put']
        >>> "pick" in query_keywords("grabbing the cup")
        True
    """
    words = mission_keywords(text)
    return words | {
        _VERB_SYNONYMS[form] for w in words for form in _forms(w) if form in _VERB_SYNONYMS
    }


def _field_forms(values: Iterable[str]) -> frozenset[str]:
    """Every form of every keyword in ``values``."""
    out: set[str] = set()
    for value in values:
        for word in mission_keywords(value):
            out |= _forms(word)
    return frozenset(out)


def _distinct(keywords: frozenset[str]) -> list[frozenset[str]]:
    """The forms of each keyword, keeping one keyword per shared form (``bowl``/``bowls``)."""
    groups: list[frozenset[str]] = []
    for word in sorted(keywords):
        forms = _forms(word)
        if not any(forms & seen for seen in groups):
            groups.append(forms)
    return groups


def _score(fields: tuple[tuple[float, frozenset[str]], ...], keywords: frozenset[str]) -> float:
    """Each keyword once, at the weight of the strongest field sharing one of its forms."""
    return sum(
        max((w for w, field in fields if forms & field), default=0.0)
        for forms in _distinct(keywords)
    )


def score_skill(entry: RSkillToolEntry, keywords: frozenset[str]) -> float:
    """Relevance of one skill to the mission keywords.

    Each keyword counts once, at the weight of the strongest field it matches
    (objects / actions 3, scenes / description / id 1). Plurals and ``-ing`` /
    ``-ed`` forms match their base (``boxes`` ↔ ``box``, ``picking`` ↔ ``pick``).

    Example:
        >>> from openral_core import RSkillAction
        >>> entry = RSkillToolEntry(
        ...     rskill_id="OpenRAL/rskill-act-franka_panda-libero-fp32",
        ...     description="ACT on LIBERO tabletop tasks.",
        ...     actions=(RSkillAction.PICK, RSkillAction.PLACE),
        ...     objects=("bowl",),
        ... )
        >>> score_skill(entry, query_keywords("picking the bowls"))
        6.0
        >>> score_skill(entry, query_keywords("open the fridge"))
        0.0
    """
    fields = (
        (_OBJECT_WEIGHT, _field_forms(entry.objects)),
        (_ACTION_WEIGHT, _field_forms(a.value for a in entry.actions)),
        (_SCENE_WEIGHT, _field_forms(entry.scenes)),
        (_TEXT_WEIGHT, _field_forms((entry.description, entry.rskill_id))),
    )
    return _score(fields, keywords)


class SkillSubset(BaseModel):
    """The skills offered on a tick and why (logged and stamped on the tick span).

    Attributes:
        palette: The narrowed palette the LLM sees and decode validates against.
        scores: ``(rskill_id, score)`` for every skill in the full palette,
            best first, so a skill left out can be diagnosed from the trace.
        query: The mission text the skills were ranked against.
        sticky: Ids kept regardless of score (in flight, then last dispatched).
        focused: The id kept as the best match for the top pending operator prompt.
        support: Ids kept as the best skill of a ``SUPPORT_ACTIONS`` action.
        tried: Ids already dispatched for the active task (demoted among ties).
        k: The cap applied.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    palette: ToolPalette
    scores: tuple[tuple[str, float], ...]
    query: str
    sticky: tuple[str, ...] = ()
    focused: tuple[str, ...] = ()
    support: tuple[str, ...] = ()
    tried: tuple[str, ...] = ()
    k: int = Field(ge=1)

    @property
    def offered(self) -> tuple[str, ...]:
        """Offered skill ids, in palette order (sorted for an id-only palette)."""
        if self.palette.skills:
            return tuple(entry.rskill_id for entry in self.palette.skills)
        return tuple(sorted(self.palette.execute_rskill_ids))

    @property
    def max_score(self) -> float:
        """The best score; ``0.0`` means the query matched no skill (tie-breaks decide)."""
        return self.scores[0][1] if self.scores else 0.0


def _support_ids(
    entries: tuple[RSkillToolEntry, ...],
    rank: Callable[[str], tuple[float, bool, int, str]],
    slots: int,
    chosen: Iterable[str],
) -> list[str]:
    """Best skill (by ``rank``) per ``SUPPORT_ACTIONS`` action, up to ``slots`` new ids.

    Only skills whose actions are all support actions qualify. An action already
    covered by a ``chosen`` id (e.g. the look-at skill in flight) needs no slot,
    and neither consumes the budget nor blocks the other action.
    """
    taken = set(chosen)
    by_id = {e.rskill_id: e for e in entries}
    picked: list[str] = []
    for action in sorted(SUPPORT_ACTIONS, key=lambda a: a.value):
        if len(picked) >= slots:
            break
        if any(action in by_id[i].actions for i in taken if i in by_id):
            continue
        candidates = [
            e.rskill_id
            for e in entries
            if action in e.actions and set(e.actions) <= SUPPORT_ACTIONS
        ]
        best = min(candidates, key=rank) if candidates else None
        if best is not None and best not in picked and best not in taken:
            picked.append(best)
    return picked


def select_skill_subset(
    palette: ToolPalette,
    *,
    query: str,
    k: int,
    sticky: Iterable[str | None] = (),
    focus: str = "",
    recent: Iterable[str | None] = (),
    tried: Iterable[str] = (),
) -> SkillSubset:
    """Offer at most ``k`` of the palette's skills, ranked by relevance to ``query``.

    Slots fill in order, all inside ``k``: the ``sticky`` ids (the skill in flight)
    so the model can wait on it; then the best match for ``focus`` (the top pending
    operator prompt, which a long task text would otherwise outvote), if it matches
    any skill; then the ``recent`` ids (the last one dispatched for the active task)
    so the model can re-dispatch it; then the best skill of each ``SUPPORT_ACTIONS``
    action (navigate, look) the system prompt's workflow uses around any object
    task, never taking the last free slot (an action already covered by a kept id
    needs none); then the highest-scoring skills, so at least one ranked skill (the
    best untried substitute) is offered whenever a slot remains after the kept ids.
    Every pick, focus and support included, orders by score, then skills not in
    ``tried`` first, then ``PRECISION_RANK`` of the run dtype, then ``rskill_id``
    (``scores`` lists that order). With one skill in
    flight, any ``k`` >= 2 leaves the focus a slot, so an operator's ask is offered
    on the tick it arrives (the tick then drains it). Sticky / recent ids not in the
    palette (a skill dropped after a permanent failure) are ignored. The narrowed
    palette keeps the full palette's skill order so the rendered tool list is
    stable. A palette whose skills carry no metadata
    (only ``execute_rskill_ids``, the single-tool fallback) is capped the same way,
    scored on the ids. A palette with ``k`` or fewer skills is returned unchanged
    (the same object).

    Args:
        palette: The full capability-matched palette.
        query: Mission text: the active task plus any pending operator prompts.
        k: Maximum number of skill tools to offer; at least 1.
        sticky: Skill ids kept first, regardless of score (the skill in flight).
        focus: Text whose best-matching skill keeps the next slot (ignored when
            blank or when it matches no skill).
        recent: Skill ids kept after the focus, regardless of score (the active
            task's last dispatch).
        tried: Skill ids already dispatched for the active task; demoted among
            equal scores.

    Returns:
        The ``SkillSubset``.

    Raises:
        ValueError: When ``k`` < 1.

    Example:
        >>> from openral_core import RSkillAction
        >>> def entry(slug, objects):
        ...     return RSkillToolEntry(
        ...         rskill_id=f"OpenRAL/rskill-{slug}",
        ...         description=slug,
        ...         actions=(RSkillAction.PICK,),
        ...         objects=objects,
        ...     )
        >>> palette = ToolPalette(
        ...     skills=(entry("a", ("cube",)), entry("b", ("bowl",)), entry("c", ("orange",)))
        ... )
        >>> subset = select_skill_subset(palette, query="put the orange in the bowl", k=2)
        >>> subset.offered
        ('OpenRAL/rskill-b', 'OpenRAL/rskill-c')
        >>> sorted(subset.palette.execute_rskill_ids)
        ['OpenRAL/rskill-b', 'OpenRAL/rskill-c']
    """
    if k < 1:
        raise ValueError(f"select_skill_subset: k must be >= 1; got {k!r}")

    def scores_for(text: str) -> dict[str, float]:
        keywords = query_keywords(text)
        if palette.skills:
            return {entry.rskill_id: score_skill(entry, keywords) for entry in palette.skills}
        # Id-only palette (single ``execute_rskill`` tool fallback): the id is the text.
        return {
            rskill_id: _score(((_TEXT_WEIGHT, _field_forms((rskill_id,))),), keywords)
            for rskill_id in palette.execute_rskill_ids
        }

    raw = scores_for(query)
    tried_set = frozenset(tried)
    precision = {e.rskill_id: PRECISION_RANK[e.quantization_dtype] for e in palette.skills}

    def ranker(scores: dict[str, float]) -> Callable[[str], tuple[float, bool, int, str]]:
        """One ordering for every pick: score, then untried, precision, id."""
        return lambda i: (-scores[i], i in tried_set, precision.get(i, 0), i)

    rank = ranker(raw)
    scored = tuple((i, raw[i]) for i in sorted(raw, key=rank))

    def present(ids: Iterable[str | None]) -> list[str]:
        out: list[str] = []
        for rskill_id in ids:
            if rskill_id and rskill_id in raw and rskill_id not in out:
                out.append(rskill_id)
        return out

    kept_sticky = present(sticky)[:k]
    support: list[str] = []
    focused: tuple[str, ...] = ()
    if len(raw) <= k:
        narrowed = palette
        kept_sticky += [i for i in present(recent) if i not in kept_sticky]
    else:
        # In flight first: the model must be able to wait on the running skill.
        chosen = list(kept_sticky)
        focus_scores = scores_for(focus) if focus.strip() else {}
        if focus_scores and len(chosen) < k:
            best_focus = min(focus_scores, key=ranker(focus_scores))
            if focus_scores[best_focus] > 0 and best_focus not in chosen:
                chosen.append(best_focus)
                focused = (best_focus,)
        for rskill_id in present(recent):
            if len(chosen) < k and rskill_id not in chosen:
                chosen.append(rskill_id)
                kept_sticky.append(rskill_id)
        # Support slots never take the last free slot: at least one ranked skill
        # (the best untried substitute) is always offered while room remains.
        support_slots = max(0, k - len(chosen) - 1)
        support = _support_ids(palette.skills, rank, support_slots, chosen)
        chosen += support
        for rskill_id, _score_value in scored:
            if len(chosen) >= k:
                break
            if rskill_id not in chosen:
                chosen.append(rskill_id)
        keep = set(chosen)
        skills = tuple(entry for entry in palette.skills if entry.rskill_id in keep)
        narrowed = palette.model_copy(
            update={"skills": skills, "execute_rskill_ids": frozenset(keep)}
        )
    return SkillSubset(
        palette=narrowed,
        scores=scored,
        query=query,
        sticky=tuple(kept_sticky),
        focused=focused,
        support=tuple(support),
        tried=tuple(sorted(tried_set & raw.keys())),
        k=k,
    )
