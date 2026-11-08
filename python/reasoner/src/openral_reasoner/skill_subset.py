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
replayable from the trace (CLAUDE.md §1.8). Ties break on ``rskill_id``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from openral_core import RSkillAction
from pydantic import BaseModel, ConfigDict, Field

from openral_reasoner.palette import RSkillToolEntry, ToolPalette

__all__ = [
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
        forms.update({word[:-2], word[:-1]})
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

    Example:
        >>> sorted(query_keywords("put the apple in the basket"))
        ['apple', 'basket', 'place', 'put']
    """
    words = mission_keywords(text)
    return words | {_VERB_SYNONYMS[w] for w in words if w in _VERB_SYNONYMS}


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
        k: The cap applied.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    palette: ToolPalette
    scores: tuple[tuple[str, float], ...]
    query: str
    sticky: tuple[str, ...] = ()
    focused: tuple[str, ...] = ()
    support: tuple[str, ...] = ()
    k: int = Field(ge=1)

    @property
    def offered(self) -> tuple[str, ...]:
        """Offered skill ids, in palette order (sorted for an id-only palette)."""
        if self.palette.skills:
            return tuple(entry.rskill_id for entry in self.palette.skills)
        return tuple(sorted(self.palette.execute_rskill_ids))

    @property
    def max_score(self) -> float:
        """The best score; ``0.0`` means the query matched no skill (alphabetical pick)."""
        return self.scores[0][1] if self.scores else 0.0


def _support_ids(
    entries: tuple[RSkillToolEntry, ...], scores: dict[str, float], slots: int
) -> list[str]:
    """Best-scoring skill per ``SUPPORT_ACTIONS`` action (only-support skills), up to ``slots``."""
    picked: list[str] = []
    for action in sorted(SUPPORT_ACTIONS, key=lambda a: a.value):
        candidates = [
            e.rskill_id
            for e in entries
            if action in e.actions and set(e.actions) <= SUPPORT_ACTIONS
        ]
        if candidates and len(picked) < slots:
            best = min(candidates, key=lambda i: (-scores[i], i))
            if best not in picked:
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
) -> SkillSubset:
    """Offer at most ``k`` of the palette's skills, ranked by relevance to ``query``.

    Slots fill in order, all inside ``k``: the ``sticky`` ids (the skill in flight)
    so the model can wait on it; then the best match for ``focus`` (the top pending
    operator prompt, which a long task text would otherwise outvote), if it matches
    any skill; then the ``recent`` ids (the last one dispatched for the active task)
    so the model can re-dispatch it; then the best skill of each ``SUPPORT_ACTIONS`` action
    (navigate, look) the system prompt's workflow uses around any object task, at
    most ``k - 1`` of them; then the highest-scoring skills, ties broken by
    ``rskill_id``. With one skill in flight, any ``k`` >= 2 leaves the focus a slot,
    so an operator's ask is offered on the tick it arrives (the tick then drains
    it). Sticky / recent ids not in the palette (a skill dropped after a permanent
    failure) are ignored. The narrowed palette keeps the full palette's skill order
    so the rendered tool list is stable. A palette whose skills carry no metadata
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
    scored = tuple(sorted(raw.items(), key=lambda item: (-item[1], item[0])))

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
            best_focus = min(focus_scores, key=lambda i: (-focus_scores[i], i))
            if focus_scores[best_focus] > 0 and best_focus not in chosen:
                chosen.append(best_focus)
                focused = (best_focus,)
        for rskill_id in present(recent):
            if len(chosen) < k and rskill_id not in chosen:
                chosen.append(rskill_id)
                kept_sticky.append(rskill_id)
        free = max(0, min(k - 1, k - len(chosen)))
        support = [i for i in _support_ids(palette.skills, raw, free) if i not in chosen]
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
        k=k,
    )
