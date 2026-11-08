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

from pydantic import BaseModel, ConfigDict, Field

from openral_reasoner.palette import RSkillToolEntry, ToolPalette

__all__ = [
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
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# Shortest tokens the plural fold touches: "ies" words longer than "ties", and
# "s" words longer than "bus"/"gas" stand-ins, so short words stay intact.
_MIN_IES_PLURAL_LEN = 5
_MIN_S_PLURAL_LEN = 4


def _normalize(token: str) -> str:
    """Fold a simple English plural onto its singular (``bowls`` → ``bowl``)."""
    if len(token) >= _MIN_IES_PLURAL_LEN and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) >= _MIN_S_PLURAL_LEN and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def mission_keywords(text: str) -> frozenset[str]:
    """Lower-cased, de-pluralized content words of ``text``.

    Example:
        >>> sorted(mission_keywords("Pick up the black bowls and place them on the plate"))
        ['black', 'bowl', 'pick', 'place', 'plate']
    """
    return frozenset(_normalize(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS)


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


def _keywords_of(values: Iterable[str]) -> frozenset[str]:
    """Union of ``mission_keywords`` over several strings."""
    out: set[str] = set()
    for value in values:
        out |= mission_keywords(value)
    return frozenset(out)


def score_skill(entry: RSkillToolEntry, keywords: frozenset[str]) -> float:
    """Relevance of one skill to the mission keywords.

    Each keyword counts once, at the weight of the strongest field it matches
    (objects / actions 3, scenes / description / id 1).

    Example:
        >>> from openral_core import RSkillAction
        >>> entry = RSkillToolEntry(
        ...     rskill_id="OpenRAL/rskill-act-franka_panda-libero-fp32",
        ...     description="ACT on LIBERO tabletop tasks.",
        ...     actions=(RSkillAction.PICK, RSkillAction.PLACE),
        ...     objects=("bowl",),
        ... )
        >>> score_skill(entry, mission_keywords("pick the bowl"))
        6.0
        >>> score_skill(entry, mission_keywords("open the fridge"))
        0.0
    """
    fields = (
        (_OBJECT_WEIGHT, _keywords_of(entry.objects)),
        (_ACTION_WEIGHT, _keywords_of(a.value for a in entry.actions)),
        (_SCENE_WEIGHT, _keywords_of(entry.scenes)),
        (_TEXT_WEIGHT, mission_keywords(entry.description) | mission_keywords(entry.rskill_id)),
    )
    score = 0.0
    for word in keywords:
        score += max((w for w, field in fields if word in field), default=0.0)
    return score


class SkillSubset(BaseModel):
    """The skills offered on a tick and why (logged and stamped on the tick span).

    Attributes:
        palette: The narrowed palette the LLM sees and decode validates against.
        offered: Offered skill ids, in palette order.
        scores: ``(rskill_id, score)`` for every skill in the full palette,
            best first, so a skill left out can be diagnosed from the trace.
        query: The mission text the skills were ranked against.
        sticky: Ids kept regardless of score (in flight / last dispatched).
        k: The cap applied.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    palette: ToolPalette
    offered: tuple[str, ...]
    scores: tuple[tuple[str, float], ...]
    query: str
    sticky: tuple[str, ...] = ()
    k: int = Field(ge=1)


def select_skill_subset(
    palette: ToolPalette,
    *,
    query: str,
    k: int,
    sticky: Iterable[str | None] = (),
) -> SkillSubset:
    """Offer at most ``k`` of the palette's skills, ranked by relevance to ``query``.

    ``sticky`` ids (the skill in flight, the last one dispatched for the active
    task) are kept first so the model can wait on or re-dispatch them; ids not
    in the palette (a skill dropped after a permanent failure) are ignored. The
    remaining slots go to the highest-scoring skills, ties broken by
    ``rskill_id``. The narrowed palette keeps the full palette's skill order so
    the rendered tool list is stable. A palette with ``k`` or fewer skills is
    returned unchanged (the same object).

    Args:
        palette: The full capability-matched palette.
        query: Mission text: the active task, else the pending operator goal.
        k: Maximum number of skill tools to offer; at least 1.
        sticky: Skill ids to keep regardless of score, in priority order.

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
    keywords = query_keywords(query)
    scored = sorted(
        ((entry.rskill_id, score_skill(entry, keywords)) for entry in palette.skills),
        key=lambda item: (-item[1], item[0]),
    )
    in_palette = {entry.rskill_id for entry in palette.skills}
    kept_sticky: list[str] = []
    for rskill_id in sticky:
        if rskill_id and rskill_id in in_palette and rskill_id not in kept_sticky:
            kept_sticky.append(rskill_id)
    kept_sticky = kept_sticky[:k]
    if len(palette.skills) <= k:
        narrowed = palette
    else:
        chosen = set(kept_sticky)
        for rskill_id, _score in scored:
            if len(chosen) >= k:
                break
            chosen.add(rskill_id)
        skills = tuple(entry for entry in palette.skills if entry.rskill_id in chosen)
        narrowed = palette.model_copy(
            update={
                "skills": skills,
                "execute_rskill_ids": frozenset(entry.rskill_id for entry in skills),
            }
        )
    return SkillSubset(
        palette=narrowed,
        offered=tuple(entry.rskill_id for entry in narrowed.skills),
        scores=tuple(scored),
        query=query,
        sticky=tuple(kept_sticky),
        k=k,
    )
