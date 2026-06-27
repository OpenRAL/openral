"""Unit slice for the collective-target detector (ADR-0075).

A skill acts on exactly ONE specific object, so the reasoner refuses to actuate
while the active task targets a collective/quantified set ("put ALL the objects
in the basket") and self-prompts the LLM to enumerate (scene_objects is already
in its context) + decompose into one subtask per object. ``is_collective_target``
is the single source of truth (``openral_core``) shared by the ``GroundedSubtask``
schema validator and the reasoner node's runtime execute gate. Pure helper — no
ROS — so this pins it directly; the full block→invite→decompose flow is exercised
by the deploy-sim run.
"""

from __future__ import annotations

import pytest
from openral_core import is_collective_target as _is_collective_target


@pytest.mark.parametrize(
    "text",
    [
        "put all the objects in the basket",
        "Put all the objects on the table into the basket.",
        "clean everything off the table",
        "stack all the bowls",
        "pick up each cup",
        "put both plates away",
        "tidy the objects",
        "move the items to the shelf",
    ],
)
def test_collective_targets_are_flagged(text: str) -> None:
    assert _is_collective_target(text)


@pytest.mark.parametrize(
    "text",
    [
        "pick up the milk and put it in the basket",
        "pick up the alphabet soup and put it in the basket",
        "pick up the ketchup",
        "place the red cube on the plate",
        "open the top drawer",
        "navigate to the kitchen",
    ],
)
def test_specific_targets_pass(text: str) -> None:
    assert not _is_collective_target(text)
