"""The tracker node's attachment revision: what bumps it, and what only republishes."""

from __future__ import annotations

from collections.abc import Callable

import pytest


def _step() -> Callable[..., tuple[bool, bool]]:
    step: Callable[..., tuple[bool, bool]] = pytest.importorskip(
        "openral_world_state_ros.object_tracker_node",
        reason="the tracker node's package (packages/world_state) is not on the path",
    ).attachment_publish_step
    return step


def test_a_place_zone_moving_with_each_grid_republishes_under_one_revision() -> None:
    """AMR aisle, ``tracking``, 2026-11-20: a zone place declaration was re-expressed on every
    grid and each re-expression bumped the attachment revision. The kernel reads a revision
    edge with no payload as a detach and retired the armed grasp region on it (65-70 times a
    run, before contact). A moving zone republishes under the same revision."""
    step = _step()
    revision = 0
    published: list[int] = []
    last_view: tuple[int, int, int] | None = None
    for grid_stamp in (10, 20, 30, 40):  # the base settles; the zone moves in the grid frame
        view = (101, 202, grid_stamp)  # (id(grasp decl), id(place decl), zone stamp)
        bump, publish = step(0, 0 if published else None, view, last_view, heartbeat=False)
        last_view = view
        revision += bump
        if publish:
            published.append(revision)
    assert published == [1, 1, 1, 1], "every grid republished, one revision"


def test_a_new_declaration_republishes_under_the_same_revision() -> None:
    """zr2 (q5090, 2026-11-20): the place declaration arrived 2 ms after the grasp region armed;
    with the declaration in the revision that edge (no payload yet) retired the region."""
    step = _step()
    assert step(0, 0, (101, 303, 10), (101, 202, 10), heartbeat=False) == (False, True)
    assert step(0, 0, (909, 202, 10), (101, 202, 10), heartbeat=False) == (False, True)


def test_only_the_attachment_set_and_a_station_edge_bump_the_revision() -> None:
    step = _step()
    view = (101, 202, 10)
    assert step(1, 0, view, view, heartbeat=False) == (True, True), "a payload binds"
    assert step(0, None, view, None, heartbeat=False) == (True, True), "a station edge"
    assert step(1, 1, view, view, heartbeat=False) == (False, False), "nothing changed"
    assert step(1, 1, view, view, heartbeat=True) == (False, True), "heartbeat"
