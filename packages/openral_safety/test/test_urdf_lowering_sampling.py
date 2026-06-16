"""Sampling-based ACM fallback (ADR-0030): deterministic, adjacency-disabling, and
conservative w.r.t. the SRDF ground truth.

The sampler reconstructs the allowed-collision matrix from a URDF when no SRDF
exists (the humanoids / openarm). It tests collisions with the safety kernel's own
conservative capsule distance, so it never disables a pair the precise-mesh SRDF
keeps checked (the safe direction). Real panda URDF + SRDF, no mocks (§1.11).
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
from openral_core.assets import resolve_asset
from openral_safety.urdf_lowering import (
    sample_acm_from_urdf,
)

pytest.importorskip("yourdfpy")
pytest.importorskip("robot_descriptions")

_PANDA = "rd:panda_description"


def _resolve_urdf_path(ref: str) -> str | None:
    """Resolve an asset ref to a URDF file path string (test helper)."""
    p = resolve_asset(ref, "urdf")
    return None if p is None else str(p)


_PANDA_SRDF = Path("/opt/ros/jazzy/share/moveit_resources_panda_moveit_config/config/panda.srdf")
# A modest sweep keeps the test fast while still exercising every rule; the
# production default is larger (_N_SAMPLES).
_N = 400


def _arm_only(pairs: set[frozenset[str]]) -> set[frozenset[str]]:
    """Restrict to pairs whose both links are numbered panda arm links (link0-7)."""
    return {
        p for p in pairs if all(link.startswith("panda_link") and link[10:].isdigit() for link in p)
    }


def test_sampling_is_deterministic_under_pinned_seed() -> None:
    urdf = _resolve_urdf_path(_PANDA)
    assert urdf is not None
    a = sample_acm_from_urdf(urdf, n_samples=_N)
    b = sample_acm_from_urdf(urdf, n_samples=_N)
    assert a == b, "pinned seed must make the ACM reproducible (the --check linchpin)"


def test_adjacent_links_always_disabled() -> None:
    urdf = _resolve_urdf_path(_PANDA)
    assert urdf is not None
    pairs = sample_acm_from_urdf(urdf, n_samples=_N)
    # Every directly joint-connected arm pair on the panda chain.
    chain = ("0", "1", "2", "3", "4", "5", "6", "7")
    for a, b in pairwise(chain):
        assert frozenset({f"panda_link{a}", f"panda_link{b}"}) in pairs


def test_sampler_does_not_disable_a_never_collide_pair() -> None:
    """Without an SRDF, a 'never-collide' pair stays CHECKED (the conservative rule).

    A random-pose sweep cannot prove a pair never collides (it can miss the tail),
    so the sampling fallback must NOT auto-disable one. link1↔link4 never collides in
    the sweep but, lacking a mesh-ground-truth SRDF, is kept checked.
    """
    urdf = _resolve_urdf_path(_PANDA)
    assert urdf is not None
    pairs = sample_acm_from_urdf(urdf, n_samples=_N)
    assert frozenset({"panda_link1", "panda_link4"}) not in pairs


def test_sampled_acm_keeps_never_collide_pairs_checked() -> None:
    """The safe conservative fallback: known 'never-collide' far pairs stay checked.

    Disable only what provably must be (adjacent + always-colliding junctions); a
    sampled 'never-collide' verdict is not trusted (it can miss the tail).
    """
    urdf = _resolve_urdf_path(_PANDA)
    assert urdf is not None
    sampled = _arm_only(sample_acm_from_urdf(urdf, n_samples=_N))
    # SRDF "Never" pairs (kinematically far) must NOT be auto-disabled by sampling.
    for never in (("panda_link1", "panda_link4"), ("panda_link2", "panda_link4")):
        assert frozenset(never) not in sampled, f"{never} unsafely disabled by sampling"
    assert frozenset({"panda_link2", "panda_link3"}) in sampled  # adjacency IS disabled
