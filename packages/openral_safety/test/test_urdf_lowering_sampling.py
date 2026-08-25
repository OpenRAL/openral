"""No-SRDF ACM fallback (part of the geometric safety collision-checking system):
deterministic, adjacency-disabling, and conservative w.r.t. the SRDF ground truth.

Reconstructs the allowed-collision matrix from a URDF alone, for robots with no
SRDF (the humanoids / openarm). It tests collisions with the safety kernel's own
predicates (`openral_safety.kernel_predicates`), and exempts a pair only when it
is adjacent or *provably* always-colliding — so it never disables a pair the
precise-mesh SRDF keeps checked (the safe direction). Real panda URDF + SRDF, no
mocks (§1.11).
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


def _arm_only(pairs: set[frozenset[str]]) -> set[frozenset[str]]:
    """Restrict to pairs whose both links are numbered panda arm links (link0-7)."""
    return {
        p for p in pairs if all(link.startswith("panda_link") and link[10:].isdigit() for link in p)
    }


def test_sampling_is_deterministic() -> None:
    """Reproducible across runs — the `--check` linchpin.

    There is no seed to pin any more: the always-colliding verdict is a proof over
    each pair's relative-DoF subspace, not a draw from an RNG. A criterion whose
    output moved with the draw order could not be reproduced after any change to
    the joint set, which is how the box under-approximation (issue #155) stayed
    invisible for so long.
    """
    urdf = _resolve_urdf_path(_PANDA)
    assert urdf is not None
    assert sample_acm_from_urdf(urdf) == sample_acm_from_urdf(urdf)


def test_adjacent_links_always_disabled() -> None:
    urdf = _resolve_urdf_path(_PANDA)
    assert urdf is not None
    pairs = sample_acm_from_urdf(urdf)
    # Every directly joint-connected arm pair on the panda chain.
    chain = ("0", "1", "2", "3", "4", "5", "6", "7")
    for a, b in pairwise(chain):
        assert frozenset({f"panda_link{a}", f"panda_link{b}"}) in pairs


def test_sampler_does_not_disable_a_never_collide_pair() -> None:
    """Without an SRDF, a 'never-collide' pair stays CHECKED (the conservative rule).

    Geometry can prove a pair *always* collides; it cannot prove one *never* does
    without mesh ground truth. So link1↔link4 — which the Franka SRDF marks
    "Never" — must stay checked on this path rather than be auto-disabled.
    """
    urdf = _resolve_urdf_path(_PANDA)
    assert urdf is not None
    pairs = sample_acm_from_urdf(urdf)
    assert frozenset({"panda_link1", "panda_link4"}) not in pairs


def test_sampled_acm_keeps_never_collide_pairs_checked() -> None:
    """The safe conservative fallback: known 'never-collide' far pairs stay checked.

    Exempt only what provably must be (adjacent + certified always-colliding); a
    geometric 'never-collide' verdict is not trusted without mesh ground truth.
    """
    urdf = _resolve_urdf_path(_PANDA)
    assert urdf is not None
    sampled = _arm_only(sample_acm_from_urdf(urdf))
    # SRDF "Never" pairs (kinematically far) must NOT be auto-disabled by sampling.
    for never in (("panda_link1", "panda_link4"), ("panda_link2", "panda_link4")):
        assert frozenset(never) not in sampled, f"{never} unsafely disabled by sampling"
    assert frozenset({"panda_link2", "panda_link3"}) in sampled  # adjacency IS disabled
