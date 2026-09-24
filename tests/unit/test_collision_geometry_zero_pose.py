"""No non-allowed capsule pair may interpenetrate at a robot's zero configuration.

The C++ safety kernel evaluates every candidate chunk against the manifest's
``collision_geometry`` capsules, skipping only ``allowed_collision_pairs``. A
pair that already overlaps with every joint at zero is not a hazard the kernel
can act on: it is a modelling error that makes the kernel refuse *every* chunk
near the rest pose. That is exactly what stopped the first OpenArm policy
dispatch on qorin1 (2026-09-22): ``openarm_*_link3``'s hand-authored capsule
was 0.22 m long from a joint 0.154 m above the elbow, so it reached 6.6 cm past
joint 4 into link 5's capsule — a constant ``-0.04575 m`` at every elbow angle,
on both arms, on the twin and on the real cell.

Real fixtures only: every ``robots/*/robot.yaml`` that declares capsules is
checked with a small forward-kinematics pass over its own ``joints`` and
``fixed_attachments`` — the same inputs the kernel lowers — and the standard
segment-segment distance. Sphere entries and entries whose ``link_name`` is not
a link of the kinematic tree (composite finger frames) are skipped, since the
kernel resolves those through its own link table.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFESTS = sorted(_REPO_ROOT.glob("robots/*/robot.yaml"))


def _rot(rpy: list[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _tf(xyz: list[float], rpy: list[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rot(rpy)
    m[:3, 3] = xyz
    return m


def _link_poses_at_zero(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    """World pose of every link at q = 0, walking joints from their parent links."""
    joints = manifest.get("joints", [])
    attachments = manifest.get("fixed_attachments", [])
    base = manifest.get("base_frame", "base_link")
    poses: dict[str, np.ndarray] = {base: np.eye(4)}
    zero3 = [0.0, 0.0, 0.0]
    pending = [
        (e["parent_link"], e["child_link"], e.get("origin_xyz", zero3), e.get("origin_rpy", zero3))
        for e in (*attachments, *joints)
    ]
    # Fixed-point pass: parents may be declared after children.
    progressed = True
    while pending and progressed:
        progressed = False
        for entry in list(pending):
            parent, child, xyz, rpy = entry
            if parent in poses:
                poses[child] = poses[parent] @ _tf(list(xyz), list(rpy))
                pending.remove(entry)
                progressed = True
    if pending:
        # Links that do not chain back to ``base_frame`` would have to be
        # placed by guessing a root, and a guessed root superimposes subtrees
        # at the origin — false overlaps. Refuse rather than guess.
        raise LookupError(sorted({child for _p, child, _x, _r in pending}))
    return poses


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Closest distance between segments p1-q1 and p2-q2 (Ericson, RTCD 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    eps = 1e-12
    if a < eps and e < eps:
        return float(np.linalg.norm(r))
    if a < eps:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e < eps:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = float(np.clip((b * f - c * e) / den, 0.0, 1.0)) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def _capsules_at_zero(manifest: dict[str, Any]) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """Every capsule as ``(link_name, a, b, radius)`` — a link may declare several."""
    poses = _link_poses_at_zero(manifest)
    out: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for entry in manifest.get("collision_geometry", []):
        shape = entry["shape"]
        if shape.get("shape") != "capsule" or entry["link_name"] not in poses:
            continue
        o = entry["origin_xyz_rpy"]
        c = poses[entry["link_name"]] @ _tf(o[:3], o[3:])
        half = shape["length_m"] / 2.0
        a = (c @ np.array([0.0, 0.0, -half, 1.0]))[:3]
        b = (c @ np.array([0.0, 0.0, half, 1.0]))[:3]
        out.append((entry["link_name"], a, b, float(shape["radius_m"])))
    return out


# Manifests whose hand-authored capsules this check finds overlapping at zero
# but which have NOT been confirmed against the C++ kernel the way OpenArm's
# pair was (the kernel's own verdict, -0.04575 m, matched this FK to 4 digits).
# Strict xfails so a fix flips them green and a regression is loud; whether
# each is a real kernel refusal at rest or a convention this FK does not
# model is the safety WG's to adjudicate before anyone edits those fixtures.
_UNADJUDICATED: dict[str, str] = {
    "franka_panda": "panda_hand/panda_link5 -0.108 m, panda_link5/panda_link7 -0.093 m",
    "g1": "several torso_link pairs, -0.037 .. -0.074 m",
    "h1": "shoulder_yaw_link/torso_link -0.0007 m both sides",
    "so100_follower": "five base/arm pairs, -0.040 .. -0.089 m",
}


@pytest.mark.parametrize(
    "manifest_path",
    [
        pytest.param(
            p,
            id=p.parent.name,
            marks=(
                pytest.mark.xfail(strict=True, reason=_UNADJUDICATED[p.parent.name])
                if p.parent.name in _UNADJUDICATED
                else ()
            ),
        )
        for p in _MANIFESTS
    ],
)
def test_no_non_allowed_capsule_pair_interpenetrates_at_zero(manifest_path: Path) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("collision_geometry"):
        pytest.skip("manifest declares no collision_geometry")
    allowed = {frozenset(p) for p in manifest.get("allowed_collision_pairs", [])}
    try:
        capsules = _capsules_at_zero(manifest)
    except LookupError as unrooted:
        pytest.skip(f"links not rooted at base_frame, FK unvalidated here: {unrooted.args[0]}")
    overlaps = []
    # The kernel keeps every capsule and checks pairs on different links, so
    # a link with two capsules contributes both; same-link pairs are skipped.
    for (a, p1, q1, r1), (b, p2, q2, r2) in itertools.combinations(
        sorted(capsules, key=lambda c: c[0]), 2
    ):
        if a == b or frozenset((a, b)) in allowed:
            continue
        surface = _segment_distance(p1, q1, p2, q2) - r1 - r2
        if surface < 0.0:
            overlaps.append((a, b, round(surface, 4)))
    assert not overlaps, (
        f"{manifest_path.parent.name}: capsule pairs interpenetrating at the zero "
        f"configuration, which the kernel would refuse on every chunk: {overlaps}"
    )
