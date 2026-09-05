"""The silhouette probe counts placed payloads per object, not per primitive.

`tools/_nav2_costmap_silhouette_probe.py` decides whether the payload half of
issue #108 was measured at all by comparing the number of attached objects it
could place against the number declared. It used to increment the placed count
once per successfully-placed *primitive* while comparing against a count of
*objects*, so any payload with more than one primitive scored `placed >
declared`, every sample was filed as a partial placement, and
`payload_silhouette_measured` stayed `false` however completely the payload had
been placed. The guard was unsatisfiable rather than unsatisfied.

That is not a hypothetical shape. `AttachedCollisionPrimitive.msg` says in its
own header that one object "may carry several of these primitives", the sim
producer builds them with `extract_body_primitives` over a whole MuJoCo body
subtree, and a live `robocasa_baguette` carry measured **16** primitives on the
single `sim:obj_main` payload. Every scene run this repo has recorded reported
the payload half unmeasured, and this is why.

Real components (CLAUDE.md §1.11): real `openral_msgs` IDL messages and the
probe's own `payload_mask`. The only thing supplied by the test is the
`base_frame <- attach_link` transform, which is a plain argument to that
function rather than a stand-in for a dependency — a real 4x4 identity.

Lives under `tests/unit/` because it needs no graph, but it is registered in
`scripts/ros_live_tests.sh` because importing the probe needs the colcon
overlay — the same reason `tests/unit/test_safety_status_msg.py` is listed
there.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("rclpy", reason="the probe imports rclpy; needs the ROS overlay")
pytest.importorskip("openral_msgs", reason="needs the colcon-generated openral_msgs")
pytest.importorskip(
    "openral_nav2_bringup", reason="the probe imports the scan filter's geometry helpers"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "tools") not in sys.path:  # tools/ is not an importable package
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

#: The measured primitive count of a live `robocasa_baguette` carry, recorded
#: off `/openral/attachment_state`. Any value above 1 reproduces the defect;
#: this one is used so the test fails the way the scene did.
_BAGUETTE_PRIMITIVES = 16

_BOX_HALF_EXTENT_M = 0.05


def _identity() -> Any:
    import numpy as np

    return np.eye(4)


def _grid() -> Any:
    """A small patch of costmap cell centres straddling the payload."""
    import numpy as np

    xs = np.linspace(-0.2, 0.2, 21)
    ys = np.linspace(-0.2, 0.2, 21)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _payload(*, n_primitives: int, malformed: bool = False) -> Any:
    """One attached object carrying ``n_primitives`` boxes at the attach link."""
    from openral_msgs.msg import AttachedCollisionObject, AttachedCollisionPrimitive

    obj = AttachedCollisionObject()
    obj.object_id = "sim:obj_main"
    obj.attach_link = "panda_link7"
    obj.pose_in_link.orientation.w = 1.0
    for _ in range(n_primitives):
        prim = AttachedCollisionPrimitive()
        prim.shape_type = AttachedCollisionPrimitive.SHAPE_BOX
        # A box takes three half-extents; two is the producer shipping a
        # malformed primitive, which must fail the whole object rather than be
        # skipped quietly.
        prim.shape_dimensions = (
            [_BOX_HALF_EXTENT_M, _BOX_HALF_EXTENT_M]
            if malformed
            else [_BOX_HALF_EXTENT_M, _BOX_HALF_EXTENT_M, _BOX_HALF_EXTENT_M]
        )
        prim.pose_in_object.orientation.w = 1.0
        obj.primitives.append(prim)
    return obj


def test_a_multi_primitive_payload_counts_as_one_placed_object() -> None:
    """The regression: 16 primitives on one object must count as one, not 16.

    With the old per-primitive count this returned 16, the caller's
    ``placed == declared`` test failed against a declared count of 1, and the
    sample was recorded as a partial placement — which is what kept
    ``payload_silhouette_measured`` false on every scene run to date.
    """
    from _nav2_costmap_silhouette_probe import payload_mask

    mask, placed, z_span = payload_mask(
        [_payload(n_primitives=_BAGUETTE_PRIMITIVES)], lambda _link: _identity(), _grid()
    )

    assert placed == 1, f"a single object with 16 primitives counted as {placed} placed objects"
    assert bool(mask.any()), "the payload projected no cells at all, so nothing was measured"
    assert z_span is not None
    lo, hi = z_span
    assert lo < hi, f"the payload z span is degenerate: {z_span}"


def test_an_object_with_no_primitives_is_never_placed() -> None:
    """It contributes no silhouette, which is the failure the guard exists for.

    Counting it as placed would let a payload the producer failed to describe
    read as a clean measurement, and its cells would be scored as obstacles
    "elsewhere on the map".
    """
    from _nav2_costmap_silhouette_probe import payload_mask

    mask, placed, _ = payload_mask([_payload(n_primitives=0)], lambda _link: _identity(), _grid())

    assert placed == 0
    assert not bool(mask.any())


def test_one_malformed_primitive_fails_the_whole_object() -> None:
    """A partial projection must not be reported as a placement.

    The other primitives still project — the mask is deliberately left as wide
    as it got, because under-covering the silhouette is the unsafe direction —
    but the object is not counted, so the run reports a partial placement rather
    than a verdict.
    """
    from _nav2_costmap_silhouette_probe import payload_mask
    from openral_msgs.msg import AttachedCollisionPrimitive

    obj = _payload(n_primitives=3)
    obj.primitives[1].shape_dimensions = [_BOX_HALF_EXTENT_M, _BOX_HALF_EXTENT_M]
    assert obj.primitives[1].shape_type == AttachedCollisionPrimitive.SHAPE_BOX

    _mask, placed, _ = payload_mask([obj], lambda _link: _identity(), _grid())

    assert placed == 0, "an object with a malformed primitive must not count as placed"


def test_an_unplaceable_attach_link_is_not_counted() -> None:
    """No TF for the attach link means the object was never placed.

    This is the case the per-object count must keep catching: it is what a
    missing ``base_frame <- attach_link`` transform looks like, and it silently
    narrows the silhouette.
    """
    from _nav2_costmap_silhouette_probe import payload_mask

    mask, placed, _ = payload_mask(
        [_payload(n_primitives=_BAGUETTE_PRIMITIVES)], lambda _link: None, _grid()
    )

    assert placed == 0
    assert not bool(mask.any())
