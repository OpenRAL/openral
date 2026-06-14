"""ADR-0044 Phase 1 — shared gaze geometry property + degenerate-case tests.

The independent oracle is ``homogeneous_from_quat_xyz`` (the existing
quat→matrix helper from object_lift): rotate the claimed view axis by the
returned quaternion and check it points from eye to target. This catches
convention mistakes a fixture table would bake in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openral_world_state.geometry import ViewAxis, compute_gaze_pose, look_at_quat_wxyz
from openral_world_state.object_lift import homogeneous_from_quat_xyz

_VIEW_VECTORS: dict[ViewAxis, tuple[float, float, float]] = {
    "-z": (0.0, 0.0, -1.0),
    "+z": (0.0, 0.0, 1.0),
    "+x": (1.0, 0.0, 0.0),
}


def _rotate_wxyz(
    quat_wxyz: tuple[float, float, float, float], vec: tuple[float, float, float]
) -> np.ndarray:
    w, x, y, z = quat_wxyz
    m = homogeneous_from_quat_xyz((0.0, 0.0, 0.0), (x, y, z, w))
    return np.asarray(m[:3, :3] @ np.asarray(vec, dtype=np.float64))


_coord = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_point = st.tuples(_coord, _coord, _coord)


@given(eye=_point, target=_point, view_axis=st.sampled_from(["-z", "+z", "+x"]))
def test_view_axis_points_at_target(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    view_axis: ViewAxis,
) -> None:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:  # degenerate inputs are covered by the fallback tests
        return
    quat = look_at_quat_wxyz(eye, target, view_axis=view_axis)
    assert math.isclose(sum(c * c for c in quat), 1.0, abs_tol=1e-9), "non-unit quaternion"
    pointed = _rotate_wxyz(quat, _VIEW_VECTORS[view_axis])
    assert float(np.dot(pointed, forward / norm)) == pytest.approx(1.0, abs=1e-6)


@given(eye=_point, target=_point)
def test_minus_z_camera_keeps_image_up(
    eye: tuple[float, float, float], target: tuple[float, float, float]
) -> None:
    """MuJoCo convention: the camera's +Y (image up) never points below world up."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6 or abs(forward[2] / norm) > 0.99:  # up-fallback cases exempt
        return
    quat = look_at_quat_wxyz(eye, target, view_axis="-z")
    image_up = _rotate_wxyz(quat, (0.0, 1.0, 0.0))
    assert float(image_up[2]) > 0.0


def test_degenerate_same_point_fallbacks() -> None:
    # -z preserves the sim composers' documented straight-down flip; the other
    # conventions return identity.
    assert look_at_quat_wxyz((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == (0.0, 1.0, 0.0, 0.0)
    assert look_at_quat_wxyz((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), view_axis="+z") == (
        1.0,
        0.0,
        0.0,
        0.0,
    )


def test_straight_down_gaze_uses_alternate_up() -> None:
    # Looking exactly along -up: would be a zero cross product without the
    # alternate-up guard. The view axis must still hit the target.
    quat = look_at_quat_wxyz((0.0, 0.0, 2.0), (0.0, 0.0, 0.0), view_axis="+z")
    pointed = _rotate_wxyz(quat, (0.0, 0.0, 1.0))
    assert float(pointed[2]) == pytest.approx(-1.0, abs=1e-9)


def test_compute_gaze_pose_carries_frame_and_position() -> None:
    pose = compute_gaze_pose((0.5, -0.2, 1.1), (1.5, -0.2, 0.1), frame_id="map")
    assert pose.frame_id == "map"
    assert pose.xyz == (0.5, -0.2, 1.1)
    # Pose6D is xyzw; its +Z (optical forward) must point at the target.
    x, y, z, w = pose.quat_xyzw
    pointed = _rotate_wxyz((w, x, y, z), (0.0, 0.0, 1.0))
    expected = np.asarray((1.0, 0.0, -1.0)) / math.sqrt(2.0)
    assert np.allclose(pointed, expected, atol=1e-9)


def test_matches_sim_composer_behaviour() -> None:
    """Regression pin: outputs captured from the pre-refactor so101_box composer.

    The three sim composers' ``_look_at_quat`` now alias this helper, so the pin
    uses literal values captured from the original implementation (2026-06-10,
    pre ADR-0044 refactor) — not a re-import that would be tautological.
    """
    cases = [
        (
            (0.2, 0.0, 0.95),
            (0.65, 0.0, 0.05),
            (0.6881909602355868, 0.16245984811645317, -0.16245984811645317, -0.6881909602355867),
        ),
        (
            (1.0, -1.0, 2.0),
            (0.0, 0.5, 0.0),
            (0.8934292207829864, 0.3432337026284389, 0.10392280320443939, 0.2705086020909677),
        ),
        (
            (-0.3, 0.4, 1.2),
            (0.2, 0.1, 0.3),
            (-0.4724843338167137, -0.13968062350953692, 0.24670257337094362, 0.8344972846006855),
        ),
    ]
    for eye, target, expected in cases:
        assert look_at_quat_wxyz(eye, target) == pytest.approx(expected, abs=1e-12)
