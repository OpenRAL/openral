# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for ``openral_core.scale_intrinsics_to`` (ADR-0035, Task 8).

The deploy-sim detection pipeline renders the same MuJoCo camera at the scene's
``observation_width``/``height`` (e.g. 256 canonical, 640 for the det640
variant). The published camera model must track the render resolution or the
depth back-projection (and the OctoMap voxels it feeds) use the wrong focal
length. ``scale_intrinsics_to`` is the pure helper that rescales a manifest's
nominal pinhole intrinsics to the render resolution; these tests pin the
linear-scaling contract the depth synth and the object lift rely on.
"""

from __future__ import annotations

import pytest
from openral_core import IntrinsicsPinhole, scale_intrinsics_to


def _panda_mobile_agentview() -> IntrinsicsPinhole:
    """The canonical panda_mobile agentview_left / front_depth intrinsics (256²)."""
    return IntrinsicsPinhole(width=256, height=256, fx=256.0, fy=256.0, cx=128.0, cy=128.0)


def test_scales_256_to_640_linearly() -> None:
    # The exact case from the plan: a 256² (fx=256, cx=128) camera rendered at
    # 640² must report fx=fy=640 and cx=cy=320 (cx == width / 2 preserved).
    base = _panda_mobile_agentview()
    scaled = scale_intrinsics_to(base, 640, 640)
    assert scaled.width == 640
    assert scaled.height == 640
    assert scaled.fx == pytest.approx(640.0)
    assert scaled.fy == pytest.approx(640.0)
    assert scaled.cx == pytest.approx(320.0)
    assert scaled.cy == pytest.approx(320.0)


def test_fov_is_preserved() -> None:
    # Linear scaling holds the field of view fixed: width / fx and the
    # normalised principal point cx / width are invariant across resolutions.
    base = _panda_mobile_agentview()
    scaled = scale_intrinsics_to(base, 640, 640)
    assert scaled.width / scaled.fx == pytest.approx(base.width / base.fx)
    assert scaled.cx / scaled.width == pytest.approx(base.cx / base.width)


def test_downscale_to_128() -> None:
    # The base scene renders at 128²; the same helper must halve a 256² model.
    base = _panda_mobile_agentview()
    scaled = scale_intrinsics_to(base, 128, 128)
    assert (scaled.width, scaled.height) == (128, 128)
    assert scaled.fx == pytest.approx(128.0)
    assert scaled.cx == pytest.approx(64.0)


def test_non_square_scales_each_axis_independently() -> None:
    base = IntrinsicsPinhole(width=256, height=256, fx=256.0, fy=256.0, cx=128.0, cy=128.0)
    scaled = scale_intrinsics_to(base, 640, 480)
    assert (scaled.width, scaled.height) == (640, 480)
    assert scaled.fx == pytest.approx(640.0)  # x scaled by 640/256
    assert scaled.fy == pytest.approx(480.0)  # y scaled by 480/256
    assert scaled.cx == pytest.approx(320.0)
    assert scaled.cy == pytest.approx(240.0)


def test_identity_when_resolution_unchanged_returns_same_object() -> None:
    base = _panda_mobile_agentview()
    assert scale_intrinsics_to(base, 256, 256) is base


def test_distortion_is_preserved() -> None:
    base = IntrinsicsPinhole(
        width=256,
        height=256,
        fx=256.0,
        fy=256.0,
        cx=128.0,
        cy=128.0,
        distortion_model="plumb_bob",
        distortion_coeffs=[0.1, -0.05, 0.0, 0.0, 0.0],
    )
    scaled = scale_intrinsics_to(base, 640, 640)
    assert scaled.distortion_model == "plumb_bob"
    assert scaled.distortion_coeffs == [0.1, -0.05, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(("width", "height"), [(0, 640), (640, 0), (-1, 640), (640, -1)])
def test_rejects_non_positive_resolution(width: int, height: int) -> None:
    base = _panda_mobile_agentview()
    with pytest.raises(ValueError, match="positive"):
        scale_intrinsics_to(base, width, height)
