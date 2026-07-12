"""Tests for ``pycuvslam_node.py``.

Two tiers:

* Pure-helper tests (always run): stereo-baseline extraction from a real
  RealSense D435i rectified calibration and the pose algebra the node
  uses to derive the ``map → odom`` correction.
* A real-engine tracking test, gated on the operator-installed PyCuVSLAM
  wheel + GPU (``pytest.importorskip`` — the NVIDIA-licensed engine is
  never bundled, so CI legitimately skips): synthetic rectified stereo
  of a textured plane translating laterally must be tracked to the
  ground-truth displacement within tolerance.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_NODE_FILE = Path(__file__).resolve().parent.parent / "openral_slam_bringup" / "pycuvslam_node.py"
_spec = importlib.util.spec_from_file_location("pycuvslam_node_under_test", _NODE_FILE)
assert _spec is not None and _spec.loader is not None
pycuvslam_node = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pycuvslam_node)

# RealSense D435i 640x480 rectified stereo calibration (fx, cx, cy from the
# factory-calibrated infra pair; Tx = -fx * 0.0499 m baseline).
_D435I_FX = 386.9
_D435I_RIGHT_P = [386.9, 0.0, 320.0, -19.30631, 0.0, 386.9, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]

_IDENTITY = ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))


def test_stereo_baseline_from_d435i_calibration() -> None:
    baseline = pycuvslam_node.stereo_baseline_m(_D435I_RIGHT_P)
    assert baseline == pytest.approx(0.0499, abs=1e-6)


@pytest.mark.parametrize(
    "bad_p, match",
    [
        ([0.0] * 12, "fx"),
        ([386.9, 0.0, 320.0, 0.0, 0.0, 386.9, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0], "baseline"),
        ([386.9, 0.0, 320.0, 19.3, 0.0, 386.9, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0], "baseline"),
        ([1.0, 2.0, 3.0], "12 entries"),
    ],
)
def test_stereo_baseline_rejects_bad_projection(bad_p: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pycuvslam_node.stereo_baseline_m(bad_p)


def test_pose_algebra_round_trip() -> None:
    # 90° about z, then translate.
    s = math.sin(math.pi / 4)
    pose = ((0.0, 0.0, s, s), (1.0, 2.0, 3.0))
    composed = pycuvslam_node.compose_pose(pose, pycuvslam_node.invert_pose(pose))
    for got, want in zip(composed[0], _IDENTITY[0], strict=True):
        assert got == pytest.approx(want, abs=1e-9)
    for got, want in zip(composed[1], _IDENTITY[1], strict=True):
        assert got == pytest.approx(want, abs=1e-9)


def test_map_from_odom_is_identity_when_odometry_agrees() -> None:
    s = math.sin(math.pi / 8)
    tracked = ((0.0, 0.0, s, math.cos(math.pi / 8)), (0.5, -0.25, 0.0))
    correction = pycuvslam_node.map_from_odom(tracked, tracked)
    assert correction[0][3] == pytest.approx(1.0, abs=1e-9)
    assert all(abs(v) < 1e-9 for v in (*correction[0][:3], *correction[1]))


def test_map_from_odom_recovers_pure_odom_drift() -> None:
    # Tracker says the rig moved 1 m in x; odometry claims no motion →
    # the whole discrepancy lands on map→odom.
    tracked = ((0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    correction = pycuvslam_node.map_from_odom(tracked, _IDENTITY)
    assert correction[1][0] == pytest.approx(1.0, abs=1e-12)


def test_tracks_synthetic_stereo_to_ground_truth() -> None:
    """Real cuVSLAM engine on GPU: track a laterally translating stereo rig."""
    cuvslam = pytest.importorskip(
        "cuvslam", reason="PyCuVSLAM wheel not installed (operator-installed, never bundled)"
    )
    np = pytest.importorskip("numpy")

    w, h, f, baseline, depth = 640, 480, 500.0, 0.12, 3.0
    disparity = f * baseline / depth
    step_px = 2
    step_m = step_px * depth / f
    n_frames = 30

    rng = np.random.default_rng(0)
    tex = rng.integers(0, 256, size=(h, w + 400), dtype=np.uint8)
    tex = (tex.astype(np.float32) + np.roll(tex, 1, 1) + np.roll(tex, 1, 0)).astype(np.uint8)

    def camera(tx: float) -> object:
        return cuvslam.Camera(
            size=[w, h],
            principal=[w / 2, h / 2],
            focal=[f, f],
            rig_from_camera=cuvslam.Pose(rotation=[0, 0, 0, 1], translation=[tx, 0, 0]),
        )

    rig = cuvslam.Rig()
    rig.cameras = [camera(0.0), camera(baseline)]
    tracker = cuvslam.Tracker(rig)

    last = None
    for i in range(n_frames):
        x0 = i * step_px
        left = tex[:, x0 : x0 + w].copy()
        right = tex[:, x0 + int(disparity) : x0 + int(disparity) + w].copy()
        pose_est, _ = tracker.track(i * 33_000_000, [left, right])
        last = pose_est

    assert last is not None and last.world_from_rig is not None, "tracker lost"
    t = [float(v) for v in last.world_from_rig.pose.translation]
    expected_x = (n_frames - 1) * step_m
    assert abs(t[0]) == pytest.approx(expected_x, rel=0.3)
    assert abs(t[1]) < 0.05 and abs(t[2]) < 0.10
