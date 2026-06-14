"""Hermetic tests for the pure mobile-base / scan helpers (issue #191 Phase 3).

``quaternion_from_yaw`` and ``constant_scan_no_hit_ranges`` are plain Python
functions exposed at module scope precisely so they can be unit-tested without
booting rclpy or the ROS msg IDL. CLAUDE.md §1.11 — real components, real
schemas; the helpers are the components under test. Phase 3 moved them out of
the (now factory-only) panda_mobile lifecycle node into the shared
``openral_hal`` bridges that own those streams generically; the frame ids are
read from the robot's :class:`~openral_core.RobotDescription`.
"""

from __future__ import annotations

import math

import pytest

# The helpers now live in the shared bridges (the panda_mobile node is a thin
# manifest-driven factory). Frame ids come from the canonical description.
from openral_hal.mobile_base_bridge import quaternion_from_yaw
from openral_hal.panda_mobile import PANDA_MOBILE_DESCRIPTION
from openral_hal.sim_sensor_bridge import constant_scan_no_hit_ranges

PANDA_MOBILE_ODOM_FRAME_ID = PANDA_MOBILE_DESCRIPTION.odom_frame
PANDA_MOBILE_BASE_FRAME_ID = PANDA_MOBILE_DESCRIPTION.base_frame
_lidar = PANDA_MOBILE_DESCRIPTION.lidar_sensor
PANDA_MOBILE_SCAN_FRAME_ID = (
    _lidar.frame_id if _lidar is not None else PANDA_MOBILE_DESCRIPTION.base_frame
)

# ── quaternion_from_yaw ─────────────────────────────────────────────


def test_quaternion_from_yaw_zero_is_identity() -> None:
    """yaw=0 → identity quaternion ``(0, 0, 0, 1)``."""
    qx, qy, qz, qw = quaternion_from_yaw(0.0)
    assert qx == 0.0
    assert qy == 0.0
    assert qz == 0.0
    assert qw == 1.0


def test_quaternion_from_yaw_pi_over_two_matches_axis_angle() -> None:
    """yaw=π/2 → ``(0, 0, sin(π/4), cos(π/4))`` (z-axis 90° rotation)."""
    qx, qy, qz, qw = quaternion_from_yaw(math.pi / 2.0)
    assert qx == 0.0
    assert qy == 0.0
    assert qz == pytest.approx(math.sin(math.pi / 4.0), abs=1e-12)
    assert qw == pytest.approx(math.cos(math.pi / 4.0), abs=1e-12)


def test_quaternion_from_yaw_unit_norm() -> None:
    """The returned quaternion is unit-norm for any yaw."""
    for yaw in (-math.pi, -1.234, 0.0, 0.5, 1.0, math.pi):
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
        assert norm_sq == pytest.approx(1.0, abs=1e-12), f"yaw={yaw}"


# ── constant_scan_no_hit_ranges ─────────────────────────────────────


def test_constant_scan_no_hit_ranges_panda_mobile_shape() -> None:
    """panda_mobile lidar (n_beams + max_range from the manifest) → all max_range."""
    assert _lidar is not None
    n_beams = int(_lidar.n_channels)
    max_range = float(_lidar.range_max_m)
    ranges = constant_scan_no_hit_ranges(n_beams=n_beams, max_range_m=max_range)
    assert isinstance(ranges, list)
    assert len(ranges) == n_beams
    assert all(isinstance(r, float) for r in ranges)
    assert all(r == max_range for r in ranges)


def test_constant_scan_no_hit_ranges_custom_beams_and_range() -> None:
    """Custom n_beams / max_range_m round-trip through the helper."""
    ranges = constant_scan_no_hit_ranges(n_beams=12, max_range_m=5.0)
    assert len(ranges) == 12
    assert all(r == 5.0 for r in ranges)


def test_constant_scan_no_hit_ranges_zero_beams_is_empty() -> None:
    """n_beams=0 returns an empty list (no crash; degenerate but allowed)."""
    ranges = constant_scan_no_hit_ranges(n_beams=0, max_range_m=1.0)
    assert ranges == []


# ── Frame id constants ──────────────────────────────────────────────


def test_frame_id_constants_match_research_doc() -> None:
    """The frame_ids match the ADR-0025 research doc + Nav2 / slam_toolbox defaults.

    `odom` is the canonical odometry frame name. `base_link` is the
    REP-105 mobile-base root frame name. `base_scan` is the
    conventional 2D-laser child frame on a mobile base (RPLIDAR /
    Hokuyo mount). slam_toolbox + Nav2 both default to these.
    """
    assert PANDA_MOBILE_ODOM_FRAME_ID == "odom"
    assert PANDA_MOBILE_BASE_FRAME_ID == "base_link"
    # Scan and base link share a frame so slam_toolbox can transform
    # scans without a separate base_link -> base_scan static TF.
    assert PANDA_MOBILE_SCAN_FRAME_ID == "base_link"
