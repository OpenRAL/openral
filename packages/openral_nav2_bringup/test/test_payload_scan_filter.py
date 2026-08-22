"""The payload half of the scan Nav2's costmaps read.

The containment predicate here is the kernel's own (``surface_distance`` in
``openral_octomap_bridge/src/payload_clearing.cpp``) at zero slack, so these
tests double as the assertion that the two have not drifted: a capsule whose
segment ran along local +X instead of +Z, or a box read as full extents instead
of half-extents, would fail here before it reached a robot.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from openral_nav2_bringup.payload_footprint_node import (
    SHAPE_BOX,
    SHAPE_CAPSULE,
    SHAPE_SPHERE,
)
from openral_nav2_bringup.payload_scan_filter_node import (
    DEFAULT_OUTPUT_TOPIC,
    filter_scan_ranges,
    points_in_primitive,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NAV2_CONFIG = (
    _REPO_ROOT / "packages" / "openral_nav2_bringup" / "config" / "nav2_panda_mobile.yaml"
)

# A 360-beam RPLIDAR-style fan, the panda_mobile HAL's own scan geometry
# (`robots/panda_mobile/robot.yaml` -> lidar_2d; `synthesize_laser_scan_2d`
# starts the fan at -pi and steps a full turn).
_N_BEAMS = 360
_ANGLE_MIN = -math.pi
_ANGLE_INCREMENT = 2.0 * math.pi / _N_BEAMS
_RANGE_MIN = 0.05
_RANGE_MAX = 12.0


def _translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _fan(*, wall_m: float) -> list[float]:
    """A scan that sees a wall at ``wall_m`` in every direction."""
    return [wall_m] * _N_BEAMS


def _beam_index(angle: float) -> int:
    return round((angle - _ANGLE_MIN) / _ANGLE_INCREMENT) % _N_BEAMS


def _filter(ranges: list[float], placements: list, **kwargs: float) -> list[float]:
    return filter_scan_ranges(
        ranges,
        angle_min=_ANGLE_MIN,
        angle_increment=_ANGLE_INCREMENT,
        range_min=_RANGE_MIN,
        range_max=_RANGE_MAX,
        placements=placements,
        **kwargs,  # type: ignore[arg-type] # reason: only margin_m is forwarded.
    )


def test_nav2_config_reads_the_filtered_scan_everywhere() -> None:
    """Every costmap source AND the collision monitor, or the payload leaks in.

    The collision monitor is the one that matters most: it reads the scan
    directly, so no costmap-side ``footprint_clearing_enabled`` protects it.
    """
    cfg = yaml.safe_load(_NAV2_CONFIG.read_text())

    local = cfg["local_costmap"]["local_costmap"]["ros__parameters"]
    global_ = cfg["global_costmap"]["global_costmap"]["ros__parameters"]
    monitor = cfg["collision_monitor"]["ros__parameters"]

    assert local["voxel_layer"]["scan"]["topic"] == DEFAULT_OUTPUT_TOPIC
    assert global_["obstacle_layer"]["scan"]["topic"] == DEFAULT_OUTPUT_TOPIC
    assert monitor["scan"]["topic"] == DEFAULT_OUTPUT_TOPIC


def test_the_payloads_own_returns_are_dropped_and_nothing_else_is() -> None:
    """A carried box 0.6 m ahead stops being an obstacle; the wall does not."""
    ranges = _fan(wall_m=3.0)
    # The forward beams hit the payload instead of the wall.
    payload_beams = [_beam_index(a) for a in np.linspace(-0.08, 0.08, 5)]
    for i in payload_beams:
        ranges[i] = 0.60
    payload = (SHAPE_BOX, (0.06, 0.06, 0.06), _translation(0.60, 0.0, 0.0))

    filtered = _filter(ranges, [payload])

    for i in payload_beams:
        assert math.isinf(filtered[i]), f"beam {i} still reports the payload"
    assert sum(1 for r in filtered if math.isinf(r)) == len(payload_beams)
    assert all(filtered[i] == 3.0 for i in range(_N_BEAMS) if i not in payload_beams)


def test_an_unattached_scan_is_returned_untouched() -> None:
    """No attachments on the wire is a pass-through, bit for bit."""
    ranges = _fan(wall_m=2.5)

    assert _filter(ranges, []) == ranges


def test_a_real_obstacle_at_the_payloads_bearing_survives() -> None:
    """Only the returns the payload's own volume explains may go."""
    ranges = _fan(wall_m=3.0)
    ahead = _beam_index(0.0)
    ranges[ahead] = 1.20  # something real, well past the payload
    payload = (SHAPE_SPHERE, (0.05,), _translation(0.60, 0.0, 0.0))

    filtered = _filter(ranges, [payload])

    assert filtered[ahead] == pytest.approx(1.20)


def test_the_capsule_segment_runs_along_local_z_like_the_kernels() -> None:
    """Same convention as ``payload_clearing.cpp``, or the shapes diverge."""
    upright = np.eye(4)
    probes = np.array([[0.0, 0.0, 0.40], [0.40, 0.0, 0.0]])

    inside = points_in_primitive(probes, SHAPE_CAPSULE, (0.03, 1.00), upright)

    assert bool(inside[0]) is True, "0.40 m up the segment must be inside"
    assert bool(inside[1]) is False, "0.40 m sideways must be outside"


def test_the_box_dimensions_are_half_extents() -> None:
    """A 10 cm half-extent box contains (0.09, 0, 0) and not (0.11, 0, 0)."""
    probes = np.array([[0.09, 0.0, 0.0], [0.11, 0.0, 0.0]])

    inside = points_in_primitive(probes, SHAPE_BOX, (0.10, 0.10, 0.10), np.eye(4))

    assert [bool(v) for v in inside] == [True, False]


def test_the_margin_only_ever_removes_more() -> None:
    ranges = _fan(wall_m=3.0)
    near_miss = _beam_index(0.0)
    ranges[near_miss] = 0.66  # 10 mm outside a 0.05 m sphere at 0.60
    payload = (SHAPE_SPHERE, (0.05,), _translation(0.60, 0.0, 0.0))

    assert _filter(ranges, [payload])[near_miss] == pytest.approx(0.66)
    assert math.isinf(_filter(ranges, [payload], margin_m=0.02)[near_miss])


def test_the_sensors_own_no_return_is_left_alone() -> None:
    """Readings already outside [range_min, range_max] are not ours to rewrite.

    A sub-``range_min`` reading back-projects to a point inside any payload
    sitting over the sensor, so without the range gate this is exactly the beam
    the filter would eat.
    """
    ranges = _fan(wall_m=3.0)
    no_return, too_close = _beam_index(0.0), _beam_index(0.05)
    ranges[no_return] = math.inf
    ranges[too_close] = 0.01
    swallows_everything = (SHAPE_SPHERE, (0.9,), _translation(0.0, 0.0, 0.0))

    filtered = _filter(ranges, [swallows_everything])

    assert math.isinf(filtered[no_return])
    assert filtered[too_close] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("shape_type", "dims"),
    [(SHAPE_SPHERE, ()), (SHAPE_CAPSULE, (0.02,)), (SHAPE_BOX, (0.1, 0.1)), (99, (0.1,))],
)
def test_a_primitive_the_kernel_would_reject_raises(
    shape_type: int, dims: tuple[float, ...]
) -> None:
    """Malformed geometry raises so the node republishes the scan unfiltered."""
    ranges = _fan(wall_m=3.0)

    with pytest.raises(ValueError, match=r"shape_type|radius|length|half-extent"):
        _filter(ranges, [(shape_type, dims, _translation(0.6, 0.0, 0.0))])
