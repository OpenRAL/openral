"""Both halves of the scan Nav2's costmaps read: the payload, and the robot.

The payload containment predicate here is the kernel's own
(``surface_distance`` in ``openral_octomap_bridge/src/payload_clearing.cpp``) at
zero slack, so these tests double as the assertion that the two have not
drifted: a capsule whose segment ran along local +X instead of +Z, or a box read
as full extents instead of half-extents, would fail here before it reached a
robot.

The self half is the opposite risk and is tested for the opposite property. The
payload half's job is to remove enough; the self half's job is to remove *only*
what it can prove is the robot, and to remove nothing at all when it cannot
place the chassis. Both directions leave more obstacles in the scan, which is
the single invariant this suite pins.
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
    base_footprint_polygon,
)
from openral_nav2_bringup.payload_scan_filter_node import (
    DEFAULT_OUTPUT_TOPIC,
    filter_scan_ranges,
    points_in_convex_polygon,
    points_in_primitive,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NAV2_CONFIG = (
    _REPO_ROOT / "packages" / "openral_nav2_bringup" / "config" / "nav2_panda_mobile.yaml"
)
_PANDA_MOBILE_YAML = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"

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


# --------------------------------------------------------------------------
# The robot half: self-returns.
#
# `panda_mobile`'s manifest declares a 0.70 x 0.50 m chassis rectangle, so the
# chassis reaches 0.35 m forward and 0.25 m sideways. Every case below is
# anchored to that real outline rather than to a hand-written polygon.
# --------------------------------------------------------------------------


def _chassis() -> list[tuple[float, float]]:
    """The real ``robots/panda_mobile/robot.yaml`` outline, CCW convex."""
    from openral_core import RobotDescription

    return base_footprint_polygon(RobotDescription.from_yaml(str(_PANDA_MOBILE_YAML)))


def _self_filter(ranges: list[float], **kwargs: object) -> list[float]:
    """Run only the self half, with the scan frame at the base origin."""
    return filter_scan_ranges(
        ranges,
        angle_min=_ANGLE_MIN,
        angle_increment=_ANGLE_INCREMENT,
        range_min=_RANGE_MIN,
        range_max=_RANGE_MAX,
        placements=[],
        self_polygon=kwargs.pop("self_polygon", _chassis()),  # type: ignore[arg-type]
        base_from_scan=kwargs.pop("base_from_scan", np.eye(4)),
        **kwargs,  # type: ignore[arg-type] # reason: only self_margin_m is forwarded.
    )


def test_the_manifests_chassis_is_the_polygon_the_self_filter_uses() -> None:
    """Pinned against the real manifest, so a re-measured chassis reaches here."""
    chassis = _chassis()

    assert sorted(chassis) == sorted([(0.35, 0.25), (-0.35, 0.25), (-0.35, -0.25), (0.35, -0.25)])
    # `base_footprint_polygon` hulls, so the half-plane test is always valid.
    probes = np.array([[0.0, 0.0], [0.34, 0.24], [0.36, 0.0], [0.0, 0.26]])
    assert [bool(v) for v in points_in_convex_polygon(probes, chassis)] == [
        True,
        True,
        False,
        False,
    ]


def test_a_self_return_goes_and_a_real_obstacle_at_the_same_bearing_stays() -> None:
    """The discriminating case, at one bearing, in the two scans it takes.

    A single beam reports one range, so a self-return and a real obstacle
    cannot occupy the same beam of the same scan — the honest construction is
    the same *bearing* across two scans. Forward, the chassis reaches 0.35 m:
    a 0.20 m return there is the robot and must go; a 0.60 m return on the
    identical beam is something real just past the chassis and must survive.
    """
    ahead = _beam_index(0.0)

    self_hit = _fan(wall_m=3.0)
    self_hit[ahead] = 0.20
    obstacle = _fan(wall_m=3.0)
    obstacle[ahead] = 0.60

    assert math.isinf(_self_filter(self_hit)[ahead]), "the chassis return survived"
    assert _self_filter(obstacle)[ahead] == pytest.approx(0.60), (
        "a real obstacle 0.25 m past the chassis was deleted as self"
    )
    # And nothing else moved in either scan.
    assert sum(1 for r in _self_filter(self_hit) if math.isinf(r)) == 1
    assert _self_filter(obstacle) == obstacle


def test_the_self_filter_removes_nothing_without_a_placed_chassis() -> None:
    """No manifest, or no TF, is `remove nothing` — never `remove everything`.

    This is the whole fail-closed direction of the robot half. The node passes
    ``self_polygon=None`` on a missing manifest and on a failed
    ``base_frame <- scan_frame`` lookup, and both must be bit-for-bit
    pass-through of the returns that are *inside* the chassis.
    """
    ranges = _fan(wall_m=3.0)
    for angle in (-0.3, 0.0, 0.3, math.pi / 2):
        ranges[_beam_index(angle)] = 0.20

    unplaced = filter_scan_ranges(
        ranges,
        angle_min=_ANGLE_MIN,
        angle_increment=_ANGLE_INCREMENT,
        range_min=_RANGE_MIN,
        range_max=_RANGE_MAX,
        placements=[],
        self_polygon=None,
    )

    assert unplaced == ranges


def test_a_chassis_polygon_that_cannot_be_placed_raises_rather_than_guessing() -> None:
    """A bad transform must reach the node as an error, not a silent identity."""
    with pytest.raises(ValueError, match=r"base_from_scan"):
        _self_filter(_fan(wall_m=3.0), base_from_scan=np.eye(3))


@pytest.mark.parametrize(
    ("name", "outline"),
    [
        # CCW but notched: the half-plane test would call the notch `robot`
        # and delete real returns out of it.
        (
            "ccw-concave",
            [(0.35, -0.25), (0.15, 0.0), (0.35, 0.25), (-0.35, 0.25), (-0.35, -0.25)],
        ),
        # Convex but wound the other way: every half-plane inverts, so the
        # test would call the whole world `robot` and empty the scan.
        (
            "clockwise",
            [(0.35, -0.25), (-0.35, -0.25), (-0.35, 0.25), (0.35, 0.25)],
        ),
    ],
)
def test_an_outline_the_half_plane_test_cannot_read_is_refused(
    name: str, outline: list[tuple[float, float]]
) -> None:
    """Both failure modes empty or over-eat the scan, so both must raise.

    ``base_footprint_polygon`` always hulls, so neither can arise in-tree —
    this pins that a future hand-supplied outline cannot slip past either.
    """
    assert name
    with pytest.raises(ValueError, match=r"counter-clockwise convex"):
        _self_filter(_fan(wall_m=3.0), self_polygon=outline)


def test_the_self_filter_follows_the_sensor_off_the_base_origin() -> None:
    """The chassis is in ``base_frame``; the beams are in the sensor's.

    A lidar mounted 0.20 m forward of ``base_link`` sees the chassis's rear
    edge 0.55 m behind it, not 0.35 m. Getting the transform direction wrong
    would delete a band of real space behind the robot, so it is pinned.
    """
    behind = _beam_index(math.pi)
    ranges = _fan(wall_m=3.0)
    ranges[behind] = 0.50  # still inside the chassis, from a forward sensor
    mounted_forward = _translation(0.20, 0.0, 0.0)

    assert math.isinf(_self_filter(ranges, base_from_scan=mounted_forward)[behind])

    just_outside = _fan(wall_m=3.0)
    just_outside[behind] = 0.60  # 0.05 m past the chassis's rear edge
    assert _self_filter(just_outside, base_from_scan=mounted_forward)[behind] == pytest.approx(0.60)


def test_both_halves_run_together_and_neither_suppresses_the_other() -> None:
    """One scan, a chassis return and a payload return, both gone; wall stays."""
    ranges = _fan(wall_m=3.0)
    chassis_beam, payload_beam = _beam_index(0.0), _beam_index(0.30)
    ranges[chassis_beam] = 0.20
    ranges[payload_beam] = 0.90
    payload_xy = (0.90 * math.cos(0.30), 0.90 * math.sin(0.30))
    payload = (SHAPE_SPHERE, (0.06,), _translation(payload_xy[0], payload_xy[1], 0.0))

    filtered = filter_scan_ranges(
        ranges,
        angle_min=_ANGLE_MIN,
        angle_increment=_ANGLE_INCREMENT,
        range_min=_RANGE_MIN,
        range_max=_RANGE_MAX,
        placements=[payload],
        self_polygon=_chassis(),
        base_from_scan=np.eye(4),
    )

    assert math.isinf(filtered[chassis_beam])
    assert math.isinf(filtered[payload_beam])
    assert sum(1 for r in filtered if math.isinf(r)) == 2


def test_the_self_margin_defaults_to_zero_and_only_ever_removes_more() -> None:
    """The margin is the dangerous direction, so it must be opt-in and explicit."""
    ranges = _fan(wall_m=3.0)
    just_outside = _beam_index(0.0)
    ranges[just_outside] = 0.36  # 10 mm past the chassis's 0.35 m front edge

    assert _self_filter(ranges)[just_outside] == pytest.approx(0.36)
    assert math.isinf(_self_filter(ranges, self_margin_m=0.02)[just_outside])

    with pytest.raises(ValueError, match=r"margin_m must be finite and non-negative"):
        _self_filter(ranges, self_margin_m=-0.01)
