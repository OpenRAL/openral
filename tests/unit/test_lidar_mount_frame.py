"""The lidar's frame and the ray that produces it must agree.

`robots/panda_mobile/robot.yaml` used to declare `frame_id: base_link` for its
2-D lidar. The sim casts that fan 0.30 m above the FLOOR, while `base_link` is
the `robot0_base_pos` platform frame at 0.700 m — so every return reached Nav2
and slam_toolbox 0.40 m above where it was measured. Kitchens hid it (cabinets
are floor-to-counter slabs, so a fan at 0.05 / 0.30 / 0.70 m returns an
identical scan) but anything living between those heights would have been mapped
0.40 m off.

These pin the fix: the lidar owns its frame, the mount is declared once, and the
declared mount is the offset that reconciles the two.

The same manifest carried a second physical dishonesty, fixed by #194:
`range_min_m` read 0.55 m — not the sensor's minimum but a radial cutoff sized
to hide a chassis whose circumscribed radius is 0.43 m. It deleted every REAL
obstacle inside 0.55 m in every direction, in the band where Nav2 has the least
room to react. Self-returns are excluded by IDENTITY instead — by kinematic-tree
root in sim, by the chassis polygon in `payload_scan_filter_node` on hardware —
so the field is free to mean what it says. The last test below is the invariant:
a `range_min_m` that reaches past the chassis is a self-filter again, whatever
the comment claims.
"""

from __future__ import annotations

import math

import pytest
import yaml
from openral_core.schemas import RobotDescription, SensorModality

#: Where the sim casts the fan, in world z — `openral_sim.backends.robocasa`
#: `_LASER_DEFAULT_HEIGHT_M`. An RPLIDAR-A1 on the Omron base's bumper.
_CAST_WORLD_Z_M = 0.30

#: `base_link`'s world z, i.e. robosuite's `robot0_base_pos[2]` for PandaMobile.
#: Measured live off `odom -> base_link`, and deliberate: `sim_attached.py`
#: records that publishing 0.0 here made pi05/rldx see the arm 0.49 m off.
_BASE_LINK_WORLD_Z_M = 0.700


def _panda_mobile() -> RobotDescription:
    with open("robots/panda_mobile/robot.yaml", encoding="utf-8") as fh:
        return RobotDescription.model_validate(yaml.safe_load(fh))


def test_lidar_has_its_own_frame_not_the_base_frame() -> None:
    """A sensor mislabelled into `base_link` cannot express a mount height."""
    lidar = _panda_mobile().lidar_sensor
    assert lidar is not None
    assert lidar.modality == SensorModality.LIDAR_2D
    assert lidar.frame_id != "base_link", (
        "the lidar must publish in its own frame — `base_link` silently places "
        "its returns at the platform height instead of the mount height"
    )
    assert lidar.frame_id == "base_scan"


def test_the_declared_mount_reconciles_the_ray_with_the_frame() -> None:
    """The offset is not decorative: it is exactly the discrepancy it fixes."""
    lidar = _panda_mobile().lidar_sensor
    assert lidar is not None
    assert lidar.parent_frame == "base_link"
    assert lidar.static_transform_xyz_rpy is not None
    x, y, z, roll, pitch, yaw = lidar.static_transform_xyz_rpy
    # A planar lidar mounted level: offset in z only.
    assert (x, y, roll, pitch, yaw) == (0.0, 0.0, 0.0, 0.0, 0.0)
    # base_link (0.700) + mount (-0.40) == where the ray is actually cast (0.30).
    assert _BASE_LINK_WORLD_Z_M + z == pytest.approx(_CAST_WORLD_Z_M, abs=1e-9), (
        f"mount {z} m puts the scan frame at {_BASE_LINK_WORLD_Z_M + z} m, but the "
        f"fan is cast at {_CAST_WORLD_Z_M} m — the ray and the frame disagree again"
    )


def test_the_mount_default_matches_the_sim_cast_height() -> None:
    """Pin the constant the manifest is reconciled against."""
    from openral_sim.backends.robocasa import _LASER_DEFAULT_HEIGHT_M

    assert _LASER_DEFAULT_HEIGHT_M == _CAST_WORLD_Z_M


def test_range_min_cannot_reach_past_the_chassis() -> None:
    """`range_min_m` is the sensor's minimum, so it must not double as a self-filter.

    Regression for #194. A `range_min_m` at or beyond the chassis's
    circumscribed radius is indistinguishable from a radial self-filter: every
    return it drops inside that radius *could* be the robot, so the value stops
    being a sensor property and starts silently deleting obstacles. Below the
    radius it can only be the sensor's own floor.
    """
    desc = _panda_mobile()
    lidar = desc.lidar_sensor
    assert lidar is not None
    assert lidar.range_min_m is not None

    # Circumscribed radius of the bare chassis outline the shaped self-filter
    # (`payload_scan_filter_node`) proves against — 0.43 m for panda_mobile.
    circumscribed_m = max(math.hypot(x, y) for x, y in desc.footprint_polygon)

    assert lidar.range_min_m < circumscribed_m, (
        f"range_min_m {lidar.range_min_m} m reaches the chassis "
        f"(circumscribed radius {circumscribed_m:.3f} m), so it is acting as a "
        "blunt radial self-filter and deleting real obstacles with it — the job "
        "`payload_scan_filter_node`'s chassis polygon does with a proof"
    )
