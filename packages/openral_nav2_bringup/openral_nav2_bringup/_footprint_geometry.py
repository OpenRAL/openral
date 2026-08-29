"""Footprint geometry shared by this package's nodes.

Extracted from the former ``payload_footprint_node`` when Nav2 went base-only.
That node grew the costmaps' footprint polygon over a carried object; it was
removed because the growth projected 3-D geometry onto a 2-D costmap whose
obstacles come from a single scan slice — forbidding exactly the place poses the
tasks require (a payload entering a fridge projects onto the fixture the base
must approach) while protecting against nothing. See this package's README,
"Nav2 is base-only".

What survives is the geometry the scan filter still needs: the shape-type
constants and the manifest's own chassis outline, which the filter uses to
decide that a return is the robot itself.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

# Mirrors openral_msgs/AttachedCollisionPrimitive. Duplicated as plain ints so
# the pure geometry below imports without the colcon message overlay (the unit
# tests run outside it); the ROS half asserts the constants match at runtime.
SHAPE_SPHERE = 1
SHAPE_CAPSULE = 2
SHAPE_BOX = 3

_MIN_POLYGON_VERTICES = 3


def convex_hull_2d(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Counter-clockwise convex hull of ``points`` (Andrew's monotone chain).

    Collinear points are dropped, so the result is a minimal vertex set. Fewer
    than three distinct points is a degenerate footprint and raises rather than
    handing Nav2 a polygon it will reject at ``setRobotFootprintPolygon``.

    Args:
        points: ``(x, y)`` pairs in metres. Duplicates are fine.

    Returns:
        Hull vertices, counter-clockwise, first vertex not repeated at the end.

    Raises:
        ValueError: If fewer than three non-collinear finite points are given.

    Example:
        >>> convex_hull_2d([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)])
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    """
    pts = sorted({(float(x), float(y)) for x, y in points})
    for x, y in pts:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"convex_hull_2d got a non-finite vertex ({x}, {y})")
    if len(pts) < _MIN_POLYGON_VERTICES:
        raise ValueError(f"convex_hull_2d needs >= 3 distinct points; got {len(pts)}")

    def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _half(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2 and _cross(out[-2], out[-1], p) <= 0.0:
                out.pop()
            out.append(p)
        return out

    lower = _half(pts)
    upper = _half(pts[::-1])
    hull = lower[:-1] + upper[:-1]
    if len(hull) < _MIN_POLYGON_VERTICES:
        raise ValueError(f"convex_hull_2d: {len(pts)} points are collinear; no polygon")
    return hull


def _circumscribed_circle_points(
    cx: float, cy: float, radius: float, *, circle_samples: int
) -> list[tuple[float, float]]:
    """N-gon vertices whose *inscribed* circle is the requested one.

    Sampling the circle itself would inscribe the polygon in it and lose up to
    ``radius * (1 - cos(pi/n))`` of the payload. Scaling by ``1/cos(pi/n)``
    puts the circle inside the polygon instead, which is the safe direction.
    """
    n = max(int(circle_samples), 4)
    scale = 1.0 / math.cos(math.pi / n)
    r = float(radius) * scale
    return [
        (cx + r * math.cos(2.0 * math.pi * i / n), cy + r * math.sin(2.0 * math.pi * i / n))
        for i in range(n)
    ]


def base_footprint_polygon(
    description: Any, *, circle_samples: int = 12
) -> list[tuple[float, float]]:
    """The robot's nominal Nav2 footprint polygon, from its manifest.

    Prefers ``RobotDescription.footprint_polygon`` (the measured outline) and
    falls back to a circumscribed N-gon around ``footprint_radius``. A manifest
    with neither declares no mobile base and has no Nav2 footprint to publish.

    Args:
        description: A loaded ``openral_core.RobotDescription``.
        circle_samples: N-gon resolution used for the ``footprint_radius``
            fallback.

    Returns:
        Counter-clockwise ``(x, y)`` vertices in the robot's ``base_frame``.

    Raises:
        ValueError: If the manifest declares neither a footprint polygon nor a
            footprint radius.
    """
    poly = getattr(description, "footprint_polygon", None)
    if poly:
        return convex_hull_2d([(float(x), float(y)) for x, y in poly])
    radius = getattr(description, "footprint_radius", None)
    if radius is not None:
        return convex_hull_2d(
            _circumscribed_circle_points(0.0, 0.0, float(radius), circle_samples=circle_samples)
        )
    raise ValueError(
        f"robot {getattr(description, 'robot_id', '?')!r} declares neither footprint_polygon "
        "nor footprint_radius; it has no Nav2 footprint to publish"
    )
