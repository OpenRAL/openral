"""Ray-cast solids into the three #418 scenes' depth cameras — the object-tracker test rigs.

Each rig is a committed camera, read from the real files (no hand-copied numbers):

* SO-101 PickOrange ``top_depth`` — ``scenes/deploy/isaac_leisaac_pick_orange.yaml``
  (``camera_mounts.top``, ``axes: ros``, on the robot ``base``; ``sim_sensors`` intrinsics);
  objects around the derived ``grasp_declaration.search_box`` (orange_1).
* DROID ``exterior_depth`` — ``scenes/deploy/isaac_arena_droid_banana_in_bowl.yaml``
  (``axes: usd`` on ``panda_link0``); objects around its ``search_box`` (the banana).
* OpenArm ``head_zed`` — ``robots/openarm/robot.yaml`` via ``_head_camera_render``; objects on
  the warehouse deck at a prop position of ``scenes/deploy/isaac_openarm_warehouse.yaml``.

Solids are exact (sphere, vertical cylinder, yawed box, and a banana-like curved union of
spheres along an arc) resting on the support plane. ``render`` returns the z-depth raster
and the 1-based solid each pixel hits first (the per-instance masks a segmenter returns);
``grid_from_depth`` voxelises what the camera saw, as a single-view octomap would. No mocks
(CLAUDE.md §1.11).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from openral_core import IntrinsicsPinhole
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_core.perception_geometry import VoxelLattice

from tests.unit._head_camera_render import head_camera

_REPO = Path(__file__).resolve().parents[2]
_PICK_ORANGE = _REPO / "scenes" / "deploy" / "isaac_leisaac_pick_orange.yaml"
_DROID = _REPO / "scenes" / "deploy" / "isaac_arena_droid_banana_in_bowl.yaml"
_WAREHOUSE = _REPO / "scenes" / "deploy" / "isaac_openarm_warehouse.yaml"
RES = 0.015  # sim octomap cells (deploy_e2e.launch.py _octomap_resolution)


@dataclass(frozen=True)
class Rig:
    """A camera in a scene, and where objects stand in front of it."""

    name: str
    frame: str
    t: NDArray[np.float64]  # base <- optical
    k: IntrinsicsPinhole
    support_z: float
    centre_xy: tuple[float, float]


def _scaled(k: dict[str, float], scale: float) -> IntrinsicsPinhole:
    return IntrinsicsPinhole(
        width=round(k["width"] * scale),
        height=round(k["height"] * scale),
        fx=k["fx"] * scale,
        fy=k["fy"] * scale,
        cx=k["cx"] * scale,
        cy=k["cy"] * scale,
    )


def _scene_rig(path: Path, sensor: str, name: str, scale: float) -> tuple[Rig, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text())
    opts = raw["scene"]["backend_options"]
    mount = opts["camera_mounts"][sensor]
    k = next(s for s in opts["sim_sensors"] if s["name"] == sensor)["intrinsics"]
    w, x, y, z = mount["quat_wxyz"]
    t = homogeneous_from_quat_xyz(tuple(mount["xyz"]), (x, y, z, w))
    if mount["axes"] == "usd":  # -z forward, +y up -> REP-103 optical
        t[:3, :3] = t[:3, :3] @ np.diag((1.0, -1.0, -1.0))
    box = raw["grasp_declaration"]["search_box"]
    cx, cy, _ = box["pose"]["xyz"]
    frame = box["frame_id"]
    rig = Rig(name, frame, np.asarray(t, np.float64), _scaled(k, scale), 0.0, (cx, cy))
    return rig, box


def so101_top(scale: float = 0.5) -> Rig:
    """PickOrange ``top_depth``; the support is the orange's bottom (search box centre - r)."""
    rig, box = _scene_rig(_PICK_ORANGE, "top_depth", "so101_top", scale)
    return Rig(**{**rig.__dict__, "support_z": box["pose"]["xyz"][2] - 0.024})


def droid_exterior(scale: float = 1.0) -> Rig:
    """DROID ``exterior_depth``; the support is the search box bottom (the table)."""
    rig, box = _scene_rig(_DROID, "exterior_depth", "droid_exterior", scale)
    z = box["pose"]["xyz"][2] - box["half_extents"][2]
    return Rig(**{**rig.__dict__, "support_z": z})


def openarm_head(scale: float = 0.25) -> Rig:
    """OpenArm ``head_zed`` over the warehouse deck (first prop position)."""
    from tests.unit.test_object_primitives import _DECK_Z, _FRAME, _PROPS_XY

    t, k = head_camera(scale)
    return Rig("openarm_head", _FRAME, t, k, _DECK_Z, _PROPS_XY[0])


RIGS = {"so101_top": so101_top, "droid_exterior": droid_exterior, "openarm_head": openarm_head}


# -- solids: (kind, params); every ray parameter is optical depth ---------------------------

Solid = tuple[str, tuple[float, ...]]
Arr = NDArray[np.float64]


def sphere(c: tuple[float, float, float], r: float) -> Solid:
    return "sphere", (*c, r)


def cylinder(xy: tuple[float, float], r: float, z0: float, z1: float) -> Solid:
    return "cylinder", (*xy, r, z0, z1)


def box(c: tuple[float, float, float], half: tuple[float, float, float], yaw: float) -> Solid:
    return "box", (*c, *half, yaw)


def banana(
    xy: tuple[float, float], z0: float, arc_r: float, r: float, span: float, yaw: float
) -> Solid:
    """A curved union of 11 spheres of radius ``r`` along an arc, lying on the support."""
    th = np.linspace(-span / 2, span / 2, 11) + yaw
    cx = xy[0] + arc_r * (np.cos(th) - math.cos(yaw))
    cy = xy[1] + arc_r * (np.sin(th) - math.sin(yaw))
    zs = z0 + r + 0.3 * r * (np.abs(th - yaw) / max(span / 2, 1e-9))  # tips curl up a little
    return "banana", tuple(float(v) for i in range(11) for v in (cx[i], cy[i], zs[i], r))


def _hit_spheres(o: Arr, d: Arr, spheres: Arr) -> Arr:
    best = np.full(len(d), np.inf)
    for cx, cy, cz, r in spheres:
        oc = o - (cx, cy, cz)
        a = np.einsum("ij,ij->i", d, d)
        b = 2 * d @ oc
        disc = b * b - 4 * a * (oc @ oc - r * r)
        t = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
        best = np.minimum(best, np.where(t > 0, t, np.inf))
    return best


def _hit(solid: Solid, o: Arr, d: Arr) -> Arr:
    kind, p = solid
    if kind in ("sphere", "banana"):
        return _hit_spheres(o, d, np.asarray(p).reshape(-1, 4))
    if kind == "box":
        cx, cy, cz, hx, hy, hz, yaw = p
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)))  # base -> box
        lo, hi = np.array((-hx, -hy, -hz)), np.array((hx, hy, hz))
        ol, dl = rot @ (o - (cx, cy, cz)), d @ rot.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - ol) / dl, (hi - ol) / dl
            near = np.nanmax(np.minimum(t1, t2), axis=1)
            far = np.nanmin(np.maximum(t1, t2), axis=1)
        return np.asarray(np.where((near <= far) & (near > 0), near, np.inf), np.float64)
    cx, cy, r, z0, z1 = p  # vertical cylinder: side, then caps
    ox, oy = o[0] - cx, o[1] - cy
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (ox * d[:, 0] + oy * d[:, 1])
    disc = b * b - 4 * a * (ox * ox + oy * oy - r * r)
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
        zs = o[2] + ts * d[:, 2]
        side = np.where((disc >= 0) & (ts > 0) & (zs >= z0) & (zs <= z1), ts, np.inf)
        tc = (z1 - o[2]) / d[:, 2]
        px, py = ox + tc * d[:, 0], oy + tc * d[:, 1]
        cap = np.where((tc > 0) & (px * px + py * py <= r * r), tc, np.inf)
    return np.asarray(np.minimum(side, cap), np.float64)


def render(solids: list[Solid], rig: Rig) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """``(depth, labels)`` of ``solids`` on the support plane (label 0 = support / nothing)."""
    k, t = rig.k, rig.t
    rows, cols = np.mgrid[0 : k.height, 0 : k.width]
    d = np.stack(((cols + 0.5 - k.cx) / k.fx, (rows + 0.5 - k.cy) / k.fy, np.ones(rows.shape)))
    d = d.reshape(3, -1).T @ t[:3, :3].T
    o = t[:3, 3]
    with np.errstate(divide="ignore"):
        plane = (rig.support_z - o[2]) / d[:, 2]
    depth = np.where(plane > 0, plane, np.inf)
    label = np.zeros(len(d), dtype=np.int64)
    for n, solid in enumerate(solids, start=1):
        hit = _hit(solid, o, d)
        closer = hit < depth
        depth[closer], label[closer] = hit[closer], n
    depth[~np.isfinite(depth)] = 0.0
    return depth.reshape(k.height, k.width), label.reshape(k.height, k.width)


def surface_samples(solid: Solid, n: int = 400) -> NDArray[np.float64]:
    """Points on the solid's surface (its convex hull's extreme points among them)."""
    kind, p = solid
    rng = np.random.default_rng(0)
    if kind in ("sphere", "banana"):
        u = rng.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        sph = np.asarray(p).reshape(-1, 4)
        return np.concatenate([s[:3] + s[3] * u for s in sph])
    if kind == "box":
        cx, cy, cz, hx, hy, hz, yaw = p
        g = np.array([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
        loc = np.concatenate((g, rng.uniform(-1, 1, (n, 3)))) * (hx, hy, hz)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.asarray(loc @ np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))).T + (cx, cy, cz))
    cx, cy, r, z0, z1 = p
    th = rng.uniform(0, 2 * math.pi, n)
    z = np.where(rng.uniform(size=n) < 0.5, z0, z1)
    return np.stack((cx + r * np.cos(th), cy + r * np.sin(th), z), axis=1)


def grid_from_depth(
    depth: NDArray[np.float64], rig: Rig, *, extra: Sequence[tuple[int, int, int]] = ()
) -> VoxelLattice:
    """What a single-view octomap of ``depth`` holds: every back-projected point's cell.

    The lattice is 0.6 m square around ``rig.centre_xy``, its layer boundary on the support.
    ``extra`` cells are marked occupied too (a body the camera did not see).
    """
    k, t = rig.k, rig.t
    rows, cols = np.nonzero(depth > 0)
    z = depth[rows, cols]
    pts = np.stack(((cols + 0.5 - k.cx) / k.fx * z, (rows + 0.5 - k.cy) / k.fy * z, z), axis=1)
    pts = pts @ t[:3, :3].T + t[:3, 3]
    origin = (rig.centre_xy[0] - 0.3, rig.centre_xy[1] - 0.3, rig.support_z - 2 * RES)
    size = (40, 40, 20)
    ijk = np.floor((pts - origin) / RES).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < size), axis=1)
    occ = np.zeros(size[0] * size[1] * size[2], dtype=np.uint8)
    for i, j, kk in [*ijk[ok].tolist(), *extra]:
        occ[i + size[0] * (j + size[1] * kk)] = 1
    return VoxelLattice(rig.frame, origin, (0.0, 0.0, 0.0, 1.0), RES, size, occ)


def inside_hull(points: NDArray[np.float64], hull_base: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether each point lies in ``conv(hull_base)`` (1e-7 m slack)."""
    from scipy.spatial import ConvexHull

    eq = ConvexHull(hull_base).equations
    return np.asarray(np.all(points @ eq[:, :3].T + eq[:, 3] <= 1e-7, axis=1))
