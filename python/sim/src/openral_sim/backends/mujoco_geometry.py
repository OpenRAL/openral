# SPDX-License-Identifier: Apache-2.0
"""MuJoCo scene geometry in the robot's base frame: static colliders and body boxes.

The MuJoCo side of the backend geometry hooks (``tools/derive_scene_inputs.py``), in
the same reply shape as the Isaac sidecar's ``static_colliders`` endpoint
(``IsaacManifestScene.static_collider_boxes``): ``{"frame": "base_frame", "boxes":
[{"path", "kind": "box", "center_m", "rotation" (3x3 rows), "half_extents_m",
"collision_enabled"}]}``. A box is the collision geom's own bound (``geom_aabb`` in the
geom frame), a superset of the geom: the fail-closed direction for geometry a scene
adds to the kernel's world.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["body_subtree_box", "static_collider_boxes"]

_MJGEOM_PLANE = 0  # mujoco.mjtGeom.mjGEOM_PLANE


def _base_from_world(data: Any, base_body_id: int) -> NDArray[np.float64]:
    m = np.eye(4)
    m[:3, :3] = np.asarray(data.xmat[base_body_id], dtype=np.float64).reshape(3, 3)
    m[:3, 3] = np.asarray(data.xpos[base_body_id], dtype=np.float64)
    inv: NDArray[np.float64] = np.asarray(np.linalg.inv(m), dtype=np.float64)
    return inv


def _geom_box(
    model: Any, data: Any, g: int, base_from_world: NDArray[np.float64]
) -> dict[str, Any]:
    """Geom ``g``'s own bound as an oriented box in the base frame."""
    aabb = np.asarray(model.geom_aabb[g], dtype=np.float64)  # (cx, cy, cz, hx, hy, hz), geom frame
    rot = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
    center_w = rot @ aabb[:3] + np.asarray(data.geom_xpos[g], dtype=np.float64)
    center = base_from_world[:3, :3] @ center_w + base_from_world[:3, 3]
    return {
        "center_m": [round(float(v), 6) for v in center],
        "rotation": [[round(float(v), 9) for v in row] for row in base_from_world[:3, :3] @ rot],
        "half_extents_m": [round(float(v), 6) for v in aabb[3:]],
    }


def _body_name(mujoco: Any, model: Any, b: int) -> str:
    return str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}")


def _collides(model: Any, g: int) -> bool:
    return bool(model.geom_contype[g]) or bool(model.geom_conaffinity[g])


def static_collider_boxes(
    model: Any, data: Any, *, base_body_id: int, exclude_body_ids: Iterable[int] = ()
) -> dict[str, Any]:
    """Every static collision geom as an oriented box in the base body's frame.

    Static = welded to the world (``body_weldid == 0``: no joint between it and the
    world body) and not mocap-driven (a mocap body is welded too, but the simulation
    moves it); collision = ``contype`` or ``conaffinity`` set. ``exclude_body_ids`` (the
    robot's own bodies) are skipped; an infinite plane has no box and is skipped.
    ``path`` is ``<fixture>/<geom>``: the fixture is the body's top-level ancestor (the
    world's child it hangs from), so every geom of a fixture made of several bodies
    shares one parent (``tools/derive_scene_inputs.py`` groups by it), and the geom is
    named ``<body>.<geom>`` below that ancestor. A geom on the world body itself is its
    own fixture (``world.<geom>/<geom>``), never one group of everything unparented.

    Example:
        >>> import mujoco
        >>> m = mujoco.MjModel.from_xml_string(
        ...     "<mujoco><worldbody><body name='base'/>"
        ...     "<body name='shelf' pos='1 0 0'><geom name='board' type='box' size='.1 .2 .3'/>"
        ...     "<body name='lip'><geom name='rim' type='box' size='.1 .01 .01'/></body></body>"
        ...     "<body name='can'><freejoint/><geom size='.03'/></body>"
        ...     "<body name='tag' mocap='true'><geom size='.01'/></body>"
        ...     "</worldbody></mujoco>"
        ... )
        >>> d = mujoco.MjData(m)
        >>> mujoco.mj_forward(m, d)
        >>> boxes = static_collider_boxes(m, d, base_body_id=1)["boxes"]
        >>> for b in boxes:
        ...     print(b["path"], b["center_m"], b["half_extents_m"])
        shelf/board [1.0, 0.0, 0.0] [0.1, 0.2, 0.3]
        shelf/lip.rim [1.0, 0.0, 0.0] [0.1, 0.01, 0.01]
    """
    import mujoco  # reason: optional sim dep

    base_from_world = _base_from_world(data, base_body_id)
    skip = set(exclude_body_ids)
    boxes = []
    for g in range(int(model.ngeom)):
        b = int(model.geom_bodyid[g])
        if int(model.body_weldid[b]) != 0 or b in skip or not _collides(model, g):
            continue
        if int(model.geom_type[g]) == _MJGEOM_PLANE or int(model.body_mocapid[b]) >= 0:
            continue
        geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        root = b
        while root != 0 and int(model.body_parentid[root]) != 0:
            root = int(model.body_parentid[root])
        if root == 0:
            path = f"world.{geom}/{geom}"
        else:
            fixture = _body_name(mujoco, model, root)
            path = (
                f"{fixture}/{geom}"
                if b == root
                else f"{fixture}/{_body_name(mujoco, model, b)}.{geom}"
            )
        boxes.append(
            {
                "path": path,
                "kind": "box",
                "collision_enabled": True,
                **_geom_box(model, data, g, base_from_world),
            }
        )
    return {"frame": "base_frame", "boxes": boxes}


def body_subtree_box(
    model: Any, data: Any, root_body_ids: Iterable[int], *, base_body_id: int
) -> dict[str, Any] | None:
    """Axis-aligned (base-frame) bound of every collision geom under ``root_body_ids``.

    For an object or a fixture made of several bodies. ``None`` when they hold no
    collision geom.

    Example:
        >>> import mujoco
        >>> m = mujoco.MjModel.from_xml_string(
        ...     "<mujoco><worldbody><body name='base'/>"
        ...     "<body name='can' pos='0.5 0 0.1'><freejoint/><geom size='.03'/></body>"
        ...     "</worldbody></mujoco>"
        ... )
        >>> d = mujoco.MjData(m)
        >>> mujoco.mj_forward(m, d)
        >>> box = body_subtree_box(m, d, [2], base_body_id=1)
        >>> box["center_m"], box["half_extents_m"]
        ([0.5, 0.0, 0.1], [0.03, 0.03, 0.03])
    """
    roots = set(root_body_ids)
    base_from_world = _base_from_world(data, base_body_id)
    corners = []
    for g in range(int(model.ngeom)):
        b = int(model.geom_bodyid[g])
        while b not in roots and b != 0:
            b = int(model.body_parentid[b])
        if b not in roots or not _collides(model, g) or int(model.geom_type[g]) == _MJGEOM_PLANE:
            continue
        box = _geom_box(model, data, g, base_from_world)
        rot = np.asarray(box["rotation"])
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        corners.append(np.asarray(box["center_m"]) + (signs * box["half_extents_m"]) @ rot.T)
    if not corners:
        return None
    pts = np.concatenate(corners)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return {
        "center_m": [round(float(v), 6) for v in (lo + hi) / 2.0],
        "rotation": np.eye(3).tolist(),
        "half_extents_m": [round(float(v), 6) for v in (hi - lo) / 2.0],
    }
