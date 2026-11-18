"""A plain GLB writer and Gprim tessellation for the console's Sim scene layer (platform#133).

Pure numpy + ``struct``: the Isaac sidecar has no trimesh, and ``omni.kit.asset_converter``
writes 5-18x bigger files (it embeds textures) and drops Gprims. The output is what the
robot console can load under its CSP: one mesh, one primitive per base colour, POSITION
and indices only (no normals, textures, Draco or meshopt). Positions stay in the caller's
frame (ROS, Z-up): no glTF Y-up conversion, the console parents the scene under that frame.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["fan_triangles", "gprim_mesh", "write_glb"]

_GLB_MAGIC = 0x46546C67  # "glTF"
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942
_FLOAT, _UINT32 = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


def fan_triangles(counts: Any, indices: Any) -> NDArray[np.uint32]:
    """Fan-triangulate USD faces (``faceVertexCounts`` / ``faceVertexIndices``) into ``(n, 3)``.

    Example:
        >>> fan_triangles([4], [0, 1, 2, 3]).tolist()
        [[0, 1, 2], [0, 2, 3]]
    """
    tris: list[tuple[int, int, int]] = []
    at = 0
    idx = [int(i) for i in indices]
    for n in (int(c) for c in counts):
        face = idx[at : at + n]
        tris.extend((face[0], face[i], face[i + 1]) for i in range(1, n - 1))
        at += n
    return np.asarray(tris, dtype=np.uint32).reshape(-1, 3)


def _ring(radius: float, z: float, seg: int) -> NDArray[np.float64]:
    a = np.linspace(0.0, 2 * np.pi, seg, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(seg, z)], axis=1)


def _tube(
    r0: float, z0: float, r1: float, z1: float, seg: int
) -> tuple[NDArray[np.float64], NDArray[np.uint32]]:
    """A capped frustum along +Z (a cylinder when ``r0 == r1``, a cone when ``r1 == 0``)."""
    verts = np.vstack([_ring(r0, z0, seg), _ring(r1, z1, seg), [[0, 0, z0], [0, 0, z1]]])
    c0, c1 = 2 * seg, 2 * seg + 1
    tris = []
    for i in range(seg):
        j = (i + 1) % seg
        tris += [(i, j, seg + j), (i, seg + j, seg + i), (c0, j, i), (c1, seg + i, seg + j)]
    return verts, np.asarray(tris, dtype=np.uint32)


def _axis_to(axis: str) -> NDArray[np.float64]:
    """Rotation taking +Z onto the Gprim's ``axis`` token."""
    if axis == "X":
        return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    if axis == "Y":
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    return np.eye(3)


def gprim_mesh(
    kind: str, *, size: float = 2.0, radius: float = 1.0, height: float = 2.0, axis: str = "Z"
) -> tuple[NDArray[np.float64], NDArray[np.uint32]] | None:
    """Local-frame triangles of a USD Gprim (``Cube``, ``Sphere``, ``Cylinder``, ``Cone``, ``Capsule``).

    Defaults are USD's. ``None`` for any other type. A capsule is drawn as its
    cylinder plus a sphere at each end (overlapping, fine for a translucent view).

    Example:
        >>> v, t = gprim_mesh("Cube", size=1.0)
        >>> float(v.min()), float(v.max()), t.shape
        (-0.5, 0.5, (12, 3))
    """
    seg = 16
    if kind == "Cube":
        h = size / 2
        v = np.array(
            [[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], dtype=np.float64
        )
        t = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]  # fmt: skip
        return v, np.asarray(t, dtype=np.uint32)
    if kind == "Sphere":
        rings = [
            _ring(radius * np.sin(p), radius * np.cos(p), seg)
            for p in np.linspace(0, np.pi, 9)[1:-1]
        ]
        v = np.vstack([[[0, 0, radius]], *rings, [[0, 0, -radius]]])
        n = len(rings)
        t = [(0, 1 + i, 1 + (i + 1) % seg) for i in range(seg)]
        for r in range(n - 1):
            a, b = 1 + r * seg, 1 + (r + 1) * seg
            for i in range(seg):
                j = (i + 1) % seg
                t += [(a + i, b + i, b + j), (a + i, b + j, a + j)]
        last, bottom = 1 + (n - 1) * seg, len(v) - 1
        t += [(bottom, last + (i + 1) % seg, last + i) for i in range(seg)]
        return v, np.asarray(t, dtype=np.uint32)
    if kind in ("Cylinder", "Cone"):
        tv, tt = _tube(radius, -height / 2, radius if kind == "Cylinder" else 0.0, height / 2, seg)
        return tv @ _axis_to(axis).T, tt
    if kind == "Capsule":
        parts = [_tube(radius, -height / 2, radius, height / 2, seg)]
        sphere = gprim_mesh("Sphere", radius=radius)
        assert sphere is not None  # reason: Sphere is handled above
        parts += [(sphere[0] + [0, 0, z], sphere[1]) for z in (-height / 2, height / 2)]
        offsets = np.cumsum([0] + [len(pv) for pv, _ in parts[:-1]])
        verts = np.vstack([pv for pv, _ in parts])
        tris = np.vstack([pt + o for (_, pt), o in zip(parts, offsets, strict=True)])
        return verts @ _axis_to(axis).T, tris.astype(np.uint32)
    return None


def write_glb(primitives: list[tuple[tuple[float, float, float, float], Any, Any]]) -> bytes:
    """One GLB holding one mesh with a primitive per ``(rgba, positions (n, 3), triangles (m, 3))``.

    Each primitive gets its own unlit-looking PBR material (``baseColorFactor`` = ``rgba``,
    metallic 0, roughness 1). Empty primitives are dropped.

    Example:
        >>> v, t = gprim_mesh("Cube")
        >>> glb = write_glb([((0.5, 0.5, 0.5, 1.0), v, t)])
        >>> glb[:4], len(glb) % 4
        (b'glTF', 0)
    """
    bin_parts: list[bytes] = []
    offset = 0
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    prims: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []

    def add(data: bytes, target: int) -> int:
        nonlocal offset
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        pad = (-len(data)) % 4
        bin_parts.append(data + b"\0" * pad)
        offset += len(data) + pad
        return len(views) - 1

    for rgba, positions, triangles in primitives:
        pos = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        tri = np.ascontiguousarray(triangles, dtype=np.uint32).reshape(-1)
        if not len(pos) or not len(tri):
            continue
        accessors.append({
            "bufferView": add(pos.tobytes(), _ARRAY_BUFFER), "componentType": _FLOAT,
            "count": len(pos), "type": "VEC3",
            "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist(),
        })  # fmt: skip
        accessors.append({
            "bufferView": add(tri.tobytes(), _ELEMENT_ARRAY_BUFFER), "componentType": _UINT32,
            "count": len(tri), "type": "SCALAR",
        })  # fmt: skip
        materials.append({
            "pbrMetallicRoughness": {
                "baseColorFactor": [float(c) for c in rgba], "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
        })  # fmt: skip
        prims.append({
            "attributes": {"POSITION": len(accessors) - 2}, "indices": len(accessors) - 1,
            "material": len(materials) - 1,
        })  # fmt: skip

    doc: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "openral scene_glb"},
        "scene": 0,
        "scenes": [{"nodes": [0] if prims else []}],
        "nodes": [{"mesh": 0}] if prims else [],
        "meshes": [{"primitives": prims}] if prims else [],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    blob = b"".join(bin_parts)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return b"".join([
        struct.pack("<III", _GLB_MAGIC, 2, total),
        struct.pack("<II", len(js), _JSON_CHUNK), js,
        struct.pack("<II", len(blob), _BIN_CHUNK), blob,
    ])  # fmt: skip
