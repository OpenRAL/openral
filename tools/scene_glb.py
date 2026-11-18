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

__all__ = [
    "diffuse_attrs",
    "diffuse_color",
    "fan_triangles",
    "gprim_mesh",
    "static_scene_glb",
    "write_glb",
]

_GLB_MAGIC = 0x46546C67  # "glTF"
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942
_FLOAT, _UINT32 = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


def fan_triangles(counts: Any, indices: Any, holes: Any = ()) -> NDArray[np.uint32]:
    """Fan-triangulate USD faces (``faceVertexCounts`` / ``faceVertexIndices``) into ``(n, 3)``.

    Faces listed in ``holes`` (``holeIndices``) are left out. A fan is exact for convex
    faces; a concave one may draw across its notch (ponytail: stage meshes are triangles
    and quads; ear-clip n-gons if a translucent overview ever shows the difference).

    Example:
        >>> fan_triangles([4, 3], [0, 1, 2, 3, 0, 2, 4], holes=[1]).tolist()
        [[0, 1, 2], [0, 2, 3]]
    """
    tris: list[tuple[int, int, int]] = []
    at = 0
    idx = [int(i) for i in indices]
    skip = {int(h) for h in holes}
    for f, n in enumerate(int(c) for c in counts):
        face = idx[at : at + n]
        if f not in skip:
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
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    return np.eye(3)


def gprim_mesh(
    kind: str,
    *,
    size: float = 2.0,
    radius: float = 1.0,
    height: float = 2.0,
    axis: str = "Z",
    width: float = 2.0,
    length: float = 2.0,
    radius_top: float | None = None,
    radius_bottom: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.uint32]] | None:
    """Local-frame triangles of a USD Gprim.

    ``Cube``, ``Sphere``, ``Cylinder`` / ``Cylinder_1``, ``Cone``, ``Capsule`` /
    ``Capsule_1`` and ``Plane``; defaults are USD's, and the ``_1`` schemas' end radii
    (``radius_top`` / ``radius_bottom``) default to ``radius``. ``None`` for any other
    type. A capsule is drawn as its tube plus a sphere at each end (overlapping, fine for
    a translucent view).

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
    r_bot = radius if radius_bottom is None else radius_bottom
    r_top = radius if radius_top is None else radius_top
    if kind in ("Cylinder", "Cylinder_1", "Cone"):
        tv, tt = _tube(r_bot, -height / 2, 0.0 if kind == "Cone" else r_top, height / 2, seg)
        return tv @ _axis_to(axis).T, tt
    if kind == "Plane":
        w, ln = width / 2, length / 2
        pv = np.array([[-w, -ln, 0], [w, -ln, 0], [w, ln, 0], [-w, ln, 0]], dtype=np.float64)
        return pv @ _axis_to(axis).T, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    if kind in ("Capsule", "Capsule_1"):
        parts = [_tube(r_bot, -height / 2, r_top, height / 2, seg)]
        for rad, z in ((r_bot, -height / 2), (r_top, height / 2)):
            sphere = gprim_mesh("Sphere", radius=rad)
            assert sphere is not None  # reason: Sphere is handled above
            parts.append((sphere[0] + [0, 0, z], sphere[1]))
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

    doc: dict[str, Any] = {"asset": {"version": "2.0", "generator": "openral scene_glb"}}
    if prims:  # glTF arrays are minItems 1: an empty scene is the asset object alone
        doc |= {
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": prims}],
            "materials": materials,
            "accessors": accessors,
            "bufferViews": views,
            "buffers": [{"byteLength": offset}],
        }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    blob = b"".join(bin_parts)
    bin_chunk = struct.pack("<II", len(blob), _BIN_CHUNK) + blob if blob else b""  # no empty BIN
    total = 12 + 8 + len(js) + len(bin_chunk)
    return b"".join([
        struct.pack("<III", _GLB_MAGIC, 2, total),
        struct.pack("<II", len(js), _JSON_CHUNK), js,
        bin_chunk,
    ])  # fmt: skip


def diffuse_attrs(material: Any, *, create: bool = False) -> list[Any]:
    """The attributes that hold a ``UsdShade.Material``'s constant diffuse colour.

    The Material's own ``diffuseColor`` interface input, then each shader's
    ``diffuseColor`` (UsdPreviewSurface) or ``diffuse_color_constant`` (OmniPBR): the
    attribute its connection resolves to (Isaac's URDF importer wires the shader to the
    interface input), else the input's own. ``create`` adds a ``diffuseColor`` input to a
    UsdPreviewSurface that has none, for a caller that writes the colour. Read with
    ``diffuse_color``, written by ``recolor_robot_materials``; one attribute per path.
    """
    from pxr import Sdf, Usd, UsdShade

    out: dict[str, Any] = {}
    iface = material.GetInput("diffuseColor")
    if iface:
        out[str(iface.GetAttr().GetPath())] = iface.GetAttr()
    for child in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(child)
        if not shader:
            continue
        for name in ("diffuseColor", "diffuse_color_constant"):
            inp = shader.GetInput(name)
            preview = shader.GetIdAttr().Get() == "UsdPreviewSurface"
            if not inp and create and name == "diffuseColor" and preview:
                inp = shader.CreateInput(name, Sdf.ValueTypeNames.Color3f)
            if not inp:
                continue
            attrs = (
                inp.GetValueProducingAttributes() if inp.HasConnectedSource() else [inp.GetAttr()]
            )
            out.update((str(a.GetPath()), a) for a in attrs)
    return list(out.values())


def diffuse_color(prim: Any) -> tuple[float, float, float]:
    """A prim's base colour: ``displayColor``, else its bound material's (``diffuse_attrs``).

    Neutral grey when neither holds a value.
    """
    from pxr import UsdGeom, UsdShade

    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    dc = pv.ComputeFlattened() if pv and pv.HasValue() else None  # indexed primvars resolved
    if dc is not None and len(dc):
        return (float(dc[0][0]), float(dc[0][1]), float(dc[0][2]))
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    for attr in diffuse_attrs(mat) if mat else []:
        val = attr.Get()
        if val is not None:
            return (float(val[0]), float(val[1]), float(val[2]))
    return (0.7, 0.7, 0.7)


def static_scene_glb(
    stage: Any,
    frame_from_world: Any,
    skip: list[str],
    nvidia_root: str = "",
    limit_bytes: int | None = None,
) -> dict[str, Any]:
    """The static part of ``stage``'s ``/World`` as one GLB in ``frame_from_world``'s frame.

    Left out: every prim at or under a ``skip`` path, anything at or under an enabled,
    non-kinematic rigid body (it can move; ``scenes/README.md``), invisible prims and
    guide/proxy purposes. ``UsdGeom.Mesh`` is fan-triangulated, Gprims tessellated
    (``gprim_mesh``); one primitive per base colour (``diffuse_color``, quantised to 1/16).

    Licence: when any exported prim takes an opinion from a layer under ``nvidia_root``
    (NVIDIA's Isaac asset root), nothing is converted. Only exported prims count, so an
    ``isaac:`` robot or task object (never exported) does not refuse a custom environment.

    ``limit_bytes`` stops the walk as soon as the geometry alone (12 bytes a vertex and a
    triangle) passes it, before a huge stage is stacked and serialised; the caller still
    checks the exact GLB size.

    Returns ``{"glb": bytes, "triangles", "log"}``, or ``{"note", "log"}`` (NVIDIA
    assets, no static geometry, or over ``limit_bytes``). ``log`` is one summary line.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    m_frame = np.asarray(frame_from_world, dtype=np.float64)
    purposes = {UsdGeom.Tokens.default_, UsdGeom.Tokens.render}
    moving: dict[str, bool] = {}

    def moves(prim: Any) -> bool:
        key = str(prim.GetPath())
        if key not in moving:
            own = False
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = UsdPhysics.RigidBodyAPI(prim)
                own = api.GetRigidBodyEnabledAttr().Get() is not False and not bool(
                    api.GetKinematicEnabledAttr().Get()
                )
            parent = prim.GetParent()
            moving[key] = own or (bool(parent) and str(parent.GetPath()) != "/" and moves(parent))
        return moving[key]

    def attr(prim: Any, name: str, default: Any) -> Any:
        a = prim.GetAttribute(name)
        v = a.Get() if a else None
        return default if v is None else v

    groups: dict[tuple[float, ...], tuple[list[Any], list[Any], list[int]]] = {}
    layers: set[str] = set()
    unsupported: dict[str, int] = {}
    triangles = 0
    vertices = 0
    root = stage.GetPrimAtPath("/World")
    it = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies())) if root else iter(())
    for prim in it:
        path = str(prim.GetPath())
        if any(path == s or path.startswith(s + "/") for s in skip) or moves(prim):
            it.PruneChildren()
            continue
        img = UsdGeom.Imageable(prim)
        if img and img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            it.PruneChildren()  # visibility is inherited: nothing below shows
            continue
        if img and img.ComputePurpose() not in purposes:
            continue  # a child may author its own purpose (render under a proxy Xform)
        kind = str(prim.GetTypeName())
        if kind == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            pts: NDArray[np.float64] = np.asarray(
                mesh.GetPointsAttr().Get() or [], dtype=np.float64
            ).reshape(-1, 3)
            tri = fan_triangles(
                mesh.GetFaceVertexCountsAttr().Get() or [],
                mesh.GetFaceVertexIndicesAttr().Get() or [],
                mesh.GetHoleIndicesAttr().Get() or [],
            )
        elif prim.IsA(UsdGeom.Gprim):
            capsule = kind.startswith("Capsule")
            got = gprim_mesh(
                kind,
                size=float(attr(prim, "size", 2.0)),
                radius=float(attr(prim, "radius", 0.5 if capsule else 1.0)),
                height=float(attr(prim, "height", 1.0 if capsule else 2.0)),
                axis=str(attr(prim, "axis", "Z")),
                width=float(attr(prim, "width", 2.0)),
                length=float(attr(prim, "length", 2.0)),
                radius_top=attr(prim, "radiusTop", None),
                radius_bottom=attr(prim, "radiusBottom", None),
            )
            if got is None:
                unsupported[kind] = unsupported.get(kind, 0) + 1
                continue
            pts, tri = got
        else:
            continue
        if not len(pts) or not len(tri):
            continue
        src = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
        for spec in src.GetPrimStack():
            layers.update(p for p in (spec.layer.identifier, spec.layer.realPath) if p)
        world_from_local = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64,
        ).T  # USD rows
        m = m_frame @ world_from_local
        rgb = tuple(round(min(max(c, 0.0), 1.0) * 16) / 16 for c in diffuse_color(prim))
        verts, tris, base = groups.setdefault(rgb, ([], [], [0]))
        verts.append(pts @ m[:3, :3].T + m[:3, 3])
        tris.append(tri + base[0])
        base[0] += len(pts)
        triangles += len(tri)
        vertices += len(pts)
        if limit_bytes is not None and 12 * (vertices + triangles) > limit_bytes:
            mb = 12 * (vertices + triangles) / 1e6
            return {
                "note": f"scene is over the {limit_bytes / 1e6:.0f} MB limit",
                "log": f"stopped at {mb:.1f} MB of geometry, over {limit_bytes / 1e6:.0f} MB",
            }

    skipped = f", unsupported {unsupported}" if unsupported else ""
    root_dir = nvidia_root.rstrip("/")
    nvidia = sorted(
        lay for lay in layers if root_dir and (lay == root_dir or lay.startswith(root_dir + "/"))
    )
    if nvidia:
        return {
            "note": "uses NVIDIA Isaac Sim assets, which may not be converted",
            "log": f"NVIDIA asset layers {nvidia[:3]}",
        }
    if not triangles:
        return {"note": "the scene has no static geometry", "log": f"0 triangles{skipped}"}
    glb = write_glb([
        ((rgb[0], rgb[1], rgb[2], 1.0), np.vstack(v), np.vstack(t))
        for rgb, (v, t, _) in groups.items()
    ])  # fmt: skip
    return {
        "glb": glb,
        "triangles": triangles,
        "log": f"{triangles} triangles, {len(groups)} colours, {len(glb) / 1e6:.1f} MB{skipped}",
    }
