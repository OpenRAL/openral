"""A plain GLB writer and Gprim tessellation for the console's Sim scene layer (platform#133).

Pure numpy + ``struct``: the Isaac sidecar has no trimesh, and ``omni.kit.asset_converter``
writes 5-18x bigger files (it embeds full-size textures) and drops Gprims. The output is
what the robot console can load under its CSP: one mesh, one primitive per base colour or
per base-colour texture (embedded, downscaled to ``TEXTURE_MAX_PX``), POSITION, indices and
TEXCOORD_0 only (no normals, Draco or meshopt). Positions stay in the caller's frame (ROS,
Z-up): no glTF Y-up conversion, the console parents the scene under that frame.

Two exporters: ``static_scene_glb`` (an Isaac USD stage, platform#133 / #137) and
``mujoco_static_scene_glb`` (a live MuJoCo model, platform#139).
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
    "encode_texture",
    "face_uvs",
    "fan_triangles",
    "gprim_mesh",
    "mujoco_static_scene_glb",
    "static_scene_glb",
    "texture_path",
    "write_glb",
]

_GLB_MAGIC = 0x46546C67  # "glTF"
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942
_FLOAT, _UINT32 = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
#: The longest side an embedded texture keeps: 512 px adds ~0.4 MB to the shelf aisle.
TEXTURE_MAX_PX = 512


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


def write_glb(primitives: list[tuple[Any, ...]]) -> bytes:
    """One GLB holding one mesh with a primitive per ``(rgba, positions (n, 3), triangles (m, 3))``.

    Each primitive gets its own unlit-looking PBR material (``baseColorFactor`` = ``rgba``,
    metallic 0, roughness 1). A textured primitive carries two more items,
    ``uvs (n, 2)`` (glTF convention: v down) and ``(image bytes, mime type)``: its image is
    embedded and becomes the material's ``baseColorTexture`` (repeat wrap). Empty
    primitives are dropped.

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
    images: list[dict[str, Any]] = []

    def add(data: bytes, target: int | None) -> int:
        nonlocal offset
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        views.append(view if target is None else view | {"target": target})
        pad = (-len(data)) % 4
        bin_parts.append(data + b"\0" * pad)
        offset += len(data) + pad
        return len(views) - 1

    for rgba, positions, triangles, *textured in primitives:
        pos = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        tri = np.ascontiguousarray(triangles, dtype=np.uint32).reshape(-1)
        if not len(pos) or not len(tri):
            continue
        attributes: dict[str, int] = {}
        pbr: dict[str, Any] = {}
        if textured:
            uvs, (data, mime) = textured
            uv = np.ascontiguousarray(uvs, dtype=np.float32).reshape(-1, 2)
            if len(uv) != len(pos):
                raise ValueError(f"{len(uv)} uvs for {len(pos)} positions")
            accessors.append({
                "bufferView": add(uv.tobytes(), _ARRAY_BUFFER), "componentType": _FLOAT,
                "count": len(uv), "type": "VEC2",
            })  # fmt: skip
            attributes["TEXCOORD_0"] = len(accessors) - 1
            images.append({"bufferView": add(data, None), "mimeType": mime})
            pbr["baseColorTexture"] = {"index": len(images) - 1}
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
            "pbrMetallicRoughness": pbr | {
                "baseColorFactor": [float(c) for c in rgba], "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
        })  # fmt: skip
        prims.append({
            "attributes": attributes | {"POSITION": len(accessors) - 2},
            "indices": len(accessors) - 1, "material": len(materials) - 1,
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
        if images:  # one texture per image, the default (repeat) sampler
            doc |= {"images": images, "textures": [{"source": i} for i in range(len(images))]}
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

    Each shader's ``diffuseColor`` (UsdPreviewSurface) or ``diffuse_color_constant``
    (OmniPBR) first: the input its connection resolves to (Isaac's URDF importer wires the
    shader to a Material interface input), else the input's own; then the Material's own
    ``diffuseColor`` interface input. Only writable inputs: a connection that resolves to
    a computed output (a texture's ``rgb``) contributes nothing. ``create`` adds a
    ``diffuseColor`` input to a UsdPreviewSurface that has none, for a caller that writes
    the colour. Read with ``diffuse_source``, written by ``recolor_robot_materials``.
    """
    from pxr import Sdf, Usd, UsdShade

    out: dict[str, Any] = {}
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
            out.update((str(a.GetPath()), a) for a in attrs if UsdShade.Input.IsInput(a))
    iface = material.GetInput("diffuseColor")
    if iface:
        out.setdefault(str(iface.GetAttr().GetPath()), iface.GetAttr())
    return list(out.values())


GREY = (0.7, 0.7, 0.7)


def diffuse_source(prim: Any) -> tuple[tuple[float, float, float], Any]:
    """A prim's base colour and the material attribute it came from.

    ``displayColor`` (flattened, so an indexed primvar resolves) with ``None``; else the
    first value among its bound material's ``diffuse_attrs``, with that attribute; else
    ``GREY`` with ``None``.
    """
    from pxr import UsdGeom, UsdShade

    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    dc = pv.ComputeFlattened() if pv and pv.HasValue() else None
    if dc is not None and len(dc):
        return (float(dc[0][0]), float(dc[0][1]), float(dc[0][2])), None
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    for attr in diffuse_attrs(mat) if mat else []:
        val = attr.Get()
        if val is not None:
            return (float(val[0]), float(val[1]), float(val[2])), attr
    return GREY, None


def diffuse_color(prim: Any) -> tuple[float, float, float]:
    """A prim's base colour (``diffuse_source`` without the attribute)."""
    return diffuse_source(prim)[0]


def _under(path: str, root: str) -> bool:
    """``path`` is ``root`` or below it (at a ``/`` boundary); never for an empty root."""
    return bool(root) and (path == root or path.startswith(root + "/"))


def texture_path(prim: Any) -> str | None:
    """The resolved file of a prim's bound base-colour texture, or ``None``.

    A ``UsdUVTexture`` wired to a UsdPreviewSurface's ``diffuseColor``, a MaterialX
    ``ND_tiledimage_color3`` ``file``, or an OmniPBR (MDL) ``diffuse_texture``. Only a file
    on disk counts (a texture packed in a ``.usdz`` is skipped).
    """
    import os

    from pxr import Usd, UsdShade

    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    for child in Usd.PrimRange(mat.GetPrim()) if mat else []:
        shader = UsdShade.Shader(child)
        if not shader:
            continue
        sid = shader.GetIdAttr().Get()
        file_input = None
        if sid == "UsdPreviewSurface":
            diffuse = shader.GetInput("diffuseColor")
            if diffuse and diffuse.HasConnectedSource():
                file_input = UsdShade.Shader(diffuse.GetConnectedSource()[0].GetPrim()).GetInput(
                    "file"
                )
        elif sid == "ND_tiledimage_color3":
            file_input = shader.GetInput("file")
        else:
            file_input = shader.GetInput("diffuse_texture")
        asset = file_input.Get() if file_input else None
        path = str(getattr(asset, "resolvedPath", "") or "")
        if path and os.path.isfile(path):
            return path
    return None


def face_uvs(prim: Any, indices: Any, n_points: int) -> NDArray[np.float64] | None:
    """A mesh's texture coordinates per face-vertex, flipped to glTF's v-down, or ``None``.

    The ``st`` / ``st0`` / ``UVMap`` primvar, else the first ``texCoord2f[]`` one;
    ``faceVarying`` is used as is, ``vertex`` / ``varying`` is gathered through
    ``indices`` (``faceVertexIndices``). Any other interpolation, or a count that does not
    match, is ``None``.
    """
    from pxr import Sdf, UsdGeom

    api = UsdGeom.PrimvarsAPI(prim)
    named = [api.GetPrimvar(n) for n in ("st", "st0", "UVMap")]
    typed = [p for p in api.GetPrimvars() if p.GetTypeName() == Sdf.ValueTypeNames.TexCoord2fArray]
    pv = next((p for p in [*named, *typed] if p and p.HasValue()), None)
    if pv is None:
        return None
    st: NDArray[np.float64] = np.asarray(pv.ComputeFlattened(), dtype=np.float64).reshape(-1, 2)
    idx = np.asarray(indices, dtype=np.int64)
    interp = pv.GetInterpolation()
    if interp == UsdGeom.Tokens.faceVarying and len(st) == len(idx):
        fv = st
    elif interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying) and len(st) == n_points:
        fv = st[idx]
    else:
        return None
    return np.stack([fv[:, 0], 1.0 - fv[:, 1]], axis=1)


def encode_texture(path: str, max_px: int = TEXTURE_MAX_PX) -> tuple[bytes, str]:
    """An image file shrunk to ``max_px`` on its longer side: JPEG, or PNG when it has alpha."""
    import io

    from PIL import Image

    with Image.open(path) as im:
        im.thumbnail((max_px, max_px))
        rgba = im.convert("RGBA")
    buf = io.BytesIO()
    if np.asarray(rgba.getchannel("A")).min() < 255:  # 8-bit alpha
        rgba.save(buf, "PNG", optimize=True)
        return buf.getvalue(), "image/png"
    rgba.convert("RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


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
    A mesh with a base-colour texture (``texture_path``) and UVs (``face_uvs``) goes in one
    primitive per texture file instead, the image embedded at ``TEXTURE_MAX_PX``
    (``encode_texture``). Without Pillow, or for a texture under ``nvidia_root``, it keeps
    its base colour.

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

    try:
        import PIL  # noqa: F401  # reason: textures are optional
    except ImportError:
        textures_on = False
    else:
        textures_on = True
    groups: dict[tuple[float, ...], tuple[list[Any], list[Any], list[int]]] = {}
    textured: dict[str, tuple[list[Any], list[Any], list[Any], list[int]]] = {}
    nvidia_textures = unreadable = 0
    layers: set[str] = set()
    unsupported: dict[str, int] = {}
    triangles = 0
    vertices = 0
    nvidia_colours = 0
    root_dir = nvidia_root.rstrip("/")
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
        uv: NDArray[np.float64] | None = None
        tex = None
        if kind == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            pts: NDArray[np.float64] = np.asarray(
                mesh.GetPointsAttr().Get() or [], dtype=np.float64
            ).reshape(-1, 3)
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
            holes = mesh.GetHoleIndicesAttr().Get() or []
            tex = texture_path(prim) if textures_on else None
            if tex is not None and _under(tex, root_dir):
                nvidia_textures += 1  # an NVIDIA texture is not embedded (licence)
                tex = None
            uv = face_uvs(prim, idx, len(pts)) if tex is not None and len(pts) else None
            if uv is not None:  # unshared vertices: a UV per face corner
                pts, tri = pts[idx], fan_triangles(counts, np.arange(len(idx)), holes)
            else:
                tri = fan_triangles(counts, idx, holes)
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
        colour, source = diffuse_source(prim)
        if source is not None and any(
            _under(p, root_dir)
            for spec in source.GetPropertyStack()
            for p in (spec.layer.identifier, spec.layer.realPath)
            if p
        ):
            colour = GREY  # an NVIDIA material's colour is not baked in (licence)
            nvidia_colours += 1
        world = pts @ m[:3, :3].T + m[:3, 3]
        if uv is not None and tex is not None:
            verts, uvs, tris, base = textured.setdefault(tex, ([], [], [], [0]))
            uvs.append(uv)
        else:
            rgb = tuple(round(min(max(c, 0.0), 1.0) * 16) / 16 for c in colour)
            verts, tris, base = groups.setdefault(rgb, ([], [], [0]))
        verts.append(world)
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
    if nvidia_colours:
        skipped += f", {nvidia_colours} NVIDIA material colours drawn grey"
    if nvidia_textures:
        skipped += f", {nvidia_textures} NVIDIA textures left out"
    if not textures_on:
        skipped += ", no Pillow: textures left out"
    nvidia = sorted(lay for lay in layers if _under(lay, root_dir))
    if nvidia:
        return {
            "note": "uses NVIDIA Isaac Sim assets, which may not be converted",
            "log": f"NVIDIA asset layers {nvidia[:3]}",
        }
    if not triangles:
        return {"note": "the scene has no static geometry", "log": f"0 triangles{skipped}"}
    prims: list[tuple[Any, ...]] = [
        ((rgb[0], rgb[1], rgb[2], 1.0), np.vstack(v), np.vstack(t))
        for rgb, (v, t, _) in groups.items()
    ]
    for path, (v, u, t, _) in textured.items():
        try:
            image = encode_texture(path)
        except OSError:  # unreadable or not an image: the geometry still draws, grey
            unreadable += 1
            prims.append(((*GREY, 1.0), np.vstack(v), np.vstack(t)))
        else:
            prims.append(((1.0, 1.0, 1.0, 1.0), np.vstack(v), np.vstack(t), np.vstack(u), image))
    if unreadable:
        skipped += f", {unreadable} unreadable textures drawn grey"
    glb = write_glb(prims)
    return {
        "glb": glb,
        "triangles": triangles,
        "log": (
            f"{triangles} triangles, {len(groups)} colours, {len(textured) - unreadable} textures, "
            f"{len(glb) / 1e6:.1f} MB{skipped}"
        ),
    }


def _mujoco_geom_mesh(
    model: Any, geom: int
) -> tuple[NDArray[np.float64], NDArray[np.uint32]] | None:
    """A MuJoCo geom's triangles in its own frame, or ``None`` (height field, SDF).

    MuJoCo sizes are half-extents; a plane of size 0 (infinite) is drawn 10 m square.
    """
    import mujoco

    g = mujoco.mjtGeom
    kind, size = int(model.geom_type[geom]), np.asarray(model.geom_size[geom], dtype=np.float64)
    if kind == g.mjGEOM_BOX:
        v, t = gprim_mesh("Cube", size=2.0)  # type: ignore[misc]  # reason: Cube is never None
        return v * size, t
    if kind in (g.mjGEOM_SPHERE, g.mjGEOM_ELLIPSOID):
        v, t = gprim_mesh("Sphere", radius=1.0)  # type: ignore[misc]  # reason: never None
        return v * (size if kind == g.mjGEOM_ELLIPSOID else size[0]), t
    if kind in (g.mjGEOM_CYLINDER, g.mjGEOM_CAPSULE):
        name = "Cylinder" if kind == g.mjGEOM_CYLINDER else "Capsule"
        return gprim_mesh(name, radius=float(size[0]), height=2 * float(size[1]))
    if kind == g.mjGEOM_PLANE:
        half = [float(s) or 5.0 for s in size[:2]]
        return gprim_mesh("Plane", width=2 * half[0], length=2 * half[1])
    if kind == g.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        return (
            np.asarray(model.mesh_vert[va : va + vn], dtype=np.float64),
            np.asarray(model.mesh_face[fa : fa + fn], dtype=np.uint32),
        )
    return None


def mujoco_static_scene_glb(
    model: Any, data: Any, frame_from_world: Any, limit_bytes: int | None = None
) -> dict[str, Any]:
    """The static part of a live MuJoCo scene as one GLB in ``frame_from_world``'s frame.

    Static: a geom on a body welded to the world (``body_weldid == 0``: no joint between it
    and the world), at ``data``'s pose. Left out: the robot (every top-level body, a child
    of the world, whose subtree holds an actuated joint, base and mount included) and
    mocap bodies with what hangs off them (they are driven). A
    top-level body with visual-only geoms (``contype == conaffinity == 0``) draws only those,
    so collision proxies do not double its surfaces. Transparent geoms, height fields and
    SDFs are skipped.

    Colour: the material's ``rgba`` (else the geom's), times its RGB texture's mean colour,
    quantised to 1/16: RoboCasa and LIBERO texture ~98 % of their geoms on a white material,
    and MuJoCo's cube and procedural textures have no UVs to embed (ponytail: the mean, not
    the image; embed 2D textures on meshes with UVs if the flat view is not enough).

    ``limit_bytes`` as in ``static_scene_glb``, and the exact GLB size is checked here.
    Returns ``{"glb": bytes, "triangles", "log"}`` or ``{"note", "log"}``.
    """
    import mujoco

    m_frame = np.asarray(frame_from_world, dtype=np.float64)
    root = np.asarray(model.body_rootid, dtype=np.int64)  # the top-level body above each body
    joint_kinds = (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT)
    robot = {
        int(root[int(model.jnt_bodyid[int(model.actuator_trnid[a, 0])])])
        for a in range(model.nu)
        if int(model.actuator_trntype[a]) in joint_kinds
    }
    mocap = {b for b in range(model.nbody) if int(model.body_mocapid[b]) >= 0}  # driven
    by_top: dict[int, list[int]] = {}
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if not model.body_weldid[body] and int(root[body]) not in robot | mocap:
            by_top.setdefault(int(root[body]), []).append(geom)

    mean_rgb: dict[int, NDArray[np.float64]] = {}
    groups: dict[tuple[float, ...], tuple[list[Any], list[Any], list[int]]] = {}
    triangles = vertices = 0
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    for geoms in by_top.values():
        visual = [g for g in geoms if not model.geom_contype[g] and not model.geom_conaffinity[g]]
        for geom in visual or geoms:
            mat = int(model.geom_matid[geom])
            rgba = np.array(model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[geom], float)
            if rgba[3] <= 0.01:  # an invisible (site-like) geom
                continue
            tex = int(model.mat_texid[mat, rgb_role]) if mat >= 0 else -1
            if tex >= 0:
                if tex not in mean_rgb:
                    adr, nch = int(model.tex_adr[tex]), int(model.tex_nchannel[tex])
                    n = int(model.tex_width[tex]) * int(model.tex_height[tex]) * nch
                    px = np.asarray(model.tex_data[adr : adr + n]).reshape(-1, nch)
                    mean_rgb[tex] = px[:, :3].mean(axis=0) / 255.0  # a grey texture: 1 channel
                rgba[:3] *= mean_rgb[tex]
            got = _mujoco_geom_mesh(model, geom)
            if got is None:
                continue
            pts, tri = got
            world_from_geom = np.eye(4)
            world_from_geom[:3, :3] = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
            world_from_geom[:3, 3] = data.geom_xpos[geom]
            m = m_frame @ world_from_geom
            rgb = tuple(round(min(max(float(c), 0.0), 1.0) * 16) / 16 for c in rgba[:3])
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
    if not triangles:
        return {"note": "the scene has no static geometry", "log": "0 triangles"}
    glb = write_glb([
        ((rgb[0], rgb[1], rgb[2], 1.0), np.vstack(v), np.vstack(t))
        for rgb, (v, t, _) in groups.items()
    ])  # fmt: skip
    log = f"{triangles} triangles, {len(groups)} colours, {len(glb) / 1e6:.1f} MB"
    if limit_bytes is not None and len(glb) > limit_bytes:
        over = f"scene is {len(glb) / 1e6:.1f} MB, over the {limit_bytes / 1e6:.0f} MB limit"
        return {"note": over, "log": log}
    return {"glb": glb, "triangles": triangles, "log": log}
