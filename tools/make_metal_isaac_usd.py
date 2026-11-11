"""Write the USD assets of the two MakerMods Metal Isaac example scenes from the MuJoCo twins.

``scenes/sim/isaac_makermods_metal_pen_holder.yaml`` (the two-arm station) and
``scenes/sim/isaac_makermods_metal_pick_place.yaml`` (the single arm, the orange cube) reuse
the MuJoCo twins' measured geometry instead of a copy of it: every size, pose, mass, friction
and colour below comes from ``MetalPenHolderOptions`` / ``MetalPickPlaceOptions`` defaults,
and the pen holder's look is ``holder_visual_mesh`` (the same tapered rounded tube, hex holes
cut through, one solid wall) with ``holder_collider_boxes`` as its colliders. Change a value
in the options model and re-run::

    python tools/make_metal_isaac_usd.py

Two Isaac 6.1 RTX traps shape the textured surfaces:

* RTX ignores a texture's ``wrap: repeat`` (the desk rendered one stretched tile), so each
  tiled texture is baked to one image covering the whole surface
  (``textures/*_baked.jpg``, ``bake_tiled``).
* RTX ignores emissive textures, so the MuJoCo twins' shadow-free ``emission`` share is not
  reproduced; the scenes' dome light carries it instead.

Output (text USDA plus baked textures, committed):

* ``scenes/sim/assets/makermods_metal_pen_holder/``: ``table.usda`` (desk with the baked
  wood top, riser shelf with the real board, back wall, the room floor; the scene's
  environment, which replaces Isaac's default ground plane), ``pen_holder.usda`` (rigid body),
  ``pen.usda`` (rigid body: square-section box collider, glossy black barrel, printed
  sleeve).
* ``scenes/sim/assets/makermods_metal_pick_place/``: ``table.usda`` (desk, wall, floor; the
  scene's environment),
  ``cube.usda`` (rigid body), ``marker.usda`` (the target sticker: visual only).

Frames: the pen-holder assets are in ``station_mid`` (the station's ``station_base``), the
pick-place ones in the single arm's ``base_link``; both x forward, z up, metres.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
for _src in ("python/sim/src", "python/core/src", "python/hal/src"):
    if str(_REPO / _src) not in sys.path:
        sys.path.insert(0, str(_REPO / _src))

from openral_sim.backends.metal_pen_holder._assets import (  # noqa: E402
    _PEN_SLEEVE_SPAN,
    _PEN_SLEEVE_WRAPS,
    MetalPenHolderOptions,
    holder_collider_boxes,
    holder_visual_mesh,
)
from openral_sim.backends.metal_pick_place._assets import MetalPickPlaceOptions  # noqa: E402

PEN_DIR = _REPO / "scenes/sim/assets/makermods_metal_pen_holder"
CUBE_DIR = _REPO / "scenes/sim/assets/makermods_metal_pick_place"
_BAKE_MAX_PX = 3072  # baked longest side (px): <= 0.5 mm/px, each image under 2 MB
_GLOSS_ROUGHNESS = 0.35  # the desks' lacquer (the MuJoCo twins' table_specular sheen)
_USD_HOLE_GRID = 64  # pen-holder wall cells per row / column in the committed USD


def linear(c: float) -> float:
    """One sRGB channel in [0, 1] as the linear value ``UsdPreviewSurface`` expects.

    Example:
        >>> round(linear(0.5), 4), linear(0.0), linear(1.0)
        (0.214, 0.0, 1.0)
    """
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _bright(rgb: tuple[float, float, float], gain: float) -> tuple[float, float, float]:
    """A linear colour times ``gain``, capped at 1."""
    r, g, b = (round(min(1.0, gain * c), 4) for c in rgb)
    return (r, g, b)


def _rgb(rgba: tuple[float, ...]) -> tuple[float, float, float]:
    return (round(linear(rgba[0]), 4), round(linear(rgba[1]), 4), round(linear(rgba[2]), 4))


def bake_tiled(src: Path, size_m: tuple[float, float], tile_m: tuple[float, float]) -> Image.Image:
    """One image of ``src`` tiled over a ``size_m`` (x, y) surface, ``tile_m`` per tile.

    Columns run along +y, rows along -x (the image's top row is the surface's far edge), so a
    quad with u = (y - y0) / size_y and v = (x - x0) / size_x shows it as the MuJoCo twin does.

    Example:
        >>> import tempfile
        >>> p = Path(tempfile.mkdtemp()) / "t.png"
        >>> Image.new("RGB", (10, 10)).save(p)
        >>> bake_tiled(p, (0.2, 0.4), (0.1, 0.1)).size
        (3072, 1536)
    """
    tile = np.asarray(Image.open(src).convert("RGB"))
    nx, ny = size_m[0] / tile_m[1], size_m[1] / tile_m[0]
    scale = _BAKE_MAX_PX / max(nx, ny) / max(tile.shape[:2])
    th, tw = max(1, round(tile.shape[0] * scale)), max(1, round(tile.shape[1] * scale))
    tile = np.asarray(Image.fromarray(tile).resize((tw, th), Image.Resampling.LANCZOS))
    reps = (math.ceil(nx) + 1, math.ceil(ny) + 1, 1)
    big = np.tile(tile, reps)[: round(nx * th), : round(ny * tw)]
    return Image.fromarray(big)


def _header(default_prim: str) -> str:
    return (
        f'#usda 1.0\n(\n    defaultPrim = "{default_prim}"\n    metersPerUnit = 1\n'
        '    upAxis = "Z"\n)\n\n'
    )


def _surface(
    name: str,
    rgb: tuple[float, float, float],
    roughness: float,
    texture: str | None = None,
    tex_scale: float = 1.0,
) -> str:
    """A ``UsdPreviewSurface`` material (diffuse colour or an sRGB texture over primvar st).

    ``tex_scale`` multiplies the texture's (linear) colour: the brightness match to the
    MuJoCo twin, whose region means were fitted to the real frames.
    """
    if texture is None:
        diffuse = f"            color3f inputs:diffuseColor = ({rgb[0]}, {rgb[1]}, {rgb[2]})\n"
        tex = ""
    else:
        diffuse = "            color3f inputs:diffuseColor.connect = <Tex.outputs:rgb>\n"
        tex = (
            '        def Shader "Tex"\n        {\n'
            '            uniform token info:id = "UsdUVTexture"\n'
            f"            asset inputs:file = @{texture}@\n"
            '            token inputs:sourceColorSpace = "sRGB"\n'
            f"            float4 inputs:scale = ({tex_scale}, {tex_scale}, {tex_scale}, 1)\n"
            "            float2 inputs:st.connect = <St.outputs:result>\n"
            "            float3 outputs:rgb\n        }\n"
            '        def Shader "St"\n        {\n'
            '            uniform token info:id = "UsdPrimvarReader_float2"\n'
            '            string inputs:varname = "st"\n'
            "            float2 outputs:result\n        }\n"
        )
    return (
        (
            f'    def Material "{name}"\n    {{\n'
            f"        token outputs:surface.connect = <Shader.outputs:surface>\n"
            '        def Shader "Shader"\n        {\n'
            '            uniform token info:id = "UsdPreviewSurface"\n'
            + diffuse
            + f"            float inputs:roughness = {roughness}\n"
            "            float inputs:metallic = 0\n"
            "            token outputs:surface\n        }\n" + tex + "    }\n"
        )
        .replace("<Shader.", f"<{{ROOT}}/{name}/Shader.")
        .replace("<Tex.", f"<{{ROOT}}/{name}/Tex.")
        .replace("<St.", f"<{{ROOT}}/{name}/St.")
    )


def _physics_material(friction: float) -> str:
    return (
        '    def Material "PhysicsMaterial" (\n'
        '        prepend apiSchemas = ["PhysicsMaterialAPI"]\n    )\n    {\n'
        f"        float physics:staticFriction = {friction}\n"
        f"        float physics:dynamicFriction = {friction}\n"
        "        float physics:restitution = 0\n    }\n"
    )


def _box(
    name: str,
    xyz: tuple[float, ...],
    half: tuple[float, ...],
    *,
    quat_wxyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    material: str | None = None,
    collide: bool = True,
    visible: bool = True,
) -> str:
    """A cube prim of half sizes ``half`` at ``xyz`` (collider and/or visual)."""
    api = '        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]\n'
    return (
        f'    def Cube "{name}"' + (f" (\n{api}    )\n" if collide else "\n") + "    {\n"
        "        double size = 2\n"
        + ("" if visible else '        uniform token purpose = "guide"\n')
        + f"        double3 xformOp:translate = ({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f})\n"
        f"        quatf xformOp:orient = ({quat_wxyz[0]:.6f}, {quat_wxyz[1]:.6f}, "
        f"{quat_wxyz[2]:.6f}, {quat_wxyz[3]:.6f})\n"
        f"        float3 xformOp:scale = ({half[0]:.6f}, {half[1]:.6f}, {half[2]:.6f})\n"
        '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", '
        '"xformOp:scale"]\n'
        + (f"        rel material:binding = <{{ROOT}}/{material}>\n" if material else "")
        + (
            "        rel material:binding:physics = <{ROOT}/PhysicsMaterial> (\n"
            '            bindMaterialAs = "weakerThanDescendants"\n        )\n'
            if collide
            else ""
        )
        + "    }\n"
    )


def _mesh(
    name: str, verts: list[float], uv: list[float] | None, faces: list[int], material: str
) -> str:
    """A visual triangle mesh (face-varying texcoords from per-vertex ``uv``)."""
    pts = ",".join(
        f"({verts[i]:.4f},{verts[i + 1]:.4f},{verts[i + 2]:.4f})" for i in range(0, len(verts), 3)
    )
    idx = ",".join(str(i) for i in faces)
    st = ""
    if uv is not None:
        st_vals = ", ".join(f"({uv[2 * i]:.5f}, {uv[2 * i + 1]:.5f})" for i in faces)
        st = f'        texCoord2f[] primvars:st = [{st_vals}] (\n            interpolation = "faceVarying"\n        )\n'
    return (
        f'    def Mesh "{name}" (\n        prepend apiSchemas = ["MaterialBindingAPI"]\n    )\n    {{\n'
        f"        int[] faceVertexCounts = [{','.join(['3'] * (len(faces) // 3))}]\n"
        f"        int[] faceVertexIndices = [{idx}]\n"
        f"        point3f[] points = [{pts}]\n"
        + st
        + '        uniform token subdivisionScheme = "none"\n'
        f"        rel material:binding = <{{ROOT}}/{material}>\n    }}\n"
    )


def _quad(
    name: str,
    x: tuple[float, float],
    y: tuple[float, float],
    z: float,
    material: str,
    *,
    u_along_x: bool = False,
) -> str:
    """A textured upward quad over [x0, x1] x [y0, y1] at height z.

    Default: u along +y, v along +x (``bake_tiled``'s layout). ``u_along_x``: u along +x, v
    along -y (an image whose columns are world x and rows world y, top row at +y).
    """
    (x0, x1), (y0, y1) = x, y
    verts = [x0, y0, z, x0, y1, z, x1, y1, z, x1, y0, z]
    if u_along_x:
        uv = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    else:
        uv = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    return _mesh(name, verts, uv, [0, 3, 2, 0, 2, 1], material)


def _rigid(root: str, mass: float, friction: float, children: str) -> str:
    # 32 position iterations: a thin pen pinched near its end between two fingers slips out
    # of the grip at the default solver count as the arm lifts it.
    body = (
        f'def Xform "{root}" (\n'
        '    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", '
        '"PhysxRigidBodyAPI"]\n)\n{\n'
        f"    float physics:mass = {mass}\n"
        "    int physxRigidBody:solverPositionIterationCount = 32\n"
        "    int physxRigidBody:solverVelocityIterationCount = 4\n"
        + children
        + _physics_material(friction)
        + "}\n"
    )
    return _header(root) + body.replace("{ROOT}", f"/{root}")


def _static(root: str, friction: float, children: str) -> str:
    body = f'def Xform "{root}"\n{{\n' + children + _physics_material(friction) + "}\n"
    return _header(root) + body.replace("{ROOT}", f"/{root}")


def _clutter(items: tuple[Any, ...]) -> str:
    """The twin's contact-free props (``ClutterItem``: drawer unit, wall pipes), visual only."""
    out = ""
    for k, c in enumerate(items):
        mat = f"Clutter{k}"
        out += _surface(mat, _rgb(c.rgba), 0.8)
        if c.kind == "box":
            out += _box(f"clutter_{k}", c.pos, c.size, material=mat, collide=False)
        else:
            out += (
                f'    def Cylinder "clutter_{k}" (\n'
                '        prepend apiSchemas = ["MaterialBindingAPI"]\n    )\n    {\n'
                '        uniform token axis = "Z"\n'
                f"        double height = {2 * c.size[1]}\n        double radius = {c.size[0]}\n"
                f"        double3 xformOp:translate = ({c.pos[0]}, {c.pos[1]}, {c.pos[2]})\n"
                '        uniform token[] xformOpOrder = ["xformOp:translate"]\n'
                f"        rel material:binding = <{{ROOT}}/{mat}>\n    }}\n"
            )
    return out


# ── pen holder (two-arm station) ─────────────────────────────────────────────────────────


def pen_holder_usda(o: MetalPenHolderOptions) -> str:
    """The cup: the MuJoCo twin's colliders (hidden) and its holed visual mesh."""
    parts = _surface("Plastic", _rgb(o.holder_rgba), 0.45)
    for name, pos, half, quat in holder_collider_boxes(o):
        parts += _box(name, pos, half, quat_wxyz=quat, visible=False)
    # A 64-cell wall grid (the MuJoCo twin draws 96): the same holes with slightly coarser
    # edges, keeping the committed USD under the repo's 2 MB file limit.
    verts, _uv, faces = holder_visual_mesh(
        o.model_copy(update={"holder_hole_grid": _USD_HOLE_GRID})
    )
    parts += _mesh("visual", verts, None, faces, "Plastic")
    return _rigid("pen_holder", o.holder_mass_kg, o.holder_friction[0], parts)


def _sleeve(o: MetalPenHolderOptions) -> tuple[list[float], list[float], list[int]]:
    """The printed sleeve just outside the barrel (the MuJoCo twin's ``pen_sleeve``)."""
    half_l, r = o.pen_length_m / 2.0, o.pen_diameter_m / 2.0 * 1.01
    x_hi, x_lo = (half_l - f * o.pen_length_m for f in _PEN_SLEEVE_SPAN)
    n = 32
    v: list[float] = []
    uv: list[float] = []
    f: list[int] = []
    for i in range(n + 1):
        a = 2.0 * math.pi * i / n
        for x, vv in ((x_hi, 0.0), (x_lo, 1.0)):
            v.extend((x, r * math.cos(a), r * math.sin(a)))
            uv.extend((_PEN_SLEEVE_WRAPS * i / n % 1.0 if i < n else 1.0, 1.0 - vv))
    for i in range(n):
        a, b = 2 * i, 2 * i + 2
        f.extend((a, a + 1, b + 1, a, b + 1, b))
    return v, uv, f


def pen_usda(o: MetalPenHolderOptions) -> str:
    """The marker: a hidden square-section box collider, glossy barrel, printed sleeve."""
    half_l, r = o.pen_length_m / 2.0, o.pen_diameter_m / 2.0
    parts = _surface("Barrel", _rgb(o.pen_rgba), 0.15)
    parts += _surface("Label", (0.8, 0.8, 0.8), 0.4, "textures/pen_label.png")
    parts += _box("collider", (0.0, 0.0, 0.0), (half_l, r, r), visible=False)
    parts += (
        '    def Cylinder "barrel" (\n        prepend apiSchemas = ["MaterialBindingAPI"]\n    )\n'
        '    {\n        uniform token axis = "X"\n'
        f"        double height = {o.pen_length_m}\n        double radius = {r}\n"
        "        rel material:binding = <{ROOT}/Barrel>\n    }\n"
    )
    v, uv, f = _sleeve(o)
    parts += _mesh("sleeve", v, uv, f, "Label")
    return _rigid("pen", o.pen_mass_kg, o.pen_friction[0], parts)


def pen_table_usda(o: MetalPenHolderOptions) -> str:
    """Desk (baked wood top), riser shelf (the real board on top), back wall."""
    (x0, x1), (y0, y1) = o.table_x_range, o.table_y_range
    top, t = o.table_top_z, o.table_thickness
    (sx0, sx1), (sy0, sy1) = o.shelf_x_range, o.shelf_y_range
    # Isaac vs the MuJoCo twin's render at the same episode-0 tick (region means): desk
    # 1.07x (left wrist 1.1x), wall 0.92x before these gains; the shelf matched.
    parts = _surface("Desk", (1, 1, 1), _GLOSS_ROUGHNESS, "textures/desk_baked.jpg", tex_scale=0.84)
    parts += _surface("ShelfTop", (1, 1, 1), 0.5, "textures/shelf_top.jpg")
    parts += _surface("ShelfSide", _rgb(o.shelf_rgba), 0.6)
    parts += _surface("Wall", _bright(_rgb(o.wall_rgba), 1.2), 0.9)
    parts += _box(
        "desk",
        ((x0 + x1) / 2, (y0 + y1) / 2, top - t / 2),
        ((x1 - x0) / 2, (y1 - y0) / 2, t / 2),
        material="Desk",
    )
    parts += _quad("desk_top", (x0, x1), (y0, y1), top + 0.0005, "Desk")
    sh = (o.shelf_top_z - top) / 2
    parts += _box(
        "shelf",
        ((sx0 + sx1) / 2, (sy0 + sy1) / 2, top + sh),
        ((sx1 - sx0) / 2, (sy1 - sy0) / 2, sh),
        material="ShelfSide",
    )
    # The board cut from the top plate: columns = world x, rows = world y (the MuJoCo twin).
    parts += _quad(
        "shelf_top", (sx0, sx1), (sy0, sy1), o.shelf_top_z + 0.0005, "ShelfTop", u_along_x=True
    )
    parts += _box(
        "wall",
        (o.wall_x + 0.01, (y0 + y1) / 2, 0.6),
        (0.01, (y1 - y0) / 2, 0.6),
        material="Wall",
    )
    parts += _surface("Floor", _rgb(o.floor_rgba), 0.9)
    parts += _box("floor", (0.0, 0.0, o.floor_z - 0.01), (4.0, 4.0, 0.01), material="Floor")
    parts += _clutter(o.clutter)
    return _static("table", 1.0, parts)


# ── orange cube (single arm) ─────────────────────────────────────────────────────────────


def cube_usda(o: MetalPickPlaceOptions) -> str:
    """The 25 mm orange cube."""
    h = o.cube_edge_m / 2
    parts = _surface("Orange", _rgb(o.cube_rgba), 0.6)
    parts += _box("cube", (0.0, 0.0, 0.0), (h, h, h), material="Orange")
    return _rigid("cube", o.cube_mass_kg, o.cube_friction[0], parts)


def marker_usda(o: MetalPickPlaceOptions) -> str:
    """The target sticker: a thin visual disc (no collider), origin at its centre."""
    r = o.marker_diameter_m / 2
    n = 48
    v = [0.0, 0.0, 0.0]
    uv = [0.5, 0.5]
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        v.extend((r * math.cos(a), r * math.sin(a), 0.0))
        uv.extend((0.5 + 0.5 * math.sin(a), 0.5 + 0.5 * math.cos(a)))
    f: list[int] = []
    for i in range(n):
        f.extend((0, 1 + i, 2 + i))
    parts = _surface("Sticker", (1, 1, 1), 0.7, "textures/marker_sticker.png")
    parts += _mesh("sticker", v, uv, f, "Sticker")
    body = 'def Xform "marker"\n{\n' + parts + "}\n"
    return _header("marker") + body.replace("{ROOT}", "/marker")


def cube_table_usda(o: MetalPickPlaceOptions) -> str:
    """Desk (baked wood top) and the wall behind it."""
    (x0, x1), (y0, y1) = o.table_x_range, o.table_y_range
    top, t = o.table_top_z, o.table_thickness
    # Isaac top view vs the MuJoCo twin's render at reset (region means): desk 1.24x, wall
    # 0.95x before these gains.
    parts = _surface("Desk", (1, 1, 1), _GLOSS_ROUGHNESS, "textures/desk_baked.jpg", tex_scale=0.67)
    parts += _surface("Wall", _bright(_rgb(o.wall_rgba), 1.11), 0.9)
    parts += _box(
        "desk",
        ((x0 + x1) / 2, (y0 + y1) / 2, top - t / 2),
        ((x1 - x0) / 2, (y1 - y0) / 2, t / 2),
        material="Desk",
    )
    parts += _quad("desk_top", (x0, x1), (y0, y1), top + 0.0005, "Desk")
    parts += _box(
        "wall",
        (o.wall_x + 0.01, (y0 + y1) / 2, 0.6),
        (0.01, (y1 - y0) / 2, 0.6),
        material="Wall",
    )
    parts += _surface("Floor", _rgb(o.floor_rgba), 0.9)
    parts += _box("floor", (0.0, 0.0, o.floor_z - 0.01), (4.0, 4.0, 0.01), material="Floor")
    parts += _clutter(o.clutter)
    return _static("table", 1.0, parts)


def main() -> None:
    """Write both asset folders (USDA + baked textures)."""
    pen, cube = MetalPenHolderOptions(), MetalPickPlaceOptions()
    for d in (PEN_DIR, CUBE_DIR):
        (d / "textures").mkdir(parents=True, exist_ok=True)
    pen_size = (
        pen.table_x_range[1] - pen.table_x_range[0],
        pen.table_y_range[1] - pen.table_y_range[0],
    )
    assert pen.table_texture is not None and pen.shelf_texture is not None
    bake_tiled(pen.asset_path(pen.table_texture), pen_size, pen.table_texture_size_m).save(
        PEN_DIR / "textures/desk_baked.jpg", quality=88
    )
    Image.open(pen.asset_path(pen.shelf_texture)).convert("RGB").save(
        PEN_DIR / "textures/shelf_top.jpg", quality=90
    )
    assert pen.pen_label_texture is not None
    Image.open(pen.asset_path(pen.pen_label_texture)).save(PEN_DIR / "textures/pen_label.png")
    cube_size = (
        cube.table_x_range[1] - cube.table_x_range[0],
        cube.table_y_range[1] - cube.table_y_range[0],
    )
    assert cube.table_texture is not None and cube.marker_texture is not None
    bake_tiled(cube.asset_path(cube.table_texture), cube_size, cube.table_texture_size_m).save(
        CUBE_DIR / "textures/desk_baked.jpg", quality=88
    )
    Image.open(cube.asset_path(cube.marker_texture)).save(CUBE_DIR / "textures/marker_sticker.png")
    for path, text in (
        (PEN_DIR / "table.usda", pen_table_usda(pen)),
        (PEN_DIR / "pen_holder.usda", pen_holder_usda(pen)),
        (PEN_DIR / "pen.usda", pen_usda(pen)),
        (CUBE_DIR / "table.usda", cube_table_usda(cube)),
        (CUBE_DIR / "cube.usda", cube_usda(cube)),
        (CUBE_DIR / "marker.usda", marker_usda(cube)),
    ):
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(_REPO)} ({path.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
