"""The console's Sim scene layer, robot side (platform#133): GLB, static rule, HAL payload."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from openral_hal.sim_scene import SceneCache, mujoco_scene_mesh, write_sim_scene
from openral_sim.sidecar import SidecarClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from scene_glb import gprim_mesh, mujoco_static_scene_glb, static_scene_glb, write_glb


def _doc(glb: bytes) -> dict[str, Any]:
    magic, version, total = struct.unpack_from("<III", glb)
    assert (magic, version, total) == (0x46546C67, 2, len(glb))
    n = struct.unpack_from("<I", glb, 12)[0]
    return cast("dict[str, Any]", json.loads(glb[20 : 20 + n]))


def test_glb_holds_one_primitive_per_colour_with_bounds_and_triangles() -> None:
    cube_mesh = gprim_mesh("Cube", size=1.0)
    cyl_mesh = gprim_mesh("Cylinder", radius=0.1, height=2.0, axis="X")
    assert cube_mesh is not None and cyl_mesh is not None
    (cube, cube_t), (cyl, cyl_t) = cube_mesh, cyl_mesh
    glb = write_glb(
        [((1, 0, 0, 1), cube, cube_t), ((0, 0, 1, 1), cyl, cyl_t), ((0, 1, 0, 1), [], [])]
    )
    doc = _doc(glb)
    prims = doc["meshes"][0]["primitives"]
    assert len(prims) == 2  # the empty one is dropped
    pos = doc["accessors"][prims[1]["attributes"]["POSITION"]]
    assert np.allclose(pos["min"][0], -1.0) and np.allclose(pos["max"][0], 1.0)  # axis X
    assert doc["accessors"][prims[0]["indices"]]["count"] == 12 * 3
    assert doc["materials"][1]["pbrMetallicRoughness"]["baseColorFactor"] == [0, 0, 1, 1]
    assert (
        sum(v["byteLength"] + (-v["byteLength"]) % 4 for v in doc["bufferViews"])
        == (doc["buffers"][0]["byteLength"])
    )
    empty = write_glb([])  # glTF arrays are minItems 1: the asset object alone, no BIN chunk
    assert _doc(empty) == {"asset": {"version": "2.0", "generator": "openral scene_glb"}}
    assert len(empty) == 12 + 8 + struct.unpack_from("<I", empty, 12)[0]


def test_plane_and_the_newer_gprim_schemas_are_drawn() -> None:
    plane = gprim_mesh("Plane", width=4.0, length=2.0, axis="Y")
    assert plane is not None
    assert np.allclose(np.ptp(plane[0], axis=0), [4.0, 0.0, 2.0])  # width x, length z, flat y
    cyl = gprim_mesh("Cylinder_1", radius_top=0.1, radius_bottom=0.3, height=1.0)
    assert cyl is not None and np.isclose(cyl[0][:, 0].max(), 0.3)
    assert gprim_mesh("Capsule_1", radius_top=0.2, radius_bottom=0.2, height=1.0) is not None
    assert gprim_mesh("HermiteCurves") is None
    for axis, tip in (("X", [1, 0, 0]), ("Y", [0, 1, 0]), ("Z", [0, 0, 1])):  # USD: tip at +axis
        cone = gprim_mesh("Cone", axis=axis)
        assert cone is not None
        assert any(np.allclose(v, tip) for v in cone[0]), axis


def test_static_scene_keeps_fixtures_and_drops_what_moves_or_is_skipped(tmp_path: Path) -> None:
    usd = pytest.importorskip("pxr.Usd")
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    # An "NVIDIA" layer: one fixture of the environment may reference it.
    nvidia = Sdf.Layer.CreateNew(str(tmp_path / "nvidia_root" / "rack.usda"))
    rack = Sdf.CreatePrimInLayer(nvidia, "/Rack")
    rack.specifier, rack.typeName = Sdf.SpecifierDef, "Cube"
    nvidia.defaultPrim = "Rack"
    nvidia.Save()

    def stage(with_nvidia: bool) -> Any:
        s = usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(s, "/World/env/wall").GetSizeAttr().Set(1.0)
        UsdGeom.Cube.Define(s, "/World/env/drawer/body")  # under a moving rigid body
        UsdPhysics.RigidBodyAPI.Apply(s.GetPrimAtPath("/World/env/drawer"))
        door = UsdGeom.Cube.Define(s, "/World/env/door")  # kinematic: static
        UsdPhysics.RigidBodyAPI.Apply(door.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdGeom.Cube.Define(s, "/World/env/hidden").MakeInvisible()
        UsdGeom.Cube.Define(s, "/World/env/tote")  # a link attachment: skipped
        UsdGeom.Sphere.Define(s, "/World/robot/base")  # the robot: skipped
        # An NVIDIA task object, never exported: must not refuse the scene.
        s.DefinePrim("/World/objects/banana").GetReferences().AddReference(nvidia.identifier)
        if with_nvidia:
            s.DefinePrim("/World/env/rack").GetReferences().AddReference(nvidia.identifier)
        # An indexed displayColor: [green, red] at index [1] is red.
        pv = UsdGeom.PrimvarsAPI(s.GetPrimAtPath("/World/env/wall")).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant
        )
        pv.Set([Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0)])
        pv.SetIndices([1])
        # A render mesh under a proxy-purpose Xform still draws.
        UsdGeom.Xform.Define(s, "/World/env/detail").CreatePurposeAttr(UsdGeom.Tokens.proxy)
        detail = UsdGeom.Cube.Define(s, "/World/env/detail/render")
        detail.CreatePurposeAttr(UsdGeom.Tokens.render)
        UsdGeom.Cube.Define(s, "/World/env/detail/proxy_only")  # inherits proxy: dropped
        return s

    skip = ["/World/robot", "/World/objects", "/World/env/tote"]
    root = str(tmp_path / "nvidia_root")
    out = static_scene_glb(stage(False), np.eye(4), skip, root)
    assert out["triangles"] == 3 * 12, out["log"]  # wall, kinematic door, render-purpose child
    doc = _doc(out["glb"])
    colours = {tuple(m["pbrMetallicRoughness"]["baseColorFactor"]) for m in doc["materials"]}
    assert (1.0, 0.0, 0.0, 1.0) in colours

    refused = static_scene_glb(stage(True), np.eye(4), skip, root)
    assert refused["note"] == "uses NVIDIA Isaac Sim assets, which may not be converted"
    assert "glb" not in refused

    # A layer beside the root, not under it, is not NVIDIA's.
    near = static_scene_glb(stage(True), np.eye(4), skip, str(tmp_path / "nvidia"))
    assert "glb" in near, near["log"]
    # A local box bound to an NVIDIA material keeps its geometry but not that colour.
    mat_layer = Sdf.Layer.CreateNew(str(tmp_path / "nvidia_root" / "paint.usda"))
    m = Sdf.CreatePrimInLayer(mat_layer, "/Paint")
    m.specifier, m.typeName = Sdf.SpecifierDef, "Material"
    sh = Sdf.CreatePrimInLayer(mat_layer, "/Paint/S")
    sh.specifier, sh.typeName = Sdf.SpecifierDef, "Shader"
    Sdf.AttributeSpec(sh, "info:id", Sdf.ValueTypeNames.Token).default = "UsdPreviewSurface"
    Sdf.AttributeSpec(sh, "inputs:diffuseColor", Sdf.ValueTypeNames.Color3f).default = Gf.Vec3f(
        0.0, 0.0, 1.0
    )
    mat_layer.defaultPrim = "Paint"
    mat_layer.Save()
    painted = stage(False)
    painted.DefinePrim("/World/Looks/paint").GetReferences().AddReference(mat_layer.identifier)
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(painted.GetPrimAtPath("/World/env/door")).Bind(
        UsdShade.Material(painted.GetPrimAtPath("/World/Looks/paint"))
    )
    grey = static_scene_glb(painted, np.eye(4), skip, root)
    assert "glb" in grey and "1 NVIDIA material colours drawn grey" in grey["log"]
    blues = [
        m for m in _doc(grey["glb"])["materials"]
        if m["pbrMetallicRoughness"]["baseColorFactor"][:3] == [0.0, 0.0, 1.0]
    ]  # fmt: skip
    assert not blues

    over = static_scene_glb(stage(False), np.eye(4), skip, root, limit_bytes=100)
    assert over["note"] == "scene is over the 0 MB limit"

    empty = static_scene_glb(stage(False), np.eye(4), [*skip, "/World/env"], root)
    assert empty["note"] == "the scene has no static geometry"


def test_a_mesh_colour_is_read_through_its_connected_material_input() -> None:
    usd = pytest.importorskip("pxr.Usd")
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    from scene_glb import diffuse_color

    s = usd.Stage.CreateInMemory()
    mat = UsdShade.Material.Define(s, "/World/Looks/red")
    tint = mat.CreateInput("tint", Sdf.ValueTypeNames.Color3f)  # not "diffuseColor": the
    tint.Set(Gf.Vec3f(0.8, 0.1, 0.1))  # red is reachable only through the connection
    sh = UsdShade.Shader.Define(s, "/World/Looks/red/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tint)
    mesh = UsdGeom.Cube.Define(s, "/World/box").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(mat)
    assert diffuse_color(mesh) == pytest.approx((0.8, 0.1, 0.1))
    assert diffuse_color(UsdGeom.Cube.Define(s, "/World/bare").GetPrim()) == (0.7, 0.7, 0.7)

    # The shader's own constant wins over an unrelated interface diffuseColor.
    blue = UsdShade.Material.Define(s, "/World/Looks/blue")
    blue.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 0))
    bsh = UsdShade.Shader.Define(s, "/World/Looks/blue/S")
    bsh.CreateIdAttr("UsdPreviewSurface")
    bsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 1))
    boxb = UsdGeom.Cube.Define(s, "/World/boxb").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(boxb).Bind(blue)
    assert diffuse_color(boxb) == pytest.approx((0.0, 0.0, 1.0))


def test_recolouring_a_textured_diffuse_reports_nothing() -> None:
    usd = pytest.importorskip("pxr.Usd")
    from isaac_manifest_scene import recolor_robot_materials
    from pxr import Sdf, UsdGeom, UsdShade

    s = usd.Stage.CreateInMemory()
    mat = UsdShade.Material.Define(s, "/World/robot/Looks/matte_black")
    tex = UsdShade.Shader.Define(s, "/World/robot/Looks/matte_black/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh = UsdShade.Shader.Define(s, "/World/robot/Looks/matte_black/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    mesh = UsdGeom.Mesh.Define(s, "/World/robot/link/visual").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(mat)
    # The texture's computed output cannot be overwritten: nothing to report.
    assert recolor_robot_materials(s, "/World/robot", {"*matte_black*": [0.0, 0.0, 0.0]}) == {}


class _UsdSceneEnv:
    """The sidecar's ``scene_mesh`` contract over a real pxr stage.

    ``IsaacManifestScene.scene_mesh`` needs a running Isaac Sim; this runs the same
    exporter (``static_scene_glb``) and the same exact-size check on an in-memory stage.
    """

    def __init__(self, stage: Any, nvidia_root: str = "") -> None:
        self.stage, self.nvidia_root, self.limits = stage, nvidia_root, cast("list[int]", [])

    def scene_mesh(self, limit_bytes: int) -> dict[str, Any]:
        self.limits.append(limit_bytes)
        out: dict[str, Any] = static_scene_glb(
            self.stage, np.eye(4), [], self.nvidia_root, limit_bytes
        )
        glb = out.get("glb")
        if isinstance(glb, bytes) and len(glb) > limit_bytes:
            return {"note": "over the limit", "log": out["log"]}
        return out


def test_hal_writes_the_exported_glb_and_sweeps_only_week_old_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usd = pytest.importorskip("pxr.Usd")
    import os
    import time

    from pxr import Sdf, UsdGeom

    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path / "cache"))
    pkg = tmp_path / "cache" / "sim_scene"
    pkg.mkdir(parents=True)
    stale, live = pkg / "0123456789ab.glb", pkg / "ba9876543210.glb"
    stale.write_bytes(b"a scene no deploy wrote for 8 days")
    os.utime(stale, (time.time() - 8 * 86400,) * 2)
    live.write_bytes(b"another deploy's current scene")

    stage = usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/World/wall")
    env = _UsdSceneEnv(stage)
    got = write_sim_scene(env.scene_mesh)
    name = str(got["uri"]).removeprefix("package://openral_sim_scene/")
    assert got == {
        "uri": f"package://openral_sim_scene/{name}",
        "bytes": got["bytes"],
        "triangles": 12,
    }
    assert _doc((pkg / name).read_bytes())["meshes"]  # the real exporter's GLB
    assert sorted(p.name for p in pkg.iterdir()) == sorted([name, live.name])  # stale swept
    assert env.limits == [10_000_000]

    nvidia = Sdf.Layer.CreateNew(str(tmp_path / "nvidia" / "rack.usda"))
    Sdf.CreatePrimInLayer(nvidia, "/Rack").typeName = "Cube"
    nvidia.defaultPrim = "Rack"
    nvidia.Save()
    stage.DefinePrim("/World/rack").GetReferences().AddReference(nvidia.identifier)
    refused = write_sim_scene(_UsdSceneEnv(stage, str(tmp_path / "nvidia")).scene_mesh)
    assert refused == {"note": "uses NVIDIA Isaac Sim assets, which may not be converted"}

    broken = write_sim_scene(_UsdSceneEnv(None).scene_mesh)
    assert str(broken["note"]).startswith("export failed: AttributeError")
    # A reply that passes as a GLB but carries no triangle count writes nothing.
    before = sorted(pkg.iterdir())
    assert write_sim_scene(lambda _: {"glb": b"glTF", "triangles": None}) == {
        "note": "export failed: the exporter sent a malformed scene"
    }
    assert sorted(pkg.iterdir()) == before


def test_a_timed_out_sidecar_call_reopens_the_req_socket() -> None:
    zmq = pytest.importorskip("zmq")
    import threading

    import msgpack

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    port = rep.bind_to_random_port("tcp://127.0.0.1")
    slow_done = threading.Event()

    def serve() -> None:
        for delay in (0.6, 0.0):  # the first reply comes after the client gave up
            msg = msgpack.unpackb(rep.recv())
            threading.Event().wait(delay)
            rep.send(msgpack.packb({"ok": True, "endpoint": msg["endpoint"]}))
            slow_done.set()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = SidecarClient(
        name="probe", host="127.0.0.1", port=port, timeout_ms=200, boot_timeout_s=1.0,
        launch_argv=[], auto_spawn=False,
    )  # fmt: skip
    client._ctx = ctx
    client._init_socket()
    try:
        with pytest.raises(zmq.Again):
            client.call("slow")
        # The sidecar is single-threaded: it answers the next call once the slow one is
        # done. Without the reopen, this send would raise (REQ stuck in EFSM).
        assert slow_done.wait(2)
        assert client.call("next") == {"ok": True, "endpoint": "next"}  # the same client
    finally:
        client.close()
        thread.join(timeout=2)
        rep.close(linger=0)


def test_an_unknown_cache_home_is_an_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from openral_hal import ros_package_overlay as rpo

    monkeypatch.setenv("OPENRAL_CACHE_DIR", "~no_such_user_openral_test/cache")
    with pytest.raises(OSError, match="OPENRAL_CACHE_DIR"):
        rpo.sim_scene_package_root()


def test_overlay_registers_this_runs_local_package_with_an_absolute_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openral_hal import ros_package_overlay as rpo

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENRAL_CACHE_DIR", "cache")  # relative on purpose
    root = rpo.sim_scene_package_root()
    assert root.is_absolute() and root == tmp_path / "cache" / "sim_scene"
    # An older overlay already indexes the package: this run's root must still win.
    old = tmp_path / "old_overlay"
    (old / "share" / "ament_index" / "resource_index" / "packages").mkdir(parents=True)
    (old / "share" / "ament_index" / "resource_index" / "packages" / "openral_sim_scene").touch()
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(old))
    prefix = rpo.public_package_overlay("", {rpo.SIM_SCENE_PACKAGE: root})
    assert prefix is not None
    assert (prefix / "share" / "openral_sim_scene").resolve() == root


def _image(doc: dict[str, Any], glb: bytes, index: int) -> bytes:
    view = doc["bufferViews"][doc["images"][index]["bufferView"]]
    start = 20 + struct.unpack_from("<I", glb, 12)[0] + 8 + view.get("byteOffset", 0)
    return glb[start : start + view["byteLength"]]


def test_a_textured_primitive_embeds_its_image_and_uvs() -> None:
    cube = gprim_mesh("Cube")
    assert cube is not None
    uv = np.zeros((len(cube[0]), 2))
    glb = write_glb([
        ((1, 1, 1, 1), *cube, uv, (b"\xff\xd8jpeg", "image/jpeg", False)),
        ((1, 1, 1, 1), *cube, uv, (b"\x89PNG", "image/png", True)),
        ((1, 1, 1, 1), *cube, uv, (b"\x89PNG", "image/png", (False, True))),
    ])  # fmt: skip
    doc = _doc(glb)
    prim = doc["meshes"][0]["primitives"][0]
    assert doc["accessors"][prim["attributes"]["TEXCOORD_0"]]["type"] == "VEC2"
    tex = doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    assert doc["textures"][tex] == {"source": 0}
    assert doc["images"][0]["mimeType"] == "image/jpeg"
    assert "target" not in doc["bufferViews"][doc["images"][0]["bufferView"]]  # glTF: images
    assert _image(doc, glb, 0) == b"\xff\xd8jpeg"
    assert "alphaMode" not in doc["materials"][0]
    assert doc["materials"][1]["alphaMode"] == "BLEND"  # a PNG carries alpha
    assert doc["textures"][1] == {"source": 1, "sampler": 0}
    assert doc["samplers"][0] == {"wrapS": 33071, "wrapT": 33071}  # clamp to edge
    assert doc["textures"][2]["sampler"] == 1
    assert doc["samplers"][1] == {"wrapS": 10497, "wrapT": 33071}  # each axis its own
    with pytest.raises(ValueError, match="uvs for"):
        write_glb([((1, 1, 1, 1), *cube, uv[:3], (b"x", "image/png", False))])


def test_a_textured_mesh_embeds_its_shrunk_texture(tmp_path: Path) -> None:
    usd = pytest.importorskip("pxr.Usd")
    image_mod = pytest.importorskip("PIL.Image")
    from pxr import Sdf, UsdGeom, UsdShade

    def textured_quad(s: Any, path: str, png: Path) -> None:
        image_mod.new("RGB", (2048, 1024), (200, 40, 40)).save(png)
        quad = UsdGeom.Mesh.Define(s, path)
        quad.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        quad.CreateFaceVertexCountsAttr([4])
        quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        UsdGeom.PrimvarsAPI(quad).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        ).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
        mat = UsdShade.Material.Define(s, f"{path}_mat")
        tex = UsdShade.Shader.Define(s, f"{path}_mat/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(png))
        rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh = UsdShade.Shader.Define(s, f"{path}_mat/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(mat)

    s = usd.Stage.CreateInMemory()
    textured_quad(s, "/World/env/poster", tmp_path / "poster.png")
    (tmp_path / "nvidia").mkdir()
    textured_quad(s, "/World/env/decal", tmp_path / "nvidia" / "decal.png")
    UsdGeom.Cube.Define(s, "/World/env/wall")
    out = static_scene_glb(s, np.eye(4), [], str(tmp_path / "nvidia"))
    assert out["triangles"] == 2 + 2 + 12, out["log"]
    assert "1 textures" in out["log"] and "1 NVIDIA textures left out" in out["log"]
    doc = _doc(out["glb"])
    assert len(doc["images"]) == 1  # the NVIDIA decal keeps its plain colour
    textured = [p for p in doc["meshes"][0]["primitives"] if "TEXCOORD_0" in p["attributes"]]
    assert len(textured) == 1
    uv = doc["accessors"][textured[0]["attributes"]["TEXCOORD_0"]]
    assert uv["count"] == 4  # one per face corner
    import io

    with image_mod.open(io.BytesIO(_image(doc, out["glb"], 0))) as im:
        assert im.format == "JPEG" and im.size == (512, 256)

    # Over the limit with its textures: rebuilt in each texture's mean colour, not refused.
    tight = static_scene_glb(s, np.eye(4), [], str(tmp_path / "nvidia"), len(out["glb"]) - 1)
    assert "textures dropped" in tight["log"] and "images" not in _doc(tight["glb"])
    factors = [
        m["pbrMetallicRoughness"]["baseColorFactor"] for m in _doc(tight["glb"])["materials"]
    ]
    assert any(np.allclose(f[:3], (200 / 255, 40 / 255, 40 / 255), atol=0.01) for f in factors)

    (tmp_path / "poster.png").write_bytes(b"not an image")
    grey = static_scene_glb(s, np.eye(4), [], str(tmp_path / "nvidia"))
    assert "1 unreadable textures drawn grey" in grey["log"] and "images" not in _doc(grey["glb"])


def test_texture_wiring_through_interface_inputs_transforms_and_nvidia_layers(
    tmp_path: Path,
) -> None:
    usd = pytest.importorskip("pxr.Usd")
    image_mod = pytest.importorskip("PIL.Image")
    from pxr import Sdf, UsdGeom, UsdShade

    png = tmp_path / "tiles.png"
    image_mod.new("RGB", (64, 64), (10, 200, 10)).save(png)
    s = usd.Stage.CreateInMemory()
    quad = UsdGeom.Mesh.Define(s, "/World/floor")
    quad.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    quad.CreateFaceVertexCountsAttr([4])
    quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.PrimvarsAPI(quad).CreatePrimvar(  # a primvar only the reader's varname names
        "uv", Sdf.ValueTypeNames.Float2Array, UsdGeom.Tokens.vertex
    ).Set([(0, 0), (1, 0), (1, 1), (0, 1)])

    def material(path: str, layer: Any = None, *, surface: bool = True) -> Any:
        if layer is not None:
            s.SetEditTarget(layer)
        mat = UsdShade.Material.Define(s, path)
        reader = UsdShade.Shader.Define(s, f"{path}/Reader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("uv")
        xf = UsdShade.Shader.Define(s, f"{path}/Xf")
        xf.CreateIdAttr("UsdTransform2d")
        xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )
        xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set((4.0, 4.0))
        tex = UsdShade.Shader.Define(s, f"{path}/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(png))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            xf.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )
        # The URDF importer's pattern: the surface reads a Material interface input.
        iface = mat.CreateInput("diffuse", Sdf.ValueTypeNames.Color3f)
        iface.ConnectToSource(tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
        sh = UsdShade.Shader.Define(s, f"{path}/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(iface)
        if surface:
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        s.SetEditTarget(s.GetRootLayer())
        return mat

    UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(material("/World/Looks/tiles"))
    out = static_scene_glb(s, np.eye(4), [], str(tmp_path / "nvidia"))
    doc = _doc(out["glb"])
    assert len(doc["images"]) == 1, out["log"]
    prim = doc["meshes"][0]["primitives"][0]
    uv = doc["accessors"][prim["attributes"]["TEXCOORD_0"]]
    assert uv["count"] == 4
    view = doc["bufferViews"][uv["bufferView"]]
    start = 20 + struct.unpack_from("<I", out["glb"], 12)[0] + 8 + view["byteOffset"]
    st = np.frombuffer(out["glb"], np.float32, 8, start).reshape(4, 2)
    assert st[:, 0].max() == pytest.approx(4.0) and st[:, 1].min() == pytest.approx(-3.0)
    assert doc["textures"][0]["sampler"] == 0
    assert doc["samplers"] == [{"wrapS": 33071, "wrapT": 10497}]  # wrapS clamp, wrapT repeat

    # A preview shader the material's surface output does not use is not its texture.
    UsdShade.MaterialBindingAPI(quad.GetPrim()).Bind(material("/World/Looks/loose", surface=False))
    assert "images" not in _doc(static_scene_glb(s, np.eye(4), [], str(tmp_path / "nv"))["glb"])

    # The same material authored in an NVIDIA layer: its texture is not embedded, even
    # though the image itself sits outside the NVIDIA root.
    (tmp_path / "nvidia").mkdir()
    sub = Sdf.Layer.CreateNew(str(tmp_path / "nvidia" / "looks.usda"))
    s.GetRootLayer().subLayerPaths.append(sub.identifier)
    UsdShade.MaterialBindingAPI(quad.GetPrim()).Bind(material("/World/Looks/nv", sub))
    nv = static_scene_glb(s, np.eye(4), [], str(tmp_path / "nvidia"))
    assert "1 NVIDIA textures left out" in nv["log"] and "images" not in _doc(nv["glb"])


_MJCF = """
<mujoco>
  <asset>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.5 0.25 0" width="8" height="8"/>
    <material name="wood" texture="wood"/>
    <material name="glass" rgba="1 1 1 0"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="marker" type="box" pos="0 0 0.05" size="0.01 0.01 0.01" contype="0" conaffinity="0"
          rgba="1 0 0 1"/>
    <body name="counter" pos="1 0 0.5">
      <geom type="box" size="0.5 0.3 0.5" material="wood"/>
      <geom type="box" size="0.5 0.3 0.5" contype="0" conaffinity="0" material="wood"/>
      <geom type="box" size="0.1 0.1 0.4" contype="0" conaffinity="0" material="wood"
            rgba="0 1 0 1"/>
      <geom type="sphere" size="0.05" rgba="1 1 1 0"/>
      <geom type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0" material="glass"
            rgba="0.500001 0.5 0.5 1"/>
    </body>
    <body name="cup" pos="1 0 1.1"><freejoint/><geom type="cylinder" size="0.04 0.05"/></body>
    <body name="target" mocap="true" pos="0 1 1"><geom type="box" size="0.1 0.1 0.1"/></body>
    <body name="robot_mount" pos="-1 0 0">
      <geom type="box" size="0.2 0.2 0.2" rgba="0 0 1 1"/>
      <body name="link1" pos="0 0 0.3">
        <joint name="j1" type="hinge"/>
        <geom type="capsule" size="0.05 0.2" rgba="0 0 1 1"/>
      </body>
    </body>
  </worldbody>
  <actuator><position joint="j1"/></actuator>
</mujoco>
"""


def _arm_description() -> Any:  # reason: openral_core imported lazily
    from openral_core import (
        ControlMode,
        EmbodimentKind,
        JointSpec,
        JointType,
        RobotCapabilities,
        RobotDescription,
        SafetyEnvelope,
    )

    return RobotDescription(
        name="arm",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(name="j1", joint_type=JointType.REVOLUTE, parent_link="a", child_link="b")
        ],
        capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        safety=SafetyEnvelope(),
    )


def test_mujoco_static_scene_keeps_world_welded_geoms_minus_robot_and_mocap() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    shift = np.eye(4)
    shift[:3, 3] = [0.0, 0.0, -1.0]
    out = mujoco_static_scene_glb(model, data, shift)
    # The floor and the world's visual-only marker (a world geom never hides another), the
    # counter's three visual boxes (not its collision box, nor the invisible ball): the
    # near-default rgba is the geom's own, opaque, not its transparent material's.
    assert out["triangles"] == 2 + 12 + 3 * 12, out["log"]
    doc = _doc(out["glb"])
    colours = {tuple(m["pbrMetallicRoughness"]["baseColorFactor"]) for m in doc["materials"]}
    # wood: its texture's mean colour; the green box: its own rgba beats its material's,
    # still times the material's texture (as MuJoCo draws it)
    assert colours == {
        (0.5, 0.5, 0.5, 1.0),
        (0.5, 0.25, 0.0, 1.0),
        (0.0, 0.25, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
    }
    zs = [doc["accessors"][p["attributes"]["POSITION"]] for p in doc["meshes"][0]["primitives"]]
    assert min(a["min"][2] for a in zs) == pytest.approx(-1.0)  # the floor, in the frame
    assert max(a["max"][2] for a in zs) == pytest.approx(0.0)  # the counter's top

    over = mujoco_static_scene_glb(model, data, np.eye(4), limit_bytes=100)
    assert over["note"] == "scene is over the 0 MB limit"


def test_a_mujoco_scene_is_baked_in_the_frame_the_base_tf_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path))
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def wood_max_x(odom_from_base: Any = None) -> float:
        out = mujoco_scene_mesh(model, data, _arm_description(), 10_000_000, odom_from_base)
        doc = _doc(out["glb"])
        wood = next(
            p for p in doc["meshes"][0]["primitives"]
            if doc["materials"][p["material"]]["pbrMetallicRoughness"]["baseColorFactor"][1]
            == 0.25
        )  # fmt: skip
        return float(doc["accessors"][wood["attributes"]["POSITION"]]["max"][0])

    # No body named for base_frame: the actuated robot's root (robot_mount, x = -1).
    assert wood_max_x() == pytest.approx(2.5)  # the counter spans x 0.5..1.5 in the world
    odom = np.eye(4)
    odom[0, 3] = 10.0  # a mobile base 10 m along its odometry frame's x
    assert wood_max_x(odom) == pytest.approx(12.5)


def test_a_mujoco_arm_hal_exports_only_when_asked_and_once_per_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mujoco")
    from openral_hal._mujoco_arm import MujocoArmHAL

    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path))
    (tmp_path / "scene.xml").write_text(_MJCF)
    xml = str(tmp_path / "scene.xml")
    arm = MujocoArmHAL(
        _arm_description(), mjcf_path=xml, joint_qpos_addr={"j1": 7}, actuator_index={"j1": 0}
    )
    assert arm.export_sim_scene() is None  # not connected
    arm.connect()
    try:
        assert not (tmp_path / "sim_scene").exists()  # connect alone exports nothing
        scene = arm.export_sim_scene()
        assert scene is not None and scene["triangles"] == 50, scene
        glb = tmp_path / "sim_scene" / str(scene["uri"]).rsplit("/", 1)[1]
        mtime = glb.stat().st_mtime_ns
        assert arm.export_sim_scene() is scene and glb.stat().st_mtime_ns >= mtime  # cached
    finally:
        arm.disconnect()
    assert arm._scene_cache is None  # the disconnected model is not kept alive


def test_a_scene_cache_redoes_an_export_it_could_not_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GLB the cache could not write, or that a sweep removed, is written again."""
    glb = write_glb([((1, 1, 1, 1), *gprim_mesh("Cube", size=1.0))])
    calls: list[int] = []

    def scene(limit_bytes: int) -> dict[str, Any]:
        calls.append(limit_bytes)
        return {"glb": glb, "triangles": 12}

    (tmp_path / "blocked").write_text("a file, not a directory")
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path / "blocked"))
    cache = SceneCache()
    assert str(cache.get("stage", scene)["note"]).startswith("export failed: ")
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path / "cache"))
    uri = str(cache.get("stage", scene)["uri"])
    path = tmp_path / "cache" / "sim_scene" / uri.rsplit("/", 1)[-1]
    assert path.read_bytes() == glb
    assert cache.get("stage", scene)["uri"] == uri and len(calls) == 2  # reused
    path.unlink()  # another deploy's week-old sweep
    assert cache.get("stage", scene)["uri"] == uri and path.read_bytes() == glb
    assert len(calls) == 3


def test_texture_encoding_and_uv_transforms_need_no_usd(tmp_path: Path) -> None:
    """The texture path's pure parts, run in the default lane (no ``scene-usd`` group)."""
    import io

    from PIL import Image
    from scene_glb import _encode_texture, _srt

    Image.new("RGB", (1024, 256), (255, 0, 0)).save(tmp_path / "red.png")
    data, mime, mean = _encode_texture(str(tmp_path / "red.png"))
    assert mime == "image/jpeg" and np.allclose(mean, (1.0, 0.0, 0.0), atol=0.02)
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (512, 128)
    decal = Image.new("RGBA", (8, 8), (0, 0, 255, 255))
    decal.putpixel((0, 0), (0, 0, 0, 0))
    decal.save(tmp_path / "decal.png")
    assert _encode_texture(str(tmp_path / "decal.png"))[1] == "image/png"  # alpha kept

    m = _srt((2.0, 3.0), 90.0, (0.5, 0.0))  # scale, then rotate, then translate
    assert np.allclose(m @ [1.0, 0.0, 1.0], [0.5, 2.0, 1.0])
