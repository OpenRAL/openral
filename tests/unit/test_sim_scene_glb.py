"""The console's Sim scene layer, robot side (platform#133): GLB, static rule, HAL payload."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from openral_hal.sim_attached import SimAttachedHAL
from openral_sim.sidecar import SidecarClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from scene_glb import gprim_mesh, mujoco_static_scene_glb, static_scene_glb, write_glb


def _hal(**kw: object) -> SimAttachedHAL:
    """Only ``_env`` is read by ``_export_sim_scene``."""
    return cast("SimAttachedHAL", SimpleNamespace(**kw))


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
    got = SimAttachedHAL._export_sim_scene(_hal(_env=env))
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
    refused = SimAttachedHAL._export_sim_scene(
        _hal(_env=_UsdSceneEnv(stage, str(tmp_path / "nvidia")))
    )
    assert refused == {"note": "uses NVIDIA Isaac Sim assets, which may not be converted"}

    broken = SimAttachedHAL._export_sim_scene(_hal(_env=_UsdSceneEnv(None)))
    assert str(broken["note"]).startswith("export failed: AttributeError")
    # A reply that passes as a GLB but carries no triangle count writes nothing.
    before = sorted(pkg.iterdir())
    bad = SimpleNamespace(scene_mesh=lambda _: {"glb": b"glTF", "triangles": None})
    assert SimAttachedHAL._export_sim_scene(_hal(_env=bad)) == {
        "note": "export failed: the sidecar sent a malformed scene"
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
    glb = write_glb([((1, 1, 1, 1), *cube, uv, (b"\xff\xd8jpeg", "image/jpeg"))])
    doc = _doc(glb)
    prim = doc["meshes"][0]["primitives"][0]
    assert doc["accessors"][prim["attributes"]["TEXCOORD_0"]]["type"] == "VEC2"
    tex = doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    assert doc["textures"][tex] == {"source": 0}
    assert doc["images"][0]["mimeType"] == "image/jpeg"
    assert "target" not in doc["bufferViews"][doc["images"][0]["bufferView"]]  # glTF: images
    assert _image(doc, glb, 0) == b"\xff\xd8jpeg"
    with pytest.raises(ValueError, match="uvs for"):
        write_glb([((1, 1, 1, 1), *cube, uv[:3], (b"x", "image/png"))])


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

    (tmp_path / "poster.png").write_bytes(b"not an image")
    grey = static_scene_glb(s, np.eye(4), [], str(tmp_path / "nvidia"))
    assert "1 unreadable textures drawn grey" in grey["log"] and "images" not in _doc(grey["glb"])


_MJCF = """
<mujoco>
  <asset>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.5 0.25 0" width="8" height="8"/>
    <material name="wood" texture="wood"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" rgba="0.5 0.5 0.5 1"/>
    <body name="counter" pos="1 0 0.5">
      <geom type="box" size="0.5 0.3 0.5" material="wood"/>
      <geom type="box" size="0.5 0.3 0.5" contype="0" conaffinity="0" material="wood"/>
      <geom type="sphere" size="0.05" rgba="1 1 1 0"/>
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


def test_mujoco_static_scene_keeps_world_welded_geoms_minus_robot_and_mocap() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    shift = np.eye(4)
    shift[:3, 3] = [0.0, 0.0, -1.0]
    out = mujoco_static_scene_glb(model, data, shift)
    assert out["triangles"] == 2 + 12, out["log"]  # floor + the counter's visual box only
    doc = _doc(out["glb"])
    colours = {tuple(m["pbrMetallicRoughness"]["baseColorFactor"]) for m in doc["materials"]}
    assert colours == {(0.5, 0.5, 0.5, 1.0), (0.5, 0.25, 0.0, 1.0)}  # wood: its mean colour
    zs = [doc["accessors"][p["attributes"]["POSITION"]] for p in doc["meshes"][0]["primitives"]]
    assert min(a["min"][2] for a in zs) == pytest.approx(-1.0)  # the floor, in the frame
    assert max(a["max"][2] for a in zs) == pytest.approx(0.0)  # the counter's top

    over = mujoco_static_scene_glb(model, data, np.eye(4), limit_bytes=100)
    assert over["note"] == "scene is over the 0 MB limit"


def test_hal_exports_a_mujoco_scene_in_the_arm_base_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path))
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    desc = SimpleNamespace(base_frame="not_in_the_mjcf", base_joints=None)
    hal = SimpleNamespace(_env=object(), _mujoco_handles=lambda: (model, data), description=desc)
    hal._mujoco_scene_mesh = lambda limit: SimAttachedHAL._mujoco_scene_mesh(
        cast("SimAttachedHAL", hal), limit
    )
    got = SimAttachedHAL._export_sim_scene(cast("SimAttachedHAL", hal))
    assert got["triangles"] == 14, got
    glb = (tmp_path / "sim_scene" / str(got["uri"]).rsplit("/", 1)[1]).read_bytes()
    doc = _doc(glb)
    wood = next(
        p for p in doc["meshes"][0]["primitives"]
        if doc["materials"][p["material"]]["pbrMetallicRoughness"]["baseColorFactor"][1] == 0.25
    )  # fmt: skip
    # The robot's root body (robot_mount, at x = -1) is the frame: the counter is at x 1.5..2.5.
    assert doc["accessors"][wood["attributes"]["POSITION"]]["max"][0] == pytest.approx(2.5)
