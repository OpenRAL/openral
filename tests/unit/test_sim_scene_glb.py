"""The console's Sim scene layer, robot side (platform#133): GLB, static rule, HAL payload."""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from openral_hal.sim_attached import SimAttachedHAL
from openral_sim.sidecar import SidecarClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from scene_glb import gprim_mesh, static_scene_glb, write_glb


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
    assert _doc(write_glb([]))["buffers"] == []  # glTF needs byteLength >= 1: no buffer at all


def test_plane_and_the_newer_gprim_schemas_are_drawn() -> None:
    plane = gprim_mesh("Plane", width=4.0, length=2.0, axis="Y")
    assert plane is not None
    assert np.allclose(np.ptp(plane[0], axis=0), [4.0, 0.0, 2.0])  # width x, length z, flat y
    cyl = gprim_mesh("Cylinder_1", radius_top=0.1, radius_bottom=0.3, height=1.0)
    assert cyl is not None and np.isclose(cyl[0][:, 0].max(), 0.3)
    assert gprim_mesh("Capsule_1", radius_top=0.2, radius_bottom=0.2, height=1.0) is not None
    assert gprim_mesh("HermiteCurves") is None


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
        UsdGeom.Cube(s.GetPrimAtPath("/World/env/wall")).CreateDisplayColorAttr(
            [Gf.Vec3f(1.0, 0.0, 0.0)]
        )
        return s

    skip = ["/World/robot", "/World/objects", "/World/env/tote"]
    root = str(tmp_path / "nvidia_root")
    out = static_scene_glb(stage(False), np.eye(4), skip, root)
    assert out["triangles"] == 2 * 12, out["log"]  # wall + kinematic door
    doc = _doc(out["glb"])
    colours = {tuple(m["pbrMetallicRoughness"]["baseColorFactor"]) for m in doc["materials"]}
    assert (1.0, 0.0, 0.0, 1.0) in colours

    refused = static_scene_glb(stage(True), np.eye(4), skip, root)
    assert refused["note"] == "uses NVIDIA Isaac Sim assets, which may not be converted"
    assert "glb" not in refused

    empty = static_scene_glb(stage(False), np.eye(4), [*skip, "/World/env"], root)
    assert empty["note"] == "the scene has no static geometry"


def test_a_mesh_colour_is_read_through_its_connected_material_input() -> None:
    usd = pytest.importorskip("pxr.Usd")
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    from scene_glb import diffuse_color

    s = usd.Stage.CreateInMemory()
    mat = UsdShade.Material.Define(s, "/World/Looks/red")
    iface = mat.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    iface.Set(Gf.Vec3f(0.8, 0.1, 0.1))
    sh = UsdShade.Shader.Define(s, "/World/Looks/red/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(iface)
    mesh = UsdGeom.Cube.Define(s, "/World/box").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(mat)
    assert diffuse_color(mesh) == pytest.approx((0.8, 0.1, 0.1))
    assert diffuse_color(UsdGeom.Cube.Define(s, "/World/bare").GetPrim()) == (0.7, 0.7, 0.7)


def test_hal_writes_the_glb_into_the_package_and_drops_older_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path))
    pkg = tmp_path / "sim_scene"
    pkg.mkdir()
    (pkg / "0123456789ab.glb").write_bytes(b"an earlier scene")
    seen: dict[str, object] = {}

    def ok(limit: int) -> dict[str, object]:
        seen["limit"] = limit
        return {"glb": b"glTF-bytes", "triangles": 2}

    got = SimAttachedHAL._export_sim_scene(_hal(_env=SimpleNamespace(scene_mesh=ok)))
    name = str(got["uri"]).removeprefix("package://openral_sim_scene/")
    assert got == {"uri": f"package://openral_sim_scene/{name}", "bytes": 10, "triangles": 2}
    assert [p.name for p in pkg.iterdir()] == [name]
    assert (pkg / name).read_bytes() == b"glTF-bytes"
    assert seen == {"limit": 10_000_000}

    def boom(limit: int) -> dict[str, object]:
        raise RuntimeError("sidecar gone")

    for env, note in [
        (SimpleNamespace(scene_mesh=lambda _: {"note": "uses NVIDIA"}), "uses NVIDIA"),
        (SimpleNamespace(scene_mesh=boom), "export failed: RuntimeError: sidecar gone"),
    ]:
        assert SimAttachedHAL._export_sim_scene(_hal(_env=env)) == {"note": note}


def test_a_timed_out_sidecar_call_reopens_the_req_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    class Again(Exception):  # noqa: N818  # reason: mirrors zmq.Again
        pass

    monkeypatch.setitem(sys.modules, "zmq", types.SimpleNamespace(Again=Again))
    reopened: list[bool] = []

    class Sock:
        def send(self, _: bytes) -> None:
            pass

        def recv(self) -> bytes:
            raise Again

    client = SimpleNamespace(
        name="isaac", _socket=Sock(), _init_socket=lambda: reopened.append(True)
    )
    with pytest.raises(Again):
        SidecarClient.call(cast("SidecarClient", client), "scene_mesh")
    assert reopened == [True]
