"""The console's Sim scene layer, robot side (platform#133): the GLB writer and the HAL payload."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from openral_hal.sim_attached import SimAttachedHAL

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from scene_glb import gprim_mesh, write_glb


def _hal(**kw: object) -> SimAttachedHAL:
    """Only ``_env`` is read by ``_export_sim_scene``."""
    return cast("SimAttachedHAL", SimpleNamespace(**kw))


def test_glb_holds_one_primitive_per_colour_with_bounds_and_triangles() -> None:
    cube_mesh = gprim_mesh("Cube", size=1.0)
    cyl_mesh = gprim_mesh("Cylinder", radius=0.1, height=2.0, axis="X")
    assert cube_mesh is not None and cyl_mesh is not None
    (cube, cube_t), (cyl, cyl_t) = cube_mesh, cyl_mesh
    glb = write_glb(
        [((1, 0, 0, 1), cube, cube_t), ((0, 0, 1, 1), cyl, cyl_t), ((0, 1, 0, 1), [], [])]
    )
    magic, version, total = struct.unpack_from("<III", glb)
    assert (magic, version, total) == (0x46546C67, 2, len(glb))
    n = struct.unpack_from("<I", glb, 12)[0]
    doc = json.loads(glb[20 : 20 + n])
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


def test_hal_turns_the_export_into_a_package_uri_or_a_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path))
    seen: dict[str, object] = {}

    def ok(out_dir: str, limit: int) -> dict[str, object]:
        seen.update(out_dir=out_dir, limit=limit)
        return {"file": "abc.glb", "bytes": 5, "triangles": 2}

    got = SimAttachedHAL._export_sim_scene(_hal(_env=SimpleNamespace(scene_mesh=ok)))
    assert got == {"uri": "package://openral_sim_scene/abc.glb", "bytes": 5, "triangles": 2}
    assert seen == {"out_dir": str(tmp_path / "sim_scene"), "limit": 10_000_000}

    def boom(out_dir: str, limit: int) -> dict[str, object]:
        raise RuntimeError("sidecar gone")

    for env, note in [
        (SimpleNamespace(scene_mesh=lambda *_: {"note": "uses NVIDIA"}), "uses NVIDIA"),
        (SimpleNamespace(scene_mesh=boom), "export failed: RuntimeError: sidecar gone"),
    ]:
        assert SimAttachedHAL._export_sim_scene(_hal(_env=env)) == {"note": note}
