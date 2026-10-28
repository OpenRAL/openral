"""Declared static geometry: schema, surface sampling, CLI forwarding, launch sampling (#349).

A scene declares its cell's fixtures (``DeployRuntime.static_geometry_file``); the launch
samples them a quarter cell apart and the octomap bridge marks them occupied in every
grid (``docs/reference/manipulation-workspace-contact.md`` §3.1). No mocks: real files,
the real cell scene, the real launch file loaded by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from openral_core import DeployRuntime, StaticGeometry
from openral_core.exceptions import ROSConfigError
from openral_safety.static_geometry import (
    load_static_geometry,
    surface_points,
    write_points_file,
)
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[2]
_CELL = _REPO / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"
_LAUNCH = _REPO / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"


def _cell_geometry(tmp_path: Path) -> Path:
    """A wall box and a one-triangle shelf mesh, written as the export tool writes them."""
    np.savez_compressed(
        tmp_path / "shelf.npz",
        vertices=np.array([[0.4, -0.2, 0.0], [0.4, 0.2, 0.0], [0.6, 0.0, 0.0]], np.float32),
        triangles=np.array([[0, 1, 2]], np.int32),
    )
    path = tmp_path / "static.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "static_geometry": [
                    {
                        "kind": "box",
                        "source": "/World/Cell/RightWall",
                        "xyz": [0.3, -0.55, 0.0],
                        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "half_extents": [0.2, 0.018, 0.3],
                    },
                    {"kind": "mesh", "source": "/World/OMAR/Rack", "file": "shelf.npz"},
                ]
            }
        )
    )
    return path


# ── schema ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "entry",
    [
        {"kind": "box", "xyz": [0, 0, 0], "half_extents": [-0.1, 0.1, 0.1]},
        {
            "kind": "box",
            "xyz": [0, 0, 0],
            "half_extents": [0.1, 0.1, 0.1],
            "quat_xyzw": [0, 0, 0, 3],
        },
        {"kind": "box", "xyz": [0, 0, float("nan")], "half_extents": [0.1, 0.1, 0.1]},
        {"kind": "mesh", "file": ""},
        {"kind": "sphere", "xyz": [0, 0, 0]},
    ],
)
def test_malformed_static_geometry_is_refused(entry: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        StaticGeometry.model_validate({"static_geometry": [entry]})


def test_the_runtime_field_defaults_off() -> None:
    assert DeployRuntime().static_geometry_file is None


# ── sampling ──────────────────────────────────────────────────────────────────


def test_every_declared_surface_point_is_within_the_spacing_of_a_sample(tmp_path: Path) -> None:
    geometry, base = load_static_geometry(_cell_geometry(tmp_path))
    spacing = 0.015 / 4.0
    pts = surface_points(geometry, base, spacing)
    rng = np.random.default_rng(0)
    # The wall's -y face and the shelf triangle, densely probed.
    face = np.column_stack(
        [rng.uniform(0.1, 0.5, 400), np.full(400, -0.568), rng.uniform(-0.3, 0.3, 400)]
    )
    w = rng.dirichlet([1.0, 1.0, 1.0], 400)
    tri = w @ np.array([[0.4, -0.2, 0.0], [0.4, 0.2, 0.0], [0.6, 0.0, 0.0]])
    for probe in (face, tri):
        nearest = np.min(np.linalg.norm(probe[:, None, :] - pts[None, :, :], axis=2), axis=1)
        assert nearest.max() <= spacing + 1e-9


def test_points_round_trip_through_the_bridge_file_format(tmp_path: Path) -> None:
    geometry, base = load_static_geometry(_cell_geometry(tmp_path))
    pts = surface_points(geometry, base, 0.01)
    out = tmp_path / "points.txt"
    write_points_file(pts, out)
    lines = out.read_text().splitlines()
    assert lines[0].startswith("#")
    back = np.loadtxt(out)
    assert back.shape == pts.shape
    assert np.allclose(back, pts, atol=1e-5)


def test_a_broken_mesh_reference_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "static.yaml"
    path.write_text(yaml.safe_dump({"static_geometry": [{"kind": "mesh", "file": "nope.npz"}]}))
    geometry, base = load_static_geometry(path)
    with pytest.raises(ROSConfigError, match=r"nope\.npz"):
        surface_points(geometry, base, 0.01)
    with pytest.raises(ROSConfigError):
        load_static_geometry(tmp_path / "missing.yaml")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _invoke(scene: Path) -> Any:
    from openral_cli.deploy_sim import resolve_launch_invocation

    return resolve_launch_invocation(
        config=scene,
        robot_override="openarm",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        enable_dashboard=False,
        hal_mode="sim",
    )


def _cell_scene(tmp_path: Path, runtime: dict[str, object]) -> Path:
    data = yaml.safe_load(_CELL.read_text(encoding="utf-8"))
    data["runtime"].update(runtime)
    scene = tmp_path / "cell.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")
    return scene


def test_deploy_sim_forwards_a_scene_relative_file_and_refuses_a_missing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "thor")
    off = _invoke(_cell_scene(tmp_path, {})).argv_template
    assert not any(a.startswith("static_geometry_file") for a in off), "off: argv unchanged"
    geometry = _cell_geometry(tmp_path)
    on = _invoke(_cell_scene(tmp_path, {"static_geometry_file": geometry.name})).argv_template
    assert f"static_geometry_file:={geometry.resolve()}" in on
    with pytest.raises(ROSConfigError, match="not found"):
        _invoke(_cell_scene(tmp_path, {"static_geometry_file": "nowhere.yaml"}))


# ── launch ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def launch_module() -> Any:
    """The real launch file, loaded by path — it is not an importable package."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_static", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_static"] = module
    spec.loader.exec_module(module)
    return module


def test_the_launch_samples_a_quarter_cell_apart_for_the_bridge(
    launch_module: Any, tmp_path: Path
) -> None:
    geometry = _cell_geometry(tmp_path)
    out = Path(launch_module._static_points_file(str(geometry), 0.015))
    try:
        pts = np.loadtxt(out)
        loaded, base = load_static_geometry(geometry)
        assert pts.shape == surface_points(loaded, base, 0.015 / 4.0).shape
    finally:
        out.unlink(missing_ok=True)


def test_the_same_geometry_reuses_one_points_file(launch_module: Any, tmp_path: Path) -> None:
    """One file per distinct geometry, not one per launch (they leaked ~350 MB a run)."""
    geometry = _cell_geometry(tmp_path)
    first = Path(launch_module._static_points_file(str(geometry), 0.015))
    try:
        stamp = first.stat().st_mtime_ns
        again = Path(launch_module._static_points_file(str(geometry), 0.015))
        assert again == first and first.stat().st_mtime_ns == stamp, "reused, not rewritten"
        other = Path(launch_module._static_points_file(str(geometry), 0.02))
        assert other != first, "a different sampling is a different file"
        other.unlink()
        assert not list(first.parent.glob(f"{first.name}.*.partial"))
    finally:
        first.unlink(missing_ok=True)


def test_a_cached_file_is_reused_without_sampling_and_an_edit_is_not(
    launch_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache is keyed on the inputs: a hit samples nothing; an edited geometry re-samples.

    Sampling a 15 M-point scene and reducing it to cells cost ~14 s per call, three calls per
    launch, even when the file already existed (the key was a digest of the sampled points).
    """
    import openral_safety.static_geometry as sg

    geometry = _cell_geometry(tmp_path)
    first = Path(launch_module._static_points_file(str(geometry), 0.015, kernel_cells=True))
    calls: list[float] = []
    real = sg.surface_points

    def counting(*a: Any, **k: Any) -> Any:
        calls.append(1.0)
        return real(*a, **k)

    monkeypatch.setattr(sg, "surface_points", counting)
    try:
        assert (
            Path(launch_module._static_points_file(str(geometry), 0.015, kernel_cells=True))
            == first
        )
        assert not calls, "a cache hit must not sample"
        geometry.write_text(geometry.read_text() + "\n# edited\n")
        edited = Path(launch_module._static_points_file(str(geometry), 0.015, kernel_cells=True))
        assert edited != first and calls, "an edited geometry is a new file"
        edited.unlink()
    finally:
        first.unlink(missing_ok=True)


def test_the_kernels_cells_file_names_exactly_the_cells_the_bridge_marks(
    launch_module: Any, tmp_path: Path
) -> None:
    """Issue #418 K7: one centre per cell of the bridge's points, on the base frame's lattice."""
    geometry = _cell_geometry(tmp_path)
    res = 0.015
    bridge = Path(launch_module._static_points_file(str(geometry), res))
    cells = Path(launch_module._static_points_file(str(geometry), res, kernel_cells=True))
    try:
        assert cells != bridge
        # The bridge's marking on a fixed-base grid: floor of each point it reads, per cell.
        marked = {tuple(k) for k in np.floor(np.loadtxt(bridge) / res).astype(np.int64)}
        centres = np.loadtxt(cells)
        keys = np.floor(centres / res).astype(np.int64)
        assert {tuple(k) for k in keys} == marked
        assert len(centres) == len(marked), "one centre per cell"
        assert np.allclose(centres, (keys + 0.5) * res), "each point is its cell's centre"
    finally:
        bridge.unlink(missing_ok=True)
        cells.unlink(missing_ok=True)


def test_the_cache_lives_in_a_directory_only_the_user_can_write(
    launch_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A predictable name in shared /tmp let another local user plant the geometry."""
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path / "cache"))
    out = Path(launch_module._static_points_file(str(_cell_geometry(tmp_path)), 0.015))
    assert out.parent == tmp_path / "cache" / "static_geometry"
    assert out.parent.stat().st_mode & 0o777 == 0o700
