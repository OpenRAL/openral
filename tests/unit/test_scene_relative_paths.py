"""A scene's relative asset paths resolve beside the scene file, from any working directory.

A scene shipped outside this checkout (OpenRAL Pro's `scenes/` + `assets/`) names its stage,
props and static geometry relative to itself; one written for this repo may still name them
relative to the repo root or the cwd, and keeps working.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_core import DeployScene, SimScene, load_scene_strict
from openral_core.scene_paths import anchor_scene_paths


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    """Another repo's layout: scenes/<tier>/x.yaml next to assets/."""
    assets = tmp_path / "repo" / "assets"
    assets.mkdir(parents=True)
    for name in ("stage.usd", "prop.usd", "static.yaml"):
        (assets / name).write_text("")
    return tmp_path / "repo", assets


def _scene(task: bool) -> dict[str, object]:
    d: dict[str, object] = {
        "robot_id": "openarm",
        "scene": {
            "id": "isaac_sim",
            "backend": "isaacsim",
            "assets_uri": "../../assets/stage.usd",
            "backend_options": {
                "objects": [
                    {"name": "a", "usd": "../../assets/prop.usd", "xyz": [0, 0, 0]},
                    {"name": "b", "usd": "isaac:Isaac/Props/x.usd", "xyz": [0, 0, 0]},
                ]
            },
        },
    }
    if task:
        d["task"] = {"id": "t/0", "scene_id": "isaac_sim", "instruction": "x", "max_steps": 10}
    return d


@pytest.mark.parametrize("tier", ["deploy", "sim"])
def test_relative_paths_resolve_beside_the_scene_from_any_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str
) -> None:
    repo, assets = _repo(tmp_path)
    path = repo / "scenes" / tier / "x.yaml"
    path.parent.mkdir(parents=True)
    raw = _scene(task=tier == "sim")
    if tier == "deploy":
        raw["runtime"] = {"static_geometry_file": "../../assets/static.yaml"}
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.chdir(tmp_path)  # neither this checkout nor the scene's repo
    loads = (
        [load_scene_strict(str(path), DeployScene), DeployScene.from_yaml(str(path))]
        if tier == "deploy"
        else [load_scene_strict(str(path), SimScene), SimScene.from_yaml(str(path))]
    )
    for scene in loads:
        assert scene.scene.assets_uri == str((assets / "stage.usd").resolve())
        objs = scene.scene.backend_options["objects"]
        assert isinstance(objs, list)
        assert [o["usd"] for o in objs] == [
            str((assets / "prop.usd").resolve()),
            "isaac:Isaac/Props/x.usd",
        ]
        if tier == "deploy":
            assert scene.runtime is not None
            assert scene.runtime.static_geometry_file == str((assets / "static.yaml").resolve())


def test_a_path_with_no_file_beside_the_scene_is_left_for_the_repo_root_lookup(
    tmp_path: Path,
) -> None:
    # This repo's own style: relative to the checkout root, resolved later by the backend.
    raw = {
        "scene": {"assets_uri": "assets/stage.usd"},
        "runtime": {"static_geometry_file": "s.yaml"},
    }
    assert anchor_scene_paths(raw, tmp_path) == raw


def test_the_isaac_robot_spec_resolves_the_urdf_outside_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openral_sim.backends.isaac_sim import _build_robot_spec
    from openral_sim.registry import ROBOTS

    monkeypatch.chdir(tmp_path)
    spec = _build_robot_spec(ROBOTS.get("openarm")(), "openarm")
    assert Path(spec["urdf_path"]).is_file()
