"""Isaac scene: a stage's own objects, its load fixes, and regions relative to an object.

The shipped ``scenes/sim/isaac_leisaac_pick_orange.yaml`` (LeIsaac's PickOrange
kitchen) is the fixture: its oranges and plate live inside the environment USD
(``stage_objects``), the USD comes from a pinned archive (``environment_archive``),
it needs ``strip_xform_colliders`` + a ``dome_light`` on Isaac 6.1, and its success
region follows the plate (``IsaacPlacementRegion.relative_to``). Host side: the
options validate and resolve into the sidecar spec. Sidecar side: the scene
module's pure helpers, imported without Isaac.
"""

from __future__ import annotations

import copy
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from openral_core import SimScene, load_scene_strict
from openral_core.exceptions import ROSConfigError
from openral_sim.backends.isaac_sim import (
    IsaacDomeLight,
    IsaacEnvironmentArchive,
    IsaacSimOptions,
    IsaacStageObject,
    _environment_spec,
    _placement,
    _placement_success_spec,
    fetch_environment_archive,
)
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[2]
_SCENE = _REPO / "scenes/sim/isaac_leisaac_pick_orange.yaml"
_PROMPT = "Pick three oranges and put them into the plate"


def _options() -> dict[str, Any]:
    return copy.deepcopy(yaml.safe_load(_SCENE.read_text())["scene"]["backend_options"])


def test_the_shipped_scene_loads_and_validates() -> None:
    scene = load_scene_strict(str(_SCENE), SimScene)
    opts = IsaacSimOptions.model_validate(scene.scene.backend_options)
    assert [o.name for o in opts.stage_objects] == ["orange_1", "orange_2", "orange_3", "plate"]
    assert opts.strip_xform_colliders is True
    assert opts.dome_light == IsaacDomeLight(intensity=3000.0, color=(0.75, 0.75, 0.75))
    assert opts.environment_archive is not None
    assert opts.environment_archive.usd == "kitchen_with_orange/scene.usd"
    assert scene.task.instruction == _PROMPT
    assert opts.placement_success is not None
    assert opts.placement_success.regions["on_plate"].relative_to == "plate"


def test_default_options_change_nothing_for_existing_scenes() -> None:
    assert _environment_spec(IsaacSimOptions()) == {
        "stage_objects": [],
        "strip_xform_colliders": False,
        "dome_light": None,
        "joint_effort_limits": {},
        "physx_scene": None,
        "collision_approximation": {},
    }


def test_the_pick_orange_scene_matches_leisaacs_contact_physics() -> None:
    # LeIsaac: convexDecomposition finger colliders; single_arm_env_cfg sim.physx values.
    spec = _environment_spec(IsaacSimOptions.model_validate(_options()))
    assert spec["collision_approximation"] == {"gripper|jaw": "convexDecomposition"}
    assert spec["physx_scene"] == {
        "friction_correlation_distance": 0.00625,
        "bounce_threshold": 0.01,
    }


def test_an_unknown_collision_approximation_is_refused() -> None:
    with pytest.raises(ValueError):
        IsaacSimOptions.model_validate({"collision_approximation": {"jaw": "meshSimplification"}})


def test_the_pick_orange_scene_caps_the_jaw_at_leisaacs_orange_effort() -> None:
    # LeIsaac: gripper effort = nearest object's mass / 0.15; every orange is 0.15 kg.
    spec = _environment_spec(IsaacSimOptions.model_validate(_options()))
    assert spec["joint_effort_limits"] == {"gripper": 1.0}


def test_an_effort_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        IsaacSimOptions(joint_effort_limits={"gripper": 0.0})


def test_spec_carries_the_stage_fixes_and_the_stage_objects() -> None:
    spec = _environment_spec(IsaacSimOptions.model_validate(_options()))
    assert spec["strip_xform_colliders"] is True
    assert spec["dome_light"] == {"intensity": 3000.0, "color": [0.75, 0.75, 0.75]}
    assert {"name": "plate", "prim": "Plate/Plate"} in spec["stage_objects"]


def test_the_rubric_keeps_the_region_relative_to_the_plate() -> None:
    opts = IsaacSimOptions.model_validate(_options())
    spec = _placement_success_spec(opts.placement_success, _PROMPT)
    assert spec is not None
    assert [p["object"] for p in spec["placements"]] == ["orange_1", "orange_2", "orange_3"]
    assert spec["placements"][0]["target"] == {
        "min": [-0.10, -0.10, -0.07],
        "max": [0.10, 0.10, 0.07],
        "relative_to": "plate",
    }


def test_a_region_relative_to_an_unknown_object_is_refused() -> None:
    opts = _options()
    opts["placement_success"]["regions"]["on_plate"]["relative_to"] = "bowl"
    with pytest.raises(ValidationError, match="bowl"):
        IsaacSimOptions.model_validate(opts)


def test_an_object_cannot_be_placed_relative_to_itself() -> None:
    opts = _options()
    opts["placement_success"]["regions"]["on_plate"]["relative_to"] = "orange_1"
    with pytest.raises(ValidationError, match="relative_to itself"):
        IsaacSimOptions.model_validate(opts)


def test_stage_objects_are_graded_like_scene_objects() -> None:
    opts = _options()
    opts["stage_objects"] = [o for o in opts["stage_objects"] if o["name"] != "orange_3"]
    with pytest.raises(ValidationError, match="orange_3"):
        IsaacSimOptions.model_validate(opts)


def test_a_stage_object_name_may_not_repeat_an_object_name() -> None:
    opts = _options()
    opts["objects"] = [{"name": "plate", "usd": "isaac:Isaac/Props/plate.usd", "xyz": [0, 0, 0]}]
    with pytest.raises(ValidationError, match="duplicate names"):
        IsaacSimOptions.model_validate(opts)


@pytest.mark.parametrize("prim", ["/Root/Plate", "Plate//Plate", "Plate/../Root", ""])
def test_stage_object_prim_is_a_relative_usd_path(prim: str) -> None:
    with pytest.raises(ValidationError):
        IsaacStageObject(name="plate", prim=prim)


@pytest.mark.parametrize(
    "field",
    [
        {"url": "http://example.org/a.zip"},  # not https
        {"url": "https://example.org/a.tar.gz"},  # not a zip
        {"sha256": "D314" + "0" * 60},  # not lowercase hex
        {"usd": "../scene.usd"},  # leaves the archive
        {"usd": "/abs/scene.usd"},
        {"usd": "kitchen/scene.obj"},  # not a USD
    ],
)
def test_environment_archive_is_pinned_and_inside(field: dict[str, str]) -> None:
    base = _options()["environment_archive"]
    with pytest.raises(ValidationError):
        IsaacEnvironmentArchive.model_validate({**base, **field})


def test_dome_light_color_is_unit_rgb() -> None:
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        IsaacDomeLight(intensity=3000.0, color=(1.5, 0.5, 0.5))


def test_an_archive_and_an_assets_uri_together_are_refused() -> None:
    scene = load_scene_strict(str(_SCENE), SimScene)
    env = scene.model_copy(
        update={"scene": scene.scene.model_copy(update={"assets_uri": "isaac:Isaac/x.usd"})}
    )
    opts = IsaacSimOptions.model_validate(env.scene.backend_options)
    with pytest.raises(ROSConfigError, match="give one"):
        _placement(env, opts)  # type: ignore[arg-type]  # reason: SimScene carries the fields


def test_a_completed_archive_extraction_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marked extraction answers without touching the network."""
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path))
    archive = IsaacEnvironmentArchive.model_validate(_options()["environment_archive"])
    root = tmp_path / "isaac-environments" / archive.sha256[:16]
    (root / "kitchen_with_orange").mkdir(parents=True)
    (root / archive.usd).write_text("#usda 1.0\n")
    (root / ".extracted").write_text(archive.sha256)
    assert fetch_environment_archive(archive) == str(root / archive.usd)


def _serve_zip(
    monkeypatch: pytest.MonkeyPatch,
    members: dict[str, str],
    compression: int = zipfile.ZIP_STORED,
) -> IsaacEnvironmentArchive:
    """A pinned archive whose download is served from memory (the network boundary)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    blob = buf.getvalue()
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(blob))
    pinned = IsaacEnvironmentArchive.model_validate(_options()["environment_archive"])
    return pinned.model_copy(update={"sha256": hashlib.sha256(blob).hexdigest()})


def test_an_archive_is_extracted_beside_the_cache_and_swapped_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale partial tree is replaced whole; no staging dir or part file is left behind."""
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path))
    archive = _serve_zip(monkeypatch, {"kitchen_with_orange/scene.usd": "#usda 1.0\n"})
    cache = tmp_path / "isaac-environments"
    root = cache / archive.sha256[:16]
    (root / "half_written").mkdir(parents=True)  # a crashed run's leftovers, no marker
    assert fetch_environment_archive(archive) == str(root / archive.usd)
    assert (root / ".extracted").read_text() == archive.sha256
    assert sorted(p.name for p in root.iterdir()) == [".extracted", "kitchen_with_orange"]
    assert sorted(p.name for p in cache.iterdir()) == [root.name, f"{root.name}.lock"]


def test_an_archive_without_its_usd_leaves_the_cache_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path))
    archive = _serve_zip(monkeypatch, {"elsewhere/scene.usd": "#usda 1.0\n"})
    with pytest.raises(ROSConfigError, match="has no"):
        fetch_environment_archive(archive)
    assert list((tmp_path / "isaac-environments").iterdir()) == []


def test_a_zip_bomb_archive_is_refused_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned digest does not bound what the zip expands to; the ratio cap does."""
    monkeypatch.setenv("OPENRAL_CACHE_DIR", str(tmp_path))
    archive = _serve_zip(
        monkeypatch,
        {"kitchen_with_orange/scene.usd": "#usda 1.0\n", "pad.bin": "\0" * (8 << 20)},
        compression=zipfile.ZIP_DEFLATED,
    )
    with pytest.raises(ROSConfigError, match="zip bomb"):
        fetch_environment_archive(archive)
    assert list((tmp_path / "isaac-environments").iterdir()) == []


# ── sidecar helpers (pure; imported without Isaac) ───────────────────────────


@pytest.fixture(scope="module")
def scene_mod() -> Any:
    tools = str(_REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import isaac_manifest_scene

    return isaac_manifest_scene


def test_a_relative_box_follows_its_reference(scene_mod: Any) -> None:
    opts = IsaacSimOptions.model_validate(_options())
    spec = _placement_success_spec(opts.placement_success, _PROMPT)
    assert spec is not None
    placement = spec["placements"][0]
    up, level = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])
    plate = np.array([2.0, -0.4, 0.9])
    for plate_at, orange, grade in [
        (plate, plate + np.array([0.05, -0.05, 0.03]), "complete"),
        (plate, plate + np.array([0.12, 0.0, 0.03]), "fail"),  # off the plate's x/y band
        (plate, plate + np.array([0.0, 0.0, 0.10]), "fail"),  # above the height band
        # The plate moved: the region follows it.
        (plate + np.array([0.3, 0.0, 0.0]), plate + np.array([0.3, 0.0, 0.0]), "complete"),
    ]:
        at = scene_mod.placement_boxes_at(placement, {"plate": plate_at})
        assert scene_mod.placement_grade(orange, level, up, at, 20.0, spec["grades"]) == grade


def test_world_boxes_pass_through(scene_mod: Any) -> None:
    placement = {"object": "o", "target": {"min": [0, 0, 0], "max": [1, 1, 1]}, "adjacent": []}
    assert scene_mod.placement_boxes_at(placement, {}) == placement


# ── round trips (hypothesis) ─────────────────────────────────────────────────

_ident = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,8}", fullmatch=True)


@given(name=_ident, parts=st.lists(_ident, min_size=1, max_size=3))
def test_stage_object_round_trips(name: str, parts: list[str]) -> None:
    obj = IsaacStageObject(name=name, prim="/".join(parts))
    assert IsaacStageObject.model_validate_json(obj.model_dump_json()) == obj


@given(
    intensity=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False),
    color=st.tuples(*(st.floats(min_value=0.0, max_value=1.0, allow_nan=False),) * 3),
)
def test_dome_light_round_trips(intensity: float, color: tuple[float, float, float]) -> None:
    light = IsaacDomeLight(intensity=intensity, color=color)
    assert IsaacDomeLight.model_validate_json(light.model_dump_json()) == light


@given(sha=st.from_regex(r"[0-9a-f]{64}", fullmatch=True), stem=_ident)
def test_environment_archive_round_trips(sha: str, stem: str) -> None:
    archive = IsaacEnvironmentArchive(
        url=f"https://example.org/{stem}.zip", sha256=sha, usd=f"{stem}/scene.usd"
    )
    assert IsaacEnvironmentArchive.model_validate_json(archive.model_dump_json()) == archive


@pytest.mark.parametrize("model", [IsaacStageObject, IsaacDomeLight, IsaacEnvironmentArchive])
def test_models_export_a_json_schema(model: type[Any]) -> None:
    schema = model.model_json_schema()
    assert schema["title"] == model.__name__
    assert set(model.model_fields) <= set(schema["properties"])
