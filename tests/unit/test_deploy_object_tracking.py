"""``object_tracking`` (issue #418) through ``deploy sim`` / ``deploy run`` and the launch.

Real committed scenes; ``resolve_launch_invocation`` is the CLI's own resolver, the launch
helper is loaded from the real launch file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from openral_cli.deploy_sim import resolve_launch_invocation
from openral_core.exceptions import ROSConfigError

_REPO = Path(__file__).resolve().parents[2]
_PICK_ORANGE = _REPO / "scenes/deploy/isaac_leisaac_pick_orange.yaml"
_REAL = _REPO / "scenes/deploy/galaxea_a1_bench.yaml"
_LAUNCH = _REPO / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"


def _argv(config: Path, mode: str, overrides: dict[str, object]) -> list[str]:
    return resolve_launch_invocation(
        config=config,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode=mode,
        runtime_overrides=overrides,
    ).argv_template


def test_deploy_sim_forwards_the_runtime_override_and_leaves_off_unchanged() -> None:
    argv = _argv(_PICK_ORANGE, "sim", {"object_tracking": "bind", "grounding_wait_s": 8})
    assert "object_tracking:=bind" in argv and "grounding_wait_s:=8.0" in argv
    assert "grounding_required:=false" in argv
    off = _argv(_PICK_ORANGE, "sim", {})
    assert not [a for a in off if a.startswith(("object_tracking:=", "grounding_"))]


def test_a_direct_dispatch_deploy_names_the_queries_from_its_preload_prompt() -> None:
    """No reasoner means no initial task; the preload prompt is the graded one."""
    import json

    argv = _argv(
        _PICK_ORANGE,
        "sim",
        {
            "object_tracking": "shadow",
            "grounding_wait_s": 8,
            "enable_reasoner": False,
            "preload_rskill_id": "OpenRAL/rskill-smolvla-so101-leisaac_pick_orange-bf16",
            "preload_prompt": "Grab orange and place into plate",
        },
    )
    (raw,) = [a.split(":=", 1)[1] for a in argv if a.startswith("object_queries_json:=")]
    assert [(q["label"], q["role"]) for q in json.loads(raw)] == [
        ("orange", "target"),
        ("plate", "destination"),
    ]


def test_an_override_must_validate_whole() -> None:
    with pytest.raises(ROSConfigError, match="grounding_wait_s"):
        _argv(_PICK_ORANGE, "sim", {"object_tracking": "shadow"})


@pytest.mark.parametrize("mode", ["bind", "on"])
def test_deploy_run_refuses_bind_and_on(mode: str) -> None:
    with pytest.raises(ROSConfigError, match="sim-only until Safety-WG"):
        _argv(_REAL, "real", {"object_tracking": mode, "grounding_wait_s": 5})


def test_deploy_run_allows_shadow() -> None:
    argv = _argv(_REAL, "real", {"object_tracking": "shadow", "grounding_wait_s": 5})
    assert "object_tracking:=shadow" in argv


@pytest.fixture(scope="module")
def object_tracking() -> object:
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_tracking", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_tracking"] = module
    spec.loader.exec_module(module)
    return module._object_tracking


_OK = {
    "hal_mode": "sim",
    "vision_attachment_enabled": True,
    "enable_octomap": True,
    "grounding_wait_s": "8",
    "cameras": "",
    "depth_cameras": ["top_depth"],
}


def test_the_launch_tracks_every_depth_camera_unless_named(object_tracking: object) -> None:
    f = object_tracking
    assert f("off", **_OK) == ("off", ())  # type: ignore[operator]
    assert f("bind", **_OK) == ("bind", ("top_depth",))  # type: ignore[operator]
    assert f("on", **{**_OK, "cameras": "a, b"}) == ("on", ("a", "b"))  # type: ignore[operator]


@pytest.mark.parametrize(
    ("mode", "change", "match"),
    [
        ("maybe", {}, "expected one of"),
        ("bind", {"hal_mode": "real"}, "sim-only"),
        ("shadow", {"grounding_wait_s": ""}, "grounding_wait_s"),
        ("shadow", {"enable_octomap": False}, "octomap"),
        ("shadow", {"vision_attachment_enabled": False}, "vision attachment"),
        ("shadow", {"depth_cameras": []}, "no depth camera"),
    ],
)
def test_the_launch_refuses_an_incomplete_posture(
    object_tracking: object, mode: str, change: dict[str, object], match: str
) -> None:
    with pytest.raises(ROSConfigError, match=match):
        object_tracking(mode, **{**_OK, **change})  # type: ignore[operator]


def test_a_locator_caches_every_tracked_camera_under_its_own_name(object_tracking: object) -> None:
    """The tracker asks locate_in_view on its depth camera; the locator must hold its frame."""
    f = sys.modules["deploy_e2e_launch_tracking"]._locator_cameras
    cams, infos = f("top", "/openral/cameras/top/image", ["top=/info"], ("top_depth", "top"))
    assert cams == ["top=/openral/cameras/top/image", "top_depth=/openral/cameras/top_depth/image"]
    assert infos == ["top=/info", "top_depth=/openral/cameras/top_depth/camera_info"]
    assert f("top", "/t", [""], ()) == (["top=/t"], [""])  # nothing tracked: unchanged
