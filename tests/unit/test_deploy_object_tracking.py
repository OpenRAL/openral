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


def _queries(scene: object) -> list[dict[str, object]]:
    import json

    from openral_cli.deploy_sim import _object_queries_json

    return list(json.loads(_object_queries_json(scene, "Grab orange and place into plate")))


def _tracking_pick_orange() -> object:
    from openral_core import DeployScene

    scene = DeployScene.from_yaml(str(_PICK_ORANGE))
    assert scene.runtime is not None
    return scene.model_copy(
        update={"runtime": scene.runtime.model_copy(update={"object_tracking": "shadow"})}
    )


def test_a_sim_ground_truth_search_box_never_seeds_the_tracker() -> None:
    """PickOrange's search_box is measured from the sim (``sim:orange_1``): no seed.

    A real deploy has no ground-truth box, so the tracker must ground the orange from its
    own sensing. The HAL's declaration keeps the box (the legacy path is unchanged).
    """
    scene = _tracking_pick_orange()
    decl = scene.grasp_declaration
    assert decl is not None and decl.search_box is not None
    assert decl.search_box.evidence_ref.startswith("sim:")
    assert [q.get("seed") for q in _queries(scene)] == [None, None]
    argv = _argv(_PICK_ORANGE, "sim", {"object_tracking": "shadow", "grounding_wait_s": 8})
    (raw,) = [a.split(":=", 1)[1] for a in argv if a.startswith("grasp_declaration_json:=")]
    assert '"evidence_ref":"sim:orange_1"' in raw


def test_a_user_declared_search_box_still_seeds_the_tracker() -> None:
    scene = _tracking_pick_orange()
    decl = scene.grasp_declaration
    assert decl is not None and decl.search_box is not None
    box = decl.search_box.model_copy(update={"evidence_ref": "operator:tote_left"})
    scene = scene.model_copy(
        update={"grasp_declaration": decl.model_copy(update={"search_box": box})}
    )
    seeds = {q["label"]: q.get("seed") for q in _queries(scene)}
    assert seeds["plate"] is None
    assert isinstance(seeds["orange"], dict)
    assert seeds["orange"]["evidence_ref"] == "operator:tote_left"


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
def launch_module() -> object:
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_tracking", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_tracking"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def object_tracking(launch_module: object) -> object:
    return launch_module._object_tracking  # type: ignore[attr-defined]


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


def test_a_locator_caches_every_tracked_camera_under_its_own_name(launch_module: object) -> None:
    """The tracker asks locate_in_view on its depth camera; the locator must hold its frame."""
    f = launch_module._locator_cameras  # type: ignore[attr-defined]
    cams, infos = f("top", "/openral/cameras/top/image", ["top=/info"], ("top_depth", "top"))
    assert cams == ["top=/openral/cameras/top/image", "top_depth=/openral/cameras/top_depth/image"]
    assert infos == ["top=/info", "top_depth=/openral/cameras/top_depth/camera_info"]
    assert f("top", "/t", [""], ()) == (["top=/t"], [""])  # nothing tracked: unchanged


def test_only_on_makes_the_tracked_objects_the_kernels_grasp_exemption_source(
    launch_module: object,
) -> None:
    f = launch_module._tracked_grasp_exemption  # type: ignore[attr-defined]
    for mode in ("off", "shadow", "bind"):
        assert f(mode, grasp_allowance_enabled=True) is False
        assert f(mode, grasp_allowance_enabled=False) is False
    assert f("on", grasp_allowance_enabled=True) is True
    with pytest.raises(ROSConfigError, match="grasp_allowance_enabled"):
        f("on", grasp_allowance_enabled=False)


@pytest.mark.parametrize(
    ("robot", "jaws"),
    [("so101_follower", 1), ("droid_franka_robotiq", 1), ("openarm", 2)],
)
def test_bind_and_on_wire_the_kernels_jaw_open_release(
    launch_module: object, robot: str, jaws: int
) -> None:
    """K5 runs in bind/on with the tracker's own gap and dwell; never in off/shadow."""
    from openral_core import RobotDescription

    f = launch_module._tracked_release_params  # type: ignore[attr-defined]
    d = RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml"))
    for mode in ("off", "shadow"):
        assert f(d, mode) == {}
    p = f(d, "bind")
    assert len(p["tracked_release_jaw_joints"]) == jaws
    assert set(p["tracked_release_jaw_open_sign"]) <= {1.0, -1.0}
    cals = [j.closure_calibration for j in d.joints if j.name in p["tracked_release_jaw_joints"]]
    assert p["tracked_release_gap"] == max(c.stall_gap / 2 for c in cals)
    assert p["tracked_release_dwell_s"] == launch_module._TRACKED_RELEASE_DWELL_S  # type: ignore[attr-defined]
    assert f(d, "on") == p
