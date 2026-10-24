"""``object_tracking`` (issue #418) through ``deploy sim`` / ``deploy run`` and the launch.

Real committed scenes; ``resolve_launch_invocation`` is the CLI's own resolver, the launch
helper is loaded from the real launch file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from openral_cli.deploy_sim import resolve_launch_invocation
from openral_core import DeployScene
from openral_core.exceptions import ROSConfigError
from openral_world_state.object_tracker.session import RELEASE_DWELL_S

_REPO = Path(__file__).resolve().parents[2]
_PICK_ORANGE = _REPO / "scenes/deploy/isaac_leisaac_pick_orange.yaml"
_REAL = _REPO / "scenes/deploy/galaxea_a1_bench.yaml"
_LAUNCH = _REPO / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"


def _argv(
    config: Path, mode: str, overrides: dict[str, object], locators: tuple[str, ...] = ()
) -> list[str]:
    return resolve_launch_invocation(
        config=config,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode=mode,
        runtime_overrides=overrides,
        object_detector_locators=list(locators),
    ).argv_template


def test_deploy_sim_forwards_the_runtime_override_and_leaves_off_unchanged() -> None:
    # The scene commits the Spark R372 posture: `on`, 8 s, grounding required.
    scene = _argv(_PICK_ORANGE, "sim", {})
    assert "object_tracking:=on" in scene and "grounding_wait_s:=8.0" in scene
    assert "grounding_required:=true" in scene
    argv = _argv(
        _PICK_ORANGE,
        "sim",
        {"object_tracking": "bind", "grounding_wait_s": 5, "grounding_required": False},
    )
    assert "object_tracking:=bind" in argv and "grounding_wait_s:=5.0" in argv
    assert "grounding_required:=false" in argv
    off = _argv(_PICK_ORANGE, "sim", {"object_tracking": "off"})
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
        # The orange has no seed: without a locator deploy sim refuses to configure.
        locators=("omdet-turbo-locator",),
    )
    (raw,) = [a.split(":=", 1)[1] for a in argv if a.startswith("object_queries_json:=")]
    assert [(q["label"], q["role"]) for q in json.loads(raw)] == [
        ("orange", "target"),
        ("plate", "destination"),
    ]


def _queries(scene: DeployScene) -> list[dict[str, Any]]:
    import json

    from openral_cli.deploy_sim import _object_queries_json

    return list(json.loads(_object_queries_json(scene, "Grab orange and place into plate")))


def _tracking_pick_orange() -> DeployScene:
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


def test_tracking_without_a_locator_or_seed_refuses_to_configure() -> None:
    """PickOrange's orange query has no seed (its box is sim ground truth): without an
    on-demand locator the tracker could never ground it, so deploy sim refuses."""
    overrides = {
        "object_tracking": "shadow",
        "grounding_wait_s": 8,
        "enable_reasoner": False,
        "preload_rskill_id": "OpenRAL/rskill-smolvla-so101-leisaac_pick_orange-bf16",
        "preload_prompt": "Grab orange and place into plate",
    }
    with pytest.raises(ROSConfigError, match="could never ground"):
        resolve_launch_invocation(
            config=_PICK_ORANGE,
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_mode="sim",
            runtime_overrides=overrides,
            object_detector_locators=[],
        )
    argv = resolve_launch_invocation(
        config=_PICK_ORANGE,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="sim",
        runtime_overrides=overrides,
        object_detector_locators=["omdet-turbo-locator"],
    ).argv_template
    assert any(a.startswith("object_queries_json:=") for a in argv)
    # Tracking off composes no queries, so nothing is refused.
    resolve_launch_invocation(
        config=_PICK_ORANGE,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="sim",
        runtime_overrides={
            **{k: v for k, v in overrides.items() if k.startswith(("enable", "pre"))},
            "object_tracking": "off",
        },
        object_detector_locators=[],
    )


def test_an_override_must_validate_whole() -> None:
    with pytest.raises(ROSConfigError, match="grounding_wait_s"):
        _argv(_PICK_ORANGE, "sim", {"grounding_wait_s": None})
    with pytest.raises(ROSConfigError, match="grounding_wait_s"):
        _argv(_PICK_ORANGE, "sim", {"object_tracking": "shadow", "grounding_wait_s": None})


def test_a_shadow_override_does_not_inherit_the_scenes_grounding_required() -> None:
    """Shadow changes no decision: the scene's committed `grounding_required: true` (for `on`)
    must not make the runner refuse goals under `--runtime object_tracking=shadow`."""
    scene = DeployScene.from_yaml(str(_PICK_ORANGE))
    assert scene.runtime is not None and scene.runtime.grounding_required is True
    argv = _argv(_PICK_ORANGE, "sim", {"object_tracking": "shadow"})
    assert "object_tracking:=shadow" in argv
    assert "grounding_required:=false" in argv
    # The scene's own posture and a bind override keep it.
    assert "grounding_required:=true" in _argv(_PICK_ORANGE, "sim", {})
    assert "grounding_required:=true" in _argv(_PICK_ORANGE, "sim", {"object_tracking": "bind"})


@pytest.mark.parametrize(
    "path",
    [_PICK_ORANGE, _REPO / "scenes/deploy/isaac_arena_droid_banana_in_bowl.yaml"],
)
def test_the_isaac_twins_commit_a_locator_for_their_unseeded_target(path: Path) -> None:
    """Their header command runs as written: the TARGET query has no seed, so without a
    committed locator `deploy sim` refuses to configure."""
    task = {
        _PICK_ORANGE: "Grab orange and place into plate",
    }.get(path, "pick up the banana and place it in the bowl")
    argv = resolve_launch_invocation(
        config=path,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="sim",
        initial_task_prompt=task,
    ).argv_template
    assert any(a.startswith("object_queries_json:=") for a in argv)
    (locators,) = [a for a in argv if a.startswith("object_detector_locators:=")]
    assert locators.endswith("rskills/omdet-turbo-locator/rskill.yaml")


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


def test_only_on_with_k2_lets_a_bound_payloads_measured_cells_stand(launch_module: object) -> None:
    f = launch_module._payload_own_cells  # type: ignore[attr-defined]
    for mode in ("off", "shadow", "bind"):
        assert f(mode, tracked_grasp_exemption_enabled=True) is False
    assert f("on", tracked_grasp_exemption_enabled=True) is True
    assert f("on", tracked_grasp_exemption_enabled=False) is False


def test_only_on_with_workspace_contact_narrows_it_to_explained_cells(
    launch_module: object,
) -> None:
    """K7 rides `on` and needs the workspace rule it narrows; never off/shadow/bind."""
    f = launch_module._workspace_explained_cells  # type: ignore[attr-defined]
    for mode in ("off", "shadow", "bind"):
        assert f(mode, workspace_contact_enabled=True) is False
    assert f("on", workspace_contact_enabled=True) is True
    assert f("on", workspace_contact_enabled=False) is False


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
    assert all(c is not None for c in cals)
    assert p["tracked_release_gap"] == max(c.stall_gap / 2 for c in cals if c is not None)
    assert p["tracked_release_dwell_s"] == RELEASE_DWELL_S
    assert f(d, "on") == p


def test_a_fixed_jaw_body_link_is_a_k2_only_extra_contact_link() -> None:
    """The SO-101's fixed jaw is its `gripper` body link (r6po_dbg3, 2026-10-17)."""
    from openral_core import JointSpec, RobotDescription
    from pydantic import ValidationError

    so101 = RobotDescription.from_yaml(str(_REPO / "robots/so101_follower/robot.yaml"))
    assert so101.fixed_jaw_links() == ("gripper",)
    assert "gripper" not in so101.grasp_contact_links(), "the legacy allowlist is unchanged"
    assert so101.fixed_jaw_links(gripper_joints=["shoulder_pan"]) == ()
    for robot in ("openarm", "droid_franka_robotiq"):
        d = RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml"))
        assert d.fixed_jaw_links() == ()
    with pytest.raises(ValidationError, match="fixed_jaw_link needs role"):
        JointSpec(name="j", joint_type="revolute", parent_link="a", child_link="b",
                  fixed_jaw_link="a")  # fmt: skip
    ok = JointSpec(name="j", joint_type="revolute", parent_link="a", child_link="b",
                   role="gripper", fixed_jaw_link="a")  # fmt: skip
    assert JointSpec.model_validate_json(ok.model_dump_json()) == ok


def test_a_placements_context_labels_ride_as_context_queries() -> None:
    """Spark R264/R265: the fingers met an untracked tote neighbour; nothing explained it."""
    scene = _tracking_pick_orange()
    raw = dict(cast("dict[str, Any]", scene.scene.backend_options["placement_success"]))
    tasks = {
        prompt: [{**p, "context": ["carton"]} for p in placements]
        for prompt, placements in raw["tasks"].items()
    }
    placed = cast("list[Any]", scene.scene.backend_options.get("objects", []))
    objects = [*placed, {"name": "carton_a"}, {"name": "carton_b"}]
    options = {
        **scene.scene.backend_options,
        "placement_success": {**raw, "tasks": tasks},
        "objects": objects,
    }
    scene = scene.model_copy(
        update={"scene": scene.scene.model_copy(update={"backend_options": options})}
    )
    queries = [(q["label"], q["role"], q["count"]) for q in _queries(scene)]
    assert queries[-1] == ("carton", "context", "all")
    assert _queries(scene)[-1]["min_instances"] == 2, "as many as the scene places"
    assert [q[:2] for q in queries[:-1]] == [("orange", "target"), ("plate", "destination")]


def test_a_context_query_is_seeded_beside_the_targets_seed() -> None:
    """Spark R282/R283: an unseeded "carton" query grounded shelf cartons, never the tote
    neighbour the finger met; seeded around the target, its prompts land beside it."""
    from openral_cli.deploy_sim import _CONTEXT_SEED_MARGIN_M

    scene = _tracking_pick_orange()
    raw = dict(cast("dict[str, Any]", scene.scene.backend_options["placement_success"]))
    tasks = {
        prompt: [{**p, "context": ["carton"]} for p in placements]
        for prompt, placements in raw["tasks"].items()
    }
    options = {**scene.scene.backend_options, "placement_success": {**raw, "tasks": tasks}}
    decl = scene.grasp_declaration
    assert decl is not None and decl.search_box is not None
    box = decl.search_box.model_copy(update={"evidence_ref": "reasoner_seed:detection 'orange'"})
    scene = scene.model_copy(
        update={
            "scene": scene.scene.model_copy(update={"backend_options": options}),
            "grasp_declaration": decl.model_copy(update={"search_box": box}),
        }
    )
    context = _queries(scene)[-1]
    assert context["role"] == "context" and context["seed"] is not None
    grown = [h + _CONTEXT_SEED_MARGIN_M for h in box.half_extents]
    assert context["seed"]["half_extents"] == pytest.approx(grown)
    assert context["seed"]["pose"]["xyz"] == pytest.approx(list(box.pose.xyz))
