"""The real-OpenArm world-voxel scene arms the kernel's voxel check at the real margin.

``scenes/deploy/openarm_real_world_voxels.yaml`` is the first scene that turns the kernel's
world-voxel check on against a real camera on a real arm. Driven the way ``openral deploy
run`` drives it: ``resolve_launch_invocation(hal_mode="real")`` produces the argv, and every
``key:=value`` in it is fed to the real ``compose_runtime_graph``. Nothing is launched.

``deploy_config`` is dropped from the real-mode composition only: the scene's ``drivers:``
include resolves ``zed_wrapper`` from the ament path, which exists on the rig overlay, not
here. The kernel/octomap parameters under test come from the argv, not from that include;
the ZED mount (the robot manifest's ``head_zed`` with the Thor unit overlay) is covered by
composing the scene on the sim path, where drivers are ignored. The scene names no
``robot_unit`` (it runs on either cell), so ``OPENRAL_ROBOT_UNIT=thor`` selects the unit.

``deploy run`` refuses this scene unless the selected unit's overlay declares ``head_zed``'s
mount (OpenRAL does not calibrate it); the real-mode tests resolve against a copy of
``robots/openarm`` served via ``OPENRAL_ROBOTS_DIR``, whose Thor overlay declares it.

Per CLAUDE.md §1.11: the committed scene, the real ``robots/openarm/robot.yaml``, the real
CLI resolver and launch composition. No mocks.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from _launch_test_common import import_launch_module

pytest.importorskip("launch")
pytest.importorskip("launch_ros")
pytest.importorskip("openral_cli")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO"),
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_FILE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
_SCENE = _REPO_ROOT / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"
_ZED_CLOUD = "/zed/zed_node/point_cloud/cloud_registered"


@pytest.fixture
def calibrated_openarm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``robots/openarm`` copied: its Thor unit overlay declares head_zed's calibrated mount."""
    robot_dir = tmp_path / "robots" / "openarm"
    shutil.copytree(_REPO_ROOT / "robots" / "openarm", robot_dir)
    monkeypatch.setenv("OPENRAL_ROBOTS_DIR", str(tmp_path / "robots"))
    return robot_dir


@pytest.fixture(autouse=True)
def _thor_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "thor")


def _launch_args(hal_mode: str, scene: Path = _SCENE) -> dict[str, str]:
    """The ``key:=value`` launch arguments ``openral deploy run|sim`` would pass."""
    from openral_cli.deploy_sim import resolve_launch_invocation

    invocation = resolve_launch_invocation(
        config=scene,
        robot_override="openarm",
        dashboard_port=4318,
        reset_to_pose_service=None,
        deploy_config=scene,
        hal_param_overrides={},
        hal_mode=hal_mode,
        enable_dashboard=False,
    )
    args = dict(tok.split(":=", 1) for tok in invocation.argv_template if ":=" in tok)
    args["hal_params_file"] = _write_hal_params(invocation.hal_params)
    return args


def _write_hal_params(hal_params: dict[str, object]) -> str:
    """The HAL params file exactly as ``deploy`` writes it: the launch reads it back for
    its HAL couplings (vision leg bounds, grasp-target producer)."""
    import tempfile

    import yaml

    path = Path(tempfile.gettempdir()) / f"openral-test-hal-params-{os.getpid()}.yaml"
    path.write_text(yaml.safe_dump({"/**": {"ros__parameters": hal_params}}), encoding="utf-8")
    return str(path)


def _compose(args: dict[str, str]) -> list[Any]:
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    module = import_launch_module(_LAUNCH_FILE)
    ctx = LaunchContext()
    ctx.launch_configurations.update(args)
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    ctx.launch_configurations.update(args)  # CLI values win over declared defaults
    ctx.launch_configurations["enable_dashboard"] = "false"
    return [ctx, list(module.compose_runtime_graph(ctx))]


def _node(entities: list[Any], package: str, executable: str | None = None) -> Any:
    from launch_ros.actions import Node

    found = [
        e
        for e in entities
        if isinstance(e, Node)
        and getattr(e, "_Node__package", None) == package
        and (executable is None or getattr(e, "_Node__node_executable", None) == executable)
    ]
    assert len(found) == 1, f"expected one {package} node, found {len(found)}"
    return found[0]


def test_deploy_run_refuses_a_unit_that_declares_no_zed_mount(calibrated_openarm: Path) -> None:
    """The gate is the unit overlay: strip head_zed's mount from Thor's and the real launch
    never resolves, whatever the manifest's nominal pose says."""
    import yaml
    from openral_core.exceptions import ROSConfigError

    unit_file = calibrated_openarm / "units" / "thor.yaml"
    doc = yaml.safe_load(unit_file.read_text(encoding="utf-8"))
    (zed,) = [s for s in doc["sensors"] if s["name"] == "head_zed"]
    del zed["static_transform_xyz_rpy"]
    unit_file.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ROSConfigError, match=r"head_zed: mount is the manifest's nominal"):
        _launch_args("real")


@pytest.mark.usefixtures("calibrated_openarm")
def test_deploy_run_resolves_the_zed_cloud_and_the_kernel_check() -> None:
    args = _launch_args("real")

    assert args["hal_mode"] == "real"
    assert args["enable_octomap"] == "true"
    assert args["enable_octomap_kernel_check"] == "true"
    assert args["octomap_cloud_topic"] == _ZED_CLOUD
    assert args["enable_reasoner"] == "false"


@pytest.mark.usefixtures("calibrated_openarm")
def test_the_kernel_gets_world_voxel_enabled_at_the_real_margin() -> None:
    from launch_ros.utilities import evaluate_parameters

    args = _launch_args("real")
    del args["deploy_config"]  # drivers: needs zed_wrapper on the ament path (rig only)
    ctx, entities = _compose(args)

    (kernel_params,) = evaluate_parameters(
        ctx, _node(entities, "openral_safety_kernel")._Node__parameters
    )
    assert kernel_params["world_voxel_enabled"] is True
    assert kernel_params["world_voxel_margin_m"] == 0.02
    assert kernel_params["world_voxel_deadline_ms"] == 1000.0
    # Real hardware has no attachment producer, so payload checking stays off.
    assert kernel_params.get("attached_collision_enabled", False) is False

    def remaps(node: Any) -> dict[str, str]:
        return {
            "".join(s.text for s in k): "".join(s.text for s in v)
            for k, v in node._Node__remappings
        }

    # Real camera: the robot self-filter sits between the ZED cloud and octomap.
    self_filter = _node(entities, "openral_octomap_bridge", "robot_self_filter")
    assert remaps(self_filter)["cloud_in"] == _ZED_CLOUD
    octomap = _node(entities, "octomap_server")
    assert remaps(octomap)["cloud_in"] == remaps(self_filter)["cloud_out"]
    (filter_params,) = evaluate_parameters(ctx, self_filter._Node__parameters)
    # The filter poses the robot from the stream the runtime nodes read: the real
    # ros2_control HAL's rate-limited republish (hal_joint_states_topic).
    assert filter_params["joint_states_topic"] == "/openral_hal_openarm/joint_states"
    (octo_params,) = evaluate_parameters(ctx, octomap._Node__parameters)
    assert octo_params["resolution"] == 0.02
    assert octo_params["frame_id"] == "openarm_base"


def test_the_unit_pose_is_the_only_mount_published_for_the_zed() -> None:
    """The selected unit's head_zed pose reaches /tf_static, once, with the manifest's parent.

    The ZED is bolted to the robot, so its mount is robot geometry: the scene declares no
    ``head_zed`` entry and may not (``check_scene_sensor_overrides``); its frames are the
    manifest's and its calibrated pose the unit overlay's (``robots/openarm/units/thor.yaml``).
    """
    import yaml
    from openral_core import load_robot_unit

    assert "sensors" not in yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    (zed,) = [
        s
        for s in load_robot_unit(_REPO_ROOT / "robots/openarm/robot.yaml", "thor").sensors
        if s.name == "head_zed"
    ]

    _, entities = _compose(_launch_args("sim"))

    mounts = []
    for e in entities:
        if getattr(e, "_Node__package", None) != "tf2_ros":
            continue
        argv = ["".join(getattr(s, "text", str(s)) for s in a) for a in e._Node__arguments]
        if argv[argv.index("--child-frame-id") + 1] == "zed_camera_link":
            mounts.append(argv)
    assert len(mounts) == 1, mounts
    (argv,) = mounts
    assert argv[argv.index("--frame-id") + 1] == "openarm_base"
    flags = ("--x", "--y", "--z", "--roll", "--pitch", "--yaw")
    assert [float(argv[argv.index(f) + 1]) for f in flags] == pytest.approx(
        list(zed.static_transform_xyz_rpy or ())
    )


def test_deploy_run_refuses_without_a_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both cells run this scene with different ZED mounts: a real run must name its cell."""
    from openral_core.exceptions import ROSConfigError

    monkeypatch.delenv("OPENRAL_ROBOT_UNIT")
    with pytest.raises(ROSConfigError, match="none is selected"):
        _launch_args("real")


@pytest.mark.parametrize("hal_mode", ["real", "sim"])
def test_deploy_refuses_a_scene_that_names_the_zed(tmp_path: Path, hal_mode: str) -> None:
    """``deploy run`` / ``deploy sim`` / ``deploy validate`` refuse before launching.

    All three go through ``resolve_launch_invocation``, which is where the rule is checked
    pre-launch: a deploy scene never touches a robot camera — a pose (or binding) hidden in
    one scene would silently not apply to the robot's others.
    """
    import yaml
    from openral_core.exceptions import ROSConfigError

    data = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    data["sensors"] = [
        {
            "name": "head_zed",
            "modality": "depth",
            "frame_id": "zed_camera_link",
            "parent_frame": "openarm_base",
            "rate_hz": 10.0,
            "static_transform_xyz_rpy": [0.013, -0.021, 0.231, 0.004, 0.771, -0.012],
        }
    ]
    scene = tmp_path / "hidden_pose.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ROSConfigError, match=r"'head_zed'.*defined by the robot manifest"):
        _launch_args(hal_mode, scene)


# ── Vision attachment leg (DeployRuntime.vision_attachment) ──────────────────

_SEGMENTER_MANIFEST = _REPO_ROOT / "rskills/rskill-sam2_1-any-grasped_object_mask-bf16/rskill.yaml"


def _scene_with_vision_leg(tmp_path: Path, *, enabled: bool | None) -> Path:
    """The committed scene with its vision leg enabled, as committed (off), or removed."""
    import yaml

    data = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    if enabled is None:
        del data["runtime"]["vision_attachment"]
    else:
        data["runtime"]["vision_attachment"]["enabled"] = enabled
    scene = tmp_path / f"vision_leg_{enabled}.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")
    return scene


def _real_graph(scene: Path) -> tuple[dict[str, object], dict[str, Any], Any, list[Any]]:
    """``(hal_params, kernel_params, ctx, entities)`` for a ``deploy run`` of ``scene``."""
    from launch_ros.utilities import evaluate_parameters
    from openral_cli.deploy_sim import resolve_launch_invocation

    invocation = resolve_launch_invocation(
        config=scene,
        robot_override="openarm",
        dashboard_port=4318,
        reset_to_pose_service=None,
        deploy_config=scene,
        hal_param_overrides={},
        hal_mode="real",
        enable_dashboard=False,
    )
    args = dict(tok.split(":=", 1) for tok in invocation.argv_template if ":=" in tok)
    args["hal_params_file"] = _write_hal_params(invocation.hal_params)
    del args["deploy_config"]  # drivers: needs zed_wrapper on the ament path (rig only)
    ctx, entities = _compose(args)
    (kernel_params,) = evaluate_parameters(
        ctx, _node(entities, "openral_safety_kernel")._Node__parameters
    )
    return invocation.hal_params, kernel_params, ctx, entities


def _segmenters(entities: list[Any]) -> list[Any]:
    return [
        e
        for e in entities
        if getattr(e, "_Node__package", None) == "openral_perception_ros"
        and getattr(e, "_Node__node_executable", None) == "segmenter_node.py"
    ]


def test_the_vision_leg_on_real_turns_the_kernel_attached_check_on(
    tmp_path: Path, calibrated_openarm: Path
) -> None:
    """Enabled on ``deploy run``: segmenter up, HAL bridge configured, kernel check on."""
    from launch_ros.actions import LifecycleNode
    from launch_ros.utilities import evaluate_parameters

    hal_params, kernel_params, ctx, entities = _real_graph(
        _scene_with_vision_leg(tmp_path, enabled=True)
    )

    # Coupling rule: the leg never runs without the kernel's attached check.
    assert kernel_params["attached_collision_enabled"] is True
    assert kernel_params["attached_collision_deadline_ms"] == 1000.0

    assert {k: v for k, v in hal_params.items() if k.startswith("vision_attachment_")} == {
        "vision_attachment_enabled": True,
        "vision_attachment_camera": "head_zed",
        "vision_attachment_depth_topic": "/zed/zed_node/depth/depth_registered",
        "vision_attachment_camera_info_topic": "/zed/zed_node/depth/camera_info",
        "vision_attachment_deadline_s": 0.25,
        "vision_attachment_evidence_timeout_s": 0.5,
        "vision_attachment_attach_effort": 0.0,
        "vision_attachment_release_effort": 0.0,
        "vision_attachment_grasp_target_enabled": False,
        "vision_attachment_place_fixture_enabled": False,
        "vision_attachment_release_timeout_s": 3.0,
        "vision_attachment_tf_frames": [
            "openarm_left_link7=openarm_left_ee_base_link",
            "openarm_right_link7=openarm_right_ee_base_link",
        ],
    }

    (segmenter,) = _segmenters(entities)
    assert isinstance(segmenter, LifecycleNode)
    (params,) = evaluate_parameters(ctx, segmenter._Node__parameters)
    assert params == {
        "robot_yaml": str(calibrated_openarm / "robot.yaml"),
        "manifest_path": str(_SEGMENTER_MANIFEST),
        "cameras": ("head_zed=/zed/zed_node/rgb/color/rect/image",),
        "camera_infos": ("head_zed=/zed/zed_node/rgb/color/rect/camera_info",),
        "primary_camera": "head_zed",
        "device": "auto",
        "use_sim_time": False,
    }


def _hal_launch_params(ctx: Any, entities: list[Any]) -> dict[str, Any]:
    """The HAL node's in-launch parameter dicts, merged (the params file is the CLI's)."""
    from launch_ros.utilities import evaluate_parameters

    hal = _node(entities, "openral_hal_node")
    merged: dict[str, Any] = {}
    for entry in evaluate_parameters(ctx, hal._Node__parameters):
        if isinstance(entry, dict):
            merged.update(entry)
    return merged


@pytest.mark.usefixtures("calibrated_openarm")
def test_the_vision_leg_bounds_derive_from_the_kernels_voxel_path(tmp_path: Path) -> None:
    """The release window and grid-age bounds come from the kernel's own voxel deadline,
    world margin and octree resolution, never a second copy; leg off, none are set."""
    _, kernel, ctx, entities = _real_graph(_scene_with_vision_leg(tmp_path, enabled=True))
    hal = _hal_launch_params(ctx, entities)
    deadline_s = kernel["world_voxel_deadline_ms"] / 1000.0
    assert hal["vision_attachment_grid_max_age_s"] == pytest.approx(deadline_s)
    # 20 mm real margin + one 20 mm cell: the release window's documented 40 mm.
    assert hal["vision_attachment_release_clear_m"] == pytest.approx(
        kernel["world_voxel_margin_m"] + 0.02
    )
    assert hal["vision_attachment_release_clear_m"] == pytest.approx(0.04)

    _, _, ctx_off, entities_off = _real_graph(_scene_with_vision_leg(tmp_path, enabled=False))
    assert not any(
        k.startswith("vision_attachment_") for k in _hal_launch_params(ctx_off, entities_off)
    )


@pytest.mark.usefixtures("calibrated_openarm")
def test_the_vision_leg_off_leaves_the_real_graph_as_without_it(tmp_path: Path) -> None:
    """Committed posture (``enabled: false``): no segmenter, no attached check, and the HAL
    and kernel parameters equal those of the scene without the block at all."""
    hal_off, kernel_off, _, entities = _real_graph(_scene_with_vision_leg(tmp_path, enabled=False))
    hal_absent, kernel_absent, _, _ = _real_graph(_scene_with_vision_leg(tmp_path, enabled=None))

    assert _segmenters(entities) == []
    assert kernel_off.get("attached_collision_enabled", False) is False
    assert not any(k.startswith("vision_attachment_") for k in hal_off)
    assert kernel_off == kernel_absent
    # Only the scene path differs (sim_env_yaml is never set on real).
    assert hal_off == hal_absent


def _voxel_bridge_params(ctx: Any, entities: list[Any]) -> dict[str, Any]:
    from launch_ros.utilities import evaluate_parameters

    bridge = _node(entities, "openral_octomap_bridge", "octomap_voxel_bridge")
    (params,) = evaluate_parameters(ctx, bridge._Node__parameters)
    return dict(params)


@pytest.mark.usefixtures("calibrated_openarm")
def test_the_voxel_bridge_finds_the_held_payload_under_the_hals_tf_frames(
    tmp_path: Path,
) -> None:
    """The published attach link is the manifest's (``openarm_left_link7``); the cell's TF
    tree names that body ``openarm_left_ee_base_link``. The bridge clears a held payload by
    looking its attach link up on TF, so it gets the scene's ``tf_frames`` as the very strings
    the HAL gets — else the payload stays an obstacle to its own gripper."""
    hal_params, _, ctx, entities = _real_graph(_scene_with_vision_leg(tmp_path, enabled=True))

    params = _voxel_bridge_params(ctx, entities)
    assert tuple(params["attach_link_tf_frames"]) == tuple(
        hal_params["vision_attachment_tf_frames"]  # type: ignore[arg-type]  # reason: dict[str, object]
    )
    assert tuple(params["attach_link_tf_frames"]) == (
        "openarm_left_link7=openarm_left_ee_base_link",
        "openarm_right_link7=openarm_right_ee_base_link",
    )


@pytest.mark.usefixtures("calibrated_openarm")
def test_the_voxel_bridge_gets_no_tf_frames_without_the_vision_leg(tmp_path: Path) -> None:
    """Leg off (committed) or block absent: the HAL gets no renames, so neither does the
    bridge — and its parameters are those of a scene that never had the block."""
    _, _, ctx_off, entities_off = _real_graph(_scene_with_vision_leg(tmp_path, enabled=False))
    _, _, ctx_absent, entities_absent = _real_graph(_scene_with_vision_leg(tmp_path, enabled=None))

    off = _voxel_bridge_params(ctx_off, entities_off)
    absent = _voxel_bridge_params(ctx_absent, entities_absent)
    assert "attach_link_tf_frames" not in absent
    assert off == absent


# ── Grasp-target exemption (DeployRuntime.grasp_allowance_enabled) ───────────

_OPENARM_FINGER_LINKS = ("openarm_left_finger_pair", "openarm_right_finger_pair")


def _scene_with_grasp_allowance(tmp_path: Path, *, enabled: bool | None) -> Path:
    """The committed scene with the flag on, explicitly off, or absent (the default)."""
    import yaml

    data = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    if enabled is None:
        data["runtime"].pop("grasp_allowance_enabled", None)
    else:
        data["runtime"]["grasp_allowance_enabled"] = enabled
    if enabled:
        # `deploy run` refuses the exemption without its producer, the vision target leg.
        data["runtime"]["vision_attachment"].update(enabled=True, grasp_target_enabled=True)
    scene = tmp_path / f"grasp_allowance_{enabled}.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")
    return scene


def _kernel_params(args: dict[str, str]) -> dict[str, Any]:
    from launch_ros.utilities import evaluate_parameters

    args = dict(args)
    args.pop("deploy_config", None)  # drivers: needs zed_wrapper on the ament path (rig only)
    ctx, entities = _compose(args)
    (params,) = evaluate_parameters(ctx, _node(entities, "openral_safety_kernel")._Node__parameters)
    return dict(params)


@pytest.mark.usefixtures("calibrated_openarm")
@pytest.mark.parametrize("hal_mode", ["real", "sim"])
def test_grasp_allowance_on_reaches_the_kernel_with_the_manifest_finger_links(
    tmp_path: Path, hal_mode: str
) -> None:
    args = _launch_args(hal_mode, _scene_with_grasp_allowance(tmp_path, enabled=True))
    assert args["grasp_allowance_enabled"] == "true"

    kernel = _kernel_params(args)
    assert kernel["grasp_allowance_enabled"] is True
    assert tuple(kernel["grasp_contact_links"]) == _OPENARM_FINGER_LINKS
    # The kernel refuses at configure a name its collision model lacks.
    assert set(_OPENARM_FINGER_LINKS) <= set(kernel["collision_link_names"])


@pytest.mark.usefixtures("calibrated_openarm")
@pytest.mark.parametrize("hal_mode", ["real", "sim"])
def test_grasp_allowance_off_still_passes_the_links_and_changes_nothing_else(
    tmp_path: Path, hal_mode: str
) -> None:
    """Off (the committed posture and the default): the flag is False, the allowlist is
    still passed, the argv carries no grasp key, and explicitly-off equals absent."""
    args_off = _launch_args(hal_mode, _scene_with_grasp_allowance(tmp_path, enabled=False))
    args_absent = _launch_args(hal_mode, _scene_with_grasp_allowance(tmp_path, enabled=None))

    assert "grasp_allowance_enabled" not in args_off
    del args_off["deploy_config"], args_absent["deploy_config"]  # the scene paths differ
    assert args_off == args_absent

    kernel_off = _kernel_params(args_off)
    assert kernel_off["grasp_allowance_enabled"] is False
    assert tuple(kernel_off["grasp_contact_links"]) == _OPENARM_FINGER_LINKS
    assert kernel_off == _kernel_params(args_absent)


@pytest.mark.usefixtures("calibrated_openarm")
@pytest.mark.parametrize(
    "hal_override",
    [
        {"vision_attachment_grasp_target_enabled": False},
        {"vision_attachment_enabled": False},
        {"vision_attachment_enabled": None, "vision_attachment_grasp_target_enabled": None},
    ],
)
def test_the_launch_refuses_grasp_allowance_when_the_hal_runs_no_target_leg(
    tmp_path: Path, hal_override: dict[str, object]
) -> None:
    """A bare ``ros2 launch ... grasp_allowance_enabled:=true`` on real is judged on the HAL's
    effective params file, not on what ``deploy run`` checked: a file whose HAL runs no
    grasp-target leg (turned off, or never set) is refused before anything starts."""
    import yaml
    from openral_core.exceptions import ROSConfigError

    args = _launch_args("real", _scene_with_grasp_allowance(tmp_path, enabled=True))
    args.pop("deploy_config")  # drivers: needs zed_wrapper on the ament path (rig only)
    hal_file = Path(args["hal_params_file"])
    params = yaml.safe_load(hal_file.read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    for key, value in hal_override.items():
        if value is None:
            params.pop(key, None)
        else:
            params[key] = value
    hal_file.write_text(yaml.safe_dump({"/**": {"ros__parameters": params}}), encoding="utf-8")

    with pytest.raises(ROSConfigError, match="grasp_target_enabled"):
        _compose(args)


@pytest.mark.usefixtures("calibrated_openarm")
def test_the_launch_refuses_grasp_allowance_on_real_without_the_vision_leg(
    tmp_path: Path,
) -> None:
    """The flag alone on a bare real launch (no ``enable_vision_attachment:=true``) is refused;
    the twin keeps it, its MuJoCo evidence tracker being the producer."""
    from openral_core.exceptions import ROSConfigError

    for hal_mode in ("real", "sim"):
        args = _launch_args(hal_mode, _scene_with_vision_leg(tmp_path, enabled=False))
        args.pop("deploy_config", None)
        args["grasp_allowance_enabled"] = "true"
        if hal_mode == "real":
            with pytest.raises(ROSConfigError, match="grasp_target_enabled"):
                _compose(args)
        else:
            assert _kernel_params(args)["grasp_allowance_enabled"] is True


@pytest.mark.usefixtures("calibrated_openarm")
@pytest.mark.parametrize(
    ("override", "verdict"),
    [
        ({"vision_attachment_grid_max_age_s": 1.5}, "world_voxel_deadline_s"),
        ({"vision_attachment_release_clear_m": 0.03}, "world_voxel_margin_m"),
        ({"vision_attachment_grid_max_age_s": 0.5, "vision_attachment_release_clear_m": 0.06}, ""),
    ],
)
def test_the_launch_refuses_vision_bounds_past_the_kernels_voxel_path(
    tmp_path: Path, override: dict[str, float], verdict: str
) -> None:
    """A params-file value wins over the launch's derived bound, so it is checked whatever its
    source: a grid older than the kernel's voxel deadline, or a release clearance under the
    kernel's margin plus one cell, fails the launch. Stricter values compose."""
    import yaml
    from openral_core.exceptions import ROSConfigError

    args = _launch_args("real", _scene_with_vision_leg(tmp_path, enabled=True))
    args.pop("deploy_config")  # drivers: needs zed_wrapper on the ament path (rig only)
    hal_file = Path(args["hal_params_file"])
    params = yaml.safe_load(hal_file.read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    params.update(override)
    hal_file.write_text(yaml.safe_dump({"/**": {"ros__parameters": params}}), encoding="utf-8")

    if verdict:
        with pytest.raises(ROSConfigError, match=verdict):
            _compose(args)
    else:
        _compose(args)
