"""Manipulation-workspace contact (issue #349): schema, manifest, launch, CLI and producer.

``docs/reference/manipulation-workspace-contact.md``. A scene may *declare* a
``GraspDeclaration.workspace_box``; only the HAL's grasp-target leg may *confirm* it
(``workspace_region``) against a fresh voxel map; the safety kernel lets the manifest's
hand links touch inside it while the hand is slow (kernel side:
``cpp/openral_safety_kernel/test``). Real OpenArm manifest, real scene fixture, real
launch file, real IDL and a real rclpy node for the producer (CLAUDE.md §1.11).
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from openral_core import (
    DeployRuntime,
    DeployScene,
    GraspDeclaration,
    PlaceRegion,
    Pose6D,
    RobotDescription,
)
from openral_core.exceptions import ROSConfigError
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[2]
_ROBOT = _REPO / "robots" / "openarm" / "robot.yaml"
_SCENE = _REPO / "tests" / "unit" / "fixtures" / "scenes" / "openarm_direct_dispatch_grasp.yaml"
_CELL = _REPO / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"
_LAUNCH = _REPO / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"
_S = 1_000_000_000


def _tote(*, stamp_ns: int = 0, frame: str = "openarm_base") -> PlaceRegion:
    """The source tote's interior: 0.2 x 0.2 x 0.12 m over its floor at z = 0."""
    return PlaceRegion(
        frame_id=frame,
        pose=Pose6D(xyz=(0.40, 0.0, 0.06), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=frame),
        half_extents=(0.10, 0.10, 0.06),
        evidence_ref="scene:source_tote",
        stamp_ns=stamp_ns,
    )


def _declaration(**update: Any) -> GraspDeclaration:
    scene = DeployScene.from_yaml(str(_SCENE)).grasp_declaration
    assert scene is not None
    fields = {**dict(scene), "stamp_ns": 10 * _S, "workspace_box": _tote(), **update}
    return GraspDeclaration.model_validate(fields)  # validated, unlike model_copy


# ── schema ────────────────────────────────────────────────────────────────────


def test_the_workspace_fields_round_trip_through_json_and_the_real_idl() -> None:
    msg_mod = pytest.importorskip("openral_msgs.msg", reason="needs the built openral_msgs IDL")
    invocation_id = "0123456789abcdef0123456789abcdef"
    declaration = _declaration(
        workspace_region=_tote(stamp_ns=11 * _S), invocation_id=invocation_id
    )
    assert GraspDeclaration.model_validate_json(declaration.model_dump_json()) == declaration
    msg = msg_mod.GraspDeclaration()
    declaration.fill_idl(msg)
    assert msg.workspace_box_valid and msg.workspace_region_valid and not msg.region_valid
    assert msg.invocation_id == invocation_id
    assert GraspDeclaration.from_idl(msg) == declaration
    bare = msg_mod.GraspDeclaration()
    _declaration(workspace_box=None).fill_idl(bare)
    assert not bare.workspace_box_valid and not bare.workspace_region_valid
    assert GraspDeclaration.from_idl(bare).workspace_box is None
    with pytest.raises(ValidationError, match="invocation_id"):
        _declaration(invocation_id="not-a-uuid")


@pytest.mark.parametrize("field", ["workspace_box", "workspace_region"])
def test_the_workspace_caps_bind(field: str) -> None:
    wide = _tote().model_copy(update={"half_extents": (0.51, 0.10, 0.10)})
    with pytest.raises(ValidationError, match=r"0\.5 m bound"):
        _declaration(**{field: wide})
    big = _tote().model_copy(update={"half_extents": (0.45, 0.45, 0.20)})
    with pytest.raises(ValidationError, match=r"m\^3 bound"):
        _declaration(**{field: big})
    # A real tote interior (0.6 x 0.4 x 0.3 m) fits both caps.
    tote = _tote().model_copy(update={"half_extents": (0.30, 0.20, 0.15)})
    assert getattr(_declaration(**{field: tote}), field) == tote


def _scene_dict() -> dict[str, Any]:
    loaded = yaml.safe_load(_SCENE.read_text())
    assert isinstance(loaded, dict)
    return copy.deepcopy(loaded)


def test_a_scene_may_declare_a_workspace_box_but_never_supply_the_confirmed_one() -> None:
    raw = _scene_dict()
    raw["grasp_declaration"]["workspace_box"] = _tote().model_dump(mode="json")
    scene = DeployScene.model_validate(raw)
    assert scene.grasp_declaration is not None
    assert scene.grasp_declaration.workspace_box == _tote()
    raw["grasp_declaration"]["workspace_region"] = _tote().model_dump(mode="json")
    with pytest.raises(ValidationError, match="workspace_region is producer-supplied"):
        DeployScene.model_validate(raw)


def test_deploy_runtime_workspace_contact_needs_the_grasp_allowance_a_speed_and_a_lead() -> None:
    assert DeployRuntime().workspace_contact_enabled is False
    with pytest.raises(ValidationError, match="requires grasp_allowance_enabled"):
        DeployRuntime(
            workspace_contact_enabled=True,
            workspace_contact_max_speed_mps=0.05,
            workspace_contact_max_lead_m=0.05,
        )
    with pytest.raises(ValidationError, match="requires workspace_contact_max_speed_mps"):
        DeployRuntime(
            workspace_contact_enabled=True,
            grasp_allowance_enabled=True,
            workspace_contact_max_lead_m=0.05,
        )
    with pytest.raises(ValidationError, match="requires workspace_contact_max_lead_m"):
        DeployRuntime(
            workspace_contact_enabled=True,
            grasp_allowance_enabled=True,
            workspace_contact_max_speed_mps=0.05,
        )
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValidationError):
            DeployRuntime(workspace_contact_max_speed_mps=bad)
    for bad in (0.0, -0.01, 0.1001, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            DeployRuntime(workspace_contact_max_lead_m=bad)
    on = DeployRuntime(
        workspace_contact_enabled=True,
        grasp_allowance_enabled=True,
        workspace_contact_max_speed_mps=0.05,
        workspace_contact_max_lead_m=0.10,
    )
    assert on.workspace_contact_max_speed_mps == 0.05
    assert on.workspace_contact_max_lead_m == 0.10


# ── manifest ──────────────────────────────────────────────────────────────────


def test_the_openarm_hand_links_are_both_jaws_the_carrier_and_the_wrist() -> None:
    robot = RobotDescription.from_yaml(str(_ROBOT))
    assert robot.hand_links() == (
        "openarm_left_finger_pair",
        "openarm_right_finger_pair",
        "openarm_left_finger_follower",
        "openarm_right_finger_follower",
        "openarm_left_link7",
        "openarm_right_link7",
        "openarm_left_link6",
        "openarm_right_link6",
    )
    # Every one is a collision link the kernel resolves, never the forearm (link5) or
    # anything behind it.
    pytest.importorskip("openral_safety", reason="the collision lowering lives in openral_safety")
    from openral_safety.envelope_loader import collision_params_from_description

    names = collision_params_from_description(robot)["collision_link_names"]
    assert isinstance(names, list)
    assert set(robot.hand_links()) <= set(names)


def test_a_second_wrist_link_is_the_next_actuated_link_up_and_capped_at_two() -> None:
    """2026-10-21 decision (Isaac R50-R82: DROID's wrist housing ``panda_link6`` commanded
    into the table or bowl rim with no physical contact): ``wrist_links=2`` adds exactly the
    next link up the chain, a collision link the kernel resolves; the DROID scene asks for
    it; nothing past two."""
    droid = RobotDescription.from_yaml(str(_REPO / "robots/droid_franka_robotiq/robot.yaml"))
    assert set(droid.hand_links(wrist_links=2)) - set(droid.hand_links()) == {"panda_link6"}
    openarm = RobotDescription.from_yaml(str(_ROBOT))
    assert set(openarm.hand_links(wrist_links=2)) - set(openarm.hand_links()) == {
        "openarm_left_link5",
        "openarm_right_link5",
    }
    with pytest.raises(ValueError, match="wrist_links must be 1 or 2"):
        droid.hand_links(wrist_links=3)
    scene = DeployScene.from_yaml(
        str(_REPO / "scenes/deploy/isaac_arena_droid_banana_in_bowl.yaml")
    )
    assert scene.runtime is not None
    assert scene.runtime.workspace_contact_wrist_links == 2
    with pytest.raises(ValidationError):
        DeployRuntime.model_validate(
            {**scene.runtime.model_dump(), "workspace_contact_wrist_links": 3}
        )
    pytest.importorskip("openral_safety", reason="the collision lowering lives in openral_safety")
    from openral_safety.envelope_loader import collision_params_from_description

    names = collision_params_from_description(droid)["collision_link_names"]
    assert isinstance(names, list)
    assert set(droid.hand_links(wrist_links=2)) <= set(names)


# ── launch ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def launch_module() -> Any:
    """The real launch file, loaded by path — it is not an importable package."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_workspace", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_workspace"] = module
    spec.loader.exec_module(module)
    return module


def test_the_launch_derives_the_kernel_params_from_the_manifest(launch_module: Any) -> None:
    robot = RobotDescription.from_yaml(str(_ROBOT))
    derive = launch_module._workspace_contact_kernel_params
    assert derive(
        robot,
        enabled=False,
        max_speed_mps=None,
        max_lead_m=None,
        grasp_allowance_enabled=False,
        hal_mode="real",
    ) == {"workspace_contact_enabled": False}
    params = derive(
        robot,
        enabled=True,
        max_speed_mps=0.05,
        max_lead_m=0.04,
        grasp_allowance_enabled=True,
        hal_mode="sim",
    )
    assert robot.control_rate_hz is not None
    assert params == {
        "workspace_contact_enabled": True,
        "workspace_contact_links": list(robot.hand_links()),
        "workspace_contact_max_speed_mps": 0.05,
        "workspace_contact_max_lead_m": 0.04,
        "workspace_contact_jaw_joints": ["left_gripper", "right_gripper"],
        "workspace_contact_brake_enabled": True,
        "workspace_contact_hold_max_consecutive": round(2.0 * robot.control_rate_hz),
        "workspace_contact_control_period_s": pytest.approx(1.0 / robot.control_rate_hz),
    }
    two = derive(
        robot,
        enabled=True,
        max_speed_mps=0.05,
        max_lead_m=0.04,
        grasp_allowance_enabled=True,
        hal_mode="sim",
        wrist_links=2,
    )
    assert two["workspace_contact_links"] == list(robot.hand_links(wrist_links=2))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"hal_mode": "real"}, "sim-only until Safety-WG review"),
        ({"grasp_allowance_enabled": False}, "requires grasp_allowance_enabled"),
        ({"max_speed_mps": None}, "max_speed_mps > 0"),
        ({"max_speed_mps": 0.0}, "max_speed_mps > 0"),
        ({"max_lead_m": None}, "max_lead_m <= 0.10"),
        ({"max_lead_m": 0.0}, "max_lead_m <= 0.10"),
        ({"max_lead_m": 0.11}, "max_lead_m <= 0.10"),
    ],
)
def test_the_launch_refuses_an_unsound_workspace_contact(
    launch_module: Any, kwargs: dict[str, Any], match: str
) -> None:
    robot = RobotDescription.from_yaml(str(_ROBOT))
    args: dict[str, Any] = {
        "enabled": True,
        "max_speed_mps": 0.05,
        "max_lead_m": 0.05,
        "grasp_allowance_enabled": True,
        "hal_mode": "sim",
        **kwargs,
    }
    with pytest.raises(ROSConfigError, match=match):
        launch_module._workspace_contact_kernel_params(robot, **args)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cell_scene(tmp_path: Path, runtime: dict[str, object]) -> Path:
    data = yaml.safe_load(_CELL.read_text(encoding="utf-8"))
    data["runtime"].update(runtime)
    scene = tmp_path / "cell.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")
    return scene


def _invoke(scene: Path, hal_mode: str) -> Any:
    from openral_cli.deploy_sim import resolve_launch_invocation

    return resolve_launch_invocation(
        config=scene,
        robot_override="openarm",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        enable_dashboard=False,
        hal_mode=hal_mode,
    )


def test_deploy_sim_forwards_workspace_contact_and_deploy_run_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "thor")
    off = _invoke(_cell_scene(tmp_path, {}), "sim").argv_template
    assert not any(a.startswith("workspace_contact") for a in off), "off: argv unchanged"
    scene = _cell_scene(
        tmp_path,
        {
            "grasp_allowance_enabled": True,
            "workspace_contact_enabled": True,
            "workspace_contact_max_speed_mps": 0.05,
            "workspace_contact_max_lead_m": 0.04,
        },
    )
    argv = _invoke(scene, "sim").argv_template
    assert "workspace_contact_enabled:=true" in argv
    assert "workspace_contact_max_speed_mps:=0.05" in argv
    assert "workspace_contact_max_lead_m:=0.04" in argv
    assert "workspace_contact_wrist_links:=1" in argv  # the default: the one wrist link
    with pytest.raises(ROSConfigError, match="sim-only until Safety-WG review"):
        _invoke(scene, "real")


def test_a_scene_may_keep_the_workspace_rule_alone_without_k7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restock (2026-10-23 decision): the tote neighbour the finger brushes is hidden from
    the one depth camera, so K7 cannot explain it; the scene keeps the workspace rule alone."""
    monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "thor")
    rule: dict[str, object] = {
        "grasp_allowance_enabled": True,
        "workspace_contact_enabled": True,
        "workspace_contact_max_speed_mps": 0.05,
        "workspace_contact_max_lead_m": 0.04,
    }
    on = _invoke(_cell_scene(tmp_path, rule), "sim").argv_template
    assert "workspace_explained_cells:=true" in on, "the default: K7 as before"
    off = _invoke(
        _cell_scene(tmp_path, {**rule, "workspace_explained_cells": False}), "sim"
    ).argv_template
    assert "workspace_explained_cells:=false" in off


# ── producer (HAL grasp-target leg) ───────────────────────────────────────────


def _tote_lattice(*, floor: bool = True, frame: str = "openarm_base") -> Any:
    """20 mm cells over the tote: its floor layer (k = 0) occupied, the rest free."""
    from openral_core.perception_geometry import VoxelLattice

    occ = np.zeros(10 * 10 * 6, dtype=np.uint8)
    if floor:
        occ[: 10 * 10] = 1
    return VoxelLattice(frame, (0.30, -0.10, 0.0), (0.0, 0.0, 0.0, 1.0), 0.02, (10, 10, 6), occ)


@contextmanager
def _leg(name: str) -> Iterator[Any]:
    """The real grasp-target leg of a real bridge on a real rclpy node."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg", reason="needs the built openral_msgs IDL")
    from openral_hal.vision_attachment_bridge import VisionAttachmentBridge, VisionAttachmentConfig

    rclpy.init()
    try:
        node = rclpy.create_node(name)
        try:
            robot = RobotDescription.from_yaml(str(_ROBOT))
            config = VisionAttachmentConfig(camera="head_zed", grasp_target_enabled=True)
            bridge = VisionAttachmentBridge(node, robot, config=config)
            assert bridge._grasp_target is not None
            yield bridge._grasp_target
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def _published(leg: Any, now_ns: int) -> GraspDeclaration:
    from openral_msgs.msg import AttachmentState

    msg = AttachmentState()
    leg.fill(msg, now_ns=now_ns)
    assert msg.grasp_declaration_valid, "the leg published no grasp declaration"
    return GraspDeclaration.from_idl(msg.grasp_declaration)


def test_the_leg_confirms_the_declared_workspace_on_a_fresh_grid_and_publishes_it() -> None:
    with _leg("test_workspace_confirm") as leg:
        now_ns = leg._now_ns()
        source_ns = now_ns - _S // 10
        leg.tracker.on_declaration(
            _declaration(stamp_ns=now_ns, workspace_region=_tote(stamp_ns=1))
        )
        assert leg.tracker.declaration.workspace_region is None, "dispatch's copy is stripped"
        assert _published(leg, now_ns).workspace_region is None, "nothing before a grid"
        leg._bridge._grid = (_tote_lattice(), source_ns, time.monotonic())
        leg._confirm_workspace(now_ns)
        published = _published(leg, now_ns)
        assert published is not None and published.region is None, "no target region needed"
        # The declared box itself, never grown, stamped with the grid's source_stamp.
        assert published.workspace_region == _tote(stamp_ns=source_ns)


def test_the_leg_retracts_the_workspace_when_the_map_no_longer_holds_it() -> None:
    with _leg("test_workspace_map_disagrees") as leg:
        now_ns = leg._now_ns()
        leg.tracker.on_declaration(_declaration(stamp_ns=now_ns))
        leg._bridge._grid = (_tote_lattice(), now_ns, time.monotonic())
        leg._confirm_workspace(now_ns)
        assert _published(leg, now_ns).workspace_region is not None
        leg._bridge._grid = (_tote_lattice(floor=False), now_ns, time.monotonic())
        leg._confirm_workspace(now_ns)
        assert leg._workspace_state == "map_disagrees"
        assert _published(leg, now_ns).workspace_region is None
        # A grid in another frame is refused the same way.
        leg._bridge._grid = (_tote_lattice(frame="world"), now_ns, time.monotonic())
        leg._confirm_workspace(now_ns)
        assert leg._workspace_state == "frame_mismatch"
        assert _published(leg, now_ns).workspace_region is None


def test_a_stale_grid_keeps_the_last_confirmation_only_until_the_freeze() -> None:
    with _leg("test_workspace_stale") as leg:
        now_ns = leg._now_ns()
        freeze_ns = int(leg._freeze_s * 1e9)
        leg.tracker.on_declaration(_declaration(stamp_ns=now_ns))
        leg._bridge._grid = (_tote_lattice(), now_ns, time.monotonic())
        leg._confirm_workspace(now_ns)
        later = now_ns + freeze_ns // 2
        leg._bridge._grid = None  # the map stops arriving
        leg._confirm_workspace(later)
        assert _published(leg, later).workspace_region is not None, "within the freeze"
        past = now_ns + freeze_ns + _S // 10
        assert _published(leg, past).workspace_region is None, "aged out at publication"
        leg._confirm_workspace(past)
        assert leg._workspace_state.startswith("aged:") and leg._workspace is None


def test_no_declared_workspace_publishes_none() -> None:
    with _leg("test_workspace_none") as leg:
        now_ns = leg._now_ns()
        leg.tracker.on_declaration(_declaration(stamp_ns=now_ns, workspace_box=None))
        leg._bridge._grid = (_tote_lattice(), now_ns, time.monotonic())
        leg._confirm_workspace(now_ns)
        assert leg._workspace is None
        assert _published(leg, now_ns).workspace_region is None
