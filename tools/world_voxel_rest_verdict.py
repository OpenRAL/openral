"""Judge a recorded world-voxel map with the real safety kernel at a robot's rest pose, offline.

The safety kernel's world-voxel check only runs on a chunk it is asked to judge, so a
cell left idle ("bring up, dispatch nothing, expect no stop") never exercises it.
This tool asks the question directly, without moving anything:

1. compose the scene's real-hardware deploy graph exactly as ``openral deploy run``
   does (``resolve_launch_invocation`` + ``compose_runtime_graph``, nothing launched;
   the scene's vendor ``drivers:`` are dropped, they do not reach the kernel) and take
   the safety kernel's parameters from it — the same collision model, world-voxel
   margin, deadlines and joint names the cell runs;
2. read the last ``/openral/world_voxels`` grid (and ``/joint_states``) from a bag
   recorded on the cell — e.g. the runbook's twin pass, motors unpowered;
3. start the real ``safety_kernel_node`` on an isolated DDS domain, feed it that grid
   (restamped as fresh) and joint states at the pose under test, and send ONE
   ``JOINT_POSITION`` row equal to that pose: a hold, which commands no motion;
4. report the verdict: accepted, refused (``KIND_COLLISION``: the link, the voxel
   cell's centre in the grid frame, the distance), or dropped (the grid was unusable).

Written for issue #356: the OpenArm torso's foot-plate slab reaches 1 mm below the
surface it stands on, and the kernel trips any 2 cm cell within 2 cm of it, while the
self-filter only removes returns within 2 cm. Whether the cell's map holds that
surface is a measurement; this is the measurement, run off the robot.

Usage (repo root, ROS 2 Jazzy + this workspace's ``install/`` sourced)::

    python tools/world_voxel_rest_verdict.py --bag ~/twin_pass_2026-10-11-1030 \\
        --scene scenes/deploy/openarm_real_world_voxels.yaml --unit thor

Exit status: 0 accepted, 1 refused, 3 dropped (grid unusable), 2 usage / setup error.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LAUNCH_FILE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
_DEFAULT_SCENE = _REPO_ROOT / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"
_VOXELS_TOPIC = "/openral/world_voxels"
_JOINT_STATES_TOPIC = "/joint_states"
_TRACE = "world_voxel_rest_verdict"


@dataclass(frozen=True)
class RestVerdict:
    """What the kernel said about one hold row against one recorded grid.

    Attributes:
        outcome: ``"accepted"``, ``"refused"`` or ``"dropped"``.
        pose_rad: The row judged, in ``collision_joint_names`` order.
        joint_names: The kernel's ``collision_joint_names``.
        occupied_cells: Occupied cells in the replayed grid.
        grid_frame: The grid's ``header.frame_id``.
        evidence: The ``FailureTrigger.evidence_json`` of a refusal, parsed.
        cell_centre_m: A refusal's ``voxel_<n>`` cell centre in the grid frame, or
            ``None`` when the refusal names no cell (a self-collision).
        drop_reason: ``SafetyStatus.drop_reason`` of a drop, else ``None``.
    """

    outcome: str
    pose_rad: list[float]
    joint_names: list[str]
    occupied_cells: int
    grid_frame: str
    evidence: dict[str, Any] = field(default_factory=dict)
    cell_centre_m: list[float] | None = None
    drop_reason: int | None = None


def _import_launch_module() -> Any:
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch", _LAUNCH_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {_LAUNCH_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


def _walk(entities: list[Any]) -> list[Any]:
    """Every launch entity, nested ones (timer, group) included."""
    out: list[Any] = []
    for entity in entities:
        out.append(entity)
        subs: list[Any] = []
        for attr in ("actions", "_TimerAction__actions", "_GroupAction__actions"):
            subs += list(getattr(entity, attr, None) or [])
        if hasattr(entity, "get_sub_entities"):
            with contextlib.suppress(Exception):  # reason: not every entity can list its own
                subs += list(entity.get_sub_entities())
        out += _walk(subs)
    return out


def deploy_node_params(scene: Path, unit: str | None) -> dict[str, dict[str, Any]]:
    """Parameters ``openral deploy run`` gives the safety kernel and the self-filter.

    Composes the scene's ``hal_mode=real`` graph without launching it. The scene's
    ``drivers:`` block is dropped first (a vendor driver package need not be on this
    host, and no driver parameter reaches the kernel); everything else is the scene.

    Returns:
        ``{"kernel": {...}, "self_filter": {...}}``; ``"self_filter"`` is empty when the
        graph has none.

    Raises:
        RuntimeError: The composed graph has no safety kernel.
    """
    import yaml
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters
    from openral_cli.deploy_sim import resolve_launch_invocation

    if unit:
        os.environ["OPENRAL_ROBOT_UNIT"] = unit
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    data.pop("drivers", None)
    # Next to the original, so any scene-relative reference still resolves.
    with tempfile.NamedTemporaryFile(
        "w", dir=scene.parent, prefix=f".{scene.stem}.", suffix=".yaml", delete=False
    ) as fp:
        yaml.safe_dump(data, fp, sort_keys=False)
        copy = Path(fp.name)
    try:
        invocation = resolve_launch_invocation(
            config=copy,
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            deploy_config=copy,
            hal_param_overrides={},
            hal_mode="real",
            enable_dashboard=False,
        )
        args = dict(tok.split(":=", 1) for tok in invocation.argv_template if ":=" in tok)
        args.setdefault("hal_params_file", os.devnull)
        module = _import_launch_module()
        ctx = LaunchContext()
        ctx.launch_configurations.update(args)
        for entity in module.generate_launch_description().entities:
            if isinstance(entity, DeclareLaunchArgument):
                entity.execute(ctx)
        ctx.launch_configurations.update(args)
        ctx.launch_configurations["enable_dashboard"] = "false"
        entities = _walk(list(module.compose_runtime_graph(ctx)))
    finally:
        copy.unlink(missing_ok=True)

    found: dict[str, dict[str, Any]] = {"kernel": {}, "self_filter": {}}
    for entity in entities:
        if not isinstance(entity, Node):
            continue
        executable = getattr(entity, "_Node__node_executable", None)
        key = {"safety_kernel_node": "kernel", "robot_self_filter": "self_filter"}.get(
            str(executable)
        )
        if key and not found[key]:
            (params,) = evaluate_parameters(ctx, entity._Node__parameters)
            found[key] = dict(params)
    if not found["kernel"]:
        raise RuntimeError(f"{scene}: the composed real graph has no safety_kernel_node")
    return found


def read_last_messages(bag: Path, topics: list[str]) -> dict[str, Any]:
    """The last message of each of ``topics`` in a rosbag2 bag (any storage), deserialised."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    # reason: rosbag2_py's pybind11 module ships no type information.
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}  # type: ignore[no-untyped-call]
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    last: dict[str, bytes] = {}
    while reader.has_next():
        topic, raw, _stamp = reader.read_next()
        last[topic] = raw
    return {t: deserialize_message(raw, get_message(types[t])) for t, raw in last.items()}


def rest_row(
    joint_names: list[str],
    aliases: list[str],
    joint_state: Any | None,
    pose: str,
) -> list[float]:
    """The row to judge: zeros, or the bag's last joint state mapped onto ``joint_names``.

    A vendor ``/joint_states`` names joints by the manifest's ``sim_joint_name``; the
    self-filter's ``collision_joint_aliases`` (parallel to ``joint_names``) carries that
    second spelling. A joint the bag does not name is an error, never a silent zero.
    """
    if pose == "zero":
        return [0.0] * len(joint_names)
    if joint_state is None:
        raise ValueError(f"--pose measured needs {_JOINT_STATES_TOPIC} in the bag")
    measured = dict(zip(joint_state.name, joint_state.position, strict=False))
    row: list[float] = []
    missing: list[str] = []
    for i, name in enumerate(joint_names):
        alias = aliases[i] if i < len(aliases) else ""
        value = measured.get(name, measured.get(alias) if alias else None)
        if value is None:
            missing.append(name)
        else:
            row.append(float(value))
    if missing:
        raise ValueError(f"the bag's last joint state does not name {missing}")
    return row


def _is_drop(status: Any) -> bool:
    """A ``SafetyStatus`` dropping our row unjudged (``DROP_*``: 100 up, 255 is ``DROP_NONE``)."""
    return bool(status.trace_id == _TRACE and 100 <= status.drop_reason < 255)


def _cell_centre(grid: Any, index: int) -> list[float]:
    o = grid.orientation
    if abs(abs(o.w) - 1.0) > 1e-6:
        raise ValueError("non-identity grid lattice: rotate the cell offset by its orientation")
    x = index % grid.size_x
    y = (index // grid.size_x) % grid.size_y
    z = index // (grid.size_x * grid.size_y)
    origin = (grid.origin.x, grid.origin.y, grid.origin.z)
    return [origin[a] + (k + 0.5) * grid.resolution for a, k in enumerate((x, y, z))]


def judge_hold_row(
    kernel_params: dict[str, Any],
    grid: Any,
    row: list[float],
    *,
    log_path: Path,
    timeout_s: float = 15.0,
) -> RestVerdict:
    """Start the real kernel, feed it ``grid`` fresh, and judge one hold row at ``row``."""
    sys.path.insert(0, str(_REPO_ROOT))
    import rclpy
    from openral_msgs.msg import ActionChunk, FailureTrigger, OccupancyVoxels, SafetyStatus
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState

    # The kernel-subprocess helpers the live tests use (start on an isolated domain,
    # lifecycle activation, SIGINT teardown); loaded at run time, as the launch file is.
    kernel_proc = importlib.import_module("tests.sim.safety._kernel_subprocess")
    start_kernel = kernel_proc.start_kernel
    activate_kernel_node = kernel_proc.activate_kernel_node
    isolated_domain_id = kernel_proc.isolated_domain_id
    terminate_kernel = kernel_proc.terminate_kernel

    joint_names = list(kernel_params["collision_joint_names"])
    if len(row) != len(joint_names):
        raise ValueError(f"row has {len(row)} values for {len(joint_names)} joints")
    params = {**kernel_params, "collision_seed_dt_s": 0.0}
    node_name = f"rest_verdict_kernel_{uuid.uuid4().hex[:8]}"
    proc = start_kernel(
        params,
        node_name,
        isolated_domain_id(),
        log_path=log_path,
        params_file=log_path.with_suffix(".params.yaml"),
    )
    try:
        time.sleep(1.5)
        rclpy.init()
        try:
            helper = rclpy.create_node("rest_verdict_helper")
            if not activate_kernel_node(node_name, helper):
                raise RuntimeError(f"kernel activation failed; log: {log_path}")
            executor = SingleThreadedExecutor()
            executor.add_node(helper)
            safe: list[Any] = []
            failures: list[Any] = []
            statuses: list[Any] = []
            chunk_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
            voxel_pub = helper.create_publisher(OccupancyVoxels, _VOXELS_TOPIC, 10)
            joint_pub = helper.create_publisher(JointState, _JOINT_STATES_TOPIC, 10)
            helper.create_subscription(
                ActionChunk,
                "/openral/safe_action",
                lambda m: safe.append(m) if m.trace_id == _TRACE else None,
                10,
            )
            helper.create_subscription(
                FailureTrigger, "/openral/failure/safety", failures.append, 10
            )
            helper.create_subscription(SafetyStatus, "/openral/safety_status", statuses.append, 10)

            def feed() -> None:
                now = helper.get_clock().now().to_msg()
                grid.header.stamp = now
                grid.source_stamp = now  # the recording is the world under test, judged now
                voxel_pub.publish(grid)
                js = JointState()
                js.header.stamp = now
                js.name = joint_names
                js.position = list(row)
                joint_pub.publish(js)

            helper.create_timer(0.1, feed)

            def spin(seconds: float) -> None:
                end = time.time() + seconds
                while time.time() < end:
                    executor.spin_once(timeout_sec=0.02)

            spin(1.5)  # discovery, then a fresh grid and joint state in the kernel
            chunk = ActionChunk()
            chunk.control_mode = 0  # JOINT_POSITION
            chunk.horizon = 1
            chunk.n_dof = len(joint_names)
            chunk.flat = list(row)
            chunk.joint_names = joint_names
            chunk.rskill_id = "openral/world-voxel-rest-verdict"
            chunk.trace_id = _TRACE
            chunk_pub.publish(chunk)
            end = time.time() + timeout_s
            while time.time() < end and not safe and not failures:
                executor.spin_once(timeout_sec=0.02)
                if any(_is_drop(s) for s in statuses):
                    break
            spin(0.3)
        finally:
            rclpy.shutdown()
    finally:
        terminate_kernel(proc)

    occupied = sum(1 for c in grid.occupancy if c)
    common = {
        "pose_rad": list(row),
        "joint_names": joint_names,
        "occupied_cells": occupied,
        "grid_frame": grid.header.frame_id,
    }
    if safe:
        return RestVerdict(outcome="accepted", **common)
    if failures:
        evidence = json.loads(failures[-1].evidence_json)
        other = str(evidence.get("link_b_or_object", ""))
        centre = _cell_centre(grid, int(other[6:])) if other.startswith("voxel_") else None
        return RestVerdict(outcome="refused", evidence=evidence, cell_centre_m=centre, **common)
    drops = [s.drop_reason for s in statuses if _is_drop(s)]
    return RestVerdict(outcome="dropped", drop_reason=drops[-1] if drops else None, **common)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bag", type=Path, required=True, help="rosbag2 bag directory or file")
    parser.add_argument("--scene", type=Path, default=_DEFAULT_SCENE, help="deploy scene YAML")
    parser.add_argument("--unit", default=os.environ.get("OPENRAL_ROBOT_UNIT"), help="robot unit")
    parser.add_argument(
        "--pose",
        choices=("measured", "zero"),
        default="measured",
        help="judge the bag's last joint state (default) or the zero configuration",
    )
    parser.add_argument("--json", action="store_true", help="print the verdict as JSON")
    args = parser.parse_args(argv)

    try:
        params = deploy_node_params(args.scene.resolve(), args.unit)
        kernel = params["kernel"]
        if not kernel.get("world_voxel_enabled"):
            print(f"{args.scene}: the real graph's kernel has the world-voxel check off")
            return 2
        messages = read_last_messages(args.bag, [_VOXELS_TOPIC, _JOINT_STATES_TOPIC])
        grid = messages.get(_VOXELS_TOPIC)
        if grid is None:
            print(f"{args.bag}: no {_VOXELS_TOPIC} message")
            return 2
        row = rest_row(
            list(kernel["collision_joint_names"]),
            list(params["self_filter"].get("collision_joint_aliases", [])),
            messages.get(_JOINT_STATES_TOPIC),
            args.pose,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}")
        return 2
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "kernel.log"
        verdict = judge_hold_row(kernel, grid, row, log_path=log)
        if args.json:
            print(json.dumps(asdict(verdict), indent=2))
        else:
            print(
                f"{verdict.outcome}: hold row at {args.pose} pose against {verdict.occupied_cells} "
                f"occupied cells ({verdict.grid_frame}), world-voxel margin "
                f"{kernel.get('world_voxel_margin_m')} m"
            )
            if verdict.outcome == "refused":
                ev = verdict.evidence
                print(
                    f"  {ev.get('collision_kind')}: {ev.get('link_a')} vs "
                    f"{ev.get('link_b_or_object')} at {ev.get('min_distance_m')} m; "
                    f"cell centre {verdict.cell_centre_m}"
                )
            elif verdict.outcome == "dropped":
                print(f"  drop_reason {verdict.drop_reason}; kernel log:\n{log.read_text()}")
    return {"accepted": 0, "refused": 1, "dropped": 3}[verdict.outcome]


if __name__ == "__main__":
    sys.exit(main())
