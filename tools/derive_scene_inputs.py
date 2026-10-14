"""Derive a deploy scene's safety inputs from the simulator instead of writing them by hand.

Boots the scene's sim env exactly as ``openral deploy sim`` does
(``openral_hal.sim_bringup.build_sim_env_from_yaml``) and writes a derived scene with:

* **declared static geometry** (``runtime.static_geometry_file``, fixed-base robots only:
  it is in ``base_frame``): every static collider
  in the manifest ``base_frame`` — Isaac: the sidecar's ``static_colliders`` (USD
  collision prims of non-robot, non-dynamic fixtures); MuJoCo: world-welded collision
  geoms (``openral_sim.backends.mujoco_geometry.static_collider_boxes``). Each box is a
  superset of its collider, the fail-closed direction.
* **grasp declaration boxes** (``grasp_declaration.search_box`` / ``workspace_box``) —
  in ``base_frame``, or for a mobile base on MuJoCo in its ``odom_frame`` (the sim's
  world), which the producer re-expresses in the grid frame every tick — grounded from the instruction's target object and destination as the sim knows them
  (the backend's ``scene_ground_truth``) through the reasoner's own grounding
  (``openral_reasoner.grounding.ground_grasp_target`` / ``ground_workspace``) — the
  functions a real cell feeds from perception instead. The workspace's source fixture
  is the static collider group the object rests on (``source_fixture``); its
  destination is the target region plus any regions the backend grades as adjacent.

Explicit wins: a value the scene already declares is kept unless ``--replace`` (then the
declared and derived boxes are both logged for comparison)::

    python tools/derive_scene_inputs.py --config scenes/deploy/local/restock.yaml \\
        --out scenes/deploy/local/restock_derived.yaml [--replace] [--max-distance 1.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from openral_reasoner.grounding import Bbox


def to_declared(box: dict[str, Any]) -> dict[str, Any]:
    """A backend ``kind: box`` collider as a declared static-geometry entry (base frame).

    Example:
        >>> e = to_declared(
        ...     {
        ...         "path": "/World/env/Wall",
        ...         "center_m": [1.0, 0.0, 0.2],
        ...         "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ...         "half_extents_m": [0.1, 1.0, 0.5],
        ...         "collision_enabled": True,
        ...     }
        ... )
        >>> e["kind"], e["xyz"], e["quat_xyzw"], e["half_extents"], e["source"]
        ('box', [1.0, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0], [0.1, 1.0, 0.5], '/World/env/Wall')
    """
    from openral_core.geometry import rotation_to_quat_wxyz

    w, x, y, z = rotation_to_quat_wxyz(np.asarray(box["rotation"], dtype=np.float64))
    return {
        "kind": "box",
        "source": box["path"],
        "xyz": [float(v) for v in box["center_m"]],
        "quat_xyzw": [round(float(v), 9) for v in (x, y, z, w)],
        "half_extents": [float(v) for v in box["half_extents_m"]],
    }


def mesh_bounds(vertices: Any) -> tuple[list[float], list[float]]:
    """Centre and half-extents of a vertex set's axis-aligned bound.

    Example:
        >>> mesh_bounds([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
        ([1.0, 0.5, 0.0], [1.0, 0.5, 0.0])
    """
    v = np.asarray(vertices, dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return [float(x) for x in (lo + hi) / 2.0], [float(x) for x in (hi - lo) / 2.0]


def nearest_distance_m(entry: dict[str, Any]) -> float:
    """Lower bound on the entry's distance from the base-frame origin (centre minus radius).

    Example:
        >>> nearest_distance_m({"xyz": [3.0, 0.0, 0.0], "half_extents": [1.0, 0.0, 0.0]})
        2.0
    """
    return max(
        0.0, float(np.linalg.norm(entry["xyz"])) - float(np.linalg.norm(entry["half_extents"]))
    )


def aabb(box: dict[str, Any]) -> Bbox:
    """The base-frame axis-aligned bound ``(x0, y0, z0, x1, y1, z1)`` of an oriented box.

    Example:
        >>> b = {"center_m": [1, 0, 0], "rotation": np.eye(3), "half_extents_m": [0.1, 0.2, 0.3]}
        >>> aabb(b)
        (0.9, -0.2, -0.3, 1.1, 0.2, 0.3)
    """
    rot = np.abs(np.asarray(box["rotation"], dtype=np.float64))
    c = np.asarray(box["center_m"], dtype=np.float64)
    r = rot @ np.asarray(box["half_extents_m"], dtype=np.float64)
    lo, hi = c - r, c + r
    return (
        round(float(lo[0]), 6),
        round(float(lo[1]), 6),
        round(float(lo[2]), 6),
        round(float(hi[0]), 6),
        round(float(hi[1]), 6),
        round(float(hi[2]), 6),
    )


def source_fixture(
    obj: Bbox, colliders: list[dict[str, Any]], tol_m: float = 0.02
) -> tuple[Bbox, str] | None:
    """The bound of the static collider group ``obj`` rests on, and that group's path.

    The support is the highest static box whose footprint holds the object's centre
    and whose top is at most ``tol_m`` above the object's bottom; its group is every
    collider sharing its parent path (a tote's floor, walls and rims; a counter body's
    geoms). ``None`` when nothing static is under the object.

    Example:
        >>> eye = np.eye(3)
        >>> floor = {
        ...     "path": "/T/Floor",
        ...     "center_m": [0, 0, 0],
        ...     "rotation": eye,
        ...     "half_extents_m": [0.2, 0.2, 0.01],
        ... }
        >>> wall = {
        ...     "path": "/T/Wall",
        ...     "center_m": [0.2, 0, 0.1],
        ...     "rotation": eye,
        ...     "half_extents_m": [0.01, 0.2, 0.1],
        ... }
        >>> source_fixture((-0.05, -0.05, 0.01, 0.05, 0.05, 0.1), [floor, wall])
        ((-0.2, -0.2, -0.01, 0.21, 0.2, 0.2), '/T')
    """
    cx, cy = (obj[0] + obj[3]) / 2.0, (obj[1] + obj[4]) / 2.0
    best: tuple[float, str] | None = None
    for c in colliders:
        b = aabb(c)
        supports = b[0] <= cx <= b[3] and b[1] <= cy <= b[4] and b[5] <= obj[2] + tol_m
        if supports and (best is None or b[5] > best[0]):
            best = (b[5], c["path"].rsplit("/", 1)[0])
    if best is None:
        return None
    group = [aabb(c) for c in colliders if c["path"].rsplit("/", 1)[0] == best[1]]
    lo = [min(b[i] for b in group) for i in range(3)]
    hi = [max(b[i + 3] for b in group) for i in range(3)]
    return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]), best[1]


def _static_reply(
    env: Any, description: Any, *, world: bool = False
) -> tuple[dict[str, Any], int | None]:
    """The backend's static colliders and, for MuJoCo, the body id they are in.

    ``world``: in the MuJoCo world body's frame (id 0) instead of the base's — the
    sim's ``odom`` frame, where a mobile base's declarations stay put.
    """
    from openral_core.exceptions import ROSConfigError

    export = getattr(env, "static_collider_boxes", None)
    if callable(export):
        return export(), None
    handles = getattr(env, "mujoco_handles", lambda: None)()
    if handles is None:
        raise ROSConfigError("the scene's backend exports no static colliders")
    import mujoco
    from openral_hal.depth_cloud import resolve_base_frame_body_name, robot_self_body_ids
    from openral_sim.backends.mujoco_geometry import static_collider_boxes

    model, data = handles
    name = None if world else resolve_base_frame_body_name(model, description=description)
    base_id = 0 if name is None else int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    robot = robot_self_body_ids(model, [j.sim_joint_name for j in description.joints])
    reply = static_collider_boxes(model, data, base_body_id=base_id, exclude_body_ids=robot)
    return reply, base_id


def write_static(reply: dict[str, Any], out: Path, max_distance: float | None) -> int:
    """Write the backend's colliders as a ``StaticGeometry`` YAML (+ mesh ``.npz``); count."""
    entries: list[dict[str, Any]] = []
    meshes: dict[str, dict[str, Any]] = {}
    for c in reply["boxes"]:
        if not c.get("collision_enabled", True):
            continue
        if c.get("kind") == "mesh":
            center, half = mesh_bounds(c["vertices_m"])
            name = c["path"].strip("/").replace("/", "__") + ".npz"
            meshes[name] = c
            entries.append(
                {
                    "kind": "mesh",
                    "source": c["path"],
                    "file": name,
                    "xyz": center,
                    "half_extents": half,
                    "triangles": len(c["triangles"]),
                }
            )
        else:
            entries.append(to_declared(c))
    if max_distance is not None:
        entries = [e for e in entries if nearest_distance_m(e) <= max_distance]
    entries.sort(key=nearest_distance_m)
    out.parent.mkdir(parents=True, exist_ok=True)
    for e in entries:
        if e["kind"] == "mesh":
            c = meshes[e["file"]]
            np.savez_compressed(
                out.parent / e["file"],
                vertices=np.asarray(c["vertices_m"], dtype=np.float32),
                triangles=np.asarray(c["triangles"], dtype=np.int32),
            )
            # Only the box-like summary goes in the YAML; the surface is in the file.
            del e["xyz"], e["half_extents"]
    out.write_text(yaml.safe_dump({"static_geometry": entries}, sort_keys=False))
    return len(entries)


def _robot_description(config: Path) -> Any:
    from openral_core import DeployScene, RobotDescription, load_scene_strict
    from openral_sim import SCENES
    from openral_sim.policies.robots import resolve_robot_manifest

    scene = load_scene_strict(str(config), DeployScene)
    robot_id = scene.robot_id or SCENES.fixed_robot(scene.scene.id)
    return RobotDescription.from_yaml(str(resolve_robot_manifest(str(robot_id))))


def _box_line(region: dict[str, Any] | None) -> str:
    if region is None:
        return "none"
    xyz = [round(float(v), 4) for v in region["pose"]["xyz"]]
    half = [round(float(v), 4) for v in region["half_extents"]]
    return f"centre {xyz} half {half}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="Deploy scene YAML (sim backend).")
    parser.add_argument("--out", required=True, help="Derived deploy scene YAML to write.")
    parser.add_argument("--instruction", help="Prompt to ground (default: preload_prompt).")
    parser.add_argument("--replace", action="store_true", help="Derived values replace declared.")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Keep only colliders whose nearest point may lie within this many metres.",
    )
    parser.add_argument(
        "--search-pad-m",
        type=float,
        default=0.045,
        help="Search box pad per side: it must hold the producer's measured region, the "
        "object grown by its fit margin and one cell's quantisation, plus pose error "
        "(15 mm each at the restock cell's 15 mm grid).",
    )
    parser.add_argument("--workspace-pad-m", type=float, default=0.02)
    parser.add_argument(
        "--workspace-headroom-m",
        type=float,
        default=0.10,
        help="Workspace height above the hull: the hand rides above what it carries.",
    )
    args = parser.parse_args(argv)

    import openral_sim.backends  # noqa: F401 — registers the sim scene factories
    from openral_core import (
        DetectedObject,
        GraspDeclaration,
        GraspTargetRef,
        Pose6D,
        gripper_hands,
    )
    from openral_core.exceptions import ROSConfigError, ROSReasonerInvalidPlan
    from openral_hal.sim_bringup import build_sim_env_from_yaml
    from openral_reasoner.grounding import ground_grasp_target, ground_workspace

    config = Path(args.config)
    scene = yaml.safe_load(config.read_text())
    runtime = scene.get("runtime") or {}
    scene["runtime"] = runtime
    instruction = args.instruction or runtime.get("preload_prompt")
    description = _robot_description(config)
    mobile = bool(description.base_joints)
    env, seed = build_sim_env_from_yaml(str(config), robot_id_fallback=None)
    try:
        env.reset(seed=seed)
        static, base_id = _static_reply(env, description, world=mobile)
        set_task = getattr(env, "set_task", None)
        if instruction and callable(set_task):
            set_task(instruction)
        truth_fn = getattr(env, "scene_ground_truth", None)
        if not callable(truth_fn):
            raise ROSConfigError("the scene's backend has no scene_ground_truth hook")
        truth = truth_fn() if base_id is None else truth_fn(base_body_id=base_id)
    finally:
        env.close()

    out = Path(args.out)
    if runtime.get("static_geometry_file") and not args.replace:
        print(f"static geometry: declared, kept ({runtime['static_geometry_file']})")
    elif mobile:
        # ponytail: base-frame fixtures; a map-frame export when a mobile cell needs one.
        print(
            "static geometry: not derived — the robot has a mobile base, and declared "
            "static geometry is in base_frame, true only at the reset pose"
        )
    else:
        static_out = out.with_name(out.stem + "_static.yaml")
        n = write_static(static, static_out, args.max_distance)
        runtime["static_geometry_file"] = str(static_out.resolve())
        print(f"static geometry: derived {n} colliders -> {static_out}")

    if not truth["placements"]:
        raise ROSConfigError(f"no target object / destination to ground for {instruction!r}")
    # A mobile base's declarations go in the world frame (MuJoCo's world body is the
    # sim's ``odom``): the producer re-expresses them in the grid frame every tick, so
    # they stay true after the base moves. Isaac reports base-frame truth only.
    base_frame = (
        description.odom_frame if mobile and base_id is not None else description.base_frame
    )
    if mobile and base_id is None:
        print(
            f"grasp declaration boxes: in {base_frame} — this backend reports base-frame "
            "truth only, true at the reset pose"
        )
    placement = truth["placements"][0]
    obj_name = str(placement["object"])
    live = [
        DetectedObject(
            label=name,
            confidence=1.0,
            pose=Pose6D(
                xyz=tuple(box["center_m"]), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=base_frame
            ),
            bbox_3d=aabb(box),
        )
        for name, box in truth["objects"].items()
    ]
    declared: dict[str, Any] = dict(scene.get("grasp_declaration") or {})
    grounded = ground_grasp_target(
        GraspTargetRef(label=obj_name, contact_links=list(declared.get("contact_links") or [])),
        live_objects=live,
        scene_graph=None,
        base_frame=base_frame,
        default_contact_links=gripper_hands(description),
        patience_s=GraspDeclaration.MAX_TIMEOUT_S,
        pad_m=args.search_pad_m,
    )
    obj_box = aabb(truth["objects"][obj_name])
    colliders = [
        c
        for c in static["boxes"]
        if c.get("kind", "box") == "box" and c.get("collision_enabled", True)
    ]
    # The destination and any regions graded as adjacent to it (the hand works there too).
    dest = [aabb(placement["target"]), *(aabb(a) for a in placement.get("adjacent", []))]
    destination: Bbox = (
        min(b[0] for b in dest),
        min(b[1] for b in dest),
        min(b[2] for b in dest),
        max(b[3] for b in dest),
        max(b[4] for b in dest),
        max(b[5] for b in dest),
    )
    source = source_fixture(obj_box, colliders)
    centre = truth["objects"][obj_name]["center_m"]
    try:
        workspace = ground_workspace(
            source[0] if source is not None else obj_box,
            destination,
            anchor_xyz=(float(centre[0]), float(centre[1]), float(centre[2])),
            base_frame=base_frame,
            pad_m=args.workspace_pad_m,
            headroom_m=args.workspace_headroom_m,
            evidence_ref=f"sim:{source[1] if source else obj_name} + destination",
        )
    except ROSReasonerInvalidPlan as exc:
        print(f"workspace_box: not derived ({exc})")
        workspace = None
    derived = {
        "search_box": grounded.search_box.model_dump(mode="json")
        if grounded.search_box is not None
        else None,
        "workspace_box": None if workspace is None else workspace.model_dump(mode="json"),
    }
    print(f"grounded {obj_name!r} on {source[1] if source else 'no static support'!r}")
    scene_declared = dict(declared)
    if not declared:
        declared = grounded.model_dump(mode="json", exclude={"region", "workspace_region"})
    for key, value in derived.items():
        old = scene_declared.get(key)
        print(f"{key}: declared {_box_line(old)} | derived {_box_line(value)}")
        if old is None or args.replace:
            declared[key] = value
    scene["grasp_declaration"] = GraspDeclaration.model_validate(declared).model_dump(
        mode="json", exclude_none=True
    )
    out.write_text(yaml.safe_dump(scene, sort_keys=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
