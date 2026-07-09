#!/usr/bin/env python3
"""Visualise a robot's *kernel* collision primitives on top of its real meshes.

Overlays the exact box/capsule geometry the C++ safety kernel checks (lowered by
``collision_params_from_description``) onto the robot's MJCF meshes, at any joint
pose. Semi-transparent so you can see the primitive fit — this is how you eyeball
whether the SO-101 ``base`` OBB (box/OBB collision primitive, issue #84) hugs the housing and
clears the folded distal links, without a `deploy run`.

This is a standalone inspection tool, NOT a pytest test (tests stay headless for
CI). Two modes:

    # interactive MuJoCo window (needs a display)
    python tools/viz_collision.py --robot so101_follower --viewer

    # offscreen screenshot (headless-safe)
    python tools/viz_collision.py --robot so101_follower --screenshot /tmp/so101.png

    # inspect a specific folded pose (degrees, manifest joint order)
    python tools/viz_collision.py --robot so101_follower --deg 0 -60 90 0 0 0 --screenshot /tmp/fold.png

    # real RViz (RobotModel + TF + collision MarkerArray) — needs ROS sourced
    python tools/viz_collision.py --robot so101_follower --rviz --deg 0 -75 100 40 0 0

IMPORTANT — env. ``openral_core`` lives in the worktree venv and ``openral_safety``
is a ROS package, so run the VENV python with the safety package on the path:

    PYTHONPATH=packages/openral_safety MUJOCO_GL=glfw \\
      .venv/bin/python tools/viz_collision.py --robot so101_follower --viewer

Use ``MUJOCO_GL=egl`` for ``--screenshot`` (offscreen). For ``--rviz`` also
``source /opt/ros/jazzy/setup.bash`` first (rclpy / rviz2 / robot_state_publisher).
"""

from __future__ import annotations

import argparse
import importlib
import math
import pathlib
from collections.abc import Sequence
from typing import cast

import numpy as np
from openral_core.assets import resolve_asset
from openral_core.schemas import RobotDescription

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - tool, not test
    raise SystemExit("mujoco is required: `just sync` then run inside the venv") from exc


_BOX_RGBA = (0.9, 0.2, 0.2, 0.35)  # translucent red for the OBB
_CAP_RGBA = (0.2, 0.5, 0.95, 0.30)  # translucent blue for capsules

_RVIZ_CONFIG = """\
Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Global Options:
    Fixed Frame: base
  Displays:
    - Class: rviz_default_plugins/Grid
      Enabled: true
      Name: Grid
    - Class: rviz_default_plugins/TF
      Enabled: true
      Name: TF
    - Class: rviz_default_plugins/RobotModel
      Enabled: true
      Name: RobotModel
      Description Source: Topic
      Description Topic:
        Value: /robot_description
        Durability Policy: Transient Local
      Visual Enabled: true
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: CollisionPrimitives
      Topic:
        Value: /collision_markers
        Durability Policy: Transient Local
  Tools:
    - Class: rviz_default_plugins/MoveCamera
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 0.9
      Focal Point:
        X: 0.0
        Y: 0.0
        Z: 0.15
"""


def _float_list(params: dict[str, object], key: str) -> list[float]:
    return cast(list[float], params.get(key, []))


def _int_list(params: dict[str, object], key: str) -> list[int]:
    return cast(list[int], params.get(key, []))


def _str_list(params: dict[str, object], key: str) -> list[str]:
    return cast(list[str], params[key])


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """Fixed-axis XYZ Euler (URDF rpy) → MuJoCo wxyz quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _add_primitives(spec: mujoco.MjSpec, params: dict[str, object]) -> int:
    """Add the kernel's box + capsule collision geoms to the spec bodies."""
    names = _str_list(params, "collision_link_names")
    bodies = {b.name: b for b in spec.bodies}
    added = 0

    box_link = _int_list(params, "collision_box_link")
    box_he = _float_list(params, "collision_box_half_extents")
    box_o = _float_list(params, "collision_box_origin_xyzrpy")
    for b, link in enumerate(box_link):
        body = bodies.get(names[link])
        if body is None:
            continue
        g = body.add_geom()
        g.name = f"col_box_{names[link]}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = box_he[3 * b : 3 * b + 3]
        g.pos = box_o[6 * b : 6 * b + 3]
        g.quat = _rpy_to_quat(*box_o[6 * b + 3 : 6 * b + 6])
        g.rgba = _BOX_RGBA
        g.contype = 0
        g.conaffinity = 0
        added += 1

    cap_link = _int_list(params, "collision_capsule_link")
    cap_r = _float_list(params, "collision_capsule_radius")
    cap_h = _float_list(params, "collision_capsule_half_length")
    cap_o = _float_list(params, "collision_capsule_origin_xyzrpy")
    for c, link in enumerate(cap_link):
        body = bodies.get(names[link])
        if body is None:
            continue
        g = body.add_geom()
        g.name = f"col_cap_{names[link]}"
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size = [cap_r[c], cap_h[c], 0.0]
        g.pos = cap_o[6 * c : 6 * c + 3]
        g.quat = _rpy_to_quat(*cap_o[6 * c + 3 : 6 * c + 6])
        g.rgba = _CAP_RGBA
        g.contype = 0
        g.conaffinity = 0
        added += 1
    return added


def _capsule_marker_parts(
    marker_id: int,
    frame: str,
    xyzrpy: Sequence[float],
    radius: float,
    half_length: float,
    rgba: Sequence[float],
) -> list[object]:
    """RViz has no capsule primitive — approximate with a cylinder + two end
    spheres (visualization_msgs/Marker list)."""
    from geometry_msgs.msg import Point, Pose, Quaternion

    Marker = importlib.import_module("visualization_msgs.msg").Marker

    q = _rpy_to_quat(*xyzrpy[3:6])
    pos = xyzrpy[0:3]
    out = []
    cyl = Marker()
    cyl.header.frame_id = frame
    cyl.ns = "capsule"
    cyl.id = marker_id
    cyl.type = Marker.CYLINDER
    cyl.action = Marker.ADD
    cyl.pose = Pose(
        position=Point(x=pos[0], y=pos[1], z=pos[2]),
        orientation=Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]),
    )
    cyl.scale.x = cyl.scale.y = 2 * radius
    cyl.scale.z = max(2 * half_length, 1e-4)
    cyl.color.r, cyl.color.g, cyl.color.b, cyl.color.a = rgba
    out.append(cyl)
    # End spheres along the local +z (the capsule axis).
    zx, zy, zz = (
        2 * (q[1] * q[3] + q[0] * q[2]),
        2 * (q[2] * q[3] - q[0] * q[1]),
        1 - 2 * (q[1] * q[1] + q[2] * q[2]),
    )
    for sign, sid in ((+1, 1000), (-1, 2000)):
        s = Marker()
        s.header.frame_id = frame
        s.ns = "capsule"
        s.id = marker_id + sid
        s.type = Marker.SPHERE
        s.action = Marker.ADD
        s.pose.position = Point(
            x=pos[0] + sign * half_length * zx,
            y=pos[1] + sign * half_length * zy,
            z=pos[2] + sign * half_length * zz,
        )
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 2 * radius
        s.color.r, s.color.g, s.color.b, s.color.a = rgba
        out.append(s)
    return out


def _run_rviz(
    rd: RobotDescription, params: dict[str, object], manifest: pathlib.Path, deg: list[float] | None
) -> None:
    """Real RViz: robot_state_publisher (RobotModel + TF from the URDF) + a
    latched collision MarkerArray + rviz2, all as child processes."""
    import subprocess
    import tempfile

    import rclpy
    from geometry_msgs.msg import Point, Pose, Quaternion
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import JointState

    marker_msg = importlib.import_module("visualization_msgs.msg")
    Marker = marker_msg.Marker
    MarkerArray = marker_msg.MarkerArray

    # URDF with absolute file:// mesh paths so RViz's RobotModel resolves them.
    if rd.assets.urdf is None:
        raise SystemExit(f"{manifest} has no URDF asset for RViz")
    try:
        urdf_src = resolve_asset(rd.assets.urdf.ref, "urdf", manifest_dir=manifest.parent)
    except Exception:
        ref = rd.assets.urdf.ref.split(":", 1)[-1]
        urdf_src = manifest.parent / ref
    if urdf_src is None:
        raise SystemExit(f"{manifest} URDF is provided only at runtime")
    assets_abs = (manifest.parent / "assets").resolve()
    urdf_text = (
        pathlib.Path(urdf_src)
        .read_text()
        .replace('filename="assets/', f'filename="file://{assets_abs}/')
    )
    tmp_urdf = pathlib.Path(tempfile.mkstemp(suffix=".urdf")[1])
    tmp_urdf.write_text(urdf_text)

    joint_names = [j.name for j in rd.joints]
    positions = [
        math.radians(deg[i]) if deg and i < len(deg) else 0.0 for i in range(len(joint_names))
    ]

    rsp = subprocess.Popen(
        ["ros2", "run", "robot_state_publisher", "robot_state_publisher", str(tmp_urdf)]
    )
    rviz_cfg = pathlib.Path(tempfile.mkstemp(suffix=".rviz")[1])
    rviz_cfg.write_text(_RVIZ_CONFIG)
    rviz = subprocess.Popen(["rviz2", "-d", str(rviz_cfg)])

    rclpy.init()
    node = Node("openral_collision_viz")
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    js_pub = node.create_publisher(JointState, "/joint_states", 10)
    mk_pub = node.create_publisher(MarkerArray, "/collision_markers", latched)

    names = _str_list(params, "collision_link_names")
    arr = MarkerArray()
    mid = 0
    box_he = _float_list(params, "collision_box_half_extents")
    box_origin = _float_list(params, "collision_box_origin_xyzrpy")
    for b, link in enumerate(_int_list(params, "collision_box_link")):
        q = _rpy_to_quat(*box_origin[6 * b + 3 : 6 * b + 6])
        m = Marker()
        m.header.frame_id = names[link]
        m.ns, m.id, m.type, m.action = "box", mid, Marker.CUBE, Marker.ADD
        m.pose = Pose(
            position=Point(x=box_origin[6 * b], y=box_origin[6 * b + 1], z=box_origin[6 * b + 2]),
            orientation=Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]),
        )
        m.scale.x = 2 * box_he[3 * b]
        m.scale.y = 2 * box_he[3 * b + 1]
        m.scale.z = 2 * box_he[3 * b + 2]
        m.color.r, m.color.g, m.color.b, m.color.a = _BOX_RGBA
        arr.markers.append(m)
        mid += 1
    cap_origin = _float_list(params, "collision_capsule_origin_xyzrpy")
    cap_radius = _float_list(params, "collision_capsule_radius")
    cap_half_length = _float_list(params, "collision_capsule_half_length")
    for c, link in enumerate(_int_list(params, "collision_capsule_link")):
        arr.markers.extend(
            _capsule_marker_parts(
                mid,
                names[link],
                cap_origin[6 * c : 6 * c + 6],
                cap_radius[c],
                cap_half_length[c],
                _CAP_RGBA,
            )
        )
        mid += 1

    js = JointState()
    js.name = joint_names
    js.position = positions

    def _tick() -> None:
        js.header.stamp = node.get_clock().now().to_msg()
        js_pub.publish(js)
        mk_pub.publish(arr)

    node.create_timer(0.1, _tick)
    node.get_logger().info("RViz collision viz up — fixed frame 'base'. Ctrl-C to quit.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        rsp.terminate()
        rviz.terminate()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="so101_follower")
    ap.add_argument("--deg", nargs="*", type=float, help="joint angles in degrees (manifest order)")
    ap.add_argument("--screenshot", type=pathlib.Path, help="write an offscreen PNG and exit")
    ap.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo window")
    ap.add_argument(
        "--rviz", action="store_true", help="open real RViz (RobotModel + TF + markers)"
    )
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    args = ap.parse_args()

    manifest = pathlib.Path("robots") / args.robot / "robot.yaml"
    rd = RobotDescription.from_yaml(str(manifest))

    from openral_safety.envelope_loader import collision_params_from_description

    params = collision_params_from_description(rd)
    if not params.get("self_collision_enabled"):
        raise SystemExit(f"{args.robot} has no self-collision geometry to visualise")

    if args.rviz:
        _run_rviz(rd, params, manifest, args.deg)
        return

    if rd.assets.mjcf is None:
        raise SystemExit(f"{manifest} has no MJCF asset to visualise")
    mjcf = resolve_asset(rd.assets.mjcf, "mjcf", manifest_dir=manifest.parent)
    if mjcf is None:
        raise SystemExit(f"{manifest} MJCF is provided only at runtime")
    spec = mujoco.MjSpec.from_file(str(mjcf))
    # The upstream MJCF ships a small offscreen buffer; size it to the request so
    # the offscreen renderer can produce the screenshot.
    spec.visual.global_.offwidth = max(args.width, 640)
    spec.visual.global_.offheight = max(args.height, 480)
    n = _add_primitives(spec, params)
    model = spec.compile()
    data = mujoco.MjData(model)

    if args.deg:
        for i, d in enumerate(args.deg[: model.nq]):
            data.qpos[i] = math.radians(d)
    mujoco.mj_forward(model, data)
    print(
        f"{args.robot}: overlaid {n} kernel collision primitives "
        f"(box=red, capsule=blue) at qpos(deg)={np.round(np.degrees(data.qpos[: model.nq]), 1)}"
    )

    if args.screenshot:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [0.0, 0.0, 0.16]
        cam.distance = 0.95
        cam.azimuth = 135
        cam.elevation = -20
        opt = mujoco.MjvOption()
        renderer.update_scene(data, camera=cam, scene_option=opt)
        pixels = renderer.render()
        try:
            from PIL import Image

            Image.fromarray(pixels).save(args.screenshot)
        except ImportError:
            import imageio.v3 as iio

            iio.imwrite(args.screenshot, pixels)
        print(f"wrote {args.screenshot}")
        return

    if args.viewer:
        from mujoco import viewer as mj_viewer

        mj_viewer.launch(model, data)
        return

    print("nothing to do — pass --screenshot PATH or --viewer")


if __name__ == "__main__":
    main()
