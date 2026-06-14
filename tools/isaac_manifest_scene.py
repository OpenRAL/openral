"""Robot-agnostic, URDF-driven Isaac Sim scene for the sidecar (ADR-0045 amendment).

Runs under the Isaac Sim py3.11 venv only; imported by ``isaac_sidecar.py`` AFTER
``SimulationApp`` is live (every import here needs a running Kit app).

Why this exists
---------------
The PoC scenes (``IsaacLiftScene`` in ``isaac_scene.py``, ``IsaacBowlPlateScene``
in ``isaac_bowl_plate_scene.py``) hardcode Isaac's built-in ``Franka`` example USD
asset and ignore the forwarded ``--robot``. That contradicts the ``DeployScene``
contract, where a scene is *environment + backend* and the robot is pluggable from
its ``RobotDescription`` (ADR-0033 / ADR-0034). This scene is the robot-agnostic
path: it builds the articulation by **importing the manifest robot's URDF**
(Isaac's ``isaacsim.asset.importer.urdf`` extension) and wires joints / sensors /
control from a plain-JSON "isaac robot spec" the openral-side backend marshals
across the venv boundary (the sidecar cannot import ``openral_core``).

Milestone M1 (this module's first cut): a fixed-base arm (``franka_panda``)
imported from its URDF, one RGB camera, and a JOINT_POSITION-delta articulation
controller, with ``/joint_states`` carrying the live imported-arm pose. Generic
depth/lidar sensors (M2) and the mobile base (M3) extend the same class.

Robot-spec contract (built by ``openral_sim.backends.isaac_sim._build_robot_spec``)::

    {
      "robot_id": str,
      "urdf_path": str,                  # resolved absolute path to the URDF file
      "fix_base": bool,                  # pin the root (True for a fixed arm)
      "joints": [{"name", "role", "joint_type"}],   # manifest order, actuated only
      "base_joints": [str] | null,       # [forward, side, yaw] for a planar base
      "base_kinematics": str | null,
      "action": {"dim": int, "control_mode": str, "arm_delta_scale": float,
                 "gripper_open_m": float, "gripper_closed_m": float},
      "sensors": [{"name", "modality", "vla_feature_key", "frame_id",
                   "parent_frame", "intrinsics": {...}, "range_min_m",
                   "range_max_m", "n_channels"}],
    }
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _isaac_scene_base import IsaacSceneBase
from numpy.typing import NDArray

# Below this magnitude a gripper action channel means HOLD (don't open/close) —
# lets a pure BODY_TWIST step (zero arm/gripper slots) leave the gripper alone.
_GRIPPER_DEADBAND = 1e-3


def map_dof_to_manifest(
    values: NDArray[np.float32],
    *,
    dof_index: dict[str, int],
    manifest_joints: list[dict[str, Any]],
    finger_dof_idx: list[int],
    base_values: list[float] | None = None,
    base_joints: list[str] | None = None,
) -> NDArray[np.float32]:
    """Map an Isaac articulation DOF vector to the full manifest joint order.

    Generic replacement for the Franka-specific ``_franka_dof_to_manifest``.
    For each manifest joint, in order:

    * a planar **base** joint (name in ``base_joints``) → the matching component
      of ``base_values`` (the kinematic base pose/twist) — it is not a URDF DOF;
    * an arm joint whose ``name`` matches a URDF DOF name → that DOF value;
    * a ``role == "gripper"`` joint with no direct DOF (the OpenRAL manifest
      collapses the two physical finger joints into one width DoF) → the mean of
      the finger DOFs;
    * anything unresolved → ``0.0``.

    Works for positions or velocities (same indexing). Returns a vector in
    ``manifest_joints`` order so ``SimAttachedHAL.read_state`` can index it
    against ``description.joints``.
    """
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    base_idx = {name: i for i, name in enumerate(base_joints or [])}
    bv = base_values or []
    out: list[float] = []
    for j in manifest_joints:
        name = str(j.get("name", ""))
        if name in base_idx and base_idx[name] < len(bv):
            out.append(float(bv[base_idx[name]]))
            continue
        idx = dof_index.get(name)
        if idx is not None and idx < v.shape[0]:
            out.append(float(v[idx]))
        elif j.get("role") == "gripper" and finger_dof_idx:
            out.append(float(np.mean([v[i] for i in finger_dof_idx if i < v.shape[0]] or [0.0])))
        else:
            out.append(0.0)
    return np.asarray(out, dtype=np.float32)


class IsaacManifestScene(IsaacSceneBase):
    """A URDF-imported, manifest-driven Isaac Sim scene (ADR-0045 amendment)."""

    warmup_steps = 4
    physics_substeps = 1

    def __init__(self, *, robot_spec: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._spec = robot_spec
        # Actuated manifest joints (the planar base is handled separately in M3;
        # it has no URDF DOF). Order is the manifest order.
        self._manifest_joints: list[dict[str, Any]] = list(robot_spec.get("joints", []))
        self._arm_names = [j["name"] for j in self._manifest_joints if j.get("role") == "arm"]
        self._base_joints: list[str] = list(robot_spec.get("base_joints") or [])
        action = robot_spec.get("action", {}) or {}
        self._control_mode = str(action.get("control_mode", "joint_position"))
        self._arm_delta_scale = float(action.get("arm_delta_scale", 0.05))
        self._gripper_open = float(action.get("gripper_open_m", 0.04))
        self._gripper_closed = float(action.get("gripper_closed_m", 0.0))
        n_gripper = sum(1 for j in self._manifest_joints if j.get("role") == "gripper")
        self._n_arm = len(self._arm_names)
        self._has_gripper = n_gripper > 0
        self._has_base = bool(action.get("has_base", False))
        self.action_dim = int(
            action.get(
                "dim",
                self._n_arm + (1 if self._has_gripper else 0) + (3 if self._has_base else 0),
            )
        )

        # Kinematic planar base (ADR-0045 amendment M3): the arm is imported
        # fix_base=True (pinned) and the whole articulation root is teleported each
        # step from an integrated (x, y, yaw) pose driven by the action's last 3
        # base-twist channels (vx, vy, wyaw, base frame). No PhysX base joints.
        # The base integrates by the BODY_TWIST command interval (one env.step =
        # one /cmd_vel command), NOT the physics dt — so a velocity command moves
        # the base v·dt per step, matching SimAttachedHAL's body_twist_dt_s.
        self._base_pose = [0.0, 0.0, 0.0]  # world x, y, yaw
        self._mount_z = float(robot_spec.get("mount_z", 0.0))
        self._base_dt = float(action.get("body_twist_dt_s", 0.05))

        self._robot: Any = None
        self._ArticulationAction: Any = None
        self._euler_to_quat: Any = None
        # One Isaac Camera per manifest RGB/depth sensor, base-relative so they
        # ride the kinematic base. `_cam_meta` is the ordered plan; `_cameras`
        # holds the live Camera objects keyed by sensor name (built in `build`).
        self._cam_meta: list[dict[str, Any]] = self._plan_cameras()
        self._cameras: dict[str, Any] = {}
        # 2-D lidar: a PhysX raycast fan from the base produces real /scan ranges
        # against the scene's obstacles (built in `build` when the manifest
        # declares a lidar_2d sensor). `_scan_z` is the beam height above base.
        self._lidar: dict[str, Any] | None = next(
            (s for s in robot_spec.get("sensors", []) if s.get("modality") == "lidar_2d"), None
        )
        self._scan_query: Any = None
        self._scan_z = 0.30
        # DOF mapping, resolved post-reset once dof_names is populated.
        self._dof_index: dict[str, int] = {}
        self._arm_dof_idx: list[int] = []
        self._finger_dof_idx: list[int] = []

    def _plan_cameras(self) -> list[dict[str, Any]]:
        """Plan one base-relative camera per manifest RGB/depth sensor.

        Each entry: ``{name, key, modality, offset_pos (base frame xyz), offset_euler
        (deg)}``. RGB sensors are keyed by their ``vla_feature_key`` suffix
        (``camera1``…) so ``SimSensorBridge`` finds them; depth sensors are keyed by
        the sensor ``name`` (the bridge's depth obs key). Offsets give each camera a
        distinct, workspace-facing viewpoint (the manifest carries no Isaac-frame
        extrinsics) — robot-mounted, so they translate/rotate with the base.
        """
        # base frame: x forward, y left, z up. (pos, euler_deg=[roll,pitch,yaw]).
        rgb_offsets = [
            ((1.5, 0.0, 1.0), (0.0, 35.0, 180.0)),  # front, look back + down
            ((1.3, -0.7, 1.1), (0.0, 38.0, 205.0)),  # front-right
            ((0.5, 0.0, 1.7), (0.0, 75.0, 180.0)),  # top-down over the workspace
        ]
        # Depth faces FORWARD (+x) and down — obstacle/ground sensing ahead of the
        # base, where a mobile manipulator drives (yaw 0, unlike the back-facing
        # manipulation agentviews above).
        depth_offset = ((0.30, 0.0, 0.75), (0.0, 20.0, 0.0))
        plan: list[dict[str, Any]] = []
        rgb_i = 0
        for s in self._spec.get("sensors", []):
            modality = s.get("modality")
            if modality == "rgb":
                pos, euler = rgb_offsets[min(rgb_i, len(rgb_offsets) - 1)]
                rgb_i += 1
                key = (
                    str(s["vla_feature_key"]).rsplit(".", 1)[-1]
                    if s.get("vla_feature_key")
                    else s.get("name", "camera1")
                )
                plan.append(
                    {
                        "name": s["name"],
                        "key": key,
                        "modality": "rgb",
                        "offset_pos": pos,
                        "offset_euler": euler,
                    }
                )
            elif modality == "depth":
                pos, euler = depth_offset
                plan.append(
                    {
                        "name": s["name"],
                        "key": s["name"],
                        "modality": "depth",
                        "offset_pos": pos,
                        "offset_euler": euler,
                        "range_max_m": s.get("range_max_m"),
                    }
                )
        return plan

    # ── build ────────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Import the manifest robot's URDF and stand up camera + ground."""
        import isaacsim.core.utils.numpy.rotations as rot_utils
        import omni.kit.commands
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera

        self._ArticulationAction = ArticulationAction

        # NOTE: no device="cuda:0" — forcing GPU PhysX hangs the first warmup for
        # minutes on an 8 GB laptop GPU (see ADR-0045 PoC notes). Default device
        # renders the same scene in ~15 s.
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()

        self._euler_to_quat = rot_utils.euler_angles_to_quats

        prim_path = self._import_urdf(omni.kit.commands)
        self._robot = self._world.scene.add(Robot(prim_path=prim_path, name="robot"))

        # A lidar robot gets a few static obstacles so /scan, slam, and Nav2 have
        # real geometry to map + avoid (a bare ground plane returns no hits).
        if self._lidar is not None:
            self._add_obstacles()

        # One Camera per planned sensor. A bare deploy scene with no RGB sensor
        # still gets a default camera1 so the obs always carries a frame.
        if not any(m["modality"] == "rgb" for m in self._cam_meta):
            self._cam_meta.insert(
                0,
                {
                    "name": "camera1",
                    "key": "camera1",
                    "modality": "rgb",
                    "offset_pos": (1.5, 0.0, 1.0),
                    "offset_euler": (0.0, 35.0, 180.0),
                },
            )
        for meta in self._cam_meta:
            cam = Camera(
                prim_path=f"/World/cam_{meta['name']}",
                resolution=(self.obs_width, self.obs_height),
            )
            self._cameras[meta["name"]] = cam

        self._world.reset()
        for meta in self._cam_meta:
            cam = self._cameras[meta["name"]]
            cam.initialize()
            if meta["modality"] == "depth":
                # distance_to_image_plane → the depth array we deproject to a cloud.
                cam.add_distance_to_image_plane_to_frame()
        self._update_camera_poses()
        self._resolve_dof_mapping()
        if self._lidar is not None:
            from omni.physx import get_physx_scene_query_interface

            self._scan_query = get_physx_scene_query_interface()

    def _add_obstacles(self) -> None:
        """Add a few static boxes around the robot for lidar/slam/Nav2 to see."""
        from isaacsim.core.api.objects import FixedCuboid

        # (center xyz, half-ish scale) — a partial enclosure + scattered boxes.
        boxes = [
            ((3.0, 0.0, 0.5), (0.3, 4.0, 1.0)),  # wall ahead
            ((-3.0, 0.0, 0.5), (0.3, 4.0, 1.0)),  # wall behind
            ((1.6, 1.8, 0.5), (0.6, 0.6, 1.0)),  # box front-left
            ((-1.5, -2.0, 0.5), (0.8, 0.8, 1.0)),  # box back-right
        ]
        for i, (pos, scale) in enumerate(boxes):
            self._world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/obstacle_{i}",
                    name=f"obstacle_{i}",
                    position=np.array(pos),
                    scale=np.array(scale),
                    color=np.array([0.4, 0.4, 0.45]),
                )
            )

    def _scan_ranges(self) -> NDArray[np.float32] | None:
        """A 2-D LaserScan range fan via PhysX raycasts from the base.

        ``n_channels`` beams from ``angle_min=-π`` to ``angle_max=+π`` (the
        bridge's convention) in the **base_link** frame, rotated to world by the
        base yaw. Each ray STARTS ``range_min_m`` beyond the base origin along the
        beam (past the robot's own chassis/arm column, which a centre-origin ray
        would hit at distance 0) and is cast for ``range_max_m - range_min_m``; the
        reported range is ``range_min_m + hit_distance``. A hit still on the robot
        (rare past ``range_min``), a miss, or an out-of-range beam reads
        ``range_max_m``. ``None`` when the manifest declares no lidar — never a
        fabricated scan.
        """
        if self._scan_query is None or self._lidar is None:
            return None
        n = int(self._lidar.get("n_channels") or 360)
        rmin = float(self._lidar.get("range_min_m") or 0.0)
        rmax = float(self._lidar.get("range_max_m") or 12.0)
        bx, by, byaw = self._base_pose
        z = float(self._scan_z)
        span = max(rmax - rmin, 0.0)
        out = np.full(n, rmax, dtype=np.float32)
        step = 2.0 * np.pi / n
        for i in range(n):
            ang = -np.pi + i * step + byaw  # base beam angle, rotated to world
            cx, cy = float(np.cos(ang)), float(np.sin(ang))
            start = (bx + rmin * cx, by + rmin * cy, z)  # past the robot footprint
            hit = self._scan_query.raycast_closest(start, (cx, cy, 0.0), span)
            # Ignore a hit still on the robot itself (imported under /panda).
            if (
                isinstance(hit, dict)
                and hit.get("hit")
                and not str(hit.get("rigidBody", "")).startswith("/panda")
            ):
                out[i] = min(rmin + float(hit.get("distance", span)), rmax)
        return out

    def _update_camera_poses(self) -> None:
        """Place every camera at its base-relative offset for the current base pose.

        The cameras are robot-mounted: as the kinematic base translates/rotates the
        viewpoints follow. base frame is planar (x, y, yaw), so the offset position
        rotates by yaw about z and the camera's euler yaw adds the base yaw.
        """
        bx, by, byaw = self._base_pose
        cos_y, sin_y = float(np.cos(byaw)), float(np.sin(byaw))
        byaw_deg = float(np.degrees(byaw))
        for meta in self._cam_meta:
            ox, oy, oz = meta["offset_pos"]
            wx = bx + cos_y * ox - sin_y * oy
            wy = by + sin_y * ox + cos_y * oy
            roll, pitch, yaw = meta["offset_euler"]
            quat = self._euler_to_quat(np.array([roll, pitch, yaw + byaw_deg]), degrees=True)
            self._cameras[meta["name"]].set_world_pose(np.array([wx, wy, oz]), quat)

    def _import_urdf(self, commands: Any) -> str:
        """Run the Isaac URDF importer; return the imported articulation prim path."""
        urdf_path = str(self._spec["urdf_path"])
        _status, import_config = commands.execute("URDFCreateImportConfig")
        # Keep the kinematic tree faithful to the manifest: do NOT merge fixed
        # joints (we map DOFs by URDF joint name), import as position-drive.
        import_config.merge_fixed_joints = False
        import_config.fix_base = bool(self._spec.get("fix_base", True))
        import_config.make_default_prim = False
        # World() already owns the PhysX scene + ground plane.
        import_config.create_physics_scene = False
        import_config.import_inertia_tensor = True
        import_config.distance_scale = 1.0
        result = commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=import_config,
        )
        # The command returns (success, prim_path); tolerate either a 2-tuple or a
        # bare path across Isaac point releases.
        prim_path = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else result
        if not prim_path:
            raise RuntimeError(f"URDF import returned no prim path for {urdf_path!r}")
        return str(prim_path)

    def _resolve_dof_mapping(self) -> None:
        """Index the imported articulation's DOFs by name (post-reset)."""
        dof_names = [str(n) for n in (self._robot.dof_names or [])]
        self._dof_index = {n: i for i, n in enumerate(dof_names)}
        self._arm_dof_idx = [self._dof_index[n] for n in self._arm_names if n in self._dof_index]
        # Finger DOFs back the collapsed manifest gripper joint.
        self._finger_dof_idx = [i for i, n in enumerate(dof_names) if "finger" in n.lower()]

    # ── IsaacSceneBase template methods ──────────────────────────────────────

    def _apply_action(self, action: NDArray[np.float32]) -> None:
        if self._control_mode != "joint_position":
            # JOINT_POSITION (+ kinematic base) is wired; CARTESIAN_DELTA (Lula IK)
            # for EE-delta VLAs is a follow-up.
            raise NotImplementedError(f"control_mode {self._control_mode!r} not wired yet")
        # Action layout: [arm deltas (n_arm), gripper (0/1), base twist (0/3)].
        current = np.asarray(self._robot.get_joint_positions(), dtype=np.float32).reshape(-1)
        target = current.copy()
        for k, dof_i in enumerate(self._arm_dof_idx):
            if k < action.shape[0]:
                target[dof_i] = current[dof_i] + float(action[k]) * self._arm_delta_scale
        if self._has_gripper and self._finger_dof_idx and action.shape[0] > self._n_arm:
            cmd = float(action[self._n_arm])
            # Deadband: a ~0 gripper command means HOLD (don't toggle), so a pure
            # base move (BODY_TWIST → zero arm/gripper slots) doesn't clamp the
            # gripper shut every /cmd_vel tick. >0 opens, <0 closes.
            if abs(cmd) > _GRIPPER_DEADBAND:
                finger = self._gripper_open if cmd > 0.0 else self._gripper_closed
                for fi in self._finger_dof_idx:
                    target[fi] = finger
        self._robot.get_articulation_controller().apply_action(
            self._ArticulationAction(joint_positions=target)
        )
        if self._has_base:
            base_i = self._n_arm + (1 if self._has_gripper else 0)
            if action.shape[0] >= base_i + 3:
                self._integrate_base(
                    float(action[base_i]), float(action[base_i + 1]), float(action[base_i + 2])
                )

    def _integrate_base(self, vx: float, vy: float, wyaw: float) -> None:
        """Advance the kinematic base by a base-frame twist and teleport the root.

        ``(vx, vy)`` are body-frame linear velocities (m/s), ``wyaw`` the yaw rate
        (rad/s). Integrated to a world ``(x, y, yaw)`` and applied as the
        articulation's root world pose — the arm rides along, giving real base
        motion + a ``base_pose`` for ``/odom`` without PhysX base joints.
        """
        x, y, yaw = self._base_pose
        dt = self._base_dt
        cos_y, sin_y = float(np.cos(yaw)), float(np.sin(yaw))
        x += (vx * cos_y - vy * sin_y) * dt
        y += (vx * sin_y + vy * cos_y) * dt
        yaw += wyaw * dt
        self._base_pose = [x, y, yaw]
        quat = np.array([float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))])  # wxyz
        self._robot.set_world_pose(np.array([x, y, self._mount_z]), quat)
        # The robot-mounted cameras ride the base.
        self._update_camera_poses()

    def _on_reset(self, rng: np.random.Generator) -> None:
        # Base returns to the origin each episode; world.reset() puts the pinned
        # arm root back at the import pose, which is the origin.
        self._base_pose = [0.0, 0.0, 0.0]
        if self._cameras:
            self._update_camera_poses()

    def _observe(self) -> dict[str, Any]:
        obs = super()._observe()
        if self._has_base:
            obs["base_pose"] = np.asarray(self._base_pose, dtype=np.float32)  # [x, y, yaw]
        clouds = self._depth_clouds()
        if clouds:
            obs["depth_points"] = clouds
        scan = self._scan_ranges()
        if scan is not None:
            obs["scan"] = scan
        return obs

    def _images(self) -> dict[str, NDArray[np.uint8]]:
        out: dict[str, NDArray[np.uint8]] = {}
        for meta in self._cam_meta:
            if meta["modality"] == "rgb":
                out[meta["key"]] = self._grab(self._cameras[meta["name"]])
        return out

    def _depth_clouds(self) -> dict[str, NDArray[np.float32]]:
        """``{sensor_name: (N, 3) base_link points}`` from each depth camera.

        Uses Isaac's ``Camera.get_pointcloud(world_frame=True)`` — Isaac owns the
        camera intrinsics + frame convention, so we never guess the
        Isaac-camera↔REP-103-optical rotation — then transforms world→base_link by
        the kinematic base pose. The openral-side ``SimSensorBridge`` publishes
        these as ``PointCloud2`` in ``base_link`` (already on /tf via /odom), so
        octomap gets a geometrically-correct cloud. Range-filtered (planar distance
        from the base) per the depth ``SensorSpec``. Empty when the manifest
        declares no depth sensor — never a fabricated cloud.
        """
        bx, by, byaw = self._base_pose
        cos_y, sin_y = float(np.cos(byaw)), float(np.sin(byaw))
        out: dict[str, NDArray[np.float32]] = {}
        for meta in self._cam_meta:
            if meta["modality"] != "depth":
                continue
            pts = self._cameras[meta["name"]].get_pointcloud(world_frame=True)
            if pts is None:
                continue
            pw = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
            if pw.size == 0:
                continue
            # world → base_link: translate by -base, rotate by -yaw about z.
            dx = pw[:, 0] - bx
            dy = pw[:, 1] - by
            xb = cos_y * dx + sin_y * dy
            yb = -sin_y * dx + cos_y * dy
            pb = np.stack([xb, yb, pw[:, 2]], axis=-1).astype(np.float32)
            rmax = float(meta.get("range_max_m") or 0.0)
            if rmax > 0.0:
                keep = (xb * xb + yb * yb) <= rmax * rmax
                pb = pb[keep]
            if pb.size:
                out[meta["name"]] = pb
        return out

    def _state(self) -> NDArray[np.float32]:
        # No task object in a bare deploy scene — proprioception is the manifest
        # joint vector (what a JOINT_POSITION policy's state head expects).
        joints = self._joint_positions()
        return joints if joints is not None else np.zeros(0, dtype=np.float32)

    def _reward_terminated(self) -> tuple[float, bool]:
        # Bare bring-up scene: no task reward / termination.
        return 0.0, False

    def _joint_positions(self) -> NDArray[np.float32] | None:
        if self._robot is None:
            return None
        return map_dof_to_manifest(
            np.asarray(self._robot.get_joint_positions(), dtype=np.float32),
            dof_index=self._dof_index,
            manifest_joints=self._manifest_joints,
            finger_dof_idx=self._finger_dof_idx,
            base_values=list(self._base_pose),  # base joints = kinematic (x, y, yaw)
            base_joints=self._base_joints,
        )

    def _joint_velocities(self) -> NDArray[np.float32] | None:
        if self._robot is None:
            return None
        # Base joint velocities are left at 0 (the kinematic base tracks pose, not
        # per-axis velocity); arm/gripper come from the articulation.
        return map_dof_to_manifest(
            np.asarray(self._robot.get_joint_velocities(), dtype=np.float32),
            dof_index=self._dof_index,
            manifest_joints=self._manifest_joints,
            finger_dof_idx=self._finger_dof_idx,
            base_values=[0.0 for _ in self._base_joints],
            base_joints=self._base_joints,
        )
