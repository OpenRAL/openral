"""``metal_pick_place`` scene rollout — MakerMods Metal cube-to-marker on the real rig.

Registers ``metal_pick_place`` as a fixed-robot scene against ``makermods_metal``.
The ``openral_sim.SimRollout`` contract:

* ``reset(seed)`` puts the arm at ``initial_joint_positions``, spawns the cube at
  the layout's start and the marker at its target (each jittered by
  ``spawn_jitter_m``) and settles;
* ``step(action)`` converts one 7-D policy action (joint1..6 + gripper, in the
  scene's ``joint_units`` / ``gripper_scale``) into the arm's 7 position
  actuators and runs one control period of physics;
* observations carry ``images.top`` (real background plate + composited robot /
  cube / marker by default), ``images.wrist``, the 7-D ``state`` in policy
  units and the task string.

Graded success, the restock contract (``IsaacPlacementSuccess``): once the cube
is released (no finger contact), on the table and below ``rest_speed_m_s`` for
``settle_s``, it is graded by how much of its footprint lies over the marker disc
(:func:`placement_grade`): ``complete`` 1.0 (>= 50 %), ``partial`` 0.5 (> 0 %),
``semi`` 0.25 (ever lifted >= 1 cm, but off the marker), else ``fail``. Any grade
but ``fail`` ends the episode with reward = its score; only ``complete`` sets
``info[task.success_key]``. Every step's ``info`` carries ``placement_grade``
(``fail`` while the cube is not judged), ``placement_score``,
``placement_overlap``, ``lifted`` and ``cube_marker_distance_m``.

``render()`` (the world video) is the top view plus a white ring around the
marker; the ring is drawn on a copy, never into a policy camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim._website_video import grade_colour
from openral_sim.backends._so_arm_units import steps_per_control_period
from openral_sim.backends.metal_pick_place._assets import (
    ARM_BODIES,
    MetalPickPlaceOptions,
    compose_metal_pick_place,
)
from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult, sim_time_ns_from_mujoco_handles

if TYPE_CHECKING:
    import mujoco
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation

__all__ = [
    "PLACEMENT_GRADE_SCORES",
    "build_metal_pick_place_scene",
    "footprint_overlap",
    "placement_grade",
    "policy_action_to_ctrl",
    "qpos_to_policy_state",
]

_ARM_DOF = 6
_ACTION_DIM = 7
_GRIPPER_QPOS = 7  # joint8 (driven finger); qpos 6 is the mimic finger joint7
_MIMIC_QPOS = 6
_FINGER_TRAVEL_M = 0.05  # joint8 range [0, 0.05] m
_LIFT_M = 0.01  # "lifted": cube centre ever this far above its resting height
_SUCCESS_OVERLAP = 0.5
_RING_GAP_M = 0.008  # world-video ring radius beyond the marker's
_RING_RGB = (255, 255, 255)

#: Grade -> score, best first; ``complete`` is the success grade (restock's names).
PLACEMENT_GRADE_SCORES: dict[str, float] = {
    "complete": 1.0,
    "partial": 0.5,
    "semi": 0.25,
    "fail": 0.0,
}


def footprint_overlap(
    cube_xy: NDArray[np.float64],
    rotation: NDArray[np.float64],
    edge_m: float,
    marker_xy: NDArray[np.float64],
    marker_diameter_m: float,
    n: int = 15,
) -> float:
    """Fraction of the cube's xy footprint over the marker disc.

    Samples an ``n^3`` grid through the cube's volume, rotates it by the cube's
    world ``rotation`` (3x3) and counts the samples whose xy falls inside the disc:
    for a cube resting on any face that is its footprint square at its yaw.

    Example:
        >>> footprint_overlap(np.zeros(2), np.eye(3), 0.025, np.zeros(2), 0.039)
        1.0
        >>> footprint_overlap(np.array([0.1, 0.0]), np.eye(3), 0.025, np.zeros(2), 0.039)
        0.0
    """
    g = (np.arange(n) + 0.5) / n - 0.5
    pts = np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3) * edge_m
    xy = pts @ np.asarray(rotation)[:2].T + np.asarray(cube_xy)[:2]
    inside = np.linalg.norm(xy - np.asarray(marker_xy)[:2], axis=1) <= marker_diameter_m / 2.0
    return float(inside.mean())


def placement_grade(overlap: float, lifted: bool) -> str:
    """Grade a released, resting cube (``PLACEMENT_GRADE_SCORES`` names).

    Example:
        >>> [placement_grade(o, lifted) for o, lifted in ((0.5, False), (0.2, True), (0.0, True))]
        ['complete', 'partial', 'semi']
        >>> placement_grade(0.0, False)
        'fail'
    """
    if overlap >= _SUCCESS_OVERLAP:
        return "complete"
    if overlap > 0.0:
        return "partial"
    return "semi" if lifted else "fail"


def policy_action_to_ctrl(
    action: NDArray[np.float64],
    *,
    joint_units: str,
    gripper_scale: float,
    gripper_offset_m: float = 0.0,
) -> NDArray[np.float64]:
    """7-D policy action -> the arm's 7 actuator targets (rad x6, joint8 metres).

    joint8 = gripper / gripper_scale * 0.05 - gripper_offset_m, clipped to [0, 0.05]
    (the offset is the reading of the fully closed real gripper).

    Example:
        >>> policy_action_to_ctrl(
        ...     np.array([0, 0, 0, 0, 0, 90.0, 100.0]), joint_units="degrees", gripper_scale=100.0
        ... ).round(4).tolist()
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.5708, 0.05]
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape != (_ACTION_DIM,):
        raise ROSConfigError(f"metal_pick_place expects a 7-D action; got shape {a.shape}.")
    ctrl = a.copy()
    if joint_units == "degrees":
        ctrl[:_ARM_DOF] = np.radians(a[:_ARM_DOF])
    ctrl[_ARM_DOF] = np.clip(
        a[_ARM_DOF] / gripper_scale * _FINGER_TRAVEL_M - gripper_offset_m, 0.0, _FINGER_TRAVEL_M
    )
    return ctrl


def qpos_to_policy_state(
    qpos: NDArray[np.float64],
    *,
    joint_units: str,
    gripper_scale: float,
    gripper_offset_m: float = 0.0,
) -> NDArray[np.float64]:
    """Arm qpos (8 slots: 6 joints, mimic, driven finger) -> 7-D policy state.

    Inverse of :func:`policy_action_to_ctrl` on the reachable range.

    Example:
        >>> q = np.array([0, 0, 0, 0, 0, np.pi / 2, -0.025, 0.025])
        >>> qpos_to_policy_state(q, joint_units="degrees", gripper_scale=100.0).round(3).tolist()
        [0.0, 0.0, 0.0, 0.0, 0.0, 90.0, 50.0]
    """
    q = np.asarray(qpos, dtype=np.float64)
    state = np.empty(_ACTION_DIM, dtype=np.float64)
    arm = q[:_ARM_DOF]
    state[:_ARM_DOF] = np.degrees(arm) if joint_units == "degrees" else arm
    state[_ARM_DOF] = (q[_GRIPPER_QPOS] + gripper_offset_m) / _FINGER_TRAVEL_M * gripper_scale
    return state


def _options(scene: SceneSpec) -> MetalPickPlaceOptions:
    opts = SCENES.validate_options(scene.id, scene.backend_options)
    if not isinstance(opts, MetalPickPlaceOptions):
        raise ROSConfigError(f"scene {scene.id!r} does not resolve to metal_pick_place.")
    return opts


def _load_plate(opts: MetalPickPlaceOptions, width: int, height: int) -> NDArray[np.uint8]:
    import imageio.v3 as iio

    path = opts.asset_path(opts.top_plate)
    try:
        plate = np.asarray(iio.imread(path), dtype=np.uint8)[..., :3]
    except OSError as exc:
        raise ROSConfigError(f"metal_pick_place: cannot read top plate {path}: {exc}") from exc
    if plate.shape[:2] != (height, width):  # nearest-neighbour resize
        ys = np.arange(height) * plate.shape[0] // height
        xs = np.arange(width) * plate.shape[1] // width
        plate = plate[ys][:, xs]
    return np.ascontiguousarray(plate)


@dataclass
class _MetalPickPlaceRollout:
    """Rollout for the metal_pick_place scene."""

    scene: SceneSpec
    task: TaskSpec
    options: MetalPickPlaceOptions
    _model: mujoco.MjModel
    _data: mujoco.MjData
    _arm_model: mujoco.MjModel  # bare arm, for the leader's forward kinematics
    _arm_data: mujoco.MjData
    _joint_ranges: NDArray[np.float64]  # (7, 2): joint1..6 + joint8
    _cube_qpos: int
    _cube_qvel: int
    _cube_geom: int
    _cube_body: int
    _top_cam: int
    _finger_geoms: frozenset[int]
    _marker_mocap: int
    _leader_mocap: list[tuple[int, int]]  # (mocap id, arm-model body id)
    _composite_geoms: NDArray[np.bool_]  # geom id -> drawn from MuJoCo on the top view
    _plate: NDArray[np.uint8] | None
    _instruction: str
    _max_steps: int
    _width: int
    _height: int
    _steps_per_action: int
    _rng: np.random.Generator
    _settle_steps: int = 1
    _step_count: int = 0
    _ring_rgb: tuple[int, int, int] = _RING_RGB  # white, then the grade colour (render())
    _rest_count: int = 0
    _lifted: bool = False
    _target_xy: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    _renderer: Any = None
    _last_top: NDArray[np.uint8] | None = None

    # ---------------------------------------------------------------- SimRollout

    def reset(self, seed: int | None = None) -> Observation:
        import mujoco

        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        opts = self.options
        mujoco.mj_resetData(self._model, self._data)
        init = self._to_ctrl(np.asarray(opts.initial_joint_positions, dtype=np.float64))
        self._data.qpos[:_ARM_DOF] = init[:_ARM_DOF]
        self._data.qpos[_GRIPPER_QPOS] = init[_ARM_DOF]
        self._data.qpos[_MIMIC_QPOS] = -init[_ARM_DOF]
        self._data.ctrl[:_ACTION_DIM] = init

        (cx, cy), (tx, ty) = opts.layouts[opts.layout]
        j = opts.spawn_jitter_m
        cube_xy = np.array([cx, cy]) + self._rng.uniform(-j, j, 2)
        self._target_xy = np.array([tx, ty]) + self._rng.uniform(-j, j, 2)
        yaw = np.radians(
            opts.cube_yaw_deg
            + self._rng.uniform(-opts.cube_yaw_jitter_deg, opts.cube_yaw_jitter_deg)
        )
        q = self._cube_qpos
        self._data.qpos[q : q + 3] = (*cube_xy, opts.table_top_z + opts.cube_edge_m / 2.0)
        self._data.qpos[q + 3 : q + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        self._data.mocap_pos[self._marker_mocap, :2] = self._target_xy
        self._pose_leader(init)

        mujoco.mj_forward(self._model, self._data)
        for _ in range(opts.settle_steps):
            mujoco.mj_step(self._model, self._data)
        self._step_count = 0
        self._ring_rgb = _RING_RGB
        self._rest_count = 0
        self._lifted = False
        return self._observation()

    def step(self, action: NDArray[np.float32]) -> StepResult:
        import mujoco

        ctrl = self._to_ctrl(np.asarray(action, dtype=np.float64))
        self._data.ctrl[:_ACTION_DIM] = ctrl
        if self.options.leader_arm == "mirror_action":
            self._pose_leader(ctrl)
        for _ in range(self._steps_per_action):
            mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        info = self._grade()
        grade = info["placement_grade"]
        if self.task.success_key is not None:
            info[self.task.success_key] = grade == "complete"
        terminated = grade != "fail"
        truncated = self._step_count >= self._max_steps
        if terminated or truncated:  # the ring turns with the video border (same colours)
            self._ring_rgb = grade_colour(float(info["placement_score"]))
        return StepResult(
            observation=self._observation(),
            reward=info["placement_score"] if terminated else 0.0,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def set_task(self, instruction: str) -> dict[str, Any]:
        """Grade ``instruction`` (``SimAttachedHAL.set_active_task``; Isaac's reply shape).

        The scene has one task, so any non-empty prompt is graded with
        ``PLACEMENT_GRADE_SCORES``.
        """
        graded = bool(instruction)
        return {
            "graded": graded,
            "grade_scores": dict(PLACEMENT_GRADE_SCORES) if graded else {},
            "prompts": [self._instruction],
        }

    def render(self) -> NDArray[np.uint8] | None:
        """The world video frame: the last top view plus a ring around the marker.

        The ring is white while the episode runs and takes the grade's colour
        (``grade_colour``: green / yellow / orange, red for a fail at the step limit)
        on the frame the episode ends, the same frame the video border turns.
        """
        if self._last_top is None:
            return None
        from PIL import Image, ImageDraw

        img = Image.fromarray(self._last_top)
        ImageDraw.Draw(img).line(
            self._ring_pixels(), fill=self._ring_rgb, width=max(2, self._width // 320)
        )
        return np.asarray(img, dtype=np.uint8)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def mujoco_handles(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Expose the model + data to ``openral sim run --view``."""
        return self._model, self._data

    def sim_time_ns(self) -> int | None:
        """Elapsed MuJoCo sim time in ns (rewinds on ``reset``)."""
        return sim_time_ns_from_mujoco_handles(self.mujoco_handles())

    @property
    def action_dim(self) -> int:
        """Flat action width ``step`` accepts: joint1..6 + gripper."""
        return _ACTION_DIM

    # ------------------------------------------------------------------ helpers

    def cube_to_marker_m(self) -> float:
        """Horizontal distance from the cube centre to the marker centre, metres."""
        q = self._cube_qpos
        return float(np.linalg.norm(self._data.qpos[q : q + 2] - self._target_xy))

    def _grade(self) -> dict[str, Any]:
        """This step's grading ``info`` (see the module docstring)."""
        o, d = self.options, self._data
        q, v = self._cube_qpos, self._cube_qvel
        rest_z = o.table_top_z + o.cube_edge_m / 2.0
        self._lifted |= bool(d.qpos[q + 2] > rest_z + _LIFT_M)
        overlap = footprint_overlap(
            d.qpos[q : q + 2],
            d.xmat[self._cube_body].reshape(3, 3),
            o.cube_edge_m,
            self._target_xy,
            o.marker_diameter_m,
        )
        at_rest = (
            d.qpos[q + 2] < o.table_top_z + o.cube_edge_m
            and np.linalg.norm(d.qvel[v : v + 3]) < o.rest_speed_m_s
            and not self._touches_fingers()
        )
        self._rest_count = self._rest_count + 1 if at_rest else 0
        settled = self._rest_count >= self._settle_steps
        grade = placement_grade(overlap, self._lifted) if settled else "fail"
        return {
            "placement_grade": grade,
            "placement_score": PLACEMENT_GRADE_SCORES[grade],
            "placement_overlap": overlap,
            "lifted": self._lifted,
            "cube_marker_distance_m": self.cube_to_marker_m(),
        }

    def _touches_fingers(self) -> bool:
        for c in self._data.contact[: self._data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if self._cube_geom in pair and pair & self._finger_geoms:
                return True
        return False

    def _ring_pixels(self) -> list[tuple[float, float]]:
        """The marker ring (table plane) projected into the top camera, pixel polyline."""
        o, d = self.options, self._data
        r = o.marker_diameter_m / 2.0 + _RING_GAP_M
        t = np.linspace(0.0, 2.0 * np.pi, 73)
        world = np.stack(
            [
                self._target_xy[0] + r * np.cos(t),
                self._target_xy[1] + r * np.sin(t),
                np.full_like(t, o.table_top_z),
            ],
            -1,
        )
        # MuJoCo camera frame: x right, y up, looking down -z.
        cam = (world - d.cam_xpos[self._top_cam]) @ d.cam_xmat[self._top_cam].reshape(3, 3)
        sx = self._width / o.top_camera_resolution[0]
        sy = self._height / o.top_camera_resolution[1]
        fx, fy = o.top_camera_focal_px * sx, o.top_camera_focal_px * sy
        cx, cy = o.top_camera_principal_px[0] * sx, o.top_camera_principal_px[1] * sy
        u = cx + fx * cam[:, 0] / -cam[:, 2]
        v = cy - fy * cam[:, 1] / -cam[:, 2]
        return list(zip(u.tolist(), v.tolist(), strict=True))

    def _to_ctrl(self, action: NDArray[np.float64]) -> NDArray[np.float64]:
        o = self.options
        ctrl = policy_action_to_ctrl(
            action,
            joint_units=o.joint_units,
            gripper_scale=o.gripper_scale,
            gripper_offset_m=o.gripper_offset_m,
        )
        return np.clip(ctrl, self._joint_ranges[:, 0], self._joint_ranges[:, 1])

    def _pose_leader(self, ctrl: NDArray[np.float64]) -> None:
        """Put the visual leader arm at ``ctrl`` (the teleop leader IS the action)."""
        import mujoco

        if not self._leader_mocap:
            return
        d = self._arm_data
        d.qpos[:_ARM_DOF] = ctrl[:_ARM_DOF]
        d.qpos[_GRIPPER_QPOS] = ctrl[_ARM_DOF]
        d.qpos[_MIMIC_QPOS] = -ctrl[_ARM_DOF]
        mujoco.mj_kinematics(self._arm_model, d)
        yaw = np.radians(self.options.leader_base_yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        q_base = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        base = np.asarray(self.options.leader_base_xyz)
        for mocap, body in self._leader_mocap:
            self._data.mocap_pos[mocap] = base + rot @ d.xpos[body]
            mujoco.mju_mulQuat(self._data.mocap_quat[mocap], q_base, d.xquat[body])

    def _observation(self) -> Observation:
        top = self._render_top()
        self._last_top = top
        q = np.asarray(self._data.qpos[:8], dtype=np.float64)
        state = qpos_to_policy_state(
            q,
            joint_units=self.options.joint_units,
            gripper_scale=self.options.gripper_scale,
            gripper_offset_m=self.options.gripper_offset_m,
        )
        return {
            "images": {"top": top, "wrist": self._render_rgb("wrist")},
            "state": state.astype(np.float32),
            "task": self._instruction,
        }

    def _ensure_renderer(self) -> Any:
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=self._height, width=self._width)
        return self._renderer

    def _render_rgb(self, camera: str) -> NDArray[np.uint8]:
        r = self._ensure_renderer()
        r.update_scene(self._data, camera=camera)
        return np.asarray(r.render(), dtype=np.uint8).copy()

    def _render_top(self) -> NDArray[np.uint8]:
        import mujoco

        m, o = self._model, self.options
        saved = (
            m.vis.headlight.ambient.copy(),
            m.vis.headlight.diffuse.copy(),
            m.light_diffuse.copy(),
            m.light_castshadow.copy(),
        )
        m.vis.headlight.ambient[:] = o.top_headlight_ambient
        m.vis.headlight.diffuse[:] = o.top_headlight_diffuse
        m.light_diffuse[:] = saved[2] * o.top_light_gain
        m.light_castshadow[:] = False
        r = self._ensure_renderer()
        try:
            r.update_scene(self._data, camera="top")
            rgb = np.asarray(r.render(), dtype=np.uint8).copy()
            if self._plate is None:
                return rgb
            r.enable_segmentation_rendering()
            try:
                r.update_scene(self._data, camera="top")
                seg = np.asarray(r.render())
            finally:
                r.disable_segmentation_rendering()
        finally:
            (
                m.vis.headlight.ambient[:],
                m.vis.headlight.diffuse[:],
                m.light_diffuse[:],
                m.light_castshadow[:],
            ) = saved

        ids, types = seg[..., 0], seg[..., 1]
        is_geom = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        mask = np.zeros(ids.shape, dtype=np.float32)
        mask[is_geom] = self._composite_geoms[ids[is_geom]]
        mask = _feather(mask, self.options.top_mask_feather_px)[..., None]
        out = mask * rgb.astype(np.float32) + (1.0 - mask) * self._plate.astype(np.float32)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _feather(mask: NDArray[np.float32], radius: int) -> NDArray[np.float32]:
    """Box-blur a hard mask by ``radius`` px (anti-aliased composite edge)."""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    padded = np.pad(mask, radius, mode="edge")
    c = padded.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = mask.shape
    s = c[k : k + h, k : k + w] - c[:h, k : k + w] - c[k : k + h, :w] + c[:h, :w]
    out: NDArray[np.float32] = (s / (k * k)).astype(np.float32)
    return out


@SCENES.register(
    "metal_pick_place",
    fixed_robot="makermods_metal",
    options_model=MetalPickPlaceOptions,
    sim_clock=True,
    converts_policy_units=True,  # backend_options.joint_units + gripper_scale
    sequential_init=True,  # makermods: asset resolve imports openral_hal -> transformers
)
def build_metal_pick_place_scene(env_cfg: SimEnvironment) -> _MetalPickPlaceRollout:
    """Build the metal_pick_place rollout from a composed ``SimEnvironment``."""
    import mujoco

    from openral_sim.factory import make_robot  # reason: defer to avoid import cycle

    opts = _options(env_cfg.scene)
    width = int(env_cfg.scene.observation_width or 640)
    height = int(env_cfg.scene.observation_height or 480)
    description = make_robot(env_cfg)
    if description is None:
        raise ROSConfigError("metal_pick_place: no robot manifest for makermods_metal.")
    model, arm_path = compose_metal_pick_place(description, opts, width=width, height=height)
    data = mujoco.MjData(model)
    arm_model = mujoco.MjModel.from_xml_path(arm_path)

    def _id(kind: Any, name: str) -> int:
        idx = int(mujoco.mj_name2id(model, kind, name))
        if idx < 0:
            raise ROSConfigError(f"metal_pick_place: {name!r} missing in the compiled model.")
        return idx

    joint_ids = [_id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{k}") for k in (1, 2, 3, 4, 5, 6, 8)]
    if [int(model.jnt_qposadr[j]) for j in joint_ids] != [0, 1, 2, 3, 4, 5, _GRIPPER_QPOS]:
        raise ROSConfigError("metal_pick_place: arm qpos layout differs from the manifest.")
    cube_joint = _id(mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    body = {n: _id(mujoco.mjtObj.mjOBJ_BODY, n) for n in ARM_BODIES}
    fingers = frozenset(
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in (body["link7"], body["link8"])
    )
    composite = np.zeros(model.ngeom, dtype=bool)
    arm_body_ids = set(body.values())
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        composite[g] = (
            int(model.geom_bodyid[g]) in arm_body_ids
            or name in ("cube", "marker")
            or name.startswith("leader_")
        )
    leader: list[tuple[int, int]] = []
    if opts.leader_arm != "hidden":
        for n in ARM_BODIES:
            mocap = int(model.body_mocapid[_id(mujoco.mjtObj.mjOBJ_BODY, f"leader_{n}")])
            leader.append((mocap, int(mujoco.mj_name2id(arm_model, mujoco.mjtObj.mjOBJ_BODY, n))))

    return _MetalPickPlaceRollout(
        scene=env_cfg.scene,
        task=env_cfg.task,
        options=opts,
        _model=model,
        _data=data,
        _arm_model=arm_model,
        _arm_data=mujoco.MjData(arm_model),
        _joint_ranges=np.asarray([model.jnt_range[j] for j in joint_ids], dtype=np.float64),
        _cube_qpos=int(model.jnt_qposadr[cube_joint]),
        _cube_qvel=int(model.jnt_dofadr[cube_joint]),
        _cube_geom=_id(mujoco.mjtObj.mjOBJ_GEOM, "cube"),
        _cube_body=_id(mujoco.mjtObj.mjOBJ_BODY, "cube"),
        _top_cam=_id(mujoco.mjtObj.mjOBJ_CAMERA, "top"),
        _finger_geoms=fingers,
        _marker_mocap=int(model.body_mocapid[_id(mujoco.mjtObj.mjOBJ_BODY, "marker")]),
        _leader_mocap=leader,
        _composite_geoms=composite,
        _plate=_load_plate(opts, width, height) if opts.top_background == "plate" else None,
        _instruction=env_cfg.task.instruction
        or "Pick up the cube and place it at the marked location",
        _max_steps=int(env_cfg.task.max_steps or 900),
        _width=width,
        _height=height,
        _steps_per_action=steps_per_control_period(
            float(model.opt.timestep), opts.control_hz, scene="metal_pick_place"
        ),
        _rng=np.random.default_rng(env_cfg.seed or 0),
        _settle_steps=max(1, round(opts.settle_s * opts.control_hz)),
    )
