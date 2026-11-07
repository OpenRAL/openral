"""``metal_pen_holder`` scene rollout — the MakerMods Metal Arm Station pen-holder task.

Registers ``metal_pen_holder`` as a fixed-robot scene against ``makermods_metal_station``.
The ``openral_sim.SimRollout`` contract:

* ``reset(seed)`` puts both arms at ``initial_joint_positions``, the cup on the shelf at
  ``holder_start`` and the pen on the desk at ``pen_start``, and settles;
* ``step(action)`` converts one 14-D policy action (per arm joint1..6 + gripper, in the
  scene's ``joint_units`` / ``gripper_scale``) into the 14 position actuators and runs
  one control period of physics;
* observations carry ``images.top`` (real background plate + composited robot / cup /
  pen by default), ``images.wrist_left``, ``images.wrist_right``, the 14-D ``state`` in
  policy units and the task string.

Graded success, the restock contract shared with ``metal_pick_place``: once the pen is
released (no finger contact on either arm) and below ``rest_speed_m_s`` for ``settle_s``,
it is graded against the cup's interior (the volume inside the leaning walls, from its
floor to its rim, in the cup's own frame) by :func:`pen_holder_grade`: ``complete`` 1.0
(the pen's centre of mass inside), ``partial`` 0.5 (some of the pen inside below the rim,
its centre outside — leaning on the rim), ``semi`` 0.25 (lifted >= 1 cm, but resting fully
outside), else ``fail``. Any grade but ``fail`` ends the episode with reward = its score;
only ``complete`` sets ``info[task.success_key]``. The cup going back to the shelf is not
graded. Every step's ``info`` carries ``placement_grade``, ``placement_score``,
``pen_inside_fraction``, ``lifted`` and ``pen_to_holder_m``.

``render()`` (the world video) is the top view plus an outline of the cup's rim; it is
white while running and takes the grade's colour on the frame the episode ends. It is
drawn on a copy, never into a policy camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim._website_video import grade_colour
from openral_sim.backends._so_arm_units import steps_per_control_period
from openral_sim.backends.metal_pen_holder._assets import (
    ARMS,
    MetalPenHolderOptions,
    compose_metal_pen_holder,
    holder_inner_half_width,
)
from openral_sim.backends.metal_pick_place._assets import ARM_BODIES
from openral_sim.backends.metal_pick_place.env import (
    PLACEMENT_GRADE_SCORES,
    _feather,
    _load_plate,
    policy_action_to_ctrl,
    qpos_to_policy_state,
)
from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult, sim_time_ns_from_mujoco_handles

if TYPE_CHECKING:
    import mujoco
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation

__all__ = [
    "build_metal_pen_holder_scene",
    "pen_holder_grade",
    "pen_points_in_holder",
]

_ARM_POLICY = 7  # per arm: joint1..6 + gripper
_ARM_QPOS = 8  # per arm: joint1..6, mimic joint7, driven joint8
_ACTION_DIM = 2 * _ARM_POLICY
_LIFT_M = 0.01
_PEN_SAMPLES = 15  # points along the pen axis for the inside test
_RING_RGB = (255, 255, 255)
_CAMERAS = ("top", "wrist_left", "wrist_right")
_DEFAULT_GEOM_RGBA = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)  # MuJoCo's: use the material


def pen_points_in_holder(
    pen_pos: NDArray[np.float64],
    pen_rot: NDArray[np.float64],
    holder_pos: NDArray[np.float64],
    holder_rot: NDArray[np.float64],
    opts: MetalPenHolderOptions,
) -> tuple[bool, float]:
    """``(centre inside, fraction of the pen axis inside)`` of the cup's interior.

    The interior, in the cup's frame (origin at its bottom centre, z up its axis): above
    the floor, below the rim, within the inner half-width the leaning walls leave at that
    height (``holder_inner_half_width``). The pen is sampled at ``_PEN_SAMPLES`` points
    along its axis (its body x).

    Example:
        >>> o = MetalPenHolderOptions()
        >>> standing = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # x up
        >>> pen_points_in_holder(np.array([0, 0, 0.08]), standing, np.zeros(3), np.eye(3), o)
        (True, 0.6)
        >>> pen_points_in_holder(np.array([0.3, 0, 0.01]), np.eye(3), np.zeros(3), np.eye(3), o)
        (False, 0.0)
    """
    s = (np.arange(_PEN_SAMPLES) + 0.5) / _PEN_SAMPLES - 0.5
    pts = pen_pos + np.outer(s * opts.pen_length_m, np.asarray(pen_rot)[:, 0])
    local = (np.vstack([pts, pen_pos]) - holder_pos) @ np.asarray(holder_rot)
    half = np.array([holder_inner_half_width(opts, float(z)) for z in local[:, 2]])
    inside = (
        (local[:, 2] > opts.holder_floor_m)
        & (local[:, 2] < opts.holder_height_m)
        & (np.abs(local[:, 0]) < half)
        & (np.abs(local[:, 1]) < half)
    )
    return bool(inside[-1]), float(inside[:-1].mean())


def pen_holder_grade(com_inside: bool, inside_fraction: float, lifted: bool) -> str:
    """Grade a released, resting pen (``PLACEMENT_GRADE_SCORES`` names).

    Example:
        >>> [
        ...     pen_holder_grade(*a)
        ...     for a in ((True, 0.6, True), (False, 0.2, True), (False, 0.0, True))
        ... ]
        ['complete', 'partial', 'semi']
        >>> pen_holder_grade(False, 0.0, False)
        'fail'
    """
    if com_inside:
        return "complete"
    if inside_fraction > 0.0:
        return "partial"
    return "semi" if lifted else "fail"


def _options(scene: SceneSpec) -> MetalPenHolderOptions:
    opts = SCENES.validate_options(scene.id, scene.backend_options)
    if not isinstance(opts, MetalPenHolderOptions):
        raise ROSConfigError(f"scene {scene.id!r} does not resolve to metal_pen_holder.")
    return opts


@dataclass
class _MetalPenHolderRollout:
    """Rollout for the metal_pen_holder scene."""

    scene: SceneSpec
    task: TaskSpec
    options: MetalPenHolderOptions
    _model: mujoco.MjModel
    _data: mujoco.MjData
    _joint_ranges: NDArray[np.float64]  # (14, 2): per arm joint1..6 + joint8
    _pen_qpos: int
    _pen_qvel: int
    _pen_geom: int
    _pen_body: int
    _holder_qpos: int
    _holder_body: int
    _top_cam: int
    _finger_geoms: frozenset[int]
    _composite_geoms: NDArray[np.bool_]
    _plate: NDArray[np.uint8] | None
    _top_k: tuple[float, float, float, float]  # fx, fy, cx, cy at the render size
    _instruction: str
    _max_steps: int
    _width: int
    _height: int
    _steps_per_action: int
    _settle_steps: int = 1
    _step_count: int = 0
    _ring_rgb: tuple[int, int, int] = _RING_RGB
    _rest_count: int = 0
    _lifted: bool = False
    _renderer: Any = None
    _last_top: NDArray[np.uint8] | None = None

    # ---------------------------------------------------------------- SimRollout

    def reset(self, seed: int | None = None) -> Observation:
        import mujoco

        opts = self.options
        mujoco.mj_resetData(self._model, self._data)
        init = self._to_ctrl(np.asarray(opts.initial_joint_positions, dtype=np.float64))
        for k in range(2):
            q, c = k * _ARM_QPOS, k * _ARM_POLICY
            self._data.qpos[q : q + 6] = init[c : c + 6]
            self._data.qpos[q + 7] = init[c + 6]
            self._data.qpos[q + 6] = -init[c + 6]
        self._data.ctrl[:_ACTION_DIM] = init
        rng = np.random.default_rng(seed)
        for qadr, (x, y, yaw) in (
            (self._holder_qpos, opts.holder_start),
            (self._pen_qpos, opts.pen_start),
        ):
            j, jy = opts.spawn_jitter_m, opts.yaw_jitter_deg
            dx, dy = rng.uniform(-j, j, 2) if j else (0.0, 0.0)
            a = np.radians(yaw + (rng.uniform(-jy, jy) if jy else 0.0))
            self._data.qpos[qadr : qadr + 2] = (x + dx, y + dy)
            self._data.qpos[qadr + 3 : qadr + 7] = (np.cos(a / 2), 0.0, 0.0, np.sin(a / 2))
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

        self._data.ctrl[:_ACTION_DIM] = self._to_ctrl(np.asarray(action, dtype=np.float64))
        for _ in range(self._steps_per_action):
            mujoco.mj_step(self._model, self._data)
        self._step_count += 1
        info = self._grade()
        grade = info["placement_grade"]
        if self.task.success_key is not None:
            info[self.task.success_key] = grade == "complete"
        terminated = grade != "fail"
        truncated = self._step_count >= self._max_steps
        if terminated or truncated:  # the outline turns with the video border
            self._ring_rgb = grade_colour(float(info["placement_score"]))
        return StepResult(
            observation=self._observation(),
            reward=info["placement_score"] if terminated else 0.0,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def set_task(self, instruction: str) -> dict[str, Any]:
        """Grade ``instruction`` (one task: any non-empty prompt is graded)."""
        graded = bool(instruction)
        return {
            "graded": graded,
            "grade_scores": dict(PLACEMENT_GRADE_SCORES) if graded else {},
            "prompts": [self._instruction],
        }

    def render(self) -> NDArray[np.uint8] | None:
        """The world video frame: the last top view plus the cup's rim outline."""
        if self._last_top is None:
            return None
        from PIL import Image, ImageDraw

        img = Image.fromarray(self._last_top)
        ImageDraw.Draw(img).line(
            self._rim_pixels(), fill=self._ring_rgb, width=max(2, self._width // 320)
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
        """Flat action width ``step`` accepts: per arm joint1..6 + gripper."""
        return _ACTION_DIM

    # ------------------------------------------------------------------ helpers

    def pen_to_holder_m(self) -> float:
        """Distance from the pen's centre to the cup's interior centre (half-way up), metres."""
        d = self._data
        mid = d.xpos[self._holder_body] + d.xmat[self._holder_body].reshape(3, 3)[:, 2] * (
            self.options.holder_height_m / 2.0
        )
        return float(np.linalg.norm(d.xpos[self._pen_body] - mid))

    def _grade(self) -> dict[str, Any]:
        o, d = self.options, self._data
        q, v = self._pen_qpos, self._pen_qvel
        rest_z = o.table_top_z + o.pen_diameter_m / 2.0
        self._lifted |= bool(d.qpos[q + 2] > rest_z + 0.01)
        com_in, frac = pen_points_in_holder(
            d.xpos[self._pen_body],
            d.xmat[self._pen_body].reshape(3, 3),
            d.xpos[self._holder_body],
            d.xmat[self._holder_body].reshape(3, 3),
            o,
        )
        at_rest = (
            np.linalg.norm(d.qvel[v : v + 3]) < o.rest_speed_m_s and not self._touches_fingers()
        )
        self._rest_count = self._rest_count + 1 if at_rest else 0
        settled = self._rest_count >= self._settle_steps
        grade = pen_holder_grade(com_in, frac, self._lifted) if settled else "fail"
        return {
            "placement_grade": grade,
            "placement_score": PLACEMENT_GRADE_SCORES[grade],
            "pen_inside_fraction": frac,
            "lifted": self._lifted,
            "pen_to_holder_m": self.pen_to_holder_m(),
        }

    def _touches_fingers(self) -> bool:
        for c in self._data.contact[: self._data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if self._pen_geom in pair and pair & self._finger_geoms:
                return True
        return False

    def _rim_pixels(self) -> list[tuple[float, float]]:
        """The cup's rim square projected into the top camera, pixel polyline."""
        o, d = self.options, self._data
        h = o.holder_top_width_m / 2.0 + 0.006
        corners = np.array([[h, h], [-h, h], [-h, -h], [h, -h], [h, h]])
        local = np.column_stack([corners, np.full(5, o.holder_height_m)])
        world = d.xpos[self._holder_body] + local @ d.xmat[self._holder_body].reshape(3, 3).T
        # MuJoCo camera frame: x right, y up, looking down -z.
        cam = (world - d.cam_xpos[self._top_cam]) @ d.cam_xmat[self._top_cam].reshape(3, 3)
        fx, fy, cx, cy = self._top_k
        u = cx + fx * cam[:, 0] / -cam[:, 2]
        v = cy - fy * cam[:, 1] / -cam[:, 2]
        return list(zip(u.tolist(), v.tolist(), strict=True))

    def _to_ctrl(self, action: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (_ACTION_DIM,):
            raise ROSConfigError(f"metal_pen_holder expects a 14-D action; got shape {a.shape}.")
        o = self.options
        ctrl = np.concatenate(
            [
                policy_action_to_ctrl(
                    a[k * _ARM_POLICY : (k + 1) * _ARM_POLICY],
                    joint_units=o.joint_units,
                    gripper_scale=o.gripper_scale,
                    gripper_offset_m=o.gripper_offset_m,
                )
                for k in range(2)
            ]
        )
        return np.clip(ctrl, self._joint_ranges[:, 0], self._joint_ranges[:, 1])

    def _observation(self) -> Observation:
        top = self._render_top()
        self._last_top = top
        o = self.options
        state = np.concatenate(
            [
                qpos_to_policy_state(
                    np.asarray(self._data.qpos[k * _ARM_QPOS : (k + 1) * _ARM_QPOS]),
                    joint_units=o.joint_units,
                    gripper_scale=o.gripper_scale,
                    gripper_offset_m=o.gripper_offset_m,
                )
                for k in range(2)
            ]
        )
        return {
            "images": {
                "top": top,
                "wrist_left": self._render_rgb("wrist_left"),
                "wrist_right": self._render_rgb("wrist_right"),
            },
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
            m.mat_rgba.copy(),
            m.geom_rgba.copy(),
        )
        for name, gain in o.top_material_gain.items():
            mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if mat >= 0:  # a geom's own (non-default) rgba overrides its material's
                m.mat_rgba[mat, :3] *= gain
                own = (m.geom_matid == mat) & np.any(m.geom_rgba != _DEFAULT_GEOM_RGBA, axis=1)
                m.geom_rgba[own, :3] *= gain
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
                m.mat_rgba[:],
                m.geom_rgba[:],
            ) = saved
        ids, types = seg[..., 0], seg[..., 1]
        is_geom = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        mask = np.zeros(ids.shape, dtype=np.float32)
        mask[is_geom] = self._composite_geoms[ids[is_geom]]
        mask = _feather(mask, o.top_mask_feather_px)[..., None]
        out = mask * rgb.astype(np.float32) + (1.0 - mask) * self._plate.astype(np.float32)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)


@SCENES.register(
    "metal_pen_holder",
    fixed_robot="makermods_metal_station",
    options_model=MetalPenHolderOptions,
    sim_clock=True,
    converts_policy_units=True,  # backend_options.joint_units + gripper_scale
    sequential_init=True,  # makermods: asset resolve imports openral_hal -> transformers
    gripper_convention="normalized_open_unit",
)
def build_metal_pen_holder_scene(env_cfg: SimEnvironment) -> _MetalPenHolderRollout:
    """Build the metal_pen_holder rollout from a composed ``SimEnvironment``."""
    import mujoco

    from openral_sim.factory import make_robot  # reason: defer to avoid import cycle

    opts = _options(env_cfg.scene)
    width = int(env_cfg.scene.observation_width or 640)
    height = int(env_cfg.scene.observation_height or 480)
    description = make_robot(env_cfg)
    if description is None:
        raise ROSConfigError("metal_pen_holder: no robot manifest for makermods_metal_station.")
    model = compose_metal_pen_holder(description, opts, width=width, height=height)

    def _id(kind: Any, name: str) -> int:
        idx = int(mujoco.mj_name2id(model, kind, name))
        if idx < 0:
            raise ROSConfigError(f"metal_pen_holder: {name!r} missing in the compiled model.")
        return idx

    joints = [f"{p}joint{k}" for p in ARMS for k in (1, 2, 3, 4, 5, 6, 8)]
    joint_ids = [_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in joints]
    want = [*range(6), 7, *range(8, 14), 15]
    if [int(model.jnt_qposadr[j]) for j in joint_ids] != want:
        raise ROSConfigError("metal_pen_holder: station qpos layout differs from the manifest.")
    arm_bodies = {_id(mujoco.mjtObj.mjOBJ_BODY, p + n) for p in ARMS for n in ARM_BODIES}
    finger_bodies = {_id(mujoco.mjtObj.mjOBJ_BODY, p + n) for p in ARMS for n in ("link7", "link8")}
    fingers = frozenset(g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in finger_bodies)
    holder_body = _id(mujoco.mjtObj.mjOBJ_BODY, "pen_holder")
    pen_body = _id(mujoco.mjtObj.mjOBJ_BODY, "pen")
    composite = np.array(
        [
            int(model.geom_bodyid[g]) in arm_bodies | {holder_body, pen_body}
            for g in range(model.ngeom)
        ]
    )
    top = next((s for s in description.sensors if s.name == "top"), None)
    if top is None or top.intrinsics is None:
        raise ROSConfigError("metal_pen_holder: the station manifest has no top camera intrinsics.")
    k = top.intrinsics
    sx, sy = width / k.width, height / k.height
    pen_joint = _id(mujoco.mjtObj.mjOBJ_JOINT, "pen_joint")
    return _MetalPenHolderRollout(
        scene=env_cfg.scene,
        task=env_cfg.task,
        options=opts,
        _model=model,
        _data=mujoco.MjData(model),
        _joint_ranges=np.asarray([model.jnt_range[j] for j in joint_ids], dtype=np.float64),
        _pen_qpos=int(model.jnt_qposadr[pen_joint]),
        _pen_qvel=int(model.jnt_dofadr[pen_joint]),
        _pen_geom=_id(mujoco.mjtObj.mjOBJ_GEOM, "pen"),
        _pen_body=pen_body,
        _holder_qpos=int(model.jnt_qposadr[_id(mujoco.mjtObj.mjOBJ_JOINT, "pen_holder_joint")]),
        _holder_body=holder_body,
        _top_cam=_id(mujoco.mjtObj.mjOBJ_CAMERA, "top"),
        _finger_geoms=fingers,
        _composite_geoms=composite,
        _plate=_load_plate(opts, width, height) if opts.top_background == "plate" else None,  # type: ignore[arg-type]  # reason: same asset_path/top_plate surface as MetalPickPlaceOptions
        _top_k=(k.fx * sx, k.fy * sy, k.cx * sx, k.cy * sy),
        _instruction=env_cfg.task.instruction
        or "Move the pen holder with the left arm and put the pen into it with the right arm",
        _max_steps=int(env_cfg.task.max_steps or 900),
        _width=width,
        _height=height,
        _steps_per_action=steps_per_control_period(
            float(model.opt.timestep), opts.control_hz, scene="metal_pen_holder"
        ),
        _settle_steps=max(1, round(opts.settle_s * opts.control_hz)),
    )
