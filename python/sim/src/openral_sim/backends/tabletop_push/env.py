"""``tabletop_push`` scene rollout — a robot-agnostic tabletop push task (ADR-0033).

This is the *greenfield* "robot as a flag" scene: registered free-axis (no
``fixed_robot``), it composes its task world around whatever compatible arm the
YAML's ``robot_id`` (or ``--robot``) names, resolving the base MJCF from that
robot's manifest (``assets.mjcf``). Unlike ``so101_box`` — which is coupled to
the SO-ARM101 MJCF schema and therefore pinned to ``so101_follower`` — this
scene works for any position-controlled arm (verified for SO-101, Franka, UR5e).

Task: push the red cube onto the green goal disc on the table. Success is purely
geometric (cube centre within ``goal_radius`` of the goal, cube still on the
table), so it makes no assumption about the robot's gripper or end-effector —
the same criterion holds for every robot.

The :class:`openral_sim.SimRollout` contract:

* ``reset(seed)`` randomises the cube spawn (on the table) and the goal-disc
  position, then settles the scene;
* ``step(action)`` writes an ``nu``-D joint-position-target action to the
  robot's actuators (``nu`` = the robot's actuator count; the scene adds none)
  and steps the simulator;
* ``render()`` returns the last top-down RGB frame;
* ``mujoco_handles()`` exposes ``MjModel`` / ``MjData`` to ``openral sim run --view``.

The action / state dimension is ``nu`` — read from the compiled model, not
hardcoded — so it adapts to the loaded robot. Indices are 1:1 with the robot's
MJCF actuator order, the same contract :meth:`openral_hal.MujocoArmHAL._sim_kwargs_for`
relies on (the appended task world never reorders the robot's actuators).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.backends.tabletop_push._assets import (
    TabletopOptions,
    compose_tabletop_mjcf,
)
from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult, sim_time_ns_from_mujoco_handles

if TYPE_CHECKING:
    import mujoco
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation

_DEFAULT_MAX_STEPS = 300
_DEFAULT_RENDER_HEIGHT = 480
_DEFAULT_RENDER_WIDTH = 640
_XY_LEN = 2
_XYZ_LEN = 3
_RANGE_PAIR_LEN = 2
_SPAWN_ATTEMPTS = 32


def _options_from_backend_options(raw: dict[str, Any] | None) -> TabletopOptions:
    """Build :class:`TabletopOptions` from the YAML ``scene.backend_options``.

    Unknown keys are rejected loudly so typos surface immediately; every field
    has a default on :class:`TabletopOptions`, so an empty block is valid.
    """
    raw = dict(raw or {})
    valid = {f.name for f in TabletopOptions.__dataclass_fields__.values()}
    unknown = set(raw) - valid
    if unknown:
        raise ROSConfigError(
            f"tabletop_push: unknown scene.backend_options keys {sorted(unknown)!r}; "
            f"valid keys: {sorted(valid)!r}.",
        )

    def _vecn(value: object, name: str, n: int) -> tuple[float, ...]:
        if not isinstance(value, (list, tuple)) or len(value) != n:
            raise ROSConfigError(
                f"tabletop_push: scene.backend_options.{name} must be a {n}-vector; got {value!r}",
            )
        return tuple(float(v) for v in value)

    def _range2(value: object, name: str) -> tuple[tuple[float, float], tuple[float, float]]:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != _RANGE_PAIR_LEN
            or not all(isinstance(p, (list, tuple)) and len(p) == _RANGE_PAIR_LEN for p in value)
        ):
            raise ROSConfigError(
                f"tabletop_push: scene.backend_options.{name} must be "
                f"((x_min, x_max), (y_min, y_max)); got {value!r}",
            )
        x, y = value
        return ((float(x[0]), float(x[1])), (float(y[0]), float(y[1])))

    xy_keys = {"table_size_xy", "table_center_xy"}
    xyz_keys = {
        "robot_base_xyz",
        "cube_size",
        "top_camera_pos",
        "front_camera_pos",
        "wrist_camera_pos_local",
        "ambient_light",
    }
    range_keys = {"cube_spawn_xy_range", "goal_spawn_xy_range"}

    parsed: dict[str, Any] = {}
    for key, value in raw.items():
        if key in xy_keys:
            parsed[key] = _vecn(value, key, _XY_LEN)
        elif key in xyz_keys:
            parsed[key] = _vecn(value, key, _XYZ_LEN)
        elif key in range_keys:
            parsed[key] = _range2(value, key)
        elif key == "settle_steps":
            parsed[key] = int(value)
        elif key == "joint_units":
            units = str(value).lower()
            if units not in ("radians", "degrees"):
                raise ROSConfigError(
                    f"tabletop_push: scene.backend_options.joint_units must be "
                    f"'radians' or 'degrees'; got {units!r}",
                )
            parsed[key] = units
        elif key in ("initial_joint_positions", "joint_offsets_deg", "joint_signs", "joint_scales"):
            if not isinstance(value, (list, tuple)):
                raise ROSConfigError(
                    f"tabletop_push: scene.backend_options.{key} must be a numeric list; "
                    f"got {value!r}",
                )
            parsed[key] = tuple(float(v) for v in value)
        elif key == "wrist_camera_mount_body":
            parsed[key] = None if value is None else str(value)
        elif key == "instruction":
            parsed[key] = str(value)
        elif key == "extra_metadata":
            if not isinstance(value, dict):
                raise ROSConfigError(
                    f"tabletop_push: scene.backend_options.extra_metadata must be a dict; "
                    f"got {type(value).__name__}",
                )
            parsed[key] = {str(k): str(v) for k, v in value.items()}
        else:
            parsed[key] = float(value)  # remaining keys are scalar floats
    return TabletopOptions(**parsed)


def _validate_joint_unit_affine(options: TabletopOptions, n_act: int) -> None:
    if options.joint_units != "degrees":
        return

    for name, values in (
        ("joint_offsets_deg", options.joint_offsets_deg),
        ("joint_signs", options.joint_signs),
        ("joint_scales", options.joint_scales),
    ):
        if values and len(values) != n_act:
            raise ROSConfigError(
                "tabletop_push: degree-trained policy mode requires "
                f"{name} to have length {n_act}; got {len(values)}.",
            )
    if any(s not in (1.0, -1.0) for s in options.joint_signs):
        raise ROSConfigError(
            "tabletop_push: scene.backend_options.joint_signs entries must be +1 or -1.",
        )
    if any(s <= 0.0 for s in options.joint_scales):
        raise ROSConfigError(
            "tabletop_push: scene.backend_options.joint_scales entries must be positive.",
        )


@dataclass
class _TabletopPushRollout:
    """Rollout for the robot-agnostic ``tabletop_push`` scene."""

    scene: SceneSpec
    task: TaskSpec
    options: TabletopOptions
    _model: mujoco.MjModel
    _data: mujoco.MjData
    # Robot actuators occupy ctrl[0 : n_act]; the scene adds none. Each actuator
    # drives one transmission joint — used for state read + action clipping.
    _n_act: int
    _act_qpos_addrs: list[int]
    _act_dof_addrs: list[int]
    _act_clip_ranges: NDArray[np.float64]
    _cube_qpos_addr: int
    _cube_body_id: int
    _goal_site_id: int
    _camera_names: tuple[str, ...]
    _instruction: str
    _max_steps: int
    _render_height: int
    _render_width: int
    _settle_steps: int
    _resting_cube_z: float
    _initial_joint_positions: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    _joint_units: str = "radians"
    _joint_offsets_deg: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    _joint_signs: NDArray[np.float64] = field(default_factory=lambda: np.ones(0))
    _joint_scales: NDArray[np.float64] = field(default_factory=lambda: np.ones(0))
    _step_count: int = 0
    _renderer_rgb: Any = None
    _last_rgb: NDArray[np.uint8] | None = None
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # ---------------------------------------------------------------- SimRollout

    def reset(self, seed: int | None = None) -> Observation:
        import mujoco

        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        mujoco.mj_resetData(self._model, self._data)
        if self._initial_joint_positions.size:
            qpos_targets = self._policy_targets_to_mujoco_radians(self._initial_joint_positions)
            for aid, qa, da, target in zip(
                range(self._n_act),
                self._act_qpos_addrs,
                self._act_dof_addrs,
                qpos_targets,
                strict=True,
            ):
                self._data.qpos[qa] = float(target)
                self._data.qvel[da] = 0.0
                self._data.ctrl[aid] = float(target)
        else:
            # Hold the robot at its MJCF default pose: seed ctrl from qpos so the
            # position actuators don't snap to the actuator centre on step 1.
            for aid, qa in enumerate(self._act_qpos_addrs):
                self._data.ctrl[aid] = float(self._data.qpos[qa])

        cube_xy = self._sample_xy(self.options.cube_spawn_xy_range)
        goal_xy = self._sample_xy_with_min_separation(
            self.options.goal_spawn_xy_range,
            other_xy=cube_xy,
            min_sep=self.options.goal_min_separation,
        )
        self._write_cube_pose(cube_xy)
        # The goal is a static worldbody site — randomise it by writing model
        # site_pos (site_xpos tracks it after mj_forward).
        self._model.site_pos[self._goal_site_id] = [
            goal_xy[0],
            goal_xy[1],
            self.options.table_top_z + 0.001,
        ]

        mujoco.mj_forward(self._model, self._data)
        for _ in range(self._settle_steps):
            mujoco.mj_step(self._model, self._data)

        self._step_count = 0
        return self._observation()

    def step(self, action: NDArray[np.float32]) -> StepResult:
        import mujoco

        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (self._n_act,):
            raise ROSConfigError(
                f"tabletop_push expects a {self._n_act}-D joint-position action "
                f"(robot actuator count); got shape {a.shape}",
            )
        self._data.ctrl[: self._n_act] = self._policy_targets_to_mujoco_radians(a)

        for _ in range(self._settle_steps):
            mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        success = self._check_on_goal()
        info: dict[str, Any] = {}
        if self.task.success_key is not None:
            info[self.task.success_key] = success
        return StepResult(
            observation=self._observation(),
            reward=1.0 if success else 0.0,
            terminated=bool(success),
            truncated=self._step_count >= self._max_steps,
            info=info,
        )

    def render(self) -> NDArray[np.uint8] | None:
        if self._last_rgb is None:
            return None
        return self._last_rgb.copy()

    def close(self) -> None:
        if self._renderer_rgb is not None:
            self._renderer_rgb.close()
            self._renderer_rgb = None

    def mujoco_handles(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Expose the model + data to ``openral sim run --view``."""
        return self._model, self._data

    def sim_time_ns(self) -> int | None:
        """Elapsed MuJoCo sim time in ns (ADR-0048 Phase 1).

        Reads ``MjData.time`` off :meth:`mujoco_handles`. Monotonic within an
        episode; rewinds on ``reset``.
        """
        return sim_time_ns_from_mujoco_handles(self.mujoco_handles())

    @property
    def action_dim(self) -> int:
        """Flat action width the env's ``step`` accepts (robot actuator count).

        ADR-0034 follow-up — ``SimAttachedHAL._probe_env_action_dim`` reads this so a
        deploy-sim action (``send_action`` / ``idle_step``) is sized to this
        robot-agnostic scene's actuator count rather than the
        robosuite-mobile-manipulator fallback (11). Without it the probe missed this
        native backend (it exposes no upstream ``action_dim``) and the next
        ``env.step`` raised a width mismatch.
        """
        return self._n_act

    # ---------------------------------------------------------------- helpers

    def _observation(self) -> Observation:
        images = {name: self._render_named_rgb(name) for name in self._camera_names}
        self._last_rgb = images[self._camera_names[0]]
        return {
            "images": images,
            "state": self._proprio_state(),
            "task": self._instruction,
        }

    def _proprio_state(self) -> NDArray[np.float32]:
        """Joint positions in the policy unit convention, in actuator order."""
        qpos = np.asarray(
            [float(self._data.qpos[a]) for a in self._act_qpos_addrs],
            dtype=np.float64,
        )
        if self._joint_units == "degrees":
            return np.asarray(
                self._joint_signs * self._joint_scales * np.degrees(qpos) + self._joint_offsets_deg,
                dtype=np.float32,
            )
        return np.asarray(qpos, dtype=np.float32)

    def _policy_targets_to_mujoco_radians(
        self, targets: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Convert policy-unit absolute joint targets to clipped MuJoCo radians."""
        if self._joint_units == "degrees":
            cmd = np.radians(
                self._joint_signs * (targets - self._joint_offsets_deg) / self._joint_scales
            )
        else:
            cmd = targets
        lo = self._act_clip_ranges[:, 0]
        hi = self._act_clip_ranges[:, 1]
        return np.clip(cmd, lo, hi)

    def _render_named_rgb(self, camera_name: str) -> NDArray[np.uint8]:
        import mujoco

        if self._renderer_rgb is None:
            self._renderer_rgb = mujoco.Renderer(
                self._model,
                height=self._render_height,
                width=self._render_width,
            )
        self._renderer_rgb.update_scene(self._data, camera=camera_name)
        return np.asarray(self._renderer_rgb.render(), dtype=np.uint8).copy()

    # ------------------------------------------------- spawn / pose utilities

    def _sample_xy(
        self,
        xy_range: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[float, float]:
        (x_lo, x_hi), (y_lo, y_hi) = xy_range
        return (
            float(self._rng.uniform(x_lo, x_hi)),
            float(self._rng.uniform(y_lo, y_hi)),
        )

    def _sample_xy_with_min_separation(
        self,
        xy_range: tuple[tuple[float, float], tuple[float, float]],
        *,
        other_xy: tuple[float, float],
        min_sep: float,
    ) -> tuple[float, float]:
        """Sample (x, y) until it is at least ``min_sep`` from ``other_xy``.

        Caps at :data:`_SPAWN_ATTEMPTS`; falls back to the last draw if every
        attempt collides (a small cube in a roomy spawn range makes this a
        one-shot path in practice).
        """
        x, y = self._sample_xy(xy_range)
        for _ in range(_SPAWN_ATTEMPTS):
            if (x - other_xy[0]) ** 2 + (y - other_xy[1]) ** 2 >= min_sep**2:
                return (x, y)
            x, y = self._sample_xy(xy_range)
        return (x, y)

    def _write_cube_pose(self, xy: tuple[float, float]) -> None:
        """Write the cube freejoint qpos: on the table at ``xy``, identity yaw."""
        addr = self._cube_qpos_addr
        self._data.qpos[addr + 0] = xy[0]
        self._data.qpos[addr + 1] = xy[1]
        self._data.qpos[addr + 2] = self._resting_cube_z
        self._data.qpos[addr + 3] = 1.0
        self._data.qpos[addr + 4] = 0.0
        self._data.qpos[addr + 5] = 0.0
        self._data.qpos[addr + 6] = 0.0

    # ----------------------------------------------------------- success check

    def _check_on_goal(self) -> bool:
        """True iff the cube rests on the table and its centre is over the goal.

        Robot-agnostic: reads only the cube body pose and the goal site pose, so
        the same criterion holds for every robot. The on-table check rejects the
        false positive where a cube knocked off the table happens to pass over
        the goal XY at the wrong height.
        """
        cube_pos = np.asarray(self._data.xpos[self._cube_body_id], dtype=np.float64)
        goal_xy = np.asarray(self._data.site_xpos[self._goal_site_id][:2], dtype=np.float64)
        if abs(float(cube_pos[2]) - self._resting_cube_z) > self.options.off_table_z_tol:
            return False
        return bool(np.linalg.norm(cube_pos[:2] - goal_xy) <= self.options.goal_radius)


@SCENES.register("tabletop_push")
def build_tabletop_push_scene(env_cfg: SimEnvironment) -> _TabletopPushRollout:
    """Build the robot-agnostic ``tabletop_push`` rollout (ADR-0033, free-axis).

    The robot is a flag: ``env_cfg.robot_id`` (set from the YAML ``robot_id:`` or
    ``--robot``) resolves a :class:`~openral_core.RobotDescription`, whose
    ``assets.mjcf`` provides the base arm MJCF. The table / cube / goal / camera
    world is composed around it via :func:`compose_tabletop_mjcf`. The robot mount
    honours ``env_cfg.base_pose`` (full 6-DOF) with the ``robot_base_xyz`` /
    ``robot_base_yaw_deg`` backend options as a yaw-only fallback.

    Raises:
        ROSConfigError: If ``env_cfg.robot_id`` has no registered manifest, or
            the manifest lacks a usable ``sim`` block.
    """
    import mujoco

    from openral_sim.factory import make_robot  # reason: defer to avoid import cycle

    options = _options_from_backend_options(env_cfg.scene.backend_options)
    description = make_robot(env_cfg)
    if description is None:
        raise ROSConfigError(
            f"tabletop_push is a free-axis scene and needs a robot, but "
            f"robot_id={env_cfg.robot_id!r} has no registered manifest. Set a "
            "valid `robot_id:` in the YAML or pass --robot <robot_id>.",
        )

    model = compose_tabletop_mjcf(description, options, base_pose=env_cfg.base_pose)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # The robot's actuators are ctrl[0 : nu] (the scene adds none). For each,
    # resolve its transmission joint to read state + clip the action to the
    # joint's range (position actuators frequently declare no ctrlrange, so the
    # [0, 0] ctrlrange sentinel cannot be used — see so101_box for the same
    # lesson).
    n_act = int(model.nu)
    if n_act == 0:
        raise ROSConfigError(
            f"tabletop_push: robot {description.name!r} compiled with no actuators; "
            "cannot drive it.",
        )
    act_qpos_addrs: list[int] = []
    act_dof_addrs: list[int] = []
    clip_ranges: list[tuple[float, float]] = []
    for aid in range(n_act):
        jid = int(model.actuator_trnid[aid, 0])
        if jid < 0:
            raise ROSConfigError(
                f"tabletop_push: actuator {aid} on {description.name!r} has no joint "
                "transmission; only joint-position arms are supported.",
            )
        act_qpos_addrs.append(int(model.jnt_qposadr[jid]))
        act_dof_addrs.append(int(model.jnt_dofadr[jid]))
        if bool(model.jnt_limited[jid]):
            lo, hi = (float(v) for v in model.jnt_range[jid])
        elif bool(model.actuator_ctrllimited[aid]):
            lo, hi = (float(v) for v in model.actuator_ctrlrange[aid])
        else:
            lo, hi = (-np.inf, np.inf)
        clip_ranges.append((lo, hi))

    _validate_joint_unit_affine(options, n_act)
    if options.initial_joint_positions and len(options.initial_joint_positions) != n_act:
        raise ROSConfigError(
            "tabletop_push: scene.backend_options.initial_joint_positions must have "
            f"length {n_act}; got {len(options.initial_joint_positions)}.",
        )

    goal_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_joint = int(model.body_jntadr[cube_body])
    if goal_site < 0 or cube_body < 0 or cube_joint < 0:
        raise ROSConfigError(
            "tabletop_push: composed model is missing the goal site / cube body / "
            "cube freejoint — the composer must be updated.",
        )

    camera_names: tuple[str, ...] = ("top", "front")
    if options.wrist_camera_mount_body is not None:
        camera_names = (*camera_names, "wrist")

    render_w = int(env_cfg.scene.observation_width or _DEFAULT_RENDER_WIDTH)
    render_h = int(env_cfg.scene.observation_height or _DEFAULT_RENDER_HEIGHT)
    max_steps = int(env_cfg.task.max_steps or _DEFAULT_MAX_STEPS)
    instruction = env_cfg.task.instruction or options.instruction

    return _TabletopPushRollout(
        scene=env_cfg.scene,
        task=env_cfg.task,
        options=options,
        _model=model,
        _data=data,
        _n_act=n_act,
        _act_qpos_addrs=act_qpos_addrs,
        _act_dof_addrs=act_dof_addrs,
        _act_clip_ranges=np.asarray(clip_ranges, dtype=np.float64),
        _cube_qpos_addr=int(model.jnt_qposadr[cube_joint]),
        _cube_body_id=int(cube_body),
        _goal_site_id=int(goal_site),
        _camera_names=camera_names,
        _instruction=instruction,
        _max_steps=max_steps,
        _render_height=render_h,
        _render_width=render_w,
        _settle_steps=options.settle_steps,
        _resting_cube_z=options.table_top_z + options.cube_size[2],
        _initial_joint_positions=np.asarray(options.initial_joint_positions, dtype=np.float64),
        _joint_units=options.joint_units,
        _joint_offsets_deg=np.asarray(
            options.joint_offsets_deg or (0.0,) * n_act, dtype=np.float64
        ),
        _joint_signs=np.asarray(options.joint_signs or (1.0,) * n_act, dtype=np.float64),
        _joint_scales=np.asarray(options.joint_scales or (1.0,) * n_act, dtype=np.float64),
        _rng=np.random.default_rng(env_cfg.seed or 0),
    )
