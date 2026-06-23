"""``so101_box`` scene rollout — SO-101 in a parameterised box arena.

Registers ``so101_box`` as a fixed-robot scene against ``so101_follower``.
The rollout implements :class:`openral_sim.SimRollout`:

* ``reset(seed)`` randomises the (x, y, yaw) pose of both the
  slotted block and the cylindrical tube, both resting on the floor
  of the arena;
* ``step(action)`` writes a 6-D joint-position-target action to the
  SO-101's upstream ``<position>`` actuators;
* ``render()`` returns the last OAK-D Pro RGB frame the policy saw;
* ``mujoco_handles()`` exposes the ``MjModel`` / ``MjData`` to
  ``openral sim run --view``.

The observation dict carries:

* ``images.oak_top``: HWC uint8 RGB from the overhead OAK-D Pro
  (bare MuJoCo camera name, matching ``scene.cameras``);
* ``images.oak_top_depth``: HW float32 depth in metres from the same
  camera;
* ``images.wrist``: HWC uint8 RGB from the terminal gripper-mounted camera;
* ``state``: 6-D float32 of the SO-101 joint qpos (rad);
* ``task``: the natural-language instruction.

The success signal — written to ``info[task.success_key]`` — fires
when the tube is inserted vertically into the slotted block's hole
within the configured tolerances (see :class:`BoxSceneOptions`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.backends.so101_box._assets import (
    BoxSceneOptions,
    compose_so101_box_mjcf,
)
from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult, sim_time_ns_from_mujoco_handles

if TYPE_CHECKING:
    import mujoco
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


# SO-101 joint qpos slots (same order as ``robots/so101_follower/robot.yaml``
# joints[]). ``MujocoArmHAL._sim_kwargs_for`` resolves by INDEX, so the
# upstream MJCF's numeric joint names ("1"…"6") don't show up here —
# we drive the 6 ``<position>`` actuators directly by their actuator
# index in the upstream actuator block (also "1"…"6").
_SO101_ARM_DOF = 6
_DEFAULT_MAX_STEPS = 500
_DEFAULT_RENDER_HEIGHT = 480
_DEFAULT_RENDER_WIDTH = 640
# Shape constants for backend_options validators.
_XYZ_LEN = 3
_RANGE_PAIR_LEN = 2


def _options_from_backend_options(raw: dict[str, Any] | None) -> BoxSceneOptions:
    """Build a :class:`BoxSceneOptions` from the YAML's ``scene.backend_options`` dict.

    Unknown keys are rejected loudly so YAML typos surface immediately.
    Every field default lives on :class:`BoxSceneOptions`, so an empty
    block is valid.
    """
    raw = dict(raw or {})
    valid = {f.name for f in BoxSceneOptions.__dataclass_fields__.values()}
    unknown = set(raw) - valid
    if unknown:
        raise ROSConfigError(
            f"so101_box: unknown scene.backend_options keys {sorted(unknown)!r}; "
            f"valid keys: {sorted(valid)!r}.",
        )

    def _tuple3(value: object, name: str) -> tuple[float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != _XYZ_LEN:
            raise ROSConfigError(
                f"so101_box: scene.backend_options.{name} must be a 3-vector; got {value!r}",
            )
        return (float(value[0]), float(value[1]), float(value[2]))

    def _range2(value: object, name: str) -> tuple[tuple[float, float], tuple[float, float]]:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != _RANGE_PAIR_LEN
            or not all(isinstance(p, (list, tuple)) and len(p) == _RANGE_PAIR_LEN for p in value)
        ):
            raise ROSConfigError(
                f"so101_box: scene.backend_options.{name} must be "
                f"((x_min, x_max), (y_min, y_max)); got {value!r}",
            )
        x, y = value
        return ((float(x[0]), float(x[1])), (float(y[0]), float(y[1])))

    def _vec6(value: object, name: str) -> tuple[float, ...]:
        if not isinstance(value, (list, tuple)) or len(value) != _SO101_ARM_DOF:
            raise ROSConfigError(
                f"so101_box: scene.backend_options.{name} must be a "
                f"{_SO101_ARM_DOF}-vector (one per joint); got {value!r}",
            )
        vec = tuple(float(v) for v in value)
        if name == "joint_signs" and any(s not in (1.0, -1.0) for s in vec):
            raise ROSConfigError(
                f"so101_box: scene.backend_options.joint_signs entries must be "
                f"+1 or -1; got {value!r}",
            )
        return vec

    parsed: dict[str, Any] = {}
    for key, value in raw.items():
        if key in (
            "box_size_xyz",
            "robot_base_xyz",
            "wrist_camera_pos_local",
            "wrist_camera_target_local",
            "wrist_camera_up_local",
            "oak_top_camera_pos",
            "oak_top_camera_target",
            "slot_block_size",
        ):
            parsed[key] = _tuple3(value, key)
        elif key in ("block_spawn_xy_range", "tube_spawn_xy_range"):
            parsed[key] = _range2(value, key)
        elif key == "extra_metadata":
            if not isinstance(value, dict):
                raise ROSConfigError(
                    f"so101_box: scene.backend_options.extra_metadata "
                    f"must be a dict; got {type(value).__name__}",
                )
            parsed[key] = {str(k): str(v) for k, v in value.items()}
        elif key == "joint_units":
            units = str(value).lower()
            if units not in ("radians", "degrees"):
                raise ROSConfigError(
                    f"so101_box: scene.backend_options.joint_units must be "
                    f"'radians' or 'degrees'; got {value!r}",
                )
            parsed[key] = units
        elif key in ("joint_offsets_deg", "joint_signs"):
            parsed[key] = _vec6(value, key)
        else:
            parsed[key] = float(value)
    return BoxSceneOptions(**parsed)


@dataclass
class _So101BoxRollout:
    """Rollout for the so101_box scene."""

    scene: SceneSpec
    task: TaskSpec
    options: BoxSceneOptions
    _model: mujoco.MjModel
    _data: mujoco.MjData
    _arm_actuator_ids: list[int]
    _arm_qpos_addrs: list[int]
    # Per-joint position limits (radians), shape (_SO101_ARM_DOF, 2). Action
    # clip bounds — the upstream MJCF's <position> actuators declare only
    # forcerange (no ctrlrange), so model.actuator_ctrlrange is the [0, 0]
    # "unlimited" sentinel; clipping to that would zero every command. We
    # clip to the joint range instead. See step().
    _arm_joint_ranges: NDArray[np.float64]
    _slot_block_qpos_addr: int
    _tube_qpos_addr: int
    _slot_block_body_id: int
    _tube_body_id: int
    _hole_site_id: int
    _tube_tip_lo_site_id: int
    _tube_tip_hi_site_id: int
    _instruction: str
    _max_steps: int
    _render_height: int
    _render_width: int
    # Joint-units convention for the proprio state the policy reads and the
    # action it returns. ``"radians"`` (default) keeps MuJoCo-native units;
    # ``"degrees"`` makes the env emit state in degrees and accept actions in
    # degrees — the convention LeRobot-trained SO-100/101 checkpoints (for
    # example MolmoAct2-SO100_101) were recorded in. Mirrors the
    # openarm_robosuite scene's ``scene.backend_options.joint_units`` knob.
    _joint_units: str = "radians"
    # Per-joint calibration affine (degrees mode only), bridging the MuJoCo URDF
    # joint convention to the checkpoint's LeRobot servo-degree convention:
    #     lerobot_deg = _joint_signs * mujoco_deg + _joint_offsets_deg
    # Identity by default. Shape (_SO101_ARM_DOF,). See BoxSceneOptions.
    _joint_offsets_deg: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(_SO101_ARM_DOF, dtype=np.float64)
    )
    _joint_signs: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(_SO101_ARM_DOF, dtype=np.float64)
    )
    _settle_steps: int = 50
    _step_count: int = 0
    _renderer_rgb: Any = None  # mujoco.Renderer — lazy, RGB-mode
    _renderer_depth: Any = None  # mujoco.Renderer — lazy, depth-mode
    _last_oak_rgb: NDArray[np.uint8] | None = None
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # ---------------------------------------------------------------- SimRollout

    def reset(self, seed: int | None = None) -> Observation:
        import mujoco

        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        mujoco.mj_resetData(self._model, self._data)
        # Seed the arm at the MJCF default zero pose; ctrl tracks qpos
        # so the first step doesn't snap to actuator centre.
        for qa, aid in zip(self._arm_qpos_addrs, self._arm_actuator_ids, strict=True):
            self._data.ctrl[aid] = float(self._data.qpos[qa])

        block_xy = self._sample_xy(self.options.block_spawn_xy_range)
        tube_xy = self._sample_xy_with_min_separation(
            self.options.tube_spawn_xy_range,
            other_xy=block_xy,
            min_sep=self.options.spawn_min_separation,
        )
        self._write_freejoint(
            qpos_addr=self._slot_block_qpos_addr,
            xyz=(block_xy[0], block_xy[1], self.options.slot_block_size[2] / 2.0),
            yaw=float(self._rng.uniform(-np.pi, np.pi)),
        )
        # Tube: lying on its side. We tip the cylinder 90° about world +X
        # so its long body axis points along world +Y, then yaw it about
        # world +Z. The composed rotation lifts the tube's CoM to
        # z = tube_radius.
        tube_yaw = float(self._rng.uniform(-np.pi, np.pi))
        self._write_freejoint_lying(
            qpos_addr=self._tube_qpos_addr,
            xyz=(tube_xy[0], tube_xy[1], self.options.tube_radius),
            yaw=tube_yaw,
        )

        mujoco.mj_forward(self._model, self._data)
        # Let the bodies settle on the floor before the first observation.
        for _ in range(self._settle_steps):
            mujoco.mj_step(self._model, self._data)

        self._step_count = 0
        return self._observation()

    def step(self, action: NDArray[np.float32]) -> StepResult:
        import mujoco

        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (_SO101_ARM_DOF,):
            raise ROSConfigError(
                f"so101_box expects a {_SO101_ARM_DOF}-D joint-position action; "
                f"got shape {a.shape}",
            )
        # Degree-trained checkpoints (LeRobot SO-100/101) emit absolute joint
        # targets in LeRobot servo degrees; MuJoCo ctrl is radians. Invert the
        # calibration affine (lerobot_deg → mujoco_deg) then convert to radians.
        # ``"radians"`` policies pass through untouched.
        if self._joint_units == "degrees":
            mujoco_deg = self._joint_signs * (a - self._joint_offsets_deg)
            cmd: NDArray[np.float64] = np.radians(mujoco_deg)
        else:
            cmd = a
        # Clip to the joint position limits (radians) before writing. The
        # MJCF's <position> actuators carry no ctrlrange, so
        # actuator_ctrlrange is the [0, 0] unlimited sentinel — clipping to
        # it would pin every command to zero (arm never moves). jnt_range is
        # the real per-joint limit.
        lo = self._arm_joint_ranges[:, 0]
        hi = self._arm_joint_ranges[:, 1]
        self._data.ctrl[self._arm_actuator_ids] = np.clip(cmd, lo, hi)

        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        success = self._check_insertion()
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
        if self._last_oak_rgb is None:
            return None
        return self._last_oak_rgb.copy()

    def close(self) -> None:
        if self._renderer_rgb is not None:
            self._renderer_rgb.close()
            self._renderer_rgb = None
        if self._renderer_depth is not None:
            self._renderer_depth.close()
            self._renderer_depth = None

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
        """Flat action width the env's ``step`` accepts (SO-101 = 6 joint targets).

        ADR-0034 follow-up — ``SimAttachedHAL._probe_env_action_dim`` reads this so a
        deploy-sim action (``send_action`` / ``idle_step``) is sized to the SO-101's
        6-D joint-position action rather than the robosuite-mobile-manipulator
        fallback (11). Without it the probe missed this native backend (it exposes
        no upstream ``action_dim``) and the next ``env.step`` raised a width mismatch.
        """
        return _SO101_ARM_DOF

    # ---------------------------------------------------------------- helpers

    def _observation(self) -> Observation:
        oak_rgb, oak_depth = self._render_oak_top()
        wrist_rgb = self._render_named_rgb("wrist")
        self._last_oak_rgb = oak_rgb
        return {
            "images": {
                # RGB is keyed by the bare MuJoCo camera name (``oak_top``)
                # so it lines up with ``scene.cameras`` — that's the key
                # the policy adapter's ``resolve_camera_keys`` pulls from
                # the observation. Depth keeps the ``_depth`` suffix as a
                # sibling stream for depth-aware skills.
                "oak_top": oak_rgb,
                "oak_top_depth": oak_depth,
                "wrist": wrist_rgb,
            },
            "state": self._proprio_state(),
            "task": self._instruction,
        }

    def _proprio_state(self) -> NDArray[np.float32]:
        """6-D SO-101 joint proprioception in the policy's expected units.

        Reads the arm qpos (MuJoCo radians). In ``"degrees"`` mode, converts to
        degrees and applies the calibration affine
        (``lerobot_deg = signs * mujoco_deg + offsets``) so the proprio matches
        the LeRobot servo-degree convention the checkpoint was trained in.
        ``"radians"`` returns qpos as-is.
        """
        qpos = np.asarray(
            [float(self._data.qpos[a]) for a in self._arm_qpos_addrs],
            dtype=np.float64,
        )
        if self._joint_units == "degrees":
            lerobot_deg: NDArray[np.float64] = (
                self._joint_signs * np.degrees(qpos) + self._joint_offsets_deg
            )
            return np.asarray(lerobot_deg, dtype=np.float32)
        return np.asarray(qpos, dtype=np.float32)

    def _render_oak_top(self) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        import mujoco

        if self._renderer_rgb is None:
            self._renderer_rgb = mujoco.Renderer(
                self._model,
                height=self._render_height,
                width=self._render_width,
            )
        if self._renderer_depth is None:
            self._renderer_depth = mujoco.Renderer(
                self._model,
                height=self._render_height,
                width=self._render_width,
            )
            self._renderer_depth.enable_depth_rendering()
        self._renderer_rgb.update_scene(self._data, camera="oak_top")
        rgb = np.asarray(self._renderer_rgb.render(), dtype=np.uint8).copy()
        self._renderer_depth.update_scene(self._data, camera="oak_top")
        depth = np.asarray(self._renderer_depth.render(), dtype=np.float32).copy()
        return rgb, depth

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
        """Sample (x, y) until the distance to ``other_xy`` is >= ``min_sep``.

        Caps at 32 attempts; falls back to the last sample if every
        draw collides. The block being a small object in a big
        workspace means this is overwhelmingly a one-shot path.
        """
        for _ in range(32):
            x, y = self._sample_xy(xy_range)
            if (x - other_xy[0]) ** 2 + (y - other_xy[1]) ** 2 >= min_sep**2:
                return (x, y)
        return (x, y)

    def _write_freejoint(
        self,
        *,
        qpos_addr: int,
        xyz: tuple[float, float, float],
        yaw: float,
    ) -> None:
        """Write a 7-D freejoint qpos slot (pos + quat).

        Uses a pure-yaw quaternion so the body lands axis-aligned with
        its body-local +Z pointing up.
        """
        self._data.qpos[qpos_addr + 0] = xyz[0]
        self._data.qpos[qpos_addr + 1] = xyz[1]
        self._data.qpos[qpos_addr + 2] = xyz[2]
        # quat = (cos(yaw/2), 0, 0, sin(yaw/2))
        self._data.qpos[qpos_addr + 3] = float(np.cos(yaw / 2.0))
        self._data.qpos[qpos_addr + 4] = 0.0
        self._data.qpos[qpos_addr + 5] = 0.0
        self._data.qpos[qpos_addr + 6] = float(np.sin(yaw / 2.0))
        # Zero the corresponding 6 qvel slots (freejoint qvel offset = 6).
        # Caller is responsible for zeroing data.qvel separately if reset
        # didn't already do it; mj_resetData does, so this method
        # leaves qvel alone.

    def _write_freejoint_lying(
        self,
        *,
        qpos_addr: int,
        xyz: tuple[float, float, float],
        yaw: float,
    ) -> None:
        """Place a cylinder lying on its side, with its long axis yawed about world +Z.

        Composed rotation: first tip 90° about world +X (cylinder axis
        flips from body +Z to world +Y), then yaw about world +Z. The
        result is the quaternion ``q_yaw * q_tip``.
        """
        # q_tip = rotation by π/2 about +X = (cos π/4, sin π/4, 0, 0)
        c_x = float(np.cos(np.pi / 4.0))
        s_x = float(np.sin(np.pi / 4.0))
        # q_yaw = (cos yaw/2, 0, 0, sin yaw/2)
        c_z = float(np.cos(yaw / 2.0))
        s_z = float(np.sin(yaw / 2.0))
        # Hamilton product q = q_z * q_x
        w = c_z * c_x - s_z * 0.0
        x = c_z * s_x + s_z * 0.0
        y = c_z * 0.0 + s_z * s_x
        z = c_z * 0.0 + s_z * c_x
        self._data.qpos[qpos_addr + 0] = xyz[0]
        self._data.qpos[qpos_addr + 1] = xyz[1]
        self._data.qpos[qpos_addr + 2] = xyz[2]
        self._data.qpos[qpos_addr + 3] = w
        self._data.qpos[qpos_addr + 4] = x
        self._data.qpos[qpos_addr + 5] = y
        self._data.qpos[qpos_addr + 6] = z

    # ----------------------------------------------------------- success check

    def _check_insertion(self) -> bool:
        """True iff the tube is inserted vertically into the block hole.

        Three conditions, AND-ed (all must hold):

        1. The tube's long axis is within
           :attr:`BoxSceneOptions.insertion_axis_tol_deg` of world -Z
           (tube is pointing down).
        2. The tube's lower tip is inside the block hole's XY footprint
           (within :attr:`BoxSceneOptions.insertion_xy_tol_m`).
        3. The lower tip's Z is below the block's top face by at least
           :attr:`BoxSceneOptions.insertion_depth_m`.
        """
        tube_z_axis_world = self._tube_z_axis_world()
        # Verticality: the tube's long axis (cylinder's body-local +Z)
        # must be ~parallel to world Z, regardless of sign. ``abs`` so
        # both "tip-down" and "tip-up" orientations count as vertical;
        # the bottom-tip geometry check below handles the actual depth.
        cos_angle = abs(float(tube_z_axis_world[2]))
        min_cos_tol = float(np.cos(np.radians(self.options.insertion_axis_tol_deg)))
        if cos_angle < min_cos_tol:
            return False

        # Tube tip (lower end) world position via the cached site id.
        tip_lo_world = np.asarray(self._data.site_xpos[self._tube_tip_lo_site_id])
        tip_hi_world = np.asarray(self._data.site_xpos[self._tube_tip_hi_site_id])
        # Pick whichever tip is geometrically lower in world Z — when
        # the policy spins the tube about its long axis the "lo" tip
        # may end up on top, but the success criterion is "the bottom
        # tip is below the hole top". This avoids a brittle sign
        # convention.
        bottom = tip_lo_world if tip_lo_world[2] < tip_hi_world[2] else tip_hi_world

        block_xy = np.asarray(self._data.site_xpos[self._hole_site_id][:2])
        block_top_z = float(
            self._data.site_xpos[self._hole_site_id][2] + self.options.slot_block_size[2] / 2.0,
        )

        if np.linalg.norm(bottom[:2] - block_xy) > self.options.insertion_xy_tol_m:
            return False
        return bool(bottom[2] <= block_top_z - self.options.insertion_depth_m)

    def _tube_z_axis_world(self) -> NDArray[np.float64]:
        """Return the tube's body-local +Z axis expressed in world coordinates."""
        # MuJoCo stores body rotation matrices in row-major xmat (9 floats).
        xmat = np.asarray(self._data.xmat[self._tube_body_id]).reshape(3, 3)
        return xmat @ np.array([0.0, 0.0, 1.0], dtype=np.float64)


@SCENES.register("so101_box", fixed_robot="so101_follower")
def build_so101_box_scene(env_cfg: SimEnvironment) -> _So101BoxRollout:
    """Build the so101_box rollout from a composed :class:`SimEnvironment`.

    ``env_cfg.scene.backend_options`` drives every dimension and
    threshold via :func:`_options_from_backend_options`. The MJCF is
    composed once at build time; the same model is reused across all
    ``reset()`` calls.

    ADR-0033 — the base arm MJCF is resolved from the robot manifest's
    ``assets.mjcf`` (the same source ``build_hal(mode="sim")`` uses), not a
    hardcode. The scene's splice anchors (``<body name="base">`` /
    ``<body name="gripper">``) + actuator naming (``"1"``..``"6"``) are still
    so_arm101-schema-specific, so ``fixed_robot`` stays ``so101_follower``
    until those anchors are parameterised (then this becomes free-axis and the
    robot is a true flag — see ADR-0033 follow-ups).
    """
    import mujoco

    from openral_sim.factory import make_robot  # reason: defer to avoid import cycle

    options = _options_from_backend_options(env_cfg.scene.backend_options)
    description = make_robot(env_cfg)
    _, output_path = compose_so101_box_mjcf(options, robot_description=description)
    model = mujoco.MjModel.from_xml_path(str(output_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Resolve actuator + joint indices. The upstream SO-101 MJCF names
    # its arm joints / actuators "1"…"6"; we don't rely on the manifest's
    # logical labels here because the rollout never sees the manifest.
    arm_actuator_ids: list[int] = []
    arm_qpos_addrs: list[int] = []
    arm_joint_ranges: list[tuple[float, float]] = []
    for k in range(1, _SO101_ARM_DOF + 1):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(k))
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(k))
        if aid < 0 or jid < 0:
            raise ROSConfigError(
                f"so101_box: SO-101 actuator/joint {k!r} missing in compiled model "
                f"(actuator_id={aid}, joint_id={jid}). Composer must be updated.",
            )
        arm_actuator_ids.append(aid)
        arm_qpos_addrs.append(int(model.jnt_qposadr[jid]))
        lo, hi = (float(x) for x in model.jnt_range[jid])
        arm_joint_ranges.append((lo, hi))

    def _resolve_named(kind: int, name: str) -> int:
        idx = mujoco.mj_name2id(model, kind, name)
        if idx < 0:
            raise ROSConfigError(
                f"so101_box: required {name!r} missing in compiled model (kind={kind}).",
            )
        return int(idx)

    slot_block_body = _resolve_named(mujoco.mjtObj.mjOBJ_BODY, "slot_block")
    tube_body = _resolve_named(mujoco.mjtObj.mjOBJ_BODY, "tube")
    slot_block_joint = _resolve_named(mujoco.mjtObj.mjOBJ_JOINT, "slot_block_joint")
    tube_joint = _resolve_named(mujoco.mjtObj.mjOBJ_JOINT, "tube_joint")
    hole_site = _resolve_named(mujoco.mjtObj.mjOBJ_SITE, "slot_block_hole")
    tip_lo_site = _resolve_named(mujoco.mjtObj.mjOBJ_SITE, "tube_tip_lo")
    tip_hi_site = _resolve_named(mujoco.mjtObj.mjOBJ_SITE, "tube_tip_hi")

    render_w = int(env_cfg.scene.observation_width or _DEFAULT_RENDER_WIDTH)
    render_h = int(env_cfg.scene.observation_height or _DEFAULT_RENDER_HEIGHT)
    max_steps = int(env_cfg.task.max_steps or _DEFAULT_MAX_STEPS)
    instruction = env_cfg.task.instruction or "insert the orange tube into the slotted block"

    return _So101BoxRollout(
        scene=env_cfg.scene,
        task=env_cfg.task,
        options=options,
        _model=model,
        _data=data,
        _arm_actuator_ids=arm_actuator_ids,
        _arm_qpos_addrs=arm_qpos_addrs,
        _arm_joint_ranges=np.asarray(arm_joint_ranges, dtype=np.float64),
        _joint_units=options.joint_units,
        _joint_offsets_deg=np.asarray(options.joint_offsets_deg, dtype=np.float64),
        _joint_signs=np.asarray(options.joint_signs, dtype=np.float64),
        _slot_block_qpos_addr=int(model.jnt_qposadr[slot_block_joint]),
        _tube_qpos_addr=int(model.jnt_qposadr[tube_joint]),
        _slot_block_body_id=slot_block_body,
        _tube_body_id=tube_body,
        _hole_site_id=hole_site,
        _tube_tip_lo_site_id=tip_lo_site,
        _tube_tip_hi_site_id=tip_hi_site,
        _instruction=instruction,
        _max_steps=max_steps,
        _render_height=render_h,
        _render_width=render_w,
        _rng=np.random.default_rng(env_cfg.seed or 0),
    )
