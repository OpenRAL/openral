"""Internal MuJoCo-backed HAL implementation shared by every bare-twin adapter.

``MujocoArmHAL`` is the ``SimAttachedHAL`` shell around a ``MujocoTwinRollout``:
the rollout owns the physics (the robot's own MJCF, its time base, command
composition and validation, ctrl writes, base driving, the one stepping loop,
joint-state reads and lazy camera renders), the shell owns everything a deploy-sim
HAL exposes around it (the send path and its slot-group staging, the idle stepper,
clock, scene export, step bookkeeping, post-step observers, the attachment API,
``reset_episode`` and per-goal grading). It is the simulation
backbone behind ``UR5eHAL``, ``FrankaPandaHAL``, the OpenArm, ALOHA, SO-100, Rizon 4,
OpenAMRobot and humanoid twins; a robot whose twin behaves differently passes its
own rollout subclass (``twin``).

The class is **not** part of the public ``openral_hal`` surface (leading
underscore module name).  Adapters subclass it and expose a typed constructor
specific to their robot.

Design notes
------------
* ``mujoco`` is imported lazily (``MujocoTwinRollout.load``) so the module loads
  cleanly on hosts without it; ``openral_sim`` is never needed.
* ``send_action`` (the shell's) hands every command to the rollout's
  ``step_action_group``: the **last** waypoint of the chunk goes into ``data.ctrl``
  and the simulator steps ``settle_steps`` times — by default one control period of
  ``description.control_rate_hz`` (issue #358), so the twin's clock is the robot's
  clock.  Trajectory interpolation is the job of the controller above it.
* Slot groups are staged in the shell's ``SlotGroupStager`` (the one staging
  machine, shared with the real ros2_control HALs); the twin keeps the HAL
  Protocol's raising ``estop``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Unpack

import structlog
from openral_core.assets import AssetRefError, resolve_asset
from openral_core.exceptions import ROSConfigError, ROSEStopRequested, ROSRuntimeError
from openral_core.schemas import RobotDescription

from openral_hal.mujoco_twin import MujocoTwinRollout, TwinKwargs
from openral_hal.sim_attached import SimAttachedHAL

if TYPE_CHECKING:
    import mujoco

__all__ = ["MujocoArmHAL"]


def _resolve_mjcf_path(desc: RobotDescription) -> str:
    """Resolve ``desc.assets.mjcf`` to an absolute MJCF path via ``resolve_asset``.

    The MujocoArmHAL is built from a ``RobotDescription`` alone (the
    manifest path is not threaded through ``from_description``), so no
    ``manifest_dir`` is supplied — every in-tree MJCF ref is a ``rd:`` /
    ``gym_aloha:`` / ``openarm:`` / ``menagerie:`` / ``unitree:`` scheme that resolves without
    one (no robot declares a ``file:`` MJCF). ``AssetRefError`` is translated to
    ``ROSConfigError`` at this HAL boundary, matching the contract
    ``_sim_kwargs_for`` documented for the old resolver.

    Raises:
        ROSConfigError: If ``desc.assets.mjcf`` is unset or cannot be resolved.
    """
    if not desc.assets.mjcf:
        raise ROSConfigError(
            f"RobotDescription '{desc.name}' has no `assets.mjcf` ref; "
            "cannot resolve a MuJoCo MJCF for MujocoArmHAL."
        )
    try:
        path = resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetRefError as exc:
        raise ROSConfigError(str(exc)) from exc
    if path is None:  # ros2:// is urdf-only; resolve_asset never returns None for mjcf
        raise ROSConfigError(
            f"assets.mjcf={desc.assets.mjcf!r} for '{desc.name}' did not resolve to a file."
        )
    return str(path)


def _kinematic_group(joint_name: str, groups: Sequence[str], *, robot: str) -> str:
    """Return the first entry of *groups* that is a substring of *joint_name*.

    Shared by the G1 and H1 humanoid adapters to classify a joint name
    (``"left_hip_yaw"``, ``"left_wrist_roll_joint"``, …) into one of the
    robot's kinematic groups for velocity/effort/PD-gain lookup tables keyed
    by group rather than by individual joint.

    Args:
        joint_name: The joint name to classify.
        groups: Candidate group tokens, checked in order.
        robot: Robot name, used only in the error message.

    Raises:
        ROSConfigError: No entry of *groups* is a substring of *joint_name*.
    """
    for token in groups:
        if token in joint_name:
            return token
    raise ROSConfigError(f"Unknown {robot} joint group for joint {joint_name!r}.")


log = structlog.get_logger(__name__)


class MujocoArmHAL(SimAttachedHAL):
    """Generic MuJoCo-backed HAL adapter for position-controlled arms.

    Supports single-arm robots, floating-base humanoids (via
    ``joint_qvel_addr`` offset), and bimanual robots (via multiple
    ``SimGripperDescription`` entries with optional
    ``mirror_actuator_index`` for parallel-jaw configurations like Aloha).

    A ``SimAttachedHAL`` whose env is a ``MujocoTwinRollout`` (or the ``twin``
    subclass a robot passes) built from these arguments; the shell drives it. What a
    robot's twin does differently lives in its rollout subclass, never here.
    ``_model`` / ``_data`` / ``_settle_steps`` are the rollout's live values.

    The keyword arguments are ``mujoco_twin.TwinKwargs`` (the one typed shape of the
    twin's wiring), forwarded to the rollout with ``backend`` set to this class's name.

    Args:
        description: Normative ``RobotDescription``; ``description.joints``
            must enumerate every controllable joint in the same order as the
            robot's MuJoCo actuators.
        mjcf_path: Filesystem path to the MJCF XML.
        joint_qpos_addr: Mapping ``joint_name -> qpos index`` in ``data.qpos``.
        actuator_index: Mapping ``joint_name -> actuator index`` (column of
            ``data.ctrl``); ``None`` for read-only joints (e.g. the mirrored
            second finger of a parallel gripper).
        joint_qvel_addr: Optional mapping ``joint_name -> qvel index``;
            defaults to ``joint_qpos_addr`` for non-floating-base arms. For
            floating-base humanoids the qvel indices are offset by 6 (vs
            qpos 7) and must be passed explicitly.
        grippers: Zero or more ``SimGripperDescription`` entries; each
            ``joint`` must also appear in ``description.joints``. Single-arm
            robots ship one (or none); bimanual robots (Aloha, OpenArm) two.
        keyframe_index: When set, ``connect`` calls
            ``mj_resetDataKeyframe(model, data, keyframe_index)`` before
            ``mj_forward`` — required for MJCFs whose default
            ``MjData.qpos`` sits outside the actuator ``ctrlrange``
            (gym-aloha).
        seed_ctrl_from_qpos: When True, ``connect`` seeds
            ``data.ctrl[actuator] = data.qpos[joint_qpos_addr]`` for every
            controllable joint so position actuators hold the initial pose
            (OpenArm v2).
        settle_steps: Number of ``mj_step`` calls in ``send_action`` to
            advance toward the new target. ``None`` (default) derives it at
            ``connect`` so one action is one control period of
            ``description.control_rate_hz`` (``physics_steps_per_tick``,
            shrinking ``opt.timestep`` to divide the period exactly); an
            explicit count is a pinned override, logged, that leaves the twin
            wall-paced (``_step_while_active``). Without a declared rate the
            legacy single step applies.
        gravity_enabled: When ``False``, gravity is zeroed at ``connect()``
            (useful for closed-loop tests asserting exact convergence).
        staleness_limit_s: Age (seconds) of the cached state above which
            ``read_state`` emits a one-shot starvation WARNING (the read
            still returns live ``MjData`` — see ``MujocoTwinRollout.read_joint_state``).
        body_twist_dt_s: Seconds one ``BODY_TWIST`` integrates a twin's base by;
            ``None`` is one control period (``MujocoTwinRollout``).
        twin: The rollout class (or a factory taking its keyword arguments) a robot's
            twin runs on; ``MujocoTwinRollout`` for a position-controlled arm.

    Raises:
        ROSConfigError: If ``description.joints`` is empty.
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        twin: Callable[..., MujocoTwinRollout] = MujocoTwinRollout,
        **kwargs: Unpack[TwinKwargs],
    ) -> None:
        """Initialise the adapter; the MJCF is not loaded until ``connect()``."""
        kwargs["backend"] = type(self).__name__
        rollout = twin(description, **kwargs)
        # The explicit width means the shell never probes the env through openral_sim.
        SimAttachedHAL.__init__(self, rollout, description, env_action_dim=rollout.action_dim)
        self._twin = rollout  # the env, typed: the twin operations (reset_to_pose, ...)

    # ── The rollout's live state, under the names subclasses use ────────────

    @property
    def _model(self) -> mujoco.MjModel | None:
        return self._twin.model

    @property
    def _data(self) -> mujoco.MjData | None:
        return self._twin.data

    @property
    def _settle_steps(self) -> int:
        return self._twin.settle_steps

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    @property
    def _step_while_active(self) -> bool:
        """Keep the wall-time idle stepper running during a skill while wall-paced.

        Needed while an action advances only a pinned / legacy step count (else
        ``/clock`` and the cameras starve); off once one action is one control period
        (``sim_dt_per_tick_s``), so the sim clock is then driven by the runner's ticks.
        """
        return self._sim_dt_per_tick_s is None

    def connect(self) -> None:
        """Load the MJCF into a fresh twin (``SimAttachedHAL.connect``: ``env.reset`` loads).

        Unlike a scene-attached env's idempotent re-connect, the HAL Protocol refuses a
        second connect.

        Raises:
            ROSRuntimeError: Already connected.
            ROSConfigError: ``mujoco`` is missing or the MJCF cannot be loaded.
        """
        if self._connected:
            raise ROSRuntimeError(
                f"{type(self).__name__}('{self.description.name}') is already connected."
            )
        try:
            super().connect()
        except BaseException:
            # A refused connect (a time base the manifest disagrees with) leaves no
            # loaded model behind: the shell's own teardown never closes the env.
            self._release_twin()
            raise

    def disconnect(self) -> None:
        """``SimAttachedHAL.disconnect``, then release the twin (model, renderers).  Idempotent."""
        if self._connected:
            log.info("hal.disconnect", robot=self.description.name)
        super().disconnect()
        self._release_twin()

    def _release_twin(self) -> None:
        """Close the rollout: its renderers' GL contexts, model and data.

        The sim time it reached is folded into the shell's cross-reset offset first,
        so the next connect's fresh model does not rewind ``sim_time_ns``.
        """
        self._accumulate_sim_time_before_reset()
        self._twin.close()
        self._scene_cache = None  # holds the model: release it too

    # ── Pose reset ─────────────────────────────────────────────────────────────

    def reset_to_pose(self, pose: list[float]) -> None:
        """Snap the MuJoCo ``qpos`` to a specific pose and re-seed ``ctrl``.

        Used by the ``rskill_runner_node`` before the first inference tick
        when the active rSkill's manifest declares a ``starting_pose:``
        teleop home.  Without this, a VLA checkpoint can see an
        out-of-distribution joint state on step 1 and drift joints into
        their mechanical stops within a few seconds.

        Args:
            pose: Joint positions in the order of ``description.joints``
                (units are whatever the underlying MJCF expects —
                radians for revolute joints, metres for prismatic).
                ``len(pose)`` must equal ``len(description.joints)``.

        Raises:
            ROSRuntimeError: If not connected.
            ROSEStopRequested: The e-stop is latched: a teleport is motion too, refused
                like every command until ``reset_estop`` (openral#700 item 9).
            ROSConfigError: If ``pose`` length doesn't match the joint
                count.

        Example:
            >>> # hal.connect()
            >>> # hal.reset_to_pose([0.0] * len(hal.description.joints))
        """
        self._require_connected("reset_to_pose")
        if self.estop_latched:
            raise ROSEStopRequested(
                f"reset_to_pose refused on robot '{self.description.name}': the e-stop is "
                "latched (reset_estop first)."
            )
        self._twin.reset_to_pose(pose)
        log.info(
            "hal.reset_to_pose",
            robot=self.description.name,
            n_joints=len(self._twin.joint_names),
        )

    # ── Safety ─────────────────────────────────────────────────────────────────

    def estop(self) -> None:
        """Latch the stop (``SimAttachedHAL.estop``), zero ``ctrl``, tear the twin down and raise.

        The twin's difference from the shell's latch: the HAL Protocol's raising stop,
        and the twin is released at once — its GL contexts closed now, because
        ``disconnect`` would find nothing connected and a renderer left to the GC is
        finalized at interpreter teardown, after its GL library (a segfault on exit).
        ``estop_latched`` reads ``True`` until ``reset_estop``: a later ``connect``
        builds a fresh twin but stays latched, so actions are dropped and idle does
        not step until the stop is explicitly cleared.

        Raises:
            ROSEStopRequested: Always.
        """
        log.critical("hal.estop", robot=self.description.name)
        super().estop()
        data = self._data
        if data is not None:
            data.ctrl[:] = 0.0
        # Disconnected before the model goes: a reader on another thread (the bridge's
        # camera timer) is refused by the connected guard, never handed a released twin.
        self._connected = False
        self._release_twin()
        raise ROSEStopRequested(f"Emergency stop triggered on robot '{self.description.name}'.")

    # ── Manifest-driven constructor ────────────────────────────────────────────

    @staticmethod
    def _sim_kwargs_for(
        description: RobotDescription,
        *,
        settle_steps: int | None = None,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
        mjcf_path_override: str | None = None,
    ) -> TwinKwargs:
        """Translate ``description.sim`` to the kwarg dict accepted by ``__init__``.

        This is the single seam where the declarative manifest is mapped to
        the imperative constructor.  Both ``from_description`` /
        ``_init_from_description`` and any caller that wants to
        post-process the kwargs delegate here.  The ``TwinKwargs`` return type
        lets ``**kwargs`` unpack into the constructor under ``mypy --strict``.

        Raises:
            ROSConfigError: If ``description.sim`` is None, or if
                ``sim.joint_qpos_addr`` references an unknown joint, or if
                ``floating_base=True`` is paired with only one of
                ``joint_qpos_addr`` / ``joint_qvel_addr`` set.
        """
        if description.sim is None:
            raise ROSConfigError(
                f"RobotDescription '{description.name}' has no `sim` block; "
                "cannot construct a MujocoArmHAL from it. Add a `sim:` section "
                "to the manifest."
            )
        sim = description.sim
        joint_names = [j.name for j in description.joints]

        qpos_offset = 7 if sim.floating_base else 0
        qvel_offset = 6 if sim.floating_base else 0

        default_qpos = {name: qpos_offset + i for i, name in enumerate(joint_names)}
        default_qvel = {name: qvel_offset + i for i, name in enumerate(joint_names)}
        default_actuator = {name: i for i, name in enumerate(joint_names)}

        if sim.floating_base and ((sim.joint_qpos_addr is None) ^ (sim.joint_qvel_addr is None)):
            raise ROSConfigError(
                f"RobotDescription '{description.name}': floating_base=True "
                "requires sim.joint_qpos_addr and sim.joint_qvel_addr to be "
                "set together (their offsets differ — 7 vs 6 — so omitting one "
                "is almost always a bug)."
            )

        joint_qpos_addr = sim.joint_qpos_addr or default_qpos
        joint_qvel_addr = sim.joint_qvel_addr or (
            joint_qpos_addr
            if sim.joint_qvel_addr is None and not sim.floating_base
            else default_qvel
        )
        actuator_index = sim.actuator_index or default_actuator

        for name in joint_qpos_addr:
            if name not in joint_names:
                raise ROSConfigError(
                    f"sim.joint_qpos_addr references unknown joint {name!r} (known: {joint_names})"
                )

        return TwinKwargs(
            mjcf_path=mjcf_path_override or _resolve_mjcf_path(description),
            joint_qpos_addr=joint_qpos_addr,
            joint_qvel_addr=joint_qvel_addr,
            actuator_index=actuator_index,
            grippers=list(sim.grippers),
            keyframe_index=sim.keyframe_index,
            seed_ctrl_from_qpos=sim.seed_ctrl_from_qpos,
            settle_steps=(sim.settle_steps_default if settle_steps is None else settle_steps),
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )

    @classmethod
    def from_description(
        cls,
        description: RobotDescription,
        *,
        settle_steps: int | None = None,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
        mjcf_path_override: str | None = None,
    ) -> MujocoArmHAL:
        """Build a ``MujocoArmHAL`` purely from ``description.sim``.

        The recommended entry point for runtime robot selection (removes
        hardcoded constants from HAL Python files); the keyword-argument
        constructor stays available for tests / one-off overrides. Default
        joint→qpos/qvel/actuator mappings are 1:1 with ``description.joints``
        order, offset by 7/6 if ``description.sim.floating_base`` is True
        (humanoids); explicit per-joint overrides in ``description.sim`` win.

        Args:
            description: A ``RobotDescription`` whose ``sim`` field is
                populated.
            settle_steps: Optional override for the number of MuJoCo
                physics steps per ``send_action``.  Defaults to
                ``description.sim.settle_steps_default``.
            gravity_enabled: Forwarded to ``__init__``.
            staleness_limit_s: Forwarded to ``__init__``.
            mjcf_path_override: Optional absolute path that wins over
                ``description.assets.mjcf`` — useful in tests that ship a
                stripped MJCF.

        Returns:
            An un-connected ``MujocoArmHAL``.

        Raises:
            ROSConfigError: If ``description.sim`` is None, or if its
                joint-index overrides reference unknown joint names, or if
                the MJCF URI cannot be resolved.

        Example:
            >>> from openral_core import RobotDescription
            >>> # desc = RobotDescription.from_yaml("robots/ur5e/robot.yaml")
            >>> # hal = MujocoArmHAL.from_description(desc)
            >>> # hal.connect()
        """
        kwargs = cls._sim_kwargs_for(
            description,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
            mjcf_path_override=mjcf_path_override,
        )
        return cls(description, **kwargs)

    def _init_from_description(
        self,
        description: RobotDescription,
        *,
        mjcf_path: str | None = None,
        settle_steps: int | None = None,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
        body_twist_dt_s: float | None = None,
        twin: Callable[..., MujocoTwinRollout] = MujocoTwinRollout,
    ) -> None:
        """Initialise *self* from a fully-populated ``description.sim`` block.

        The seam every thin per-robot subclass (UR5e/UR10e, Franka, ALOHA,
        OpenArm, Rizon4, G1, H1, SO-100) uses to drop the boilerplate
        ``super().__init__(DESC, **MujocoArmHAL._sim_kwargs_for(DESC, ...))``
        dance — subclasses keep their typed ``__init__`` signature (so IDEs
        surface ``mjcf_path``/``settle_steps``/``gravity_enabled``/
        ``staleness_limit_s`` as the four user-tunable knobs) and forward
        straight to here. ``settle_steps=None`` (default) lets
        ``_sim_kwargs_for`` substitute
        ``description.sim.settle_steps_default``.

        Args:
            description: A ``RobotDescription`` whose ``sim`` field is
                populated.
            mjcf_path: Optional override for the MJCF file path; wins over
                ``description.assets.mjcf`` when set.
            settle_steps: Optional override for the number of MuJoCo
                physics steps per ``send_action``.
            gravity_enabled: When ``False``, gravity is zeroed at
                ``connect`` time.
            staleness_limit_s: Age above which ``read_state`` emits a
                one-shot starvation WARNING (it still returns live state).
            body_twist_dt_s: Seconds one ``BODY_TWIST`` integrates the base by;
                ``None`` is one control period (``MujocoTwinRollout``).
            twin: The rollout class (or factory) the robot's twin runs on.

        Raises:
            ROSConfigError: Propagated from ``_sim_kwargs_for`` /
                ``__init__`` for malformed ``description.sim``.
        """
        kwargs = MujocoArmHAL._sim_kwargs_for(
            description,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
            mjcf_path_override=mjcf_path,
        )
        kwargs["body_twist_dt_s"] = body_twist_dt_s
        MujocoArmHAL.__init__(self, description, twin=twin, **kwargs)
