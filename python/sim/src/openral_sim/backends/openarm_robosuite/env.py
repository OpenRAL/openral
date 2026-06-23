"""OpenArm v2 bimanual tabletop scene with direct joint-position targets.

Pairs with :mod:`._assets` (which generates the MJCF) to expose a
:class:`SimRollout` driven through the upstream OpenArm v2 MJCF position
actuators. The 16-D action chunk emitted by the pi05 OpenArm checkpoints
flows in as ``[left_j1..7, left_grip, right_j1..7, right_grip]``; the env
splits it per arm, clips each joint target to the MJCF limits, writes
``data.ctrl`` directly, and steps the MuJoCo sim.

Honest scope note
-----------------
This is the first cut. The structural integration is verified by
``tests/sim/test_openarm_scene_pnp.py`` (loads the scene, runs zero
actions, reads the observation dict, renders a frame). The closed-loop
task score still depends on the checkpoint being in-distribution for this
simple drawer/cube layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError


# `mujoco` is imported lazily inside `_build_openarm_tabletop_scene` —
# the parent `openral_sim` package eagerly registers every backend at
# import time, and a top-level `import mujoco` here puts mujoco in the
# import path of `openral_cli.main` (which the
# `test_bh_cli_import_is_light` regression guard refuses).
from openral_sim.backends.openarm_robosuite._assets import (
    compose_openarm_tabletop_mjcf,
    load_openarm_description,
)
from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult, sim_time_ns_from_mujoco_handles

if TYPE_CHECKING:
    import mujoco
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


# Per-arm joint counts. 7 revolute arm joints + 1 driven finger
# (the second finger mirrors via the upstream <equality> constraint).
_ARM_JOINT_COUNT = 7
_GRIPPER_CTRL_COUNT = 1
_ACTION_PER_ARM = 7  # 3 pos + 3 rot + 1 gripper

# Per-arm joint count *including* the driven gripper — robot.yaml's left arm
# is ``[L_j1..L_j7, L_grip]``, eight slots. The full bimanual state vector
# is therefore ``2 * _DOF_PER_ARM == 16``, but the env never hardcodes 16:
# it consults the loaded ``RobotDescription.joints`` count and (when
# available) the rSkill manifest's ``state_contract.dim`` /
# ``action_contract.dim``.
_DOF_PER_ARM = _ARM_JOINT_COUNT + _GRIPPER_CTRL_COUNT

_DEFAULT_RENDER_WIDTH = 256
_DEFAULT_RENDER_HEIGHT = 256
_GRIPPER_ENCODER_DEADBAND = 0.05
# Wrist camera local pose in the ``openarm_{side}_ee_base_link`` body frame.
# EEF frame conventions (at tabletop reset pose):
#   body +X  → world +Z (up)
#   body +Y  → world +Y (lateral, jaw opening/closing axis)
#   body -Z  → world +X (approach axis, toward workspace)
#
# Both left and right EEF bodies share identical world-frame orientation
# (bimanual kinematics are Y-mirrored, so the same body frame holds for
# both arms throughout symmetric motions).
#
# pos: 14 cm in body +X (= 14 cm above EEF in world Z) and 6 cm in body
# -Z (= 6 cm forward toward the fingertips along the approach axis).
# This clears the gripper housing and places the camera at roughly
# fingertip height when looking straight down.
#
# quat (wxyz, MuJoCo convention): computed from look/up vectors so the
# camera is orthogonal to the jaw opening direction:
#   image right = body -Y = world -Y  (jaw axis → jaws open left↔right)
#   look        = mostly body -X (world -Z, downward) + slight body -Z
#                 (world +X, forward) — no body-Y component, so the look
#                 direction is strictly perpendicular to the jaw axis.
#   image up    = body +Z (world +X, toward workspace) × slight body +X
# Quaternion derived analytically from R = [right | up | -look] column matrix.
_WRIST_CAM_LOCAL_POS = np.asarray([0.14, 0.0, -0.06], dtype=np.float64)
_WRIST_CAM_LOCAL_QUAT_WXYZ = np.asarray([0.611289, -0.355423, 0.355423, -0.611289], dtype=np.float64)
_WRIST_CAMERA_FOVY = 85.0


def _parse_xyz(raw: object, field_name: str) -> tuple[float, float, float] | None:
    """Validate a 3-vector from ``scene.backend_options``.

    Accepts ``None`` (returns ``None``), a YAML list / tuple, or a
    whitespace-separated string. Raises :class:`ROSConfigError` with a
    descriptive message for any other shape.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = raw.split()
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        raise ROSConfigError(
            f"scene.backend_options.{field_name} must be a list[float] of "
            f"length 3 or a whitespace-separated string; got {type(raw).__name__}",
        )
    if len(parts) != 3:
        raise ROSConfigError(
            f"scene.backend_options.{field_name} must have exactly 3 values; "
            f"got {len(parts)}: {parts!r}",
        )
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except (TypeError, ValueError) as exc:
        raise ROSConfigError(
            f"scene.backend_options.{field_name} entries must be numeric; got {parts!r}",
        ) from exc


_IDENTITY_QUAT_TOL: float = 1e-6


def _openarm_gripper_target(raw: float, ctrl_range: NDArray[np.float64]) -> float:
    """Map OpenArm checkpoint gripper values to the MJCF hinge endpoint.

    The pi05 OpenArm normalizer stores gripper slots in the upstream OpenArm
    encoder convention, not in MuJoCo hinge radians. Values near/below zero are
    the neutral open state in that dataset; clipping them as radians reset both
    grippers closed and hid the wrist-camera view behind the fingers.
    """
    lo = float(ctrl_range[0])
    hi = float(ctrl_range[1])
    open_pos = hi if abs(hi) >= abs(lo) else lo
    closed_pos = lo if open_pos == hi else hi
    if raw < lo or raw > hi or abs(raw) <= _GRIPPER_ENCODER_DEADBAND:
        return open_pos if raw <= 0.0 else closed_pos
    return float(np.clip(raw, lo, hi))


def _resolve_base_translation(env_cfg: SimEnvironment) -> tuple[float, float]:
    """Resolve the (z lift, x forward) MJCF base translation from ``base_pose``.

    ADR-0002 Amendment 3. ``env_cfg.base_pose`` is the only knob —
    there is no legacy ``backend_options`` fallback and no hand-tuned
    default. A YAML that omits ``base_pose`` for this scene is a
    :class:`ROSConfigError` at compose time.

    The (z, x) projection is deliberate: the underlying
    :func:`_lift_robot_bases` helper is translation-only, so non-zero
    ``y`` and non-identity quaternion are rejected. Full 6-DOF
    mounting waits on the MJCF helper learning to rotate the bases.
    """
    if env_cfg.base_pose is None:
        raise ROSConfigError(
            "openarm_tabletop_pnp scene requires `base_pose:` to be set on "
            "the SimScene YAML (ADR-0002 Amendment 3; ADR-0041 renamed "
            "SceneEnvironment → SimScene). Example: "
            "`base_pose: {xyz: [0.20, 0.0, 0.55], quat_xyzw: [0.0, 0.0, "
            "0.0, 1.0], frame_id: world}`. There is no implicit default "
            "— the previous `robot_lift_z` / `robot_forward_x` knobs and "
            "the hand-tuned defaults have been removed.",
        )

    x, y, z = env_cfg.base_pose.xyz
    if abs(y) > 1e-9:
        raise ROSConfigError(
            "openarm_tabletop_pnp scene only supports translation along x/z "
            f"(the MJCF helper does not rotate or shift the bases sideways); "
            f"got base_pose.xyz=({x}, {y}, {z}). Drop the y component or "
            "wait for the rotated-mounting follow-up.",
        )
    qx, qy, qz, qw = env_cfg.base_pose.quat_xyzw
    if (
        abs(qx) > _IDENTITY_QUAT_TOL
        or abs(qy) > _IDENTITY_QUAT_TOL
        or abs(qz) > _IDENTITY_QUAT_TOL
        or abs(qw - 1.0) > _IDENTITY_QUAT_TOL
    ):
        raise ROSConfigError(
            "openarm_tabletop_pnp scene only supports identity-rotation "
            f"base_pose (the MJCF helper is translation-only); got "
            f"quat_xyzw=({qx}, {qy}, {qz}, {qw}). Set "
            "quat_xyzw=[0, 0, 0, 1] or wait for the rotated-mounting "
            "follow-up.",
        )
    return float(z), float(x)


def _resolve_state_dim(
    *,
    weights_uri: str | None,
    fallback: int,
) -> int:
    """Resolve the proprioception / action vector dimension for the rollout.

    Source of truth precedence:

    1. ``manifest.state_contract.dim`` — the rSkill's declared state
       vector width (per CLAUDE.md §6.4 every rSkill that wants to
       drive the dataset bridge must set this).
    2. ``manifest.action_contract.dim`` — used as a cross-check when
       both are present; the env asserts they agree because this
       backend writes the policy's action vector directly into the
       observation-shaped ``state``.
    3. ``fallback`` — usually ``len(RobotDescription.joints)``. Used
       when no rSkill manifest is resolvable (e.g. a smoke-test path
       with ``weights_uri="mock://noop"``).

    Args:
        weights_uri: The :attr:`VLASpec.weights_uri` from the eval YAML.
            Only bare rSkill references are inspected; explicit-scheme
            URIs (``hf://``, ``local://``, etc.) drop to ``fallback``.
        fallback: Dimension to return when no manifest is available
            or the manifest does not declare contracts.

    Returns:
        A positive integer state dimension.

    Raises:
        ROSConfigError: If the rSkill manifest declares
            ``state_contract.dim`` and ``action_contract.dim`` with
            different values (the env feeds the action vector back
            into the proprioception slot, so they must match).
    """
    if not weights_uri or weights_uri.startswith(
        ("hf://", "local://", "file://", "http://", "https://")
    ):
        return fallback
    try:
        from openral_rskill.loader import load_rskill_manifest
    except ImportError:
        return fallback
    try:
        manifest = load_rskill_manifest(weights_uri)
    except Exception:
        # The downstream pipeline will surface a more useful error
        # when it tries to actually load the rSkill; the env defaults
        # to the robot manifest's joint count rather than re-raising
        # so that test fixtures without network access stay green.
        return fallback

    state_dim = getattr(getattr(manifest, "state_contract", None), "dim", None)
    action_dim = getattr(getattr(manifest, "action_contract", None), "dim", None)
    if state_dim is not None and action_dim is not None and state_dim != action_dim:
        raise ROSConfigError(
            f"rSkill {weights_uri!r} declares state_contract.dim={state_dim} but "
            f"action_contract.dim={action_dim}; the openarm_tabletop_pnp backend "
            "feeds the action vector through the observation.state slot, so they "
            "must match.",
        )
    if state_dim is not None:
        return int(state_dim)
    if action_dim is not None:
        return int(action_dim)
    return fallback


def _resolve_initial_pose_from_rskill(
    weights_uri: str | None,
    action_layout: str,
    state_dim: int,
) -> NDArray[np.float32] | None:
    """Resolve the initial episode pose for this rSkill.

    Reads ``manifest.starting_pose`` — an explicit ``state_dim``-long
    list in ``rskill.yaml``, in the checkpoint's
    ``action_feature_names`` order, units radians. Maintainer
    ground-truth (a teleop-recorded "episode 0" pose, or an
    analytically tuned centre such as the normalizer's
    ``observation.state.q50``).

    Returns ``None`` (the env falls back to its default elbow-bent
    home pose) when the rskill does not declare ``starting_pose`` —
    no on-the-fly Hub fetch, no implicit q50 lookup. If the
    checkpoint maintainer wants a specific start pose, they declare
    it. Otherwise the env's default applies.

    The returned vector is always in robot.yaml left-first order
    (``[L_j1..L_grip, R_j1..R_grip]``). ``action_layout`` describes
    the rSkill side; the function reorders for the caller so the env
    never has to think about layout swap.
    """
    if not weights_uri or weights_uri.startswith(
        ("hf://", "local://", "file://", "http://", "https://")
    ):
        return None
    try:
        from openral_rskill.loader import load_rskill_manifest
    except ImportError:
        return None
    try:
        manifest = load_rskill_manifest(weights_uri)
    except Exception:
        return None

    manifest_pose = getattr(manifest, "starting_pose", None)
    if manifest_pose is None or len(manifest_pose) != state_dim:
        return None
    try:
        pose = np.asarray(manifest_pose, dtype=np.float32)
    except Exception:  # reason: malformed YAML — surface as "no pose"
        return None

    # Reorder from the rSkill's layout to robot.yaml left-first. The
    # bimanual layout splits the vector in two equal halves; per the
    # cross-check in :func:`_resolve_state_dim`, ``state_dim`` must be
    # the total bimanual width so each arm's slice is ``state_dim // 2``.
    if action_layout == "right_first":
        half = state_dim // 2
        # rSkill order [R..R_grip, L..L_grip] → robot [L..L_grip, R..R_grip]
        pose = np.concatenate([pose[half:state_dim], pose[0:half]])
    return pose.astype(np.float32)


@dataclass
class _ArmHandles:
    """Bookkeeping for one arm's actuator / joint indices in the composed MJCF.

    The composed MJCF keeps the upstream OpenArm v2 ``<position>``
    actuators (per-joint kp/kv tuned by the upstream MJCF class
    definitions). The env step writes joint position targets directly
    to ``data.ctrl`` — same contract the live
    ``OpenArmMujocoHAL.send_action`` path uses in
    ``packages/openral_hal_openarm/`` — so the sim's force model
    exactly matches the live ROS launch.
    """

    side: str  # "left" | "right"
    arm_actuator_ids: list[int]  # 7 position actuator ids for the arm joints
    arm_qpos_ix: list[int]
    arm_qvel_ix: list[int]
    grip_actuator_id: int


def _arm_joint_names_for_side(
    side: str,
    description: RobotDescription | None,
) -> list[str]:
    """Return the 7 MJCF arm joint names for ``side``.

    Reads :attr:`~openral_core.JointSpec.sim_joint_name` off every
    ``robot.yaml`` joint whose ``name`` starts with ``"{side}_joint"``,
    falling back to the legacy hardcoded ``openarm_{side}_joint{i}``
    pattern when no description is passed (lets hermetic tests build
    the env without a description). ADR-0025.
    """
    if description is None:
        return [f"openarm_{side}_joint{i}" for i in range(1, _ARM_JOINT_COUNT + 1)]
    prefix = f"{side}_joint"
    sim_names: list[str] = []
    for spec in description.joints:
        if not spec.name.startswith(prefix):
            continue
        if spec.sim_joint_name is None:
            continue
        sim_names.append(spec.sim_joint_name)
        if len(sim_names) == _ARM_JOINT_COUNT:
            break
    if len(sim_names) != _ARM_JOINT_COUNT:
        # Manifest is incomplete — fall back loudly. The composer
        # always passes a description in production; an incomplete
        # robot.yaml is a real misconfiguration, not "best effort".
        raise ROSConfigError(
            f"_arm_joint_names_for_side: robot description has "
            f"{len(sim_names)} sim_joint_name overrides for side={side!r}; "
            f"expected {_ARM_JOINT_COUNT}. Populate `sim_joint_name` on every "
            f"`{prefix}*` joint in robots/openarm/robot.yaml."
        )
    return sim_names


def _build_arm_handles(
    model: mujoco.MjModel,
    side: str,
    description: RobotDescription | None = None,
) -> _ArmHandles:
    """Look up actuator / joint indices for one arm in the composed MJCF.

    Joint names come from the per-joint ``sim_joint_name`` overrides
    in :class:`~openral_core.RobotDescription` (ADR-0025) — falling
    back to the previous hardcoded ``openarm_{side}_joint{i}`` strings
    when no description is passed (legacy / hermetic-test paths).

    Actuator names follow the upstream OpenArm v2 ``<position>``
    convention (``{side}_joint{i}_ctrl`` for the 7 arm joints,
    ``{side}_finger1_ctrl`` for the driven finger). The schema doesn't
    carry sim-actuator overrides today, so those remain in code; if a
    future fork renames actuators, add a ``sim_actuator_name`` field
    on :class:`~openral_core.JointSpec`.

    Args:
        model: Compiled MuJoCo model (composed with
            ``strip_actuators=False`` so the upstream position
            actuators survive).
        side: ``"left"`` or ``"right"``.
        description: Loaded ``robots/openarm/robot.yaml`` — used to
            source ``sim_joint_name`` overrides.
    """
    import mujoco

    arm_joint_names = _arm_joint_names_for_side(side, description)
    arm_actuator_names = [f"{side}_joint{i}_ctrl" for i in range(1, _ARM_JOINT_COUNT + 1)]
    grip_actuator_name = f"{side}_finger1_ctrl"

    arm_actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in arm_actuator_names
    ]
    qpos_ix = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in arm_joint_names
    ]
    qvel_ix = [
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in arm_joint_names
    ]
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, grip_actuator_name)
    if any(i < 0 for i in [*arm_actuator_ids, *qpos_ix, *qvel_ix, grip_id]):
        raise ROSConfigError(
            f"OpenArm tabletop scene: missing actuator / joint for side={side!r}. "
            f"Check _assets.py against the upstream MJCF — composer must run "
            f"with strip_actuators=False so the upstream <position> actuators "
            f"named {arm_actuator_names!r} + {grip_actuator_name!r} survive. "
            f"Joint names resolved (from description.sim_joint_name): "
            f"{arm_joint_names!r}.",
        )

    return _ArmHandles(
        side=side,
        arm_actuator_ids=arm_actuator_ids,
        arm_qpos_ix=qpos_ix,
        arm_qvel_ix=qvel_ix,
        grip_actuator_id=grip_id,
    )


@dataclass
class _OpenArmTabletopRollout:
    """Bimanual OpenArm tabletop scene driven by MJCF position actuators."""

    scene: SceneSpec
    task: TaskSpec
    _model: mujoco.MjModel
    _sim: Any  # robosuite.utils.binding_utils.MjSim
    _left: _ArmHandles
    _right: _ArmHandles
    _instruction: str
    _renderer: mujoco.Renderer | None = None
    _max_steps: int = 500
    _step_count: int = 0
    _last_pixels: NDArray[np.uint8] | None = None
    _image_keys: tuple[str, str, str] = ("top", "wrist_left", "wrist_right")
    _render_width: int = _DEFAULT_RENDER_WIDTH
    _render_height: int = _DEFAULT_RENDER_HEIGHT
    # ``"left_first"`` (default — matches robots/openarm/robot.yaml and
    # mddoai/pi05_openarm_vast's ``action_feature_names``) or
    # ``"right_first"`` (matches yuto-urushima / AdrianLlopart pickplace
    # checkpoints whose ``action_feature_names`` are right-first). Set
    # via ``scene.backend_options.action_layout`` in the YAML. Applied
    # symmetrically to both the inbound action and the outbound state
    # observation so the env's I/O contract matches the policy's order.
    _action_layout: str = "left_first"
    # ``"radians"`` (the MuJoCo qpos / ctrl native unit and the
    # robots/openarm/robot.yaml convention) or ``"degrees"`` (the
    # LeRobot OpenArm dataset convention — yuto-urushima / AdrianLlopart
    # pickplace checkpoints record state in degrees and emit actions
    # in degrees; verified against the on-disk normalizer pack's
    # observation.state.q50 elbow value ≈90 — clearly degrees). When
    # ``"degrees"``, the env converts qpos → state (rad → deg) before
    # handing observation to the policy and action → ctrl (deg → rad)
    # before writing to MuJoCo. Grippers are pass-through — the policy
    # encodes them in a custom motor-encoder unit (q01..q99 ≈ -50..-1)
    # which is neither radians nor degrees, and the actuator ctrlrange
    # clips into the joint's valid range. Set via
    # ``scene.backend_options.joint_units``.
    _joint_units: str = "radians"
    # Initial ``state_dim``-D arm pose in robot.yaml left-first order
    # [L_j1..L_j7, L_grip, R_j1..R_j7, R_grip] — units always radians,
    # clipped to actuator ctrlrange before write. Set by the env factory
    # from the rSkill's ``starting_pose`` so the episode starts at the
    # policy's training-distribution centre instead of the MJCF default
    # zero pose. ``None`` falls back to the "elbows at π/2" home pose
    # (matches the live ``sim_e2e.launch.py`` HAL).
    _initial_pose_robot_order: NDArray[np.float32] | None = None
    # Total state / action vector width, sourced from the rSkill
    # manifest's ``state_contract.dim`` (or ``action_contract.dim``)
    # when the eval YAML's ``vla.weights_uri`` points at an rSkill,
    # falling back to ``len(RobotDescription.joints)`` otherwise.
    # Defaults to ``2 * _DOF_PER_ARM`` (i.e. 16 for the OpenArm v2) for
    # dataclasses constructed in tests without an explicit value.
    _state_dim: int = 2 * _DOF_PER_ARM

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            np.random.seed(seed)
        # robosuite wraps MjData in its own proxy; use the wrapper's
        # reset / forward so the proxy stays in sync. Raw mujoco.mj_*
        # functions reject the wrapper.
        self._sim.reset()
        # Apply the initial pose. Precedence:
        #   1. ``_initial_pose_robot_order`` (set by the factory from the
        #      rSkill's normalizer state.q50 — the training distribution
        #      centre so the policy's first inference lands on a familiar
        #      observation).
        #   2. Elbows at +π/2 fallback — same elbow-bent home the live
        #      ``sim_e2e.launch.py`` HAL uses.
        if self._initial_pose_robot_order is not None:
            pose = self._initial_pose_robot_order
            half = self._state_dim // 2
            for handles, slice_lo in ((self._left, 0), (self._right, half)):
                for slot, qa in enumerate(handles.arm_qpos_ix):
                    # Clip per-joint to the actuator's ctrlrange so a
                    # q50 just past the hardware limit (training data
                    # from a wider real-hardware range than the MJCF
                    # joint range) doesn't error out.
                    aid = handles.arm_actuator_ids[slot]
                    lo, hi = self._model.actuator_ctrlrange[aid]
                    self._sim.data.qpos[qa] = float(np.clip(pose[slice_lo + slot], lo, hi))
                # Gripper qpos sits immediately after the arm joints.
                self._sim.data.qpos[handles.arm_qpos_ix[-1] + 1] = float(
                    _openarm_gripper_target(
                        float(pose[slice_lo + _ARM_JOINT_COUNT]),
                        self._model.actuator_ctrlrange[handles.grip_actuator_id],
                    ),
                )
        else:
            self._sim.data.qpos[self._left.arm_qpos_ix[3]] = float(np.pi / 2)
            self._sim.data.qpos[self._right.arm_qpos_ix[3]] = float(np.pi / 2)
        # Seed ``ctrl`` from the new ``qpos`` so the upstream position
        # actuators hold the home pose on step 0 instead of pulling
        # every joint back to zero (the actuator default ctrl).
        for h in (self._left, self._right):
            for qa, aid in zip(h.arm_qpos_ix, h.arm_actuator_ids, strict=True):
                self._sim.data.ctrl[aid] = float(self._sim.data.qpos[qa])
            self._sim.data.ctrl[h.grip_actuator_id] = float(
                self._sim.data.qpos[h.arm_qpos_ix[-1] + 1],
            )
        self._sim.forward()
        self._step_count = 0
        return self._observation()

    def step(self, action: NDArray[np.float32]) -> StepResult:
        """Apply a ``state_dim``-D joint-position-target action.

        Layout: ``[left_j1..7, left_grip, right_j1..7, right_grip]``
        (the URDF / ``robots/openarm/robot.yaml`` order — the same
        order the live ``OpenArmMujocoHAL.send_action`` consumes and
        the same order the ``openral sim run`` shim aligns the policy
        output to before this method is reached).

        The expected width is ``self._state_dim``, derived at backend
        init from the rSkill's ``action_contract.dim`` (or the loaded
        :class:`RobotDescription`'s joint count as the fallback).
        Targets are written *directly* to the upstream OpenArm v2
        ``<position>`` actuators (``{side}_joint{i}_ctrl`` +
        ``{side}_finger1_ctrl``), inheriting the upstream MJCF's
        per-joint kp/kv tuning. No custom PD law, no OSC, no torque
        rewrite — the same control contract the live ROS launch's
        HAL uses, so both paths converge on identical dynamics.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (self._state_dim,):
            raise ROSConfigError(
                f"openarm_tabletop_pnp expects a {self._state_dim}-D joint-position "
                f"action (from the rSkill's action_contract.dim / robot manifest's "
                f"joint count); got shape {a.shape}",
            )

        # The checkpoint's ``action_feature_names`` declares the order:
        #   left_first  : [L_j1..7, L_grip, R_j1..7, R_grip]
        #   right_first : [R_j1..7, R_grip, L_j1..7, L_grip]
        # The YAML's ``scene.backend_options.action_layout`` selects.
        half = self._state_dim // 2
        if self._action_layout == "right_first":
            right_targets = a[0:half]
            left_targets = a[half : self._state_dim]
        else:
            left_targets = a[0:half]
            right_targets = a[half : self._state_dim]

        for handles, targets in ((self._left, left_targets), (self._right, right_targets)):
            arm_targets = np.asarray(targets[:_ARM_JOINT_COUNT], dtype=np.float64)
            grip_target = float(targets[_ARM_JOINT_COUNT])
            # Convert policy-space arm action → MuJoCo qpos units. The
            # LeRobot OpenArm dataset records actions in degrees; MuJoCo
            # ctrl is radians. The gripper is in a per-encoder motor
            # unit that is neither — pass it through and let ctrlrange
            # clipping handle it.
            if self._joint_units == "degrees":
                arm_targets = np.radians(arm_targets)
            # Clip to the actuator's ctrlrange so the position actuator
            # never gets a setpoint past its mechanical limit. The
            # upstream MJCF's ctrlrange == the joint's position_limits;
            # MuJoCo silently clips internally, but writing pre-clipped
            # makes the read-back ctrl == the actual commanded target.
            lo = self._model.actuator_ctrlrange[handles.arm_actuator_ids, 0]
            hi = self._model.actuator_ctrlrange[handles.arm_actuator_ids, 1]
            self._sim.data.ctrl[handles.arm_actuator_ids] = np.clip(arm_targets, lo, hi)
            self._sim.data.ctrl[handles.grip_actuator_id] = float(
                _openarm_gripper_target(
                    grip_target,
                    self._model.actuator_ctrlrange[handles.grip_actuator_id],
                ),
            )

        self._sim.step()
        self._step_count += 1

        return StepResult(
            observation=self._observation(),
            reward=0.0,  # Reward shaping is a follow-up.
            terminated=False,
            truncated=self._step_count >= self._max_steps,
            info={"drawer_pos": float(self._drawer_position())},
        )

    def render(self) -> NDArray[np.uint8] | None:
        if self._last_pixels is None:
            return None
        return self._last_pixels.copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ helpers

    def _observation(self) -> Observation:
        images = self._render_cameras()
        self._last_pixels = images[self._image_keys[0]]
        return {
            "images": images,
            "state": self._proprio_state(),
            "task": self._instruction,
        }

    def _render_cameras(self) -> dict[str, NDArray[np.uint8]]:
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self._model,
                width=self._render_width,
                height=self._render_height,
            )
        out: dict[str, NDArray[np.uint8]] = {}
        # Per ADR-0069 the sensor / output keys are canonical (``top`` /
        # ``wrist_left`` / ``wrist_right``); the openarm MJCF composer renames
        # the upstream wrist cameras to match, so the same name is used for both
        # the renderer lookup and the output dict.
        for cam_name in self._image_keys:
            if cam_name in ("wrist_left", "wrist_right"):
                self._update_dynamic_wrist_camera(mujoco, cam_name)
            # mujoco.Renderer wants the raw mujoco.MjData, not robosuite's wrapper.
            self._renderer.update_scene(self._sim.data._data, camera=cam_name)
            out[cam_name] = self._renderer.render().copy()
        return out

    def _update_dynamic_wrist_camera(self, mujoco: Any, cam_name: str) -> None:
        """Fix the wrist camera to be properly body-parented to the EEF link.

        The upstream OpenArm MJCF places the wrist cameras inside the gripper
        shell at fingertip level with only a -90° Z-rotation, which renders
        mostly gripper geometry from an uninformative angle. This function
        overrides position and orientation **in the EEF body-local frame**
        using ``_WRIST_CAM_LOCAL_POS`` / ``_WRIST_CAM_LOCAL_QUAT_WXYZ`` so
        the camera sits 14 cm above the wrist and 6 cm forward, looking down
        at the workspace with the jaw axis horizontal in the image frame.
        The camera stays rigidly attached to the EEF body rather than floating
        at a world-space offset. Called once per render step so the camera
        tracks the live EEF pose.

        Both left and right cameras use the same quaternion because the
        bimanual EEF bodies maintain identical world-frame orientations
        throughout symmetric motions (Y-mirrored kinematics → same body frame).
        """
        side = cam_name.split("_", 1)[1]  # "wrist_left" → "left"
        body_name = f"openarm_{side}_ee_base_link"
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if cam_id < 0 or body_id < 0:
            raise ROSConfigError(
                f"openarm_tabletop_pnp could not resolve {cam_name!r} camera "
                f"or {body_name!r} body in the composed MJCF.",
            )
        # Keep the camera body-parented (follows the EEF through the trajectory).
        self._model.cam_bodyid[cam_id] = body_id
        self._model.cam_pos[cam_id] = _WRIST_CAM_LOCAL_POS.copy()
        self._model.cam_quat[cam_id] = _WRIST_CAM_LOCAL_QUAT_WXYZ.copy()
        self._model.cam_fovy[cam_id] = _WRIST_CAMERA_FOVY
        mujoco.mj_forward(self._model, self._sim.data._data)

    def _proprio_state(self) -> NDArray[np.float32]:
        """``state_dim``-D proprioception in the policy's expected layout + units.

        ``state_dim`` is ``2 * _DOF_PER_ARM`` for any rSkill whose
        ``state_contract.dim`` matches the robot's joint inventory.

        Layout follows ``_action_layout``:

        * ``"left_first"`` → ``[L_j1..7, L_grip, R_j1..7, R_grip]``
          (the robot.yaml / mddoai pi05_openarm_vast convention).
        * ``"right_first"`` → ``[R_j1..7, R_grip, L_j1..7, L_grip]``
          (the yuto-urushima / AdrianLlopart pickplace convention,
          matching their ``config.json action_feature_names``).

        Units follow ``_joint_units``: ``"radians"`` returns qpos as-is;
        ``"degrees"`` converts arm joints (rad → deg) while leaving the
        gripper qpos untouched. The gripper qpos is in MuJoCo radians
        (joint range [0, 0.7854] for left, [-0.7854, 0] for right) but
        the LeRobot OpenArm dataset records the gripper in a custom
        motor-encoder unit — there is no closed-form conversion, so the
        env passes it through and downstream consumers (normalizer +
        clipping) handle the unit mismatch.
        """
        # Assume the finger qpos sits immediately after joint7's qpos
        # slot. Verified by the upstream MJCF body order; the smoke test
        # asserts this invariant so a future MJCF reorder fails loudly.
        left = np.empty(_ARM_JOINT_COUNT + 1, dtype=np.float32)
        left[:_ARM_JOINT_COUNT] = self._sim.data.qpos[self._left.arm_qpos_ix]
        left[_ARM_JOINT_COUNT] = float(
            self._sim.data.qpos[self._left.arm_qpos_ix[-1] + 1],
        )
        right = np.empty(_ARM_JOINT_COUNT + 1, dtype=np.float32)
        right[:_ARM_JOINT_COUNT] = self._sim.data.qpos[self._right.arm_qpos_ix]
        right[_ARM_JOINT_COUNT] = float(
            self._sim.data.qpos[self._right.arm_qpos_ix[-1] + 1],
        )

        if self._joint_units == "degrees":
            left[:_ARM_JOINT_COUNT] = np.degrees(left[:_ARM_JOINT_COUNT])
            right[:_ARM_JOINT_COUNT] = np.degrees(right[:_ARM_JOINT_COUNT])

        if self._action_layout == "right_first":
            return np.concatenate([right, left]).astype(np.float32)
        return np.concatenate([left, right]).astype(np.float32)

    def _drawer_position(self) -> float:
        import mujoco

        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        return float(self._sim.data.qpos[self._model.jnt_qposadr[jid]])

    def mujoco_handles(self) -> tuple[mujoco.MjModel, Any]:
        """Expose (model, raw data) for ``openral sim run --view`` passive viewer.

        The viewer uses ``mujoco.viewer.launch_passive(model, data)`` which
        wants the raw mujoco MjData, not robosuite's MjSim wrapper. The
        wrapper's ``.data._data`` attribute is the underlying object.
        """
        return self._model, self._sim.data._data

    def sim_time_ns(self) -> int | None:
        """Elapsed MuJoCo sim time in ns (ADR-0048 Phase 1).

        Reads ``MjData.time`` off :meth:`mujoco_handles`. Monotonic within an
        episode; rewinds on ``reset``.
        """
        return sim_time_ns_from_mujoco_handles(self.mujoco_handles())

    @property
    def action_dim(self) -> int:
        """Flat action width the env's ``step`` accepts (bimanual state_dim, e.g. 16).

        ADR-0034 follow-up — ``SimAttachedHAL._probe_env_action_dim`` reads this so a
        deploy-sim action (``send_action`` / ``idle_step``) is sized to the OpenArm's
        joint-position width (``state_dim``, derived from the rSkill's
        ``action_contract.dim`` / manifest joint count) rather than the
        robosuite-mobile-manipulator fallback (11). Without it the probe missed this
        native backend (it exposes no upstream ``action_dim``) and the next
        ``env.step`` raised a width mismatch.
        """
        return self._state_dim


def _build_openarm_tabletop_scene(env_cfg: SimEnvironment) -> _OpenArmTabletopRollout:
    """SCENES factory for ``openarm_tabletop_pnp``.

    Reads optional ``scene.backend_options`` keys:
        ``render_width`` / ``render_height`` (default 256 each).
        ``max_steps`` (default 500).
    """
    # Run the install-plan preflight BEFORE the mujoco / robosuite imports
    # so a fresh venv (post `uv sync --all-packages` without `--group
    # robocasa`) gets the Rich license banner + typer.confirm prompt
    # instead of a bare ModuleNotFoundError. Mirrors the robocasa /
    # libero / metaworld / aloha / maniskill3 scene factories.
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("openarm_robosuite")

    import mujoco
    from robosuite.utils.binding_utils import MjSim

    # Read MJCF compose knobs from the scene config. Defaults mirror
    # the live ROS launch (``sim_e2e.launch.py``) so ``openral sim run``
    # and ``ros2 launch ... sim_e2e.launch.py`` build the SAME
    # scene — bases lifted above the table top, shifted forward into
    # the workspace, white skybox so the dataset's lighting reads
    # cleanly.
    opts_compose: dict[str, Any] = dict(env_cfg.scene.backend_options or {})

    # Robot mounting pose: required, must be set via `base_pose:` in the
    # SimScene YAML (ADR-0002 Amendment 3; ADR-0041 renamed
    # SceneEnvironment → SimScene). No legacy knobs, no implicit defaults.
    lift_z, forward_x = _resolve_base_translation(env_cfg)
    white_bg = bool(opts_compose.get("white_background", True))
    top_camera_pos = _parse_xyz(opts_compose.get("top_camera_pos"), "top_camera_pos")
    top_camera_target = _parse_xyz(opts_compose.get("top_camera_target"), "top_camera_target")
    top_camera_fovy_raw = opts_compose.get("top_camera_fovy")
    top_camera_fovy = float(top_camera_fovy_raw) if top_camera_fovy_raw is not None else None

    # Load the OpenArm v2 manifest once and feed it to the composer
    # (so the actuator inventory + ``scene_defaults.top_camera`` come
    # from the description, not module-level hardcodes) and to the
    # state-dim resolver (so the policy's shape contract drives the
    # action / observation width).
    robot_description = load_openarm_description()

    # Compose with strip_actuators=False so the upstream OpenArm v2
    # ``<position>`` actuators (per-joint kp/kv tuned classes) survive
    # — same contract the live ``OpenArmMujocoHAL.send_action`` uses
    # via the sim_e2e.launch.py path. The custom PD / OSC paths
    # are deliberately gone; this env writes position targets straight
    # to ``data.ctrl`` and lets the MJCF's PD law do the work.
    xml, meshdir = compose_openarm_tabletop_mjcf(
        strip_actuators=False,
        robot_lift_z=lift_z,
        robot_forward_x=forward_x,
        white_background=white_bg,
        top_camera_pos=top_camera_pos,
        top_camera_target=top_camera_target,
        top_camera_fovy=top_camera_fovy,
        robot_description=robot_description,
    )
    # Drop the composed XML next to the upstream meshdir so the relative
    # ``meshdir="assets"`` resolves at compile time without copying meshes.
    generated_path = meshdir.parent / "openarm_tabletop_pnp_generated.xml"
    generated_path.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(generated_path))
    sim = MjSim(model)

    left = _build_arm_handles(model, "left", robot_description)
    right = _build_arm_handles(model, "right", robot_description)

    opts: dict[str, Any] = dict(env_cfg.scene.backend_options or {})
    render_w = int(opts.get("render_width", _DEFAULT_RENDER_WIDTH))
    render_h = int(opts.get("render_height", _DEFAULT_RENDER_HEIGHT))
    max_steps = int(opts.get("max_steps", 500))
    action_layout = str(opts.get("action_layout", "left_first")).lower()
    if action_layout not in ("left_first", "right_first"):
        raise ROSConfigError(
            f"scene.backend_options.action_layout must be 'left_first' or "
            f"'right_first'; got {action_layout!r}",
        )
    joint_units = str(opts.get("joint_units", "radians")).lower()
    if joint_units not in ("radians", "degrees"):
        raise ROSConfigError(
            f"scene.backend_options.joint_units must be 'radians' or "
            f"'degrees'; got {joint_units!r}",
        )

    instruction = env_cfg.task.instruction or "pick the red cube and place it in the drawer"

    # Resolve the policy's state / action dimension. The rSkill manifest
    # is the source of truth (CLAUDE.md §6.4); the robot manifest's
    # joint count is the fallback for test paths that do not point at
    # an rSkill (e.g. ``mock://noop``).
    weights_uri = getattr(env_cfg.vla, "weights_uri", None)
    state_dim = _resolve_state_dim(
        weights_uri=weights_uri,
        fallback=len(robot_description.joints),
    )

    # Resolve the per-rSkill q50 home pose. The rSkill's normalizer
    # stats centre the training distribution; starting from there
    # means the policy's first observation is in-distribution by
    # construction. Falls back to the default elbow-bent home when the
    # rSkill repo doesn't ship the normalizer pack.
    initial_pose = _resolve_initial_pose_from_rskill(
        weights_uri=weights_uri,
        action_layout=action_layout,
        state_dim=state_dim,
    )

    return _OpenArmTabletopRollout(
        scene=env_cfg.scene,
        task=env_cfg.task,
        _model=model,
        _sim=sim,
        _left=left,
        _right=right,
        _instruction=instruction,
        _render_width=render_w,
        _render_height=render_h,
        _max_steps=max_steps,
        _action_layout=action_layout,
        _joint_units=joint_units,
        _initial_pose_robot_order=initial_pose,
        _state_dim=state_dim,
    )


_SCENE_ID = "openarm_tabletop_pnp"
SCENES.register(_SCENE_ID)(_build_openarm_tabletop_scene)
