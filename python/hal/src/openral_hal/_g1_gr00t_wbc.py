"""NVIDIA GR00T Decoupled WBC lower-body controller for the G1 sim HAL (sim-only S0).

Port of the upstream MuJoCo runner
``decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py`` of
``NVlabs/GR00T-WholeBodyControl`` at the pinned commit below: two ONNX
networks (``Balance`` for a zero velocity command, ``Walk`` otherwise) own the
15 lower-body joints (12 legs + 3 waist) of the 29-DoF G1 at 50 Hz, and a
joint-space PD law ``tau = kp * (q* - q) - kd * dq`` turns their targets into
torques every physics step. The 14 arm joints stay free for an S1 skill.

Licence lineage (CLAUDE.md §1.9): the runner code is Apache-2.0; the ONNX
weights are released under the **NVIDIA Open Model License** (commercial use
permitted, ``RSkillLicensePosture.NVIDIA_OPEN_MODEL``), whose text ships next to
them upstream and is fetched and verified alongside them here. Nothing is
bundled in this repository: the files are downloaded at first use into
``$OPENRAL_CACHE_DIR`` from a pinned commit and SHA-256-verified, fail-closed.

This is a **simulation-only** controller. It must never be wired to a real
HAL: the production G1 S0 is a C++ ``ros2_control`` controller (CLAUDE.md §3
dual-system pattern).
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import RSkillLicensePosture

from openral_hal._g1_walking import _download, _InferenceSession

if TYPE_CHECKING:
    import mujoco

log = structlog.get_logger(__name__)

__all__ = [
    "GR00T_WBC_LICENSE_POSTURE",
    "GR00T_WBC_PINNED_SHA",
    "Gr00tWbcAssets",
    "Gr00tWbcController",
    "ensure_gr00t_wbc_assets",
    "gr00t_wbc_observation",
    "position_actuators_to_torque_motors",
]

#: Pinned ``NVlabs/GR00T-WholeBodyControl`` commit the weights are fetched from.
GR00T_WBC_PINNED_SHA = "1e1ec4f5f27e6c3e03048f693bd7c60cce04afda"
#: Licence posture of the ONNX weights (code is Apache-2.0).
GR00T_WBC_LICENSE_POSTURE = RSkillLicensePosture.NVIDIA_OPEN_MODEL

_POLICY_DIR = "decoupled_wbc/sim2mujoco/resources/robots/g1/policy"
# The ONNX files are Git LFS objects: the raw.githubusercontent URL serves the
# pointer, media.githubusercontent the bytes. The licence text is a plain file.
_LFS_ROOT = f"https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/{GR00T_WBC_PINNED_SHA}"
_RAW_ROOT = (
    f"https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/{GR00T_WBC_PINNED_SHA}"
)
_WALK_ONNX = f"{_POLICY_DIR}/GR00T-WholeBodyControl-Walk.onnx"
_BALANCE_ONNX = f"{_POLICY_DIR}/GR00T-WholeBodyControl-Balance.onnx"
_LICENSE = f"{_POLICY_DIR}/NVIDIA%20Open%20Model%20License"
_ASSETS: dict[str, tuple[str, str]] = {
    _WALK_ONNX: (_LFS_ROOT, "7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407"),
    _BALANCE_ONNX: (_LFS_ROOT, "f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18"),
    _LICENSE: (_RAW_ROOT, "a0c4f6a35dd2c858c86069d84140052831820e78f3074a2fb758bb4156b66994"),
}

# ── Policy contract (upstream g1_gear_wbc.yaml @ the pinned commit) ──────────
#: Joints the policy owns: menagerie indices 0..14 (12 legs + waist yaw/roll/pitch).
S0_JOINT_COUNT = 15
_N_JOINTS = 29
_OBS_DIM = 86
_HISTORY_LEN = 6
_INPUT_DIM = _OBS_DIM * _HISTORY_LEN  # 516
#: Below this raw ``||(vx, vy, wz)||`` the Balance network runs, else Walk (upstream ``<= 0.05``).
WALK_COMMAND_THRESHOLD = 0.05
_ACTION_SCALE = 0.25
_CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
_ANG_VEL_SCALE = 0.5
_DOF_VEL_SCALE = 0.05
#: Upstream ``height_cmd`` default (m) and ``rpy_cmd`` default (rad).
DEFAULT_BASE_HEIGHT_M = 0.74
_DEFAULT_LEG = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0)
#: Stance the policy's action is an offset from (legs per side, then waist).
S0_DEFAULT_ANGLES: NDArray[np.float64] = np.array(
    [*_DEFAULT_LEG, *_DEFAULT_LEG, 0.0, 0.0, 0.0], dtype=np.float64
)
_DEFAULT_Q29: NDArray[np.float64] = np.concatenate([S0_DEFAULT_ANGLES, np.zeros(14)])
#: PD gains for the policy-owned joints (upstream sim ``kps`` / ``kds``).
S0_KP: NDArray[np.float64] = np.array(
    [150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250], dtype=np.float64
)
S0_KD: NDArray[np.float64] = np.array(
    [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5], dtype=np.float64
)
#: Arm PD gains (per side: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) —
#: GR00T's real-robot ``MOTOR_KP`` / ``MOTOR_KD`` (``g1_29dof_gear_wbc.yaml``).
ARM_KP: NDArray[np.float64] = np.array([100, 100, 40, 40, 20, 20, 20] * 2, dtype=np.float64)
ARM_KD: NDArray[np.float64] = np.array([5, 5, 2, 2, 2, 2, 2] * 2, dtype=np.float64)
#: Policy update rate (upstream: 5 ms physics, decimation 4).
CONTROL_RATE_HZ = 50.0


@dataclass(frozen=True)
class Gr00tWbcAssets:
    """Verified local paths of the pinned GR00T WBC files."""

    walk_onnx: Path
    balance_onnx: Path
    license_text: Path


def ensure_gr00t_wbc_assets() -> Gr00tWbcAssets:
    """Fetch (or reuse) the pinned, SHA-256-verified GR00T WBC weights and licence.

    Files land in ``$OPENRAL_CACHE_DIR/g1_gr00t_wbc/<sha>/`` (default
    ``~/.cache/openral``); a verified cached copy is reused offline.

    Returns:
        The local Walk / Balance ONNX and licence-text paths.

    Raises:
        ROSConfigError: A download fails or a file fails SHA-256 verification.
    """
    root = (
        Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
        / "g1_gr00t_wbc"
        / GR00T_WBC_PINNED_SHA
    )
    paths: dict[str, Path] = {}
    for relative_path, (url_root, expected_sha256) in _ASSETS.items():
        destination = root / relative_path.replace("%20", "_")
        _download(
            relative_path,
            expected_sha256,
            destination,
            root=url_root,
            label="GR00T WBC asset",
        )
        paths[relative_path] = destination
    return Gr00tWbcAssets(
        walk_onnx=paths[_WALK_ONNX],
        balance_onnx=paths[_BALANCE_ONNX],
        license_text=paths[_LICENSE],
    )


def position_actuators_to_torque_motors(model: mujoco.MjModel) -> None:
    """Turn the Menagerie ``<position kp=500>`` actuators into plain torque motors in memory.

    ``ctrl`` then is a joint torque (gain 1, no bias, ``ctrlrange`` off); the
    joint ``actuatorfrcrange`` from the MJCF still clamps it. The MJCF on disk
    is not touched.
    """
    import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dependency

    model.actuator_gaintype[:] = mj.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[:] = mj.mjtBias.mjBIAS_NONE
    model.actuator_gainprm[:] = 0.0
    model.actuator_gainprm[:, 0] = 1.0
    model.actuator_biasprm[:] = 0.0
    model.actuator_ctrllimited[:] = 0


def gr00t_wbc_observation(
    *,
    command: tuple[float, float, float],
    base_height: float,
    torso_rpy: tuple[float, float, float],
    base_quat_wxyz: NDArray[np.float64],
    base_ang_vel_body: NDArray[np.float64],
    joint_pos: NDArray[np.float64],
    joint_vel: NDArray[np.float64],
    last_action: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Build one 86-D GR00T WBC observation frame (upstream ``compute_observation``).

    Layout: ``[vx*2, vy*2, wz*0.5, height, roll, pitch, yaw, omega*0.5 (3),
    projected gravity R^T[0,0,-1] (3), q29 - default29 (29), dq29*0.05 (29),
    last raw action (15)]``.

    Args:
        command: Raw base velocity command ``(vx, vy, wz)``.
        base_height: Pelvis height command (m).
        torso_rpy: Torso roll/pitch/yaw command (rad), relative to the yaw-only pelvis frame.
        base_quat_wxyz: Floating-base orientation ``qpos[3:7]``.
        base_ang_vel_body: Floating-base angular velocity in the body frame ``qvel[3:6]``.
        joint_pos: The 29 joint positions, Menagerie order.
        joint_vel: The 29 joint velocities, Menagerie order.
        last_action: The previous raw 15-D network output.

    Returns:
        The float32 frame.

    Example:
        >>> import numpy as np
        >>> frame = gr00t_wbc_observation(
        ...     command=(0.5, 0.0, 0.0),
        ...     base_height=0.74,
        ...     torso_rpy=(0.0, 0.0, 0.0),
        ...     base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        ...     base_ang_vel_body=np.zeros(3),
        ...     joint_pos=np.zeros(29),
        ...     joint_vel=np.zeros(29),
        ...     last_action=np.zeros(15, dtype=np.float32),
        ... )
        >>> frame.shape, float(frame[0]), [float(g) + 0.0 for g in frame[10:13]]
        ((86,), 1.0, [0.0, 0.0, -1.0])
    """
    frame = np.zeros(_OBS_DIM, dtype=np.float32)
    frame[0:3] = np.asarray(command, dtype=np.float32) * _CMD_SCALE
    frame[3] = base_height
    frame[4:7] = torso_rpy
    frame[7:10] = np.asarray(base_ang_vel_body, dtype=np.float64) * _ANG_VEL_SCALE
    w, x, y, z = (float(v) for v in base_quat_wxyz)
    # R(q)^T @ [0, 0, -1] (upstream ``get_gravity_orientation``).
    frame[10:13] = (2.0 * (-z * x + w * y), -2.0 * (z * y + w * x), 1.0 - 2.0 * (w * w + z * z))
    frame[13:42] = np.asarray(joint_pos, dtype=np.float64) - _DEFAULT_Q29
    frame[42:71] = np.asarray(joint_vel, dtype=np.float64) * _DOF_VEL_SCALE
    frame[71:86] = last_action
    return frame


class Gr00tWbcController:
    """Stateful 50 Hz GR00T WBC policy: observation history, net selection, targets.

    Args:
        assets: Verified local weight paths from ``ensure_gr00t_wbc_assets``.

    Raises:
        ROSConfigError: ``onnxruntime`` is missing or a network does not load
            with the pinned ``[B, 516] -> [B, 15]`` signature.
    """

    def __init__(self, assets: Gr00tWbcAssets) -> None:
        """Load both ONNX networks on the CPU execution provider."""
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]  # noqa: PLC0415  # reason: optional backend has no py.typed
        except ImportError as exc:
            raise ROSConfigError(
                "The G1 gr00t_wbc S0 controller needs onnxruntime. Install the HAL sim "
                "extras with: just sync --group sim"
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._sessions: dict[str, _InferenceSession] = {}
        for name, path in (("walk", assets.walk_onnx), ("balance", assets.balance_onnx)):
            try:
                session = ort.InferenceSession(
                    str(path), options, providers=["CPUExecutionProvider"]
                )
            except Exception as exc:  # reason: onnxruntime raises untyped load errors
                raise ROSConfigError(
                    f"Could not load GR00T WBC {name} policy {path}: {exc}"
                ) from exc
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if (
                len(inputs) != 1
                or inputs[0].name != "input"
                or list(inputs[0].shape)[-1] != _INPUT_DIM
                or len(outputs) != 1
                or outputs[0].name != "output"
                or list(outputs[0].shape)[-1] != S0_JOINT_COUNT
            ):
                raise ROSConfigError(
                    f"GR00T WBC {name} policy {path} does not match the pinned "
                    f"'input'[B,{_INPUT_DIM}] -> 'output'[B,{S0_JOINT_COUNT}] signature."
                )
            self._sessions[name] = session
        self._history: deque[NDArray[np.float32]] = deque(maxlen=_HISTORY_LEN)
        self._last_action: NDArray[np.float32] = np.zeros(S0_JOINT_COUNT, dtype=np.float32)
        self.used_walk = False
        self.reset()

    def reset(self) -> None:
        """Zero the observation history and the last action (fresh episode)."""
        self._history.clear()
        self._history.extend(np.zeros(_OBS_DIM, dtype=np.float32) for _ in range(_HISTORY_LEN))
        self._last_action = np.zeros(S0_JOINT_COUNT, dtype=np.float32)
        self.used_walk = False

    def policy_input(self, frame: NDArray[np.float32]) -> NDArray[np.float32]:
        """Append ``frame`` to the history; return the ``[1, 516]`` network input, oldest first."""
        self._history.append(frame)
        return np.concatenate(list(self._history)).astype(np.float32).reshape(1, _INPUT_DIM)

    def step(
        self,
        data: mujoco.MjData,
        *,
        command: tuple[float, float, float],
        base_height: float = DEFAULT_BASE_HEIGHT_M,
        torso_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> NDArray[np.float64]:
        """Run one 50 Hz policy step on the live state; return the 15 joint targets.

        Raises:
            ROSConfigError: The network emitted a non-finite action.
        """
        frame = gr00t_wbc_observation(
            command=command,
            base_height=base_height,
            torso_rpy=torso_rpy,
            base_quat_wxyz=np.asarray(data.qpos[3:7]),
            base_ang_vel_body=np.asarray(data.qvel[3:6]),
            joint_pos=np.asarray(data.qpos[7 : 7 + _N_JOINTS]),
            joint_vel=np.asarray(data.qvel[6 : 6 + _N_JOINTS]),
            last_action=self._last_action,
        )
        batch = self.policy_input(frame)
        walk = float(np.linalg.norm(np.asarray(command, dtype=np.float64))) > WALK_COMMAND_THRESHOLD
        session = self._sessions["walk" if walk else "balance"]
        raw = np.asarray(session.run(["output"], {"input": batch})[0], dtype=np.float32).reshape(-1)
        if raw.shape != (S0_JOINT_COUNT,) or not np.isfinite(raw).all():
            raise ROSConfigError(
                f"GR00T WBC policy must emit {S0_JOINT_COUNT} finite targets; got "
                f"shape={raw.shape}, finite={bool(np.isfinite(raw).all())}."
            )
        self.used_walk = walk
        self._last_action = raw
        return raw.astype(np.float64) * _ACTION_SCALE + S0_DEFAULT_ANGLES
