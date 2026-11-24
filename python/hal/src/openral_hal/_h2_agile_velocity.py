"""NVIDIA WBC-AGILE velocity policy as the H2 twin's sim-only S0 (a LEAPP bundle on onnxruntime).

The bundle is the one the real H2 runs (``config/h2/policies/agile_velocity.yaml``:
Hugging Face ``nvidia/agile_wbc_unitree_h2`` at a pinned revision, every file SHA-256
pinned, NVIDIA Open Model License weights — downloaded, never vendored). This module
steps its ONNX the way NVIDIA's ``InferenceController`` does: the inputs are mapped by
the LEAPP ``element_names``, the ``pipeline.feedback_flow`` outputs (5-step histories,
last action) loop back to their inputs every tick, seeded from the bundle's
``initial_values`` safetensors.

Simulation only: on hardware this policy runs in C++ inside ``ros2_control``
(CLAUDE.md §3 dual-system pattern); this runner must never drive a real H2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import RSkillLicensePosture

from openral_hal.unitree_s0_policy import load_s0_policy, resolve_leapp_config, s0_policy_path

__all__ = [
    "AGILE_LICENSE_POSTURE",
    "LeappVelocityPolicy",
    "h2_agile_velocity_config",
]

#: Licence posture of the AGILE weights (the runner code is Apache-2.0).
AGILE_LICENSE_POSTURE = RSkillLicensePosture.NVIDIA_OPEN_MODEL

# The input layouts this runner maps MuJoCo state into; a bundle that names them
# differently is refused instead of being fed a mis-ordered vector.
_QUAT_ELEMENTS = ["qx", "qy", "qz", "qw"]
_COMMAND_ELEMENTS = ["lin_vel_x", "lin_vel_y", "ang_vel_z"]


def h2_agile_velocity_config() -> Path:
    """The H2 AGILE bundle's LEAPP YAML, SHA-256-verified against its policy file pins.

    Raises:
        ROSConfigError: The bundle is not downloaded (the message names the ``hf download``
            command) or a file fails its pin.
    """
    return resolve_leapp_config(load_s0_policy(s0_policy_path("h2", "agile_velocity")), "")


class LeappVelocityPolicy:
    """One LEAPP velocity-policy export, stepped with its feedback tensors looped.

    Args:
        leapp_yaml: The bundle's LEAPP YAML (its ONNX and initial values sit beside it).

    Attributes:
        obs_joint_names: Joints of the ``robot_joint_pos`` / ``robot_joint_vel`` inputs,
            in network order (Isaac Lab breadth-first).
        action_joint_names: Joints of the ``joint_pos`` / kp / kd outputs, in output order.
        default_joint_pos: ``joint -> default position`` (rad) the policy was trained around.
        default_kp: ``joint -> stiffness`` it was trained with (also the non-policy joints').
        default_kd: ``joint -> damping`` it was trained with.
        frequency_hz: The policy rate (``pipeline.configs.frequency``).
        last_outputs: Every output of the last ``step`` by name (feedback included).

    Raises:
        ROSConfigError: ``onnxruntime`` / ``safetensors`` missing, or the bundle is not the
            single-model velocity layout this runner maps.

    Example:
        >>> policy = LeappVelocityPolicy(h2_agile_velocity_config())  # doctest: +SKIP
        >>> q = np.array([policy.default_joint_pos[j] for j in policy.obs_joint_names])
        >>> target, kp, kd = policy.step(  # doctest: +SKIP
        ...     ang_vel_b=np.zeros(3),
        ...     quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        ...     command=(0.0, 0.0, 0.0),
        ...     joint_pos=q,
        ...     joint_vel=np.zeros(31),
        ... )
        >>> target.shape  # doctest: +SKIP
        (14,)
    """

    def __init__(self, leapp_yaml: Path) -> None:
        """Load the LEAPP spec, the ONNX session and the feedback seed."""
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]  # noqa: PLC0415  # reason: optional backend has no py.typed
            from safetensors.numpy import load_file  # noqa: PLC0415  # reason: optional dep
        except ImportError as exc:
            raise ROSConfigError(
                "The H2 agile_velocity S0 controller needs onnxruntime and safetensors. "
                "Install the HAL sim extras with: just sync --group sim"
            ) from exc
        spec: dict[str, Any] = yaml.safe_load(leapp_yaml.read_text(encoding="utf-8"))
        try:
            ((name, model),) = spec["models"].items()
            inputs = {i["name"]: i for i in model["inputs"]}
            outputs = {o["name"]: o for o in model["outputs"]}
            self.obs_joint_names: tuple[str, ...] = tuple(
                inputs["robot_joint_pos"]["element_names"][0]
            )
            vel_names = tuple(inputs["robot_joint_vel"]["element_names"][0])
            self.action_joint_names: tuple[str, ...] = tuple(
                outputs["joint_pos"]["element_names"][0]
            )
            quat = inputs["robot_root_quat_w"]["element_names"][0]
            command = inputs["base_velocity"]["element_names"][0]
            gains = [
                outputs[g]["element_names"][0] for g in ("joint_pos_kp_gains", "joint_pos_kd_gains")
            ]
            art = spec["agile"]["articulations"]["robot"]
            joints = art["joint_names"]
            self.default_joint_pos = dict(zip(joints, art["default_joint_pos"], strict=True))
            self.default_kp = dict(zip(joints, art["default_joint_stiffness"], strict=True))
            self.default_kd = dict(zip(joints, art["default_joint_damping"], strict=True))
            self.frequency_hz = float(spec["pipeline"]["configs"]["frequency"])
            onnx_path = leapp_yaml.parent / model["parameters"]["model_path"]
            seed_path = leapp_yaml.parent / spec["pipeline"]["initial_values"]
            prefix = f"{name}/"
            self._loops = {
                out.removeprefix(prefix): ins[0].removeprefix(prefix)
                for out, ins in spec["pipeline"]["feedback_flow"].items()
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ROSConfigError(
                f"LEAPP bundle {leapp_yaml} is not a velocity export: {exc}"
            ) from exc
        if (
            vel_names != self.obs_joint_names
            or quat != _QUAT_ELEMENTS
            or command != _COMMAND_ELEMENTS
            or any(list(g) != list(self.action_joint_names) for g in gains)
        ):
            raise ROSConfigError(
                f"LEAPP bundle {leapp_yaml}: input/output element_names differ from the "
                f"layout this runner maps (quat {_QUAT_ELEMENTS}, command {_COMMAND_ELEMENTS}, "
                "joint vel = joint pos order, kp/kd = joint_pos order)."
            )
        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._out_names = [o.name for o in self._session.get_outputs()]
        self._seed = {k.removeprefix(prefix): v for k, v in load_file(str(seed_path)).items()}
        self._feedback: dict[str, NDArray[np.float32]] = {}
        self.last_outputs: dict[str, NDArray[np.float32]] = {}
        self.reset()

    def reset(self) -> None:
        """Re-seed the feedback tensors from the bundle's initial values (fresh episode)."""
        self._feedback = {k: v.copy() for k, v in self._seed.items()}
        self.last_outputs = {}

    def step(
        self,
        *,
        ang_vel_b: NDArray[np.float64],
        quat_xyzw: NDArray[np.float64],
        command: tuple[float, float, float],
        joint_pos: NDArray[np.float64],
        joint_vel: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """One policy tick; ``joint_pos`` / ``joint_vel`` in ``obs_joint_names`` order.

        Returns:
            ``(targets, kp, kd)``, each in ``action_joint_names`` order (absolute rad).

        Raises:
            ROSConfigError: The network emitted a non-finite target or gain.
        """

        def row(values: Any) -> NDArray[np.float32]:  # noqa: ANN401  # reason: array-like
            return np.asarray(values, dtype=np.float32).reshape(1, -1)

        state = {
            "robot_root_ang_vel_b": row(ang_vel_b),
            "robot_root_quat_w": row(quat_xyzw),
            "base_velocity": row(command),
            "robot_joint_pos": row(joint_pos),
            "robot_joint_vel": row(joint_vel),
        }
        result = dict(
            zip(self._out_names, self._session.run(None, {**state, **self._feedback}), strict=True)
        )
        self._feedback = {dst: result[src] for src, dst in self._loops.items()}
        self.last_outputs = result
        out = tuple(
            np.asarray(result[k], dtype=np.float64).reshape(-1)
            for k in ("joint_pos", "joint_pos_kp_gains", "joint_pos_kd_gains")
        )
        if not all(np.isfinite(a).all() for a in out):
            raise ROSConfigError("H2 agile_velocity policy emitted a non-finite target or gain.")
        return out[0], out[1], out[2]
