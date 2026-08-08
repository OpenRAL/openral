"""LingBot-VA adapter for the Runtime-owned Galaxea A1 policy gateway."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from openral_core import VLASpec
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_observability import inference_span
from openral_rskill.loader import load_rskill_manifest
from openral_runner.backends.galaxea_a1_ipc import (
    RuntimeLocalClient,
    decode_array,
    encode_array,
)

from openral_sim.registry import POLICIES

_PROTOCOL_VERSION = "galaxea_a1_openral_policy_v1"
_SOCKET_NAME = "a1-openral-policy.sock"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_ACTION_DIM = 7
_COLOR_NDIM = 3
_COLOR_CHANNELS = 3


def _joint_contract(
    spec: VLASpec,
    robot_description: Any,
) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.float64], float]:
    """Validate the rSkill replay bound against the active A1 HAL contract."""
    if robot_description is None or robot_description.name != "galaxea_a1":
        raise ROSConfigError("lingbot_va_a1 requires the Galaxea A1 RobotDescription")
    joints = tuple(robot_description.joints)
    names = tuple(joint.name for joint in joints)
    lower: list[float] = []
    upper: list[float] = []
    for joint in joints:
        limits = joint.position_limits
        if limits is None:
            raise ROSConfigError(f"OpenRAL joint {joint.name!r} requires finite position limits")
        lo, hi = (float(item) for item in limits)
        if not np.isfinite((lo, hi)).all() or lo >= hi:
            raise ROSConfigError(f"OpenRAL joint {joint.name!r} requires finite position limits")
        lower.append(lo)
        upper.append(hi)
    raw_step = spec.extra.get("max_joint_substep_rad")
    if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
        raise ROSConfigError("lingbot_va_a1 requires numeric policy_extras.max_joint_substep_rad")
    step = float(raw_step)
    try:
        defaults = robot_description.hal.parameters.defaults
        phase_limits = (
            float(defaults["max_target_step_rad"]),
            float(defaults["initial_alignment_tolerance_rad"]),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ROSConfigError(
            "Galaxea A1 HAL requires target-step and initial-alignment limits"
        ) from exc
    if (
        not np.isfinite(step)
        or step <= 0
        or any(not np.isfinite(limit) or limit <= 0 for limit in phase_limits)
        or step > min(phase_limits)
    ):
        raise ROSConfigError(
            "policy max_joint_substep_rad must be positive and no greater than "
            "the A1 HAL target-step and initial-alignment limits"
        )
    return (
        names,
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        step,
    )


def _model_identity(spec: VLASpec) -> tuple[str, str]:
    """Return the immutable model repo and revision from the loaded rSkill."""
    manifest = load_rskill_manifest(spec.weights_uri)
    uri = manifest.weights_uri
    if not isinstance(uri, str) or not uri.startswith("hf://") or "@" not in uri:
        raise ROSConfigError(
            "lingbot_va_a1 rSkill weights_uri must pin an hf:// repository revision"
        )
    repo_id, revision = uri.removeprefix("hf://").rsplit("@", 1)
    if not repo_id or not revision:
        raise ROSConfigError("lingbot_va_a1 rSkill model identity is incomplete")
    return repo_id, revision


class _LingBotVaA1Adapter:
    """Proxy A1 observations to Runtime and return its typed action proposal."""

    device = "external:a1-runtime"

    def __init__(self, spec: VLASpec, *, robot_description: Any) -> None:
        self.spec = spec
        names, lower, upper, max_step = _joint_contract(spec, robot_description)
        latency_budget_ms = spec.extra.get("latency_budget_ms")
        if isinstance(latency_budget_ms, bool) or not isinstance(
            latency_budget_ms,
            (int, float),
        ):
            raise ROSConfigError(
                "lingbot_va_a1 requires its manifest latency budget in VLASpec.extra"
            )
        self._timeout_s = float(latency_budget_ms) / 1000.0
        if not np.isfinite(self._timeout_s) or self._timeout_s <= 0:
            raise ROSConfigError("lingbot_va_a1 latency budget must be positive")
        self._client = RuntimeLocalClient(
            _SOCKET_NAME,
            label="Galaxea A1 OpenRAL policy gateway",
            max_response_bytes=_MAX_RESPONSE_BYTES,
        )
        self._instruction: str | None = None
        try:
            self._client.connect(timeout_s=5.0)
            description = self._client.call(
                {"protocol": _PROTOCOL_VERSION, "op": "describe"},
                timeout_s=5.0,
            )
            self._validate_description(
                description.get("metadata"),
                joint_names=names,
                expected_model=_model_identity(spec),
            )
            self._client.call(
                {
                    "protocol": _PROTOCOL_VERSION,
                    "op": "configure",
                    "contract": {
                        "joint_names": list(names),
                        "lower_limits": lower.tolist(),
                        "upper_limits": upper.tolist(),
                        "max_joint_step_rad": max_step,
                    },
                },
                timeout_s=5.0,
            )
        except BaseException:
            self._client.close()
            raise

    def reset(self) -> None:
        """Clear the local instruction identity; Runtime resets on the next step."""
        self._instruction = None

    def close(self) -> None:
        """Close only this client; the Runtime-owned gateway remains alive."""
        self._client.close()

    def step(
        self,
        observation: dict[str, Any],
        instruction: str,
    ) -> NDArray[np.float32]:
        """Return one absolute A1 joint/gripper action."""
        if not instruction.strip():
            raise ROSRuntimeError("lingbot_va_a1 instruction must be non-empty")
        if self._instruction != instruction:
            self._client.call(
                {
                    "protocol": _PROTOCOL_VERSION,
                    "op": "reset",
                    "instruction": instruction,
                },
                timeout_s=self._timeout_s,
            )
            self._instruction = instruction
        joints = np.asarray(observation.get("state"), dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ROSRuntimeError(
                f"lingbot_va_a1 requires six finite A1 joint positions, got {joints.shape}"
            )
        images = observation.get("images")
        if not isinstance(images, dict):
            raise ROSRuntimeError("lingbot_va_a1 requires OpenRAL image observations")
        front = _rgb_image(images.get("front"), name="front")
        wrist = _rgb_image(images.get("wrist"), name="wrist")
        with inference_span(kind="single", engine="sidecar"):
            response = self._client.call(
                {
                    "protocol": _PROTOCOL_VERSION,
                    "op": "step",
                    "instruction": instruction,
                    "timeout_s": self._timeout_s,
                    "joints": encode_array(joints),
                    "front_rgb": encode_array(front),
                    "wrist_rgb": encode_array(wrist),
                },
                timeout_s=self._timeout_s,
            )
        action = decode_array(
            response.get("action"),
            shape=(7,),
            dtype=np.dtype(np.float32),
            label="Galaxea A1 OpenRAL policy action",
        )
        if not np.isfinite(action).all() or not 0.0 <= float(action[6]) <= 1.0:
            raise ROSRuntimeError(
                "Galaxea A1 OpenRAL policy returned an invalid joint/gripper action"
            )
        return action

    @staticmethod
    def _validate_description(
        value: Any,
        *,
        joint_names: tuple[str, ...],
        expected_model: tuple[str, str],
    ) -> None:
        if not isinstance(value, dict):
            raise ROSConfigError("Galaxea A1 OpenRAL policy metadata is missing")
        if value.get("protocol") != _PROTOCOL_VERSION:
            raise ROSConfigError("Galaxea A1 OpenRAL policy protocol mismatch")
        if value.get("joint_names") != list(joint_names):
            raise ROSConfigError(
                "Galaxea A1 OpenRAL policy joint contract does not match the robot"
            )
        if value.get("action_dim") != _ACTION_DIM or value.get("gripper_range") != [
            0.0,
            1.0,
        ]:
            raise ROSConfigError("Galaxea A1 OpenRAL policy action contract mismatch")
        actual_model = (
            value.get("model_repo_id"),
            value.get("model_revision"),
        )
        if actual_model != expected_model:
            raise ROSConfigError(
                "Galaxea A1 OpenRAL policy serves a different model: "
                f"expected {expected_model[0]}@{expected_model[1]}, "
                f"got {actual_model[0]}@{actual_model[1]}"
            )


def _rgb_image(value: Any, *, name: str) -> NDArray[np.uint8]:
    image = np.asarray(value)
    if image.ndim != _COLOR_NDIM or image.shape[2] != _COLOR_CHANNELS or image.dtype != np.uint8:
        raise ROSRuntimeError(
            f"lingbot_va_a1 image {name!r} must be HWC uint8 RGB, "
            f"got shape={image.shape} dtype={image.dtype}"
        )
    return image


@POLICIES.register("lingbot_va_a1")
def _build_lingbot_va_a1(env_cfg: Any) -> _LingBotVaA1Adapter:
    return _LingBotVaA1Adapter(
        env_cfg.vla,
        robot_description=getattr(env_cfg, "robot_description", None),
    )
