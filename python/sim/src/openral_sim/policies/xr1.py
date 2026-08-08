"""Xiaomi Robotics XR-1 policy adapter via an isolated sidecar.

The released benchmark checkpoints use ``MiBoTForActionGeneration`` custom
Transformers code and pin torch 2.8, transformers 4.57.1, and FlashAttention
2.8.3. Those versions cannot coexist with the OpenRAL workspace, so inference
runs in ``tools/_xr1_server.py`` and this adapter only owns observation packing,
action replay, and the ZMQ wire.

Quantization follows the same two-stage pattern as π0.5 and MolmoAct2:

1. The manifest selects ``quantization.dtype: int4`` and the policy loader
   defines the exact bitsandbytes NF4 rule (double quantization, BF16 compute).
2. For repeated use, export the loaded NF4 model once and point a local rSkill
   at it; reload then consumes packed 4-bit weights instead of quantizing the
   upstream BF16 shards at every startup::

       OPENRAL_ALLOW_REMOTE_CODE=1 uv run python tools/xr1_sidecar.py \
         --model hf://XiaomiRobotics/Xiaomi-Robotics-1-VLABench@<sha> \
         --profile vlabench_choice --quantization nf4 \
         --export-dir outputs/run_artifacts/xr1-vlabench-nf4-rskill/weights
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_observability import inference_span
from openral_rskill._vla_core import resolve_camera_keys

from openral_sim.policies._policy_loading import load_manifest_for_spec
from openral_sim.registry import POLICIES
from openral_sim.sidecar import SidecarClient

if TYPE_CHECKING:
    from openral_core import RSkillManifest, SimEnvironment, VLASpec

    from openral_sim.rollout import Observation

_ALLOW_REMOTE_CODE_ENV = "OPENRAL_ALLOW_REMOTE_CODE"
_PROFILES = {"robocasa_mg", "robocasa365", "vlabench_choice"}
_PROFILE_STATE_DIMS = {"robocasa_mg": 8, "robocasa365": 14, "vlabench_choice": 7}
_PROFILE_ACTION_DIMS = {"robocasa_mg": 7, "robocasa365": 12, "vlabench_choice": 7}
_PROFILE_REPLAN_STEPS = {"robocasa_mg": 10, "robocasa365": 16, "vlabench_choice": 5}
_PROFILE_CROP_RATIOS = {"robocasa_mg": 0.95, "robocasa365": 0.95, "vlabench_choice": 1.0}
_RC365_HISTORY_LEN = 7
_RC365_HISTORY_OFFSETS = (0, 2, 4, 6)
_RC365_SOURCE_STATE_DIM = 16
_ACTION_CHUNK_RANK = 2
_CAMERA_COUNT = 3
_QUAT_EPSILON = 1e-8
_VLABENCH_GRIPPER_THRESHOLD = 0.2
_PORT_MIN = 20_000
_PORT_MAX = 39_999
_DEFAULT_TIMEOUT_MS = 120_000
_DEFAULT_BOOT_TIMEOUT_S = 1200.0


def _require_remote_code_ack() -> None:
    if os.environ.get(_ALLOW_REMOTE_CODE_ENV) != "1":
        raise ROSConfigError(
            "XR-1 checkpoints execute repository-shipped Transformers custom code. "
            f"Set {_ALLOW_REMOTE_CODE_ENV}=1 after reviewing the pinned checkpoint revision."
        )


def _profile_port(model_id: str, profile: str) -> int:
    digest = int.from_bytes(
        hashlib.sha256(f"xr1|{model_id}|{profile}".encode()).digest()[:4],
        "big",
    )
    return _PORT_MIN + digest % (_PORT_MAX - _PORT_MIN)


def _quat_xyzw_to_axis_angle(quat: NDArray[np.float32]) -> NDArray[np.float32]:
    q = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ROSConfigError("XR-1 RoboCasa365 received a zero quaternion.")
    normalized = np.asarray(q / norm, dtype=np.float32)
    if normalized[3] < 0.0:
        normalized = -normalized
    vector_norm = float(np.linalg.norm(normalized[:3]))
    if vector_norm < _QUAT_EPSILON:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(vector_norm, float(normalized[3]))
    return np.asarray(normalized[:3] / vector_norm * angle, dtype=np.float32)


def _rc365_state(state: object) -> NDArray[np.float32]:
    """Convert OpenRAL's human300/rc365 16-D layout to XR-1's 14-D layout."""
    src = np.asarray(state, dtype=np.float32).reshape(-1)
    if src.shape[0] != _RC365_SOURCE_STATE_DIM:
        raise ROSConfigError(
            f"XR-1 RoboCasa365 requires a 16-D source state, got {src.shape[0]}-D."
        )
    return np.concatenate(
        [
            src[0:3],
            _quat_xyzw_to_axis_angle(src[3:7]),
            src[14:16],
            src[7:10],
            _quat_xyzw_to_axis_angle(src[10:14]),
        ]
    ).astype(np.float32)


def _history_sample(values: deque[NDArray[np.float32]]) -> NDArray[np.float32]:
    if not values:
        raise ROSConfigError("XR-1 RoboCasa365 history is empty.")
    padded = [values[0]] * (_RC365_HISTORY_LEN - len(values)) + list(values)
    return np.stack([padded[i] for i in _RC365_HISTORY_OFFSETS])


def _vlabench_targets(
    deltas: NDArray[np.float32],
    state: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Integrate XR-1 VLABench deltas into the absolute targets the env consumes."""
    current = np.asarray(state, dtype=np.float32).reshape(7).copy()
    targets: list[NDArray[np.float32]] = []
    for delta in np.asarray(deltas, dtype=np.float32):
        current[:6] += delta[:6]
        current[3:6] = (current[3:6] + np.pi) % (2.0 * np.pi) - np.pi
        current[6] = 1.0 if delta[6] >= _VLABENCH_GRIPPER_THRESHOLD else 0.0
        targets.append(current.copy())
    return np.stack(targets)


def _preprocess_image(image: object, *, flip_180: bool) -> NDArray[np.uint8]:
    """Apply the manifest's explicit orientation transform; false is byte-preserving."""
    arr = np.asarray(image, dtype=np.uint8)
    if flip_180:
        arr = np.flip(arr, axis=(0, 1))
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _resolve_model_id(spec: VLASpec) -> str:
    manifest = load_manifest_for_spec(spec)
    if manifest is None:
        raise ROSConfigError("XR-1 requires a locally resolvable rSkill manifest.")
    from openral_rskill.loader import resolve_rskill_to_hf_with_revision

    target, revision = resolve_rskill_to_hf_with_revision(spec.weights_uri)
    if revision is not None:
        return f"hf://{target}@{revision}"
    return target


def _quantization_mode(manifest: RSkillManifest) -> str:
    dtype = manifest.quantization.dtype.value
    if dtype == "int4":
        if manifest.policy_extras.get("prequantized_nf4") is True:
            return "prequantized_nf4"
        return "nf4"
    if dtype == "bf16":
        return "none"
    raise ROSConfigError(
        f"XR-1 supports manifest quantization.dtype 'int4' (NF4) or 'bf16', got {dtype!r}."
    )


def _extra_int(extra: dict[str, object], key: str, default: int) -> int:
    value = extra.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ROSConfigError(f"XR-1 policy_extras.{key} must be an integer, got {value!r}.")
    try:
        result = int(value)
    except ValueError as exc:
        raise ROSConfigError(
            f"XR-1 policy_extras.{key} must be an integer, got {value!r}."
        ) from exc
    if result <= 0:
        raise ROSConfigError(f"XR-1 policy_extras.{key} must be > 0, got {result}.")
    return result


def _extra_float(extra: dict[str, object], key: str, default: float) -> float:
    value = extra.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ROSConfigError(f"XR-1 policy_extras.{key} must be numeric, got {value!r}.")
    try:
        result = float(value)
    except ValueError as exc:
        raise ROSConfigError(f"XR-1 policy_extras.{key} must be numeric, got {value!r}.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ROSConfigError(f"XR-1 policy_extras.{key} must be finite and > 0.")
    return result


def _extra_bool(extra: dict[str, object], key: str, default: bool) -> bool:
    value = extra.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ROSConfigError(f"XR-1 policy_extras.{key} must be a bool, got {value!r}.")


def _locate_sidecar_script() -> Path:
    override = os.environ.get("OPENRAL_XR1_SIDECAR_SCRIPT")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise ROSConfigError(f"OPENRAL_XR1_SIDECAR_SCRIPT={override!r} is not a file.")
        return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "xr1_sidecar.py"
        if candidate.is_file():
            return candidate
    raise ROSConfigError(
        f"Could not locate tools/xr1_sidecar.py upwards from {here}; "
        "set OPENRAL_XR1_SIDECAR_SCRIPT."
    )


@dataclass
class _XR1Adapter:
    spec: VLASpec
    device: str
    _client: SidecarClient
    _profile: str
    _camera_keys: tuple[str, ...]
    _replan_steps: int
    _crop_ratio: float
    _flip_180: bool
    _queue: deque[NDArray[np.float32]] = field(default_factory=deque)
    _image_history: dict[str, deque[NDArray[np.float32]]] = field(default_factory=dict)
    _state_history: deque[NDArray[np.float32]] = field(
        default_factory=lambda: deque(maxlen=_RC365_HISTORY_LEN)
    )
    _last_input: NDArray[np.uint8] | None = None

    def reset(self) -> None:
        self._queue.clear()
        self._image_history.clear()
        self._state_history.clear()
        with contextlib.suppress(Exception):
            self._client.call("reset")

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return None if self._last_input is None else self._last_input.copy()

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        images, state = self._read_observation(observation)
        self._record_history(images, state)
        if not self._queue:
            self._refill(images, state, instruction)
        return np.asarray(self._queue.popleft(), dtype=np.float32)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.call("close")
        self._client.close()

    def _read_observation(
        self,
        observation: Observation,
    ) -> tuple[dict[str, NDArray[np.uint8]], NDArray[np.float32]]:
        raw_images = observation.get("images", {})
        images: dict[str, NDArray[np.uint8]] = {}
        for key in self._camera_keys:
            image = raw_images.get(key)
            if image is None:
                raise ROSConfigError(
                    f"XR-1 expected camera {key!r}; observation provides {sorted(raw_images)}."
                )
            images[key] = _preprocess_image(image, flip_180=self._flip_180)
        self._last_input = images[self._camera_keys[0]]

        raw_state = observation.get("state")
        if raw_state is None:
            raise ROSConfigError(f"XR-1 profile {self._profile!r} requires proprioceptive state.")
        state = (
            _rc365_state(raw_state)
            if self._profile == "robocasa365"
            else np.asarray(raw_state, dtype=np.float32).reshape(-1)
        )
        expected = _PROFILE_STATE_DIMS[self._profile]
        if state.shape != (expected,):
            raise ROSConfigError(
                f"XR-1 profile {self._profile!r} requires {expected}-D state, "
                f"got {state.shape[0]}-D."
            )
        return images, state

    def _record_history(
        self,
        images: dict[str, NDArray[np.uint8]],
        state: NDArray[np.float32],
    ) -> None:
        if self._profile != "robocasa365":
            return
        for key, image in images.items():
            history = self._image_history.setdefault(key, deque(maxlen=_RC365_HISTORY_LEN))
            history.append(np.asarray(image, dtype=np.float32))
        self._state_history.append(state)

    def _refill(
        self,
        images: dict[str, NDArray[np.uint8]],
        state: NDArray[np.float32],
        instruction: str,
    ) -> None:
        payload_images: dict[str, NDArray[np.float32]] = {}
        if self._profile == "robocasa365":
            for key in self._camera_keys:
                payload_images[key] = _history_sample(self._image_history[key])
            payload_state = _history_sample(self._state_history)
        else:
            payload_images = {
                key: np.asarray(image, dtype=np.float32) for key, image in images.items()
            }
            payload_state = state

        # The sidecar round-trip IS the inference — span it here, as rldx and
        # every other sidecar adapter does. `openral.inference.duration` is
        # emitted only by `inference_span`, so an unspanned adapter reports no
        # latency at all (tests/unit/test_every_adapter_is_instrumented.py).
        with inference_span(kind="foreground"):
            reply = self._client.call(
                "get_action",
                {
                    "profile": self._profile,
                    "images": payload_images,
                    "state": payload_state,
                    "task": instruction,
                    "seed": 42,
                    "crop_ratio": self._crop_ratio,
                },
            )
        chunk = np.asarray(self._client.require(reply, "action"), dtype=np.float32)
        expected_dim = _PROFILE_ACTION_DIMS[self._profile]
        if chunk.ndim != _ACTION_CHUNK_RANK or chunk.shape[1] != expected_dim:
            raise ROSConfigError(
                f"XR-1 sidecar returned {chunk.shape}; expected (chunk, {expected_dim})."
            )
        if self._profile == "vlabench_choice":
            chunk = _vlabench_targets(chunk, state)
        for action in chunk[: self._replan_steps]:
            self._queue.append(np.asarray(action, dtype=np.float32))
        if not self._queue:
            raise ROSConfigError("XR-1 sidecar returned an empty action chunk.")


@POLICIES.register("xr1")
def _build_xr1(env_cfg: SimEnvironment) -> _XR1Adapter:
    _require_remote_code_ack()
    spec = env_cfg.vla
    extra = dict(spec.extra or {})
    profile = str(extra.get("xr1_profile", ""))
    if profile not in _PROFILES:
        raise ROSConfigError(
            f"XR-1 policy_extras.xr1_profile must be one of {sorted(_PROFILES)}, got {profile!r}."
        )
    manifest = load_manifest_for_spec(spec)
    if manifest is None:
        raise ROSConfigError("XR-1 requires a locally resolvable rSkill manifest.")
    model_id = _resolve_model_id(spec)
    quantization = _quantization_mode(manifest)
    scene_cameras = getattr(env_cfg.scene, "cameras", None)
    camera_keys = resolve_camera_keys(
        load_manifest_for_spec(spec),
        spec.extra,
        scene_cameras=scene_cameras,
    )
    if len(camera_keys) != _CAMERA_COUNT:
        raise ROSConfigError(f"XR-1 requires exactly three cameras, got {camera_keys}.")

    port = _extra_int(extra, "port", _profile_port(model_id, profile))
    timeout_ms = _extra_int(extra, "timeout_ms", _DEFAULT_TIMEOUT_MS)
    replan_steps = _extra_int(extra, "replan_steps", _PROFILE_REPLAN_STEPS[profile])
    crop_ratio = _extra_float(extra, "crop_ratio", _PROFILE_CROP_RATIOS[profile])
    if crop_ratio > 1.0:
        raise ROSConfigError("XR-1 policy_extras.crop_ratio must be <= 1.0.")
    auto_spawn = _extra_bool(extra, "auto_spawn", True)
    launch_argv = [
        sys.executable,
        str(_locate_sidecar_script()),
        "--model",
        model_id,
        "--profile",
        profile,
        "--quantization",
        quantization,
        "--port",
        str(port),
    ]
    client = SidecarClient(
        name="xr1",
        host="127.0.0.1",
        port=port,
        timeout_ms=timeout_ms,
        boot_timeout_s=_extra_float(
            extra,
            "boot_timeout_s",
            _DEFAULT_BOOT_TIMEOUT_S,
        ),
        launch_argv=launch_argv,
        auto_spawn=auto_spawn,
        expected_identity={
            "model": model_id,
            "profile": profile,
            "quantization": quantization,
        },
    )
    client.connect()
    return _XR1Adapter(
        spec=spec,
        device="cuda",
        _client=client,
        _profile=profile,
        _camera_keys=tuple(camera_keys),
        _replan_steps=replan_steps,
        _crop_ratio=crop_ratio,
        _flip_180=(
            manifest.image_preprocessing.flip_180
            if manifest.image_preprocessing is not None
            else False
        ),
    )
