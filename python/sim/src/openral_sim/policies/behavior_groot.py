"""Official BEHAVIOR-1K GR00T policy adapter via an isolated sidecar."""

from __future__ import annotations

import contextlib
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

from openral_sim import _behavior_wire
from openral_sim.sidecar import SidecarClient

if TYPE_CHECKING:
    from openral_core import RSkillManifest, VLASpec

    from openral_sim.rollout import Observation

_SIDECAR_PYTHON_ENV = "OPENRAL_BEHAVIOR_GROOT_SIDECAR_PYTHON"
_SIDECAR_SCRIPT_ENV = "OPENRAL_BEHAVIOR_GROOT_SIDECAR_SCRIPT"
_CHECKPOINT_ENV = "OPENRAL_BEHAVIOR_GROOT_CHECKPOINT"
_HOST_ENV = "OPENRAL_BEHAVIOR_GROOT_HOST"
_PORT_ENV = "OPENRAL_BEHAVIOR_GROOT_PORT"
_AUTO_SPAWN_ENV = "OPENRAL_BEHAVIOR_GROOT_AUTO_SPAWN"

_DEFAULT_SIDECAR_PYTHON = (
    Path.home() / ".cache" / "openral" / "behavior-groot" / ".venv" / "bin" / "python"
)
_DEFAULT_CHECKPOINT = Path("checkpoints/behavior-groot-turning-on-radio")
_DEFAULT_HOST = "127.0.0.1"
_PORT_MIN = 22_000
_PORT_MAX = 22_999
_DEFAULT_TIMEOUT_MS = 120_000
_DEFAULT_BOOT_TIMEOUT_S = 600.0
_IMPLEMENTATION = "behavior_b1k_sidecar"

_STATE_KEY = _behavior_wire.STATE_KEY
_CAMERA_KEYS = _behavior_wire.CAMERA_RGB_KEYS
_STATE_DIM = _behavior_wire.STATE_DIM
_ACTION_DIM = _behavior_wire.ACTION_DIM


class _EnvCfg(Protocol):
    vla: VLASpec


def _policy_default_port(
    task: str, checkpoint: str, quantization: str, control_mode: str, nf4_min_params: int
) -> int:
    # Every option that changes the served policy's behavior is part of the
    # port key, so e.g. an nf4→int8 A/B rerun can never silently adopt the
    # still-running sidecar quantized the old way (the ping-identity check is
    # lenient about absent keys; the port hash is the primary stale-reuse
    # guard).
    key = (
        f"behavior-groot|{task}|{checkpoint}|{quantization}|{control_mode}|{nf4_min_params}"
    ).encode()
    digest = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return _PORT_MIN + (digest % (_PORT_MAX - _PORT_MIN))


def _opt_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _opt_float(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _checkpoint_path(manifest: RSkillManifest) -> Path:
    override = os.environ.get(_CHECKPOINT_ENV)
    if override:
        return Path(override).expanduser()
    weights_uri = str(manifest.weights_uri or "")
    if weights_uri.startswith("local://"):
        return Path(weights_uri.removeprefix("local://")).expanduser()
    return _DEFAULT_CHECKPOINT


def _sidecar_python() -> Path:
    override = os.environ.get(_SIDECAR_PYTHON_ENV)
    path = Path(override).expanduser() if override else _DEFAULT_SIDECAR_PYTHON
    if path.is_file():
        return path
    raise ROSConfigError(
        "BEHAVIOR GR00T sidecar venv not found. Provision the pinned "
        "wensi-ai/Isaac-GR00T behavior branch under Python 3.10, then set "
        f"{_SIDECAR_PYTHON_ENV} to its .venv/bin/python. See "
        "rskills/gr00t-n17-b1k-turning-on-radio/README.md."
    )


def _locate_sidecar_script() -> Path:
    override = os.environ.get(_SIDECAR_SCRIPT_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise ROSConfigError(f"{_SIDECAR_SCRIPT_ENV}={override!r} is not a file.")
        return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "behavior_groot_sidecar.py"
        if candidate.is_file():
            return candidate
    raise ROSConfigError(
        f"Could not locate tools/behavior_groot_sidecar.py upwards from {here}. "
        f"Set {_SIDECAR_SCRIPT_ENV} to its absolute path."
    )


def _behavior_wire_observation(
    observation: Observation,
    *,
    instruction: str,
) -> dict[str, object]:
    raw = observation.get("behavior_raw")
    if isinstance(raw, dict) and all(isinstance(key, str) for key in raw):
        return {**raw, "openral_instruction": instruction}

    state_raw = observation.get("state")
    if state_raw is None:
        raise ROSRuntimeError(
            "BEHAVIOR GR00T requires either observation['behavior_raw'] from the "
            "official evaluator or a 61-D observation['state'] vector."
        )
    state = np.asarray(state_raw, dtype=np.float32).reshape(-1)
    if state.shape != (_STATE_DIM,):
        raise ROSRuntimeError(
            f"BEHAVIOR GR00T state must have {_STATE_DIM} values, got {state.shape[0]}."
        )

    images = observation.get("images", {})
    wire: dict[str, object] = {
        _STATE_KEY: state,
        "openral_instruction": instruction,
    }
    for role, wire_key in _CAMERA_KEYS.items():
        image = images.get(role)
        if image is None:
            raise ROSRuntimeError(f"BEHAVIOR GR00T observation is missing camera {role!r}.")
        wire[wire_key] = np.asarray(image, dtype=np.uint8)
    return wire


@dataclass
class _BehaviorGrootAdapter:
    """PolicyAdapter proxying the official B1K GR00T runtime."""

    spec: VLASpec
    device: str
    _client: SidecarClient
    _last_input: NDArray[np.uint8] | None = field(default=None)

    def reset(self) -> None:
        self._client.call("reset")

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        images = observation.get("images", {})
        head = images.get("head")
        if head is not None:
            self._last_input = np.asarray(head, dtype=np.uint8)
        reply = self._client.call(
            "get_action",
            {"observation": _behavior_wire_observation(observation, instruction=instruction)},
        )
        action = np.asarray(self._client.require(reply, "action"), dtype=np.float32).reshape(-1)
        if action.shape != (_ACTION_DIM,):
            raise ROSRuntimeError(
                f"BEHAVIOR GR00T sidecar returned {action.shape[0]} actions; "
                f"expected {_ACTION_DIM}."
            )
        if not np.isfinite(action).all():
            raise ROSRuntimeError("BEHAVIOR GR00T sidecar returned a non-finite action.")
        return action

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return None if self._last_input is None else self._last_input.copy()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.call("close")
        self._client.close()


def build_behavior_groot_policy(
    env_cfg: _EnvCfg,
    manifest: RSkillManifest,
    extra: dict[str, object],
) -> _BehaviorGrootAdapter:
    """Build the official B1K GR00T adapter behind its Python 3.10 sidecar."""
    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("behavior_groot_client")
    try:
        import msgpack  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: opt-in sidecar wire
        import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401  reason: opt-in sidecar wire
    except ImportError as exc:  # pragma: no cover - runtime install error
        raise ROSConfigError(
            "BEHAVIOR GR00T needs pyzmq + msgpack on the OpenRAL venv: "
            "just sync --all-packages --group behavior-groot"
        ) from exc

    spec = env_cfg.vla
    task = str(extra.get("task", "turning_on_radio"))
    instruction = str(extra.get("instruction", task.replace("_", " ")))
    control_mode = str(extra.get("control_mode", "temporal_ensemble"))
    quantization = str(extra.get("quantization", "nf4"))
    nf4_min_params = _opt_int(extra.get("nf4_min_params"), 4_000_000)
    host = os.environ.get(_HOST_ENV, str(extra.get("host", _DEFAULT_HOST)))
    checkpoint = _checkpoint_path(manifest)
    default_port = _policy_default_port(
        task, str(checkpoint), quantization, control_mode, nf4_min_params
    )
    port = _behavior_wire.explicit_port(
        os.environ.get(_PORT_ENV), extra.get("port"), default_port, env_var=_PORT_ENV
    )
    auto_spawn = os.environ.get(_AUTO_SPAWN_ENV, "1") != "0"

    launch_argv: list[str] = []
    if auto_spawn:
        if not checkpoint.is_dir():
            raise ROSConfigError(
                f"BEHAVIOR GR00T checkpoint not found at {checkpoint}. Download the official "
                "turning_on_radio checkpoint and set "
                f"{_CHECKPOINT_ENV}, or place it at {_DEFAULT_CHECKPOINT}."
            )
        device = "cuda" if spec.device == "auto" else str(spec.device)
        launch_argv = [
            "env",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            str(_sidecar_python()),
            str(_locate_sidecar_script()),
            "--checkpoint",
            str(checkpoint),
            "--task",
            task,
            "--instruction",
            instruction,
            "--control-mode",
            control_mode,
            "--device",
            device,
            "--quantization",
            quantization,
            "--nf4-min-params",
            str(nf4_min_params),
            "--host",
            host,
            "--port",
            str(port),
        ]

    client = SidecarClient(
        name="behavior-groot",
        host=host,
        port=port,
        timeout_ms=_DEFAULT_TIMEOUT_MS,
        boot_timeout_s=_opt_float(extra.get("boot_timeout_s"), _DEFAULT_BOOT_TIMEOUT_S),
        launch_argv=launch_argv,
        auto_spawn=auto_spawn,
        expected_identity={
            "model": "behavior_groot",
            "task": task,
            "quantization": quantization,
            "control_mode": control_mode,
        },
    )
    client.connect()
    return _BehaviorGrootAdapter(spec=spec, device="sidecar", _client=client)


__all__ = ["build_behavior_groot_policy"]
