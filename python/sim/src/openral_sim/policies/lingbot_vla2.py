r"""LingBot-VLA 2.0 policy adapter — auto-managed ZMQ sidecar.

Robbyant's LingBot-VLA 2.0 (``robbyant/lingbot-vla-v2-6b``, 6.38 B params,
Apache-2.0 code + weights) is a Qwen3-VL-4B backbone with a sparse-MoE
flow-matching action expert (10 denoising steps, 55-D unified state/action
vector, 30 Hz). Its reference inference path is the upstream
``lingbotvla`` package (https://github.com/robbyant/lingbot-vla-v2), a
VeOmni-derived training framework whose model class pulls in custom Triton
MoE kernels (``lingbotvla.ops.robby_moe`` / ``group_gemm``), ``flex_attention``,
and a hard pin on ``torch==2.8.0`` / ``transformers==4.57.3`` / ``triton==3.4.0``.

Why an out-of-process sidecar (not in-process, not vendored)
-----------------------------------------------------------
Mirrors the :mod:`openral_sim.policies.rldx` rationale:

* **Dep stack cannot coexist.** ``lingbotvla`` pins ``torch==2.8.0`` +
  ``triton==3.4.0`` + ``transformers==4.57.3``; the openral workspace is
  ``torch>=2.9`` / ``transformers>=5`` (CLAUDE.md §3). Force-installing would
  clobber smolvla / pi05 / ACT / GR00T (the documented ``--group`` clobber).
* **Vendoring is intractable.** The inference path is ~6 kLOC of tightly
  coupled model code (``modeling_lingbot_vla_v2`` + ``qwen2_action_expert`` +
  ``qwen3vl_in_vla`` + ``flex_attention``) plus ``lingbotvla/ops`` Triton
  kernels and a ``sys.path``-based package layout — not the handful of files
  the in-process MolmoAct2 / SmolVLA adapters vendor.
* **The HF release ships no ``config.json`` architecture.** ``config.json`` is
  a 31-byte ``{"vlm_family":"qwen3_vl"}`` stub; the real architecture dims live
  in the repo's ``configs/vla/robotwin/robotwin.yaml`` training config, which
  the sidecar owns in its repo checkout. There is no ``trust_remote_code``
  escape.

So the sidecar (``tools/lingbot_vla2_sidecar.py``) runs the upstream
``LingbotVLAv2Server`` in its own py3.12 + torch-2.8 venv and answers
``ping`` / ``reset`` / ``get_action`` over ZMQ + msgpack, exactly like the
rldx / rlbench-3dda sidecars.

Observation / action contract (verified against the upstream configs)
---------------------------------------------------------------------
The default embodiment is ``robotwin`` (dual-arm Agilex Cobot Magic), the only
robot config shipped upstream (``configs/robot_configs/robotwin.yaml`` +
``assets/norm_stats/robotwin.json``):

* **3 cameras**, mapped positionally onto the upstream origin keys
  ``cam_high`` / ``cam_left_wrist`` / ``cam_right_wrist`` (model-space
  ``camera_top`` / ``camera_wrist_left`` / ``camera_wrist_right``), HWC uint8,
  resized to 256×256 by the server's ``FeatureTransform``.
* **14-D proprio state**: ``[arm_l(6) grip_l(1) arm_r(6) grip_r(1)]``, padded to
  the model's 55-D unified vector server-side.
* **Action chunk**: the server returns unnormalized ``action.arm.position(12)``
  + ``action.effector.position(2)`` per step; the sidecar flattens them (in that
  key order) to a ``(chunk, 14)`` array under ``"action"``. This adapter replays
  one 14-D step per :meth:`step`, refilling from the sidecar when the queue
  drains.

Quantization: the 8 GB-class dev GPU cannot hold the 6.38 B model (fp32 25.5 GB
/ bf16 12.8 GB), so the sidecar NF4-quantizes the Qwen3-VL backbone and keeps
the MoE expert in bf16 (``--quantization nf4``, the default); ``none`` for
≥16 GB cards.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.registry import POLICIES
from openral_sim.sidecar import SidecarClient

if TYPE_CHECKING:
    from openral_core import SimEnvironment, VLASpec

    from openral_sim.rollout import Observation

_log = structlog.get_logger(__name__)

_DEFAULT_MODEL_ID = "robbyant/lingbot-vla-v2-6b"
_DEFAULT_ROBO_NAME = "robotwin"
# Upstream FeatureTransform origin image keys, positionally aligned with the
# scene-side camera keys resolved below.
_LINGBOT_IMAGE_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
_DEFAULT_CAMERA_KEYS = ("camera_top", "camera_wrist_left", "camera_wrist_right")
_LINGBOT_STATE_DIM = 14
# Action chunk is rank-2: (chunk_len, action_dim).
_CHUNK_RANK_2D = 2
# 30 Hz policy; the upstream RoboTwin deploy replays a 25-step slice of the
# 50-step predicted chunk before replanning (``--use_length 25``).
_DEFAULT_REPLAN_STEPS = 25

_PORT_MIN = 20_000
_PORT_MAX = 39_999
_DEFAULT_TIMEOUT_MS = 120_000
_DEFAULT_BOOT_TIMEOUT_S = 1200.0  # first boot clones the repo + builds a torch-2.8 venv


def _opt_int(value: object, default: int) -> int:
    """Coerce a ``vla.extra`` value (typed ``object``) to int, else ``default``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _policy_default_port(model_id: str, robo_name: str) -> int:
    """Deterministically map (model, embodiment) to a default sidecar port.

    Two runs of the same checkpoint + embodiment share one sidecar; distinct
    checkpoints land on distinct ports so they never adopt each other's server.
    """
    key = f"lingbot_vla2|{model_id}|{robo_name}".encode()
    digest = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return _PORT_MIN + (digest % (_PORT_MAX - _PORT_MIN))


def _resolve_camera_keys(env_cfg: SimEnvironment, extra: dict[str, Any]) -> tuple[str, ...]:
    """Resolve the 3 scene-side camera keys aligned with the LingBot views.

    Precedence: ``vla.extra.camera_keys`` > scene ``cameras`` (first three) >
    the upstream default triple. The order is load-bearing — index 0 is the
    top/high camera, 1/2 are the left/right wrist cameras.
    """
    override = extra.get("camera_keys")
    if isinstance(override, (list, tuple)) and override:
        keys = tuple(str(k) for k in override)
    else:
        scene_cams = getattr(env_cfg.scene, "cameras", None)
        keys = tuple(str(k) for k in scene_cams) if scene_cams else _DEFAULT_CAMERA_KEYS
    if len(keys) < len(_LINGBOT_IMAGE_KEYS):
        raise ROSConfigError(
            f"lingbot_vla2 needs {len(_LINGBOT_IMAGE_KEYS)} cameras "
            f"({_LINGBOT_IMAGE_KEYS}); got {keys}. Set vla.extra.camera_keys."
        )
    return keys[: len(_LINGBOT_IMAGE_KEYS)]


_SIDECAR_VENV = Path.home() / ".cache" / "openral" / "lingbot-vla2-sidecar" / "source" / ".venv"


def _locate_sidecar_script() -> Path:
    override = os.environ.get("OPENRAL_LINGBOT_VLA2_SIDECAR_SCRIPT")
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise ROSConfigError(f"OPENRAL_LINGBOT_VLA2_SIDECAR_SCRIPT={override!r} is not a file.")
        return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "lingbot_vla2_sidecar.py"
        if candidate.is_file():
            return candidate
    raise ROSConfigError(
        f"Could not locate tools/lingbot_vla2_sidecar.py upwards from {here}. "
        "Set OPENRAL_LINGBOT_VLA2_SIDECAR_SCRIPT to its absolute path."
    )


def _sidecar_python() -> Path:
    """Resolve the py3.12 + torch-2.8 sidecar venv interpreter.

    ``lingbotvla`` pins torch==2.8.0 / transformers==4.57.3 / triton==3.4.0, so
    it lives in its own venv (default under
    ``~/.cache/openral/lingbot-vla2-sidecar``); override with
    ``OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON``. Phase 2 wires the git-clone + venv
    build (``tools/lingbot_vla2_sidecar.py`` provisioning) into this path.
    """
    override = os.environ.get("OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON")
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise ROSConfigError(f"OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON={override!r} is not a file.")
        return p
    default = _SIDECAR_VENV / "bin" / "python"
    if default.is_file():
        return default
    raise ROSConfigError(
        "lingbot_vla2 sidecar venv not found. It needs a py3.12 venv with the "
        "upstream `lingbotvla` package (torch==2.8.0). Provision it and set "
        "OPENRAL_LINGBOT_VLA2_SIDECAR_PYTHON, or run "
        "`python tools/lingbot_vla2_sidecar.py --help`."
    )


@dataclass
class _LingBotVla2Adapter:
    """:class:`PolicyAdapter` proxying the LingBot-VLA 2.0 sidecar.

    Predicts a full action chunk on the sidecar and replays one 16-D step per
    :meth:`step`; the queue refills (a new sidecar inference) when it drains.
    """

    spec: VLASpec
    device: str
    _client: SidecarClient
    _camera_keys: tuple[str, ...]
    _replan_steps: int = _DEFAULT_REPLAN_STEPS
    _queue: deque[NDArray[np.float32]] = field(default_factory=deque)
    _last_input: NDArray[np.uint8] | None = field(default=None)

    def reset(self) -> None:
        self._queue.clear()
        with contextlib.suppress(Exception):
            self._client.call("reset", {"robo_name": _DEFAULT_ROBO_NAME})

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        if not self._queue:
            self._refill(observation, instruction)
        return np.asarray(self._queue.popleft(), dtype=np.float32)

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return None if self._last_input is None else self._last_input.copy()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.call("close")
        self._client.close()

    def _refill(self, observation: Observation, instruction: str) -> None:
        images = observation.get("images", {})
        payload_images: dict[str, NDArray[np.uint8]] = {}
        for lingbot_key, scene_key in zip(_LINGBOT_IMAGE_KEYS, self._camera_keys, strict=True):
            img = images.get(scene_key)
            if img is None:
                raise ROSConfigError(
                    f"lingbot_vla2 expected camera {scene_key!r} on the observation; "
                    f"scene provides {list(images)}."
                )
            arr = np.asarray(img, dtype=np.uint8)
            payload_images[lingbot_key] = arr
            if lingbot_key == "cam_high":
                self._last_input = arr

        state = observation.get("state")
        if state is None:
            raise ROSConfigError("lingbot_vla2 requires a 14-D proprio `state` on the observation.")
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != _LINGBOT_STATE_DIM:
            raise ROSConfigError(
                f"lingbot_vla2 robotwin expects a {_LINGBOT_STATE_DIM}-D state "
                f"(arm_l6+grip1+arm_r6+grip1), got {state_arr.shape[0]}-D."
            )

        reply = self._client.call(
            "get_action",
            {
                "observation": {
                    "images": payload_images,
                    "state": state_arr,
                    "task": instruction,
                }
            },
        )
        chunk = np.asarray(self._client.require(reply, "action"), dtype=np.float32)
        if chunk.ndim != _CHUNK_RANK_2D:
            raise ROSConfigError(
                f"lingbot_vla2 sidecar returned action of shape {chunk.shape}; expected 2-D."
            )
        for action in chunk[: self._replan_steps]:
            self._queue.append(np.asarray(action, dtype=np.float32))


@POLICIES.register("lingbot_vla2")
def _build_lingbot_vla2(env_cfg: SimEnvironment) -> _LingBotVla2Adapter:
    """Build the LingBot-VLA 2.0 adapter behind its auto-managed ZMQ sidecar.

    YAML knobs (``vla.extra``): ``model_id``, ``robo_name``, ``camera_keys``,
    ``quantization`` (``nf4`` default / ``int8`` / ``none``), ``port``,
    ``replan_steps``, ``auto_spawn``. Environment overrides:
    ``OPENRAL_LINGBOT_VLA2_{QUANTIZATION,PORT,AUTO_SPAWN}``.
    """
    spec = env_cfg.vla
    extra = dict(spec.extra or {})

    model_id = str(extra.get("model_id") or spec.weights_uri or _DEFAULT_MODEL_ID)
    robo_name = str(extra.get("robo_name") or _DEFAULT_ROBO_NAME)
    quantization = str(
        os.environ.get("OPENRAL_LINGBOT_VLA2_QUANTIZATION") or extra.get("quantization", "nf4")
    ).lower()
    replan_steps = _opt_int(extra.get("replan_steps"), _DEFAULT_REPLAN_STEPS)
    camera_keys = _resolve_camera_keys(env_cfg, extra)

    host = "127.0.0.1"
    port_env = os.environ.get("OPENRAL_LINGBOT_VLA2_PORT")
    port = (
        int(port_env)
        if port_env
        else _opt_int(extra.get("port"), _policy_default_port(model_id, robo_name))
    )
    auto_spawn = os.environ.get("OPENRAL_LINGBOT_VLA2_AUTO_SPAWN", "1") != "0"

    launch_argv = [
        "env",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        str(_sidecar_python()),
        str(_locate_sidecar_script()),
        "--model",
        model_id,
        "--robo-name",
        robo_name,
        "--quantization",
        quantization,
        "--host",
        host,
        "--port",
        str(port),
    ]

    client = SidecarClient(
        name="lingbot-vla2",
        host=host,
        port=port,
        timeout_ms=_DEFAULT_TIMEOUT_MS,
        boot_timeout_s=_DEFAULT_BOOT_TIMEOUT_S,
        launch_argv=launch_argv,
        auto_spawn=auto_spawn,
        expected_identity={"model": model_id, "robo_name": robo_name},
    )
    _log.info("lingbot_vla2_connecting", model=model_id, robo_name=robo_name, port=port)
    client.connect()
    return _LingBotVla2Adapter(
        spec=spec,
        device="cuda",
        _client=client,
        _camera_keys=camera_keys,
        _replan_steps=replan_steps,
    )
