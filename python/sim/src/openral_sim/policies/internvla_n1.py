"""InternVLA-N1 policy adapter — dual-system VLN inference via a sidecar.

InternVLA-N1 / DualVLN (InternRobotics, arXiv:2512.08186) is a navigation
foundation model: a Qwen2.5-VL-7B System-2 plans a pixel goal from RGB history +
instruction, and a diffusion-transformer System-1 turns it into a base
trajectory. Upstream InternNav pins ``transformers==4.51.0``, incompatible with
the py3.12 workspace, so — like the ``rldx`` adapter — inference runs in an
isolated venv as a long-lived ZMQ process (``tools/internvla_n1_sidecar.py``
boots ``tools/_internvla_n1_server.py``).

Selected by the rSkill manifest ``model_family: "internvla_n1"``.

The adapter consumes ``observation["images"][<cam>]`` (RGB uint8) and returns a
6-D ``BODY_TWIST`` action row ``[vx, vy, vz, wx, wy, wz]`` in the base frame.
After System-2 emits STOP the adapter latches and returns zero twist for the
rest of the episode (the base halts; episode termination is the env's /
reasoner's call).

**Depth** is sourced monocularly from the DA3 sidecar
(:mod:`openral_sim.da3_depth`) — the same ``depth-anything/DA3-SMALL`` model the
SLAM/nvblox stack uses — so it is real metric depth and works identically on a
sim-rendered RGB frame and a real camera (no robot depth sensor required). The
DualVLN checkpoint's ``nextdit`` System-1 is RGB+latent conditioned and does not
actually consume depth (verified against the model source), so
``OPENRAL_INTERNVLA_N1_DEPTH=none`` skips DA3 for it; a depth-consuming ``navdp``
System-1 checkpoint keeps the default ``da3``.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_observability import inference_span
from openral_rskill._vla_core import resolve_rskill_repo_id

from openral_sim.da3_depth import DEFAULT_DA3_PORT, Da3DepthClient
from openral_sim.policies._policy_loading import load_manifest_for_spec
from openral_sim.registry import POLICIES
from openral_sim.sidecar import SidecarClient

if TYPE_CHECKING:
    from openral_core import SimEnvironment, VLASpec

    from openral_sim.rollout import Observation

_log = structlog.get_logger(__name__)

_PORT_ENV = "OPENRAL_INTERNVLA_N1_SIDECAR_PORT"
_AUTO_SPAWN_ENV = "OPENRAL_INTERNVLA_N1_AUTO_SPAWN"
# Depth source. `da3` (default) pulls real monocular metric depth from the DA3
# sidecar (openral_sim.da3_depth) — the SAME model the SLAM/nvblox stack uses,
# shared via ping-reuse; works identically on a sim-rendered RGB frame and a
# real camera. `none` sends a unit-depth placeholder: valid ONLY for the
# depth-agnostic `nextdit` System-1 (which InternVLA-N1-DualVLN uses — its
# trajectory head is RGB+latent conditioned, verified against the model source);
# a `navdp` System-1 checkpoint genuinely consumes depth and needs `da3`.
_DEPTH_SOURCE_ENV = "OPENRAL_INTERNVLA_N1_DEPTH"
_DA3_PORT_ENV = "OPENRAL_INTERNVLA_N1_DA3_PORT"
_PORT_MIN = 22_000
_PORT_MAX = 22_499
# One S2 replan is a full 7B-VLM generate on an NF4 8 GiB card — keep the
# per-step timeout generous; S1-only steps are fast.
_DEFAULT_TIMEOUT_MS = 180_000
# First boot provisions the sidecar venv (torch + transformers wheels).
_DEFAULT_BOOT_TIMEOUT_S = 1_800.0
_TWIST_DIM = 6


def _derive_port(model: str, quantization: str) -> int:
    digest = int.from_bytes(
        hashlib.sha256(f"internvla_n1|{model}|{quantization}".encode()).digest()[:4], "big"
    )
    return _PORT_MIN + (digest % (_PORT_MAX - _PORT_MIN))


def _locate_sidecar_script() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "internvla_n1_sidecar.py"
        if candidate.is_file():
            return candidate
    raise ROSConfigError(f"Could not locate tools/internvla_n1_sidecar.py upwards from {here}.")


@dataclass
class _InternVLAN1Adapter:
    """:class:`PolicyAdapter` proxying the InternVLA-N1 sidecar.

    Depth is sourced monocularly from the DA3 sidecar (``_da3``) rather than a
    robot depth sensor, so the same adapter runs on a sim-rendered RGB frame and
    a real RGB camera. When ``_da3`` is ``None`` (``OPENRAL_INTERNVLA_N1_DEPTH=
    none``) a unit-depth placeholder is sent — valid only for DualVLN's
    depth-agnostic ``nextdit`` System-1.
    """

    spec: VLASpec
    device: str
    _client: SidecarClient
    _camera_key: str
    _da3: Da3DepthClient | None
    _stopped: bool = False
    _last_input: NDArray[np.uint8] | None = field(default=None)

    def reset(self) -> None:
        self._client.call("reset")
        self._stopped = False

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        twist = np.zeros(_TWIST_DIM, dtype=np.float32)
        if self._stopped:
            return twist

        images = observation.get("images") or {}
        if self._camera_key not in images:
            raise ROSConfigError(
                f"internvla_n1: camera {self._camera_key!r} not in observation images "
                f"{sorted(images)}. Set vla.extra.camera_keys or fix scene cameras."
            )
        rgb = np.ascontiguousarray(images[self._camera_key], dtype=np.uint8)
        self._last_input = rgb
        # Real DA3 metric depth (same model the SLAM stack uses), or a unit
        # placeholder for the depth-agnostic nextdit checkpoint.
        depth = (
            self._da3.infer(rgb)
            if self._da3 is not None
            else np.ones(rgb.shape[:2], dtype=np.float32)
        )

        with inference_span(kind="single", engine="sidecar"):
            reply = self._client.call(
                "step", {"rgb": rgb, "depth": depth, "instruction": instruction}
            )
        v, w = (float(x) for x in self._client.require(reply, "twist"))
        if bool(reply.get("stop")):
            self._stopped = True
            _log.info("internvla_n1.stop", instruction=instruction)
            return twist
        twist[0] = v  # forward, base frame
        twist[5] = w  # yaw rate
        return twist

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return None if self._last_input is None else self._last_input.copy()

    def close(self) -> None:
        # Only shut the sidecar server down if THIS client spawned it; a reused
        # external sidecar (auto_spawn=0, e.g. a benchmark's shared instance)
        # must outlive one episode. SidecarClient.close() reaps a spawned child.
        if getattr(self._client, "_child", None) is not None:
            with contextlib.suppress(Exception):
                self._client.call("close")
        self._client.close()
        if self._da3 is not None:
            self._da3.close()


@POLICIES.register("internvla_n1")
def _build_internvla_n1(env_cfg: SimEnvironment) -> _InternVLAN1Adapter:
    """Build the InternVLA-N1 adapter behind its auto-spawned sidecar.

    Resolves the checkpoint repo + quantization from the rSkill manifest
    (``vla.quantization`` overrides), derives a per-identity port so two
    checkpoints never share a sidecar, and connects (spawning
    ``tools/internvla_n1_sidecar.py`` under the current interpreter when no
    sidecar answers).
    """
    manifest = load_manifest_for_spec(env_cfg.vla)
    if manifest is None:
        raise ROSConfigError(
            "internvla_n1 requires an rSkill manifest reference in weights_uri "
            "(e.g. rskills/internvla-n1-mobile-vln-nf4) — raw URIs carry no "
            "action contract."
        )
    repo = resolve_rskill_repo_id(str(env_cfg.vla.weights_uri), adapter_name="InternVLA-N1")

    quant_cfg = env_cfg.vla.quantization or manifest.quantization
    # ``dtype`` is a QuantizationDtype str-enum whose ``str()`` is
    # ``"QuantizationDtype.INT4"`` — take ``.value`` (falls back to a plain str).
    dt = getattr(quant_cfg, "dtype", None)
    dtype = str(getattr(dt, "value", dt) or "none").lower()
    quantization = {"int4": "nf4", "nf4": "nf4", "int8": "int8"}.get(dtype, "none")

    camera_keys = env_cfg.vla.extra.get("camera_keys")
    if isinstance(camera_keys, (list, tuple)) and camera_keys:
        camera_key = str(camera_keys[0])
    else:
        # InternVLA-N1 is a VLN *navigation* policy: it needs a forward
        # egocentric view down the base's travel direction, which the
        # robocasa backend synthesizes as `observation.images.head`
        # (see robots/panda_mobile/robot.yaml `head` sensor +
        # `openral_sim.backends.robocasa.render_head_view`, gated on
        # OPENRAL_ROBOCASA_HEAD_CAM=1). The scene's camera1/2/3 are the
        # arm-mounted agentview / eye_in_hand streams that stare at the
        # counter manipulation workspace — useless for navigation. Set
        # `vla.extra.camera_keys` to override.
        camera_key = "head"

    port_env = os.environ.get(_PORT_ENV)
    port = int(port_env) if port_env else _derive_port(repo, quantization)
    auto_spawn = os.environ.get(_AUTO_SPAWN_ENV, "1") != "0"

    launch_argv = [
        sys.executable,
        str(_locate_sidecar_script()),
        "--model",
        repo,
        "--port",
        str(port),
        "--quantization",
        quantization,
    ]
    client = SidecarClient(
        name="internvla-n1",
        host="127.0.0.1",
        port=port,
        timeout_ms=_DEFAULT_TIMEOUT_MS,
        boot_timeout_s=_DEFAULT_BOOT_TIMEOUT_S,
        launch_argv=launch_argv,
        auto_spawn=auto_spawn,
        expected_identity={
            "family": "internvla_n1",
            "model": repo,
            "quantization": quantization,
        },
    )
    client.connect()

    # Depth source: DA3 (real monocular metric depth, shared with SLAM) by
    # default; `none` skips it (DualVLN's nextdit System-1 ignores depth).
    depth_source = os.environ.get(_DEPTH_SOURCE_ENV, "da3").lower()
    da3: Da3DepthClient | None = None
    if depth_source == "da3":
        da3_port_env = os.environ.get(_DA3_PORT_ENV)
        da3 = Da3DepthClient(
            port=int(da3_port_env) if da3_port_env else DEFAULT_DA3_PORT,
            auto_spawn=auto_spawn,
        )
        da3.connect()
    elif depth_source != "none":
        raise ROSConfigError(
            f"{_DEPTH_SOURCE_ENV}={depth_source!r} is invalid; use 'da3' or 'none'."
        )

    return _InternVLAN1Adapter(
        spec=env_cfg.vla, device="cuda", _client=client, _camera_key=camera_key, _da3=da3
    )
