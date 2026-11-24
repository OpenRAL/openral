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

**Depth** is declared by the rSkill manifest, never fetched from a private
sidecar: ``policy_extras.depth_input`` must be ``"none"``, which states that the
checkpoint's System-1 never reads depth. That holds for DualVLN: its
``config.json`` sets ``system1: "nextdit_async"``, and InternNav's
``generate_traj`` (pinned commit, see ``tools/internvla_n1_sidecar.py``) feeds
``depths_dp`` only to the ``navdp`` branch. The sidecar's step wire still takes
a depth map, so a unit-metre placeholder is sent. A ``navdp`` System-1
checkpoint would need metric depth for its camera from the robot (an RGB-D
sensor, or ``DeployRuntime.estimated_depth_camera`` with a ``depth_estimator``
rSkill), checked by ``_check_depth_input`` via
``openral_core.depth_extrinsic.policy_depth_source``; it is still refused at
build time, because the policy observation carries RGB only today.
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
from openral_rskill._vla_core import resolve_rskill_repo_revision

from openral_sim._quantization import resolve_quant_plan, sidecar_quant_token
from openral_sim.policies._policy_loading import load_manifest_for_spec
from openral_sim.registry import POLICIES
from openral_sim.sidecar import SidecarClient

if TYPE_CHECKING:
    from openral_core import SimEnvironment, VLASpec

    from openral_sim.rollout import Observation

_log = structlog.get_logger(__name__)

_PORT_ENV = "OPENRAL_INTERNVLA_N1_SIDECAR_PORT"
_AUTO_SPAWN_ENV = "OPENRAL_INTERNVLA_N1_AUTO_SPAWN"
# The manifest's `policy_extras` key declaring what depth the checkpoint reads.
_DEPTH_INPUT_KEY = "depth_input"
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
    """``PolicyAdapter`` proxying the InternVLA-N1 sidecar.

    Only built for a manifest declaring ``policy_extras.depth_input: none``, so
    the sidecar gets a unit-metre depth placeholder (its step wire takes one and
    refuses all-zero depth); the checkpoint's System-1 never reads it.
    """

    spec: VLASpec
    device: str
    _client: SidecarClient
    _camera_key: str
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
        depth = np.ones(rgb.shape[:2], dtype=np.float32)  # depth_input: none

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


def _check_depth_input(
    env_cfg: SimEnvironment, skill: str, policy_extras: dict[str, object], camera: str
) -> None:
    """Refuse, before any sidecar boots, a checkpoint whose depth this deployment cannot feed.

    ``policy_extras.depth_input`` is required. ``none`` (System-1 never reads depth) needs
    nothing. ``metric`` needs depth for ``camera`` from the robot or scene
    (``openral_core.depth_extrinsic.policy_depth_source``: an RGB-D sensor, or
    ``vla.extra["estimated_depth_camera"]`` with a ``depth_estimator`` rSkill); even when
    that holds it is refused, because the policy observation carries RGB only today.

    Raises:
        ROSConfigError: ``depth_input`` is missing or unknown, the robot gives ``camera``
            no depth, or ``metric`` is declared at all (naming the missing transport).
    """
    from openral_core.depth_extrinsic import policy_depth_source

    depth_input = policy_extras.get(_DEPTH_INPUT_KEY)
    if depth_input == "none":
        _log.info("internvla_n1.depth_unused", skill=skill, depth_input="none")
        return
    if depth_input != "metric":
        raise ROSConfigError(
            f"internvla_n1: rSkill {skill!r} must declare policy_extras.{_DEPTH_INPUT_KEY} "
            f"'none' (System-1 never reads depth, config.json system1: nextdit*) or 'metric' "
            f"(navdp*), got {depth_input!r}."
        )
    description = getattr(env_cfg, "robot_description", None)
    if description is None:
        raise ROSConfigError(
            f"internvla_n1: rSkill {skill!r} reads metric depth; its depth source is checked "
            "against the robot manifest, and none was given."
        )
    estimated = env_cfg.vla.extra.get("estimated_depth_camera")
    source = policy_depth_source(
        list(description.sensors),
        camera,
        estimated_depth_camera=str(estimated) if estimated else None,
    )
    raise ROSConfigError(
        f"internvla_n1: rSkill {skill!r} reads metric depth for {camera!r}, which "
        f"{description.name!r} provides on {source!r}'s depth stream, but the policy "
        "observation carries RGB only (WorldState ingests no depth images), so this "
        "checkpoint cannot run yet."
    )


@POLICIES.register(
    "internvla_n1",
    install_groups=("rldx",),
    required_imports=("zmq", "msgpack"),
    install_note=(
        "The policy runs in tools/internvla_n1_sidecar.py's own auto-provisioned "
        "Python 3.12 venv (transformers 4.51 pin)."
    ),
)
def _build_internvla_n1(env_cfg: SimEnvironment) -> _InternVLAN1Adapter:
    """Build the InternVLA-N1 adapter behind its auto-spawned sidecar.

    Resolves the checkpoint repo from the rSkill manifest and the
    quantization via ``resolve_quant_plan`` (shared precedence), derives a per-identity port so two
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
    # The manifest's ``@<sha>`` pin must reach both loaders in the sidecar; HF
    # ignores one glued onto the repo id.
    repo, revision = resolve_rskill_repo_revision(
        str(env_cfg.vla.weights_uri), adapter_name="InternVLA-N1"
    )
    model_ref = f"{repo}@{revision}" if revision else repo

    quantization = sidecar_quant_token(
        resolve_quant_plan(env_cfg.vla, manifest, default="bf16"),
        frozenset({"none", "nf4", "int8"}),
        "internvla_n1",
    )

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

    _check_depth_input(env_cfg, manifest.name, manifest.policy_extras, camera_key)

    port_env = os.environ.get(_PORT_ENV)
    port = int(port_env) if port_env else _derive_port(model_ref, quantization)
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
        *(["--revision", revision] if revision else []),
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
            "revision": revision,
            "quantization": quantization,
        },
    )
    client.connect()
    return _InternVLAN1Adapter(
        spec=env_cfg.vla, device="cuda", _client=client, _camera_key=camera_key
    )
