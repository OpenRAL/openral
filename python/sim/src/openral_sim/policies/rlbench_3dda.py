r"""3D Diffuser Actor policy adapter — RLBench keyframe inference via a sidecar.

ADR-0061. 3D Diffuser Actor (Ke et al., 2024, arXiv:2402.10885, MIT) predicts
end-effector keyposes for RLBench by diffusion. Its released PerAct 18-task
checkpoint pins an older stack (the ``MohitShridhar/RLBench@peract`` fork + CLIP
+ an Ada-compatible torch build) incompatible with the openral py3.12 workspace,
so — like the RLDX-1 adapter (:mod:`openral_sim.policies.rldx`) — it runs in its
own venv as a long-lived process driven over ZMQ. This module is the **openral
side**: a :class:`PolicyAdapter` that marshals each observation to the sidecar
(``tools/rlbench_3dda_sidecar.py``) and returns the 8-D keyframe the RLBench
scene backend executes.

Selected by the rSkill manifest ``model_family: "diffuser_actor"``.

The action is an 8-D keyframe ``[x y z qx qy qz qw gripper_open]`` consumed by
:mod:`openral_sim.backends.rlbench` (whose sidecar plans+executes it). The policy
keeps a 3-step observation history server-side, so this adapter is thin.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.registry import POLICIES
from openral_sim.sidecar import SidecarClient

if TYPE_CHECKING:
    from openral_core import SimEnvironment, VLASpec

    from openral_sim.rollout import Observation

_SIDECAR_PYTHON_ENV = "OPENRAL_RLBENCH_SIDECAR_PYTHON"
_SIDECAR_SCRIPT_ENV = "OPENRAL_RLBENCH_3DDA_SIDECAR_SCRIPT"
_REPO_ENV = "OPENRAL_3DDA_REPO"
_CKPT_ENV = "OPENRAL_3DDA_CHECKPOINT"
_INSTR_ENV = "OPENRAL_3DDA_INSTRUCTIONS"
_BOUNDS_ENV = "OPENRAL_3DDA_BOUNDS"

_RLBENCH_VENV = Path.home() / ".cache" / "openral" / "rlbench-policy" / ".venv"
_DEFAULT_REPO = Path.home() / ".cache" / "openral" / "policy-src" / "3d_diffuser_actor"
_PORT_MIN = 21_500
_PORT_MAX = 21_999
_DEFAULT_TIMEOUT_MS = 120_000
_DEFAULT_BOOT_TIMEOUT_S = 300.0


def _policy_default_port(rlbench_task: str, variation: int) -> int:
    key = f"3dda|{rlbench_task}|{variation}".encode()
    digest = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return _PORT_MIN + (digest % (_PORT_MAX - _PORT_MIN))


def _opt_int(value: object, default: int) -> int:
    """Coerce a ``backend_options`` value (typed ``object``) to int, else default."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _resolve_file(env_name: str, default: Path, what: str) -> Path:
    override = os.environ.get(env_name)
    p = Path(override).expanduser() if override else default
    if not p.exists():
        raise ROSConfigError(
            f"3D Diffuser Actor {what} not found at {p}. Set {env_name}. "
            "See docs/adr/0061-rlbench-benchmark-backend.md for the provisioning recipe."
        )
    return p


def _default_checkpoint() -> Path:
    import glob

    base = Path.home() / ".cache/huggingface/hub/models--katefgroup--3d_diffuser_actor/snapshots"
    hits = glob.glob(str(base / "*" / "diffuser_actor_peract.pth"))
    return Path(hits[0]) if hits else Path("/__missing_3dda_checkpoint__")


@dataclass
class _Diffuser3DActorAdapter:
    """:class:`PolicyAdapter` proxying the 3D Diffuser Actor policy sidecar."""

    spec: VLASpec
    device: str
    _client: SidecarClient
    _last_input: NDArray[np.uint8] | None = field(default=None)

    def reset(self) -> None:
        self._client.call("reset")

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        images = {
            k: np.asarray(v, dtype=np.uint8) for k, v in observation.get("images", {}).items()
        }
        if images:
            self._last_input = next(iter(images.values()))
        payload = {
            "observation": {
                "images": images,
                "point_clouds": {
                    k: np.asarray(v, dtype=np.float32)
                    for k, v in observation.get("point_clouds", {}).items()
                },
                "gripper_pose": np.asarray(observation["gripper_pose"], dtype=np.float32),
                "gripper_open": float(observation.get("gripper_open", 1.0)),
                "instruction": instruction,
            }
        }
        reply = self._client.call("get_action", payload)
        return np.asarray(self._client.require(reply, "action"), dtype=np.float32).reshape(-1)

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return None if self._last_input is None else self._last_input.copy()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.call("close")
        self._client.close()


def _sidecar_python() -> Path:
    override = os.environ.get(_SIDECAR_PYTHON_ENV)
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_PYTHON_ENV}={override!r} is not a file.")
        return p
    default = _RLBENCH_VENV / "bin" / "python"
    if default.is_file():
        return default
    raise ROSConfigError(
        "3D Diffuser Actor sidecar venv not found. It shares the rlbench-policy "
        f"py3.10 venv. Provision it and set {_SIDECAR_PYTHON_ENV} (ADR-0061)."
    )


def _locate_sidecar_script() -> Path:
    override = os.environ.get(_SIDECAR_SCRIPT_ENV)
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise ROSConfigError(f"{_SIDECAR_SCRIPT_ENV}={override!r} is not a file.")
        return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "rlbench_3dda_sidecar.py"
        if candidate.is_file():
            return candidate
    raise ROSConfigError(
        f"Could not locate tools/rlbench_3dda_sidecar.py upwards from {here}. Set "
        f"{_SIDECAR_SCRIPT_ENV} to its absolute path."
    )


@POLICIES.register("diffuser_actor")
def _build_diffuser_actor(env_cfg: SimEnvironment) -> _Diffuser3DActorAdapter:
    """Build the 3D Diffuser Actor adapter behind the policy sidecar.

    Reads the RLBench task + variation from ``scene.backend_options`` (the policy
    needs them to select the matching CLIP instruction embedding); resolves the
    sidecar venv + 3DDA repo + checkpoint/instructions/bounds (externally
    provisioned); spawns the sidecar and connects.
    """
    repo = _resolve_file(_REPO_ENV, _DEFAULT_REPO, "repo checkout")
    checkpoint = _resolve_file(_CKPT_ENV, _default_checkpoint(), "checkpoint")
    instructions = _resolve_file(
        _INSTR_ENV,
        Path.home()
        / ".cache/openral/rlbench-policy/instructions/instructions/peract/instructions.pkl",
        "instructions.pkl",
    )
    bounds = _resolve_file(
        _BOUNDS_ENV, repo / "tasks" / "18_peract_tasks_location_bounds.json", "bounds json"
    )

    opts = env_cfg.scene.backend_options
    rlbench_task = str(opts.get("rlbench_task") or "")
    if not rlbench_task:
        raise ROSConfigError(
            "diffuser_actor policy requires the rlbench scene's "
            "backend_options.rlbench_task to select the instruction embedding."
        )
    variation = _opt_int(opts.get("variation"), 0)
    image_size = int(env_cfg.scene.observation_height)
    host = "127.0.0.1"
    port = _opt_int(opts.get("policy_port"), _policy_default_port(rlbench_task, variation))
    auto_spawn = os.environ.get("OPENRAL_RLBENCH_AUTO_SPAWN", "1") != "0"

    launch_argv = [
        "env",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        str(_sidecar_python()),
        str(_locate_sidecar_script()),
        "--repo",
        str(repo),
        "--checkpoint",
        str(checkpoint),
        "--instructions",
        str(instructions),
        "--bounds",
        str(bounds),
        "--rlbench-task",
        rlbench_task,
        "--variation",
        str(variation),
        "--image-size",
        str(image_size),
        "--host",
        host,
        "--port",
        str(port),
    ]

    client = SidecarClient(
        name="rlbench-3dda",
        host=host,
        port=port,
        timeout_ms=_DEFAULT_TIMEOUT_MS,
        boot_timeout_s=_DEFAULT_BOOT_TIMEOUT_S,
        launch_argv=launch_argv,
        auto_spawn=auto_spawn,
        expected_identity={"model": "3d_diffuser_actor"},
    )
    client.connect()
    device = "cuda"
    return _Diffuser3DActorAdapter(spec=env_cfg.vla, device=device, _client=client)
