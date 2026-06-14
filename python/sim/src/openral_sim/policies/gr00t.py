"""NVIDIA Isaac GR00T policy adapter — out-of-process ZMQ sidecar (ADR-0046).

GR00T (N1.x / N2) cannot load in the Python 3.12-only workspace: Isaac-GR00T
pins Python 3.10 + flash-attn + CUDA. So, exactly like the ``rldx`` adapter,
GR00T runs in its own Python 3.10 sidecar venv and is driven over a ZMQ +
msgpack wire. This is a natural fit because **RLDX-1 is itself a GR00T-N1.5
finetune** — its sidecar mimics the upstream GR00T ``PolicyServer`` contract
(LIBERO-flat ``state.x`` / ``action.x`` keys under ``--use-sim-policy-wrapper``,
``--embodiment-tag``). We therefore *reuse* :class:`_Gr00tFamilySidecarAdapter`
verbatim (its ``_build_libero_obs`` already emits the exact GR00T LIBERO keys),
parameterized with ``family="gr00t"`` so it forks ``tools/gr00t_sidecar.py`` and
reads the ``OPENRAL_GR00T_*`` env namespace. The obs/action layout is dispatched
off the rSkill manifest's ``state_contract.layout`` (shared with the rldx
factory), so a non-LIBERO GR00T finetune is assembled with the right contract
rather than force-fed LIBERO keys.

The boot helper ``tools/gr00t_sidecar.py`` clones ``NVIDIA/Isaac-GR00T`` and
serves ``Gr00tPolicy`` over the same wire. Live boot + sim eval are
operator-run on a Python-3.10 GPU host (the model is unverifiable on the 8 GB
reference laptop without NF4); see ADR-0046 PR2.
"""

from __future__ import annotations

import os
from typing import Any

from openral_rskill._vla_core import resolve_image_preprocessing

from openral_sim.policies._policy_loading import load_manifest_for_spec
from openral_sim.policies.rldx import (
    _RLDX_CAMERA_PAIR_LEN,
    _RLDX_CHUNK_LEN,
    _env_bool,
    _Gr00tFamilySidecarAdapter,
    _require_scene_cameras,
    _resolve_sidecar_port,
    _resolve_state_layout,
)
from openral_sim.registry import POLICIES

# The gr00t-n17-libero rSkill targets nvidia/GR00T-N1.7-LIBERO, whose
# embodiment tag is ``LIBERO_PANDA`` (enum value ``libero_sim``) — confirmed
# against the checkpoint's modality configs. (The *base* GR00T-N1.7-3B only
# carries pretrain tags like ``OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`` /
# ``XDOF``; ``LIBERO_PANDA`` is finetune-only.) Overridable via
# ``OPENRAL_GR00T_EMBODIMENT_TAG`` / ``vla.extra.embodiment_tag`` for other
# checkpoints.
_GR00T_DEFAULT_EMBODIMENT_TAG = "LIBERO_PANDA"


@POLICIES.register("gr00t")  # type: ignore[arg-type]
def _build_gr00t(env_cfg: Any) -> _Gr00tFamilySidecarAdapter:
    """Build the auto-managed GR00T adapter from a SimEnvironment.

    Mirrors ``rldx._build_rldx`` but reads the ``OPENRAL_GR00T_*`` env
    namespace and forks ``tools/gr00t_sidecar.py``. The obs/action layout is
    dispatched off ``manifest.state_contract.layout`` (LIBERO by default, the
    contract RLDX-1 — itself a GR00T-N1.5 finetune — shares), so a GR00T
    checkpoint targeting RoboCasa / SimplerEnv / GR1 gets its own contract.

    YAML knobs (via ``vla.extra``): host, port, replan_steps, image_size,
    timeout_ms, camera_keys, auto_spawn, boot_timeout_s, quantization,
    embodiment_tag, model_id (set this to pick a LIBERO suite, e.g.
    ``nvidia/GR00T-N1.7-LIBERO/libero_spatial``).

    Environment overrides: ``OPENRAL_GR00T_HOST``, ``OPENRAL_GR00T_PORT``,
    ``OPENRAL_GR00T_AUTO_SPAWN``, ``OPENRAL_GR00T_BOOT_TIMEOUT_S``,
    ``OPENRAL_GR00T_QUANTIZATION``, ``OPENRAL_GR00T_EMBODIMENT_TAG``,
    ``OPENRAL_GR00T_MODEL_ID``, ``OPENRAL_GR00T_SIDECAR_SCRIPT``.
    """
    spec = env_cfg.vla
    extra = dict(spec.extra or {})
    host = os.environ.get("OPENRAL_GR00T_HOST", str(extra.get("host", "127.0.0.1")))

    manifest = load_manifest_for_spec(spec)
    manifest_steps = getattr(manifest, "n_action_steps", None) if manifest else None
    replan_steps = int(
        extra.get(
            "replan_steps",
            manifest_steps if manifest_steps is not None else _RLDX_CHUNK_LEN // 2,
        )
    )
    image_size = int(extra.get("image_size", 256))
    timeout_ms = int(extra.get("timeout_ms", 60_000))
    cam_keys_raw = extra.get("camera_keys")
    if isinstance(cam_keys_raw, (list, tuple)) and len(cam_keys_raw) == _RLDX_CAMERA_PAIR_LEN:
        camera_keys = (str(cam_keys_raw[0]), str(cam_keys_raw[1]))
    else:
        camera_keys = ("camera1", "camera2")
    auto_spawn = _env_bool("OPENRAL_GR00T_AUTO_SPAWN", bool(extra.get("auto_spawn", True)))
    boot_timeout_s = float(
        os.environ.get("OPENRAL_GR00T_BOOT_TIMEOUT_S") or extra.get("boot_timeout_s", 900.0)
    )
    quantization = str(
        os.environ.get("OPENRAL_GR00T_QUANTIZATION") or extra.get("quantization", "nf4")
    )
    embodiment_tag = str(
        os.environ.get("OPENRAL_GR00T_EMBODIMENT_TAG")
        or extra.get("embodiment_tag", _GR00T_DEFAULT_EMBODIMENT_TAG)
    )
    model_id_override = os.environ.get("OPENRAL_GR00T_MODEL_ID") or extra.get("model_id")

    ip = resolve_image_preprocessing(manifest, spec.extra)

    # Dispatch the obs/action layout off the manifest, exactly like the rldx
    # factory. GR00T checkpoints are mostly LIBERO today (the default), but a
    # RoboCasa / SimplerEnv / GR1 GR00T finetune declares its own
    # ``state_contract.layout`` and must not be force-fed the LIBERO obs
    # contract — that mismatch is the "always loads LIBERO" breakage. The
    # embodiment tag stays GR00T-specific (LIBERO_PANDA default, overridable),
    # since GR00T's tag enum differs from RLDX's GENERAL_EMBODIMENT family.
    layout = _resolve_state_layout(manifest)

    # Reject a too-few-cameras scene up front (before the multi-minute sidecar
    # boot) instead of failing opaquely on the first obs assembly.
    _require_scene_cameras(env_cfg, layout=layout, camera_keys=camera_keys, family="gr00t")

    # Port precedence: OPENRAL_GR00T_PORT > vla.extra.port > per-identity
    # default — keeps a GR00T sidecar off whatever port an RLDX run is using.
    port = _resolve_sidecar_port(
        port_env=os.environ.get("OPENRAL_GR00T_PORT"),
        extra_port=extra.get("port"),
        family="gr00t",
        model=str(model_id_override or getattr(manifest, "weights_uri", "") or spec.id),
        embodiment_tag=embodiment_tag,
        quantization=quantization,
        layout=layout,
    )

    return _Gr00tFamilySidecarAdapter(
        spec=spec,
        host=host,
        port=port,
        replan_steps=replan_steps,
        image_size=image_size,
        timeout_ms=timeout_ms,
        flip_180=ip.flip_180,
        flip_vertical=ip.flip_vertical,
        state_layout=layout,
        auto_spawn=auto_spawn,
        boot_timeout_s=boot_timeout_s,
        quantization=quantization,
        embodiment_tag=embodiment_tag,
        model_id=str(model_id_override) if model_id_override else None,
        family="gr00t",
        # GR00T's LIBERO_PANDA video modality is single-frame (horizon 1),
        # unlike RLDX-1's 4-frame LIBERO history.
        video_offsets=(0,),
        _camera_keys=camera_keys,
    )
