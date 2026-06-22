r"""RLDX-1 auto-managed sidecar adapter via ZMQ REQ/REP + msgpack.

Background
----------
RLWRLD publishes the RLDX-1 family of VLAs (a Qwen3-VL-8B backbone with
a Multi-Stream Action Transformer flow-matching head, ~6.9 B params,
bf16 on disk ≈ 14 GiB). The reference inference path ships in
https://github.com/RLWRLD/RLDX-1 as ``rldx/eval/run_rldx_server.py`` —
a ZMQ REP server (``tcp://<host>:<port>``) that holds the ``RLDXPolicy``
in memory and answers ``get_action`` / ``reset`` / ``ping`` requests.

Why an out-of-process sidecar (and not in-process)
--------------------------------------------------
In-process loading of ``rldx`` into the openral Python 3.12 venv was
evaluated and rejected as impractical:

* ``rldx`` pins ``requires-python = "==3.10.*"`` plus strict majors on
  ``numpy==1.26.4`` / ``torch==2.7.0`` / ``transformers==4.57.0`` /
  ``flash-attn==2.7.4.post1``. The openral workspace is Python 3.12
  with ``numpy>=2`` / ``torch>=2.10`` / ``transformers>=5`` (CLAUDE.md
  §3) — downgrading would break smolvla, pi05, xVLA, ACT, DP.
* The HF checkpoints (``RLWRLD/RLDX-1-FT-*``) do NOT ship
  ``modeling_rldx.py``, so there is no ``trust_remote_code`` escape —
  ``AutoModel.from_pretrained`` strictly requires the local ``rldx``
  package, which registers ``architectures=["RLDX"]`` at import time.
* Force-installing ``rldx`` with ``--no-deps`` cascades through 15+
  packages with major-version-incompatible APIs (albumentations 2.x vs
  1.4 pinned, lmdb, av, dm-tree, …) — and even past the imports the
  model load goes through transformers 5.x against rldx code written
  for 4.57.
* Reimplementing the policy inference layer in openral would mean
  porting ~25 kLOC of custom Triton kernels + MSAT flow-matching code
  (``rldx/inference/``, ``rldx/model/``) — out of scope.

So we run the upstream server in its own Python 3.10 venv (the only
3.10 environment on disk; one venv reused across every RLDX-1
checkpoint) and talk to it from this adapter over ZMQ. The friction
the user pays for "extra venvs" is exactly that one cached environment
at ``~/.cache/openral/rldx-sidecar/source/.venv``; we do **not** create
one per rSkill.

Auto-managed lifecycle (the openral piece)
------------------------------------------
This adapter manages the sidecar process so end users never invoke the
boot helper by hand:

* On ``__post_init__`` we ping the server at ``host:port``.
* If the ping fails and ``auto_spawn=True`` (the default; toggle via
  ``OPENRAL_RLDX_AUTO_SPAWN=0`` or ``vla.extra.auto_spawn: false``), we
  ``Popen`` :mod:`tools.rldx_sidecar` with the manifest-resolved model
  id, port, quantization, and embodiment tag, then poll the ping until
  the server answers or ``boot_timeout_s`` elapses (default 900 s —
  the first boot on a fresh host includes the ``git clone`` and
  ``uv sync`` of the upstream repo, which is the slow path).
* ``close()`` terminates the child we spawned. If the server was
  already running when we connected (an operator launched the sidecar
  by hand, or another openral process is sharing it), ``close()`` is a
  no-op for the subprocess — we don't reach into other people's PIDs.

The wire-format and obs-layout code below is unchanged from the
non-auto-managed variant; the auto-spawn block is a thin wrapper that
gives users a single-command experience without breaking the manual
boot path for debugging or shared-host setups.

This adapter is registered as a ``POLICIES`` entry; the rSkill
manifest describes the upstream RLDX checkpoint and
``model_family: "rldx"`` selects this adapter.

Wire protocol
-------------
Source of truth: ``rldx/policy/server_client.py`` and
``rldx/eval/run_rldx_server.py`` in the upstream repo (Apache-2.0).

Transport: ``zmq.REQ`` ↔ ``zmq.REP``, framed by ``msgpack``. NumPy
arrays are encoded via ``np.save`` into an in-memory buffer wrapped in
``{"__ndarray_class__": True, "as_npy": <bytes>}``.

Request::

    {"endpoint": "get_action",
     "data": {"observation": <obs_dict>, "options": None},
     "api_token": None}

Response::

    [action_dict, info_dict]

For LIBERO eval the server is booted with ``--use-sim-policy-wrapper``,
which wraps the policy in ``RLDXSimPolicyWrapper`` (rldx/policy/...).
The wrapper consumes a flat-keyed batched-temporal observation::

    video.image:        (B=1, T=1, 256, 256, 3) uint8  -- agentview, post-flip
    video.wrist_image:  (B=1, T=1, 256, 256, 3) uint8  -- eye-in-hand, post-flip
    state.x / y / z:                (B=1, T=1, 1) float32
    state.roll / pitch / yaw:       (B=1, T=1, 1) float32
    state.gripper:                  (B=1, T=1, 2) float32
    annotation.human.action.task_description: tuple[str] of len B

…and returns the canonical LIBERO action chunk as flat keys::

    action.x / y / z / roll / pitch / yaw / gripper: (B=1, T=16, 1) float32

The adapter assembles those into the 7-D LIBERO action vector
``[dx, dy, dz, droll, dpitch, dyaw, gripper]`` and queues the chunk
(default 16 actions) for replay.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_observability import inference_span

from openral_sim.policies._policy_loading import load_manifest_for_spec
from openral_sim.registry import POLICIES

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation


_log = structlog.get_logger(__name__)


_RLDX_CHUNK_LEN = 16
# Server-side flat keys the upstream RLDXSimPolicyWrapper aliases to
# `video.front_view` / `video.left_wrist_view` (see
# rldx/policy/rldx_policy.py:314-317).
_RLDX_AGENTVIEW_KEY = "video.image"
_RLDX_WRIST_KEY = "video.wrist_image"
_RLDX_TASK_KEY = "annotation.human.action.task_description"
_RLDX_ACTION_AXES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
# The FT-LIBERO checkpoint's processor pins the general_embodiment
# modality config: video.delta_indices = [-6, -4, -2, 0]. We pass T=4
# stacked frames per camera (oldest first); the adapter maintains a
# rolling 7-frame history per camera and samples at offsets {-6,-4,-2,0}.
_RLDX_VIDEO_HORIZON = 4
_RLDX_VIDEO_OFFSETS = (-6, -4, -2, 0)
_RLDX_VIDEO_HISTORY = -_RLDX_VIDEO_OFFSETS[0] + 1  # = 7
# LIBERO state contract (matches openral_sim.backends.libero._wrap_obs):
# eef_pos(3) ‖ axisangle(3) ‖ gripper_qpos(2). Encoded as a constant so the
# 8-D check in `_pick_state` does not earn a magic-number flag.
_LIBERO_STATE_DIM = 8
# RLDXSimPolicyWrapper returns each action axis as a (B=1, T, 1) tensor;
# these constants name the rank checks in `_assemble_action_chunk`.
_RLDX_ACTION_RANK_BT1 = 3
_RLDX_ACTION_RANK_T1 = 2
# `camera_keys` from the SimEnvironment must be a 2-tuple (agentview, wrist).
_RLDX_CAMERA_PAIR_LEN = 2
# Server response envelope is exactly ``[action_dict, info_dict]``.
_RLDX_RESPONSE_ENVELOPE_LEN = 2

# ─── GR1 (Fourier ArmsAndWaistFourierHands) state/action layout ──────────
#
# RLDX-1-FT-GR1 is NATIVE to the Fourier GR-1 humanoid. The model card's
# inference example uses `EmbodimentTag.GENERAL_EMBODIMENT`; that slot's
# modality keys in processor_config.json + statistics.json match the
# Fourier BASIC composite exactly:
#
#   state.right_arm:  7-D  (right arm joints)
#   state.left_arm:   7-D  (left arm joints)
#   state.waist:      3-D
#   state.right_hand: 6-D  (Fourier dexhand)
#   state.left_hand:  6-D  (Fourier dexhand)
#                    ────
#   total:           29-D  = Fourier BASIC composite
#
# (Aside: the FT-GR1 checkpoint ALSO carries `humanoid_everyday_g1` /
# `_h1` / `neural_gr1` modality configs as cross-embodiment slots used
# during pretraining — `humanoid_everyday_g1` is NVIDIA's Unitree-G1
# dataset, NOT the deployment target. The README is explicit:
# "Fourier GR-1 humanoid platform … arms + waist + Fourier hands".)
#
# The openral GR1 RoboCasa scene emits a 29-D state in the order
# ``[waist(3) | right_arm(7) | left_arm(7) | right_hand(6) | left_hand(6)]``,
# matching the upstream GR1ArmsAndWaistKeyConverter.map_obs output
# (see openral_sim.backends.robocasa._wrap_obs_gr1). The dims are
# already Fourier-native (6-DoF dexhand, NOT 11-D qpos) so no
# trimming/clipping is needed.
_GR1_STATE_SLICES: dict[str, tuple[int, int]] = {
    "state.waist": (0, 3),
    "state.right_arm": (3, 10),
    "state.left_arm": (10, 17),
    "state.right_hand": (17, 23),
    "state.left_hand": (23, 29),
}
# Ordered to match the openral GR1 state ordering (waist first):
# ``[waist(3) | right_arm(7) | left_arm(7) | right_hand(6) | left_hand(6)]``.
# This is the same layout the openral RoboCasa GR1 backend unflattens
# back into a robosuite-keyed dict for env.step
# (see openral_sim.backends.robocasa._to_gr1_action_dict).
_GR1_ACTION_KEYS = ("waist", "right_arm", "left_arm", "right_hand", "left_hand")
# FT-GR1 general_embodiment registers the camera as `ego_view`.
_GR1_VIDEO_KEY = "video.ego_view"
# Language modality for general_embodiment is "annotation.human.coarse_action",
# distinct from the LIBERO-flat wrapper's "annotation.human.action.task_description".
_GR1_TASK_KEY = "annotation.human.coarse_action"
# Fourier GR-1 BASIC composite (29-D): right_arm(7) + left_arm(7) + waist(3)
# + right_hand(6) + left_hand(6). Total adds up to 29.
_GR1_BASIC_DIM = 29

# ─── RC365 (RoboCasa-365 on PandaMobile) state/action layout ──────────────
#
# RLDX-1-FT-RC365 targets the RoboCasa-365 cross-task benchmark on
# PandaMobile. The model card's inference example uses
# `EmbodimentTag.GENERAL_EMBODIMENT`; that slot's modality config
# (verified from processor_config.json + statistics.json) is:
#
#   video (×3, T=4 each):
#       video.robot0_agentview_left
#       video.robot0_agentview_right
#       video.robot0_eye_in_hand
#   state (5 keys, 16-D total):
#       state.end_effector_position_relative  (3)
#       state.end_effector_rotation_relative  (4)  — quaternion
#       state.gripper_qpos                    (2)
#       state.base_position                   (3)
#       state.base_rotation                   (4)  — quaternion
#   action (5 keys, 12-D total):
#       action.end_effector_position  (3)  — delta
#       action.end_effector_rotation  (3)  — delta axis-angle
#       action.gripper_close          (1)
#       action.base_motion            (4)  — delta (dx, dy, dyaw, dz)
#       action.control_mode           (1)
#   language: annotation.human.task_description
#
# Our openral RoboCasa scene already emits a 16-D `human300_16d` state and
# the same three camera streams (camera1/2/3 = agentview_left / right /
# eye_in_hand). human300 layout: eef_pos(3) + eef_quat(4) + base_pos(3) +
# base_rot(4) + grip(2). RC365 puts gripper BEFORE base, so we re-slice.
_RC365_STATE_SLICES_FROM_HUMAN300: dict[str, tuple[int, int]] = {
    "state.end_effector_position_relative": (0, 3),
    "state.end_effector_rotation_relative": (3, 7),
    "state.gripper_qpos": (14, 16),
    "state.base_position": (7, 10),
    "state.base_rotation": (10, 14),
}
_RC365_VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
_RC365_TASK_KEY = "annotation.human.task_description"
# PandaMobile BASIC composite layout the openral RoboCasa adapter
# consumes: arm_osc(6) + gripper(1) + base(3) + torso/mode(1) -> 11-D.
# RC365 outputs 12-D total across 5 keys; we concatenate in the natural
# order and let `robocasa.py`'s env-dim skew-handler (lines ~190-218)
# absorb the 12→11 trimming (drops the last dim, which is the
# control_mode flag the env doesn't consume directly).
_RC365_ACTION_KEYS = (
    "end_effector_position",  # 3
    "end_effector_rotation",  # 3
    "gripper_close",  # 1
    "base_motion",  # 4
    "control_mode",  # 1  (last — robocasa.py trims it for env_dim==11)
)
_RC365_ACTION_DIM = 12

# ─── SimplerEnv (RLDX-1-FT-SIMPLER-{WIDOWX,GOOGLE}) wire schema ─────────────
#
# Both checkpoints were trained against the SimplerEnv legacy MS2
# observation shape — an 8-D ``obs['agent']['eef_pos']``: ``[x, y, z,
# qw, qx, qy, qz, gripper]``. The openral SimplerEnv backend rebuilds
# that vector from the MS3 v3.0.x ``ee_gripper_link`` (WidowX) /
# ``link_ee`` (Google) pose + last qpos channel — see
# ``openral_sim.backends.simpler_env._compute_eef_pos``.
#
# The upstream wire schema is defined by ``rldx/eval/sim/SimplerEnv/
# simpler_env.py``:
#
# WidowX / Bridge-Data ("oxe_widowx" / "bridge_orig"):
#   video.image_0           — single 256x320 RGB
#   state.x / .y / .z       — TCP world position
#   state.roll/pitch/yaw    — bridge-rotation-corrected Euler RPY
#   state.pad               — constant 0 (sentinel, schema requires it)
#   state.gripper           — raw last qpos value (0 closed → 0.037 open)
#   action.{x,y,z,roll,pitch,yaw,gripper}
#
# Google / Fractal20220817 ("oxe_google" / "fractal20220817_data"):
#   video.image             — single 256x320 RGB
#   state.x / .y / .z       — TCP world position
#   state.r{x,y,z,w}        — TCP orientation as xyzw quat
#   state.gripper           — gripper "closedness" = 1 - raw_open
#   action.{x,y,z,roll,pitch,yaw,gripper}
#
# The upstream Bridge env's orientation rotation matrix takes the SAPIEN
# wxyz quat → rotation matrix → multiplies by ``default_rot.T`` →
# extracts ``(roll, pitch, yaw)`` Euler. ``default_rot`` here matches
# the upstream constant exactly.
_SIMPLER_BRIDGE_DEFAULT_ROT: NDArray[np.float32] = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
)
_SIMPLER_EEF_DIM = 8
_SIMPLER_LANG_KEY = "annotation.human.action.task_description"
_SIMPLER_VIDEO_KEY_WIDOWX = "video.image_0"
_SIMPLER_VIDEO_KEY_GOOGLE = "video.image"
_SIMPLER_WIDOWX_IMAGE_HW: tuple[int, int] = (256, 320)
_SIMPLER_GOOGLE_IMAGE_HW: tuple[int, int] = (256, 320)
# Layout → upstream ``EmbodimentTag`` enum name (drives the modality
# config the sidecar uses to validate obs / route actions). Verified
# against ``rldx/data/embodiment_tags.py`` at the RLDX-1 commit pinned
# by ``tools/rldx_sidecar.py``. ALSO cross-checked against each FT
# checkpoint's `processor_config.json` `modality_configs` keys:
# `OXE_WIDOWX` / `OXE_GOOGLE` are enum *names* that exist in the enum
# module, but the published FT-SIMPLER-* checkpoints' processor only
# contains the OXE_BRIDGE_ORIG (`bridge_orig`) and OXE_FRACTAL
# (`fractal20220817_data`) modality buckets — `PolicyLoader.load` does
# `modality_configs[embodiment_tag.value]` and crashes with `KeyError:
# oxe_widowx` if we pass the wrong name. The map below uses the names
# that actually round-trip through the FT checkpoints we ship rSkills
# for.
_RLDX_LAYOUT_TO_EMBODIMENT_TAG: dict[str, str] = {
    "libero": "GENERAL_EMBODIMENT",
    "gr1": "GENERAL_EMBODIMENT",
    "rc365": "GENERAL_EMBODIMENT",
    "simpler_widowx": "OXE_BRIDGE_ORIG",
    "simpler_google": "OXE_FRACTAL",
}
# Non-default obs/action layouts the sidecar adapter knows how to assemble.
# Anything not listed here (including a missing/unknown manifest layout) falls
# back to the LIBERO-flat contract.
_RLDX_NON_DEFAULT_LAYOUTS: tuple[str, ...] = ("gr1", "rc365", "simpler_widowx", "simpler_google")

# Distinct camera streams each layout's obs builder reads off the scene. The
# Gr00t-family sidecar reads ``_camera_keys`` positionally and has NO single-view
# fallback (unlike the in-process lerobot adapters, which resolve their camera
# list from ``scene.cameras`` and adapt) — a missing stream surfaces as a cryptic
# mid-rollout ``observation.images[...]`` KeyError *after* the multi-minute
# sidecar boot. We use this to reject the pairing up front instead. Layouts:
# libero=agentview+wrist, rc365=agentview_left/right+eye_in_hand, gr1/simpler=ego.
_RLDX_LAYOUT_CAMERA_COUNT: dict[str, int] = {
    "libero": 2,
    "gr1": 1,
    "rc365": 3,
    "simpler_widowx": 1,
    "simpler_google": 1,
}

# Deterministic per-identity default port range. When the user pins neither
# OPENRAL_<FAMILY>_PORT nor vla.extra.port, we derive the sidecar port from the
# policy identity so two *different* checkpoints never collide on a shared
# default (the old hard 5555 default) while two evals of the *same* checkpoint
# still land on the same port and legitimately share one sidecar. 20000–39999
# sits clear of well-known ports and the usual ephemeral range.
_SIDECAR_PORT_MIN = 20000
_SIDECAR_PORT_MAX = 40000


def _resolve_state_layout(manifest: Any) -> str:
    """Map an rSkill manifest's ``state_contract.layout`` to a sidecar layout.

    Returns one of :data:`_RLDX_NON_DEFAULT_LAYOUTS` when the manifest declares
    it, else ``"libero"`` (the flat LIBERO contract). Shared by the ``rldx``
    and ``gr00t`` factories so both dispatch obs assembly off the manifest
    instead of hardcoding a single embodiment.
    """
    if manifest is not None and getattr(manifest, "state_contract", None) is not None:
        manifest_layout = getattr(manifest.state_contract, "layout", None)
        if manifest_layout in _RLDX_NON_DEFAULT_LAYOUTS:
            return str(manifest_layout)
    return "libero"


def _require_scene_cameras(
    env_cfg: Any, *, layout: str, camera_keys: tuple[str, ...], family: str
) -> None:
    """Reject early when a scene declares too few cameras for ``layout``.

    GR00T / RLDX take a fixed number of *distinct* camera streams (LIBERO=2
    agentview+wrist, RC365=3, GR1/Simpler=1) and read them positionally with no
    single-view fallback. On a scene that renders too few, the missing stream
    only surfaces as an opaque ``observation.images[...]`` error mid-rollout —
    after the (multi-minute) sidecar boot. Surfacing it here turns that into a
    clear, upfront :class:`ROSCapabilityMismatch`.

    A scene that declares **no** ``cameras`` is the adapter-default case (LIBERO
    renders ``camera1``+``camera2`` itself), so only an *explicit, too-short*
    camera list is treated as a real mismatch — never a false-reject of the
    LIBERO sim scenes that omit the field.

    Raises:
        openral_core.exceptions.ROSCapabilityMismatch: ``scene.cameras`` is
            declared with fewer distinct entries than ``layout`` requires.
    """
    required = _RLDX_LAYOUT_CAMERA_COUNT.get(layout, _RLDX_CAMERA_PAIR_LEN)
    if required <= 1:
        return
    scene_cameras = list(getattr(env_cfg.scene, "cameras", []) or [])
    if not scene_cameras:
        return
    distinct = len(set(scene_cameras))
    if distinct >= required:
        return
    raise ROSCapabilityMismatch(
        f"{family} checkpoint (state_layout={layout!r}) consumes {required} distinct "
        f"camera views {list(camera_keys[:required])}, but scene "
        f"{env_cfg.scene.id!r} declares only {distinct}: {scene_cameras}. Render the "
        f"missing view(s) (extend the scene's `cameras:`) or run a single-view "
        f"checkpoint here — this family has no single-camera fallback."
    )


def _derive_sidecar_port(
    *, family: str, model: str, embodiment_tag: str, quantization: str, layout: str
) -> int:
    """Deterministically map a policy identity to a default sidecar port.

    Non-cryptographic — SHA-1 is used only to spread identities evenly across
    the port range; it never guards a security boundary.
    """
    key = "|".join((family, model, embodiment_tag, quantization, layout))
    # SHA-1 here is non-cryptographic — only used to spread identities evenly
    # across the port range, never as a security boundary.
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    span = _SIDECAR_PORT_MAX - _SIDECAR_PORT_MIN
    return _SIDECAR_PORT_MIN + (int.from_bytes(digest[:4], "big") % span)


def _resolve_sidecar_port(
    *,
    port_env: str | None,
    extra_port: Any,
    family: str,
    model: str,
    embodiment_tag: str,
    quantization: str,
    layout: str,
) -> int:
    """Resolve the sidecar port: env pin > vla.extra pin > per-identity default.

    The per-identity default (:func:`_derive_sidecar_port`) replaces the old
    hard 5555 so different checkpoints don't collide on one port and silently
    reuse each other's sidecar.
    """
    if port_env is not None:
        return int(port_env)
    if extra_port is not None:
        return int(extra_port)
    return _derive_sidecar_port(
        family=family,
        model=model,
        embodiment_tag=embodiment_tag,
        quantization=quantization,
        layout=layout,
    )


# Numerical tolerances for the hand-rolled bridge-orientation Euler
# helpers — exposed as module constants so the lint magic-value rule
# doesn't fire on the comparisons inside the math.
_QUAT_NORM_EPS = 1e-12  # below this we fall back to the identity matrix.
_EULER_GIMBAL_EPS = 1e-6  # below this we drop into the gimbal-lock branch.
# WidowX RLDS gripper threshold — 0.5 cuts cleanly between "fully open" and
# "fully closed" RLDS values. Matches the upstream
# ``WidowXBridgeEnv._postprocess_gripper`` constant.
_RLDS_GRIPPER_THRESHOLD = 0.5
# WidowX gripper qpos range vs bridge_data_v2 stats. The MS3 widowx
# bridge env configures the gripper finger joints with
# ``lower=0.015, upper=0.037`` (see
# ``mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/base_env.py``
# `gripper_pd_joint_pos`). Bridge_data_v2's published statistics for
# ``state.gripper_position`` give range [0.046, 1.115] and mean
# 0.708, i.e. the policy was trained on a gripper state expressed in
# a different unit than MS3's raw joint position. Without rescaling
# the policy sees a "fully open" reset state value (raw 0.037) that's
# ~20× smaller than its training "fully open" (~1.0), reads it as
# strongly out-of-distribution, and emits ~zero arm deltas (the
# arm-frozen-but-gripper-active failure mode observed at run time).
_BRIDGE_GRIPPER_RAW_LOW = 0.015
_BRIDGE_GRIPPER_RAW_HIGH = 0.037
_BRIDGE_GRIPPER_NORM_LOW = 0.046
_BRIDGE_GRIPPER_NORM_HIGH = 1.115


# WidowX sticky-gripper params. Mirrors the upstream
# ``rldx/eval/sim/SimplerEnv/simpler_env.py`` Google fractal handler
# (``_postprocess_gripper`` + ``sticky_gripper_num_repeat = 15``).
# The WidowX upstream env wrapper is "stateless" — it just binarizes
# ``2*(close>0.5)-1`` — but that path was evaluated at bf16 precision
# where the policy's gripper_close outputs were crisp (~0 or ~1). On
# our nf4-quantized path the outputs hover around the threshold and
# binarization oscillates, which prevents the gripper from holding
# closed long enough to grasp. Adopting the Google-style sticky
# state machine recovers a working grasp without re-quantizing.
_STICKY_GRIPPER_NUM_REPEAT = 15
# Confidence band: a transition only fires when the policy is
# strongly biased (>0.75 close or <0.25 close). Values inside
# [0.25, 0.75] mean "no opinion" and we keep the current state.
_STICKY_GRIPPER_CLOSE_CONF = 0.75
_STICKY_GRIPPER_OPEN_CONF = 0.25


def _normalize_bridge_gripper_state(raw_qpos: float) -> float:
    """Map MS3 widowx finger qpos to bridge_data_v2's gripper_position units.

    Linear interpolation between the MS3 joint range
    ``[lower, upper]`` and the bridge_data_v2 statistics range
    ``[q01_min, max]``. Anchors at both endpoints so:

      * ``raw_qpos = 0.015`` (MS3 fully closed) → ``0.046`` (bridge min)
      * ``raw_qpos = 0.037`` (MS3 fully open)   → ``1.115`` (bridge max)
      * mid-stroke values land near the bridge mean ``0.708``.

    Values slightly outside MS3's nominal range (the
    ``extra_gripper_clearance`` cushion of ±0.001) extrapolate
    linearly — the policy's training tails extend past the nominal
    bounds anyway (q01/q99 wider than min/max would suggest).
    """
    span_raw = _BRIDGE_GRIPPER_RAW_HIGH - _BRIDGE_GRIPPER_RAW_LOW
    span_norm = _BRIDGE_GRIPPER_NORM_HIGH - _BRIDGE_GRIPPER_NORM_LOW
    return _BRIDGE_GRIPPER_NORM_LOW + (raw_qpos - _BRIDGE_GRIPPER_RAW_LOW) / span_raw * span_norm


def _encode_ndarray(obj: Any) -> Any:
    """Msgpack ``default`` hook: serialize ndarrays via ``np.save``.

    Mirrors ``MsgSerializer.encode_ndarray`` in ``rldx/policy/server_client.py``.
    Wrapping in a sentinel dict keeps msgpack-only on the wire (no
    msgpack-numpy dependency).
    """
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode_ndarray(obj: dict[str, Any]) -> Any:
    """Msgpack ``object_hook``: reverse :func:`_encode_ndarray`."""
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


@dataclass
class _Gr00tFamilySidecarAdapter:
    """ZMQ-sidecar-backed adapter for the GR00T policy family.

    Drives both ``rldx`` (RLWRLD RLDX-1, a GR00T-N1.5 finetune) and ``gr00t``
    (NVIDIA Isaac GR00T) — they share the upstream ``PolicyServer`` wire, so
    the ``family`` field selects the boot helper + env namespace (ADR-0046).
    Lives in this module for history (RLDX-1 landed first); ``_RLDXSidecarAdapter``
    is a back-compat alias defined below.

    Replays a 16-action chunk on the openral side so the env runner
    sees a single per-step action; refills query the sidecar only when
    the queue is empty.
    """

    spec: VLASpec
    host: str
    port: int
    replan_steps: int
    image_size: int
    timeout_ms: int
    flip_180: bool = False
    flip_vertical: bool = False
    # Manifest-driven dispatch: "libero" sends 8-D LIBERO-flat state +
    # two 256x256 RGB streams; "gr1" sends 39-D state split across the
    # humanoid_everyday_g1 keys + one ego_view stream.
    state_layout: str = "libero"
    # Auto-managed lifecycle knobs (see module docstring §"Auto-managed
    # lifecycle"). The adapter forks tools/rldx_sidecar.py when the
    # initial ping fails and `auto_spawn` is on; first boot on a fresh
    # host triggers the upstream `git clone` + `uv sync` which can run
    # for several minutes, so `boot_timeout_s` defaults large.
    auto_spawn: bool = True
    boot_timeout_s: float = 900.0
    quantization: str = "nf4"
    embodiment_tag: str = "GENERAL_EMBODIMENT"
    # When None we derive the HF id from spec.weights_uri / the rSkill
    # manifest's weights_uri. Explicit override available via
    # vla.extra.model_id for offline / mirrored checkpoints.
    model_id: str | None = None
    # Sidecar family — drives the boot-helper script name
    # (``tools/<family>_sidecar.py``) and the env-var namespace
    # (``OPENRAL_<FAMILY>_*``) used in spawn/locate/error paths. Defaults
    # to ``"rldx"`` so existing RLDX behavior is unchanged; the GR00T
    # adapter (ADR-0046) reuses this class with ``family="gr00t"`` because
    # RLDX-1 is itself a GR00T-N1.5 finetune sharing the wire contract.
    family: str = "rldx"
    # LIBERO video-history offsets (frames sampled relative to the current
    # step). RLDX-1's LIBERO config wants 4 frames spaced 2 steps apart;
    # GR00T's ``LIBERO_PANDA`` wants a single current frame (horizon 1).
    video_offsets: tuple[int, ...] = _RLDX_VIDEO_OFFSETS
    _camera_keys: tuple[str, str] = ("camera1", "camera2")
    _ctx: Any = None
    _socket: Any = None
    # PID we forked; None if we connected to a pre-existing sidecar.
    # close() only tears down what we spawned ourselves.
    _child: subprocess.Popen[bytes] | None = None
    _chunk: collections.deque[NDArray[np.float32]] = field(default_factory=collections.deque)
    _last_input_frame: NDArray[np.uint8] | None = None
    # Per-camera rolling frame history (len=_RLDX_VIDEO_HISTORY). On reset
    # we clear; on the first frame after reset we pad by repeating the
    # current frame so the model sees a static 4-frame stack instead of
    # garbage from a previous episode. Per ADR-0069 the canonical scene
    # camera names (e.g. ``front`` / ``wrist``) live on ``_camera_keys``;
    # the keys below are INTERNAL buffer aliases that the obs assembler
    # uses to refer to "first camera" / "second camera" / "third camera"
    # positionally — they are deliberately decoupled from the scene-side
    # names so the buffer wiring does not need to be re-aliased per robot.
    _frame_buffers: dict[str, collections.deque[NDArray[np.uint8]]] = field(
        default_factory=lambda: {
            "camera1": collections.deque(maxlen=_RLDX_VIDEO_HISTORY),
            "camera2": collections.deque(maxlen=_RLDX_VIDEO_HISTORY),
            "camera3": collections.deque(maxlen=_RLDX_VIDEO_HISTORY),
        }
    )
    # Sticky-gripper state machine — see `_apply_widowx_sticky_gripper`.
    # Initial target = +1.0 (MS3 "open") so the first applied command
    # of an episode behaves as if the gripper just opened, which
    # matches the reset state (raw qpos 0.037 = fully open). Mirrors
    # the upstream Google fractal env's `_postprocess_gripper`
    # initial conditions.
    _sticky_gripper_target: float = 1.0
    _sticky_gripper_lock: int = 0

    def __post_init__(self) -> None:
        # pyzmq + msgpack can disappear out from under us if the user
        # ran `uv sync --group <other>` between rldx invocations and
        # that group's resolved set did not list them (the robocasa
        # group is the canonical case — surfaced by the rSkill audit
        # GPU smoke tests). Treat them as a backend dep so the auto-
        # installer reproduces them on demand instead of leaving the
        # user with a `uv add` hint.
        from openral_sim._deps import ensure_backend_deps

        ensure_backend_deps("rldx_client")
        try:
            import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # reason: opt-in rldx group
        except ImportError as exc:  # pragma: no cover - opt-in
            raise ROSConfigError(
                "rldx adapter requires pyzmq + msgpack: install with "
                "`uv sync --all-packages --group rldx --inexact`"
            ) from exc
        self._ctx = zmq.Context.instance()
        self._init_socket()
        endpoint = f"tcp://{self.host}:{self.port}"
        _log.info("rldx_sidecar_connecting", endpoint=endpoint)
        # Try the existing-server path first. If anyone is already
        # serving on the port we reuse them — operator manual boots and
        # cross-process sharing keep working.
        if self._try_ping():
            self._verify_existing_identity(endpoint)
            _log.info("rldx_sidecar_connected", endpoint=endpoint, mode="existing")
            return
        if not self.auto_spawn:
            raise ROSConfigError(
                f"{self.family} sidecar at {endpoint} did not answer ping within "
                f"{self.timeout_ms} ms and auto_spawn is disabled. Boot "
                f"the sidecar manually with `python tools/{self.family}_sidecar.py "
                f"--model {self.model_id or '<HF id>'} --port {self.port} "
                f"--quantization {self.quantization} --embodiment-tag "
                f"{self.embodiment_tag}` or set "
                f"OPENRAL_{self.family.upper()}_AUTO_SPAWN=1."
            )
        # No server up — pre-stage the sidecar source + venv (git clone
        # + uv sync + bitsandbytes ≈ 80 s on a cold cache, ~0 s warm)
        # behind one ensure_backend_deps prompt so it doesn't happen
        # mid-rollout. The launcher's own ``_ensure_source`` /
        # ``_install_deps`` short-circuit when the cache is warm, so
        # this is additive rather than duplicative. Only the rldx sidecar
        # registers a pre-stage step; the gr00t sidecar (ADR-0046) clones
        # NVIDIA/Isaac-GR00T itself at boot.
        if self.family == "rldx":
            ensure_backend_deps("rldx_sidecar_setup")
        # No server up — fork the boot helper and wait for it to bind.
        self._spawn_sidecar()
        if not self._wait_for_boot():
            # Capture the child's exit status BEFORE _terminate_child() nulls
            # it: a crashed child (poll() != None) is not a slow bootstrap, so
            # the "did not answer ping within {timeout}s / slow path" text would
            # send the operator down the wrong road. Mirrors
            # SidecarClient._boot_failure_error (sidecar.py).
            returncode = self._child.poll() if self._child is not None else None
            self._terminate_child()
            raise self._boot_failure_error(endpoint, returncode)
        _log.info("rldx_sidecar_connected", endpoint=endpoint, mode="auto-spawned")

    # ─── Public PolicyAdapter contract ───────────────────────────────

    def last_input_frame(self) -> NDArray[np.uint8] | None:
        return self._last_input_frame

    def reset(self) -> None:
        """Clear the queued chunk + frame history and notify the sidecar."""
        self._chunk.clear()
        for buf in self._frame_buffers.values():
            buf.clear()
        # Reset the WidowX sticky-gripper state machine to the
        # episode's initial pose (gripper open = +1 in MS3 convention).
        self._sticky_gripper_target = 1.0
        self._sticky_gripper_lock = 0
        try:
            self._call("reset", {"options": None})
        except Exception as exc:  # pragma: no cover - server-dependent
            # A reset failure should not kill the whole rollout, but it's
            # diagnostic gold: server-side memory not flushed between
            # episodes will quietly degrade success rates.
            _log.warning("rldx_sidecar_reset_failed", error=str(exc))

    def step(self, observation: Observation, instruction: str) -> NDArray[np.float32]:
        if not self._chunk:
            self._refill_chunk(observation, instruction)
        action = np.asarray(self._chunk.popleft(), dtype=np.float32)
        # For simpler_widowx the chunk's gripper column carries the raw
        # policy `gripper_close` in [0, 1]; we apply the sticky state
        # machine + binarisation here (across applied steps) rather
        # than in `_assemble_simpler_chunk` (which only sees one chunk
        # at a time and would lose state across replan boundaries).
        if self.state_layout == "simpler_widowx":
            action[6] = self._apply_widowx_sticky_gripper(float(action[6]))
        return action

    def _apply_widowx_sticky_gripper(self, raw_close: float) -> float:
        """Sticky-gripper state machine for the WidowX bridge eval path.

        Inputs the policy's raw ``action.gripper_close`` value in
        [0, 1] (1 = close, 0 = open). Returns the MS3 widowx
        controller's action[6] in [-1, +1] (-1 = close, +1 = open).

        Behavior (mirrors Google fractal's ``_postprocess_gripper``):

        * If we are currently locked (``_sticky_gripper_lock > 0``),
          return the locked target and decrement the counter — the
          gripper holds its state until the timer expires regardless
          of policy chatter near the binarisation threshold.
        * Otherwise, only transition when the policy is *confident*:
          ``raw_close > 0.75`` (close) or ``raw_close < 0.25``
          (open) triggers a transition AND locks for
          ``_STICKY_GRIPPER_NUM_REPEAT`` (15) steps.
        * Values inside [0.25, 0.75] mean "no opinion" and we hold
          the current target.
        """
        if self._sticky_gripper_lock > 0:
            self._sticky_gripper_lock -= 1
            return self._sticky_gripper_target
        if raw_close > _STICKY_GRIPPER_CLOSE_CONF:
            self._sticky_gripper_target = -1.0  # MS3 widowx: -1 = close
            self._sticky_gripper_lock = _STICKY_GRIPPER_NUM_REPEAT
        elif raw_close < _STICKY_GRIPPER_OPEN_CONF:
            self._sticky_gripper_target = 1.0  # MS3 widowx: +1 = open
            self._sticky_gripper_lock = _STICKY_GRIPPER_NUM_REPEAT
        return self._sticky_gripper_target

    def close(self) -> None:
        # Idempotent. Tear down the ZMQ socket first so the child's exit
        # cannot race a pending send/recv. Then signal the spawned child
        # (no-op when we didn't fork one).
        if self._socket is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - teardown
                self._socket.close()
            self._socket = None
        self._terminate_child()

    # ─── Internal helpers ────────────────────────────────────────────

    def _init_socket(self) -> None:
        """Create (or recreate) the REQ socket and connect to the sidecar.

        Idempotent: closes any existing socket first. We need to be able to
        recreate the socket because a ZMQ REQ socket whose ``recv()`` timed
        out (RCVTIMEO) is left in ``EFSM`` -- the strict REQ state machine
        requires send→recv pairs, and a half-finished send blocks every
        subsequent ``send()`` with ``Operation cannot be accomplished in
        current state`` until the socket is reopened. ``_try_ping`` calls
        this on failure so ``_wait_for_boot`` doesn't loop forever against
        a permanently-dead socket once the first ping times out.
        """
        import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # reason: opt-in rldx group

        if self._socket is not None:
            with contextlib.suppress(Exception):
                self._socket.close(linger=0)
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def _try_ping(self) -> bool:
        """One ZMQ ping with the configured RCVTIMEO. True iff server answers.

        ZMQ REQ is lazy on tcp:// — ``send()`` queues locally and
        ``recv()`` blocks for the full ``RCVTIMEO`` (default 60 s) when
        nothing is on the other end, instead of returning the raw
        ECONNREFUSED a TCP connect would surface in microseconds. To
        keep cold-start connects fast (no sidecar yet → spawn one), we
        gate the ZMQ leg behind the cheap :meth:`_is_port_busy` TCP
        probe. Side benefit: :meth:`_wait_for_boot` now polls at the
        ~2 s sleep rate instead of being throttled to the 60 s RCVTIMEO,
        which also shortens child-death detection during boot.

        On ZMQ-leg failure we recreate the REQ socket -- see
        :meth:`_init_socket` for why the EFSM state of REQ makes
        per-attempt resets mandatory.
        """
        if not self._is_port_busy():
            return False
        try:
            self._call("ping", {})
        except Exception:
            self._init_socket()
            return False
        return True

    def _verify_existing_identity(self, endpoint: str) -> None:
        """Refuse to reuse a pre-existing sidecar serving a different checkpoint.

        The upstream RLDX / GR00T ZMQ servers answer ``ping`` regardless of
        which model they loaded, so a stale sidecar (e.g. an earlier RLDX
        LIBERO run still holding the port) would otherwise be silently adopted
        for a RoboCasa / SimplerEnv / GR00T eval and serve the wrong policy —
        the "always loads RLDX's environment" failure. We cross-check the
        sidecar's recorded identity (written by ``run_sidecar``) and fail
        closed on a mismatch rather than hiding it (CLAUDE.md §1.4).

        A missing record (sidecar booted before this control, or by some
        other path) is unverifiable, not a mismatch: we log and proceed,
        preserving the operator-managed-sidecar workflow.

        Raises:
            ROSConfigError: when the recorded identity disagrees with what
                this adapter is configured to serve.
        """
        from openral_sim._sidecar_common import read_sidecar_identity

        recorded = read_sidecar_identity(self.port)
        if recorded is None:
            _log.warning(
                "rldx_sidecar_identity_unverified",
                endpoint=endpoint,
                port=self.port,
                reason="no identity record on disk; trusting pre-existing sidecar",
            )
            return
        want = {
            "family": self.family,
            "model": self._resolve_model_id(),
            "embodiment_tag": self.embodiment_tag,
            "quantization": self.quantization,
        }
        mismatched = {k: (recorded.get(k), v) for k, v in want.items() if recorded.get(k) != v}
        if mismatched:
            detail = ", ".join(
                f"{k}: serving {found!r}, want {wanted!r}"
                for k, (found, wanted) in mismatched.items()
            )
            raise ROSConfigError(
                f"A different {recorded.get('family', '?')} sidecar (pid "
                f"{recorded.get('pid', '?')}) is already serving on {endpoint}: "
                f"{detail}. Reusing it would run the wrong policy for this "
                "environment. Stop that sidecar, or point this run at a free "
                f"port via OPENRAL_{self.family.upper()}_PORT / vla.extra.port. "
                "By default each checkpoint derives its own port, so this "
                "usually means two runs pinned the same port explicitly."
            )
        _log.info(
            "rldx_sidecar_identity_verified",
            endpoint=endpoint,
            family=self.family,
            model=recorded.get("model"),
        )

    def _spawn_sidecar(self) -> None:
        """Fork ``tools/rldx_sidecar.py`` for the rSkill's checkpoint.

        We resolve everything we need from the (already-validated)
        rSkill manifest + the adapter's own fields:
          * model HF id     — manifest.weights_uri (``hf://``) or
                              ``vla.extra.model_id``,
          * port            — ``self.port``,
          * quantization    — ``self.quantization``,
          * embodiment tag  — ``self.embodiment_tag``.

        The child is launched in its own session/process group so a
        Ctrl-C against the openral process (e.g. ``openral sim run``) does
        not also abort the multi-minute first-boot ``uv sync``. We rely
        on :meth:`close` / atexit to clean up.
        """
        if self._is_port_busy():
            # Race: somebody else just bound the port between our ping
            # failure and this call. Skip the spawn and let the next
            # ping attempt reuse them.
            _log.info("rldx_sidecar_spawn_skipped_port_busy", port=self.port)
            return

        script = self._locate_sidecar_script()
        model_id = self._resolve_model_id()
        cmd = [
            sys.executable,
            str(script),
            "--model",
            model_id,
            "--port",
            str(self.port),
            "--quantization",
            self.quantization,
            "--embodiment-tag",
            self.embodiment_tag,
        ]
        _log.info(
            "rldx_sidecar_spawning",
            model=model_id,
            port=self.port,
            quantization=self.quantization,
            embodiment_tag=self.embodiment_tag,
            script=str(script),
        )
        # start_new_session=True puts the child in its own pgid so SIGINT
        # to openral doesn't propagate down to the in-flight bootstrap.
        # We forward stdout/stderr to the openral process's streams so
        # the user can watch the boot progress without extra plumbing.
        self._child = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _wait_for_boot(self) -> bool:
        """Poll ``ping`` until success, child death, or timeout."""
        deadline = time.monotonic() + self.boot_timeout_s
        # First ping after spawn: be patient — the server may not have
        # bound the socket yet. We sleep before retrying so we don't
        # spin the GIL while the sidecar is doing heavy weight loads.
        while time.monotonic() < deadline:
            if self._child is not None and self._child.poll() is not None:
                _log.error(
                    "rldx_sidecar_died_during_boot",
                    returncode=self._child.returncode,
                )
                return False
            if self._try_ping():
                return True
            time.sleep(2.0)
        return False

    def _boot_failure_error(self, endpoint: str, returncode: int | None) -> ROSConfigError:
        """Classify a boot failure off the child's exit status.

        ``returncode`` is None when the child is still running (a genuine
        timeout — slow first-boot ``git clone`` + ``uv sync``) and a non-zero
        int when it crashed during boot. A crash is NOT a slow bootstrap, so
        reporting "did not answer ping within {timeout}s" is misleading —
        surface the exit code and the common causes instead. The signature
        cause on an 8 GB host is CUDA OOM when the {family} weights load
        bf16-resident alongside another GPU sidecar (e.g. the Isaac renderer):
        the rldx upstream loader does not NF4-quantise, so a 3B checkpoint
        needs ~6 GiB and will not co-fit. ``returncode=-9`` is the OOM-killer
        (SIGKILL); a positive code is usually a Python-level fault (a pretrain
        base such as RLDX-1-PT whose processor has no modality config for the
        requested embodiment KeyErrors at boot).
        """
        if returncode is not None and returncode != 0:
            return ROSConfigError(
                f"{self.family} sidecar process exited with code {returncode} during boot "
                f"on {endpoint} — it crashed, it did not time out. Inspect the sidecar "
                "stdout above. Common causes: CUDA OOM (the bf16-resident weights do not "
                "co-fit with another GPU sidecar such as the Isaac renderer on an 8 GB host "
                "— rc=-9 is the OOM-killer), missing/incompatible weights, or a pretrain "
                "base (e.g. RLDX-1-PT) whose processor lacks a modality config for the "
                "requested embodiment. Use a task finetune and/or a larger GPU."
            )
        return ROSConfigError(
            f"{self.family} sidecar spawned but did not answer ping within "
            f"{self.boot_timeout_s:.0f} s on {endpoint}. The first boot "
            f"on a fresh host runs `git clone` + `uv sync` of the "
            f"{self.family} sidecar source (slow path); raise "
            "vla.extra.boot_timeout_s if the bootstrap is still in "
            "progress, or inspect the sidecar's stdout for the actual "
            "failure mode."
        )

    def _terminate_child(self) -> None:
        """Best-effort SIGTERM → SIGKILL of any child we own."""
        child = self._child
        if child is None or child.poll() is not None:
            self._child = None
            return
        _log.info("rldx_sidecar_terminating", pid=child.pid)
        with contextlib.suppress(Exception):  # pragma: no cover - teardown
            child.terminate()
            try:
                child.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5.0)
        self._child = None

    def _is_port_busy(self) -> bool:
        """Cheap TCP probe — returns True if something is listening on the port.

        Avoids spawning a duplicate sidecar in racy multi-rSkill setups.
        """
        with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            s.connect((self.host, self.port))
            return True
        return False

    def _locate_sidecar_script(self) -> Path:
        """Find ``tools/<family>_sidecar.py`` relative to the repo root.

        Resolution order:
          1. ``OPENRAL_<FAMILY>_SIDECAR_SCRIPT`` env var (absolute path).
          2. Walking upwards from this file until we hit ``tools/``.

        Raises ``ROSConfigError`` when neither resolves to an existing file.
        """
        script_name = f"{self.family}_sidecar.py"
        script_env = f"OPENRAL_{self.family.upper()}_SIDECAR_SCRIPT"
        env_override = os.environ.get(script_env)
        if env_override:
            p = Path(env_override).expanduser().resolve()
            if not p.is_file():
                raise ROSConfigError(f"{script_env}={env_override!r} is not a file.")
            return p
        # __file__ lives at python/sim/src/openral_sim/policies/rldx.py.
        # Walk up looking for a sibling tools/<family>_sidecar.py — robust
        # against the user installing the package out-of-tree.
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "tools" / script_name
            if candidate.is_file():
                return candidate
        raise ROSConfigError(
            f"Could not locate tools/{script_name} upwards from "
            f"{here}. Set {script_env} to the absolute "
            "path of the boot helper, or disable auto-spawn via "
            f"OPENRAL_{self.family.upper()}_AUTO_SPAWN=0 and run it by hand."
        )

    def _resolve_model_id(self) -> str:
        """Pick the HF id to hand the sidecar.

        Precedence:
          1. ``self.model_id`` (from ``vla.extra.model_id`` — supports
             local-path overrides),
          2. ``rSkill manifest.weights_uri`` when it starts with
             ``hf://`` (the canonical path — every rldx1-* manifest
             carries one),
          3. ``self.spec.weights_uri`` when the user supplied a bare
             ``hf://`` directly.

        Raises ``ROSConfigError`` when no HF id is available — the
        sidecar requires one to know what checkpoint to load.
        """
        if self.model_id:
            return self.model_id

        manifest = load_manifest_for_spec(self.spec)
        candidate = getattr(manifest, "weights_uri", None) if manifest else None
        candidate = candidate or getattr(self.spec, "weights_uri", "")
        candidate = str(candidate or "")
        if candidate.startswith("hf://"):
            return candidate[len("hf://") :]
        raise ROSConfigError(
            "rldx adapter cannot resolve a HuggingFace model id. Set "
            "the rSkill manifest's weights_uri to `hf://RLWRLD/...` or "
            "pass `vla.extra.model_id` explicitly."
        )

    def _call(self, endpoint: str, data: dict[str, Any]) -> Any:
        """Issue one REQ/REP round trip; raise on transport / server error."""
        import msgpack  # type: ignore[import-not-found,import-untyped,unused-ignore]  # reason: opt-in rldx group

        msg = {"endpoint": endpoint, "data": data, "api_token": None}
        self._socket.send(msgpack.packb(msg, default=_encode_ndarray, use_bin_type=True))
        raw = self._socket.recv()
        return msgpack.unpackb(raw, object_hook=_decode_ndarray, raw=False)

    def _refill_chunk(self, observation: Observation, instruction: str) -> None:
        """Build the wire obs (LIBERO / GR1 / RC365 / SimplerEnv), ship, queue the chunk."""
        if self.state_layout == "gr1":
            obs = self._build_gr1_obs(observation, instruction)
        elif self.state_layout == "rc365":
            obs = self._build_rc365_obs(observation, instruction)
        elif self.state_layout == "simpler_widowx":
            obs = self._build_simpler_widowx_obs(observation, instruction)
        elif self.state_layout == "simpler_google":
            obs = self._build_simpler_google_obs(observation, instruction)
        else:
            obs = self._build_libero_obs(observation, instruction)

        t0 = time.monotonic()
        with inference_span(kind="chunk"):
            response = self._call("get_action", {"observation": obs, "options": None})
        elapsed = time.monotonic() - t0

        # The upstream PolicyServer answers `get_action` with either:
        #   - a dict {"error": "..."} when validation / inference fails,
        #   - a flat action dict {"action.x": ..., "action.y": ..., ...}
        #     keyed as the LIBERO 7-D layout when --use-sim-policy-wrapper
        #     is in effect (this is the path we drive), or
        #   - a [action_dict, info_dict] list on builds with debug info.
        # Accept any of the three shapes; raise loud if the server reports
        # an error so we don't silently swallow a misconfigured embodiment.
        if isinstance(response, dict) and "error" in response and len(response) == 1:
            raise ROSConfigError(
                f"rldx sidecar refused observation: {response['error']!r}. "
                "Most likely cause: the server was booted with the wrong "
                "--embodiment-tag for the scene — for LIBERO use "
                "`--embodiment-tag libero_panda`."
            )
        if isinstance(response, list) and len(response) == _RLDX_RESPONSE_ENVELOPE_LEN:
            action_dict, _info = response
        elif isinstance(response, dict):
            action_dict = response
        else:
            raise ROSConfigError(
                "rldx sidecar returned a malformed response (expected an "
                "action dict, an error dict, or [action_dict, info_dict]; "
                f"got {type(response).__name__})."
            )
        if not isinstance(action_dict, dict):
            raise ROSConfigError(
                "rldx sidecar returned a non-dict action payload "
                f"(got {type(action_dict).__name__})."
            )

        chunk = self._assemble_action_chunk(action_dict)
        if chunk.shape[0] < self.replan_steps:
            raise ROSConfigError(
                f"rldx sidecar returned only {chunk.shape[0]} actions but "
                f"replan_steps={self.replan_steps}; lower replan_steps or "
                "rebuild the checkpoint with a longer action_horizon."
            )
        for a in chunk[: self.replan_steps]:
            self._chunk.append(a)
        _log.info(
            "rldx_sidecar_chunk",
            host=self.host,
            elapsed_s=round(elapsed, 2),
            n_queued=len(self._chunk),
            action_dim=int(chunk.shape[-1]),
        )

    def _build_libero_obs(self, observation: Observation, instruction: str) -> dict[str, Any]:
        """Wire obs in LIBERO-flat layout (two cameras + per-axis state scalars)."""
        agentview, wrist = self._pick_images(observation)
        self._last_input_frame = wrist[0, 0]

        state_vec = self._pick_state(observation, _LIBERO_STATE_DIM)
        x, y, z = state_vec[0:3]
        roll, pitch, yaw = state_vec[3:6]
        gripper = state_vec[6:8].astype(np.float32)

        def _bt1(val: float) -> NDArray[np.float32]:
            return np.asarray([[[val]]], dtype=np.float32)

        return {
            _RLDX_AGENTVIEW_KEY: agentview,
            _RLDX_WRIST_KEY: wrist,
            "state.x": _bt1(float(x)),
            "state.y": _bt1(float(y)),
            "state.z": _bt1(float(z)),
            "state.roll": _bt1(float(roll)),
            "state.pitch": _bt1(float(pitch)),
            "state.yaw": _bt1(float(yaw)),
            "state.gripper": gripper.reshape(1, 1, 2),
            _RLDX_TASK_KEY: (instruction or observation.get("task", "") or "",),
        }

    def _build_gr1_obs(self, observation: Observation, instruction: str) -> dict[str, Any]:
        """Wire obs in Fourier GR-1 native layout: one ego_view + five state buckets.

        Matches the FT-GR1 checkpoint's ``general_embodiment`` modality
        config (5 state keys: waist + per-arm + per-hand; 1 video key
        ``ego_view``; language key ``annotation.human.coarse_action``).
        """
        ego = self._pick_single_camera(observation, self._camera_keys[0], buf_key="camera1")
        self._last_input_frame = ego[0, 0]

        state_vec = self._pick_state(observation, expected_dim=29)
        obs: dict[str, Any] = {_GR1_VIDEO_KEY: ego}
        for key, (lo, hi) in _GR1_STATE_SLICES.items():
            obs[key] = state_vec[lo:hi].astype(np.float32).reshape(1, 1, hi - lo)
        obs[_GR1_TASK_KEY] = (instruction or observation.get("task", "") or "",)
        return obs

    def _build_rc365_obs(self, observation: Observation, instruction: str) -> dict[str, Any]:
        """Wire obs in RoboCasa-365 PandaMobile layout: 3 cameras + 5 state buckets.

        Matches FT-RC365's ``general_embodiment`` modality config:
        * three RGB streams (agentview_left / right + eye_in_hand) at T=4
        * five state keys re-sliced from openral's 16-D human300 layout
        * language key ``annotation.human.task_description``
        """
        # Pick the scene-side camera keys (default camera1/2/3); fall back
        # cleanly when fewer than 3 are present.
        scene_cams = [
            self._camera_keys[0] if len(self._camera_keys) > 0 else "camera1",
            self._camera_keys[1] if len(self._camera_keys) > 1 else "camera2",
            "camera3",
        ]
        videos = [
            self._pick_single_camera(observation, scene_cams[i], buf_key=f"camera{i + 1}")
            for i in range(3)
        ]
        # Top-down ("wrist" equivalent) is the eye_in_hand stream — index 2.
        self._last_input_frame = videos[2][0, 0]

        state_vec = self._pick_state(observation, expected_dim=16)
        obs: dict[str, Any] = {_RC365_VIDEO_KEYS[i]: videos[i] for i in range(3)}
        for key, (lo, hi) in _RC365_STATE_SLICES_FROM_HUMAN300.items():
            obs[key] = state_vec[lo:hi].astype(np.float32).reshape(1, 1, hi - lo)
        obs[_RC365_TASK_KEY] = (instruction or observation.get("task", "") or "",)
        return obs

    def _build_simpler_widowx_obs(
        self, observation: Observation, instruction: str
    ) -> dict[str, Any]:
        """Wire obs in SimplerEnv WidowX (bridge_orig) layout.

        Matches the canonical modality config registered against
        ``EmbodimentTag.OXE_BRIDGE_ORIG`` in
        ``rldx/configs/data/simpler_widowx_config.py``:

        * ``state.end_effector_position`` — (B=1, T=1, 3) xyz
        * ``state.end_effector_rotation`` — (B=1, T=1, 3) bridge-rotated Euler RPY
        * ``state.gripper_position``      — (B=1, T=1, 1) raw last-qpos
        * ``video.image_0``               — (B=1, T=1, H, W, 3) uint8

        The earlier split-scalar layout (``state.x``/``state.y``/…/
        ``state.pad``/``state.gripper``) targeted an obsolete env-wrapper
        schema that the upstream ``WidowXBridgeEnv`` still produces, but
        the published FT-SIMPLER-WIDOWX checkpoint's
        ``processor_config.json`` does not consume — its
        ``modality_configs["bridge_orig"]["state"].modality_keys`` lists
        only the three vector keys above and ``PolicyLoader.load``
        crashes with ``KeyError: 'state.end_effector_position'``
        otherwise.
        """
        proprio = _resolve_simpler_eef_pos(observation)
        image = _resize_to_hw(
            _pick_simpler_image(observation, self._camera_keys[0]),
            _SIMPLER_WIDOWX_IMAGE_HW,
        )
        self._last_input_frame = image

        # Bridge orientation correction: SAPIEN wxyz quat → rotation matrix
        # → multiply by ``default_rot.T`` → Euler RPY. Matches the upstream
        # ``WidowXBridgeEnv._process_observation`` exactly.
        roll, pitch, yaw = _bridge_quat_to_euler(proprio[3:7])
        pos = np.asarray(proprio[:3], dtype=np.float32).reshape(3)
        rot = np.asarray([roll, pitch, yaw], dtype=np.float32).reshape(3)
        gripper_norm = _normalize_bridge_gripper_state(float(proprio[7]))
        gripper = np.asarray([gripper_norm], dtype=np.float32)

        return {
            _SIMPLER_VIDEO_KEY_WIDOWX: image[None, None, ...],
            "state.end_effector_position": pos.reshape(1, 1, 3),
            "state.end_effector_rotation": rot.reshape(1, 1, 3),
            "state.gripper_position": gripper.reshape(1, 1, 1),
            _SIMPLER_LANG_KEY: (instruction or observation.get("task", "") or "",),
        }

    def _build_simpler_google_obs(
        self, observation: Observation, instruction: str
    ) -> dict[str, Any]:
        """Wire obs in SimplerEnv Google Robot (fractal20220817_data) layout.

        Matches the canonical modality config registered against
        ``EmbodimentTag.OXE_FRACTAL`` in
        ``rldx/configs/data/simpler_google_config.py``:

        * ``state.end_effector_position`` — (B=1, T=1, 3) xyz
        * ``state.end_effector_rotation`` — (B=1, T=1, 4) xyzw quat
          (SAPIEN wxyz rolled by -1)
        * ``state.gripper_position``      — (B=1, T=1, 1) closedness
          (1 − raw open)
        * ``video.image``                  — (B=1, T=1, H, W, 3) uint8

        The earlier split-scalar layout (``state.x``/``state.y``/…/
        ``state.rw``/``state.gripper``) does not satisfy the published
        FT-SIMPLER-GOOGLE checkpoint's processor_config — see the
        WidowX docstring for the same rationale.
        """
        proprio = _resolve_simpler_eef_pos(observation)
        image = _resize_to_hw(
            _pick_simpler_image(observation, self._camera_keys[0]),
            _SIMPLER_GOOGLE_IMAGE_HW,
        )
        self._last_input_frame = image

        # SAPIEN reports quat as wxyz; the upstream wrapper rolls -1 to
        # get xyzw.
        quat_wxyz = np.asarray(proprio[3:7], dtype=np.float32)
        quat_xyzw = np.roll(quat_wxyz, -1).astype(np.float32).reshape(4)
        pos = np.asarray(proprio[:3], dtype=np.float32).reshape(3)
        gripper_closedness = np.asarray([1.0 - float(proprio[7])], dtype=np.float32)

        return {
            _SIMPLER_VIDEO_KEY_GOOGLE: image[None, None, ...],
            "state.end_effector_position": pos.reshape(1, 1, 3),
            "state.end_effector_rotation": quat_xyzw.reshape(1, 1, 4),
            "state.gripper_position": gripper_closedness.reshape(1, 1, 1),
            _SIMPLER_LANG_KEY: (instruction or observation.get("task", "") or "",),
        }

    def _pick_single_camera(
        self, observation: Observation, scene_key: str, *, buf_key: str
    ) -> NDArray[np.uint8]:
        """Single-camera variant of :meth:`_pick_images` for GR1 / RC365."""
        images = observation.get("images", {})
        img = images.get(scene_key)
        if img is None:
            raise ROSConfigError(
                f"rldx adapter expects observation.images[{scene_key!r}]; got "
                f"{list(images.keys())}."
            )
        arr = np.asarray(img, dtype=np.uint8)
        if arr.shape[:2] != (self.image_size, self.image_size):
            arr = self._resize(arr, self.image_size)
        if self.flip_180:
            arr = arr[::-1, ::-1].copy()
        if self.flip_vertical:
            # RoboCasa's gymnasium_basic.process_img does
            # ``np.copy(img[::-1, :, :])`` (H-only reverse) before
            # passing frames to the policy. RLDX-1 RC365 / ROBOCASA
            # finetunes were trained against that pipeline, so the
            # openral camera frames need the same vertical flip.
            arr = arr[::-1, :, :].copy()
        buf = self._frame_buffers[buf_key]
        buf.append(arr)
        stack = []
        for off in _RLDX_VIDEO_OFFSETS:
            idx = max(0, len(buf) - 1 + off)
            stack.append(buf[idx])
        return np.stack(stack, axis=0)[None, ...]  # (1, T=4, H, W, 3)

    def _pick_images(self, observation: Observation) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Extract the two RGB streams in (B=1, T=4, H, W, 3) uint8 form.

        The general_embodiment / libero modality config the FT-LIBERO
        checkpoint expects has ``video.delta_indices = [-6, -4, -2, 0]``
        — four frames spaced two simulator steps apart. We maintain a
        per-camera rolling buffer of the last 7 frames and sample at the
        four target offsets. On cold start the buffer is short, so we
        pad by repeating the oldest available frame; that produces a
        static stack on step 0 and a true motion stack once
        ``_RLDX_VIDEO_HISTORY`` steps have accumulated.
        """
        images = observation.get("images", {})
        # Fixed alias from scene key (which the user can override via
        # `camera_keys`) to the rolling-buffer key.
        cam_buf_alias = ("camera1", "camera2")

        def _take(scene_key: str, buf_key: str) -> NDArray[np.uint8]:
            img = images.get(scene_key)
            if img is None:
                raise ROSConfigError(
                    f"rldx adapter expects observation.images[{scene_key!r}]; got "
                    f"{list(images.keys())}. Set vla.extra.camera_keys to "
                    "match your scene."
                )
            arr = np.asarray(img, dtype=np.uint8)
            if arr.shape[:2] != (self.image_size, self.image_size):
                arr = self._resize(arr, self.image_size)
            if self.flip_180:
                # LIBERO's upstream env (rldx/eval/sim/LIBERO/libero_env.py)
                # rotates both agentview and wrist 180° before sending to
                # the policy: `obs[...][::-1, ::-1]`. Mirror that here so
                # the FT-LIBERO checkpoint sees the orientation it was
                # trained on. Driven by rskill manifest
                # `image_preprocessing.flip_180`.
                arr = arr[::-1, ::-1].copy()
            if self.flip_vertical:
                arr = arr[::-1, :, :].copy()
            buf = self._frame_buffers[buf_key]
            buf.append(arr)
            stack = []
            for off in self.video_offsets:
                idx = max(0, len(buf) - 1 + off)
                stack.append(buf[idx])
            return np.stack(stack, axis=0)[None, ...]  # (1, T=4, H, W, 3)

        return (
            _take(self._camera_keys[0], cam_buf_alias[0]),
            _take(self._camera_keys[1], cam_buf_alias[1]),
        )

    @staticmethod
    def _resize(img: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
        """Bilinear resize HWC uint8 to ``size×size`` (PIL backend)."""
        from PIL import Image as _PILImage

        return np.asarray(
            _PILImage.fromarray(img).resize((size, size), _PILImage.Resampling.BILINEAR),
            dtype=np.uint8,
        )

    def _pick_state(
        self, observation: Observation, expected_dim: int = _LIBERO_STATE_DIM
    ) -> NDArray[np.float32]:
        """Pull the proprio state vector and assert the expected width."""
        state = observation.get("state")
        if state is None:
            raise ROSConfigError(
                f"rldx adapter requires a non-empty {expected_dim}-D `state` on "
                "the observation. The scene adapter must populate it."
            )
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.shape[0] != expected_dim:
            raise ROSConfigError(
                f"rldx adapter expects a {expected_dim}-D state for "
                f"state_layout={self.state_layout!r}, got {arr.shape[0]}-D. "
                "Either pick the right rskill (rldx1-ft-libero-nf4 needs 8-D, "
                "rldx1-ft-gr1-nf4 needs 39-D) or update the scene."
            )
        return arr

    def _assemble_action_chunk(self, action_dict: dict[str, Any]) -> NDArray[np.float32]:
        """Layout-aware chunk assembly.

        * LIBERO → stacks ``action.x .. action.gripper`` into ``(T, 7)``.
        * GR1    → concatenates the five Fourier groups into ``(T, 29)``.
        * RC365  → concatenates eef_pos + eef_rot + gripper + base_motion
          + control_mode into ``(T, 12)``; openral_sim.backends.robocasa
          trims to env_dim (11) by dropping the last (control_mode) slot.
        """
        if self.state_layout == "gr1":
            return self._assemble_gr1_chunk(action_dict)
        if self.state_layout == "rc365":
            return self._assemble_rc365_chunk(action_dict)
        if self.state_layout in ("simpler_widowx", "simpler_google"):
            return self._assemble_simpler_chunk(action_dict)
        return self._assemble_libero_chunk(action_dict)

    @staticmethod
    def _normalize_action_column(arr: NDArray[Any], key: str) -> NDArray[np.float32]:
        """Squeeze a server-side ``(1, T, D)`` action column to ``(T, D)``."""
        a: NDArray[np.float32] = np.asarray(arr, dtype=np.float32)
        if a.ndim == _RLDX_ACTION_RANK_BT1 and a.shape[0] == 1:
            return np.asarray(a[0], dtype=np.float32)  # (T, D)
        if a.ndim == _RLDX_ACTION_RANK_T1:
            return a
        raise ROSConfigError(
            f"rldx sidecar action[{key!r}] has unexpected shape {a.shape}; "
            "expected (1, T, D) or (T, D)."
        )

    def _assemble_libero_chunk(self, action_dict: dict[str, Any]) -> NDArray[np.float32]:
        """Stack LIBERO-flat ``action.x .. action.gripper`` into ``(T, 7)``.

        The gripper column is rescaled to the LIBERO/robosuite convention
        — see :func:`_rldx_gripper_to_libero` — because the RLDX policy
        emits gripper in the RLDS dataset convention (``[0, 1]`` with
        ``0=close, 1=open``) while ``LiberoEnv.step`` consumes
        ``[-1, +1]`` with ``-1=open, +1=close`` (GH-133). Without this
        transform every gripper command lands at ~0 (mid-range) and the
        Franka gripper never actuates.
        """
        columns: list[NDArray[np.float32]] = []
        for axis in _RLDX_ACTION_AXES:
            key = f"action.{axis}"
            col = action_dict.get(key)
            if col is None:
                raise ROSConfigError(
                    f"rldx sidecar returned action dict without {key!r}; "
                    f"got keys {list(action_dict.keys())!r}. The server "
                    "must be booted with --use-sim-policy-wrapper for "
                    "LIBERO-shaped outputs."
                )
            arr = self._normalize_action_column(col, key)
            # LIBERO axes are 1-D per step.
            if arr.shape[-1] != 1:
                raise ROSConfigError(
                    f"rldx LIBERO action[{key!r}] has trailing dim {arr.shape[-1]}, expected 1."
                )
            columns.append(arr[..., 0])
        chunk = np.stack(columns, axis=-1).astype(np.float32)
        # Rescale gripper column (axis index 6 = "gripper" in
        # _RLDX_ACTION_AXES) from RLDS [0,1] (0=close, 1=open) to LIBERO
        # [-1,1] (-1=open, +1=close). Mirrors the two-step transform the
        # upstream RLDX LIBERO eval env applies before env.step (see
        # rldx/eval/sim/LIBERO/libero_env.py::{normalize,invert}_gripper_action).
        chunk[..., 6] = _rldx_gripper_to_libero(chunk[..., 6])
        return chunk

    def _assemble_gr1_chunk(self, action_dict: dict[str, Any]) -> NDArray[np.float32]:
        """Concatenate general_embodiment per-group actions into 29-D.

        Layout (waist-first, matching openral's GR1 state ordering AND
        what openral_sim.backends.robocasa._to_gr1_action_dict unflattens):

            [waist(3) | right_arm(7) | left_arm(7) | right_hand(6) | left_hand(6)]

        FT-GR1 ``general_embodiment`` is native to the Fourier GR-1
        (arms 7-D, waist 3-D, dexhands 6-D) — no dimension fudging.
        """
        cols: dict[str, NDArray[np.float32]] = {}
        for group in _GR1_ACTION_KEYS:
            key = f"action.{group}"
            col = action_dict.get(key)
            if col is None:
                raise ROSConfigError(
                    f"rldx sidecar returned action dict without {key!r}; "
                    f"got keys {list(action_dict.keys())!r}. Boot the sidecar "
                    "with --embodiment-tag GENERAL_EMBODIMENT so the FT-GR1 "
                    "wrapper produces the Fourier-native action groups."
                )
            cols[group] = self._normalize_action_column(col, key)  # (T, D)

        horizon = cols["waist"].shape[0]
        out = np.zeros((horizon, _GR1_BASIC_DIM), dtype=np.float32)
        out[:, 0:3] = cols["waist"]
        out[:, 3:10] = cols["right_arm"]
        out[:, 10:17] = cols["left_arm"]
        out[:, 17:23] = cols["right_hand"]
        out[:, 23:29] = cols["left_hand"]
        return out

    def _assemble_rc365_chunk(self, action_dict: dict[str, Any]) -> NDArray[np.float32]:
        """Concatenate FT-RC365's five action groups into ``(T, 12)``.

        Output order: ``[eef_pos(3) | eef_rot(3) | gripper(1) | base_motion(4)
        | control_mode(1)]`` — matches the order RoboCasa's PandaMobile BASIC
        composite expects after the openral_sim.backends.robocasa skew-handler
        trims the last (control_mode) slot for an 11-D env action_dim.
        """
        cols: list[NDArray[np.float32]] = []
        for group in _RC365_ACTION_KEYS:
            key = f"action.{group}"
            col = action_dict.get(key)
            if col is None:
                raise ROSConfigError(
                    f"rldx sidecar returned action dict without {key!r}; "
                    f"got keys {list(action_dict.keys())!r}. Boot the sidecar "
                    "with --embodiment-tag GENERAL_EMBODIMENT so the FT-RC365 "
                    "wrapper produces the per-component action groups."
                )
            arr = self._normalize_action_column(col, key)  # (T, D)
            cols.append(arr)
        return np.concatenate(cols, axis=-1).astype(np.float32)

    def _assemble_simpler_chunk(self, action_dict: dict[str, Any]) -> NDArray[np.float32]:
        """Stack SimplerEnv vector actions into ``(T, 7)`` with per-layout gripper postproc.

        Reads the canonical modality_keys from the OXE_BRIDGE_ORIG /
        OXE_FRACTAL ``ModalityConfig.action`` (see
        ``rldx/configs/data/simpler_{widowx,google}_config.py``):

        * ``action.end_effector_position`` — (B, T, 3) DELTA EEF
        * ``action.end_effector_rotation`` — (B, T, 3) DELTA EEF (Euler)
        * ``action.gripper_close``         — (B, T, 1) ABSOLUTE NON_EEF
          (closedness; 1 = fully closed, 0 = fully open)

        Both layouts leave the gripper column AS-IS at the raw policy
        ``gripper_close`` value in [0, 1]. WidowX's binarisation + the
        Google-style sticky-gripper state machine are applied in
        :meth:`step` against the per-chunk popleft, not here — the
        state machine spans applied steps and would otherwise lose
        its across-chunk continuity at every replan boundary. The
        environment-side step path then receives MS3's
        ``[-1, +1]`` convention (``-1`` = close, ``+1`` = open).
        """
        per_key_dims: tuple[tuple[str, int], ...] = (
            ("action.end_effector_position", 3),
            ("action.end_effector_rotation", 3),
            ("action.gripper_close", 1),
        )
        slabs: list[NDArray[np.float32]] = []
        for key, expected_dim in per_key_dims:
            col = action_dict.get(key)
            if col is None:
                raise ROSConfigError(
                    f"rldx sidecar returned action dict without {key!r}; "
                    f"got keys {list(action_dict.keys())!r}. Boot the sidecar "
                    f"with --embodiment-tag {_RLDX_LAYOUT_TO_EMBODIMENT_TAG[self.state_layout]}"
                    " so the SimplerEnv wrapper produces the expected keys."
                )
            arr = self._normalize_action_column(col, key)
            if arr.shape[-1] != expected_dim:
                raise ROSConfigError(
                    f"rldx SimplerEnv action[{key!r}] has trailing dim "
                    f"{arr.shape[-1]}, expected {expected_dim}."
                )
            slabs.append(arr)
        chunk = np.concatenate(slabs, axis=-1).astype(np.float32)  # (T, 7)
        # No per-chunk gripper postproc — the per-step sticky machine
        # in :meth:`step` reads the raw closedness directly. Google's
        # sticky-gripper (when wired) will share the same per-step
        # hook for symmetry.
        return chunk


def _resolve_simpler_eef_pos(observation: Observation) -> NDArray[np.float32]:
    """Pull the 8-D legacy SimplerEnv ``eef_pos`` vector from the obs.

    The openral SimplerEnv backend rebuilds this from the SAPIEN TCP
    link pose + last qpos channel — see
    ``openral_sim.backends.simpler_env._compute_eef_pos``.
    """
    raw = observation.get("raw")
    agent = raw.get("agent") if isinstance(raw, dict) else None
    eef = agent.get("eef_pos") if isinstance(agent, dict) else None
    if eef is None:
        raise ROSConfigError(
            "rldx adapter expects observation['raw']['agent']['eef_pos'] for "
            f"state_layout in {{simpler_widowx, simpler_google}}; got "
            f"keys {list(observation.keys())!r}. The SimplerEnv backend must "
            "be the obs source — other scenes do not publish this field."
        )
    arr = np.asarray(eef, dtype=np.float32).reshape(-1)
    if arr.shape[0] != _SIMPLER_EEF_DIM:
        raise ROSConfigError(
            f"rldx simpler_* layout expects an 8-D eef_pos vector "
            f"(x,y,z + wxyz quat + gripper); got shape {arr.shape}."
        )
    return arr


def _pick_simpler_image(observation: Observation, key: str) -> NDArray[np.uint8]:
    """Pull the single camera stream the SimplerEnv layouts expect."""
    images = observation.get("images", {})
    img = images.get(key)
    if img is None and images:
        # Fall back to whichever camera the scene published — the
        # default ``camera1`` is what ``openral_sim.backends.simpler_env``
        # always emits.
        img = next(iter(images.values()))
    if img is None:
        raise ROSConfigError(
            "rldx simpler_* layout requires at least one RGB camera on "
            f"observation.images; got {list(images.keys())}."
        )
    return np.asarray(img, dtype=np.uint8)


def _resize_to_hw(img: NDArray[np.uint8], target_hw: tuple[int, int]) -> NDArray[np.uint8]:
    """Bilinear resize HWC uint8 to ``(H, W)``."""
    h, w = target_hw
    if img.shape[:2] == (h, w):
        return img
    from PIL import Image as _PILImage

    return np.asarray(
        _PILImage.fromarray(img).resize((w, h), _PILImage.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _bridge_quat_to_euler(quat_wxyz: NDArray[np.float32]) -> tuple[float, float, float]:
    """SAPIEN wxyz quat → bridge-corrected Euler RPY (radians).

    Implements the upstream ``WidowXBridgeEnv._process_observation``
    formula: ``rm = quat2mat(quat_wxyz); rpy = mat2euler(rm @
    default_rot.T)``. Hand-rolled instead of delegating to
    ``transforms3d.euler.mat2euler`` because the latest tagged
    transforms3d (0.4.2) calls ``np.array(..., copy=False)`` which
    raises under NumPy 2.0's strict copy semantics. The math here
    mirrors ``transforms3d``'s default ``sxyz`` Euler convention so
    drop-in parity with the upstream eval env stays intact.
    """
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    rm_bridge = _quat_wxyz_to_mat(quat)
    rotated = rm_bridge @ _SIMPLER_BRIDGE_DEFAULT_ROT.astype(np.float64).T
    return _mat_to_euler_sxyz(rotated)


def _quat_wxyz_to_mat(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    """SAPIEN-style wxyz quaternion → 3x3 rotation matrix.

    Matches ``transforms3d.quaternions.quat2mat`` for a normalised unit
    quaternion; matches the upstream WidowXBridgeEnv reference closely
    enough that downstream Euler outputs are bit-identical for the
    fixtures the unit tests pin.
    """
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < _QUAT_NORM_EPS:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    xx, xy, xz = x * x * s, x * y * s, x * z * s
    yy, yz, zz = y * y * s, y * z * s, z * z * s
    return np.asarray(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _mat_to_euler_sxyz(mat: NDArray[np.float64]) -> tuple[float, float, float]:
    """3x3 rotation matrix → ``sxyz`` Euler angles (roll, pitch, yaw).

    Equivalent to ``transforms3d.euler.mat2euler(mat, axes='sxyz')`` but
    NumPy 2.0-compatible. ``sxyz`` is the upstream default and what
    ``rldx/eval/sim/SimplerEnv/simpler_env.py`` consumes.
    """
    m = np.asarray(mat, dtype=np.float64)
    sy = math.hypot(m[0, 0], m[1, 0])
    if sy > _EULER_GIMBAL_EPS:
        roll = math.atan2(m[2, 1], m[2, 2])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:
        roll = math.atan2(-m[1, 2], m[1, 1])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def _rldx_gripper_to_libero(gripper: NDArray[np.float32]) -> NDArray[np.float32]:
    """Map an RLDS-convention gripper column to the LIBERO convention.

    RLDX-1 emits ``action.gripper`` in the RLDS dataset convention —
    ``[0, 1]`` where ``0=close`` and ``1=open`` — because that is the
    space its training data was standardized into. LIBERO / robosuite's
    OSC controller however consumes a gripper command in ``[-1, +1]``
    with the **opposite** sign convention (``-1=open`` / ``+1=close``).

    The upstream RLDX LIBERO eval env composes two transforms before
    stepping the env (see
    ``rldx/eval/sim/LIBERO/libero_env.py::normalize_gripper_action``
    and ``::invert_gripper_action``):

      1. ``y = 2 * (x - 0) / (1 - 0) - 1``   — rescale [0, 1] → [-1, 1].
      2. ``y = sign(y)``                     — binarize to {-1, +1}.
      3. ``y = -y``                          — flip sign so -1=open.

    Net effect: ``out = -sign(2 * g - 1)`` — i.e. any RLDS gripper
    value above 0.5 becomes ``-1`` (open) and any value below 0.5
    becomes ``+1`` (close). Values exactly at 0.5 stay at ``0`` (the
    only zero of ``sign``), which preserves the "no-op" semantics of a
    truly indecisive policy output.

    Args:
        gripper: Raw RLDS-convention gripper column shaped ``(T,)``.

    Returns:
        Float32 array of the same shape with values in ``{-1, 0, +1}``,
        ready to feed straight into LIBERO's 7-D action vector at
        index ``6`` (see :data:`_RLDX_ACTION_AXES`).
    """
    g = np.asarray(gripper, dtype=np.float32)
    out: NDArray[np.float32] = (-np.sign(2.0 * g - 1.0)).astype(np.float32)
    return out


def _env_bool(name: str, default: bool) -> bool:
    """Parse a permissive bool env var (``1`` / ``true`` / ``yes``)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Back-compat alias: the adapter was named for RLDX-1 (which landed first) before
# it was generalized to the GR00T family (ADR-0046). Existing imports + the
# rldx-specific tests keep using this name.
_RLDXSidecarAdapter = _Gr00tFamilySidecarAdapter


@POLICIES.register("rldx")  # type: ignore[arg-type]
def _build_rldx(env_cfg: Any) -> _Gr00tFamilySidecarAdapter:
    """Build the auto-managed RLDX-1 adapter from a SimEnvironment.

    YAML knobs (via ``vla.extra``):
        host             -- ZMQ bind host of the sidecar (default 127.0.0.1).
        port             -- ZMQ port of the sidecar. Default is derived
                            per-checkpoint (a stable value in 20000–39999)
                            so two different checkpoints never share one
                            port; pin this to share a hand-managed sidecar.
        replan_steps     -- env steps replayed per server forward
                            (default 8; half the model's 16-action chunk,
                            matching the RTC ``exec_horizon`` convention).
        image_size       -- video frame resize target (default 256, the
                            LiberoEnv native resolution).
        timeout_ms       -- per-request ZMQ recv timeout (default 60_000).
                            First call after sidecar boot is slow because
                            the policy hasn't loaded; raise this for big
                            checkpoints.
        camera_keys      -- override the (agentview, wrist) camera pair
                            when your scene doesn't follow the LIBERO
                            convention.
        auto_spawn       -- when True (default) the adapter forks
                            ``tools/rldx_sidecar.py`` if no server is
                            already listening on ``host:port``; set False
                            to preserve the older "boot it yourself"
                            workflow.
        boot_timeout_s   -- max seconds to wait for the spawned sidecar
                            to answer ``ping`` (default 900 — the first
                            boot includes the upstream ``git clone`` +
                            ``uv sync`` of RLDX-1, several minutes).
        quantization     -- backbone quantization scheme for the spawned
                            sidecar (default ``nf4``; ``int8`` or
                            ``none`` also accepted — see
                            ``tools/rldx_sidecar.py --quantization``).
        embodiment_tag   -- ``EmbodimentTag`` value handed to the
                            sidecar (default ``GENERAL_EMBODIMENT``, the
                            FT-LIBERO / FT-GR1 / FT-RC365 contract).
        model_id         -- explicit HF id override; default is the
                            rSkill manifest's ``weights_uri``
                            (``hf://RLWRLD/...``).

    Environment overrides (ergonomic; no YAML edit required):
        OPENRAL_RLDX_HOST, OPENRAL_RLDX_PORT, OPENRAL_RLDX_AUTO_SPAWN,
        OPENRAL_RLDX_BOOT_TIMEOUT_S, OPENRAL_RLDX_QUANTIZATION,
        OPENRAL_RLDX_EMBODIMENT_TAG, OPENRAL_RLDX_MODEL_ID,
        OPENRAL_RLDX_SIDECAR_SCRIPT.
    """
    spec = env_cfg.vla
    extra = dict(spec.extra or {})
    host = os.environ.get("OPENRAL_RLDX_HOST", str(extra.get("host", "127.0.0.1")))
    # Port is resolved *after* layout/embodiment/model below, so the
    # per-identity default can hash them in. Explicit env / vla.extra pins
    # still win.
    # Replan precedence: vla.extra.replan_steps > manifest.n_action_steps >
    # half-chunk default (``_RLDX_CHUNK_LEN // 2 = 8``). The manifest
    # field was previously declared but unused — the adapter only read
    # vla.extra, so a value set in rskill.yaml was silently ignored.
    # Honour it now so a checkpoint can ship its own tested cadence
    # (e.g. 16 = "replay the full chunk, half as many inference
    # round-trips, twice the open-loop horizon between observations").
    #
    # The half-chunk fallback is **still reachable in production**:
    # ``rskills/rldx1-ft-libero-nf4``, ``rldx1-ft-gr1-nf4``, and
    # ``rldx1-ft-rc365-nf4`` deliberately omit ``n_action_steps`` (the
    # manifest comment "equals chunk_size (full chunk replay)" is a lie
    # the schema does not enforce — ``RSkillManifest.n_action_steps``
    # defaults to ``None``, not to ``chunk_size``). With no
    # ``vla.extra.replan_steps`` override either, those rskills depend
    # on this 8-step half-chunk default to actually replay anything,
    # so the fallback stays.
    _manifest = load_manifest_for_spec(spec)
    _manifest_steps = getattr(_manifest, "n_action_steps", None) if _manifest else None
    replan_steps = int(
        extra.get(
            "replan_steps",
            _manifest_steps if _manifest_steps is not None else _RLDX_CHUNK_LEN // 2,
        )
    )
    image_size = int(extra.get("image_size", 256))
    timeout_ms = int(extra.get("timeout_ms", 60_000))
    cam_keys_raw = extra.get("camera_keys")
    if isinstance(cam_keys_raw, (list, tuple)) and len(cam_keys_raw) == _RLDX_CAMERA_PAIR_LEN:
        camera_keys = (str(cam_keys_raw[0]), str(cam_keys_raw[1]))
    else:
        # Per ADR-0069: fall back to the scene's canonical camera names
        # (e.g. ``("front", "wrist")`` on franka_panda) when the rskill
        # manifest does not pin ``vla.extra.camera_keys`` explicitly. The
        # ordinal ``("camera1", "camera2")`` legacy default is retained
        # only when the scene leaves ``cameras`` empty.
        _scene_cams = list(getattr(env_cfg.scene, "cameras", []) or [])
        camera_keys = (
            _scene_cams[0] if len(_scene_cams) > 0 else "camera1",
            _scene_cams[1] if len(_scene_cams) > 1 else "camera2",
        )
    auto_spawn = _env_bool("OPENRAL_RLDX_AUTO_SPAWN", bool(extra.get("auto_spawn", True)))
    boot_timeout_s = float(
        os.environ.get("OPENRAL_RLDX_BOOT_TIMEOUT_S") or extra.get("boot_timeout_s", 900.0)
    )
    quantization = str(
        os.environ.get("OPENRAL_RLDX_QUANTIZATION") or extra.get("quantization", "nf4")
    )
    embodiment_tag = str(
        os.environ.get("OPENRAL_RLDX_EMBODIMENT_TAG")
        or extra.get("embodiment_tag", "GENERAL_EMBODIMENT")
    )
    model_id_override = os.environ.get("OPENRAL_RLDX_MODEL_ID") or extra.get("model_id")
    # Honour the rskill manifest's image_preprocessing.flip_180 — LIBERO
    # checkpoints want the 180° rotation that the upstream eval env
    # applies to agentview / wrist frames.
    from openral_rskill._vla_core import resolve_image_preprocessing

    manifest = load_manifest_for_spec(spec)
    ip = resolve_image_preprocessing(manifest, spec.extra)

    # Dispatch obs / action layout off the manifest's state_contract.
    #   "gr1"             → Fourier GR-1 general_embodiment (1 ego_view, 5 state groups)
    #   "rc365"           → RoboCasa-365 general_embodiment (3 cams, 5 state groups, 12-D action)
    #   "simpler_widowx"  → SimplerEnv WidowX bridge_orig (1 cam, 8 state scalars, 7-D action)
    #   "simpler_google"  → SimplerEnv Google fractal20220817 (1 cam, 8 state scalars, 7-D action)
    #   default           → LIBERO-flat (2 cams, 7 state scalars)
    layout = _resolve_state_layout(manifest)

    # Reject a too-few-cameras scene up front (before the multi-minute sidecar
    # boot) instead of failing opaquely on the first obs assembly.
    _require_scene_cameras(env_cfg, layout=layout, camera_keys=camera_keys, family="rldx")

    # Layout drives the upstream EmbodimentTag — unless the user pinned
    # one explicitly via env / vla.extra (kept as an escape hatch for
    # bespoke checkpoints that ship a custom tag).
    if (
        "OPENRAL_RLDX_EMBODIMENT_TAG" not in os.environ
        and extra.get("embodiment_tag") is None
        and layout in _RLDX_LAYOUT_TO_EMBODIMENT_TAG
    ):
        embodiment_tag = _RLDX_LAYOUT_TO_EMBODIMENT_TAG[layout]

    # Port precedence: OPENRAL_RLDX_PORT > vla.extra.port > per-identity
    # default. The derived default keeps two different checkpoints off the
    # same port so neither silently reuses the other's sidecar.
    port = _resolve_sidecar_port(
        port_env=os.environ.get("OPENRAL_RLDX_PORT"),
        extra_port=extra.get("port"),
        family="rldx",
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
        _camera_keys=camera_keys,
    )
