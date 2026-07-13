"""InternVLA-N1 dual-system navigation inference server (sidecar process).

Runs InternRobotics' InternVLA-N1 / DualVLN (arXiv:2512.08186) — a
Qwen2.5-VL-7B System-2 waypoint planner + NavDP DiT System-1 trajectory
policy — in its own isolated venv (upstream InternNav pins
``transformers==4.51.0``, incompatible with the py3.12 workspace) and serves
velocity commands over a ZMQ REP socket. The openral-side client is the
``internvla_n1`` policy adapter
(:mod:`openral_sim.policies.internvla_n1`), which speaks the canonical
:class:`openral_sim.sidecar.SidecarClient` frame
(``{"endpoint": str, "data": {...}}`` + the ``__ndarray__``/npy codec).

Wire protocol (msgpack, ZMQ REQ/REP):
  in : {"endpoint": "ping"}
       -> {"ok": True, "model": str, "family": "internvla_n1",
           "quantization": str}
  in : {"endpoint": "reset"}                    -> {"ok": True}
  in : {"endpoint": "step", "data": {
            "rgb": <ndarray HxWx3 uint8>,
            "depth": <ndarray HxW float32, METERS>,
            "instruction": str}}
       -> {"ok": True, "kind": "discrete"|"trajectory",
           "twist": [v_forward_mps, w_yaw_radps], "stop": bool,
           "actions": [int, ...] | None,
           "trajectory": [[x, y, theta], ...] | None,
           "pixel": [row, col] | None, "latency_ms": float}
  in : {"endpoint": "close"} / {"endpoint": "shutdown"} -> {"ok": True} (exits)
  on error -> {"error": str}

The look-down dance is handled server-side: when System-2 emits the
look-down action (``[5]``) the agent is immediately re-stepped with
``look_down=True`` on the same frame, mirroring the upstream realworld
deployment loop (InternNav ``scripts/realworld/http_internvla_server.py``).

Depth is REQUIRED and must be metric meters (upstream feeds
``uint16/10000.0``); this server refuses all-zero depth rather than
letting System-1 hallucinate trajectories from a broken sensor.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

# msgpack + zmq are imported lazily inside main() so this module imports (and
# its pure helpers unit-test) under the workspace interpreter, which does not
# ship the sidecar-venv wire deps.

_NDARRAY_SENTINEL = "__ndarray__"

# Discrete VLN-CE action ids emitted by System-2 (see upstream
# ``InternVLAN1AsyncAgent.actions2idx``): STOP=0, forward=1 (0.25 m),
# turn-left=2 (15 deg), turn-right=3 (15 deg), look-down=5.
_ACTION_STOP = 0
_ACTION_FORWARD = 1
_ACTION_LEFT = 2
_ACTION_RIGHT = 3
_ACTION_LOOK_DOWN = 5

# Discrete-action → twist magnitudes. VLN-CE's forward step is 0.25 m and
# its turn is 15 deg; expressed as a velocity held for ~1 s of wall time
# between S2 replans (PLAN_STEP_GAP frames). Tunable via CLI for real bases.
_DEFAULT_FORWARD_MPS = 0.25
_DEFAULT_TURN_RADPS = float(np.deg2rad(15.0))


def discrete_action_to_twist(
    actions: list[int], *, forward_mps: float, turn_radps: float
) -> tuple[float, float, bool]:
    """Map a VLN-CE discrete action sequence to ``(v_forward, w_yaw, stop)``.

    Only the FIRST action drives motion this step (the sequence is a plan the
    base executes one command at a time). STOP anywhere in the sequence latches
    ``stop=True``. Pure + side-effect-free so it is unit-testable without the
    8.3B model — see ``tests/unit/test_internvla_n1_action_mapping.py``.
    """
    stop = _ACTION_STOP in actions
    if stop or not actions:
        return 0.0, 0.0, stop
    first = actions[0]
    if first == _ACTION_FORWARD:
        return forward_mps, 0.0, False
    if first == _ACTION_LEFT:
        return 0.0, turn_radps, False
    if first == _ACTION_RIGHT:
        return 0.0, -turn_radps, False
    return 0.0, 0.0, False  # look-down / unknown → no base motion this step


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {_NDARRAY_SENTINEL: True, "npy": buf.getvalue()}
    return obj


def _decode(obj: dict[str, Any]) -> Any:
    if _NDARRAY_SENTINEL in obj and "npy" in obj:
        return np.load(io.BytesIO(obj["npy"]), allow_pickle=False)
    return obj


def _ensure_aux_depth_checkpoint() -> None:
    """Fetch the DepthAnythingV2 aux checkpoint the System-1 RGB encoder needs.

    ``internnav...internvla_n1_arch.build_depthanythingv2`` hard-codes
    ``torch.load("checkpoints/depth_anything_v2_metric_hypersim_vits.pth")``
    (a ~50 MB ViT-S metric-depth encoder used as the System-1 RGB feature
    extractor) relative to cwd. We chdir to ``--work-dir`` before load, so
    materialize it under ``<work-dir>/checkpoints/`` via ``hf_hub_download``
    (idempotent — skipped if already present) instead of asking the operator to
    place a file by hand.
    """
    dst = Path("checkpoints") / "depth_anything_v2_metric_hypersim_vits.pth"
    if dst.exists():
        return
    from huggingface_hub import hf_hub_download

    dst.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
        filename="depth_anything_v2_metric_hypersim_vits.pth",
    )
    os.symlink(path, dst)
    print(f"[internvla-n1-server] linked aux depth encoder -> {dst}", flush=True)


def _build_agent(args: argparse.Namespace) -> Any:
    """Load InternVLA-N1 with the requested quantization and wrap the agent.

    Mirrors ``InternVLAN1AsyncAgent.__init__`` (InternNav
    ``internnav/agent/internvla_n1_agent_realworld.py``) but swaps the
    hardcoded ``bfloat16 + flash_attention_2`` load for a quantization-aware
    one: NF4/int8 via bitsandbytes for ≤ 8 GiB GPUs, ``sdpa`` attention so
    flash-attn need not build in the sidecar venv. Everything else (history
    handling, S2 prompting, S1 latents→trajectory) is reused from upstream
    unchanged.
    """
    import importlib.util

    import torch

    _ensure_aux_depth_checkpoint()

    # InternVLA-N1's NextDiT System-1 calls `enable_gradient_checkpointing()` in
    # its constructor — a TRAINING-only feature. In diffusers 0.33.x the base
    # `enable_gradient_checkpointing` calls `_set_gradient_checkpointing(enable=
    # ...)`, but `LuminaNextDiT2DModel` (which NextDiT subclasses) still has the
    # old positional signature, so model construction crashes at load. We do
    # inference only, so no-op it before the model builds.
    import diffusers.models.modeling_utils as _dm

    _dm.ModelMixin.enable_gradient_checkpointing = lambda self, *a, **k: None

    import internnav
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
    from transformers import AutoProcessor

    # Load the realworld agent module BY FILE, not via `internnav.agent` —
    # that package's __init__ eagerly imports every agent (CMA, RDP, dialog)
    # and drags in eval-stack deps the inference venv deliberately lacks.

    agent_py = Path(internnav.__file__).parent / "agent" / "internvla_n1_agent_realworld.py"
    spec = importlib.util.spec_from_file_location("_internvla_n1_agent_realworld", agent_py)
    assert spec is not None and spec.loader is not None
    agent_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_mod)
    InternVLAN1AsyncAgent = agent_mod.InternVLAN1AsyncAgent

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": {"": args.device},
    }
    if args.quantization in {"nf4", "int8"}:
        from transformers import BitsAndBytesConfig

        # Keep the System-1 head (NavDP DiT + latent/traj projections) and
        # the token embeddings at bf16 — only the Qwen2.5-VL backbone
        # linears are worth quantizing, and bnb-packed uint8 weights inside
        # the DiT would break its dtype-inference paths.
        skip = [
            "lm_head",
            "embed_tokens",
            "traj_dit",
            "navdp",
            "action_encoder",
            "action_decoder",
        ]
        if args.quantization == "nf4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=skip,
            )
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_skip_modules=skip
            )

    class _QuantizedAgent(InternVLAN1AsyncAgent):
        def __init__(self, a: SimpleNamespace) -> None:
            # Reimplements upstream __init__ (load call swapped); do NOT
            # call super().__init__.
            self.device = torch.device(a.device)
            self.save_dir = "test_data/" + time.strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.save_dir, exist_ok=True)
            self.model = InternVLAN1ForCausalLM.from_pretrained(a.model_path, **load_kwargs)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(a.model_path)
            self.processor.tokenizer.padding_side = "left"
            self.resize_w = a.resize_w
            self.resize_h = a.resize_h
            self.num_history = a.num_history
            self.PLAN_STEP_GAP = a.plan_step_gap
            prompt = (
                "You are an autonomous navigation assistant. Your task is to "
                "<instruction>. Where should you go next to stay on track? "
                "Please output the next waypoint's coordinates in the image. "
                "Please output STOP when you have successfully completed the task."
            )
            self.conversation = [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": ""},
            ]
            self.conjunctions = [
                "you can see ",
                "in front of you is ",
                "there is ",
                "you can spot ",
                "you are toward the ",
                "ahead of you is ",
                "in your sight is ",
            ]
            from collections import OrderedDict

            self.actions2idx = OrderedDict({"STOP": [0], "↑": [1], "←": [2], "→": [3], "↓": [5]})
            self.rgb_list: list[Any] = []
            self.depth_list: list[Any] = []
            self.pose_list: list[Any] = []
            self.episode_idx = 0
            self.conversation_history: list[Any] = []
            self.llm_output = ""
            self.past_key_values = None
            self.last_s2_idx = -100
            self.output_action = None
            self.output_latent = None
            self.output_pixel = None
            self.pixel_goal_rgb = None
            self.pixel_goal_depth = None

    agent_args = SimpleNamespace(
        device=args.device,
        model_path=args.model,
        resize_w=args.resize,
        resize_h=args.resize,
        num_history=args.num_history,
        plan_step_gap=args.plan_step_gap,
    )
    return _QuantizedAgent(agent_args)


def _step_agent(
    agent: Any,
    rgb: NDArray[np.uint8],
    depth: NDArray[np.float32],
    instruction: str,
    *,
    forward_mps: float,
    turn_radps: float,
) -> dict[str, Any]:
    """One dual-system step → uniform reply dict with a twist command."""
    pose = np.eye(4)  # upstream realworld loop feeds identity — pose is logged, unused
    out = agent.step(rgb, depth, pose, instruction, intrinsic=None)
    if out.output_action is not None and list(out.output_action) == [_ACTION_LOOK_DOWN]:
        out = agent.step(rgb, depth, pose, instruction, intrinsic=None, look_down=True)

    if out.output_action is not None:
        actions = [int(a) for a in out.output_action]
        v, w, stop = discrete_action_to_twist(
            actions, forward_mps=forward_mps, turn_radps=turn_radps
        )
        return {
            "kind": "discrete",
            "actions": actions,
            "trajectory": None,
            "pixel": None,
            "twist": [float(v), float(w)],
            "stop": bool(stop),
        }

    traj = np.asarray(out.output_trajectory, dtype=np.float32).reshape(-1, 3)
    # Upstream Go2 loop: velocity from the last waypoint (robot frame).
    v, w = agent.trajectory_tovw(traj)
    pixel = None if out.output_pixel is None else [int(p) for p in out.output_pixel]
    return {
        "kind": "trajectory",
        "actions": None,
        "trajectory": traj.tolist(),
        "pixel": pixel,
        "twist": [float(v), float(w)],
        "stop": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF id or local path of the checkpoint.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5781)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quantization", choices=("none", "nf4", "int8"), default="nf4")
    ap.add_argument("--resize", type=int, default=384)
    ap.add_argument("--num-history", type=int, default=8)
    ap.add_argument("--plan-step-gap", type=int, default=4)
    ap.add_argument("--forward-mps", type=float, default=_DEFAULT_FORWARD_MPS)
    ap.add_argument("--turn-radps", type=float, default=_DEFAULT_TURN_RADPS)
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=Path.home() / ".cache" / "openral" / "internvla-n1-sidecar" / "runs",
        help="cwd for the agent's per-episode debug frame dumps.",
    )
    args = ap.parse_args()

    import msgpack
    import zmq

    args.work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.work_dir)  # upstream agent dumps debug jpgs relative to cwd

    print(f"[internvla-n1-server] loading {args.model} quant={args.quantization}...", flush=True)
    t0 = time.perf_counter()
    agent = _build_agent(args)
    import torch

    print(
        f"[internvla-n1-server] loaded in {time.perf_counter() - t0:.1f}s; "
        f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[internvla-n1-server] listening on tcp://{args.host}:{args.port}", flush=True)

    running = True
    while running:
        req = msgpack.unpackb(sock.recv(), object_hook=_decode, raw=False)
        endpoint = req.get("endpoint") if isinstance(req, dict) else None
        data = req.get("data") or {} if isinstance(req, dict) else {}
        try:
            if endpoint == "ping":
                reply: dict[str, Any] = {
                    "ok": True,
                    "model": args.model,
                    "family": "internvla_n1",
                    "quantization": args.quantization,
                }
            elif endpoint == "reset":
                agent.reset()
                reply = {"ok": True}
            elif endpoint == "step":
                t0 = time.perf_counter()
                rgb = np.ascontiguousarray(data["rgb"], dtype=np.uint8)
                depth = np.ascontiguousarray(data["depth"], dtype=np.float32)
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise ValueError(f"rgb must be HxWx3 uint8, got {rgb.shape}")
                if depth.shape != rgb.shape[:2]:
                    raise ValueError(f"depth {depth.shape} does not match rgb {rgb.shape[:2]}")
                if not np.any(depth > 0.0):
                    raise ValueError(
                        "depth is all-zero — a broken depth sensor would make System-1 "
                        "hallucinate trajectories; refusing (depth must be metric meters)."
                    )
                reply = _step_agent(
                    agent,
                    rgb,
                    depth,
                    str(data["instruction"]),
                    forward_mps=args.forward_mps,
                    turn_radps=args.turn_radps,
                )
                reply["ok"] = True
                reply["latency_ms"] = (time.perf_counter() - t0) * 1000.0
            elif endpoint in ("close", "shutdown"):
                reply = {"ok": True}
                running = False
            else:
                reply = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as exc:  # server must reply, not die, on a bad frame
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        sock.send(msgpack.packb(reply, default=_encode, use_bin_type=True))

    print("[internvla-n1-server] shutting down", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
