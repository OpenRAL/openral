r"""3D Diffuser Actor policy sidecar — RLBench keyframe inference over ZMQ.

ADR-0061. 3D Diffuser Actor (Ke et al., 2024, arXiv:2402.10885, MIT) is a
diffusion policy over end-effector keyposes for RLBench. Its released PerAct
18-task checkpoint pins an older stack (the ``MohitShridhar/RLBench@peract``
fork + CLIP + a torch build that must be Ada-compatible) that cannot live in the
openral py3.12 workspace — so, like the RLDX-1 policy adapter
(:mod:`openral_sim.policies.rldx`), it runs in its own venv as a long-lived
process and is driven over ZMQ REQ/REP framed by msgpack.

This file is the **sidecar side** (no openral import). It owns the
``DiffuserActor`` model, the CLIP-encoded instruction embeddings, and the
per-episode observation history (the policy is trained with ``num_history=3``).
The openral side is :mod:`openral_sim.policies.rlbench_3dda`.

Wire protocol::

    ping       -> {"ok": True, "model": "3d_diffuser_actor"}
    reset      -> {"ok": True}                       # clears per-episode history
    get_action -> {"action": (8,) float32}           # [x y z qx qy qz qw gripper_open]

``get_action`` request data carries the RLBench observation the scene sidecar
produced: per-camera ``images`` (HWC uint8) + ``point_clouds`` (HWC float32) +
``gripper_pose`` (7) + ``gripper_open``. The instruction is the precomputed CLIP
token embedding for the launched task (matches the upstream evaluator exactly).

VRAM: inference peaks ~0.43 GB (8 GB-host friendly); runs under ``no_grad`` (the
100-step diffusion loop would otherwise build a graph and OOM).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import random
import sys
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

_CAMERAS = ("left_shoulder", "right_shoulder", "wrist", "front")
_NUM_HISTORY = 3
_INTERPOLATION_LENGTH = 2


def _encode_ndarray(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray__": True, "npy": buf.getvalue()}
    return obj


def _decode_ndarray(obj: dict[str, Any]) -> Any:
    if "__ndarray__" in obj:
        return np.load(io.BytesIO(obj["npy"]), allow_pickle=False)
    return obj


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenRAL 3D Diffuser Actor policy sidecar")
    p.add_argument("--repo", required=True, help="path to the 3d_diffuser_actor checkout")
    p.add_argument("--checkpoint", required=True, help="path to diffuser_actor_peract.pth")
    p.add_argument("--instructions", required=True, help="path to peract instructions.pkl")
    p.add_argument("--bounds", required=True, help="path to 18_peract_tasks_location_bounds.json")
    p.add_argument("--rlbench-task", required=True, help="task stem for instruction lookup")
    p.add_argument("--variation", type=int, default=0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=21200)
    return p.parse_args(argv)


class _Diffuser3DActor:
    """Loads the model + instruction embedding; keeps per-episode obs history."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        sys.path.insert(0, args.repo)
        import torch

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._build()
        self.reset()

    def _build(self) -> None:
        torch = self._torch
        from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor

        with open(self._args.bounds) as bf:
            d = json.load(bf)
        mins = np.array([v[0] for v in d.values()]).min(0)
        maxs = np.array([v[1] for v in d.values()]).max(0)
        bounds = np.stack([mins, maxs])

        model = DiffuserActor(
            backbone="clip",
            image_size=(self._args.image_size, self._args.image_size),
            embedding_dim=120,
            num_vis_ins_attn_layers=2,
            use_instruction=True,
            fps_subsampling_factor=5,
            gripper_loc_bounds=bounds,
            rotation_parametrization="6D",
            quaternion_format="wxyz",
            diffusion_timesteps=100,
            nhist=_NUM_HISTORY,
        ).to(self._device)
        ck = self._args.checkpoint
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        sd = sd.get("weight", sd)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model.eval()
        self._model = model

        with open(self._args.instructions, "rb") as fh:
            instructions = pickle.load(fh)
        task = self._args.rlbench_task
        var = self._args.variation
        instrs = list(instructions[task][var])
        self._instr = random.choice(instrs).unsqueeze(0).to(self._device)
        print(f"[rlbench_3dda_sidecar] model ready on {self._device}", flush=True)

    def reset(self) -> None:
        # Per-episode rolling history of (gripper) state; rgb/pcd are Markovian.
        self._gripper_hist: list[NDArray[np.float32]] = []

    def get_action(self, obs: dict[str, Any]) -> NDArray[np.float32]:
        torch = self._torch

        # Assemble (1, ncam, 3, H, W) rgb [-1,1] and pcd from the scene obs.
        rgbs: list[NDArray[np.float32]] = []
        pcds: list[NDArray[np.float32]] = []
        for cam in _CAMERAS:
            rgb = np.asarray(obs["images"][cam], dtype=np.float32).transpose(2, 0, 1)
            rgb = 2.0 * (rgb / 255.0) - 1.0  # transform() normalisation
            pcd = np.asarray(obs["point_clouds"][cam], dtype=np.float32).transpose(2, 0, 1)
            rgbs.append(rgb)
            pcds.append(pcd)
        rgb_t = torch.from_numpy(np.stack(rgbs)).unsqueeze(0).to(self._device)  # (1,N,3,H,W)
        pcd_t = torch.from_numpy(np.stack(pcds)).unsqueeze(0).to(self._device)

        gripper = np.concatenate(
            [np.asarray(obs["gripper_pose"], dtype=np.float32), [float(obs["gripper_open"])]]
        ).astype(np.float32)  # concat with a py float promotes to float64 — force float32
        self._gripper_hist.append(gripper)
        hist = self._gripper_hist[-_NUM_HISTORY:]
        while len(hist) < _NUM_HISTORY:
            hist = [hist[0], *hist]
        gripper_t = torch.from_numpy(np.stack(hist)).unsqueeze(0).to(self._device)  # (1,nhist,8)

        with torch.no_grad():
            rgbs_in = rgb_t / 2 + 0.5  # back to [0,1]; the encoder applies CLIP norm
            fake_traj = torch.zeros(
                1, _INTERPOLATION_LENGTH - 1, gripper_t.shape[-1], device=self._device
            )
            traj_mask = torch.zeros(
                1, _INTERPOLATION_LENGTH - 1, dtype=torch.bool, device=self._device
            )
            traj = self._model(
                fake_traj,
                traj_mask,
                rgbs_in,
                pcd_t,
                self._instr,
                gripper_t[..., :7],
                run_inference=True,
            )
        keypose = traj[-1].cpu().numpy()  # (interp_len-1, 8)
        action = np.asarray(keypose[-1], dtype=np.float32).reshape(-1)
        action[-1] = float(round(action[-1]))  # gripper open/close
        return cast(NDArray[np.float32], action)

    def close(self) -> None:
        pass


def _serve(policy: _Diffuser3DActor, *, host: str, port: int) -> int:
    import msgpack
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[rlbench_3dda_sidecar] serving on tcp://{host}:{port}", flush=True)
    running = True
    while running:
        raw = sock.recv()
        req = msgpack.unpackb(raw, object_hook=_decode_ndarray, raw=False)
        endpoint = req.get("endpoint")
        data = req.get("data", {}) or {}
        try:
            if endpoint == "ping":
                reply: dict[str, Any] = {"ok": True, "model": "3d_diffuser_actor"}
            elif endpoint == "reset":
                policy.reset()
                reply = {"ok": True}
            elif endpoint == "get_action":
                reply = {"action": policy.get_action(data["observation"])}
            elif endpoint == "close":
                reply = {"ok": True}
                running = False
            else:
                reply = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as exc:
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        sock.send(msgpack.packb(reply, default=_encode_ndarray, use_bin_type=True))
    sock.close(linger=0)
    ctx.term()
    return 0


def main(argv: list[str]) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = _parse_args(argv)
    policy = _Diffuser3DActor(args)
    return _serve(policy, host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
