#!/usr/bin/env python
"""Isaac Sim scene sidecar — runs Isaac Lab in its own py3.11 venv (ADR-0045).

This is the **Isaac side** of the Isaac Sim backend. It is launched (auto-spawned)
by :mod:`openral_sim.backends.isaac_sim` running under the openral py3.12 venv,
and it runs under the separate Isaac Sim py3.11 venv whose interpreter is named
by ``OPENRAL_ISAAC_SIDECAR_PYTHON``.

It launches the Omniverse Kit app headless, builds a minimal Isaac Lab
manipulation scene (a Franka arm + a liftable cube + a tiled RGB camera), and
serves a ZMQ REP loop speaking the same msgpack + ndarray framing the openral
side uses:

    ping  -> {"ok": True, "action_dim": int, "task": str, "layout": str}
    reset -> {"observation": {...}}
    step  -> {"observation": {...}, "reward", "terminated", "truncated", "info"}
    render-> {"frame": <uint8 HWC>|None}
    close -> {"ok": True}

Observation dict shape (eval-layer contract):
    {"images": {"camera1": <H,W,3 uint8>}, "state": <1-D float32>, "task": str}

IMPORTANT (Isaac import order): ``SimulationApp`` MUST be constructed before any
``omni.*`` / ``isaaclab`` import. We therefore do all heavy imports *inside*
:func:`main`, after the app is up. Do not hoist them to module scope.

Licensing: this process sets ``OMNI_KIT_ACCEPT_EULA=YES`` — running it is the
user's acceptance of the NVIDIA Omniverse license. The Omniverse Kit components
are proprietary and are never vendored into the repo (CLAUDE.md §1.9); this
launcher only drives an externally-provisioned install.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Any

import numpy as np

# Accept the NVIDIA Omniverse EULA non-interactively. Without this the Kit
# bootstrap blocks on a stdin "Do you accept the EULA?" prompt and dies on EOF
# when spawned with a closed stdin.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


# ── msgpack ndarray codec (matches openral_sim.backends.isaac_sim) ────────────


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
    p = argparse.ArgumentParser(description="OpenRAL Isaac Sim scene sidecar")
    p.add_argument("--task", required=True, help="task id, e.g. isaac_sim/lift_cube")
    p.add_argument("--robot", default="franka_panda")
    p.add_argument("--instruction", default="")
    p.add_argument("--obs-height", type=int, default=256)
    p.add_argument("--obs-width", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--success-key", default="is_success")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5757)
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--layout",
        default="lift_cube",
        choices=["lift_cube", "bowl_plate", "manifest"],
        help=(
            "lift_cube = 8-D joint-delta PoC; bowl_plate = LIBERO-shaped 7-D "
            "EE-delta scene; manifest = robot-agnostic URDF-driven scene "
            "(needs --robot-spec, ADR-0045 amendment)"
        ),
    )
    p.add_argument(
        "--robot-spec",
        default=None,
        help="path to the JSON isaac robot spec (manifest layout only)",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    # 1) Launch the Kit app FIRST — every omni.* / isaaclab import below depends
    #    on a live SimulationApp.
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": bool(args.headless)})

    try:
        # 2) Heavy imports, only valid post-launch. Pick the scene by --layout.
        if args.layout == "manifest":
            import json

            from isaac_manifest_scene import IsaacManifestScene

            if not args.robot_spec:
                raise SystemExit("--layout manifest requires --robot-spec <path>")
            with open(args.robot_spec, encoding="utf-8") as fh:
                robot_spec = json.load(fh)
            scene: Any = IsaacManifestScene(
                robot_spec=robot_spec,
                obs_height=args.obs_height,
                obs_width=args.obs_width,
                instruction=args.instruction,
                success_key=args.success_key,
                max_steps=args.max_steps,
            )
        elif args.layout == "bowl_plate":
            from isaac_bowl_plate_scene import IsaacBowlPlateScene

            scene = IsaacBowlPlateScene(
                obs_height=args.obs_height,
                obs_width=args.obs_width,
                instruction=args.instruction,
                success_key=args.success_key,
                max_steps=args.max_steps,
            )
        else:
            from isaac_scene import IsaacLiftScene

            scene = IsaacLiftScene(
                obs_height=args.obs_height,
                obs_width=args.obs_width,
                instruction=args.instruction,
                success_key=args.success_key,
                max_steps=args.max_steps,
            )
        scene.build()

        # 3) Serve the ZMQ REP loop.
        return _serve(
            scene,
            host=args.host,
            port=args.port,
            sim_app=sim_app,
            task=args.task,
            layout=args.layout,
        )
    finally:
        sim_app.close()


def _serve(scene: Any, *, host: str, port: int, sim_app: Any, task: str, layout: str) -> int:
    import msgpack
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[isaac_sidecar] serving on tcp://{host}:{port}", flush=True)

    running = True
    while running:
        raw = sock.recv()
        req = msgpack.unpackb(raw, object_hook=_decode_ndarray, raw=False)
        endpoint = req.get("endpoint")
        data = req.get("data", {}) or {}
        try:
            if endpoint == "ping":
                # Identity lets the client reject a stale sidecar serving a
                # different scene on this port (SidecarClient.expected_identity).
                reply: dict[str, Any] = {
                    "ok": True,
                    "action_dim": scene.action_dim,
                    "task": task,
                    "layout": layout,
                }
            elif endpoint == "reset":
                # ADR-0048 Phase 2 — carry sim time on reset too (≈0 after the
                # world reset) so the HAL's cross-reset offset stays monotonic.
                reply = {
                    "observation": scene.reset(seed=data.get("seed")),
                    "sim_time_ns": scene.sim_time_ns(),
                }
            elif endpoint == "step":
                reply = scene.step(np.asarray(data["action"], dtype=np.float32))
            elif endpoint == "render":
                reply = {"frame": scene.render()}
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


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
