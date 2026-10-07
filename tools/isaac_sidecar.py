#!/usr/bin/env python
"""Isaac Sim scene sidecar — runs Isaac Lab in its own py3.11 venv.

Isaac side of the backend in ``openral_sim.backends.isaac_sim`` (py3.12),
auto-spawned under the venv named by ``OPENRAL_ISAAC_SIDECAR_PYTHON``. Launches
Omniverse Kit headless, builds the robot-agnostic manifest scene
(``isaac_manifest_scene``: the manifest robot from its URDF, plus an optional
environment USD and objects), and serves ZMQ REP + msgpack/ndarray framing:

    ping->{"ok","action_dim","task","layout","environment","spawn","robot","objects","robot_spec_hash"}
    reset->{"observation"}
    step->{"observation","reward","terminated","truncated","info"}
    render->{"frame": uint8 HWC|None}   close->{"ok"}
    observation = {"images": {"camera1": <H,W,3 uint8>}, "state": 1-D float32, "task": str}

IMPORTANT: construct ``SimulationApp`` before any ``omni.*``/``isaaclab`` import
— heavy imports live inside ``main``, not module scope.

Sets ``OMNI_KIT_ACCEPT_EULA=YES`` (running this accepts the NVIDIA Omniverse
license). Kit is proprietary and never vendored (CLAUDE.md §1.9) — this only
drives an externally-provisioned install.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sys
import traceback
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
    p.add_argument("--task", required=True, help="task id, e.g. isaac_sim/_hal_deploy_noop")
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
        default="manifest",
        choices=["manifest"],
        help="manifest = robot-agnostic URDF-driven scene (needs --robot-spec)",
    )
    p.add_argument(
        "--robot-spec",
        default=None,
        help="path to the JSON isaac robot spec (manifest layout only)",
    )
    p.add_argument(
        "--environment-usd",
        default=None,
        help=(
            "environment USD referenced under /World/environment (manifest layout "
            "only): a local path, an http(s):// or omniverse:// URL, or "
            "isaac:<path> under the installed Isaac Sim asset root"
        ),
    )
    p.add_argument(
        "--spawn-pose",
        type=float,
        nargs=4,
        default=[0.0, 0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z", "YAW"),
        help="robot world placement, metres + radians (manifest layout only)",
    )
    p.add_argument(
        "--objects-json",
        default=None,
        help=(
            "JSON list of extra scene objects (manifest layout only): "
            '[{"usd", "name", "xyz", "yaw", "dynamic"}, ...]'
        ),
    )
    p.add_argument(
        "--site-dir",
        default=None,
        help=(
            "extra import dir prepended to sys.path (a binary Isaac Sim install's "
            "pyzmq/msgpack, installed beside it rather than into it)"
        ),
    )
    p.add_argument(
        "--require-min",
        action="append",
        default=[],
        metavar="DIST>=VERSION",
        help=(
            "installed-version floor to verify before Kit boots (repeatable). "
            "The openral side passes the CUDA runtime pins its venv provisioning "
            "applies, so a venv that predates a pin correction fails in a second "
            "with the actual versions instead of hanging (issue #89)."
        ),
    )
    args = p.parse_args(argv)
    return args


def _check_required_versions(requirements: list[str]) -> None:
    """Fail loudly when an installed dist is older than the floor openral pins.

    A too-old ``nvidia-*`` CUDA wheel doesn't raise — it leaves Kit alive but
    never serving, burning the full ``boot_timeout_s`` silently. Versions
    compare as digit tuples (``12.6.85`` -> ``(12, 6, 85)``) to avoid a
    ``packaging`` dependency the sidecar venv isn't guaranteed to carry.
    """
    from importlib.metadata import PackageNotFoundError, version

    def _tuple(v: str) -> tuple[int, ...]:
        return tuple(int(part) for part in re.findall(r"\d+", v))

    problems: list[str] = []
    for requirement in requirements:
        dist, _, floor = requirement.partition(">=")
        if not floor:
            raise SystemExit(f"--require-min expects 'DIST>=VERSION', got {requirement!r}")
        try:
            installed = version(dist)
        except PackageNotFoundError:
            problems.append(f"{dist}: not installed (need >={floor})")
            continue
        if _tuple(installed) < _tuple(floor):
            problems.append(f"{dist}: {installed} installed, need >={floor}")
    if problems:
        raise SystemExit(
            "[isaac_sidecar] sidecar venv fails its dependency floors:\n  "
            + "\n  ".join(problems)
            + f"\nRepair it:\n  {sys.executable} -m pip install --upgrade --no-deps "
            + " ".join(f"'{r},<13'" for r in requirements)
            + "\nOr delete the venv and re-run with OPENRAL_ISAAC_AUTO_PROVISION=1."
        )


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.site_dir:
        sys.path.insert(0, args.site_dir)
    # Before the ~50 s Kit boot: a stale venv must not cost a full boot timeout.
    _check_required_versions(args.require_min)

    # A shared, root-owned binary install (/opt/isaac-sim) can't take Kit's
    # caches + documents: point Kit at a per-user data root instead (Kit reads
    # --portable-root off sys.argv).
    isaac_root = os.environ.get("ISAAC_PATH")
    if isaac_root and not os.access(os.path.join(isaac_root, "kit"), os.W_OK):
        kit_data = os.path.expanduser("~/.cache/openral/isaac-sidecar/kit-data")
        os.makedirs(kit_data, exist_ok=True)
        sys.argv += ["--portable-root", kit_data]

    # 1) Launch the Kit app FIRST — every omni.* / isaaclab import below depends
    #    on a live SimulationApp.
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": bool(args.headless)})

    try:
        # 2) Heavy imports, only valid post-launch.
        import json

        from isaac_manifest_scene import IsaacManifestScene

        if not args.robot_spec:
            raise SystemExit("--layout manifest requires --robot-spec <path>")
        with open(args.robot_spec, encoding="utf-8") as fh:
            robot_spec_json = fh.read()
        robot_spec = json.loads(robot_spec_json)
        scene = IsaacManifestScene(
            robot_spec=robot_spec,
            environment_usd=args.environment_usd,
            spawn_pose=tuple(args.spawn_pose),
            objects=json.loads(args.objects_json) if args.objects_json else None,
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
            environment=args.environment_usd or "",
            spawn=list(args.spawn_pose),
            robot=args.robot,
            objects=args.objects_json or "",
            robot_spec_hash=hashlib.sha256(robot_spec_json.encode("utf-8")).hexdigest(),
        )
    except BaseException:
        # Print before close(): Kit's fast shutdown ends the process inside
        # close(), so an exception left to propagate is never printed — the
        # sidecar would just vanish with exit code 0 mid-boot.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        sim_app.close()


def _serve(
    scene: Any,
    *,
    host: str,
    port: int,
    sim_app: Any,
    task: str,
    layout: str,
    environment: str,
    spawn: list[float],
    robot: str,
    objects: str,
    robot_spec_hash: str,
) -> int:
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
                    "environment": environment,
                    "spawn": spawn,
                    "robot": robot,
                    "objects": objects,
                    "robot_spec_hash": robot_spec_hash,
                }
            elif endpoint == "reset":
                # Carry sim time on reset too (≈0 after the
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
