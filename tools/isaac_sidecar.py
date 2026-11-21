#!/usr/bin/env python
"""Isaac Sim scene sidecar — runs Isaac Lab in its own py3.11 venv.

Isaac side of the backend in ``openral_sim.backends.isaac_sim`` (py3.12),
auto-spawned under the venv named by ``OPENRAL_ISAAC_SIDECAR_PYTHON``. Launches
Omniverse Kit headless, builds the robot-agnostic manifest scene
(``isaac_manifest_scene``: the manifest robot from its URDF, plus an optional
environment USD and objects), and serves msgpack/ndarray framing on a ZMQ ROUTER socket
(clients are REQ sockets; ``step`` replies wait on the lockstep, ``_Lockstep``):

    ping->{"ok","action_dim","task","layout","environment","spawn","robot","objects","robot_spec_hash","launch_id"}
        (+ "robots" when serving several)
    reset->{"observation"}
    step{action, frames?, robot?, idle?}->{"observation","reward","terminated","truncated","info","renders"}
        frames: "render" (default) | "fresh" | "none" (no frame keys in the observation)
    render->{"frame": uint8 HWC|None}   close->{"ok"}
    contacts->PhysX robot-vs-world contacts, last physics step (scene.contact_ground_truth)
    link_poses{links}->{"sim_time_ns","dofs","links"}: every DOF's position + drive target,
        the named links' base-frame poses as PhysX holds them (scene.link_poses)
    footprint_overlaps->{"overlaps": [[a, b]], "events": [{"robots", "sim_time_ns"}]}
    observation = {"images": {"camera1": <H,W,3 uint8>}, "state": 1-D float32, "task": str}

Several robots in one stage (``--robots``, openral-pro#99): every request about one robot
names it in its ``robot`` field (``""`` when absent: the one robot of a ``--robot-spec``
sidecar), and the lockstep's released steps advance the world once (``scene.step_many``).

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
import json
import os
import re
import sys
import time
import traceback
from typing import Any

import numpy as np

# Accept the NVIDIA Omniverse EULA non-interactively. Without this the Kit
# bootstrap blocks on a stdin "Do you accept the EULA?" prompt and dies on EOF
# when spawned with a closed stdin.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


# ── msgpack ndarray codec (matches openral_sim.backends.isaac_sim) ────────────


class _StepTiming:
    """Mean per-step wall time (ms) by frame mode, printed every 100 steps of a mode
    (``OPENRAL_SIDECAR_TIMING=1``): physics, catch-up renders, observation assembly,
    reply packing, send, whole handler, reply size."""

    def __init__(self) -> None:
        self._sums: dict[str, dict[str, float]] = {}
        self._n: dict[str, int] = {}
        self.sized = False

    def add(self, mode: str, parts: dict[str, float]) -> None:
        sums = self._sums.setdefault(mode, {})
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        self._n[mode] = self._n.get(mode, 0) + 1
        if self._n[mode] % 100 == 0:
            n = 100.0
            line = " ".join(f"{k}={v / n:.1f}" for k, v in sums.items())
            print(f"[sidecar_timing] frames={mode} n={self._n[mode]} mean_ms: {line}", flush=True)
            sums.clear()


def _nbytes_tree(obj: Any, prefix: str = "") -> dict[str, float]:
    """MB of every ndarray in a reply, by key path (depth_frames/head_zed/rgb ...)."""
    if isinstance(obj, np.ndarray):
        return {prefix: obj.nbytes / 1e6}
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_nbytes_tree(v, f"{prefix}/{k}"))
    return out


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
    p.add_argument("--robot", default=None, help="robot id (default franka_panda)")
    p.add_argument("--instruction", default="")
    p.add_argument("--obs-height", type=int, default=256)
    p.add_argument("--obs-width", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--success-key", default="is_success")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5757)
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--launch-id",
        default="",
        help="opaque launch identity echoed by ping (openral_sim isaac_sim._launch_id)",
    )
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
        "--robots",
        default=None,
        help=(
            "several robots in one stage, a JSON list "
            '[{"name", "robot_spec": <path>, "spawn_pose": [x, y, z, yaw]}, ...]; '
            "replaces --robot / --robot-spec / --spawn-pose"
        ),
    )
    p.add_argument(
        "--spawn-pose",
        type=float,
        nargs=4,
        default=None,
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
    if args.robots is not None:
        given = [f for f in ("robot", "robot_spec", "spawn_pose") if getattr(args, f) is not None]
        if given:
            p.error(f"--robots replaces {', '.join('--' + f.replace('_', '-') for f in given)}")
    if args.robot is None:
        args.robot = "franka_panda"
    if args.spawn_pose is None:
        args.spawn_pose = [0.0, 0.0, 0.0, 0.0]
    return args


#: A ``--robots`` name: it names the robot's prim, graph and fleet entry.
_ROBOT_NAME = re.compile(r"[a-z0-9]+")


def _parse_robots(text: str) -> list[dict[str, Any]]:
    """``--robots``: ``[{name, robot_spec, spawn_pose?}]`` read into name, spec, spawn, spec hash.

    Read before Kit boots, so a bad list fails in a second. Raises ``SystemExit`` on a
    malformed list, a bad or duplicate name, or a spawn that is not ``[x, y, z, yaw]``.

    Example:
        >>> _parse_robots('[{"name": "South", "robot_spec": "s.json"}]')
        Traceback (most recent call last):
        ...
        SystemExit: --robots: robot name 'South' must be lowercase letters and digits
    """
    entries = json.loads(text)
    if not isinstance(entries, list) or not entries:
        raise SystemExit("--robots expects a non-empty JSON list of {name, robot_spec, spawn_pose}")
    out: list[dict[str, Any]] = []
    for entry in entries:
        name = str(entry.get("name", ""))
        if not _ROBOT_NAME.fullmatch(name):
            raise SystemExit(f"--robots: robot name {name!r} must be lowercase letters and digits")
        spawn = [float(v) for v in entry.get("spawn_pose", [0.0, 0.0, 0.0, 0.0])]
        if len(spawn) != 4:
            raise SystemExit(f"--robots: robot {name!r} spawn_pose must be [x, y, z, yaw]")
        with open(entry["robot_spec"], encoding="utf-8") as fh:
            spec_json = fh.read()
        out.append(
            {
                "name": name,
                "robot_spec": json.loads(spec_json),
                "spawn_pose": spawn,
                "robot_spec_hash": hashlib.sha256(spec_json.encode("utf-8")).hexdigest(),
            }
        )
    names = [r["name"] for r in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"--robots: duplicate robot names in {names}")
    return out


def _robots_identity(robots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What ``ping`` reports of each robot: name, spec hash and spawn (``SidecarClient`` identity)."""
    return [{k: r[k] for k in ("name", "robot_spec_hash", "spawn_pose")} for r in robots]


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


def _zero_delay_experience(isaac_root: str | None) -> str | None:
    """Isaac's zero-delay Kit experience file, when the install ships one.

    It runs the RTX pipeline without the default frames in flight, so a camera frame
    trails the physics by ~2 renders instead of ~4 (measured on Isaac 6.1): a
    ``frames="fresh"`` step then needs about half the catch-up renders, which are
    most of its cost in a deploy scene. ``None`` keeps Kit's default experience.

    Example:
        >>> _zero_delay_experience("/nonexistent") is None
        True
    """
    candidates = []
    if isaac_root:
        candidates.append(os.path.join(isaac_root, "apps", "isaacsim.exp.base.zero_delay.kit"))
    try:
        import isaacsim  # pip install: the apps live beside the package

        package_file = getattr(isaacsim, "__file__", None)  # None for a namespace package
        if package_file:
            candidates.append(
                os.path.join(
                    os.path.dirname(package_file), "apps", "isaacsim.exp.base.zero_delay.kit"
                )
            )
    except ImportError:
        pass
    return next((c for c in candidates if os.path.isfile(c)), None)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.site_dir:
        sys.path.insert(0, args.site_dir)
    # Before the ~50 s Kit boot: a stale venv must not cost a full boot timeout.
    _check_required_versions(args.require_min)
    robots = _parse_robots(args.robots) if args.robots is not None else None

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

    experience = _zero_delay_experience(isaac_root)
    print(f"[isaac_sidecar] Kit experience: {experience or 'default'}", flush=True)
    sim_app = (
        SimulationApp({"headless": bool(args.headless)}, experience=experience)
        if experience
        else SimulationApp({"headless": bool(args.headless)})
    )
    if args.headless:
        # Kit keeps rendering its 1280x720 editor viewport every frame even headless;
        # nothing reads it and it costs ~20 % of a scene step (26 -> 20.5 ms measured
        # on GB10). The scene's own cameras render through their render products.
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.updates_enabled = False

    try:
        # 2) Heavy imports, only valid post-launch.
        from isaac_manifest_scene import IsaacManifestScene

        if robots is None:
            if not args.robot_spec:
                raise SystemExit("--layout manifest requires --robot-spec <path>")
            with open(args.robot_spec, encoding="utf-8") as fh:
                robot_spec_json = fh.read()
            one = {
                "name": "",
                "robot_spec": json.loads(robot_spec_json),
                "spawn_pose": list(args.spawn_pose),
                "robot_spec_hash": hashlib.sha256(robot_spec_json.encode("utf-8")).hexdigest(),
            }
        scenes = [
            IsaacManifestScene(
                robot_spec=r["robot_spec"],
                name=r["name"],
                environment_usd=args.environment_usd,
                spawn_pose=tuple(r["spawn_pose"]),
                objects=json.loads(args.objects_json) if args.objects_json else None,
                obs_height=args.obs_height,
                obs_width=args.obs_width,
                instruction=args.instruction,
                success_key=args.success_key,
                max_steps=args.max_steps,
            )
            for r in (robots or [one])
        ]
        IsaacManifestScene.build_all(scenes)

        # 3) Serve the ZMQ ROUTER loop (``_serve``).
        identity = None if robots is None else _robots_identity(robots)
        return _serve(
            scenes[0],
            host=args.host,
            port=args.port,
            sim_app=sim_app,
            task=args.task,
            layout=args.layout,
            environment=args.environment_usd or "",
            # Several robots: their spawns and specs are in `robots`, hashed together.
            spawn=list(args.spawn_pose) if robots is None else [],
            robot=args.robot if robots is None else "",
            objects=args.objects_json or "",
            robot_spec_hash=(
                one["robot_spec_hash"]
                if robots is None
                else hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            ),
            launch_id=args.launch_id,
            robots=identity,
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


#: The largest Sim scene GLB the ``scene_mesh`` endpoint sends (platform#133).
_SCENE_MESH_MAX_BYTES = 10_000_000


#: A commanding robot that sends no step for this long (wall time) is dropped from the lockstep:
#: the world goes on without it (openral-pro#99).
_BARRIER_TIMEOUT_S = 5.0


class _Lockstep:
    """Which robots' ``step`` requests to answer together (openral-pro#99).

    One action is one control period for every robot of the scene, so the world advances
    once **every commanding robot** has a step waiting; then every waiting step is answered.
    A robot is commanding from a step until it sends one marked ``idle`` (its HAL's hold
    stepper), and an idle robot never holds the world back. A commanding robot silent for
    ``timeout_s`` is dropped (``drop_late``). With one robot every step is answered at once,
    as before the lockstep existed. Robots are told apart by the ``robot`` field of a
    request (``""`` when absent, as every client before #99 sends).

    Example:
        >>> lock = _Lockstep(timeout_s=5.0)
        >>> lock.submit("a", b"ra", {"action": 1}, now=0.0)
        >>> [r for r, *_ in lock.release()]  # alone: answered at once
        ['a']
        >>> lock.submit("b", b"rb", {"action": 2}, now=0.1)  # b commands too: it waits for a
        >>> lock.release()
        []
        >>> lock.submit("a", b"ra", {"action": 3}, now=0.2)
        >>> [r for r, *_ in lock.release()]  # one period for both
        ['b', 'a']
        >>> lock.submit("b", b"rb", {"action": 4}, now=0.3)  # and a goes silent
        >>> lock.drop_late(now=5.3), [r for r, *_ in lock.release()]
        (['a'], ['b'])
    """

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self.waiting: dict[str, tuple[bytes, dict[str, Any]]] = {}  # robot -> (route, data)
        self.commanding: dict[str, float] = {}  # robot -> when its last step came

    def submit(self, robot: str, route: bytes, data: dict[str, Any], *, now: float) -> None:
        """Hold ``robot``'s step until ``release`` says the world may advance."""
        if robot in self.waiting:
            raise ValueError(f"robot {robot!r} already has a step waiting")
        if data.get("idle"):
            self.commanding.pop(robot, None)
        else:
            self.commanding[robot] = now
        self.waiting[robot] = (route, data)

    def release(self) -> list[tuple[str, bytes, dict[str, Any]]]:
        """The steps to answer now, in arrival order: all waiting, once every commanding robot is."""
        if not self.waiting or any(r not in self.waiting for r in self.commanding):
            return []
        out = [(r, route, data) for r, (route, data) in self.waiting.items()]
        self.waiting.clear()
        return out

    def drop_late(self, *, now: float) -> list[str]:
        """Drop the commanding robots silent past the timeout while another robot waits."""
        if not self.waiting:
            return []
        late = [
            r
            for r, t in self.commanding.items()
            if r not in self.waiting and now - t > self.timeout_s
        ]
        for r in late:
            del self.commanding[r]
        return late

    def wait_s(self, *, now: float) -> float | None:
        """How long the server may block before ``drop_late`` has work, or ``None``."""
        if not self.waiting:
            return None
        silent = [t for r, t in self.commanding.items() if r not in self.waiting]
        return max(0.0, min(silent) + self.timeout_s - now) if silent else None


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
    launch_id: str = "",
    robots: list[dict[str, Any]] | None = None,
) -> int:
    """Serve ``scene`` (the world's first robot; ``scene.robot(name)`` reaches each) until ``close``.

    ``robots``: the robot list's identity (``_robots_identity``) when serving several, which
    ``ping`` reports. A ``reset`` resets the whole world, unless another robot stepped in the
    last ``_BARRIER_TIMEOUT_S``: then that robot is running, and the asking robot gets its
    observation of the running world instead (its graph joined late, or restarted).
    """
    import msgpack
    import zmq

    ctx = zmq.Context()
    # ROUTER, not REP: a robot's step can wait for the others' (``_Lockstep``) while other
    # requests are answered. The clients stay REQ sockets: [route, b"", payload] frames.
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[isaac_sidecar] serving on tcp://{host}:{port}", flush=True)

    lock = _Lockstep(_BARRIER_TIMEOUT_S)
    timing = _StepTiming() if os.environ.get("OPENRAL_SIDECAR_TIMING") == "1" else None
    last_step: dict[str, float] = {}  # robot -> when its last step was answered (the reset rule)
    overlaps: list[list[str]] = []  # robot pairs whose base footprints overlap now
    overlap_events: list[dict[str, Any]] = []  # each overlap's start since the last world reset

    def send(route: bytes, reply: dict[str, Any]) -> float:
        payload = msgpack.packb(reply, default=_encode_ndarray, use_bin_type=True)
        sock.send_multipart([route, b"", payload])
        return len(payload) / 1e6

    def answer_steps() -> None:
        released = lock.release()
        if not released:
            return
        t_req = time.perf_counter()
        try:
            # One world advance for every released robot (openral-pro#99).
            replies = scene.step_many(
                {
                    name: (
                        np.asarray(data["action"], dtype=np.float32),
                        str(data.get("frames", "render")),
                    )
                    for name, _route, data in released
                }
            )
        except Exception as exc:
            replies = {name: {"error": f"{type(exc).__name__}: {exc}"} for name, *_ in released}
        now = time.monotonic()
        for name, *_ in released:
            last_step[name] = now
        if robots is not None:
            try:
                check_overlaps()
            except Exception as exc:  # a diagnostic: never fail the steps over it
                print(f"[isaac_sidecar] footprint overlap check failed: {exc!r}", flush=True)
        t_pack = time.perf_counter()
        for name, route, data in released:
            mb = send(route, replies[name])
            if timing is not None:
                _time_step(timing, scene.robot(name), data, replies[name], t_req, t_pack, mb)

    def check_overlaps() -> None:
        """Log each new base-footprint overlap between robots (ground truth, never a robot's)."""
        now = scene.footprint_overlaps()
        for pair in now:
            if pair not in overlaps:
                overlap_events.append({"robots": pair, "sim_time_ns": scene.sim_time_ns()})
                print(f"[isaac_sidecar] robots {pair} base footprints overlap", flush=True)
        overlaps[:] = now

    running = True
    while running:
        wait = lock.wait_s(now=time.monotonic())
        if not sock.poll(None if wait is None else max(1, int(wait * 1000))):
            for late in lock.drop_late(now=time.monotonic()):
                print(
                    f"[isaac_sidecar] robot {late!r} sent no step for "
                    f"{_BARRIER_TIMEOUT_S:g} s: dropped from the lockstep",
                    flush=True,
                )
            answer_steps()
            continue
        route, _, raw = sock.recv_multipart()
        req = msgpack.unpackb(raw, object_hook=_decode_ndarray, raw=False)
        endpoint = req.get("endpoint")
        data = req.get("data", {}) or {}
        name = str(data.get("robot", ""))
        if endpoint == "step":
            try:
                scene.robot(name)
                lock.submit(name, route, data, now=time.monotonic())
            except (KeyError, ValueError) as exc:
                send(route, {"error": str(exc.args[0])})
            answer_steps()
            continue
        try:
            if endpoint == "ping":
                # Identity lets the client reject a stale sidecar serving a
                # different scene on this port (SidecarClient.expected_identity).
                try:
                    action_dim = scene.robot(name).action_dim
                except KeyError:  # a client of another robot set: the identity tells it so
                    action_dim = None
                reply: dict[str, Any] = {
                    "ok": action_dim is not None,
                    "action_dim": action_dim,
                    # Configured nominal seconds per step; reset later replaces
                    # this with the observed callback cadence before HAL checks it.
                    "sim_dt_per_tick_s": scene.sim_dt_per_tick_s,
                    "task": task,
                    "layout": layout,
                    "environment": environment,
                    "spawn": spawn,
                    "robot": robot,
                    "objects": objects,
                    "robot_spec_hash": robot_spec_hash,
                    "launch_id": launch_id,
                }
                if robots is not None:
                    reply["robots"] = robots
            elif endpoint == "reset":
                asking = scene.robot(name)
                t = time.monotonic()
                busy = sorted(
                    r for r, at in last_step.items() if r != name and t - at < _BARRIER_TIMEOUT_S
                )
                if busy:
                    print(
                        f"[isaac_sidecar] reset from robot {name!r} while {busy} run: "
                        "the world is not reset, it gets the running world's observation",
                        flush=True,
                    )
                    observation = asking.refresh_frames()
                else:
                    observation = asking.reset(seed=data.get("seed"))
                    overlaps.clear()
                    overlap_events.clear()
                # Carry sim time on reset too (≈0 after the
                # world reset) so the HAL's cross-reset offset stays monotonic.
                reply = {
                    "observation": observation,
                    "sim_time_ns": asking.sim_time_ns(),
                    # Same clock as World.current_time stamps; ping remains the
                    # configured nominal period used to build physics substeps.
                    "sim_dt_per_tick_s": asking._observed_sim_dt_per_tick_s,
                }
            elif endpoint == "refresh_frames":
                reply = {"observation": scene.robot(name).refresh_frames()}
            elif endpoint == "render":
                reply = {"frame": scene.robot(name).render()}
            elif endpoint == "contacts":
                reply = scene.robot(name).contact_ground_truth()
            elif endpoint == "static_colliders":
                reply = scene.robot(name).static_collider_boxes()
            elif endpoint == "scene_mesh":
                # The client may ask for less, never more: the GLB rides the same
                # connection as the steps.
                limit = min(int(data["limit_bytes"]), _SCENE_MESH_MAX_BYTES)
                reply = scene.robot(name).scene_mesh(limit)
            elif endpoint == "scene_ground_truth":
                reply = scene.robot(name).scene_ground_truth()
            elif endpoint == "link_poses":
                reply = scene.robot(name).link_poses([str(n) for n in data.get("links", [])])
            elif endpoint == "set_task":
                reply = scene.robot(name).set_task(str(data.get("instruction", "")))
            elif endpoint == "footprint_overlaps":
                reply = {"overlaps": overlaps, "events": overlap_events}
            elif endpoint == "set_object_position":
                reply = scene.set_object_position(str(data["name"]), list(data["xyz"]))
            elif endpoint == "close":
                reply = {"ok": True}
                running = False
            else:
                reply = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as exc:
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        send(route, reply)

    sock.close(linger=0)
    ctx.term()
    return 0


def _time_step(
    timing: _StepTiming,
    scene: Any,
    data: dict[str, Any],
    reply: dict[str, Any],
    t_req: float,
    t_pack: float,
    mb: float,
) -> None:
    """``OPENRAL_SIDECAR_TIMING=1``: one step's handler, pack+send time and reply size."""
    if data.get("frames") == "fresh" and not timing.sized:
        timing.sized = True
        sizes = _nbytes_tree(reply)
        print(
            "[sidecar_timing] reply MB by array: "
            + ", ".join(
                f"{k}={v:.2f}" for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:10]
            ),
            flush=True,
        )
    timing.add(
        str(data.get("frames", "render")),
        {
            **getattr(scene, "last_step_timing", {}),
            "pack_send": (time.perf_counter() - t_pack) * 1e3,
            "bytes": mb,
            "handler": (t_pack - t_req) * 1e3,
        },
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
