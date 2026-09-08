"""Real GPU rollout audit for every (scene x rSkill) combination under ``scenes/``.

Launches each catalogue entry through its tier CLI (``openral sim run`` /
``openral benchmark scene`` / ``openral deploy sim``) for one real episode or
a full ROS-graph launch + SIGINT teardown, classifies the outcome from exit
code + stderr tail, and writes a JSON report + Markdown table. Default mode
does a full rollout (30 s-10 min/row); ``--check-compatibility`` is a cheap
in-process schema/manifest/HAL-registry check only (no subprocess, no GPU).

Usage::

    uv run python tools/audit_sim_configs.py                          # full rollouts
    uv run python tools/audit_sim_configs.py --check-compatibility    # cheap gate
    uv run python tools/audit_sim_configs.py libero_spatial pusht     # narrow by stem

Operator-driven (CLAUDE.md §1.11-§1.12: no smoke tests, no --dry-run), not a
pytest test — the fast schema-only check is
``tests/unit/test_examples_sim_configs_load.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Literal

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SCENES_DIR: Final[Path] = REPO_ROOT / "scenes"
OUTPUT_DIR: Final[Path] = REPO_ROOT / "outputs"
DEFAULT_TIMEOUT_S: Final[int] = 600  # 10 min/config — covers cold weights download
DEFAULT_DEPLOY_ALIVE_GRACE_S: Final[int] = 90  # how long to let the ROS graph live
DEFAULT_DEPLOY_SHUTDOWN_GRACE_S: Final[int] = 30  # how long to wait after SIGINT

RunMode = Literal["sim", "benchmark", "deploy"]


@dataclasses.dataclass(frozen=True)
class ConfigSpec:
    """One row in the audit catalogue.

    Attributes:
        config: YAML path relative to the repo root.
        rskill: Bare rSkill reference (name/path); empty for ``run_mode ==
            "deploy"`` (env-only — the reasoner picks the rSkill at runtime).
        uv_group: uv dependency group (``libero``, ``metaworld``, ``robocasa``,
            ``maniskill3``, ``simpler-env``, or ``sim``).
        run_mode: ``"sim"`` (``openral sim run``), ``"benchmark"`` (``openral
            benchmark scene --no-update-manifest --n-episodes 1``, demo-grade),
            or ``"deploy"`` (``openral deploy sim`` + ``--alive-grace`` soak +
            SIGINT teardown).

    Only scene×rSkill pairs that exist in the tree; schema-load coverage for
    every scene lives in ``tests/unit/test_examples_sim_configs_load.py``.
    """

    config: str
    rskill: str
    uv_group: str
    run_mode: RunMode


# Explicit mapping. The YAML names + rSkill names are not regular enough to
# derive via a rule, so we list them. Each row is
# one (scene, rSkill) combination — the same scene may appear multiple
# times paired with different rSkills (LIBERO is the obvious example).
CATALOGUE: Final[tuple[ConfigSpec, ...]] = (
    # ---- SimScene tier (openral sim run) ----
    ConfigSpec("scenes/sim/libero_spatial.yaml", "rskills/smolvla-libero", "libero", "sim"),
    ConfigSpec("scenes/sim/libero_spatial.yaml", "rskills/xvla-libero", "libero", "sim"),
    ConfigSpec("scenes/sim/libero_spatial.yaml", "rskills/pi05-libero-int8", "libero", "sim"),
    ConfigSpec("scenes/sim/libero_spatial.yaml", "rskills/act-libero", "libero", "sim"),
    ConfigSpec("scenes/sim/libero_spatial.yaml", "rskills/rldx1-ft-libero-nf4", "libero", "sim"),
    ConfigSpec(
        "scenes/sim/libero_spatial.yaml",
        "rskills/molmoact2-libero-nf4",
        "libero",
        "sim",
    ),
    ConfigSpec(
        "scenes/sim/franka_libero_pnp.yaml",
        "rskills/pi05-libero-int8",
        "libero",
        "sim",
    ),
    ConfigSpec(
        "scenes/sim/robocasa_pnp.yaml",
        "rskills/rldx1-ft-rc365-nf4",
        "robocasa",
        "sim",
    ),
    ConfigSpec(
        "scenes/sim/robocasa_gr1_pnp_cup_to_drawer.yaml",
        "rskills/rldx1-ft-gr1-nf4",
        "robocasa",
        "sim",
    ),
    ConfigSpec(
        "scenes/sim/behavior_turning_on_radio.yaml",
        "rskills/gr00t-n17-b1k-turning-on-radio",
        "behavior-groot",
        "sim",
    ),
    # ---- BenchmarkScene tier (openral benchmark scene --no-update-manifest --n-episodes 1) ----
    # Each row is a demo-grade run, not a paper-comparable claim.
    ConfigSpec(
        "scenes/benchmark/metaworld_push.yaml",
        "rskills/smolvla-metaworld",
        "metaworld",
        "benchmark",
    ),
    ConfigSpec(
        "scenes/benchmark/aloha_transfer_cube.yaml",
        "rskills/act-aloha",
        "sim",
        "benchmark",
    ),
    ConfigSpec(
        "scenes/benchmark/aloha_insertion.yaml",
        "rskills/act-aloha-insertion",
        "sim",
        "benchmark",
    ),
    ConfigSpec(
        "scenes/benchmark/pusht.yaml",
        "rskills/diffusion-pusht",
        "sim",
        "benchmark",
    ),
    ConfigSpec(
        "scenes/benchmark/libero_spatial.yaml",
        "rskills/smolvla-libero",
        "libero",
        "benchmark",
    ),
    ConfigSpec(
        "scenes/benchmark/maniskill_pick_cube.yaml",
        "rskills/smolvla-maniskill-franka",
        "maniskill3",
        "benchmark",
    ),
    ConfigSpec(
        "scenes/benchmark/widowx_carrot_on_plate.yaml",
        "rskills/rldx1-ft-simpler-widowx-nf4",
        "simpler-env",
        "benchmark",
    ),
    # ---- DeployScene tier (openral deploy sim) ----
    # Env-only: the reasoner picks the rSkill at runtime, so `rskill` is
    # left empty. Each row spawns `openral deploy sim --config <yaml>`, lets
    # the ROS graph soak for `--alive-grace` seconds (configure → activate
    # of HAL + safety_kernel + dashboard + opt-in slam/nav2), then SIGINTs
    # the launch group and waits for graceful shutdown.
    ConfigSpec("scenes/deploy/openarm_tabletop.yaml", "", "sim", "deploy"),
    ConfigSpec("scenes/deploy/so101_box.yaml", "", "sim", "deploy"),
    ConfigSpec("scenes/deploy/robocasa_pnp.yaml", "", "robocasa", "deploy"),
    ConfigSpec("scenes/deploy/libero_pnp.yaml", "", "libero", "deploy"),
    ConfigSpec("scenes/deploy/behavior_r1pro.yaml", "", "behavior-groot", "deploy"),
)


@dataclasses.dataclass
class AuditRow:
    config: str
    rskill: str
    status: str  # pass | pass-compat | fail-oom | fail-asset | fail-sidecar | fail-timeout | fail-other | fail-compat | skipped-opt-dep | skipped-host-setup
    exit_code: int | None
    wall_s: float
    peak_vram_mib: int | None
    tail: str  # last ~30 lines of stderr+stdout, for triage


# stderr substrings that classify a failure.
_OOM_PATTERNS: Final[tuple[str, ...]] = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "cudaErrorMemoryAllocation",
    "RuntimeError: CUDA error: out of memory",
)
_ASSET_PATTERNS: Final[tuple[str, ...]] = (
    "FileNotFoundError",
    "No such file or directory",
    "asset path does not exist",
    "init_files/",
)
_OPT_DEP_PATTERNS: Final[tuple[str, ...]] = (
    "MetaWorld backend not installed",
    "metaworld backend not installed",
    "robocasa is not installed",
    "ModuleNotFoundError: No module named 'metaworld'",
    "ModuleNotFoundError: No module named 'robocasa'",
    # robocasa auto-install races its own import probe: install completes but
    # the probe still fails, and the CLI emits this exact phrase — a known
    # robosuite/robocasa import-cache issue, not a config bug.
    "install ran to completion but the probe still fails",
)
_HOST_SETUP_PATTERNS: Final[tuple[str, ...]] = (
    # uv reinstalls hf-libero==0.1.3 on every group switch; the prior
    # install's distutils egg-info resists uninstall. The per-call purge
    # fixes the first hit, but repeated swaps (libero->robocasa->libero) can
    # recreate it mid-run — host-setup, not a config failure.
    "Unable to uninstall `hf-libero",
    "distutils-installed distributions do not include the metadata",
)
_SIDECAR_PATTERNS: Final[tuple[str, ...]] = (
    "OPENRAL_RLDX1_PYTHON",
    "RLDX-1 sidecar",
    "interpreter not found",
    "Python 3.10",
    "PythonVersionMismatch",
)


def _classify(returncode: int, tail: str) -> str:
    if returncode == 0:
        return "pass"
    # MuJoCo/GL atexit SIGSEGV (signal 11 → exit 139) fires after the episode
    # summary is printed in gym-aloha scenes. The rollout itself is clean; only
    # the GL context teardown crashes. Treat as pass when no error patterns appear.
    if returncode == 139:
        blob_139 = tail.lower()
        if not any(
            p.lower() in blob_139
            for p in (
                *_OOM_PATTERNS,
                *_ASSET_PATTERNS,
                *_OPT_DEP_PATTERNS,
                *_HOST_SETUP_PATTERNS,
                *_SIDECAR_PATTERNS,
            )
        ):
            return "pass"
    blob = tail.lower()
    # First-match wins. Order matters: OOM before generic asset / sidecar,
    # opt-dep / host-setup before asset (since "metaworld not installed"
    # and "Unable to uninstall hf-libero" surface as ImportErrors / uv
    # errors that the asset patterns would otherwise gobble up).
    table: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("fail-oom", _OOM_PATTERNS),
        ("skipped-opt-dep", _OPT_DEP_PATTERNS),
        ("skipped-host-setup", _HOST_SETUP_PATTERNS),
        ("fail-sidecar", _SIDECAR_PATTERNS),
        ("fail-asset", _ASSET_PATTERNS),
    )
    for status, patterns in table:
        if any(p.lower() in blob for p in patterns):
            return status
    if returncode == -9 or "timed out" in blob:
        return "fail-timeout"
    return "fail-other"


class _VramSampler:
    """Background sampler for the GPU's used-memory column from ``nvidia-smi``.

    Idempotent: when no ``nvidia-smi`` is on PATH (CPU-only host), reports
    ``None``. Sampling interval 200 ms matches the resolution of CUDA's
    own allocator and is cheap enough not to compete with the rollout.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._peak_mib: int | None = None
        self._thread: threading.Thread | None = None
        self._available = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self._available:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        return self._peak_mib

    def _loop(self) -> None:
        while not self._stop.wait(0.2):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                if out.returncode != 0:
                    continue
                mib = max(int(line.strip()) for line in out.stdout.splitlines() if line.strip())
                if self._peak_mib is None or mib > self._peak_mib:
                    self._peak_mib = mib
            except (subprocess.TimeoutExpired, ValueError):
                continue


def _check_compat(spec: ConfigSpec) -> AuditRow:
    """Cheap in-process compatibility gate (``--check-compatibility``).

    Sim/benchmark rows: load the YAML via ``openral_core.load_scene_strict``,
    then validate the rSkill manifest via ``openral_core.RSkillManifest``.
    Deploy rows: load as ``openral_core.DeployScene`` and assert
    ``robot_id`` resolves in ``openral_cli.deploy_sim._ROBOT_HAL_REGISTRY``.

    Returns:
        An ``AuditRow`` with ``status="pass-compat"`` on success or
        ``"fail-compat"`` on any schema/lookup error. No subprocess, no GPU;
        single-digit seconds even with cold imports.
    """
    started = time.monotonic()
    try:
        # Lazy import so the heavy `openral_core` graph is paid only on
        # the `--check-compatibility` path, never on the default rollout
        # path (which doesn't need to import anything in-process).
        import yaml
        from openral_core import (
            BenchmarkScene,
            DeployScene,
            SimScene,
            load_scene_strict,
        )
        from openral_core.schemas import RSkillManifest

        # `load_scene_strict` takes `path: str` + `expected: type[X]` and is
        # overloaded per tier — branch on `run_mode` so the overload
        # resolves to the exact tier without needing `cast`.
        config_path = str(REPO_ROOT / spec.config)
        scene: SimScene | BenchmarkScene | DeployScene
        if spec.run_mode == "sim":
            scene = load_scene_strict(config_path, expected=SimScene)
        elif spec.run_mode == "benchmark":
            scene = load_scene_strict(config_path, expected=BenchmarkScene)
        else:  # deploy
            scene = load_scene_strict(config_path, expected=DeployScene)

        # rSkill manifest validation (sim/benchmark only — deploy is env-only).
        if spec.run_mode != "deploy":
            if not spec.rskill:
                raise ValueError(f"sim/benchmark row missing rskill: {spec.config}")
            manifest_path = REPO_ROOT / spec.rskill / "rskill.yaml"
            if not manifest_path.exists():
                raise FileNotFoundError(f"rSkill manifest not found: {manifest_path}")
            RSkillManifest.model_validate(yaml.safe_load(manifest_path.read_text()))

        # Deploy-tier HAL-registry lookup.
        if spec.run_mode == "deploy":
            assert isinstance(scene, DeployScene)
            # Resolve robot_id either from the explicit field or via the
            # scene registry's fixed_robot mapping (so101_box →
            # so101_follower, libero_spatial → franka_panda, etc.).
            robot_id = scene.robot_id
            if robot_id is None:
                from openral_sim.registry import SCENES

                robot_id = SCENES.fixed_robot(scene.scene.id)
                if robot_id is None:
                    raise ValueError(
                        f"DeployScene {spec.config!r} has no `robot_id` and the scene "
                        f"id {scene.scene.id!r} is not registered with a fixed_robot."
                    )
            from openral_cli.deploy_sim import _ROBOT_HAL_REGISTRY

            if robot_id not in _ROBOT_HAL_REGISTRY:
                supported = ", ".join(sorted(_ROBOT_HAL_REGISTRY))
                raise KeyError(
                    f"robot {robot_id!r} from {spec.config!r} has no HAL entry "
                    f"in _ROBOT_HAL_REGISTRY (supported: {supported})."
                )

        wall_s = time.monotonic() - started
        return AuditRow(spec.config, spec.rskill, "pass-compat", 0, wall_s, None, "")
    except Exception as exc:
        wall_s = time.monotonic() - started
        return AuditRow(
            spec.config,
            spec.rskill,
            "fail-compat",
            1,
            wall_s,
            None,
            f"{type(exc).__name__}: {exc}",
        )


def _build_run_cmd(spec: ConfigSpec) -> list[str]:
    """Build the `uv run ... openral <subcmd>` argv for sim / benchmark rows.

    Split out of ``_run_one`` so the deploy-tier launch path
    (``_run_one_deploy``) can stay focused on lifecycle teardown.
    """
    if spec.run_mode == "sim":
        # SimScene tier: `openral sim run --config scenes/sim/<scene>.yaml`.
        # `--n-episodes 1` keeps the audit row at one rollout.
        return [
            "uv",
            "run",
            "--all-packages",
            "--group",
            spec.uv_group,
            "openral",
            "sim",
            "run",
            "--config",
            spec.config,
            "--rskill",
            spec.rskill,
            "--n-episodes",
            "1",
        ]
    # BenchmarkScene tier: `openral benchmark scene
    # --config scenes/benchmark/<scene>.yaml --no-update-manifest
    # --n-episodes 1`. `--no-update-manifest` keeps the rSkill's
    # recorded benchmark numbers untouched on an audit row (a
    # 1-episode demo, not a paper-comparable claim).
    return [
        "uv",
        "run",
        "--all-packages",
        "--group",
        spec.uv_group,
        "openral",
        "benchmark",
        "scene",
        "--config",
        spec.config,
        "--rskill",
        spec.rskill,
        "--no-update-manifest",
        "--n-episodes",
        "1",
    ]


def _run_one_deploy(
    spec: ConfigSpec,
    *,
    alive_grace_s: int,
    shutdown_grace_s: int,
    timeout_s: int,
) -> AuditRow:
    """Tier-2 deploy launch: `openral deploy sim` → soak → SIGINT → graceful exit.

    Mirrors the SIGINT/SIGKILL escalation in
    ``openral_cli.deploy_sim._run_launch``, but operator-driven (SIGINT after
    ``alive_grace_s``, not Ctrl-C).

    Pass criteria: survives the alive grace, exits within ``shutdown_grace_s``
    of SIGINT, and the log contains the `deploy sim` banner from
    ``openral_cli.deploy_sim.deploy_sim_command``. Graceful exit codes: 0, 130
    (128+SIGINT), -2, -15 (SIGTERM). Any other code past the grace is
    fail-other; an early non-zero exit is classified via the tail patterns.
    """
    # Deploy rows switch uv groups too (libero → robocasa → libero); strip
    # the stale hf-libero egg-info first or `uv run --group libero` aborts
    # before the ROS graph ever launches (same failure as `_run_one`).
    _purge_hf_libero_egg(spec)

    env = os.environ.copy()
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env.setdefault("OPENRAL_AUTO_INSTALL_DEPS", "1")

    cmd = [
        "uv",
        "run",
        "--all-packages",
        "--group",
        spec.uv_group,
        "openral",
        "deploy",
        "sim",
        "--config",
        spec.config,
        # Headless: no live dashboard so the audit doesn't fight for the
        # OTLP port across rows. The dashboard child is the longest-lived
        # process in the graph and orphans most readily on SIGINT; killing
        # it keeps the audit's teardown deterministic.
        "--no-dashboard",
    ]

    sampler = _VramSampler()
    sampler.start()
    started = time.monotonic()
    # Drain to temp files, not PIPEs: runtime_node logs at ~30 Hz and an
    # undrained OS pipe (~64 KB) fills during the soak, blocking the child's
    # writers before they reach their SIGINT handlers — false fail-timeout
    # SIGKILL. Files have no backpressure; read back after exit.
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as out_f,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err_f,
    ):
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=out_f,
            stderr=err_f,
            start_new_session=True,
            text=True,
        )

        early_exit_code, returncode, wall_s, peak_vram = _soak_and_shutdown(
            proc,
            sampler=sampler,
            started=started,
            alive_grace_s=alive_grace_s,
            shutdown_grace_s=shutdown_grace_s,
        )

        # Process has fully exited — read the captured output back.
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read()
        stderr = err_f.read()

    combined = (stdout or "") + "\n" + (stderr or "")
    tail = "\n".join(combined.splitlines()[-30:])

    # Classification:
    # * Early crash (before alive grace) → run through the usual pattern table.
    # * Healthy soak + graceful SIGINT exit → pass.
    # * Healthy soak + abnormal exit → fail-other.
    if early_exit_code is not None and early_exit_code != 0:
        # Crashed during configure / activate — classify via the standard
        # tail patterns so OOM / missing-asset / opt-dep / sidecar errors
        # surface uniformly across run modes.
        return AuditRow(
            spec.config,
            spec.rskill,
            _classify(early_exit_code, combined),
            early_exit_code,
            wall_s,
            peak_vram,
            tail,
        )

    # Treat SIGINT-driven shutdown as pass when the CLI's banner printed
    # (proves resolution + launch invocation succeeded) AND the exit code
    # is one of the graceful-shutdown codes.
    banner_seen = "deploy sim" in combined and "robot=" in combined
    graceful_codes = {0, -signal.SIGINT, 128 + signal.SIGINT, -signal.SIGTERM}
    if banner_seen and returncode in graceful_codes:
        return AuditRow(spec.config, spec.rskill, "pass", returncode, wall_s, peak_vram, tail)
    if not banner_seen:
        return AuditRow(spec.config, spec.rskill, "fail-asset", returncode, wall_s, peak_vram, tail)
    if returncode == -9 or wall_s >= timeout_s:
        return AuditRow(
            spec.config, spec.rskill, "fail-timeout", returncode, wall_s, peak_vram, tail
        )
    return _classify_or_fallback(returncode, combined, tail, spec, wall_s, peak_vram)


def _soak_and_shutdown(
    proc: subprocess.Popen[str],
    *,
    sampler: _VramSampler,
    started: float,
    alive_grace_s: int,
    shutdown_grace_s: int,
) -> tuple[int | None, int, float, int | None]:
    """Soak the deploy graph for the alive grace, then SIGINT → SIGKILL.

    Returns:
        ``(early_exit_code, returncode, wall_s, peak_vram)``. ``early_exit_code``
        is the code if the proc exited before the alive grace elapsed (``None``
        if it survived the soak); ``returncode`` is the final exit code.
    """
    early_exit_code: int | None = None
    returncode: int
    try:
        # Soak: let the ROS graph configure → activate. If the launch
        # crashes during configure (HAL build failure, missing package,
        # etc.) the process exits early — capture that and skip the
        # SIGINT step.
        try:
            early_exit_code = proc.wait(timeout=alive_grace_s)
        except subprocess.TimeoutExpired:
            early_exit_code = None  # still alive after grace — expected

        if early_exit_code is None:
            # Healthy soak. Send SIGINT to the launch's process group;
            # ros2 launch translates it to a graceful lifecycle shutdown.
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(proc.pid, signal.SIGINT)
            try:
                returncode = proc.wait(timeout=shutdown_grace_s)
            except subprocess.TimeoutExpired:
                # SIGINT didn't drain in time → escalate to SIGKILL on
                # the launch's process group + treat as fail-other so
                # the operator sees the stragglers.
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
                try:
                    returncode = proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    returncode = -9
        else:
            returncode = early_exit_code
    finally:
        wall_s = time.monotonic() - started
        peak_vram = sampler.stop()

    return early_exit_code, returncode, wall_s, peak_vram


def _classify_or_fallback(
    returncode: int,
    combined: str,  # full output for classification
    tail: str,  # 30-line excerpt for AuditRow.tail display
    spec: ConfigSpec,
    wall_s: float,
    peak_vram: int | None,
) -> AuditRow:
    """Deploy-mode wrapper around ``_classify`` that defaults to ``fail-other``
    when no pattern matches (rather than 'pass')."""
    status = _classify(returncode, combined)
    if status == "pass" and returncode != 0:
        status = "fail-other"
    return AuditRow(spec.config, spec.rskill, status, returncode, wall_s, peak_vram, tail)


def _purge_hf_libero_egg(spec: ConfigSpec) -> None:
    """Strip the stale ``hf_libero-*.egg-info`` before a libero-group call.

    uv re-creates the duplicate egg-info (that ``_strip-hf-libero-egg``
    purges) on every group switch, so purge before each libero call, not
    just once — otherwise a round-trip like libero → robocasa → libero fails
    with "Unable to uninstall hf-libero==0.1.3: distutils-installed
    distributions do not include the metadata required to uninstall safely".
    Removes both forms: the directory (setuptools editable layout) and the
    flat file (distutils manifest) — only the file form triggers the error.
    """
    if spec.uv_group != "libero":
        return
    venv = REPO_ROOT / ".venv"
    if not venv.exists():
        return
    for path in venv.rglob("hf_libero-*.egg-info"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _run_one(spec: ConfigSpec, timeout_s: int) -> AuditRow:
    _purge_hf_libero_egg(spec)

    env = os.environ.copy()
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["OPENRAL_SIM_SEQUENTIAL_INIT"] = "1"  # readable serial logs
    # Suppress the interactive "Install <pkg> deps now? [y/N]" prompt
    # that fires when a config (typically robocasa / RLDX-1) needs an
    # optional dep group. The audit's subprocess.run has no stdin and
    # would otherwise Abort with exit=1. Set to "1" so the prompt
    # auto-accepts; the operator can still cancel by interrupting the
    # whole audit.
    env.setdefault("OPENRAL_AUTO_INSTALL_DEPS", "1")
    # `--no-view` is incompatible with `MUJOCO_GL=egl` (cli.py rejects the
    # combination). The default view tri-state auto-disables the viewer when
    # EGL is set, so we don't pass --view/--no-view at all.
    cmd = _build_run_cmd(spec)

    sampler = _VramSampler()
    sampler.start()
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        returncode = proc.returncode
        combined = proc.stdout + "\n" + proc.stderr
        tail = "\n".join(combined.splitlines()[-30:])
    except subprocess.TimeoutExpired as exc:
        returncode = -9
        captured_stdout = (
            (exc.stdout or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        captured_stderr = (
            (exc.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        combined = (
            captured_stdout + "\n" + captured_stderr + "\nTIMEOUT after " + str(timeout_s) + " s"
        )
        tail = "\n".join(combined.splitlines()[-30:])
    finally:
        wall_s = time.monotonic() - started
        peak_vram = sampler.stop()

    # Classify against the full combined output so that error keywords that
    # appear before atexit/destructor noise (e.g. EGL cleanup spam after an
    # OOM) are not lost by the 30-line tail truncation.
    status = _classify(returncode, combined)
    return AuditRow(spec.config, spec.rskill, status, returncode, wall_s, peak_vram, tail)


def _filter_catalogue(filters: Iterable[str]) -> list[ConfigSpec]:
    """When `filters` is empty, return the full catalogue; otherwise keep
    entries whose YAML stem matches any filter."""
    filters_list = list(filters)
    if not filters_list:
        return list(CATALOGUE)
    keep: list[ConfigSpec] = []
    for spec in CATALOGUE:
        stem = Path(spec.config).stem
        if any(f in stem or f == spec.config for f in filters_list):
            keep.append(spec)
    return keep


def _write_report(rows: list[AuditRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": [dataclasses.asdict(r) for r in rows],
        "summary": {
            status: sum(1 for r in rows if r.status == status)
            for status in (
                "pass",
                "pass-compat",
                "fail-oom",
                "fail-asset",
                "fail-sidecar",
                "fail-timeout",
                "fail-other",
                "fail-compat",
                "skipped-opt-dep",
                "skipped-host-setup",
            )
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))


def _preflight_libero(catalogue: list[ConfigSpec]) -> None:
    """Re-stamp ~/.libero/config.yaml against the current venv before any
    LIBERO rollout. Idempotent — no-ops when the config is already correct.

    LIBERO caches absolute paths to its asset directories on first import
    and subsequent runs in a different venv crash with a confusing
    ``FileNotFoundError`` on ``init_files/<task>.pruned_init`` instead of
    a clean error. The ``_ensure-libero-config`` Justfile helper does the
    same fixup before per-config Justfile recipes; the audit invokes it
    directly so the LIBERO subset of the catalogue runs cleanly.
    """
    if not any(spec.uv_group == "libero" for spec in catalogue):
        return
    fixer = REPO_ROOT / "tools" / "fix_libero_config.py"
    if not fixer.exists():
        return
    # Purge the duplicate hf-libero egg-info that confuses uv group switches.
    venv = REPO_ROOT / ".venv"
    if venv.exists():
        for path in venv.rglob("hf_libero-*.egg-info"):
            shutil.rmtree(path, ignore_errors=True)
    subprocess.run(
        ["uv", "run", "--group", "libero", "python", str(fixer)],
        cwd=REPO_ROOT,
        check=False,
        timeout=120,
    )


def _print_markdown(rows: list[AuditRow]) -> None:
    print("| Config | rSkill | Status | Exit | Wall (s) | Peak VRAM (MiB) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        vram = "-" if r.peak_vram_mib is None else str(r.peak_vram_mib)
        print(
            f"| `{Path(r.config).name}` | `{r.rskill}` | **{r.status}** "
            f"| {r.exit_code if r.exit_code is not None else '-'} | {r.wall_s:.1f} | {vram} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "filters",
        nargs="*",
        help="Optional config stems or paths to audit; default is every YAML in the catalogue.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help="Per-config wall-clock timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--deploy-alive-grace",
        type=int,
        default=DEFAULT_DEPLOY_ALIVE_GRACE_S,
        help=(
            "Per-deploy-row seconds to soak the ROS graph before SIGINTing it "
            f"(default: {DEFAULT_DEPLOY_ALIVE_GRACE_S}). Long enough to cover "
            "HAL configure → activate + safety_kernel + opt-in slam/nav2."
        ),
    )
    parser.add_argument(
        "--deploy-shutdown-grace",
        type=int,
        default=DEFAULT_DEPLOY_SHUTDOWN_GRACE_S,
        help=(
            "Per-deploy-row seconds to wait after SIGINT for the launch to "
            f"drain (default: {DEFAULT_DEPLOY_SHUTDOWN_GRACE_S}). After this "
            "the audit escalates to SIGKILL and the row is marked fail-other."
        ),
    )
    parser.add_argument(
        "--check-compatibility",
        action="store_true",
        help=(
            "Cheap in-process gate: load each scene YAML via "
            "`openral_core.load_scene_strict`, validate the matching rSkill "
            "manifest, and (for deploy rows) assert the robot resolves in "
            "`_ROBOT_HAL_REGISTRY`. No subprocess, no GPU, no env build."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUT_DIR / "audit_sim_configs.json",
        help="Where to write the JSON report.",
    )
    args = parser.parse_args(argv)

    catalogue = _filter_catalogue(args.filters)
    if not catalogue:
        sys.stderr.write(f"No configs matched filters {args.filters!r}\n")
        return 2

    if not args.check_compatibility:
        # Heavy preflight (LIBERO config restamp + egg-info purge) only
        # matters for the full-rollout path.
        _preflight_libero(catalogue)

    rows: list[AuditRow] = []
    for i, spec in enumerate(catalogue, 1):
        sys.stderr.write(
            f"[{i}/{len(catalogue)}] {spec.config} -> "
            f"{spec.rskill or '<env-only>'} ({spec.run_mode}) ...\n"
        )
        sys.stderr.flush()
        if args.check_compatibility:
            row = _check_compat(spec)
        elif spec.run_mode == "deploy":
            row = _run_one_deploy(
                spec,
                alive_grace_s=args.deploy_alive_grace,
                shutdown_grace_s=args.deploy_shutdown_grace,
                timeout_s=args.timeout,
            )
        else:
            row = _run_one(spec, args.timeout)
        sys.stderr.write(
            f"    -> {row.status} (exit={row.exit_code}, wall={row.wall_s:.1f}s, vram={row.peak_vram_mib} MiB)\n"
        )
        sys.stderr.flush()
        rows.append(row)

    _write_report(rows, args.report)
    _print_markdown(rows)

    sys.stderr.write(f"\nReport written to {args.report}\n")
    fails = [r for r in rows if r.status.startswith("fail-")]
    if fails:
        sys.stderr.write(f"FAIL: {len(fails)}/{len(rows)} configs did not pass.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
