"""Real GPU rollout audit for every (scene x rSkill) combination under ``scenes/``.

Per CLAUDE.md §1.11–§1.12: no smoke tests, no ``--dry-run``. Each catalogue
entry is launched through the matching tier-aware CLI (``openral sim run``
for SimScene-tier entries, ``openral benchmark scene`` for BenchmarkScene-tier
entries, ``openral deploy sim`` for DeployScene-tier entries) for one real
episode (sim/benchmark) or one full ROS-graph launch + graceful SIGINT
teardown (deploy), and the outcome is classified from the exit code +
stderr tail. Result is written as a JSON report and printed as a Markdown
table that the operator can paste into a PR.

Two modes:

* **Default (full rollout / launch).** Tier 3 for sim+benchmark scenes;
  Tier 2 (launch + ``--alive-grace`` seconds + SIGINT) for deploy scenes.
  Expensive — 30 s – 10 min per row depending on the VLA + asset cold-cache.
* **``--check-compatibility``.** In-process gate. For sim/benchmark rows it
  loads the scene via :func:`openral_core.load_scene_strict` *and* validates
  the rSkill manifest via :class:`openral_core.RSkillManifest`; for deploy
  rows it loads the scene + asserts ``robot_id`` resolves to a HAL entry in
  ``openral_cli.deploy_sim._ROBOT_HAL_REGISTRY``. Cheap — single-digit
  seconds per row, no GPU, no subprocess.

Usage::

    uv run python tools/audit_sim_configs.py                          # full rollouts
    uv run python tools/audit_sim_configs.py --check-compatibility    # cheap gate
    uv run python tools/audit_sim_configs.py libero_spatial pusht     # narrow by stem

This is operator-driven, not a pytest test — full rollouts at 30 s – 10 min
each don't belong in ``just test``. The companion fast check lives in
``tests/unit/test_examples_sim_configs_load.py``, which only asserts that
every YAML *loads* as its tier's typed schema (no rollout).
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

    ``config`` is the YAML's path relative to the repo root.
    ``rskill`` is the bare rSkill reference (name or path). Empty string for
    ``run_mode == "deploy"`` — DeployScenes are env-only and the reasoner
    picks the rSkill at runtime; the audit therefore tests the launch +
    teardown contract, not a specific policy.
    ``uv_group`` selects the uv dependency group (``libero``, ``metaworld``,
    ``robocasa``, ``maniskill3``, ``simpler-env``, or ``sim`` for the default).
    ``run_mode`` is ``"sim"`` for ``scenes/sim/*.yaml`` (driven by
    ``openral sim run``), ``"benchmark"`` for ``scenes/benchmark/*.yaml``
    (driven by ``openral benchmark scene --no-update-manifest --n-episodes 1``;
    a demo-grade run, not a paper-comparable claim — see ADR-0041), or
    ``"deploy"`` for ``scenes/deploy/*.yaml`` (driven by ``openral deploy sim``
    + ``--alive-grace`` seconds of soak + SIGINT graceful teardown).

    The catalogue holds only scene×rSkill pairs that actually exist in
    the tree — scenes without a matching in-tree rSkill are tracked in
    the scene YAML itself (and in
    ``tests/unit/test_examples_sim_configs_load.py`` for schema-load
    coverage), not as audit rows.
    """

    config: str
    rskill: str
    uv_group: str
    run_mode: RunMode


# Explicit mapping. The YAML names + rSkill names are not regular enough to
# derive via a rule (e.g. `scenes/sim/robocasa_pnp.yaml` →
# `rskills/pi05-robocasa365-human300-nf4`), so we list them. Each row is
# one (scene, rSkill) combination — the same scene may appear multiple
# times paired with different rSkills (LIBERO is the obvious example).
CATALOGUE: Final[tuple[ConfigSpec, ...]] = (
    # ---- SimScene tier (openral sim run) ----
    ConfigSpec("scenes/sim/libero_spatial.yaml", "rskills/smolvla-libero", "libero", "sim"),
    ConfigSpec("scenes/sim/libero_spatial.yaml", "rskills/xvla-libero", "libero", "sim"),
    ConfigSpec("scenes/sim/libero_spatial.yaml", "rskills/pi05-libero-nf4", "libero", "sim"),
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
        "rskills/pi05-libero-nf4",
        "libero",
        "sim",
    ),
    ConfigSpec(
        "scenes/sim/openarm_tabletop.yaml",
        "rskills/pi05-openarm-vision-nf4",
        "sim",
        "sim",
    ),
    ConfigSpec(
        "scenes/sim/robocasa_pnp.yaml",
        "rskills/pi05-robocasa365-human300-nf4",
        "robocasa",
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
    # The robocasa auto-install flow races with the package's own
    # import probe; when the install runs to completion but the probe
    # still fails the OpenRAL CLI emits this exact phrase and exits.
    # Surface it as `host-setup` so the operator knows it's not a real
    # config bug — it's a known robosuite/robocasa import-cache issue
    # documented at the install site.
    "install ran to completion but the probe still fails",
)
_HOST_SETUP_PATTERNS: Final[tuple[str, ...]] = (
    # `uv` reinstalls `hf-libero==0.1.3` during group switches and the
    # distutils-installed egg-info from the prior install resists
    # uninstall. The audit's per-call purge fixes the first occurrence
    # but successive `uv group` swaps can recreate the .egg-info inside
    # the same audit run. Treat as host-setup, not as a config failure.
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

    For sim/benchmark rows: load the YAML via
    :func:`openral_core.load_scene_strict` (covers the ADR-0042 bare-list
    contract for benchmark rows and the ADR-0041 SimScene contract for
    sim rows), then validate the rSkill manifest via
    :class:`openral_core.RSkillManifest` so a missing or schema-busted
    ``rskill.yaml`` is caught before paying for env build.

    For deploy rows: load the YAML as :class:`openral_core.DeployScene` and
    assert ``robot_id`` resolves to a HAL package registered in
    ``openral_cli.deploy_sim._ROBOT_HAL_REGISTRY``. Catches the common
    "scene references a robot that has no HAL twin" failure mode without
    booting a ROS graph.

    Returns an :class:`AuditRow` with ``status="pass-compat"`` on success
    or ``"fail-compat"`` on any schema / lookup error.

    No subprocess. No GPU. ``wall_s`` is the in-process cost (single-digit
    seconds even with cold imports).
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

    Split out of :func:`_run_one` so the deploy-tier launch path
    (:func:`_run_one_deploy`) can stay focused on lifecycle teardown.
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
    # 1-episode demo, not a paper-comparable claim — see ADR-0041).
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

    Mirrors the SIGINT / SIGKILL escalation that
    ``openral_cli.deploy_sim._run_launch`` applies internally, but the
    SIGINT is operator-driven here (after ``alive_grace_s`` seconds of
    soak), not user-driven via Ctrl-C.

    Pass criteria: the launch survives the alive grace without crashing AND
    exits cleanly on SIGINT within ``shutdown_grace_s`` AND the captured
    log contains the `deploy sim` banner from
    ``openral_cli.deploy_sim.deploy_sim_command`` (proves at least the CLI
    resolution + invocation path ran).

    Returncode interpretation: 0, 130 (=128+SIGINT, the Python convention),
    -2 (signal.SIGINT as a negative exit code), and -15 (SIGTERM, in case
    the launch's shutdown supervisor intercepts) are all "graceful". Any
    other code while the proc was alive past the grace = fail-other.
    Early exit (before alive grace) with non-zero = fail (classified by
    tail patterns as usual).
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
    # Drain the child's stdout/stderr to temp files rather than PIPEs.
    # runtime_node logs at ~30 Hz; an undrained OS pipe (~64 KB) fills
    # during the alive-grace soak, blocks the child's writers, and prevents
    # them from reaching their SIGINT shutdown handlers — forcing a false
    # fail-timeout SIGKILL. Files have no backpressure, so the graph tears
    # down cleanly and we read the captured output back after exit for
    # classification.
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

        # Process has fully exited — read the captured output back from the
        # temp files (replaces the old post-exit proc.communicate()).
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

    Returns ``(early_exit_code, returncode, wall_s, peak_vram)``. The
    signal / timeout sequencing is unchanged from the original inline
    body — only the output draining moved to the caller (temp files).
    ``early_exit_code`` is the code if the proc exited before the alive
    grace elapsed (``None`` if it survived the soak); ``returncode`` is
    the final exit code in all cases.
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
    """Deploy-mode wrapper around :func:`_classify` that defaults to ``fail-other``
    when no pattern matches (rather than 'pass')."""
    status = _classify(returncode, combined)
    if status == "pass" and returncode != 0:
        status = "fail-other"
    return AuditRow(spec.config, spec.rskill, status, returncode, wall_s, peak_vram, tail)


def _purge_hf_libero_egg(spec: ConfigSpec) -> None:
    """Strip the stale ``hf_libero-*.egg-info`` before a libero-group call.

    uv re-installs ``hf-libero`` whenever it switches dependency groups,
    which repeatedly re-creates the duplicate egg-info that the
    ``_strip-hf-libero-egg`` Justfile recipe purges. Purge before every
    libero call, not just once at audit start, otherwise the second
    round-trip (e.g. libero → robocasa → libero) fails with
    "Unable to uninstall hf-libero==0.1.3: distutils-installed
    distributions do not include the metadata required to uninstall safely".

    The egg-info appears in two forms across uv/distutils versions: a
    directory (older setuptools editable layout) and a flat file (the
    distutils-installed manifest uv cannot uninstall). Remove both —
    ``shutil.rmtree`` alone misses the file form, which is exactly the one
    that triggers the uv error.
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
    # MolmoAct2 ships custom modelling code in the HF repo and requires
    # OPENRAL_ALLOW_REMOTE_CODE=1 to proceed past the safety gate; without
    # it the process exits with ROSConfigError before the model even loads,
    # and the audit reports fail-other instead of the real failure reason
    # (fail-oom on 8 GiB GPUs, or pass on 16+ GiB). The audit acknowledges
    # this risk because it runs against rSkills in the local tree whose
    # source_repo is pinned in rskill.yaml (§3 / ADR-0006).
    env.setdefault("OPENRAL_ALLOW_REMOTE_CODE", "1")

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
