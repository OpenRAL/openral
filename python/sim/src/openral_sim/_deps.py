"""Lazy auto-install helpers for sim backends with bespoke install chains.

Backends like RoboCasa (kitchen + GR1 fork) and LIBERO need install
recipes that go beyond ``uv sync --group <name>``: editable installs of
git clones, ``--no-deps`` pins of robosuite master, compiler env
overrides (``CC=/usr/bin/gcc`` for LIBERO's robosuite==1.4 C
extensions). The historical UX was a 10-line install hint baked into a
``ROSConfigError`` and the user pasted commands by hand. This module
turns that into a one-prompt-and-go flow:

* :class:`BackendInstallPlan` declares a sequence of subprocess steps
  plus probe imports to detect "already installed".
* :func:`ensure_backend_deps` short-circuits when the probes succeed,
  otherwise prints a Rich banner with the full plan and auto-installs
  (default). Set ``OPENRAL_AUTO_INSTALL_DEPS=0`` to prompt instead.
  Failures raise a typed :class:`ROSConfigError` with the verbatim
  commands so the user can finish out-of-band. Failures raise a typed
  :class:`ROSConfigError` with the verbatim commands so the user can
  finish out-of-band.

Sibling of :mod:`openral_sim._assets` which handles the lazy-download
of large CC-BY asset bundles after deps are in place.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sysconfig
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from openral_core.exceptions import ROSConfigError
from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from collections.abc import Callable


_DEFAULT_CACHE_HOME = Path.home() / ".cache" / "openral"
_AUTO_INSTALL_ENV = "OPENRAL_AUTO_INSTALL_DEPS"

_INSTALL_LOCK = threading.Lock()
"""Serialises :func:`ensure_backend_deps` across threads.

``SimRunner._build_env_and_policy`` builds env + policy concurrently on a
2-worker ``ThreadPoolExecutor``. Without this lock both workers race into
``ensure_backend_deps`` and interleave their Rich banners + ``typer.confirm``
prompts on the same stdout/stdin pair — the user sees two prompts back-to-back
on one line and the single ``y`` is consumed by only one of them. Holding the
lock for the whole probe→prompt→install→re-probe sequence means a single
banner is shown, the user answers once, and the second thread re-probes after
the install has run (and short-circuits if the same plan covered both)."""
"""Set to ``1`` to skip the confirmation prompt and run the plan straight away.

Honoured the same way :data:`openral_sim._assets._ROBOCASA_ALLOW_ENV`
gates the asset-download prompt — useful in CI / Dockerfiles where
interactive prompts are not available.
"""


def _cache_home() -> Path:
    raw = os.environ.get("OPENRAL_CACHE_HOME")
    return Path(raw) if raw else _DEFAULT_CACHE_HOME


@dataclass(frozen=True)
class InstallStep:
    """One subprocess step in a backend install plan.

    Attributes:
        description: One-line human description shown in the banner.
        argv: Command + args list, executed via :func:`subprocess.run`
            (``shell=False``). The first entry is resolved through
            :func:`shutil.which` so the plan can refer to ``"uv"`` /
            ``"git"`` without hardcoding paths.
        env: Extra env vars layered on top of the parent process's
            environment for this step only. ``CC=/usr/bin/gcc`` for
            LIBERO's robosuite C extensions is the motivating case.
        cwd: Working directory for the step. ``None`` keeps the parent
            cwd. Used to chain a ``git clone <path>`` step into a
            subsequent ``uv pip install -e <path>``.
    """

    description: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None


@dataclass(frozen=True)
class BackendInstallPlan:
    """A backend's "what to install + how to check it's there" recipe.

    Attributes:
        backend_id: Short slug used in log messages, e.g.
            ``"robocasa_gr1"``.
        display_name: Human-readable title for the Rich banner.
        license_note: One- or two-line license posture surfaced in the
            banner; declines to confirm count as a license refusal.
        probe: Zero-arg callable returning ``True`` when the backend
            is already installed. Runs at the start of
            :func:`ensure_backend_deps` and again after the plan has
            executed; the second probe gates the success message.
        steps: Ordered install steps. Each runs sequentially, stopping
            on the first non-zero exit.
        manual_hint: String shown inside the :class:`ROSConfigError`
            when the user refuses or a step fails -- gives them the
            exact commands to finish manually.
    """

    backend_id: str
    display_name: str
    license_note: str
    probe: Callable[[], bool]
    steps: tuple[InstallStep, ...]
    manual_hint: str


# ── probes ───────────────────────────────────────────────────────────────────


def _has_module(module: str) -> bool:
    """True iff ``module`` resolves via ``importlib.util.find_spec``.

    Uses ``find_spec`` instead of ``importlib.import_module`` so we do
    not trigger robocasa's import-time version assertions (which would
    raise AssertionError under a real numpy 2.x install before the
    backend's adapter has a chance to spoof ``__version__``).
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# Runtime imports robocasa / robosuite perform lazily but which
# `find_spec("robocasa")` will not flag as missing. Each entry has
# surfaced as a post-install bug at least once:
#   lxml, h5py            -- MJCF munging + episode logging
#   llvmlite, numba       -- robosuite -> mujoco_mjx -> scipy
#   robosuite             -- the obvious one
#   robosuite.examples    -- master-branch only; the PyPI wheel drops
#                            `robosuite.examples`, which is where
#                            `WholeBodyMinkIK` lives. Probing for it
#                            re-triggers the plan whenever a stray
#                            `uv sync` evicts master and reinstalls
#                            the PyPI wheel.
#   robosuite_models      -- robosuite logs WARN at import without it
#   mink, qpsolvers       -- robosuite's WholeBodyMinkIK runtime deps
# mimicgen is intentionally NOT in this list. robocasa.__init__
# prints "WARNING: mimicgen environments not imported …" when it is
# missing, but installing mimicgen brings two NEW print statements
# (robosuite_environments + robosuite_task_zoo) because mimicgen's
# own __init__ imports module paths the current robosuite master
# has renamed / removed. Net noise is worse with mimicgen installed
# than without, so we accept the one informational print line.
_ROBOCASA_RUNTIME_DEPS = (
    "lxml",
    "h5py",
    "llvmlite",
    "numba",
    "robosuite",
    "robosuite.examples",
    "robosuite_models",
    "mink",
    "qpsolvers",
)


def _has_robocasa_kitchen() -> bool:
    """Robocasa is installed AND the kitchen variant ships `download_kitchen_assets`."""
    if not _has_module("robocasa"):
        return False
    spec = importlib.util.find_spec("robocasa")
    if spec is None or spec.origin is None:
        return False
    scripts = Path(spec.origin).parent / "scripts"
    if not (scripts / "download_kitchen_assets.py").is_file():
        return False
    if (scripts / "download_tabletop_assets.py").is_file():
        # GR1 fork installed instead -- kitchen probe must say "no" so
        # the kitchen plan does not get skipped on a GR1 host.
        return False
    return all(_has_module(m) for m in _ROBOCASA_RUNTIME_DEPS)


def _has_robocasa_gr1() -> bool:
    """Robocasa is installed AND the GR1 fork ships `download_tabletop_assets`."""
    if not _has_module("robocasa"):
        return False
    spec = importlib.util.find_spec("robocasa")
    if spec is None or spec.origin is None:
        return False
    scripts = Path(spec.origin).parent / "scripts"
    # Filesystem check on robocasa.utils too -- the fork's setup.py
    # uses find_packages() which drops that directory under a
    # non-editable install, leaving a half-broken `import robocasa`.
    utils = Path(spec.origin).parent / "utils"
    if not (
        (scripts / "download_tabletop_assets.py").is_file()
        and utils.is_dir()
        and (utils / "placement_samplers.py").is_file()
    ):
        return False
    return all(_has_module(m) for m in _ROBOCASA_RUNTIME_DEPS)


def _has_simpler_env() -> bool:
    """SimplerEnv requires the upstream ``simpler_env`` package plus ManiSkill3.

    The ``simpler_env`` package has no PyPI release — it ships as a git
    install of the ``maniskill3`` branch — and rides on top of
    ``mani_skill`` (gymnasium-registered bridge digital twins). We probe
    both so a partial install (uv group synced but the git pip step
    skipped) re-triggers the plan.
    """
    return _has_module("simpler_env") and _has_module("mani_skill")


def _has_libero() -> bool:
    """LIBERO requires lerobot.envs.libero + a compatible robosuite (==1.4 series)."""
    if not _has_module("lerobot.envs.libero"):
        return False
    if not _has_module("robosuite"):
        return False
    # LIBERO pins robosuite==1.4; if a >=1.5 robosuite is installed
    # (e.g. from a robocasa install) we still treat LIBERO as missing
    # since LiberoEnv.reset() crashes on the newer API. Probe the version
    # via importlib.metadata rather than ``robosuite.__version__`` — the
    # openral-vendored robosuite 1.5.x builds don't expose ``__version__``,
    # and the bare attribute access crashed the libero readiness probe with
    # AttributeError on a robocasa-provisioned venv.
    try:
        installed = importlib.metadata.version("robosuite")
    except importlib.metadata.PackageNotFoundError:
        return False
    return installed.startswith("1.4")


def _has_maniskill3() -> bool:
    """ManiSkill3 needs ``mani_skill`` (carries SAPIEN) + ``gymnasium``."""
    return _has_module("mani_skill") and _has_module("gymnasium")


def _has_aloha() -> bool:
    """gym-aloha needs ``gym_aloha`` (the bimanual MuJoCo env registry) + ``gymnasium``."""
    return _has_module("gym_aloha") and _has_module("gymnasium")


def _has_metaworld() -> bool:
    """MetaWorld needs both ``lerobot.envs.metaworld`` and the ``metaworld`` package.

    ``metaworld==3.0.0`` ships as a separate ``--no-deps`` pip install (its
    transitive deps conflict with the workspace lock), so we probe both so a
    partial install — group synced but the ``pip install metaworld`` step
    skipped — re-triggers the plan.
    """
    return _has_module("lerobot.envs.metaworld") and _has_module("metaworld")


def _has_openarm_robosuite() -> bool:
    """OpenArm tabletop needs ``robosuite>=1.5`` + ``mujoco``.

    The backend uses ``robosuite.utils.binding_utils.MjSim`` as a thin MJCF
    wrapper (no robocasa envs, no OSC composite controllers, no
    WholeBodyMinkIK), so we deliberately do NOT probe the full
    ``_ROBOCASA_RUNTIME_DEPS`` set — robosuite-master,
    robosuite_models, mink, qpsolvers are all unnecessary here.

    A LIBERO-installed ``robosuite==1.4`` mis-trips a bare ``find_spec``
    check (the symbol resolves but the 1.5 controller / sampler API is
    missing); pin the major.minor floor so that case re-triggers the
    plan and the user gets the libero ↔ robocasa swap prompt.
    """
    if not _has_module("robosuite") or not _has_module("mujoco"):
        return False
    import robosuite

    try:
        major, minor, *_ = (int(p) for p in robosuite.__version__.split(".")[:2])
    except ValueError:
        # Unparsable version string — treat as "needs reinstall" rather
        # than silently accepting; the plan is cheap to re-run when
        # robosuite is actually fine.
        return False
    return (major, minor) >= (1, 5)


# ── plans ────────────────────────────────────────────────────────────────────


def _refresh_editable_finders() -> None:
    """Refresh ``sys.meta_path`` after ``uv pip install -e`` swaps editables in-process.

    setuptools-editable ships one ``__editable___<pkg>_<ver>_finder.py``
    module per editable package, registered via an
    ``__editable__.<pkg>-<ver>.pth`` shim that calls
    ``<finder_module>.install()``. The finder bakes a ``MAPPING`` dict
    at import time pointing at the source directory. When ``uv pip
    install -e`` swaps an editable install mid-process (e.g. the
    RoboCasa plan steps from the kitchen fork ``robocasa==1.0.1`` to
    the GR1 fork ``robocasa==0.2.0``), the old ``.pth`` + finder
    ``.py`` are removed from site-packages but the old
    ``_EditableFinder`` stays on ``sys.meta_path`` with its stale
    ``MAPPING``. ``importlib.invalidate_caches()`` only flushes
    path-importer caches, not ``sys.meta_path``, so ``find_spec("<pkg>")``
    keeps resolving to the old source directory and the post-install
    probe falsely reports the install never landed.

    This helper:

    1. Removes any ``sys.meta_path`` finder backed by a module whose
       ``__file__`` no longer exists on disk (the marker that uv
       deleted its ``.pth`` + ``_finder.py`` pair).
    2. Re-executes every current ``__editable__.*.pth`` ``import``
       line in the active venv's site-packages so the replacement
       finder modules register themselves via ``install()`` (which is
       idempotent: it no-ops when an equivalent finder is already
       present).
    """
    import sys

    purelib = Path(sysconfig.get_paths()["purelib"])

    # 1) Drop sys.meta_path entries whose backing module file is gone.
    for finder in list(sys.meta_path):
        mod_name = getattr(finder, "__module__", None)
        if not mod_name or not mod_name.startswith("__editable__"):
            continue
        mod = sys.modules.get(mod_name)
        mod_file = getattr(mod, "__file__", None) if mod else None
        if mod_file and not Path(mod_file).is_file():
            with contextlib.suppress(ValueError):
                sys.meta_path.remove(finder)
            sys.modules.pop(mod_name, None)

    # 2) Re-run the current setuptools-editable .pth shims. setuptools
    #    emits two shapes depending on the project layout:
    #      a) Finder-shim mode (strict / multi-package layouts, used by
    #         the robocasa-gr1 + robocasa-kitchen forks): a single
    #         ``import <finder_mod>; <finder_mod>.install()`` line. The
    #         import loads the (possibly newly-written) finder with
    #         the correct MAPPING, and install() is idempotent (no-ops
    #         when an equivalent finder is already on sys.meta_path).
    #      b) Path-based mode (simple single-package src layouts): a
    #         single line containing the source directory. Mirrors
    #         site.addsitedir's path-line handling.
    import sys as _sys

    if not purelib.is_dir():
        return
    for pth in purelib.glob("__editable__.*.pth"):
        try:
            text = pth.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("import ", "import\t")):
                # exec is the documented mechanism for setuptools-editable
                # .pth shims (mirrors what site.addpackage does); we already
                # filter to lines starting with "import ", and the .pth
                # files come from the active venv's site-packages.
                with contextlib.suppress(Exception):
                    exec(line, {})
                break
            candidate = Path(line)
            if candidate.is_dir() and str(candidate) not in _sys.path:
                _sys.path.insert(0, str(candidate))
            break


def _uv() -> str:
    """Return the path to the ``uv`` executable or raise ROSConfigError."""
    found = shutil.which("uv")
    if found is None:
        raise ROSConfigError(
            "uv is required for auto-install but was not found on PATH. "
            "Install with `curl -LsSf https://astral.sh/uv/install.sh | sh` "
            "and re-run."
        )
    return found


def _git() -> str:
    """Return the path to ``git`` or raise ROSConfigError."""
    found = shutil.which("git")
    if found is None:
        raise ROSConfigError(
            "git is required to install the RoboCasa GR1 fork but was not found on PATH."
        )
    return found


def _remove_editable_shadow_step(pkg_name: str) -> InstallStep:
    """A post-editable-install step that removes a ``<site_packages>/<pkg>/`` shadow.

    Closes a real install-side wedge surfaced by the rSkill audit GPU smoke tests:
    ``uv pip install --force-reinstall -e <fork>`` removes files it tracks but
    leaves untracked siblings behind. The robosuite + robocasa-gr1 + robocasa
    kitchen forks all ship a sibling ``macros_private.py`` that the
    package writes at first import (or that the install plan's patch step
    pre-creates), and that file lives in a ``<site_packages>/<pkg>/`` dir
    without an ``__init__.py``. Python's site-packages walk then treats
    the shadow as a **namespace package** and intercepts the import
    *before* the ``__editable__.{pkg}.pth`` finder gets a chance, so
    ``importlib.util.find_spec("<pkg>").origin`` returns ``None`` and
    every probe in :func:`_has_robocasa_kitchen` / :func:`_has_robocasa_gr1`
    fails — even though the editable install is correct on disk.

    The step removes the shadow iff it exists and has no ``__init__.py``
    (i.e. it is the namespace-package stub, not a legitimate install).
    Idempotent: a missing shadow exits silently.
    """
    uv = _uv()
    script = (
        "import sys, sysconfig, shutil\n"
        "from pathlib import Path\n"
        f"pkg = {pkg_name!r}\n"
        "sp = Path(sysconfig.get_paths()['purelib'])\n"
        "shadow = sp / pkg\n"
        "if (\n"
        "    shadow.is_dir()\n"
        "    and not shadow.is_symlink()\n"
        "    and not (shadow / '__init__.py').is_file()\n"
        "):\n"
        "    shutil.rmtree(shadow)\n"
        "    print(f'removed editable-install shadow dir: {shadow}')\n"
        "else:\n"
        "    print(f'no shadow to clean for {pkg}')\n"
    )
    return InstallStep(
        description=(
            f"clean site-packages namespace shadow for {pkg_name!r} (post-editable; "
            "a sibling macros_private.py without __init__.py hides the editable "
            "finder and breaks importlib.find_spec)"
        ),
        argv=[uv, "run", "python", "-c", script],
    )


# Pin robosuite to a specific master commit instead of floating HEAD.
#
# Both robocasa forks install robosuite from an editable clone of
# ARISE-Initiative/robosuite *master*. The kitchen fork (robocasa 1.0.1)
# tracks recent master, but the GR1 fork (robocasa-gr1-tabletop-tasks
# 0.2.0, NVIDIA's GR00T-N1 release) was authored against robosuite
# 1.5.0/1.5.1 and only declares support for those. Riding floating
# master means a future master commit that refactors the robot
# base-class API silently breaks the GR1 env build with
# ``ValueError: Invalid base type to add to robot!`` at
# ``robot_model.py:add_base`` (issue #44) while the kitchen fork keeps
# working — and the two cannot be told apart by version string because
# master always reports ``"1.5.2"``. Pinning to a single verified commit
# makes both forks deterministic and lets them share one robosuite
# install (no per-scene robosuite swap).
#
# This SHA is validated end-to-end by ``openral sim run`` on
# ``robocasa_gr1_pnp_cup_to_drawer`` (GR1 + RLDX-1-FT-GR1, full episode)
# AND by the kitchen scenes. Bump it only after re-running both on the
# new commit.
#
# MUST equal the ``[tool.uv.sources] robosuite = { rev = ... }`` pin in
# ``pyproject.toml``: ``uv sync --group robocasa`` lands robosuite at the
# uv.sources rev, then the steps below ``uv pip install -e`` the local
# clone OVER it. If the clone rode floating master (the bug — issue #44)
# the editable reinstall silently replaced the pinned tree with a
# drifting one. Keep the two in lockstep when bumping.
_ROBOSUITE_PIN = "232ce7d4a6ed89c949a9aba024a05c8c32fdd08b"  # master @ 2026-05-09


def _robosuite_clone_step(git: str, rs_dir: Path) -> InstallStep:
    """Clone robosuite (if absent) and pin it to :data:`_ROBOSUITE_PIN`.

    Idempotent: a shallow master clone seeds the directory on first run,
    then we shallow-fetch the pinned commit and check it out detached.
    Re-running on an existing clone just re-asserts the checkout. Shared
    by the kitchen and GR1 plans so both forks land on the same commit
    (issue #44 — a drifting master broke the GR1 fork's ``add_base``).
    """
    return InstallStep(
        description=(
            f"git clone robosuite + pin to {_ROBOSUITE_PIN[:12]} → {rs_dir} "
            "(idempotent; pinned commit instead of floating master — issue #44)"
        ),
        argv=[
            "bash",
            "-c",
            f"set -e; [ -d {rs_dir}/.git ] || {git} clone --depth=1 "
            f"https://github.com/ARISE-Initiative/robosuite.git {rs_dir}; "
            f"{git} -C {rs_dir} fetch --depth=1 origin {_ROBOSUITE_PIN}; "
            f"{git} -C {rs_dir} checkout --quiet --detach {_ROBOSUITE_PIN}",
        ],
    )


def _robosuite_clone_hint(git: str, rs_dir: Path) -> str:
    """Manual-install one-liner mirroring :func:`_robosuite_clone_step`."""
    return (
        f"{git} clone --depth=1 https://github.com/ARISE-Initiative/robosuite.git {rs_dir} ; "
        f"{git} -C {rs_dir} fetch --depth=1 origin {_ROBOSUITE_PIN} && "
        f"{git} -C {rs_dir} checkout --detach {_ROBOSUITE_PIN}"
    )


def _robocasa_kitchen_plan() -> BackendInstallPlan:
    uv = _uv()
    git = _git()
    rs_dir = _robosuite_clone_dir()
    rc_dir = _robocasa_kitchen_clone_dir()
    return BackendInstallPlan(
        backend_id="robocasa_kitchen",
        display_name="RoboCasa kitchen (~100 atomic PnP / door / drawer tasks)",
        license_note=(
            "Pulls robocasa (MIT), robosuite master (MIT), robosuite-models "
            "(MIT). Kitchen asset bundle (CC-BY-4.0, ~11 GB) is fetched "
            "separately on first env build."
        ),
        probe=_has_robocasa_kitchen,
        steps=(
            InstallStep(
                description=(
                    "uv sync --group robocasa --inexact (workspace deps + robosuite>=1.5; "
                    "--inexact preserves packages from sibling groups already in the venv "
                    "— e.g. pyzmq/msgpack from --group rldx — that would otherwise be "
                    "uninstalled because they are not in the robocasa group's resolved set)"
                ),
                argv=[
                    uv,
                    "sync",
                    "--all-packages",
                    "--group",
                    "robocasa",
                    "--inexact",
                ],
            ),
            InstallStep(
                description=f"mkdir -p {rs_dir.parent} (cache for editable installs)",
                argv=["mkdir", "-p", str(rs_dir.parent)],
            ),
            _robosuite_clone_step(git, rs_dir),
            InstallStep(
                description=(
                    "patch robosuite clone: touch missing __init__.py under "
                    "robosuite/examples/ + robosuite/examples/third_party_controller/ "
                    "(upstream find_packages() drops these so robosuite/__init__.py:37 "
                    "always fails to import WholeBodyMinkIK without this patch), "
                    "and write empty macros_private.py"
                ),
                argv=[
                    "bash",
                    "-c",
                    f"touch {rs_dir}/robosuite/examples/__init__.py "
                    f"{rs_dir}/robosuite/examples/third_party_controller/__init__.py "
                    f"{rs_dir}/robosuite/macros_private.py",
                ],
            ),
            InstallStep(
                description=(
                    f"uv pip install -e {rs_dir} "
                    "(editable so the __init__.py touches above survive)"
                ),
                argv=[
                    uv,
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-deps",
                    "-e",
                    str(rs_dir),
                ],
            ),
            _remove_editable_shadow_step("robosuite"),
            InstallStep(
                description="uv pip install robosuite-models",
                argv=[
                    uv,
                    "pip",
                    "install",
                    "--no-deps",
                    "robosuite-models @ git+https://github.com/ARISE-Initiative/robosuite_models.git",
                ],
            ),
            InstallStep(
                description=(
                    "uv pip install runtime extras "
                    "(h5py lxml llvmlite numba qpsolvers pyopengl-accelerate)"
                ),
                argv=[
                    uv,
                    "pip",
                    "install",
                    "h5py>=3.16",
                    "lxml>=5",
                    "llvmlite",
                    "numba",
                    # robosuite's WholeBodyMinkIK loader needs both
                    # mink (Python QP wrapper) and qpsolvers (the QP
                    # solver). Without them robosuite logs
                    # "Could not load the mink-based whole-body IK".
                    "qpsolvers",
                    # Optional perf module for PyOpenGL; silences the
                    # "No OpenGL_accelerate module loaded" INFO log on
                    # every offscreen-render setup.
                    "pyopengl-accelerate",
                ],
            ),
            InstallStep(
                description=(
                    "uv pip install mink==0.0.5 --no-deps. Pin to 0.0.5 because "
                    "robosuite's mink_controller.py imports "
                    "`from mink.tasks.exceptions import TargetNotSet`, a "
                    "path that newer mink (1.x) restructured. --no-deps "
                    "avoids forcing numpy<2 (mink 0.0.5's pin), which "
                    "would downgrade and break every other backend."
                ),
                argv=[uv, "pip", "install", "--no-deps", "mink==0.0.5"],
            ),
            InstallStep(
                description=(
                    f"git clone --depth=1 robocasa (kitchen) → {rc_dir} "
                    "(idempotent: skipped if already cloned). Editable "
                    "install pattern matches the robosuite clone above — "
                    "the upstream wheel/sdist excludes the models/assets/ "
                    "tree (including box_links/box_links_assets.json that "
                    "the kitchen download script reads), so a non-editable "
                    '`pip install "robocasa @ git+..."` succeeds but every '
                    "subsequent env build fails on missing arena XMLs."
                ),
                argv=[
                    "bash",
                    "-c",
                    f"[ -d {rc_dir}/.git ] || {git} clone --depth=1 "
                    f"https://github.com/robocasa/robocasa.git {rc_dir}",
                ],
            ),
            InstallStep(
                description=(
                    "patch robocasa kitchen clone: write empty macros_private.py "
                    "(silences `[robosuite WARNING] No private macro file found`)"
                ),
                argv=["bash", "-c", f"touch {rc_dir}/robocasa/macros_private.py"],
            ),
            InstallStep(
                description=(
                    f"uv pip install -e {rc_dir} --no-deps "
                    "(editable so models/assets/ is reachable via "
                    "import robocasa; --no-deps keeps the carefully "
                    "constructed venv intact)"
                ),
                argv=[uv, "pip", "install", "--no-deps", "-e", str(rc_dir)],
            ),
            _remove_editable_shadow_step("robocasa"),
        ),
        manual_hint=(
            "just sync --all-packages --group robocasa --inexact && "
            f"mkdir -p {rs_dir.parent} && "
            f"{_robosuite_clone_hint(git, rs_dir)} && "
            f"touch {rs_dir}/robosuite/examples/__init__.py "
            f"{rs_dir}/robosuite/examples/third_party_controller/__init__.py "
            f"{rs_dir}/robosuite/macros_private.py && "
            f"uv pip install --force-reinstall --no-deps -e {rs_dir} && "
            'uv pip install --no-deps "robosuite-models @ '
            'git+https://github.com/ARISE-Initiative/robosuite_models.git" && '
            'uv pip install "h5py>=3.16" "lxml>=5" llvmlite '
            "numba qpsolvers pyopengl-accelerate && "
            'uv pip install --no-deps "mink==0.0.5" && '
            f"git clone --depth=1 https://github.com/robocasa/robocasa.git {rc_dir} && "
            f"touch {rc_dir}/robocasa/macros_private.py && "
            f"uv pip install --no-deps -e {rc_dir}"
        ),
    )


def _gr1_clone_dir() -> Path:
    """Stable on-disk path for the GR1 fork clone (editable install anchor).

    Lives under ``$OPENRAL_CACHE_HOME/repos/robocasa-gr1-tabletop-tasks``.
    Editable installs anchor to this path forever, so it MUST stay
    stable across runs -- moving the clone would silently break
    ``import robocasa``.
    """
    return _cache_home() / "repos" / "robocasa-gr1-tabletop-tasks"


def _robocasa_kitchen_clone_dir() -> Path:
    """Stable on-disk path for the robocasa kitchen clone (editable install anchor).

    Lives under ``$OPENRAL_CACHE_HOME/repos/robocasa-kitchen``. Same
    rationale as :func:`_gr1_clone_dir` and :func:`_robosuite_clone_dir`:
    editable installs anchor here forever. The earlier non-editable
    `uv pip install --no-deps "robocasa @ git+https://..."` path was a
    real wedge — the upstream sdist excludes the ``models/assets/`` tree
    AND the ``models/assets/box_links/box_links_assets.json`` data file
    that ``download_kitchen_assets.py`` reads to know what to fetch, so
    the post-install asset download crashed with a confusing
    FileNotFoundError. Editable install keeps the full source tree
    (including ``models/assets/``) on disk, so ``import robocasa`` sees
    the assets directly.
    """
    return _cache_home() / "repos" / "robocasa-kitchen"


def _robosuite_clone_dir() -> Path:
    """Stable on-disk path for the robosuite master clone.

    Same rationale as :func:`_gr1_clone_dir`: editable installs anchor
    here. We need a patched master clone (with empty ``__init__.py``
    files dropped into ``robosuite/examples/`` and
    ``robosuite/examples/third_party_controller/``) because upstream
    `setup.py` uses `find_packages()` which excludes those directories
    from the wheel -- but robosuite's own `__init__.py:37` then tries
    to `from robosuite.examples.third_party_controller.mink_controller
    import WholeBodyMinkIK` and logs WARN when that import fails. The
    PyPI wheel hits the WARN unconditionally for the same reason.
    """
    return _cache_home() / "repos" / "robosuite"


def _robocasa_gr1_plan() -> BackendInstallPlan:
    uv = _uv()
    git = _git()
    clone_dir = _gr1_clone_dir()
    clone_parent = clone_dir.parent
    rs_dir = _robosuite_clone_dir()
    return BackendInstallPlan(
        backend_id="robocasa_gr1",
        display_name="RoboCasa GR1 tabletop (24 PnP envs from NVIDIA's GR00T-N1 release)",
        license_note=(
            "Pulls the robocasa-gr1-tabletop-tasks fork (MIT) editable from "
            "$OPENRAL_CACHE_HOME/repos/, robosuite master (MIT), "
            "robosuite-models (MIT). Tabletop asset bundle "
            "(CC-BY-4.0, ~1 GB) is fetched separately on first env build."
        ),
        probe=_has_robocasa_gr1,
        steps=(
            InstallStep(
                description=(
                    "uv sync --group robocasa --inexact (workspace deps + robosuite>=1.5; "
                    "--inexact preserves packages from sibling groups already in the venv "
                    "— e.g. pyzmq/msgpack from --group rldx — that would otherwise be "
                    "uninstalled because they are not in the robocasa group's resolved set)"
                ),
                argv=[
                    uv,
                    "sync",
                    "--all-packages",
                    "--group",
                    "robocasa",
                    "--inexact",
                ],
            ),
            InstallStep(
                description=f"mkdir -p {clone_parent} (cache for editable installs)",
                argv=["mkdir", "-p", str(clone_parent)],
            ),
            _robosuite_clone_step(git, rs_dir),
            InstallStep(
                description=(
                    "patch robosuite clone: touch missing __init__.py under "
                    "robosuite/examples/ + robosuite/examples/third_party_controller/ "
                    "(upstream find_packages() drops these; without them "
                    "`from robosuite.examples.third_party_controller.mink_controller "
                    "import WholeBodyMinkIK` fails at robosuite/__init__.py:37 and "
                    "logs WARN every run), and write empty macros_private.py "
                    "(robosuite/macros.py emits a 3-line WARN at import when "
                    "the sibling macros_private.py is missing)"
                ),
                argv=[
                    "bash",
                    "-c",
                    f"touch {rs_dir}/robosuite/examples/__init__.py "
                    f"{rs_dir}/robosuite/examples/third_party_controller/__init__.py "
                    f"{rs_dir}/robosuite/macros_private.py",
                ],
            ),
            InstallStep(
                description=(
                    f"uv pip install -e {rs_dir} "
                    "(editable so the __init__.py touches above survive)"
                ),
                argv=[
                    uv,
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-deps",
                    "-e",
                    str(rs_dir),
                ],
            ),
            _remove_editable_shadow_step("robosuite"),
            InstallStep(
                description="uv pip install robosuite-models",
                argv=[
                    uv,
                    "pip",
                    "install",
                    "--no-deps",
                    "robosuite-models @ git+https://github.com/ARISE-Initiative/robosuite_models.git",
                ],
            ),
            InstallStep(
                description=(
                    "uv pip install runtime extras "
                    "(h5py lxml llvmlite numba qpsolvers pyopengl-accelerate)"
                ),
                argv=[
                    uv,
                    "pip",
                    "install",
                    "h5py>=3.16",
                    "lxml>=5",
                    "llvmlite",
                    "numba",
                    # robosuite's WholeBodyMinkIK loader needs both
                    # mink (Python QP wrapper) and qpsolvers (the QP
                    # solver). Without them robosuite logs
                    # "Could not load the mink-based whole-body IK".
                    "qpsolvers",
                    # Optional perf module for PyOpenGL; silences the
                    # "No OpenGL_accelerate module loaded" INFO log on
                    # every offscreen-render setup.
                    "pyopengl-accelerate",
                ],
            ),
            InstallStep(
                description=(
                    "uv pip install mink==0.0.5 --no-deps. Pin to 0.0.5 because "
                    "robosuite's mink_controller.py imports "
                    "`from mink.tasks.exceptions import TargetNotSet`, a "
                    "path that newer mink (1.x) restructured. --no-deps "
                    "avoids forcing numpy<2 (mink 0.0.5's pin), which "
                    "would downgrade and break every other backend."
                ),
                argv=[uv, "pip", "install", "--no-deps", "mink==0.0.5"],
            ),
            InstallStep(
                description=(
                    f"git clone --depth=1 robocasa-gr1-tabletop-tasks → {clone_dir} "
                    "(idempotent: skipped if already cloned)"
                ),
                argv=[
                    "bash",
                    "-c",
                    f"[ -d {clone_dir}/.git ] || {git} clone --depth=1 "
                    "https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git "
                    f"{clone_dir}",
                ],
            ),
            InstallStep(
                description=(
                    "patch robocasa fork: write empty macros_private.py "
                    "(robocasa/macros.py emits a 3-line WARN at import when "
                    "the sibling macros_private.py is missing)"
                ),
                argv=[
                    "bash",
                    "-c",
                    f"touch {clone_dir}/robocasa/macros_private.py",
                ],
            ),
            InstallStep(
                description=(
                    f"uv pip install -e {clone_dir} (editable -- the fork's "
                    "setup.py uses find_packages() which drops robocasa.utils "
                    "under a non-editable install)"
                ),
                argv=[uv, "pip", "install", "--no-deps", "-e", str(clone_dir)],
            ),
            _remove_editable_shadow_step("robocasa"),
        ),
        manual_hint=(
            "just sync --all-packages --group robocasa --inexact && "
            f"mkdir -p {clone_parent} && "
            f"{_robosuite_clone_hint(git, rs_dir)} && "
            f"touch {rs_dir}/robosuite/examples/__init__.py "
            f"{rs_dir}/robosuite/examples/third_party_controller/__init__.py "
            f"{rs_dir}/robosuite/macros_private.py && "
            f"uv pip install --force-reinstall --no-deps -e {rs_dir} && "
            'uv pip install --no-deps "robosuite-models @ '
            'git+https://github.com/ARISE-Initiative/robosuite_models.git" && '
            'uv pip install "h5py>=3.16" "lxml>=5" llvmlite '
            "numba qpsolvers pyopengl-accelerate && "
            'uv pip install --no-deps "mink==0.0.5" && '
            "git clone --depth=1 "
            "https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git "
            f"{clone_dir} && "
            f"touch {clone_dir}/robocasa/macros_private.py && "
            f"uv pip install --no-deps -e {clone_dir}"
        ),
    )


def _libero_plan() -> BackendInstallPlan:
    uv = _uv()
    # LIBERO pulls hf-libero which builds C extensions transitively;
    # on systems where the default compiler is clang the build picks
    # up a flag set the C extension rejects. Pinning CC=gcc mirrors
    # the existing install hint and the bootstrap script.
    cc = shutil.which("gcc") or "/usr/bin/gcc"
    # hf-libero==0.1.3 ships distutils-installed metadata with no RECORD.
    # We deliberately do NOT pass `--reinstall-package hf-libero`: that
    # forces uv to *uninstall* hf-libero first, hitting the very barrier
    # it was meant to dodge (`error: Unable to uninstall hf-libero==0.1.3:
    # distutils-installed distributions do not include the metadata
    # required to uninstall safely`) and wedging the whole libero install
    # — exactly what happens when swapping in from a robocasa (robosuite
    # 1.5) venv. A plain `--inexact` sync installs/overwrites hf-libero
    # with proper dist-info when it's absent and leaves it untouched when
    # already satisfied; it never forces an uninstall, so the barrier
    # never fires. hf-libero is pure-python — robosuite owns the C
    # extensions and is version-swapped separately — so no forced rebuild
    # is needed. --inexact preserves other backend groups in the venv.
    libero_args = [
        uv,
        "sync",
        "--all-packages",
        "--group",
        "libero",
        "--inexact",
    ]
    return BackendInstallPlan(
        backend_id="libero",
        display_name="LIBERO (10/object/spatial/goal suites, Franka panda)",
        license_note=(
            "Pulls lerobot[libero] (Apache-2.0) which transitively bundles "
            "the LIBERO bddl task files and robosuite==1.4. Requires gcc "
            "for the robosuite C extensions; CC=gcc is forced for this run."
        ),
        probe=_has_libero,
        steps=(
            InstallStep(
                description=(
                    "CC=gcc uv sync --group libero --inexact (compiles robosuite==1.4 "
                    "C extensions; --inexact preserves other backend groups already in "
                    "the venv and avoids the hf-libero distutils-uninstall barrier that "
                    "--reinstall-package hf-libero would trigger)"
                ),
                argv=libero_args,
                env={"CC": cc},
            ),
        ),
        manual_hint=(f"CC={cc} just sync --all-packages --group libero --inexact"),
    )


_SIMPLER_ENV_GIT_URL = "simpler-env @ git+https://github.com/simpler-env/SimplerEnv.git@maniskill3"


def _simpler_env_plan() -> BackendInstallPlan:
    uv = _uv()
    return BackendInstallPlan(
        backend_id="simpler_env",
        display_name="SimplerEnv (real-to-sim correlator: WidowX / Google Robot bridge tasks)",
        license_note=(
            "Pulls ManiSkill3 (Apache-2.0) + SAPIEN + gymnasium via the "
            "`simpler-env` extras group, then installs `simpler-env` itself "
            "from its `maniskill3` git branch (MIT, no PyPI release)."
        ),
        probe=_has_simpler_env,
        steps=(
            InstallStep(
                description=(
                    "uv sync --all-packages --group simpler-env --inexact "
                    "(pulls ManiSkill3 + gymnasium; --inexact preserves other "
                    "backend groups already in the venv)"
                ),
                argv=[
                    uv,
                    "sync",
                    "--all-packages",
                    "--group",
                    "simpler-env",
                    "--inexact",
                ],
            ),
            # `simpler-env` has no PyPI release. We use `uv pip install`
            # rather than `uv run pip install`: pip-on-uv-venvs silently
            # exits 0 after "Preparing metadata" without ever building
            # the wheel for this package, while uv's native pip shim
            # handles the PEP 517 build correctly.
            InstallStep(
                description=(
                    f"uv pip install '{_SIMPLER_ENV_GIT_URL}' "
                    "(no PyPI release; pinned to the maniskill3 branch where "
                    "the bridge tasks are ManiSkill3-registered)"
                ),
                argv=[uv, "pip", "install", _SIMPLER_ENV_GIT_URL],
            ),
        ),
        manual_hint=(
            "just sync --all-packages --group simpler-env --inexact && "
            f"uv pip install '{_SIMPLER_ENV_GIT_URL}'"
        ),
    )


def _has_rldx_client() -> bool:
    """RLDX client side needs pyzmq + msgpack on the openral venv.

    Sidecar venv (Python 3.10) has its own copy; this probe is for the
    Python 3.12 workspace where the adapter lives.
    """
    return _has_module("zmq") and _has_module("msgpack")


def _has_isaac_client() -> bool:
    """Isaac sidecar client side needs pyzmq + msgpack on the openral venv.

    The heavy ``isaacsim`` / ``isaaclab`` install lives in a separate py3.11
    sidecar venv (ADR-0045), provisioned out-of-band; this probe only covers the
    openral-side wire, same shape as :func:`_has_rldx_client`.
    """
    return _has_module("zmq") and _has_module("msgpack")


def _isaac_client_plan() -> BackendInstallPlan:
    uv = _uv()
    return BackendInstallPlan(
        backend_id="isaac_client",
        display_name="Isaac Sim adapter wire (pyzmq + msgpack on the openral venv)",
        license_note=(
            "Pulls pyzmq (LGPL+ZeroMQ exception → effectively permissive) and "
            "msgpack (Apache-2.0). The Isaac Sim / Isaac Lab sidecar itself is an "
            "externally-provisioned py3.11 venv (NVIDIA Omniverse Kit components "
            "are proprietary, non-redistributable; ADR-0045 / CLAUDE.md §1.9) and "
            "is NOT installed by this plan."
        ),
        probe=_has_isaac_client,
        steps=(
            InstallStep(
                description=(
                    "uv sync --group isaacsim --inexact (adds pyzmq + msgpack to the "
                    "openral venv; --inexact keeps other backend deps in place)"
                ),
                argv=[
                    uv,
                    "sync",
                    "--all-packages",
                    "--group",
                    "isaacsim",
                    "--inexact",
                ],
            ),
        ),
        manual_hint="just sync --all-packages --group isaacsim --inexact",
    )


def _has_robotwin_client() -> bool:
    """RoboTwin sidecar client side needs pyzmq + msgpack on the openral venv.

    The heavy SAPIEN + RoboTwin install lives in a separate py3.10 sidecar venv
    (ADR-0061), provisioned out-of-band; this probe only covers the openral-side
    wire, same shape as :func:`_has_isaac_client`.
    """
    return _has_module("zmq") and _has_module("msgpack")


def _has_rlbench_client() -> bool:
    """RLBench sidecar client side needs pyzmq + msgpack on the openral venv.

    CoppeliaSim/PyRep + the peract RLBench fork live in a separate py3.10 sidecar
    venv (ADR-0062), provisioned out-of-band (CoppeliaSim is proprietary and never
    vendored); this probe only covers the openral-side wire, same shape as
    :func:`_has_isaac_client`.
    """
    return _has_module("zmq") and _has_module("msgpack")


def _robotwin_client_plan() -> BackendInstallPlan:
    uv = _uv()
    return BackendInstallPlan(
        backend_id="robotwin_client",
        display_name="RoboTwin adapter wire (pyzmq + msgpack on the openral venv)",
        license_note=(
            "Pulls pyzmq (LGPL+ZeroMQ exception → effectively permissive) and "
            "msgpack (Apache-2.0). The SAPIEN + RoboTwin 2.0 sidecar itself is an "
            "externally-provisioned py3.10 venv (SAPIEN/RoboTwin/CuRobo are large + "
            "CUDA-12.1-pinned; ADR-0061 / CLAUDE.md §1.9) and is NOT installed by "
            "this plan. RoboTwin is MIT-licensed."
        ),
        probe=_has_robotwin_client,
        steps=(
            InstallStep(
                description=(
                    "uv sync --group robotwin --inexact (adds pyzmq + msgpack to the "
                    "openral venv; --inexact keeps other backend deps in place)"
                ),
                argv=[
                    uv,
                    "sync",
                    "--all-packages",
                    "--group",
                    "robotwin",
                    "--inexact",
                ],
            ),
        ),
        manual_hint="just sync --all-packages --group robotwin --inexact",
    )


def _rlbench_client_plan() -> BackendInstallPlan:
    uv = _uv()
    return BackendInstallPlan(
        backend_id="rlbench_client",
        display_name="RLBench adapter wire (pyzmq + msgpack on the openral venv)",
        license_note=(
            "Pulls pyzmq (LGPL+ZeroMQ exception → effectively permissive) and "
            "msgpack (Apache-2.0). The CoppeliaSim/PyRep + peract-RLBench sidecar "
            "(plus the 3D Diffuser Actor checkpoint) is an externally-provisioned "
            "py3.10 venv — CoppeliaSim is proprietary, free-EDU, NEVER vendored "
            "(ADR-0062 / CLAUDE.md §1.9) — and is NOT installed by this plan."
        ),
        probe=_has_rlbench_client,
        steps=(
            InstallStep(
                description=(
                    "uv sync --group rlbench --inexact (adds pyzmq + msgpack to the "
                    "openral venv; --inexact keeps other backend deps in place)"
                ),
                argv=[uv, "sync", "--all-packages", "--group", "rlbench", "--inexact"],
            ),
        ),
        manual_hint="just sync --all-packages --group rlbench --inexact",
    )


def _rldx_client_plan() -> BackendInstallPlan:
    uv = _uv()
    return BackendInstallPlan(
        backend_id="rldx_client",
        display_name="RLDX adapter wire (pyzmq + msgpack on the openral venv)",
        license_note=(
            "Pulls pyzmq (LGPL+ZeroMQ exception → effectively permissive) "
            "and msgpack (Apache-2.0). Sidecar weights/license posture "
            "(non-commercial) is gated separately at rSkill load time."
        ),
        probe=_has_rldx_client,
        steps=(
            InstallStep(
                description=(
                    "uv sync --group rldx --inexact (adds pyzmq + msgpack to the "
                    "openral venv; --inexact keeps other backend deps in place)"
                ),
                argv=[
                    uv,
                    "sync",
                    "--all-packages",
                    "--group",
                    "rldx",
                    "--inexact",
                ],
            ),
        ),
        manual_hint="just sync --all-packages --group rldx --inexact",
    )


# ── RLDX sidecar source + Python 3.10 venv ───────────────────────────────────
#
# These constants mirror ``tools/rldx_sidecar.py`` (which still owns the
# server launch path) so the install plan and the launcher agree on
# where things live on disk. Keep them in sync if the sidecar layout
# ever changes.
_RLDX_SIDECAR_REPO_URL = "https://github.com/RLWRLD/RLDX-1.git"
_RLDX_SIDECAR_HOME = _DEFAULT_CACHE_HOME / "rldx-sidecar"


def _rldx_sidecar_source_dir() -> Path:
    """Stable on-disk path for the RLDX-1 sidecar clone.

    Matches ``tools/rldx_sidecar.py:_ensure_source``. The directory is
    used as a uv project root — its own ``pyproject.toml`` produces a
    sibling ``.venv`` on first ``uv sync``. Moving this path would
    orphan the (~6 GB) sidecar venv.
    """
    home = os.environ.get("OPENRAL_CACHE_HOME")
    base = Path(home) / "rldx-sidecar" if home else _RLDX_SIDECAR_HOME
    return base / "source"


def _has_rldx_sidecar_setup() -> bool:
    """RLDX sidecar source clone + Python 3.10 venv + bitsandbytes are on disk.

    All three are prerequisites for ``tools/rldx_sidecar.py`` to launch
    the server without paying the multi-minute install cost on first
    use. We probe each one explicitly so a half-completed setup
    (clone-but-no-venv, venv-but-no-bnb) re-triggers the plan.
    """
    source = _rldx_sidecar_source_dir()
    if not (source / ".git").is_dir():
        return False
    venv_python = source / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        return False
    # bitsandbytes is not in the upstream RLDX-1 lockfile; the launcher
    # adds it via a follow-up ``uv pip install``. Probe its on-disk
    # presence in the sidecar venv (we cannot ``import`` it from this
    # Python 3.12 process — wrong ABI).
    bnb_dist = source / ".venv" / "lib"
    if not bnb_dist.is_dir():
        return False
    return any(p.name.startswith("bitsandbytes") for p in bnb_dist.rglob("bitsandbytes*"))


def _rldx_sidecar_setup_plan() -> BackendInstallPlan:
    """Install the RLDX-1 sidecar source + Python 3.10 venv up-front.

    The sidecar launcher (``tools/rldx_sidecar.py``) is currently doing
    the git clone + ``uv sync`` lazily on the first ``openral sim run`` —
    the user sees ~80 s of subprocess output (clone + ~6 GB of CUDA
    wheel downloads) before the policy even starts loading. Moving
    those steps behind ``ensure_backend_deps`` gives the same one-prompt
    UX as LIBERO / RoboCasa / simpler-env, fails fast on network errors,
    and leaves the launcher's existing idempotent short-circuits as a
    safety net (a second clone / sync on launch is a no-op).

    The launcher still owns the per-launch wrapper write + server boot.
    What this plan does NOT shrink is the per-launch ~65 s of Python
    startup + transformers import + Qwen3-VL load + NF4 quantize —
    those live inside the upstream RLDX-1 codebase.
    """
    uv = _uv()
    git = _git()
    source = _rldx_sidecar_source_dir()
    venv_python = source / ".venv" / "bin" / "python"
    return BackendInstallPlan(
        backend_id="rldx_sidecar_setup",
        display_name="RLDX-1 sidecar source + Python 3.10 venv (~6 GB CUDA deps)",
        license_note=(
            "Clones https://github.com/RLWRLD/RLDX-1 (Apache-2.0 + per-checkpoint "
            "non-commercial weight licenses; the rSkill loader enforces the latter "
            "via OPENRAL_ALLOW_NONCOMMERCIAL). Builds an isolated Python 3.10 "
            "venv at <source>/.venv with the upstream lockfile + bitsandbytes."
        ),
        probe=_has_rldx_sidecar_setup,
        steps=(
            InstallStep(
                description=f"mkdir -p {source.parent} (sidecar work directory)",
                argv=["mkdir", "-p", str(source.parent)],
            ),
            # Idempotent: only clone if the .git directory is absent.
            # Mirrors ``tools/rldx_sidecar.py:_ensure_source`` — we run
            # this here so the slow first-run download happens behind
            # one typer.confirm() rather than mid-rollout.
            InstallStep(
                description=(
                    f"git clone --depth 1 {_RLDX_SIDECAR_REPO_URL} → {source} "
                    "(idempotent: skipped when .git already exists)"
                ),
                argv=[
                    "sh",
                    "-c",
                    f"[ -d {source}/.git ] || {git} clone --depth 1 "
                    f"{_RLDX_SIDECAR_REPO_URL} {source}",
                ],
            ),
            # uv sync inside the sidecar's own pyproject creates
            # <source>/.venv (Python 3.10) with the full RLDX-1 + CUDA
            # stack. The upstream lockfile pulls flash-attn, transformers,
            # torch, etc. -- ~6 GB on a clean cache.
            InstallStep(
                description=(
                    f"uv sync in {source} (builds sidecar venv at <source>/.venv "
                    "with the upstream Python 3.10 + CUDA dep set)"
                ),
                argv=[uv, "sync"],
                cwd=source,
            ),
            # bitsandbytes is not in the upstream lockfile; the launcher
            # adds it for the NF4 path. We pin it here so the launcher's
            # post-clone install becomes a fast no-op on subsequent
            # boots. ``--python <venv>/bin/python`` keeps uv from
            # routing the install to this Python 3.12 workspace by
            # mistake.
            InstallStep(
                description=(
                    f"uv pip install --python {venv_python} bitsandbytes>=0.43.0 "
                    "(NF4 quantization dep; not in the upstream RLDX-1 lockfile)"
                ),
                argv=[
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(venv_python),
                    "bitsandbytes>=0.43.0",
                ],
                cwd=source,
            ),
        ),
        manual_hint=(
            f"mkdir -p {source.parent} && "
            f"git clone --depth 1 {_RLDX_SIDECAR_REPO_URL} {source} && "
            f"cd {source} && uv sync && "
            f"uv pip install --python {venv_python} 'bitsandbytes>=0.43.0'"
        ),
    )


def _maniskill3_plan() -> BackendInstallPlan:
    uv = _uv()
    return BackendInstallPlan(
        backend_id="maniskill3",
        display_name="ManiSkill3 (Apache-2.0 SAPIEN-based GPU physics + manipulation tasks)",
        license_note=(
            "Pulls mani-skill (Apache-2.0) which transitively bundles "
            "SAPIEN (Vulkan-backed GPU physics, ~200 MB wheel) and gymnasium. "
            "Also carries lerobot + transformers + accelerate + num2words so "
            "every in-tree policy adapter (SmolVLA / π0.5 / xVLA / ACT / "
            "diffusion) can load against an MS3 scene."
        ),
        probe=_has_maniskill3,
        steps=(
            InstallStep(
                description=(
                    "uv sync --all-packages --group maniskill3 --inexact "
                    "(pulls mani-skill + SAPIEN + lerobot + transformers; "
                    "--inexact preserves other backend groups already in the venv)"
                ),
                argv=[
                    uv,
                    "sync",
                    "--all-packages",
                    "--group",
                    "maniskill3",
                    "--inexact",
                ],
            ),
        ),
        manual_hint="just sync --all-packages --group maniskill3 --inexact",
    )


def _aloha_plan() -> BackendInstallPlan:
    uv = _uv()
    return BackendInstallPlan(
        backend_id="aloha",
        display_name="gym-aloha (Apache-2.0 MuJoCo bimanual ALOHA env: cube transfer, insertion)",
        license_note=(
            "Pulls gym-aloha (Apache-2.0) via the `sim` group, which also "
            "carries mujoco, gymnasium[mujoco], transformers, and "
            "bitsandbytes (the standard sim baseline used by ACT / "
            "diffusion / SmolVLA / π0.5 adapters)."
        ),
        probe=_has_aloha,
        steps=(
            InstallStep(
                description=(
                    "uv sync --all-packages --group sim --inexact "
                    "(pulls gym-aloha + mujoco + lerobot baseline; "
                    "--inexact preserves other backend groups already in the venv)"
                ),
                argv=[
                    uv,
                    "sync",
                    "--all-packages",
                    "--group",
                    "sim",
                    "--inexact",
                ],
            ),
        ),
        manual_hint="just sync --all-packages --group sim --inexact",
    )


def _metaworld_plan() -> BackendInstallPlan:
    uv = _uv()
    return BackendInstallPlan(
        backend_id="metaworld",
        display_name="MetaWorld MT-50 (MIT — Sawyer manipulation benchmark suite)",
        license_note=(
            "Pulls lerobot (Apache-2.0) + transformers + mujoco via the "
            "`metaworld` group, then installs the MetaWorld benchmark "
            "package (MIT) pinned to ==3.0.0 with --no-deps because its "
            "transitive dep set conflicts with the workspace lock."
        ),
        probe=_has_metaworld,
        steps=(
            InstallStep(
                description=(
                    "uv sync --all-packages --group metaworld --inexact "
                    "(pulls lerobot + transformers + mujoco; --inexact preserves "
                    "other backend groups already in the venv)"
                ),
                argv=[
                    uv,
                    "sync",
                    "--all-packages",
                    "--group",
                    "metaworld",
                    "--inexact",
                ],
            ),
            # MetaWorld's own deps conflict with the workspace lock; the
            # documented workaround is a --no-deps pip install. We use
            # `uv pip install` rather than `uv run pip install` for the
            # same reason as the simpler_env plan: `uv run pip` silently
            # skips PEP 517 build for git-installed wheel-less packages.
            InstallStep(
                description=(
                    "uv pip install metaworld==3.0.0 --no-deps "
                    "(transitive deps conflict with the workspace lock — "
                    "the lerobot.envs.metaworld wrapper only needs the "
                    "task suite itself)"
                ),
                argv=[uv, "pip", "install", "metaworld==3.0.0", "--no-deps"],
            ),
        ),
        manual_hint=(
            "just sync --all-packages --group metaworld --inexact && "
            "uv pip install metaworld==3.0.0 --no-deps"
        ),
    )


def _openarm_robosuite_plan() -> BackendInstallPlan:
    """OpenArm v2 tabletop scene — needs ``robosuite>=1.5`` from the ``robocasa`` group.

    The scene uses ``robosuite.utils.binding_utils.MjSim`` as a thin MJCF
    wrapper around a hand-composed bimanual scene; it does NOT use
    robocasa envs, OSC composite controllers, or WholeBodyMinkIK. The
    only place ``robosuite>=1.5`` is declared in the workspace is the
    ``[project.optional-dependencies] robocasa`` extra, so we reuse
    that group — leaner steps than ``_robocasa_kitchen_plan`` (no
    robosuite-master clone, no mink / qpsolvers, no robocasa repo).

    Mutual exclusion with LIBERO (which pins ``robosuite==1.4``) is
    enforced by uv's solver, not here; the probe pins ``>=1.5`` so a
    libero-installed venv re-triggers the plan and the user sees the
    swap.
    """
    uv = _uv()
    return BackendInstallPlan(
        backend_id="openarm_robosuite",
        display_name="OpenArm v2 tabletop (robosuite>=1.5 MJCF wrapper, no robocasa envs)",
        license_note=(
            "Pulls robosuite>=1.5 (MIT) via the `robocasa` extras group. "
            "Mutually exclusive with the `libero` group (robosuite==1.4 pin) "
            "in a single venv — see ADR-0011."
        ),
        probe=_has_openarm_robosuite,
        steps=(
            InstallStep(
                description=(
                    "uv sync --all-packages --group robocasa --inexact "
                    "(pulls robosuite>=1.5 + mujoco; --inexact preserves "
                    "other backend groups already in the venv)"
                ),
                argv=[
                    uv,
                    "sync",
                    "--all-packages",
                    "--group",
                    "robocasa",
                    "--inexact",
                ],
            ),
        ),
        manual_hint="just sync --all-packages --group robocasa --inexact",
    )


_PLANS: dict[str, Callable[[], BackendInstallPlan]] = {
    "robocasa_kitchen": _robocasa_kitchen_plan,
    "robocasa_gr1": _robocasa_gr1_plan,
    "libero": _libero_plan,
    "simpler_env": _simpler_env_plan,
    "maniskill3": _maniskill3_plan,
    "aloha": _aloha_plan,
    "metaworld": _metaworld_plan,
    "openarm_robosuite": _openarm_robosuite_plan,
    "rldx_client": _rldx_client_plan,
    "rldx_sidecar_setup": _rldx_sidecar_setup_plan,
    "isaac_client": _isaac_client_plan,
    "rlbench_client": _rlbench_client_plan,
    "robotwin_client": _robotwin_client_plan,
}


def get_plan(backend_id: str) -> BackendInstallPlan:
    """Resolve a backend id to its install plan, building lazily.

    Raises:
        KeyError: If ``backend_id`` is not registered. The set of
            registered ids is a deliberate closed enum so a typo in a
            backend's adapter surfaces at adapter-load time, not at
            install time.
    """
    if backend_id not in _PLANS:
        raise KeyError(f"unknown backend_id {backend_id!r}; registered: {sorted(_PLANS)}")
    return _PLANS[backend_id]()


# ── runner ───────────────────────────────────────────────────────────────────


def _display_install_banner(plan: BackendInstallPlan) -> None:
    body = (
        f"[bold]{plan.display_name}[/bold]\n"
        f"License: {plan.license_note}\n"
        "\n"
        "Plan ([cyan]each step runs as a subprocess[/cyan]):\n"
    )
    for idx, step in enumerate(plan.steps, start=1):
        env_note = (
            ""
            if not step.env
            else " (env: " + " ".join(f"{k}={v}" for k, v in step.env.items()) + ")"
        )
        body += f"  {idx}. {step.description}{env_note}\n"
    body += f"\nSet {_AUTO_INSTALL_ENV}=0 to suppress auto-install."
    Console().print(
        Panel.fit(
            body,
            title=f"{plan.backend_id} — install on first use",
            border_style="yellow",
        )
    )


def ensure_backend_deps(backend_id: str) -> None:
    """Install ``backend_id`` deps if missing, after the user confirms.

    Probes via the plan's :attr:`BackendInstallPlan.probe` and short-
    circuits when it returns ``True``. Otherwise prints a Rich banner
    listing every subprocess step + the license posture, asks
    auto-installs by default (set ``OPENRAL_AUTO_INSTALL_DEPS=0`` to
    prompt via :func:`typer.confirm` instead), runs the steps in order,
    and re-probes.

    Thread-safe: ``SimRunner._build_env_and_policy`` builds env + policy
    on a 2-worker ``ThreadPoolExecutor``. When both sides need a first-
    install the two banners + ``typer.confirm`` reads would interleave on
    one stdout/stdin pair and the single ``y`` would only land in one of
    the two prompts. The module-level :data:`_INSTALL_LOCK` serialises
    the probe-prompt-install-reprobe sequence so the second waiter sees
    the first install's result and short-circuits when its own probe
    flips to ``True``.

    Raises:
        ROSConfigError: When the user refuses, ``uv`` / ``git`` is
            missing, a step exits non-zero, or the post-install probe
            still fails. The error message embeds
            :attr:`BackendInstallPlan.manual_hint` so the user has a
            ready-to-paste fallback.
    """
    plan = get_plan(backend_id)
    if plan.probe():
        return

    # Serialise the prompt + install so concurrent callers from the env-
    # builder + policy-builder threads in ``SimRunner._build_env_and_policy``
    # do not interleave Rich banners + ``typer.confirm`` reads on stdin —
    # see ``_INSTALL_LOCK``. Re-probe inside the lock so the second waiter
    # short-circuits when the first thread already finished the install.
    with _INSTALL_LOCK:
        if plan.probe():
            return

        bypass = os.environ.get(_AUTO_INSTALL_ENV, "1") == "1"
        if not bypass:
            _display_install_banner(plan)
            if not typer.confirm(f"Install {plan.backend_id} deps now?", default=False):
                raise ROSConfigError(
                    f"{plan.backend_id} backend not installed. Either set "
                    f"{_AUTO_INSTALL_ENV}=1 and re-run, or install manually:\n  " + plan.manual_hint
                )

        for step in plan.steps:
            run_env = {**os.environ, **step.env}
            try:
                subprocess.run(
                    step.argv,
                    env=run_env,
                    cwd=str(step.cwd) if step.cwd else None,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                raise ROSConfigError(
                    f"{plan.backend_id} install failed at step "
                    f"{step.description!r}: {exc}. Finish manually:\n  " + plan.manual_hint
                ) from exc

        # Some packages need importlib's caches invalidated for newly-installed
        # site-packages to be discoverable in this same process.
        importlib.invalidate_caches()
        # `uv pip install -e` swaps an editable install via the
        # setuptools-editable .pth + finder shim. The old _EditableFinder
        # stays on sys.meta_path with a stale MAPPING dict baked in at
        # import time — invalidate_caches() does not refresh meta_path —
        # so find_spec(<pkg>) keeps returning the old source directory.
        # Drop dead finders and re-process the current .pth shims.
        _refresh_editable_finders()
        # Drop any cached sys.modules entries for the packages we just
        # (re)installed. Without this, `find_spec("robosuite.examples")`
        # keeps returning None on the post-install probe because the
        # pre-install attempt cached a negative result; flushing
        # sys.modules forces a fresh resolve against the now-present
        # site-packages tree.
        import sys

        # `pop(..., None)` is idempotent — required because a module like
        # ``robosuite.examples`` matches BOTH the exact ``"robosuite.examples"``
        # entry AND the ``mod_name.startswith("robosuite" + ".")`` check, so the
        # earlier ``del sys.modules[mod_name]`` raised ``KeyError`` on the second
        # match and aborted the cache flush midway. Breaking after the first match
        # would mask the intent (we still want every dep checked); ``pop`` is
        # the minimal robust fix.
        flush_prefixes = (*_ROBOCASA_RUNTIME_DEPS, "robocasa")
        for mod_name in list(sys.modules):
            for dep in flush_prefixes:
                if mod_name == dep or mod_name.startswith(dep + "."):
                    sys.modules.pop(mod_name, None)
                    break

        if not plan.probe():
            raise ROSConfigError(
                f"{plan.backend_id} install ran to completion but the probe "
                f"still fails. Inspect the install output above and finish "
                f"manually:\n  " + plan.manual_hint
            )

        # Post-install: silence the "No private macro file found" warnings
        # by creating the empty `macros_private.py` files robosuite and
        # robocasa expect. Idempotent -- skipped when the files already
        # exist (e.g. on second run after a prior auto-install).
        if backend_id in {"robocasa_kitchen", "robocasa_gr1"}:
            _ensure_robocasa_macros_private()


def _ensure_robocasa_macros_private() -> None:
    """Create empty ``macros_private.py`` for robosuite + robocasa.

    Both packages emit a 3-line WARN at import time when the
    ``macros_private.py`` sibling of their ``macros.py`` is missing,
    nagging the user to run ``setup_macros.py``. That upstream script
    just copies ``macros.py`` to ``macros_private.py`` verbatim; an
    empty Python module suffices since ``macros.py`` does
    ``from .macros_private import *`` only as a soft override. Doing
    this in-process avoids spawning yet another subprocess and works
    even when the upstream setup_macros script has a sibling-import
    bug (the GR1 fork's robocasa.scripts.setup_macros does not exist
    on disk despite the warning suggesting it).
    """
    for pkg_name in ("robosuite", "robocasa"):
        spec = importlib.util.find_spec(pkg_name)
        if spec is None or spec.origin is None:
            continue
        pkg_dir = Path(spec.origin).parent
        # Sanity check: only write into a real package directory (must
        # contain __init__.py). Without this guard we'd write
        # `macros_private.py` into a leftover stub directory in
        # site-packages -- which would then look like a namespace
        # package and mask the editable install's real
        # __init__.py.
        if not (pkg_dir / "__init__.py").is_file():
            continue
        target = pkg_dir / "macros_private.py"
        if target.is_file():
            continue
        # Read-only site-packages or permission denied -- leave the
        # warning intact; not worth raising.
        with contextlib.suppress(OSError):
            target.write_text(
                "# Auto-generated by openral_sim._deps to silence the "
                f"'No private macro file found' nag from {pkg_name}.\n"
                "# Override any default macro here.\n"
            )
