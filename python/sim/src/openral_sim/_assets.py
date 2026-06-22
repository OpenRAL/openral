"""Lazy asset-fetch helpers for sim backends with large CC-BY downloads.

Today the only consumer is the RoboCasa adapter; the module
is named generically so a future MuJoCo backend that ships its own
~GB asset bundle can reuse the same readiness-sentinel + license-banner
pattern without re-inventing it.

Conventions
-----------
- Asset caches live under ``$OPENRAL_CACHE_HOME`` (defaults to
  ``~/.cache/openral/``), mirroring the rSkill cache convention in
  ``openral_rskill.loader``.
- Each backend gets its own subdirectory (``<cache_home>/robocasa/``).
- A readiness sentinel file (``.openral-ready``) marks "assets
  fully unpacked"; the helper short-circuits on subsequent runs.
- A Rich license banner surfaces the upstream license + URL + target
  path + size *before* asking the user to confirm.
- An env-var bypass (``OPENRAL_ALLOW_ROBOCASA_ASSETS=1`` for
  RoboCasa) skips the prompt for CI.

Refusal raises :class:`ROSConfigError` with the manual-fetch command so
users can authorise the download out-of-band.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer
from openral_core.exceptions import ROSConfigError
from rich.console import Console
from rich.panel import Panel

# ── Cache layout ──────────────────────────────────────────────────────────────


_DEFAULT_CACHE_HOME = Path.home() / ".cache" / "openral"
_READY_SENTINEL = ".openral-ready"

_ROBOCASA_ALLOW_ENV = "OPENRAL_ALLOW_ROBOCASA_ASSETS"
_ROBOCASA_ASSETS_SIZE_GB = 11  # upstream README says ~10-11 GB depending on version


def _cache_home() -> Path:
    """Resolve the openral cache root from env or its default."""
    raw = os.environ.get("OPENRAL_CACHE_HOME")
    return Path(raw) if raw else _DEFAULT_CACHE_HOME


def _display_robocasa_license_banner(target: Path) -> None:
    """Print a Rich license panel before the typer.confirm() prompt.

    Separate function so unit tests can monkeypatch the prompt without
    suppressing the user-visible banner (the banner is a UX
    requirement, not implementation detail).
    """
    Console().print(
        Panel.fit(
            (
                "[bold]RoboCasa kitchen assets[/bold]\n"
                "License: [bold]CC-BY-4.0[/bold] "
                "(https://creativecommons.org/licenses/by/4.0/)\n"
                f"Target:  {target}\n"
                f"Approx.  ~{_ROBOCASA_ASSETS_SIZE_GB} GB on disk\n"
                "\n"
                "By continuing you accept the upstream RoboCasa CC-BY-4.0\n"
                "license for the kitchen asset pack. Derivative artefacts\n"
                "(videos, traces) must carry attribution. To skip this\n"
                f"prompt in CI export {_ROBOCASA_ALLOW_ENV}=1."
            ),
            title="RoboCasa asset download (first-use)",
            border_style="yellow",
        )
    )


def _detect_robocasa_variant() -> str:
    """Return ``"kitchen"`` or ``"gr1_tabletop"`` based on the installed robocasa.

    The two python packages share the name ``robocasa`` so a host
    installs ONE or the OTHER. We pick the variant by which download
    script the package ships:

    * Kitchen (github.com/robocasa/robocasa) — ``download_kitchen_assets``.
    * GR1 tabletop fork (github.com/robocasa/robocasa-gr1-tabletop-tasks)
      — ``download_tabletop_assets`` (plus the kitchen one as a sibling
      since the fork is a soft fork of robocasa).

    Detection is purely filesystem-based so we don't trigger robocasa's
    own import-time version assertions. If neither script is present
    we default to ``"kitchen"`` -- the historical behaviour -- and the
    subsequent subprocess call will surface a typed ImportError.
    """
    try:
        import robocasa  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError:
        return "kitchen"
    scripts = Path(robocasa.__file__).parent / "scripts"
    if (scripts / "download_tabletop_assets.py").is_file():
        return "gr1_tabletop"
    return "kitchen"


def ensure_robocasa_assets() -> Path:  # noqa: PLR0915  # reason: orchestrates two variants (kitchen + gr1) each with a short-circuit + a download subprocess path; splitting hurts readability more than the length does
    """Make sure the RoboCasa assets are on disk; download if needed.

    Handles both robocasa variants -- the upstream kitchen package and
    the GR1 tabletop fork -- by detecting which is installed and
    invoking the matching download script. The version asserts are
    relaxed in the editable clone at provision time
    (``_deps._relax_robocasa_version_asserts_step``), so no version
    spoof is needed here.

    The first call triggers a Rich license banner + ``typer.confirm()``
    prompt unless ``OPENRAL_ALLOW_ROBOCASA_ASSETS=1`` is set. On
    confirm we run the variant's download script and touch the
    readiness sentinel. Subsequent calls are silent.

    Returns:
        Path to the ``<cache_home>/robocasa/`` directory.

    Raises:
        ROSConfigError: When the user refuses the prompt OR when the
            upstream downloader fails (with the exact subprocess
            stderr embedded so the user can debug).
    """
    # Variant-scoped sentinel. A previous shared `<cache>/robocasa/.openral-ready`
    # was incorrect: the kitchen and GR1 variants ship different `models/assets/`
    # trees and live in different `import robocasa` paths, so a sentinel
    # touched by the GR1 short-circuit (line ~152) silently masked a
    # missing kitchen asset bundle on the next swap to the kitchen
    # backend — and vice versa. Per-variant cache dirs decouple them.
    variant = _detect_robocasa_variant()
    target = _cache_home() / f"robocasa_{variant}"
    sentinel = target / _READY_SENTINEL
    if sentinel.is_file():
        return target

    # GR1 fork: the user runs the upstream
    # `python robocasa/scripts/download_tabletop_assets.py -y` once at
    # install time per the fork's README (because that script does a
    # sibling import `from download_groot_assets import …` that only
    # resolves when run from inside the scripts/ directory -- it does
    # NOT survive `python -m` invocation). Detect already-downloaded
    # assets in the editable install's models/assets/objects/ and
    # short-circuit with a sentinel touch so subsequent runs are silent.
    if variant == "gr1_tabletop":
        try:
            import robocasa  # type: ignore[import-not-found,import-untyped,unused-ignore]

            assert robocasa.__file__ is not None
            robocasa_dir = Path(robocasa.__file__).parent
            objects_dir = robocasa_dir / "models" / "assets" / "objects"
            if objects_dir.is_dir() and any(objects_dir.iterdir()):
                target.mkdir(parents=True, exist_ok=True)
                sentinel.touch()
                return target
        except ImportError:
            # fall through; the deps path will reinstall robocasa and
            # this function gets called again on the next env build.
            raise ROSConfigError(
                "RoboCasa GR1 tabletop adapter requires the editable "
                "fork clone; openral_sim._deps.ensure_backend_deps "
                "('robocasa_gr1') installs it. Re-run the same command "
                "and accept the deps-install prompt."
            ) from None

        # Drive the upstream downloader as a subprocess with the
        # script's parent dir as cwd (the upstream script does a
        # sibling import `from download_groot_assets import …` that
        # only resolves when invoked from inside scripts/ -- it does
        # NOT survive `python -m`). Bypass the prompt with the same
        # env-var that gates the kitchen path.
        bypass_gr1 = (
            os.environ.get(_ROBOCASA_ALLOW_ENV) == "1"
            or os.environ.get("OPENRAL_AUTO_INSTALL_DEPS") == "1"
        )
        if not bypass_gr1:
            _display_robocasa_license_banner(target)
            if not typer.confirm("Download RoboCasa GR1 tabletop assets now?", default=False):
                raise ROSConfigError(
                    "RoboCasa GR1 tabletop assets not downloaded. Either "
                    f"set {_ROBOCASA_ALLOW_ENV}=1 (or "
                    "OPENRAL_AUTO_INSTALL_DEPS=1) and re-run, or fetch "
                    "them manually:\n"
                    f"  cd {robocasa_dir.parent} && uv run python "
                    "robocasa/scripts/download_tabletop_assets.py -y"
                )

        scripts_dir = robocasa_dir / "scripts"
        try:
            subprocess.run(
                [sys.executable, "download_tabletop_assets.py", "-y"],
                cwd=str(scripts_dir),
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise ROSConfigError(
                f"RoboCasa GR1 asset download failed: {exc}. Re-run "
                f"manually:\n  cd {robocasa_dir.parent} && uv run python "
                "robocasa/scripts/download_tabletop_assets.py -y"
            ) from exc

        target.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
        return target

    # Kitchen variant: short-circuit ONLY when every downloaded bundle is
    # on disk. The editable install path (`_deps._robocasa_kitchen_plan`
    # → clone + `uv pip install -e`) ships the static skeleton
    # (``arenas/empty_kitchen_arena.xml``, ``box_links/...json``) from the
    # cloned source, but the heavyweight content (~11 GB) ships
    # separately via ``robocasa.scripts.download_kitchen_assets``'s Box
    # URLs and lands under the dirs enumerated in that script's
    # ``DOWNLOAD_ASSET_REGISTRY``. Per-task object sampling in
    # ``kitchen_object_utils.sample_kitchen_object`` reads the default
    # ``obj_registries=("objaverse", "lightwheel")``, and the AI-gen
    # category set is also referenced by many tasks — so a partial
    # install (e.g. only ``objects/lightwheel/``) reads "ready" but later
    # divides by zero candidates in the sampler and surfaces as
    # ``ValueError: probabilities contain NaN`` at first ``env.reset()``.
    # Require every non-source download target to be non-empty.
    try:
        import robocasa  # type: ignore[import-not-found,import-untyped,unused-ignore]

        assert robocasa.__file__ is not None
        robocasa_dir = Path(robocasa.__file__).parent
        kitchen_arena = robocasa_dir / "models" / "assets" / "arenas" / "empty_kitchen_arena.xml"
        box_links_assets = (
            robocasa_dir / "models" / "assets" / "box_links" / "box_links_assets.json"
        )
        # Downloaded bundles, matching DOWNLOAD_ASSET_REGISTRY keys in
        # robocasa.scripts.download_kitchen_assets. Each must be a
        # non-empty directory.
        assets_root = robocasa_dir / "models" / "assets"
        downloaded_bundles = (
            assets_root / "textures",
            assets_root / "generative_textures",
            assets_root / "fixtures",
            assets_root / "objects" / "objaverse",
            assets_root / "objects" / "aigen_objs",
            assets_root / "objects" / "lightwheel",
        )

        def _populated(p: Path) -> bool:
            return p.is_dir() and any(p.iterdir())

        if (
            kitchen_arena.is_file()
            and box_links_assets.is_file()
            and all(_populated(p) for p in downloaded_bundles)
        ):
            target.mkdir(parents=True, exist_ok=True)
            sentinel.touch()
            return target
    except ImportError:
        pass

    # ``OPENRAL_AUTO_INSTALL_DEPS=1`` (used by ``openral deploy sim``
    # for CI-style runs) implies acceptance of the RoboCasa asset
    # license + download. The license banner still prints to stderr
    # so the operator has a record of what was downloaded.
    bypass = (
        os.environ.get(_ROBOCASA_ALLOW_ENV) == "1"
        or os.environ.get("OPENRAL_AUTO_INSTALL_DEPS") == "1"
    )
    if not bypass:
        _display_robocasa_license_banner(target)
        if not typer.confirm("Download RoboCasa assets now?", default=False):
            script = (
                "robocasa.scripts.download_tabletop_assets"
                if variant == "gr1_tabletop"
                else "robocasa.scripts.download_kitchen_assets"
            )
            raise ROSConfigError(
                "RoboCasa assets not downloaded. Either set "
                f"{_ROBOCASA_ALLOW_ENV}=1 (or OPENRAL_AUTO_INSTALL_DEPS=1) "
                f"and re-run, or fetch them manually with: "
                f"`uv run python -m {script}`"
            )

    target.mkdir(parents=True, exist_ok=True)
    # The upstream download script prompts on stdin when called
    # directly, so we drive it as a subprocess with stdin confirmation
    # pre-baked. The script writes assets to its site-package-internal
    # `models/assets/` directory (not `target`); we touch the sentinel
    # under `target` afterwards as a readiness marker.
    #
    # robocasa's download scripts ``import robocasa``, which used to hard-assert
    # exact mujoco/numpy/robosuite micro versions at import. The install plan now
    # relaxes those asserts in the editable clone at provision time
    # (``_deps._relax_robocasa_version_asserts_step``), so the download runs on
    # the workspace's real versions with NO version spoof -- a plain ``runpy``
    # shim suffices. Only the procedural ``download_*_assets`` target differs by
    # variant (kitchen vs the GR1 tabletop fork).
    if variant == "gr1_tabletop":
        script_hint = "robocasa.scripts.download_tabletop_assets"
    else:
        script_hint = "robocasa.scripts.download_kitchen_assets"
    shim = f'import runpy; runpy.run_module("{script_hint}", run_name="__main__")'

    try:
        subprocess.run(
            [sys.executable, "-c", shim],
            input="y\n",
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ROSConfigError(
            f"RoboCasa asset download failed: {exc}. Run "
            f"`uv run python -m {script_hint}` manually and re-run the sim."
        ) from exc

    sentinel.touch()
    return target
