"""Unit tests for ``scripts/install_cli.py`` — the ``openral`` launcher writer.

The generated ``~/.local/bin/openral`` wrapper runs ``set -euo pipefail`` and
then sources the ROS 2 distro overlay and the colcon workspace overlay before
``exec``-ing ``.venv/bin/openral``. Those overlays are ament-generated and are
NOT nounset-safe — ``/opt/ros/<distro>/setup.bash`` line 8 reads
``$AMENT_TRACE_SETUP_FILES`` with no default. Under ``set -u`` that aborts the
wrapper with::

    /opt/ros/jazzy/setup.bash: line 8: AMENT_TRACE_SETUP_FILES: unbound variable

…so the user never reaches the REPL. These tests build a real, throwaway repo
shape under ``tmp_path`` (no mocks / no monkey-patching — CLAUDE.md §1.11),
including a deliberately nounset-unsafe ``install/setup.bash`` overlay and a
stub ``.venv/bin/openral``, render the wrapper against it, run it with real
bash, and assert it sources the unsafe overlay and still reaches ``exec``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

_INSTALL_CLI = Path(__file__).resolve().parents[2] / "scripts" / "install_cli.py"


def _load_install_cli() -> ModuleType:
    """Import ``scripts/install_cli.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("install_cli", _INSTALL_CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_fake_repo(tmp_path: Path, *, with_venv: bool = True) -> Path:
    """Build a throwaway OpenRAL checkout shape that the wrapper can drive.

    The ``install/setup.bash`` overlay is intentionally nounset-unsafe — it
    reproduces the exact ament idiom (``[ -n "$AMENT_TRACE_SETUP_FILES" ]``)
    that aborts the wrapper under ``set -u``. The stub ``.venv/bin/openral``
    prints a sentinel and echoes its args so the test can prove ``exec`` ran.
    """
    repo = tmp_path / "openral"
    install = repo / "install"
    install.mkdir(parents=True)
    (install / "setup.bash").write_text(
        "# ament-style overlay: NOT nounset-safe (reads an unset var).\n"
        'if [ -n "$AMENT_TRACE_SETUP_FILES" ]; then echo trace; fi\n'
        "export OPENRAL_OVERLAY_SOURCED=1\n"
    )
    if with_venv:
        venv_bin = repo / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        cli = venv_bin / "openral"
        cli.write_text(
            "#!/usr/bin/env bash\n"
            'echo "REACHED_EXEC overlay=${OPENRAL_OVERLAY_SOURCED:-unset} args=$*"\n'
        )
        cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return repo


def _write_rendered_wrapper(install_cli: ModuleType, repo: Path, dest: Path) -> Path:
    """Render the wrapper for ``repo`` and write an executable script to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(install_cli.render_wrapper(repo))
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def _run_wrapper(wrapper: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # reason: running our own generated bash launcher
        [str(wrapper), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_wrapper_sources_nounset_unsafe_overlay_and_reaches_exec(tmp_path: Path) -> None:
    """Regression: `set -u` + nounset-unsafe overlay must NOT abort before exec.

    This is the bug report — ``openral`` printed
    ``AMENT_TRACE_SETUP_FILES: unbound variable`` and dropped the user back to
    the shell instead of the REPL.
    """
    install_cli = _load_install_cli()
    repo = _make_fake_repo(tmp_path)
    wrapper = _write_rendered_wrapper(install_cli, repo, tmp_path / "bin" / "openral")

    proc = _run_wrapper(wrapper, "doctor", "--flag")

    assert proc.returncode == 0, proc.stderr
    assert "AMENT_TRACE_SETUP_FILES: unbound variable" not in proc.stderr
    # exec reached, overlay env survived the function-scoped source, args forwarded.
    assert "REACHED_EXEC overlay=1 args=doctor --flag" in proc.stdout


def test_wrapper_keeps_strict_mode_prologue(tmp_path: Path) -> None:
    """The wrapper still runs its own logic under `set -euo pipefail`."""
    install_cli = _load_install_cli()
    script = install_cli.render_wrapper(tmp_path / "bin" / "openral")
    assert "set -euo pipefail" in script
    # Strict mode is only relaxed via the dedicated overlay helper.
    assert "_source_overlay" in script
    assert "set +u +e" in script
    assert "set -u -e" in script


def test_wrapper_errors_clearly_when_venv_cli_missing(tmp_path: Path) -> None:
    """No ``.venv/bin/openral`` → exit 1 with an actionable message, not a crash."""
    install_cli = _load_install_cli()
    repo = _make_fake_repo(tmp_path, with_venv=False)
    wrapper = _write_rendered_wrapper(install_cli, repo, tmp_path / "bin" / "openral")

    proc = _run_wrapper(wrapper)

    assert proc.returncode == 1
    assert ".venv/bin/openral not found" in proc.stderr
    assert "just sync --all-packages" in proc.stderr


def test_render_wrapper_substitutes_repo_token(tmp_path: Path) -> None:
    """``__REPO__`` is fully replaced by the target repo path."""
    install_cli = _load_install_cli()
    repo = tmp_path / "openral"
    script = install_cli.render_wrapper(repo)
    assert "__REPO__" not in script
    assert f'_OPENRAL_DIR="{repo}"' in script


def test_bin_dir_env_override_targets_custom_dir(tmp_path: Path) -> None:
    """``OPENRAL_CLI_BIN_DIR`` redirects the install target (so tests/CI never
    clobber a developer's real ``~/.local/bin/openral``)."""
    if shutil.which("python3") is None:  # pragma: no cover - python3 always present in CI
        import pytest

        pytest.skip("python3 not on PATH")
    bin_dir = tmp_path / "custom-bin"
    env = {**os.environ, "OPENRAL_CLI_BIN_DIR": str(bin_dir), "HOME": str(tmp_path)}
    proc = subprocess.run(  # reason: invoking our own installer with a redirected bin dir
        [sys.executable, str(_INSTALL_CLI)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    wrapper = bin_dir / "openral"
    assert wrapper.is_file()
    assert os.access(wrapper, os.X_OK)
    # The wrapper points back at the real repo this script lives in.
    assert "set -euo pipefail" in wrapper.read_text()
