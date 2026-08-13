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

The second family of tests covers *provenance*. The wrapper bakes in the
checkout that generated it, so running ``openral`` from a second checkout used
to silently execute the first one's venv, overlay and ``robots/`` manifests —
which is how a DGX Spark validation run got attributed to the wrong branch.
Those tests build two real checkouts under ``tmp_path`` (one of them a real
``git init`` repo, because the mismatch guard probes ``git rev-parse``) and
pin down the three-way behaviour matrix: baked default runs silently,
``OPENRAL_REPO_ROOT`` redirects and announces itself, and a cwd inside a
different checkout warns without changing which tree runs.
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


def _make_fake_repo(
    tmp_path: Path,
    *,
    with_venv: bool = True,
    name: str = "openral",
    git_init: bool = False,
) -> Path:
    """Build a throwaway OpenRAL checkout shape that the wrapper can drive.

    The ``install/setup.bash`` overlay is intentionally nounset-unsafe — it
    reproduces the exact ament idiom (``[ -n "$AMENT_TRACE_SETUP_FILES" ]``)
    that aborts the wrapper under ``set -u``. Both the overlay and the stub
    ``.venv/bin/openral`` stamp their own checkout path into the output, so a
    test can prove *which* tree supplied the overlay and which supplied the
    CLI — that is exactly what the repo-root override has to get right.

    ``git_init`` makes the checkout a real git repository, which is what the
    wrapper's cwd-mismatch guard probes with ``git rev-parse --show-toplevel``.
    """
    repo = (tmp_path / name).resolve()
    install = repo / "install"
    install.mkdir(parents=True)
    (install / "setup.bash").write_text(
        "# ament-style overlay: NOT nounset-safe (reads an unset var).\n"
        'if [ -n "$AMENT_TRACE_SETUP_FILES" ]; then echo trace; fi\n'
        "export OPENRAL_OVERLAY_SOURCED=1\n"
        f'export OPENRAL_OVERLAY_FROM="{repo}"\n'
    )
    if with_venv:
        venv_bin = repo / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        cli = venv_bin / "openral"
        cli.write_text(
            "#!/usr/bin/env bash\n"
            'echo "REACHED_EXEC overlay=${OPENRAL_OVERLAY_SOURCED:-unset} args=$*"\n'
            f'echo "EXEC_REPO={repo}"\n'
            'echo "OVERLAY_REPO=${OPENRAL_OVERLAY_FROM:-unset}"\n'
        )
        cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if git_init:
        subprocess.run(  # reason: a real git repo is what the wrapper's guard probes
            ["git", "init", "-q", str(repo)],
            check=True,
            capture_output=True,
        )
    return repo


def _write_rendered_wrapper(install_cli: ModuleType, repo: Path, dest: Path) -> Path:
    """Render the wrapper for ``repo`` and write an executable script to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(install_cli.render_wrapper(repo))
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def _run_wrapper(
    wrapper: Path,
    *args: str,
    cwd: Path,
    repo_root_env: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the generated launcher from ``cwd`` with a controlled environment.

    ``cwd`` is always explicit because the wrapper now inspects it (the
    different-checkout guard), and ``OPENRAL_REPO_ROOT`` is always set or
    explicitly cleared so an exported value in the developer's shell cannot
    leak into the run.
    """
    env = {**os.environ}
    env.pop("OPENRAL_REPO_ROOT", None)
    if repo_root_env is not None:
        env["OPENRAL_REPO_ROOT"] = repo_root_env
    return subprocess.run(  # reason: running our own generated bash launcher
        [str(wrapper), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(cwd),
        env=env,
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

    proc = _run_wrapper(wrapper, "doctor", "--flag", cwd=tmp_path)

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

    proc = _run_wrapper(wrapper, cwd=tmp_path)

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


def test_repo_root_override_redirects_venv_and_overlay_and_announces_it(tmp_path: Path) -> None:
    """``OPENRAL_REPO_ROOT`` wins over the baked-in checkout, visibly.

    Provenance regression: on a DGX Spark a validation run launched from a git
    worktree silently executed the *parent* checkout's venv, overlay and
    ``robots/`` manifests, so the result was attributed to the wrong branch.
    The override must redirect BOTH the colcon overlay and the exec'd CLI, and
    must name the root it chose on stderr so the run log records which tree ran.
    """
    install_cli = _load_install_cli()
    baked = _make_fake_repo(tmp_path, name="openral-main")
    other = _make_fake_repo(tmp_path, name="openral-worktree")
    wrapper = _write_rendered_wrapper(install_cli, baked, tmp_path / "bin" / "openral")

    proc = _run_wrapper(wrapper, "doctor", cwd=tmp_path, repo_root_env=str(other))

    assert proc.returncode == 0, proc.stderr
    assert f"EXEC_REPO={other}" in proc.stdout
    assert f"OVERLAY_REPO={other}" in proc.stdout
    assert f"EXEC_REPO={baked}" not in proc.stdout
    # One stderr line names the root actually used (and the default it replaced).
    assert f"openral: repo root {other}" in proc.stderr
    assert "OPENRAL_REPO_ROOT override" in proc.stderr
    assert str(baked) in proc.stderr


def test_repo_root_override_without_a_venv_cli_fails_instead_of_falling_back(
    tmp_path: Path,
) -> None:
    """A bogus override is a hard error — never a silent fall back to the baked tree.

    Falling back would recreate the exact hazard the override exists to close:
    the caller asks for checkout B and gets checkout A's code with no signal.
    """
    install_cli = _load_install_cli()
    baked = _make_fake_repo(tmp_path, name="openral-main")
    unsynced = _make_fake_repo(tmp_path, name="openral-unsynced", with_venv=False)
    wrapper = _write_rendered_wrapper(install_cli, baked, tmp_path / "bin" / "openral")

    proc = _run_wrapper(wrapper, cwd=tmp_path, repo_root_env=str(unsynced))

    assert proc.returncode == 1
    assert "REACHED_EXEC" not in proc.stdout  # the baked CLI must NOT have run
    assert f"OPENRAL_REPO_ROOT={unsynced}" in proc.stderr
    assert "no executable .venv/bin/openral" in proc.stderr
    assert "just sync --all-packages" in proc.stderr


def test_warns_when_cwd_is_a_different_openral_checkout(tmp_path: Path) -> None:
    """cwd in another usable checkout → WARNING on stderr, baked tree still runs.

    This is the Spark footgun made visible. It stays a warning, not an error:
    the baked default must keep working, and the launcher must not silently
    *switch* trees either — it only tells the operator what it is about to do.
    """
    install_cli = _load_install_cli()
    baked = _make_fake_repo(tmp_path, name="openral-main")
    worktree = _make_fake_repo(tmp_path, name="openral-worktree", git_init=True)
    wrapper = _write_rendered_wrapper(install_cli, baked, tmp_path / "bin" / "openral")

    proc = _run_wrapper(wrapper, "deploy", cwd=worktree)

    assert proc.returncode == 0, proc.stderr
    assert "WARNING: cwd is inside the OpenRAL checkout" in proc.stderr
    assert str(worktree) in proc.stderr
    assert f"baked to {baked}" in proc.stderr
    assert f"export OPENRAL_REPO_ROOT={worktree}" in proc.stderr
    # Behaviour is unchanged: the baked checkout is what actually executes.
    assert f"EXEC_REPO={baked}" in proc.stdout


def test_no_mismatch_warning_for_the_normal_single_checkout_case(tmp_path: Path) -> None:
    """cwd inside the baked checkout itself → silent, exactly as before."""
    install_cli = _load_install_cli()
    baked = _make_fake_repo(tmp_path, name="openral-main", git_init=True)
    wrapper = _write_rendered_wrapper(install_cli, baked, tmp_path / "bin" / "openral")

    proc = _run_wrapper(wrapper, cwd=baked)

    assert proc.returncode == 0, proc.stderr
    assert "WARNING: cwd is inside" not in proc.stderr
    assert "OPENRAL_REPO_ROOT override" not in proc.stderr
    assert f"EXEC_REPO={baked}" in proc.stdout


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
