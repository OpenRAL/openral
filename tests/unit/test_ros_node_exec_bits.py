"""Guard: every ``install(PROGRAMS …)`` node ships committed executable.

``just ros2-build`` runs ``colcon build --merge-install --symlink-install``.
Under ``--symlink-install``, CMake's ``install(PROGRAMS …)`` does not copy the
file into ``lib/<pkg>/`` — it symlinks it — so the "make it executable on
install" half of ``PROGRAMS`` never happens and the libexec entry inherits the
*source* file's mode. ``launch_ros`` resolves ``executable=`` with
``shutil.which()`` over that directory, which returns ``None`` for a 0644 file,
and raises ``SubstitutionFailure``. launch does not skip the offending node: it
abandons the whole launch description.

That is how six nodes shipped un-launchable on master — including
``openral_safety_watchdog``'s deadman + hardware-estop pair and
``openral_human_estop``'s forwarder, i.e. three quarters of the
defense-in-depth estop path. The defect is invisible without
``--symlink-install`` (a plain copy build silently repairs the mode), so it
gets a source-level guard rather than a build-time one.

Run with:
    uv run pytest tests/unit/test_ros_node_exec_bits.py -v
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES = _REPO_ROOT / "packages"


def _committed_modes() -> dict[str, str]:
    """Repo-relative path -> committed mode, straight out of the git index."""
    proc = subprocess.run(
        ["git", "ls-files", "-s"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=True,
    )
    modes: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        meta, _, path = line.partition("\t")
        if path:
            modes[path] = meta.split(" ", 1)[0]
    return modes


def _install_programs_sources() -> dict[str, Path]:
    """Every file any package installs via ``install(PROGRAMS …)``.

    Keyed by repo-relative path. Discovery is by parsing the CMakeLists, so a
    node added to any package tomorrow is covered without editing this test.
    """
    sources: dict[str, Path] = {}
    for cmakelists in sorted(_PACKAGES.glob("*/CMakeLists.txt")):
        body = cmakelists.read_text(encoding="utf-8")
        project = re.search(r"project\(\s*([^)\s]+)", body)
        name = project.group(1) if project else cmakelists.parent.name
        # Strip comments first: several CMakeLists explain the PROGRAMS
        # pattern in prose ("whereas install(PROGRAMS ...) copies the file"),
        # and an unstripped scan picks the literal `...` up as a target.
        body = re.sub(r"(?m)#.*$", "", body)
        for block in re.finditer(r"install\s*\(\s*PROGRAMS(.*?)DESTINATION", body, re.DOTALL):
            for token in block.group(1).split():
                src = (cmakelists.parent / token.replace("${PROJECT_NAME}", name)).resolve()
                sources[str(src.relative_to(_REPO_ROOT))] = src
    return sources


def test_install_programs_targets_exist() -> None:
    """A CMakeLists cannot name a file that is not there."""
    sources = _install_programs_sources()
    assert sources, "no install(PROGRAMS …) targets found — did packages/ move?"
    missing = sorted(rel for rel, src in sources.items() if not src.is_file())
    assert not missing, f"install(PROGRAMS …) names files that do not exist: {missing}"


def test_install_programs_targets_are_committed_executable() -> None:
    """0644 + ``--symlink-install`` = a node launch_ros can never resolve."""
    modes = _committed_modes()
    not_executable = sorted(
        (rel, modes.get(rel, "untracked"))
        for rel in _install_programs_sources()
        if modes.get(rel) != "100755"
    )
    assert not not_executable, (
        "install(PROGRAMS …) targets are not committed 100755. Under "
        "--symlink-install their libexec entry symlinks straight back to these "
        "modes, so launch_ros raises \"executable '…' not found on the libexec "
        'directory" and abandons the entire launch description. Fix with '
        f"`git update-index --chmod=+x <file>`: {not_executable}"
    )


def test_install_programs_targets_carry_a_shebang() -> None:
    """The +x bit only means anything with an interpreter line to go with it.

    ``install(PROGRAMS …)`` is used precisely so the file keeps its own
    ``#!/usr/bin/env python3`` (a setuptools console_script would be rewritten
    to the system interpreter, which lacks the workspace packages).
    """
    no_shebang = sorted(
        rel
        for rel, src in _install_programs_sources().items()
        if src.is_file() and not src.read_bytes().startswith(b"#!")
    )
    assert not no_shebang, (
        "install(PROGRAMS …) targets are exec'd directly out of lib/<pkg>/ and "
        f"need an interpreter line: {no_shebang}"
    )
