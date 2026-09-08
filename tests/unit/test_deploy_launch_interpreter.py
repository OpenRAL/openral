"""The ``ros2 launch`` parser must run under the workspace venv interpreter.

``/opt/ros/<distro>/bin/ros2`` has a ``#!/usr/bin/python3`` shebang, so a bare
``ros2 launch`` parses under the *system* interpreter — ``PYTHONPATH`` only
prepends the venv, so apt-installed extensions can mix with the venv's NumPy.

Observed on a Jetson AGX Thor with ``python3-pandas``: ``openral deploy run``
aborted with ``ValueError: numpy.dtype size changed`` from
``openral_hal.sim_bringup`` → ``lerobot`` → ``deepdiff`` → ``import pandas``
(``deepdiff`` guards that import against ``ImportError`` only, not ``ValueError``).

Running under ``sys.executable`` makes the venv's ``include-system-site-packages
= false`` apply, removing the failure class. These tests pin that the resolution
layer keeps doing so — no ``ros2 launch`` is actually run.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import sys
from pathlib import Path

from openral_cli.deploy_sim import _ros2_argv_head, resolve_launch_invocation


def _ros2_is_python_script(path: str) -> bool:
    with open(path, "rb") as handle:
        shebang = handle.readline(256)
    return shebang.startswith(b"#!") and b"python" in shebang


def _wrappable() -> str | None:
    """The ``ros2`` this host would wrap, or ``None`` when it must not be wrapped."""
    ros2_bin = shutil.which("ros2")
    if ros2_bin is None or importlib.util.find_spec("ros2cli") is None:
        return None
    return ros2_bin if _ros2_is_python_script(ros2_bin) else None


class TestRos2ArgvHead:
    def test_head_matches_what_this_host_actually_ships(self) -> None:
        """Head is ``[sys.executable, <ros2>]`` iff ``ros2`` is a Python script.

        Asserted against the host's real ``PATH`` and the real file on disk
        rather than a patched ``shutil.which`` (CLAUDE.md §1.11), so whichever
        branch this host takes is the branch that gets exercised.
        """
        ros2_bin = _wrappable()
        head = _ros2_argv_head()
        if ros2_bin is None:
            assert head == ["ros2"]
        else:
            assert head == [sys.executable, ros2_bin]

    def test_a_python_ros2_is_never_shelled_bare(self) -> None:
        """A Python ``ros2`` shelled bare is the bug this guards against."""
        if _wrappable() is None:
            return  # covered by the branch above on this host
        assert _ros2_argv_head()[0] == sys.executable

    def test_ros2cli_must_be_importable_by_this_interpreter(self) -> None:
        """The wrap is only taken when ``sys.executable`` can import ``ros2cli``.

        A conda/RoboStack ROS satisfies the shebang test (its own Python owns
        the script) while its site-packages are absent from the workspace
        ``PYTHONPATH``, so wrapping there would replace a working bare ``ros2``
        with ``PackageNotFoundError: ros2cli``.
        """
        if importlib.util.find_spec("ros2cli") is None:
            assert _ros2_argv_head() == ["ros2"]
        else:
            assert _wrappable() is None or _ros2_argv_head()[0] == sys.executable

    def test_a_non_python_ros2_falls_back_to_the_bare_name(self, tmp_path: Path) -> None:
        """A shell-wrapper ``ros2`` must not be exec'd under an interpreter.

        Uses a real executable file on a real PATH entry — no patching of
        ``shutil`` — so the shebang branch is exercised as it would be on a
        distro that ships a wrapper instead of a console script.
        """
        wrapper = tmp_path / "ros2"
        wrapper.write_text('#!/bin/sh\nexec /opt/ros/jazzy/bin/ros2 "$@"\n')
        wrapper.chmod(0o755)
        original = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{tmp_path}{os.pathsep}{original}"
        try:
            assert _ros2_argv_head() == ["ros2"]
        finally:
            os.environ["PATH"] = original


class TestLaunchInvocationArgv:
    def test_openarm_real_invocation_uses_the_venv_interpreter(self) -> None:
        """The real OpenArm deploy scene resolves to a venv-interpreter argv.

        Uses the committed ``scenes/deploy/openarm_restock_shelf.yaml`` fixture
        — the bimanual cell whose launch the pandas ABI abort was found on.
        """
        inv = resolve_launch_invocation(
            config=Path("scenes/deploy/openarm_restock_shelf.yaml"),
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_mode="real",
        )
        argv = inv.argv_template
        assert "launch" in argv
        assert argv[argv.index("launch") + 1] == "openral_rskill_ros"
        if _wrappable() is not None:
            assert argv[0] == sys.executable

    def test_the_echoed_argv_stays_parseable_by_the_validation_matrix(self) -> None:
        """``validation_matrix.parse_launch_argv`` must still recover this argv.

        The round-provenance importer reads the CLI's own ``argv:`` echo. It
        used to match the literal ``argv: ros2 launch``, which the venv-wrapped
        head no longer produces — and it fails *silently*, recording an empty
        stack rather than an error, so nothing else catches the regression.
        """
        sys.path.insert(0, str(Path("tools").resolve()))
        try:
            from validation_matrix import parse_launch_argv, robot_facts_from_launch_argv
        finally:
            sys.path.pop(0)

        inv = resolve_launch_invocation(
            config=Path("scenes/deploy/openarm_restock_shelf.yaml"),
            robot_override=None,
            dashboard_port=4318,
            reset_to_pose_service=None,
            hal_mode="real",
        )
        echoed = f"  argv: {shlex.join(inv.argv_template)}"
        parsed = parse_launch_argv([echoed])
        assert parsed == inv.argv_template
        assert robot_facts_from_launch_argv(parsed)["robot_id"] == "openarm"
