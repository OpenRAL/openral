"""Shared off-screen-renderer availability probe for the ``python/hal`` tests.

Every test in this directory that builds a HAL (native-so101, LIBERO or
robocasa) renders an off-screen camera frame inside ``connect()``.  On a
headless runner (no GPU/display) that either ``abort()``s the process at the C
level (native MuJoCo ``Renderer`` -> SIGABRT) or raises (robosuite's EGL path ->
``eglQueryString`` AttributeError), so every render-dependent test must skip
when no off-screen renderer is available.  The ``robosuite``/``libero``
``importorskip``s alone are not enough: a host can have them installed and
still lack a working GL/EGL stack.

This module previously existed as five byte-identical copies (one per test
module).  Hoisting it here also means the subprocess probe runs **once** per
pytest session instead of once per module.

Import it via the ``sys.path`` shim in this directory's ``conftest.py``::

    from _renderer_probe import requires_renderer
"""

from __future__ import annotations

import os

import pytest

# Force EGL (off-screen) rendering so hosts without a display don't abort.
# The classic renderer calls glXOpenDisplay() and raises SIGABRT on headless
# runners; EGL avoids the display requirement entirely.  This must happen
# before anything imports ``mujoco``, hence module scope.
os.environ.setdefault("MUJOCO_GL", "egl")


def mujoco_renderer_probe_error() -> str | None:
    """Return ``None`` if a MuJoCo off-screen renderer can be created, else a reason.

    Creating a ``mujoco.Renderer`` on a headless host without a working GL/EGL
    stack calls ``abort()`` at the C level (SIGABRT), which a Python
    ``try/except`` cannot catch — an in-process probe therefore crashes pytest
    outright (``Fatal Python error: Aborted``) and takes the whole partition
    down with it. Running the probe in a subprocess turns that abort into a
    non-zero exit code we can detect and convert into a clean skip reason,
    leaving collection alive. Mirrors ``tests/sim/conftest`` (a sibling test
    root we cannot import across).
    """
    import subprocess
    import sys

    probe = (
        "import mujoco;"
        "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody></worldbody></mujoco>');"
        "r = mujoco.Renderer(m, 1, 1); r.close()"
    )
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )
    except FileNotFoundError:  # mujoco import unavailable in the probe interpreter
        return "mujoco unavailable for renderer probe"
    except subprocess.TimeoutExpired:
        return "mujoco renderer probe timed out (120s)"
    if proc.returncode == 0:
        return None
    stderr_lines = (proc.stderr or "").strip().splitlines()
    detail = stderr_lines[-1] if stderr_lines else "no stderr"
    return f"renderer probe exited {proc.returncode}: {detail}"


RENDERER_ERROR = mujoco_renderer_probe_error()
"""``None`` when an off-screen renderer is available, else the skip reason."""

requires_renderer = pytest.mark.skipif(
    RENDERER_ERROR is not None,
    reason=f"mujoco renderer unavailable: {RENDERER_ERROR}",
)
"""Skip marker for a test that renders an off-screen frame."""
