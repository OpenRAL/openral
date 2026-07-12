"""Regression: LIBERO's first-run ``input()`` prompt must not crash a
non-interactive / captured-stdin process.

``libero.libero.__init__`` writes ``~/.libero/config.yaml`` on first import,
prompting via ``input()`` when it is absent. Under pytest's captured stdin (and
any subprocess with a closed stdin — a deploy-sim launch, a cron CI runner) that
``input()`` raises ``OSError: reading from stdin while output is captured``,
crashing the very first ``from lerobot.envs.libero import LiberoEnv`` inside
:func:`openral_sim.backends.libero._build_libero_scene`.

``_seed_libero_config_if_absent`` writes the config non-interactively before
that import so the prompt never fires. This test reproduces the exact failure
mode — a fresh interpreter, an empty config dir, ``stdin`` closed — and asserts
the seed lets ``libero.libero`` import cleanly. Real components only
(CLAUDE.md §1.11): it exercises the genuine ``libero`` package and skips where
``lerobot[libero]`` is not installed.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# `import libero` (the parent package) does NOT trigger the prompting
# `libero.libero` submodule, so this importorskip is safe under captured stdin.
pytest.importorskip("libero")


def test_seed_lets_libero_import_with_closed_stdin(tmp_path: object) -> None:
    """A fresh process with an empty config dir + closed stdin imports clean."""
    config_dir = os.fspath(tmp_path)  # type: ignore[arg-type]  # reason: pytest tmp_path
    script = (
        "from openral_sim.backends.libero import _seed_libero_config_if_absent\n"
        "_seed_libero_config_if_absent()\n"
        "import libero.libero  # must not prompt now that the config exists\n"
        "print('LIBERO_IMPORT_OK')\n"
    )
    env = {**os.environ, "LIBERO_CONFIG_PATH": config_dir}
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        stdin=subprocess.DEVNULL,  # reproduce the captured/non-interactive stdin
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # we assert on returncode below with the captured stderr
    )
    assert result.returncode == 0, f"libero import crashed under closed stdin:\n{result.stderr}"
    assert "LIBERO_IMPORT_OK" in result.stdout
    assert os.path.exists(os.path.join(config_dir, "config.yaml")), (
        "seed did not write LIBERO config.yaml"
    )
