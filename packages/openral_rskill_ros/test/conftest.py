"""Make this test directory's shared helper modules importable under direct pytest.

The repo runs pytest with ``--import-mode=importlib`` (see the ``addopts`` in
``pyproject.toml``), which deliberately does *not* inject a test file's own
directory into ``sys.path``. Without that injection a test module here cannot
``from _launch_test_common import ...`` its sibling helper, so this shim adds
the directory explicitly — the same trick ``python/hal/tests/conftest.py``
and the sibling ROS package test roots use
(``packages/openral_hal_scene_attached/test/conftest.py`` and friends).
"""

from __future__ import annotations

import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent

if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))
