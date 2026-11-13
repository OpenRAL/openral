"""Make this test directory's shared helper modules importable under direct pytest.

The repo runs pytest with ``--import-mode=importlib`` (see the ``addopts`` in
``pyproject.toml``), which deliberately does *not* inject a test file's own
directory into ``sys.path``. Without that injection a test module here cannot
``from _launch_test_common import ...`` its sibling helper, so this shim adds
the directory explicitly — the same trick ``python/hal/tests/conftest.py``
and the sibling ROS package test roots use
(``packages/openral_hal_node/test/conftest.py`` and friends).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve().parent

if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))


@pytest.fixture
def host_confined_ros_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ROS scope ``openral deploy run`` gives a real robot: discovery on this host only.

    ``deploy_e2e.launch.py`` refuses ``hal_mode:=real`` on a graph that reaches other
    hosts (``openral_core.ros_graph_scope``), and sourcing ROS's ``setup.bash`` exports
    ``SUBNET``, so a test composing the real graph sets the deploy's scope first. The
    refusal itself is covered with explicit envs in ``test_deploy_e2e_real_graph_scope.py``.
    Every other scope variable is cleared, so a shell's Zenoh RMW, discovery server,
    peer-listing XML or SROS2 keystore cannot fail the composition either. Nothing is
    launched, so no process outside this environment has to find the graph.
    """
    from _launch_test_common import set_ros_scope

    set_ros_scope(monkeypatch, {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"})
