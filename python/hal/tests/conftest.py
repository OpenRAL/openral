"""Make this test directory's shared helper modules importable under direct pytest.

The repo runs pytest with ``--import-mode=importlib`` (see the ``addopts`` in
``pyproject.toml``), which deliberately does *not* inject a test file's own
directory into ``sys.path``.  Without that injection a test module here cannot
``from _renderer_probe import ...`` its sibling helper, so this shim adds the
directory explicitly — the same trick the sibling ROS package test roots use
(``packages/openral_hal_node/test/conftest.py`` and friends).

Importing the helper through ``conftest`` itself is *not* an option: the
repo-root ``conftest.py`` already owns the top-level module name ``conftest``,
so ``from conftest import ...`` here would resolve to the wrong file.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

_TEST_DIR = Path(__file__).resolve().parent

if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))


@pytest.fixture
def _build_so101_hal() -> Callable[[], object]:
    """Return a builder for a connected ``SimAttachedHAL`` over the native so101 box scene.

    The backend exposes no introspectable ``action_dim``, so the builder passes
    ``env_action_dim=6`` explicitly — the documented path for non-introspectable
    envs (mirrors what the lifecycle node would resolve).
    """

    def _build_so101_hal() -> object:
        from openral_core import RobotDescription
        from openral_hal.sim_attached import SimAttachedHAL
        from openral_hal.sim_bringup import build_sim_env_from_yaml

        env, seed = build_sim_env_from_yaml(
            "scenes/sim/so101_tube_insertion.yaml", robot_id_fallback="so101_follower"
        )
        desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
        hal = SimAttachedHAL(env, desc, env_reset_seed=seed, env_action_dim=6)
        hal.connect()
        return hal

    return _build_so101_hal


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root (``robots/``, ``scenes/``)."""
    return _TEST_DIR.parents[2]


@pytest.fixture
def robot_description(repo_root: Path) -> Callable[[str], object]:
    """Load ``robots/<robot>/robot.yaml`` as a real ``RobotDescription``."""

    def _load(robot: str) -> object:
        from openral_core import RobotDescription

        return RobotDescription.from_yaml(str(repo_root / "robots" / robot / "robot.yaml"))

    return _load


@pytest.fixture
def connect_twin(robot_description: Callable[[str], object]) -> Iterator[Callable[..., Any]]:
    """Connect a sim HAL: ``build_hal(robots/<robot>, mode="sim")``, or a zero-arg builder.

    Skips only when an optional asset or package is unavailable on this host
    (``_twin_assets.skip_if_unavailable``); a broken twin fails. Every HAL it
    connected is disconnected at teardown.
    """
    from _twin_assets import skip_if_unavailable
    from openral_core.exceptions import ROSConfigError

    connected: list[Any] = []

    def _connect(robot: str | Callable[[], Any]) -> Any:  # reason: any sim HAL
        try:
            # Construction resolves assets too, so it shares the classified handling.
            if isinstance(robot, str):
                from openral_hal import build_hal

                hal = build_hal(robot_description(robot), mode="sim")  # type: ignore[arg-type]  # reason: a RobotDescription
            else:
                hal = robot()
            hal.connect()
        except ROSConfigError as exc:
            skip_if_unavailable(exc, f"{robot} twin")
        connected.append(hal)
        return hal

    yield _connect
    for hal in reversed(connected):
        hal.disconnect()
