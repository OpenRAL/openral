"""The deploy launch seeds the reasoner from the same ``rskills/`` tree as the CLI preflights.

``openral deploy sim`` reads ``$OPENRAL_REPO_ROOT``'s skills for its palette-deps and
reward VRAM gates (``_repo_root_from``); the launch must hand the reasoner that tree too,
or the gate can pass on a skill the launched palette does not contain.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH = REPO_ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"


@pytest.fixture(scope="module")
def launch_module() -> object:
    """The real launch file, loaded by path — it is not an importable package."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_rskills_dir", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_rskills_dir"] = module
    spec.loader.exec_module(module)
    return module


def test_override_checkout_supplies_the_rskills_dir(
    launch_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid ``$OPENRAL_REPO_ROOT`` moves the skill tree, not the launch checkout."""
    shutil.copytree(
        REPO_ROOT / "rskills" / "smolvla-libero", tmp_path / "rskills" / "smolvla-libero"
    )
    shutil.copytree(REPO_ROOT / "robots" / "franka_panda", tmp_path / "robots" / "franka_panda")
    monkeypatch.setenv("OPENRAL_REPO_ROOT", str(tmp_path))
    assert launch_module._resolve_rskills_dir() == str(tmp_path.resolve() / "rskills")  # type: ignore[attr-defined]  # reason: module loaded by path
    assert launch_module._REPO_ROOT == REPO_ROOT  # type: ignore[attr-defined]  # reason: module loaded by path


def test_no_or_invalid_override_keeps_the_launch_checkout(
    launch_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset, or pointing at a directory without ``robots/`` + ``rskills/``: this checkout."""
    expected = str(launch_module._REPO_ROOT / "rskills")  # type: ignore[attr-defined]  # reason: module loaded by path
    monkeypatch.delenv("OPENRAL_REPO_ROOT", raising=False)
    assert launch_module._resolve_rskills_dir() == expected  # type: ignore[attr-defined]  # reason: module loaded by path
    monkeypatch.setenv("OPENRAL_REPO_ROOT", str(tmp_path))
    assert launch_module._resolve_rskills_dir() == expected  # type: ignore[attr-defined]  # reason: module loaded by path
