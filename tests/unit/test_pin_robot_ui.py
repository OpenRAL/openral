# SPDX-License-Identifier: Apache-2.0
"""``tools/pin_robot_ui.py`` rewrites the real ``ui_assets.py`` pin, and nothing else."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("pin_robot_ui", _REPO / "tools" / "pin_robot_ui.py")
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
sys.modules["pin_robot_ui"] = tool
_spec.loader.exec_module(tool)


def test_the_pin_is_rewritten_in_place_and_parses(tmp_path: Path) -> None:
    src = (_REPO / tool.UI_ASSETS).read_text(encoding="utf-8")
    url = "https://github.com/OpenRAL/openral/releases/download/robot-ui-abc1234/openral-robot-ui-robot-ui-abc1234.tar.gz"
    out = tool.pin(src, "robot-ui-abc1234", url, "a" * 64, 184_000)
    changed = [(a, b) for a, b in zip(src.splitlines(), out.splitlines(), strict=True) if a != b]
    assert len(changed) == 4  # url, sha256, size, and the docstring's version
    assert "``robot-ui.yml``" in out and "``robot-ui-abc1234``" in out
    assert f'url="{url}"' in out and 'sha256="' + "a" * 64 + '"' in out and "size=184000," in out
    # Idempotent over its own output (the next bump finds the block again).
    assert tool.pin(out, "robot-ui-def5678", url, "b" * 64, 1).count("robot-ui-def5678") == 1
    target = tmp_path / "ui_assets.py"
    target.write_text(src, encoding="utf-8")
    assert tool.main(["robot-ui-abc1234", url, "c" * 64, "9", str(target)]) == 0
    compile(target.read_text(encoding="utf-8"), str(target), "exec")


@pytest.mark.parametrize(
    ("version", "url", "sha256", "size"),
    [
        ("robot-ui-x", "https://x", "nothex", 1),
        ("robot-ui-x", "http://x", "a" * 64, 1),
        ("bad tag\n", "https://x", "a" * 64, 1),
        ("v0.5.0", "https://x", "a" * 64, 1),
        ("robot-ui-x", "https://x", "a" * 64, 0),
    ],
)
def test_malformed_arguments_are_refused(version: str, url: str, sha256: str, size: int) -> None:
    src = (_REPO / tool.UI_ASSETS).read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        tool.pin(src, version, url, sha256, size)
