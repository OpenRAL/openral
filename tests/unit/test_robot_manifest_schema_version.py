"""``RobotDescription`` loads only ``schema_version: "0.2"`` (no migrator)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ur5e() -> dict[str, object]:
    raw = yaml.safe_load((REPO_ROOT / "robots/ur5e/robot.yaml").read_text(encoding="utf-8"))
    assert raw["schema_version"] == "0.2"
    return raw


def test_a_current_manifest_loads() -> None:
    assert RobotDescription.model_validate(_ur5e()).schema_version == "0.2"


@pytest.mark.parametrize("version", ["0.1", "0.3", "1.0", "garbage"])
def test_any_other_schema_version_is_refused(version: str) -> None:
    raw = _ur5e()
    raw["schema_version"] = version
    with pytest.raises(ROSConfigError, match=r"only '0\.2' loads"):
        RobotDescription.model_validate(raw)
