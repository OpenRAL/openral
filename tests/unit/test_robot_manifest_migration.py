"""``RobotDescription`` schema_version 0.1 -> 0.2 migration (CLAUDE.md §1.6).

Fixtures are verbatim 0.1 manifests from ``robots/`` at commit 4d589d1
(``tests/unit/fixtures/robot_manifest_v0_1/SOURCE.txt``).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from openral_core import RobotDescription, migrate_robot_manifest
from openral_core.exceptions import ROSConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
V01 = REPO_ROOT / "tests/unit/fixtures/robot_manifest_v0_1"


def _raw(name: str) -> dict[str, object]:
    data = yaml.safe_load((V01 / f"{name}.yaml").read_text(encoding="utf-8"))
    assert data["schema_version"] == "0.1"
    return data


def test_a_real_0_1_manifest_migrates_on_load_and_validates() -> None:
    desc = RobotDescription.from_yaml(str(V01 / "ur5e.yaml"))
    assert desc.schema_version == "0.2"
    assert desc.safety.joint_state_staleness_limit_s == 0.5
    assert "staleness_limit_s" not in desc.hal.parameters.defaults
    # Same content as the shipped 0.2 manifest, bar comments.
    current = RobotDescription.from_yaml(str(REPO_ROOT / "robots/ur5e/robot.yaml"))
    assert desc.model_dump() == current.model_dump()


def test_the_migrator_is_pure_and_only_moves_the_staleness() -> None:
    raw = _raw("ur5e")
    before = copy.deepcopy(raw)
    out = migrate_robot_manifest(raw)
    assert raw == before
    assert out["schema_version"] == "0.2"
    assert out["safety"]["joint_state_staleness_limit_s"] == 0.5  # type: ignore[index]
    assert out["hal"]["parameters"]["defaults"] == {}  # type: ignore[index]
    # Nothing else changed.
    for key in set(before) - {"schema_version", "safety", "hal"}:
        assert out[key] == before[key]


def test_a_0_1_real_prismatic_manifest_missing_the_m_fields_is_refused() -> None:
    with pytest.raises(ROSConfigError) as err:
        RobotDescription.from_yaml(str(V01 / "sawyer.yaml"))
    msg = str(err.value)
    assert "safety.starting_pose_max_joint_speed_m_s" in msg
    assert "safety.starting_pose_tolerance_m" in msg
    assert "never invents a safety value" in msg


def test_a_current_manifest_is_not_migrated() -> None:
    raw = yaml.safe_load((REPO_ROOT / "robots/ur5e/robot.yaml").read_text(encoding="utf-8"))
    assert raw["schema_version"] == "0.2"
    assert migrate_robot_manifest(raw) == raw


@pytest.mark.parametrize("version", ["0.3", "1.0", "garbage"])
def test_an_unknown_schema_version_is_refused(version: str) -> None:
    raw = _raw("ur5e")
    raw["schema_version"] = version
    with pytest.raises(ROSConfigError, match="unknown schema_version"):
        RobotDescription.model_validate(raw)
