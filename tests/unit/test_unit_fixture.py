"""Surveyed cell fixtures on the robot unit overlay (real pick-and-place design §2.3).

A ``UnitFixture`` is a declared volume in a fixed-base robot's base frame. These tests load
a unit file carrying one through the real ``load_robot_unit`` against the real OpenArm
manifest, and check ``fixture_problems`` against real fixed-base and mobile manifests.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from openral_core import (
    AttachmentEvidenceKind,
    RobotDescription,
    RobotUnit,
    UnitFixture,
    fixture_problems,
    load_robot_unit,
)
from openral_core.exceptions import ROSConfigError
from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parents[2]
_OPENARM = _ROOT / "robots" / "openarm" / "robot.yaml"
_SHELF_UNIT = _ROOT / "tests" / "unit" / "fixtures" / "robot_units" / "openarm_shelf_cell.yaml"


def _openarm_with_unit(tmp_path: Path, doc: dict[str, object]) -> Path:
    robot_dir = tmp_path / "openarm"
    (robot_dir / "units").mkdir(parents=True)
    shutil.copy(_OPENARM, robot_dir / "robot.yaml")
    (robot_dir / "units" / "shelf_cell.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    return robot_dir / "robot.yaml"


def _shelf_doc() -> dict[str, object]:
    doc = yaml.safe_load(_SHELF_UNIT.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def test_an_openarm_unit_with_a_shelf_fixture_loads(tmp_path: Path) -> None:
    unit = load_robot_unit(_openarm_with_unit(tmp_path, _shelf_doc()), "shelf_cell")
    shelf = unit.fixture("cell:shelf_top")
    assert shelf.frame_id == "openarm_base"
    assert shelf.top_plane_normal == (0.0, 0.0, 1.0)
    assert shelf.surveyed_on.isoformat() == "2026-10-02"
    with pytest.raises(ROSConfigError, match="surveys no fixture 'cell:front_table'"):
        unit.fixture("cell:front_table")


def test_a_fixture_outside_the_base_frame_is_refused(tmp_path: Path) -> None:
    doc = _shelf_doc()
    fixture = doc["fixtures"][0]  # type: ignore[index]
    fixture["frame_id"] = fixture["pose"]["frame_id"] = "world"
    unit = RobotUnit.model_validate(doc)
    problems = fixture_problems(RobotDescription.from_yaml(str(_OPENARM)), unit)
    assert problems == [
        "cell:shelf_top: frame_id 'world' is not the robot's base_frame 'openarm_base'"
    ]
    with pytest.raises(ROSConfigError, match="is not the robot's base_frame"):
        load_robot_unit(_openarm_with_unit(tmp_path, doc), "shelf_cell")


@pytest.mark.parametrize("robot", ["panda_mobile", "g1"])
def test_a_mobile_robot_takes_no_fixture(robot: str) -> None:
    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / robot / "robot.yaml"))
    doc = _shelf_doc()
    fixture = doc["fixtures"][0]  # type: ignore[index]
    fixture["frame_id"] = fixture["pose"]["frame_id"] = desc.base_frame
    problems = fixture_problems(desc, RobotUnit.model_validate(doc))
    assert len(problems) == 1
    assert "is not fixed-base" in problems[0]


def test_the_committed_openarm_units_carry_no_fixture_yet() -> None:
    # A fixture needs a survey and live map verification first (thor.yaml's footer).
    for unit in ("thor", "orin"):
        assert load_robot_unit(_OPENARM, unit).fixtures == []


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"id": "shelf_top"}, "must be 'cell:<name>'"),
        ({"id": "cell:"}, "must be 'cell:<name>'"),
        ({"frame_id": "world"}, "pose is in 'openarm_base'"),
        ({"half_extents": [0.2, 0.0, 0.01]}, "finite and positive"),
        ({"half_extents": [1.6, 0.1, 0.01]}, "1.5 m place-region bound"),
        ({"half_extents": [1.5, 1.5, 1.5]}, "8.0 m\\^3 place-region bound"),
        ({"top_plane_normal": [0.0, 0.0, 2.0]}, "unit vector"),
        ({"survey_uncertainty_m": 0.0}, "greater than 0"),
        ({"survey_uncertainty_m": 0.06}, "less than or equal to 0.05"),
        ({"method": "  "}, "not provenance"),
        ({"surveyed_on": "yesterday"}, "date"),
        (
            {"pose": {"xyz": [0, 0, 0], "quat_xyzw": [0, 0, 0, 2], "frame_id": "openarm_base"}},
            "unit length",
        ),
        ({"extra": 1}, "Extra inputs"),
    ],
)
def test_a_malformed_fixture_is_refused(patch: dict[str, object], match: str) -> None:
    raw = {**_shelf_doc()["fixtures"][0], **patch}  # type: ignore[index]
    with pytest.raises(ValidationError, match=match):
        UnitFixture.model_validate(raw)


def test_fixture_ids_are_unique() -> None:
    doc = _shelf_doc()
    doc["fixtures"] = [doc["fixtures"][0]] * 2  # type: ignore[index]
    with pytest.raises(ValidationError, match="declared more than once"):
        RobotUnit.model_validate(doc)


def test_declared_fixture_evidence_kind_is_on_the_wire_string() -> None:
    assert AttachmentEvidenceKind("declared_fixture") is AttachmentEvidenceKind.DECLARED_FIXTURE


def test_thor_candidate_bench_fixture_template_loads_once_uncommented(tmp_path: Path) -> None:
    # thor.yaml ends in a commented candidate fixture; uncommenting it (stripping "#   ") must
    # yield a loadable unit (Pose6D needs quat_xyzw; an rpy pose breaks every Thor deploy).
    lines = (_ROOT / "robots" / "openarm" / "units" / "thor.yaml").read_text().splitlines()
    start = lines.index("#   fixtures:")
    block = [line.removeprefix("#   ") for line in lines[start:] if line.startswith("#   ")]
    assert len(block) == len(lines) - start
    robot_dir = tmp_path / "openarm"
    (robot_dir / "units").mkdir(parents=True)
    shutil.copy(_OPENARM, robot_dir / "robot.yaml")
    (robot_dir / "units" / "thor.yaml").write_text(
        "\n".join([*lines[:start], *block, ""]), encoding="utf-8"
    )
    bench = load_robot_unit(robot_dir / "robot.yaml", "thor").fixture("cell:bench_top")
    assert bench.pose.quat_xyzw == (0.0, 0.0, 0.0, 1.0)
