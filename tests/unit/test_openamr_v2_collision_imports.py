"""OpenAMR v2's arm collision geometry is OpenArm's, imported at load and renamed.

``robots/openamr_v2/robot.yaml`` declares ``collision_geometry_imports`` from
``robots/openarm`` instead of a copy, so an OpenArm re-lowering reaches this robot.
The import is valid only because the arm link frames are the same in both
manifests: every arm joint's and mimic's origin and axis must match.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import CollisionGeometryImport, RobotDescription
from openral_core._safe_yaml import safe_load
from pydantic import ValidationError

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"
_AMR = _ROBOTS / "openamr_v2" / "robot.yaml"


@pytest.fixture(scope="module")
def amr() -> RobotDescription:
    return RobotDescription.from_yaml(str(_AMR))


@pytest.fixture(scope="module")
def openarm() -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))


def _link_map(amr: RobotDescription) -> dict[str, str]:
    (imp,) = amr.collision_geometry_imports
    assert imp.robot == "openarm"
    return imp.link_map


def test_every_imported_volume_is_openarms_renamed(
    amr: RobotDescription, openarm: RobotDescription
) -> None:
    link_map = _link_map(amr)
    for src, dst in link_map.items():
        want = [
            g.model_copy(update={"link_name": dst})
            for g in openarm.collision_geometry
            if g.link_name == src
        ]
        got = [g for g in amr.collision_geometry if g.link_name == dst]
        assert want and got == want, dst


def test_the_arm_link_frames_match_between_the_two_robots(
    amr: RobotDescription, openarm: RobotDescription
) -> None:
    """Same joint origin and axis for every arm joint and mimic, so link frames agree."""
    link_map = _link_map(amr)
    amr_joints = {j.child_link: j for j in amr.joints}
    for j in openarm.joints:
        if j.child_link not in link_map or j.parent_link not in link_map:
            continue
        mine = amr_joints[link_map[j.child_link]]
        assert mine.parent_link == link_map[j.parent_link], j.name
        assert mine.origin_xyz == pytest.approx(j.origin_xyz), j.name
        assert mine.origin_rpy == pytest.approx(j.origin_rpy), j.name
        assert mine.axis_xyz == pytest.approx(j.axis_xyz), j.name
    amr_mimics = {m.child_link: m for m in amr.mimic_joints}
    for m in openarm.mimic_joints:
        if m.child_link in link_map:
            mine_m = amr_mimics[link_map[m.child_link]]
            assert mine_m.origin_xyz == pytest.approx(m.origin_xyz), m.child_link
            assert mine_m.origin_rpy == pytest.approx(m.origin_rpy), m.child_link
            assert mine_m.axis_xyz == pytest.approx(m.axis_xyz), m.child_link


def test_unresolved_imports_are_refused() -> None:
    """Validated from a bare dict, nothing reads robots/openarm: refuse, never no geometry."""
    with _AMR.open(encoding="utf-8") as fh:
        raw = safe_load(fh)
    with pytest.raises(ValidationError, match="unresolved"):
        RobotDescription.model_validate(raw)


def test_a_link_both_authored_and_imported_is_refused(tmp_path: Path) -> None:
    with _AMR.open(encoding="utf-8") as fh:
        raw = safe_load(fh)
    assert isinstance(raw, dict)
    raw["collision_geometry"].append(
        {
            "link_name": "openarm_left_link1",
            "shape": {"shape": "sphere", "radius_m": 0.05},
            "origin_xyz_rpy": [0.0] * 6,
        }
    )
    robot_dir = tmp_path / "robots" / "openamr_v2"
    robot_dir.mkdir(parents=True)
    (tmp_path / "robots" / "openarm").symlink_to(_ROBOTS / "openarm")
    import yaml

    (robot_dir / "robot.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="both authored and imported"):
        RobotDescription.from_yaml(str(robot_dir / "robot.yaml"))


def test_import_model_refuses_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        CollisionGeometryImport.model_validate({"robot": "openarm", "links": {"a": "b"}})


def test_a_loaded_manifest_dumped_and_loaded_again_round_trips(tmp_path: Path) -> None:
    """A resolved manifest carries the imported volumes; reloading it must not refuse them."""
    import yaml

    loaded = RobotDescription.from_yaml(str(_AMR))
    robot_dir = tmp_path / "robots" / "openamr_v2"
    robot_dir.mkdir(parents=True)
    (tmp_path / "robots" / "openarm").symlink_to(_ROBOTS / "openarm")
    (robot_dir / "robot.yaml").write_text(
        yaml.safe_dump(loaded.model_dump(mode="json")), encoding="utf-8"
    )
    again = RobotDescription.from_yaml(str(robot_dir / "robot.yaml"))
    assert again.collision_geometry == loaded.collision_geometry


def test_two_imports_onto_one_link_are_refused(tmp_path: Path) -> None:
    import yaml

    with _AMR.open(encoding="utf-8") as fh:
        raw = safe_load(fh)
    assert isinstance(raw, dict)
    raw["collision_geometry_imports"].append(
        {"robot": "openarm", "link_map": {"openarm_left_link2": "openarm_left_link1"}}
    )
    robot_dir = tmp_path / "robots" / "openamr_v2"
    robot_dir.mkdir(parents=True)
    (tmp_path / "robots" / "openarm").symlink_to(_ROBOTS / "openarm")
    (robot_dir / "robot.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="map two links onto"):
        RobotDescription.from_yaml(str(robot_dir / "robot.yaml"))
