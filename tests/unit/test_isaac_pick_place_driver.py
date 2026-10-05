"""The scripted Isaac pick/place driver's pure geometry, and its scene + rSkill fixtures.

``tools/isaac_pick_place/driver.py`` chooses the closing axis from the voxel-map
footprint, centres the jaws from the measured grasp region and re-aims only on a region
that saw the whole footprint. These are the three decisions that made or broke a grasp
in the Isaac pick/place runs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _REPO_ROOT / "tools" / "isaac_pick_place"
sys.path.insert(0, str(_TOOL))

import driver  # noqa: E402


def _brick_columns(
    yaw: float, length: float = 0.075, width: float = 0.05, res: float = 0.015
) -> np.ndarray:
    """Occupied column centres of a ``length`` x ``width`` footprint rotated by ``yaw``."""
    xs = np.arange(-length / 2 + res / 2, length / 2, res)
    ys = np.arange(-width / 2 + res / 2, width / 2, res)
    pts = np.array([(x, y) for x in xs for y in ys])
    c, s = math.cos(yaw), math.sin(yaw)
    return pts @ np.array([[c, s], [-s, c]]) + np.array([0.27, -0.2])


def _axis_err_deg(a: float, b: float) -> float:
    """Angle between two undirected horizontal axes, in degrees."""
    return abs(math.degrees((a - b + math.pi / 2) % math.pi - math.pi / 2))


@pytest.mark.parametrize("yaw_deg", [0.0, 20.0, -35.0, 90.0])
def test_footprint_minor_yaw_is_the_narrow_axis(yaw_deg: float) -> None:
    yaw = math.radians(yaw_deg)
    minor = driver.footprint_minor_yaw(_brick_columns(yaw))
    # The narrow (width) axis is the footprint's local y: yaw + 90 deg.
    assert _axis_err_deg(minor, yaw + math.pi / 2) < 1e-6


def test_footprint_counts_each_column_once() -> None:
    """Walls the camera sees add cells but no columns: the axis must not lean toward them."""
    cols = _brick_columns(0.0)
    # Camera-facing walls: four stacked cells on the two columns of one corner.
    corner = cols[(cols[:, 0] >= np.sort(cols[:, 0])[-4]) & (cols[:, 1] == cols[:, 1].max())]
    cells = np.vstack([cols, *(corner for _ in range(4))])
    # Per cell, the stack drags the axis off the brick's; per column it does not.
    assert _axis_err_deg(driver.footprint_minor_yaw(cells), math.pi / 2) > 3.0
    assert _axis_err_deg(driver.footprint_minor_yaw(np.unique(cells, axis=0)), math.pi / 2) < 1e-6


def test_grasp_centre_takes_the_region_across_the_jaws_and_the_cluster_along_them() -> None:
    closing_yaw = math.pi / 2  # jaws close along y
    cluster = np.array([0.27, 0.20])
    # The region sits 1 cm off across the jaws (y) and 2 cm short along them (x).
    region = cluster + np.array([-0.02, 0.01])
    centre = driver.grasp_centre(region, cluster, closing_yaw)
    np.testing.assert_allclose(centre, [0.27, 0.21], atol=1e-12)


def _region(xy: tuple[float, float], half: tuple[float, float], yaw: float) -> dict[str, object]:
    return {"xyz": [xy[0], xy[1], -0.44], "half": [half[0], half[1], 0.025], "yaw": yaw}


def test_a_region_that_saw_the_whole_footprint_covers_it() -> None:
    cols = _brick_columns(0.0)
    assert driver.region_covers_footprint(_region((0.27, -0.2), (0.0375, 0.025), 0.0), cols, 0.015)


def test_a_partial_view_region_does_not_cover_the_footprint() -> None:
    """The hovering hand hides the far half: such a region may not re-aim the jaws."""
    cols = _brick_columns(0.0)
    near_half = _region((0.27 - 0.02, -0.2), (0.018, 0.025), math.radians(40.0))
    assert not driver.region_covers_footprint(near_half, cols, 0.015)


def test_the_validation_scene_and_driver_rskill_load() -> None:
    from openral_core import DeployScene, RSkillManifest, load_scene_strict

    scene = load_scene_strict(str(_TOOL / "scene.yaml"), DeployScene)
    assert scene.robot_id == "openarm"
    assert scene.runtime is not None and scene.runtime.grasp_allowance_enabled
    manifest = RSkillManifest.from_yaml(str(_TOOL / "rskill" / "rskill.yaml"))
    assert manifest.model_family == "isaac_pick_place_scripted"
