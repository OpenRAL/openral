"""The nodes' own perception caps equal ``DeployRuntime``'s (a deliberate C++/Pydantic mirror).

``DeployRuntime`` refuses a scene whose rig values exceed the hard caps, but the safety
kernel, the robot self-filter and the octomap voxel bridge can be launched without it
(``ros2 run``, another launch file), so each node refuses the same values itself. The C++
side cannot import the schema, so the numbers are written twice; this pins them equal.
Drift in either direction is a hole: a looser node accepts what the schema would refuse, a
tighter one refuses a scene the schema passed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from openral_core import DeployRuntime
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KERNEL_H = (
    _REPO_ROOT / "cpp/openral_safety_kernel/include/openral_safety_kernel/lifecycle_kernel.hpp"
)
_BRIDGE_INC = _REPO_ROOT / "packages/openral_octomap_bridge/include/openral_octomap_bridge"

# (header, C++ constant, scale to the schema's unit, DeployRuntime field)
_MIRRORS = [
    (_KERNEL_H, "kMaxWorldVoxelDeadlineMs", 1e-3, "world_voxel_deadline_s"),
    (_KERNEL_H, "kMaxWorldVoxelDataAgeBudgetMs", 1e-3, "world_voxel_data_age_budget_s"),
    (_BRIDGE_INC / "octree_freshness.hpp", "kMaxOctreeAgeS", 1.0, "world_voxel_deadline_s"),
    (_BRIDGE_INC / "self_filter.hpp", "kMaxSelfFilterPaddingM", 1.0, "robot_self_filter_padding_m"),
]


def _cpp_constant(header: Path, name: str) -> float:
    match = re.search(
        rf"inline constexpr double {name} = ([0-9.eE+-]+);", header.read_text(encoding="utf-8")
    )
    assert match, f"{name} not found in {header}"
    return float(match.group(1))


@pytest.mark.parametrize(("header", "name", "scale", "field"), _MIRRORS)
def test_node_cap_equals_the_deploy_runtime_cap(
    header: Path, name: str, scale: float, field: str
) -> None:
    cap = _cpp_constant(header, name) * scale
    schema = DeployRuntime.model_json_schema()["properties"][field]
    assert schema["maximum"] == pytest.approx(cap), f"{name} drifted from {field}"
    DeployRuntime.model_validate({field: cap})
    with pytest.raises(ValidationError, match=field):
        DeployRuntime.model_validate({field: cap * 1.001})


def test_kernel_data_age_budget_default_equals_the_deploy_runtime_default() -> None:
    default_s = _cpp_constant(_KERNEL_H, "kDefaultWorldVoxelDataAgeBudgetMs") * 1e-3
    assert DeployRuntime().world_voxel_data_age_budget_s == pytest.approx(default_s)
