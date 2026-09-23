# SPDX-License-Identifier: Apache-2.0
"""The skill runner ticks at the manifest's control rate unless told otherwise.

Issue #303: the real ros2_control HAL derives every trajectory deadline from
`action_spec.control_freq_hz`, so the runner must tick at that same value or
the controller is again asked to cover each step in the wrong time.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from openral_core.schemas import RobotDescription

REPO_ROOT = Path(__file__).resolve().parents[2]

runner = importlib.import_module("openral_rskill_ros.rskill_runner_node")


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [("robots/openarm/robot.yaml", 30.0), ("robots/aloha_bimanual/robot.yaml", 50.0)],
)
def test_an_unset_param_reads_the_manifest(manifest: str, expected: float) -> None:
    desc = RobotDescription.from_yaml(str(REPO_ROOT / manifest))
    assert runner.resolve_control_rate_hz(0.0, desc) == expected


def test_an_explicit_param_overrides_the_manifest() -> None:
    desc = RobotDescription.from_yaml(str(REPO_ROOT / "robots/openarm/robot.yaml"))
    assert runner.resolve_control_rate_hz(15.0, desc) == 15.0


def test_a_manifest_without_a_control_rate_falls_back_to_30_hz() -> None:
    desc = RobotDescription.from_yaml(str(REPO_ROOT / "robots/openarm/robot.yaml"))
    rateless = desc.model_copy(update={"action_spec": None})
    assert runner.resolve_control_rate_hz(0.0, rateless) == 30.0
    assert runner.resolve_control_rate_hz(0.0, None) == 30.0
