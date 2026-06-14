"""Unit tests for the manifest-driven HAL lifecycle node (ADR-0032).

`make_lifecycle_main_from_manifest` builds a node whose `_create_hal()` reads
the `robot_yaml` + `hal_mode` ROS params and routes through the single resolver
seam `openral_hal.build_hal`. These tests pin that the node builds the *sim* HAL
in sim mode and the *real* HAL in real mode for every robot, driven only by the
manifest's `hal{sim,real}` block — and that a sim-only robot asked for real (or a
missing `robot_yaml`) fails with a typed error.

Skipped where ROS 2 (`rclpy`) is unavailable (e.g. a GPU-less CI runner).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rclpy")

import rclpy  # reason: import-after-skip is the rclpy test idiom
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_hal.lifecycle import _ManifestHALLifecycleNode
from rclpy.parameter import Parameter

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def _rclpy_ctx() -> object:
    rclpy.init()
    yield
    rclpy.shutdown()


def _build(node_name: str, robot_id: str, mode: str) -> object:
    node = _ManifestHALLifecycleNode(node_name)
    node.set_parameters(
        [
            Parameter("robot_yaml", value=str(REPO_ROOT / "robots" / robot_id / "robot.yaml")),
            Parameter("hal_mode", value=mode),
        ]
    )
    try:
        return node._create_hal()
    finally:
        node.destroy_node()


@pytest.mark.usefixtures("_rclpy_ctx")
class TestManifestNode:
    """sim_mode → sim HAL; real_mode → real HAL; missing → typed errors."""

    def test_sim_mode_builds_sim_hal(self) -> None:
        from openral_hal.franka_panda import FrankaPandaHAL

        assert isinstance(_build("t_franka_sim", "franka_panda", "sim"), FrankaPandaHAL)

    def test_real_mode_builds_real_hal(self) -> None:
        from openral_hal.franka_panda_real import FrankaPandaRealHAL

        assert isinstance(_build("t_franka_real", "franka_panda", "real"), FrankaPandaRealHAL)

    def test_sim_only_robot_real_mode_raises(self) -> None:
        with pytest.raises(ROSCapabilityMismatch, match="g1"):
            _build("t_g1_real", "g1", "real")

    def test_missing_robot_yaml_raises(self) -> None:
        node = _ManifestHALLifecycleNode("t_noyaml")
        try:
            with pytest.raises(ROSConfigError, match="robot_yaml"):
                node._create_hal()
        finally:
            node.destroy_node()
