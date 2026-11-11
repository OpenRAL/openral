"""Unit tests for the manifest-driven HAL lifecycle node.

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

from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_hal.lifecycle import _ManifestHALLifecycleNode
from rclpy.parameter import Parameter

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build(node_name: str, robot_id: str, mode: str) -> object:
    node = _ManifestHALLifecycleNode(
        node_name,
        parameter_overrides=[
            Parameter("robot_yaml", value=str(REPO_ROOT / "robots" / robot_id / "robot.yaml")),
            Parameter("hal_mode", value=mode),
        ],
    )
    try:
        return node._create_hal()
    finally:
        node.destroy_node()


@pytest.mark.usefixtures("_rclpy_ctx")
class TestManifestNode:
    """sim_mode → sim HAL; real_mode → real HAL; missing → typed errors."""

    def test_sim_mode_builds_sim_hal(self) -> None:
        from openral_hal._mujoco_arm import MujocoArmHAL

        assert type(_build("t_franka_sim", "franka_panda", "sim")) is MujocoArmHAL

    def test_hal_transport_json_reaches_the_hal(self) -> None:
        """An un-allowlisted kwarg (galaxea ``host``) is forwarded, not dropped."""
        node = _ManifestHALLifecycleNode(
            "t_galaxea_json",
            parameter_overrides=[
                Parameter("robot_yaml", value=str(REPO_ROOT / "robots/galaxea_a1/robot.yaml")),
                Parameter("hal_mode", value="real"),
                Parameter("hal_transport_json", value='{"host": "127.0.0.2", "port": 46099}'),
            ],
        )
        try:
            hal = node._create_hal()
        finally:
            node.destroy_node()
        assert hal._host == "127.0.0.2"  # type: ignore[attr-defined] # reason: HAL-private introspection
        assert hal._port == 46099  # type: ignore[attr-defined] # reason: HAL-private introspection

    def test_hal_transport_json_must_be_an_object(self) -> None:
        node = _ManifestHALLifecycleNode(
            "t_bad_json",
            parameter_overrides=[
                Parameter("robot_yaml", value=str(REPO_ROOT / "robots/galaxea_a1/robot.yaml")),
                Parameter("hal_mode", value="real"),
                Parameter("hal_transport_json", value="[1, 2]"),
            ],
        )
        try:
            with pytest.raises(ROSConfigError, match="JSON object"):
                node._create_hal()
        finally:
            node.destroy_node()

    def test_real_mode_builds_real_hal(self) -> None:
        from openral_hal.franka_panda_real import FrankaPandaRealHAL

        assert isinstance(_build("t_franka_real", "franka_panda", "real"), FrankaPandaRealHAL)

    @pytest.mark.parametrize("robot_id", ["h2"])
    def test_sim_only_robot_real_mode_raises(self, robot_id: str) -> None:
        with pytest.raises(ROSCapabilityMismatch, match=robot_id):
            _build(f"t_{robot_id}_real", robot_id, "real")

    def test_missing_robot_yaml_raises(self) -> None:
        node = _ManifestHALLifecycleNode("t_noyaml")
        try:
            with pytest.raises(ROSConfigError, match="robot_yaml"):
                node._create_hal()
        finally:
            node.destroy_node()


# Every parameter that selects code, files, hardware addresses or the sim/real mode;
# `on_configure` re-reads them, so a writable one lets any DDS peer pick what the
# next configure loads.
_LOCKED_PARAMS = [
    ("robot_yaml", "/tmp/attacker/robot.yaml"),
    ("hal_mode", "real"),
    ("sim_env_yaml", "/tmp/attacker/scene.yaml"),
    ("scene_composition_json", '{"composer": "os:system", "params": {}}'),
    ("hal_transport_json", '{"host": "203.0.113.7"}'),
    ("port", "/dev/ttyS9"),
    ("robot_ip", "203.0.113.7"),
    ("fci_ip", "203.0.113.7"),
    ("id", "attacker"),
    ("calibration_dir", "/tmp/attacker"),
    ("walking_enabled", True),
    ("tracking_fault_window_s", 0.0),
]


@pytest.mark.usefixtures("_rclpy_ctx")
class TestParametersAreReadOnly:
    """A peer's ``~/set_parameters`` cannot change what the next configure loads."""

    def test_a_remote_peer_cannot_set_hal_parameters(self) -> None:
        import rclpy
        import rclpy.executors
        from rclpy.parameter_client import AsyncParameterClient

        robot_yaml = str(REPO_ROOT / "robots/so101_follower/robot.yaml")
        node = _ManifestHALLifecycleNode(
            "t_readonly_hal", parameter_overrides=[Parameter("robot_yaml", value=robot_yaml)]
        )
        peer = rclpy.create_node("t_readonly_peer")
        try:
            client = AsyncParameterClient(peer, "t_readonly_hal")
            executor = rclpy.executors.SingleThreadedExecutor()
            executor.add_node(node)
            executor.add_node(peer)
            assert client.wait_for_services(timeout_sec=5.0)
            future = client.set_parameters(
                [Parameter(name, value=value) for name, value in _LOCKED_PARAMS]
            )
            executor.spin_until_future_complete(future, timeout_sec=5.0)
            results = future.result().results
            assert [r.successful for r in results] == [False] * len(_LOCKED_PARAMS)
            assert all("read-only" in r.reason for r in results)
            # The launch value stands; nothing the peer sent took effect.
            assert node.get_parameter("robot_yaml").value == robot_yaml
            assert node.get_parameter("hal_mode").value == "sim"
        finally:
            peer.destroy_node()
            node.destroy_node()

    def test_a_late_declaration_stays_dynamic_and_leaves_the_descriptor_alone(self) -> None:
        """A name-only declare keeps rclpy's dynamic typing; a passed descriptor is copied."""
        from rcl_interfaces.msg import ParameterDescriptor

        node = _ManifestHALLifecycleNode("t_readonly_late")
        shared = ParameterDescriptor(description="shared with another node")
        try:
            node.declare_parameter("late_untyped")
            node.declare_parameter("late_typed", 1.5, shared)
            assert node.describe_parameter("late_untyped").read_only
            assert node.describe_parameter("late_untyped").dynamic_typing
            assert node.describe_parameter("late_typed").read_only
            assert shared.read_only is False
        finally:
            node.destroy_node()

    def test_every_declared_parameter_is_read_only(self) -> None:
        node = _ManifestHALLifecycleNode("t_readonly_all")
        try:
            names = list(node._parameters)
            assert {n for n, _ in _LOCKED_PARAMS} <= set(names)
            writable = [n for n in names if not node.describe_parameter(n).read_only]
            assert writable == []
        finally:
            node.destroy_node()


@pytest.mark.usefixtures("_rclpy_ctx")
def test_the_node_rechecks_a_composer_that_skipped_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composer import seam refuses foreign code even past the schema check."""
    from openral_core import RobotDescription, SceneComposition
    from openral_core.code_refs import ALLOW_EXTERNAL_PLUGINS_ENV

    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    description = RobotDescription.from_yaml(str(REPO_ROOT / "robots/openarm/robot.yaml"))
    hostile = SceneComposition.model_construct(composer="os:system", params={"command": "true"})
    node = _ManifestHALLifecycleNode("t_composer_recheck")
    try:
        with pytest.raises(ROSConfigError, match=r"composition\.composer = 'os:system'"):
            node._compose_scene_mjcf(description, hostile)
    finally:
        node.destroy_node()
