"""The reasoner palette keeps a 16-D OpenArm rSkill on the 20-joint OpenAMRobot v2.

``_maybe_seed_palette_from_search_paths`` drops a VLA whose ``state_contract.dim`` is not
the robot's joint count. The OpenAMRobot v2 adds three base joints and a lift to the exact
OpenArm 16, and the runner feeds an OpenArm skill exactly the joints its
``action_contract.slots`` name (``slot_state_joint_names``), so the palette admits it.
A 16-D skill whose slots do not account for its state is still dropped.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like
``test_reasoner_palette_preserves_detector_coverage.py``::

    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest \\
        tests/integration/test_reasoner_palette_slot_state_subset.py -v \\
        -p no:launch_testing -p no:launch_ros
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from openral_core import RSkillManifest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT_YAML = _REPO_ROOT / "robots" / "openamrobot_v2" / "robot.yaml"


@pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy node construction — set OPENRAL_TEST_ROS_LIVE=1 and source "
    "install/setup.bash first.",
)
def test_palette_admits_the_openarm_skill_and_drops_an_unaccounted_state(
    tmp_path: Path, require_policy_runtime: Callable[[RSkillManifest], None]
) -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_reasoner_ros import ReasonerNode

    from tests.integration.fakes.fake_llm import FakeToolUseClient
    from tests.unit.test_openarm_skill_on_openamrobot_v2 import openarm_pi05_manifest

    keep = openarm_pi05_manifest()
    require_policy_runtime(keep)
    # Same checkpoint contract with the gripper slots folded into whole-vector
    # joint actions: the 16-D state no longer names its joints, so on a 20-joint
    # robot it is the misfeed the filter exists to refuse.
    assert keep.action_contract is not None
    drop = openarm_pi05_manifest(
        name="OpenRAL/rskill-pi05-openarm-restock_shelf-unslotted-bf16",
        action_contract={"dim": 16, "joint_units": "radians"},
    )
    for manifest in (keep, drop):
        skill_dir = tmp_path / manifest.name.split("/")[-1]
        skill_dir.mkdir()
        (skill_dir / "rskill.yaml").write_text(
            yaml.safe_dump(manifest.model_dump(mode="json", exclude_none=True))
        )

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"robot_yaml:={_ROBOT_YAML}",
            "-p",
            "grasp_target_voxel_m:=0.05",
            "-p",
            f"rskill_search_paths:=[{tmp_path}]",
        ],
    )
    try:
        reasoner = ReasonerNode(tick_hz=2.0, client=FakeToolUseClient(responses=[]))
        reasoner.trigger_configure()
        ids = set(reasoner._palette.execute_rskill_ids)
        reasoner.trigger_cleanup()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()
    assert keep.name in ids, sorted(ids)
    assert drop.name not in ids, sorted(ids)
