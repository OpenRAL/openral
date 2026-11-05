"""A policy-state skill enters the seeded palette only while the topic is published (#474).

The in-tree GR00T N1.7 BEHAVIOR-1K radio skill (``state_contract.dim: 61``,
``use_world_state_policy_state``) on the real 22-joint R1Pro: before #474 the
state-contract gate compared 61 with the joint count and always dropped it. It is now
gated on a ``/openral/policy_state`` publisher instead. With none it is dropped; when one
appears the reasoner's watch re-seeds the palette and admits it, and when it goes (a HAL
deactivating) the skill is dropped again. The HAL and
the reasoner autostart independently in ``deploy_e2e``, so the publisher can show up
after the seed. ``kind: wam`` is the same manifest re-kinded (no in-tree WAM sets the flag).

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like
``test_reasoner_palette_preserves_detector_coverage.py``::

    just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest \\
        tests/integration/test_reasoner_palette_policy_state_gate.py -v \\
        -p no:launch_testing -p no:launch_ros
"""

from __future__ import annotations

import os
import pathlib
import shutil
import time

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy node construction — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "and source install/setup.bash first."
)

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_RADIO = _REPO_ROOT / "rskills" / "gr00t-n17-b1k-turning-on-radio"
_ROBOT_YAML = _REPO_ROOT / "robots" / "r1pro" / "robot.yaml"
_SKILL_ID = "OpenRAL/rskill-gr00t_n17-r1pro-turning_on_radio-bf16"


def _skill_dir(tmp_path: pathlib.Path, kind: str) -> pathlib.Path:
    root = tmp_path / kind
    skill = root / _RADIO.name
    shutil.copytree(_RADIO, skill)
    manifest = skill / "rskill.yaml"
    text = manifest.read_text()
    assert 'kind: "vla"' in text
    manifest.write_text(text.replace('kind: "vla"', f'kind: "{kind}"', 1))
    return root


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
@pytest.mark.parametrize("kind", ["vla", "wam"])
def test_policy_state_skill_follows_the_publisher(tmp_path: pathlib.Path, kind: str) -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_reasoner_ros import ReasonerNode
    from std_msgs.msg import Float32MultiArray

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"robot_yaml:={_ROBOT_YAML}",
            "-p",
            "grasp_target_voxel_m:=0.05",
            "-p",
            f"rskill_search_paths:=[{_skill_dir(tmp_path, kind)}]",
        ],
    )
    hal_stand_in = None
    try:
        reasoner = ReasonerNode(tick_hz=2.0, client=FakeToolUseClient(responses=[]))
        reasoner.trigger_configure()
        assert _SKILL_ID not in reasoner._palette.execute_rskill_ids, (
            "no /openral/policy_state publisher: the skill's every dispatch would fail"
        )

        # The HAL's publisher, as it appears on the first frame carrying a policy state.
        hal_stand_in = rclpy.create_node("openral_policy_state_hal_stand_in")
        hal_stand_in.create_publisher(Float32MultiArray, "/openral/policy_state", 10)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _SKILL_ID not in reasoner._palette.execute_rskill_ids:
            rclpy.spin_once(reasoner, timeout_sec=0.1)
        assert _SKILL_ID in reasoner._palette.execute_rskill_ids, (
            "the policy-state watch did not re-seed the palette once the topic appeared; "
            f"got {sorted(reasoner._palette.execute_rskill_ids)!r}"
        )
        assert reasoner._policy_state_watch is not None, "the watch must outlive admission"

        # The HAL deactivates and destroys its publisher: the skill must leave the palette
        # before its last sample goes stale under a dispatch.
        hal_stand_in.destroy_node()
        hal_stand_in = None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _SKILL_ID in reasoner._palette.execute_rskill_ids:
            rclpy.spin_once(reasoner, timeout_sec=0.1)
        assert _SKILL_ID not in reasoner._palette.execute_rskill_ids, (
            "the skill stayed in the palette after /openral/policy_state lost its publisher"
        )

        reasoner.trigger_cleanup()
        reasoner.destroy_node()
    finally:
        if hal_stand_in is not None:
            hal_stand_in.destroy_node()
        rclpy.shutdown()
