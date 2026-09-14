"""The palette seed must not silently drop on-demand locators / continuous detectors.

``_maybe_seed_palette_from_search_paths`` computes an early ``capability_palette``
from the FULL manifest scan purely to derive ``execute_rskill_ids`` (the
capability-matching narrowing step), then rebuilds the final palette from
``importable`` — manifests filtered down through the ros-server-availability,
state-contract, action-mode and import-deps gates, ALL of which apply only to
``kind: vla`` / ``kind: ros_action`` / ``kind: ros_service`` ExecuteRskill
candidates. A ``kind: detector`` manifest is never in ``execute_rskill_ids``
(``build_tool_palette``'s own docstring: detectors are perception producers,
not ExecuteRskill-dispatchable) and so was silently dropped from
``importable`` before the final ``build_tool_palette`` call — meaning the
final palette's ``continuous_detectors`` / ``on_demand_detectors`` were always
empty, no matter what detectors were actually installed and matched the robot.

Reproduced live: ``openral deploy sim --config scenes/deploy/libero_pnp.yaml
--object-detector-locator rskills/omdet-turbo-locator/rskill.yaml
--initial-task "..."`` correctly offered the ``locate_in_view`` tool
(``detector_available`` was true), and the reasoner correctly dispatched it —
but its ``known_aliases`` (``{d.alias for d in palette.on_demand_detectors}``)
was empty, so an LLM-supplied alias could never validate and the tool's own
description never listed the locator as a selectable option.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like
``test_reasoner_palette_primes_vram_gate.py``, which this mirrors::

    just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest \\
        tests/integration/test_reasoner_palette_preserves_detector_coverage.py -v \\
        -p no:launch_testing -p no:launch_ros
"""

from __future__ import annotations

import os
import pathlib

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy node construction — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "and source install/setup.bash first."
)

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_RSKILLS = _REPO_ROOT / "rskills"
# franka_panda, not so101_follower — the exact live reproducer's robot, and
# omdet-turbo-locator's embodiment_tags=["any"] covers either.
_ROBOT_YAML = _REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_seeded_palette_keeps_the_on_demand_locator_selectable() -> None:
    """The exact live symptom: a real on-demand locator must survive the seed."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_reasoner_ros import ReasonerNode

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"robot_yaml:={_ROBOT_YAML}",
            "-p",
            f"rskill_search_paths:=[{_RSKILLS}]",
        ],
    )
    try:
        reasoner = ReasonerNode(tick_hz=2.0, client=FakeToolUseClient(responses=[]))
        reasoner.trigger_configure()

        aliases = {d.alias for d in reasoner._palette.on_demand_detectors}
        assert "omdet_turbo-any-locator-fp16" in aliases, (
            "the seeded palette's on_demand_detectors is missing the in-tree "
            "omdet-turbo-locator — a real, capability-matched locator, silently dropped "
            "by the execute_rskill-only narrowing between the early capability_palette "
            "scan and the final build_tool_palette call. Reproduced live: locate_in_view "
            f"was offered with zero selectable options. Got: {sorted(aliases)!r}"
        )

        continuous_ids = {d.rskill_id for d in reasoner._palette.continuous_detectors}
        assert continuous_ids, (
            "the seeded palette's continuous_detectors is empty — same drop, the "
            "always-on background bank (e.g. rtdetr-coco-r18 / omdet-turbo-indoor)."
        )

        reasoner.trigger_cleanup()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()
