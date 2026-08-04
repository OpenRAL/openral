"""The palette seed must prime the VRAM gate's manifest cache.

`_refuse_unfittable_vla` refuses a VLA that cannot co-reside with the reward
model — but only when `_manifest_for_rskill` can resolve the dispatched id. That
lookup used to consult the install registry
(`~/.local/share/openral/rskills.json`) alone, which a palette seeded from
`rskill_search_paths` never populates: on the reference host the palette carried
48 manifests while the registry held 3 detector skills and no VLA at all, so the
gate early-returned `False` for **every** VLA and both its tiers were dead.

The refusal test itself could not catch this — it stubbed `_manifest_for_rskill`
out. This one drives the real seed against the repo's own `rskills/` tree and
asserts the production lookup resolves what the LLM is offered.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` like the rest of the live reasoner suite::

    just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest \\
        tests/integration/test_reasoner_palette_primes_vram_gate.py -v \\
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
_ROBOT_YAML = _REPO_ROOT / "robots" / "so101_follower" / "robot.yaml"


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_seeded_palette_resolves_every_vla_for_the_vram_gate() -> None:
    """Every VLA the palette offers must resolve through `_manifest_for_rskill`.

    An id the LLM can dispatch but the gate cannot resolve is an unguarded
    dispatch — the exact shape of the live molmoact2 failure.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_reasoner_ros import ReasonerNode

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    # `_seed_palette` needs BOTH a robot_yaml (it capability-filters against the
    # description) and a non-empty rskill_search_paths, or it returns early.
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
        # A client must be injected or `on_configure` aborts on the missing LLM
        # env before `_seed_palette` ever runs — which would make this test fail
        # for a reason unrelated to what it is asserting.
        reasoner = ReasonerNode(tick_hz=2.0, client=FakeToolUseClient(responses=[]))
        reasoner.trigger_configure()
        assert reasoner._manifests_by_id, (
            "on_configure did not seed the manifest cache — either the palette "
            "seed was skipped or it stopped priming the VRAM gate"
        )

        vla_names = {
            m.name
            for path in sorted(_RSKILLS.glob("*/rskill.yaml"))
            for m in [_load(path)]
            if m is not None and m.kind == "vla"
        }
        assert vla_names, f"no VLA manifests under {_RSKILLS} — fixture is empty"

        unresolvable = sorted(
            name for name in vla_names if reasoner._manifest_for_rskill(name) is None
        )
        assert not unresolvable, (
            f"{len(unresolvable)} VLA(s) in the palette cannot be resolved by the "
            f"VRAM gate, so their dispatch is unguarded: {unresolvable}"
        )

        # And the resolution came from the seed, not the install registry —
        # otherwise this passes only on a host that happens to have them installed.
        sample = next(iter(vla_names))
        assert sample in reasoner._manifests_by_id

        reasoner.trigger_cleanup()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()


def _load(path: pathlib.Path) -> object | None:
    from openral_core import RSkillManifest

    try:
        return RSkillManifest.from_yaml(str(path))
    except (OSError, ValueError):
        return None
