"""Two perception capabilities that existed but never reached the reasoner.

1. ``query_scene`` was never offered by ``deploy sim``. It's gated on a
   ``scene_query_available`` ROS param that no launch file or CLI path ever
   set, so the Qwen VLM rSkill (``rskills/qwen35-4b-nf4``), ``scene_vlm_node``
   (serves ``/openral/perception/query_scene``), and the reasoner's
   ``_dispatch_query_scene`` were all wired to each other but unreachable.
2. ``resolve_place`` never escalated to the open-vocab detector like
   ``recall_object`` does via ``locate_in_view``. Its miss fell through to a
   "not in memory" re-prompt, so the LLM just re-picked the same tool.
   Observed live on ``robocasa_baguette``: ticks 10, 11, 22 all re-selected
   ``resolve_place`` for 'the counter' while an ``any-indoor`` detector sat
   idle in the same graph.

No mocks (CLAUDE.md §1.11): real in-tree scene YAML, real ``DeployRuntime``
schema, real launch-argv construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.schemas import DeployRuntime

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BAGUETTE = _REPO_ROOT / "scenes" / "deploy" / "robocasa_baguette.yaml"
_QWEN_MANIFEST = _REPO_ROOT / "rskills" / "qwen35-4b-nf4" / "rskill.yaml"


def test_scene_vlm_rskill_and_node_exist() -> None:
    """The pieces query_scene depends on are really in-tree."""
    assert _QWEN_MANIFEST.is_file(), "the kind:vlm rSkill backing query_scene is missing"
    node = (
        _REPO_ROOT
        / "packages"
        / "openral_perception_ros"
        / "openral_perception_ros"
        / "scene_vlm_node.py"
    )
    assert node.is_file(), "scene_vlm_node serves /openral/perception/query_scene"
    cmake = (_REPO_ROOT / "packages" / "openral_perception_ros" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    assert "scene_vlm_node.py" in cmake, (
        "scene_vlm_node must be install(PROGRAMS ...)'d or the launch "
        "executable= lookup fails at runtime"
    )


def test_deploy_runtime_accepts_the_scene_vlm_knobs() -> None:
    """A scene may pin the scene VLM; omitting it stays exactly as before."""
    assert "enable_scene_vlm" in DeployRuntime.model_fields
    assert "scene_vlm_manifest" in DeployRuntime.model_fields
    # Backward-compatible addition: an existing scene that says nothing is unchanged.
    assert DeployRuntime().enable_scene_vlm is None
    assert DeployRuntime().scene_vlm_manifest is None
    assert DeployRuntime(enable_scene_vlm=True).enable_scene_vlm is True


def _launch_argv(**kwargs: object) -> str:
    """Resolve a real deploy-sim invocation for the baguette scene."""
    from openral_cli.deploy_sim import resolve_launch_invocation

    inv = resolve_launch_invocation(
        config=_BAGUETTE,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        **kwargs,  # type: ignore[arg-type]
    )
    return " ".join(inv.argv_template)


def test_scene_vlm_off_by_default_and_on_with_the_flag() -> None:
    """The launch arg tracks the CLI flag — the whole point of the wiring."""
    assert "enable_scene_vlm:=false" in _launch_argv()
    assert "enable_scene_vlm:=true" in _launch_argv(enable_scene_vlm=True)


def test_scene_vlm_manifest_only_forwarded_when_enabled() -> None:
    """An explicit manifest rides along only when the leg is actually up."""
    argv_on = _launch_argv(enable_scene_vlm=True, scene_vlm_manifest=str(_QWEN_MANIFEST))
    assert f"scene_vlm_manifest:={_QWEN_MANIFEST}" in argv_on
    # ros2 launch rejects an empty ``name:=`` value, so an unset manifest must
    # not be forwarded at all; the launch file owns the in-tree default.
    assert "scene_vlm_manifest:=" not in _launch_argv(enable_scene_vlm=True)
    assert "scene_vlm_manifest:=" not in _launch_argv(scene_vlm_manifest=str(_QWEN_MANIFEST))


def test_search_term_reads_both_variants() -> None:
    """Both search tools yield an escalation term, despite naming it differently.

    ``recall_object`` carries ``query`` and ``resolve_place`` carries
    ``reference``; that mismatch is why the ``locate_in_view`` escalation was
    once gated on ``isinstance(call, RecallObjectTool)`` and skipped places.
    """
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    from openral_core import RecallObjectTool, ResolvePlaceTool
    from openral_reasoner_ros.reasoner_node import _search_term

    assert _search_term(RecallObjectTool(query="baguette")) == "baguette"
    # The regression: this used to raise AttributeError if reached at all, and
    # was simply never reached — a place miss looped on "not in memory" instead.
    assert _search_term(ResolvePlaceTool(reference="the counter")) == "the counter"
