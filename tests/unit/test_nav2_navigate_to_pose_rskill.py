"""``rskill-nav2-navigate-to-pose`` manifest, loaded via the real loader.

Its MoveIt ``ros_action`` siblings each have a dedicated test naming them
(``test_look_at_rskill.py`` for ``rskill-moveit-look-at``; the generic
per-kind checks in ``test_rskill_manifest.py``), but nothing named this one.
No mocks (CLAUDE.md §1.11): loads the real on-disk manifest through
``openral_rskill.loader.load_rskill_manifest`` — the same resolver every
runtime dispatch path uses (``discover_intree_rskills``,
``openral_sim`` policy loading, ``openral benchmark run``) — and cross-checks
its declared contract against the real ROS wrapper package
(``packages/openral_nav2_bringup``) and its own README.
"""

from __future__ import annotations

import pathlib

from openral_core.schemas import RSkillAction
from openral_rskill.loader import load_rskill_manifest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SKILL_DIR = _REPO / "rskills" / "rskill-nav2-navigate-to-pose"
_URI = "rskills/rskill-nav2-navigate-to-pose"


def test_manifest_loads_through_the_real_loader() -> None:
    """The runtime-facing loader resolves the in-tree directory, not a stub."""
    m = load_rskill_manifest(_URI)
    assert m.name == "OpenRAL/rskill-nav2-mobile_base-navigate_to_pose"
    assert m.kind == "ros_action"
    assert m.role == "s1"
    assert m.license == "apache-2.0"


def test_embodiment_and_actuator_contract_matches_body_twist() -> None:
    """Generic ``mobile_base`` tag + a ``body_twist`` actuator, per the README."""
    m = load_rskill_manifest(_URI)
    assert m.embodiment_tags == ["mobile_base"]
    assert len(m.actuators_required) == 1
    actuator = m.actuators_required[0]
    assert actuator.kind == "body_twist"
    assert actuator.control_mode_semantics is not None
    assert actuator.control_mode_semantics.mode == "absolute"
    # `chunk_size: 1` is a hard schema requirement for every ros_action/
    # ros_service manifest (the safety supervisor only checks row 0 of a
    # chunk); assert the manifest actually declares it rather than relying
    # on the validator to have caught a regression silently.
    assert m.chunk_size == 1
    assert [a.value for a in m.actions] == [RSkillAction.NAVIGATE.value]
    assert m.scenes == ["indoor"]
    assert m.objects == []


def test_ros_integration_matches_the_wrapped_nav2_bringup_package() -> None:
    """The manifest's wrapped-server pointer matches what ``openral_nav2_bringup``
    and its Nav2 params file actually bring up (result-only mode: no
    ``result_trajectory_field``, so the adapter never emits an ``Action``
    chunk for Nav2's own ``/cmd_vel`` stream — see the manifest's own
    docstring on why that bypasses the safety supervisor)."""
    m = load_rskill_manifest(_URI)
    ri = m.ros_integration
    assert ri is not None
    assert ri.package == "nav2_msgs"
    assert ri.interface_type == "NavigateToPose"
    assert ri.interface_name == "/navigate_to_pose"
    assert ri.result_trajectory_field is None
    # nav2_panda_mobile.yaml's bt_navigator lists `navigate_to_pose` as one
    # of its `navigators`, and the config's velocity-smoother output topic
    # is `cmd_vel` — the exact bypass path the manifest documents.
    bringup_params = (
        _REPO / "packages" / "openral_nav2_bringup" / "config" / "nav2_panda_mobile.yaml"
    ).read_text(encoding="utf-8")
    assert '"navigate_to_pose"' in bringup_params
    assert 'cmd_vel_out_topic: "cmd_vel"' in bringup_params


def test_default_goal_is_stay_in_place_identity() -> None:
    """The default goal is a documented no-op (base_link identity), not the
    map origin — a misconfigured dispatch must not drive the robot."""
    import json

    m = load_rskill_manifest(_URI)
    ri = m.ros_integration
    assert ri is not None
    default_goal = json.loads(ri.default_goal_json)
    assert default_goal["pose"]["header"]["frame_id"] == "base_link"
    position = default_goal["pose"]["pose"]["position"]
    assert (position["x"], position["y"], position["z"]) == (0.0, 0.0, 0.0)
    orientation = default_goal["pose"]["pose"]["orientation"]
    assert (orientation["z"], orientation["w"]) == (0.0, 1.0)


def test_readme_documents_result_only_mode_and_safety_caveat() -> None:
    """The README (required packaging, CLAUDE.md §6.4) states the exact
    safety implication the manifest's own comments describe: Nav2 bypasses
    the OpenRAL safety supervisor for velocity commands."""
    readme = (_SKILL_DIR / "README.md").read_text(encoding="utf-8")
    assert "Result-only mode" in readme
    assert "the OpenRAL safety supervisor does NOT" in readme
    assert "navigate_to_pose" in readme
