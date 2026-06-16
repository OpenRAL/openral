"""ADR-0024 — :class:`ROSActionRskill` adapter behaviour.

The hot-path tests use the adapter's existing fallback-setter and
dotted-accessor implementations against synthetic message-shaped
objects so we don't depend on a sourced ROS 2 workspace just to verify
the trajectory-replay + termination logic. The end-to-end ROS path
(real ``rclpy.action.ActionClient`` against a real ``ActionServer``)
is gated on rclpy + example_interfaces being importable and skips
cleanly when they aren't, per CLAUDE.md §1.11.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from openral_core import (
    ActuatorRequirement,
    ControlMode,
    ControlModeSemantics,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    ROSConfigError,
    RosIntegration,
    ROSRskillGoalSatisfied,
    RSkillAction,
    RSkillLatencyBudget,
    RSkillLicensePosture,
    RSkillManifest,
    RSkillState,
    SafetyEnvelope,
)
from openral_rskill.ros_action_rskill import (
    ROSActionRskill,
    _merge_nested,
    _resolve_dotted,
    _set_message_fields,
    build_joint_permutation_from_names,
)

# ── build_joint_permutation_from_names ───────────────────────────────────────


def test_joint_permutation_identity() -> None:
    perm, unmoved = build_joint_permutation_from_names(
        source_names=["a", "b", "c"],
        target_names=["a", "b", "c"],
    )
    assert perm == [0, 1, 2]
    assert unmoved == []


def test_joint_permutation_reorders() -> None:
    perm, unmoved = build_joint_permutation_from_names(
        source_names=["c", "a", "b"],
        target_names=["a", "b", "c"],
    )
    # target[0] == "a" == source[1] → perm[0] = 1
    assert perm == [1, 2, 0]
    assert unmoved == []


def test_joint_permutation_subset_marks_unmoved_slots() -> None:
    """ADR-0024 — MoveIt's panda_arm group (7 joints) is a STRICT subset of
    franka_panda's RobotDescription (7 arm + 1 gripper). Slots not in the
    source come back as unmoved indices, with perm[i] == -1.
    """
    perm, unmoved = build_joint_permutation_from_names(
        source_names=["panda_joint1", "panda_joint2", "panda_joint3"],
        target_names=["panda_joint1", "panda_joint2", "panda_joint3", "panda_gripper"],
    )
    assert perm == [0, 1, 2, -1]
    assert unmoved == [3]


def test_joint_permutation_rejects_source_extra() -> None:
    """The wrapped planner commanding a joint the host doesn't know about
    is a real misconfiguration, not silently swappable bytes."""
    with pytest.raises(ROSConfigError, match="extra_in_source"):
        build_joint_permutation_from_names(
            source_names=["a", "b", "c"],
            target_names=["a", "b", "d"],
        )


# ── _resolve_dotted ──────────────────────────────────────────────────────────


def test_resolve_dotted_walks_attributes() -> None:
    @dataclass
    class _Inner:
        joints: list[str] = field(default_factory=lambda: ["x", "y"])

    @dataclass
    class _Outer:
        inner: _Inner = field(default_factory=_Inner)

    obj = _Outer()
    assert _resolve_dotted(obj, "inner.joints") == ["x", "y"]


# ── ROSActionRskill — trajectory mode, synthetic IDL ────────────────────────


@dataclass
class _FakeJointTrajectoryPoint:
    positions: list[float]


@dataclass
class _FakeJointTrajectory:
    joint_names: list[str]
    points: list[_FakeJointTrajectoryPoint]


@dataclass
class _FakeMoveGroupResult:
    planned_trajectory: Any  # carries .joint_trajectory


@dataclass
class _FakeMoveGroupResultWrapper:
    result: _FakeMoveGroupResult


@dataclass
class _FakePlannedTrajectory:
    joint_trajectory: _FakeJointTrajectory


def _two_dof_robot() -> RobotDescription:
    """Two-joint canonical embodiment for joint-permutation tests."""
    return RobotDescription(
        name="custom_2dof",
        embodiment_kind="manipulator",
        joints=[
            JointSpec(
                name="j_a",
                joint_type=JointType.REVOLUTE,
                parent_link="base",
                child_link="link_a",
            ),
            JointSpec(
                name="j_b",
                joint_type=JointType.REVOLUTE,
                parent_link="link_a",
                child_link="link_b",
            ),
        ],
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION],
            embodiment_tags=["custom"],
        ),
        safety=SafetyEnvelope(),
    )


def _trajectory_manifest() -> RSkillManifest:
    return RSkillManifest(
        name="openral/rskill-test-traj",
        version="0.1.0",
        license=RSkillLicensePosture.APACHE_2_0,
        role="s1",
        kind="ros_action",
        embodiment_tags=["franka_panda"],
        actuators_required=[
            ActuatorRequirement(
                kind=ControlMode.JOINT_POSITION,
                control_mode_semantics=ControlModeSemantics(mode="absolute"),
            ),
        ],
        chunk_size=1,
        latency_budget=RSkillLatencyBudget(per_chunk_ms=100.0),
        description="Test wrapped trajectory.",
        actions=[RSkillAction.REACH],
        ros_integration=RosIntegration(
            package="moveit_msgs",
            interface_type="MoveGroup",
            interface_name="/move_action",
            result_trajectory_field="planned_trajectory.joint_trajectory",
            default_goal_json=json.dumps({"request": {"group_name": "panda_arm"}}),
        ),
    )


def _result_only_manifest() -> RSkillManifest:
    return RSkillManifest(
        name="openral/rskill-test-result-only",
        version="0.1.0",
        license=RSkillLicensePosture.APACHE_2_0,
        role="s1",
        kind="ros_action",
        embodiment_tags=["panda_mobile"],
        actuators_required=[
            ActuatorRequirement(
                kind=ControlMode.BODY_TWIST,
                control_mode_semantics=ControlModeSemantics(mode="absolute"),
            ),
        ],
        chunk_size=1,
        latency_budget=RSkillLatencyBudget(per_chunk_ms=100.0),
        description="Test wrapped result-only.",
        actions=[RSkillAction.NAVIGATE],
        ros_integration=RosIntegration(
            package="nav2_msgs",
            interface_type="NavigateToPose",
            interface_name="/navigate_to_pose",
            result_trajectory_field=None,
            default_goal_json='{"pose": {"header": {"frame_id": "map"}}}',
        ),
    )


def _make_skill(manifest: RSkillManifest, description: RobotDescription | None) -> ROSActionRskill:
    """Construct the adapter without driving its lifecycle."""
    return ROSActionRskill(
        manifest=manifest,
        ros_node=None,  # unused on the synthetic paths below
        robot_description=description,
        prompt="test",
        prompt_metadata_json="",
    )


def test_trajectory_mode_replays_waypoints_then_signals_completion() -> None:
    """Three-point trajectory → three Actions, then ``ROSRskillGoalSatisfied``."""
    skill = _make_skill(_trajectory_manifest(), _two_dof_robot())
    # Inject the wrapped trajectory we'd otherwise obtain from
    # ActionClient.get_result_async() so the test stays hermetic.
    skill._result_consumed = True
    skill._waypoints = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
    # Drive into ACTIVE state so step() doesn't reject the call.
    skill._info = skill._info.model_copy(update={"state": RSkillState.ACTIVE})

    a0 = skill.step(world_state=None)  # type: ignore[arg-type]
    assert a0.control_mode is ControlMode.JOINT_POSITION
    assert a0.horizon == 1
    assert a0.joint_targets == [[0.1, 0.2]]

    a1 = skill.step(world_state=None)  # type: ignore[arg-type]
    assert a1.joint_targets == [[0.3, 0.4]]

    a2 = skill.step(world_state=None)  # type: ignore[arg-type]
    assert a2.joint_targets == [[0.5, 0.6]]

    with pytest.raises(ROSRskillGoalSatisfied, match="emitted all 3 waypoints"):
        skill.step(world_state=None)  # type: ignore[arg-type]


def test_dispatch_and_cache_reorders_joints_into_robot_order() -> None:
    """Wrapped server returns joints in [j_b, j_a]; adapter reorders to [j_a, j_b]."""
    skill = _make_skill(_trajectory_manifest(), _two_dof_robot())
    # Patch the ActionClient interaction with a synthetic result.
    fake_result = _FakeMoveGroupResult(
        planned_trajectory=_FakePlannedTrajectory(
            joint_trajectory=_FakeJointTrajectory(
                joint_names=["j_b", "j_a"],  # reversed vs robot order
                points=[
                    _FakeJointTrajectoryPoint(positions=[0.8, 0.1]),  # j_b=0.8, j_a=0.1
                    _FakeJointTrajectoryPoint(positions=[0.9, 0.2]),
                ],
            )
        )
    )
    skill._interface_kind = "action"

    def _stub_send() -> Any:
        return fake_result

    skill._send_action_goal_and_await_result = _stub_send  # type: ignore[method-assign]
    skill._dispatch_and_cache_result()

    # Robot order is [j_a, j_b]; positions must be [j_a_val, j_b_val] post-reorder.
    assert skill._waypoints == [[0.1, 0.8], [0.2, 0.9]]


def test_result_only_mode_signals_completion_on_first_step() -> None:
    """Nav2 shape: no trajectory → first step() raises ``ROSRskillGoalSatisfied``."""
    skill = _make_skill(_result_only_manifest(), description=None)
    # Pretend the dispatch finished cleanly without producing a trajectory.
    called = {"n": 0}

    def _stub_dispatch() -> None:
        called["n"] += 1

    skill._dispatch_and_cache_result = _stub_dispatch  # type: ignore[method-assign]
    skill._info = skill._info.model_copy(update={"state": RSkillState.ACTIVE})

    with pytest.raises(ROSRskillGoalSatisfied, match="result-only action completed"):
        skill.step(world_state=None)  # type: ignore[arg-type]
    assert called["n"] == 1


@dataclass
class _FakeResultWrapper:
    """Minimal stand-in for ``GetResult_Response``.

    The real rclpy wrapper carries both ``status`` (action_msgs/GoalStatus
    int) and ``result`` (the per-action IDL result). The adapter's
    ``_send_action_goal_and_await_result`` checks status and either
    returns ``result`` or raises ``ROSRuntimeError``.
    """

    status: int
    result: Any


@dataclass
class _FakeGoalHandle:
    accepted: bool = True

    def get_result_async(self) -> Any:
        return _DoneFuture(self._wrapper)

    def __init__(self, wrapper: _FakeResultWrapper) -> None:
        self._wrapper = wrapper
        self.accepted = True


@dataclass
class _DoneFuture:
    _value: Any

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self._value


@dataclass
class _FakeActionClient:
    """Minimal stand-in for ``rclpy.action.ActionClient``."""

    wrapper: _FakeResultWrapper

    def send_goal_async(self, goal: Any) -> Any:
        del goal
        return _DoneFuture(_FakeGoalHandle(self.wrapper))


def _make_skill_with_fake_action(
    manifest: RSkillManifest,
    wrapper: _FakeResultWrapper,
) -> ROSActionRskill:
    """Build the adapter with a faked ActionClient that returns ``wrapper``.

    Bypasses the live rclpy wiring so we can drive the status-check
    code path against synthetic SUCCEEDED / ABORTED / CANCELED inputs.
    """
    skill = _make_skill(manifest, description=None)
    skill._interface_kind = "action"  # type: ignore[attr-defined]
    skill._interface_type = type(
        "_FakeIDL", (), {"Goal": dataclass(type("Goal", (), {})), "Result": dict}
    )
    skill._client = _FakeActionClient(wrapper)
    skill._goal_dict = {}
    skill._result_deadline_s = 1.0
    return skill


def test_result_only_mode_raises_on_action_aborted() -> None:
    """ADR-0024 — Nav2 ABORTED must surface as a runtime failure.

    Before this check, ``_send_action_goal_and_await_result`` returned
    ``wrapper.result`` regardless of ``wrapper.status``. A Nav2 goal
    that aborted (planner failure, costmap rejection, controller
    timeout) was indistinguishable from success: the wrapped action
    completed, just with ``status=STATUS_ABORTED=6``, and the
    reasoner logged ``execute_rskill succeeded`` on a failed nav.
    """
    from openral_core.exceptions import ROSRuntimeError

    # Mirror the Nav2 result shape (``error_code``, ``error_msg``).
    @dataclass
    class _NavResult:
        error_code: int = 0
        error_msg: str = "Goal Coordinates of(0.0, 0.0) was outside bounds"

    wrapper = _FakeResultWrapper(status=6, result=_NavResult())  # STATUS_ABORTED
    skill = _make_skill_with_fake_action(_result_only_manifest(), wrapper)

    with pytest.raises(ROSRuntimeError, match="STATUS_ABORTED"):
        skill._send_action_goal_and_await_result()


def test_result_only_mode_raises_on_action_canceled() -> None:
    """STATUS_CANCELED (5) also fails — the operator cancelled the goal."""
    from openral_core.exceptions import ROSRuntimeError

    @dataclass
    class _NavResult:
        error_code: int = 0
        error_msg: str = ""

    wrapper = _FakeResultWrapper(status=5, result=_NavResult())  # STATUS_CANCELED
    skill = _make_skill_with_fake_action(_result_only_manifest(), wrapper)

    with pytest.raises(ROSRuntimeError, match="STATUS_CANCELED"):
        skill._send_action_goal_and_await_result()


def test_result_only_mode_returns_result_on_action_succeeded() -> None:
    """STATUS_SUCCEEDED (4) → return the result; happy path stays unchanged."""

    @dataclass
    class _NavResult:
        error_code: int = 0
        error_msg: str = ""

    wrapper = _FakeResultWrapper(status=4, result=_NavResult())  # STATUS_SUCCEEDED
    skill = _make_skill_with_fake_action(_result_only_manifest(), wrapper)

    result = skill._send_action_goal_and_await_result()
    assert isinstance(result, _NavResult)


# ── ADR-0026 — _merge_nested + goal_params_json merge ───────────────────────


def test_merge_nested_leaves_replace() -> None:
    """Scalar leaves: overrides win."""
    base = {"a": 1, "b": 2, "c": 3}
    overrides = {"b": 99}
    assert _merge_nested(base, overrides) == {"a": 1, "b": 99, "c": 3}
    # Original inputs untouched.
    assert base == {"a": 1, "b": 2, "c": 3}


def test_merge_nested_recurses_dicts() -> None:
    """Dict-vs-dict at the same key recurses; sibling keys preserved."""
    base = {"pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "frame_id": "map"}}
    overrides = {"pose": {"position": {"x": 1.5}}}
    out = _merge_nested(base, overrides)
    assert out == {
        "pose": {
            "position": {"x": 1.5, "y": 0.0, "z": 0.0},  # y/z from base
            "frame_id": "map",  # frame_id from base
        }
    }


def test_merge_nested_arrays_replace_verbatim() -> None:
    """Arrays don't element-wise merge — overrides win whole."""
    base = {"joint_targets": [0.1, 0.2, 0.3], "speed": 1.0}
    overrides = {"joint_targets": [0.5, 0.6]}
    assert _merge_nested(base, overrides) == {
        "joint_targets": [0.5, 0.6],  # length-2 wins; no padding from base
        "speed": 1.0,
    }


def test_merge_nested_override_only_keys_get_added() -> None:
    """Keys present only in overrides land in the output."""
    base = {"a": 1}
    overrides = {"b": 2, "nested": {"x": 10}}
    assert _merge_nested(base, overrides) == {"a": 1, "b": 2, "nested": {"x": 10}}


def test_merge_nested_dict_replaces_scalar_in_base() -> None:
    """If base has a scalar but override has a dict, override wins."""
    base = {"k": 5}
    overrides = {"k": {"nested": True}}
    assert _merge_nested(base, overrides) == {"k": {"nested": True}}


def test_rosaction_rskill_constructor_accepts_goal_params_json() -> None:
    """ADR-0026 — adapter accepts the new kwarg + stores it on self."""
    skill = ROSActionRskill(
        manifest=_result_only_manifest(),
        ros_node=None,
        robot_description=None,
        prompt="navigate",
        prompt_metadata_json="",
        goal_params_json='{"pose": {"position": {"x": 1.5}}}',
    )
    assert skill._goal_params_json == '{"pose": {"position": {"x": 1.5}}}'


def test_rosaction_rskill_default_goal_params_json_is_empty() -> None:
    """Backward-compat: omitting the kwarg keeps today's empty default."""
    skill = ROSActionRskill(
        manifest=_result_only_manifest(),
        ros_node=None,
        robot_description=None,
        prompt="navigate",
        prompt_metadata_json="",
    )
    assert skill._goal_params_json == ""


def test_constructor_rejects_manifest_without_ros_integration() -> None:
    """Defence-in-depth: even if a synthetic manifest sneaks past the
    schema-side validator, the adapter rejects it."""
    bad_manifest = _trajectory_manifest().model_copy(update={"ros_integration": None})
    with pytest.raises(ROSConfigError, match="ros_integration"):
        _make_skill(bad_manifest, _two_dof_robot())


# ── #nav goal-build: nested IDL application + flattened-params rejection ───────


def test_set_message_fields_applies_correct_nesting() -> None:
    # The manifest's nested default_goal_json shape must apply cleanly to a real
    # geometry_msgs/PoseStamped (the rosidl_runtime_py production path).
    geometry_msgs = pytest.importorskip("geometry_msgs.msg")
    ps = geometry_msgs.PoseStamped()
    _set_message_fields(
        ps, {"header": {"frame_id": "map"}, "pose": {"position": {"x": 3.5, "y": 2.1}}}
    )
    assert ps.header.frame_id == "map"
    assert ps.pose.position.x == 3.5
    assert ps.pose.position.y == 2.1


def test_set_message_fields_rejects_flattened_pose() -> None:
    # The exact LLM mistake that broke nav2-navigate-to-pose: position/orientation
    # hoisted to the PoseStamped level (instead of under pose.pose). PoseStamped has
    # no such field, so application must raise — this is what the adapter wraps into
    # an actionable ROSConfigError the reasoner can replan on.
    geometry_msgs = pytest.importorskip("geometry_msgs.msg")
    ps = geometry_msgs.PoseStamped()
    flattened = {
        "header": {"frame_id": "map"},
        "pose": {"position": {"x": 0.0}},
        "position": {"x": 3.5},  # misplaced
        "orientation": {"w": 1.0},  # misplaced
    }
    with pytest.raises((AttributeError, TypeError, ValueError)):
        _set_message_fields(ps, flattened)


def test_merge_nested_flattened_override_surfaces_bad_keys() -> None:
    # A flattened override deep-merges into extra top-level keys on the PoseStamped
    # (header, pose, position, orientation) — the structural smell the goal-build
    # then rejects.
    default = {"pose": {"header": {"frame_id": "base_link"}, "pose": {"position": {"x": 0.0}}}}
    override = {"pose": {"position": {"x": 3.5}, "orientation": {"w": 1.0}}}
    merged = _merge_nested(default, override)
    assert sorted(merged["pose"]) == ["header", "orientation", "pose", "position"]
