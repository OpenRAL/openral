"""``ROSActionRskill`` adapter behaviour.

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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from openral_core import (
    Action,
    ActuatorRequirement,
    ControlMode,
    ControlModeSemantics,
    JointSpec,
    JointState,
    JointType,
    RobotCapabilities,
    RobotDescription,
    ROSConfigError,
    ROSDeadlineMissed,
    RosIntegration,
    ROSRskillGoalSatisfied,
    RSkillAction,
    RSkillLatencyBudget,
    RSkillLicensePosture,
    RSkillManifest,
    RSkillState,
    SafetyEnvelope,
    WorldState,
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
    """MoveIt's panda_arm group (7 joints) is a STRICT subset of
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


def _one(step: Action | list[Action]) -> Action:
    assert isinstance(step, Action)
    return step


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
    """Untimed three-point trajectory → one waypoint per call, then success once reached."""
    skill = _make_skill(_trajectory_manifest(), _two_dof_robot())
    # Inject the wrapped trajectory we'd otherwise obtain from
    # ActionClient.get_result_async() so the test stays hermetic.
    skill._result_consumed = True
    skill._waypoints = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
    skill._planned_joints = [(0, "j_a"), (1, "j_b")]
    # Drive into ACTIVE state so step() doesn't reject the call.
    skill._info = skill._info.model_copy(update={"state": RSkillState.ACTIVE})
    at_final = WorldState(
        stamp_ns=0, joint_state=JointState(name=["j_a", "j_b"], position=[0.5, 0.6], stamp_ns=0)
    )

    a0 = _one(skill.step(at_final))
    assert a0.control_mode is ControlMode.JOINT_POSITION
    assert a0.horizon == 1
    assert a0.joint_targets == [[0.1, 0.2]]

    a1 = _one(skill.step(at_final))
    assert a1.joint_targets == [[0.3, 0.4]]

    a2 = _one(skill.step(at_final))
    assert a2.joint_targets == [[0.5, 0.6]]

    # No measured joint state: never "reached", the final waypoint is held.
    assert _one(skill.step(world_state=None)).joint_targets == [[0.5, 0.6]]  # type: ignore[arg-type]

    with pytest.raises(ROSRskillGoalSatisfied, match="reached the final of 3 waypoints"):
        skill.step(at_final)


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
    skill._interface_kind = "action"
    skill._interface_type = type(
        "_FakeIDL", (), {"Goal": dataclass(type("Goal", (), {})), "Result": dict}
    )
    skill._client = _FakeActionClient(wrapper)
    skill._goal_dict = {}
    skill._result_deadline_s = 1.0
    return skill


def test_result_only_mode_raises_on_action_aborted() -> None:
    """Nav2 ABORTED must surface as a runtime failure.

    Before this check, ``_send_action_goal_and_await_result`` returned
    ``wrapper.result`` regardless of ``wrapper.status``, so a Nav2 goal
    aborted at status=STATUS_ABORTED=6 read as success in the reasoner log.
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


# ── _merge_nested + goal_params_json merge ──────────────────────────────────


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
    """Adapter accepts the new kwarg + stores it on self."""
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


# ── Trajectory replay: paced by time_from_start, done only once reached ──────
#
# Regression for the MoveIt replay reporting success as soon as the last
# waypoint was *sent*: the runner ticks ``step()`` at its control rate, the
# adapter advanced one waypoint per call regardless of the plan's
# ``time_from_start`` or the measured joints, and raised
# ``ROSRskillGoalSatisfied`` one tick after the last waypoint went out — with the
# arm nowhere near it (DsslRobot/openral 1d5b739a7: tool 0.6 m from the goal).
# Real in-tree manifest + robot, real ``trajectory_msgs`` / ``builtin_interfaces``
# IDL for the planned trajectory; the MoveGroup result envelope arrives through
# the ``_FakeActionClient`` graph-boundary double above (``moveit_msgs`` is not
# required).

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PANDA_ARM = [f"panda_joint{i}" for i in range(1, 8)]
#: ``rskills/rskill-moveit-joints`` default joint goal (Franka home).
_PANDA_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
_TICK_S = 1.0 / 30.0  # the runner's default control rate


def _panda_world_state(robot: RobotDescription, positions: list[float]) -> WorldState:
    """A measured snapshot in ``robot.joints`` order (what the runner hands ``step``)."""
    return WorldState(
        stamp_ns=0,
        joint_state=JointState(
            name=[j.name for j in robot.joints], position=list(positions), stamp_ns=0
        ),
    )


def _moveit_replay_skill(times_s: list[float]) -> tuple[ROSActionRskill, RobotDescription]:
    """``rskill-moveit-joints`` on the Franka, the wrapped server answering a timed plan.

    The plan moves the 7 arm joints linearly from all-zero to ``_PANDA_HOME``,
    waypoint ``k`` at ``times_s[k]`` — the shape MoveIt's time parameterisation
    returns.
    """
    trajectory_msgs = pytest.importorskip("trajectory_msgs.msg")
    builtin_interfaces = pytest.importorskip("builtin_interfaces.msg")
    manifest = RSkillManifest.from_yaml(
        str(_REPO_ROOT / "rskills/rskill-moveit-joints/rskill.yaml")
    )
    robot = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/franka_panda/robot.yaml"))
    jt = trajectory_msgs.JointTrajectory(joint_names=_PANDA_ARM)
    for k, t in enumerate(times_s):
        frac = k / (len(times_s) - 1)
        point = trajectory_msgs.JointTrajectoryPoint(positions=[frac * g for g in _PANDA_HOME])
        point.time_from_start = builtin_interfaces.Duration(
            sec=int(t), nanosec=round((t - int(t)) * 1e9)
        )
        jt.points.append(point)
    result = _FakeMoveGroupResult(planned_trajectory=_FakePlannedTrajectory(joint_trajectory=jt))
    skill = ROSActionRskill(
        manifest=manifest,
        ros_node=None,  # no node → the replay is timed on time.monotonic()
        robot_description=robot,
        prompt="move to home",
        prompt_metadata_json="",
    )
    skill._interface_kind = "action"
    skill._interface_type = type("_FakeIDL", (), {"Goal": dataclass(type("Goal", (), {}))})
    skill._client = _FakeActionClient(_FakeResultWrapper(status=4, result=result))
    skill._goal_dict = {}
    skill._info = skill._info.model_copy(update={"state": RSkillState.ACTIVE})
    return skill, robot


def test_replay_never_reports_goal_satisfied_while_the_arm_has_not_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drives that never follow → a typed deadline failure, never ``GoalSatisfied``.

    On the pre-fix adapter the fourth ``step()`` of a three-waypoint plan raised
    ``ROSRskillGoalSatisfied("emitted all 3 waypoints")`` while every measured
    joint still sat at the start, up to 2.356 rad from the final waypoint.
    """
    from openral_rskill import ros_action_rskill

    monkeypatch.setattr(ros_action_rskill, "_REPLAY_SETTLE_TIMEOUT_S", 0.3, raising=False)
    skill, robot = _moveit_replay_skill([0.0, 0.2, 0.4])
    stuck = _panda_world_state(robot, [0.0] * len(robot.joints))  # the arm never moves
    t0 = time.monotonic()
    with pytest.raises(ROSDeadlineMissed, match="final waypoint"):
        for tick in range(300):
            try:
                skill.step(stuck)
            except ROSRskillGoalSatisfied as exc:
                pytest.fail(
                    f"goal satisfied on tick {tick} with every arm joint still at 0.0 "
                    f"(final waypoint up to 2.356 rad away): {exc}"
                )
            time.sleep(_TICK_S)
    # Not before the plan's 0.4 s plus the 0.3 s settle bound.
    assert time.monotonic() - t0 >= 0.7


def test_replay_paces_by_time_from_start_and_completes_once_reached() -> None:
    """An arm that tracks every command: waypoints go out on schedule, then success."""
    times_s = [0.0, 0.3, 0.6]
    skill, robot = _moveit_replay_skill(times_s)
    measured = _panda_world_state(robot, [0.0] * len(robot.joints))
    emitted: list[tuple[float, list[float]]] = []
    t0 = time.monotonic()
    with pytest.raises(ROSRskillGoalSatisfied, match="reached the final of 3 waypoints"):
        for _ in range(300):
            action = _one(skill.step(measured))
            assert action.joint_targets is not None
            emitted.append((time.monotonic() - t0, action.joint_targets[0]))
            measured = _panda_world_state(robot, action.joint_targets[0])  # drives follow
            time.sleep(_TICK_S)
    waypoints = skill._waypoints
    order = [waypoints.index(target) for _, target in emitted]
    assert order == sorted(order), "replay went backwards"
    assert sorted(set(order)) == [0, 1, 2], "every waypoint must reach the drives"
    for elapsed, target in emitted:
        assert elapsed >= times_s[waypoints.index(target)], "waypoint sent before it was due"
    # Held, not streamed: ~0.6 s of plan at 30 Hz is far more than 3 ticks.
    assert len(emitted) >= 10


# ── unmoved-joint padding is read by joint name ──────────────────────────────


def test_unmoved_joint_padding_is_read_by_name_not_index() -> None:
    """``/joint_states`` order is the publisher's, not ``RobotDescription.joints``.

    The planner moves only ``j_a``; ``j_b`` must be held at *its* measured value
    even when the measurement lists the joints in another order.
    """
    skill = _make_skill(_trajectory_manifest(), _two_dof_robot())
    skill._interface_kind = "action"
    reordered = WorldState(
        stamp_ns=0, joint_state=JointState(name=["j_b", "j_a"], position=[0.7, 0.1], stamp_ns=0)
    )
    skill._capture_unmoved_joint_padding(reordered)
    plan = _FakeMoveGroupResult(
        planned_trajectory=_FakePlannedTrajectory(
            joint_trajectory=_FakeJointTrajectory(
                joint_names=["j_a"], points=[_FakeJointTrajectoryPoint(positions=[0.3])]
            )
        )
    )
    skill._send_action_goal_and_await_result = lambda: plan  # type: ignore[method-assign]
    skill._dispatch_and_cache_result()
    assert skill._waypoints == [[0.3, 0.7]]


def test_unmeasured_unmoved_joint_is_refused_not_zero_filled() -> None:
    """A held joint with no measurement must not be commanded to a made-up 0.0."""
    from openral_core import ROSPerceptionStale

    skill = _make_skill(_trajectory_manifest(), _two_dof_robot())
    skill._interface_kind = "action"
    skill._capture_unmoved_joint_padding(
        WorldState(stamp_ns=0, joint_state=JointState(name=["j_a"], position=[0.1], stamp_ns=0))
    )
    plan = _FakeMoveGroupResult(
        planned_trajectory=_FakePlannedTrajectory(
            joint_trajectory=_FakeJointTrajectory(
                joint_names=["j_a"], points=[_FakeJointTrajectoryPoint(positions=[0.3])]
            )
        )
    )
    skill._send_action_goal_and_await_result = lambda: plan  # type: ignore[method-assign]
    with pytest.raises(ROSPerceptionStale, match="j_b"):
        skill._dispatch_and_cache_result()


# ── a MoveGroup replay goal stays plan_only ──────────────────────────────────


@pytest.mark.parametrize(
    "override",
    [
        {"planning_options": {"plan_only": False}},
        {"planning_options": {"plan_only": 0}},
        {"planning_options": None},
    ],
)
def test_goal_params_cannot_turn_off_plan_only(override: dict[str, Any]) -> None:
    """``goal_params_json`` must not let move_group execute a plan OpenRAL replays."""
    manifest = RSkillManifest.from_yaml(
        str(_REPO_ROOT / "rskills/rskill-moveit-joints/rskill.yaml")
    )
    skill = ROSActionRskill(
        manifest=manifest,
        ros_node=None,
        robot_description=None,
        prompt="move",
        prompt_metadata_json="",
        goal_params_json=json.dumps(override),
    )
    with pytest.raises(ROSConfigError, match="plan_only"):
        skill._configure_impl()


# ── a blocking wrapped wait sees the enclosing goal's cancel / deadline ──────


class _PendingFuture:
    def done(self) -> bool:
        return False

    def result(self) -> Any:
        return None


class _PendingGoalHandle:
    """An accepted goal whose result never arrives (Nav2 still driving)."""

    accepted = True

    def __init__(self) -> None:
        self.cancelled = False

    def get_result_async(self) -> Any:
        return _PendingFuture()

    def cancel_goal_async(self) -> Any:
        self.cancelled = True
        return _DoneFuture(None)


def test_result_wait_stops_and_cancels_when_the_enclosing_goal_ends() -> None:
    """The runner cannot interrupt a blocking ``step()``; the adapter must poll ``stop_check``.

    Without it a cancelled / timed-out ``ExecuteRskill`` left Nav2 driving until the
    adapter's own result deadline.
    """
    handle = _PendingGoalHandle()
    skill = _make_skill_with_fake_action(
        _result_only_manifest(), _FakeResultWrapper(status=4, result=None)
    )
    skill._client = type("_C", (), {"send_goal_async": lambda self, g: _DoneFuture(handle)})()
    skill._result_deadline_s = 3.0
    skill.stop_check = lambda: "cancel requested"
    t0 = time.monotonic()
    with pytest.raises(ROSDeadlineMissed, match="cancel requested"):
        skill._send_action_goal_and_await_result()
    assert time.monotonic() - t0 < 1.0
    assert handle.cancelled, "the wrapped goal must be cancelled, not left running"
