"""Unit tests for WorldStateAggregator.

Tests cover:
- Fresh snapshot before any updates (zeroed joint state, all stale diagnostics)
- Joint state update → snapshot reflects data and marks ``"ok"``
- Staleness detection for joint state and sensors
- Image topic tracking
- EE pose tracking
- Base pose + twist
- Battery percentage
- Forced error latching and clearing
- Thread safety: concurrent updates + snapshots don't corrupt state
- Snapshot rate: 30 Hz nominal (clock injection)
"""

from __future__ import annotations

import threading
import time

import pytest
from openral_core import (
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_core.schemas import (
    EndEffectorSpec,
    JointState,
    Pose6D,
    SensorBundle,
    SensorModality,
    SensorSpec,
    WorldState,
)
from openral_world_state import DEFAULT_RATE_HZ, DEFAULT_STALENESS_S, WorldStateAggregator

# ── Helpers ───────────────────────────────────────────────────────────────────

_JOINT_NAMES = ["shoulder", "elbow", "wrist"]


def _make_description(
    n_joints: int = 3,
    sensor_names: list[str] | None = None,
    ee_names: list[str] | None = None,
) -> RobotDescription:
    """Minimal RobotDescription with optional sensor bundles and EEs."""
    joints = [
        JointSpec(
            name=_JOINT_NAMES[i] if i < len(_JOINT_NAMES) else f"j{i}",
            joint_type=JointType.REVOLUTE,
            parent_link="base" if i == 0 else f"link_{i - 1}",
            child_link=f"link_{i}",
        )
        for i in range(n_joints)
    ]
    bundles: list[SensorBundle] = []
    if sensor_names:
        bundles = [
            SensorBundle(
                bundle_name=f"{n}_bundle",
                sensors=[
                    SensorSpec(
                        name=n,
                        modality=SensorModality.RGB,
                        frame_id=f"{n}_frame",
                        rate_hz=30.0,
                        ros2_topic=f"/{n}/image_raw",
                        ros2_msg_type="sensor_msgs/Image",
                    )
                ],
            )
            for n in sensor_names
        ]
    ee_list: list[EndEffectorSpec] = []
    if ee_names:
        ee_list = [EndEffectorSpec(name=n, kind="parallel_gripper") for n in ee_names]

    return RobotDescription(
        name="test_robot",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=joints,
        capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        safety=SafetyEnvelope(),
        sensor_bundles=bundles,
        end_effectors=ee_list,
    )


def _js(positions: list[float], stamp_ns: int | None = None) -> JointState:
    n = len(positions)
    names = _JOINT_NAMES[:n] if n <= len(_JOINT_NAMES) else [f"j{i}" for i in range(n)]
    return JointState(
        name=names,
        position=positions,
        velocity=[0.0] * n,
        effort=[0.0] * n,
        stamp_ns=stamp_ns if stamp_ns is not None else time.time_ns(),
    )


def _pose(x: float = 0.0) -> Pose6D:
    return Pose6D(xyz=(x, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="world")


def _make_agg(
    n_joints: int = 3,
    sensor_names: list[str] | None = None,
    ee_names: list[str] | None = None,
    staleness_limit_s: float = DEFAULT_STALENESS_S,
    clock_fn: None = None,
) -> WorldStateAggregator:
    return WorldStateAggregator(
        _make_description(n_joints, sensor_names, ee_names),
        staleness_limit_s=staleness_limit_s,
        clock_fn=clock_fn,
    )


# ── Module-level constants ────────────────────────────────────────────────────


def test_default_rate_hz() -> None:
    assert DEFAULT_RATE_HZ == 30.0


def test_default_staleness_s() -> None:
    # 0.5 s, not 0.1 s: a 0.1 s window equals the 10 Hz camera period and made
    # the per-sensor diagnostics flap OK↔STALE on every snapshot.
    assert pytest.approx(0.5) == DEFAULT_STALENESS_S


# ── Snapshot before any updates ───────────────────────────────────────────────


class TestSnapshotBeforeUpdates:
    def test_returns_world_state(self) -> None:
        agg = _make_agg()
        ws = agg.snapshot()
        assert isinstance(ws, WorldState)

    def test_joint_state_zeroed(self) -> None:
        agg = _make_agg(n_joints=3)
        ws = agg.snapshot()
        assert ws.joint_state.position == [0.0, 0.0, 0.0]

    def test_joint_state_names_from_description(self) -> None:
        agg = _make_agg(n_joints=2)
        ws = agg.snapshot()
        assert ws.joint_state.name == ["shoulder", "elbow"]

    def test_joint_state_stale_before_update(self) -> None:
        agg = _make_agg()
        ws = agg.snapshot()
        assert ws.diagnostics["joint_state"] == "stale"

    def test_stamp_ns_positive(self) -> None:
        agg = _make_agg()
        ws = agg.snapshot()
        assert ws.stamp_ns > 0

    def test_no_sensor_data_initially(self) -> None:
        agg = _make_agg(sensor_names=["cam"])
        ws = agg.snapshot()
        assert ws.diagnostics["cam"] == "stale"
        assert "cam" not in ws.images

    def test_no_ee_pose_initially(self) -> None:
        # EE poses are lazily registered: a declared end-effector that has never
        # received a pose is absent from diagnostics entirely (not reported as
        # "stale"). Nothing currently feeds EE poses on the sim deploy path, so
        # pre-populating them just produced a permanently-STALE pill (noise).
        agg = _make_agg(ee_names=["gripper"])
        ws = agg.snapshot()
        assert "gripper" not in ws.diagnostics
        assert "gripper" not in ws.ee_poses

    def test_ee_pose_appears_in_diagnostics_after_update(self) -> None:
        # Once an EE pose is observed it joins the diagnostics ledger and is
        # classified ok/stale by age like any other component.
        agg = _make_agg(ee_names=["gripper"])
        agg.update_ee_pose("gripper", _pose(x=0.1))
        ws = agg.snapshot()
        assert ws.diagnostics["gripper"] == "ok"
        assert "gripper" in ws.ee_poses

    def test_battery_none_initially(self) -> None:
        agg = _make_agg()
        ws = agg.snapshot()
        assert ws.battery_pct is None

    def test_base_pose_none_initially(self) -> None:
        agg = _make_agg()
        ws = agg.snapshot()
        assert ws.base_pose is None


# ── Joint state updates ───────────────────────────────────────────────────────


class TestJointStateUpdates:
    def test_update_reflects_in_snapshot(self) -> None:
        agg = _make_agg(n_joints=3)
        agg.update_joint_state(_js([1.0, 2.0, 3.0]))
        ws = agg.snapshot()
        assert ws.joint_state.position == pytest.approx([1.0, 2.0, 3.0])

    def test_fresh_joint_state_is_ok(self) -> None:
        agg = _make_agg()
        agg.update_joint_state(_js([0.0, 0.0, 0.0]))
        ws = agg.snapshot()
        assert ws.diagnostics["joint_state"] == "ok"

    def test_sequential_updates_keep_latest(self) -> None:
        agg = _make_agg(n_joints=2)
        agg.update_joint_state(_js([1.0, 0.0]))
        agg.update_joint_state(_js([2.0, 0.0]))
        ws = agg.snapshot()
        assert ws.joint_state.position[0] == pytest.approx(2.0)

    def test_stale_joint_state_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Advance the clock past staleness_limit_s and verify 'stale'."""
        now_ns = time.time_ns()
        clock_calls = [now_ns]

        def fake_clock() -> int:
            return clock_calls[0]

        agg = WorldStateAggregator(
            _make_description(),
            staleness_limit_s=0.1,
            clock_fn=fake_clock,
        )
        agg.update_joint_state(_js([0.0, 0.0, 0.0]))
        # Advance clock by 200 ms
        clock_calls[0] = now_ns + 200_000_000
        ws = agg.snapshot()
        assert ws.diagnostics["joint_state"] == "stale"

    def test_fresh_then_stale_then_fresh(self) -> None:
        """Update → ok, wait → stale, update again → ok."""
        now_ns = [time.time_ns()]

        def fake_clock() -> int:
            return now_ns[0]

        agg = WorldStateAggregator(
            _make_description(),
            staleness_limit_s=0.1,
            clock_fn=fake_clock,
        )
        agg.update_joint_state(_js([0.0, 0.0, 0.0]))
        assert agg.snapshot().diagnostics["joint_state"] == "ok"

        now_ns[0] += 200_000_000  # +200 ms
        assert agg.snapshot().diagnostics["joint_state"] == "stale"

        agg.update_joint_state(_js([1.0, 0.0, 0.0]))
        assert agg.snapshot().diagnostics["joint_state"] == "ok"


# ── Image / sensor updates ────────────────────────────────────────────────────


class TestImageUpdates:
    def test_image_topic_stored(self) -> None:
        agg = _make_agg(sensor_names=["head"])
        agg.update_image("head", "/head/image_raw", time.time_ns())
        ws = agg.snapshot()
        assert ws.images["head"] == "/head/image_raw"

    def test_fresh_sensor_is_ok(self) -> None:
        agg = _make_agg(sensor_names=["head"])
        agg.update_image("head", "/head/image_raw", time.time_ns())
        ws = agg.snapshot()
        assert ws.diagnostics["head"] == "ok"

    def test_unknown_sensor_not_tracked(self) -> None:
        """Sensors not in description are stored but not in sensor_names."""
        agg = _make_agg(sensor_names=["cam_a"])
        agg.update_image("cam_a", "/cam_a/image_raw", time.time_ns())
        agg.update_image("cam_b", "/cam_b/image_raw", time.time_ns())
        ws = agg.snapshot()
        # cam_a is tracked (in description)
        assert "cam_a" in ws.images
        # cam_b received but not in description's sensor_names → no diag entry
        assert "cam_b" not in ws.diagnostics

    def test_stale_sensor_after_timeout(self) -> None:
        now_ns = [time.time_ns()]

        def fake_clock() -> int:
            return now_ns[0]

        desc = _make_description(sensor_names=["wrist"])
        agg = WorldStateAggregator(desc, staleness_limit_s=0.1, clock_fn=fake_clock)
        agg.update_image("wrist", "/wrist/image_raw", now_ns[0])
        assert agg.snapshot().diagnostics["wrist"] == "ok"

        now_ns[0] += 200_000_000
        assert agg.snapshot().diagnostics["wrist"] == "stale"

    def test_multiple_sensors_tracked_independently(self) -> None:
        now_ns = [time.time_ns()]

        def fake_clock() -> int:
            return now_ns[0]

        desc = _make_description(sensor_names=["cam_a", "cam_b"])
        agg = WorldStateAggregator(desc, staleness_limit_s=0.1, clock_fn=fake_clock)
        agg.update_image("cam_a", "/a", now_ns[0])
        agg.update_image("cam_b", "/b", now_ns[0])

        now_ns[0] += 200_000_000
        # Update only cam_a
        agg.update_image("cam_a", "/a", now_ns[0])

        ws = agg.snapshot()
        assert ws.diagnostics["cam_a"] == "ok"
        assert ws.diagnostics["cam_b"] == "stale"


# ── EE pose updates ───────────────────────────────────────────────────────────


class TestEEPoseUpdates:
    def test_ee_pose_stored(self) -> None:
        agg = _make_agg(ee_names=["gripper"])
        pose = _pose(x=0.5)
        agg.update_ee_pose("gripper", pose)
        ws = agg.snapshot()
        assert ws.ee_poses["gripper"].xyz == pytest.approx((0.5, 0.0, 0.0))

    def test_fresh_ee_is_ok(self) -> None:
        agg = _make_agg(ee_names=["gripper"])
        agg.update_ee_pose("gripper", _pose())
        ws = agg.snapshot()
        assert ws.diagnostics["gripper"] == "ok"

    def test_multiple_ees_tracked(self) -> None:
        agg = _make_agg(ee_names=["left", "right"])
        agg.update_ee_pose("left", _pose(0.1))
        agg.update_ee_pose("right", _pose(0.2))
        ws = agg.snapshot()
        assert ws.diagnostics["left"] == "ok"
        assert ws.diagnostics["right"] == "ok"
        assert ws.ee_poses["left"].xyz[0] == pytest.approx(0.1)
        assert ws.ee_poses["right"].xyz[0] == pytest.approx(0.2)


# ── Base pose + battery ───────────────────────────────────────────────────────


class TestBasePoseAndBattery:
    def test_base_pose_stored(self) -> None:
        agg = _make_agg()
        agg.update_base_pose(_pose(1.0))
        ws = agg.snapshot()
        assert ws.base_pose is not None
        assert ws.base_pose.xyz[0] == pytest.approx(1.0)

    def test_base_twist_stored(self) -> None:
        agg = _make_agg()
        twist = (1.0, 0.0, 0.0, 0.0, 0.0, 0.5)
        agg.update_base_pose(_pose(), twist=twist)
        ws = agg.snapshot()
        assert ws.base_twist == pytest.approx(twist)

    def test_battery_stored(self) -> None:
        agg = _make_agg()
        agg.update_battery(72.5)
        ws = agg.snapshot()
        assert ws.battery_pct == pytest.approx(72.5)

    def test_battery_overwritten_by_latest(self) -> None:
        agg = _make_agg()
        agg.update_battery(90.0)
        agg.update_battery(80.0)
        ws = agg.snapshot()
        assert ws.battery_pct == pytest.approx(80.0)


# ── Forced diagnostics ────────────────────────────────────────────────────────


class TestForcedDiagnostics:
    def test_set_error_overrides_ok(self) -> None:
        agg = _make_agg()
        agg.update_joint_state(_js([0.0, 0.0, 0.0]))
        agg.set_error("joint_state")
        ws = agg.snapshot()
        assert ws.diagnostics["joint_state"] == "error"

    def test_set_warn_overrides_stale(self) -> None:
        agg = _make_agg(sensor_names=["cam"])
        agg.set_error("cam", "warn")
        ws = agg.snapshot()
        assert ws.diagnostics["cam"] == "warn"

    def test_clear_error_restores_staleness_check(self) -> None:
        agg = _make_agg()
        agg.set_error("joint_state")
        agg.clear_error("joint_state")
        agg.update_joint_state(_js([0.0, 0.0, 0.0]))
        ws = agg.snapshot()
        assert ws.diagnostics["joint_state"] == "ok"

    def test_clear_nonexistent_is_noop(self) -> None:
        agg = _make_agg()
        agg.clear_error("does_not_exist")  # must not raise

    def test_forced_error_persists_across_snapshots(self) -> None:
        agg = _make_agg()
        agg.set_error("joint_state")
        for _ in range(3):
            ws = agg.snapshot()
            assert ws.diagnostics["joint_state"] == "error"


# ── Thread safety ─────────────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_updates_and_snapshots(self) -> None:
        """Many threads updating state while another thread polls snapshots."""
        agg = _make_agg(n_joints=2, sensor_names=["cam"])
        errors: list[Exception] = []
        snapshots: list[WorldState] = []
        stop = threading.Event()

        def updater() -> None:
            for i in range(50):
                try:
                    agg.update_joint_state(_js([float(i), float(i)]))
                    agg.update_image("cam", "/cam/image_raw", time.time_ns())
                    agg.update_battery(float(i % 100))
                except Exception as exc:
                    errors.append(exc)

        def poller() -> None:
            while not stop.is_set():
                try:
                    ws = agg.snapshot()
                    snapshots.append(ws)
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=updater) for _ in range(4)]
        poll_thread = threading.Thread(target=poller)
        poll_thread.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        poll_thread.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(snapshots) > 0

    def test_snapshot_always_returns_valid_world_state(self) -> None:
        """snapshot() must never return a malformed WorldState."""
        agg = _make_agg(n_joints=1)
        for _ in range(20):
            agg.update_joint_state(_js([0.5]))
            ws = agg.snapshot()
            assert isinstance(ws, WorldState)
            assert ws.joint_state.name == ["shoulder"]


# ── Clock injection (30 Hz simulation) ───────────────────────────────────────


class TestClockInjection:
    def test_custom_clock_used_for_stamps(self) -> None:
        fixed_ns = 1_000_000_000

        agg = WorldStateAggregator(
            _make_description(),
            clock_fn=lambda: fixed_ns,
        )
        ws = agg.snapshot()
        assert ws.stamp_ns == fixed_ns

    def test_30hz_snapshots_advance_stamp(self) -> None:
        """Simulate 30 snapshots 33 ms apart; stamps must be strictly increasing."""
        step_ns = int(1e9 / 30)  # ~33.3 ms
        tick = [0]

        def clock() -> int:
            return tick[0]

        agg = WorldStateAggregator(_make_description(), clock_fn=clock)
        agg.update_joint_state(
            JointState(name=["shoulder", "elbow", "wrist"], position=[0.0, 0.0, 0.0], stamp_ns=0)
        )
        stamps = []
        for _ in range(30):
            tick[0] += step_ns
            ws = agg.snapshot()
            stamps.append(ws.stamp_ns)

        assert stamps == sorted(stamps)
        assert stamps[-1] - stamps[0] == pytest.approx(29 * step_ns)
