"""`OpenArmRealHAL` — the actuation path to the physical OpenArm v2.

Per CLAUDE.md §1.11 the only doubles here are at the transport boundary:
``publish_fn`` / ``state_fn`` are the injection points the adapter already
exposes for exactly this purpose, and every schema, manifest and joint name
is the real one.  The CAN preflight reads a real sysfs-shaped directory
tree with ``openral_core.can.SYSFS_NET`` redirected at it.

What these pin down:

- One 16-DoF ``Action`` must reach **four** controllers, sliced correctly.
  The OpenArm's bimanual bringup commands each arm and each gripper
  separately; a single-topic publish would move nothing.
- ``/joint_states`` must be read by **name**, not by index.  Four
  controllers publish into that topic independently and the message carries
  no ordering guarantee, so a positional read silently scrambles the arm.
- ``connect()`` must refuse a motor bus that is down.  A HAL reporting
  "connected" over a dead bus makes every ``send_action`` succeed while the
  arm stands still — undetectable from upstream.
"""

from __future__ import annotations

from pathlib import Path

import openral_core.can as core_can
import pytest

pytest.importorskip("lerobot", reason="openral_hal imports lerobot at package import")

from openral_core.exceptions import ROSConfigError, ROSEStopRequested, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, RobotDescription
from openral_hal.openarm_real import OPENARM_REAL_DESCRIPTION, OpenArmRealHAL

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENARM_MANIFEST = REPO_ROOT / "robots" / "openarm" / "robot.yaml"

_ARPHRD_CAN = "280"


# ── Fixtures ──────────────────────────────────────────────────────────────────


class _Recorder:
    """Records every (topic, message) the HAL publishes."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, object]]] = []

    def __call__(self, topic: str, msg: dict[str, object]) -> None:
        self.sent.append((topic, msg))

    def by_topic(self, topic: str) -> dict[str, object]:
        return next(m for t, m in self.sent if t == topic)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _make_can_link(root: Path, name: str, *, arphrd: str = _ARPHRD_CAN, up: bool = True) -> None:
    _write(root / name / "type", arphrd)
    _write(root / name / "operstate", "up" if up else "down")


@pytest.fixture
def both_buses_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "net"
    _make_can_link(root, "openarm_left")
    _make_can_link(root, "openarm_right")
    monkeypatch.setattr(core_can, "SYSFS_NET", str(root))
    return root


@pytest.fixture
def hal(both_buses_up: Path) -> OpenArmRealHAL:
    return OpenArmRealHAL()


def _action(*, horizon: int = 1) -> Action:
    """A 16-DoF action whose every entry is its own joint index.

    Values double as their index, so a mis-sliced or reordered publish is
    visible in the assertion rather than hidden behind equal numbers.
    """
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=horizon,
        joint_targets=[[float(i) for i in range(16)] for _ in range(horizon)],
    )


# ── Construction ──────────────────────────────────────────────────────────────


class TestConstruction:
    def test_defaults_to_the_real_description(self, both_buses_up: Path) -> None:
        assert OpenArmRealHAL().description.name == "openarm_v2"
        assert OPENARM_REAL_DESCRIPTION.sdk_kind == "open"

    def test_real_and_sim_descriptions_share_kinematics(self) -> None:
        from openral_hal import OPENARM_DESCRIPTION

        assert OPENARM_REAL_DESCRIPTION.joints == OPENARM_DESCRIPTION.joints
        assert OPENARM_REAL_DESCRIPTION.safety == OPENARM_DESCRIPTION.safety

    def test_joint_names_come_from_the_manifest_not_a_second_copy(self) -> None:
        hal = OpenArmRealHAL(require_can_links=False)
        assert hal.ros2_control_joint_names() == [
            str(j.sim_joint_name) for j in OPENARM_REAL_DESCRIPTION.joints
        ]

    def test_urdf_joint_names_match_the_bringup_controllers(self) -> None:
        names = OpenArmRealHAL(require_can_links=False).ros2_control_joint_names()
        assert names[0:7] == [f"openarm_left_joint{i}" for i in range(1, 8)]
        assert names[7] == "openarm_left_finger_joint1"
        assert names[8:15] == [f"openarm_right_joint{i}" for i in range(1, 8)]
        assert names[15] == "openarm_right_finger_joint1"

    def test_a_non_bimanual_manifest_is_rejected(self) -> None:
        single_arm = OPENARM_REAL_DESCRIPTION.model_copy(
            update={"joints": OPENARM_REAL_DESCRIPTION.joints[:8]}, deep=True
        )
        with pytest.raises(ROSConfigError, match="16-joint"):
            OpenArmRealHAL(single_arm, require_can_links=False)

    def test_a_manifest_without_sim_joint_names_is_rejected(self) -> None:
        # Without sim_joint_name there is no bridge to ros2_control naming,
        # and guessing one would silently command the wrong joints.
        stripped = OPENARM_REAL_DESCRIPTION.model_copy(
            update={
                "joints": [
                    j.model_copy(update={"sim_joint_name": None})
                    for j in OPENARM_REAL_DESCRIPTION.joints
                ]
            },
            deep=True,
        )
        with pytest.raises(ROSConfigError, match="sim_joint_name"):
            OpenArmRealHAL(stripped, require_can_links=False)


# ── CAN preflight ─────────────────────────────────────────────────────────────


class TestBusPreflight:
    def test_connect_succeeds_when_both_buses_are_up(self, hal: OpenArmRealHAL) -> None:
        hal.connect()
        assert hal.health().fields["left_can"] == "openarm_left"
        hal.disconnect()

    def test_connect_refuses_a_down_bus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "net"
        _make_can_link(root, "openarm_left")
        _make_can_link(root, "openarm_right", up=False)
        monkeypatch.setattr(core_can, "SYSFS_NET", str(root))
        with pytest.raises(ROSConfigError, match="openarm_right"):
            OpenArmRealHAL().connect()

    def test_connect_refuses_a_missing_bus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "net"
        _make_can_link(root, "openarm_left")
        monkeypatch.setattr(core_can, "SYSFS_NET", str(root))
        with pytest.raises(ROSConfigError, match="no network interface"):
            OpenArmRealHAL().connect()

    def test_connect_refuses_a_non_can_interface_of_the_right_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A veth or bridge that happens to carry the name is not a motor bus.
        root = tmp_path / "net"
        _make_can_link(root, "openarm_left")
        _make_can_link(root, "openarm_right", arphrd="1")
        monkeypatch.setattr(core_can, "SYSFS_NET", str(root))
        with pytest.raises(ROSConfigError, match="not a CAN link"):
            OpenArmRealHAL().connect()

    def test_a_failed_preflight_leaves_the_hal_disconnected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "net"
        monkeypatch.setattr(core_can, "SYSFS_NET", str(root))
        hal = OpenArmRealHAL()
        with pytest.raises(ROSConfigError):
            hal.connect()
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.read_state()

    def test_preflight_can_be_skipped_for_a_simulated_bringup(self) -> None:
        hal = OpenArmRealHAL(require_can_links=False)
        hal.connect()
        assert "skipped" in hal.health().fields["preflight"]
        hal.disconnect()


# ── send_action fan-out ───────────────────────────────────────────────────────


class TestSendAction:
    def test_one_action_reaches_all_four_controllers(self, both_buses_up: Path) -> None:
        recorder = _Recorder()
        hal = OpenArmRealHAL(publish_fn=recorder)
        hal.connect()
        hal.send_action(_action())
        assert [t for t, _ in recorder.sent] == hal.command_topics()

    def test_each_controller_receives_its_own_slice(self, both_buses_up: Path) -> None:
        recorder = _Recorder()
        hal = OpenArmRealHAL(publish_fn=recorder)
        hal.connect()
        hal.send_action(_action())
        left_arm, left_grip, right_arm, right_grip = hal.command_topics()
        assert recorder.by_topic(left_arm)["joint_targets"] == [[0.0, 1, 2, 3, 4, 5, 6]]
        assert recorder.by_topic(left_grip)["joint_targets"] == [[7.0]]
        assert recorder.by_topic(right_arm)["joint_targets"] == [[8.0, 9, 10, 11, 12, 13, 14]]
        assert recorder.by_topic(right_grip)["joint_targets"] == [[15.0]]

    def test_each_message_names_its_own_joints(self, both_buses_up: Path) -> None:
        recorder = _Recorder()
        hal = OpenArmRealHAL(publish_fn=recorder)
        hal.connect()
        hal.send_action(_action())
        left_arm, left_grip, _, right_grip = hal.command_topics()
        assert recorder.by_topic(left_arm)["joint_names"] == [
            f"openarm_left_joint{i}" for i in range(1, 8)
        ]
        assert recorder.by_topic(left_grip)["joint_names"] == ["openarm_left_finger_joint1"]
        assert recorder.by_topic(right_grip)["joint_names"] == ["openarm_right_finger_joint1"]

    def test_a_multi_step_chunk_is_sliced_per_step(self, both_buses_up: Path) -> None:
        recorder = _Recorder()
        hal = OpenArmRealHAL(publish_fn=recorder)
        hal.connect()
        hal.send_action(_action(horizon=3))
        left_grip = hal.command_topics()[1]
        assert recorder.by_topic(left_grip)["joint_targets"] == [[7.0], [7.0], [7.0]]

    def test_a_wrong_width_action_publishes_nothing(self, both_buses_up: Path) -> None:
        # All-or-nothing: a rejected action must not leave one arm commanded
        # from the new chunk and the other holding an older setpoint.
        recorder = _Recorder()
        hal = OpenArmRealHAL(publish_fn=recorder)
        hal.connect()
        with pytest.raises(ROSConfigError):
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[[0.0] * 8],
                )
            )
        assert recorder.sent == []

    def test_an_unsupported_control_mode_publishes_nothing(self, both_buses_up: Path) -> None:
        recorder = _Recorder()
        hal = OpenArmRealHAL(publish_fn=recorder)
        hal.connect()
        with pytest.raises(ROSConfigError):
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_VELOCITY,
                    horizon=1,
                    joint_targets=[[0.0] * 16],
                )
            )
        assert recorder.sent == []

    def test_send_action_before_connect_is_refused(self, both_buses_up: Path) -> None:
        recorder = _Recorder()
        hal = OpenArmRealHAL(publish_fn=recorder)
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.send_action(_action())
        assert recorder.sent == []


# ── read_state ────────────────────────────────────────────────────────────────


class TestReadState:
    @staticmethod
    def _shuffled_state(hal: OpenArmRealHAL) -> dict[str, object]:
        """`/joint_states` in an order no controller would coincidentally emit.

        Position values encode the joint's *manifest* index, so a positional
        read produces visibly wrong numbers.
        """
        names = hal.ros2_control_joint_names()
        order = list(reversed(range(16)))
        return {
            "name": [names[i] for i in order],
            "position": [float(i) for i in order],
            "velocity": [float(i) * 0.1 for i in order],
            "effort": [float(i) * 0.01 for i in order],
        }

    def test_state_is_matched_by_name_not_index(self, both_buses_up: Path) -> None:
        hal = OpenArmRealHAL(require_can_links=False)
        hal._state_fn = lambda: self._shuffled_state(hal)
        hal.connect()
        state = hal.read_state()
        assert state.position == [float(i) for i in range(16)]
        assert state.velocity == pytest.approx([i * 0.1 for i in range(16)])

    def test_state_is_returned_in_manifest_joint_order(self, both_buses_up: Path) -> None:
        hal = OpenArmRealHAL(require_can_links=False)
        hal._state_fn = lambda: self._shuffled_state(hal)
        hal.connect()
        assert hal.read_state().name == [j.name for j in hal.description.joints]

    def test_a_partial_joint_state_is_an_error_not_a_zero_fill(self, both_buses_up: Path) -> None:
        # A gripper controller that failed to spawn must not read as "gripper
        # at 0.0" — that is a plausible pose, so it would go unnoticed.
        hal = OpenArmRealHAL(require_can_links=False)
        names = hal.ros2_control_joint_names()[:15]
        hal._state_fn = lambda: {
            "name": names,
            "position": [0.0] * 15,
        }
        hal.connect()
        with pytest.raises(ROSRuntimeError, match="openarm_right_finger_joint1"):
            hal.read_state()

    def test_no_subscriber_yields_a_zeroed_state_in_manifest_order(
        self, both_buses_up: Path
    ) -> None:
        hal = OpenArmRealHAL(require_can_links=False)
        hal.connect()
        state = hal.read_state()
        assert state.name == [j.name for j in hal.description.joints]
        assert state.position == [0.0] * 16


# ── Safety ────────────────────────────────────────────────────────────────────


class TestSafety:
    def test_estop_raises_and_drops_the_connection(self, both_buses_up: Path) -> None:
        recorder = _Recorder()
        hal = OpenArmRealHAL(publish_fn=recorder)
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        with pytest.raises(ROSRuntimeError):
            hal.send_action(_action())

    def test_estop_recovery_is_declared_resettable(self, both_buses_up: Path) -> None:
        from openral_hal.protocol import EStopRecovery, LifecycleEStopHAL

        hal = OpenArmRealHAL(require_can_links=False)
        assert isinstance(hal, LifecycleEStopHAL)
        assert hal.estop_recovery is EStopRecovery.RESETTABLE


# ── Manifest wiring ───────────────────────────────────────────────────────────


class TestManifestWiring:
    def test_committed_manifest_points_at_this_adapter(self) -> None:
        manifest = RobotDescription.from_yaml(str(OPENARM_MANIFEST))
        assert manifest.hal.real == "openral_hal.openarm_real:OpenArmRealHAL"

    def test_manifest_hal_parameters_are_accepted_by_the_constructor(self) -> None:
        # build_hal filters unknown keys, but the ones the manifest declares
        # for the real HAL must genuinely be constructor arguments — a typo
        # here would be silently dropped and the rig would use the defaults.
        import inspect

        manifest = RobotDescription.from_yaml(str(OPENARM_MANIFEST))
        accepted = set(inspect.signature(OpenArmRealHAL.__init__).parameters)
        declared = set(manifest.hal.parameters.defaults)
        real_only = {k for k in declared if k.endswith(("_can_interface", "_controller"))}
        assert real_only, "manifest declares no real-HAL parameters"
        assert real_only <= accepted

    def test_the_hal_builds_from_the_committed_manifest(self, both_buses_up: Path) -> None:
        manifest = RobotDescription.from_yaml(str(OPENARM_MANIFEST))
        params = manifest.hal.parameters.defaults
        import inspect

        accepted = set(inspect.signature(OpenArmRealHAL.__init__).parameters)
        kwargs = {k: v for k, v in params.items() if k in accepted}
        hal = OpenArmRealHAL(manifest, **kwargs)  # type: ignore[arg-type]  # reason: manifest-typed
        hal.connect()
        assert hal.command_topics()[0] == "/left_joint_trajectory_controller/joint_trajectory"
        hal.disconnect()
