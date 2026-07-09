"""Sim tests for :class:`openral_hal.AnvilOpenArmV2MujocoHAL` against real MuJoCo physics.

These tests load the **Anvil OpenARM 2.0** bimanual MJCF — the standard
Enactic OpenArm v2 with Anvil's documented range deltas (J1 +/-135 deg,
J6 -45..+70 deg of radial deviation) and the wrist support bracket — via
:mod:`openral_hal._anvil_openarm_v2_assets`, and exercise the full HAL
lifecycle (connect → read_state → send_action → estop / disconnect)
against a real ``mj_step`` loop.  No mocks; the closed-loop behaviour
comes from the MJCF's own native ``<position>`` actuators, exactly like
the Enactic v2 suite.

The point of this suite is the Anvil OpenARM "real hardware first day"
contract (CLAUDE.md §1.11): if these tests pass, the 16-DoF
``left arm 7 + left gripper 1 + right arm 7 + right gripper 1`` action
layout, the joint indexing (driven finger per side, follower finger via
``<equality>``, hidden from the public surface), the **Anvil-only
ranges** (a J6 target of +66 deg converges here and is impossible on
the stock v2 arm), and the ``RobotDescription`` round-trip are
guaranteed to match what a future ``AnvilOpenArmV2RealHAL`` (wrapping
Anvil's driver stack) will see on the physical arm.

Gravity is disabled in the closed-loop tests so the joint positions
converge exactly to the commanded pose; the lifecycle / read_state
tests run gravity-on to mirror production.
"""

from __future__ import annotations

import os
import time

import pytest

# Use try/except → boolean + `pytestmark.skipif` rather than module-level
# `pytest.skip(allow_module_level=True)`: with `tests/sim/__init__.py`
# making this directory a Package, a Skipped raised at module-import time
# poisons the whole `tests/sim` Package collection.  Our HAL never
# renders in these tests — we only need the physics — so deferring the
# decision to `pytestmark` keeps this module importable on hosts where
# MuJoCo's eager OSMesa probe or the pinned-clone fetch fails.
try:
    import mujoco
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

# The Anvil 2.0 MJCF is fetched by ``ensure_anvil_openarm_v2_mjcf`` on
# first use (pinned clone of bensonlee5/anvil-openarm-mujoco plus its
# enactic mesh submodule).  Skip the suite cleanly on hosts where
# ``git`` isn't on the PATH or the clone can't complete.
try:
    from openral_hal._anvil_openarm_v2_assets import ensure_anvil_openarm_v2_mjcf

    _ANVIL_MJCF: str = ensure_anvil_openarm_v2_mjcf()
    if not os.path.isfile(_ANVIL_MJCF):
        raise FileNotFoundError(f"missing MJCF at {_ANVIL_MJCF}")
    _MJCF_ERROR: str | None = None
except Exception as exc:
    _ANVIL_MJCF = ""  # type: ignore[assignment]
    _MJCF_ERROR = str(exc)

from openral_core import (
    Action,
    ControlMode,
    EmbodimentKind,
    JointState,
    JointType,
    ROSConfigError,
    ROSRuntimeError,
)
from openral_hal import ANVIL_OPENARM_V2_DESCRIPTION, AnvilOpenArmV2MujocoHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"Anvil OpenARM 2.0 MJCF unavailable: {_MJCF_ERROR}",
    ),
]


_EXPECTED_ANVIL_JOINT_ORDER: list[str] = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
    "left_gripper",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
    "right_gripper",
]

# The two Anvil deltas vs the standard v2 arm, verbatim from the docs
# comparison table (J1 +/-135 deg; J6 -45..+70 deg).
_ANVIL_J1_RANGE = (-2.3562, 2.3562)
_ANVIL_J6_RANGE = (-0.7854, 1.2217)


# ── Schema-level checks (cheap; do not require connect) ───────────────────────


class TestAnvilDescription:
    def test_canonical_description_shape(self) -> None:
        desc = ANVIL_OPENARM_V2_DESCRIPTION
        assert desc.name == "anvil_openarm_v2"
        assert desc.embodiment_kind == EmbodimentKind.BIMANUAL.value
        # 7 arm + 1 gripper per side = 16
        assert len(desc.joints) == 16

    def test_joint_names_match_expected_layout(self) -> None:
        names = [j.name for j in ANVIL_OPENARM_V2_DESCRIPTION.joints]
        assert names == _EXPECTED_ANVIL_JOINT_ORDER

    def test_all_joints_are_revolute(self) -> None:
        # 2.0-era grippers are hinge jaws, same as the Enactic v2 arm.
        for j in ANVIL_OPENARM_V2_DESCRIPTION.joints:
            assert j.joint_type == JointType.REVOLUTE.value, j.name

    def test_anvil_range_deltas(self) -> None:
        """J1 and J6 carry the Anvil ranges on BOTH arms — the entire
        kinematic difference vs robots/openarm (Enactic v2)."""
        by_name = {j.name: j for j in ANVIL_OPENARM_V2_DESCRIPTION.joints}
        for side in ("left", "right"):
            assert by_name[f"{side}_joint1"].position_limits == pytest.approx(_ANVIL_J1_RANGE)
            assert by_name[f"{side}_joint6"].position_limits == pytest.approx(_ANVIL_J6_RANGE)

    def test_capabilities_advertise_joint_position(self) -> None:
        modes = ANVIL_OPENARM_V2_DESCRIPTION.capabilities.supported_control_modes
        assert ControlMode.JOINT_POSITION.value in modes

    def test_capabilities_bimanual(self) -> None:
        assert ANVIL_OPENARM_V2_DESCRIPTION.capabilities.bimanual is True

    def test_embodiment_tags_include_anvil(self) -> None:
        tags = ANVIL_OPENARM_V2_DESCRIPTION.capabilities.embodiment_tags
        assert "openarm" in tags
        assert "anvil_openarm_v2" in tags
        assert "anvil" in tags


# ── MJCF schema invariants (catches pinned-asset drift) ──────────────────────


class TestUpstreamSchema:
    """Guard against silent drift of the pinned Anvil 2.0 MJCF.

    The wiring relies on the v2-era 18-joint / 16-actuator layout, on
    every actuator being ``<position>`` mode, and on the Anvil range
    deltas plus the bracket geoms being present.  The asset is pinned
    by SHA, so drift only arrives via a deliberate pin bump — these
    guards fail before the closed-loop tests do.
    """

    def test_joint_count_and_order(self) -> None:
        model = mujoco.MjModel.from_xml_path(_ANVIL_MJCF)
        # 18 joints — 16 actuated + 2 follower fingers (one per
        # gripper) coupled via an ``<equality>`` constraint.
        assert model.njnt == 18
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        assert names == [
            "openarm_left_joint1",
            "openarm_left_joint2",
            "openarm_left_joint3",
            "openarm_left_joint4",
            "openarm_left_joint5",
            "openarm_left_joint6",
            "openarm_left_joint7",
            "openarm_left_finger_joint1",  # driven (act[7])
            "openarm_left_finger_joint2",  # follower (no actuator)
            "openarm_right_joint1",
            "openarm_right_joint2",
            "openarm_right_joint3",
            "openarm_right_joint4",
            "openarm_right_joint5",
            "openarm_right_joint6",
            "openarm_right_joint7",
            "openarm_right_finger_joint1",  # driven (act[15])
            "openarm_right_finger_joint2",  # follower (no actuator)
        ]

    def test_actuators_are_position_mode(self) -> None:
        """Every actuator is ``<position>`` mode — the property that lets
        the HAL stay a thin manifest-driven wrapper (no software PD)."""
        model = mujoco.MjModel.from_xml_path(_ANVIL_MJCF)
        assert model.nu == 16
        for i in range(model.nu):
            gain = float(model.actuator_gainprm[i][0])
            bias_p = float(model.actuator_biasprm[i][1])
            assert gain > 1.0, f"act[{i}] gain={gain} suggests torque mode"
            assert bias_p == pytest.approx(-gain, abs=1e-3), (
                f"act[{i}] gainprm={gain}, biasprm[1]={bias_p}; "
                "expected position actuator (gainprm == -biasprm[1])"
            )

    def test_anvil_range_deltas_in_mjcf(self) -> None:
        """The two Anvil deltas are baked into the pinned MJCF on both
        arms (identical numerics; upstream mirrors via flipped axes)."""
        model = mujoco.MjModel.from_xml_path(_ANVIL_MJCF)
        for side in ("left", "right"):
            for jname, expected in (
                (f"openarm_{side}_joint1", _ANVIL_J1_RANGE),
                (f"openarm_{side}_joint6", _ANVIL_J6_RANGE),
            ):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                lo, hi = model.jnt_range[jid]
                assert (lo, hi) == pytest.approx(expected, abs=1e-3), jname

    def test_wrist_bracket_present(self) -> None:
        """The Anvil wrist bracket (the part that enables J6's +70 deg)
        ships as visual-only CAD mesh geoms hosted on both link5 forearm
        bodies (it clamps the J6 hub, so it must not rotate with J6)."""
        model = mujoco.MjModel.from_xml_path(_ANVIL_MJCF)
        for side in ("left", "right"):
            gname = f"anvil_wrist_bracket_{side}"
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            assert gid >= 0, f"bracket geom {gname!r} missing from the pinned MJCF"
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_link5")
            assert model.geom_bodyid[gid] == bid
            # Visual-only: must not take part in collision.
            assert model.geom_contype[gid] == 0
            assert model.geom_conaffinity[gid] == 0
            # Machined aluminum (via its material): bright and neutral.
            rgba = model.mat_rgba[model.geom_matid[gid]]
            assert all(c > 0.5 for c in rgba[:3]), rgba
            assert max(rgba[:3]) - min(rgba[:3]) < 0.1, rgba

    def test_wrist_cameras_present(self) -> None:
        model = mujoco.MjModel.from_xml_path(_ANVIL_MJCF)
        for name in ("camera_wrist_left", "camera_wrist_right"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0, name


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def hal() -> AnvilOpenArmV2MujocoHAL:
    """Fresh Anvil 2.0 HAL with gravity off and enough settle steps for
    the MJCF's native position-actuator PD to converge."""
    return AnvilOpenArmV2MujocoHAL(gravity_enabled=False, settle_steps=2000)


@pytest.fixture()
def connected_hal(hal: AnvilOpenArmV2MujocoHAL) -> AnvilOpenArmV2MujocoHAL:
    hal.connect()
    yield hal
    hal.disconnect()


def _action(target: list[float]) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[target],
        stamp_ns=time.time_ns(),
    )


# ── Anvil-specific lifecycle tests ────────────────────────────────────────────
#
# Shared protocol compliance and standard lifecycle tests are consolidated in
# tests/sim/test_hal_protocol_contracts.py (parametrized across all MuJoCo HALs).
# Keep only Anvil-specific tests here.


class TestAnvilLifecycle:
    def test_connect_loads_mujoco_model(self, hal: AnvilOpenArmV2MujocoHAL) -> None:
        hal.connect()
        try:
            assert hal._connected is True
            assert hal._model is not None
            assert hal._data is not None
            assert hal._model.nu == 16  # 16 position actuators, v2-era layout
        finally:
            hal.disconnect()

    def test_connect_seeds_ctrl_from_qpos(self, hal: AnvilOpenArmV2MujocoHAL) -> None:
        # Driven by ``ANVIL_OPENARM_V2_DESCRIPTION.sim.seed_ctrl_from_qpos``
        # (ADR-0023) — the native <position> actuators must hold the
        # rest pose on the first mj_step rather than yanking to ctrl=0.
        hal.connect()
        try:
            assert hal._data is not None
            for name in hal._joint_names:
                qpos_idx = hal._joint_qpos_addr[name]
                act_idx = hal._actuator_index[name]
                assert hal._data.ctrl[act_idx] == pytest.approx(
                    float(hal._data.qpos[qpos_idx]), abs=1e-6
                )
        finally:
            hal.disconnect()


# ── read_state ────────────────────────────────────────────────────────────────


class TestReadState:
    def test_returns_jointstate_with_sixteen_joints(
        self, connected_hal: AnvilOpenArmV2MujocoHAL
    ) -> None:
        state = connected_hal.read_state()
        assert isinstance(state, JointState)
        assert len(state.name) == 16
        assert state.name == [j.name for j in ANVIL_OPENARM_V2_DESCRIPTION.joints]
        assert len(state.position) == 16
        assert len(state.velocity) == 16
        assert state.stamp_ns > 0


# ── send_action ───────────────────────────────────────────────────────────────


class TestSendAction:
    def test_rejects_wrong_joint_count(self, connected_hal: AnvilOpenArmV2MujocoHAL) -> None:
        bad = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * 15],
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="16 joints"):
            connected_hal.send_action(bad)


# ── Closed-loop physics (native v2 PD — exact convergence) ───────────────────


class TestClosedLoopMujoco:
    """Real MuJoCo physics — no mocks.  The MJCF's native ``<position>``
    actuators with per-class kp/kv track step inputs cleanly, so every
    closed-loop test asserts **exact** convergence to the commanded
    pose — the same bar the Enactic v2 suite sets.
    """

    def test_hold_zero_pose(self, connected_hal: AnvilOpenArmV2MujocoHAL) -> None:
        connected_hal.send_action(_action([0.0] * 16))
        state = connected_hal.read_state()
        for i, q in enumerate(state.position):
            assert abs(q) < 5e-3, f"joint {state.name[i]!r} drifted to {q:.4f}"

    # 0.15 rad fits inside every arm joint's range on the side each
    # per-joint test commands it (J2's narrow side is ~0.17 rad).
    _ANVIL_SAFE_ARM_TARGET: float = 0.15

    def test_left_arm_per_joint_convergence(self, connected_hal: AnvilOpenArmV2MujocoHAL) -> None:
        for slot in range(7):
            # left_joint2's positive range ends at +0.17 — command its
            # wide (negative) side; left_joint4 is positive-only.
            sign = -1.0 if slot == 1 else 1.0
            expected = sign * self._ANVIL_SAFE_ARM_TARGET
            target = [0.0] * 16
            target[slot] = expected
            connected_hal.send_action(_action(target))
            state = connected_hal.read_state()
            assert state.position[slot] == pytest.approx(expected, abs=1e-2), (
                f"left-arm slot {slot} ({state.name[slot]!r}) failed to converge: "
                f"read {state.position[slot]!r}"
            )

    def test_right_arm_per_joint_convergence(self, connected_hal: AnvilOpenArmV2MujocoHAL) -> None:
        for slot in range(8, 15):
            # right_joint2's negative range ends at -0.17 (mirror of the
            # left) — command its wide (positive) side.
            expected = self._ANVIL_SAFE_ARM_TARGET
            target = [0.0] * 16
            target[slot] = expected
            connected_hal.send_action(_action(target))
            state = connected_hal.read_state()
            assert state.position[slot] == pytest.approx(expected, abs=1e-2), (
                f"right-arm slot {slot} ({state.name[slot]!r}) failed to converge: "
                f"read {state.position[slot]!r}"
            )

    def test_anvil_only_ranges_converge(self, connected_hal: AnvilOpenArmV2MujocoHAL) -> None:
        """Targets that are mechanically impossible on the stock v2 arm.

        J6 at +66 deg (stock v2 clamps at +45) and J1 at +/-115 deg on
        the side stock v2 clamps at 80 deg — the two Anvil deltas,
        exercised end-to-end through the HAL.  If the pin ever regresses
        to stock v2 ranges, the actuator-side ctrlrange clamp masks the
        target and this fails loudly.
        """
        target = [0.0] * 16
        target[0] = -2.0  # left_joint1 — beyond stock v2's -1.40 clamp
        target[5] = 1.15  # left_joint6 — beyond stock v2's +0.785 clamp
        target[8] = 2.0  # right_joint1 — beyond stock v2's +... mirrored clamp
        target[13] = 1.15  # right_joint6
        connected_hal.send_action(_action(target))
        state = connected_hal.read_state()
        for i in (0, 5, 8, 13):
            assert state.position[i] == pytest.approx(target[i], abs=1e-2), (
                f"Anvil-range slot {i} ({state.name[i]!r}) read {state.position[i]!r}, "
                f"expected {target[i]!r} — has the pin regressed to stock v2 ranges?"
            )

    def test_both_arms_converge_simultaneously(
        self, connected_hal: AnvilOpenArmV2MujocoHAL
    ) -> None:
        target = [0.0] * 16
        target[0] = -0.3
        target[1] = -0.2
        target[3] = 0.5  # left_joint4 is positive-only
        target[8] = 0.3
        target[9] = 0.2
        target[11] = 0.5  # right_joint4 is positive-only
        connected_hal.send_action(_action(target))
        state = connected_hal.read_state()
        for i in (0, 1, 3, 8, 9, 11):
            assert state.position[i] == pytest.approx(target[i], abs=1e-2), (
                f"slot {i} ({state.name[i]!r}) read {state.position[i]!r}, expected {target[i]!r}"
            )

    def test_left_gripper_opens(self, connected_hal: AnvilOpenArmV2MujocoHAL) -> None:
        # Left gripper ctrlrange is [0, 0.7854] rad (closed → open).
        target = [0.0] * 16
        target[7] = 0.5
        connected_hal.send_action(_action(target))
        state = connected_hal.read_state()
        assert state.position[7] == pytest.approx(0.5, abs=1e-2)
        # Right gripper stays closed.
        assert state.position[15] == pytest.approx(0.0, abs=1e-2)

    def test_right_gripper_opens(self, connected_hal: AnvilOpenArmV2MujocoHAL) -> None:
        # Right gripper ctrlrange is [-0.7854, 0] rad (mirrored).
        target = [0.0] * 16
        target[15] = -0.5
        connected_hal.send_action(_action(target))
        state = connected_hal.read_state()
        assert state.position[7] == pytest.approx(0.0, abs=1e-2)
        assert state.position[15] == pytest.approx(-0.5, abs=1e-2)

    def test_follower_finger_tracks_driver(self, connected_hal: AnvilOpenArmV2MujocoHAL) -> None:
        """The MJCF couples ``finger_joint2`` to ``finger_joint1`` via an
        ``<equality>`` constraint per side.  Commanding the driven finger
        must move the follower in tandem."""
        target = [0.0] * 16
        target[7] = 0.4  # left gripper
        target[15] = -0.4  # right gripper
        connected_hal.send_action(_action(target))
        # Reach into the MJCF state for the follower finger qpos
        # (left_finger_joint2 = qpos[8]; right_finger_joint2 = qpos[17]).
        assert connected_hal._data is not None
        left_driver = float(connected_hal._data.qpos[7])
        left_follower = float(connected_hal._data.qpos[8])
        right_driver = float(connected_hal._data.qpos[16])
        right_follower = float(connected_hal._data.qpos[17])
        assert left_driver == pytest.approx(left_follower, abs=1e-2), (
            f"left finger driver {left_driver!r} vs follower {left_follower!r} — "
            "equality constraint missing?"
        )
        assert right_driver == pytest.approx(right_follower, abs=1e-2), (
            f"right finger driver {right_driver!r} vs follower {right_follower!r} — "
            "equality constraint missing?"
        )

    def test_per_slot_identity_wiring(self, connected_hal: AnvilOpenArmV2MujocoHAL) -> None:
        """Wiring check — every action slot drives exactly its joint.

        Sweeps the 16 slots one at a time with a fresh HAL per
        iteration (no state accumulation) and asserts the commanded
        joint converges to the commanded value while every other slot
        stays at zero.  Alternates signs to catch sign flips, L/R
        swaps, and off-by-one wiring errors.
        """
        # left_joint2's positive range ends at +0.17 and right_joint2's
        # negative range at -0.17 (mirrored); left_joint4 / right_joint4
        # are positive-only; left gripper positive-only, right gripper
        # negative-only.  Pin those slots; alternate the rest.
        forced_sign = {1: -1.0, 3: 1.0, 7: 1.0, 9: 1.0, 11: 1.0, 15: -1.0}
        gripper_slots = {7, 15}
        for slot in range(16):
            sign = forced_sign.get(slot, 1.0 if slot % 2 == 0 else -1.0)
            magnitude = 0.3 if slot in gripper_slots else self._ANVIL_SAFE_ARM_TARGET
            expected = sign * magnitude
            hal = AnvilOpenArmV2MujocoHAL(gravity_enabled=False, settle_steps=2000)
            hal.connect()
            try:
                target = [0.0] * 16
                target[slot] = expected
                hal.send_action(_action(target))
                state = hal.read_state()
                assert state.position[slot] == pytest.approx(expected, abs=1e-2), (
                    f"slot {slot} ({state.name[slot]!r}) read {state.position[slot]!r}, "
                    f"expected {expected!r} — wiring mismatch?"
                )
                for other in range(16):
                    if other == slot:
                        continue
                    assert abs(state.position[other]) < 2e-2, (
                        f"slot {other} ({state.name[other]!r}) drifted to "
                        f"{state.position[other]} when only slot {slot} was commanded"
                    )
            finally:
                hal.disconnect()


# ── Full lifecycle sequence ───────────────────────────────────────────────────


class TestFullLifecycle:
    def test_connect_read_send_disconnect(self, hal: AnvilOpenArmV2MujocoHAL) -> None:
        hal.connect()
        try:
            state0 = hal.read_state()
            assert len(state0.name) == 16

            target = [0.0] * 16
            target[0] = -0.3  # left_joint1
            target[5] = 1.0  # left_joint6 — inside the Anvil-only range
            target[7] = 0.4  # left gripper open
            target[8] = 0.3  # right_joint1
            target[13] = 1.0  # right_joint6 — inside the Anvil-only range
            target[15] = -0.4  # right gripper open
            hal.send_action(_action(target))
            state1 = hal.read_state()
            for i in (0, 5, 7, 8, 13, 15):
                assert state1.position[i] == pytest.approx(target[i], abs=1e-2), (
                    f"slot {i} ({state1.name[i]!r}) read {state1.position[i]!r}, "
                    f"expected {target[i]!r}"
                )
        finally:
            hal.disconnect()
        with pytest.raises(ROSRuntimeError):
            hal.read_state()


# ── Wrist cameras (manifest SensorSpec → MJCF camera wiring) ─────────────────


class TestWristCameras:
    def test_read_images_renders_both_wrists(self, connected_hal: AnvilOpenArmV2MujocoHAL) -> None:
        """``read_images`` renders the manifest's two wrist sensors.

        Pins the ``sim_camera_name`` → MJCF camera wiring and the
        640x400 sensor intrinsics (sized to fit MuJoCo's default
        offscreen framebuffer — the MJCF declares no ``offwidth``).
        Offscreen GL is host-specific, so probe it with a real
        Renderer first and skip honestly when absent (CLAUDE.md
        §1.11) rather than asserting on the HAL's failure fallback.
        """
        assert connected_hal._model is not None
        try:
            probe = mujoco.Renderer(connected_hal._model, 64, 64)
        except Exception as exc:  # GL probe is env-specific
            pytest.skip(f"offscreen renderer unavailable: {exc}")
        probe.close()

        images = connected_hal.read_images()
        assert set(images) == {"wrist_left", "wrist_right"}
        for name, frame in images.items():
            assert frame.shape == (400, 640, 3), (name, frame.shape)
            assert frame.dtype.name == "uint8", (name, frame.dtype)
