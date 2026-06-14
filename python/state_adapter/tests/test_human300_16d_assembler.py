"""ADR-0027 — human300_16d assembler. Per-field correctness pinned.

No mocks (CLAUDE.md §1.11). The TF lookup is a real callable
(``tf_lookup: TfLookup``), implemented in-test as a dict-backed function
that returns real ``TransformView`` dataclasses. The schema models
(:class:`StateContractBindings`) are real Pydantic instances. The only
thing absent is the live ROS graph — the skill_runner integration test
exercises the rclpy boundary.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from openral_core import ROSConfigError, StateContractBindings
from openral_state_adapter import (
    TfLookup,
    TransformView,
    assemble_state,
    registered_layouts,
)
from openral_state_adapter.layouts.human300_16d import assemble_human300_16d


def _make_tf_lookup(table: dict[tuple[str, str], TransformView]) -> TfLookup:
    """Build a real ``TfLookup`` from a frame-pair table.

    Raises :class:`LookupError` for missing pairs — same semantics as a
    real ``tf2_ros.Buffer.lookup_transform`` failure (the production
    wrapper translates ``tf2.LookupException`` into Python ``KeyError``
    / ``LookupError``).
    """

    def _lookup(target_frame: str, source_frame: str) -> TransformView:
        try:
            return table[(target_frame, source_frame)]
        except KeyError as exc:
            raise LookupError(
                f"no transform from {source_frame!r} to {target_frame!r}",
            ) from exc

    return _lookup


_PANDA_BINDINGS = StateContractBindings(
    eef_frame="panda_hand",
    base_frame="base_link",
    world_frame="map",
    gripper_qpos_joints=["panda_finger_joint1", "panda_finger_joint2"],
    quaternion_convention="xyzw",
)


class TestAssembleHuman300:
    def test_returns_16d_float32(self) -> None:
        tf = _make_tf_lookup(
            {
                ("base_link", "panda_hand"): TransformView(
                    position=(0.5, 0.0, 0.3),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                ("map", "base_link"): TransformView(
                    position=(1.0, 2.0, 0.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            },
        )
        joint_positions = {
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }
        out = assemble_human300_16d(_PANDA_BINDINGS, joint_positions, tf)
        assert out.shape == (16,)
        assert out.dtype == np.float32

    def test_field_order_matches_robocasa_backend(self) -> None:
        """Field order pinned verbatim to robocasa.py:508 — picking the
        wrong order silently feeds quaternions into gripper slots and
        vice versa (see the warning at line 503 in that file).
        """
        tf = _make_tf_lookup(
            {
                ("base_link", "panda_hand"): TransformView(
                    position=(0.1, 0.2, 0.3),
                    quaternion_xyzw=(0.4, 0.5, 0.6, 0.7),
                ),
                ("map", "base_link"): TransformView(
                    position=(1.1, 2.2, 0.0),
                    quaternion_xyzw=(0.8, 0.9, 1.0, 1.1),
                ),
            },
        )
        joint_positions = {
            "panda_finger_joint1": 0.030,
            "panda_finger_joint2": 0.029,
        }
        out = assemble_human300_16d(_PANDA_BINDINGS, joint_positions, tf)
        # base_to_eef.position
        np.testing.assert_allclose(out[0:3], [0.1, 0.2, 0.3])
        # base_to_eef.quaternion (xyzw)
        np.testing.assert_allclose(out[3:7], [0.4, 0.5, 0.6, 0.7])
        # world_to_base.position
        np.testing.assert_allclose(out[7:10], [1.1, 2.2, 0.0])
        # world_to_base.quaternion (xyzw)
        np.testing.assert_allclose(out[10:14], [0.8, 0.9, 1.0, 1.1])
        # gripper_qpos
        np.testing.assert_allclose(out[14:16], [0.030, 0.029])

    def test_wxyz_convention_permutes_quaternions(self) -> None:
        """When the manifest declares ``quaternion_convention: "wxyz"``,
        the assembler rotates each quaternion ONCE at the boundary.
        """
        bindings = StateContractBindings(
            eef_frame="panda_hand",
            base_frame="base_link",
            world_frame="map",
            gripper_qpos_joints=["g1", "g2"],
            quaternion_convention="wxyz",
        )
        tf = _make_tf_lookup(
            {
                ("base_link", "panda_hand"): TransformView(
                    position=(0.0, 0.0, 0.0),
                    quaternion_xyzw=(0.1, 0.2, 0.3, 0.4),
                ),
                ("map", "base_link"): TransformView(
                    position=(0.0, 0.0, 0.0),
                    quaternion_xyzw=(0.5, 0.6, 0.7, 0.8),
                ),
            },
        )
        out = assemble_human300_16d(bindings, {"g1": 0.0, "g2": 0.0}, tf)
        # wxyz = (w, x, y, z) = (0.4, 0.1, 0.2, 0.3)
        np.testing.assert_allclose(out[3:7], [0.4, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(out[10:14], [0.8, 0.5, 0.6, 0.7])

    def test_missing_gripper_joint_raises_key_error(self) -> None:
        """A topic-name skew between manifest bindings and the live
        ``/joint_states`` MUST surface, not silently zero-fill.
        """
        tf = _make_tf_lookup(
            {
                ("base_link", "panda_hand"): TransformView((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                ("map", "base_link"): TransformView((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            },
        )
        with pytest.raises(KeyError):
            assemble_human300_16d(_PANDA_BINDINGS, {"panda_finger_joint1": 0.0}, tf)

    def test_wrong_gripper_count_raises_config_error(self) -> None:
        # 1 joint = parallel-gripper abstraction (mirrored to [v, -v]) ✓
        # 2 joints = per-finger ✓
        # Anything else is a malformed manifest.
        bindings = StateContractBindings(
            eef_frame="panda_hand",
            base_frame="base_link",
            world_frame="map",
            gripper_qpos_joints=["a", "b", "c"],
            quaternion_convention="xyzw",
        )
        tf = _make_tf_lookup(
            {
                ("base_link", "panda_hand"): TransformView((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                ("map", "base_link"): TransformView((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            },
        )
        with pytest.raises(ROSConfigError, match="gripper joints"):
            assemble_human300_16d(bindings, {"a": 0.0, "b": 0.0, "c": 0.0}, tf)

    def test_parallel_gripper_single_joint_mirrors(self) -> None:
        """1-joint binding → mirrored to ``[v, -v]`` per the parallel
        gripper abstraction. Matches robosuite's franka two-finger mimic
        convention (opposite-sign open/close on the parallel mechanism).
        """
        bindings = StateContractBindings(
            eef_frame="panda_hand",
            base_frame="base_link",
            world_frame="map",
            gripper_qpos_joints=["panda_gripper"],
            quaternion_convention="xyzw",
        )
        tf = _make_tf_lookup(
            {
                ("base_link", "panda_hand"): TransformView((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                ("map", "base_link"): TransformView((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            },
        )
        out = assemble_human300_16d(bindings, {"panda_gripper": 0.04}, tf)
        assert out[14] == pytest.approx(0.04)
        assert out[15] == pytest.approx(-0.04)

    def test_quaternion_sign_canonicalized_to_positive_w(self) -> None:
        """Sign-flipped quats (q vs -q, mathematically equivalent
        rotations) must land in the same hemisphere bytewise — the
        deploy_sim TF chain and the sim_run RoboCasa proprio
        otherwise disagreed on sign for the same physical pose,
        producing a spurious ``max_abs_diff = 2.0`` on every quat
        slot of the dump-diff regression.
        """
        # Same rotation expressed with negative w — assembler must flip.
        tf = _make_tf_lookup(
            {
                ("base_link", "panda_hand"): TransformView(
                    position=(0.0, 0.0, 0.0),
                    quaternion_xyzw=(-0.1, -0.2, -0.3, -0.9),  # w < 0
                ),
                ("map", "base_link"): TransformView(
                    position=(0.0, 0.0, 0.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            },
        )
        out = assemble_human300_16d(
            _PANDA_BINDINGS,
            {"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0},
            tf,
        )
        # Canonicalised to w >= 0: all four components flip.
        np.testing.assert_allclose(out[3:7], [0.1, 0.2, 0.3, 0.9])

    def test_unit_quaternion_norm_preserved(self) -> None:
        """The assembler MUST NOT renormalize. Whatever TF says is what
        the policy sees — picking up a numerical drift is the upstream
        publisher's bug, not ours to mask.
        """
        # 45-degree yaw quaternion: (0, 0, sin(π/8), cos(π/8))
        sin_h, cos_h = math.sin(math.pi / 8), math.cos(math.pi / 8)
        tf = _make_tf_lookup(
            {
                ("base_link", "panda_hand"): TransformView(
                    position=(0.0, 0.0, 0.0),
                    quaternion_xyzw=(0.0, 0.0, sin_h, cos_h),
                ),
                ("map", "base_link"): TransformView(
                    position=(0.0, 0.0, 0.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            },
        )
        out = assemble_human300_16d(
            _PANDA_BINDINGS, {"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0}, tf
        )
        norm = math.sqrt(sum(float(x) ** 2 for x in out[3:7]))
        assert math.isclose(norm, 1.0, abs_tol=1e-6)


class TestRegistryIntegration:
    def test_human300_16d_registered_on_import(self) -> None:
        assert "human300_16d" in registered_layouts()

    def test_assemble_state_routes_through_registry(self) -> None:
        tf = _make_tf_lookup(
            {
                ("base_link", "panda_hand"): TransformView(
                    position=(0.5, 0.0, 0.3),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                ("map", "base_link"): TransformView(
                    position=(0.0, 0.0, 0.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            },
        )
        out = assemble_state(
            "human300_16d",
            _PANDA_BINDINGS,
            {"panda_finger_joint1": 0.02, "panda_finger_joint2": 0.02},
            tf,
        )
        assert out.shape == (16,)
        np.testing.assert_allclose(out[0:3], [0.5, 0.0, 0.3])

    def test_assemble_state_unknown_layout_raises(self) -> None:
        with pytest.raises(ROSConfigError, match="no assembler registered"):
            assemble_state(
                "smolvla_9d",  # joint-space, not in registry
                StateContractBindings(eef_frame="a", base_frame="b"),
                {},
                _make_tf_lookup({}),
            )
