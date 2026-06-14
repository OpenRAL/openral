"""Wire-shape unit tests for the RLDX SimplerEnv layouts.

The RLDX-1-FT-SIMPLER-{WIDOWX,GOOGLE} checkpoints both ship a strict
modality config in ``processor_config.json`` (consumed by the upstream
``RLDXSimPolicyWrapper`` at sidecar boot). Wire-shape regressions
between the openral side and the upstream wrapper land as opaque
``check_action`` failures over ZMQ that take a sidecar bootstrap (~minutes
+ GPU) to surface.

These tests pin the wire shape against synthetic obs so any drift in
the adapter's ``_build_simpler_widowx_obs`` / ``_build_simpler_google_obs``
fails locally without paying the sidecar boot.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_adapter(state_layout: str):
    from openral_core import VLASpec
    from openral_sim.policies.rldx import _RLDXSidecarAdapter

    # The dataclass calls ``__post_init__`` which tries to ping the
    # sidecar; we monkeypatch the post-init for these wire-only tests.
    adapter = _RLDXSidecarAdapter.__new__(_RLDXSidecarAdapter)
    object.__setattr__(adapter, "spec", VLASpec(id="rldx", weights_uri="stub"))
    object.__setattr__(adapter, "host", "127.0.0.1")
    object.__setattr__(adapter, "port", 0)
    object.__setattr__(adapter, "replan_steps", 8)
    object.__setattr__(adapter, "image_size", 256)
    object.__setattr__(adapter, "timeout_ms", 60_000)
    object.__setattr__(adapter, "flip_180", False)
    object.__setattr__(adapter, "state_layout", state_layout)
    object.__setattr__(adapter, "auto_spawn", False)
    object.__setattr__(adapter, "boot_timeout_s", 1.0)
    object.__setattr__(adapter, "quantization", "nf4")
    object.__setattr__(adapter, "embodiment_tag", "OXE_BRIDGE_ORIG")
    object.__setattr__(adapter, "model_id", None)
    object.__setattr__(adapter, "_camera_keys", ("camera1", "camera2"))
    object.__setattr__(adapter, "_last_input_frame", None)
    # Sticky-gripper state machine — same defaults as the dataclass field
    # so wire-shape tests start from a known "open and unlocked" state.
    object.__setattr__(adapter, "_sticky_gripper_target", 1.0)
    object.__setattr__(adapter, "_sticky_gripper_lock", 0)
    return adapter


def _make_simpler_obs(eef_pos: np.ndarray) -> dict:
    return {
        "images": {"camera1": np.zeros((480, 640, 3), dtype=np.uint8)},
        "state": np.zeros(16, dtype=np.float32),
        "task": "put the carrot on the plate",
        "raw": {"agent": {"eef_pos": eef_pos.astype(np.float32)}},
    }


class TestSimplerWidowXWireShape:
    """Pin the bridge_orig (SIMPLER-WIDOWX) wire schema.

    The canonical modality config is registered against
    ``EmbodimentTag.OXE_BRIDGE_ORIG`` in
    ``rldx/configs/data/simpler_widowx_config.py``:

    * ``state.end_effector_position`` (3-vec)
    * ``state.end_effector_rotation`` (3-vec Euler RPY)
    * ``state.gripper_position`` (1-vec)
    * ``video.image_0``
    """

    def test_obs_carries_expected_keys(self) -> None:
        from openral_sim.policies.rldx import (
            _SIMPLER_LANG_KEY,
            _SIMPLER_VIDEO_KEY_WIDOWX,
        )

        adapter = _make_adapter("simpler_widowx")
        obs = _make_simpler_obs(np.asarray([0.1, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.037]))
        wire = adapter._build_simpler_widowx_obs(obs, "put the carrot on the plate")
        assert set(wire.keys()) == {
            _SIMPLER_VIDEO_KEY_WIDOWX,
            "state.end_effector_position",
            "state.end_effector_rotation",
            "state.gripper_position",
            _SIMPLER_LANG_KEY,
        }

    def test_state_vectors_have_canonical_shapes(self) -> None:
        adapter = _make_adapter("simpler_widowx")
        obs = _make_simpler_obs(np.asarray([0.1, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5]))
        wire = adapter._build_simpler_widowx_obs(obs, "")
        # (B=1, T=1, D) for each — matches the canonical modality_config.
        assert wire["state.end_effector_position"].shape == (1, 1, 3)
        assert wire["state.end_effector_rotation"].shape == (1, 1, 3)
        assert wire["state.gripper_position"].shape == (1, 1, 1)

    def test_position_carries_proprio_xyz(self) -> None:
        adapter = _make_adapter("simpler_widowx")
        # SAPIEN proprio: [x, y, z, qw, qx, qy, qz, gripper]
        obs = _make_simpler_obs(np.asarray([0.1, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.037]))
        wire = adapter._build_simpler_widowx_obs(obs, "")
        np.testing.assert_array_almost_equal(
            wire["state.end_effector_position"].reshape(3),
            np.asarray([0.1, 0.2, 1.0]),
        )

    def test_video_is_resized_to_256x320(self) -> None:
        from openral_sim.policies.rldx import _SIMPLER_VIDEO_KEY_WIDOWX

        adapter = _make_adapter("simpler_widowx")
        obs = _make_simpler_obs(np.zeros(8))
        wire = adapter._build_simpler_widowx_obs(obs, "")
        video = wire[_SIMPLER_VIDEO_KEY_WIDOWX]
        # (B=1, T=1, H=256, W=320, C=3)
        assert video.shape == (1, 1, 256, 320, 3)
        assert video.dtype == np.uint8

    def test_gripper_is_rescaled_to_bridge_units(self) -> None:
        adapter = _make_adapter("simpler_widowx")
        # raw qpos at MS3's "fully open" upper bound → bridge max.
        obs_open = _make_simpler_obs(np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.037]))
        wire_open = adapter._build_simpler_widowx_obs(obs_open, "")
        assert float(wire_open["state.gripper_position"].reshape(-1)[0]) == pytest.approx(
            1.115, abs=1e-4
        )
        # raw qpos at MS3's "fully closed" lower bound → bridge min.
        obs_closed = _make_simpler_obs(np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.015]))
        wire_closed = adapter._build_simpler_widowx_obs(obs_closed, "")
        assert float(wire_closed["state.gripper_position"].reshape(-1)[0]) == pytest.approx(
            0.046, abs=1e-4
        )
        # Mid-stroke maps near bridge_data_v2 mean ~0.71.
        obs_mid = _make_simpler_obs(np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.027]))
        wire_mid = adapter._build_simpler_widowx_obs(obs_mid, "")
        mid_norm = float(wire_mid["state.gripper_position"].reshape(-1)[0])
        assert 0.5 < mid_norm < 0.8


class TestSimplerGoogleWireShape:
    """Pin the fractal20220817_data (SIMPLER-GOOGLE) wire schema.

    The canonical modality config is registered against
    ``EmbodimentTag.OXE_FRACTAL`` in
    ``rldx/configs/data/simpler_google_config.py``:

    * ``state.end_effector_position`` (3-vec)
    * ``state.end_effector_rotation`` (4-vec xyzw quat — Google ships
      raw quaternion, unlike WidowX which uses bridge-rotated Euler)
    * ``state.gripper_position`` (1-vec closedness)
    * ``video.image``
    """

    def test_obs_carries_expected_keys(self) -> None:
        from openral_sim.policies.rldx import (
            _SIMPLER_LANG_KEY,
            _SIMPLER_VIDEO_KEY_GOOGLE,
        )

        adapter = _make_adapter("simpler_google")
        obs = _make_simpler_obs(np.asarray([0.1, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.037]))
        wire = adapter._build_simpler_google_obs(obs, "pick the coke can")
        assert set(wire.keys()) == {
            _SIMPLER_VIDEO_KEY_GOOGLE,
            "state.end_effector_position",
            "state.end_effector_rotation",
            "state.gripper_position",
            _SIMPLER_LANG_KEY,
        }

    def test_state_vectors_have_canonical_shapes(self) -> None:
        adapter = _make_adapter("simpler_google")
        obs = _make_simpler_obs(np.asarray([0.1, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5]))
        wire = adapter._build_simpler_google_obs(obs, "")
        assert wire["state.end_effector_position"].shape == (1, 1, 3)
        # Google: 4-vec xyzw quat (vs WidowX's 3-vec Euler).
        assert wire["state.end_effector_rotation"].shape == (1, 1, 4)
        assert wire["state.gripper_position"].shape == (1, 1, 1)

    def test_quaternion_is_rolled_to_xyzw(self) -> None:
        """SAPIEN reports quat in wxyz order; the wire schema expects xyzw."""
        adapter = _make_adapter("simpler_google")
        # wxyz quat = [0.5, 0.5, 0.5, 0.5] (a 120° rotation)
        obs = _make_simpler_obs(np.asarray([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5]))
        wire = adapter._build_simpler_google_obs(obs, "")
        # Rolled by -1 → xyzw = [0.5, 0.5, 0.5, 0.5]; with a non-degenerate
        # input the roll is unambiguous: [w, x, y, z] → [x, y, z, w].
        rot = wire["state.end_effector_rotation"].reshape(4)
        np.testing.assert_array_almost_equal(rot, np.asarray([0.5, 0.5, 0.5, 0.5]))

    def test_gripper_is_closedness(self) -> None:
        adapter = _make_adapter("simpler_google")
        # raw_open = 0.037 (almost closed) → closedness = 1 - 0.037 = 0.963
        obs = _make_simpler_obs(np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.037]))
        wire = adapter._build_simpler_google_obs(obs, "")
        assert float(wire["state.gripper_position"].reshape(-1)[0]) == pytest.approx(1.0 - 0.037)


def _make_simpler_action_dict(*, chunk_len: int, gripper_close: list[float]) -> dict:
    """Build a fake action dict in the canonical vector layout for chunk tests."""
    return {
        "action.end_effector_position": np.zeros((1, chunk_len, 3), dtype=np.float32),
        "action.end_effector_rotation": np.zeros((1, chunk_len, 3), dtype=np.float32),
        "action.gripper_close": np.asarray(gripper_close, dtype=np.float32).reshape(
            1, chunk_len, 1
        ),
    }


class TestSimplerChunkAssembly:
    """Pin the action chunk assembly for both SimplerEnv layouts.

    The canonical action keys are vector-shaped per
    ``ModalityConfig.action`` in the upstream configs:
    ``end_effector_position`` (3-vec DELTA), ``end_effector_rotation``
    (3-vec DELTA), ``gripper_close`` (1-vec ABSOLUTE).
    """

    def test_chunk_preserves_raw_gripper_close(self) -> None:
        """Chunk assembly leaves gripper_close raw; per-step sticky machine handles it."""
        adapter = _make_adapter("simpler_widowx")
        action_dict = _make_simpler_action_dict(chunk_len=3, gripper_close=[0.1, 0.5, 0.9])
        chunk = adapter._assemble_simpler_chunk(action_dict)
        assert chunk.shape == (3, 7)
        np.testing.assert_array_almost_equal(chunk[:, 6], np.asarray([0.1, 0.5, 0.9]))

    def test_google_gripper_is_raw(self) -> None:
        adapter = _make_adapter("simpler_google")
        action_dict = _make_simpler_action_dict(chunk_len=2, gripper_close=[0.3, 0.7])
        chunk = adapter._assemble_simpler_chunk(action_dict)
        # Google: raw closedness, no binarization (sticky-gripper is applied
        # by the env / wrapper, not the chunk-assembler).
        assert chunk.shape == (2, 7)
        np.testing.assert_array_almost_equal(chunk[:, 6], np.asarray([0.3, 0.7]))


class TestWidowXStickyGripper:
    """Pin the per-step WidowX sticky-gripper state machine.

    The state machine is applied in ``_RLDXSidecarAdapter.step`` (not
    in chunk assembly) so it spans replan boundaries cleanly. nf4
    quantisation makes the policy's gripper_close output noisy near
    the 0.5 binarisation threshold, which caused the gripper to
    oscillate every step and never hold closed long enough to grasp.
    The sticky machine locks transitions for 15 steps after each
    confident (>0.75 or <0.25) command, mirroring the upstream Google
    fractal env's ``_postprocess_gripper`` contract.
    """

    def test_initial_state_is_open(self) -> None:
        adapter = _make_adapter("simpler_widowx")
        # First confident-open command should pass through as +1.
        result = adapter._apply_widowx_sticky_gripper(0.1)
        assert result == pytest.approx(1.0)

    def test_close_then_lock_for_15_steps(self) -> None:
        adapter = _make_adapter("simpler_widowx")
        # Confident close transitions to -1 and locks.
        assert adapter._apply_widowx_sticky_gripper(0.95) == pytest.approx(-1.0)
        # Next 15 calls must stay at -1 regardless of policy noise.
        for _ in range(15):
            assert adapter._apply_widowx_sticky_gripper(0.1) == pytest.approx(-1.0)
        # After lock expires, a confident-open command transitions back.
        assert adapter._apply_widowx_sticky_gripper(0.05) == pytest.approx(1.0)

    def test_unconfident_command_holds_current_state(self) -> None:
        adapter = _make_adapter("simpler_widowx")
        # No transition fires for values in [0.25, 0.75].
        for raw in (0.3, 0.5, 0.7):
            assert adapter._apply_widowx_sticky_gripper(raw) == pytest.approx(1.0)
        # A confident close still works after a run of "no opinion" values.
        assert adapter._apply_widowx_sticky_gripper(0.95) == pytest.approx(-1.0)


class TestLayoutToEmbodimentTag:
    """Pin the layout → upstream embodiment_tag map."""

    def test_simpler_layouts_dispatch_correctly(self) -> None:
        from openral_sim.policies.rldx import _RLDX_LAYOUT_TO_EMBODIMENT_TAG

        # The published FT-SIMPLER-* checkpoints' `processor_config.json`
        # only ship `bridge_orig` / `fractal20220817_data` modality
        # buckets; the unused OXE_WIDOWX / OXE_GOOGLE enum names crash
        # PolicyLoader.load with KeyError. Use OXE_BRIDGE_ORIG /
        # OXE_FRACTAL — the names whose `.value` matches a real bucket.
        assert _RLDX_LAYOUT_TO_EMBODIMENT_TAG["simpler_widowx"] == "OXE_BRIDGE_ORIG"
        assert _RLDX_LAYOUT_TO_EMBODIMENT_TAG["simpler_google"] == "OXE_FRACTAL"
        # LIBERO / GR1 / RC365 keep GENERAL_EMBODIMENT (no regression).
        assert _RLDX_LAYOUT_TO_EMBODIMENT_TAG["libero"] == "GENERAL_EMBODIMENT"
        assert _RLDX_LAYOUT_TO_EMBODIMENT_TAG["gr1"] == "GENERAL_EMBODIMENT"
        assert _RLDX_LAYOUT_TO_EMBODIMENT_TAG["rc365"] == "GENERAL_EMBODIMENT"
