"""Unit tests for the pure helpers inside the eval adapters.

These cover the parts of ``openral_sim.{policies,backends}.{libero, metaworld, smolvla}``
that don't require heavyweight optional deps (``lerobot``, ``torch``,
``transformers``).  The lazy-imported builders themselves are exercised via
their ``ROSConfigError`` failure paths so the no-backend message is locked in.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from openral_core import (
    PhysicsBackend,
    SceneSpec,
    SimEnvironment,
    TaskSpec,
    VLASpec,
)
from openral_core.exceptions import ROSConfigError
from openral_sim.backends import libero as libero_mod
from openral_sim.backends import metaworld as metaworld_mod
from openral_sim.policies import smolvla as smolvla_mod

# ── libero helpers ───────────────────────────────────────────────────────────


class TestLiberoParseTaskId:
    def test_valid_returns_int_index(self) -> None:
        assert libero_mod._parse_task_id("libero_spatial/3", "libero_spatial") == 3

    def test_missing_slash_raises(self) -> None:
        with pytest.raises(ROSConfigError, match="<suite>/<int>"):
            libero_mod._parse_task_id("just-a-name", "libero_spatial")

    def test_suite_mismatch_raises(self) -> None:
        with pytest.raises(ROSConfigError, match="does not match scene id"):
            libero_mod._parse_task_id("libero_object/1", "libero_spatial")

    def test_non_integer_index_raises(self) -> None:
        with pytest.raises(ROSConfigError, match="not an integer"):
            libero_mod._parse_task_id("libero_spatial/abc", "libero_spatial")


class TestQuatToAxisangle:
    def test_identity_quat_returns_zero(self) -> None:
        # [x,y,z,w] = [0,0,0,1] → no rotation
        out = libero_mod._quat_to_axisangle(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        assert out.shape == (3,)
        np.testing.assert_array_almost_equal(out, np.zeros(3))

    def test_180_about_z_axis(self) -> None:
        # quat for 180° about +Z → [0, 0, 1, 0]; axis-angle = [0, 0, π]
        out = libero_mod._quat_to_axisangle(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(out, np.array([0.0, 0.0, np.pi]), decimal=5)

    def test_clipping_handles_w_outside_range(self) -> None:
        # w slightly above 1.0 must be clipped — should not produce NaN
        out = libero_mod._quat_to_axisangle(np.array([0.0, 0.0, 0.0, 1.0001], dtype=np.float32))
        assert np.all(np.isfinite(out))


class TestLiberoSimWrapObs:
    def _make_sim(self) -> libero_mod._LiberoSim:
        return libero_mod._LiberoSim(
            scene=SceneSpec(id="libero_spatial", backend=PhysicsBackend.MUJOCO),
            task=TaskSpec(
                id="libero_spatial/0",
                scene_id="libero_spatial",
                instruction="pick block",
                max_steps=5,
            ),
            _env=MagicMock(task_description="pick the red block"),
            _last_pixels={},
        )

    def test_wraps_full_observation(self) -> None:
        sim = self._make_sim()
        obs_in = {
            "pixels": {
                "image": np.full((256, 256, 3), 7, dtype=np.uint8),
                "image2": np.full((256, 256, 3), 9, dtype=np.uint8),
            },
            "robot_state": {
                "eef": {
                    "pos": [0.1, 0.2, 0.3],
                    "quat": [0.0, 0.0, 0.0, 1.0],
                },
                "gripper": {"qpos": [0.04, 0.04]},
            },
        }
        wrapped = sim._wrap_obs(obs_in)
        assert wrapped["images"]["camera1"].shape == (256, 256, 3)
        assert wrapped["images"]["camera2"][0, 0, 0] == 9
        assert wrapped["state"].shape == (8,)
        assert wrapped["state"].dtype == np.float32
        # eef pos must round-trip
        np.testing.assert_array_almost_equal(wrapped["state"][:3], [0.1, 0.2, 0.3])
        # identity quat → zero axis-angle
        np.testing.assert_array_almost_equal(wrapped["state"][3:6], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(wrapped["state"][6:8], [0.04, 0.04])
        assert wrapped["task"] == "pick the red block"

    def test_missing_eef_returns_zero_state(self) -> None:
        sim = self._make_sim()
        wrapped = sim._wrap_obs({"pixels": {}, "robot_state": {}})
        assert wrapped["state"].shape == (8,)
        np.testing.assert_array_equal(wrapped["state"], np.zeros(8, dtype=np.float32))
        # Defaults for missing camera frames
        assert wrapped["images"]["camera1"].shape == (256, 256, 3)

    def test_render_returns_none_without_pixels(self) -> None:
        sim = self._make_sim()
        assert sim.render() is None

    def test_render_returns_copy_of_image(self) -> None:
        sim = self._make_sim()
        sim._last_pixels = {"image": np.full((10, 10, 3), 5, dtype=np.uint8)}
        out = sim.render()
        assert out is not None
        assert out.dtype == np.uint8
        assert out.shape == (10, 10, 3)

    def test_close_delegates_to_env(self) -> None:
        sim = self._make_sim()
        sim.close()
        sim._env.close.assert_called_once()

    def test_reset_calls_env_reset(self) -> None:
        sim = self._make_sim()
        sim._env.reset.return_value = ({"pixels": {}, "robot_state": {}}, {})
        wrapped = sim.reset(seed=42)
        sim._env.reset.assert_called_once_with(seed=42)
        assert "images" in wrapped

    def test_step_wraps_step_result(self) -> None:
        sim = self._make_sim()
        sim._env.step.return_value = (
            {"pixels": {}, "robot_state": {}},
            0.5,
            False,
            True,
            {"info_key": "v"},
        )
        result = sim.step(np.zeros(7, dtype=np.float32))
        assert result.reward == 0.5
        assert result.terminated is False
        assert result.truncated is True
        assert result.info["info_key"] == "v"


class TestLiberoBuilder:
    def test_unknown_suite_raises(self) -> None:
        env_cfg = SimEnvironment(
            robot_id="franka",
            scene=SceneSpec(id="not_a_suite", backend=PhysicsBackend.MUJOCO),
            task=TaskSpec(id="not_a_suite/0", scene_id="not_a_suite", instruction="x", max_steps=1),
            vla=VLASpec(id="zero", weights_uri="mock://"),
        )
        with pytest.raises(ROSConfigError, match="libero scene id must be one of"):
            libero_mod._build_libero_scene(env_cfg)


class TestLiberoControlModeResolution:
    """`_resolve_control_mode` picks scene pin > manifest > relative default."""

    def _env(self, *, backend_options: dict[str, Any] | None, weights_uri: str) -> SimEnvironment:
        return SimEnvironment(
            robot_id="franka_panda",
            scene=SceneSpec(
                id="libero_spatial",
                backend=PhysicsBackend.MUJOCO,
                backend_options=backend_options or {},
            ),
            task=TaskSpec(
                id="libero_spatial/0", scene_id="libero_spatial", instruction="", max_steps=220
            ),
            vla=VLASpec(id="x", weights_uri=weights_uri),
        )

    def test_scene_pin_wins(self) -> None:
        env_cfg = self._env(
            backend_options={"control_mode": "absolute"}, weights_uri="rskills/smolvla-libero"
        )
        assert libero_mod._resolve_control_mode(env_cfg) == "absolute"

    def test_xvla_manifest_declares_absolute(self) -> None:
        # No scene pin: xVLA's manifest sim_env_control_mode drives the env.
        # Under the old (scene-only) behaviour this returned "relative" and the
        # arm saturated static — the exact bug the deleted libero_spatial_xvla
        # scene worked around.
        env_cfg = self._env(backend_options=None, weights_uri="rskills/xvla-libero")
        assert libero_mod._resolve_control_mode(env_cfg) == "absolute"

    def test_delta_policy_defaults_relative(self) -> None:
        # smolvla-libero declares no sim_env_control_mode → OSC delta default.
        env_cfg = self._env(backend_options=None, weights_uri="rskills/smolvla-libero")
        assert libero_mod._resolve_control_mode(env_cfg) == "relative"


# ── metaworld helpers ────────────────────────────────────────────────────────


class TestMetaworldParseTaskId:
    def test_valid_returns_task_name(self) -> None:
        assert metaworld_mod._parse_task_id("metaworld/push-v3") == "push-v3"

    def test_wrong_prefix_raises(self) -> None:
        with pytest.raises(ROSConfigError, match="metaworld/<task-name>"):
            metaworld_mod._parse_task_id("libero/0")

    def test_missing_slash_raises(self) -> None:
        with pytest.raises(ROSConfigError, match="metaworld/<task-name>"):
            metaworld_mod._parse_task_id("push-v3")


class TestMetaworldSimWrapObs:
    def _make_sim(self) -> metaworld_mod._MetaworldSim:
        return metaworld_mod._MetaworldSim(
            scene=SceneSpec(
                id="metaworld",
                backend=PhysicsBackend.MUJOCO,
                observation_height=64,
                observation_width=64,
            ),
            task=TaskSpec(
                id="metaworld/push-v3",
                scene_id="metaworld",
                instruction="push the puck",
                max_steps=5,
            ),
            _env=MagicMock(),
        )

    def test_wraps_image_and_agent_pos(self) -> None:
        sim = self._make_sim()
        wrapped = sim._wrap_obs(
            {
                "pixels": np.full((480, 480, 3), 11, dtype=np.uint8),
                "agent_pos": np.array([0.1, 0.2, 0.3, 0.5], dtype=np.float32),
            }
        )
        assert wrapped["images"]["camera1"].shape == (480, 480, 3)
        assert wrapped["state"].shape == (4,)
        assert wrapped["state"].dtype == np.float32
        assert wrapped["task"] == "push the puck"
        assert sim._last_image is not None

    def test_missing_pixels_uses_zeros_at_scene_resolution(self) -> None:
        sim = self._make_sim()
        wrapped = sim._wrap_obs({})
        assert wrapped["images"]["camera1"].shape == (64, 64, 3)
        assert wrapped["state"].shape == (4,)
        np.testing.assert_array_equal(wrapped["state"], np.zeros(4, dtype=np.float32))

    def test_render_returns_none_before_step(self) -> None:
        sim = self._make_sim()
        assert sim.render() is None

    def test_render_returns_copy_after_obs(self) -> None:
        sim = self._make_sim()
        sim._wrap_obs({"pixels": np.full((10, 10, 3), 3, dtype=np.uint8)})
        out = sim.render()
        assert out is not None
        assert out.dtype == np.uint8
        assert out.shape == (10, 10, 3)

    def test_close_delegates_to_env(self) -> None:
        sim = self._make_sim()
        sim.close()
        sim._env.close.assert_called_once()

    def test_step_wraps_step_result(self) -> None:
        sim = self._make_sim()
        sim._env.step.return_value = (
            {"agent_pos": np.zeros(4, dtype=np.float32)},
            1.0,
            True,
            False,
            {},
        )
        result = sim.step(np.zeros(4, dtype=np.float32))
        assert result.reward == 1.0
        assert result.terminated is True
        assert result.truncated is False

    def test_reset_calls_env_reset(self) -> None:
        sim = self._make_sim()
        sim._env.reset.return_value = ({"agent_pos": np.zeros(4, dtype=np.float32)}, {})
        sim.reset(seed=7)
        sim._env.reset.assert_called_once_with(seed=7)


# ── smolvla helpers ──────────────────────────────────────────────────────────


def _spec(weights_uri: str, device: str = "cpu", **extra: Any) -> VLASpec:
    return VLASpec(id="smolvla", weights_uri=weights_uri, device=device, extra=extra)


class TestResolveDevice:
    def test_explicit_device_returned_as_is(self) -> None:
        from openral_rskill._vla_core import resolve_device

        assert resolve_device(_spec("hf://x", device="cpu")) == "cpu"
        assert resolve_device(_spec("hf://x", device="cuda:1")) == "cuda:1"

    def test_auto_falls_back_to_cpu_without_torch(self) -> None:
        from openral_rskill._vla_core import resolve_device

        with patch.dict("sys.modules", {"torch": None}):
            assert resolve_device(_spec("hf://x", device="auto")) == "cpu"

    def test_auto_picks_cuda_when_available(self) -> None:
        from openral_rskill._vla_core import resolve_device

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": fake_torch}):
            assert resolve_device(_spec("hf://x", device="auto")) == "cuda:0"

    def test_auto_picks_mps_when_cuda_unavailable(self) -> None:
        from openral_rskill._vla_core import resolve_device

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": fake_torch}):
            assert resolve_device(_spec("hf://x", device="auto")) == "mps"

    def test_auto_falls_back_to_cpu_when_neither_accelerator(self) -> None:
        from openral_rskill._vla_core import resolve_device

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": fake_torch}):
            assert resolve_device(_spec("hf://x", device="auto")) == "cpu"


class TestResolveRepoId:
    def test_bare_ref_resolves_via_loader(self) -> None:
        from openral_rskill import loader
        from openral_rskill._vla_core import resolve_rskill_repo_id

        with patch.object(loader, "resolve_rskill_to_hf", return_value="org/repo"):
            assert resolve_rskill_repo_id("my-skill", adapter_name="SmolVLA") == "org/repo"

    def test_hf_scheme_rejected(self) -> None:
        from openral_rskill._vla_core import resolve_rskill_repo_id

        with pytest.raises(ROSConfigError, match="hf://"):
            resolve_rskill_repo_id("hf://lerobot/smolvla", adapter_name="SmolVLA")

    def test_file_scheme_rejected(self) -> None:
        from openral_rskill._vla_core import resolve_rskill_repo_id

        with pytest.raises(ROSConfigError, match="file://"):
            resolve_rskill_repo_id("file:///tmp/model", adapter_name="SmolVLA")


class TestSmolVLAAdapterBuildBatch:
    def _make_adapter(self, **overrides: Any) -> smolvla_mod._SmolVLAAdapter:
        import torch

        kwargs: dict[str, Any] = {
            "spec": _spec("hf://x"),
            "device": "cpu",
            "_policy": MagicMock(),
            "_preprocessor": MagicMock(side_effect=lambda b: b),
            "_postprocessor": MagicMock(side_effect=lambda t: t),
            "_torch": torch,
            "_flip_images_180": False,
        }
        kwargs.update(overrides)
        return smolvla_mod._SmolVLAAdapter(**kwargs)

    def test_build_batch_default_alias_is_passthrough(self) -> None:
        """Default ``_cam_alias`` is empty — keys flow through unchanged.

        After the ``feat(skill): _vla_core resolvers (no auto-derive)``
        commit, the historical LIBERO-only default ``{camera1: image,
        camera2: image2}`` is gone from the dataclass. That mapping is
        now ONLY applied when the rSkill manifest's
        ``image_preprocessing.aliases`` field declares it (see
        ``rskills/{pi05,smolvla}-libero/rskill.yaml``).
        """
        adapter = self._make_adapter()
        batch = adapter._build_batch(
            {
                "images": {"camera1": np.zeros((4, 4, 3), dtype=np.uint8)},
                "state": np.zeros(8, dtype=np.float32),
            },
            "do thing",
        )
        assert "observation.images.camera1" in batch
        assert "observation.images.image" not in batch
        assert batch["task"] == "do thing"

    def test_build_batch_applies_manifest_aliases(self) -> None:
        """When ``_cam_alias`` is populated (manifest path), keys are renamed."""
        adapter = self._make_adapter(_cam_alias={"camera1": "image", "camera2": "image2"})
        batch = adapter._build_batch(
            {
                "images": {"camera1": np.zeros((4, 4, 3), dtype=np.uint8)},
                "state": np.zeros(8, dtype=np.float32),
            },
            "do thing",
        )
        assert "observation.images.image" in batch

    def test_build_batch_pads_state_dim(self) -> None:
        adapter = self._make_adapter(_state_dim=10)
        batch = adapter._build_batch(
            {"images": {}, "state": np.ones(6, dtype=np.float32)},
            "x",
        )
        assert batch["observation.state"].shape == (1, 10)

    def test_build_batch_truncates_state_dim(self) -> None:
        adapter = self._make_adapter(_state_dim=4)
        batch = adapter._build_batch(
            {"images": {}, "state": np.ones(8, dtype=np.float32)},
            "x",
        )
        assert batch["observation.state"].shape == (1, 4)

    def test_build_batch_falls_back_to_obs_task(self) -> None:
        adapter = self._make_adapter()
        batch = adapter._build_batch({"images": {}, "task": "from_obs"}, "")
        assert batch["task"] == "from_obs"

    def test_reset_delegates_when_policy_has_reset(self) -> None:
        policy = MagicMock()
        adapter = self._make_adapter(_policy=policy)
        adapter.reset()
        policy.reset.assert_called_once()

    def test_reset_no_op_without_policy_reset(self) -> None:
        # An object without a 'reset' attribute must not raise.
        class _NoReset:
            pass

        adapter = self._make_adapter(_policy=_NoReset())
        adapter.reset()  # must be a no-op

    def test_close_on_cuda_calls_empty_cache(self) -> None:
        fake_torch = MagicMock()
        adapter = self._make_adapter(device="cuda:0", _torch=fake_torch)
        adapter.close()
        fake_torch.cuda.empty_cache.assert_called_once()

    def test_close_on_cpu_does_not_call_cuda(self) -> None:
        fake_torch = MagicMock()
        adapter = self._make_adapter(device="cpu", _torch=fake_torch)
        adapter.close()
        fake_torch.cuda.empty_cache.assert_not_called()


class TestSmolvlaBuilderWithoutBackend:
    def test_missing_lerobot_raises_actionable_message(self) -> None:
        env_cfg = SimEnvironment(
            robot_id="franka",
            scene=SceneSpec(id="mock", backend=PhysicsBackend.MOCK),
            task=TaskSpec(id="mock/0", scene_id="mock", instruction="x", max_steps=1),
            vla=VLASpec(id="smolvla", weights_uri="hf://x"),
        )
        with (
            patch.dict(
                "sys.modules",
                {
                    "lerobot": None,
                    "lerobot.policies": None,
                    "lerobot.policies.factory": None,
                    "lerobot.policies.smolvla": None,
                    "lerobot.policies.smolvla.modeling_smolvla": None,
                },
            ),
            pytest.raises(ROSConfigError, match="SmolVLA adapter requires"),
        ):
            smolvla_mod._build_smolvla(env_cfg)
