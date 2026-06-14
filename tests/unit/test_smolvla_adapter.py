"""Unit tests for SmolVLAAdapter, SO100SmolVLASkill, and ChunkedExecutor.

All tests run without a GPU, without lerobot, and without network access.
A ``NullPolicy`` stub stands in for ``SmolVLAPolicy``, satisfying the same
interface that ``ChunkedExecutor`` and ``SmolVLAAdapter`` require.

Test strategy
-------------
- ``TestChunkedExecutor``: exercises the state machine (reset, prefetch
  trigger, step counting) using a NullPolicy whose ``select_action`` is
  instrumented to count calls.  Asserts the prefetch fires exactly when
  expected and that background errors propagate cleanly.
- ``TestSmolVLAAdapterLifecycle``: exercises the full Skill lifecycle
  (configure → activate → step → deactivate → shutdown) by monkey-patching
  the lerobot imports so ``on_load_weights`` doesn't hit the network.
- ``TestSO100SmolVLASkill``: smoke test for the convenience subclass;
  asserts the correct name and embodiment_tags are set by default.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_core.schemas import (
    Action,
    ControlMode,
    JointState,
    RSkillState,
    WorldState,
)
from openral_rskill.smolvla import (
    ChunkedExecutor,
    SmolVLAAdapter,
    SO100SmolVLASkill,
)

# ── Fixtures / helpers ────────────────────────────────────────────────────────


class _FakeConfig:
    """Minimal stand-in for SmolVLAPolicy.config."""

    def __init__(self, chunk_size: int = 10, n_dof: int = 6) -> None:
        self.chunk_size = chunk_size
        self.n_action_steps = chunk_size
        self.n_dof = n_dof
        self.input_features = {
            "observation.state": MagicMock(shape=(n_dof,)),
            "observation.images.camera1": MagicMock(shape=(3, 256, 256)),
        }
        self.output_features = {
            "action": MagicMock(shape=(n_dof,)),
        }


class _NullPolicy:
    """Minimal lerobot policy stub for unit tests.

    ``select_action`` returns a (1, n_dof) zero tensor and increments an
    internal call counter.  A ``chunk_size``-step internal queue is simulated
    so :class:`ChunkedExecutor` behaves correctly.
    """

    def __init__(self, chunk_size: int = 10, n_dof: int = 6) -> None:
        self.config = _FakeConfig(chunk_size=chunk_size, n_dof=n_dof)
        self._call_count = 0
        self._queue: int = 0  # remaining steps in the current chunk

    def reset(self) -> None:
        """Reset internal queue (simulates policy.reset())."""
        self._queue = 0

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        """Return a zero action tensor; simulate internal queue depletion.

        Args:
            batch: Ignored (no real inference performed).

        Returns:
            (1, n_dof) float32 zero tensor.
        """
        self._call_count += 1
        if self._queue == 0:
            self._queue = self.config.n_action_steps
        self._queue -= 1
        return torch.zeros(1, self.config.n_dof)  # shape: (1, n_dof)


def _make_world_state(n_joints: int = 6) -> WorldState:
    """Construct a minimal WorldState with ``n_joints`` zero-position joints."""
    return WorldState(
        stamp_ns=time.time_ns(),
        joint_state=JointState(
            name=[f"j{i}" for i in range(n_joints)],
            position=[0.0] * n_joints,
            stamp_ns=time.time_ns(),
        ),
    )


def _dummy_obs_fn(device: str = "cpu") -> Callable[[WorldState], dict[str, Any]]:
    """Return a simple obs_fn that builds a minimal SmolVLA batch on CPU."""

    def obs_fn(ws: WorldState) -> dict[str, Any]:
        state = torch.tensor(ws.joint_state.position[:6], dtype=torch.float32).unsqueeze(0)
        img = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
        return {
            "observation.state": state,
            "observation.images.camera1": img,
            "task": ["pick the cube"],
        }

    return obs_fn


def _make_patched_adapter(
    policy: _NullPolicy | None = None,
    device: str = "cpu",
    n_dof: int = 6,
) -> SmolVLAAdapter:
    """Build a SmolVLAAdapter with lerobot patched out.

    Returns the adapter with ``_policy`` and ``_preprocessor`` already set so
    ``configure()`` → ``activate()`` → ``step()`` can run without network or GPU.
    """
    if policy is None:
        policy = _NullPolicy(chunk_size=10, n_dof=n_dof)

    def _noop_preprocessor(batch: dict[str, Any]) -> dict[str, Any]:
        return batch

    adapter = SmolVLAAdapter(
        repo_id="fake/smolvla",
        obs_fn=_dummy_obs_fn(device),
        prompt="pick the cube",
        device=device,
        n_dof=n_dof,
        prefetch_at=2,  # small for test speed
    )
    # Bypass on_load_weights by pre-injecting the stub.
    adapter._policy = policy
    adapter._preprocessor = _noop_preprocessor

    return adapter


# ── ChunkedExecutor tests ─────────────────────────────────────────────────────


class TestChunkedExecutor:
    def test_first_call_hits_policy(self) -> None:
        """The very first select_action call must invoke the policy (no cache)."""
        p = _NullPolicy(chunk_size=5)
        ex = ChunkedExecutor(p, prefetch_at=2)
        ex.start()
        batch: dict[str, Any] = {}
        ex.select_action(batch)
        assert p._call_count == 1

    def test_subsequent_calls_hit_policy_queue(self) -> None:
        """Calls 2..chunk_size must drain the internal queue (call_count stays at 1).

        We use prefetch_at=chunk_size+1 so the background pre-fetch never fires
        within the tested range, giving us a clean call-count check.
        """
        chunk_size = 8
        p = _NullPolicy(chunk_size=chunk_size)
        # prefetch_at > chunk_size disables prefetching in this window.
        ex = ChunkedExecutor(p, prefetch_at=chunk_size + 1)
        ex.start()
        batch: dict[str, Any] = {}
        for _ in range(chunk_size - 1):  # steps 1..7, no prefetch triggered
            ex.select_action(batch)
        # Each step calls select_action exactly once — no extra background calls.
        assert p._call_count == chunk_size - 1

    def test_reset_clears_state(self) -> None:
        """reset() must set step_in_chunk back to 0 and call policy.reset()."""
        p = _NullPolicy(chunk_size=5)
        ex = ChunkedExecutor(p, prefetch_at=1)
        ex.start()
        ex.select_action({})
        ex.select_action({})
        assert ex._step_in_chunk == 2
        ex.reset()
        assert ex._step_in_chunk == 0

    def test_stop_is_idempotent(self) -> None:
        """stop() must be callable multiple times without error."""
        p = _NullPolicy()
        ex = ChunkedExecutor(p)
        ex.start()
        ex.stop()
        ex.stop()  # second stop must not raise

    def test_background_error_propagates_as_ros_runtime_error(self) -> None:
        """If the pre-fetch thread raises, the next foreground call must re-raise."""
        chunk_size = 4
        p = _NullPolicy(chunk_size=chunk_size)
        ex = ChunkedExecutor(p, prefetch_at=2)
        ex.start()
        batch: dict[str, Any] = {}

        # Drain through step chunk_size - prefetch_at to trigger the BG thread.
        for _ in range(chunk_size - 2):
            ex.select_action(batch)

        # Inject a pre-baked error into the executor state (simulates BG failure).
        ex._bg_event.set()
        ex._bg_error = ValueError("simulated GPU OOM")
        ex._step_in_chunk = chunk_size  # exhaust the chunk so next call reads BG result

        with pytest.raises(ROSRuntimeError, match="simulated GPU OOM"):
            ex.select_action(batch)

    def test_prefetch_triggers_background_thread(self) -> None:
        """A background thread must be launched when remaining steps == prefetch_at."""
        chunk_size = 6
        prefetch_at = 2
        p = _NullPolicy(chunk_size=chunk_size)
        ex = ChunkedExecutor(p, prefetch_at=prefetch_at)
        ex.start()
        batch: dict[str, Any] = {}

        # Advance to the trigger point (chunk_size - prefetch_at steps drained).
        trigger_step = chunk_size - prefetch_at
        for _ in range(trigger_step):
            ex.select_action(batch)

        # Allow the daemon thread to start.
        time.sleep(0.05)
        assert ex._bg_thread is not None
        ex.stop()


# ── SmolVLAAdapter lifecycle tests ────────────────────────────────────────────


class TestSmolVLAAdapterLifecycle:
    def test_initial_state_is_unconfigured(self) -> None:
        """Adapter must start in UNCONFIGURED state before configure()."""
        adapter = _make_patched_adapter()
        assert adapter.state is RSkillState.UNCONFIGURED

    def test_configure_transitions_to_inactive(self) -> None:
        """configure() must set state=INACTIVE after calling on_load_weights etc."""
        adapter = _make_patched_adapter()
        # Bypass on_load_weights (already pre-set) — patch hooks that would fail.
        with patch.object(adapter, "on_load_weights"), patch.object(adapter, "on_quantize"):
            adapter.configure()
        assert adapter.state is RSkillState.INACTIVE
        assert adapter.info.weights_loaded is True

    def test_activate_transitions_to_active_and_starts_executor(self) -> None:
        """activate() must set state=ACTIVE and create a ChunkedExecutor."""
        adapter = _make_patched_adapter()
        with (
            patch.object(adapter, "on_load_weights"),
            patch.object(adapter, "on_quantize"),
            patch.object(adapter, "on_warmup"),
        ):
            adapter.configure()
            adapter.activate()
        assert adapter.state is RSkillState.ACTIVE
        assert adapter._executor is not None

    def test_step_returns_joint_position_action(self) -> None:
        """step() must return an Action with JOINT_POSITION control mode and horizon=1."""
        adapter = _make_patched_adapter(n_dof=6)
        with (
            patch.object(adapter, "on_load_weights"),
            patch.object(adapter, "on_quantize"),
            patch.object(adapter, "on_warmup"),
        ):
            adapter.configure()
            adapter.activate()

        ws = _make_world_state(n_joints=6)
        action = adapter.step(ws)

        assert isinstance(action, Action)
        assert action.control_mode is ControlMode.JOINT_POSITION
        assert action.horizon == 1
        assert action.joint_targets is not None
        assert len(action.joint_targets) == 1
        assert len(action.joint_targets[0]) == 6

    def test_step_before_activate_raises(self) -> None:
        """step() before activate() must raise ROSRuntimeError."""
        adapter = _make_patched_adapter()
        ws = _make_world_state()
        with pytest.raises(ROSRuntimeError, match="must be 'active'"):
            adapter.step(ws)

    def test_deactivate_stops_executor(self) -> None:
        """deactivate() must stop the ChunkedExecutor thread."""
        adapter = _make_patched_adapter()
        with (
            patch.object(adapter, "on_load_weights"),
            patch.object(adapter, "on_quantize"),
            patch.object(adapter, "on_warmup"),
        ):
            adapter.configure()
            adapter.activate()

        assert adapter._executor is not None
        adapter.deactivate()
        assert adapter.state is RSkillState.INACTIVE

    def test_shutdown_releases_policy(self) -> None:
        """shutdown() must set _policy to None (releases GPU reference)."""
        adapter = _make_patched_adapter()
        with (
            patch.object(adapter, "on_load_weights"),
            patch.object(adapter, "on_quantize"),
            patch.object(adapter, "on_warmup"),
        ):
            adapter.configure()
            adapter.activate()

        adapter.shutdown()
        assert adapter.state is RSkillState.FINALIZED
        assert adapter._policy is None

    def test_shutdown_is_idempotent(self) -> None:
        """Calling shutdown() twice must not raise."""
        adapter = _make_patched_adapter()
        with patch.object(adapter, "on_load_weights"), patch.object(adapter, "on_quantize"):
            adapter.configure()

        adapter.shutdown()
        adapter.shutdown()  # second call must be a no-op

    def test_configure_errors_enter_error_state(self) -> None:
        """If _configure_impl raises, state must be ERROR."""
        adapter = _make_patched_adapter(n_dof=6)
        # Force n_dof mismatch: policy reports shape (14,) but adapter expects 6.
        adapter._policy.config.output_features["action"].shape = (14,)
        with (
            patch.object(adapter, "on_load_weights"),
            patch.object(adapter, "on_quantize"),
            pytest.raises(ROSConfigError, match="does not match n_dof"),
        ):
            adapter.configure()
        assert adapter.state is RSkillState.ERROR

    def test_on_load_weights_raises_ros_config_error_without_lerobot(self) -> None:
        """on_load_weights must raise ROSConfigError when lerobot is not installed."""
        adapter = SmolVLAAdapter(
            repo_id="fake/smolvla",
            obs_fn=_dummy_obs_fn(),
            prompt="test",
            device="cpu",
        )
        # Patch the import inside on_load_weights to simulate missing lerobot.
        with (
            patch.dict(
                "sys.modules",
                {
                    "lerobot": None,
                    "lerobot.policies": None,
                    "lerobot.policies.smolvla": None,
                    "lerobot.policies.smolvla.modeling_smolvla": None,
                    "lerobot.policies.factory": None,
                },
            ),
            pytest.raises(ROSConfigError, match="requires 'lerobot'"),
        ):
            adapter.on_load_weights()

    def test_multiple_steps_cycle_through_chunk(self) -> None:
        """Running chunk_size steps must cycle through the internal queue."""
        chunk_size = 5
        policy = _NullPolicy(chunk_size=chunk_size)
        adapter = _make_patched_adapter(policy=policy, n_dof=6)
        with (
            patch.object(adapter, "on_load_weights"),
            patch.object(adapter, "on_quantize"),
            patch.object(adapter, "on_warmup"),
        ):
            adapter.configure()
            adapter.activate()

        ws = _make_world_state()
        actions = [adapter.step(ws) for _ in range(chunk_size)]
        assert len(actions) == chunk_size
        for a in actions:
            assert a.control_mode is ControlMode.JOINT_POSITION


# ── SO100SmolVLASkill tests ───────────────────────────────────────────────────


class TestSO100SmolVLASkill:
    def test_default_name_and_embodiment_tags(self) -> None:
        """SO100SmolVLASkill must default to name='smolvla_so100' and SO-100 tags."""
        skill = SO100SmolVLASkill(prompt="test", device="cpu")
        assert skill.name == "smolvla_so100"
        assert "so100_follower" in skill.info.embodiment_tags

    def test_custom_name_override(self) -> None:
        """Passing name= kwarg must override the default."""
        skill = SO100SmolVLASkill(prompt="test", device="cpu", name="my_skill")
        assert skill.name == "my_skill"

    def test_starts_unconfigured(self) -> None:
        """A freshly constructed SO100SmolVLASkill must be in UNCONFIGURED state."""
        skill = SO100SmolVLASkill(prompt="test", device="cpu")
        assert skill.state is RSkillState.UNCONFIGURED

    def test_obs_fn_builds_valid_batch(self) -> None:
        """The internal obs_fn must produce a dict with the expected keys."""
        skill = SO100SmolVLASkill(prompt="pick up the cube", device="cpu")
        ws = _make_world_state(n_joints=6)
        batch = skill._obs_fn(ws)
        assert "observation.state" in batch
        assert "observation.images.camera1" in batch
        assert batch["task"] == ["pick up the cube"]
        assert batch["observation.state"].shape == (1, 6)
