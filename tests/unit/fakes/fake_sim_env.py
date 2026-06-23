"""SimRollout-shaped fake for `SimAttachedHAL` unit tests.

CLAUDE.md §1.11 boundary double — the
:class:`openral_sim.rollout.SimRollout` Protocol is a process boundary
between the HAL layer and the simulator layer, so a narrow fake at
that boundary is allowed.

The fake implements the **subset** of SimRollout that
:class:`openral_hal.sim_attached.SimAttachedHAL` actually consumes:

* ``reset(seed=...)`` — returns an empty obs dict; records the seed.
* ``step(action)`` — records the action vector; returns a step result
  with an empty observation.
* ``action_dim: int`` — surface for
  :meth:`SimAttachedHAL._probe_env_action_dim`.
* ``mujoco_handles() -> tuple | None`` — pluggable so tests can
  exercise both the "MJCF reachable" and "MJCF unreachable" branches
  of :meth:`SimAttachedHAL.read_state`.

The real :class:`openral_sim.rollout.SimRollout` Protocol carries
additional methods (``render``, ``close``, ``task``, metrics
accessors). The fake does NOT implement those because
``SimAttachedHAL`` never calls them. If a future test or refactor
needs the wider surface, extend this fake — don't reach into the
real backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["FakeSimEnv", "FakeStepResult"]


@dataclass
class FakeStepResult:
    """Minimal step-result shape — only ``observation`` is consumed."""

    observation: dict[str, Any]


@dataclass
class FakeSimEnv:
    """In-test recorder satisfying the SimRollout subset SimAttachedHAL uses.

    Attributes:
        action_dim: The env's flat action dimensionality.
            :meth:`SimAttachedHAL._probe_env_action_dim` reads this on
            ``connect``.
        last_action: The action vector from the most recent
            :meth:`step` call. ``None`` until the first step.
        reset_calls: List of seeds passed to :meth:`reset`. Lets tests
            assert "connect was called with the right seed" without
            poking at private state.
        step_calls: Counter of :meth:`step` invocations.
        handles: The tuple :meth:`mujoco_handles` returns. Tests vary
            this to exercise both branches of
            :meth:`SimAttachedHAL.read_state`.
    """

    action_dim: int = 11
    last_action: np.ndarray[Any, np.dtype[np.float32]] | None = None
    reset_calls: list[int | None] = field(default_factory=list)
    step_calls: int = 0
    handles: tuple[Any, Any] | None = None
    # Opt-in: emit a ``raw_proprio`` block (``robot0_base_pos`` +
    # ``robot0_base_quat``) derived from the live base qpos, mirroring
    # what the real RoboCasa backend's observable produces. Lets tests
    # exercise the ``base_pose_6dof`` (``/odom`` source) path — including
    # the BODY_TWIST refresh that keeps it from going stale. Off by
    # default so existing tests see no ``refresh_obs`` effect (and
    # ``step_calls`` stays 0 after a BODY_TWIST).
    emit_proprio: bool = False
    base_joint_names: tuple[str, str, str] | None = None
    base_z: float = 0.7
    # ADR-0048 Phase 1 — opt-in fake sim clock for ``sim_time_ns`` tests.
    #   * ``has_sim_clock=True`` → :meth:`sim_time_ns` returns a per-episode
    #     elapsed-time counter that advances by ``sim_dt_ns`` on every ``step``
    #     and REWINDS to 0 on every ``reset`` (modelling robocasa's MjData.time
    #     reset). This exercises ``SimAttachedHAL``'s cross-reset offset.
    #   * ``has_sim_clock=False`` (default) → the method is still defined but
    #     returns ``None`` (clock-less backend) so the fake doubles as the
    #     "no sim clock" case without a second fake class.
    has_sim_clock: bool = False
    sim_time_from_mujoco: bool = False
    sim_dt_ns: int = 20_000_000  # 20 ms — a typical robosuite control step
    _sim_time_ns: int = 0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset stub — returns an empty Observation-shaped dict."""
        self.reset_calls.append(seed)
        # Model the backend rewinding its physics clock to 0 on reset.
        self._sim_time_ns = 0
        obs: dict[str, Any] = {"images": {}, "state": np.zeros(0, dtype=np.float32), "task": ""}
        proprio = self._proprio_from_qpos()
        if proprio is not None:
            obs["raw_proprio"] = proprio
        return obs

    def step(self, action: np.ndarray[Any, np.dtype[np.float32]]) -> FakeStepResult:
        """Step stub — records the action; returns an empty observation."""
        self.last_action = np.asarray(action, dtype=np.float32)
        self.step_calls += 1
        self._sim_time_ns += self.sim_dt_ns
        return FakeStepResult(observation={"images": {}, "state": np.zeros(0, dtype=np.float32)})

    def sim_time_ns(self) -> int | None:
        """Per-episode elapsed sim time in ns, or ``None`` when clock-less.

        Models the :class:`openral_sim.rollout.SimRollout` contract: monotonic
        within an episode, rewinds on ``reset``. ``None`` when
        ``has_sim_clock`` is ``False`` (the clock-less-backend case).
        """
        if self.has_sim_clock and self.sim_time_from_mujoco and self.handles is not None:
            _model, data = self.handles
            return round(float(data.time) * 1_000_000_000)
        return self._sim_time_ns if self.has_sim_clock else None

    def refresh_obs(self) -> dict[str, Any] | None:
        """Return a fresh Observation with ``raw_proprio`` from live qpos.

        Models the real backend's contract: after a direct base-qpos
        write, ``SimAttachedHAL`` calls ``refresh_obs`` to recompute the
        authoritative pose. Returns ``None`` when proprio emission is
        off (then the HAL keeps its cached obs — the pre-fix behaviour
        that froze ``/odom``). Does NOT call :meth:`step`, so tests that
        assert ``step_calls == 0`` after a BODY_TWIST stay valid.
        """
        proprio = self._proprio_from_qpos()
        if proprio is None:
            return None
        return {"images": {}, "state": np.zeros(0, dtype=np.float32), "raw_proprio": proprio}

    def _proprio_from_qpos(self) -> dict[str, Any] | None:
        """Compute ``robot0_base_pos`` / ``robot0_base_quat`` from base qpos."""
        if not self.emit_proprio or self.handles is None or self.base_joint_names is None:
            return None
        import math

        import mujoco

        model, data = self.handles
        addrs = [
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in self.base_joint_names
        ]
        x, y, yaw = (float(data.qpos[a]) for a in addrs)
        return {
            "robot0_base_pos": np.array([x, y, self.base_z], dtype=np.float64),
            "robot0_base_quat": np.array(
                [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], dtype=np.float64
            ),
        }

    def mujoco_handles(self) -> tuple[Any, Any] | None:
        """Return whatever was configured on the dataclass field."""
        return self.handles
