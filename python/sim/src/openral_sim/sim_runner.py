"""SimRunner — per-step :class:`InferenceRunner` for the simulation runtime.

Sim and hardware share one tick semantic — one inference step per tick.
:class:`DeployRunner` ticks at e.g. 30 Hz on a real robot; :class:`SimRunner`
ticks as fast as the env + policy let it, with each tick advancing one
``env.step``. Episodes are a derived view: SimRunner accumulates per-tick
data into a private buffer and emits an :class:`EpisodeResult` whenever the
env terminates / truncates or the per-episode step budget is reached.
"""

from __future__ import annotations

import contextlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import DeadlineOverrunPolicy, TickResult
from openral_core.exceptions import ROSConfigError
from openral_observability import metrics as ral_metrics
from openral_observability import semconv
from openral_runner.base import InferenceRunnerBase
from opentelemetry import trace

from openral_sim.factory import make_env, make_policy
from openral_sim.rollout import EpisodeResult

if TYPE_CHECKING:
    from openral_core import RSkillManifest, SimEnvironment
    from openral_dataset import RolloutRecorder

    from openral_sim.policy import PolicyAdapter
    from openral_sim.rollout import Observation, SimRollout


__all__ = ["SimRunner"]


def _tracer() -> trace.Tracer:
    # Per-call resolution — caching at module import binds to whichever
    # TracerProvider was global at import time and silently swallows
    # spans after a provider swap (matches the world-state aggregator
    # fix in the OTel rollout branch).
    return trace.get_tracer("openral_sim.sim_runner")


_log = structlog.get_logger(__name__)
_VIDEO_FRAMES_INFO_KEY = "_openral_video_frames"
_IMAGE_NDIM = 3
_GRAYSCALE_CHANNELS = 1
_RGB_CHANNELS = 3
_RGBA_CHANNELS = 4


def _append_adapter_video_frames(
    target: list[NDArray[np.uint8]],
    payload: object,
) -> None:
    """Append adapter-provided intra-step frames to the episode video buffer."""
    if payload is None:
        return
    if not isinstance(payload, list):
        _log.warning(
            "adapter_video_frames_invalid",
            reason="payload_not_list",
            payload_type=type(payload).__name__,
        )
        return
    for frame in payload:
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != _IMAGE_NDIM or arr.shape[2] not in (
            _GRAYSCALE_CHANNELS,
            _RGB_CHANNELS,
            _RGBA_CHANNELS,
        ):
            _log.warning(
                "adapter_video_frame_invalid",
                shape=tuple(arr.shape),
                dtype=str(arr.dtype),
            )
            continue
        if arr.shape[2] == _GRAYSCALE_CHANNELS:
            arr = np.repeat(arr, _RGB_CHANNELS, axis=2)
        elif arr.shape[2] == _RGBA_CHANNELS:
            arr = arr[:, :, :_RGB_CHANNELS]
        target.append(np.ascontiguousarray(arr, dtype=np.uint8))


def _dump_obs_for_step(
    *,
    tick: int,
    obs: Any,
    raw_policy_action: Any,
    prompt: str,
) -> None:
    """Debug-only side-by-side dump matching rskill_runner_node's pickle.

    Activated by ``OPENRAL_DUMP_OBS_TICK`` (comma-separated tick indices,
    1-based to match the skill_runner log). Output dir from
    ``OPENRAL_DUMP_OBS_PATH`` (defaults to ``/tmp/openral_obs_dump``).
    Writes ``simrun_tick<NN>.pkl`` + per-camera ``simrun_tick<NN>_camera<key>.npy``
    so a ``deploy_sim`` pickle (named ``<rskill>_tickNN.pkl``) and a
    ``sim run`` pickle for the same tick can be diffed directly.

    Same shape as ``_PolicyAdapterSkill._dump_obs_to_disk`` in
    ``packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py``.
    Failures swallowed — never load-bearing.
    """
    raw = os.environ.get("OPENRAL_DUMP_OBS_TICK", "").strip()
    if not raw:
        return
    ticks = {int(t) for t in raw.split(",") if t.strip().isdigit()}
    if tick not in ticks:
        return
    import pickle  # reason: defer; debug-only
    from pathlib import Path

    try:
        root = Path(os.environ.get("OPENRAL_DUMP_OBS_PATH", "/tmp/openral_obs_dump"))
        root.mkdir(parents=True, exist_ok=True)
        stem = f"simrun_tick{tick:04d}"
        images = obs.get("images") if isinstance(obs, dict) else None
        image_shapes: dict[str, tuple[int, ...]] = {}
        if isinstance(images, dict):
            for k, v in images.items():
                arr = np.asarray(v)
                image_shapes[str(k)] = tuple(arr.shape)
                np.save(root / f"{stem}_camera{k}.npy", arr)
        state = obs.get("state") if isinstance(obs, dict) else None
        payload = {
            "tick": tick,
            "source": "sim_run",
            "prompt": prompt,
            "obs_state": np.asarray(state) if state is not None else None,
            "raw_policy_action": np.asarray(raw_policy_action),
            "image_keys": sorted(image_shapes.keys()),
            "image_shapes": image_shapes,
        }
        with (root / f"{stem}.pkl").open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[sim_runner] obs_dump tick={tick} wrote {root / f'{stem}.pkl'} + "
            f"{len(image_shapes)} camera npy(s)",
            flush=True,
        )
    except Exception as exc:  # reason: debug dump never load-bearing
        print(f"[sim_runner] obs_dump tick={tick} failed: {exc!r}", flush=True)


def _resolve_step_instruction(
    *,
    instruction_override: str | None,
    obs_task: object,
    task_instruction: str,
) -> str:
    """Pick the natural-language instruction handed to the policy each step.

    Three candidate sources, in strict precedence order:

    1. ``instruction_override`` — the user's explicit ``--instruction`` CLI
       flag (``None`` when not passed). An explicit override ALWAYS wins,
       even over a scene's per-episode language. ``--instruction`` was
       previously silent on scenes that publish ``obs["task"]`` (for example,
       RoboCasa sampled-object language), so the
       robot ignored it (CLAUDE.md §1.4 — explicit beats implicit).
    2. ``obs["task"]`` — the env's per-episode language, when the scene
       adapter populated it with a non-empty ``str``. RoboCasa interpolates
       the *sampled* object name here, so it is genuinely more correct than
       the static YAML for those scenes.
    3. ``task_instruction`` — the static YAML ``task.instruction`` fallback.

    Args:
        instruction_override: Explicit CLI override, or ``None``.
        obs_task: ``obs.get("task")``; only honoured when a non-blank ``str``.
        task_instruction: The static ``TaskSpec.instruction`` fallback.

    Returns:
        The instruction string to prompt the policy with this step.

    Example:
        >>> _resolve_step_instruction(
        ...     instruction_override="pick the orange juice",
        ...     obs_task="pick the milk",
        ...     task_instruction="",
        ... )
        'pick the orange juice'
    """
    if instruction_override is not None and instruction_override.strip():
        return instruction_override
    if isinstance(obs_task, str) and obs_task.strip():
        return obs_task
    return task_instruction


def _count_policy_input_cameras(policy: object, env_cfg: SimEnvironment) -> int:
    """Best-effort count of distinct camera streams consumed by the policy."""
    camera_keys = getattr(policy, "_camera_keys", None)
    if isinstance(camera_keys, (list, tuple)) and camera_keys:
        return len(camera_keys)
    cam_keys = env_cfg.vla.extra.get("camera_keys")
    if isinstance(cam_keys, (list, tuple)) and cam_keys:
        return len(cam_keys)
    return 1


@dataclass
class _EpisodeBuffer:
    """Per-episode accumulation buffer used by :class:`SimRunner`.

    Mirrors the lists ``run_episode`` used to materialise locally, but as
    an instance owned by the runner so it survives between ``_tick_impl``
    calls. Reset on each episode boundary by :meth:`SimRunner._finalize_episode`.
    """

    latencies: list[float] = field(default_factory=list)
    frames: list[NDArray[np.uint8]] = field(default_factory=list)
    vla_input_frames: list[NDArray[np.uint8]] = field(default_factory=list)
    joint_positions: list[NDArray[np.float32]] = field(default_factory=list)
    actions: list[NDArray[np.float32]] = field(default_factory=list)
    total_reward: float = 0.0
    max_step_reward: float = float("-inf")
    success: bool = False
    steps_done: int = 0
    budget_violations: int = 0

    @property
    def has_data(self) -> bool:
        """True if at least one step has been accumulated."""
        return self.steps_done > 0


class SimRunner(InferenceRunnerBase):
    """One-tick = one-env-step :class:`InferenceRunner` for sim rollouts.

    Drives a single (robot × scene × task × VLA) :class:`SimEnvironment`
    for ``env_cfg.n_episodes`` episodes. Each call to :meth:`tick`
    advances by one tick of one of two flavours:

      * **Reset tick** — emitted at the start of each episode (and once
        more after a terminated / truncated step). Calls ``env.reset``
        and ``policy.reset``, finalises the previous :class:`EpisodeResult`
        if any, and returns a :class:`TickResult` with
        ``action_applied=False`` and ``inference_ms == 0.0``.
      * **Step tick** — runs ``policy.step`` + ``env.step``. Records
        latency, reward, ``terminated`` / ``truncated``, and (when
        ``env_cfg.record_video`` is set) the rendered frame, VLA input
        frame, joint state, and action. When the env signals termination
        the next tick will be a reset tick.

    :meth:`_should_terminate` returns ``True`` once ``n_episodes``
    EpisodeResults have been emitted, so callers can pass an upper-bound
    ``max_ticks`` (typically ``n_episodes * (max_steps + 1)``) and rely
    on the hook for the real stop condition.

    Args:
        env_cfg: Validated :class:`openral_core.SimEnvironment`.
        view: Open a passive ``mujoco.viewer`` window during the rollout.
            Only valid for MuJoCo-backed scenes that expose
            ``mujoco_handles()``.
        strict_view: When ``view`` is True and the env adapter does not
            expose ``mujoco_handles()``, raise :class:`ROSConfigError`
            instead of warning and continuing offscreen.
        deadline_overrun_policy: Forwarded to the base class. Defaults to
            WARN since sim rollouts are not real-time.

    Attributes:
        episode_results: One :class:`EpisodeResult` per completed episode,
            populated as ticks complete.
        manifest: The validated rSkill manifest, or ``None`` for mock
            policies that carry no weights.
    """

    episode_results: list[EpisodeResult]
    manifest: RSkillManifest | None

    def __init__(
        self,
        env_cfg: SimEnvironment,
        *,
        view: bool = False,
        strict_view: bool = False,
        instruction_override: str | None = None,
        deadline_overrun_policy: DeadlineOverrunPolicy = DeadlineOverrunPolicy.WARN,
        recorder: RolloutRecorder | None = None,
    ) -> None:
        """Build the runner; defer env / policy construction to :meth:`activate`.

        Args:
            env_cfg: Validated :class:`openral_core.SimEnvironment`.
            view: Open a passive ``mujoco.viewer`` window during the rollout.
            strict_view: Raise on missing viewer handles instead of warning.
            instruction_override: An explicit ``--instruction`` CLI value that
                must win over a scene's per-episode ``obs["task"]`` language
                (for example, RoboCasa sampled object).
                ``None`` when the user passed nothing — the env/YAML language
                then takes over. See :func:`_resolve_step_instruction`.
            deadline_overrun_policy: Forwarded to the base class.
            recorder: ADR-0019 optional :class:`openral_dataset.RolloutRecorder`.
                When set, per-step state / images / action plus episode
                boundaries are fanned out to the recorder's sinks in
                addition to the existing :class:`_EpisodeBuffer`. The
                buffer drives the in-memory video / json / benchmark
                pipeline; the recorder drives the durable LeRobotDataset
                v3 path. The two are additive — the recorder is never
                a substitute for the buffer.
        """
        # Sim rollouts run as fast as policy + env permit; rate_hz only
        # parameterises the base class' deadline machinery. 1000 Hz keeps
        # the period at 1 ms so a single env.step never naturally
        # over-runs the deadline (deadline policy is WARN by default
        # anyway).
        super().__init__(
            rate_hz=1000.0,
            deadline_overrun_policy=deadline_overrun_policy,
            runner_name=f"sim_runner:{env_cfg.scene.id}:{env_cfg.task.id}",
            save_dir=env_cfg.save_dir,
        )
        self._env_cfg = env_cfg
        self._view = view
        self._strict_view = strict_view
        self._instruction_override = instruction_override
        self._recorder = recorder

        # Lifecycle state. Initialised properly in activate() so the runner
        # can be re-used across runs.
        self.episode_results = []
        self.manifest = None
        self._env: SimRollout | None = None
        self._policy: PolicyAdapter | None = None
        self._obs: Observation = {}
        self._viewer: Any = None
        # True when the scene adapter draws its own window inside env.step /
        # env.reset (e.g. gym_pusht with render_mode="human"). Suppresses
        # the lazy mujoco-viewer-open path in _reset_tick.
        self._intrinsic_viewer: bool = False
        self._frame_dt_s: float | None = None
        self._episode_idx = 0
        self._step_idx = 0
        self._needs_reset = True
        self._buf = _EpisodeBuffer()
        self._budget_ms: float | None = None
        self._run_span_ctx: Any = None
        # Tracks whether the current episode is currently "open" on the
        # recorder. The recorder requires an explicit episode_start before
        # record_frame and episode_end at the boundary; we mirror the
        # _EpisodeBuffer.has_data state here.
        self._recorder_episode_open: bool = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def activate(self) -> None:
        """Validate the manifest, build env + policy, and arm the first reset tick.

        ``make_env`` (MuJoCo XML compile / dataset prefetch — ~30–60 s on
        LIBERO / RoboCasa) and ``make_policy`` (PaliGemma 3.4B graph
        allocation + NF4 quantization — ~100–150 s on π0.5) are independent:
        both only read the immutable :class:`SimEnvironment` and share no
        mutable state. By default they run concurrently on a 2-worker
        ``ThreadPoolExecutor`` (the GIL is released by MuJoCo C calls and
        the torch / safetensors load path), so the wall-clock for
        :meth:`activate` collapses to ``max(env_ms, policy_ms)`` instead
        of their sum. See GH-134.

        Set ``OPENRAL_SIM_SEQUENTIAL_INIT=1`` to force the legacy
        sequential path (debugging, profiling, or when interleaved logs
        would obscure a root cause).

        Exceptions from either side propagate verbatim — no silent retry,
        no swallowed traceback. If the policy build raises while the env
        build is still running, the env future is left to finish on the
        worker thread (cancellation of a started future is a no-op in
        Python's executor model); :meth:`deactivate` is responsible for
        closing it if :meth:`activate` raised after a partial assignment.
        """
        # Validate before the (expensive) env / policy build so misconfigs
        # fail fast without burning GPU time.
        self.manifest, self._env_cfg = _check_rskill_compatibility(self._env_cfg)
        # SAPIEN-backed scenes (simpler-env / ManiSkill3 bridge envs)
        # need to know at gym.make() time whether to advertise
        # `render_mode='human'` — that toggle is what tells SAPIEN to
        # build a live viewer instead of an offscreen render target. The
        # SCENES factory only sees ``env_cfg``, so we publish the flag
        # through an env var scoped to the build window. MuJoCo-backed
        # adapters ignore the var (their viewer opens lazily from
        # ``mujoco_handles()`` on first reset, not at construct time).
        prev_view_env = os.environ.get(_VIEW_ENV)
        if self._view:
            os.environ[_VIEW_ENV] = "1"
        try:
            self._env, self._policy = _build_env_and_policy(self._env_cfg)
        finally:
            if prev_view_env is None:
                os.environ.pop(_VIEW_ENV, None)
            else:
                os.environ[_VIEW_ENV] = prev_view_env
        self._viewer = None
        self._intrinsic_viewer = False
        self._frame_dt_s = None

        # Scenes whose engine draws its own window (gym_pusht via
        # render_mode="human") expose enable_intrinsic_viewer(); for
        # those the mujoco-viewer path in _reset_tick is bypassed.
        if self._view:
            enable_fn = getattr(self._env, "enable_intrinsic_viewer", None)
            if callable(enable_fn):
                enable_fn()
                self._intrinsic_viewer = True
        self._episode_idx = 0
        self._step_idx = 0
        self._needs_reset = True
        self._buf = _EpisodeBuffer()
        self.episode_results = []
        self._budget_ms = (
            self.manifest.latency_budget.per_chunk_ms if self.manifest is not None else None
        )

        # Outer OTel span equivalent to the deleted run_evaluation's
        # ``eval.run_evaluation``. Held open until deactivate() so child
        # ``rskill.tick`` spans from the base class nest under it.
        self._run_span_ctx = _tracer().start_as_current_span(
            "sim.run",
            attributes={
                "robot.id": self._env_cfg.robot_id,
                "scene.id": self._env_cfg.scene.id,
                "task.id": self._env_cfg.task.id,
                "vla.id": self._env_cfg.vla.id,
                "vla.weights_uri": self._env_cfg.vla.weights_uri,
                "n_episodes": self._env_cfg.n_episodes,
                "seed": self._env_cfg.seed,
            },
        )
        self._run_span_ctx.__enter__()

        super().activate()

    def deactivate(self) -> None:
        """Finalise any trailing episode, close env + policy, idempotently."""
        # Flush partial episode (e.g. when max_ticks cut us short).
        if self._buf.has_data:
            self._finalize_episode()
        # ADR-0019: if an episode opened on the recorder never reached
        # _finalize_episode (e.g. activate() succeeded but no ticks ran),
        # close it as a failure before finalising the recorder.
        if self._recorder is not None:
            if self._recorder_episode_open:
                with contextlib.suppress(ValueError, RuntimeError):
                    self._recorder.episode_end(success=False)
                self._recorder_episode_open = False
            try:
                self._recorder.finalize()
            except (ValueError, RuntimeError) as exc:
                _log.warning(
                    "rollout_recorder_finalize_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._policy is not None:
            self._policy.close()
            self._policy = None
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._run_span_ctx is not None:
            self._run_span_ctx.__exit__(None, None, None)
            self._run_span_ctx = None
        super().deactivate()

    # ── Termination hook ────────────────────────────────────────────────────

    def _should_terminate(self) -> bool:
        """Stop once ``n_episodes`` EpisodeResults have been emitted."""
        return len(self.episode_results) >= self._env_cfg.n_episodes

    # ── ADR-0019 PR3 — episode boundary overrides (sim no-ops) ──────────────
    #
    # SimRunner derives episode boundaries from the env's terminated /
    # truncated flags inside _reset_tick / _finalize_episode; the
    # RolloutRecorder is already driven there. The explicit
    # episode_start / episode_end API exists for DeployRunner. We
    # override them as no-ops on the sim runner so a higher-level driver
    # (e.g. a BT executor) can call the same method on either runner
    # uniformly without dispatching on type.

    def episode_start(self, task_string: str) -> int:
        """No-op on SimRunner: sim derives episodes from env signals.

        Returns the current ``_episode_idx`` so callers expecting an
        integer don't crash. The recorder is opened by `_reset_tick`,
        not by this method.
        """
        return self._episode_idx

    def episode_end(self, *, success: bool) -> None:
        """No-op on SimRunner: sim closes episodes inside `_finalize_episode`.

        Argument intentionally unused — sim's success flag comes from
        ``step_result.info[task.success_key]``, not from the caller.
        """
        _ = success

    def _on_deadline_overrun(self, result: TickResult) -> None:
        """Sim ticks intentionally ignore the base-class rate budget.

        :class:`DeployRunner` ticks at a real-time cadence so deadline
        overruns matter — they mean the robot is starving on stale
        actions. Sim is not real-time: it runs as fast as the policy +
        env allow. The base class still enforces ``rate_hz`` because we
        share the loop, but the WARN logs would fire on every tick
        (env.step alone is well over 1 ms even on a tiny scene). When
        the caller explicitly opts into ``RAISE``, we still raise; for
        the default WARN, we no-op.
        """
        if self.deadline_overrun_policy == DeadlineOverrunPolicy.RAISE:
            super()._on_deadline_overrun(result)

    # ── Hot path ────────────────────────────────────────────────────────────

    def _tick_impl(self, tick_idx: int) -> TickResult:
        """Advance one tick: either reset the env or run policy.step + env.step."""
        assert self._env is not None, "SimRunner._tick_impl called before activate()"
        assert self._policy is not None
        if self._needs_reset:
            return self._reset_tick(tick_idx)
        return self._step_tick(tick_idx)

    # ── Reset tick ──────────────────────────────────────────────────────────

    def _reset_tick(self, tick_idx: int) -> TickResult:
        """env.reset + policy.reset, plus finalising the prior episode if any."""
        assert self._env is not None
        assert self._policy is not None
        tick_t0 = time.perf_counter()
        stamp_ns = time.time_ns()

        if self._buf.has_data:
            self._finalize_episode()

        seed = self._env_cfg.seed + self._episode_idx
        _seed_global_rngs(seed)
        self._obs = self._env.reset(seed=seed)
        self._policy.reset()
        self._step_idx = 0
        self._needs_reset = False

        # ADR-0019: open a new episode on the recorder so the first
        # _step_tick's record_frame has a target. The recorder is
        # additive — _EpisodeBuffer continues to drive in-memory video /
        # benchmark output as before.
        if self._recorder is not None and not self._recorder_episode_open:
            self._recorder.episode_start(task_string=self._env_cfg.task.instruction)
            self._recorder_episode_open = True

        # Open the MuJoCo viewer once, after the first reset (LIBERO et al.
        # rebuild ``MjModel`` on every reset; binding pre-reset hands you a
        # stale window that doesn't update). Skipped when the scene already
        # drives its own window (gym_pusht render_mode="human").
        if self._viewer is None and self._view and not self._intrinsic_viewer:
            self._viewer, self._frame_dt_s = _open_viewer_and_pacing(
                self._env, self._env_cfg, strict_view=self._strict_view
            )

        tick_ms = (time.perf_counter() - tick_t0) * 1000.0
        return TickResult(
            stamp_ns=stamp_ns,
            tick_idx=tick_idx,
            inference_ms=0.0,
            tick_ms=tick_ms,
            action_applied=False,
            episode_idx=self._episode_idx,
            step_idx=None,
            reward=None,
            terminated=None,
            truncated=None,
        )

    # ── Step tick ───────────────────────────────────────────────────────────

    def _step_tick(self, tick_idx: int) -> TickResult:  # noqa: PLR0915
        """One env step under a policy inference call.

        Long because the step span carries every per-stage timing /
        reward attribute and the frame / vla-input capture lives in the
        same critical section; splitting hurts readability.
        """
        assert self._env is not None
        assert self._policy is not None
        tick_t0 = time.perf_counter()
        stamp_ns = time.time_ns()

        record = self._env_cfg.record_video
        if record:
            frame = self._env.render()
            if frame is not None:
                self._buf.frames.append(frame)
            state = self._obs.get("state") if isinstance(self._obs, dict) else None
            if state is not None:
                self._buf.joint_positions.append(np.asarray(state, dtype=np.float32).copy())

        with _tracer().start_as_current_span(
            "eval.step", attributes={"step": self._step_idx}
        ) as step_span:
            t0 = time.perf_counter()
            # Resolve the instruction the policy is prompted with. An explicit
            # ``--instruction`` override wins over everything; otherwise the
            # env's per-episode ``obs["task"]`` language (e.g. RoboCasa
            # interpolates the sampled object name into ``get_ep_meta()["lang"]``) wins,
            # falling back to the static YAML instruction. See
            # :func:`_resolve_step_instruction`.
            obs_task = self._obs.get("task") if isinstance(self._obs, dict) else None
            instruction = _resolve_step_instruction(
                instruction_override=self._instruction_override,
                obs_task=obs_task,
                task_instruction=self._env_cfg.task.instruction,
            )
            action = self._policy.step(self._obs, instruction)
            # Debug-only obs/action capture — mirror of rskill_runner_node's
            # ``_dump_obs_to_disk`` so a deploy_sim pickle and a sim_run
            # pickle for the same tick can be diffed byte-for-byte.
            # Gated on ``OPENRAL_DUMP_OBS_TICK`` env var so the production
            # path costs nothing.
            _dump_obs_for_step(
                tick=self._step_idx + 1,
                obs=self._obs,
                raw_policy_action=action,
                prompt=instruction,
            )
            inference_ms = (time.perf_counter() - t0) * 1000.0
            self._buf.latencies.append(inference_ms)
            step_span.set_attribute("policy_latency_ms", inference_ms)
            if self._budget_ms is not None and inference_ms > self._budget_ms:
                self._buf.budget_violations += 1
                step_span.set_attribute("budget_exceeded", True)

            if record:
                self._buf.actions.append(np.asarray(action, dtype=np.float32).copy())
                vla_frame_fn = getattr(self._policy, "last_input_frame", None)
                vla_frame = vla_frame_fn() if vla_frame_fn is not None else None
                if vla_frame is not None:
                    self._buf.vla_input_frames.append(vla_frame)

            # Child span isolates physics time from policy time so reviewers
            # can answer "is the rollout slow because of inference or because
            # of MuJoCo?" without ad-hoc instrumentation.
            with _tracer().start_as_current_span(semconv.SPAN_PHYSICS_STEP) as physics_span:
                physics_t0 = time.perf_counter()
                step_result = self._env.step(action)
                physics_ms = (time.perf_counter() - physics_t0) * 1000.0
                physics_span.set_attribute("physics.step_ms", physics_ms)
            if record:
                _append_adapter_video_frames(
                    self._buf.frames,
                    step_result.info.get(_VIDEO_FRAMES_INFO_KEY),
                )
            self._obs = step_result.observation
            self._buf.total_reward += step_result.reward
            self._buf.max_step_reward = max(self._buf.max_step_reward, step_result.reward)
            self._step_idx += 1
            self._buf.steps_done = self._step_idx
            step_span.set_attribute("reward", step_result.reward)
            step_span.set_attribute("cum_reward", self._buf.total_reward)

            success_key = self._env_cfg.task.success_key
            if (
                success_key in step_result.info
                and bool(step_result.info[success_key])
                and not self._buf.success
            ):
                self._buf.success = True
                step_span.add_event("episode.first_success", attributes={"step": self._step_idx})

            if self._viewer is not None:
                self._viewer.sync()
                if self._frame_dt_s is not None:
                    elapsed = time.perf_counter() - t0
                    remaining = self._frame_dt_s - elapsed
                    if remaining > 0.0:
                        time.sleep(remaining)

            terminated = step_result.terminated
            truncated = step_result.truncated
            if terminated or truncated:
                step_span.set_attribute("terminated", terminated)
                step_span.set_attribute("truncated", truncated)
                self._needs_reset = True
            elif (
                self._env_cfg.task.max_steps is not None
                and self._step_idx >= self._env_cfg.task.max_steps
            ):
                # Step budget hit without env signal: treat as truncation.
                truncated = True
                self._needs_reset = True

            # ADR-0019: fan out per-step state / images / action to the
            # recorder. We render here unconditionally when a recorder
            # is attached — the existing _buf.frames branch above only
            # captures frames when record_video=True. The recorder is
            # the durable path, independent of the in-memory video
            # pipeline.
            if self._recorder is not None and self._recorder_episode_open:
                self._record_to_recorder(action, step_result.reward, terminated, truncated)

        tick_ms = (time.perf_counter() - tick_t0) * 1000.0
        return TickResult(
            stamp_ns=stamp_ns,
            tick_idx=tick_idx,
            inference_ms=inference_ms,
            tick_ms=tick_ms,
            action_applied=True,
            episode_idx=self._episode_idx,
            step_idx=self._step_idx - 1,
            reward=float(step_result.reward),
            terminated=terminated,
            truncated=truncated,
        )

    # ── ADR-0019 recorder fan-out ───────────────────────────────────────────

    def _record_to_recorder(
        self,
        action: NDArray[np.float32],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Fan one step's payload out to the attached :class:`RolloutRecorder`.

        Splits into its own helper to keep ``_step_tick`` readable;
        called only when ``self._recorder is not None``. The state /
        image / action extraction mirrors what _EpisodeBuffer captures
        when ``record_video=True`` — same data, durable destination.
        """
        assert self._recorder is not None
        assert self._env is not None
        # Proprioception. Falls back to a zero vector of the expected
        # shape when the env doesn't expose state in the obs dict —
        # better to land a row of zeros than crash mid-rollout.
        state = self._obs.get("state") if isinstance(self._obs, dict) else None
        if state is None:
            state_shape = self._recorder.expected_state_shape
            state_array = np.zeros(state_shape, dtype=np.float32)
        else:
            state_array = np.asarray(state, dtype=np.float32).copy()

        # Camera frame. Sim envs typically expose ONE rendered viewpoint
        # (env.render()); we fan it out to every camera key the robot
        # declares so the dataset's schema is satisfied. PR3 hardware
        # path will replace this with per-camera streams from ROS topics.
        frame = self._env.render()
        images: dict[str, NDArray[np.uint8]] = {}
        if frame is not None:
            frame_u8 = np.asarray(frame, dtype=np.uint8)
            for cam_key in self._recorder.expected_image_keys():
                images[cam_key] = frame_u8
        try:
            self._recorder.record_frame(
                observation_state=state_array,
                images=images,
                action=np.asarray(action, dtype=np.float32).copy(),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        except (ValueError, RuntimeError) as exc:
            # Recorder errors must NOT kill the rollout — the sim path
            # is the primary deliverable; dataset writes are optional
            # side-effects. Log loudly and continue.
            _log.warning(
                "rollout_recorder_record_frame_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                episode_idx=self._episode_idx,
                step_idx=self._step_idx,
            )

    # ── Episode finalisation ────────────────────────────────────────────────

    def _finalize_episode(self) -> None:
        """Build an :class:`EpisodeResult` from the per-step buffer and emit it."""
        env_cfg = self._env_cfg
        latencies = self._buf.latencies
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        max_lat = float(np.max(latencies)) if latencies else 0.0
        max_step_reward = (
            float(self._buf.max_step_reward) if self._buf.max_step_reward != float("-inf") else 0.0
        )
        num_cams = _count_policy_input_cameras(self._policy, env_cfg)

        out = EpisodeResult(
            success=self._buf.success,
            steps=self._buf.steps_done,
            total_reward=self._buf.total_reward,
            max_step_reward=max_step_reward,
            mean_step_latency_ms=mean_lat,
            max_step_latency_ms=max_lat,
            latency_budget_ms=self._budget_ms,
            budget_violations=self._buf.budget_violations,
            frames=self._buf.frames,
            vla_input_frames=self._buf.vla_input_frames,
            joint_positions=self._buf.joint_positions,
            actions=self._buf.actions,
            num_input_cameras=num_cams,
            metadata=dict(env_cfg.metadata),
        )
        self.episode_results.append(out)
        # ``scene.id`` / ``task.id`` / ``vla.id`` are closed sets per the
        # sim registries (``ROBOTS`` / ``POLICIES`` / ``SCENES``), so they
        # are safe metric labels — no per-prompt cardinality leak.
        episode_attrs = {
            "scene.id": env_cfg.scene.id,
            "task.id": env_cfg.task.id,
            "vla.id": env_cfg.vla.id,
        }
        ral_metrics.get_sim_episode_count().add(1, episode_attrs)
        if out.success:
            ral_metrics.get_sim_episode_success().add(1, episode_attrs)
        _log.info(
            "episode_done",
            scene=env_cfg.scene.id,
            task=env_cfg.task.id,
            vla=env_cfg.vla.id,
            success=out.success,
            steps=out.steps,
            mean_lat_ms=round(out.mean_step_latency_ms, 2),
            budget_violations=out.budget_violations,
        )

        # ADR-0019: close the recorder's view of the episode AFTER the
        # buffer has been drained but BEFORE the index advances, so the
        # recorder's episode_idx aligns with the EpisodeResult that was
        # just appended.
        if self._recorder is not None and self._recorder_episode_open:
            try:
                self._recorder.episode_end(success=out.success)
            except (ValueError, RuntimeError) as exc:
                _log.warning(
                    "rollout_recorder_episode_end_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    episode_idx=self._episode_idx,
                )
            self._recorder_episode_open = False

        self._episode_idx += 1
        self._buf = _EpisodeBuffer()


# ── rSkill / RNG / viewer helpers ────────────────────────────────────────────
#
# These were private to the deleted ``runner.py`` module. They moved here
# verbatim — no behaviour change. Kept module-private (single underscore)
# because nothing outside this file should call them directly.


_MOCK_POLICY_IDS = frozenset({"zero", "random"})
_MOCK_PLACEHOLDER_URI = "placeholder"


def _required_render_resolution(
    manifest: RSkillManifest, env_cfg: SimEnvironment
) -> tuple[int, int]:
    """Render (width, height) that satisfies the rSkill's camera minimums.

    A sim scene renders every camera at ``scene.observation_{width,height}``;
    each ``sensors_required`` entry may declare ``min_width`` / ``min_height``.
    When an rSkill needs more than the scene's default (e.g. rldx-mt-droid wants
    >=320 but LIBERO renders 256) we bump the render size to its requirement
    instead of failing the sensor gate. We bump ONLY to what the rSkill asks for
    — never a blanket upscale — because rendering above a checkpoint's training
    resolution is out-of-distribution for policies that resize internally
    (smolvla / pi05; native 512 LIBERO was measured OOD and reverted, PR #83).
    """
    w = int(env_cfg.scene.observation_width)
    h = int(env_cfg.scene.observation_height)
    req_w = max((s.min_width or 0 for s in manifest.sensors_required), default=0)
    req_h = max((s.min_height or 0 for s in manifest.sensors_required), default=0)
    return max(w, req_w), max(h, req_h)


def _with_render_resolution(env_cfg: SimEnvironment, width: int, height: int) -> SimEnvironment:
    """Return ``env_cfg`` with the scene rendered at ``width`` x ``height``."""
    if (width, height) == (
        env_cfg.scene.observation_width,
        env_cfg.scene.observation_height,
    ):
        return env_cfg
    scene = env_cfg.scene.model_copy(
        update={"observation_width": width, "observation_height": height}
    )
    return env_cfg.model_copy(update={"scene": scene})


def _check_rskill_compatibility(
    env_cfg: SimEnvironment,
) -> tuple[RSkillManifest | None, SimEnvironment]:
    """Load the rSkill manifest, bump render resolution to its needs, verify compat.

    Strict-by-construction: every sim eval flows through the rSkill manifest
    so the embodiment / sensor / capability contract is exercised on every
    rollout. The render resolution is raised to satisfy the rSkill's camera
    minimums (see :func:`_required_render_resolution`), and the robot's RGB
    sensor intrinsics are synced to the actual render size before the sensor
    gate — robot.yaml declares a nominal size, but in sim the rendered frame is
    really ``scene.observation_*``.

    Raises:
        openral_core.exceptions.ROSConfigError: ``vla.weights_uri`` carries
            an ``hf://`` scheme (sim runner requires a locally-resolvable
            rSkill), or ``robot_id`` is not registered in
            :data:`openral_sim.ROBOTS`.
        openral_core.exceptions.ROSCapabilityMismatch: The manifest's
            embodiment tags, capability flags, or sensor requirements do
            not intersect the robot's :class:`RobotDescription`.

    Returns:
        ``(manifest, env_cfg)`` where ``env_cfg`` may carry a bumped render
        resolution. ``manifest`` is ``None`` for built-in mock policies.
    """
    from openral_sim.factory import make_robot

    weights_uri = env_cfg.vla.weights_uri
    if env_cfg.vla.id in _MOCK_POLICY_IDS and weights_uri == _MOCK_PLACEHOLDER_URI:
        _log.info("rskill_compat_skipped_mock", vla_id=env_cfg.vla.id)
        return None, env_cfg

    if weights_uri.startswith("hf://"):
        raise ROSConfigError(
            f"vla.weights_uri {weights_uri!r} carries an 'hf://' scheme; "
            "the sim runner requires a locally-resolvable rSkill — "
            "pass a bare name or path (rskills/<name>) instead."
        )

    from openral_rskill.loader import load_rskill_manifest, rSkill

    manifest = load_rskill_manifest(weights_uri)

    # Raise the render resolution to the rSkill's camera minimums (no-op when
    # the scene already meets them). Done before make_robot + the sensor gate.
    width, height = _required_render_resolution(manifest, env_cfg)
    if (width, height) != (env_cfg.scene.observation_width, env_cfg.scene.observation_height):
        _log.info(
            "sim_render_resolution_bumped",
            skill=manifest.name,
            width=width,
            height=height,
            scene_default=(env_cfg.scene.observation_width, env_cfg.scene.observation_height),
        )
        env_cfg = _with_render_resolution(env_cfg, width, height)

    robot = make_robot(env_cfg)
    if robot is None:
        raise ROSConfigError(
            f"robot_id={env_cfg.robot_id!r} is not registered in openral_sim.ROBOTS; "
            "the sim runner requires a RobotDescription for compatibility validation"
        )

    # Sync RGB sensor intrinsics to the actual render size so the sensor gate
    # sees what the scene really produces (robot.yaml's nominal size is static).
    synced_sensors = []
    for sensor in robot.sensors:
        intr = sensor.intrinsics
        needs_resize = (
            sensor.modality == "rgb"
            and intr is not None
            and (intr.width, intr.height) != (width, height)
        )
        if needs_resize and intr is not None:
            synced_sensors.append(
                sensor.model_copy(
                    update={
                        "intrinsics": intr.model_copy(update={"width": width, "height": height})
                    }
                )
            )
        else:
            synced_sensors.append(sensor)
    robot = robot.model_copy(update={"sensors": synced_sensors})

    rSkill.check_compatibility(manifest, robot)
    _log.info(
        "rskill_compat_ok",
        skill=manifest.name,
        skill_tags=manifest.embodiment_tags,
        sensor_reqs=len(manifest.sensors_required),
        robot_id=robot.name,
        robot_tags=robot.capabilities.embodiment_tags,
        robot_sensors=[s.name for s in robot.sensors],
    )
    return manifest, env_cfg


_SEQUENTIAL_INIT_ENV = "OPENRAL_SIM_SEQUENTIAL_INIT"
# Set to "1" by :meth:`SimRunner.activate` for the duration of the
# scene build when ``--view`` is on. Read by scene factories whose
# underlying engine builds its live viewer at construct time
# (currently the SAPIEN/ManiSkill3 ``simpler_env`` backend, which
# accepts ``render_mode='human'`` to ``gym.make``). MuJoCo-backed
# adapters open the viewer lazily after ``reset()`` via
# :func:`_open_viewer_and_pacing` and so ignore this var.
_VIEW_ENV = "OPENRAL_SIM_VIEW"

# Scene-id prefixes that are known to race against the lerobot/transformers
# import chain when ``make_env`` and ``make_policy`` run on parallel
# threads. Two distinct race classes have been observed:
#
# 1. ``openarm_`` / ``tabletop_push`` — the scene factory imports a module
#    that transitively pulls ``transformers`` onto the env thread, racing the
#    same lazy ``transformers`` submodule attributes lerobot's policy factory is
#    resolving on the policy thread. The trigger differs per scene:
#      * ``openarm_`` imports robosuite + robosuite_models in its env factory
#        (robosuite's deps pull transformers).
#      * ``tabletop_push`` resolves ``assets.mjcf`` (via ``resolve_asset``)
#        to load the arm MJCF,
#        which imports ``openral_hal._mujoco_arm`` → ``openral_hal._base``;
#        that chain transitively imports transformers. (Verified:
#        ``import openral_hal._mujoco_arm`` leaves ``transformers`` in
#        ``sys.modules``.) Note ``so101_box`` is NOT affected — it resolves the
#        SO-101 MJCF by importing ``robot_descriptions`` directly, never
#        ``_mujoco_arm``, so nothing pulls transformers onto its env thread.
#    Either way the ``transformers._LazyModule`` attr lookup is not thread-safe
#    under concurrent import; symptom is ``ImportError: cannot import name
#    'AutoConfig' from 'transformers'`` surfaced as ROSConfigError("requires
#    torch + lerobot[libero]").
#
# 2. ``maniskill3`` / ``simpler_env`` (both SAPIEN-backed) — the policy
#    factory's ``torch.set_default_dtype(bfloat16)`` window (transient
#    inside smolvla / pi05 / lerobot processor build) leaks into the env
#    thread's SAPIEN ``gym.make`` call. SAPIEN renders an internal tensor
#    using the active default dtype during env construction, and bf16
#    is not a supported ScalarType for its image path. Symptom is
#    ``TypeError: Got unsupported ScalarType BFloat16 was raised from
#    the environment creator for PickCube-v1`` (or any SAPIEN env).
#    ``openral benchmark run --suite maniskill3_panda`` and
#    ``openral sim run --config scenes/simpler_env_widowx_*``
#    both trip this when the policy is bf16; sequential init avoids
#    the dtype window overlap entirely.
#
# We force sequential init for any scene whose id starts with one of
# these prefixes; the user-facing ``OPENRAL_SIM_SEQUENTIAL_INIT=1`` env
# var still works as an explicit manual override for combos we haven't
# yet catalogued. Add new prefixes here as concurrency races surface;
# the alternative (always-sequential) would cost the LIBERO / MetaWorld
# combos their ~5-10 s parallel-init win for no benefit. (RoboCasa was
# previously assumed safe here too, but its robosuite import races the
# same way openarm's does — see prefix 3 below.)
_RACE_PRONE_SCENE_PREFIXES: tuple[str, ...] = (
    "openarm_",
    "tabletop_push",
    "maniskill3",
    "simpler_env",
    # 3. ``robocasa/`` (kitchen ``robocasa/<task>`` + GR1 ``robocasa/gr1/<task>``)
    #    — same class-1 race as ``openarm_``: the RoboCasa env factory imports
    #    robosuite (``ensure_backend_deps`` → ``_has_module("robosuite")`` →
    #    ``importlib.util.find_spec`` which *executes* ``robosuite/__init__``)
    #    on the env thread while the policy thread imports robosuite-adjacent
    #    modules. Concurrent import of the same submodule trips CPython's
    #    ``_load_unlocked`` ``sys.modules.pop`` → ``KeyError:
    #    'robosuite.renderers.viewer.mjviewer_renderer'`` (non-deterministic:
    #    rldx-ft-rc365 won the race, rldx-ft-robocasa lost it on the same
    #    scene). The earlier "RoboCasa+RLDX benefits from parallel init" note
    #    was wrong — robosuite's heavy import makes it as race-prone as
    #    openarm. The ~5-10 s parallel win is not worth a hard import crash.
    "robocasa/",
)


def _scene_requires_sequential_init(env_cfg: SimEnvironment) -> bool:
    """Return True when ``env_cfg.scene.id`` matches a known race-prone prefix."""
    scene_id = str(env_cfg.scene.id)
    return any(scene_id.startswith(prefix) for prefix in _RACE_PRONE_SCENE_PREFIXES)


def _build_env_and_policy(
    env_cfg: SimEnvironment,
) -> tuple[SimRollout, PolicyAdapter]:
    """Build (env, policy) — concurrently by default, sequentially on opt-out.

    Encapsulates the GH-134 parallelisation so :meth:`SimRunner.activate`
    stays readable and so the behaviour is unit-testable in isolation
    (see ``tests/unit/test_sim_runner_parallel_init.py``).

    The two side effects we care about are bounded:

    * **No shared mutable state** between :func:`make_env` and
      :func:`make_policy` — both only read the immutable
      :class:`SimEnvironment` Pydantic model and dispatch to their
      respective registries (``SCENES`` / ``POLICIES``).
    * **Exception propagation** is verbatim — the helper does not catch
      :class:`ROSError` (or anything else) on either side. The first
      exception observed wins; the other future is allowed to finish
      so its resources can be cleaned up on the worker thread without
      racing with the caller.

    Args:
        env_cfg: Validated :class:`openral_core.SimEnvironment`.

    Returns:
        ``(env, policy)`` — both fully constructed and ready for the
        runner to drive.

    Raises:
        openral_core.exceptions.ROSError: Whatever :func:`make_env` or
            :func:`make_policy` raises, re-raised verbatim. If both sides
            raise, the env-build exception wins (it is awaited first);
            the policy-build exception is suppressed via
            :class:`contextlib.suppress` to surface the env failure
            cleanly without losing the env traceback to a chained
            policy traceback.
    """
    env_opt_in = os.environ.get(_SEQUENTIAL_INIT_ENV, "").strip() == "1"
    scene_forced = _scene_requires_sequential_init(env_cfg)
    sequential = env_opt_in or scene_forced
    t0 = time.perf_counter()
    if sequential:
        if scene_forced and not env_opt_in:
            _log.info(
                "sim_init_force_sequential",
                scene_id=str(env_cfg.scene.id),
                reason="race_prone_scene_prefix",
                opt_out_env=_SEQUENTIAL_INIT_ENV,
            )
        env_t0 = time.perf_counter()
        env = make_env(env_cfg)
        env_ms = (time.perf_counter() - env_t0) * 1000.0
        policy_t0 = time.perf_counter()
        policy = make_policy(env_cfg)
        policy_ms = (time.perf_counter() - policy_t0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0
        _log.info(
            "sim_init_sequential",
            env_ms=round(env_ms, 1),
            policy_ms=round(policy_ms, 1),
            total_ms=round(total_ms, 1),
            opt_out_env=_SEQUENTIAL_INIT_ENV,
        )
        return env, policy

    # Parallel path. 2 workers is the natural fan-out; we do not size
    # by os.cpu_count() because there are only ever two tasks and a
    # larger pool would only confuse profilers.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="openral_sim_init") as pool:
        env_t0 = time.perf_counter()
        policy_t0 = time.perf_counter()
        env_future = pool.submit(make_env, env_cfg)
        policy_future = pool.submit(make_policy, env_cfg)
        try:
            env = env_future.result()
        except BaseException:
            # The policy future may still be running. Allow it to finish
            # so its resources get cleaned up on the worker thread; we
            # suppress its exception here because the caller is already
            # going to receive the env failure (the first one we saw)
            # and chaining the policy traceback onto it would muddy the
            # report. The worker thread itself does NOT swallow the
            # exception — Future.exception() captures it on the future,
            # and the executor logs it via the worker's default
            # __exit__ shutdown(wait=True).
            with contextlib.suppress(BaseException):
                policy_future.result()
            raise
        env_ms = (time.perf_counter() - env_t0) * 1000.0
        policy = policy_future.result()
        policy_ms = (time.perf_counter() - policy_t0) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0
    # ``max(env_ms, policy_ms)`` is the wall-clock lower bound; the
    # difference vs ``env_ms + policy_ms`` is the parallelisation win.
    parallel_floor_ms = max(env_ms, policy_ms)
    serial_estimate_ms = env_ms + policy_ms
    saved_ms = serial_estimate_ms - total_ms
    _log.info(
        "sim_init_parallel",
        env_ms=round(env_ms, 1),
        policy_ms=round(policy_ms, 1),
        total_ms=round(total_ms, 1),
        parallel_floor_ms=round(parallel_floor_ms, 1),
        serial_estimate_ms=round(serial_estimate_ms, 1),
        saved_ms=round(saved_ms, 1),
        opt_out_env=_SEQUENTIAL_INIT_ENV,
    )
    return env, policy


def _seed_global_rngs(seed: int) -> None:
    """Seed Python / NumPy / Torch RNGs so stochastic policies reproduce per seed.

    The env's ``reset(seed=...)`` only controls the env's RNG. Stochastic
    policies (Diffusion Policy's iterative denoiser, π0.5's flow-matching
    head, anything calling ``torch.randn`` inside ``select_action``) pull
    from the global torch RNG state instead. Without seeding here, two
    invocations of the same YAML produce different trajectories.

    Torch import is deferred — pure-Python adapters (mock, random) don't
    need torch and shouldn't pay for the import.
    """
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch
    except ImportError:  # pragma: no cover  # reason: torch is required by every VLA adapter
        return
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)


class _SapienViewerProxy:
    """Adapt a SAPIEN/ManiSkill3 env with ``render_mode='human'`` to the runner's viewer contract.

    The runner only invokes :meth:`sync` (after each applied step) and
    :meth:`close` (during teardown). For SAPIEN the live window is owned
    by the env itself — :meth:`sync` just pumps the next frame via
    ``env.viewer_render()`` (which in turn calls ``self._env.render()``
    on the underlying gym env), and :meth:`close` is a no-op since the
    rollout's own :meth:`close` tears down the env and its viewer
    together.
    """

    def __init__(self, env: SimRollout) -> None:
        self._env = env

    def sync(self) -> None:
        viewer_render = getattr(self._env, "viewer_render", None)
        if viewer_render is not None:
            viewer_render()

    def close(self) -> None:
        return None


def _open_viewer_and_pacing(
    env: SimRollout,
    env_cfg: SimEnvironment,
    *,
    strict_view: bool,
) -> tuple[Any, float | None]:
    """Open the per-step live viewer and compute the per-step sleep budget.

    Two duck-typed paths, in priority order:

    * Adapter exposes ``viewer_render(self) -> None`` — the underlying
      engine owns the viewer (SAPIEN / ManiSkill3 ``render_mode='human'``
      etc.). Return a :class:`_SapienViewerProxy` that the runner can
      ``.sync()`` after each step.
    * Adapter exposes ``mujoco_handles(self) -> (MjModel, MjData) | None``
      — open a passive ``mujoco.viewer`` and pace from
      ``MjModel.opt.timestep``.

    Returns ``(viewer, frame_dt_s)``. ``viewer`` may be ``None`` when
    neither path is available and ``strict_view`` is False; the runner
    then continues offscreen with a warning.
    """
    viewer_render = getattr(env, "viewer_render", None)
    if callable(viewer_render):
        # Engine-owned viewer (SAPIEN today). We do not synthesise a
        # per-step sleep budget — SAPIEN's ``env.render()`` paces the
        # viewer against the simulator's wall-clock and the runner's
        # tick loop already sleeps via the base class' rate limiter.
        return _SapienViewerProxy(env), None

    handles_fn = getattr(env, "mujoco_handles", None)
    handles = handles_fn() if callable(handles_fn) else None
    if handles is None:
        msg = (
            f"scene {env_cfg.scene.id!r} does not expose MuJoCo handles "
            "or a viewer_render() hook; --view is unsupported for this backend"
        )
        if strict_view:
            raise ROSConfigError(msg)
        _log.warning("viewer_unsupported", scene=env_cfg.scene.id)
        return None, None
    try:
        import mujoco.viewer
    except ImportError as exc:
        if strict_view:
            raise ROSConfigError(
                "mujoco.viewer is unavailable; install mujoco>=3 to use --view"
            ) from exc
        _log.warning("viewer_import_failed", error=str(exc))
        return None, None
    mj_model, mj_data = handles
    # Hide both side panels (left settings / right info) so the window shows
    # only the simulation render; still toggleable at runtime via Tab/Shift+Tab.
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, show_left_ui=False, show_right_ui=False
    )
    _aim_viewer_camera(viewer, env, mj_model, mj_data)
    n_substeps = int(getattr(mj_model.opt, "nsubsteps", 1)) or 1
    frame_dt_s = float(mj_model.opt.timestep) * n_substeps
    return viewer, frame_dt_s


def _aim_viewer_camera(viewer: Any, env: SimRollout, mj_model: Any, mj_data: Any) -> None:
    """Set the viewer's opening camera + geom visibility.

    * Hides robosuite/RoboCasa collision shells (the red kitchen / green robot)
      via :func:`apply_robosuite_visual_geomgroups` so textures render; no-op on
      dm_control/gym scenes.
    * Sets the **free** camera's opening pose via :func:`initial_viewer_camera`
      (eye at the authored overview camera, orbit pivot on the robot base, else
      the base-aligned default). The camera stays ``mjCAMERA_FREE`` so the user
      keeps full mouse control — drag to orbit, scroll to zoom; we only set the
      initial viewpoint.

    Best effort: any failure leaves MuJoCo's default free camera untouched.
    """
    try:
        import mujoco

        # Lazy import: openral_hal.__init__ pulls torch/lerobot, so we pay that
        # cost only at interactive viewer-open (mirrors openarm_robosuite assets).
        from openral_hal.depth_cloud import (
            apply_robosuite_visual_geomgroups,
            initial_viewer_camera,
        )

        lookat, distance, azimuth, elevation = initial_viewer_camera(
            model=mj_model, data=mj_data, description=getattr(env, "description", None)
        )
        with viewer.lock():
            # Hide robosuite/RoboCasa collision shells so textures render (the
            # red kitchen / green robot are collision geoms); no-op elsewhere.
            apply_robosuite_visual_geomgroups(viewer.opt, mj_model)
            cam = viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = lookat
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = elevation
    except Exception as exc:  # reason: camera aiming is cosmetic, never fatal
        _log.warning("viewer_camera_aim_failed", error=str(exc))
