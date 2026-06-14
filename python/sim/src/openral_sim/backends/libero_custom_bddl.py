"""Robosuite-backed custom-scene adapter for pi0.5-LIBERO.

This adapter routes through **robosuite + LIBERO's `OffScreenRenderEnv`**
so the user gets pi0.5's exact training pipeline (OSC_POSE
controller, robosuite renderer, sRGB framebuffer, lerobot-style state
encoding) on a *custom* scene specified via a BDDL file.

Unlike the four raw-mujoco custom scenes
(``franka_cereal_pickplace``, ``franka_libero_pickplace``,
``franka_libero_floor_pickplace``, ``franka_libero_arena_pickplace``)
which reimplement OSC + rendering in Python and inevitably drift from
the training distribution, this adapter delegates 100 % of the
control and rendering work to robosuite — the policy sees the same
pixels and the same dynamics it did at training time, just with
*your* objects, positions, and target predicate.

Usage — write a custom BDDL file in LIBERO's format (see
``libero/bddl_files/libero_object/*.bddl`` for examples), then::

    scene:
      id: franka_libero_custom_bddl
      backend: mujoco
      backend_options:
        bddl_file: "/path/to/your_task.bddl"
        # Optional — path to a .pruned_init file with hand-tuned
        # initial qpos. Omit to use robosuite's default init
        # (panda starts at panda_robot.init_qpos).
        init_state_file: "/path/to/your_task.pruned_init"
        init_state_index: 0   # which row of the init file to use
      observation_height: 256
      observation_width: 256

    vla:
      id: pi05
      weights_uri: "rskills/pi05-libero-nf4"
      device: cpu
      extra:
        flip_images_180: true   # robosuite outputs in OpenGL conv.
        state_dim: 8
        camera_keys: ["camera1", "camera2"]
        n_action_steps: 50

The state encoding matches lerobot's :class:`LiberoEnvProcessorStep`
verbatim (verified earlier in this session — sub-mm parity with real
LIBERO when wired correctly).

Registered as the scene id ``franka_libero_custom_bddl`` in
:data:`openral_sim.SCENES`.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult, sim_time_ns_from_mujoco_handles

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


_CUSTOM_SCENE_ID = "franka_libero_custom_bddl"
_PI05_STATE_DIM = 8
# A LIBERO ``init_state_file`` is a pickled checkpoint loaded with
# ``torch.load`` — which executes arbitrary code from the file. The path is
# supplied by the (possibly shared/downloaded) scene config, so this is a
# remote-code-execution sink. Gate it behind the same explicit acknowledgement
# used for the other pickle sinks (security audit 2026-06, C2). A hard in-tree
# path restriction was rejected: init-state files are documented to live at
# absolute external paths, so an allowlist would break legitimate use.
_ALLOW_UNSAFE_PICKLE_ENV = "OPENRAL_ALLOW_UNSAFE_PICKLE"


def _load_init_states(init_state_path: Path) -> Any:
    """Load a pickled LIBERO init-state file, gated against pickle RCE.

    Args:
        init_state_path: Path to the ``.init`` / ``.pruned_init`` checkpoint
            named by ``scene.backend_options.init_state_file``.

    Returns:
        The deserialized init-state object (typically an array/dict of states).

    Raises:
        ROSConfigError: If the file is missing, or ``OPENRAL_ALLOW_UNSAFE_PICKLE``
            is not set to ``"1"`` to acknowledge the trust assumption.
    """
    if not init_state_path.exists():
        raise ROSConfigError(f"init_state_file does not exist: {init_state_path}")
    if os.environ.get(_ALLOW_UNSAFE_PICKLE_ENV, "0") != "1":
        raise ROSConfigError(
            f"Loading init_state_file '{init_state_path}' deserializes a pickle via "
            "torch.load, which executes arbitrary code from the file (remote-code-execution "
            "risk for an untrusted scene config). To load a TRUSTED init-state file, set: "
            f"export {_ALLOW_UNSAFE_PICKLE_ENV}=1"
        )
    import torch

    return torch.load(str(init_state_path), weights_only=False)


# Numerical guard for quaternion → axis-angle conversion. When the
# quaternion is near identity (|w| ≈ 1), the rotation-axis denominator
# ``sqrt(1 - w^2)`` collapses to zero and the axis is undefined; the
# canonical convention is to return a zero rotation. Threshold matches
# lerobot's ``LiberoEnvProcessorStep._quat2axisangle``.
_QUAT_AXIS_DEN_EPS = 1e-10


def _quat_to_axisangle_xyzw(quat_xyzw: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert (x, y, z, w) quaternion → 3-D axis-angle vector.

    Mirrors :meth:`lerobot.processor.env_processor.LiberoEnvProcessorStep._quat2axisangle`.
    robosuite returns ``robot0_eef_quat`` in xyzw order.
    """
    w = float(np.clip(quat_xyzw[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den < _QUAT_AXIS_DEN_EPS:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    axis = quat_xyzw[:3] / den
    out: NDArray[np.float32] = (axis * angle).astype(np.float32)
    return out


@dataclass
class _LiberoCustomBDDLSim:
    """Wraps :class:`libero.libero.envs.env_wrapper.OffScreenRenderEnv`.

    All control + rendering goes through robosuite (the same pipeline
    that the in-tree :mod:`openral_sim.backends.libero` adapter
    uses for benchmark tasks), so pi0.5 sees the exact distribution
    it was trained against.
    """

    scene: SceneSpec
    task: TaskSpec
    _env: Any  # libero.libero.envs.env_wrapper.OffScreenRenderEnv
    _last_pixels: dict[str, NDArray[np.uint8]] = field(default_factory=dict)
    _init_states: Any = None  # numpy array of shape (N, ?) or None
    _init_state_index: int = 0

    def reset(self, seed: int | None = None) -> Observation:
        """Reset robosuite + optionally apply a pickled init state."""
        if seed is not None:
            # robosuite's env.seed() exists on most env classes but is not
            # universal (some wrappers omit it); be permissive.
            with contextlib.suppress(AttributeError, TypeError):
                self._env.seed(int(seed))
        raw = self._env.reset()
        if self._init_states is not None:
            n = len(self._init_states)
            row = self._init_states[self._init_state_index % n]
            raw = self._env.set_init_state(row)
        return self._wrap_obs(raw)

    def step(self, action: NDArray[np.float32]) -> StepResult:
        """Pass action straight through robosuite's OSC_POSE controller."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        raw, reward, done, info = self._env.step(a)
        terminated = bool(info.get("success", done))
        truncated = bool(done and not terminated)
        step_info: dict[str, object] = dict(info)
        if self.task.success_key is not None:
            step_info[self.task.success_key] = terminated
        return StepResult(
            observation=self._wrap_obs(raw),
            reward=float(reward) if reward is not None else 0.0,
            terminated=terminated,
            truncated=truncated,
            info=step_info,
        )

    @property
    def action_dim(self) -> int:
        """Flat action width the env's ``step`` accepts (LIBERO OSC_POSE = 7).

        ADR-0036 — ``SimAttachedHAL._probe_env_action_dim`` reads this so a
        cartesian rSkill's slot-packed action is sized to the LIBERO action
        space (6-D OSC end-effector delta + gripper) rather than the
        robosuite-mobile-manipulator fallback (11). robosuite exposes the
        per-robot action width on each ``robots`` entry; summing is correct
        for single- and multi-arm robosuite envs and is robust to the
        ``OffScreenRenderEnv`` ``.env`` / ``._env`` wrapper layering.
        """
        return int(sum(r.action_dim for r in self._env.robots))

    def render(self) -> NDArray[np.uint8] | None:
        return self._last_pixels.get("camera1")

    def close(self) -> None:
        # robosuite occasionally throws on close (EGL teardown, missing
        # sim handle on already-closed env) — best-effort, never raise.
        with contextlib.suppress(Exception):
            self._env.close()

    def mujoco_handles(self) -> tuple[Any, Any] | None:
        """Reach robosuite's underlying ``MjModel`` / ``MjData`` for the viewer."""
        sim = getattr(self._env.env, "sim", None) or getattr(self._env, "sim", None)
        if sim is None:
            return None
        model = getattr(getattr(sim, "model", None), "_model", None)
        data = getattr(getattr(sim, "data", None), "_data", None)
        if model is None or data is None:
            return None
        return model, data

    def sim_time_ns(self) -> int | None:
        """Elapsed MuJoCo sim time in ns (ADR-0048 Phase 1), or None.

        Reads ``MjData.time`` off :meth:`mujoco_handles`. Monotonic within an
        episode; rewinds on ``reset``.
        """
        return sim_time_ns_from_mujoco_handles(self.mujoco_handles())

    def _wrap_obs(self, raw: dict[str, Any]) -> Observation:
        """Match lerobot's LiberoEnvProcessor state composition exactly."""
        # Cameras — robosuite returns ``agentview_image`` and
        # ``robot0_eye_in_hand_image``; expose under our naming.
        camera1 = raw.get("agentview_image")
        camera2 = raw.get("robot0_eye_in_hand_image")
        if camera1 is not None:
            self._last_pixels["camera1"] = np.asarray(camera1, dtype=np.uint8)
        if camera2 is not None:
            self._last_pixels["camera2"] = np.asarray(camera2, dtype=np.uint8)

        eef_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
        eef_quat_xyzw = np.asarray(raw["robot0_eef_quat"], dtype=np.float32)
        eef_axisangle = _quat_to_axisangle_xyzw(eef_quat_xyzw)
        gripper_qpos = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)

        state = np.zeros(_PI05_STATE_DIM, dtype=np.float32)
        state[0:3] = eef_pos
        state[3:6] = eef_axisangle
        state[6:8] = gripper_qpos

        # Task description comes from the BDDL; fall back to user's
        # YAML-specified instruction if missing.
        instr = getattr(self._env, "language_instruction", None) or self.task.instruction

        return {
            "images": {
                "camera1": self._last_pixels.get(
                    "camera1", np.zeros((256, 256, 3), dtype=np.uint8)
                ),
                "camera2": self._last_pixels.get(
                    "camera2", np.zeros((256, 256, 3), dtype=np.uint8)
                ),
            },
            "state": state,
            "task": instr,
            "raw": raw,
        }


def _build_libero_custom_bddl_scene(env_cfg: SimEnvironment) -> _LiberoCustomBDDLSim:
    """Construct a robosuite + LIBERO OffScreenRenderEnv from a custom BDDL.

    Required ``scene.backend_options.bddl_file``: absolute path to a
    BDDL file. Optional ``init_state_file`` + ``init_state_index``
    for a hand-tuned starting pose.

    Raises:
        ROSConfigError: ``bddl_file`` missing or unreadable; user
            declines the LIBERO auto-install prompt; install plan
            ran but ``libero.libero.envs.env_wrapper`` still refuses
            to import (post-install probe failure).
    """
    backend_opts = env_cfg.scene.backend_options or {}
    bddl_file = backend_opts.get("bddl_file")
    if not bddl_file:
        raise ROSConfigError(
            "franka_libero_custom_bddl requires scene.backend_options.bddl_file "
            "(absolute path to a LIBERO-format BDDL file describing the task)."
        )
    bddl_path = Path(str(bddl_file))
    if not bddl_path.exists():
        raise ROSConfigError(f"bddl_file does not exist: {bddl_path}")

    from openral_sim._deps import ensure_backend_deps

    ensure_backend_deps("libero")

    try:
        from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    except ImportError as exc:  # pragma: no cover - tested via runtime error path
        # ensure_backend_deps re-probes after running its plan so this
        # branch is only reached when the install ran but
        # libero.libero.envs.env_wrapper still refuses to import (e.g.
        # partial C-extension compile failure). Keep the typed error so
        # the user has somewhere to look.
        raise ROSConfigError(
            "LIBERO backend installed but libero.libero.envs.env_wrapper still "
            "refuses to import. Inspect the auto-install output above and re-run: "
            "CC=/usr/bin/gcc just sync --all-packages --group libero"
        ) from exc

    h = env_cfg.scene.observation_height
    w = env_cfg.scene.observation_width

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        robots=["Panda"],
        controller="OSC_POSE",
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=h,
        camera_widths=w,
        camera_depths=False,
        horizon=int(env_cfg.task.max_steps or 200),
        ignore_done=False,
        control_freq=20,
        use_camera_obs=True,
        has_offscreen_renderer=True,
        has_renderer=False,
    )

    # Optional: load a pre-pickled init state (.init or .pruned_init).
    init_states = None
    init_state_index = 0
    init_state_file = backend_opts.get("init_state_file")
    if init_state_file:
        init_states = _load_init_states(Path(str(init_state_file)))
        raw_index = backend_opts.get("init_state_index", 0)
        if not isinstance(raw_index, (int, str)):
            raise ROSConfigError(
                f"init_state_index must be an int (got {type(raw_index).__name__}: {raw_index!r})"
            )
        init_state_index = int(raw_index)

    return _LiberoCustomBDDLSim(
        scene=env_cfg.scene,
        task=env_cfg.task,
        _env=env,
        _init_states=init_states,
        _init_state_index=init_state_index,
    )


# LIBERO's MuJoCo physics hard-wire the Franka Panda.
SCENES.register(_CUSTOM_SCENE_ID, fixed_robot="franka_panda")(_build_libero_custom_bddl_scene)
