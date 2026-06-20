"""Sim-only free-running idle stepper for deploy-sim (ADR-0034 amendment).

In ``openral deploy sim`` the MuJoCo env lives only in the HAL node via
:class:`~openral_hal.sim_attached.SimAttachedHAL`, and ``env.step()`` runs only
from ``send_action`` — reached only while a skill is executing. When idle the
env froze, so camera frames went stale and the ADR-0035 perception /
object-detector bus saw a dead scene. :meth:`SimAttachedHAL.idle_step` advances
the env one tick with a zero/HOLD action so cameras keep rendering.

These tests exercise the real LIBERO (robosuite OSC_POSE) digital twin the way
``test_sim_attached_action_dim.py`` does — no mocks (CLAUDE.md §1.11). The
``should_idle_step`` predicate test is pure (no sim) and always runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from openral_hal.sim_sensor_bridge import should_idle_step

# Force EGL (off-screen) rendering so CI hosts without a display don't abort.
# The classic renderer calls glXOpenDisplay() and raises SIGABRT on headless
# runners; EGL avoids the display requirement entirely.
os.environ.setdefault("MUJOCO_GL", "egl")


def _mujoco_renderer_probe_error() -> str | None:
    """Return ``None`` if a MuJoCo off-screen renderer can be created, else a reason.

    Creating a ``mujoco.Renderer`` on a headless host without a working GL/EGL
    stack calls ``abort()`` at the C level (SIGABRT), which a Python
    ``try/except`` cannot catch — an in-process probe therefore crashes pytest
    outright (``Fatal Python error: Aborted``) and takes the whole partition
    down with it. Running the probe in a subprocess turns that abort into a
    non-zero exit code we can detect and convert into a clean skip reason,
    leaving collection alive. Mirrors ``tests/sim/conftest`` (a sibling test
    root we cannot import across).
    """
    import subprocess
    import sys

    probe = (
        "import mujoco;"
        "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody></worldbody></mujoco>');"
        "r = mujoco.Renderer(m, 1, 1); r.close()"
    )
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )
    except FileNotFoundError:  # mujoco import unavailable in the probe interpreter
        return "mujoco unavailable for renderer probe"
    except subprocess.TimeoutExpired:
        return "mujoco renderer probe timed out (120s)"
    if proc.returncode == 0:
        return None
    stderr_lines = (proc.stderr or "").strip().splitlines()
    detail = stderr_lines[-1] if stderr_lines else "no stderr"
    return f"renderer probe exited {proc.returncode}: {detail}"


# Every test below that builds a HAL (native-so101 or LIBERO) renders an
# off-screen camera frame inside ``connect()``. On a headless CI runner
# (no GPU/display) that either abort()s the process at the C level (native
# MuJoCo Renderer → SIGABRT) or raises (robosuite's EGL path →
# ``eglQueryString`` AttributeError) — so every render-dependent test must
# skip when no off-screen renderer is available. The robosuite/libero
# importorskips alone are not enough: a host can have robosuite installed yet
# still lack a working GL/EGL stack.
_RENDERER_ERROR = _mujoco_renderer_probe_error()
_requires_renderer = pytest.mark.skipif(
    _RENDERER_ERROR is not None,
    reason=f"mujoco renderer unavailable: {_RENDERER_ERROR}",
)


def test_should_idle_step_yields_within_hold_engages_after() -> None:
    """Pure predicate: yields within idle_hold of a recent action, engages after.

    Independent of rclpy / sim — the single-threaded executor hand-off rests on
    this timestamp comparison alone.
    """
    hold = 200_000_000  # 200 ms
    now = 1_000_000_000

    # A real action 50 ms ago — still inside the hold → yield (False).
    assert should_idle_step(now, last_action_ns=now - 50_000_000, idle_hold_ns=hold) is False
    # Exactly at the boundary (200 ms ago) → engage (>= is True).
    assert should_idle_step(now, last_action_ns=now - hold, idle_hold_ns=hold) is True
    # 500 ms ago — well past the hold → engage.
    assert should_idle_step(now, last_action_ns=now - 500_000_000, idle_hold_ns=hold) is True
    # Never actuated (last_action_ns == 0) → engage so an idle scene starts.
    assert should_idle_step(now, last_action_ns=0, idle_hold_ns=hold) is True


def _build_libero_hal() -> object:
    """Build a connected SimAttachedHAL over the real LIBERO digital twin.

    Mirrors the fixture idiom in ``test_sim_attached_action_dim.py``.
    """
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/libero_spatial.yaml", robot_id_fallback="franka_panda"
    )
    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed)
    hal.connect()
    return hal


def _build_so101_hal() -> object:
    """Build a connected SimAttachedHAL over the native-MuJoCo so101 box scene.

    A native (non-robosuite/non-libero) MuJoCo backend, so it runs in envs
    without LIBERO. Its action width is not introspectable (the backend exposes
    no ``action_dim``), so we pass ``env_action_dim=6`` explicitly — the
    documented path the :class:`SimAttachedHAL` constructor supports for
    non-introspectable envs (mirrors what the lifecycle node would resolve).
    """
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/so101_tube_insertion.yaml", robot_id_fallback="so101_follower"
    )
    desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed, env_action_dim=6)
    hal.connect()
    return hal


def _first_frame(images: dict[str, object]) -> np.ndarray:
    for arr in images.values():
        a = np.asarray(arr)
        if a.ndim == 3:
            return a
    raise AssertionError(f"no HWC frame in images keys={sorted(images)}")


@_requires_renderer
def test_idle_step_advances_render_and_changes_frame() -> None:
    """Idle → idle_step() returns True and the rendered frame advances (un-freeze)."""
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    pytest.importorskip("libero")
    hal = _build_libero_hal()

    before = _first_frame(hal.read_images()).copy()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    # Step several idle ticks so settling dynamics produce a visible delta.
    for _ in range(5):
        assert hal.idle_step() is True  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    after = _first_frame(hal.read_images())  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface

    assert before.shape == after.shape
    # Physics + render advanced — the frozen-scene regression would leave these
    # byte-identical.
    assert not np.array_equal(before, after)


@_requires_renderer
def test_idle_step_advances_render_native_so101() -> None:
    """Idle → idle_step() advances the frame on the native-MuJoCo so101 backend.

    The same regression as the LIBERO test, against a backend that needs no
    robosuite/libero so it runs in this env.
    """
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    hal = _build_so101_hal()

    before = _first_frame(hal.read_images()).copy()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    for _ in range(5):
        assert hal.idle_step() is True  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    after = _first_frame(hal.read_images())  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    assert before.shape == after.shape
    assert not np.array_equal(before, after)


@_requires_renderer
def test_idle_step_suppressed_when_estop_latched_native_so101() -> None:
    """Estop latched → idle_step() returns False, frame unchanged (native so101)."""
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    hal = _build_so101_hal()

    before = _first_frame(hal.read_images()).copy()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    hal.estop()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    assert hal.idle_step() is False  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    after = _first_frame(hal.read_images())  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    assert np.array_equal(before, after)


@_requires_renderer
def test_idle_step_resets_then_steps_after_episode_termination() -> None:
    """Terminated episode → idle_step does reset-then-zero-step (no raise; latch cleared).

    Mirrors the ADR-0036 reset test for ``send_action``.
    """
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    pytest.importorskip("libero")
    hal = _build_libero_hal()

    # Simulate the env having reported a terminal step on the previous tick.
    hal._episode_done = True  # type: ignore[attr-defined]  # reason: white-box latch set
    # Must NOT raise — the reset branch resets the env before zero-stepping.
    assert hal.idle_step() is True  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    # A single zero step on the fresh episode does not terminate → latch clear.
    assert hal._episode_done is False  # type: ignore[attr-defined]  # reason: white-box latch read


@_requires_renderer
def test_idle_step_suppressed_when_estop_latched() -> None:
    """Estop latched → idle_step() returns False and _last_obs is unchanged (freeze)."""
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    pytest.importorskip("libero")
    hal = _build_libero_hal()

    before = _first_frame(hal.read_images()).copy()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    hal.estop()  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    assert hal.idle_step() is False  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    after = _first_frame(hal.read_images())  # type: ignore[attr-defined]  # reason: SimAttachedHAL surface
    # Estopped HAL freezes — the cached frame must not advance.
    assert np.array_equal(before, after)


def test_real_hal_has_no_idle_step_and_sim_yaml_on_real_raises() -> None:
    """SAFETY: a real HAL must NOT define idle_step; sim_env_yaml + mode=real is rejected.

    The idle-step timer is gated on ``getattr(hal, 'idle_step', None)`` so against
    a real HAL it is never created — this is the primary real-hardware exclusion
    (a zero vector is a HOLD in sim but "drive to 0 rad" on a real position arm).
    The ``sim_env_yaml`` + ``mode='real'`` rejection locks the secondary gate.
    """
    from openral_core import RobotDescription
    from openral_core.exceptions import ROSConfigError
    from openral_hal.resolver import build_hal

    desc = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    if desc.hal.real is None:
        pytest.skip("franka_panda has no real HAL declared; cannot exercise the real path")

    real_hal = build_hal(desc, mode="real")
    # The method-only-on-SimAttachedHAL exclusion: a real HAL never defines it.
    assert getattr(real_hal, "idle_step", None) is None

    # Secondary backstop: a sim scene can never attach to a real-hardware HAL.
    with pytest.raises(ROSConfigError):
        build_hal(desc, mode="real", sim_env_yaml="scenes/sim/libero_spatial.yaml")


class _RecordingTimer:
    """Stand-in for an rclpy timer handle — records cancellation."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def info(self, msg: str) -> None:  # pragma: no cover - unused in this test
        pass


class _RecordingNode:
    """Minimal stand-in for the rclpy LifecycleNode (the framework boundary).

    The HAL under test is a *real* :class:`SimAttachedHAL`; only the rclpy node
    (a process/framework boundary, CLAUDE.md §1.11) is stubbed, exposing just
    the ``create_timer`` / ``get_logger`` surface ``SimSensorBridge`` touches.
    """

    def __init__(self) -> None:
        self._logger = _RecordingLogger()

    def create_timer(
        self, period: float, callback: object, *, clock: object | None = None
    ) -> _RecordingTimer:
        # Mirror rclpy's ``Node.create_timer`` surface: ADR-0048 Phase 2 passes a
        # SYSTEM_TIME ``clock=`` so the idle stepper runs on wall time. The fake
        # just has to accept (and ignore) it — the timer itself is recorded.
        return _RecordingTimer()

    def get_logger(self) -> _RecordingLogger:
        return self._logger


@_requires_renderer
def test_idle_tick_disables_timer_after_action_dim_mismatch_native_so101() -> None:
    """CONTAINMENT: an idle_step raise → one WARNING, timer cancelled + disabled.

    Force the documented probe gap by constructing the HAL with the wrong
    ``env_action_dim`` (11 vs so101_box's true 6) so ``idle_step`` raises a
    width mismatch on ``env.step``. ``_idle_step_tick`` must contain it: log a
    single warning and disable the idle timer (so it cannot crash-loop the
    graph every tick), not propagate or silently swallow.
    """
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")
    from openral_core import RobotDescription
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml
    from openral_hal.sim_sensor_bridge import SimSensorBridge

    env, seed = build_sim_env_from_yaml(
        "scenes/sim/so101_tube_insertion.yaml", robot_id_fallback="so101_follower"
    )
    desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    hal = SimAttachedHAL(env, desc, env_reset_seed=seed, env_action_dim=11)  # WRONG on purpose
    hal.connect()

    node = _RecordingNode()
    bridge = SimSensorBridge(node, hal, desc, viewer_enabled=False)
    # Wire the idle timer directly (both gates hold: idle_step is callable +
    # so101_box exposes live MuJoCo handles).
    bridge._setup_idle_stepper()
    timer = bridge._idle_timer
    assert timer is not None, "idle timer should have been created (both gates hold)"

    # last_action_ns == 0 → the predicate engages immediately; idle_step raises.
    bridge._idle_step_tick()

    assert timer.cancelled is True
    assert bridge._idle_timer is None
    assert len(node.get_logger().warnings) == 1
    assert "idle stepper disabled" in node.get_logger().warnings[0]

    # A second tick is a no-op (timer already disabled) — no second warning,
    # no crash-loop.
    bridge._idle_step_tick()
    assert len(node.get_logger().warnings) == 1
