"""Cross-HAL integration test: drive every HAL twin through ``HardwareRunner``.

This is the strongest "all wiring is correct" signal in the repo.  Each
per-HAL ``tests/sim/test_<robot>_hal_mujoco.py`` validates one HAL in
isolation; this suite parametrizes the **production**
:class:`openral_runner.HardwareRunner` over every HAL twin and runs a
real rate-limited inference loop end-to-end:

    activate → for max_ticks:
        sensors → HAL.read_state → aggregator.snapshot →
        skill.step → safety.check_action → HAL.send_action
    → deactivate

If a HAL's joint indexing, units, lifecycle, or schema doesn't line up
with the rest of the stack (``WorldStateAggregator``, ``rSkillBase``,
``SafetyClient``, ``HardwareRunner``), this suite catches it.  Every
per-robot sim test could pass while this one still fails — the
integration boundary is the new contract being exercised here.

No mocks (CLAUDE.md §1.11).  Real ``MujocoArmHAL`` / ``AlohaMujocoHAL``
subclasses, real ``WorldStateAggregator``, real ``HardwareRunner`` with
``NullSafetyClient`` (the same default the production CLI uses for
twins).  The skill is a tiny ``EchoCurrentPoseSkill`` that returns the
robot's current joint state as a 1-step JOINT_POSITION action chunk —
that's always a valid, safety-envelope-clean command for any
embodiment, regardless of joint count, units, or asymmetric
conventions.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

# Module-level guards mirror the per-robot HAL sim tests.  We need both
# ``mujoco`` and ``robot_descriptions`` for the menagerie-backed HALs;
# ALOHA additionally needs ``gym_aloha`` — that check stays inside the
# ALOHA fixture so the other six HALs still run.
#
# Use try/except → boolean + `pytestmark.skipif` rather than module-level
# `pytest.skip(allow_module_level=True)`: with `tests/sim/__init__.py`
# making this directory a Package, a Skipped raised at module-import time
# poisons the whole `tests/sim` Package collection ("found no collectors
# for ..." on every sibling). Deferring the decision to `pytestmark`
# keeps this module importable when optional deps fail.
try:
    import mujoco  # noqa: F401
except Exception as exc:  # pragma: no cover - exercised when mujoco missing
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

try:
    # Probe one menagerie MJCF to confirm robot_descriptions is on disk
    # and the menagerie repo is cloned.  All four MuJoCo-backed HAL twins
    # (UR / Franka / SO-100 / G1 / H1) use the same loader, so probing
    # one short-circuits collection if the optional dep is missing.
    from robot_descriptions import ur5e_mj_description as _menagerie_probe

    _ = _menagerie_probe.MJCF_PATH  # triggers the lazy clone / cache lookup
    _MENAGERIE_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - cold-cache / no network
    _MENAGERIE_ERROR = str(exc)

from openral_core import Action, ControlMode
from openral_core.schemas import WorldState
from openral_hal import (
    AlohaMujocoHAL,
    FrankaPandaHAL,
    G1MujocoHAL,
    H1MujocoHAL,
    SO100FollowerHAL,
    SO100MujocoHAL,
    UR5eHAL,
    UR10eHAL,
)
from openral_hal.protocol import HAL
from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
from openral_rskill.base import rSkillBase
from openral_runner import HardwareRunner
from openral_world_state.aggregator import WorldStateAggregator

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MENAGERIE_ERROR is not None,
        reason=f"robot_descriptions unavailable: {_MENAGERIE_ERROR}",
    ),
]


# ── Trivial skill: echo current pose ─────────────────────────────────────────
# Reads ``world_state.joint_state.position`` and returns it verbatim as a
# 1-step JOINT_POSITION chunk.  This is the smallest skill that's also a
# **valid** skill for every HAL — the units are whatever the HAL's
# ``read_state`` returns (rad for arms, [0, 1] for SO-100 / Franka
# grippers, m for ALOHA fingers, ...).  Echoing the read pose is by
# construction inside every safety envelope, so the safety client never
# trips and we isolate the test signal to "does the wiring work".


class _EchoCurrentPoseSkill(rSkillBase):
    """Read the current joint pose, send it back unchanged.

    Counts ``step()`` invocations so the test can assert the runner is
    driving the skill at every tick.
    """

    def __init__(self, n_joints: int, embodiment_tags: list[str]) -> None:
        super().__init__(name="echo_current_pose", embodiment_tags=embodiment_tags)
        self._n_joints = n_joints
        self.step_count = 0

    def _configure_impl(self) -> None: ...
    def _activate_impl(self) -> None: ...
    def _deactivate_impl(self) -> None: ...
    def _shutdown_impl(self) -> None: ...

    def _step_impl(self, world_state: WorldState) -> Action:
        self.step_count += 1
        # ``world_state.joint_state`` is always populated by the runner
        # before ``skill.step`` — the aggregator's snapshot fans the
        # HAL's just-read JointState into the WorldState before the
        # inference phase (see ``HardwareRunner._tick_impl``).
        js = world_state.joint_state
        assert js is not None, "runner must populate joint_state before step()"
        positions = list(js.position)[: self._n_joints]
        # Pad just in case a future refactor changes the description
        # joint count out from under the skill — the test skill should
        # never be the failure cause.
        while len(positions) < self._n_joints:
            positions.append(0.0)
        return Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[positions],
            confidence=1.0,
            stamp_ns=time.time_ns(),
        )


# ── Per-HAL factories ────────────────────────────────────────────────────────
# Each factory returns a HAL instance with gravity disabled and a small
# ``settle_steps`` budget so the integration test runs fast (the per-robot
# HAL tests already validate closed-loop convergence; this one only needs
# to verify that connect / read / step / send / disconnect form a coherent
# stack with the runner).


def _so100_follower_factory() -> HAL:
    """SO-100 follower over the in-process digital twin (no serial port,
    no MuJoCo).  Validates that lerobot-driven HALs also wire through
    the runner — the kinematic twin is a different code path from the
    MuJoCo-backed ``SO100MujocoHAL``."""
    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    return SO100FollowerHAL(robot=twin)


def _so100_mujoco_factory() -> HAL:
    return SO100MujocoHAL(gravity_enabled=False, settle_steps=10)


def _ur5e_factory() -> HAL:
    return UR5eHAL(gravity_enabled=False, settle_steps=10)


def _ur10e_factory() -> HAL:
    return UR10eHAL(gravity_enabled=False, settle_steps=10)


def _franka_factory() -> HAL:
    return FrankaPandaHAL(gravity_enabled=False, settle_steps=10)


def _aloha_factory() -> HAL:
    try:
        import gym_aloha  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip(
            "gym-aloha not installed; ALOHA twin requires just sync --all-packages --group sim"
        )
    return AlohaMujocoHAL(gravity_enabled=False, settle_steps=10)


def _g1_factory() -> HAL:
    return G1MujocoHAL(gravity_enabled=False, settle_steps=10)


def _h1_factory() -> HAL:
    return H1MujocoHAL(gravity_enabled=False, settle_steps=10)


# ``(label, factory, n_joints, embodiment_tag)`` per HAL.  The label is
# the pytest test id; the embodiment tag must match the description's
# tags so a future ``rSkill.check_compatibility`` call upstream of the
# skill passes (today it's not strictly required, but staying in
# contract keeps the test honest).
_HAL_CASES: list[tuple[str, Callable[[], HAL], int, str]] = [
    ("so100_follower", _so100_follower_factory, 6, "so100_follower"),
    ("so100_mujoco", _so100_mujoco_factory, 6, "so100_follower"),
    ("ur5e", _ur5e_factory, 6, "ur5e"),
    ("ur10e", _ur10e_factory, 6, "ur10e"),
    ("franka_panda", _franka_factory, 8, "franka_panda"),
    ("aloha_bimanual", _aloha_factory, 14, "aloha"),
    ("g1", _g1_factory, 29, "g1"),
    ("h1", _h1_factory, 19, "h1"),
]


# ── The matrix test ──────────────────────────────────────────────────────────
# A SINGLE parametrized test that proves end-to-end runner wiring for
# every HAL.  10 ticks at 30 Hz on a gravity-off twin — long enough to
# exercise the rate-limited loop multiple times, short enough to keep
# the suite under a second per robot in steady state.


@pytest.mark.parametrize(
    "factory, n_joints, embodiment_tag",
    [(f, n, e) for (_, f, n, e) in _HAL_CASES],
    ids=[label for (label, _, _, _) in _HAL_CASES],
)
def test_hal_drives_through_hardware_runner_end_to_end(
    factory: Callable[[], HAL],
    n_joints: int,
    embodiment_tag: str,
) -> None:
    """For each HAL: ``HardwareRunner.run(max_ticks=10)`` completes without
    exception, the skill sees one ``step()`` per tick, and the runner's
    ``RunResult.n_ticks`` matches.  Any breakage in the joint-indexing /
    units / lifecycle / aggregator / safety wiring surfaces here as an
    exception or a step-count mismatch.
    """
    hal = factory()
    aggregator = WorldStateAggregator(hal.description)
    assert len(hal.description.joints) == n_joints, (
        f"Description for {hal.description.name!r} declares "
        f"{len(hal.description.joints)} joints, test expected {n_joints}"
    )

    skill = _EchoCurrentPoseSkill(n_joints=n_joints, embodiment_tags=[embodiment_tag])
    skill.configure()
    skill.activate()

    # Use a rate the cheap settle_steps=10 budget can keep — at 30 Hz
    # each tick has ~33 ms; mj_step iterations on a 29-DoF humanoid
    # take a fraction of that.  No latency budget — this test isn't
    # about timing, it's about wiring.
    runner = HardwareRunner(
        hal=hal,
        skill=skill,
        aggregator=aggregator,
        rate_hz=30.0,
        runner_name=f"hardware_runner.{hal.description.name}",
    )

    max_ticks = 10
    runner.activate()
    try:
        result = runner.run(max_ticks=max_ticks)
    finally:
        runner.deactivate()
        if skill.state.value == "active":
            skill.deactivate()
        if skill.state.value != "finalized":
            skill.shutdown()

    # End-to-end wiring assertions
    assert result.n_ticks == max_ticks, f"runner ran {result.n_ticks} ticks, expected {max_ticks}"
    assert skill.step_count == max_ticks, (
        f"skill.step() was called {skill.step_count} times, expected {max_ticks}"
    )
    # Sanity: aggregated timings are populated and finite.
    assert result.avg_tick_ms > 0
    assert result.p99_tick_ms >= result.avg_tick_ms


# ── Lifecycle-only smoke ─────────────────────────────────────────────────────
# Cheaper sibling that just exercises activate → 1 tick → deactivate.
# Useful when the full 10-tick run is too expensive on a constrained
# host; also serves as the "fast failure" path so a broken HAL shows up
# in the first second of the suite rather than after eight other HALs
# have run.


@pytest.mark.parametrize(
    "factory, n_joints, embodiment_tag",
    [(f, n, e) for (_, f, n, e) in _HAL_CASES],
    ids=[label for (label, _, _, _) in _HAL_CASES],
)
def test_hal_activate_one_tick_deactivate(
    factory: Callable[[], HAL],
    n_joints: int,
    embodiment_tag: str,
) -> None:
    """One-tick smoke: build the full stack, run a single tick, tear down."""
    hal = factory()
    aggregator = WorldStateAggregator(hal.description)
    skill = _EchoCurrentPoseSkill(n_joints=n_joints, embodiment_tags=[embodiment_tag])
    skill.configure()
    skill.activate()
    runner = HardwareRunner(
        hal=hal,
        skill=skill,
        aggregator=aggregator,
        rate_hz=30.0,
        runner_name=f"hardware_runner.{hal.description.name}.smoke",
    )
    runner.activate()
    try:
        result = runner.run(max_ticks=1)
    finally:
        runner.deactivate()
        if skill.state.value == "active":
            skill.deactivate()
        if skill.state.value != "finalized":
            skill.shutdown()

    assert result.n_ticks == 1
    assert skill.step_count == 1
