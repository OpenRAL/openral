"""Unit tests for ``openral_sim.factory`` and the eval-layer Protocol contracts.

Companion to ``tests/unit/test_eval_registry_and_runner.py`` — that file covers
the registry semantics and the runner's happy path against the mock adapters.
This file pins the **error paths** of ``make_env`` / ``make_policy`` /
``make_robot`` and the **runtime conformance** of the mock adapters against
the ``SimRollout`` and ``PolicyAdapter`` ``runtime_checkable`` Protocols, so a
typo or signature drift in either Protocol is caught at the unit lane rather
than waiting for a sim-test failure.

Coverage
--------
- ``make_env``    raises ``ROSConfigError`` on an unknown scene id and the
  message lists the registered ids (so the runner's CLI shows a helpful hint).
- ``make_policy`` raises ``ROSConfigError`` on an unknown vla id.
- ``make_robot``  returns ``None`` when the robot is not registered and a
  ``RobotDescription`` when it is.
- Mock scene satisfies the :class:`SimRollout` Protocol at runtime.
- Mock zero / random policies satisfy the :class:`PolicyAdapter` Protocol.
- ``EpisodeResult.summary()`` formats the expected fields.
"""

from __future__ import annotations

import pytest
from openral_core import (
    EmbodimentKind,
    JointSpec,
    JointType,
    PhysicsBackend,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SceneSpec,
    SimEnvironment,
    TaskSpec,
    VLASpec,
)
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import ControlMode
from openral_sim import make_env, make_policy
from openral_sim.factory import make_robot
from openral_sim.policy import PolicyAdapter
from openral_sim.registry import ROBOTS
from openral_sim.rollout import EpisodeResult, SimRollout

# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_env(**overrides: object) -> SimEnvironment:
    base: dict[str, object] = {
        "robot_id": "so100_follower",
        "scene": SceneSpec(
            id="mock",
            backend=PhysicsBackend.MOCK,
            backend_options={"action_dim": 7},
        ),
        "task": TaskSpec(id="mock/0", scene_id="mock", instruction="x", max_steps=10),
        "vla": VLASpec(id="zero", weights_uri="mock://noop", extra={"action_dim": 7}),
    }
    base.update(overrides)
    return SimEnvironment(**base)  # type: ignore[arg-type]


# ── make_env / make_policy error paths ──────────────────────────────────────


def test_make_env_unknown_scene_raises_rosconfigerror() -> None:
    env_cfg = _mock_env(
        scene=SceneSpec(id="not_a_real_scene", backend=PhysicsBackend.MOCK),
        task=TaskSpec(id="mock/0", scene_id="not_a_real_scene", instruction="x", max_steps=10),
    )
    with pytest.raises(ROSConfigError) as excinfo:
        make_env(env_cfg)
    assert "scene" in str(excinfo.value)
    assert "not_a_real_scene" in str(excinfo.value)


def test_make_env_error_message_lists_registered_scenes() -> None:
    """Helpful-error contract: the message MUST list the known ids."""
    env_cfg = _mock_env(
        scene=SceneSpec(id="typo", backend=PhysicsBackend.MOCK),
        task=TaskSpec(id="mock/0", scene_id="typo", instruction="x", max_steps=10),
    )
    with pytest.raises(ROSConfigError) as excinfo:
        make_env(env_cfg)
    msg = str(excinfo.value)
    # At least the built-in 'mock' scene must appear in the suggestion list.
    assert "mock" in msg


def test_make_policy_unknown_vla_raises_rosconfigerror() -> None:
    env_cfg = _mock_env(
        vla=VLASpec(id="not_a_real_policy", weights_uri="mock://noop"),
    )
    with pytest.raises(ROSConfigError) as excinfo:
        make_policy(env_cfg)
    assert "policy" in str(excinfo.value)
    assert "not_a_real_policy" in str(excinfo.value)


def test_make_policy_error_message_lists_registered_policies() -> None:
    env_cfg = _mock_env(
        vla=VLASpec(id="typo", weights_uri="mock://noop"),
    )
    with pytest.raises(ROSConfigError) as excinfo:
        make_policy(env_cfg)
    msg = str(excinfo.value)
    # The built-in 'zero' / 'random' policies should be in the suggestion list.
    assert "zero" in msg or "random" in msg


# ── make_robot ───────────────────────────────────────────────────────────────


def test_make_robot_returns_none_for_unregistered_id() -> None:
    """Eval does not require a robot description; unknown ids return None."""
    env_cfg = _mock_env(robot_id="completely-unregistered-robot-xyz")
    assert make_robot(env_cfg) is None


def test_make_robot_returns_description_when_registered() -> None:
    robot_id = "test/factory-robot-fixture"
    desc = RobotDescription(
        name="factory_robot",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name="j0",
                joint_type=JointType.REVOLUTE,
                parent_link="base",
                child_link="link0",
            ),
        ],
        capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        safety=SafetyEnvelope(),
    )

    @ROBOTS.register(robot_id)
    def _factory() -> RobotDescription:
        return desc

    try:
        env_cfg = _mock_env(robot_id=robot_id)
        got = make_robot(env_cfg)
        assert got is not None
        assert got.name == "factory_robot"
    finally:
        # Clean up the registry mutation so other tests are unaffected.
        ROBOTS._items.pop(robot_id, None)  # type: ignore[attr-defined]


# ── Protocol runtime conformance ────────────────────────────────────────────


def test_make_env_returns_object_satisfying_simrollout_protocol() -> None:
    env_cfg = _mock_env()
    sim = make_env(env_cfg)
    assert isinstance(sim, SimRollout)
    # SimRollout is structural; reset / step / render / close must be callable.
    obs = sim.reset(seed=0)
    assert isinstance(obs, dict)


@pytest.mark.parametrize("vla_id", ["zero", "random"])
def test_make_policy_returns_object_satisfying_policyadapter_protocol(vla_id: str) -> None:
    env_cfg = _mock_env(
        vla=VLASpec(
            id=vla_id,
            weights_uri="mock://noop",
            extra={"action_dim": 7, "seed": 0},
        ),
    )
    pol = make_policy(env_cfg)
    assert isinstance(pol, PolicyAdapter)
    assert pol.spec.id == vla_id
    assert pol.device == "cpu"


# ── EpisodeResult.summary() ─────────────────────────────────────────────────


def test_episode_result_summary_format() -> None:
    r = EpisodeResult(
        success=True,
        steps=42,
        total_reward=1.5,
        mean_step_latency_ms=12.34,
        budget_violations=0,
    )
    s = r.summary()
    assert "success=True" in s
    assert "steps=42" in s
    assert "1.500" in s
    assert "12.3ms" in s
    assert "budget_viol=0" in s


def test_episode_result_summary_with_budget_violation() -> None:
    r = EpisodeResult(
        success=False,
        steps=10,
        total_reward=0.0,
        mean_step_latency_ms=0.0,
        budget_violations=3,
    )
    s = r.summary()
    assert "success=False" in s
    assert "budget_viol=3" in s
