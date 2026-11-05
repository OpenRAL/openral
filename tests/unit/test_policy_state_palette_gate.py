"""The palette's state-contract gate follows the runner's ``obs["state"]`` source (issue #474).

A manifest with ``policy_extras.use_world_state_policy_state`` is fed
``WorldState.policy_state`` by the skill runner, not the robot's joint vector, so the
palette must not drop it for a ``state_contract.dim`` that differs from the joint count.
The runner still validates that vector at step time.

Real fixtures only (CLAUDE.md §1.11): the in-tree GR00T N1.7 BEHAVIOR-1K radio skill
(``dim: 61``, flag set) on the real R1Pro (27 joints). The ``kind: wam`` variant is that
manifest re-kinded, as ``test_rskill_manifest_kinds`` does: no in-tree WAM sets the flag.
The only double is the adapter at the ``PolicyAdapter`` protocol boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_core import JointState, RobotDescription, RSkillManifest, VLASpec, WorldState
from openral_core.exceptions import ROSPerceptionStale, ROSRuntimeError
from openral_reasoner_ros.reasoner_node import _state_contract_verdict
from openral_rskill.loader import load_rskill_manifest

_ROOT = Path(__file__).resolve().parents[2]
_RADIO = _ROOT / "rskills" / "gr00t-n17-b1k-turning-on-radio"
_R1PRO = _ROOT / "robots" / "r1pro" / "robot.yaml"


def _radio(kind: str = "vla", *, flag: bool = True) -> RSkillManifest:
    manifest = load_rskill_manifest(str(_RADIO))
    extras = dict(manifest.policy_extras)
    if not flag:
        extras.pop("use_world_state_policy_state")
    return manifest.model_copy(update={"kind": kind, "policy_extras": extras})


def _r1pro() -> RobotDescription:
    return RobotDescription.from_yaml(str(_R1PRO))


def test_fixture_state_dim_differs_from_the_joint_count() -> None:
    manifest, robot = _radio(), _r1pro()
    assert manifest.policy_extras["use_world_state_policy_state"] is True
    assert manifest.state_contract is not None
    assert manifest.state_contract.dim != len(robot.joints)


@pytest.mark.parametrize("kind", ["vla", "wam"])
def test_policy_state_skill_survives_the_palette(kind: str) -> None:
    n_joints = len(_r1pro().joints)
    assert _state_contract_verdict(_radio(kind), n_joints) == "policy_state"


@pytest.mark.parametrize("kind", ["vla", "wam"])
def test_joint_state_dim_mismatch_still_drops(kind: str) -> None:
    n_joints = len(_r1pro().joints)
    assert _state_contract_verdict(_radio(kind, flag=False), n_joints) == "dim_mismatch"
    disabled = _radio(kind, flag=False)
    disabled.policy_extras["use_world_state_policy_state"] = False
    assert _state_contract_verdict(disabled, n_joints) == "dim_mismatch"


def test_raw_contract_without_dim_keeps_the_joint_vector_default() -> None:
    manifest = _radio(flag=False)
    assert manifest.state_contract is not None
    raw = manifest.model_copy(
        update={"state_contract": manifest.state_contract.model_copy(update={"dim": None})}
    )
    assert _state_contract_verdict(raw, len(_r1pro().joints)) == "admit"


def test_registered_assembler_wins_over_the_flag() -> None:
    """The runner's layout assembler runs after the policy_state read and overwrites it."""
    path = _ROOT / "rskills" / "rldx1-ft-rc365-nf4" / "rskill.yaml"
    manifest = RSkillManifest.from_yaml(str(path))
    flagged = manifest.model_copy(
        update={"policy_extras": {**manifest.policy_extras, "use_world_state_policy_state": True}}
    )
    assert _state_contract_verdict(flagged, len(_r1pro().joints)) == "assembler"


class _Adapter:
    """``PolicyAdapter`` boundary double: records the state each step was fed."""

    spec = VLASpec(id="gr00t_b1k", weights_uri=str(_RADIO))
    device = "cpu"

    def __init__(self) -> None:
        self.states: list[NDArray[np.float32]] = []

    def reset(self) -> None:
        return None

    def step(self, observation: dict[str, Any], instruction: str) -> NDArray[np.float32]:
        del instruction
        self.states.append(np.asarray(observation["state"]))
        return np.zeros(23, dtype=np.float32)

    def close(self) -> None:
        return None


def _skill(kind: str) -> tuple[Any, _Adapter, RobotDescription]:
    from openral_rskill_ros.rskill_runner_node import _make_policy_adapter_skill

    adapter, robot = _Adapter(), _r1pro()
    skill = _make_policy_adapter_skill(
        manifest=_radio(kind), adapter=adapter, prompt="turn on the radio", description=robot
    )
    adapter.states.clear()  # drop the activate-time warm-up step
    return skill, adapter, robot


def _world_state(
    robot: RobotDescription, policy_state: list[float] | None, freshness: str = "ok"
) -> WorldState:
    n = len(robot.joints)
    return WorldState(
        stamp_ns=1,
        joint_state=JointState(
            name=[j.name for j in robot.joints],
            position=[0.0] * n,
            velocity=[0.0] * n,
            stamp_ns=1,
        ),
        policy_state=policy_state,
        diagnostics={"policy_state": freshness},
    )


@pytest.mark.parametrize("kind", ["vla", "wam"])
def test_runner_feeds_the_61d_policy_state(kind: str) -> None:
    skill, adapter, robot = _skill(kind)
    skill.step(_world_state(robot, [0.5] * 61))
    assert adapter.states[-1].shape == (61,)


@pytest.mark.parametrize("kind", ["vla", "wam"])
def test_runner_rejects_missing_stale_or_missized_policy_state(kind: str) -> None:
    skill, _, robot = _skill(kind)
    with pytest.raises(ROSRuntimeError, match=r"requires WorldState\.policy_state"):
        skill.step(_world_state(robot, None))
    with pytest.raises(ROSPerceptionStale, match="fresh policy_state"):
        skill.step(_world_state(robot, [0.0] * 61, freshness="stale"))
    wrong_size = f"requires policy_state dim 61, got {len(robot.joints)}"
    with pytest.raises(ROSRuntimeError, match=wrong_size):
        skill.step(_world_state(robot, [0.0] * len(robot.joints)))
