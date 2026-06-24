"""ADR-0027 — StateContract.bindings + the wrapped-task-space requirement.

Pins the per-layout binding policy: task-space layouts (``human300_16d`` /
``rc365``) require ``bindings``; joint-space layouts
(``smolvla_9d`` / ``gr1`` / ``simpler_*``) forbid them. The reasoner's
admit-with-adapter filter and the runtime state assembler both depend
on this invariant.

No mocks (CLAUDE.md §1.11) — drives real :class:`StateContract` Pydantic
validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import (
    WRAPPED_TASK_SPACE_LAYOUTS,
    RSkillManifest,
    StateContract,
    StateContractBindings,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PI05_MANIFEST = _REPO_ROOT / "rskills" / "pi05-robocasa365-human300-nf4" / "rskill.yaml"


class TestStateContractBindingsValidator:
    def test_task_space_layout_requires_bindings(self) -> None:
        with pytest.raises(ValueError, match="REQUIRES `bindings`"):
            StateContract(layout="human300_16d", dim=16)

    def test_task_space_layout_with_bindings_accepted(self) -> None:
        contract = StateContract(
            layout="human300_16d",
            dim=16,
            bindings=StateContractBindings(
                eef_frame="panda_hand",
                base_frame="base_link",
                world_frame="map",
                gripper_qpos_joints=["panda_finger_joint1", "panda_finger_joint2"],
            ),
        )
        assert contract.bindings is not None
        assert contract.bindings.eef_frame == "panda_hand"

    def test_joint_space_layout_forbids_bindings(self) -> None:
        with pytest.raises(ValueError, match="joint-space layout"):
            StateContract(
                layout="gr1",
                dim=29,
                bindings=StateContractBindings(eef_frame="dummy"),
            )

    def test_joint_space_layout_without_bindings_accepted(self) -> None:
        contract = StateContract(layout="gr1", dim=29)
        assert contract.bindings is None

    def test_none_layout_without_bindings_accepted(self) -> None:
        contract = StateContract(layout=None, dim=None)
        assert contract.bindings is None

    @pytest.mark.parametrize("missing_field", ["eef_frame", "base_frame"])
    def test_human300_requires_eef_and_base(self, missing_field: str) -> None:
        kwargs = {"eef_frame": "panda_hand", "base_frame": "base_link"}
        kwargs[missing_field] = None  # type: ignore[assignment]
        with pytest.raises(ValueError, match="requires bindings"):
            StateContract(
                layout="human300_16d",
                dim=16,
                bindings=StateContractBindings(**kwargs),
            )


class TestStateContractBindingsDefaults:
    def test_world_frame_defaults_to_map(self) -> None:
        bindings = StateContractBindings(eef_frame="ee", base_frame="base")
        assert bindings.world_frame == "map"

    def test_quaternion_convention_defaults_to_xyzw(self) -> None:
        bindings = StateContractBindings(eef_frame="ee", base_frame="base")
        assert bindings.quaternion_convention == "xyzw"

    def test_gripper_joints_defaults_to_empty_list(self) -> None:
        bindings = StateContractBindings(eef_frame="ee", base_frame="base")
        assert bindings.gripper_qpos_joints == []


class TestRSkillManifestRoundTrip:
    def test_pi05_manifest_carries_panda_mobile_bindings(self) -> None:
        """The in-tree pi05 manifest validates AND surfaces the expected
        per-robot bindings — guards against a future PR dropping the
        bindings block and silently re-introducing the 10-vs-16 dim filter."""
        manifest = RSkillManifest.from_yaml(str(_PI05_MANIFEST))
        assert manifest.state_contract is not None
        assert manifest.state_contract.layout == "human300_16d"
        assert manifest.state_contract.bindings is not None
        bindings = manifest.state_contract.bindings
        assert bindings.eef_frame == "panda_hand_tcp"
        assert bindings.base_frame == "base_link"
        assert bindings.gripper_qpos_joints == ["panda_gripper"]


def test_wrapped_set_matches_task_space_definition() -> None:
    """Pins the canonical set so a future PR that adds a new layout to
    StateLayout has to consciously choose joint-space vs task-space."""
    assert {"human300_16d", "rc365", "libero_eef8d"} == WRAPPED_TASK_SPACE_LAYOUTS
    assert "human300_16d" in WRAPPED_TASK_SPACE_LAYOUTS
    assert "rc365" in WRAPPED_TASK_SPACE_LAYOUTS
    assert "libero_eef8d" in WRAPPED_TASK_SPACE_LAYOUTS  # ADR-0027 LIBERO task-space proprio
    # pi0_16d / eef_pose_7d / base_pose_7d were robocasa sim-observation
    # layouts with no state-adapter assembler — removed (to be recreated
    # later); they must not advertise as wrapped-task-space layouts.
    assert "pi0_16d" not in WRAPPED_TASK_SPACE_LAYOUTS
    assert "gr1" not in WRAPPED_TASK_SPACE_LAYOUTS, (
        "gr1 is a 29-D joint-space composite (waist+arms+hands), not a "
        "task-space (FK-derived) layout — putting it in the set would "
        "wrongly require StateContractBindings on every GR1 rSkill."
    )
    assert "smolvla_9d" not in WRAPPED_TASK_SPACE_LAYOUTS
