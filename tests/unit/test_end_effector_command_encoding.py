"""``EndEffectorSpec.command_convention`` / ``command_range`` validation."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from openral_core import GRIPPER_CONVENTION_RANGES, EndEffectorSpec
from pydantic import ValidationError


@pytest.mark.parametrize(("convention", "expected"), sorted(GRIPPER_CONVENTION_RANGES.items()))
def test_normalized_conventions_default_their_range(
    convention: str, expected: tuple[float, float]
) -> None:
    ee = EndEffectorSpec(name="g", kind="parallel_gripper", command_convention=convention)  # type: ignore[arg-type]  # reason: parametrized literal
    assert ee.resolved_command_range() == expected


@pytest.mark.parametrize("convention", ["raw_joint_rad", "width_meters"])
def test_physical_conventions_require_a_range(convention: str) -> None:
    with pytest.raises(ValidationError, match="command_range is required"):
        EndEffectorSpec(name="g", kind="parallel_gripper", command_convention=convention)  # type: ignore[arg-type]  # reason: parametrized literal


def test_range_without_convention_is_refused() -> None:
    with pytest.raises(ValidationError, match="needs command_convention"):
        EndEffectorSpec(name="g", kind="parallel_gripper", command_range=(0.0, 1.0))


@pytest.mark.parametrize("bad", [(1.0, 0.0), (0.5, 0.5), (0.0, float("inf"))])
def test_empty_or_unbounded_range_is_refused(bad: tuple[float, float]) -> None:
    with pytest.raises(ValidationError, match="min < max"):
        EndEffectorSpec(
            name="g", kind="parallel_gripper", command_convention="raw_joint_rad", command_range=bad
        )


def test_range_cannot_exceed_a_normalized_convention() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        EndEffectorSpec(
            name="g",
            kind="parallel_gripper",
            command_convention="normalized_open_unit",
            command_range=(-0.5, 1.0),
        )


def test_undeclared_end_effector_has_no_bound() -> None:
    assert EndEffectorSpec(name="g", kind="parallel_gripper").resolved_command_range() is None


@given(
    lo=st.floats(min_value=-3.0, max_value=3.0),
    width=st.floats(min_value=1e-3, max_value=3.0),
)
def test_physical_range_round_trips(lo: float, width: float) -> None:
    ee = EndEffectorSpec(
        name="jaw",
        kind="parallel_gripper",
        command_convention="raw_joint_rad",
        command_range=(lo, lo + width),
    )
    again = EndEffectorSpec.model_validate_json(ee.model_dump_json())
    assert again == ee
    assert again.resolved_command_range() == (lo, lo + width)
