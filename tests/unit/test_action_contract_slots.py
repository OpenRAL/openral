"""Unit + property tests for ``ActionContract.slots`` (ADR-0028b).

Covers:

* Per-mode field requirements on :class:`ActionSlot` (ee / frame /
  joint_names presence per control_mode).
* Coverage invariants on :class:`ActionContract`: every index in
  ``[0, dim)`` belongs to exactly one slot, no gaps, no overlaps.
* Discard-slot semantics: no Action emitted, but coverage still
  honoured.
* Round-trip JSON serialisation.
* Property: any partition of ``[0, dim)`` into non-overlapping
  slot ranges + correct per-mode fields validates.

The full slot dispatcher (``rskill_runner_node._step_impl``) is tested
in a sibling module (``test_skill_runner_slot_dispatch.py``) once
ADR-0028b step 3 lands. This file is the schema-side guard.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st
from openral_core import ActionContract, ActionSlot, ControlMode
from pydantic import ValidationError


def _robocasa_slots() -> list[ActionSlot]:
    """Canonical RoboCasa365 pi0.5 layout from ADR-0028b §"Manifest update"."""
    return [
        ActionSlot(
            range=(0, 5),
            control_mode=ControlMode.CARTESIAN_DELTA,
            ee="panda_hand",
            frame="panda_link0",
        ),
        ActionSlot(
            range=(6, 6),
            control_mode=ControlMode.GRIPPER_POSITION,
            ee="panda_gripper",
        ),
        ActionSlot(range=(7, 7), discard=True),
        ActionSlot(
            range=(8, 10),
            control_mode=ControlMode.BODY_TWIST,
            frame="base_link",
        ),
        ActionSlot(range=(11, 11), discard=True),
    ]


# ─── Happy path ──────────────────────────────────────────────────────────────


def test_legacy_action_contract_without_slots_still_works() -> None:
    """Manifests without ``slots`` keep the existing single-Action path."""
    c = ActionContract(dim=7)
    assert c.slots is None


def test_robocasa_12d_layout_validates() -> None:
    """The canonical RoboCasa365 slot block from ADR-0028b validates."""
    c = ActionContract(dim=12, slots=_robocasa_slots())
    assert len(c.slots or []) == 5
    assert c.dim == 12


def test_pure_joint_position_single_slot_layout() -> None:
    """ALOHA-style 14-D pure joint-position layout — one slot, no discard."""
    c = ActionContract(
        dim=14,
        slots=[
            ActionSlot(
                range=(0, 13),
                control_mode=ControlMode.JOINT_POSITION,
                joint_names=[f"j{i}" for i in range(14)],
            )
        ],
    )
    assert c.slots is not None
    assert c.slots[0].range == (0, 13)


def test_round_trip_json_preserves_slots() -> None:
    """``model_dump_json`` ↔ ``model_validate_json`` round-trip is lossless."""
    c = ActionContract(dim=12, slots=_robocasa_slots())
    payload = c.model_dump_json()
    restored = ActionContract.model_validate_json(payload)
    assert restored == c


# ─── ActionSlot per-mode field requirements ──────────────────────────────────


def test_cartesian_delta_requires_ee_and_frame() -> None:
    with pytest.raises(ValidationError, match="ee is required"):
        ActionSlot(range=(0, 5), control_mode=ControlMode.CARTESIAN_DELTA, frame="f")
    with pytest.raises(ValidationError, match="frame is required"):
        ActionSlot(range=(0, 5), control_mode=ControlMode.CARTESIAN_DELTA, ee="ee")


def test_body_twist_requires_frame_forbids_ee() -> None:
    with pytest.raises(ValidationError, match="frame is required"):
        ActionSlot(range=(0, 5), control_mode=ControlMode.BODY_TWIST)
    with pytest.raises(ValidationError, match="ee is forbidden"):
        ActionSlot(
            range=(0, 5),
            control_mode=ControlMode.BODY_TWIST,
            frame="base_link",
            ee="panda_hand",
        )


def test_gripper_position_requires_ee_forbids_frame() -> None:
    with pytest.raises(ValidationError, match="ee is required"):
        ActionSlot(range=(6, 6), control_mode=ControlMode.GRIPPER_POSITION)
    with pytest.raises(ValidationError, match="frame is forbidden"):
        ActionSlot(
            range=(6, 6),
            control_mode=ControlMode.GRIPPER_POSITION,
            ee="panda_gripper",
            frame="panda_link0",
        )


def test_joint_position_forbids_ee_and_frame() -> None:
    with pytest.raises(ValidationError, match="ee is forbidden"):
        ActionSlot(range=(0, 6), control_mode=ControlMode.JOINT_POSITION, ee="x")
    with pytest.raises(ValidationError, match="frame is forbidden"):
        ActionSlot(range=(0, 6), control_mode=ControlMode.JOINT_POSITION, frame="x")


def test_joint_position_joint_names_length_must_match_slot_width() -> None:
    """``joint_names`` length must equal slot width when supplied."""
    with pytest.raises(ValidationError, match="must equal slot width"):
        ActionSlot(
            range=(0, 6),
            control_mode=ControlMode.JOINT_POSITION,
            joint_names=["a", "b"],  # 2 names for a 7-wide slot
        )


def test_joint_position_joint_names_optional_when_omitted() -> None:
    """Empty ``joint_names`` is allowed (runner falls back to robot order)."""
    s = ActionSlot(range=(0, 6), control_mode=ControlMode.JOINT_POSITION)
    assert s.joint_names == []


def test_discard_forbids_everything_else() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        ActionSlot(range=(7, 7), discard=True, control_mode=ControlMode.JOINT_POSITION)
    with pytest.raises(ValidationError, match="forbids ee"):
        ActionSlot(range=(7, 7), discard=True, ee="x")
    with pytest.raises(ValidationError, match="forbids ee"):
        ActionSlot(range=(7, 7), discard=True, frame="x")
    with pytest.raises(ValidationError, match="forbids ee"):
        ActionSlot(range=(7, 7), discard=True, joint_names=["a"])


def test_non_discard_requires_control_mode() -> None:
    with pytest.raises(ValidationError, match="control_mode is required"):
        ActionSlot(range=(0, 0))


def test_range_start_must_not_exceed_end() -> None:
    with pytest.raises(ValidationError, match="start <= end"):
        ActionSlot(range=(5, 3), control_mode=ControlMode.JOINT_POSITION)


def test_range_start_must_be_non_negative() -> None:
    with pytest.raises(ValidationError, match="start must be >= 0"):
        ActionSlot(range=(-1, 3), control_mode=ControlMode.JOINT_POSITION)


# ─── ActionContract coverage invariants ──────────────────────────────────────


def test_slots_must_cover_full_dim() -> None:
    """Gap-coverage is a load-time error."""
    with pytest.raises(ValidationError, match="not covered"):
        ActionContract(
            dim=4,
            slots=[
                ActionSlot(range=(0, 1), control_mode=ControlMode.JOINT_POSITION),
                # missing [2, 3]
            ],
        )


def test_slots_must_not_overlap() -> None:
    with pytest.raises(ValidationError, match="multiple slots"):
        ActionContract(
            dim=4,
            slots=[
                ActionSlot(range=(0, 2), control_mode=ControlMode.JOINT_POSITION),
                ActionSlot(range=(2, 3), control_mode=ControlMode.JOINT_POSITION),
            ],
        )


def test_slot_range_must_fit_within_dim() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        ActionContract(
            dim=4,
            slots=[ActionSlot(range=(0, 5), control_mode=ControlMode.JOINT_POSITION)],
        )


def test_empty_slots_list_rejected() -> None:
    """An empty list silently loses the policy vector — reject."""
    with pytest.raises(ValidationError, match="must be omitted"):
        ActionContract(dim=4, slots=[])


# ─── Property test ───────────────────────────────────────────────────────────

# Hypothesis: build a random partition of [0, dim) into contiguous ranges
# with random control_modes and the right per-mode fields. The result
# must always validate.


@st.composite
def _valid_action_contract(draw: st.DrawFn) -> ActionContract:
    """Generate a syntactically valid ActionContract with slots."""
    dim = draw(st.integers(min_value=1, max_value=32))
    # Random partition: pick a sorted set of cut-points strictly inside
    # (0, dim). When dim == 1 the partition is forced to one slot
    # spanning [0, 0] — no interior cuts possible.
    if dim == 1:
        cuts: list[int] = []
    else:
        n_cuts = draw(st.integers(min_value=0, max_value=min(dim - 1, 6)))
        cuts = sorted(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=dim - 1),
                    min_size=n_cuts,
                    max_size=n_cuts,
                    unique=True,
                )
            )
        )
    bounds = [0, *cuts, dim]
    slots: list[ActionSlot] = []
    for lo, hi_exclusive in itertools.pairwise(bounds):
        hi = hi_exclusive - 1
        # 30% chance of a discard slot; otherwise a random valid mode +
        # the right per-mode fields.
        if draw(st.booleans()) and draw(st.booleans()):
            slots.append(ActionSlot(range=(lo, hi), discard=True))
            continue
        mode = draw(
            st.sampled_from(
                [
                    ControlMode.JOINT_POSITION,
                    ControlMode.CARTESIAN_DELTA,
                    ControlMode.BODY_TWIST,
                    ControlMode.GRIPPER_POSITION,
                ]
            )
        )
        if mode is ControlMode.JOINT_POSITION:
            slots.append(ActionSlot(range=(lo, hi), control_mode=mode))
        elif mode is ControlMode.CARTESIAN_DELTA:
            slots.append(ActionSlot(range=(lo, hi), control_mode=mode, ee="ee", frame="frame"))
        elif mode is ControlMode.BODY_TWIST:
            slots.append(ActionSlot(range=(lo, hi), control_mode=mode, frame="frame"))
        else:  # GRIPPER_POSITION
            slots.append(ActionSlot(range=(lo, hi), control_mode=mode, ee="ee"))
    return ActionContract(dim=dim, slots=slots)


@given(_valid_action_contract())
def test_property_any_valid_partition_validates(contract: ActionContract) -> None:
    """Any valid partition + correct per-mode fields validates."""
    assert contract.slots is not None
    # Verify coverage: every index appears in exactly one slot.
    covered: list[int] = [0] * contract.dim
    for slot in contract.slots:
        lo, hi = slot.range
        for i in range(lo, hi + 1):
            covered[i] += 1
    assert all(n == 1 for n in covered)


@given(_valid_action_contract())
def test_property_round_trip_json(contract: ActionContract) -> None:
    """JSON round-trip is lossless for any valid contract."""
    payload = contract.model_dump_json()
    restored = ActionContract.model_validate_json(payload)
    assert restored == contract
