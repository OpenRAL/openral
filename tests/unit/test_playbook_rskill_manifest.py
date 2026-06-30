"""Tests for the ``kind: "playbook"`` rSkill manifest variant (ADR-0072).

Covers:
- :class:`~openral_core.schemas.PlaybookContract` Hypothesis round-trip +
  JSON-Schema validation + field guards.
- Contract test: load ``rskills/find-object/rskill.yaml`` via
  :meth:`~openral_core.schemas.RSkillManifest.from_yaml` and assert the key
  guarantees.
- Validator boundary tests: each required/forbidden rule for ``kind=playbook``
  raises :exc:`pydantic.ValidationError` exactly, and a non-playbook kind that
  carries a ``playbook`` block is rejected.

Run with:
    uv run pytest tests/unit/test_playbook_rskill_manifest.py -v
"""

from __future__ import annotations

import copy
import json
import pathlib

import jsonschema
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from openral_core.schemas import PlaybookContract, RSkillManifest
from pydantic import ValidationError

# ─── Fixture paths ─────────────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_FIND_OBJECT_FIXTURE = _REPO_ROOT / "rskills" / "find-object" / "rskill.yaml"
# A real VLA manifest, used to prove a non-playbook kind rejects a playbook block.
_VLA_FIXTURE = _REPO_ROOT / "rskills" / "act-aloha" / "rskill.yaml"


def _load(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text())


def _playbook_manifest_dict() -> dict:
    """A known-good playbook manifest dict, freshly copied per test to mutate."""
    return copy.deepcopy(_load(_FIND_OBJECT_FIXTURE))


# ─── PlaybookContract fuzz + field guards ──────────────────────────────────────

_FUZZ_SETTINGS = settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)

_text = st.text(min_size=1, max_size=200)
_playbook_contract_st = st.builds(
    PlaybookContract,
    trigger=st.text(min_size=1, max_size=500),
    body_uri=_text,
    composes_tools=st.lists(_text, min_size=1, max_size=8),
    done_predicate=st.text(min_size=1, max_size=500),
    max_steps=st.integers(min_value=1, max_value=1000),
)


@_FUZZ_SETTINGS
@given(_playbook_contract_st)
def test_fuzz_playbook_contract(instance: PlaybookContract) -> None:
    """PlaybookContract round-trips through JSON and validates against its schema."""
    serialized = instance.model_dump_json()
    reloaded = PlaybookContract.model_validate_json(serialized)
    assert reloaded == instance, "Round-trip failed for PlaybookContract"
    jsonschema.validate(json.loads(serialized), PlaybookContract.model_json_schema())


def test_playbook_contract_max_steps_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        PlaybookContract(
            trigger="t",
            body_uri="./PLAYBOOK.md",
            composes_tools=["x"],
            done_predicate="d",
            max_steps=0,
        )


def test_playbook_contract_composes_tools_non_empty() -> None:
    with pytest.raises(ValidationError):
        PlaybookContract(
            trigger="t",
            body_uri="./PLAYBOOK.md",
            composes_tools=[],
            done_predicate="d",
            max_steps=4,
        )


def test_playbook_contract_is_frozen() -> None:
    p = PlaybookContract(
        trigger="t",
        body_uri="./PLAYBOOK.md",
        composes_tools=["x"],
        done_predicate="d",
        max_steps=4,
    )
    with pytest.raises(ValidationError):
        p.max_steps = 9  # type: ignore[misc]  # reason: frozen model assignment must raise


def test_playbook_contract_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        PlaybookContract.model_validate(
            {
                "trigger": "t",
                "body_uri": "./PLAYBOOK.md",
                "composes_tools": ["x"],
                "done_predicate": "d",
                "max_steps": 4,
                "bogus": 1,
            }
        )


# ─── Contract test: the real in-tree fixture ───────────────────────────────────


def test_find_object_fixture_loads() -> None:
    m = RSkillManifest.from_yaml(str(_FIND_OBJECT_FIXTURE))
    assert m.kind == "playbook"
    assert m.role == "s2"
    assert m.chunk_size == 1
    assert "plan" in [a.value for a in m.actions]
    assert not m.actuators_required
    assert m.weights_uri is None
    assert m.embodiment_tags == ["any"]  # explicit embodiment-agnostic wildcard
    assert m.playbook is not None
    assert m.playbook.max_steps == 12
    assert "recall_object" in m.playbook.composes_tools


# ─── Validator boundary tests (kind=playbook) ──────────────────────────────────


def test_missing_playbook_block_raises() -> None:
    data = _playbook_manifest_dict()
    del data["playbook"]
    with pytest.raises(ValidationError, match=r"requires a .playbook. block"):
        RSkillManifest.model_validate(data)


def test_non_s2_role_raises() -> None:
    data = _playbook_manifest_dict()
    data["role"] = "s1"
    with pytest.raises(ValidationError, match="requires role='s2'"):
        RSkillManifest.model_validate(data)


def test_missing_plan_action_raises() -> None:
    data = _playbook_manifest_dict()
    data["actions"] = ["navigate"]
    with pytest.raises(ValidationError, match="requires the 'plan' action verb"):
        RSkillManifest.model_validate(data)


def test_chunk_size_not_one_raises() -> None:
    data = _playbook_manifest_dict()
    data["chunk_size"] = 4
    with pytest.raises(ValidationError, match="requires chunk_size=1"):
        RSkillManifest.model_validate(data)


def test_actuators_required_non_empty_raises() -> None:
    data = _playbook_manifest_dict()
    data["actuators_required"] = [
        {"kind": "body_twist", "control_mode_semantics": {"mode": "absolute"}}
    ]
    with pytest.raises(ValidationError, match=r"actuators_required. to be empty"):
        RSkillManifest.model_validate(data)


def test_forbidden_weights_uri_raises() -> None:
    data = _playbook_manifest_dict()
    data["weights_uri"] = "hf://OpenRAL/some-weights"
    with pytest.raises(ValidationError, match="kind='playbook' forbids"):
        RSkillManifest.model_validate(data)


def test_non_playbook_kind_rejects_playbook_block() -> None:
    """A vla manifest carrying a `playbook` block is rejected by the top guard."""
    data = _load(_VLA_FIXTURE)
    assert data["kind"] == "vla"
    data["playbook"] = {
        "trigger": "t",
        "body_uri": "./PLAYBOOK.md",
        "composes_tools": ["x"],
        "done_predicate": "d",
        "max_steps": 4,
    }
    with pytest.raises(ValidationError, match=r"forbids a .playbook. block"):
        RSkillManifest.model_validate(data)


# ─── Embodiment-approval contract (ADR-0072) ───────────────────────────────────


def test_playbook_any_tag_is_embodiment_agnostic() -> None:
    """A playbook declares ``embodiment_tags: ["any"]`` to run on every robot
    (ADR-0072); an EMPTY list is rejected (agnosticism is declared, not derived),
    and declared specific tags still gate. Exercises the real loader gate.
    """
    from openral_core.exceptions import ROSCapabilityMismatch
    from openral_core.schemas import RobotCapabilities
    from openral_rskill.loader import rSkill

    m = RSkillManifest.from_yaml(str(_FIND_OBJECT_FIXTURE))
    assert m.embodiment_tags == ["any"]
    caps = RobotCapabilities(embodiment_tags=["totally_unrelated_robot"])
    # "any" → match-any → must NOT raise on a non-matching robot.
    rSkill.check_embodiment_tags(m, caps)
    # A playbook that declares specific tags is still gated to them.
    tagged = m.model_copy(update={"embodiment_tags": ["so100_follower"]})
    with pytest.raises(ROSCapabilityMismatch):
        rSkill.check_embodiment_tags(tagged, caps)
    # And an EMPTY tag list is rejected at validation time (all kinds).
    data = _playbook_manifest_dict()
    data["embodiment_tags"] = []
    with pytest.raises(ValidationError, match="embodiment_tags must be non-empty"):
        RSkillManifest.model_validate(data)
