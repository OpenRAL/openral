"""Tests for the ``kind: "detector"`` rSkill manifest variant (ADR-0037).

Covers:
- :class:`~openral_core.schemas.DetectorContract` Hypothesis round-trip +
  JSON-Schema validation.
- Contract test: load ``rskills/rtdetr-coco-r18/rskill.yaml`` via
  :meth:`~openral_core.schemas.RSkillManifest.model_validate` and assert
  the key guarantees.
- Validator boundary tests: each required/forbidden field rule raises
  :exc:`pydantic.ValidationError` exactly.

Run with:
    uv run pytest tests/unit/test_rskill_detector_kind.py -v
"""

from __future__ import annotations

import copy
import pathlib

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from openral_core.schemas import (
    DetectorContract,
    RSkillAction,
    RSkillManifest,
)
from pydantic import ValidationError

# ─── Fixture path ──────────────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_RTDETR_FIXTURE = _REPO_ROOT / "rskills" / "rtdetr-coco-r18" / "rskill.yaml"

# ─── Fuzz strategy helpers ────────────────────────────────────────────────────

_FUZZ_SETTINGS = settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)

_label_st = st.text(min_size=1, max_size=32)
_pos_int = st.integers(min_value=1, max_value=4096)
_prob = st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1.0)

_detector_contract_st = st.builds(
    DetectorContract,
    labels=st.lists(_label_st, min_size=1, max_size=20),
    input_size=st.tuples(_pos_int, _pos_int),
    score_threshold=_prob,
)


# ─── DetectorContract fuzz ────────────────────────────────────────────────────


@_FUZZ_SETTINGS
@given(_detector_contract_st)
def test_fuzz_detector_contract(instance: DetectorContract) -> None:
    """DetectorContract round-trips through JSON and validates against its schema.

    Property: for any valid DetectorContract, serialise → deserialise round-
    trips losslessly and the serialised dict satisfies the model's own
    JSON Schema.
    """
    import json

    import jsonschema

    serialized = instance.model_dump_json()
    reloaded = DetectorContract.model_validate_json(serialized)
    assert reloaded == instance, "Round-trip failed for DetectorContract"

    data = json.loads(serialized)
    schema = DetectorContract.model_json_schema()
    jsonschema.validate(data, schema)


# ─── Contract test: real fixture ──────────────────────────────────────────────


def test_rtdetr_coco_fixture_validates() -> None:
    """Load rskills/rtdetr-coco-r18/rskill.yaml and assert key invariants.

    This is the canonical contract test for the ``kind: "detector"``
    manifest shape introduced by ADR-0037.  It exercises a real fixture,
    not synthetic data, per CLAUDE.md §1.11.
    """
    assert _RTDETR_FIXTURE.exists(), f"Fixture not found: {_RTDETR_FIXTURE}"

    with open(_RTDETR_FIXTURE, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    manifest = RSkillManifest.model_validate(data)

    assert manifest.kind == "detector"
    assert manifest.detector is not None
    assert len(manifest.detector.labels) >= 1, "labels must be non-empty"
    assert manifest.weights_uri is not None, "detector requires weights_uri"
    assert manifest.actuators_required == [], "detector must have no actuators"
    assert RSkillAction.DETECT in manifest.actions, "detector should declare DETECT action"
    # A detector has no VLA identity fields
    assert manifest.model_family is None
    assert manifest.action_contract is None
    assert manifest.state_contract is None
    assert manifest.ros_integration is None


# ─── Validator boundary tests ─────────────────────────────────────────────────

# Minimal valid detector manifest dict — used as baseline for each mutation.
_VALID_DETECTOR: dict = {
    "schema_version": "0.1",
    "name": "openral/test-detector",
    "version": "0.1.0",
    "license": "apache-2.0",
    "role": "s1",
    "kind": "detector",
    "embodiment_tags": ["franka_panda"],
    "sensors_required": [
        {
            "modality": "rgb",
            "vla_feature_key": "observation.images.camera1",
            "min_width": 640,
            "min_height": 480,
        }
    ],
    "actuators_required": [],
    "runtime": "onnx",
    "weights_uri": "local://rskills/rtdetr-coco-r18",
    "chunk_size": 1,
    "latency_budget": {"per_chunk_ms": 50.0},
    "description": "Test detector for unit tests.",
    "actions": ["detect"],
    "detector": {
        "labels": ["person", "car"],
        "input_size": [640, 640],
        "score_threshold": 0.5,
    },
}


def test_detector_valid_baseline() -> None:
    """The baseline detector dict validates without errors."""
    m = RSkillManifest.model_validate(copy.deepcopy(_VALID_DETECTOR))
    assert m.kind == "detector"
    assert m.detector is not None
    assert m.detector.labels == ["person", "car"]


def test_detector_missing_detector_block_raises() -> None:
    """A ``kind: "detector"`` manifest without a ``detector`` block is rejected."""
    bad = copy.deepcopy(_VALID_DETECTOR)
    del bad["detector"]
    with pytest.raises(ValidationError, match="requires a `detector` block"):
        RSkillManifest.model_validate(bad)


def test_detector_with_model_family_raises() -> None:
    """A ``kind: "detector"`` manifest with ``model_family`` set is rejected."""
    bad = copy.deepcopy(_VALID_DETECTOR)
    bad["model_family"] = "smolvla"
    with pytest.raises(ValidationError, match="model_family"):
        RSkillManifest.model_validate(bad)


def test_detector_with_action_contract_raises() -> None:
    """A ``kind: "detector"`` manifest with ``action_contract`` set is rejected."""
    bad = copy.deepcopy(_VALID_DETECTOR)
    bad["action_contract"] = {"dim": 7}
    with pytest.raises(ValidationError, match="action_contract"):
        RSkillManifest.model_validate(bad)


def test_detector_with_state_contract_raises() -> None:
    """A ``kind: "detector"`` manifest with ``state_contract`` set is rejected."""
    bad = copy.deepcopy(_VALID_DETECTOR)
    bad["state_contract"] = {"dim": 8}
    with pytest.raises(ValidationError, match="state_contract"):
        RSkillManifest.model_validate(bad)


def test_detector_with_processors_raises() -> None:
    """A ``kind: "detector"`` manifest with a ``processors`` block is rejected.

    ``processors`` is a VLA-inference-lifecycle field (the lerobot
    PolicyProcessorPipeline) — meaningless for a perception producer.
    """
    bad = copy.deepcopy(_VALID_DETECTOR)
    bad["processors"] = {
        "preprocessor_uri": "hf://openral/test/policy_preprocessor.json",
        "postprocessor_uri": "hf://openral/test/policy_postprocessor.json",
    }
    with pytest.raises(ValidationError, match="processors"):
        RSkillManifest.model_validate(bad)


def test_detector_with_n_action_steps_raises() -> None:
    """A ``kind: "detector"`` manifest with ``n_action_steps`` set is rejected.

    ``n_action_steps`` is the VLA chunk-replay cadence — a detector emits
    no actions, so there is nothing to replay.
    """
    bad = copy.deepcopy(_VALID_DETECTOR)
    bad["n_action_steps"] = 25
    with pytest.raises(ValidationError, match="n_action_steps"):
        RSkillManifest.model_validate(bad)


def test_detector_with_starting_pose_raises() -> None:
    """A ``kind: "detector"`` manifest with ``starting_pose`` set is rejected.

    ``starting_pose`` is the VLA episode reset pose — a detector has no
    actuators and therefore no pose to start from.
    """
    bad = copy.deepcopy(_VALID_DETECTOR)
    bad["starting_pose"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValidationError, match="starting_pose"):
        RSkillManifest.model_validate(bad)


_JOINT_POS_ACTUATOR = {
    "kind": "joint_position",
    "control_mode_semantics": {"mode": "absolute"},
}


def test_detector_with_actuators_raises() -> None:
    """A ``kind: "detector"`` manifest that lists actuators is rejected."""
    bad = copy.deepcopy(_VALID_DETECTOR)
    bad["actuators_required"] = [_JOINT_POS_ACTUATOR]
    with pytest.raises(ValidationError, match=r"actuates nothing"):
        RSkillManifest.model_validate(bad)


def test_detector_missing_weights_uri_raises() -> None:
    """A ``kind: "detector"`` manifest without ``weights_uri`` is rejected."""
    bad = copy.deepcopy(_VALID_DETECTOR)
    del bad["weights_uri"]
    with pytest.raises(ValidationError, match=r"weights_uri"):
        RSkillManifest.model_validate(bad)


_DETECTOR_BLOCK = {
    "labels": ["person"],
    "input_size": [640, 640],
    "score_threshold": 0.5,
}


def test_non_detector_with_detector_block_raises_vla() -> None:
    """A ``kind: "vla"`` manifest with a ``detector`` block is rejected."""
    bad = {
        "schema_version": "0.1",
        "name": "openral/test-vla",
        "version": "0.1.0",
        "license": "apache-2.0",
        "role": "s1",
        "kind": "vla",
        "model_family": "smolvla",
        "embodiment_tags": ["franka_panda"],
        "actuators_required": [_JOINT_POS_ACTUATOR],
        "runtime": "pytorch",
        "weights_uri": "hf://openral/test",
        "chunk_size": 16,
        "latency_budget": {"per_chunk_ms": 100.0},
        "description": "Test VLA for unit tests.",
        "actions": ["pick"],
        # smolvla requires processors
        "processors": {
            "preprocessor_uri": "hf://openral/test/policy_preprocessor.json",
            "postprocessor_uri": "hf://openral/test/policy_postprocessor.json",
        },
        # Forbidden on vla
        "detector": _DETECTOR_BLOCK,
    }
    with pytest.raises(ValidationError, match="detector"):
        RSkillManifest.model_validate(bad)


def test_non_detector_with_detector_block_raises_ros_action() -> None:
    """A ``kind: "ros_action"`` manifest with a ``detector`` block is rejected."""
    bad = {
        "schema_version": "0.1",
        "name": "openral/test-ros",
        "version": "0.1.0",
        "license": "apache-2.0",
        "role": "s1",
        "kind": "ros_action",
        "embodiment_tags": ["franka_panda"],
        "actuators_required": [_JOINT_POS_ACTUATOR],
        "runtime": "pytorch",
        "chunk_size": 1,
        "latency_budget": {"per_chunk_ms": 5000.0},
        "description": "Test ROS action for unit tests.",
        "actions": ["grasp"],
        "ros_integration": {
            "package": "control_msgs",
            "interface_type": "GripperCommand",
            "interface_name": "/panda_gripper/gripper_action",
            "default_goal_json": '{"command": {"position": 0.0, "max_effort": 20.0}}',
        },
        # Forbidden on ros_action
        "detector": _DETECTOR_BLOCK,
    }
    with pytest.raises(ValidationError, match="detector"):
        RSkillManifest.model_validate(bad)


# ─── DetectorContract edge-case tests ────────────────────────────────────────


def test_detector_contract_requires_at_least_one_label() -> None:
    """DetectorContract with empty labels raises ValidationError."""
    with pytest.raises(ValidationError):
        DetectorContract(labels=[], input_size=(640, 640), score_threshold=0.5)


def test_detector_contract_input_size_must_be_positive() -> None:
    """DetectorContract with non-positive input_size raises ValidationError."""
    with pytest.raises(ValidationError, match="dimensions > 0"):
        DetectorContract(labels=["person"], input_size=(0, 640), score_threshold=0.5)
    with pytest.raises(ValidationError, match="dimensions > 0"):
        DetectorContract(labels=["person"], input_size=(640, -1), score_threshold=0.5)


def test_detector_contract_score_threshold_bounds() -> None:
    """DetectorContract score_threshold must be in [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        DetectorContract(labels=["person"], input_size=(640, 640), score_threshold=-0.1)
    with pytest.raises(ValidationError):
        DetectorContract(labels=["person"], input_size=(640, 640), score_threshold=1.1)
    # Boundary values are valid
    dc_low = DetectorContract(labels=["person"], input_size=(640, 640), score_threshold=0.0)
    assert dc_low.score_threshold == 0.0
    dc_high = DetectorContract(labels=["person"], input_size=(640, 640), score_threshold=1.0)
    assert dc_high.score_threshold == 1.0
