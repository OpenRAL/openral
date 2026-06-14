"""Unit tests for :data:`openral_core.PerceptionEventMetadata` (ADR-0018 F6).

Real Pydantic models — no mocks, no fixtures fabricated with placeholder
strings (CLAUDE.md §1.11). Each test exercises the actual discriminator
contract that :class:`PerceptionEventPublisher` relies on.
"""

from __future__ import annotations

import pytest
from openral_core import (
    MotionMetadata,
    ObjectDetection2D,
    ObjectsMetadata,
    OcrMetadata,
    PerceptionEventMetadata,
    SceneChangeMetadata,
)
from pydantic import TypeAdapter, ValidationError

ADAPTER = TypeAdapter(PerceptionEventMetadata)


def test_motion_round_trip() -> None:
    """A motion event survives JSON round-trip via the union adapter."""
    src = MotionMetadata(
        sensor_id="wrist_rgb",
        magnitude=0.12,
        threshold=0.02,
        region_bbox=(10, 20, 100, 200),
    )
    decoded = ADAPTER.validate_json(src.model_dump_json())
    assert isinstance(decoded, MotionMetadata)
    assert decoded == src


def test_scene_change_round_trip() -> None:
    """A scene-change event survives JSON round-trip and keeps its metric label."""
    src = SceneChangeMetadata(
        sensor_id="overhead",
        distance=0.73,
        threshold=0.5,
        metric="chisqr_alt",
    )
    decoded = ADAPTER.validate_json(src.model_dump_json())
    assert isinstance(decoded, SceneChangeMetadata)
    assert decoded.metric == "chisqr_alt"
    assert decoded == src


def test_objects_round_trip_preserves_detection_order() -> None:
    """Objects metadata carries the detections list verbatim (incl. order)."""
    src = ObjectsMetadata(
        sensor_id="wrist_rgb",
        model_id="yolov8n",
        detections=[
            ObjectDetection2D(label="cup", confidence=0.91, bbox_xyxy=(0, 0, 50, 50)),
            ObjectDetection2D(label="hand", confidence=0.72, bbox_xyxy=(60, 60, 120, 130)),
        ],
        frame_width=640,
        frame_height=480,
    )
    decoded = ADAPTER.validate_json(src.model_dump_json())
    assert isinstance(decoded, ObjectsMetadata)
    assert [d.label for d in decoded.detections] == ["cup", "hand"]
    assert decoded == src


def test_ocr_round_trip_with_no_region() -> None:
    """OCR metadata accepts an absent region bbox."""
    src = OcrMetadata(
        sensor_id="overhead",
        text="E-STOP",
        confidence=0.98,
        region_bbox=None,
    )
    decoded = ADAPTER.validate_json(src.model_dump_json())
    assert isinstance(decoded, OcrMetadata)
    assert decoded.region_bbox is None
    assert decoded == src


def test_unknown_kind_is_rejected() -> None:
    """A JSON payload whose discriminator is unknown fails validation."""
    payload = '{"kind": "telepathy", "sensor_id": "wrist_rgb"}'
    with pytest.raises(ValidationError):
        ADAPTER.validate_json(payload)


def test_confidence_out_of_range_is_rejected() -> None:
    """Object detection confidence > 1 fails validation (defensive bound)."""
    with pytest.raises(ValidationError):
        ObjectDetection2D(label="cup", confidence=1.5, bbox_xyxy=(0, 0, 1, 1))


def test_motion_threshold_bound() -> None:
    """Motion magnitude/threshold are constrained to [0, 1]."""
    with pytest.raises(ValidationError):
        MotionMetadata(
            sensor_id="wrist_rgb",
            magnitude=1.5,
            threshold=0.02,
            region_bbox=None,
        )


def test_models_are_frozen() -> None:
    """Frozen=True so a consumer can't mutate a routed event in-flight."""
    src = MotionMetadata(
        sensor_id="wrist_rgb",
        magnitude=0.1,
        threshold=0.02,
        region_bbox=None,
    )
    with pytest.raises(ValidationError):
        src.magnitude = 0.5  # type: ignore[misc]  # reason: frozen enforced at runtime


def test_extra_fields_forbidden() -> None:
    """``extra='forbid'`` keeps producers from sneaking ad-hoc fields onto the wire."""
    with pytest.raises(ValidationError):
        MotionMetadata.model_validate(
            {
                "kind": "motion",
                "sensor_id": "wrist_rgb",
                "magnitude": 0.1,
                "threshold": 0.02,
                "region_bbox": None,
                "stash": "should-not-survive",
            },
        )
