"""ObjectsMetadata frame-dimension fields (ADR-0035)."""

from __future__ import annotations

import pytest
from openral_core.schemas import ObjectDetection2D, ObjectsMetadata
from pydantic import ValidationError


def test_objects_metadata_carries_frame_dims_and_roundtrips():
    md = ObjectsMetadata(
        sensor_id="head_rgb",
        detections=[ObjectDetection2D(label="cup", confidence=0.9, bbox_xyxy=(1, 2, 3, 4))],
        model_id="rtdetr",
        frame_width=640,
        frame_height=480,
    )
    assert md.frame_width == 640
    assert md.frame_height == 480
    back = ObjectsMetadata.model_validate_json(md.model_dump_json())
    assert back == md


def test_objects_metadata_rejects_nonpositive_frame_dims():
    with pytest.raises(ValidationError):
        ObjectsMetadata(
            sensor_id="head_rgb",
            detections=[],
            model_id="m",
            frame_width=0,
            frame_height=480,
        )
