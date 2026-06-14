"""Unit tests for the CPU-tier object detector (ADR-0037 PR5).

All tests run real ONNXRuntime against a real deterministic ONNX fixture built
with ``onnx.helper`` — no mocks, no stubs, per CLAUDE.md §1.11.

The ONNX fixture mimics an RT-DETR / D-FINE export: two ``Constant`` outputs
(logits shape ``(1, 3, 4)`` and boxes shape ``(1, 3, 4)`` cxcywh normalised)
plus a trivial ``Identity`` consuming the ``images`` input (required so
OnnxRuntime accepts the model despite the unused input).

Fixture constants:
    - 3 queries, 4 classes: ``["person", "bicycle", "car", "dog"]``
    - q0 logits ``[-5, -5, 3.0, -5]``   → car, sigmoid(3)  ≈ 0.953 (above 0.5)
    - q1 logits ``[2.0, -5, -5, -5]``   → person, sigmoid(2) ≈ 0.881 (above 0.5)
    - q2 logits ``[-5, -5, -5, -5]``    → max≈0.007 (below 0.5)
    - boxes q0 cxcywh ``[0.5, 0.5, 0.2, 0.4]``     → xyxy on 640×480: (256,144,384,336)
    - boxes q1 cxcywh ``[0.25, 0.25, 0.1, 0.1]``   → xyxy on 640×480: (128, 96,192,144)
    - boxes q2 cxcywh ``[0.8, 0.8, 0.1, 0.1]``     → filtered (below threshold)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
onnx_mod = pytest.importorskip("onnx")

import onnx  # noqa: E402  # reason: after importorskip guard
import onnx.helper as h  # noqa: E402  # reason: after importorskip guard
import onnx.numpy_helper as nph  # noqa: E402  # reason: after importorskip guard
from openral_core import (  # noqa: E402  # reason: after importorskip guard
    ObjectDetection2D,
    ObjectsMetadata,
)
from openral_core.exceptions import ROSConfigError  # noqa: E402  # reason: after importorskip guard
from openral_runner.backends.gstreamer.objects_detector import (  # noqa: E402  # reason: after importorskip guard
    DetectorTier,
    ObjectsDetector,
    identify_rtdetr_outputs,
    make_objects_detector,
    postprocess_rtdetr,
    select_detector_tier,
)
from openral_runner.backends.gstreamer.pipeline import (  # noqa: E402  # reason: after importorskip guard
    Platform,
    inspect_element_present,
)

# ── Labels used by the ONNX fixtures ─────────────────────────────────────────

# 4-class fixture: logits and boxes are both (1, N, 4) → exercises the
# index-order tiebreaker (shapes are ambiguous when num_classes == 4).
_LABELS = ["person", "bicycle", "car", "dog"]

# 5-class fixture: logits (1, N, 5) ≠ boxes (1, N, 4) → exercises the real
# production shape-based identification (last_dim != 4 → logits). This is the
# path every COCO/80-class model hits.
_LABELS_5 = ["person", "bicycle", "car", "dog", "cat"]

# ── ONNX fixture helper ───────────────────────────────────────────────────────


def _write_rtdetr_like_onnx(path: pathlib.Path) -> None:
    """Write a deterministic RT-DETR-like ONNX to *path*.

    The model has one input (``images`` float32 ``[1,3,640,640]``) and three
    outputs:

    * ``logits``  float32 ``(1, 3, 4)`` — constant pre-sigmoid class scores.
    * ``boxes``   float32 ``(1, 3, 4)`` — constant cxcywh normalised.
    * ``images_passthrough``  float32 ``(1,3,640,640)`` — identity of the
      input, added so ONNXRuntime accepts the model despite the ``Constant``
      nodes not referencing ``images``.
    """
    # Logits: shape (1, 3, 4) — 3 queries, 4 classes.
    logits_data = np.array(
        [[[-5.0, -5.0, 3.0, -5.0], [2.0, -5.0, -5.0, -5.0], [-5.0, -5.0, -5.0, -5.0]]],
        dtype=np.float32,
    )
    # Boxes: cxcywh normalised [0, 1], shape (1, 3, 4).
    boxes_data = np.array(
        [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]],
        dtype=np.float32,
    )

    logits_tensor = nph.from_array(logits_data, name="logits_const")
    boxes_tensor = nph.from_array(boxes_data, name="boxes_const")

    images_input = h.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 640, 640])
    logits_out = h.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 3, 4])
    boxes_out = h.make_tensor_value_info("boxes", onnx.TensorProto.FLOAT, [1, 3, 4])
    passthrough_out = h.make_tensor_value_info(
        "images_passthrough", onnx.TensorProto.FLOAT, [1, 3, 640, 640]
    )

    # Consume images via Identity so ORT accepts the input.
    id_node = h.make_node("Identity", inputs=["images"], outputs=["images_passthrough"])
    logits_node = h.make_node("Constant", inputs=[], outputs=["logits"], value=logits_tensor)
    boxes_node = h.make_node("Constant", inputs=[], outputs=["boxes"], value=boxes_tensor)

    graph = h.make_graph(
        nodes=[id_node, logits_node, boxes_node],
        name="rtdetr_test",
        inputs=[images_input],
        outputs=[logits_out, boxes_out, passthrough_out],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _write_rtdetr_like_onnx_5class(path: pathlib.Path) -> None:
    """Write a deterministic RT-DETR-like ONNX with 5 classes to *path*.

    Unlike :func:`_write_rtdetr_like_onnx`, ``num_classes`` is 5 so the logits
    output is ``(1, 2, 5)`` while boxes stays ``(1, 2, 4)`` — the two outputs
    are now distinguishable purely by shape (``last_dim != 4 → logits``), which
    is the production identification path. Output order is deliberately
    boxes-then-logits to prove ordering is irrelevant.

    * q0 logits ``[-5, -5, 4.0, -5, -5]``  → car (index 2), sigmoid(4) ≈ 0.982
    * q1 logits ``[-5, -5, -5, -5, -5]``   → max ≈ 0.007 (below 0.5)
    * q0 box cxcywh ``[0.5, 0.5, 0.4, 0.2]`` → xyxy on 800×600: (240, 240, 560, 360)
    * q1 box cxcywh ``[0.1, 0.1, 0.05, 0.05]`` (filtered by threshold)
    """
    logits_data = np.array(
        [[[-5.0, -5.0, 4.0, -5.0, -5.0], [-5.0, -5.0, -5.0, -5.0, -5.0]]],
        dtype=np.float32,
    )
    boxes_data = np.array(
        [[[0.5, 0.5, 0.4, 0.2], [0.1, 0.1, 0.05, 0.05]]],
        dtype=np.float32,
    )

    logits_tensor = nph.from_array(logits_data, name="logits_const")
    boxes_tensor = nph.from_array(boxes_data, name="boxes_const")

    images_input = h.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 640, 640])
    logits_out = h.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 2, 5])
    boxes_out = h.make_tensor_value_info("boxes", onnx.TensorProto.FLOAT, [1, 2, 4])
    passthrough_out = h.make_tensor_value_info(
        "images_passthrough", onnx.TensorProto.FLOAT, [1, 3, 640, 640]
    )

    id_node = h.make_node("Identity", inputs=["images"], outputs=["images_passthrough"])
    logits_node = h.make_node("Constant", inputs=[], outputs=["logits"], value=logits_tensor)
    boxes_node = h.make_node("Constant", inputs=[], outputs=["boxes"], value=boxes_tensor)

    graph = h.make_graph(
        nodes=[id_node, boxes_node, logits_node],
        name="rtdetr_test_5class",
        inputs=[images_input],
        # boxes listed before logits to prove output order doesn't matter.
        outputs=[boxes_out, logits_out, passthrough_out],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Write the test ONNX fixture once per test module and return its path."""
    p = tmp_path_factory.mktemp("onnx") / "rtdetr_test.onnx"
    _write_rtdetr_like_onnx(p)
    return p


@pytest.fixture(scope="module")
def detector(onnx_path: pathlib.Path) -> ObjectsDetector:
    """A real ``ObjectsDetector`` backed by the test ONNX fixture."""
    return ObjectsDetector(
        onnx_path,
        labels=_LABELS,
        model_id="rtdetr-test",
        input_size=(640, 640),
        score_threshold=0.5,
    )


@pytest.fixture(scope="module")
def onnx_path_5class(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Write the 5-class ONNX fixture once per test module and return its path."""
    p = tmp_path_factory.mktemp("onnx5") / "rtdetr_test_5class.onnx"
    _write_rtdetr_like_onnx_5class(p)
    return p


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestObjectsDetectorDetect:
    """Tests for :meth:`ObjectsDetector.detect`."""

    def test_detect_returns_thresholded_detections(self, detector: ObjectsDetector) -> None:
        """Two queries pass the 0.5 threshold; q2 (≈0.007) is filtered out."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8).tobytes()
        md = detector.detect(frame, 640, 480, "head_rgb")

        assert md is not None
        assert isinstance(md, ObjectsMetadata)
        assert md.kind == "objects"
        assert md.sensor_id == "head_rgb"
        assert md.model_id == "rtdetr-test"
        assert len(md.detections) == 2

        labels = {d.label for d in md.detections}
        assert labels == {"car", "person"}

        # q2 (max ≈0.007) must not appear.
        assert all(d.label != "dog" for d in md.detections)
        assert all(d.label != "bicycle" for d in md.detections)

        # Sorted descending by confidence: car (≈0.953) before person (≈0.881).
        assert md.detections[0].label == "car"
        assert md.detections[1].label == "person"
        assert md.detections[0].confidence > md.detections[1].confidence

        # bbox_xyxy for q0 (car): cxcywh=[0.5,0.5,0.2,0.4] on 640×480
        # → x:[0.4,0.6]×640=[256,384], y:[0.3,0.7]×480=[144,336]
        car = md.detections[0]
        assert abs(car.bbox_xyxy[0] - 256) <= 1  # x_min
        assert abs(car.bbox_xyxy[1] - 144) <= 1  # y_min
        assert abs(car.bbox_xyxy[2] - 384) <= 1  # x_max
        assert abs(car.bbox_xyxy[3] - 336) <= 1  # y_max

        # bbox_xyxy for q1 (person): cxcywh=[0.25,0.25,0.1,0.1] on 640×480
        # → x:[0.2,0.3]×640=[128,192], y:[0.2,0.3]×480=[96,144]
        person = md.detections[1]
        assert abs(person.bbox_xyxy[0] - 128) <= 1  # x_min
        assert abs(person.bbox_xyxy[1] - 96) <= 1  # y_min
        assert abs(person.bbox_xyxy[2] - 192) <= 1  # x_max
        assert abs(person.bbox_xyxy[3] - 144) <= 1  # y_max

    def test_detect_shape_based_identification_5class(self, onnx_path_5class: pathlib.Path) -> None:
        """5-class model: logits (1,N,5) ≠ boxes (1,N,4) → shape-based id, not tiebreaker.

        Output order is boxes-then-logits in the fixture; the detector must
        still identify logits by ``last_dim != 4`` and return the single
        above-threshold "car" detection with the correct bbox. This is the
        production path every COCO/80-class model exercises.
        """
        det = ObjectsDetector(
            onnx_path_5class,
            labels=_LABELS_5,
            model_id="rtdetr-5class",
            input_size=(640, 640),
            score_threshold=0.5,
        )
        # The detector must have identified logits (last_dim 5) vs boxes (last_dim 4)
        # by shape. Output order is boxes-then-logits, so the index-order tiebreaker
        # would mislabel them (logits<-boxes); these asserts pass ONLY via shape id.
        assert det._logits_name == "logits"
        assert det._boxes_name == "boxes"

        frame = np.zeros((600, 800, 3), dtype=np.uint8).tobytes()
        md = det.detect(frame, 800, 600, "front_cam")

        assert md is not None
        assert isinstance(md, ObjectsMetadata)
        assert md.model_id == "rtdetr-5class"
        # Only q0 (car, sigmoid(4)≈0.982) is above threshold; q1 (≈0.007) filtered.
        assert len(md.detections) == 1
        car = md.detections[0]
        assert car.label == "car"
        assert car.confidence > 0.5  # above the 0.5 threshold (sigmoid(4)≈0.982)

        # bbox for q0: cxcywh=[0.5,0.5,0.4,0.2] on 800×600
        # → x:[0.3,0.7]×800=[240,560], y:[0.4,0.6]×600=[240,360]
        assert abs(car.bbox_xyxy[0] - 240) <= 1  # x_min
        assert abs(car.bbox_xyxy[1] - 240) <= 1  # y_min
        assert abs(car.bbox_xyxy[2] - 560) <= 1  # x_max
        assert abs(car.bbox_xyxy[3] - 360) <= 1  # y_max

    def test_detect_wrong_frame_size_returns_none(self, detector: ObjectsDetector) -> None:
        """A buffer whose length ≠ width*height*3 must return ``None``."""
        bad_frame = b"\x00" * (640 * 480 * 3 - 1)  # one byte too short
        result = detector.detect(bad_frame, 640, 480, "head_rgb")
        assert result is None

    def test_detect_all_below_threshold_returns_none(self, onnx_path: pathlib.Path) -> None:
        """With threshold=0.99 all three queries are filtered → ``None``."""
        det = ObjectsDetector(
            onnx_path,
            labels=_LABELS,
            model_id="rtdetr-test",
            score_threshold=0.99,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8).tobytes()
        result = det.detect(frame, 640, 480, "head_rgb")
        assert result is None


class TestObjectsDetectorSummarise:
    """Tests for :meth:`ObjectsDetector.summarise`."""

    def test_summarise_counts_labels(self, detector: ObjectsDetector) -> None:
        """Counter aggregation: 2x car + 1x person, and sensor_id in output."""
        md = ObjectsMetadata(
            sensor_id="head_rgb",
            detections=[
                ObjectDetection2D(label="car", confidence=0.9, bbox_xyxy=(0, 0, 10, 10)),
                ObjectDetection2D(label="car", confidence=0.85, bbox_xyxy=(20, 20, 30, 30)),
                ObjectDetection2D(label="person", confidence=0.7, bbox_xyxy=(50, 50, 60, 60)),
            ],
            model_id="rtdetr-test",
            frame_width=640,
            frame_height=480,
        )
        summary = detector.summarise(md)
        assert "2x car" in summary
        assert "1x person" in summary
        assert "on head_rgb" in summary

    def test_summarise_empty_detections(self, detector: ObjectsDetector) -> None:
        """Empty detection list returns graceful 'none' string."""
        md = ObjectsMetadata(
            sensor_id="front_cam",
            detections=[],
            model_id="rtdetr-test",
            frame_width=640,
            frame_height=480,
        )
        summary = detector.summarise(md)
        assert "none" in summary
        assert "front_cam" in summary

    def test_summarise_wrong_type_raises(self, detector: ObjectsDetector) -> None:
        """Passing a non-ObjectsMetadata raises TypeError."""
        from openral_core import MotionMetadata

        wrong = MotionMetadata(sensor_id="x", magnitude=0.1, threshold=0.02)
        with pytest.raises(TypeError):
            detector.summarise(wrong)


class TestTierSelection:
    """Tests for :func:`select_detector_tier` and :func:`make_objects_detector`."""

    def test_select_tier_cpu_without_nvinfer(self) -> None:
        """On CPU_ONLY platform, tier is CPU_ONNX — unless nvinfer is present."""
        if inspect_element_present("nvinfer"):
            pytest.skip("nvinfer is present on this host — NVINFER tier would be returned")
        result = select_detector_tier(platform=Platform.CPU_ONLY)
        assert result is DetectorTier.CPU_ONNX

    def test_make_detector_cpu_default(self, onnx_path: pathlib.Path) -> None:
        """Passing tier=CPU_ONNX explicitly returns an ObjectsDetector."""
        det = make_objects_detector(
            onnx_path,
            labels=_LABELS,
            model_id="m",
            tier=DetectorTier.CPU_ONNX,
        )
        assert isinstance(det, ObjectsDetector)

    def test_make_detector_nvinfer_raises(self, onnx_path: pathlib.Path) -> None:
        """Requesting NVINFER raises ROSConfigError (spike-gated PR D)."""
        with pytest.raises(ROSConfigError, match="nvinfer"):
            make_objects_detector(
                onnx_path,
                labels=_LABELS,
                model_id="m",
                tier=DetectorTier.NVINFER,
            )

    def test_make_detector_nvmm_aggregator_dispatches_to_nvmm(self, tmp_path: pathlib.Path) -> None:
        """NVMM_AGGREGATOR dispatches to NvmmObjectsDetector, not the tier guard.

        Point at a non-existent onnx so ``NvmmObjectsDetector.__init__``'s own
        ``p.exists()`` guard fires first — host-independent (runs before any
        TRT/GPU work) and positively proves dispatch reached the NVMM detector.
        """
        missing = tmp_path / "does_not_exist.onnx"
        with pytest.raises(ROSConfigError, match="NvmmObjectsDetector"):
            make_objects_detector(
                missing, labels=["a"], model_id="m", tier=DetectorTier.NVMM_AGGREGATOR
            )


class TestEventDetectorProtocol:
    """Verify ObjectsDetector satisfies the EventDetector protocol shape."""

    def test_objects_detector_satisfies_event_detector_shape(
        self, detector: ObjectsDetector
    ) -> None:
        """ObjectsDetector must have kind='objects', callable detect, callable summarise."""
        assert detector.kind == "objects"
        assert callable(detector.detect)
        assert callable(detector.summarise)


class TestConstructorValidation:
    """Tests for constructor validation in ObjectsDetector."""

    def test_missing_onnx_file_raises(self, tmp_path: pathlib.Path) -> None:
        """Non-existent ONNX path raises ROSConfigError."""
        with pytest.raises(ROSConfigError, match="not found"):
            ObjectsDetector(
                tmp_path / "does_not_exist.onnx",
                labels=_LABELS,
                model_id="m",
            )

    def test_invalid_score_threshold_raises(self, onnx_path: pathlib.Path) -> None:
        """Score threshold outside [0,1] raises ROSConfigError."""
        with pytest.raises(ROSConfigError, match="score_threshold"):
            ObjectsDetector(onnx_path, labels=_LABELS, model_id="m", score_threshold=1.5)

    def test_empty_labels_raises(self, onnx_path: pathlib.Path) -> None:
        """Empty labels list raises ROSConfigError."""
        with pytest.raises(ROSConfigError, match="labels"):
            ObjectsDetector(onnx_path, labels=[], model_id="m")


def test_identify_rtdetr_outputs_by_last_dim_four():
    """The 3-D output whose last dim is 4 is boxes; the other is logits."""
    logits_name, boxes_name = identify_rtdetr_outputs(
        [("logits", (1, 300, 80)), ("boxes", (1, 300, 4))]
    )
    assert (logits_name, boxes_name) == ("logits", "boxes")


def test_identify_rtdetr_outputs_index_order_tiebreak():
    """When num_classes==4 both outputs end in 4 → index order (0=logits,1=boxes)."""
    logits_name, boxes_name = identify_rtdetr_outputs([("a", (1, 300, 4)), ("b", (1, 300, 4))])
    assert (logits_name, boxes_name) == ("a", "b")


def test_postprocess_rtdetr_thresholds_and_scales():
    """One query above threshold → one ObjectDetection2D in pixel coords."""
    logits = np.array([[[-9.0, 9.0], [-9.0, -9.0]]], dtype=np.float32)  # (1,2,2)
    boxes = np.array([[[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]], dtype=np.float32)
    md = postprocess_rtdetr(
        logits,
        boxes,
        labels=["bg", "car"],
        model_id="rtdetr-test",
        sensor_id="head_rgb",
        score_threshold=0.5,
        frame_width=100,
        frame_height=100,
    )
    assert md is not None
    assert len(md.detections) == 1
    d = md.detections[0]
    assert d.label == "car"
    assert d.bbox_xyxy == (25, 25, 75, 75)
    assert d.confidence > 0.99


def test_postprocess_rtdetr_returns_none_when_empty():
    """All queries below threshold → postprocess_rtdetr returns None."""
    logits = np.array([[[-9.0, -9.0]]], dtype=np.float32)
    boxes = np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32)
    md = postprocess_rtdetr(
        logits,
        boxes,
        labels=["bg", "car"],
        model_id="m",
        sensor_id="s",
        score_threshold=0.5,
        frame_width=10,
        frame_height=10,
    )
    assert md is None


def test_identify_rtdetr_outputs_raises_with_one_three_d_output():
    """Fewer than two 3-D outputs → ROSConfigError."""
    from openral_core.exceptions import ROSConfigError

    with pytest.raises(ROSConfigError, match="three-dimensional"):
        identify_rtdetr_outputs([("only", (1, 300, 80)), ("img", (1, 3, 640, 640))])


def test_make_objects_detector_nvinfer_still_raises(tmp_path):
    """nvinfer tier is PR D / spike-gated → typed config error mentioning nvinfer."""
    from openral_core.exceptions import ROSConfigError
    from openral_runner.backends.gstreamer.objects_detector import (
        DetectorTier,
        make_objects_detector,
    )

    onnx = tmp_path / "x.onnx"
    onnx.write_bytes(b"\x00")
    with pytest.raises(ROSConfigError, match="nvinfer"):
        make_objects_detector(onnx, labels=["a"], model_id="m", tier=DetectorTier.NVINFER)
