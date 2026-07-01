"""TDD tests for ADR-0074 §5 completion-adjudication helpers.

Three test sections:

1. **Pure** (no ROS): ``parse_yes_no`` truth table and ``image_msg_to_jpeg``
   encode/decode — importable from ``openral_reasoner.completion`` without a
   ROS install.

2. **Adjudication flow** (no full node): ``ReasonerNode._adjudicate_completion``
   via a thin fake object that provides the two attrs (``_latest_completion_frame``,
   ``_tool_use_client``) and a trivial ``get_logger()``. Skipped if rclpy is absent.

3. **DRY faithfulness** (no full node): confirms ``_complete_active_and_advance``
   exists and is callable on the node class (structural check). Skipped if rclpy
   is absent.
"""

from __future__ import annotations

import io

import pytest
from openral_reasoner.completion import (
    COMPLETION_QUESTION,
    image_msg_to_jpeg,
    is_frame_fresh,
    is_reward_wake,
    parse_yes_no,
    resolve_band_edges,
    resolve_patience_s,
)

# Optional ROS import — guarded so pure tests always run.
try:
    import openral_msgs as _openral_msgs  # noqa: F401  # reason: availability-check only
    import rclpy as _rclpy  # noqa: F401  # reason: availability-check only
    from openral_reasoner_ros.reasoner_node import ReasonerNode as _ReasonerNode

    _HAS_ROS = True
except ImportError:
    _HAS_ROS = False
    _ReasonerNode = None  # type: ignore[assignment, misc]

_ROS_SKIP = pytest.mark.skipif(not _HAS_ROS, reason="requires rclpy + openral_msgs")


# ── 1. parse_yes_no truth table ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "answer,expected",
    [
        # Clear affirmatives
        ("yes", True),
        ("Yes", True),
        ("YES", True),
        ("Yes, the task is complete.", True),
        ("complete", True),
        ("Complete!", True),
        ("Task complete.", True),
        ("done", True),
        ("Done!", True),
        ("Done. The object is placed.", True),
        ("success", True),
        ("Success — object on coaster.", True),
        ("finished", True),
        ("The task is finished.", True),
        # Clear negatives
        ("no", False),
        ("No", False),
        ("NO", False),
        ("No, not yet.", False),
        ("not yet", False),
        ("Not done yet.", False),
        ("not complete", False),
        ("Not complete.", False),
        ("not success", False),
        ("cannot confirm", False),
        ("can't tell", False),
        ("isn't done", False),
        ("wasn't finished", False),
        ("hasn't been placed", False),
        ("haven't finished", False),
        ("doesn't look done", False),
        ("incomplete", False),
        # Ambiguous / empty → False (default-to-not-complete)
        ("", False),
        ("   ", False),
        ("I'm not sure", False),
        ("maybe", False),
        ("unclear", False),
        ("pending", False),
    ],
)
def test_parse_yes_no(answer: str, expected: bool) -> None:
    """``parse_yes_no`` returns the expected boolean for every entry."""
    assert parse_yes_no(answer) is expected


def test_parse_yes_no_default_false_on_nonsense() -> None:
    """Anything that is not a clear affirmative must return False."""
    assert parse_yes_no("xyzzy plover") is False


# ── 2. image_msg_to_jpeg ────────────────────────────────────────────────────


def _make_raw_rgb(height: int = 2, width: int = 2) -> bytes:
    """2×2 flat-colour raw RGB byte array."""
    pixel = b"\x80\x40\xc0"  # R=128 G=64 B=192
    return pixel * (height * width)


def test_image_msg_to_jpeg_rgb8_produces_jpeg() -> None:
    """A well-formed 2×2 rgb8 buffer converts to a non-empty JPEG."""
    pytest.importorskip("PIL", reason="Pillow not installed")
    pytest.importorskip("numpy", reason="numpy not installed")
    data = _make_raw_rgb(2, 2)
    jpeg = image_msg_to_jpeg(data=data, height=2, width=2, encoding="rgb8")
    # JPEG files start with the SOI marker 0xFFD8
    assert jpeg[:2] == b"\xff\xd8", "Expected JPEG SOI marker"
    assert len(jpeg) > 10


def test_image_msg_to_jpeg_bgr8_produces_jpeg() -> None:
    """A well-formed 2×2 bgr8 buffer converts to a non-empty JPEG (channels swapped)."""
    pytest.importorskip("PIL", reason="Pillow not installed")
    pytest.importorskip("numpy", reason="numpy not installed")
    # bgr8: blue first
    pixel = b"\xc0\x40\x80"  # B=192 G=64 R=128
    data = pixel * 4  # 2×2
    jpeg = image_msg_to_jpeg(data=data, height=2, width=2, encoding="bgr8")
    assert jpeg[:2] == b"\xff\xd8"
    assert len(jpeg) > 10


def test_image_msg_to_jpeg_unsupported_encoding_raises() -> None:
    """An unsupported encoding raises ValueError."""
    pytest.importorskip("PIL", reason="Pillow not installed")
    pytest.importorskip("numpy", reason="numpy not installed")
    data = _make_raw_rgb(2, 2)
    with pytest.raises(ValueError, match="unsupported encoding"):
        image_msg_to_jpeg(data=data, height=2, width=2, encoding="mono8")


def test_image_msg_to_jpeg_malformed_buffer_raises() -> None:
    """Malformed data (wrong size) raises so the callback can leave cache unchanged."""
    pytest.importorskip("PIL", reason="Pillow not installed")
    pytest.importorskip("numpy", reason="numpy not installed")
    # Only 3 bytes for a 2×2 rgb8 image (needs 12) — numpy reshape raises ValueError
    with pytest.raises((ValueError, RuntimeError)):
        image_msg_to_jpeg(data=b"\x00\x01\x02", height=2, width=2, encoding="rgb8")


def test_image_msg_to_jpeg_flip_180_inverts_vertically() -> None:
    """flip_180 rotates the image 180°: a top-bright/bottom-dark frame inverts.

    JPEG is lossy, so assert on region means (robust to compression) rather than
    exact pixels. Without flip the top half is bright; with flip_180 it is dark.
    """
    np = pytest.importorskip("numpy", reason="numpy not installed")
    PIL_Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    h, w = 16, 16
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2, :, :] = 255  # bright top half, dark bottom half
    raw = img.tobytes()

    def _top_mean(jpeg: bytes) -> float:
        arr = np.asarray(PIL_Image.open(io.BytesIO(jpeg)).convert("RGB"))
        return float(arr[: h // 2].mean())

    plain = image_msg_to_jpeg(data=raw, height=h, width=w, encoding="rgb8")
    flipped = image_msg_to_jpeg(data=raw, height=h, width=w, encoding="rgb8", flip_180=True)
    assert _top_mean(plain) > 200.0, "unflipped: top half should stay bright"
    assert _top_mean(flipped) < 55.0, "flip_180: top half should become dark (was bottom)"


@pytest.mark.parametrize(
    "age_s,max_age_s,expected",
    [
        (0.0, 2.0, True),
        (0.5, 2.0, True),
        (2.0, 2.0, True),  # boundary inclusive
        (2.1, 2.0, False),
        (3.0, 2.0, False),
        (999.0, 0.0, True),  # guard disabled
        (999.0, -1.0, True),  # negative also disables
    ],
)
def test_is_frame_fresh(age_s: float, max_age_s: float, expected: bool) -> None:
    """The freshness guard accepts fresh frames, rejects stale ones, off when <=0."""
    assert is_frame_fresh(age_s=age_s, max_age_s=max_age_s) is expected


def test_completion_question_contains_task_placeholder() -> None:
    """COMPLETION_QUESTION formats correctly with a task string."""
    q = COMPLETION_QUESTION.format(task="pick the cup")
    assert "pick the cup" in q
    # The question must prompt for yes/no
    assert "yes" in q.lower() or "no" in q.lower()


# ── 2b. is_reward_wake (the cancel-in-flight predicate, ADR-0074 §2) ─────────


def test_is_reward_wake_critic_fail_is_a_wake() -> None:
    """A critic-source trigger at >= SEVERITY_FAIL is a reward wake."""
    assert is_reward_wake(source="critic", severity=2, severity_fail=2) is True
    assert is_reward_wake(source="critic", severity=3, severity_fail=2) is True


def test_is_reward_wake_below_fail_is_not_a_wake() -> None:
    """A critic trigger below SEVERITY_FAIL (e.g. WARN) is not a wake."""
    assert is_reward_wake(source="critic", severity=1, severity_fail=2) is False


def test_is_reward_wake_non_critic_source_is_never_a_wake() -> None:
    """hal/sensor/rskill/safety/wam failures are ordinary, never reward wakes."""
    for src in ("safety", "hal", "sensor", "rskill", "wam"):
        assert is_reward_wake(source=src, severity=2, severity_fail=2) is False


# ── 2c. resolve_band_edges / resolve_patience_s (ADR-0074 §1/§2/§3) ──────────


def test_resolve_band_edges_prefers_contract() -> None:
    """The reward model's calibration wins over the system fallback."""
    assert resolve_band_edges(
        contract_threshold=0.9, contract_floor=0.6, fallback_threshold=0.8, fallback_floor=0.5
    ) == (0.9, 0.6)


def test_resolve_band_edges_falls_back_when_no_contract() -> None:
    """No contract → system fallback edges."""
    assert resolve_band_edges(
        contract_threshold=None, contract_floor=None, fallback_threshold=0.8, fallback_floor=0.5
    ) == (0.8, 0.5)


def test_resolve_band_edges_half_contract_falls_back() -> None:
    """A half-populated contract never mixes sources — it falls back wholesale."""
    assert resolve_band_edges(
        contract_threshold=0.9, contract_floor=None, fallback_threshold=0.8, fallback_floor=0.5
    ) == (0.8, 0.5)


def test_resolve_patience_authority_stack() -> None:
    """LLM override > reward-model default > legacy deadline_s."""
    # LLM override wins.
    assert resolve_patience_s(override=12.0, contract_default=30.0, legacy_deadline_s=5.0) == 12.0
    # No override → reward model's calibrated default.
    assert resolve_patience_s(override=None, contract_default=30.0, legacy_deadline_s=5.0) == 30.0
    # No override, no contract → the LLM's legacy deadline_s.
    assert resolve_patience_s(override=None, contract_default=None, legacy_deadline_s=5.0) == 5.0
    # No override, no contract, deadline 0 → 0 (runner resolves its own ceiling).
    assert resolve_patience_s(override=None, contract_default=None, legacy_deadline_s=0.0) == 0.0


# ── 3. _adjudicate_completion thin-harness tests (no full node) ─────────────
#
# Call _adjudicate_completion as an unbound method on a minimal fake object
# that supplies just the two required attrs and a no-op logger.  Skipped when
# rclpy / openral_msgs are absent (the import of reasoner_node requires both).


class _NoOpLogger:
    """Minimal logger that satisfies ``node.get_logger().*`` calls."""

    def debug(self, *a: object, **kw: object) -> None:
        pass

    def warning(self, *a: object, **kw: object) -> None:
        pass

    def info(self, *a: object, **kw: object) -> None:
        pass


class _FakeDescribeClient:
    """Minimal ToolUseClient-alike that supports describe_image only."""

    def __init__(self, result: str | Exception) -> None:
        self._result = result

    def describe_image(self, *, image_jpeg: bytes, question: str) -> str:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result  # type: ignore[return-value]


class _FakeAdjudicationNode:
    """Thin fake object accepted by ``_adjudicate_completion`` as ``self``."""

    def __init__(
        self,
        *,
        frame: bytes | None,
        describe_result: str | Exception | None = None,
    ) -> None:
        self._latest_completion_frame = frame
        # None disables the freshness branch in _adjudicate_completion (the guard
        # logic itself is covered by test_is_frame_fresh on the pure helper).
        self._latest_completion_frame_time = None
        self._completion_frame_max_age_s = 0.0
        if describe_result is not None:
            self._tool_use_client: object | None = _FakeDescribeClient(describe_result)
        else:
            self._tool_use_client = None

    def get_logger(self) -> _NoOpLogger:
        return _NoOpLogger()


def _make_jpeg() -> bytes:
    """Produce a minimal real JPEG for the frame-present branches."""
    try:
        from PIL import Image as _PILImage  # reason: conditional import for JPEG fixture
    except ImportError:
        pytest.skip("Pillow not installed")
    buf = io.BytesIO()
    _PILImage.new("RGB", (1, 1), color=(128, 64, 192)).save(buf, "JPEG")
    return buf.getvalue()


@_ROS_SKIP
def test_adjudicate_no_frame_returns_none() -> None:
    """No cached frame → None (cannot adjudicate)."""
    assert _ReasonerNode is not None
    fake = _FakeAdjudicationNode(frame=None, describe_result="yes")
    result = _ReasonerNode._adjudicate_completion(fake, "pick the cup")  # type: ignore[arg-type]
    assert result is None


@_ROS_SKIP
def test_adjudicate_no_client_returns_none() -> None:
    """No VLM client → None (cannot adjudicate)."""
    assert _ReasonerNode is not None
    jpeg = _make_jpeg()
    fake = _FakeAdjudicationNode(frame=jpeg)  # no describe_result → client is None
    result = _ReasonerNode._adjudicate_completion(fake, "pick the cup")  # type: ignore[arg-type]
    assert result is None


@_ROS_SKIP
def test_adjudicate_vlm_yes_returns_true() -> None:
    """VLM answers 'yes' → True."""
    assert _ReasonerNode is not None
    jpeg = _make_jpeg()
    fake = _FakeAdjudicationNode(frame=jpeg, describe_result="yes")
    result = _ReasonerNode._adjudicate_completion(fake, "pick the cup")  # type: ignore[arg-type]
    assert result is True


@_ROS_SKIP
def test_adjudicate_vlm_done_returns_true() -> None:
    """VLM answers 'Done!' → True (affirmative word)."""
    assert _ReasonerNode is not None
    jpeg = _make_jpeg()
    fake = _FakeAdjudicationNode(frame=jpeg, describe_result="Done!")
    result = _ReasonerNode._adjudicate_completion(fake, "place the cup")  # type: ignore[arg-type]
    assert result is True


@_ROS_SKIP
def test_adjudicate_vlm_no_returns_false() -> None:
    """VLM answers 'no' → False."""
    assert _ReasonerNode is not None
    jpeg = _make_jpeg()
    fake = _FakeAdjudicationNode(frame=jpeg, describe_result="no")
    result = _ReasonerNode._adjudicate_completion(fake, "pick the cup")  # type: ignore[arg-type]
    assert result is False


@_ROS_SKIP
def test_adjudicate_vlm_raises_returns_none() -> None:
    """Provider error (describe_image raises) → None, never a false True."""
    assert _ReasonerNode is not None
    jpeg = _make_jpeg()
    fake = _FakeAdjudicationNode(frame=jpeg, describe_result=RuntimeError("timeout"))
    result = _ReasonerNode._adjudicate_completion(fake, "pick the cup")  # type: ignore[arg-type]
    assert result is None


@_ROS_SKIP
def test_adjudicate_vlm_ambiguous_returns_false() -> None:
    """An ambiguous answer → False (default not-complete, never false positive)."""
    assert _ReasonerNode is not None
    jpeg = _make_jpeg()
    fake = _FakeAdjudicationNode(frame=jpeg, describe_result="I cannot tell from the image")
    result = _ReasonerNode._adjudicate_completion(fake, "pick the cup")  # type: ignore[arg-type]
    assert result is False


# ── 4. DRY faithfulness — _complete_active_and_advance exists ───────────────


@_ROS_SKIP
def test_complete_active_and_advance_exists_on_node() -> None:
    """``_complete_active_and_advance`` is present so both paths can reuse it."""
    assert _ReasonerNode is not None
    assert callable(getattr(_ReasonerNode, "_complete_active_and_advance", None))
