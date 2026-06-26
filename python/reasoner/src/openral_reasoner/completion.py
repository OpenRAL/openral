"""ADR-0074 §5 — VLM adjudication helpers (pure, no rclpy dependency).

These functions are imported by ``reasoner_node`` and tested standalone
(without a ROS install) because they carry no rclpy dependency.
"""

from __future__ import annotations

import io

__all__ = [
    "COMPLETION_QUESTION",
    "image_msg_to_jpeg",
    "is_reward_wake",
    "parse_yes_no",
]


def is_reward_wake(*, source: str, severity: int, severity_fail: int) -> bool:
    """Whether a ``FailureTrigger`` is a reward-watcher wake (ADR-0074 §2).

    The reward-watcher rides the ADR-0064 critic path: a ``critic``-source
    trigger at ``SEVERITY_FAIL`` is the "attempt is over" signal (success,
    plateau, or patience) that should preempt — and, while a VLA is in
    flight, stop it *now* rather than at the ``deadline_s`` clock.  A
    non-critic source (hal/sensor/rskill/safety/wam) is an ordinary
    failure, never a reward wake.

    Args:
        source: The failure-bus source name (the topic leaf).
        severity: The trigger's severity.
        severity_fail: The ``SEVERITY_FAIL`` constant to compare against.

    Returns:
        ``True`` iff ``source == "critic"`` and ``severity >= severity_fail``.

    Example:
        >>> is_reward_wake(source="critic", severity=2, severity_fail=2)
        True
        >>> is_reward_wake(source="critic", severity=1, severity_fail=2)
        False
        >>> is_reward_wake(source="sensor", severity=2, severity_fail=2)
        False
    """
    return source == "critic" and severity >= severity_fail

# ADR-0074 §5 — VLM adjudication prompt for the ambiguous reward band.
# Kept short and binary so the provider returns a parseable answer;
# {task!r} is a repr-quoted task string so embedded quotes are escaped.
COMPLETION_QUESTION: str = (
    "Has the robot finished this task: {task!r}?"
    " Look at the scene and answer only 'yes' or 'no'."
)

_NEGATIONS: tuple[str, ...] = (
    "no",
    "not",
    "cannot",
    "can't",
    "isn't",
    "wasn't",
    "hasn't",
    "haven't",
    "doesn't",
    "don't",
    "never",
    "incomplete",
    "not complete",
    "not done",
    "not success",
    "not finished",
)
_AFFIRMATIVES: tuple[str, ...] = ("yes", "complete", "done", "success", "finished")


def parse_yes_no(answer: str) -> bool:
    """Parse a VLM yes/no answer to a boolean.

    Returns ``True`` iff the answer contains a clear affirmative
    (``"yes"``, ``"complete"``, ``"done"``, ``"success"``, or
    ``"finished"``) without an obvious negation prefix (``"no"`` /
    ``"not"`` / ``"cannot"`` / ``"isn't"`` / ``"wasn't"`` / ``"hasn't"``
    / ``"haven't"`` / ``"doesn't"``).  Returns ``False`` on any ambiguous
    or empty input — the default is *not complete* (never a false positive).

    Args:
        answer: Raw text returned by the VLM.

    Returns:
        ``True`` for affirmative; ``False`` for negative or ambiguous.

    Example:
        >>> parse_yes_no("Yes, the task is complete.")
        True
        >>> parse_yes_no("No, the cup is still on the table.")
        False
        >>> parse_yes_no("")
        False
        >>> parse_yes_no("Not done yet.")
        False
        >>> parse_yes_no("Done! The object is placed.")
        True
    """
    lowered = answer.strip().lower()
    if not lowered:
        return False
    for neg in _NEGATIONS:
        # Match at word boundary (space-prefixed or at start) to avoid
        # triggering on "not" inside a word like "annotation".
        if lowered == neg or lowered.startswith(neg + " ") or (" " + neg + " ") in lowered:
            return False
    return any(aff in lowered for aff in _AFFIRMATIVES)


def image_msg_to_jpeg(
    *,
    data: bytes,
    height: int,
    width: int,
    encoding: str,
) -> bytes:
    r"""Convert a raw ``sensor_msgs/Image`` payload to JPEG bytes.

    Supports ``"rgb8"`` and ``"bgr8"`` encodings (the two common camera
    encodings used by the OpenRAL HAL).  No ``cv_bridge`` — uses
    ``numpy`` + ``PIL`` so the conversion works without a full ROS
    install.

    Args:
        data: The raw pixel bytes from ``sensor_msgs/Image.data``.
        height: Image height in pixels.
        width: Image width in pixels.
        encoding: Pixel encoding, e.g. ``"rgb8"`` or ``"bgr8"``.

    Returns:
        JPEG-encoded bytes.

    Raises:
        ValueError: When ``encoding`` is not ``"rgb8"`` or ``"bgr8"``.
        Exception: Propagates any numpy / PIL error so the caller can
            log and ignore (never cache a bad frame).

    Example:
        >>> data = bytes([128, 0, 64]) * 4  # 2x2 rgb8
        >>> jpeg = image_msg_to_jpeg(data=data, height=2, width=2, encoding="rgb8")
        >>> jpeg[:2]  # JPEG SOI marker
        b'\xff\xd8'
    """
    import numpy as np  # noqa: PLC0415  # reason: deferred — keep node import light
    from PIL import Image as _PILImage  # noqa: PLC0415  # reason: deferred — keep node import light

    raw = np.frombuffer(data, dtype=np.uint8).reshape(height, width, -1)
    enc = encoding.lower()
    if enc == "bgr8":
        rgb = raw[..., ::-1].astype(np.uint8)  # BGR -> RGB copy
    elif enc == "rgb8":
        rgb = raw.astype(np.uint8)
    else:
        raise ValueError(
            f"image_msg_to_jpeg: unsupported encoding {encoding!r} (expected rgb8/bgr8)"
        )
    buf = io.BytesIO()
    _PILImage.fromarray(rgb, mode="RGB").save(buf, "JPEG")
    return buf.getvalue()
