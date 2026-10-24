"""VLM adjudication helpers (pure, no rclpy dependency).

These functions are imported by ``reasoner_node`` and tested standalone
(without a ROS install) because they carry no rclpy dependency.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

__all__ = [
    "COMPLETION_QUESTION",
    "CompletionTrigger",
    "image_msg_to_jpeg",
    "is_frame_fresh",
    "is_reward_wake",
    "parse_yes_no",
    "resolve_band_edges",
    "resolve_patience_s",
    "tiled_jpeg",
]


def is_frame_fresh(*, age_s: float, max_age_s: float) -> bool:
    """Whether a cached completion frame is recent enough to adjudicate.

    A stale frame (e.g. one cached from a *previous* attempt while the camera
    topic briefly stalled) would make the VLM judge the wrong scene, so an
    over-age frame is treated as "cannot adjudicate" (degrades to the ladder,
    never a false verdict). ``max_age_s <= 0`` disables the guard (always fresh).

    Args:
        age_s: Seconds since the cached frame was received.
        max_age_s: Maximum tolerated age; ``<= 0`` disables the check.

    Returns:
        ``True`` if the frame may be used; ``False`` if it is too old.

    Example:
        >>> is_frame_fresh(age_s=0.5, max_age_s=2.0)
        True
        >>> is_frame_fresh(age_s=3.0, max_age_s=2.0)
        False
        >>> is_frame_fresh(age_s=999.0, max_age_s=0.0)  # guard disabled
        True
    """
    if max_age_s <= 0.0:
        return True
    return age_s <= max_age_s


def resolve_band_edges(
    *,
    contract_threshold: float | None,
    contract_floor: float | None,
    fallback_threshold: float,
    fallback_floor: float,
) -> tuple[float, float]:
    """Resolve the three-tier verdict band edges.

    The active reward model's calibration wins when present; otherwise the
    system fallback. Both contract values must be present to use the contract
    (a half-populated contract falls back, never mixes sources).

    Args:
        contract_threshold: The reward model's ``success_threshold`` (or None).
        contract_floor: The reward model's ``check_floor`` (or None).
        fallback_threshold: System-default success threshold.
        fallback_floor: System-default check floor.

    Returns:
        ``(success_threshold, check_floor)``.

    Example:
        >>> resolve_band_edges(
        ...     contract_threshold=0.9,
        ...     contract_floor=0.6,
        ...     fallback_threshold=0.8,
        ...     fallback_floor=0.5,
        ... )
        (0.9, 0.6)
        >>> resolve_band_edges(
        ...     contract_threshold=None,
        ...     contract_floor=None,
        ...     fallback_threshold=0.8,
        ...     fallback_floor=0.5,
        ... )
        (0.8, 0.5)
    """
    if contract_threshold is not None and contract_floor is not None:
        return (contract_threshold, contract_floor)
    return (fallback_threshold, fallback_floor)


def resolve_patience_s(
    *,
    override: float | None,
    contract_default: float | None,
    legacy_deadline_s: float,
) -> float:
    """Resolve the patience ceiling for a dispatch.

    Authority stack — system fallback < reward-model default < LLM override:
    the LLM's per-task ``patience_s`` override wins; else the reward model's
    calibrated ``default_patience_s``; else the LLM's legacy ``deadline_s`` time
    budget (``0.0`` → the runner resolves its own ceiling from the manifest).

    Args:
        override: The LLM's ``ExecuteRskillTool.patience_s`` (or None).
        contract_default: The reward model's ``default_patience_s`` (or None).
        legacy_deadline_s: The LLM's legacy ``ExecuteRskillTool.deadline_s``.

    Returns:
        The effective patience in seconds.

    Example:
        >>> resolve_patience_s(override=12.0, contract_default=30.0, legacy_deadline_s=5.0)
        12.0
        >>> resolve_patience_s(override=None, contract_default=30.0, legacy_deadline_s=5.0)
        30.0
        >>> resolve_patience_s(override=None, contract_default=None, legacy_deadline_s=5.0)
        5.0
    """
    if override is not None:
        return float(override)
    if contract_default is not None:
        return float(contract_default)
    return float(legacy_deadline_s)


def is_reward_wake(*, source: str, severity: int, severity_fail: int) -> bool:
    """Whether a ``FailureTrigger`` is a reward-watcher wake.

    The reward-watcher rides the critic path: a ``critic``-source
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


# VLM adjudication prompt for the ambiguous reward band.
# Kept short and binary so the provider returns a parseable answer;
# {task!r} is a repr-quoted task string so embedded quotes are escaped.
COMPLETION_QUESTION: str = (
    "Has the robot finished this task: {task!r}? Look at the scene and answer only 'yes' or 'no'."
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
_AFFIRMATIVES: tuple[str, ...] = (
    "yes",
    "complete",
    "completed",
    "done",
    "success",
    "succeeded",
    "successful",
    "successfully",
    "finished",
)


def parse_yes_no(answer: str) -> bool:
    """Parse a VLM yes/no answer to a boolean.

    Returns ``True`` iff the answer contains a clear affirmative token
    (``"yes"``, ``"complete"``/``"completed"``, ``"done"``,
    ``"success"``/``"succeeded"``/``"successfully"``, or ``"finished"``)
    without an obvious negation (``"no"`` / ``"not"`` / ``"cannot"`` /
    ``"isn't"`` / …). ``False`` on ambiguous or empty input (never a false
    positive). Matching is token-based over punctuation-normalised text —
    substring matching false-completed on ``"No. It is done."`` (the
    ``"No."`` never matched the space-delimited negation) and
    ``"The task was abandoned"`` (``"abandoned"`` contains ``"done"``).

    Args:
        answer: Raw text returned by the VLM.

    Returns:
        ``True`` for affirmative; ``False`` for negative or ambiguous.

    Example:
        >>> parse_yes_no("Yes, the task is complete.")
        True
        >>> parse_yes_no("No, the cup is still on the table.")
        False
        >>> parse_yes_no("No. It is done.")
        False
        >>> parse_yes_no("The task was abandoned.")
        False
        >>> parse_yes_no("")
        False
        >>> parse_yes_no("Not done yet.")
        False
        >>> parse_yes_no("Done! The object is placed.")
        True
        >>> parse_yes_no("The task was completed successfully.")
        True
        >>> parse_yes_no("The task isn\u2019t complete.")  # typographic apostrophe
        False
    """
    lowered = answer.strip().lower()
    if not lowered:
        return False
    # Fold typographic apostrophes (U+2018/U+2019 — the quotes LLMs routinely
    # emit) into ASCII so "isn't" matches the negation tokens; without this
    # the contraction splits to "isn t", the negation is missed, and
    # a typographic "isn't complete" would parse as a FALSE POSITIVE via
    # "complete".
    lowered = lowered.replace("\u2018", "'").replace("\u2019", "'")
    # Normalise punctuation to spaces so "no," / "no." / "done!" tokenize
    # cleanly; keep the apostrophe so contractions ("isn't") stay one token.
    normalised = "".join(c if c.isalnum() or c == "'" else " " for c in lowered)
    padded = f" {' '.join(normalised.split())} "
    if any(f" {neg} " in padded for neg in _NEGATIONS):
        return False
    return any(f" {aff} " in padded for aff in _AFFIRMATIVES)


def image_msg_to_jpeg(
    *,
    data: bytes,
    height: int,
    width: int,
    encoding: str,
    flip_180: bool = False,
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
        flip_180: Rotate the image 180° before encoding. The HAL publishes
            LIBERO/MuJoCo frames bottom-up (the published Image is *raw* — the
            ``OPENRAL_DASHBOARD_FLIP_180`` flip is applied only to the dashboard
            thumbnail, not the topic; ``sim_sensor_bridge``), so a VLM judging
            the scene needs the same 180° correction the dashboard and the VLA
            apply. A 180° rotation = flip both spatial axes; channels untouched.

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
    if flip_180:
        # 180° rotation = reverse rows and columns; ascontiguousarray because the
        # negative-stride view is not C-contiguous and PIL.fromarray needs it.
        rgb = np.ascontiguousarray(rgb[::-1, ::-1])
    buf = io.BytesIO()
    _PILImage.fromarray(rgb, mode="RGB").save(buf, "JPEG")
    return buf.getvalue()


@dataclass
class CompletionTrigger:
    """When to ask a VLM whether the running task is complete (2026-10-22).

    A reward model's success head alone is no completion signal: on Spark it peaked mid-carry
    as often as at the placement (R221-R223). It is a cheap trigger instead, and so is a
    released grasp; a VLM confirms. Fires on a success score at or above
    ``success_threshold`` or on a release, at most once per ``cooldown_s`` (source seconds),
    and never while a hand holds an object (``holding``): a judge asked then said "done" with
    the banana still in the gripper over the bowl (R235, R237). A release seen while the hand
    still holds (the released payload stays on the wire for the kernel's release window)
    fires once the hand is empty. A release is asked again ``release_rechecks`` times, one
    cooldown apart, while the hand stays empty: asked as the gripper opened, the judge said
    "no" to a banana lying in the bowl and was never asked again (Spark R329, 2026-10-23).

    Example:
        >>> t = CompletionTrigger(success_threshold=0.6, cooldown_s=10.0)
        >>> t.fire(0.0, success=0.4) is None
        True
        >>> t.fire(1.0, success=0.65)
        'success'
        >>> t.fire(5.0, released=True) is None  # cooling down
        True
        >>> t.fire(12.0, success=0.9, holding=True) is None  # in the hand: never
        True
        >>> t.fire(13.0, released=True, holding=True) is None  # released, still on the wire
        True
        >>> t.fire(14.0)  # the hand is empty: the pending release fires
        'release'
        >>> t.fire(24.0)  # still empty a cooldown later: asked again
        'release_recheck'
    """

    success_threshold: float = 0.6
    cooldown_s: float = 3.0
    release_rechecks: int = 3
    _last_s: float = field(default=-math.inf, repr=False)
    _release_pending: bool = field(default=False, repr=False)
    _rechecks_left: int = field(default=0, repr=False)

    def mark_released(self) -> None:
        """Remember a release seen while a check was already running (it fires next)."""
        self._release_pending = True

    def fire(
        self,
        now_s: float,
        *,
        success: float | None = None,
        released: bool = False,
        holding: bool = False,
    ) -> str | None:
        """The trigger's reason if a check is due, else ``None``.

        ``"release"``, ``"success"`` or ``"release_recheck"`` (a release asked again while the
        hand stays empty; a hold ends the rechecks).
        """
        self._release_pending = self._release_pending or released
        if holding:
            self._rechecks_left = 0
            return None
        if now_s - self._last_s < self.cooldown_s:
            return None
        if self._release_pending:
            reason = "release"
            self._rechecks_left = self.release_rechecks
        elif success is not None and success >= self.success_threshold:
            reason = "success"
        elif self._rechecks_left > 0:
            reason = "release_recheck"
            self._rechecks_left -= 1
        else:
            return None
        self._release_pending = False
        self._last_s = now_s
        return reason


def tiled_jpeg(images: list[tuple[bytes, int, int, str]], *, flip_180: bool = False) -> bytes:
    """Raw camera images ``(data, height, width, encoding)`` side by side as one JPEG.

    Each is scaled to the first one's height, left to right in order: one image a
    single-image VLM call (``describe_image``) can judge every view from. ``flip_180``
    rotates every view upright first (``image_msg_to_jpeg``: LIBERO/MuJoCo publish
    bottom-up, the host's ``OPENRAL_DASHBOARD_FLIP_180``).

    Example:
        >>> import io
        >>> from PIL import Image
        >>> red = (bytes([255, 0, 0]) * 4, 2, 2, "rgb8")
        >>> blue = (bytes([0, 0, 255]) * 16, 4, 4, "rgb8")
        >>> Image.open(io.BytesIO(tiled_jpeg([red, blue]))).size
        (4, 2)
    """
    from PIL import Image  # noqa: PLC0415 — lazy, as image_msg_to_jpeg: PIL stays optional

    tiles = [
        Image.open(
            io.BytesIO(
                image_msg_to_jpeg(data=data, height=h, width=w, encoding=enc, flip_180=flip_180)
            )
        ).convert("RGB")
        for data, h, w, enc in images
    ]
    height = tiles[0].height
    tiles = [
        t if t.height == height else t.resize((max(1, round(t.width * height / t.height)), height))
        for t in tiles
    ]
    canvas = Image.new("RGB", (sum(t.width for t in tiles), height))
    x = 0
    for t in tiles:
        canvas.paste(t, (x, 0))
        x += t.width
    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=90)
    return out.getvalue()
