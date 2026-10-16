"""The segmenter serves the cached frame a request's stamp names, not the newest one.

Reproduces Isaac DROID deploy-sim trial t3 (2026-10-16): the HAL's grasp-target leg
stamps its ``SegmentInView`` request with the depth frame it holds, the slow
segmenter serves it 2-3 frames later, and masks cut from the newest RGB frame sat
0.133-0.200 s from that depth (15 Hz), so the HAL's own check
(``vision_attachment_bridge.mask_depth_skew_reason``, 0.1 s) refused every
pre-measurement. With the frame history the request is served from the frame its
stamp names: the DROID scene's ``exterior_depth`` RGB and depth of one render
carry one stamp, so the pair the HAL checks has zero skew. The HAL check is
unchanged and still refuses a frame that is gone.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from openral_core import DeployScene, RobotDescription, load_scene_strict
from openral_hal.vision_attachment_bridge import mask_depth_skew_reason
from openral_perception_ros.segmenter_node import (
    DEFAULT_FRAME_WAIT_S,
    DEFAULT_MAX_FRAME_SKEW_S,
    FRAME_HISTORY,
    frame_nearest_stamp,
    wait_for_frame,
)

_SCENE = "scenes/deploy/isaac_arena_droid_banana_in_bowl.yaml"
_ROBOT = "robots/droid_franka_robotiq/robot.yaml"


def _tick_ns() -> int:
    """One control tick of the DROID deploy twin (``action_spec.control_freq_hz``)."""
    spec = RobotDescription.from_yaml(_ROBOT).action_spec
    assert spec is not None
    hz = spec.control_freq_hz
    assert hz == 15.0
    return round(1e9 / hz)


def _stream(n: int) -> deque[tuple[int, str]]:
    """A camera cache fed ``n`` frames, one per tick, the way the node's callback fills it."""
    tick = _tick_ns()
    cache: deque[tuple[int, str]] = deque(maxlen=FRAME_HISTORY)
    for k in range(n):
        cache.append((26_000_000_000 + k * tick, f"frame{k}"))
    return cache


def test_the_scene_feeds_the_segmenter_the_depth_cameras_rgb() -> None:
    """The precondition: the leg's RGB and depth are one render, so they share stamps."""
    scene = load_scene_strict(_SCENE, DeployScene)
    assert scene.runtime is not None
    va = scene.runtime.vision_attachment
    assert va is not None
    assert va.rgb_topic == "/openral/cameras/exterior_depth/image"
    assert va.depth_topic == "/openral/cameras/exterior_depth/depth/image"


def test_a_late_request_is_served_from_the_frame_its_stamp_names() -> None:
    cache = _stream(12)
    tick = _tick_ns()
    depth_stamp_ns = cache[-3][0]  # served 2 frames late, as in t3
    frame, reason = frame_nearest_stamp(
        list(cache), depth_stamp_ns, max_skew_s=DEFAULT_MAX_FRAME_SKEW_S
    )
    assert (frame, reason) == ("frame9", "")
    served_ns = next(s for s, f in cache if f == frame)
    assert mask_depth_skew_reason([served_ns], depth_stamp_ns, max_skew_s=0.1) == ""
    # What the newest frame gave: refused by the HAL, unchanged.
    assert mask_depth_skew_reason([cache[-1][0]], depth_stamp_ns, max_skew_s=0.1)
    assert cache[-1][0] - depth_stamp_ns == 2 * tick


def test_a_stamp_older_than_the_history_is_refused() -> None:
    cache = _stream(30)
    assert len(cache) == FRAME_HISTORY
    gone_ns = 26_000_000_000 + 5 * _tick_ns()
    frame, reason = frame_nearest_stamp(list(cache), gone_ns, max_skew_s=DEFAULT_MAX_FRAME_SKEW_S)
    assert frame is None
    assert reason.startswith("ROSPerceptionStale:")


def test_an_unstamped_request_takes_the_newest_frame() -> None:
    cache = _stream(4)
    assert frame_nearest_stamp(list(cache), 0, max_skew_s=DEFAULT_MAX_FRAME_SKEW_S) == (
        "frame3",
        "",
    )


def test_an_empty_cache_is_refused() -> None:
    frame, reason = frame_nearest_stamp([], 1, max_skew_s=DEFAULT_MAX_FRAME_SKEW_S)
    assert frame is None
    assert reason.startswith("ROSPerceptionStale:")


def test_a_request_ahead_of_the_cache_waits_for_its_frame() -> None:
    """DROID t7: the attach request named a depth frame whose RGB twin had not arrived.

    The newest cached frame was one tick (0.133 s at the 7.5 Hz sim sensor) older than
    the stamp, beyond the 0.1 s skew. The request now waits for the frame it names; it
    arrives 50 ms later and is served, at zero skew to the HAL's depth.
    """
    cache = _stream(4)
    tick = _tick_ns()
    depth_stamp_ns = cache[-1][0] + 2 * tick  # its RGB twin is still on its way
    arrived = threading.Condition()

    def deliver() -> None:
        time.sleep(0.05)
        with arrived:
            cache.append((depth_stamp_ns, "twin"))
            arrived.notify_all()

    threading.Thread(target=deliver, daemon=True).start()
    start = time.monotonic()
    frame, reason = wait_for_frame(
        lambda: cache, arrived, depth_stamp_ns,
        max_skew_s=DEFAULT_MAX_FRAME_SKEW_S, wait_s=DEFAULT_FRAME_WAIT_S,
    )  # fmt: skip
    assert (frame, reason) == ("twin", "")
    assert time.monotonic() - start < DEFAULT_FRAME_WAIT_S
    assert mask_depth_skew_reason([depth_stamp_ns], depth_stamp_ns, max_skew_s=0.1) == ""


def test_a_frame_that_never_arrives_is_refused_after_the_wait() -> None:
    cache = _stream(4)
    ahead_ns = cache[-1][0] + 3 * _tick_ns()
    start = time.monotonic()
    frame, reason = wait_for_frame(
        lambda: cache, threading.Condition(), ahead_ns,
        max_skew_s=DEFAULT_MAX_FRAME_SKEW_S, wait_s=0.1,
    )  # fmt: skip
    assert frame is None
    assert reason.startswith("ROSPerceptionStale:") and "waited 0.100 s" in reason
    assert time.monotonic() - start >= 0.1


def test_a_request_behind_the_cache_is_refused_without_waiting() -> None:
    """The frame it names is gone (older than the history), not on its way: no wait."""
    cache = _stream(30)
    gone_ns = 26_000_000_000
    start = time.monotonic()
    frame, reason = wait_for_frame(
        lambda: cache, threading.Condition(), gone_ns,
        max_skew_s=DEFAULT_MAX_FRAME_SKEW_S, wait_s=DEFAULT_FRAME_WAIT_S,
    )  # fmt: skip
    assert frame is None and reason.startswith("ROSPerceptionStale:")
    assert time.monotonic() - start < 0.05


def test_a_stamped_request_on_an_empty_cache_waits_for_the_first_frame() -> None:
    """A request that beats the camera's first frame is served when that frame lands."""
    cache: deque[tuple[int, str]] = deque(maxlen=FRAME_HISTORY)
    depth_stamp_ns = 26_000_000_000
    arrived = threading.Condition()

    def deliver() -> None:
        time.sleep(0.05)
        with arrived:
            cache.append((depth_stamp_ns, "first"))
            arrived.notify_all()

    threading.Thread(target=deliver, daemon=True).start()
    frame, reason = wait_for_frame(
        lambda: cache, arrived, depth_stamp_ns,
        max_skew_s=DEFAULT_MAX_FRAME_SKEW_S, wait_s=DEFAULT_FRAME_WAIT_S,
    )  # fmt: skip
    assert (frame, reason) == ("first", "")


def test_an_empty_cache_is_refused_after_the_wait_or_at_once_when_unstamped() -> None:
    start = time.monotonic()
    frame, reason = wait_for_frame(
        lambda: [], threading.Condition(), 26_000_000_000,
        max_skew_s=DEFAULT_MAX_FRAME_SKEW_S, wait_s=0.1,
    )  # fmt: skip
    assert frame is None
    assert reason.startswith("ROSPerceptionStale:") and "waited 0.100 s" in reason
    assert time.monotonic() - start >= 0.1
    start = time.monotonic()
    frame, reason = wait_for_frame(
        lambda: [], threading.Condition(), 0,
        max_skew_s=DEFAULT_MAX_FRAME_SKEW_S, wait_s=DEFAULT_FRAME_WAIT_S,
    )  # fmt: skip
    assert frame is None and reason == "ROSPerceptionStale: no frame cached"
    assert time.monotonic() - start < 0.05
