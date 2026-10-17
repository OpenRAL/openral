"""The tracker's starvation fill (issue #418 WP C): a real object, a real clock argument."""

from __future__ import annotations

import pytest
from openral_world_state.object_tracker.heartbeat import StarvationFill


def _live() -> StarvationFill[str]:
    f = StarvationFill[str](gap_s=0.4, silent_s=5.0)
    f.touch(0.0)
    f.joints(0.0)
    f.published(0.0, tracked="set", attach="att")
    return f


def test_nothing_is_due_while_the_main_heartbeat_is_on_time() -> None:
    f = _live()
    assert f.due(0.3) == (None, None)
    f.published(0.5, tracked="set2", attach="att2")  # the main thread was on time
    assert f.due(0.8) == (None, None)


def test_a_starved_main_heartbeat_is_filled_with_the_last_messages_once_per_gap() -> None:
    f = _live()
    assert f.due(2.0) == ("set", "att")  # 2 s without a main publish (r10rs02_camlong)
    assert f.due(2.1) == (None, None)  # spaced by the gap, not every tick
    assert f.due(2.5) == ("set", "att")


def test_the_newest_published_message_is_the_one_republished() -> None:
    f = _live()
    f.published(1.0, tracked="set2")
    assert f.due(1.5) == ("set2", "att")


def test_a_hung_main_thread_is_never_covered_for() -> None:
    f = _live()
    f.touch(1.0)
    f.joints(1.0)
    assert f.due(5.9) == ("set", "att")
    assert f.due(6.5) == (None, None)  # silent for > 5 s: the kernel's gate must trip


def test_the_attachment_stream_needs_the_arm_watched() -> None:
    f = StarvationFill[str](gap_s=0.4, silent_s=5.0)
    f.touch(0.0)
    f.published(0.0, tracked="set", attach="att")  # no joint state ever
    assert f.due(1.0) == ("set", None)
    f.joints(1.0)
    f.touch(1.0)
    assert f.due(1.5) == ("set", "att")


def test_an_unsound_window_is_refused() -> None:
    with pytest.raises(ValueError):
        StarvationFill[str](gap_s=1.0, silent_s=0.5)
