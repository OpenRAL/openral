"""``CompletionTrigger``: when the completion check asks the VLM (Spark R221-R223, 2026-10-22).

Robometer's success head peaked mid-carry as often as at the placement, so it only triggers
a VLM check; so does a release. One check per cooldown.
"""

from __future__ import annotations

from openral_reasoner.completion import CompletionTrigger


def test_a_success_at_the_bar_fires_and_below_it_does_not() -> None:
    t = CompletionTrigger(success_threshold=0.6, cooldown_s=10.0)
    assert t.fire(0.0, success=0.599) is None
    assert t.fire(1.0, success=0.6) == "success"


def test_a_release_fires_whatever_the_score() -> None:
    t = CompletionTrigger(success_threshold=0.6, cooldown_s=10.0)
    assert t.fire(0.0, success=0.1, released=True) == "release"


def test_one_check_per_cooldown_whichever_trigger() -> None:
    t = CompletionTrigger(success_threshold=0.6, cooldown_s=10.0)
    assert t.fire(0.0, success=0.9) == "success"
    assert t.fire(9.9, released=True) is None
    assert t.fire(9.9, success=0.95) is None
    assert t.fire(10.0, success=0.95) == "release", "a release is never lost to the cooldown"
    assert t.fire(20.0, success=0.95) == "success"


def test_nothing_to_go_on_fires_nothing_and_keeps_the_cooldown_open() -> None:
    t = CompletionTrigger(success_threshold=0.6, cooldown_s=10.0)
    assert t.fire(0.0) is None
    assert t.fire(0.5, success=0.7) == "success", "a non-firing call starts no cooldown"


def test_nothing_fires_while_a_hand_holds_an_object() -> None:
    """Spark R235 / R237: asked with the banana still in the gripper, the judge said done."""
    t = CompletionTrigger(success_threshold=0.6, cooldown_s=10.0)
    assert t.fire(0.0, success=0.95, holding=True) is None
    assert t.fire(1.0, success=0.95) == "success", "holding started no cooldown"


def test_a_release_still_on_the_wire_fires_once_the_hand_is_empty() -> None:
    """The kernel keeps a released payload on the wire for its release window."""
    t = CompletionTrigger(success_threshold=0.6, cooldown_s=10.0)
    assert t.fire(0.0, released=True, holding=True) is None
    assert t.fire(2.0, holding=True) is None
    assert t.fire(3.0) == "release"
    assert t.fire(4.0, holding=True) is None
    assert t.fire(20.0) is None, "fired once: a new hold ended its rechecks"


def test_the_default_cooldown_re_checks_inside_a_short_placed_window() -> None:
    """Spark R270 (2026-10-23), sim seconds: the release check at 0 said "not yet" (the banana
    still settling); it lay in the bowl from 2.2 s to 5.4 s at success 0.79-0.89 before the
    arm tipped the bowl. The default cooldown asks again inside that window."""
    t = CompletionTrigger()
    assert t.fire(0.0, released=True) == "release"
    fired = [s for s in (1.0, 2.2, 3.1, 4.0, 5.4) if t.fire(s, success=0.85) == "success"]
    assert fired and 2.2 <= fired[0] <= 5.4


def test_a_release_is_asked_again_while_the_hand_stays_empty() -> None:
    """Spark R329 (2026-10-23): asked as the gripper opened, the judge said "no" to a banana
    lying in the bowl, and nothing asked again; the policy knocked it out minutes later."""
    t = CompletionTrigger(success_threshold=0.6, cooldown_s=3.0, release_rechecks=3)
    assert t.fire(0.0, released=True) == "release"
    assert t.fire(2.0) is None, "one cooldown apart"
    fired = [s for s in (3.0, 4.0, 6.0, 9.0, 12.0, 15.0) if t.fire(s) is not None]
    assert fired == [3.0, 6.0, 9.0], fired
    again = CompletionTrigger(cooldown_s=3.0, release_rechecks=3)
    assert again.fire(0.0, released=True) == "release"
    assert again.fire(1.0, holding=True) is None
    assert again.fire(10.0) is None, "a new hold ends the rechecks"
