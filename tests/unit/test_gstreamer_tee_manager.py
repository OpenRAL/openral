"""Live integration tests for the runtime tee-branch manager (ADR-0037).

No mocks (CLAUDE.md §1.11): these drive a **real** GStreamer pipeline
(``videotestsrc ! tee`` set to PLAYING) and exercise real dynamic pad
add / remove via :class:`TeeManager`. Skipped when ``gi`` (the
``openral-runner[gstreamer]`` extra) is not importable, or when the core
``videotestsrc`` / ``tee`` plugins are absent.
"""

from __future__ import annotations

import time

import pytest

gi = pytest.importorskip("gi", reason="PyGObject not installed (openral-runner[gstreamer])")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402  # gi requires a version-pin before import
from openral_core.exceptions import ROSConfigError, ROSRuntimeError  # noqa: E402
from openral_runner.backends.gstreamer.pipeline import TEE_NAME, leaky_branch  # noqa: E402
from openral_runner.backends.gstreamer.tee_manager import BranchHandle, TeeManager  # noqa: E402

Gst.init(None)

_APPSINK = "appsink name={name} emit-signals=false sync=false"


def _require_elements(*names: str) -> None:
    """Skip the test when any core GStreamer element is missing."""
    registry = Gst.Registry.get()
    for name in names:
        if registry.find_feature(name, Gst.ElementFactory.__gtype__) is None:
            pytest.skip(f"GStreamer element {name!r} not available in this install")


@pytest.fixture
def playing_pipeline() -> object:
    """A live ``videotestsrc ! tee name=<TEE_NAME> ! <leaky> fakesink`` pipeline.

    Yields the pipeline (already PLAYING) and sets it to NULL on teardown.
    """
    _require_elements("videotestsrc", "tee", "queue", "fakesink", "appsink")
    pipeline = Gst.parse_launch(
        "videotestsrc is-live=true ! video/x-raw,framerate=30/1 ! "
        f"tee name={TEE_NAME}  " + leaky_branch("fakesink sync=false")
    )
    pipeline.set_state(Gst.State.PLAYING)
    # Block until the state change completes so the streaming thread is running
    # (the IDLE detach probe fires from it).
    _, state, _ = pipeline.get_state(Gst.SECOND)
    assert state == Gst.State.PLAYING, "pipeline did not reach PLAYING"
    try:
        yield pipeline
    finally:
        pipeline.set_state(Gst.State.NULL)


def _is_playing(pipeline: object) -> bool:
    """True once the pipeline has settled in PLAYING.

    Uses a bounded blocking ``get_state`` rather than a zero-timeout query so a
    transient ASYNC window (e.g. just after a branch is removed) settles instead
    of being read as "not playing".
    """
    ret, state, _pending = pipeline.get_state(Gst.SECOND)  # type: ignore[attr-defined]
    return ret == Gst.StateChangeReturn.SUCCESS and state == Gst.State.PLAYING


def test_init_raises_when_tee_absent() -> None:
    """Constructing against a pipeline with no bus tee is a config error."""
    _require_elements("videotestsrc", "fakesink")
    pipeline = Gst.parse_launch("videotestsrc num-buffers=1 ! fakesink")
    with pytest.raises(ROSConfigError, match="no element named"):
        TeeManager(pipeline)


def test_attach_then_detach_roundtrip(playing_pipeline: object) -> None:
    """attach adds a live branch; detach removes it; the policy leg is untouched."""
    tee = playing_pipeline.get_by_name(TEE_NAME)  # type: ignore[attr-defined]
    tm = TeeManager(playing_pipeline)

    assert tm.branch_count == 0
    base_pads = tee.numsrcpads  # the static policy leg

    handle = tm.attach(_APPSINK.format(name="det_sink"), name="det")
    assert isinstance(handle, BranchHandle)
    assert tm.branch_count == 1
    assert tee.numsrcpads == base_pads + 1
    assert playing_pipeline.get_by_name("det_sink") is not None  # type: ignore[attr-defined]
    time.sleep(0.2)
    assert _is_playing(playing_pipeline), "policy leg must keep PLAYING during attach"

    tm.detach(handle)
    assert tm.branch_count == 0
    assert tee.numsrcpads == base_pads, "tee request pad released on detach"
    assert playing_pipeline.get_by_name("det_sink") is None  # type: ignore[attr-defined]
    assert _is_playing(playing_pipeline), "policy leg must keep PLAYING after detach"


def test_attach_detach_is_repeatable(playing_pipeline: object) -> None:
    """Two attach/detach cycles leave the tee back at its base pad count."""
    tee = playing_pipeline.get_by_name(TEE_NAME)  # type: ignore[attr-defined]
    tm = TeeManager(playing_pipeline)
    base_pads = tee.numsrcpads

    for i in range(2):
        handle = tm.attach(_APPSINK.format(name=f"det_sink_{i}"), name=f"det_{i}")
        assert tee.numsrcpads == base_pads + 1
        time.sleep(0.1)
        tm.detach(handle)
        assert tee.numsrcpads == base_pads
    assert tm.branch_count == 0
    assert _is_playing(playing_pipeline)


def test_multiple_concurrent_branches(playing_pipeline: object) -> None:
    """Two branches can be attached at once (forward-looking: detector + VLA)."""
    tee = playing_pipeline.get_by_name(TEE_NAME)  # type: ignore[attr-defined]
    tm = TeeManager(playing_pipeline)
    base_pads = tee.numsrcpads

    h1 = tm.attach(_APPSINK.format(name="sink_a"), name="a")
    h2 = tm.attach(_APPSINK.format(name="sink_b"), name="b")
    assert tm.branch_count == 2
    assert tee.numsrcpads == base_pads + 2
    time.sleep(0.2)
    assert _is_playing(playing_pipeline)

    tm.detach(h1)
    tm.detach(h2)
    assert tm.branch_count == 0
    assert tee.numsrcpads == base_pads


def test_detach_is_idempotent(playing_pipeline: object) -> None:
    """Detaching an already-detached handle is a harmless no-op."""
    tm = TeeManager(playing_pipeline)
    handle = tm.attach(_APPSINK.format(name="det_sink"), name="det")
    tm.detach(handle)
    # Second detach of the same handle must not raise or change state.
    tm.detach(handle)
    assert tm.branch_count == 0
    assert _is_playing(playing_pipeline)


def test_attach_duplicate_name_raises(playing_pipeline: object) -> None:
    """Re-using an attached branch name is a config error (ambiguous detach)."""
    tm = TeeManager(playing_pipeline)
    handle = tm.attach(_APPSINK.format(name="det_sink"), name="det")
    try:
        with pytest.raises(ROSConfigError, match="already attached"):
            tm.attach(_APPSINK.format(name="det_sink_2"), name="det")
    finally:
        tm.detach(handle)
    assert tm.branch_count == 0


def test_attach_unparseable_elements_raises(playing_pipeline: object) -> None:
    """A bogus branch description is rejected without leaking a tee pad."""
    tee = playing_pipeline.get_by_name(TEE_NAME)  # type: ignore[attr-defined]
    tm = TeeManager(playing_pipeline)
    base_pads = tee.numsrcpads
    with pytest.raises((ROSConfigError, ROSRuntimeError)):
        tm.attach("this_is_not_a_real_gst_element_xyz", name="bad")
    assert tm.branch_count == 0
    assert tee.numsrcpads == base_pads, "a failed attach must not leak a tee src pad"
