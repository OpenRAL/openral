"""Runtime tee-branch manager for the GStreamer perception bus (ADR-0037).

A :class:`TeeManager` owns the named ``tee`` of a *running* camera pipeline
(:data:`~openral_runner.backends.gstreamer.pipeline.TEE_NAME`) and attaches /
detaches consumer branches on the live pipeline via dynamic pad add / remove.
This is the mechanism the S2 reasoner drives through ``ExecuteSkill``: activating
a detector rSkill attaches a branch; deactivating it detaches the branch.

Each branch is built as ``<leaky queue> ! <caller elements>`` so a stalled or
crashing consumer drops its own frames instead of backpressuring the policy leg
(ADR-0018 §3) — the same isolation policy the static pipeline builder applies
via :func:`~openral_runner.backends.gstreamer.pipeline.leaky_branch`, shared here
through :data:`~openral_runner.backends.gstreamer.pipeline.LEAKY_BRANCH_QUEUE`.

Lifecycle:

* :meth:`attach` requests a ``tee`` src pad, parses the branch into a bin, adds
  it to the pipeline, links the pad, and syncs the branch to PLAYING. Adding is
  safe on a live tee — the leaky queue absorbs the brief pre-roll window.
* :meth:`detach` installs an ``IDLE`` pad probe on the branch's ``tee`` pad; when
  the pad next goes idle (between buffers, on the streaming thread) the probe
  unlinks the branch, releases the request pad, and tears the bin down to
  ``NULL``. This is the canonical safe-detach pattern for a flowing pipeline.

Like :mod:`~openral_runner.backends.gstreamer.reader`, this module imports
``gi`` at load and therefore requires the ``gstreamer`` optional-extra; the
:mod:`~openral_runner.backends.gstreamer.pipeline` builder it depends on is
import-safe everywhere.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Final

import gi
import structlog

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402  # gi requires a version-pin before import
from openral_core.exceptions import ROSConfigError, ROSRuntimeError  # noqa: E402

from openral_runner.backends.gstreamer.pipeline import (  # noqa: E402
    LEAKY_BRANCH_QUEUE,
    TEE_NAME,
)

# Idempotent; mirrors reader.py's eager init to keep GStreamer process state
# initialised before any later ``import rclpy`` in the same interpreter.
Gst.init(None)

log = structlog.get_logger(__name__)

__all__ = ["BranchHandle", "TeeManager"]

# Request-pad template name for ``GstTee`` src pads.
_REQUEST_PAD_TEMPLATE: Final[str] = "src_%u"

# How long :meth:`TeeManager.detach` waits for its IDLE probe to fire before
# logging a warning. On a flowing pipeline the probe fires within a frame or
# two; the timeout only guards a stalled / NULL pipeline.
_DETACH_TIMEOUT_S: Final[float] = 5.0


@dataclass
class BranchHandle:
    """Opaque handle to a branch attached by :meth:`TeeManager.attach`.

    Returned by :meth:`TeeManager.attach` and passed back to
    :meth:`TeeManager.detach`. Callers treat it as opaque; the private fields
    carry the GStreamer objects the manager needs to tear the branch down.

    Attributes:
        name: Caller-supplied branch name (also the bin's element name).
    """

    name: str
    _tee_pad: Any = field(repr=False)  # Gst.Pad — the tee request src pad
    _bin: Any = field(repr=False)  # Gst.Bin — ``queue ! <elements>``


class TeeManager:
    """Attaches / detaches consumer branches on a running pipeline's named tee.

    Args:
        pipeline: A ``Gst.Pipeline`` that is (or will be) PLAYING and contains a
            ``tee`` named *tee_name*.
        tee_name: Name of the bus tee to manage. Defaults to
            :data:`~openral_runner.backends.gstreamer.pipeline.TEE_NAME`.

    Raises:
        ROSConfigError: If *pipeline* has no element named *tee_name*.

    Example:
        >>> # Exercised live in tests/unit/test_gstreamer_tee_manager.py:
        >>> import contextlib
        >>> with contextlib.suppress(ImportError):
        ...     from openral_runner.backends.gstreamer.tee_manager import TeeManager
    """

    def __init__(self, pipeline: Any, *, tee_name: str = TEE_NAME) -> None:  # noqa: ANN401  # reason: Gst.Pipeline — duck-typed to avoid a gi type dependency at the signature
        """Look up and retain the named tee; raise if it is absent."""
        tee = pipeline.get_by_name(tee_name)
        if tee is None:
            raise ROSConfigError(
                f"TeeManager: pipeline has no element named {tee_name!r}; "
                "the camera pipeline must declare the bus tee (see pipeline.TEE_NAME)."
            )
        self._pipeline = pipeline
        self._tee = tee
        self._tee_name = tee_name
        self._lock = threading.Lock()
        self._branches: dict[str, BranchHandle] = {}

    @property
    def branch_count(self) -> int:
        """Number of branches currently attached by this manager."""
        with self._lock:
            return len(self._branches)

    def attach(self, elements: str, *, name: str) -> BranchHandle:
        """Attach ``<leaky queue> ! elements`` as a new branch on the live tee.

        Args:
            elements: The branch body downstream of the shared leaky queue —
                e.g. ``"appsink name=det_sink emit-signals=true sync=false"`` or
                a ``nvvidconv ! ... ! appsink`` chain.
            name: Unique branch name (also the bin's element name). Reused as the
                detach key.

        Returns:
            A :class:`BranchHandle` to pass to :meth:`detach`.

        Raises:
            ROSConfigError: If *name* is already attached, or *elements* is not a
                parseable GStreamer bin description.
            ROSRuntimeError: If the tee refuses a request pad or the link fails.
        """
        branch_desc = f"{LEAKY_BRANCH_QUEUE} ! {elements}"
        with self._lock:
            if name in self._branches:
                raise ROSConfigError(f"TeeManager: branch {name!r} is already attached.")
            try:
                # ghost_unlinked_pads=True exposes the leading queue's sink as a
                # ghost "sink" pad we can link the tee request pad to.
                branch = Gst.parse_bin_from_description(branch_desc, True)
            except GLib.Error as exc:
                raise ROSConfigError(
                    f"TeeManager: branch {name!r} description is not parseable: "
                    f"{branch_desc!r}: {exc}"
                ) from exc
            branch.set_property("name", name)

            # Add the bin BEFORE requesting a tee pad, so an add failure needs no
            # pad cleanup. A False return means a name collision — most likely an
            # orphan bin left by a prior detach that timed out (see detach()).
            if not self._pipeline.add(branch):
                raise ROSRuntimeError(
                    f"TeeManager: pipeline.add failed for branch {name!r}; a prior "
                    "detach may have timed out and left an orphan bin in the pipeline."
                )
            tee_pad = self._tee.request_pad_simple(_REQUEST_PAD_TEMPLATE)
            if tee_pad is None:  # pragma: no cover — tee always grants src pads
                branch.set_state(Gst.State.NULL)
                self._pipeline.remove(branch)
                raise ROSRuntimeError(
                    f"TeeManager: tee {self._tee_name!r} refused a request pad for {name!r}."
                )
            link_ret = tee_pad.link(branch.get_static_pad("sink"))
            if link_ret != Gst.PadLinkReturn.OK:
                # Roll back so a failed attach leaves no orphan pad / bin.
                self._tee.release_request_pad(tee_pad)
                branch.set_state(Gst.State.NULL)
                self._pipeline.remove(branch)
                raise ROSRuntimeError(
                    f"TeeManager: linking tee {self._tee_name!r} to branch {name!r} "
                    f"failed: {link_ret!r}."
                )
            branch.sync_state_with_parent()

            handle = BranchHandle(name=name, _tee_pad=tee_pad, _bin=branch)
            self._branches[name] = handle
            log.debug(
                "tee_manager.attached",
                branch=name,
                tee=self._tee_name,
                src_pads=self._tee.numsrcpads,
            )
            return handle

    def detach(self, handle: BranchHandle) -> None:
        """Detach a branch from the live tee and tear it down to ``NULL``.

        Idempotent: detaching an already-detached (or unknown) handle is a no-op.
        Blocks until the branch is removed (the IDLE probe fires on the streaming
        thread of a flowing pipeline).

        Args:
            handle: The handle returned by :meth:`attach`.

        Raises:
            ROSRuntimeError: If removal does not complete within
                :data:`_DETACH_TIMEOUT_S` — the pipeline is stalled. The IDLE
                probe remains installed and finishes removal once data flows
                again; the caller should treat the branch as not-yet-removed.
        """
        with self._lock:
            if self._branches.get(handle.name) is not handle:
                # Already detached, or never ours — nothing to do.
                return
            del self._branches[handle.name]

        removed = threading.Event()

        def _on_idle(pad: Any, info: Any) -> Any:  # noqa: ANN401  # reason: Gst.Pad / Gst.PadProbeInfo — duck-typed
            """Unlink + release the request pad + NULL the bin, on the streaming thread.

            Returns ``REMOVE`` to drop this probe — the canonical way to remove a
            probe from inside its own callback (calling ``pad.remove_probe`` here
            would re-enter the probe lock).
            """
            pad.unlink(handle._bin.get_static_pad("sink"))
            self._tee.release_request_pad(pad)
            handle._bin.set_state(Gst.State.NULL)
            self._pipeline.remove(handle._bin)
            removed.set()
            return Gst.PadProbeReturn.REMOVE

        handle._tee_pad.add_probe(Gst.PadProbeType.IDLE, _on_idle)
        if not removed.wait(timeout=_DETACH_TIMEOUT_S):
            log.warning(
                "tee_manager.detach_timeout",
                branch=handle.name,
                tee=self._tee_name,
                timeout_s=_DETACH_TIMEOUT_S,
            )
            raise ROSRuntimeError(
                f"TeeManager: detach of branch {handle.name!r} did not complete within "
                f"{_DETACH_TIMEOUT_S}s; the pipeline may be stalled. The IDLE probe remains "
                "installed and will finish removal once data flows again."
            )
        log.debug(
            "tee_manager.detached",
            branch=handle.name,
            tee=self._tee_name,
            src_pads=self._tee.numsrcpads,
        )
