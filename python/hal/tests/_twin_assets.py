"""Skip a twin test only when an optional prerequisite is unavailable.

A twin's ``connect`` (or its ``from_description``) raises ``ROSConfigError`` both
when an optional asset or package cannot be had on this host — an un-fetchable MJCF
or policy, a pinned ``git clone`` that fails offline, an optional package that is
not installed — and when the twin itself is broken (an MJCF that does not compile, a
camera rig or time base it refuses). Only the first may skip; the second must fail.

Import it via the ``sys.path`` shim in this directory's ``conftest.py``::

    from _twin_assets import skip_if_unavailable
"""

from __future__ import annotations

import subprocess
import urllib.error

import pytest
from openral_core.assets import AssetRefError
from openral_core.exceptions import ROSConfigError

#: Causes that mean "not available on this host", not "broken".
_UNAVAILABLE = (AssetRefError, ImportError, subprocess.CalledProcessError, urllib.error.URLError)


def skip_if_unavailable(exc: ROSConfigError, label: str) -> None:
    """``pytest.skip`` when ``exc`` comes from an unavailable prerequisite, else re-raise it.

    Raises:
        ROSConfigError: ``exc``, when nothing in its cause chain marks it unavailable.
    """
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, _UNAVAILABLE) or "needs `git` on the PATH" in str(cause):
            pytest.skip(f"{label} unavailable: {exc}")
        cause = cause.__cause__
    raise exc
