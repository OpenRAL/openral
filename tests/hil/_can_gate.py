# SPDX-License-Identifier: Apache-2.0
"""Shared CAN-link gate for ``tests/hil/``.

Lives here rather than in ``conftest.py`` because a conftest is not importable
as a module: pytest registers it under its own private name, so
``from tests.hil.conftest import ...`` raises ``ModuleNotFoundError`` on the
lab hosts these tests actually run on. The ``tests/hil/_*.py`` helpers
(``_ros_control_transport``, ``_openarm_ros_transport``) are the convention
that works.
"""

from __future__ import annotations


def _can_links_up(can_links: tuple[str, ...]) -> bool:
    """True if every link in *can_links* is up (per ``enumerate_can_interfaces``).

    Args:
        can_links: SocketCAN interface names the robot needs, e.g.
            ``("openarm_left", "openarm_right")``.

    Returns:
        True when every named link exists and is up.

    Example:
        >>> _can_links_up(("no_such_can_link",))
        False
    """
    from openral_cli.autodetect import enumerate_can_interfaces

    up = {i.name for i in enumerate_can_interfaces() if i.is_up}
    return set(can_links) <= up
