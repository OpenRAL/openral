# SPDX-License-Identifier: Apache-2.0
"""Whether an ament package is installed in the sourced ROS 2 overlay, for skip decisions.

Pure-ament packages (an RMW, a vendor launch package) have no Python module to
``importorskip``, so they are resolved with ``ros2 pkg prefix``.
"""

from __future__ import annotations

import shutil
import subprocess


def ros_pkg_available(name: str) -> bool:
    """Whether ``ros2 pkg prefix name`` resolves; False without a ``ros2`` CLI on PATH.

    Example:
        >>> ros_pkg_available("openral_no_such_package")
        False
    """
    if shutil.which("ros2") is None:
        return False
    result = subprocess.run(
        ["ros2", "pkg", "prefix", name], check=False, capture_output=True, text=True, timeout=10
    )
    return result.returncode == 0 and result.stdout.strip() != ""
