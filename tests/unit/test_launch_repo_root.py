# SPDX-License-Identifier: Apache-2.0
"""The launch file must find the repo root from its *installed* copy too.

`ros2 launch` resolves `deploy_e2e.launch.py` through the ament index, so it
normally runs from `<repo>/install/share/openral_rskill_ros/launch/` rather than
`<repo>/packages/openral_rskill_ros/launch/`. Both are four levels deep, so the
original `parents[3]` silently returned `<repo>/install` from the installed copy
and every repo-relative path built from it — `tools/lifecycle_autostart.py`,
`rskills/`, `.venv/bin/openral` — pointed somewhere that does not exist.

A `--symlink-install` layout resolves back through the symlink to the source
tree, which is why this survived: it only bites on a *copied* install. Found on
the OpenArm cell after a clean `colcon build`, where the graph came up looking
healthy while the safety-kernel and reasoner autostart processes each died with
exit code 2 (python: no such file), leaving both nodes unconfigured.
"""

from __future__ import annotations

import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_the_source_tree_layout_is_four_deep() -> None:
    """Pins the arithmetic the fallback still uses."""
    launch = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
    assert launch.is_file()
    assert launch.resolve().parents[3] == _REPO_ROOT


def test_the_install_layout_is_also_four_deep() -> None:
    """Which is exactly why `parents[3]` could not tell them apart.

    `<repo>/install/share/<pkg>/launch/` has the same depth as
    `<repo>/packages/<pkg>/launch/`, so the arithmetic yields `<repo>/install`
    with no error anywhere — the failure only shows up as missing files later.
    """
    installed = _REPO_ROOT / "install" / "share" / "openral_rskill_ros" / "launch" / "x.launch.py"
    assert installed.parents[3] == _REPO_ROOT / "install"
    assert installed.parents[3] != _REPO_ROOT


def test_marker_search_finds_the_repo_root_from_both_layouts() -> None:
    """The search the launch file uses must agree from source *and* install.

    Anchored on the real helper rather than a copy of its rule, because the
    launch file reuses that helper — if its markers change, this moves with it.
    """
    loader = pytest.importorskip("openral_rskill.loader")
    find = loader._find_repo_root_from

    from_source = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch"
    assert find(from_source) == _REPO_ROOT

    # The installed copy need not exist for the walk to be well-defined: the
    # helper tests each ancestor, and `<repo>` is an ancestor of both.
    from_install = _REPO_ROOT / "install" / "share" / "openral_rskill_ros" / "launch"
    assert find(from_install) == _REPO_ROOT


def test_the_repo_root_actually_carries_what_the_launch_reads() -> None:
    """A root without these is the wrong root, however it was derived."""
    assert (_REPO_ROOT / "tools" / "lifecycle_autostart.py").is_file()
    assert (_REPO_ROOT / "rskills").is_dir()
