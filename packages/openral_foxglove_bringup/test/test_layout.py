"""Guards for the generated Foxglove layout (``openral_foxglove_bringup.layout``).

Two things are being protected here:

1. **The layout can only ever show what the bridge exposes.** Every topic a
   generated layout references must match ``BUCKET1_TOPIC_WHITELIST`` — for any
   camera-slot list, not just the shipped default. A layout referencing an
   unexposed topic is a panel that silently sits empty.
2. **The surface stays read-only.** No generated layout may contain a
   write-capable panel (Publish, Teleop, Call Service, Parameters), and the 3D
   panels must not carry a click-to-publish target. Actuation from a viewer is
   out of scope for this package (see ``README.md`` and CLAUDE.md §3).

Like the sibling launch tests, these load the target modules by path so they
run without the ament package installed.
"""

from __future__ import annotations

import doctest
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_PKG_DIR = Path(__file__).resolve().parent.parent
_LAYOUT = _PKG_DIR / "config" / "openral_layout.json"


def _load(name: str, filename: str) -> ModuleType:
    """Load a package module by path (ament package, not pip-installed)."""
    spec = importlib.util.spec_from_file_location(
        name, _PKG_DIR / "openral_foxglove_bringup" / filename
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_topics = _load("_fxbringup_topics_layout", "topics.py")
_layout = _load("_fxbringup_layout", "layout.py")

BUCKET1_TOPIC_WHITELIST: list[str] = _topics.BUCKET1_TOPIC_WHITELIST
build_layout = _layout.build_layout
DEFAULT_CAMERAS: tuple[str, ...] = _layout.DEFAULT_CAMERAS

#: Foxglove panel types that can write to the robot. None may appear in a
#: layout this package ships or generates.
#:
#: Cross-checked against ``foxglove-sdk``'s layout panel inventory
#: (``python/foxglove/layouts/__init__.py``), which names the write-capable
#: panels ``Teleop``, ``Publish``, ``Parameters`` and ``ServiceCall``. The SDK
#: is a *different* serialisation from the ``configById`` format this layout
#: uses — it writes the 3D panel as ``ThreeDee`` where ``configById`` writes
#: ``3D`` — so its spellings are not authoritative for our prefixes. Both
#: spellings of the service-call panel are therefore listed: whichever the
#: viewer's format uses, the guard trips.
_WRITE_CAPABLE_PANELS = ("Publish", "Teleop", "Parameters", "ServiceCall", "CallService")


def _referenced_topics(layout: dict[str, Any]) -> set[str]:
    """Every ROS topic name a layout mentions, message-path suffixes stripped.

    Foxglove message paths carry field accessors (``/joint_states.position[:]``),
    so the topic is the leading ``/``-rooted, word-and-slash run.
    """
    blob = json.dumps(layout)
    return {t for t in re.findall(r"/[\w/]+", blob) if t.count("/") >= 1 and len(t) > 1}


def _is_exposed(topic: str) -> bool:
    """Whether the bridge's allowlist matches ``topic`` (``std::regex_match``)."""
    return any(re.fullmatch(pat, topic) for pat in BUCKET1_TOPIC_WHITELIST)


# ---------------------------------------------------------------------------
# The generated layout only shows exposed topics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cameras",
    [
        DEFAULT_CAMERAS,
        ("top",),  # isaac_franka
        ("top", "wrist"),  # isaac_franka_bowl
        ("head", "left_wrist", "right_wrist"),  # behavior_r1pro
        ("agentview", "agentview_left", "cam_l", "cam_r"),  # LIBERO-style four-up
    ],
)
def test_generated_layout_references_only_exposed_topics(cameras: tuple[str, ...]) -> None:
    """Whatever the scene's camera slots are, every panel's topic is on the allowlist."""
    layout = build_layout(list(cameras))
    for topic in _referenced_topics(layout):
        # TF frame ids and panel ids are not topics; only check /-rooted names
        # that look like ours or like the standard nav/robot set.
        if topic.startswith(("/openral", "/octomap", "/rosout", "/diagnostics")) or topic in {
            "/map",
            "/joint_states",
            "/odom",
            "/scan",
            "/tf",
            "/tf_static",
            "/robot_description",
        }:
            assert _is_exposed(topic), (
                f"generated layout references {topic!r}, which the bridge does not expose"
            )


@pytest.mark.parametrize("compressed", [False, True])
def test_camera_panels_track_the_requested_slots(compressed: bool) -> None:
    """One Image panel per camera slot, in order, on the right transport."""
    cameras = ["head", "left_wrist", "right_wrist"]
    layout = build_layout(cameras, compressed=compressed)
    suffix = "/compressed" if compressed else ""
    for i, name in enumerate(cameras):
        panel = layout["configById"][f"Image!cam_{i}"]
        assert panel["imageMode"]["imageTopic"] == f"/openral/cameras/{name}/image{suffix}"


def test_empty_camera_list_is_rejected() -> None:
    """A layout with no Image panel is a bug, not a degenerate case."""
    with pytest.raises(ValueError, match="cameras must not be empty"):
        build_layout([])


def test_every_panel_id_is_placed_in_the_layout_tree() -> None:
    """No configured panel is orphaned — each id appears in the tree or a tab."""
    layout = build_layout()
    placed = set(re.findall(r"[A-Za-z0-9]+![a-z0-9_]+", json.dumps(layout["layout"])))
    for panel_id, config in layout["configById"].items():
        if panel_id.startswith("Tab!"):
            placed.update(tab["layout"] for tab in config["tabs"])
    missing = set(layout["configById"]) - placed
    assert not missing, f"panels configured but never placed: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Read-only invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("panel_type", _WRITE_CAPABLE_PANELS)
def test_generated_layout_has_no_write_capable_panel(panel_type: str) -> None:
    """A viewer must not get a Publish / Teleop / Call Service / Parameters panel."""
    layout = build_layout()
    for panel_id in layout["configById"]:
        assert not panel_id.startswith(f"{panel_type}!"), (
            f"{panel_id} is a write-capable panel; the surface is read-only"
        )


def test_three_d_panels_have_no_publish_target() -> None:
    """The 3D panels' click-to-publish topics stay empty."""
    layout = build_layout()
    for panel_id, config in layout["configById"].items():
        if not panel_id.startswith("3D!"):
            continue
        publish = config.get("publish", {})
        assert not publish.get("poseTopic")
        assert not publish.get("pointTopic")


def test_no_user_scripts_are_shipped() -> None:
    """``userNodes`` stays empty — the layout ships no executable script."""
    assert build_layout()["userNodes"] == {}


# ---------------------------------------------------------------------------
# The shipped file is the generator's output
# ---------------------------------------------------------------------------


def test_shipped_layout_matches_the_generator() -> None:
    """``config/openral_layout.json`` is ``build_layout(DEFAULT_CAMERAS)``.

    Regenerate with ``python -m openral_foxglove_bringup.layout --write-default``
    rather than hand-editing, so the file and the generator cannot drift.
    """
    assert json.loads(_LAYOUT.read_text()) == build_layout(list(DEFAULT_CAMERAS))


def test_layout_module_docstring_examples_run() -> None:
    """Every ``Example`` in the module's docstrings executes (CLAUDE.md §2)."""
    results = doctest.testmod(_layout, verbose=False)
    assert results.failed == 0, f"{results.failed} doctest failure(s) in layout.py"
