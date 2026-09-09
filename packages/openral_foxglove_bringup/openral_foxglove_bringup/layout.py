"""Generator for the OpenRAL Foxglove layout (``config/openral_layout.json``).

A Foxglove layout is imported *client-side*, so a ROS launch argument cannot
reach it — the camera panels in a shipped layout are baked in. Scenes disagree
about camera names (``behavior_r1pro`` publishes ``head`` /``left_wrist`` /
``right_wrist``; ``isaac_franka`` publishes ``top``), so this module builds the
layout from the scene's actual camera list instead:

.. code-block:: bash

   python -m openral_foxglove_bringup.layout --cameras head left_wrist right_wrist

The shipped ``config/openral_layout.json`` is the output of this module for
:data:`DEFAULT_CAMERAS`; regenerate it with ``--write-default``.

Layout shape — one hero scene, the cameras beside it, everything else tabbed:

.. code-block:: text

   ┌───────────────────────────┬──────────────┐
   │ 3D · robot + environment  │ camera 0     │
   │ (URDF · TF · map · voxels │ camera 1     │
   │  · collisions · odom)     │ camera 2     │
   ├───────────────────────────┼──────────────┤
   │ tabs: nav map · joints ·  │ tabs: log ·  │
   │ collisions · policy state │ health · …   │
   └───────────────────────────┴──────────────┘

Every topic referenced here is on ``topics.BUCKET1_TOPIC_WHITELIST`` — the
bridge exposes nothing else, and ``test/test_foxglove_launch.py`` enforces it.
The surface stays read-only: no Publish, Teleop, Call Service or Parameters
panel appears in any generated layout.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Camera slots used by the shipped layout. ``top`` leads because it is the
#: 3rd-person overview slot verified against the LIBERO + Franka-Panda deploy
#: scene; the two wrist slots cover the three-camera humanoid scenes. A slot a
#: scene does not publish renders as an empty panel — pick another from the
#: panel's topic dropdown, or regenerate with ``--cameras``.
#: Camera slots the shipped layout is generated for. These are *sensor names*
#: from the robot manifest, not free labels — `camera_image_topic` builds
#: `/openral/cameras/<slot>/image` from them, and a slot that does not exist
#: renders as "Image topic does not exist" in an otherwise healthy panel.
#: `wrist_left` / `wrist_right` is the spelling `robots/*/robot.yaml` uses
#: (openarm, and two other manipulators); the transposed `left_wrist` this
#: shipped with matched no camera on any robot in the repo.
DEFAULT_CAMERAS: tuple[str, ...] = ("top", "wrist_left", "wrist_right")

#: Where the generated layout is shipped, relative to the package root.
DEFAULT_LAYOUT_PATH: str = "config/openral_layout.json"


def camera_image_topic(camera: str, *, compressed: bool = False) -> str:
    """Return the image topic for one camera slot.

    Args:
        camera: Camera slot name as it appears in a scene's ``cameras:`` list
            (e.g. ``"left_wrist"``).
        compressed: Point at the ``image_transport`` ``/compressed`` sibling
            (published only when ``foxglove.launch.py`` runs with
            ``republish_compressed:=true``).

    Returns:
        The full ROS topic name.

    Example:
        >>> camera_image_topic("left_wrist")
        '/openral/cameras/left_wrist/image'
        >>> camera_image_topic("top", compressed=True)
        '/openral/cameras/top/image/compressed'
    """
    suffix = "/compressed" if compressed else ""
    return f"/openral/cameras/{camera}/image{suffix}"


def _scene_panel(follow_frame: str) -> dict[str, Any]:
    """3D panel config for the hero view: the robot inside its environment."""
    return {
        "followTf": follow_frame,
        "scene": {"transforms": {"showLabel": False}},
        "cameraState": {
            "perspective": True,
            "distance": 5,
            "phi": 60,
            "thetaOffset": 45,
            "targetOffset": [0, 0, 0],
            "fovy": 45,
            "near": 0.05,
            "far": 5000,
        },
        "topics": {
            # The robot itself: URDF from /robot_description, posed by /tf.
            "/robot_description": {"visible": True},
            # The environment around it.
            "/map": {"visible": True},
            "/octomap_point_cloud_centers": {
                "visible": True,
                "colorField": "z",
                "colorMode": "colormap",
                "colorMap": "turbo",
                "pointSize": 0.03,
            },
            "/openral/world_voxels_cloud": {
                "visible": True,
                "colorField": "z",
                "colorMode": "colormap",
                "colorMap": "turbo",
                "pointSize": 0.04,
            },
            "/openral/world_collisions_markers": {"visible": True},
            "/odom": {"visible": True},
            "/scan": {"visible": True},
        },
        "layers": {},
    }


def _nav_panel() -> dict[str, Any]:
    """3D panel config for the top-down 2D navigation view."""
    return {
        "followTf": "map",
        "scene": {},
        "cameraState": {
            "perspective": False,
            "distance": 12,
            "phi": 0,
            "thetaOffset": 0,
            "targetOffset": [0, 0, 0],
            "fovy": 45,
            "near": 0.5,
            "far": 5000,
        },
        "topics": {
            "/map": {"visible": True},
            "/odom": {"visible": True},
            "/scan": {"visible": True},
        },
        "layers": {},
        # Read-only: no publish click-target is configured.
        "publish": {"type": "point", "poseTopic": "", "pointTopic": ""},
    }


def _bucket2_panel(follow_frame: str) -> dict[str, Any]:
    """3D panel config for the Bucket-2 converter outputs, close in on the robot."""
    return {
        "followTf": follow_frame,
        "scene": {},
        "cameraState": {
            "perspective": True,
            "distance": 4,
            "phi": 45,
            "thetaOffset": 30,
            "targetOffset": [0, 0, 0],
            "fovy": 45,
            "near": 0.05,
            "far": 5000,
        },
        "topics": {
            "/openral/world_collisions_markers": {
                "visible": True,
                "colorField": "z",
                "colorMode": "rgba",
            },
            "/openral/world_voxels_cloud": {
                "visible": True,
                "colorField": "z",
                "colorMode": "colormap",
                "colorMap": "turbo",
                "pointSize": 0.04,
            },
        },
        "layers": {},
    }


def _plot(paths: Sequence[tuple[str, bool]], *, seconds: float = 30) -> dict[str, Any]:
    """Plot panel config from ``(message_path, enabled)`` pairs."""
    return {
        "paths": [
            {"value": value, "enabled": enabled, "timestampMethod": "receiveTime"}
            for value, enabled in paths
        ],
        "showLegend": True,
        "xAxisVal": "timestamp",
        "followingViewWidth": seconds,
    }


def _tab(tabs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Tab panel config from ``(title, child_panel_id)`` pairs."""
    return {
        "activeTabIdx": 0,
        "tabs": [{"title": title, "layout": panel_id} for title, panel_id in tabs],
    }


def _split(direction: str, first: Any, second: Any, percentage: float) -> dict[str, Any]:
    """One node of Foxglove's binary layout tree."""
    return {
        "direction": direction,
        "first": first,
        "second": second,
        "splitPercentage": percentage,
    }


def _stack(panel_ids: Sequence[str], direction: str = "column") -> Any:
    """Split ``panel_ids`` evenly along ``direction`` as a nested layout tree."""
    if not panel_ids:
        raise ValueError("panel_ids must not be empty")
    if len(panel_ids) == 1:
        return panel_ids[0]
    # The first panel takes 1/n of the space; the rest split the remainder.
    head, *tail = panel_ids
    return _split(direction, head, _stack(tail, direction), round(100.0 / len(panel_ids), 2))


def build_layout(
    cameras: Sequence[str] = DEFAULT_CAMERAS,
    *,
    compressed: bool = False,
    follow_frame: str = "base_link",
) -> dict[str, Any]:
    """Build the OpenRAL Foxglove layout for a scene's camera slots.

    The hero 3D panel draws the robot (URDF + TF) inside its environment (map,
    octomap voxels, Bucket-2 collision markers, odometry, laser). One Image
    panel per entry in ``cameras`` stacks beside it. The remaining telemetry —
    node logs, diagnostics, world state, mission/episode transitions, reward,
    detected objects, the topic graph — is tabbed so it is one click away
    without crowding the scene.

    Args:
        cameras: Camera slot names, in display order. Typically a deploy
            scene's ``cameras:`` list.
        compressed: Point the Image panels at the ``/compressed`` siblings
            (needs ``republish_compressed:=true`` on the bridge launch).
        follow_frame: TF frame the 3D panels follow. Must be a frame the
            robot actually broadcasts: Foxglove renders nothing at all — no
            model, no clouds, no markers — when the follow frame is absent
            from TF, so the ROS-conventional ``base_link`` default is wrong
            for any robot that names its root otherwise (OpenArm broadcasts
            ``openarm_base``). Pass ``RobotDescription.base_frame``. ``map``
            pins the view to the world origin instead.

    Returns:
        A Foxglove layout dict, ready to serialise to JSON and import.

    Raises:
        ValueError: If ``cameras`` is empty.

    Example:
        >>> layout = build_layout(["head", "left_wrist"])
        >>> sorted(k for k in layout["configById"] if k.startswith("Image"))
        ['Image!cam_0', 'Image!cam_1']
        >>> layout["configById"]["Image!cam_0"]["imageMode"]["imageTopic"]
        '/openral/cameras/head/image'
        >>> layout["configById"]["3D!scene"]["followTf"]
        'base_link'
    """
    if not cameras:
        raise ValueError("cameras must not be empty — the layout needs at least one Image panel")

    camera_ids = [f"Image!cam_{i}" for i, _ in enumerate(cameras)]
    config_by_id: dict[str, Any] = {
        # ---- hero: the robot in its environment -------------------------
        "3D!scene": _scene_panel(follow_frame),
        # ---- the cameras -------------------------------------------------
        **{
            panel_id: {"imageMode": {"imageTopic": camera_image_topic(name, compressed=compressed)}}
            for panel_id, name in zip(camera_ids, cameras, strict=True)
        },
        # ---- scene-side tabs ---------------------------------------------
        "3D!nav": _nav_panel(),
        # Bucket-2 close-up: the converter's collision capsules + voxel grid on
        # their own, tight on the robot. The hero panel shows the same two
        # topics in world context; this one is for inspecting the geometry.
        "3D!bucket2": _bucket2_panel(follow_frame),
        # ``[:]`` slices every joint, so the plot fits any DOF count instead
        # of the six indices the hand-written layout hard-coded.
        "Plot!joints": _plot(
            [("/joint_states.position[:]", True), ("/joint_states.velocity[:]", False)]
        ),
        "Plot!state": _plot(
            [
                ("/openral/policy_state.data[:]", True),
                ("/openral/world_state_fast.staleness_ms[:]", False),
                ("/openral/world_state_fast.battery_pct", False),
            ]
        ),
        # ---- telemetry tabs ----------------------------------------------
        # Node log stream — the closest native equivalent to the dashboard's
        # event log. (The dashboard keeps its OTel-fed log; this is the ROS one.)
        "RosOut!log": {
            "searchTerms": [],
            "minLogLevel": 2,  # INFO and above
            "topicToRender": "/rosout",
        },
        "DiagnosticSummary!health": {
            "minLevel": 0,
            "pinnedIds": [],
            "topicToRender": "/diagnostics",
            "hardwareIdFilter": "",
            "sortByLevel": True,
        },
        # WorldStateStamped carries joint state, base pose/twist, EE poses,
        # per-component diagnostics + staleness, battery and detected objects
        # in one message — most of the dashboard's World state card.
        "RawMessages!world": {"topicPath": "/openral/world_state_fast"},
        # Mission traceability: episode boundaries, phase and success as a
        # timeline rather than a number that scrolls past.
        "StateTransitions!mission": {
            "paths": [
                {"value": "/openral/episode.phase", "timestampMethod": "receiveTime"},
                {"value": "/openral/episode.success", "timestampMethod": "receiveTime"},
                {"value": "/openral/episode.task_string", "timestampMethod": "receiveTime"},
            ],
            "isSynced": True,
        },
        "Plot!reward": _plot(
            [("/openral/critic/score.score", True), ("/openral/critic/score.threshold", True)],
            seconds=120,
        ),
        "RawMessages!objects": {"topicPath": "/openral/perception/objects"},
        "RawMessages!attachment": {"topicPath": "/openral/attachment_state"},
        "TopicGraph!graph": {},
        # ---- the two tab containers --------------------------------------
        "Tab!scene_aux": _tab(
            [
                ("Nav · 2D map", "3D!nav"),
                ("Joints", "Plot!joints"),
                ("Collisions · voxels", "3D!bucket2"),
                ("Policy state", "Plot!state"),
            ]
        ),
        "Tab!telemetry": _tab(
            [
                ("Log", "RosOut!log"),
                ("Health", "DiagnosticSummary!health"),
                ("World state", "RawMessages!world"),
                ("Mission", "StateTransitions!mission"),
                ("Reward", "Plot!reward"),
                ("Objects", "RawMessages!objects"),
                ("Attachments", "RawMessages!attachment"),
                ("Topics", "TopicGraph!graph"),
            ]
        ),
    }

    return {
        "configById": config_by_id,
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1},
        "layout": _split(
            "row",
            _split("column", "3D!scene", "Tab!scene_aux", 62),
            _split("column", _stack(camera_ids), "Tab!telemetry", 58),
            62,
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the generator CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m openral_foxglove_bringup.layout",
        description="Generate an OpenRAL Foxglove layout for a scene's camera slots.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        metavar="NAME",
        help=f"camera slot names, in display order (default: {' '.join(DEFAULT_CAMERAS)})",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="point the Image panels at the /compressed siblings",
    )
    parser.add_argument(
        "--follow-frame",
        default="base_link",
        help="TF frame the hero 3D panel follows (default: base_link)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the layout here (default: stdout)",
    )
    parser.add_argument(
        "--write-default",
        action="store_true",
        help=f"overwrite the shipped {DEFAULT_LAYOUT_PATH} in this package",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render a layout to stdout, to ``--output``, or over the shipped default.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        Process exit code (``0`` on success).

    Example:
        >>> main(["--cameras", "top", "-o", "/dev/null"])
        0
    """
    args = _parse_args(argv)
    layout = build_layout(args.cameras, compressed=args.compressed, follow_frame=args.follow_frame)
    text = json.dumps(layout, indent=1) + "\n"

    destination: Path | None = args.output
    if args.write_default:
        destination = Path(__file__).resolve().parent.parent / DEFAULT_LAYOUT_PATH
    if destination is None:
        print(text, end="")
    else:
        destination.write_text(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
