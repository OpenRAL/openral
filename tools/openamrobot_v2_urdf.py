"""Build ``robots/openamrobot_v2/openamrobot_v2.urdf``: upstream's URDF plus the camera overlay.

``robots/openamrobot_v2/upstream/combined_robot.urdf`` is upstream's own expansion of
``openamr-upperbody-sw`` ``combined_robot.xacro`` with its default arguments plus
``sim_dynamics:=on``, vendored byte-for-byte. OpenRAL changes exactly two things:

* the wrist cameras: upstream's are placeholders (marker boxes 5 cm beside the hand that
  never see the fingers); they go to OpenArm's wrist-camera mount, the view the OpenArm
  rSkills were trained on, and their marker boxes are dropped;
* the head camera: upstream's assembly moves rigidly onto the head ZED lens at the OpenArm
  shelf cell's pose (``robots/openamrobot_v2/README.md`` "Head camera").

It also drops upstream's ``<ros2_control>`` and ``<transmission>`` blocks:
controller_manager configuration OpenRAL never reads, whose ``<joint>`` entries would
otherwise look like URDF joints to a parser walking the whole tree. The ``<gazebo>``
blocks stay: their ``DiffDrive`` plugin is upstream's own declaration of the drive
wheels, track, radius and acceleration limits, which the simulators read.
Everything else, comments aside, is upstream's.

Usage::

    python tools/openamrobot_v2_urdf.py                # rewrite openamrobot_v2.urdf
    python tools/openamrobot_v2_urdf.py --check        # exit 1 if it is stale
    python tools/openamrobot_v2_urdf.py --expand UB PLATFORM OPENARM_DESC
        # re-vendor upstream/combined_robot.urdf from three checkouts (needs ROS 2 xacro)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

_DIR = Path(__file__).resolve().parents[1] / "robots" / "openamrobot_v2"
UPSTREAM_URDF: Path = _DIR / "upstream" / "combined_robot.urdf"
OVERLAID_URDF: Path = _DIR / "openamrobot_v2.urdf"

#: Every xacro argument at upstream's default except ``sim_dynamics`` (its MuJoCo output).
XACRO_ARGS: tuple[str, ...] = ("sim_dynamics:=on",)

# joint -> (parent link, xyz, rpy). OpenArm twin `camera_wrist_left/right` mount on
# `openarm_*_ee_base_link` (openarm_v20_bimanual.xml); the head lens pose 2.5 mm behind
# the lens with upstream's 30 deg pitch undone (README "Regenerating").
CAMERA_MOUNTS: dict[str, tuple[str | None, str, str]] = {
    "left_wrist_camera_joint": (
        "openarm_left_ee_base_link",
        "0.023 0.0182 -0.10236",
        "-2.100862 1.142449 -2.143093",
    ),
    "right_wrist_camera_joint": (
        "openarm_right_ee_base_link",
        "0.023 -0.0182 -0.10236",
        "2.10087 1.142449 2.143093",
    ),
    "head_camera_mount_joint": (None, "0.258568 0.027277 0.267553", "-0.028403 0.621812 -0.00224"),
    "head_camera_optical_joint": (None, "0.0025 0 0", "-1.5707963267948966 0 -1.5707963267948966"),
}
#: Top-level elements that configure ros2_control, not the robot's kinematics.
NON_DESCRIPTION_TAGS: tuple[str, ...] = ("ros2_control", "transmission")
#: Links whose upstream placeholder visual/collision boxes are dropped.
MARKERLESS_LINKS: tuple[str, ...] = ("left_wrist_camera_link", "right_wrist_camera_link")


def overlay(upstream_xml: str) -> str:
    """Return ``upstream_xml`` with the camera overlay applied (comments, ros2_control dropped).

    Example:
        >>> out = overlay(UPSTREAM_URDF.read_text())
        >>> 'xyz="0.023 0.0182 -0.10236"' in out
        True
    """
    root = ET.fromstring(upstream_xml)
    for tag in NON_DESCRIPTION_TAGS:
        for el in root.findall(tag):
            root.remove(el)
    joints = {j.get("name"): j for j in root.findall("joint")}
    links = {link.get("name"): link for link in root.findall("link")}
    for name, (parent, xyz, rpy) in CAMERA_MOUNTS.items():
        joint = joints[name]
        if parent is not None:
            joint.find("parent").set("link", parent)  # type: ignore[union-attr]  # reason: URDF joints always carry <parent>
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin")
        origin.set("xyz", xyz)
        origin.set("rpy", rpy)
    for name in MARKERLESS_LINKS:
        for tag in ("visual", "collision"):
            for el in links[name].findall(tag):
                links[name].remove(el)
    ET.indent(root, space="  ")
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def expand(upperbody: Path, platform: Path, openarm_description: Path) -> str:
    """Run upstream's ``combined_robot.xacro`` against three checkouts (needs ``xacro`` on PATH)."""
    pkgs = {
        "openarm_description": openarm_description,
        "openamrobot_description": platform / "ros2/src/openamrobot_description",
        "openamrobot_upperbody_description": upperbody
        / "ros2/src/openamrobot_upperbody_description",
        "openamrobot_upperbody_control": upperbody / "ros2/src/openamrobot_upperbody_control",
    }
    with tempfile.TemporaryDirectory() as prefix:
        index = Path(prefix) / "share/ament_index/resource_index/packages"
        index.mkdir(parents=True)
        for name, src in pkgs.items():
            (Path(prefix) / "share" / name).symlink_to(src.resolve())
            (index / name).touch()
        env = dict(
            os.environ, AMENT_PREFIX_PATH=f"{prefix}:{os.environ.get('AMENT_PREFIX_PATH', '')}"
        )
        # Relative to the prefix, so xacro's "autogenerated from" header names no local path.
        xacro = "share/openamrobot_upperbody_description/urdf/combined_robot.xacro"
        return subprocess.run(
            ["xacro", xacro, *XACRO_ARGS],
            cwd=prefix,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout


def main(argv: list[str] | None = None) -> int:
    """CLI entry: rewrite (default), ``--check`` or ``--expand``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if openamrobot_v2.urdf is stale"
    )
    parser.add_argument(
        "--expand", nargs=3, type=Path, metavar=("UPPERBODY", "PLATFORM", "OPENARM_DESC")
    )
    args = parser.parse_args(argv)
    if args.expand:
        UPSTREAM_URDF.write_text(expand(*args.expand))
    want = overlay(UPSTREAM_URDF.read_text())
    if args.check:
        if OVERLAID_URDF.read_text() != want:
            print(
                f"{OVERLAID_URDF} is stale: run python tools/openamrobot_v2_urdf.py",
                file=sys.stderr,
            )
            return 1
        return 0
    OVERLAID_URDF.write_text(want)
    return 0


if __name__ == "__main__":
    sys.exit(main())
