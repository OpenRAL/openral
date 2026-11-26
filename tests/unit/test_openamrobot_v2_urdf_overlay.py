"""openamrobot_v2.urdf is upstream's vendored expansion plus the camera overlay, nothing else."""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "openamrobot_v2_urdf.py"
_spec = importlib.util.spec_from_file_location("openamrobot_v2_urdf", _TOOL)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_committed_urdf_is_the_overlay_of_the_vendored_upstream() -> None:
    assert tool.main(["--check"]) == 0


def test_overlay_touches_only_the_camera_joints_and_markers() -> None:
    up = ET.parse(tool.UPSTREAM_URDF).getroot()
    ours = ET.parse(tool.OVERLAID_URDF).getroot()
    touched = set(tool.CAMERA_MOUNTS) | set(tool.MARKERLESS_LINKS)

    def by_name(root: ET.Element) -> dict[str, str]:
        return {
            f"{el.tag}:{el.get('name')}": "".join(ET.tostring(el, encoding="unicode").split())
            for el in root
            if el.get("name") not in touched and el.tag not in tool.NON_DESCRIPTION_TAGS
        }

    assert by_name(up) == by_name(ours)
