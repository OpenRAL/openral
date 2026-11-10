"""The tracker's locator query carries contrast labels only for a class-naming locator.

Review of PR #521: LocateAnything names a box with no ``<ref>`` with the whole query
(``locateanything_detector.build_objects_metadata``'s fallback label), so a query with the
contrast labels appended came back labelled "stock object, can, ..." and the tracker's label
filter (``contrasted``) dropped every detection. Real parser, real filter, real launch file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from openral_runner.backends.gstreamer.locateanything_detector import build_objects_metadata

_node = pytest.importorskip(
    "openral_world_state_ros.object_tracker_node",
    reason="the tracker node's package (packages/world_state) is not on the path",
)
_LAUNCH = (
    Path(__file__).resolve().parents[2] / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"
)
_ANSWER = "<box><100><200><300><400></box>"  # LocateAnything with no <ref>: no class label


def _kept(answer: str, label: str, *, contrast: bool) -> int:
    query = _node.locate_query(label, contrast=contrast)
    md = build_objects_metadata(
        answer, width=640, height=360, model_id="m", sensor_id="head", fallback_label=query
    )
    assert md is not None
    own = _node.own_detections(md.model_dump_json(), label, contrast=contrast)
    return len(json.loads(own)["detections"])


def test_an_unlabelled_locateanything_box_survives_only_without_contrast_labels() -> None:
    assert _kept(_ANSWER, "stock object", contrast=True) == 0, "the bug: no name of the label"
    assert _kept(_ANSWER, "stock object", contrast=False) == 1


def test_without_contrast_labels_every_locateanything_box_is_kept() -> None:
    """Review of #521: the label filter ran without contrast labels too, so a synonym prompt's
    fallback label ("orange, citrus") or a ``<ref>`` phrase ("the orange") was dropped."""
    assert _kept(_ANSWER, "orange, citrus", contrast=False) == 1
    ref = "<ref>the orange</ref><box><100><200><300><400></box>"
    assert _kept(ref, "orange", contrast=False) == 1


def test_the_launch_asks_for_contrast_labels_only_from_an_omdet_locator() -> None:
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_contrast", _LAUNCH)
    assert spec is not None and spec.loader is not None
    launch = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_contrast"] = launch
    spec.loader.exec_module(launch)
    pick = launch._locate_contrast_labels
    assert pick([{"engine": "zeroshot_hf"}, {"engine": ""}]) is True
    assert pick([{"engine": ""}, {"engine": "zeroshot_hf"}]) is False
    assert pick([]) is False  # no locator: no locate requests at all
