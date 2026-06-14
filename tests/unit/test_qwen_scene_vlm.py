"""Tests for the Qwen3.5-4B scene-VLM backend + query_scene tool (ADR-0047).

Two tiers:

* **Schema / wiring** (always run, no GPU): the ``query_scene`` reasoner tool
  round-trips through the ``ReasonerToolCall`` union, the palette gates it on
  ``scene_query_available``, the ZMQ client transport is a declared dependency,
  and ``build_scene_vlm`` rejects a non-``vlm`` manifest.
* **Real end-to-end** (gated): boot the actual sidecar, run NF4 Qwen3.5-4B on a
  real image fixture, and assert a text answer comes back over ZMQ. Skipped
  unless a sidecar venv is provisioned (``OPENRAL_QWEN_VLM_SIDECAR_VENV``) and a
  GPU is present — the legitimate CI skip path (CLAUDE.md §12).
"""

from __future__ import annotations

import os
import pathlib
import shutil

import pytest
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import RSkillManifest

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"
_REPO = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "rskills" / "qwen35-4b-nf4" / "rskill.yaml"
_RTDETR_MANIFEST = _REPO / "rskills" / "rtdetr-coco-r18" / "rskill.yaml"


# --------------------------------------------------------------------------
# Schema / wiring (always run).
# --------------------------------------------------------------------------


def test_query_scene_tool_schema_round_trips() -> None:
    """QuerySceneTool parses via the ReasonerToolCall discriminated union (ADR-0047)."""
    from openral_core import QuerySceneTool, ReasonerToolCall
    from pydantic import TypeAdapter

    parsed = TypeAdapter(ReasonerToolCall).validate_python(
        {"tool": "query_scene", "question": "Has the robot grasped the mug?", "camera": "wrist"}
    )
    assert isinstance(parsed, QuerySceneTool)
    assert parsed.question == "Has the robot grasped the mug?"
    assert parsed.camera == "wrist"
    # camera is optional (camera-agnostic): empty default, not a hardcoded name.
    assert (
        TypeAdapter(ReasonerToolCall)
        .validate_python({"tool": "query_scene", "question": "Is the table clear?"})
        .camera
        == ""
    )


def test_query_scene_requires_non_empty_question() -> None:
    from openral_core import QuerySceneTool
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QuerySceneTool(question="")


def test_query_scene_palette_gated_on_scene_query_available() -> None:
    """The LLM sees query_scene only when a scene VLM is available (ADR-0047)."""
    from openral_reasoner.palette import ToolPalette
    from openral_reasoner.tool_use import _tool_palette_to_anthropic_tools

    off = [d["name"] for d in _tool_palette_to_anthropic_tools(ToolPalette())]
    on = [
        d["name"] for d in _tool_palette_to_anthropic_tools(ToolPalette(scene_query_available=True))
    ]
    assert "query_scene" not in off
    assert "query_scene" in on


def test_query_scene_independent_of_detector_available() -> None:
    """scene_query and locate_in_view are independently provisioned (ADR-0047)."""
    from openral_reasoner.palette import ToolPalette
    from openral_reasoner.tool_use import _tool_palette_to_anthropic_tools

    tools = _tool_palette_to_anthropic_tools(ToolPalette(detector_available=True))
    names = [d["name"] for d in tools]
    # A detector being up must NOT imply the scene-VLM tool is offered.
    assert "locate_in_view" in names
    assert "query_scene" not in names


def test_qwen_vlm_extra_declares_sidecar_client_deps() -> None:
    """The node-side ZMQ client transport is a declared dependency (ADR-0047).

    Regression guard: ``QwenSceneVlm`` lazily imports ``zmq`` + ``msgpack``;
    they must ship in a real optional-dependency group so the scene_vlm_node's
    query_scene service doesn't fail per-request with a bare "No module named
    'zmq'".
    """
    import tomllib

    with (_REPO / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    groups = pyproject["dependency-groups"]
    assert "qwen-vlm" in groups, "missing `qwen-vlm` dependency group"
    group = " ".join(groups["qwen-vlm"])
    assert "pyzmq" in group and "msgpack" in group


def test_build_scene_vlm_rejects_non_vlm_manifest() -> None:
    """build_scene_vlm only accepts kind='vlm' manifests (ADR-0047)."""
    from openral_runner.backends.gstreamer.qwen_scene_vlm import build_scene_vlm

    m = RSkillManifest.from_yaml(str(_RTDETR_MANIFEST))  # kind: detector
    with pytest.raises(ROSConfigError, match="kind='vlm'"):
        build_scene_vlm(m)


def test_build_scene_vlm_from_qwen_manifest() -> None:
    """build_scene_vlm wires model_id + weights_source from the rSkill manifest."""
    from openral_runner.backends.gstreamer.qwen_scene_vlm import QwenSceneVlm, build_scene_vlm

    m = RSkillManifest.from_yaml(str(_MANIFEST))
    vlm = build_scene_vlm(m, port=5759)
    assert isinstance(vlm, QwenSceneVlm)
    # Construction is side-effect-free: no sidecar spawned, no socket opened.
    assert vlm._sock is None
    assert vlm._child is None


def test_query_empty_question_raises_before_connect() -> None:
    """An empty question fails fast without touching the sidecar."""
    from openral_runner.backends.gstreamer.qwen_scene_vlm import QwenSceneVlm

    vlm = QwenSceneVlm(model_id="OpenRAL/rskill-qwen35-4b-nf4", auto_spawn=False)
    with pytest.raises(ROSConfigError, match="non-empty"):
        vlm.query(b"\x00" * 12, 2, 2, "   ")


# --------------------------------------------------------------------------
# Real end-to-end through the sidecar (gated).
# --------------------------------------------------------------------------


def _gpu_present() -> bool:
    return shutil.which("nvidia-smi") is not None


@pytest.mark.skipif(
    not os.environ.get("OPENRAL_QWEN_VLM_SIDECAR_VENV") or not _gpu_present(),
    reason="needs a provisioned Qwen VLM sidecar venv + a local GPU "
    "(set OPENRAL_QWEN_VLM_SIDECAR_VENV).",
)
def test_e2e_query_coco_sample() -> None:
    """Real NF4 Qwen3.5-4B answers a scene question about the COCO fixture over ZMQ."""
    import numpy as np
    from openral_runner.backends.gstreamer.qwen_scene_vlm import QwenSceneVlm
    from PIL import Image

    img = Image.open(_FIXTURES / "coco_sample.jpg").convert("RGB")
    w, h = img.size
    bgr = np.asarray(img)[:, :, ::-1].tobytes()  # RGB -> BGR bytes

    vlm = QwenSceneVlm(
        model_id="OpenRAL/rskill-qwen35-4b-nf4",
        port=5760,
        boot_timeout_s=1800,
    )
    try:
        answer = vlm.query(bgr, w, h, "How many cats are in this image? Answer with a number.")
        assert isinstance(answer, str) and answer.strip()
        # coco_sample.jpg is the canonical two-cats image; the answer should
        # mention two / 2. Proves the real multimodal path returns grounded text.
        assert "2" in answer or "two" in answer.lower()
    finally:
        vlm.close()
