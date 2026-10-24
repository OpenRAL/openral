"""Tests for the Qwen3.5-4B scene-VLM backend + query_scene tool.

Two tiers:

* **Schema / wiring** (always run, no GPU): the ``query_scene`` reasoner tool
  round-trips through the ``ReasonerToolCall`` union, the palette gates it on
  ``scene_query_available``, the ZMQ client transport is a declared dependency,
  and ``build_scene_vlm`` rejects a non-``vlm`` manifest.
* **Real end-to-end** (gated): boot the actual sidecar, run NF4 Qwen3.5-4B on a
  real image fixture, and assert a text answer comes back over ZMQ. Skipped
  unless a sidecar venv is provisioned (``OPENRAL_QWEN_VLM_SIDECAR_VENV``) and a
  GPU is present — the legitimate CI skip path (CLAUDE.md §1.12).
"""

from __future__ import annotations

import os
import pathlib
import shutil
from collections.abc import Callable

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
    """QuerySceneTool parses via the ReasonerToolCall discriminated union."""
    from openral_core import QuerySceneTool, ReasonerToolCall
    from pydantic import TypeAdapter

    parsed: ReasonerToolCall = TypeAdapter(ReasonerToolCall).validate_python(
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
    """The LLM sees query_scene only when a scene VLM is available."""
    from openral_reasoner.palette import ToolPalette
    from openral_reasoner.tool_use import _tool_palette_to_anthropic_tools

    off = [d["name"] for d in _tool_palette_to_anthropic_tools(ToolPalette())]
    on = [
        d["name"] for d in _tool_palette_to_anthropic_tools(ToolPalette(scene_query_available=True))
    ]
    assert "query_scene" not in off
    assert "query_scene" in on


def test_query_scene_independent_of_detector_available() -> None:
    """scene_query and locate_in_view are independently provisioned."""
    from openral_reasoner.palette import ToolPalette
    from openral_reasoner.tool_use import _tool_palette_to_anthropic_tools

    tools = _tool_palette_to_anthropic_tools(ToolPalette(detector_available=True))
    names = [d["name"] for d in tools]
    # A detector being up must NOT imply the scene-VLM tool is offered.
    assert "locate_in_view" in names
    assert "query_scene" not in names


def test_qwen_vlm_extra_declares_sidecar_client_deps(
    expand_dependency_group: Callable[..., list[str]],
) -> None:
    """The node-side ZMQ client transport is a declared dependency.

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

    group = " ".join(expand_dependency_group(groups, "qwen-vlm"))
    assert "pyzmq" in group and "msgpack" in group


def test_build_scene_vlm_rejects_non_vlm_manifest() -> None:
    """build_scene_vlm only accepts kind='vlm' manifests."""
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

    vlm = QwenSceneVlm(model_id="OpenRAL/rskill-qwen35_4b-any-general-nf4", auto_spawn=False)
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
        model_id="OpenRAL/rskill-qwen35_4b-any-general-nf4",
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


def test_a_default_client_leaves_the_token_budget_to_the_sidecar() -> None:
    """Spark R230 (2026-10-22): a 256 sent with every query overrode the sidecar's 1024.

    Thinking is on, so a trace (~801 tokens) never closed in 256 and every answer failed.
    A ZMQ reply socket at the process boundary records what the client sends.
    """
    msgpack = pytest.importorskip("msgpack")
    zmq = pytest.importorskip("zmq")
    import threading

    from openral_runner.backends.gstreamer.qwen_scene_vlm import QwenSceneVlm

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    port = rep.bind_to_random_port("tcp://127.0.0.1")
    seen: list[dict[str, object]] = []

    def serve() -> None:
        while len(seen) < 4:  # a ping + a query per client
            req = msgpack.unpackb(rep.recv(), raw=False)
            seen.append(req)
            reply = {"ok": True} if req.get("op") == "ping" else {"ok": True, "answer": "yes"}
            rep.send(msgpack.packb(reply, use_bin_type=True))

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        default = QwenSceneVlm(model_id="m", port=port, auto_spawn=False)
        assert default.query(b"\x00" * 12, 2, 2, "done?") == "yes"
        default.close()
        explicit = QwenSceneVlm(model_id="m", port=port, auto_spawn=False, max_new_tokens=64)
        explicit.query(b"\x00" * 12, 2, 2, "done?")
        explicit.close()
    finally:
        worker.join(timeout=5)
        rep.close(linger=0)
    queries = [r for r in seen if r.get("op") == "query"]
    assert "max_new_tokens" not in queries[0], "the sidecar's own budget applies"
    assert queries[1]["max_new_tokens"] == 64


def test_a_sidecar_that_never_boots_is_ended_with_the_failed_warm_up(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boot timeout raised with the spawned child still running (#421 review): a
    failed warm-up left a process holding the model's GPU memory behind."""
    pytest.importorskip("msgpack")
    pytest.importorskip("zmq")
    import socket

    from openral_runner.backends.gstreamer.qwen_scene_vlm import QwenSceneVlm

    pid_file = tmp_path / "pid"
    script = tmp_path / "silent_sidecar.py"  # the sidecar's argv, never answers a ping
    script.write_text(
        "import os, sys, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(120)\n"
    )
    monkeypatch.setenv("OPENRAL_QWEN_VLM_SIDECAR", str(script))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    vlm = QwenSceneVlm(model_id="m", port=port, boot_timeout_s=3.0)
    with pytest.raises(ROSConfigError, match="not ready"):
        vlm.warm_up()
    pid = int(pid_file.read_text())
    assert not pathlib.Path(f"/proc/{pid}").exists(), "the spawned sidecar outlived the failed boot"
