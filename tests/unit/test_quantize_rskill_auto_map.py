"""Tests for the quantize_rskill ``auto_map`` completeness check.

A ``trust_remote_code`` rSkill whose ``auto_map`` references a ``*.py`` module
that did not make it into the quantized bundle fails ``from_pretrained`` at load
time in a production deploy ("<repo> does not appear to have a file named
<module>.py" — the molmoact2-libero-nf4 image/video_processing.py regression).
``_verify_auto_map_complete`` catches that at quantization time instead.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "quantize_rskill", REPO_ROOT / "tools" / "quantize_rskill.py"
)
assert _spec is not None and _spec.loader is not None
quantize_rskill = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = quantize_rskill
_spec.loader.exec_module(quantize_rskill)


def _write(out_dir: Path, name: str, payload: dict) -> None:
    (out_dir / name).write_text(json.dumps(payload))


def test_auto_map_modules_collects_from_all_configs(tmp_path: Path) -> None:
    # The molmoact2 shape: config.json names the model/config code; the
    # processor_config.json names the (image/video) processors — including a
    # [slow, fast] tuple and a None slot.
    _write(
        tmp_path,
        "config.json",
        {
            "auto_map": {
                "AutoConfig": "configuration_molmoact2.MolmoAct2Config",
                "AutoModelForCausalLM": "modeling_molmoact2.MolmoAct2ForConditionalGeneration",
            }
        },
    )
    _write(
        tmp_path,
        "processor_config.json",
        {
            "auto_map": {
                "AutoProcessor": "processing_molmoact2.MolmoAct2Processor",
                "AutoImageProcessor": [
                    "image_processing_molmoact2.Slow",
                    "image_processing_molmoact2.Fast",
                ],
                "AutoVideoProcessor": "video_processing_molmoact2.MolmoAct2VideoProcessor",
                "AutoFeatureExtractor": None,
            }
        },
    )
    assert quantize_rskill._auto_map_modules(tmp_path) == {
        "configuration_molmoact2.py",
        "modeling_molmoact2.py",
        "processing_molmoact2.py",
        "image_processing_molmoact2.py",
        "video_processing_molmoact2.py",
    }


def test_verify_raises_on_missing_auto_map_module(tmp_path: Path) -> None:
    # The exact molmoact2 regression: processor references video_processing but
    # the bundle only shipped processing_molmoact2.py.
    _write(
        tmp_path,
        "processor_config.json",
        {
            "auto_map": {
                "AutoProcessor": "processing_molmoact2.MolmoAct2Processor",
                "AutoVideoProcessor": "video_processing_molmoact2.MolmoAct2VideoProcessor",
            }
        },
    )
    (tmp_path / "processing_molmoact2.py").write_text("# present")
    with pytest.raises(RuntimeError, match=r"video_processing_molmoact2\.py"):
        quantize_rskill._verify_auto_map_complete(tmp_path)


def test_verify_passes_when_all_modules_present(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "processor_config.json",
        {
            "auto_map": {
                "AutoProcessor": "processing_molmoact2.MolmoAct2Processor",
                "AutoVideoProcessor": "video_processing_molmoact2.X",
            }
        },
    )
    for stem in ("processing_molmoact2", "video_processing_molmoact2"):
        (tmp_path / f"{stem}.py").write_text("# present")
    quantize_rskill._verify_auto_map_complete(tmp_path)  # must not raise


def test_verify_noop_when_no_auto_map(tmp_path: Path) -> None:
    # A plain (non-trust_remote_code) checkpoint has no auto_map → nothing to check.
    _write(tmp_path, "config.json", {"model_type": "smolvla"})
    quantize_rskill._verify_auto_map_complete(tmp_path)  # must not raise
