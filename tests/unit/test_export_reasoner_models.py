# SPDX-License-Identifier: Apache-2.0
"""``tools/export_reasoner_models.py``: openral.com's model drop-down, from the real registry."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from openral_core import REASONER_MODELS

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "export_reasoner_models", _REPO / "tools" / "export_reasoner_models.py"
)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
sys.modules["export_reasoner_models"] = tool
_spec.loader.exec_module(tool)


def test_every_offered_model_is_curated_and_needs_a_key_the_website_has() -> None:
    models = tool.export()
    assert json.loads(json.dumps(models)) == models
    ids = [m["id"] for m in models]
    assert len(ids) == len(set(ids)) and set(ids) <= set(REASONER_MODELS)
    assert {m["needs"] for m in models} <= {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", None}
    for m in models:
        entry = REASONER_MODELS[m["id"]]
        assert (m["needs"] is None) == (not entry.auth_required)
        # The provider rides in the key it needs, never in the label.
        assert not any(p in m["label"] for p in ("(Anthropic)", "(via OpenRouter)", "(OpenAI)"))
    # Every model a robot runs with an Anthropic or OpenAI key, or none, is offered.
    labels = {m["id"]: m["label"] for m in models}
    assert (labels["claude-opus-4-8"], labels["gpt-5.5"]) == ("Claude Opus 4.8", "GPT-5.5")
    by_id = {m["id"]: m["needs"] for m in models}
    assert by_id["claude-sonnet-5-5"] == "ANTHROPIC_API_KEY"
    assert by_id["gpt-5.5"] == "OPENAI_API_KEY"
    assert by_id["cosmos3-edge"] is None
