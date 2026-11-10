"""Export the curated reasoner models for openral.com's Account → Settings (ADR-0129).

The website offers ``REASONER_MODELS`` as its model drop-down and asks for the key each
one needs: ``ANTHROPIC_API_KEY`` for an Anthropic-dialect model, ``OPENAI_API_KEY`` for
an OpenAI-dialect cloud model (sent to OpenAI's own API), none for one that runs on the
robot. ``.github/workflows/reasoner-models.yml`` writes this to the website's
``server/reasoner-models.json`` whenever the list changes on master.

Usage: python tools/export_reasoner_models.py > reasoner-models.json
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from openral_core import REASONER_MODELS


def export() -> list[dict[str, Any]]:
    """``[{id, label, needs}]`` in the registry's order.

    Example:
        >>> {m["id"]: m["needs"] for m in export()}["claude-opus-4-8"]
        'ANTHROPIC_API_KEY'
        >>> next(m for m in export() if m["id"] == "gpt-5.5")["label"]
        'GPT-5.5'
    """
    out = []
    for model in REASONER_MODELS.values():
        if not model.auth_required:
            needs = None
        elif model.dialect == "anthropic":
            needs = "ANTHROPIC_API_KEY"
        elif model.served_model_id.startswith("openai/"):
            needs = "OPENAI_API_KEY"
        else:  # a provider the website has no key for (Gemini, xAI, ...): not offered there
            continue
        # The provider rides in the key it needs: "(Anthropic)", "(via OpenRouter)" would mislead.
        label = re.sub(r"\s*\((?:Anthropic|via OpenRouter|OpenAI)\)$", "", model.display_name)
        out.append({"id": model.id, "label": label, "needs": needs})
    return out


if __name__ == "__main__":
    json.dump(export(), sys.stdout, indent=2)
    sys.stdout.write("\n")
