"""The reasoner settings an operator sets in the robot console (ADR-0129).

The robot keeps them in ``robot-settings.json`` beside its enrolment
(``openral_core.enrolment_path()``'s folder, mode 0600): the provider API keys and the
reasoner model, the same three settings openral.com's Account → Settings holds, in the
same shape (``GET``/``PUT /api/settings``), so the console's settings pane edits either.
A deploy applies them at start (:func:`reasoner_env`); an environment variable already
set wins, and they win over the openral.com account settings.

Secrets never leave this module in the clear: :func:`shown` gives ``set`` and the last
four characters only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from openral_core import (
    ANTHROPIC_KEY_SETTING,
    OPENAI_KEY_SETTING,
    enrolment_path,
    reasoner_model_options,
)

__all__ = [
    "MODEL_SETTING",
    "SETTINGS",
    "load_settings",
    "reasoner_env",
    "save_setting",
    "settings_path",
    "shown",
]

#: The model setting's name (the reasoner's own variable).
MODEL_SETTING = "OPENRAL_REASONER_MODEL"

#: name -> {label, secret, options?}: what the settings pane draws.
SETTINGS: dict[str, dict[str, object]] = {
    ANTHROPIC_KEY_SETTING: {"label": "Anthropic API key", "secret": True},
    OPENAI_KEY_SETTING: {"label": "OpenAI API key", "secret": True},
    MODEL_SETTING: {
        "label": "Reasoner model",
        "secret": False,
        "options": reasoner_model_options(),
    },
}

_MAX_VALUE = 4096


def settings_path() -> Path:
    """``robot-settings.json`` beside the robot's enrolment file."""
    return enrolment_path().parent / "robot-settings.json"


def load_settings(path: Path | None = None) -> dict[str, str]:
    """The saved settings (known names, string values); ``{}`` when none or unreadable."""
    try:
        raw = json.loads((path or settings_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k in SETTINGS and isinstance(v, str) and v}


def save_setting(name: str, value: str, path: Path | None = None) -> dict[str, str]:
    """Set ``name`` to ``value`` (``""`` clears it) and return the settings.

    Raises:
        ValueError: An unknown name, a value over 4096 characters, or a model that is
            not one of OpenRAL's curated reasoner models.

    Example:
        >>> import tempfile
        >>> p = Path(tempfile.mkdtemp()) / "robot-settings.json"
        >>> save_setting("OPENRAL_REASONER_MODEL", "claude-sonnet-5-5", p)
        {'OPENRAL_REASONER_MODEL': 'claude-sonnet-5-5'}
        >>> oct(p.stat().st_mode & 0o777)
        '0o600'
    """
    if name not in SETTINGS:
        raise ValueError("unknown setting")
    value = value.strip()
    if len(value) > _MAX_VALUE:
        raise ValueError("that value is too long")
    if name == MODEL_SETTING and value and value not in {o["id"] for o in reasoner_model_options()}:
        raise ValueError("pick one of the models listed")
    target = path or settings_path()
    current = load_settings(target)
    if value:
        current[name] = value
    else:
        current.pop(name, None)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(current, fh)
    os.chmod(tmp, 0o600)
    tmp.replace(target)
    return current


def shown(values: dict[str, str]) -> dict[str, dict[str, object]]:
    """What the pane may see: a secret only as set with its last four characters.

    Example:
        >>> shown({"ANTHROPIC_API_KEY": "sk-ant-secret-1234"})
        {'ANTHROPIC_API_KEY': {'set': True, 'hint': '…1234'}}
    """
    out: dict[str, dict[str, object]] = {}
    for name, value in values.items():
        if name not in SETTINGS:
            continue
        out[name] = (
            {"set": True, "hint": f"…{value[-4:]}"}
            if SETTINGS[name]["secret"]
            else {"set": True, "value": value}
        )
    return out


def reasoner_env(values: dict[str, str]) -> dict[str, str]:
    """The reasoner's environment from the saved settings (openral.com's ``robotEnv``).

    The model, the key its provider needs as ``OPENRAL_REASONER_API_KEY``, and for an
    OpenAI model OpenAI's own API (openral's default for those is OpenRouter). Nothing
    without a curated model.

    Example:
        >>> reasoner_env({"OPENRAL_REASONER_MODEL": "gpt-5.5", "OPENAI_API_KEY": "sk-x"})
        {'OPENRAL_REASONER_MODEL': 'gpt-5.5', 'OPENRAL_REASONER_API_KEY': 'sk-x', \
'OPENRAL_REASONER_ENDPOINT': 'openai'}
    """
    chosen = values.get(MODEL_SETTING)
    model = next((o for o in reasoner_model_options() if o["id"] == chosen), None)
    if model is None:
        return {}
    out = {MODEL_SETTING: str(model["id"])}
    needs = model["needs"]
    if needs and values.get(needs):
        out["OPENRAL_REASONER_API_KEY"] = values[needs]
    if needs == OPENAI_KEY_SETTING:
        out["OPENRAL_REASONER_ENDPOINT"] = "openai"
    return out
