"""Compatibility helpers for checkpoints saved by a different lerobot version.

lerobot parses a checkpoint's ``config.json`` through draccus, which rejects any
key the installed config dataclass does not declare. Checkpoints saved by
another lerobot version or a training fork (e.g. ``kafarobotics/lerobot``'s
``n_state_dim`` on ACT) carry such keys, so ``from_pretrained`` fails before it
builds the policy. ``strip_unknown_config_keys`` drops them in place.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def strip_unknown_config_keys(config_cls: type[Any], cfg_path: Path) -> list[str]:
    """Rewrite ``cfg_path`` without the keys ``config_cls`` does not declare.

    Keeps the draccus ``type`` discriminator. Idempotent: a missing, unparsable
    or already-clean file is left untouched.

    Args:
        config_cls: The installed lerobot config dataclass (e.g. ``ACTConfig``).
        cfg_path: The checkpoint's ``config.json``.

    Returns:
        The keys that were dropped (empty when nothing changed).

    Example:
        >>> import dataclasses, json, tempfile
        >>> from pathlib import Path
        >>> @dataclasses.dataclass
        ... class Cfg:
        ...     chunk_size: int = 1
        >>> p = Path(tempfile.mkdtemp()) / "config.json"
        >>> _ = p.write_text(json.dumps({"type": "x", "chunk_size": 2, "n_state_dim": 6}))
        >>> strip_unknown_config_keys(Cfg, p)
        ['n_state_dim']
        >>> json.loads(p.read_text())
        {'type': 'x', 'chunk_size': 2}
    """
    if not cfg_path.is_file():
        return []
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict):
        return []
    valid = {f.name for f in dataclasses.fields(config_cls)} | {"type"}
    dropped = sorted(k for k in raw if k not in valid)
    if dropped:
        keep = {k: v for k, v in raw.items() if k in valid}
        cfg_path.write_text(json.dumps(keep, indent=2), encoding="utf-8")
        log.info("lerobot.config_sanitized", path=str(cfg_path), dropped=dropped)
    return dropped


def sanitize_smolvla_config(repo_id: str, *, revision: str | None = None) -> None:
    """Strip fields lerobot's ``SmolVLAConfig`` rejects from a checkpoint config.

    Resolves the checkpoint's ``config.json`` (local dir or HF cache) and runs
    ``strip_unknown_config_keys`` on it with ``SmolVLAConfig``. No-op when the
    config is already clean or lerobot / the config are unavailable. Call it
    immediately before ``SmolVLAPolicy.from_pretrained(repo_id)``.

    Args:
        repo_id: HF checkpoint id or a local checkpoint directory.
        revision: Optional revision passed to ``hf_hub_download`` for the id case.
    """
    try:
        from lerobot.policies.smolvla.configuration_smolvla import (  # noqa: PLC0415
            SmolVLAConfig,
        )
    except ImportError:
        return  # lerobot absent — the real load will raise a clearer error.

    src = Path(repo_id)
    if src.is_dir():
        cfg_path = src / "config.json"
    else:
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            cfg_path = Path(hf_hub_download(repo_id, "config.json", revision=revision))
        except Exception:
            return
    strip_unknown_config_keys(SmolVLAConfig, cfg_path)
