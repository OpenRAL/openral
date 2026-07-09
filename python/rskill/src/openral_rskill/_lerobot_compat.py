"""Compatibility shim for ``lerobot.policies`` import side-effects.

Importing ``lerobot.policies`` (any submodule) eagerly initialises
``lerobot.policies.groot``. Upstream ``lerobot==0.5.0/0.5.1`` shipped a
``GR00TN15Config`` dataclass that failed to construct on Python 3.12 with
``transformers>=5.3`` (``TypeError: non-default argument 'backbone_cfg'
follows default argument``), which crashed the whole package import.

For that broken combination this module installs a stub
``lerobot.policies.groot.modeling_groot`` so the package initialises.

On ``lerobot>=0.6.0`` GR00T-N1.7 is first-party and imports cleanly, so the
shim self-disables: it must NOT install the stub, because a stub
``GrootPolicy`` would shadow the real class and break the in-process GR00T
backend (``openral_sim.policies.gr00t``). Importing this module is therefore a
harmless no-op on a modern lerobot.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
import types
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

_STUB_NAME = "lerobot.policies.groot.modeling_groot"


def _install_stub() -> None:
    if _STUB_NAME in sys.modules:
        return
    try:
        # lerobot >= 0.6.0: the real modeling_groot imports fine — use it, so the
        # genuine GrootPolicy (with from_pretrained) is what callers resolve.
        importlib.import_module(_STUB_NAME)
        return
    except Exception:
        # Only the broken 0.5.x combination (or a groot-deps-less install) lands
        # here; install the empty stub so `lerobot.policies` can still initialise.
        stub = types.ModuleType(_STUB_NAME)
        stub.GrootPolicy = type("GrootPolicy", (), {})  # type: ignore[attr-defined]
        sys.modules[_STUB_NAME] = stub


_install_stub()


def sanitize_smolvla_config(repo_id: str, *, revision: str | None = None) -> None:
    """Strip fields lerobot's ``SmolVLAConfig`` rejects from a checkpoint config.

    Some published SO-101 SmolVLA checkpoints were saved by a different lerobot
    version and carry config keys the installed one no longer accepts — e.g.
    ``nota-gmbh/so101_pick_place_pen_smolvla`` ships ``pretrained_revision``,
    which lerobot 0.5.x's draccus-backed ``SmolVLAConfig`` rejects with
    ``DecodingError`` before ``from_pretrained`` can build the policy.

    This resolves the checkpoint's ``config.json`` (local dir or HF cache) and
    rewrites it in place with only the keys valid for ``SmolVLAConfig`` (plus
    the draccus ``type`` discriminator). Idempotent, no-op when the config is
    already clean or lerobot / the config are unavailable. Call it immediately
    before ``SmolVLAPolicy.from_pretrained(repo_id)``.

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
    if not cfg_path.exists():
        return

    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    valid = {f.name for f in dataclasses.fields(SmolVLAConfig)}
    keep = {k: v for k, v in raw.items() if k in valid or k == "type"}
    dropped = sorted(set(raw) - set(keep))
    if dropped:
        cfg_path.write_text(json.dumps(keep, indent=2), encoding="utf-8")
        log.info("smolvla.config_sanitized", repo_id=repo_id, dropped=dropped)
