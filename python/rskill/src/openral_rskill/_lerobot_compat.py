"""Compatibility shim for ``lerobot.policies`` import side-effects.

Importing ``lerobot.policies`` (any submodule) eagerly initialises
``lerobot.policies.groot.__init__``, which imports ``modeling_groot``.
Upstream ``lerobot==0.5.0/0.5.1`` ships a ``GR00TN15Config`` dataclass that
fails to construct on Python 3.12 with ``transformers>=5.3``: the
``@dataclass`` decorator rejects ``field(init=False)`` fields without
defaults when the parent ``PretrainedConfig`` carries class-level attributes
that dataclass treats as defaults — yielding ``TypeError: non-default
argument 'backbone_cfg' follows default argument``.

Importing this module **before** any ``lerobot.policies`` import installs a
stub for ``lerobot.policies.groot.modeling_groot`` so the package
initialises. Skills that actually depend on GR00T cannot use this shim and
must wait for an upstream fix.
"""

from __future__ import annotations

import sys
import types

_STUB_NAME = "lerobot.policies.groot.modeling_groot"


def _install_stub() -> None:
    if _STUB_NAME in sys.modules:
        return
    stub = types.ModuleType(_STUB_NAME)
    stub.GrootPolicy = type("GrootPolicy", (), {})  # type: ignore[attr-defined]
    sys.modules[_STUB_NAME] = stub


_install_stub()
