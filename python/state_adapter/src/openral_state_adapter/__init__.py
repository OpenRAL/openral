"""Public surface for the layout-adapter registry (ADR-0027).

Importing this package registers every shipped layout in the registry.
Consumers (skill_runner, reasoner palette filter) call
:func:`assemble_state` / :func:`registered_layouts`.
"""

from openral_state_adapter import (
    layouts as _layouts,  # reason: side-effect import — registers every shipped layout
)
from openral_state_adapter._protocol import Assembler, TfLookup, TransformView
from openral_state_adapter._registry import (
    assemble_state,
    register,
    registered_layouts,
)

__all__ = [
    "Assembler",
    "TfLookup",
    "TransformView",
    "assemble_state",
    "register",
    "registered_layouts",
]
__version__ = "0.1.0"
