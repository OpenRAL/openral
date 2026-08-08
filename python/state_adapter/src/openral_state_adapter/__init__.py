"""Public surface for the layout-adapter registry.

Importing this package registers every shipped layout in the registry.
Consumers (skill_runner, reasoner palette filter) call
:func:`assemble_state` / :func:`registered_layouts`.
"""

from importlib.metadata import version as _pkg_version

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
__version__ = _pkg_version("openral-state-adapter")
