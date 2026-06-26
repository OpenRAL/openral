"""Layout assemblers (ADR-0027).

Each module here implements one :data:`openral_core.StateLayout` value
and calls :func:`openral_state_adapter._registry.register` at module
load. Importing the package implicitly registers every layout.
"""

from openral_state_adapter.layouts import (
    human300_16d as _human300_16d,  # reason: side-effect import — registers the assembler
)
from openral_state_adapter.layouts import (
    libero_eef8d as _libero_eef8d,  # reason: side-effect import — registers the assembler
)
from openral_state_adapter.layouts import (
    rc365 as _rc365,  # side-effect import: re-registers the human300_16d assembler under "rc365"
)

__all__: list[str] = []
