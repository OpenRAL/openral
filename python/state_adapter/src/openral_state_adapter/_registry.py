"""Layout-adapter registry (ADR-0027).

Single mapping from the closed :data:`openral_core.StateLayout` literal
to an :class:`~openral_state_adapter._protocol.Assembler` function.
Layout files (one per literal value) register themselves at import via
:func:`register`. The reasoner palette filter and the skill_runner both
consult this registry — when a layout is present, the wrapped-task-space
drop flips to admit-with-adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openral_core import ROSConfigError, StateLayout

from openral_state_adapter._protocol import Assembler

if TYPE_CHECKING:
    from numpy import float32
    from numpy.typing import NDArray
    from openral_core import StateContractBindings

    from openral_state_adapter._protocol import TfLookup


_LAYOUT_ASSEMBLERS: dict[StateLayout, Assembler] = {}


def register(layout: StateLayout, assembler: Assembler) -> None:
    """Bind ``assembler`` to ``layout``. Overrides any prior registration.

    Layouts MUST be registered before
    :meth:`openral_state_adapter.assemble_state` is invoked with that
    layout — typically by importing the matching ``layouts/<layout>.py``
    module (each layout file calls ``register`` at module scope).
    """
    _LAYOUT_ASSEMBLERS[layout] = assembler


def registered_layouts() -> frozenset[StateLayout]:
    """Snapshot of the layouts that currently have an assembler.

    The reasoner palette filter calls this to decide whether to admit a
    wrapped-task-space rSkill: if its ``state_contract.layout`` is in
    the returned set, the skill is admitted (with the adapter inline);
    otherwise it falls through to the existing "wrapped task-space
    layout" drop path.
    """
    return frozenset(_LAYOUT_ASSEMBLERS.keys())


def assemble_state(
    layout: StateLayout,
    bindings: StateContractBindings,
    joint_positions: dict[str, float],
    tf_lookup: TfLookup,
) -> NDArray[float32]:
    """Look up the assembler for ``layout`` and run it.

    Raises:
        ROSConfigError: When no assembler is registered for ``layout`` —
            the skill_runner should pre-check via
            :func:`registered_layouts` so the dispatch failure becomes
            a palette-time drop instead of a 5 Hz runtime error.
    """
    assembler = _LAYOUT_ASSEMBLERS.get(layout)
    if assembler is None:
        raise ROSConfigError(
            f"openral_state_adapter: no assembler registered for "
            f"state_contract.layout={layout!r}. Available: "
            f"{sorted(_LAYOUT_ASSEMBLERS.keys())!r}. "
            f"Register one in python/state_adapter/src/openral_state_adapter/"
            f"layouts/<layout>.py, or drop the rSkill from the palette.",
        )
    return assembler(bindings, joint_positions, tf_lookup)
