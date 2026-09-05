"""Layout-adapter registry for per-checkpoint state-vector assembly.

Single mapping from the closed :data:`openral_core.StateLayout` literal
to an :class:`~openral_state_adapter._protocol.Assembler` function.
Layout files (one per literal value) register themselves at import via
:func:`register`. The reasoner palette filter and the skill_runner both
consult this registry — when a layout is present, the wrapped-task-space
drop flips to admit-with-adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openral_core import ROSConfigError, ROSPerceptionStale, StateLayout

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

    Every layout assembler indexes ``joint_positions`` by the names the
    manifest bound, so the presence check belongs here rather than in each
    layout: one guard covers every registered layout, including ones added
    later.

    Raises:
        ROSConfigError: When no assembler is registered for ``layout`` —
            the skill_runner should pre-check via
            :func:`registered_layouts` so the dispatch failure becomes
            a palette-time drop instead of a 5 Hz runtime error. Also when
            the robot is publishing joints but not the bound ones, which no
            amount of waiting will fix.
        ROSPerceptionStale: When no joint frame has arrived at all. Kept
            distinct from the config error above because an empty frame says
            nothing about the manifest, whereas a populated frame that lacks
            the bound joint does.
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
    missing = [name for name in bindings.gripper_qpos_joints if name not in joint_positions]
    if missing:
        # Two different faults, and the message has to tell them apart because
        # the second is the one that actually happened (2026-09-05): a sim on
        # DDS domain 0 discovered a LIVE OpenArm on another machine, and the
        # populated frame it read carried `openarm_*` joints instead of
        # `panda_*`. A bare KeyError from inside a layout said nothing; naming
        # what the robot DOES publish is what identified the foreign robot.
        if not joint_positions:
            raise ROSPerceptionStale(
                f"openral_state_adapter: no joint state has arrived yet, so "
                f"state_contract.layout={layout!r} cannot be assembled "
                f"(needs {sorted(missing)!r}).",
            )
        raise ROSConfigError(
            f"openral_state_adapter: state_contract.layout={layout!r} binds "
            f"gripper_qpos_joints={sorted(missing)!r}, which the robot does not "
            f"publish. It publishes {sorted(joint_positions)!r}. Either the "
            f"manifest's bindings are wrong for this robot, or these joints "
            f"belong to ANOTHER robot sharing the DDS graph — check "
            f"`ros2 topic info /joint_states --no-daemon` for a second publisher.",
        )
    return assembler(bindings, joint_positions, tf_lookup)
