"""Policy adapter protocol — the contract every VLA backend must satisfy.

A :class:`PolicyAdapter` is the eval-layer wrapper around a VLA / scripted
policy / mock. It hides the differences between SmolVLA, π0.5, xVLA, and
random/zero baselines behind a uniform interface so the runner does not need
to know which one is in use.

Implementations live under :mod:`openral_sim.{policies,backends}`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from openral_core import VLASpec

    from openral_sim.rollout import Observation


@runtime_checkable
class PolicyAdapter(Protocol):
    """Uniform VLA / policy interface used by the runner.

    Attributes:
        spec: The :class:`openral_core.VLASpec` this adapter was built for.
        device: Resolved torch / numpy device string (``"cuda:0"``, ``"cpu"``).
    """

    spec: VLASpec
    device: str

    def reset(self) -> None:
        """Reset internal state (action queue, RNG) at the start of each episode."""

    def step(
        self,
        observation: Observation,
        instruction: str,
    ) -> NDArray[np.float32]:
        """Produce the next action vector for the given observation.

        Args:
            observation: Adapter-specific observation dict produced by the env.
            instruction: Natural-language task string from
                :attr:`openral_core.TaskSpec.instruction`.

        Returns:
            1-D float32 NumPy array of length ``action_dim``. Adapters must
            return a flat per-step action even if their underlying VLA emits
            chunks — chunk caching belongs in the adapter.
        """

    def close(self) -> None:
        """Release any GPU memory / file handles."""

    # Visuomotor adapters MAY also implement::
    #
    #     def last_input_frame(self) -> NDArray[np.uint8] | None:
    #         '''HWC uint8 RGB image of what the VLA saw last step.'''
    #
    # The runner checks for this method via ``getattr`` so scripted /
    # mock policies don't need to implement it. See
    # :class:`openral_sim.SimRunner`.
