"""Public types the layout assemblers consume.

ADR-0027. Kept rclpy-free so the assemblers are pure functions and the
unit tests don't need a running ROS graph — the skill_runner wraps
``tf2_ros.Buffer.lookup_transform`` into the :class:`TfLookup` Protocol
at call time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from numpy import float32
    from numpy.typing import NDArray
    from openral_core import StateContractBindings


@dataclass(frozen=True)
class TransformView:
    """rclpy-free view of a ``geometry_msgs/TransformStamped``.

    Attributes:
        position: ``(x, y, z)`` translation in metres.
        quaternion_xyzw: Rotation as ``(x, y, z, w)`` — the
            ``geometry_msgs/Quaternion`` field order. The assembler
            permutes to ``wxyz`` itself iff
            ``StateContractBindings.quaternion_convention == "wxyz"``.
    """

    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


class TfLookup(Protocol):
    """Callable that returns the transform from ``source`` expressed in ``target``.

    The skill_runner's adapter wraps ``tf2_ros.Buffer.lookup_transform``
    (which takes ``target_frame``, ``source_frame``, ``time``,
    ``timeout``) into this Protocol. Implementations MUST raise an
    exception when the transform is unavailable — the assembler does
    NOT silently substitute identity.
    """

    def __call__(self, target_frame: str, source_frame: str) -> TransformView: ...


class Assembler(Protocol):
    """Pure-function signature every layout file implements.

    Joins ``bindings`` (per-robot source binding from the manifest) +
    ``joint_positions`` (``JointState.name → JointState.position``) +
    ``tf_lookup`` (per-call live TF) into the per-checkpoint state
    vector. Returns ``float32`` per ADR-0019.
    """

    def __call__(
        self,
        bindings: StateContractBindings,
        joint_positions: dict[str, float],
        tf_lookup: TfLookup,
    ) -> NDArray[float32]: ...
