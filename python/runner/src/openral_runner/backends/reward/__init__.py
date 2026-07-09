"""Reward / progress-monitor runtime backend (``kind: "reward"``).

A reward rSkill (Robometer-4B) runs in parallel with a VLA policy and scores the
rollout: per-frame normalized progress + per-frame success probability. This
package holds the **node-side** pieces — a transport-agnostic rolling frame
buffer (:class:`~openral_runner.backends.reward.frame_source.RollingFrameBuffer`,
fed by the same ``sensor_msgs/Image`` topic the VLA uses, in sim or real) and
the in-process Robometer scorer
(:class:`~openral_runner.backends.reward.robometer_reward.RobometerInProcessReward`).

Nothing here imports torch / transformers, so the package stays importable on
any host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openral_runner.backends.reward.frame_source import RollingFrameBuffer
    from openral_runner.backends.reward.robometer_reward import (
        RobometerInProcessReward,
        build_reward_monitor,
    )
    from openral_runner.backends.reward.topreward_reward import (
        TOPRewardMonitor,
        build_topreward_monitor,
    )

__all__ = [
    "RobometerInProcessReward",
    "RollingFrameBuffer",
    "TOPRewardMonitor",
    "build_reward_monitor",
    "build_topreward_monitor",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401 — lazy re-export; concrete types are in the submodules
    """Lazy re-export so importing the package pulls in no optional deps."""
    if name == "RollingFrameBuffer":
        from openral_runner.backends.reward.frame_source import (  # noqa: PLC0415
            RollingFrameBuffer,
        )

        return RollingFrameBuffer
    if name in {"RobometerInProcessReward", "build_reward_monitor"}:
        from openral_runner.backends.reward import robometer_reward  # noqa: PLC0415

        return getattr(robometer_reward, name)
    if name in {"TOPRewardMonitor", "build_topreward_monitor"}:
        from openral_runner.backends.reward import topreward_reward  # noqa: PLC0415

        return getattr(topreward_reward, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
