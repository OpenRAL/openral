"""Reward / progress-monitor runtime backend (``kind: "reward"``, ADR-0057).

A reward rSkill (Robometer-4B) runs in parallel with a VLA policy and scores the
rollout: per-frame normalized progress + per-frame success probability. This
package holds the **node-side** pieces — a transport-agnostic rolling frame
buffer (:class:`~openral_runner.backends.reward.frame_source.RollingFrameBuffer`,
fed by the same ``sensor_msgs/Image`` topic the VLA uses, in sim or real) and a
ZMQ client to a stateless scoring sidecar
(:class:`~openral_runner.backends.reward.robometer_reward.RobometerReward`).

The heavy NF4 model runs out-of-process in its own venv
(``tools/robometer_sidecar.py``); nothing here imports torch / transformers, so
the package stays importable on any host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openral_runner.backends.reward.frame_source import RollingFrameBuffer
    from openral_runner.backends.reward.robometer_reward import (
        RobometerReward,
        build_reward_monitor,
    )

__all__ = ["RobometerReward", "RollingFrameBuffer", "build_reward_monitor"]


def __getattr__(name: str) -> Any:
    """Lazy re-export so importing the package pulls in no optional deps."""
    if name == "RollingFrameBuffer":
        from openral_runner.backends.reward.frame_source import (  # noqa: PLC0415
            RollingFrameBuffer,
        )

        return RollingFrameBuffer
    if name in {"RobometerReward", "build_reward_monitor"}:
        from openral_runner.backends.reward import robometer_reward  # noqa: PLC0415

        return getattr(robometer_reward, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
