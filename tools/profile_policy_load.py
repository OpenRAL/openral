#!/usr/bin/env python3
"""Profile one policy load and print a phase-by-phase wall-time breakdown.

Use when a policy load takes a surprising amount of time and you want to know
which phase to attack. Drives ``openral_sim.factory.make_policy`` end-to-end
against the given rSkill manifest, captures every
``<prefix>_<name>_{start,heartbeat,done}`` event from
``openral_rskill._diagnostics.phase_timer``, and renders a phase/elapsed_s/
share table. Captures the same events ``openral dashboard`` ingests via OTel,
so the numbers match what an operator sees live.

Usage::

    uv run tools/profile_policy_load.py --rskill rskills/rldx1-ft-rc365-nf4

    # Bypass HF cache validation for inner lerobot calls not covered by our
    # local_files_only=True fast-path:
    HF_HUB_OFFLINE=1 uv run tools/profile_policy_load.py \\
        --rskill rskills/rldx1-ft-rc365-nf4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import structlog

# ── structlog capture --------------------------------------------------------


class _PhaseCapture:
    """``structlog`` processor that buffers `_start` / `_done` events.

    Each `phase_timer(name=…, prefix=…)` emits a ``<prefix>_<name>_start`` and
    matching ``..._done`` event (``elapsed_s`` float on the done side); matched
    by event name into ``(phase_name, elapsed_s)`` pairs in run order.
    """

    def __init__(self) -> None:
        self.pairs: list[tuple[str, float]] = []
        # Fallback start time, in case event_dict ever lacks elapsed_s.
        self._open: dict[str, float] = {}

    def __call__(self, logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        del logger, method_name
        event: str = str(event_dict.get("event", ""))
        if event.endswith("_start"):
            self._open[event[: -len("_start")]] = time.monotonic()
        elif event.endswith("_done"):
            phase = event[: -len("_done")]
            elapsed_s_raw = event_dict.get("elapsed_s")
            if isinstance(elapsed_s_raw, (int, float)):
                elapsed = float(elapsed_s_raw)
            else:
                started = self._open.get(phase)
                elapsed = (time.monotonic() - started) if started is not None else 0.0
            self.pairs.append((phase, elapsed))
            self._open.pop(phase, None)
        return event_dict


# ── argparse + manifest → SimEnvironment stub ------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="profile_policy_load",
        description="Print a phase-by-phase wall-time breakdown of one policy load.",
    )
    p.add_argument(
        "--rskill",
        required=True,
        help="Path to an rSkill directory (containing rskill.yaml) under rskills/.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="VLASpec.device (auto, cpu, cuda:0, …). Default: auto.",
    )
    return p.parse_args(argv)


class _SimpleSceneCfg:
    """Minimal `env_cfg.scene` carrier — only `.cameras` is read."""

    def __init__(self) -> None:
        self.cameras: tuple[str, ...] = ()


class _SimpleEnvCfg:
    """Minimal env_cfg carrier mirroring `rskill_runner_node._SimpleEnvCfg`."""

    def __init__(self, *, vla: Any) -> None:
        self.vla = vla
        self.scene = _SimpleSceneCfg()
        self.robot_id = ""


def _build_env_cfg(rskill_dir: Path, *, device: str) -> _SimpleEnvCfg:
    from openral_core import RSkillManifest, VLASpec

    manifest_path = rskill_dir / "rskill.yaml"
    if not manifest_path.is_file():
        raise SystemExit(f"no rskill.yaml under {rskill_dir!r}")
    manifest = RSkillManifest.from_yaml(str(manifest_path))
    vla = VLASpec(
        id=manifest.model_family,
        weights_uri=str(rskill_dir),
        device=device,
        extra={},
    )
    return _SimpleEnvCfg(vla=vla)


# ── rendering ---------------------------------------------------------------


def _render(pairs: list[tuple[str, float]], total_s: float) -> str:
    if not pairs:
        return f"(no phase_timer events captured; end-to-end {total_s:.1f} s)"
    name_w = max(len(name) for name, _ in pairs)
    name_w = max(name_w, len("phase"))
    lines: list[str] = []
    lines.append(f"{'phase':<{name_w}}  {'elapsed_s':>9}  {'share':>5}")
    lines.append(f"{'─' * name_w}  {'─' * 9}  {'─' * 5}")
    covered = 0.0
    for name, elapsed in pairs:
        share = (elapsed / total_s * 100.0) if total_s > 0 else 0.0
        lines.append(f"{name:<{name_w}}  {elapsed:>9.1f}  {share:>4.0f}%")
        covered += elapsed
    lines.append(f"{'─' * name_w}  {'─' * 9}  {'─' * 5}")
    lines.append(f"{'end-to-end':<{name_w}}  {total_s:>9.1f}  {100:>4d}%")
    unaccounted = total_s - covered
    if unaccounted > 1.0:
        lines.append(
            f"{'(unaccounted)':<{name_w}}  {unaccounted:>9.1f}  "
            f"{unaccounted / total_s * 100.0:>4.0f}%"
        )
    return "\n".join(lines)


# ── main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    capture = _PhaseCapture()
    structlog.configure(
        processors=[capture, structlog.dev.ConsoleRenderer(colors=False)],  # type: ignore[list-item]  # reason: _PhaseCapture is a structlog processor by duck-typing; structlog's Processor type is too narrow to admit our recorder
    )

    rskill_dir = Path(args.rskill).resolve()
    print(f"# rskill: {rskill_dir}")
    env_cfg = _build_env_cfg(rskill_dir, device=args.device)
    print(f"# model_family: {env_cfg.vla.id!r}")
    print(f"# device: {args.device!r}")
    print(f"# HF_HUB_OFFLINE={'1' if _hf_offline() else '0'}")
    print("# loading...")
    t0 = time.monotonic()
    # Late import — make_policy pulls torch + lerobot, which is one of
    # the phases we want to measure. Doing it here keeps that cost
    # inside the profiled wall-time.
    from openral_sim.factory import make_policy

    policy = make_policy(env_cfg)  # type: ignore[arg-type]  # reason: _SimpleEnvCfg is a duck-typed stand-in for SimEnvironment for the profile-load case (no real task / scene attached)
    total_s = time.monotonic() - t0
    print()
    print(_render(capture.pairs, total_s))
    return 0


def _hf_offline() -> bool:
    import os

    return os.environ.get("HF_HUB_OFFLINE", "").strip() in {"1", "true", "True", "yes"}


if __name__ == "__main__":
    sys.exit(main())
